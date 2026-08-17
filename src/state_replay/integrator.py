"""A small, deterministic state integrator and persistent replay journal.

The integrator deliberately knows nothing about Counter-Strike rendering.  It
owns the other side of that boundary: complete checkpoints and ordered,
hash-chained entity transactions.  A renderer consumes immutable snapshots.

Only JSON data is accepted.  An omitted field in an update means "unchanged";
an explicit JSON null is retained as a value.  Entity/component removal is
therefore always an explicit operation, never an overloaded null.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

SNAPSHOT_SCHEMA = "tardigrade/state-snapshot/v1"
MANIFEST_SCHEMA = "tardigrade/state-manifest/v1"
TRANSACTION_SCHEMA = "tardigrade/state-transaction/v1"


class ProtocolError(ValueError):
    """The document is malformed or cannot be deterministically represented."""


class ConflictError(ProtocolError):
    """The document conflicts with the already committed history."""


class NotFoundError(ProtocolError):
    """A stream, state position, or action was not found."""


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"value is not canonical JSON: {exc}") from exc


def _clone(value: Any) -> Any:
    return copy.deepcopy(value)


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ProtocolError(f"{name} must be an integer >= {minimum}")
    return value


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ProtocolError(f"{name} must be a non-empty string")
    return value


def _position(doc: Mapping[str, Any]) -> tuple[int, int]:
    return (_integer(doc.get("tick"), "tick"),
            _integer(doc.get("subtick", 0), "subtick"))


def _validate_components(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ProtocolError("components must be an object")
    for name in value:
        _string(name, "component name")
    _canonical_json(value)
    return _clone(value)


def _validate_entity(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ProtocolError("each entity must be an object")
    entity_id = _string(value.get("id"), "entity id")
    generation = _integer(value.get("generation"), "entity generation")
    entity_class = _string(value.get("class"), "entity class")
    components = _validate_components(value.get("components"))
    allowed = {"id", "generation", "class", "components"}
    extras = set(value) - allowed
    if extras:
        raise ProtocolError(f"unknown entity fields: {sorted(extras)}")
    return {"id": entity_id, "generation": generation,
            "class": entity_class, "components": components}


def _entity_list(state: Mapping[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [_clone(state[key]) for key in sorted(state)]


def state_hash(entities: Sequence[Mapping[str, Any]]) -> str:
    """Hash canonical entity state, independent of JSON key/list order."""
    checked: dict[str, dict[str, Any]] = {}
    for raw in entities:
        entity = _validate_entity(raw)
        if entity["id"] in checked:
            raise ProtocolError(f"duplicate active entity id {entity['id']!r}")
        checked[entity["id"]] = entity
    payload = {"entities": _entity_list(checked)}
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _hash_state(state: Mapping[str, dict[str, Any]]) -> str:
    payload = {"entities": _entity_list(state)}
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _snapshot(stream_id: str, tick: int, subtick: int, sequence: int,
              state: Mapping[str, dict[str, Any]], *,
              discontinuity: Mapping[str, Any] | None = None) -> dict[str, Any]:
    result = {
        "schema": SNAPSHOT_SCHEMA,
        "stream_id": stream_id,
        "tick": tick,
        "subtick": subtick,
        "sequence": sequence,
        "state_hash": _hash_state(state),
        "entities": _entity_list(state),
    }
    if discontinuity is not None:
        result["discontinuity"] = _clone(dict(discontinuity))
    return result


def _merge_patch(target: Any, patch: Any) -> Any:
    """Deep object merge in which null is data, not a deletion sentinel."""
    if not isinstance(patch, dict) or not isinstance(target, dict):
        return _clone(patch)
    result = _clone(target)
    for key, value in patch.items():
        _string(key, "component field")
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_patch(result[key], value)
        else:
            result[key] = _clone(value)
    return result


class StateIntegrator:
    """One stream backed by ``<storage_root>/state-replay.sqlite3``.

    Use :meth:`create` once, then :meth:`open` after process restarts.  SQLite
    WAL storage permits concurrent readers while a capture process appends.
    """

    DB_NAME = "state-replay.sqlite3"

    def __init__(self, storage_root: os.PathLike[str] | str,
                 stream_id: str) -> None:
        self.storage_root = Path(storage_root).expanduser().resolve()
        self.stream_id = _string(stream_id, "stream_id")
        self._lock = threading.RLock()
        db = self.storage_root / self.DB_NAME
        if not db.is_file():
            raise NotFoundError(f"state store does not exist: {db}")
        self._conn = sqlite3.connect(db, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._configure()
        self._load()

    @classmethod
    def create(cls, storage_root: os.PathLike[str] | str,
               manifest: Mapping[str, Any],
               initial_checkpoint: Mapping[str, Any]) -> "StateIntegrator":
        root = Path(storage_root).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        if not root.is_dir():
            raise ProtocolError(f"storage root is not a directory: {root}")
        checked_manifest = cls._check_manifest(manifest)
        checked_checkpoint, state = cls._check_checkpoint(
            initial_checkpoint, checked_manifest["stream_id"])
        if checked_checkpoint["sequence"] != 0:
            raise ProtocolError("initial checkpoint sequence must be 0")
        if checked_checkpoint.get("discontinuity"):
            raise ProtocolError("initial checkpoint is not a discontinuity")

        conn = sqlite3.connect(root / cls.DB_NAME)
        conn.row_factory = sqlite3.Row
        cls._configure_connection(conn)
        cls._create_tables(conn)
        stream_id = checked_manifest["stream_id"]
        internal = {
            "entities": _entity_list(state),
            "generation_watermarks": {
                key: entity["generation"] for key, entity in state.items()
            },
        }
        try:
            with conn:
                conn.execute(
                    "INSERT INTO manifests(stream_id, document) VALUES (?, ?)",
                    (stream_id, _canonical_json(checked_manifest)),
                )
                conn.execute(
                    "INSERT INTO checkpoints(stream_id,tick,subtick,sequence,"
                    "state_hash,document,internal) VALUES (?,?,?,?,?,?,?)",
                    (stream_id, checked_checkpoint["tick"],
                     checked_checkpoint["subtick"], 0,
                     checked_checkpoint["state_hash"],
                     _canonical_json(checked_checkpoint),
                     _canonical_json(internal)),
                )
        except sqlite3.IntegrityError as exc:
            raise ConflictError(f"stream already exists: {stream_id}") from exc
        finally:
            conn.close()
        return cls(root, stream_id)

    @classmethod
    def open(cls, storage_root: os.PathLike[str] | str,
             stream_id: str) -> "StateIntegrator":
        return cls(storage_root, stream_id)

    @staticmethod
    def _configure_connection(conn: sqlite3.Connection) -> None:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")

    def _configure(self) -> None:
        self._configure_connection(self._conn)
        self._create_tables(self._conn)

    @staticmethod
    def _create_tables(conn: sqlite3.Connection) -> None:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS manifests(
              stream_id TEXT PRIMARY KEY,
              document TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS transactions(
              stream_id TEXT NOT NULL,
              sequence INTEGER NOT NULL,
              tick INTEGER NOT NULL,
              subtick INTEGER NOT NULL,
              base_hash TEXT NOT NULL,
              post_hash TEXT NOT NULL,
              document TEXT NOT NULL,
              PRIMARY KEY(stream_id, sequence),
              FOREIGN KEY(stream_id) REFERENCES manifests(stream_id)
            );
            CREATE INDEX IF NOT EXISTS transaction_position
              ON transactions(stream_id, tick, subtick, sequence);
            CREATE TABLE IF NOT EXISTS checkpoints(
              stream_id TEXT NOT NULL,
              tick INTEGER NOT NULL,
              subtick INTEGER NOT NULL,
              sequence INTEGER NOT NULL,
              state_hash TEXT NOT NULL,
              document TEXT NOT NULL,
              internal TEXT NOT NULL,
              PRIMARY KEY(stream_id, sequence),
              FOREIGN KEY(stream_id) REFERENCES manifests(stream_id)
            );
            CREATE INDEX IF NOT EXISTS checkpoint_position
              ON checkpoints(stream_id, tick, subtick, sequence);
            CREATE TABLE IF NOT EXISTS actions(
              stream_id TEXT NOT NULL,
              action_id TEXT NOT NULL,
              sequence INTEGER NOT NULL,
              op_index INTEGER NOT NULL,
              document TEXT NOT NULL,
              pre_hash TEXT NOT NULL,
              post_hash TEXT NOT NULL,
              PRIMARY KEY(stream_id, action_id),
              FOREIGN KEY(stream_id, sequence)
                REFERENCES transactions(stream_id, sequence)
            );
        """)

    @staticmethod
    def _check_manifest(doc: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(doc, Mapping):
            raise ProtocolError("manifest must be an object")
        if doc.get("schema") != MANIFEST_SCHEMA:
            raise ProtocolError(f"manifest schema must be {MANIFEST_SCHEMA}")
        stream_id = _string(doc.get("stream_id"), "stream_id")
        rate = _integer(doc.get("tick_rate_hz"), "tick_rate_hz", 1)
        checked = _clone(dict(doc))
        checked["stream_id"] = stream_id
        checked["tick_rate_hz"] = rate
        _canonical_json(checked)
        return checked

    @staticmethod
    def _check_checkpoint(
        doc: Mapping[str, Any], stream_id: str
    ) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
        if not isinstance(doc, Mapping):
            raise ProtocolError("checkpoint must be an object")
        if doc.get("schema") != SNAPSHOT_SCHEMA:
            raise ProtocolError(f"checkpoint schema must be {SNAPSHOT_SCHEMA}")
        if doc.get("stream_id") != stream_id:
            raise ProtocolError("checkpoint stream_id does not match")
        tick, subtick = _position(doc)
        sequence = _integer(doc.get("sequence", 0), "sequence")
        raw_entities = doc.get("entities")
        if not isinstance(raw_entities, list):
            raise ProtocolError("checkpoint entities must be an array")
        state: dict[str, dict[str, Any]] = {}
        for raw in raw_entities:
            entity = _validate_entity(raw)
            if entity["id"] in state:
                raise ProtocolError(f"duplicate entity id {entity['id']!r}")
            state[entity["id"]] = entity
        actual_hash = _hash_state(state)
        if doc.get("state_hash") != actual_hash:
            raise ConflictError(
                f"checkpoint state_hash mismatch: expected {actual_hash}, "
                f"received {doc.get('state_hash')!r}")
        checked = _snapshot(stream_id, tick, subtick, sequence, state)
        if "discontinuity" in doc:
            discontinuity = doc["discontinuity"]
            if not isinstance(discontinuity, Mapping):
                raise ProtocolError("discontinuity must be an object")
            kind = discontinuity.get("kind")
            if kind not in {"gap", "seek", "reset", "source-switch"}:
                raise ProtocolError("invalid discontinuity kind")
            checked["discontinuity"] = {
                "kind": kind,
                "reason": _string(discontinuity.get("reason"),
                                  "discontinuity reason"),
            }
        _canonical_json(checked)
        return checked, state

    @staticmethod
    def make_checkpoint(stream_id: str, tick: int, subtick: int,
                        sequence: int,
                        entities: Sequence[Mapping[str, Any]],
                        *, discontinuity: bool | str = False,
                        reason: str | None = None) -> dict[str, Any]:
        state: dict[str, dict[str, Any]] = {}
        for raw in entities:
            entity = _validate_entity(raw)
            if entity["id"] in state:
                raise ProtocolError(f"duplicate entity id {entity['id']!r}")
            state[entity["id"]] = entity
        result = _snapshot(_string(stream_id, "stream_id"),
                           _integer(tick, "tick"),
                           _integer(subtick, "subtick"),
                           _integer(sequence, "sequence"), state)
        if discontinuity:
            kind = "reset" if discontinuity is True else discontinuity
            if kind not in {"gap", "seek", "reset", "source-switch"}:
                raise ProtocolError("invalid discontinuity kind")
            result["discontinuity"] = {
                "kind": kind,
                "reason": _string(reason, "reason"),
            }
        return result

    def _load(self) -> None:
        row = self._conn.execute(
            "SELECT document FROM manifests WHERE stream_id=?",
            (self.stream_id,),
        ).fetchone()
        if row is None:
            raise NotFoundError(f"unknown stream {self.stream_id!r}")
        self.manifest = json.loads(row["document"])
        cp = self._conn.execute(
            "SELECT * FROM checkpoints WHERE stream_id=? "
            "ORDER BY sequence DESC LIMIT 1", (self.stream_id,),
        ).fetchone()
        if cp is None:
            raise ProtocolError("stored stream has no checkpoint")
        internal = json.loads(cp["internal"])
        self._state = {e["id"]: e for e in internal["entities"]}
        self._watermarks = {
            str(key): int(value)
            for key, value in internal["generation_watermarks"].items()
        }
        self._tick, self._subtick = int(cp["tick"]), int(cp["subtick"])
        self._sequence = int(cp["sequence"])
        self._hash = str(cp["state_hash"])
        self._discontinuity = json.loads(cp["document"]).get("discontinuity")
        rows = self._conn.execute(
            "SELECT document,post_hash FROM transactions "
            "WHERE stream_id=? AND sequence>? ORDER BY sequence",
            (self.stream_id, self._sequence),
        )
        for row in rows:
            result = self._apply_checked(
                json.loads(row["document"]), self._state, self._watermarks,
                self._tick, self._subtick, self._sequence, self._hash,
            )
            (self._state, self._watermarks, self._tick, self._subtick,
             self._sequence, self._hash, _) = result
            self._discontinuity = None
            if self._hash != row["post_hash"]:
                raise ProtocolError("persisted transaction post_hash is corrupt")

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def current_snapshot(self) -> dict[str, Any]:
        with self._lock:
            return _snapshot(self.stream_id, self._tick, self._subtick,
                             self._sequence, self._state,
                             discontinuity=self._discontinuity)

    def _apply_checked(
        self, doc: Mapping[str, Any], state: Mapping[str, dict[str, Any]],
        watermarks: Mapping[str, int], tick: int, subtick: int,
        sequence: int, current_hash: str,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, int], int, int, int,
               str, list[tuple[str, int, dict[str, Any], str]]]:
        if not isinstance(doc, Mapping):
            raise ProtocolError("transaction must be an object")
        if doc.get("schema") != TRANSACTION_SCHEMA:
            raise ProtocolError(f"transaction schema must be {TRANSACTION_SCHEMA}")
        if doc.get("stream_id") != self.stream_id:
            raise ProtocolError("transaction stream_id does not match")
        new_sequence = _integer(doc.get("sequence"), "sequence")
        if new_sequence != sequence + 1:
            raise ConflictError(
                f"sequence must be {sequence + 1}, received {new_sequence}")
        new_tick, new_subtick = _position(doc)
        if (new_tick, new_subtick) < (tick, subtick):
            raise ConflictError("transaction position moves backwards")
        if doc.get("base_hash") != current_hash:
            raise ConflictError(
                f"base_hash mismatch: expected {current_hash}, "
                f"received {doc.get('base_hash')!r}")
        ops = doc.get("ops")
        if not isinstance(ops, list) or not ops:
            raise ProtocolError("transaction ops must be a non-empty array")

        next_state = _clone(dict(state))
        next_watermarks = dict(watermarks)
        actions: list[tuple[str, int, dict[str, Any], str]] = []
        for index, raw_op in enumerate(ops):
            if not isinstance(raw_op, dict):
                raise ProtocolError(f"operation {index} must be an object")
            op = raw_op.get("op")
            if op == "create":
                entity = _validate_entity(raw_op.get("entity"))
                entity_id, generation = entity["id"], entity["generation"]
                if entity_id in next_state:
                    raise ConflictError(f"create of active entity {entity_id!r}")
                if entity_id in next_watermarks and generation <= next_watermarks[entity_id]:
                    raise ConflictError(
                        f"generation {generation} does not advance retired "
                        f"entity {entity_id!r} generation "
                        f"{next_watermarks[entity_id]}")
                next_state[entity_id] = entity
                next_watermarks[entity_id] = generation
            elif op == "update":
                entity_id = _string(raw_op.get("id"), "update id")
                generation = _integer(raw_op.get("generation"),
                                      "update generation")
                if entity_id not in next_state:
                    raise ConflictError(f"update of inactive entity {entity_id!r}")
                entity = next_state[entity_id]
                if generation != entity["generation"]:
                    raise ConflictError(f"stale generation for entity {entity_id!r}")
                if "components" in raw_op:
                    patch = _validate_components(raw_op["components"])
                    entity["components"] = _merge_patch(entity["components"], patch)
                removals = raw_op.get("remove_components", [])
                if not isinstance(removals, list):
                    raise ProtocolError("remove_components must be an array")
                for name in removals:
                    name = _string(name, "removed component")
                    if name not in entity["components"]:
                        raise ConflictError(
                            f"cannot remove absent component {name!r} from {entity_id!r}")
                    del entity["components"][name]
                if "components" not in raw_op and not removals:
                    raise ProtocolError("update has no component changes")
            elif op == "destroy":
                entity_id = _string(raw_op.get("id"), "destroy id")
                generation = _integer(raw_op.get("generation"),
                                      "destroy generation")
                if entity_id not in next_state:
                    raise ConflictError(f"destroy of inactive entity {entity_id!r}")
                if generation != next_state[entity_id]["generation"]:
                    raise ConflictError(f"stale generation for entity {entity_id!r}")
                del next_state[entity_id]
            elif op == "action":
                action_id = _string(raw_op.get("action_id"), "action_id")
                kind = _string(raw_op.get("kind"), "action kind")
                action = _clone(raw_op)
                action["kind"] = kind
                _canonical_json(action)
                actions.append((action_id, index, action,
                                _hash_state(next_state)))
            else:
                raise ProtocolError(f"unknown operation {op!r} at index {index}")

        new_hash = _hash_state(next_state)
        expected_post = doc.get("post_hash")
        if expected_post is not None and expected_post != new_hash:
            raise ConflictError(
                f"post_hash mismatch: expected {new_hash}, received {expected_post!r}")
        return (next_state, next_watermarks, new_tick, new_subtick,
                new_sequence, new_hash, actions)

    def ingest(self, transaction: Mapping[str, Any]) -> dict[str, Any]:
        return self.ingest_many([transaction])[-1]

    def ingest_many(self, transactions: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """Validate and atomically commit a batch of ordered transactions."""
        docs = list(transactions)
        if not docs:
            return []
        with self._lock:
            state, watermarks = _clone(self._state), dict(self._watermarks)
            tick, subtick = self._tick, self._subtick
            sequence, current_hash = self._sequence, self._hash
            staged = []
            seen_actions: set[str] = set()
            for doc in docs:
                result = self._apply_checked(
                    doc, state, watermarks, tick, subtick, sequence, current_hash)
                (state, watermarks, tick, subtick,
                 sequence, current_hash, actions) = result
                for action_id, _, _, _ in actions:
                    if action_id in seen_actions:
                        raise ConflictError(f"duplicate action_id {action_id!r} in batch")
                    seen_actions.add(action_id)
                staged.append((dict(doc), result))

            try:
                with self._conn:
                    for doc, result in staged:
                        (_, _, dtick, dsubtick, dsequence,
                         post_hash, actions) = result
                        self._conn.execute(
                            "INSERT INTO transactions(stream_id,sequence,tick,"
                            "subtick,base_hash,post_hash,document) VALUES(?,?,?,?,?,?,?)",
                            (self.stream_id, dsequence, dtick, dsubtick,
                             doc["base_hash"], post_hash, _canonical_json(doc)),
                        )
                        for action_id, op_index, action, pre_hash in actions:
                            self._conn.execute(
                                "INSERT INTO actions(stream_id,action_id,sequence,"
                                "op_index,document,pre_hash,post_hash) "
                                "VALUES(?,?,?,?,?,?,?)",
                                (self.stream_id, action_id, dsequence, op_index,
                                 _canonical_json(action), pre_hash, post_hash),
                            )
            except sqlite3.IntegrityError as exc:
                raise ConflictError(f"journal identity already exists: {exc}") from exc

            self._state, self._watermarks = state, watermarks
            self._tick, self._subtick = tick, subtick
            self._sequence, self._hash = sequence, current_hash
            self._discontinuity = None
            return [
                _snapshot(self.stream_id, r[2], r[3], r[4], r[0])
                for _, r in staged
            ]

    def checkpoint(self, document: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Persist current state, or install an explicit discontinuity snapshot."""
        with self._lock:
            if document is None:
                checked = _snapshot(self.stream_id, self._tick, self._subtick,
                                    self._sequence, self._state)
                state = _clone(self._state)
            else:
                checked, state = self._check_checkpoint(document, self.stream_id)
                if checked.get("discontinuity"):
                    if checked["sequence"] != self._sequence + 1:
                        raise ConflictError("discontinuity sequence must advance by one")
                    if (checked["tick"], checked["subtick"]) <= (self._tick,
                                                                  self._subtick):
                        raise ConflictError("discontinuity must advance the position")
                    for entity_id, entity in state.items():
                        generation = entity["generation"]
                        if entity_id in self._state:
                            minimum = self._state[entity_id]["generation"]
                            if generation < minimum:
                                raise ConflictError(
                                    f"discontinuity regresses active entity "
                                    f"{entity_id!r} generation below {minimum}")
                        elif (entity_id in self._watermarks and generation <=
                              self._watermarks[entity_id]):
                            raise ConflictError(
                                f"discontinuity reactivates retired entity "
                                f"{entity_id!r} without advancing generation")
                else:
                    if (checked["sequence"], checked["tick"], checked["subtick"],
                            checked["state_hash"]) != (
                                self._sequence, self._tick, self._subtick, self._hash):
                        raise ConflictError(
                            "ordinary checkpoint must exactly match current state")
            watermarks = dict(self._watermarks)
            for entity_id, entity in state.items():
                watermarks[entity_id] = max(
                    watermarks.get(entity_id, -1), entity["generation"])
            internal = {"entities": _entity_list(state),
                        "generation_watermarks": watermarks}
            try:
                with self._conn:
                    self._conn.execute(
                        "INSERT INTO checkpoints(stream_id,tick,subtick,sequence,"
                        "state_hash,document,internal) VALUES(?,?,?,?,?,?,?)",
                        (self.stream_id, checked["tick"], checked["subtick"],
                         checked["sequence"], checked["state_hash"],
                         _canonical_json(checked), _canonical_json(internal)),
                    )
            except sqlite3.IntegrityError as exc:
                raise ConflictError("checkpoint already exists at this sequence") from exc
            if checked.get("discontinuity"):
                self._state, self._watermarks = state, watermarks
                self._tick, self._subtick = checked["tick"], checked["subtick"]
                self._sequence, self._hash = checked["sequence"], checked["state_hash"]
                self._discontinuity = _clone(checked["discontinuity"])
            return _clone(checked)

    def _base_for_position(
        self, tick: int, subtick: int
    ) -> tuple[dict[str, dict[str, Any]], dict[str, int], int, int, int, str,
               dict[str, Any] | None]:
        cp = self._conn.execute(
            "SELECT * FROM checkpoints WHERE stream_id=? AND "
            "(tick < ? OR (tick=? AND subtick<=?)) "
            "ORDER BY tick DESC,subtick DESC,sequence DESC LIMIT 1",
            (self.stream_id, tick, tick, subtick),
        ).fetchone()
        if cp is None:
            raise NotFoundError("requested position precedes the first checkpoint")
        internal = json.loads(cp["internal"])
        state = {e["id"]: e for e in internal["entities"]}
        watermarks = {str(k): int(v) for k, v in
                      internal["generation_watermarks"].items()}
        discontinuity = json.loads(cp["document"]).get("discontinuity")
        return (state, watermarks, int(cp["tick"]), int(cp["subtick"]),
                int(cp["sequence"]), str(cp["state_hash"]), discontinuity)

    def state_at(self, tick: int, subtick: int = 2**31 - 1) -> dict[str, Any]:
        tick, subtick = _integer(tick, "tick"), _integer(subtick, "subtick")
        with self._lock:
            state, watermarks, btick, bsub, sequence, current_hash, discontinuity = \
                self._base_for_position(tick, subtick)
            rows = self._conn.execute(
                "SELECT document FROM transactions WHERE stream_id=? AND sequence>? "
                "AND (tick < ? OR (tick=? AND subtick<=?)) ORDER BY sequence",
                (self.stream_id, sequence, tick, tick, subtick),
            )
            for row in rows:
                result = self._apply_checked(
                    json.loads(row["document"]), state, watermarks,
                    btick, bsub, sequence, current_hash)
                (state, watermarks, btick, bsub,
                 sequence, current_hash, _) = result
                discontinuity = None
            return _snapshot(self.stream_id, btick, bsub, sequence, state,
                             discontinuity=discontinuity)

    def batch_states(self, positions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """Return requested positions in input order with one forward replay.

        Transactions between the first checkpoint and last requested position
        are decoded and applied once, irrespective of batch cardinality.
        """
        if not isinstance(positions, Sequence):
            raise ProtocolError("positions must be an array")
        normalized = [_position(position) for position in positions]
        if not normalized:
            return []
        targets = sorted(set(normalized))
        with self._lock:
            (state, watermarks, tick, subtick, sequence, current_hash,
             discontinuity) = self._base_for_position(*targets[0])
            last_tick, last_subtick = targets[-1]
            events: list[tuple[int, str, sqlite3.Row]] = []
            for row in self._conn.execute(
                "SELECT * FROM transactions WHERE stream_id=? AND sequence>? "
                "AND (tick < ? OR (tick=? AND subtick<=?)) ORDER BY sequence",
                (self.stream_id, sequence, last_tick, last_tick, last_subtick),
            ):
                events.append((int(row["sequence"]), "transaction", row))
            for row in self._conn.execute(
                "SELECT * FROM checkpoints WHERE stream_id=? AND sequence>? "
                "AND (tick < ? OR (tick=? AND subtick<=?)) ORDER BY sequence",
                (self.stream_id, sequence, last_tick, last_tick, last_subtick),
            ):
                if json.loads(row["document"]).get("discontinuity"):
                    events.append((int(row["sequence"]), "checkpoint", row))
            events.sort(key=lambda item: item[0])

            cache: dict[tuple[int, int], dict[str, Any]] = {}
            event_index = 0
            for target in targets:
                while event_index < len(events):
                    _, kind, row = events[event_index]
                    position = (int(row["tick"]), int(row["subtick"]))
                    if position > target:
                        break
                    if kind == "checkpoint":
                        internal = json.loads(row["internal"])
                        state = {e["id"]: e for e in internal["entities"]}
                        watermarks = {str(k): int(v) for k, v in
                                      internal["generation_watermarks"].items()}
                        tick, subtick, sequence, current_hash = (
                            position[0], position[1], int(row["sequence"]),
                            str(row["state_hash"]))
                        discontinuity = json.loads(
                            row["document"])["discontinuity"]
                    else:
                        result = self._apply_checked(
                            json.loads(row["document"]), state, watermarks,
                            tick, subtick, sequence, current_hash)
                        (state, watermarks, tick, subtick, sequence,
                         current_hash, _) = result
                        discontinuity = None
                    event_index += 1
                cache[target] = _snapshot(
                    self.stream_id, tick, subtick, sequence, state,
                    discontinuity=discontinuity)
            return [_clone(cache[position]) for position in normalized]

    def _state_before_sequence(
        self, target_sequence: int
    ) -> tuple[dict[str, dict[str, Any]], dict[str, int], int, int, int, str]:
        cp = self._conn.execute(
            "SELECT * FROM checkpoints WHERE stream_id=? AND sequence<? "
            "ORDER BY sequence DESC LIMIT 1", (self.stream_id, target_sequence),
        ).fetchone()
        if cp is None:
            raise ProtocolError("no checkpoint precedes action transaction")
        internal = json.loads(cp["internal"])
        state = {e["id"]: e for e in internal["entities"]}
        watermarks = {str(k): int(v) for k, v in
                      internal["generation_watermarks"].items()}
        tick, subtick, sequence, current_hash = (
            int(cp["tick"]), int(cp["subtick"]), int(cp["sequence"]),
            str(cp["state_hash"]))
        rows = self._conn.execute(
            "SELECT document FROM transactions WHERE stream_id=? "
            "AND sequence>? AND sequence<? ORDER BY sequence",
            (self.stream_id, sequence, target_sequence),
        )
        for row in rows:
            result = self._apply_checked(json.loads(row["document"]), state,
                                         watermarks, tick, subtick, sequence,
                                         current_hash)
            (state, watermarks, tick, subtick,
             sequence, current_hash, _) = result
        return state, watermarks, tick, subtick, sequence, current_hash

    def action_context(self, action_id: str) -> dict[str, Any]:
        action_id = _string(action_id, "action_id")
        with self._lock:
            action_row = self._conn.execute(
                "SELECT * FROM actions WHERE stream_id=? AND action_id=?",
                (self.stream_id, action_id),
            ).fetchone()
            if action_row is None:
                raise NotFoundError(f"unknown action {action_id!r}")
            tx_row = self._conn.execute(
                "SELECT document FROM transactions WHERE stream_id=? AND sequence=?",
                (self.stream_id, int(action_row["sequence"])),
            ).fetchone()
            tx = json.loads(tx_row["document"])
            state, watermarks, tick, subtick, sequence, current_hash = \
                self._state_before_sequence(int(action_row["sequence"]))
            target = int(action_row["op_index"])
            prefix = dict(tx)
            prefix["ops"] = tx["ops"][:target]
            prefix.pop("post_hash", None)
            if prefix["ops"]:
                result = self._apply_checked(prefix, state, watermarks, tick,
                                             subtick, sequence, current_hash)
                (state, watermarks, tick, subtick,
                 _, _, _) = result
            else:
                tick, subtick = _position(tx)
            pre = _snapshot(self.stream_id, tick, subtick,
                            int(tx["sequence"]) - 1, state)
            # Apply the complete transaction from the actual previous state.
            (base_state, base_watermarks, base_tick, base_subtick,
             base_sequence, base_hash) = self._state_before_sequence(
                 int(action_row["sequence"]))
            result = self._apply_checked(tx, base_state, base_watermarks,
                                         base_tick, base_subtick,
                                         base_sequence, base_hash)
            post = _snapshot(self.stream_id, result[2], result[3], result[4],
                             result[0])
            if pre["state_hash"] != action_row["pre_hash"]:
                raise ProtocolError("stored action pre_hash is corrupt")
            if post["state_hash"] != action_row["post_hash"]:
                raise ProtocolError("stored action post_hash is corrupt")
            return {
                "stream_id": self.stream_id,
                "action": json.loads(action_row["document"]),
                "transaction_sequence": int(action_row["sequence"]),
                "operation_index": target,
                "pre": pre,
                "post": post,
            }

    def stats(self) -> dict[str, Any]:
        with self._lock:
            counts = {}
            for table in ("transactions", "checkpoints", "actions"):
                counts[table] = int(self._conn.execute(
                    f"SELECT count(*) FROM {table} WHERE stream_id=?",
                    (self.stream_id,),
                ).fetchone()[0])
            return {
                "stream_id": self.stream_id,
                "tick": self._tick,
                "subtick": self._subtick,
                "sequence": self._sequence,
                "state_hash": self._hash,
                **counts,
            }
