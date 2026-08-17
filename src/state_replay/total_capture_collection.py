"""Transactional ingestion for multi-process/map total-capture collections."""
from __future__ import annotations

import os
import json
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

from .total_capture import TotalCaptureIngestor, _sha256_file
from .total_capture_collection_contract import (
    TOTAL_CAPTURE_COLLECTION_RENDER_SCHEMA,
    TotalCaptureContractError,
    epoch_context_entity,
    epoch_key,
    epoch_stream_id,
    validate_collection_manifest,
    validate_collection_terminal,
    validate_coverage_ledger,
    validate_salvage_document,
)
from .total_capture_contract import canonical_json, validate_manifest, validate_record
from .source1_demo import Source1DemoError, Source1DemoReader


TOTAL_CAPTURE_COLLECTION_INDEX_SCHEMA = "tardigrade/total-capture-collection-index/v1"
MAX_COVERAGE_LEDGER_BYTES = 16 * 1024 * 1024


@dataclass
class CollectionEpochInput:
    """One recurrence domain: either a complete capture or explicit salvage."""

    manifest: Mapping[str, object] | None = None
    records: Iterable[Mapping[str, object]] | None = None
    artifact_paths: Mapping[str, os.PathLike[str] | str] = field(default_factory=dict)
    salvage: Mapping[str, object] | None = None


@dataclass(frozen=True)
class TotalCaptureCollectionSummary:
    collection_id: str
    corpus_status: str
    continuity_status: str
    complete_epochs: int
    salvage_epochs: int
    complete_target_active_ns: int
    output: str


def _verify_salvage_artifacts(salvage: Mapping[str, object],
                              paths: Mapping[str, os.PathLike[str] | str]) -> None:
    expected = salvage["artifact_sha256s"]
    if set(paths) != set(expected):
        raise TotalCaptureContractError(
            f"salvage artifact paths must exactly be {sorted(expected)}")
    for artifact_id, digest in expected.items():
        path = Path(paths[str(artifact_id)]).expanduser().resolve()
        if not path.is_file():
            raise TotalCaptureContractError(
                f"salvage artifact {artifact_id!r} is not a regular file")
        if _sha256_file(path) != digest:
            raise TotalCaptureContractError(
                f"salvage artifact {artifact_id!r} sha256 mismatch")


def _clock_index(records: list[Mapping[str, object]],
                 host_horizon: Mapping[str, object]) -> list[dict[str, object]]:
    start = int(host_horizon["start_monotonic_ns"])
    end = int(host_horizon["end_monotonic_ns"])
    result: list[dict[str, object]] = []
    for record in records:
        now = int(record["clock"]["monotonic_ns"])
        if not start <= now <= end:
            raise TotalCaptureContractError(
                "epoch record clock escapes declared host horizon")
        result.append({
            "monotonic_ns": now, "tick": int(record["tick"]),
            "subtick": int(record["subtick"]),
            "stream_id": str(record["stream_id"]),
            "stream_sequence": int(record["sequence"]),
            "record_sha256": str(record["record_sha256"]),
        })
    result.sort(key=lambda item: (
        item["monotonic_ns"], item["tick"], item["subtick"],
        item["stream_id"], item["stream_sequence"]))
    return result


class TotalCaptureCollectionIngestor:
    """Publish a complete corpus or an explicitly requested incomplete index."""

    def __init__(self) -> None:
        self._used = False

    def ingest(self, manifest_document: Mapping[str, object],
               terminal_document: Mapping[str, object],
               epoch_inputs: Mapping[str, CollectionEpochInput],
               output: os.PathLike[str] | str, *,
               outer_artifact_paths: Mapping[str, os.PathLike[str] | str],
               allow_target_not_met: bool = False) -> TotalCaptureCollectionSummary:
        if self._used:
            raise TotalCaptureContractError("TotalCaptureCollectionIngestor is one-shot")
        self._used = True
        manifest = validate_collection_manifest(manifest_document)
        terminal = validate_collection_terminal(terminal_document, manifest)
        expected_outer = {str(item["id"]): item
                          for item in manifest["outer_artifacts"]}
        if set(outer_artifact_paths) != set(expected_outer):
            raise TotalCaptureContractError(
                "outer artifact paths must exactly match collection manifest")
        for artifact_id, descriptor in expected_outer.items():
            path = Path(outer_artifact_paths[artifact_id]).expanduser().resolve()
            if not path.is_file() or _sha256_file(path) != descriptor["sha256"]:
                raise TotalCaptureContractError(
                    f"outer artifact {artifact_id!r} hash/path mismatch")
            if descriptor["role"] == "coverage-ledger":
                if path.stat().st_size > MAX_COVERAGE_LEDGER_BYTES:
                    raise TotalCaptureContractError(
                        "coverage ledger exceeds 16 MiB protocol limit")
                try:
                    ledger = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise TotalCaptureContractError("coverage ledger is not JSON") from exc
                validate_coverage_ledger(ledger, manifest)
        if (manifest["coverage"]["corpus_status"] != "target-met" and
                not allow_target_not_met):
            raise TotalCaptureContractError(
                "collection target not met; explicit allow_target_not_met required for salvage index")
        expected_keys = {
            epoch_key(str(item["process_epoch_id"]), str(item["map_epoch_id"]),
                      str(item["capture_epoch_id"]))
            for item in manifest["epochs"]
        }
        if set(epoch_inputs) != expected_keys:
            raise TotalCaptureContractError(
                "epoch inputs must exactly match declared capture epochs")
        final = Path(output).expanduser().resolve()
        if final.exists():
            raise TotalCaptureContractError("collection output already exists")
        final.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=f".{final.name}.partial-", dir=final.parent))
        index_entries: list[dict[str, object]] = []
        complete_epochs = 0
        salvage_epochs = 0
        try:
            epoch_root = stage / "epochs"
            epoch_root.mkdir()
            for ordinal, descriptor in enumerate(manifest["epochs"]):
                key = epoch_key(str(descriptor["process_epoch_id"]),
                                str(descriptor["map_epoch_id"]),
                                str(descriptor["capture_epoch_id"]))
                supplied = epoch_inputs[key]
                if descriptor["status"] == "complete":
                    if supplied.manifest is None or supplied.records is None or supplied.salvage is not None:
                        raise TotalCaptureContractError(
                            f"complete epoch {key!r} lacks complete capture input")
                    epoch_manifest = validate_manifest(supplied.manifest)
                    if epoch_manifest["manifest_sha256"] != descriptor[
                            "capture_manifest_sha256"]:
                        raise TotalCaptureContractError("epoch manifest hash mismatch")
                    if epoch_manifest["artifacts"]["demo_sha256"] != descriptor[
                            "native_demo_sha256"]:
                        raise TotalCaptureContractError("epoch native demo hash mismatch")
                    if epoch_manifest["producer_closure"]["native_demo"][
                            "map_name"] != descriptor["map_name"]:
                        raise TotalCaptureContractError("epoch map name mismatch")
                    try:
                        with Source1DemoReader.open(
                                supplied.artifact_paths["demo"]) as reader:
                            playback_ns = round(
                                reader.header.playback_time * 1_000_000_000)
                    except (OSError, Source1DemoError) as exc:
                        raise TotalCaptureContractError(
                            "epoch demo cannot prove playback duration") from exc
                    if playback_ns != descriptor["demo_playback_ns"]:
                        raise TotalCaptureContractError(
                            "epoch demo playback duration mismatch")
                    records = [validate_record(item, epoch_manifest)
                               for item in supplied.records]
                    clock_index = _clock_index(records, descriptor["host_horizon"])
                    stream_id = epoch_stream_id(
                        str(manifest["collection_id"]), ordinal,
                        str(descriptor["capture_epoch_id"]))
                    relative = Path("epochs") / f"{ordinal:06d}"
                    TotalCaptureIngestor().ingest(
                        epoch_manifest, records, supplied.artifact_paths,
                        stage / relative, stream_id,
                        initial_entities=[epoch_context_entity(manifest, ordinal)])
                    (stage / relative / "clock-index.json").write_text(
                        canonical_json({
                            "schema": "tardigrade/total-capture-epoch-clock-index/v1",
                            "stream_id": stream_id, "entries": clock_index,
                        }) + "\n", encoding="utf-8")
                    index_entries.append({
                        "ordinal": ordinal,
                        "process_epoch_id": descriptor["process_epoch_id"],
                        "map_epoch_id": descriptor["map_epoch_id"],
                        "capture_epoch_id": descriptor["capture_epoch_id"],
                        "status": "complete", "stream_id": stream_id,
                        "journal": str(relative),
                        "clock_index": str(relative / "clock-index.json"),
                        "salvage": None,
                    })
                    complete_epochs += 1
                else:
                    if (supplied.salvage is None or supplied.manifest is not None or
                            supplied.records is not None):
                        raise TotalCaptureContractError(
                            f"incomplete epoch {key!r} must be salvage-only")
                    salvage = validate_salvage_document(supplied.salvage)
                    if salvage["salvage_sha256"] != descriptor["salvage_sha256"]:
                        raise TotalCaptureContractError("epoch salvage hash mismatch")
                    _verify_salvage_artifacts(salvage, supplied.artifact_paths)
                    if (descriptor["native_demo_sha256"] is not None and
                            descriptor["native_demo_sha256"] not in
                            salvage["artifact_sha256s"].values()):
                        raise TotalCaptureContractError(
                            "incomplete epoch demo hash is absent from salvage")
                    relative = Path("epochs") / f"{ordinal:06d}-salvage.json"
                    (stage / relative).write_text(canonical_json(salvage) + "\n",
                                                  encoding="utf-8")
                    index_entries.append({
                        "ordinal": ordinal,
                        "process_epoch_id": descriptor["process_epoch_id"],
                        "map_epoch_id": descriptor["map_epoch_id"],
                        "capture_epoch_id": descriptor["capture_epoch_id"],
                        "status": descriptor["status"], "stream_id": None,
                        "journal": None, "clock_index": None,
                        "salvage": str(relative),
                    })
                    salvage_epochs += 1
            (stage / "collection-manifest.json").write_text(
                canonical_json(manifest) + "\n", encoding="utf-8")
            (stage / "collection-terminal.json").write_text(
                canonical_json(terminal) + "\n", encoding="utf-8")
            (stage / "collection-index.json").write_text(canonical_json({
                "schema": TOTAL_CAPTURE_COLLECTION_INDEX_SCHEMA,
                "render_schema": TOTAL_CAPTURE_COLLECTION_RENDER_SCHEMA,
                "collection_id": manifest["collection_id"],
                "collection_manifest_sha256": manifest["collection_manifest_sha256"],
                "terminal_sha256": terminal["terminal_sha256"],
                "corpus_status": manifest["coverage"]["corpus_status"],
                "continuity_status": manifest["coverage"]["continuity_status"],
                "epochs": index_entries,
            }) + "\n", encoding="utf-8")
            if final.exists():
                raise TotalCaptureContractError("collection output appeared during ingest")
            os.replace(stage, final)
            return TotalCaptureCollectionSummary(
                str(manifest["collection_id"]),
                str(manifest["coverage"]["corpus_status"]),
                str(manifest["coverage"]["continuity_status"]),
                complete_epochs, salvage_epochs,
                int(manifest["coverage"]["complete_target_active_ns"]), str(final))
        except Exception:
            shutil.rmtree(stage, ignore_errors=True)
            raise


__all__ = [
    "CollectionEpochInput", "MAX_COVERAGE_LEDGER_BYTES",
    "TOTAL_CAPTURE_COLLECTION_INDEX_SCHEMA",
    "TotalCaptureCollectionIngestor", "TotalCaptureCollectionSummary",
]
