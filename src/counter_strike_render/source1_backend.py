"""Configuration-selectable Source 1 reference rendering backend.

The public entry point, :func:`render_session`, deliberately resembles a
subprocess invocation so the render service can select it in place of the CS2
renderer.  This module is a deterministic CPU reference renderer, not a claim
of Source 1 GPU/material parity.  It rasterizes explicit triangles, VBSP world
faces, and MDL 48/49 geometry assembled from VVD v4 plus DX90.VTX v7 and
skinned with recorded final bone matrices.  Source animation evaluation,
materials, lightmaps, flexes, particles, and HUD reconstruction remain absent.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
import math
import os
import subprocess
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from .replay_bridge import (
    BridgeMode,
    ProvenancedValue,
    RenderBatch,
    RenderEntityRecord,
    RenderFrame,
    StateToRenderBridge,
)
from state_replay.integrator import ProtocolError, state_hash as canonical_state_hash
from state_replay.total_capture_contract import TotalCaptureContractError
from state_replay.total_capture_collection_contract import (
    TOTAL_CAPTURE_COLLECTION_RENDER_SCHEMA,
    validate_collection_render_envelope,
)
from state_replay.source1_resource_contract import (
    MODEL_BINDING_DERIVATION,
    MODEL_BINDING_PROVENANCE,
    MODEL_PRECACHE_ENTITY_ID,
    STRING_TABLE_RESOURCE_DERIVATION,
    STRING_TABLE_RESOURCE_PROVENANCE,
    Source1ResourceContractError,
    string_table_entry_document_sha256,
    validate_string_table_entry,
)

from .source1_formats import (
    Source1BSP,
    Source1FormatError,
    Source1MDL,
    Source1VPK,
    Source1VTX,
    Source1VVD,
)
from .source1_total_capture import (
    TotalCaptureEpochContext,
    TotalCaptureFrame,
    TotalCapturePov,
    inspect_epoch_context,
    inspect_total_capture,
    require_closed_horizon,
)


BACKEND_NAME = "source1-reference"
SCENE_SCHEMA = "tardigrade/source1-reference-scene/v1"
MANIFEST_SCHEMA = "tardigrade/source1-reference-render/v1"
class Source1RenderRefusal(RuntimeError):
    """A missing fact or unimplemented Source 1 subsystem forbids rendering."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


@dataclass(frozen=True)
class Source1RenderConfig:
    asset_root: Path | None = None
    map_path: Path | None = None
    width: int = 320
    height: int = 180
    mode: BridgeMode | str = BridgeMode.CANONICAL
    allow_synthetic_geometry: bool = False
    reference_fov_degrees: float | None = None
    vpk_paths: tuple[Path, ...] = ()
    map_asset: str | None = None
    fidelity_mode: str = "canonical-reference"
    expected_manifest_sha256: str | None = None
    expected_collection_manifest_sha256: str | None = None
    expected_build_id: str | None = None
    expected_build_sha256: str | None = None
    expected_content_manifest_sha256: str | None = None
    content_model_bindings: tuple[tuple[str, str], ...] = ()

    def normalized(self, *, cwd: str | os.PathLike[str] | None = None) -> "Source1RenderConfig":
        try:
            mode = BridgeMode(self.mode)
        except ValueError as exc:
            raise Source1RenderRefusal("invalid-mode", str(exc)) from exc
        if self.fidelity_mode not in {
            "canonical-reference", "total-capture-authoritative"
        }:
            raise Source1RenderRefusal(
                "invalid-fidelity-mode",
                "fidelity_mode must be canonical-reference or "
                "total-capture-authoritative",
            )
        expected_joins = (
            self.expected_build_id,
            self.expected_build_sha256,
            self.expected_content_manifest_sha256,
        )
        for label, digest in (
            ("expected_manifest_sha256", self.expected_manifest_sha256),
            (
                "expected_collection_manifest_sha256",
                self.expected_collection_manifest_sha256,
            ),
            ("expected_build_sha256", self.expected_build_sha256),
            (
                "expected_content_manifest_sha256",
                self.expected_content_manifest_sha256,
            ),
        ):
            if digest is not None and (
                len(digest) != 64
                or any(ch not in "0123456789abcdef" for ch in digest)
            ):
                raise Source1RenderRefusal(
                    "invalid-build-join", f"{label} must be lowercase SHA-256"
                )
        if self.expected_build_id is not None and not self.expected_build_id:
            raise Source1RenderRefusal(
                "invalid-build-join", "expected_build_id must be nonempty"
            )
        if self.fidelity_mode == "total-capture-authoritative":
            if mode is not BridgeMode.AUTHORITATIVE:
                raise Source1RenderRefusal(
                    "total-capture-mode-mismatch",
                    "total-capture-authoritative fidelity requires mode=authoritative",
                )
            if any(value is None for value in expected_joins):
                raise Source1RenderRefusal(
                    "missing-build-join",
                    "total-capture-authoritative fidelity requires expected manifest, "
                    "build id, build hash, and content-manifest hash",
                )
            if (
                self.expected_manifest_sha256 is None
                and self.expected_collection_manifest_sha256 is None
            ):
                raise Source1RenderRefusal(
                    "missing-build-join",
                    "total-capture-authoritative fidelity requires an expected "
                    "capture manifest or collection manifest hash",
                )
        normalized_bindings: list[tuple[str, str]] = []
        seen_content_hashes: set[str] = set()
        for content_hash, source_path in self.content_model_bindings:
            if (
                len(content_hash) != 64
                or any(ch not in "0123456789abcdef" for ch in content_hash)
                or content_hash in seen_content_hashes
            ):
                raise Source1RenderRefusal(
                    "invalid-content-model-binding",
                    "content model hashes must be unique lowercase SHA-256",
                )
            seen_content_hashes.add(content_hash)
            normalized_bindings.append(
                (content_hash, _source_asset_path(source_path, "content model path"))
            )
        if isinstance(self.width, bool) or not isinstance(self.width, int) or self.width < 1:
            raise Source1RenderRefusal("invalid-resolution", "width must be an integer > 0")
        if isinstance(self.height, bool) or not isinstance(self.height, int) or self.height < 1:
            raise Source1RenderRefusal("invalid-resolution", "height must be an integer > 0")
        if self.width * self.height > 16_777_216:
            raise Source1RenderRefusal("invalid-resolution", "reference frame exceeds 16M pixels")
        reference_fov = self.reference_fov_degrees
        if reference_fov is not None:
            reference_fov = _number(reference_fov, "reference_fov_degrees")
            if not 1.0 <= reference_fov < 179.0:
                raise Source1RenderRefusal(
                    "invalid-reference-fov",
                    "reference_fov_degrees must be in [1, 179)",
                )
            if mode is BridgeMode.AUTHORITATIVE:
                raise Source1RenderRefusal(
                    "reference-fov-forbidden",
                    "profile-derived FOV is forbidden in authoritative mode",
                )
        base = Path(cwd or os.getcwd()).resolve()
        root = _resolve_optional(self.asset_root, base)
        map_path = _resolve_optional(self.map_path, root or base)
        if root is not None and not root.is_dir():
            raise Source1RenderRefusal("missing-asset-root", f"asset root does not exist: {root}")
        if map_path is not None:
            if root is not None and not _is_beneath(map_path, root):
                raise Source1RenderRefusal(
                    "asset-escape", f"map path {map_path} escapes asset root {root}"
                )
            if not map_path.is_file():
                raise Source1RenderRefusal("missing-map", f"map does not exist: {map_path}")
            if map_path.suffix.casefold() != ".bsp":
                raise Source1RenderRefusal("wrong-map-format", "Source 1 map must end in .bsp")
        resolved_vpks: list[Path] = []
        for raw_vpk in self.vpk_paths:
            vpk = _resolve_optional(Path(raw_vpk), root or base)
            assert vpk is not None
            if not vpk.is_file():
                raise Source1RenderRefusal(
                    "missing-vpk", f"configured VPK directory does not exist: {vpk}"
                )
            if not vpk.name.casefold().endswith("_dir.vpk"):
                raise Source1RenderRefusal(
                    "invalid-vpk-path",
                    f"configured VPK must name an explicit _dir.vpk: {vpk}",
                )
            resolved_vpks.append(vpk)
        map_asset = (
            _source_asset_path(self.map_asset, "map_asset")
            if self.map_asset is not None
            else None
        )
        if map_asset is not None and not map_asset.casefold().endswith(".bsp"):
            raise Source1RenderRefusal(
                "wrong-map-format", "Source 1 map_asset must end in .bsp"
            )
        if map_path is not None and map_asset is not None:
            raise Source1RenderRefusal(
                "ambiguous-map", "configure either loose map_path or map_asset, not both"
            )
        return Source1RenderConfig(
            root,
            map_path,
            self.width,
            self.height,
            mode,
            bool(self.allow_synthetic_geometry),
            reference_fov,
            tuple(resolved_vpks),
            map_asset,
            self.fidelity_mode,
            self.expected_manifest_sha256,
            self.expected_collection_manifest_sha256,
            self.expected_build_id,
            self.expected_build_sha256,
            self.expected_content_manifest_sha256,
            tuple(normalized_bindings),
        )


@dataclass(frozen=True)
class Source1Camera:
    entity_id: str | int
    origin: tuple[float, float, float]
    yaw_degrees: float
    pitch_degrees: float
    fov_degrees: float
    fov_provenance: str
    vantage_id: str | None = None
    roll_degrees: float = 0.0
    projection: str = "perspective"
    near: float = 0.01
    far: float = 1_000_000.0
    viewport: tuple[int, int] | None = None
    view_matrix_4x4: tuple[float, ...] | None = None
    projection_matrix_4x4: tuple[float, ...] | None = None
    matrix_convention: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "origin": list(self.origin),
            "yaw_degrees": self.yaw_degrees,
            "pitch_degrees": self.pitch_degrees,
            "fov_degrees": self.fov_degrees,
            "fov_provenance": self.fov_provenance,
            "vantage_id": self.vantage_id,
            "roll_degrees": self.roll_degrees,
            "projection": self.projection,
            "near": self.near,
            "far": self.far,
            "viewport": list(self.viewport) if self.viewport is not None else None,
            "view_matrix_4x4": (
                list(self.view_matrix_4x4)
                if self.view_matrix_4x4 is not None else None
            ),
            "projection_matrix_4x4": (
                list(self.projection_matrix_4x4)
                if self.projection_matrix_4x4 is not None else None
            ),
            "matrix_convention": (
                dict(self.matrix_convention)
                if self.matrix_convention is not None else None
            ),
        }


@dataclass(frozen=True)
class Source1TriangleMesh:
    entity_id: str | int
    source: str
    vertices: tuple[tuple[float, float, float], ...]
    triangles: tuple[tuple[int, int, int], ...]
    color: tuple[int, int, int]
    space: str = "world"
    projection_fov_degrees: float | None = None
    near: float | None = None
    far: float | None = None
    projection_matrix_4x4: tuple[float, ...] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "source": self.source,
            "vertices": [list(vertex) for vertex in self.vertices],
            "triangles": [list(triangle) for triangle in self.triangles],
            "color": list(self.color),
            "space": self.space,
            "projection_fov_degrees": self.projection_fov_degrees,
            "near": self.near,
            "far": self.far,
            "projection_matrix_4x4": (
                list(self.projection_matrix_4x4)
                if self.projection_matrix_4x4 is not None else None
            ),
        }


@dataclass(frozen=True)
class Source1Scene:
    stream_id: str
    tick: int
    subtick: int
    state_hash: str
    mode: str
    camera: Source1Camera
    meshes: tuple[Source1TriangleMesh, ...]
    omissions: tuple[Mapping[str, Any], ...] = ()
    capture_manifest_sha256: str | None = None
    discontinuity: Mapping[str, Any] | None = None
    vantage_state_hash: str | None = None
    epoch_context: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCENE_SCHEMA,
            "backend": BACKEND_NAME,
            "stream_id": self.stream_id,
            "tick": self.tick,
            "subtick": self.subtick,
            "state_hash": self.state_hash,
            "mode": self.mode,
            "camera": self.camera.to_dict(),
            "meshes": [mesh.to_dict() for mesh in self.meshes],
            "omissions": [dict(item) for item in self.omissions],
            "capture_manifest_sha256": self.capture_manifest_sha256,
            "discontinuity": (
                dict(self.discontinuity) if self.discontinuity is not None else None
            ),
            "vantage_state_hash": self.vantage_state_hash,
            "epoch_context": (
                dict(self.epoch_context)
                if self.epoch_context is not None else None
            ),
        }


@dataclass(frozen=True)
class Source1RenderedFrame:
    scene: Source1Scene
    ppm: bytes
    depth_u64le: bytes
    color_sha256: str
    depth_sha256: str

    def manifest(self) -> dict[str, Any]:
        epoch = self.scene.epoch_context
        return {
            "state_hash": self.scene.state_hash,
            "tick": self.scene.tick,
            "subtick": self.scene.subtick,
            "vantage_id": self.scene.camera.vantage_id,
            "vantage_state_hash": self.scene.vantage_state_hash,
            "epoch_context": (
                dict(epoch) if epoch is not None else None
            ),
            "epoch_state_locator": (
                {
                    "process_epoch_id": epoch["process_epoch_id"],
                    "map_epoch_id": epoch["map_epoch_id"],
                    "capture_epoch_id": epoch["capture_epoch_id"],
                    "tick": self.scene.tick,
                    "subtick": self.scene.subtick,
                    "state_hash": self.scene.state_hash,
                    "action_locator_shape": [
                        "process_epoch_id", "map_epoch_id",
                        "capture_epoch_id", "action_id",
                    ],
                }
                if epoch is not None else None
            ),
            "color_sha256": self.color_sha256,
            "depth_sha256": self.depth_sha256,
            "depth_encoding": {
                "scalar": "uint64-le",
                "row_order": "top-to-bottom",
                "units": "source-units-times-1000000",
                "background": 18446744073709551615,
            },
            "depth_bytes": len(self.depth_u64le),
            "width": _ppm_dimensions(self.ppm)[0],
            "height": _ppm_dimensions(self.ppm)[1],
        }


@dataclass(frozen=True)
class Source1RenderedCollection:
    manifest: Mapping[str, Any]
    terminal: Mapping[str, Any]
    frames: tuple[Source1RenderedFrame, ...]
    salvage_epochs: tuple[Mapping[str, Any], ...]

    @property
    def is_continuous(self) -> bool:
        coverage = self.manifest["coverage"]
        return (
            coverage["continuity_status"] == "gap-free"
            and coverage["corpus_status"] == "target-met"
        )

    def summary(self) -> dict[str, Any]:
        return {
            "collection_id": self.manifest["collection_id"],
            "collection_manifest_sha256": self.manifest[
                "collection_manifest_sha256"
            ],
            "collection_terminal_sha256": self.terminal["terminal_sha256"],
            "close_reason": self.terminal["close_reason"],
            "corpus_status": self.manifest["coverage"]["corpus_status"],
            "continuity_status": self.manifest["coverage"][
                "continuity_status"
            ],
            "continuous": self.is_continuous,
            "target_complete": (
                self.manifest["coverage"]["corpus_status"] == "target-met"
            ),
            "rendered_complete_epochs": len({
                item.scene.epoch_context["ordinal"]
                for item in self.frames
                if item.scene.epoch_context is not None
            }),
            "salvage_epochs": [dict(item) for item in self.salvage_epochs],
            "presentation": (
                "continuous-collection"
                if self.is_continuous
                else (
                    "discrete-complete-epochs-target-incomplete"
                    if self.manifest["coverage"]["corpus_status"]
                    != "target-met"
                    else "discrete-complete-epochs-with-explicit-gaps"
                )
            ),
        }


def backend_descriptor() -> Mapping[str, Any]:
    """Stable registry record consumed by the engine-selection lane."""

    return MappingProxyType({
        "name": BACKEND_NAME,
        "engine": "source1",
        "game": "csgo-legacy",
        "entrypoint": "counter_strike_render.source1_backend:render_session",
        "input_schema": "tardigrade/state-snapshot/v1",
        "collection_input_schema": TOTAL_CAPTURE_COLLECTION_RENDER_SCHEMA,
        "scene_schema": SCENE_SCHEMA,
        "output_formats": ("ppm-p6", "depth-u64le", "json"),
        "fidelity": "deterministic-reference",
        "implemented": (
            "canonical-bridge",
            "state-hash-verification",
            "total-capture-closure-and-build-join",
            "isolated-process-map-capture-epoch-collections",
            "explicit-salvage-and-gap-accounting",
            "batched-scored-pov-raster",
            "final-pose-attachments-viewmodels-rigid-bodies",
            "anchor-validated-angular-target-history",
            "vbsp-world-faces",
            "explicit-vpk-v1/v2-assets-with-crc",
            "explicit-triangles",
            "mdl48/49-vvd4-vtx7-lod0-skinning",
            "flat-raster",
            "verifiable-depth-output",
        ),
        "refused": (
            "animation-evaluation",
            "flex-evaluation",
            "materials",
            "lightmaps",
            "particles",
            "hud",
            "nonempty-effects-and-diegetic-ui",
            "temporal-render-history-samples",
            "authoritative-engine-parity",
        ),
    })


class _Source1Assets:
    """Deterministic Source search path: loose root, then listed VPKs."""

    def __init__(self, config: Source1RenderConfig) -> None:
        self.root = config.asset_root
        self.vpks = tuple(_cached_vpk(path) for path in config.vpk_paths)

    def read(self, source_path: str) -> tuple[bytes, str]:
        normalized = _source_asset_path(source_path, "Source asset path")
        if self.root is not None:
            loose = (self.root / normalized).resolve()
            if not _is_beneath(loose, self.root):
                raise Source1RenderRefusal(
                    "asset-escape", f"asset path escapes root: {source_path!r}"
                )
            if loose.is_file():
                return loose.read_bytes(), f"loose:{loose}"
        for path, archive in zip(self._vpk_paths(), self.vpks):
            if normalized.casefold() in archive.entries:
                try:
                    return archive.read(normalized, verify_crc=True), f"vpk:{path}:{normalized}"
                except (OSError, Source1FormatError) as exc:
                    raise Source1RenderRefusal("invalid-vpk-entry", str(exc)) from exc
        searched = [str(self.root)] if self.root is not None else []
        searched.extend(str(path) for path in self._vpk_paths())
        raise Source1RenderRefusal(
            "missing-source1-asset",
            f"{normalized!r} was not found in configured Source search paths "
            f"{searched!r}",
        )

    def _vpk_paths(self) -> tuple[Path, ...]:
        return tuple(Path(archive.path) for archive in self.vpks)


@lru_cache(maxsize=16)
def _cached_vpk_versioned(path: str, mtime_ns: int, size: int) -> Source1VPK:
    del mtime_ns, size
    return Source1VPK(path)


def _cached_vpk(path: Path) -> Source1VPK:
    stat = path.stat()
    return _cached_vpk_versioned(str(path), stat.st_mtime_ns, stat.st_size)


def render_snapshot(
    snapshot: Mapping[str, Any], config: Source1RenderConfig
) -> Source1RenderedFrame:
    cfg = config.normalized()
    frame = StateToRenderBridge(cfg.mode).convert(snapshot)
    return render_frame(frame, cfg)


def render_batch(
    batch: RenderBatch | Iterable[Mapping[str, Any]], config: Source1RenderConfig
) -> tuple[Source1RenderedFrame, ...]:
    cfg = config.normalized()
    converted = batch if isinstance(batch, RenderBatch) else StateToRenderBridge(cfg.mode).batch(batch)
    streams = {frame.stream_id for frame in converted.frames}
    if len(streams) > 1:
        raise Source1RenderRefusal(
            "mixed-stream-batch",
            "one Source 1 render batch may contain exactly one stream_id; "
            "split streams before assigning tick/subtick output names",
        )
    positions = [(frame.tick, frame.subtick) for frame in converted.frames]
    if len(set(positions)) != len(positions):
        raise Source1RenderRefusal(
            "duplicate-frame-position",
            "a Source 1 render batch contains duplicate tick/subtick positions",
        )
    rendered: list[Source1RenderedFrame] = []
    for frame in converted.frames:
        rendered.extend(render_frame_vantages(frame, cfg))
    return tuple(rendered)


def render_collection(
    document: Mapping[str, Any], config: Source1RenderConfig
) -> Source1RenderedCollection:
    """Render complete epochs independently from a validated outer collection.

    Crashed and incomplete epochs contribute labelled salvage evidence only.
    Each complete epoch gets a fresh bridge cursor, so entity identity, state
    recurrence, and previous-hash checks can never cross an epoch boundary.
    """

    cfg = config.normalized()
    try:
        envelope = validate_collection_render_envelope(document)
    except TotalCaptureContractError as exc:
        raise Source1RenderRefusal("invalid-capture-collection", str(exc)) from exc
    manifest = envelope["manifest"]
    if cfg.expected_manifest_sha256 is not None:
        raise Source1RenderRefusal(
            "ambiguous-collection-build-join",
            "collection rendering uses expected_collection_manifest_sha256; "
            "a single-epoch expected_manifest_sha256 would be ambiguous",
        )
    expected_collection = cfg.expected_collection_manifest_sha256
    if (
        expected_collection is not None
        and manifest["collection_manifest_sha256"] != expected_collection
    ):
        raise Source1RenderRefusal(
            "collection-manifest-mismatch",
            f"captured {manifest['collection_manifest_sha256']!r}, configured "
            f"{expected_collection!r}",
        )
    if (
        cfg.fidelity_mode == "total-capture-authoritative"
        and expected_collection is None
    ):
        raise Source1RenderRefusal(
            "missing-collection-build-join",
            "authoritative collection rendering requires an expected collection "
            "manifest hash",
        )

    rendered: list[Source1RenderedFrame] = []
    salvage_epochs: list[Mapping[str, Any]] = []
    for entry, descriptor in zip(envelope["epochs"], manifest["epochs"]):
        if entry["status"] != "complete":
            salvage = entry["salvage"]
            salvage_epochs.append({
                "ordinal": entry["ordinal"],
                "process_epoch_id": entry["process_epoch_id"],
                "map_epoch_id": entry["map_epoch_id"],
                "capture_epoch_id": entry["capture_epoch_id"],
                "status": entry["status"],
                "salvage_sha256": salvage["salvage_sha256"],
                "reason": salvage["reason"],
                "presentation": "salvage-evidence-only-no-canonical-frames",
            })
            continue
        # Deliberately instantiate a new bridge for every capture/demo segment.
        # A reconnect or recording rotation may keep process+map IDs unchanged.
        bridge = StateToRenderBridge(cfg.mode)
        try:
            batch = bridge.batch(entry["frames"])
        except (ValueError, TypeError) as exc:
            raise Source1RenderRefusal(
                "invalid-capture-epoch", str(exc)
            ) from exc
        positions = [(frame.tick, frame.subtick) for frame in batch.frames]
        if len(set(positions)) != len(positions):
            raise Source1RenderRefusal(
                "duplicate-frame-position",
                f"capture epoch {entry['capture_epoch_id']!r} contains duplicate "
                "tick/subtick positions",
            )
        epoch_cfg = replace(
            cfg, expected_manifest_sha256=descriptor["capture_manifest_sha256"]
        )
        for frame in batch.frames:
            if frame.discontinuity is not None:
                raise Source1RenderRefusal(
                    "discontinuous-complete-epoch",
                    f"complete capture epoch {entry['capture_epoch_id']!r} "
                    "contains an internal state discontinuity",
                )
            rendered.extend(render_frame_vantages(
                frame, epoch_cfg, collection_manifest=manifest
            ))
    return Source1RenderedCollection(
        manifest, envelope["terminal"], tuple(rendered), tuple(salvage_epochs)
    )


def render_frame(frame: RenderFrame, config: Source1RenderConfig) -> Source1RenderedFrame:
    rendered = render_frame_vantages(frame, config)
    if len(rendered) != 1:
        raise Source1RenderRefusal(
            "multi-vantage-frame",
            f"frame contains {len(rendered)} scored vantages; use "
            "render_frame_vantages or render_batch",
        )
    return rendered[0]


def render_frame_vantages(
    frame: RenderFrame, config: Source1RenderConfig,
    *, collection_manifest: Mapping[str, Any] | None = None,
) -> tuple[Source1RenderedFrame, ...]:
    cfg = config.normalized()
    _verify_frame_hash(frame)
    if frame.mode is not BridgeMode(cfg.mode):
        raise Source1RenderRefusal(
            "mode-mismatch", f"frame mode {frame.mode.value!r} differs from backend mode {BridgeMode(cfg.mode).value!r}"
        )
    try:
        capture = inspect_total_capture(frame)
        epoch_context = inspect_epoch_context(frame, collection_manifest)
    except TotalCaptureContractError as exc:
        raise Source1RenderRefusal("invalid-total-capture", str(exc)) from exc
    if collection_manifest is not None and epoch_context is None:
        raise Source1RenderRefusal(
            "missing-epoch-context",
            "collection frame has no validated capture epoch context",
        )
    if epoch_context is not None:
        if capture is None:
            raise Source1RenderRefusal(
                "epoch-without-total-capture",
                "capture epoch context cannot decorate a non-total-capture frame",
            )
        if epoch_context.capture_manifest_sha256 != capture.manifest[
            "manifest_sha256"
        ]:
            raise Source1RenderRefusal(
                "epoch-capture-manifest-mismatch",
                "epoch context does not name the frame's total-capture manifest",
            )
        captured_map_name = capture.manifest["producer_closure"]["native_demo"][
            "map_name"
        ]
        if epoch_context.map_name != captured_map_name:
            raise Source1RenderRefusal(
                "epoch-map-mismatch",
                "collection epoch map_name differs from parsed native demo evidence",
            )
        if collection_manifest is not None:
            descriptor = collection_manifest["epochs"][epoch_context.ordinal]
            if descriptor["native_demo_sha256"] != capture.manifest["artifacts"][
                "demo_sha256"
            ]:
                raise Source1RenderRefusal(
                    "epoch-native-demo-mismatch",
                    "collection epoch native demo hash differs from total-capture "
                    "artifact join",
                )
        expected_epoch_collection = cfg.expected_collection_manifest_sha256
        if (
            expected_epoch_collection is not None
            and epoch_context.collection_manifest_sha256
            != expected_epoch_collection
        ):
            raise Source1RenderRefusal(
                "collection-manifest-mismatch",
                "epoch context collection hash differs from configured hash",
            )
        if cfg.fidelity_mode == "total-capture-authoritative":
            if expected_epoch_collection is None:
                raise Source1RenderRefusal(
                    "missing-collection-build-join",
                    "an authoritative frame carrying epoch context requires an "
                    "expected collection manifest hash",
                )
    if cfg.fidelity_mode == "total-capture-authoritative":
        if capture is None:
            raise Source1RenderRefusal(
                "missing-total-capture",
                "total-capture-authoritative rendering requires capture metadata",
            )
        try:
            require_closed_horizon(frame, capture)
        except TotalCaptureContractError as exc:
            raise Source1RenderRefusal("incomplete-total-capture", str(exc)) from exc
        _verify_total_capture_build_join(capture, cfg)
    elif frame.mode is BridgeMode.AUTHORITATIVE:
        raise Source1RenderRefusal(
            "authoritative-parity-unimplemented",
            "flat BSP/triangle rasterization cannot claim Source 1 material, lightmap, animation, particle, or HUD parity",
        )
    if capture is not None:
        povs: tuple[TotalCapturePov | None, ...] = tuple(capture.povs)
        camera_records = tuple(pov.record for pov in capture.povs)
    else:
        if not frame.cameras:
            raise Source1RenderRefusal("camera-cardinality", "frame has no camera")
        povs = (None,) * len(frame.cameras)
        camera_records = frame.cameras
    return tuple(
        _render_frame_for_vantage(
            frame, cfg, camera_record, capture, pov, epoch_context
        )
        for camera_record, pov in zip(camera_records, povs)
    )


def _render_frame_for_vantage(
    frame: RenderFrame,
    cfg: Source1RenderConfig,
    camera_record: RenderEntityRecord,
    capture: TotalCaptureFrame | None,
    pov: TotalCapturePov | None,
    epoch_context: TotalCaptureEpochContext | None,
) -> Source1RenderedFrame:
    camera, camera_omissions = _camera(frame, cfg, camera_record, pov)
    meshes: list[Source1TriangleMesh] = []
    omissions: list[Mapping[str, Any]] = list(camera_omissions)
    if epoch_context is not None:
        omissions.append({
            "code": "capture-epoch-reset-boundary",
            "ordinal": epoch_context.ordinal,
            "process_epoch_id": epoch_context.process_epoch_id,
            "map_epoch_id": epoch_context.map_epoch_id,
            "capture_epoch_id": epoch_context.capture_epoch_id,
            "map_name": epoch_context.map_name,
            "reset": dict(epoch_context.reset),
            "action_locator": (
                "(process_epoch_id,map_epoch_id,capture_epoch_id,action_id)"
            ),
            "detail": "state recurrence is scoped to this independently seeded epoch",
        })
    _consume_total_capture_faculties(frame, capture, pov, cfg, omissions)
    assets = _Source1Assets(cfg)
    if cfg.map_path is not None or cfg.map_asset is not None:
        try:
            if cfg.map_path is not None:
                map_data = cfg.map_path.read_bytes()
                map_label = f"loose:{cfg.map_path}"
            else:
                assert cfg.map_asset is not None
                map_data, map_label = assets.read(cfg.map_asset)
            world = Source1BSP(map_data, label=map_label).world_mesh()
        except (OSError, Source1FormatError) as exc:
            raise Source1RenderRefusal("invalid-bsp", str(exc)) from exc
        meshes.append(Source1TriangleMesh(
            "worldspawn", f"bsp:{map_label}", world.vertices, world.triangles, (92, 104, 112)
        ))
        if cfg.fidelity_mode == "total-capture-authoritative":
            raise Source1RenderRefusal(
                "world-materials-unimplemented",
                "the reference backend cannot claim authoritative BSP material/lightmap fidelity",
            )
        omissions.append({"code": "bsp-materials-unimplemented", "detail": "reference output flat-shades parsed world faces; VMT/VTF/lightmaps are not evaluated"})
    for record in frame.entities:
        if capture is not None and (
            record.entity_id == "total-capture:metadata"
            or record.entity_id == "total-capture:epoch-context"
            or str(record.entity_id).startswith("total-capture:pov:")
        ):
            continue
        viewmodel = record.components.get("viewmodel")
        if viewmodel is not None and pov is not None:
            viewmodel_value = _available_mapping(
                viewmodel, f"viewmodel entity {record.entity_id!r}"
            )
            if viewmodel_value.get("pov_id") != pov.pov_id:
                continue
        if capture is not None and pov is not None and not _visible_to_pov(record, pov):
            omissions.append({
                "code": "entity-not-visible-to-vantage",
                "entity_id": record.entity_id,
                "pov_id": pov.pov_id,
                "action": "entity-skipped",
            })
            continue
        geometry = record.components.get("source1_geometry")
        model = _model_component(record)
        animation = record.components.get("animation") or record.components.get(
            "final_pose"
        )
        visible = _visibility(record, omissions)
        if visible is False:
            continue
        if model is not None:
            if cfg.fidelity_mode == "total-capture-authoritative":
                raise Source1RenderRefusal(
                    "model-materials-unimplemented",
                    f"entity {record.entity_id!r} has captured material selection, "
                    "but the reference backend only flat-shades model triangles",
                )
            model_meshes = _model_meshes(
                record, model, animation, cfg, omissions, assets, frame, capture
            )
            meshes.extend(model_meshes)
            omissions.append({
                "code": "mdl-materials-unimplemented",
                "entity_id": record.entity_id,
                "detail": "MDL/VVD/VTX geometry and final-bone skinning are exact inputs; material meshes are deterministically flat-shaded",
            })
        if geometry is not None:
            if not cfg.allow_synthetic_geometry:
                raise Source1RenderRefusal(
                    "synthetic-geometry-disabled", "source1_geometry requires allow_synthetic_geometry=true"
                )
            meshes.append(_explicit_mesh(record, geometry))
        elif model is None and _is_visual_record(record):
            raise Source1RenderRefusal(
                "missing-visual-asset",
                f"visual entity {record.entity_id!r} generation {record.generation} "
                "has no source1_geometry or Source 1 .mdl reference; no "
                "placeholder or CS2 asset will be selected",
            )
    if not meshes:
        raise Source1RenderRefusal(
            "no-renderable-geometry", "snapshot and configuration provide no BSP world or explicit Source 1 geometry"
        )
    scene = Source1Scene(
        frame.stream_id, frame.tick, frame.subtick, frame.state_hash, frame.mode.value,
        camera, tuple(meshes), tuple(omissions),
        (capture.manifest["manifest_sha256"] if capture is not None else None),
        (frame.discontinuity.to_dict() if frame.discontinuity is not None else None),
        _vantage_state_hash(frame, pov),
        (epoch_context.to_dict() if epoch_context is not None else None),
    )
    ppm, depth_bytes = _rasterize(scene, cfg.width, cfg.height)
    return Source1RenderedFrame(
        scene,
        ppm,
        depth_bytes,
        hashlib.sha256(ppm).hexdigest(),
        hashlib.sha256(depth_bytes).hexdigest(),
    )


def _vantage_state_hash(frame: RenderFrame, pov: TotalCapturePov | None) -> str:
    if pov is None:
        return frame.state_hash
    document = {
        "schema": "tardigrade/source1-vantage-state/v1",
        "state_hash": frame.state_hash,
        "pov_id": pov.pov_id,
        "camera": dict(pov.camera),
        "visibility": dict(pov.visibility),
        "diegetic_ui": dict(pov.diegetic_ui),
        "render_history": dict(pov.render_history),
    }
    return hashlib.sha256(
        json.dumps(
            document, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()
def _verify_frame_hash(frame: RenderFrame) -> None:
    entities = [
        {
            "id": record.entity_id,
            "generation": record.generation,
            "class": record.entity_class,
            "components": {
                name: component.to_dict()
                for name, component in record.components.items()
            },
        }
        for record in frame.entities
    ]
    try:
        computed = canonical_state_hash(entities)
    except ProtocolError as exc:
        raise Source1RenderRefusal("unhashable-state", str(exc)) from exc
    if frame.state_hash != computed:
        raise Source1RenderRefusal(
            "state-hash-mismatch",
            f"snapshot declares {frame.state_hash!r}, canonical entities hash to "
            f"{computed!r}",
        )


def _camera(
    frame: RenderFrame,
    config: Source1RenderConfig,
    record: RenderEntityRecord | None = None,
    pov: TotalCapturePov | None = None,
) -> tuple[Source1Camera, tuple[Mapping[str, Any], ...]]:
    if record is None:
        if len(frame.cameras) != 1:
            raise Source1RenderRefusal(
                "camera-cardinality", f"reference backend requires exactly one camera, found {len(frame.cameras)}"
            )
        record = frame.cameras[0]
    if pov is not None:
        value = pov.camera
        if value.get("projection") != "perspective":
            raise Source1RenderRefusal(
                "unsupported-camera-projection",
                f"POV {pov.pov_id!r} projection must be perspective",
            )
        near = _number(value.get("near"), "camera.near")
        far = _number(value.get("far"), "camera.far")
        if near <= 0.0 or far <= near:
            raise Source1RenderRefusal(
                "invalid-camera", "captured camera requires 0 < near < far"
            )
        fov = _number(value.get("fov"), "camera.fov")
        if not 1.0 <= fov < 179.0:
            raise Source1RenderRefusal("invalid-camera", "camera.fov must be in [1, 179)")
        viewport_value = value.get("viewport")
        if (
            not isinstance(viewport_value, Sequence)
            or isinstance(viewport_value, (str, bytes, bytearray))
            or len(viewport_value) != 2
        ):
            raise Source1RenderRefusal("invalid-camera", "camera.viewport must have two values")
        viewport = tuple(
            _index_unbounded(item, "camera.viewport") for item in viewport_value
        )
        if viewport != (config.width, config.height):
            raise Source1RenderRefusal(
                "camera-viewport-mismatch",
                f"recorded viewport {viewport!r} differs from render output "
                f"{(config.width, config.height)!r}",
            )
        view_matrix = tuple(
            _number(item, "camera.view_matrix_4x4")
            for item in value["view_matrix_4x4"]
        )
        projection_matrix = tuple(
            _number(item, "camera.projection_matrix_4x4")
            for item in value["projection_matrix_4x4"]
        )
        convention = value["matrix_convention"]
        assert isinstance(convention, Mapping)
        viewed_origin = _camera_view_transform(
            view_matrix, (*_vec3(value.get("origin"), "camera.origin"), 1.0),
            convention,
        )
        if any(abs(item) > 1e-5 for item in viewed_origin[:3]):
            raise Source1RenderRefusal(
                "camera-view-matrix-mismatch",
                f"POV {pov.pov_id!r} view matrix does not map recorded origin to zero",
            )
        return (
            Source1Camera(
                record.entity_id,
                _vec3(value.get("origin"), "camera.origin"),
                _number(value.get("yaw_degrees"), "camera.yaw_degrees"),
                _number(value.get("pitch_degrees"), "camera.pitch_degrees"),
                fov,
                "recorded",
                pov.pov_id,
                _number(value.get("roll_degrees"), "camera.roll_degrees"),
                "perspective",
                near,
                far,
                viewport, view_matrix, projection_matrix, dict(convention),
            ),
            (),
        )
    component = record.components["camera"]
    value = _available_mapping(component, f"camera entity {record.entity_id!r}")
    origin_value = value.get("origin")
    if origin_value is not None:
        origin = _vec3(origin_value, "camera.origin")
    else:
        origin = tuple(_number(value.get(axis), f"camera.{axis}") for axis in ("x", "y", "z"))
        if "eye_z" in value:
            origin = (origin[0], origin[1], origin[2] + _number(value["eye_z"], "camera.eye_z"))
    yaw = _number(value.get("yaw_degrees", value.get("yaw")), "camera.yaw_degrees")
    pitch = _number(value.get("pitch_degrees", value.get("pitch")), "camera.pitch_degrees")
    omissions: list[Mapping[str, Any]] = []
    raw_fov = value.get("fov")
    unavailable_fov_reason: str | None = None
    if isinstance(raw_fov, Mapping) and "provenance" in raw_fov:
        nested_provenance = raw_fov.get("provenance")
        if nested_provenance == "unavailable":
            unavailable_fov_reason = str(
                raw_fov.get("reason") or "capture marked projection FOV unavailable"
            )
            raw_fov = None
        elif nested_provenance in ("recorded", "derived"):
            if "value" not in raw_fov:
                raise Source1RenderRefusal(
                    "invalid-camera-fov", "camera.fov provenance envelope has no value"
                )
            raw_fov, fov_provenance = raw_fov["value"], str(nested_provenance)
        else:
            raise Source1RenderRefusal(
                "invalid-camera-fov", "camera.fov has an invalid provenance envelope"
            )
    else:
        fov_provenance = component.provenance
    if raw_fov is not None:
        fov = _number(raw_fov, "camera.fov")
    elif config.reference_fov_degrees is not None:
        fov = config.reference_fov_degrees
        fov_provenance = "profile-derived"
        omissions.append({
            "code": "camera-fov-not-recorded",
            "entity_id": record.entity_id,
            "field": "camera.fov",
            "provenance": "profile-derived",
            "value": fov,
            "reason": unavailable_fov_reason,
            "detail": "democmdinfo did not record projection FOV; the explicit canonical render profile supplied reference_fov_degrees",
        })
    else:
        raise Source1RenderRefusal(
            "missing-camera-fov",
            "camera.fov is absent and no canonical reference_fov_degrees was configured",
        )
    if not 1.0 <= fov < 179.0:
        raise Source1RenderRefusal("invalid-camera", "camera.fov must be in [1, 179)")
    return (
        Source1Camera(
            record.entity_id, origin, yaw, pitch, fov, fov_provenance,
            str(record.entity_id),
        ),
        tuple(omissions),
    )


def _verify_total_capture_build_join(
    capture: TotalCaptureFrame, config: Source1RenderConfig
) -> None:
    manifest = capture.manifest
    engine = manifest["engine"]
    checks = (
        ("manifest_sha256", manifest["manifest_sha256"], config.expected_manifest_sha256),
        ("engine.build_id", engine["build_id"], config.expected_build_id),
        ("engine.build_sha256", engine["build_sha256"], config.expected_build_sha256),
        (
            "engine.content_manifest_sha256",
            engine["content_manifest_sha256"],
            config.expected_content_manifest_sha256,
        ),
    )
    mismatches = [
        f"{name}: captured {captured!r}, configured {expected!r}"
        for name, captured, expected in checks
        if captured != expected
    ]
    if mismatches:
        raise Source1RenderRefusal(
            "total-capture-build-mismatch", "; ".join(mismatches)
        )


def _visible_to_pov(record: RenderEntityRecord, pov: TotalCapturePov) -> bool:
    if record.entity_id == "total-capture:metadata" or str(record.entity_id).startswith(
        "total-capture:pov:"
    ):
        return True
    raw = pov.visibility.get("visible_entity_ids")
    if (
        isinstance(raw, (str, bytes, bytearray, Mapping))
        or not isinstance(raw, Sequence)
    ):
        raise Source1RenderRefusal(
            "invalid-vantage-visibility", "visibility.visible_entity_ids must be an array"
        )
    if len(set(map(str, raw))) != len(raw):
        raise Source1RenderRefusal(
            "invalid-vantage-visibility", "visibility entity IDs must be unique"
        )
    return record.entity_id in raw or str(record.entity_id) in {
        str(item) for item in raw
    }


def _consume_total_capture_faculties(
    frame: RenderFrame,
    capture: TotalCaptureFrame | None,
    pov: TotalCapturePov | None,
    config: Source1RenderConfig,
    omissions: list[Mapping[str, Any]],
) -> None:
    if capture is None:
        return
    if pov is None:
        raise Source1RenderRefusal("missing-vantage", "total capture has no scored POV")
    if capture.completeness.get("status") != "complete":
        omissions.append({
            "code": "total-capture-horizon-incomplete",
            "status": capture.completeness.get("status"),
            "action": "canonical-reference-only",
            "detail": (
                "capture stream closure is incomplete; authoritative rendering "
                "is forbidden"
            ),
        })
    _consume_angular_target_history(frame, pov, config, omissions)
    ui_elements = pov.diegetic_ui.get("elements")
    if not isinstance(ui_elements, Sequence) or isinstance(
        ui_elements, (str, bytes, bytearray)
    ):
        raise Source1RenderRefusal(
            "invalid-diegetic-ui", "diegetic_ui.elements must be an array"
        )
    if ui_elements:
        raise Source1RenderRefusal(
            "diegetic-ui-unimplemented",
            f"POV {pov.pov_id!r} declares diegetic UI elements the reference "
            "backend cannot rasterize",
        )
    history = pov.render_history
    samples = history.get("samples")
    if not isinstance(samples, Sequence) or isinstance(
        samples, (str, bytes, bytearray)
    ):
        raise Source1RenderRefusal(
            "invalid-render-history", "render_history.samples must be an array"
        )
    if samples:
        raise Source1RenderRefusal(
            "render-history-unimplemented",
            "captured temporal render-history samples cannot be reproduced by "
            "the static reference rasterizer",
        )
    omissions.append({
        "code": "reference-render-history",
        "pov_id": pov.pov_id,
        "sample_count": len(samples),
        "detail": "captured history was validated; the reference renderer has no temporal pass",
    })
    for record in frame.entities:
        world_transform = record.components.get("world_transform")
        if world_transform is not None:
            world_value = _available_mapping(
                world_transform, f"world_transform entity {record.entity_id!r}"
            )
            _flat_matrix_3x4(
                world_value.get("matrix_3x4"), "world_transform"
            )
            omissions.append({
                "code": "captured-world-transform-consumed",
                "entity_id": record.entity_id,
            })
        if "final_pose" in record.components and "model_binding" not in record.components:
            raise Source1RenderRefusal(
                "missing-pose-model-binding",
                f"entity {record.entity_id!r} has final_pose without model_binding",
            )
        effect = record.components.get("effects")
        if effect is not None:
            value = _available_mapping(effect, f"effects entity {record.entity_id!r}")
            instances = value.get("instances")
            if not isinstance(instances, Sequence) or isinstance(
                instances, (str, bytes, bytearray)
            ):
                raise Source1RenderRefusal(
                    "invalid-effects", "effects.instances must be an array"
                )
            if instances:
                raise Source1RenderRefusal(
                    "effects-unimplemented",
                    f"entity {record.entity_id!r} declares captured effects the "
                    "reference backend cannot rasterize",
                )
        ragdoll = record.components.get("ragdoll")
        if ragdoll is not None:
            value = _available_mapping(ragdoll, f"ragdoll entity {record.entity_id!r}")
            if value.get("active") is True and "final_pose" not in record.components:
                raise Source1RenderRefusal(
                    "missing-ragdoll-pose",
                    f"active ragdoll {record.entity_id!r} has no final_pose",
                )
        constraints = record.components.get("constraints")
        if constraints is not None:
            value = _available_mapping(
                constraints, f"constraints entity {record.entity_id!r}"
            )
            items = value.get("items")
            assert isinstance(items, Sequence)
            omissions.append({
                "code": "captured-constraints-consumed",
                "entity_id": record.entity_id,
                "count": len(items),
                "detail": "final rigid transforms are raster inputs; constraints remain recurrence state",
            })
    if frame.discontinuity is not None:
        omissions.append({
            "code": "render-history-discontinuity",
            "kind": frame.discontinuity.kind,
            "reason": frame.discontinuity.reason,
            "action": "canonical-history-reset",
        })


def _consume_angular_target_history(
    frame: RenderFrame,
    pov: TotalCapturePov,
    config: Source1RenderConfig,
    omissions: list[Mapping[str, Any]],
) -> None:
    history = pov.angular_target_history
    if history is None:
        if config.fidelity_mode == "total-capture-authoritative":
            raise Source1RenderRefusal(
                "missing-angular-target-history",
                f"POV {pov.pov_id!r} has no anchor-defined angular trajectory",
            )
        omissions.append({
            "code": "angular-target-history-absent",
            "pov_id": pov.pov_id,
        })
        return
    anchors = history["anchors"]
    camera_clock = pov.camera.get("clock")
    if not isinstance(camera_clock, Mapping) or camera_clock.get("engine_tick") != frame.tick:
        raise Source1RenderRefusal(
            "camera-clock-mismatch",
            f"POV {pov.pov_id!r} camera clock does not equal render tick",
        )
    requested_ns = camera_clock.get("monotonic_ns")
    candidates = [
        item for item in anchors
        if item["angle"]["kind"] == "render_camera"
        and item["clock"]["engine_tick"] == frame.tick
        and (requested_ns is None or item["clock"]["monotonic_ns"] == requested_ns)
    ]
    if len(candidates) != 1:
        raise Source1RenderRefusal(
            "ambiguous-render-camera-anchor",
            f"POV {pov.pov_id!r} requires exactly one render_camera anchor at "
            f"tick {frame.tick}, found {len(candidates)}",
        )
    anchor = candidates[0]
    captured = tuple(
        float(pov.camera[field])
        for field in ("yaw_degrees", "pitch_degrees", "roll_degrees")
    )
    anchored = tuple(
        float(anchor["angle"][field])
        for field in ("yaw_degrees", "pitch_degrees", "roll_degrees")
    )
    if captured != anchored:
        raise Source1RenderRefusal(
            "render-camera-anchor-mismatch",
            f"POV {pov.pov_id!r} camera angles do not equal its authoritative anchor",
        )
    omissions.append({
        "code": "angular-target-history-consumed",
        "pov_id": pov.pov_id,
        "focus_segment_id": history["focus_segment_id"],
        "anchor_monotonic_ns": anchor["clock"]["monotonic_ns"],
        "hid_interval_semantics": "relative_counts_are_interval_integrals",
        "camera_source": "authoritative-render-anchor",
    })


def _model_component(record: RenderEntityRecord) -> ProvenancedValue | None:
    return (
        record.components.get("source1_model")
        or record.components.get("model")
        or record.components.get("model_binding")
    )


def _model_meshes(
    record: RenderEntityRecord,
    component: ProvenancedValue,
    animation: ProvenancedValue | None,
    config: Source1RenderConfig,
    omissions: list[Mapping[str, Any]],
    assets: _Source1Assets,
    frame: RenderFrame,
    capture: TotalCaptureFrame | None,
) -> tuple[Source1TriangleMesh, ...]:
    total_binding = record.components.get("model_binding") is component
    if total_binding:
        if component.provenance != "recorded" or capture is None:
            raise Source1RenderRefusal(
                "invalid-content-model-binding",
                "total-capture model_binding must be recorded and joined to metadata",
            )
    elif component.provenance == MODEL_BINDING_PROVENANCE:
        if component.derivation != MODEL_BINDING_DERIVATION:
            raise Source1RenderRefusal(
                "untrusted-model-derivation",
                f"entity {record.entity_id!r} derived model binding must use "
                f"{MODEL_BINDING_DERIVATION!r}",
            )
    elif component.provenance != "recorded":
        raise Source1RenderRefusal(
            "unrecorded-model-binding",
            f"entity {record.entity_id!r} Source 1 model binding is unavailable",
        )
    value = _available_mapping(component, f"model entity {record.entity_id!r}")
    if total_binding:
        engine_name = capture.manifest["engine"]["name"]
        if value.get("engine") != engine_name:
            raise Source1RenderRefusal(
                "content-model-engine-mismatch",
                f"entity {record.entity_id!r} model engine does not equal capture engine",
            )
        model_hash = value.get("model_content_sha256")
        bindings = dict(config.content_model_bindings)
        raw_path = bindings.get(model_hash)
        if raw_path is None:
            raise Source1RenderRefusal(
                "missing-content-model-resolver",
                f"no explicit SHA-to-Source-path binding for captured model "
                f"{value.get('model_id')!r} ({model_hash})",
            )
    else:
        raw_path = value.get("source1_mdl")
    if component.provenance == MODEL_BINDING_PROVENANCE and not total_binding:
        _validate_model_join_metadata(value, record.entity_id, frame)
    if not isinstance(raw_path, str) or not raw_path.casefold().endswith(".mdl"):
        raise Source1RenderRefusal(
            "ambiguous-model-engine", f"entity {record.entity_id!r} must name source1_mdl ending in .mdl; CS2 VMDL substitution is forbidden"
        )
    raw_path = _source_asset_path(raw_path, "source1_mdl")
    raw_lod = value.get("lod")
    lod_policy_reason: str | None = None
    if raw_lod is None:
        lod = 0
        lod_policy_reason = "demo/entity state contains no render LOD"
    elif isinstance(raw_lod, Mapping) and "provenance" in raw_lod:
        if raw_lod.get("provenance") == "unavailable":
            lod = 0
            lod_policy_reason = str(
                raw_lod.get("reason") or "capture marked render LOD unavailable"
            )
        elif raw_lod.get("provenance") in ("recorded", "derived"):
            if "value" not in raw_lod:
                raise Source1RenderRefusal(
                    "invalid-model-lod", "model.lod provenance envelope has no value"
                )
            lod = raw_lod["value"]
        else:
            raise Source1RenderRefusal(
                "invalid-model-lod", "model.lod has an invalid provenance envelope"
            )
    else:
        lod = raw_lod
    if lod != 0 or isinstance(lod, bool):
        raise Source1RenderRefusal(
            "unsupported-model-lod",
            "source1_mdl rendering currently supports only lod=0",
        )
    if lod_policy_reason is not None:
        omissions.append({
            "code": "reference-model-lod-policy",
            "entity_id": record.entity_id,
            "field": "model.lod",
            "provenance": "renderer-policy",
            "value": 0,
            "reason": lod_policy_reason,
            "detail": "LOD is presentation state, not present in the demo; the deterministic reference renderer selected highest-detail LOD0",
        })
    bodygroups = value.get("bodygroups")
    if bodygroups is not None and (
        isinstance(bodygroups, (str, bytes, Mapping))
        or not isinstance(bodygroups, Sequence)
    ):
        raise Source1RenderRefusal(
            "invalid-bodygroups", "model.bodygroups must be an array of integers"
        )
    try:
        mdl_data, mdl_label = assets.read(raw_path)
        if total_binding and hashlib.sha256(mdl_data).hexdigest() != model_hash:
            raise Source1RenderRefusal(
                "content-model-hash-mismatch",
                f"resolved bytes for {value.get('model_id')!r} do not match "
                "model_content_sha256",
            )
        mdl = Source1MDL.parse(mdl_data, label=mdl_label)
        stem = raw_path[:-4]
        vvd_data, vvd_label = assets.read(stem + ".vvd")
        vtx_data, vtx_label = assets.read(stem + ".dx90.vtx")
        vvd = Source1VVD.parse(
            vvd_data,
            expected_checksum=mdl.checksum,
            root_lod=0,
            label=vvd_label,
        )
        topology = Source1VTX(
            vtx_data, expected_checksum=mdl.checksum, label=vtx_label
        ).assemble_lod0(mdl, vvd, bodygroups=bodygroups)
    except (OSError, Source1FormatError) as exc:
        raise Source1RenderRefusal("invalid-model", str(exc)) from exc
    rigid_component = record.components.get("rigid_body")
    rigid_matrix: tuple[tuple[float, float, float, float], ...] | None = None
    if animation is None and rigid_component is not None:
        if rigid_component.provenance != "recorded":
            raise Source1RenderRefusal(
                "unrecorded-rigid-body",
                f"entity {record.entity_id!r} rigid_body must be recorded",
            )
        if len(mdl.bones) > 1:
            raise Source1RenderRefusal(
                "rigid-model-needs-final-pose",
                f"rigid entity {record.entity_id!r} has {len(mdl.bones)} bones; "
                "a final_pose is required",
            )
        rigid_value = _available_mapping(
            rigid_component, f"rigid_body entity {record.entity_id!r}"
        )
        rigid_matrix = _flat_matrix_3x4(
            rigid_value.get("transform_3x4"), "rigid_body"
        )
        omissions.append({
            "code": "captured-rigid-body-consumed",
            "entity_id": record.entity_id,
            "body_id": rigid_value.get("body_id"),
            "sleeping": rigid_value.get("sleeping"),
        })
    elif animation is None:
        raise Source1RenderRefusal(
            "missing-animation-state", f"skinned entity {record.entity_id!r} has no animation component"
        )
    if animation is not None and animation.provenance != "recorded":
        raise Source1RenderRefusal(
            "unrecorded-animation-state",
            f"entity {record.entity_id!r} final bone matrices must be recorded",
        )
    animation_value = (
        _available_mapping(animation, f"animation entity {record.entity_id!r}")
        if animation is not None
        else {}
    )
    total_pose = animation is not None and record.components.get("final_pose") is animation
    if rigid_matrix is not None:
        bones = ()
        pose_space = "world"
    elif total_pose:
        bones = animation_value.get("bones")
        pose_space = animation_value.get("space")
        expected_space = "view" if "viewmodel" in record.components else "world"
        if pose_space != expected_space:
            raise Source1RenderRefusal(
                "final-pose-space-mismatch",
                f"entity {record.entity_id!r} final_pose.space must be "
                f"{expected_space!r}",
            )
        if animation_value.get("model_content_sha256") != hashlib.sha256(
            mdl_data
        ).hexdigest():
            raise Source1RenderRefusal(
                "pose-model-content-mismatch",
                f"entity {record.entity_id!r} pose does not name the resolved model bytes",
            )
    else:
        bones = animation_value.get("evaluated_bones")
        pose_space = "world"
    if rigid_matrix is None and bones is None:
        raise Source1RenderRefusal(
            "unevaluated-animation-state",
            f"entity {record.entity_id!r} records no final evaluated bones; "
            "sequence/cycle/pose/layers alone are not substituted",
        )
    if rigid_matrix is None and (
        isinstance(bones, (str, bytes, Mapping)) or not isinstance(bones, Sequence)
    ):
        raise Source1RenderRefusal("invalid-animation-state", "evaluated_bones must be an array")
    if rigid_matrix is None and len(bones) != len(mdl.bones):
        raise Source1RenderRefusal(
            "bone-count-mismatch", f"recorded {len(bones)} evaluated bones but {raw_path!r} declares {len(mdl.bones)}"
        )
    if rigid_matrix is None and not total_pose and animation_value.get("model_checksum") != mdl.checksum:
        raise Source1RenderRefusal(
            "animation-model-mismatch",
            f"animation.model_checksum must equal MDL checksum {mdl.checksum}",
        )
    if rigid_matrix is None and not total_pose and (
        animation_value.get("matrix_kind")
        != "source1_bone_to_world_skinning_3x4"
    ):
        raise Source1RenderRefusal(
            "ambiguous-bone-matrices",
            "animation.matrix_kind must be "
            "'source1_bone_to_world_skinning_3x4'; local/model-space or "
            "unskinned bone poses are not interchangeable",
        )
    if rigid_matrix is not None:
        matrices = ()
    elif total_pose:
        matrices_list = []
        for index, item in enumerate(bones):
            if not isinstance(item, Mapping):
                raise Source1RenderRefusal(
                    "invalid-final-pose", f"final_pose.bones[{index}] must be an object"
                )
            if item.get("name") != mdl.bones[index].name:
                raise Source1RenderRefusal(
                    "bone-name-mismatch",
                    f"final_pose.bones[{index}].name must equal "
                    f"{mdl.bones[index].name!r}",
                )
            if item.get("parent") != mdl.bones[index].parent:
                raise Source1RenderRefusal(
                    "bone-parent-mismatch",
                    f"final_pose.bones[{index}].parent does not match MDL hierarchy",
                )
            matrices_list.append(
                _flat_matrix_3x4(item.get("matrix_3x4"), f"final_pose.bones[{index}]")
            )
        matrices = tuple(matrices_list)
        attachments = animation_value.get("attachments")
        assert isinstance(attachments, Sequence)
        bone_names = {bone.name for bone in mdl.bones}
        for index, attachment in enumerate(attachments):
            assert isinstance(attachment, Mapping)
            if attachment.get("bone") not in bone_names:
                raise Source1RenderRefusal(
                    "attachment-bone-mismatch",
                    f"final_pose.attachments[{index}] names an absent bone",
                )
            _flat_matrix_3x4(
                attachment.get("matrix_3x4"), f"final_pose.attachments[{index}]"
            )
        omissions.append({
            "code": "captured-attachments-consumed",
            "entity_id": record.entity_id,
            "space": pose_space,
            "count": len(attachments),
            "pose_sequence": animation_value.get("pose_sequence"),
        })
    else:
        matrices = tuple(
            _bone_matrix(item, mdl.bones[index].name, index)
            for index, item in enumerate(bones)
        )
    skinned_cache: dict[int, tuple[float, float, float]] = {}

    def skin(global_index: int) -> tuple[float, float, float]:
        cached = skinned_cache.get(global_index)
        if cached is not None:
            return cached
        vertex = vvd.vertices[global_index]
        if rigid_matrix is not None:
            x, y, z = vertex.position
            final = tuple(
                rigid_matrix[row][0] * x
                + rigid_matrix[row][1] * y
                + rigid_matrix[row][2] * z
                + rigid_matrix[row][3]
                for row in range(3)
            )
            skinned_cache[global_index] = final
            return final  # type: ignore[return-value]
        result = [0.0, 0.0, 0.0]
        for weight, bone_id in zip(vertex.weights, vertex.bones):
            if bone_id >= len(matrices):
                raise Source1RenderRefusal(
                    "vertex-bone-out-of-range",
                    f"VVD vertex {global_index} names bone {bone_id}, but MDL "
                    f"has {len(matrices)} bones",
                )
            matrix = matrices[bone_id]
            x, y, z = vertex.position
            for row in range(3):
                result[row] += weight * (
                    matrix[row][0] * x
                    + matrix[row][1] * y
                    + matrix[row][2] * z
                    + matrix[row][3]
                )
        final = (result[0], result[1], result[2])
        if not all(math.isfinite(item) for item in final):
            raise Source1RenderRefusal(
                "nonfinite-skinned-vertex",
                f"skinning VVD vertex {global_index} produced a non-finite point",
            )
        skinned_cache[global_index] = final
        return final

    meshes: list[Source1TriangleMesh] = []
    viewmodel_value = (
        _available_mapping(
            record.components["viewmodel"], f"viewmodel entity {record.entity_id!r}"
        )
        if "viewmodel" in record.components
        else None
    )
    for item in topology:
        vertices = tuple(skin(index) for index in item.vertex_indices)
        digest = hashlib.sha256(
            f"{mdl.checksum}:{item.material}".encode("ascii")
        ).digest()
        color = tuple(64 + byte % 160 for byte in digest[:3])
        meshes.append(
            Source1TriangleMesh(
                record.entity_id,
                f"mdl:{raw_path}:bodypart={item.body_part}:model={item.model}:"
                f"mesh={item.mesh}:material={item.material}",
                vertices,
                item.triangles,
                color,  # type: ignore[arg-type]
                str(pose_space),
                (
                    _number(viewmodel_value.get("viewmodel_fov"), "viewmodel.viewmodel_fov")
                    if viewmodel_value is not None
                    else None
                ),
                (
                    _number(viewmodel_value.get("near"), "viewmodel.near")
                    if viewmodel_value is not None
                    else None
                ),
                (
                    _number(viewmodel_value.get("far"), "viewmodel.far")
                    if viewmodel_value is not None
                    else None
                ),
                (
                    tuple(
                        _number(item, "viewmodel.projection_matrix_4x4")
                        for item in viewmodel_value["projection_matrix_4x4"]
                    )
                    if viewmodel_value is not None
                    else None
                ),
            )
        )
    if not meshes or not any(mesh.triangles for mesh in meshes):
        raise Source1RenderRefusal(
            "empty-model-topology", f"{raw_path!r} LOD0 contains no triangles"
        )
    return tuple(meshes)


def _flat_matrix_3x4(
    value: Any, label: str
) -> tuple[tuple[float, float, float, float], ...]:
    if (
        isinstance(value, (str, bytes, bytearray, Mapping))
        or not isinstance(value, Sequence)
        or len(value) != 12
    ):
        raise Source1RenderRefusal(
            "invalid-final-pose", f"{label}.matrix_3x4 must contain 12 numbers"
        )
    numbers = tuple(_number(item, f"{label}.matrix_3x4") for item in value)
    return (numbers[0:4], numbers[4:8], numbers[8:12])


def _bone_matrix(
    value: Any, expected_name: str, index: int
) -> tuple[tuple[float, float, float, float], ...]:
    if not isinstance(value, Mapping):
        raise Source1RenderRefusal(
            "invalid-bone-matrix",
            f"evaluated_bones[{index}] must be an object with name and matrix3x4",
        )
    if value.get("name") != expected_name:
        raise Source1RenderRefusal(
            "bone-name-mismatch",
            f"evaluated_bones[{index}].name must equal MDL bone {expected_name!r}",
        )
    matrix = value.get("matrix3x4")
    if (
        isinstance(matrix, (str, bytes, Mapping))
        or not isinstance(matrix, Sequence)
        or len(matrix) != 3
    ):
        raise Source1RenderRefusal(
            "invalid-bone-matrix",
            f"evaluated_bones[{index}].matrix3x4 must contain three rows",
        )
    rows: list[tuple[float, float, float, float]] = []
    for row_no, raw_row in enumerate(matrix):
        if (
            isinstance(raw_row, (str, bytes, Mapping))
            or not isinstance(raw_row, Sequence)
            or len(raw_row) != 4
        ):
            raise Source1RenderRefusal(
                "invalid-bone-matrix",
                f"evaluated_bones[{index}].matrix3x4[{row_no}] must contain four numbers",
            )
        rows.append(
            tuple(
                _number(item, f"evaluated_bones[{index}].matrix3x4[{row_no}]")
                for item in raw_row
            )  # type: ignore[arg-type]
        )
    return tuple(rows)


def _validate_model_join_metadata(
    value: Mapping[str, Any], entity_id: str | int, frame: RenderFrame
) -> None:
    model_index = value.get("model_index")
    revision = value.get("source_table_revision")
    entry_hash = value.get("source_entry_sha256")
    if (
        isinstance(model_index, bool)
        or not isinstance(model_index, int)
        or model_index < 0
    ):
        raise Source1RenderRefusal(
            "invalid-model-join",
            f"entity {entity_id!r} derived model_index must be a nonnegative integer",
        )
    if value.get("source_table_entity_id") != MODEL_PRECACHE_ENTITY_ID:
        raise Source1RenderRefusal(
            "invalid-model-join",
            f"entity {entity_id!r} derived model binding does not identify the "
            "canonical modelprecache table",
        )
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
        raise Source1RenderRefusal(
            "invalid-model-join",
            f"entity {entity_id!r} source_table_revision must be nonnegative",
        )
    if (
        not isinstance(entry_hash, str)
        or len(entry_hash) != 64
        or any(character not in "0123456789abcdef" for character in entry_hash)
    ):
        raise Source1RenderRefusal(
            "invalid-model-join",
            f"entity {entity_id!r} source_entry_sha256 must be lowercase SHA-256",
        )

    table_entity_id = value["source_table_entity_id"]
    table_record = next(
        (record for record in frame.entities if record.entity_id == table_entity_id),
        None,
    )
    if table_record is None:
        raise Source1RenderRefusal(
            "model-join-mismatch",
            f"entity {entity_id!r} derived model binding references absent "
            f"canonical entity {table_entity_id!r}",
        )
    if table_record.entity_class != "Source1StringTable":
        raise Source1RenderRefusal(
            "model-join-mismatch",
            f"entity {table_entity_id!r} is not a Source1StringTable",
        )
    table_component = table_record.components.get("source1_string_table")
    if (
        table_component is None
        or table_component.provenance != STRING_TABLE_RESOURCE_PROVENANCE
        or table_component.derivation != STRING_TABLE_RESOURCE_DERIVATION
    ):
        raise Source1RenderRefusal(
            "model-join-mismatch",
            f"entity {table_entity_id!r} is not the exact trusted canonical "
            "Source 1 string-table materialization",
        )
    table = _available_mapping(
        table_component, f"string table entity {table_entity_id!r}"
    )
    if table.get("name") != "modelprecache" or table.get("revision") != revision:
        raise Source1RenderRefusal(
            "model-join-mismatch",
            f"entity {entity_id!r} derived model binding revision does not "
            "match the recorded modelprecache table",
        )
    field_origins = table.get("field_origins")
    expected_origins = {
        "revision": {
            "provenance": "derived",
            "derivation": "monotonic reducer revision for hashed content",
        },
        "name": {
            "provenance": "recorded",
            "source": "Source 1 table name retained across ordered history",
        },
        "server_entries": {
            "provenance": "derived",
            "derivation": (
                "accumulated server map from recorded create/update/snapshot operations"
            ),
            "entry_content_provenance": "recorded",
        },
    }
    if not isinstance(field_origins, Mapping) or any(
        field_origins.get(name) != origin
        for name, origin in expected_origins.items()
    ):
        raise Source1RenderRefusal(
            "model-join-mismatch",
            f"entity {table_entity_id!r} lacks the exact field provenance "
            "required to authenticate a modelprecache join",
        )
    entries = table.get("server_entries")
    if (
        isinstance(entries, (str, bytes, bytearray, Mapping))
        or not isinstance(entries, Sequence)
    ):
        raise Source1RenderRefusal(
            "model-join-mismatch",
            "recorded modelprecache server_entries must be an array",
        )
    matches = [
        entry
        for entry in entries
        if isinstance(entry, Mapping) and entry.get("index") == model_index
    ]
    if len(matches) != 1:
        raise Source1RenderRefusal(
            "model-join-mismatch",
            f"entity {entity_id!r} model_index {model_index} resolves to "
            f"{len(matches)} recorded server entries",
        )
    entry = matches[0]
    try:
        validated_index, validated_string, _ = validate_string_table_entry(entry)
        computed_entry_hash = string_table_entry_document_sha256(entry)
    except Source1ResourceContractError as exc:
        raise Source1RenderRefusal(
            "model-join-mismatch",
            f"modelprecache entry {model_index} violates the shared canonical "
            f"resource contract: {exc}",
        ) from exc
    if validated_index != model_index:
        raise Source1RenderRefusal(
            "model-join-mismatch",
            f"modelprecache entry key {validated_index} does not equal model_index "
            f"{model_index}",
        )
    source1_mdl = value.get("source1_mdl")
    if validated_string != source1_mdl:
        raise Source1RenderRefusal(
            "model-join-mismatch",
            f"entity {entity_id!r} source1_mdl does not equal recorded "
            f"modelprecache entry {model_index}",
        )
    if computed_entry_hash != entry_hash:
        raise Source1RenderRefusal(
            "model-join-mismatch",
            f"entity {entity_id!r} source_entry_sha256 does not authenticate "
            f"recorded modelprecache entry {model_index}",
        )


def _visibility(
    record: RenderEntityRecord, omissions: list[Mapping[str, Any]]
) -> bool | None:
    component = record.components.get("visibility")
    if component is None:
        return None
    if component.provenance == "unavailable":
        omissions.append({
            "code": "visibility-unavailable",
            "entity_id": record.entity_id,
            "provenance": "unavailable",
            "detail": component.reason,
            "action": "entity-skipped",
        })
        return False
    value = _available_mapping(component, f"visibility entity {record.entity_id!r}")
    in_pvs = value.get("in_pvs")
    if not isinstance(in_pvs, bool):
        raise Source1RenderRefusal(
            "invalid-visibility",
            f"entity {record.entity_id!r} visibility.in_pvs must be boolean",
        )
    if not in_pvs:
        omissions.append({
            "code": "entity-outside-pvs",
            "entity_id": record.entity_id,
            "provenance": component.provenance,
            "action": "entity-skipped",
        })
    elif component.provenance != "recorded":
        omissions.append({
            "code": "derived-visibility",
            "entity_id": record.entity_id,
            "provenance": component.provenance,
            "derivation": component.derivation,
        })
    return in_pvs


def _explicit_mesh(record: RenderEntityRecord, component: ProvenancedValue) -> Source1TriangleMesh:
    value = _available_mapping(component, f"source1_geometry entity {record.entity_id!r}")
    space = value.get("space")
    if space not in {"world", "local"}:
        raise Source1RenderRefusal(
            "ambiguous-geometry-space",
            "source1_geometry.space must be world or local",
        )
    raw_vertices = value.get("vertices")
    raw_triangles = value.get("triangles")
    if isinstance(raw_vertices, (str, bytes, Mapping)) or not isinstance(raw_vertices, Sequence):
        raise Source1RenderRefusal("invalid-geometry", "vertices must be an array")
    if isinstance(raw_triangles, (str, bytes, Mapping)) or not isinstance(raw_triangles, Sequence):
        raise Source1RenderRefusal("invalid-geometry", "triangles must be an array")
    vertices = tuple(_vec3(vertex, "source1_geometry.vertices[]") for vertex in raw_vertices)
    if space == "local":
        rigid = record.components.get("rigid_body")
        if rigid is None or rigid.provenance != "recorded":
            raise Source1RenderRefusal(
                "missing-rigid-transform",
                "local source1_geometry requires a recorded rigid_body transform",
            )
        rigid_value = _available_mapping(rigid, f"rigid_body entity {record.entity_id!r}")
        matrix = _flat_matrix_3x4(rigid_value.get("transform_3x4"), "rigid_body")
        vertices = tuple(
            tuple(
                matrix[row][0] * vertex[0]
                + matrix[row][1] * vertex[1]
                + matrix[row][2] * vertex[2]
                + matrix[row][3]
                for row in range(3)
            )
            for vertex in vertices
        )  # type: ignore[assignment]
    triangles: list[tuple[int, int, int]] = []
    for raw in raw_triangles:
        if isinstance(raw, (str, bytes, Mapping)) or not isinstance(raw, Sequence) or len(raw) != 3:
            raise Source1RenderRefusal("invalid-geometry", "every triangle must contain three indices")
        tri = tuple(_index(index, len(vertices)) for index in raw)
        if len(set(tri)) != 3:
            raise Source1RenderRefusal("invalid-geometry", "degenerate triangle indices are refused")
        triangles.append(tri)
    if not triangles:
        raise Source1RenderRefusal("invalid-geometry", "mesh has no triangles")
    raw_color = value.get("color", (196, 128, 48))
    if isinstance(raw_color, (str, bytes, Mapping)) or not isinstance(raw_color, Sequence) or len(raw_color) != 3:
        raise Source1RenderRefusal("invalid-geometry", "color must contain three byte values")
    color = tuple(_byte(item, "source1_geometry.color") for item in raw_color)
    return Source1TriangleMesh(record.entity_id, "canonical:source1_geometry", vertices, tuple(triangles), color)


def _is_visual_record(record: RenderEntityRecord) -> bool:
    # Animation telemetry alone does not assert that an entity has renderable
    # geometry.  Asset-bearing/player/physics/render components do.  The
    # Source 1 adapter names its canonical player subset ``player_state``;
    # additionally, a decoded nonzero m_nModelIndex is direct evidence that
    # the edict expected a model even when the resource join is unavailable.
    if {
        "player",
        "player_state",
        "dynamic_physics",
        "rigid_body",
        "viewmodel",
        "ragdoll",
        "render",
    } & set(record.components):
        return True
    netprops = record.components.get("source1_netprops")
    if netprops is None or netprops.provenance == "unavailable":
        return False
    if not isinstance(netprops.value, Mapping):
        return False
    return any(
        key.endswith("m_nModelIndex")
        and isinstance(value, int)
        and not isinstance(value, bool)
        and value > 0
        for key, value in netprops.value.items()
    )


def _available_mapping(component: ProvenancedValue, label: str) -> Mapping[str, Any]:
    if component.provenance == "unavailable":
        raise Source1RenderRefusal("unavailable-state", f"{label} is unavailable: {component.reason}")
    if not isinstance(component.value, Mapping):
        raise Source1RenderRefusal("invalid-component", f"{label} value must be an object")
    return component.value


def _rasterize(scene: Source1Scene, width: int, height: int) -> tuple[bytes, bytes]:
    background = (18, 20, 24)
    pixels = bytearray(background * (width * height))
    infinity = (1 << 63) - 1
    depth = [infinity] * (width * height)
    camera = scene.camera
    yaw = math.radians(camera.yaw_degrees)
    pitch = math.radians(camera.pitch_degrees)
    forward = (math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), -math.sin(pitch))
    right = (-math.sin(yaw), math.cos(yaw), 0.0)
    # Source coordinates are X-forward at zero yaw and Z-up.  With ``right``
    # pointing +Y, forward x right is therefore the camera's +Z axis.
    up = _cross(forward, right)
    roll = math.radians(camera.roll_degrees)
    if roll:
        base_right, base_up = right, up
        right = tuple(
            math.cos(roll) * base_right[index] + math.sin(roll) * base_up[index]
            for index in range(3)
        )
        up = tuple(
            -math.sin(roll) * base_right[index] + math.cos(roll) * base_up[index]
            for index in range(3)
        )
    def project(
        vertex: tuple[float, float, float], mesh: Source1TriangleMesh
    ) -> tuple[float, float, float] | None:
        convention = camera.matrix_convention
        matrix_projection = (
            mesh.projection_matrix_4x4
            if mesh.space == "view" and mesh.projection_matrix_4x4 is not None
            else camera.projection_matrix_4x4
        )
        if mesh.space == "view":
            viewed = (*vertex, 1.0)
        elif camera.view_matrix_4x4 is not None and convention is not None:
            viewed = _camera_view_transform(
                camera.view_matrix_4x4, (*vertex, 1.0), convention
            )
        else:
            rel = tuple(vertex[i] - camera.origin[i] for i in range(3))
            z = _dot(rel, forward)
            lateral, vertical = _dot(rel, right), _dot(rel, up)
            viewed = None
        if viewed is not None and matrix_projection is not None and convention is not None:
            axes = convention["view_axis_order"]
            forward_index = list(axes).index("forward")
            z = float(viewed[forward_index])
            if convention["view_handedness"] == "right-handed":
                z = -z
            clip = _matrix4_transform(
                matrix_projection, viewed, convention
            )
            if abs(clip[3]) < 1e-12:
                return None
            ndc_x, ndc_y, ndc_z = (
                clip[0] / clip[3], clip[1] / clip[3], clip[2] / clip[3]
            )
            low_z = (
                0.0 if convention["clip_depth_range"] == "zero-to-one" else -1.0
            )
            if not low_z - 1e-6 <= ndc_z <= 1.0 + 1e-6:
                return None
            x = (ndc_x + 1.0) * width / 2.0
            y = (
                (1.0 - ndc_y) * height / 2.0
                if convention["ndc_y_direction"] == "up"
                else (ndc_y + 1.0) * height / 2.0
            )
            near = mesh.near if mesh.near is not None else camera.near
            far = mesh.far if mesh.far is not None else camera.far
            if z < near or z > far:
                return None
            return x, y, z
        near = mesh.near if mesh.near is not None else camera.near
        far = mesh.far if mesh.far is not None else camera.far
        if z < near or z > far:
            return None
        fov = (
            mesh.projection_fov_degrees
            if mesh.projection_fov_degrees is not None
            else camera.fov_degrees
        )
        scale = (width / 2.0) / math.tan(math.radians(fov) / 2.0)
        x = width / 2.0 + lateral * scale / z
        y = height / 2.0 - vertical * scale / z
        return x, y, z

    for mesh in scene.meshes:
        for triangle in mesh.triangles:
            projected = tuple(
                project(mesh.vertices[index], mesh) for index in triangle
            )
            if any(item is None for item in projected):
                continue
            a, b, c = projected  # type: ignore[misc]
            area = _edge(a, b, c[0], c[1])
            if abs(area) < 1e-12:
                continue
            min_x = max(0, int(math.floor(min(a[0], b[0], c[0]))))
            max_x = min(width - 1, int(math.ceil(max(a[0], b[0], c[0]))))
            min_y = max(0, int(math.floor(min(a[1], b[1], c[1]))))
            max_y = min(height - 1, int(math.ceil(max(a[1], b[1], c[1]))))
            if min_x > max_x or min_y > max_y:
                continue
            for py in range(min_y, max_y + 1):
                sy = py + 0.5
                for px in range(min_x, max_x + 1):
                    sx = px + 0.5
                    w0 = _edge(b, c, sx, sy) / area
                    w1 = _edge(c, a, sx, sy) / area
                    w2 = 1.0 - w0 - w1
                    if w0 < 0.0 or w1 < 0.0 or w2 < 0.0:
                        continue
                    inv_z = w0 / a[2] + w1 / b[2] + w2 / c[2]
                    z_fixed = int(round((1.0 / inv_z) * 1_000_000.0))
                    pixel = py * width + px
                    if z_fixed < depth[pixel]:
                        depth[pixel] = z_fixed
                        offset = pixel * 3
                        pixels[offset : offset + 3] = bytes(mesh.color)
    depth_bytes = b"".join(value.to_bytes(8, "little", signed=False) for value in depth)
    return f"P6\n{width} {height}\n255\n".encode("ascii") + bytes(pixels), depth_bytes


def _edge(a: Sequence[float], b: Sequence[float], x: float, y: float) -> float:
    return (x - a[0]) * (b[1] - a[1]) - (y - a[1]) * (b[0] - a[0])


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(a[i] * b[i] for i in range(3))


def _cross(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def render_session(
    argv: Sequence[str], cwd: str | os.PathLike[str] | None = None
) -> subprocess.CompletedProcess[str]:
    """Run the Source 1 backend using a subprocess-compatible contract.

    One frame writes ``--frame-out`` directly.  A multi-frame batch treats it
    as a directory and writes ``frame-TICK-SUBTICK.ppm``.  ``--scene-out`` is
    always an atomic JSON manifest containing every state hash.
    """

    args_list = [os.fspath(item) for item in argv]
    parser = argparse.ArgumentParser(prog=BACKEND_NAME, add_help=False)
    parser.add_argument("--snapshot-json", required=True)
    parser.add_argument("--asset-root")
    parser.add_argument("--map")
    parser.add_argument("--map-asset")
    parser.add_argument("--vpk", action="append", default=[])
    parser.add_argument("--scene-out", required=True)
    parser.add_argument("--frame-out", required=True)
    parser.add_argument("--depth-out")
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--mode", choices=("canonical", "authoritative"), default="canonical")
    parser.add_argument("--allow-synthetic-geometry", action="store_true")
    parser.add_argument("--reference-fov-degrees", type=float)
    parser.add_argument(
        "--fidelity-mode",
        choices=("canonical-reference", "total-capture-authoritative"),
        default="canonical-reference",
    )
    parser.add_argument("--expected-manifest-sha256")
    parser.add_argument("--expected-collection-manifest-sha256")
    parser.add_argument("--expected-build-id")
    parser.add_argument("--expected-build-sha256")
    parser.add_argument("--expected-content-manifest-sha256")
    parser.add_argument("--content-model-binding", action="append", default=[])
    try:
        ns, unknown = parser.parse_known_args(args_list)
        if unknown:
            raise Source1RenderRefusal("unknown-arguments", repr(unknown))
        base = Path(cwd or os.getcwd()).resolve()
        snapshot_path = _resolve_required(ns.snapshot_json, base)
        if not snapshot_path.is_file():
            raise Source1RenderRefusal("missing-snapshot", f"snapshot JSON does not exist: {snapshot_path}")
        try:
            document = json.loads(snapshot_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise Source1RenderRefusal("invalid-snapshot-json", str(exc)) from exc
        is_collection = (
            isinstance(document, Mapping)
            and document.get("schema") == TOTAL_CAPTURE_COLLECTION_RENDER_SCHEMA
        )
        snapshots = [] if is_collection else _snapshot_documents(document)
        config = Source1RenderConfig(
            asset_root=Path(ns.asset_root) if ns.asset_root else None,
            map_path=Path(ns.map) if ns.map else None,
            width=ns.width,
            height=ns.height,
            mode=ns.mode,
            allow_synthetic_geometry=ns.allow_synthetic_geometry,
            reference_fov_degrees=ns.reference_fov_degrees,
            vpk_paths=tuple(Path(path) for path in ns.vpk),
            map_asset=ns.map_asset,
            fidelity_mode=ns.fidelity_mode,
            expected_manifest_sha256=ns.expected_manifest_sha256,
            expected_collection_manifest_sha256=(
                ns.expected_collection_manifest_sha256
            ),
            expected_build_id=ns.expected_build_id,
            expected_build_sha256=ns.expected_build_sha256,
            expected_content_manifest_sha256=ns.expected_content_manifest_sha256,
            content_model_bindings=tuple(
                _parse_content_model_binding(item)
                for item in ns.content_model_binding
            ),
        ).normalized(cwd=base)
        collection = render_collection(document, config) if is_collection else None
        rendered = (
            collection.frames if collection is not None
            else render_batch(snapshots, config)
        )
        if not rendered and collection is None:
            raise Source1RenderRefusal("empty-batch", "snapshot input contains no frames")
        scene_path = _resolve_required(ns.scene_out, base)
        frame_path = _resolve_required(ns.frame_out, base)
        explicit_depth_path = (
            _resolve_required(ns.depth_out, base) if ns.depth_out else None
        )
        outputs: list[str] = []
        depth_outputs: list[str] = []
        if len(rendered) == 1 and collection is None:
            _atomic_bytes(frame_path, rendered[0].ppm)
            depth_path = explicit_depth_path or frame_path.with_suffix(
                ".depth.u64le"
            )
            _atomic_bytes(depth_path, rendered[0].depth_u64le)
            outputs.append(str(frame_path))
            depth_outputs.append(str(depth_path))
        else:
            frame_path.mkdir(parents=True, exist_ok=True)
            if explicit_depth_path is not None:
                explicit_depth_path.mkdir(parents=True, exist_ok=True)
            for item in rendered:
                vantage = item.scene.camera.vantage_id or str(
                    item.scene.camera.entity_id
                )
                vantage_token = hashlib.sha256(vantage.encode("utf-8")).hexdigest()
                if collection is not None:
                    epoch = item.scene.epoch_context
                    assert epoch is not None
                    collection_token = hashlib.sha256(
                        str(epoch["collection_id"]).encode("utf-8")
                    ).hexdigest()
                    process_token = hashlib.sha256(
                        str(epoch["process_epoch_id"]).encode("utf-8")
                    ).hexdigest()
                    map_token = hashlib.sha256(
                        str(epoch["map_epoch_id"]).encode("utf-8")
                    ).hexdigest()
                    map_name_token = hashlib.sha256(
                        str(epoch["map_name"]).encode("utf-8")
                    ).hexdigest()
                    capture_token = hashlib.sha256(
                        str(epoch["capture_epoch_id"]).encode("utf-8")
                    ).hexdigest()
                    relative = (
                        Path(
                            f"collection-{collection_token}-"
                            f"{epoch['collection_manifest_sha256']}"
                        )
                        / f"epoch-{int(epoch['ordinal']):06d}-{capture_token}"
                        / f"process-{process_token}"
                        / f"map-{map_token}-{map_name_token}"
                        / (
                            f"frame-{item.scene.tick:010d}-"
                            f"{item.scene.subtick:06d}-pov-{vantage_token}"
                        )
                    )
                else:
                    relative = Path(
                        f"frame-{item.scene.tick:010d}-{item.scene.subtick:06d}"
                        f"-pov-{vantage_token[:12]}"
                    )
                target = frame_path / relative.with_suffix(".ppm")
                depth_root = explicit_depth_path or frame_path
                depth_target = depth_root / relative.with_suffix(".depth.u64le")
                _atomic_bytes(target, item.ppm)
                _atomic_bytes(depth_target, item.depth_u64le)
                outputs.append(str(target))
                depth_outputs.append(str(depth_target))
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "backend": dict(backend_descriptor()),
            "frames": [
                item.manifest()
                | {
                    "scene": item.scene.to_dict(),
                    "output": outputs[index],
                    "depth_output": depth_outputs[index],
                }
                for index, item in enumerate(rendered)
            ],
            "collection": (
                collection.summary() if collection is not None else None
            ),
        }
        _atomic_json(scene_path, manifest)
        return subprocess.CompletedProcess(
            args_list,
            0,
            json.dumps(
                {
                    "scene_out": str(scene_path),
                    "outputs": outputs,
                    "depth_outputs": depth_outputs,
                },
                sort_keys=True,
            ),
            "",
        )
    except (Source1RenderRefusal, Source1FormatError, ValueError, TypeError, OSError) as exc:
        return subprocess.CompletedProcess(args_list, 2, "", f"{BACKEND_NAME} refused: {exc}")


def _snapshot_documents(document: Any) -> list[Mapping[str, Any]]:
    if isinstance(document, Mapping) and document.get("schema") == "tardigrade/state-snapshot/v1":
        return [document]
    if isinstance(document, Mapping) and "frames" in document:
        document = document["frames"]
    if isinstance(document, (str, bytes, Mapping)) or not isinstance(document, Sequence):
        raise Source1RenderRefusal("invalid-snapshot-json", "expected a snapshot, array, or {frames: [...]} object")
    if any(not isinstance(item, Mapping) for item in document):
        raise Source1RenderRefusal("invalid-snapshot-json", "every frame must be an object")
    return list(document)


def _parse_content_model_binding(value: str) -> tuple[str, str]:
    digest, separator, source_path = value.partition("=")
    if not separator or not source_path:
        raise Source1RenderRefusal(
            "invalid-content-model-binding",
            "--content-model-binding must be SHA256=source/path.mdl",
        )
    return digest, source_path


def _atomic_json(path: Path, document: Mapping[str, Any]) -> None:
    data = (json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    _atomic_bytes(path, data)


def _atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _resolve_optional(path: Path | None, base: Path) -> Path | None:
    if path is None:
        return None
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _source_asset_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise Source1RenderRefusal(
            "invalid-source1-path", f"{label} must be a nonempty Source-relative path"
        )
    normalized = value.replace("\\", "/")
    parts = normalized.split("/")
    if (
        normalized.startswith("/")
        or "\0" in normalized
        or any(part in ("", ".", "..") for part in parts)
        or ":" in parts[0]
    ):
        raise Source1RenderRefusal(
            "invalid-source1-path",
            f"{label} must be normalized, relative, and traversal-free: {value!r}",
        )
    return "/".join(parts)


def _resolve_required(path: str | os.PathLike[str], base: Path) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (base / candidate).resolve()


def _is_beneath(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Source1RenderRefusal("invalid-number", f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise Source1RenderRefusal("invalid-number", f"{label} must be finite")
    return result


def _index_unbounded(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Source1RenderRefusal(
            "invalid-camera", f"{label} values must be positive integers"
        )
    return value


def _matrix4_transform(
    flat: Sequence[float],
    vector: Sequence[float],
    convention: Mapping[str, Any],
) -> tuple[float, float, float, float]:
    if len(flat) != 16 or len(vector) != 4:
        raise Source1RenderRefusal("invalid-matrix", "4x4 transform shape mismatch")
    if convention.get("layout") == "row-major":
        matrix = tuple(tuple(flat[row * 4 + column] for column in range(4))
                       for row in range(4))
    elif convention.get("layout") == "column-major":
        matrix = tuple(tuple(flat[column * 4 + row] for column in range(4))
                       for row in range(4))
    else:
        raise Source1RenderRefusal("invalid-matrix-convention", "unknown matrix layout")
    if convention.get("vectors") == "column-vectors":
        result = tuple(
            sum(matrix[row][column] * vector[column] for column in range(4))
            for row in range(4)
        )
    elif convention.get("vectors") == "row-vectors":
        result = tuple(
            sum(vector[row] * matrix[row][column] for row in range(4))
            for column in range(4)
        )
    else:
        raise Source1RenderRefusal("invalid-matrix-convention", "unknown vector convention")
    return result  # type: ignore[return-value]


def _camera_view_transform(
    flat: Sequence[float],
    vector: Sequence[float],
    convention: Mapping[str, Any],
) -> tuple[float, float, float, float]:
    if convention.get("view_transform_direction") == "world-to-view":
        return _matrix4_transform(flat, vector, convention)
    if convention.get("view_transform_direction") != "view-to-world":
        raise Source1RenderRefusal(
            "invalid-matrix-convention", "unknown view transform direction"
        )
    inverse = _inverse_matrix4_flat(flat, convention)
    return _matrix4_transform(inverse, vector, convention)


def _inverse_matrix4_flat(
    flat: Sequence[float], convention: Mapping[str, Any]
) -> tuple[float, ...]:
    if convention.get("layout") == "row-major":
        rows = [list(float(flat[row * 4 + column]) for column in range(4))
                for row in range(4)]
    else:
        rows = [list(float(flat[column * 4 + row]) for column in range(4))
                for row in range(4)]
    augmented = [row + [1.0 if row_no == column else 0.0 for column in range(4)]
                 for row_no, row in enumerate(rows)]
    for column in range(4):
        pivot = max(range(column, 4), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            raise Source1RenderRefusal("singular-view-matrix", "camera view matrix is singular")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [item / divisor for item in augmented[column]]
        for row in range(4):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                augmented[row][item] - factor * augmented[column][item]
                for item in range(8)
            ]
    inverse_rows = [row[4:] for row in augmented]
    if convention.get("layout") == "row-major":
        return tuple(inverse_rows[row][column] for row in range(4) for column in range(4))
    return tuple(inverse_rows[row][column] for column in range(4) for row in range(4))


def _vec3(value: Any, label: str) -> tuple[float, float, float]:
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Sequence) or len(value) != 3:
        raise Source1RenderRefusal("invalid-vector", f"{label} must contain three numbers")
    return tuple(_number(item, label) for item in value)  # type: ignore[return-value]


def _index(value: Any, count: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < count:
        raise Source1RenderRefusal("invalid-geometry", f"triangle index {value!r} is outside [0, {count})")
    return value


def _byte(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 255:
        raise Source1RenderRefusal("invalid-color", f"{label} values must be integer bytes")
    return value


def _ppm_dimensions(ppm: bytes) -> tuple[int, int]:
    header = ppm.split(b"\n", 3)
    width, height = header[1].split()
    return int(width), int(height)


__all__ = [
    "BACKEND_NAME",
    "MANIFEST_SCHEMA",
    "MODEL_BINDING_DERIVATION",
    "SCENE_SCHEMA",
    "Source1Camera",
    "Source1RenderConfig",
    "Source1RenderedFrame",
    "Source1RenderedCollection",
    "Source1RenderRefusal",
    "Source1Scene",
    "Source1TriangleMesh",
    "backend_descriptor",
    "render_batch",
    "render_collection",
    "render_frame",
    "render_frame_vantages",
    "render_session",
    "render_snapshot",
]
