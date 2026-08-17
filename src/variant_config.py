"""Authoritative Counter-Strike product/backend configuration.

Variant selection is deliberately closed: callers name one of the canonical
IDs below and receive exactly one renderer backend.  There is no platform,
asset, import-success, or file-extension fallback between engines.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Any, Mapping


class VariantConfigurationError(ValueError):
    """A product variant is absent, unknown, or contradictory."""


@dataclass(frozen=True, slots=True)
class RendererBackend:
    kind: str
    module: str
    callable: str

    def descriptor(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class GameVariant:
    id: str
    display_name: str
    engine: str
    renderer: RendererBackend
    foreground_names_macos: tuple[str, ...]
    foreground_bundle_ids_macos: tuple[str, ...]
    client_executables_macos: tuple[str, ...]

    def public_descriptor(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "display_name": self.display_name,
            "engine": self.engine,
            "renderer": self.renderer.descriptor(),
        }


_VARIANTS: dict[str, GameVariant] = {
    "cs2": GameVariant(
        id="cs2",
        display_name="Counter-Strike 2",
        engine="source2",
        renderer=RendererBackend(
            kind="python_subprocess",
            module="counter_strike_render.gpu_render",
            callable="main",
        ),
        foreground_names_macos=("cs2", "Counter-Strike 2"),
        foreground_bundle_ids_macos=(),
        # Valve does not ship a native macOS CS2 client.  Keep this empty so
        # discovery cannot resurrect speculative osx64 paths as evidence.
        client_executables_macos=(),
    ),
    "csgo_legacy": GameVariant(
        id="csgo_legacy",
        display_name="Counter-Strike: Global Offensive (Legacy)",
        engine="source1",
        renderer=RendererBackend(
            kind="python_callable",
            module="counter_strike_render.source1_backend",
            callable="render_session",
        ),
        foreground_names_macos=(
            "csgo_osx64",
            "Counter-Strike: Global Offensive",
        ),
        foreground_bundle_ids_macos=("com.valvesoftware.csgo",),
        client_executables_macos=(
            "csgo_osx64",
            "csgo.app/Contents/MacOS/csgo_osx64",
            "Counter-Strike Global Offensive.app/Contents/MacOS/csgo_osx64",
        ),
    ),
}

VARIANTS: Mapping[str, GameVariant] = MappingProxyType(_VARIANTS)
VARIANT_IDS = tuple(_VARIANTS)


def get_variant(value: object, *, source: str = "game_variant") -> GameVariant:
    if not isinstance(value, str) or not value.strip():
        raise VariantConfigurationError(
            f"{source} is required and must be one of {list(VARIANT_IDS)!r}"
        )
    key = value.strip()
    try:
        return VARIANTS[key]
    except KeyError as error:
        raise VariantConfigurationError(
            f"unknown {source} {key!r}; supported variants are "
            f"{list(VARIANT_IDS)!r}"
        ) from error


def resolve_profile_request_variant(
    profile_value: object,
    request_value: object | None,
) -> GameVariant:
    """Resolve only from an explicit profile, optionally cross-checking wire.

    In particular this function never guesses from a filename, installed
    client, available backend, or whichever import happens to succeed.
    """
    profile = get_variant(profile_value, source="profile game_variant")
    if request_value is None:
        return profile
    request = get_variant(request_value, source="request game_variant")
    if request.id != profile.id:
        raise VariantConfigurationError(
            "request/profile game_variant mismatch: request names "
            f"{request.id!r}, profile names {profile.id!r}"
        )
    return profile


def public_variant_descriptors() -> list[dict[str, Any]]:
    return [VARIANTS[key].public_descriptor() for key in VARIANT_IDS]
