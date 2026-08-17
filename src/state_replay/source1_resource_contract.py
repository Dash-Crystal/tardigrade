"""Shared canonical identity contract for Source 1 string-table resources."""
from __future__ import annotations

import base64
import binascii
import hashlib
import json
from typing import Mapping

STRING_TABLE_RESOURCE_PROVENANCE = "derived"
STRING_TABLE_RESOURCE_DERIVATION = (
    "canonical resource materialized from decoded Source 1 string-table state"
)
MODEL_BINDING_PROVENANCE = "derived"
MODEL_BINDING_DERIVATION = (
    "lossless join of decoded m_nModelIndex to recorded modelprecache entry"
)
MODEL_PRECACHE_ENTITY_ID = "source1:stringtable:modelprecache"
ENTRY_CONTENT_PROVENANCE = "recorded"
ENTRY_PRESENCE_PROVENANCE = "derived-from-ordered-table-history"
ENTRY_KEYS = frozenset({
    "index", "string", "user_data_base64", "content_provenance",
    "canonical_presence_provenance",
})


class Source1ResourceContractError(ValueError):
    """A canonical Source 1 resource document violates the shared contract."""


def make_string_table_entry(index: int, string: str,
                            user_data: bytes) -> dict[str, object]:
    """Return the exact five-key entry stored in canonical resource state."""
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise Source1ResourceContractError(
            "string-table entry index must be an integer >= 0")
    if not isinstance(string, str):
        raise Source1ResourceContractError(
            "string-table entry string must be text")
    if not isinstance(user_data, bytes):
        raise Source1ResourceContractError(
            "string-table entry user_data must be bytes")
    return {
        "index": index,
        "string": string,
        "user_data_base64": base64.b64encode(user_data).decode("ascii"),
        "content_provenance": ENTRY_CONTENT_PROVENANCE,
        "canonical_presence_provenance": ENTRY_PRESENCE_PROVENANCE,
    }


def validate_string_table_entry(
    document: Mapping[str, object],
) -> tuple[int, str, bytes]:
    """Validate an exact canonical entry and return decoded identity/content."""
    if not isinstance(document, Mapping):
        raise Source1ResourceContractError("string-table entry must be an object")
    keys = set(document)
    if keys != ENTRY_KEYS:
        raise Source1ResourceContractError(
            f"string-table entry keys must be exactly {sorted(ENTRY_KEYS)}")
    index = document["index"]
    string = document["string"]
    encoded = document["user_data_base64"]
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise Source1ResourceContractError(
            "string-table entry index must be an integer >= 0")
    if not isinstance(string, str):
        raise Source1ResourceContractError(
            "string-table entry string must be text")
    if not isinstance(encoded, str):
        raise Source1ResourceContractError(
            "string-table entry user_data_base64 must be text")
    if document["content_provenance"] != ENTRY_CONTENT_PROVENANCE:
        raise Source1ResourceContractError(
            f"content_provenance must equal {ENTRY_CONTENT_PROVENANCE!r}")
    if document["canonical_presence_provenance"] != ENTRY_PRESENCE_PROVENANCE:
        raise Source1ResourceContractError(
            "canonical_presence_provenance does not match the shared contract")
    try:
        data = base64.b64decode(encoded.encode("ascii"), validate=True)
    except (UnicodeEncodeError, binascii.Error, ValueError) as exc:
        raise Source1ResourceContractError(
            "user_data_base64 is not canonical RFC4648 base64") from exc
    if base64.b64encode(data).decode("ascii") != encoded:
        raise Source1ResourceContractError(
            "user_data_base64 does not round-trip canonically")
    return index, string, data


def string_table_entry_document_sha256(document: Mapping[str, object]) -> str:
    """Validate and hash an exact entry document without reconstructing it."""
    validate_string_table_entry(document)
    payload = json.dumps(dict(document), sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def string_table_entry_sha256(index: int, string: str,
                              user_data: bytes) -> str:
    """Hash the exact canonical entry document, including provenance keys."""
    document = make_string_table_entry(index, string, user_data)
    return string_table_entry_document_sha256(document)


__all__ = [
    "ENTRY_CONTENT_PROVENANCE",
    "ENTRY_KEYS",
    "ENTRY_PRESENCE_PROVENANCE",
    "MODEL_BINDING_DERIVATION",
    "MODEL_BINDING_PROVENANCE",
    "MODEL_PRECACHE_ENTITY_ID",
    "STRING_TABLE_RESOURCE_DERIVATION",
    "STRING_TABLE_RESOURCE_PROVENANCE",
    "Source1ResourceContractError",
    "make_string_table_entry",
    "string_table_entry_document_sha256",
    "string_table_entry_sha256",
    "validate_string_table_entry",
]
