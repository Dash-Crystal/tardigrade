#!/usr/bin/env python3
"""Decode exact-device M575 HID reports into interval-integrated constraints.

The raw JSONL remains authoritative.  This decoder validates the captured HID
report descriptor before assigning bit meanings, then emits lossless signed
relative counts.  A report is an integral over an interval, not an
instantaneous translation.  The optional count-rate fields are interval
averages only; a continuous angular target history must additionally be
constrained by recorded engine view-angle/camera anchors.
"""
from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Iterable, Mapping


SCHEMA = "tardigrade/hid-relative-motion/v1"
M575_VID = 0x046D
M575_PID = 0xB041
M575_DESCRIPTOR_SHA256 = (
    "d5d52a6dcf81eb642653291a5c7a16df1f9c9656cec30e44625fd4052b0573a0"
)


class MotionDecodeError(ValueError):
    pass


def _integer(value: object, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise MotionDecodeError(f"{name} must be an integer >= {minimum}")
    return value


def _signed(value: int, bits: int) -> int:
    sign = 1 << (bits - 1)
    return value - (1 << bits) if value & sign else value


def decode_m575_report(payload: bytes) -> dict[str, int]:
    """Decode report ID 2 using the validated 97-byte M575 descriptor.

    Layout after the report ID is 16 button bits, signed relative 12-bit X and
    Y, signed 8-bit vertical wheel, and signed 8-bit horizontal pan.
    """
    if len(payload) != 8 or payload[0] != 2:
        raise MotionDecodeError(
            f"M575 motion report must be 8 bytes with report ID 2, got "
            f"length={len(payload)} id={payload[0] if payload else None}"
        )
    packed_xy = int.from_bytes(payload[3:6], "little")
    return {
        "buttons": int.from_bytes(payload[1:3], "little"),
        "dx_counts": _signed(packed_xy & 0xFFF, 12),
        "dy_counts": _signed((packed_xy >> 12) & 0xFFF, 12),
        "wheel_counts": _signed(payload[6], 8),
        "pan_counts": _signed(payload[7], 8),
    }


def _decode_descriptor(header: Mapping[str, object]) -> tuple[dict, int, int]:
    if header.get("schema") != "counter-strike-ego-hid-v3":
        raise MotionDecodeError("input is not counter-strike-ego-hid-v3")
    if header.get("effective_backend") != "hid-report":
        raise MotionDecodeError("relative-motion decoding requires hid-report")
    devices = header.get("selected_devices")
    if not isinstance(devices, list) or len(devices) != 1 or not isinstance(devices[0], dict):
        raise MotionDecodeError("capture must select exactly one physical device")
    device = devices[0]
    if device.get("vendor_id") != M575_VID or device.get("product_id") != M575_PID:
        raise MotionDecodeError("selected device is not Logitech ERGO M575S 046d:b041")
    raw_descriptor = device.get("report_descriptor_base64")
    if not isinstance(raw_descriptor, str):
        raise MotionDecodeError("capture header omitted the HID report descriptor")
    try:
        descriptor = base64.b64decode(raw_descriptor, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise MotionDecodeError(f"invalid descriptor base64: {exc}") from exc
    digest = hashlib.sha256(descriptor).hexdigest()
    if len(descriptor) != 97 or digest != M575_DESCRIPTOR_SHA256:
        raise MotionDecodeError(
            f"unrecognized M575 report descriptor length/hash: {len(descriptor)}/{digest}"
        )
    return device, _integer(header.get("mach_timebase_numer", 0), "timebase numer"), _integer(
        header.get("mach_timebase_denom", 0), "timebase denom"
    )


def decode_documents(documents: Iterable[Mapping[str, object]]) -> list[dict]:
    iterator = iter(documents)
    try:
        capture_header = next(iterator)
    except StopIteration as exc:
        raise MotionDecodeError("capture JSONL is empty") from exc

    # Timebase lives on the clock anchor, while identity/descriptor live on the
    # capture header.  Consume metadata until the first report without losing it.
    if capture_header.get("type") != "capture_header":
        raise MotionDecodeError("first JSONL record must be capture_header")
    device = capture_header.get("selected_devices", [{}])[0]
    if not isinstance(device, dict):
        raise MotionDecodeError("selected device identity is malformed")
    descriptor_header = dict(capture_header)
    numer = denom = 0
    pending: list[Mapping[str, object]] = []
    for document in iterator:
        if document.get("type") == "clock_anchor":
            numer = _integer(document.get("mach_timebase_numer"), "timebase numer", 1)
            denom = _integer(document.get("mach_timebase_denom"), "timebase denom", 1)
        pending.append(document)
    descriptor_header["mach_timebase_numer"] = numer
    descriptor_header["mach_timebase_denom"] = denom
    validated_device, numer, denom = _decode_descriptor(descriptor_header)

    output: list[dict] = [{
        "schema": SCHEMA,
        "type": "motion_header",
        "source_schema": capture_header["schema"],
        "device": validated_device,
        "descriptor_sha256": M575_DESCRIPTOR_SHA256,
        "coordinate_space": "signed-relative-hid-counts",
        "interval_average_rate_units": "counts-per-second",
        "finite_difference_rate_units": "counts-per-second-squared",
        "hid_tick_timebase": {
            "nanoseconds_numerator": numer,
            "nanoseconds_denominator": denom,
        },
        "provenance": {
            "counts": "decoded-losslessly-from-validated-hid-report",
            "interval_average_rate":
                "derived-from-integrated-counts-and-consecutive-report-timestamps",
            "finite_difference_of_interval_average_rate":
                "derived-summary-not-an-instantaneous-acceleration-observation",
        },
        "limitations": [
            "counts are not physical trackball angle without sensor calibration",
            "HID timestamps are host report-delivery time, not engine-consumption time",
            "macOS/game acceleration and filtering are not applied to these raw counts",
            "the exact stop time inside a report-silent interval is unobserved",
            "count divided by interval is only an interval average, not instantaneous velocity",
            "continuous angular motion requires authoritative engine view-angle anchors",
        ],
    }]
    previous_ticks: int | None = None
    previous_velocity: tuple[float, float] | None = None
    previous_midpoint_ns: float | None = None
    reports = moving = 0
    for document in pending:
        if document.get("type") != "hid_report":
            continue
        encoded = document.get("report_base64")
        if not isinstance(encoded, str):
            raise MotionDecodeError("hid_report omitted report_base64")
        try:
            payload = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise MotionDecodeError(f"invalid report base64: {exc}") from exc
        fields = decode_m575_report(payload)
        report_device = document.get("device")
        if not isinstance(report_device, dict) or any(
            report_device.get(name) != validated_device.get(name)
            for name in ("vendor_id", "product_id", "location_id")
        ):
            raise MotionDecodeError("hid_report device identity changed within capture")
        ticks = _integer(document.get("hid_report_timestamp_ticks"), "HID timestamp")
        timestamp_ns = ticks * numer / denom
        interval_ns: float | None = None
        interval_ticks: int | None = None
        velocity: tuple[float, float] | None = None
        acceleration: tuple[float, float] | None = None
        if previous_ticks is not None:
            interval_ticks = ticks - previous_ticks
            if interval_ticks <= 0:
                raise MotionDecodeError("HID report timestamps must strictly increase")
            interval_ns = interval_ticks * numer / denom
            seconds = interval_ns / 1_000_000_000.0
            velocity = (
                fields["dx_counts"] / seconds,
                fields["dy_counts"] / seconds,
            )
            midpoint_ns = timestamp_ns - interval_ns / 2.0
            if previous_velocity is not None and previous_midpoint_ns is not None:
                acceleration_seconds = (midpoint_ns - previous_midpoint_ns) / 1_000_000_000.0
                if acceleration_seconds <= 0 or not math.isfinite(acceleration_seconds):
                    raise MotionDecodeError("velocity midpoint timestamps are not increasing")
                acceleration = (
                    (velocity[0] - previous_velocity[0]) / acceleration_seconds,
                    (velocity[1] - previous_velocity[1]) / acceleration_seconds,
                )
            previous_velocity = velocity
            previous_midpoint_ns = midpoint_ns
        record = {
            "schema": SCHEMA,
            "type": "relative_motion",
            "sequence": _integer(document.get("sequence"), "sequence"),
            "source_sequence": _integer(document.get("source_sequence"), "source sequence"),
            "hid_report_timestamp_ticks": ticks,
            "hid_timestamp_ns": timestamp_ns,
            "received_monotonic_raw_ns": _integer(
                document.get("received_monotonic_raw_ns"), "received monotonic timestamp"
            ),
            **fields,
            "interval_ns": interval_ns,
            "interval_hid_ticks": interval_ticks,
            "interval_average_count_rate_per_second": list(velocity) if velocity else None,
            "finite_difference_of_interval_average_count_rate_per_second_squared": (
                list(acceleration) if acceleration else None
            ),
            "raw_report_base64": encoded,
        }
        output.append(record)
        reports += 1
        moving += int(fields["dx_counts"] != 0 or fields["dy_counts"] != 0)
        previous_ticks = ticks
    if not reports:
        raise MotionDecodeError("capture contains no M575 HID reports")
    output.append({
        "schema": SCHEMA,
        "type": "motion_summary",
        "reports": reports,
        "reports_with_nonzero_xy": moving,
        "terminal_observed": any(item.get("type") == "capture_end" for item in pending),
    })
    return output


def load_jsonl(path: Path) -> list[dict]:
    documents: list[dict] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            try:
                document = json.loads(line)
            except json.JSONDecodeError as exc:
                raise MotionDecodeError(f"line {line_number}: {exc}") from exc
            if not isinstance(document, dict):
                raise MotionDecodeError(f"line {line_number}: record must be an object")
            documents.append(document)
    return documents


def _atomic_write(path: Path, documents: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            for document in documents:
                output.write(json.dumps(document, sort_keys=True, separators=(",", ":")))
                output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output must not already exist")
    try:
        decoded = decode_documents(load_jsonl(args.capture))
        _atomic_write(args.output, decoded)
    except (OSError, MotionDecodeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
