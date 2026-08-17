# Passive Counter-Strike ego-input capture

This tool records timestamped raw HID reports and decoded HID usage/value
transitions from explicitly selected devices while the selected game is foreground. A
listen-only CGEvent stream can be enabled as an independent aggregate
cross-check. For `csgo_legacy` it is a producer owned by the unified native
coordinator, not an engine hook or evidence of when the game consumed an input. See
[BACKEND_AUDIT.md](BACKEND_AUDIT.md) for the exact observability boundary.

## Safety boundary

The Swift process uses a normal, non-seizing `IOHIDManager` input-value
callback. It does not inject or modify events, seize a device, read or debug
game memory, load a library into the game, install persistence, change a game or
Steam configuration, send console commands, inspect network traffic, or
upload anything. It records numeric HID usage page, usage, and integer value;
it never maps keyboard usages to text or characters.

Capture requires all of the following:

- an explicit `--consent` acknowledgement;
- one or more exact `VID:PID` selectors;
- either target-process lifecycle mode for a complete session or a finite
  duration of at most four hours for probes;
- a target foreground application name or bundle ID;
- the relevant macOS Input Monitoring permission granted by the operator.
  IOHID and CGEvent permission states are diagnosed independently. Capture
  never calls a permission-request API implicitly; the separate, explicit
  `--consent --request-permission KIND` action is available if desired.

The persistent terminal banner shows whether capture is active or paused.
Focus transitions are written to the stream. In `--session` mode the recorder
binds the first matching foreground PID, pauses when it is not foreground, and
finishes when that same PID terminates. It posts a visible warning immediately
and every configured interval while focused; an interval that expires while
unfocused is delivered on refocus. Under cmux it uses the launching workspace's
notification UI, otherwise it uses a standard macOS notification.

## Inventory and direct use

Build outside this repository:

```sh
mkdir -p /Users/mdot/dox/runs/cs2-ego-capture-bin
swiftc -swift-version 6 -O \
  -framework AppKit -framework CoreGraphics -framework IOKit \
  tools/cs2_ego_capture/Capture.swift \
  -o /Users/mdot/dox/runs/cs2-ego-capture-bin/cs2-ego-capture
```

Inventory visible HID devices without recording values:

```sh
/Users/mdot/dox/runs/cs2-ego-capture-bin/cs2-ego-capture --list-devices
```

Diagnose both public access mechanisms without prompting:

```sh
/Users/mdot/dox/runs/cs2-ego-capture-bin/cs2-ego-capture --diagnose-permissions
```

An operator may explicitly ask macOS for one permission; this is never done by
inventory, diagnostics, capture setup, or the supervisor:

```sh
/Users/mdot/dox/runs/cs2-ego-capture-bin/cs2-ego-capture \
  --consent --request-permission iohid
```

The ERGO M575S presently attached to the development Mac identifies as
`0x046d:0xb041`. Capture a complete CS2 process session, regardless of length,
with a visible reminder every 15 minutes:

```sh
/Users/mdot/dox/runs/cs2-ego-capture-bin/cs2-ego-capture \
  --consent \
  --game-variant cs2 \
  --backend hid-report \
  --device 0x046d:0xb041 \
  --foreground-name cs2 \
  --session \
  --reminder-minutes 15 \
  --output /Users/mdot/dox/runs/example-session/hid-events.jsonl
```

Valve ships no native macOS CS2 client, so that CS2 command applies only when
observing an explicitly documented external supported execution context.
Legacy CS:GO uses the same passive lifecycle capture with exact Source 1
application selectors:

```sh
/Users/mdot/dox/runs/cs2-ego-capture-bin/cs2-ego-capture \
  --consent \
  --game-variant csgo_legacy \
  --backend hid-report \
  --device 0x046d:0xb041 \
  --foreground-name csgo_osx64 \
  --foreground-name "Counter-Strike: Global Offensive" \
  --session \
  --reminder-minutes 15 \
  --output /Users/mdot/dox/runs/example-csgo-session/hid-events.jsonl
```

A location-qualified selector, `VID:PID:LOCATION`, disambiguates identical
devices. A keyboard is captured only if separately selected. The program still
records report bytes, HID usages, or numeric virtual keycodes—not characters.

For a ten-second backend/calibration measurement, keep the named terminal app
foreground and move/click/scroll the selected device:

```sh
python3 scripts/cs2_hid_probe.py \
  --consent --duration-seconds 10 --backend calibration \
  --device 0x046d:0xb041 --foreground-name cmux
```

The external probe report gives per-backend counts, timestamp inter-arrival
distributions, observed rates, and callback-minus-HID timestamp distributions.
It does not launch CS2.

## Passive diagnostic session

The older supervisor creates a new directory below
`/Users/mdot/dox/runs/cs2-capture-sessions` by default, builds the Swift binary
at the stable external path
`/Users/mdot/dox/runs/cs2-ego-capture-bin/cs2-ego-capture`, records
host/build/device provenance, and snapshots the chosen demo directory. The
stable path matters because macOS privacy grants should not be aimed at a new
ephemeral executable every session. A source change rebuilds the binary and
may require the operator to review Input Monitoring permission again.

It is deliberately only a passive HID diagnostic. It is not an acceptable
legacy total-capture path and now refuses `--game-variant csgo_legacy` instead
of asking the operator to create a demo manually. For a monitor-owned legacy
bot session use the unified coordinator:

```sh
python3 tools/csgo_total_capture/monitor.py \
  --consent --offline-only --insecure \
  --target-active-gameplay-seconds 3600 \
  --device 0x046d:0xb041
```

That coordinator starts HID before observing ordinary allowlisted game processes,
starts/stops the native demo from the in-process lifecycle observer, and joins
the demo and state sidecars without menu or console intervention. On exit, any
artifacts are copied after stability checks and SHA-256 hashed. The JSONL,
capture source/binary identity, device inventory, demo copy, and final
`manifest.json` are recorded in the external session. The manifest is
published by an atomic rename.

The supervisor refuses `--game-variant cs2` on macOS unless
`--external-client-context DESCRIPTION` explicitly records the supported
external context being observed. It never invents a native CS2 executable;
there is none. Legacy CS:GO total sessions belong to the unified coordinator
shown above; the standalone Swift binary retains `csgo_osx64` and
`com.valvesoftware.csgo` matching only for explicit HID diagnostics.

## Stream contract

`hid-events.jsonl` contains:

- a schema/capture-policy header and selected device identities;
- a simultaneous clock anchor containing Mach continuous, Mach absolute,
  `CLOCK_MONOTONIC_RAW`, and wall clocks plus the Mach timebase;
- foreground application transitions;
- the bound target PID, its termination, and every attempted visible logging
  notification with delivery backend/status;
- ordered raw interrupt reports with the public IOHID arrival timestamp,
  report ID/type/bytes, exact device identity, and callback clocks;
- decoded HID values with OS-absolute timestamps, usage page/usage/value, and
  exact device identity;
- optionally, independent listen-only CGEvents with elapsed-nanosecond
  timestamps and numeric deltas/buttons/scroll/keycodes. They have no physical
  device attribution;
- a terminal record with the stop reason.

Optional supervisor arguments of the form
`--alignment-anchor MONOTONIC_NS:DEMO_TICK` preserve independently established
alignment pairs. With two or more distinct anchors, the manifest includes a
least-squares affine estimate. This is bookkeeping for later comparison, not a
claim of exact timing: passive HID callback delivery, OS event processing,
client input sampling, user-command creation, prediction, server receipt, and
demo serialization are different clock-domain events. The native demo remains
authoritative for game state and outcomes.
