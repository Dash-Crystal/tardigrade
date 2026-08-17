# macOS passive-input backend audit

## Outcome

The best polite host observation is a timestamped IOHID interrupt report from
an explicitly selected physical device, accompanied by IOHID's decoded element
values. It is finer than a native demo's user-command serialization and retains
the device's actual report cadence. It still precedes and cannot observe the
game's input-thread sampling, event accumulation, sensitivity transform,
user-command construction, prediction, or command transmission.

The production default is `hid-report`, retaining the finest packet-level
evidence without multiplying rows. `hid-dual` retains both raw reports and
decoded values as parallel views for validation. They are not two actions and
must never be summed. `calibration` adds a passive WindowServer event stream to
both IOHID views to measure what survives into the aggregate application-facing path. `auto` prefers both IOHID
views and falls back to CGEvent only, with the loss of physical attribution
written into the stream.

## Public APIs and what they expose

| Backend | Physical attribution | Source timestamp | Payload | Appropriate role |
|---|---|---|---|---|
| `IOHIDManagerRegisterInputReportWithTimeStampCallback` | Exact selected `VID:PID`, optionally `LOCATION` | IOHID report-arrival timestamp in OS absolute-time ticks | Raw interrupt report, report ID/type | Primary packet-level evidence |
| `IOHIDManagerRegisterInputValueCallback` | Exact selected device and element | `IOHIDValueGetTimeStamp`, documented as OS absolute time | Decoded usage page, usage, integer value | Primary action-readable evidence |
| `CGEvent.tapCreate(..., .listenOnly, ...)` at the session tap | None; all devices are aggregated by WindowServer | Nanoseconds elapsed since startup | Mouse deltas/buttons, scroll, virtual keycode/flags | Independent fallback and calibration only |
| `NSEvent.addGlobalMonitorForEvents` | None | Seconds since startup | Higher-level asynchronous AppKit event | Rejected for primary capture |

Apple describes IOHID input values as normally issued by interrupt-driven
reports, and its timestamped report callback explicitly supplies the time the
report arrived. The stream preserves the report descriptor and maximum input
report size so raw bytes remain decodable offline. It does not set a report
interval or otherwise configure the device.

NSEvent was not implemented as another nominal backend because it adds no
physical attribution, is asynchronous, excludes events delivered to the
monitoring app itself, requires Accessibility trust for key monitoring, and is
farther downstream than the listen-only Quartz event tap. Adding it would add
duplicate records without a stronger timing or identity claim.

Primary Apple references:

- [IOHID timestamped input-report callback](https://developer.apple.com/documentation/iokit/3042788-iohidmanagerregisterinputreportw)
- [IOHID input-value callback](https://developer.apple.com/documentation/iokit/1438367-iohidmanagerregisterinputvalueca)
- [IOHID access check](https://developer.apple.com/documentation/iokit/3181573-iohidcheckaccess)
- [IOHID explicit access request](https://developer.apple.com/documentation/iokit/3181574-iohidrequestaccess)
- [CGEvent tap creation](https://developer.apple.com/documentation/coregraphics/cgevent/tapcreate%28tap%3Aplace%3Aoptions%3Aeventsofinterest%3Acallback%3Auserinfo%3A%29)
- [CGEvent listen-only option](https://developer.apple.com/documentation/coregraphics/cgeventtapoptions/listenonly)
- [CGEvent timestamp definition](https://developer.apple.com/documentation/coregraphics/cgeventtimestamp)
- [NSEvent global monitor limitations](https://developer.apple.com/documentation/appkit/nsevent/addglobalmonitorforevents%28matching%3Ahandler%3A%29)

## Permissions and failure behavior

IOHID listen access is checked using
`IOHIDCheckAccess(kIOHIDRequestTypeListenEvent)`. CGEvent access is separately
reported by `CGPreflightListenEventAccess`. The old architecture incorrectly
used the CG preflight as a prerequisite for IOHID; version 2 removes that
cross-backend gate and attempts only the selected backend.

`--diagnose-permissions` is read-only and never prompts. Permission request APIs
are called only through the conspicuous standalone command
`--consent --request-permission iohid|cgevent|both`. Failure to open IOHID or
create a passive event tap reports the backend-specific access state and
remediation. No root escalation, private entitlement, Accessibility bypass, or
anti-cheat workaround exists.

On the audited macOS 26.3.2 host, the current cmux-launched stable recorder now
reports IOHID listen access `granted` and CGEvent listen preflight `true`.
Earlier denied/false diagnostics were therefore a permission state, not an API
limit. Inventory is safely available. The attached ERGO M575S reports:

- vendor/product `0x046d:0xb041`;
- Bluetooth Low Energy transport;
- 20-byte maximum input report;
- 97-byte HID report descriptor;
- reported interval 8,000 microseconds, corresponding to 125 Hz.

That 125 Hz is the device/driver report interval—not a claim that CS2 polls,
consumes, or emits user commands at 125 Hz.

## Timestamp and duplicate policy

Each stream record carries a global sequence, a backend-local sequence,
callback `mach_absolute_time`, `mach_continuous_time`, and
`CLOCK_MONOTONIC_RAW`. The opening anchor includes the Mach timebase and wall
clock. IOHID source timestamps can therefore be compared with callback
absolute time without substituting wall time. CGEvent's source clock is already
defined in elapsed nanoseconds since startup.

The recorder does not collapse reports, values, and CGEvents. A single report
can yield several element values; WindowServer may combine, accelerate, or
coalesce motion into a different number of CGEvents. Backend type and local
sequence make this multiplicity explicit. Offline consumers choose one
canonical source and use the others only for descriptor decoding or calibrated
correspondence.

The JSONL writer batches up to 256 KiB and flushes every 250 ms rather than
synchronously writing each callback. Foreground state is cached, updated on
application-activation notification, and rechecked at 10 Hz; callbacks do not
query AppKit per input. The terminal record includes
per-backend callback counts, callback errors, writer totals, and the internal
drop counter. A zero internal-drop count does not prove the OS, Bluetooth link,
driver, WindowServer, or application consumed every physical sample.

## Empirical probe

`scripts/cs2_hid_probe.py` runs for at most 30 seconds and never starts CS2. It
records the requested backends to `/Users/mdot/dox/runs/cs2-hid-probes`, then
reports:

- records per backend;
- per-source inter-arrival minimum, median, p95, maximum, and median rate;
- callback observation time minus IOHID timestamp distributions;
- decoded HID usages observed;
- permission/backend failure results and terminal drop counters.

The probe is intentionally empirical: permission failures produce a durable
report rather than being converted into a theory that fine-grained capture is
impossible.
