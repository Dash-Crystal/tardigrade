import AppKit
import CoreGraphics
import Darwin
import Foundation
import IOKit.hid

private let schemaVersion = "counter-strike-ego-hid-v3"

private enum CaptureBackend: String {
    case hidValue = "hid-value"
    case hidReport = "hid-report"
    case hidDual = "hid-dual"
    case cgEvent = "cgevent"
    case calibration = "calibration"
    case auto = "auto"

    var usesHIDValue: Bool { self == .hidValue || self == .hidDual || self == .calibration }
    var usesHIDReport: Bool { self == .hidReport || self == .hidDual || self == .calibration }
    var usesCGEvent: Bool { self == .cgEvent || self == .calibration }
    var usesHID: Bool { usesHIDValue || usesHIDReport }
}

private struct DeviceSelector: Hashable {
    let vendorID: UInt32
    let productID: UInt32
    let locationID: UInt32?

    var canonical: String {
        let base = String(format: "0x%04x:0x%04x", vendorID, productID)
        guard let locationID else { return base }
        return base + String(format: ":0x%08x", locationID)
    }

    static func parse(_ text: String) -> DeviceSelector? {
        let pieces = text.split(separator: ":", omittingEmptySubsequences: false)
        guard (pieces.count == 2 || pieces.count == 3),
              let vendor = parseInteger(String(pieces[0])),
              let product = parseInteger(String(pieces[1])) else {
            return nil
        }
        let location = pieces.count == 3 ? parseInteger(String(pieces[2])) : nil
        if pieces.count == 3 && location == nil { return nil }
        return DeviceSelector(vendorID: vendor, productID: product, locationID: location)
    }

    private static func parseInteger(_ text: String) -> UInt32? {
        if text.lowercased().hasPrefix("0x") {
            return UInt32(text.dropFirst(2), radix: 16)
        }
        return UInt32(text, radix: 10)
    }

    func matches(_ device: IOHIDDevice) -> Bool {
        guard numberProperty(device, kIOHIDVendorIDKey as CFString) == vendorID,
              numberProperty(device, kIOHIDProductIDKey as CFString) == productID else {
            return false
        }
        guard let locationID else { return true }
        return numberProperty(device, kIOHIDLocationIDKey as CFString) == locationID
    }
}

private struct Options {
    var consent = false
    var listDevices = false
    var diagnosePermissions = false
    var permissionRequest: String?
    var output: String?
    var durationSeconds: Double?
    var sessionLifecycle = false
    var outerSessionOwnerPID: pid_t?
    var eligibilityControlPath: String?
    var reminderMinutes: Double = 15
    var backend = CaptureBackend.hidReport
    var selectors: Set<DeviceSelector> = []
    var foregroundBundleIDs: Set<String> = []
    var foregroundNames: Set<String> = []
    var gameVariant = "cs2"

    var targetDisplayName: String {
        gameVariant == "csgo_legacy"
            ? "Counter-Strike: Global Offensive (Legacy)"
            : "Counter-Strike 2"
    }

    static let usage = """
    Passive, non-seizing macOS HID observer for a consented Counter-Strike capture session.

    Inventory:
      cs2-ego-capture --list-devices

    Complete target-lifecycle capture:
      cs2-ego-capture --consent --output EVENTS.jsonl --session \\
        --reminder-minutes 15 --device 0x046d:0xb041 --foreground-name cs2

    Options:
      --consent                       Required acknowledgement for capture mode.
      --game-variant ID               cs2 (default) or csgo_legacy; recorded in
                                      the stream and used in visible notices.
      --device VID:PID[:LOCATION]     Exact selector; location disambiguates identical devices.
      --backend NAME                  hid-value, hid-report (default), hid-dual,
                                      cgevent, calibration, or auto.
      --session                       Bind the first matching foreground PID and
                                      record until that application terminates.
      --outer-session-owner-pid PID   Persistent session: allow successive target
                                      PIDs and stop only when the monitor PID exits.
      --eligibility-control PATH      Newline-delimited PIDs authorized by the
                                      outer offline monitor; all others stay paused.
      --reminder-minutes K            Visible reminder interval in session mode
                                      (1...1440 minutes; default 15), deferred
                                      to refocus if its deadline passes off-focus.
      --duration-seconds N            Finite probe/test mode (1...14400 seconds),
                                      mutually exclusive with --session.
      --foreground-bundle-id ID       Log values only while this app is foreground.
      --foreground-name NAME          Foreground executable/localized-name fallback.
      --output PATH                   New JSONL file; existing files are refused.
      --list-devices                  Print device inventory as JSON and exit.
      --diagnose-permissions          Print IOHID and CGEvent access state; do not prompt.
      --request-permission KIND       With --consent: explicitly request iohid,
                                      cgevent, or both, then exit.
      --help                          Show this message.

    Only raw report bytes and numeric input fields are recorded. No character or
    text interpretation is performed. The program does not seize devices, inject
    input, inspect another process, alter events, or communicate over the network.
    """

    static func parse(_ arguments: [String]) throws -> Options {
        var result = Options()
        var index = 0
        while index < arguments.count {
            let argument = arguments[index]
            func value(after option: String) throws -> String {
                guard index + 1 < arguments.count else {
                    throw CaptureError.arguments("missing value after \(option)")
                }
                index += 1
                return arguments[index]
            }
            switch argument {
            case "--consent":
                result.consent = true
            case "--game-variant":
                let raw = try value(after: argument)
                guard ["cs2", "csgo_legacy"].contains(raw) else {
                    throw CaptureError.arguments("--game-variant must be cs2 or csgo_legacy")
                }
                result.gameVariant = raw
            case "--list-devices":
                result.listDevices = true
            case "--diagnose-permissions":
                result.diagnosePermissions = true
            case "--request-permission":
                let raw = try value(after: argument).lowercased()
                guard ["iohid", "cgevent", "both"].contains(raw) else {
                    throw CaptureError.arguments("--request-permission must be iohid, cgevent, or both")
                }
                result.permissionRequest = raw
            case "--backend":
                let raw = try value(after: argument)
                guard let backend = CaptureBackend(rawValue: raw) else {
                    throw CaptureError.arguments("unknown backend '\(raw)'")
                }
                result.backend = backend
            case "--output":
                result.output = try value(after: argument)
            case "--session":
                result.sessionLifecycle = true
            case "--outer-session-owner-pid":
                let raw = try value(after: argument)
                guard let pid = Int32(raw), pid > 0 else {
                    throw CaptureError.arguments("--outer-session-owner-pid must be a positive PID")
                }
                result.sessionLifecycle = true
                result.outerSessionOwnerPID = pid
            case "--eligibility-control":
                result.eligibilityControlPath = try value(after: argument)
            case "--reminder-minutes":
                let raw = try value(after: argument)
                guard let minutes = Double(raw), minutes >= 1, minutes <= 1_440 else {
                    throw CaptureError.arguments("--reminder-minutes must be in 1...1440")
                }
                result.reminderMinutes = minutes
            case "--duration-seconds":
                let raw = try value(after: argument)
                guard let duration = Double(raw), duration >= 1, duration <= 14_400 else {
                    throw CaptureError.arguments("--duration-seconds must be in 1...14400")
                }
                result.durationSeconds = duration
            case "--device":
                let raw = try value(after: argument)
                guard let selector = DeviceSelector.parse(raw) else {
                    throw CaptureError.arguments("invalid device selector '\(raw)'; use VID:PID")
                }
                result.selectors.insert(selector)
            case "--foreground-bundle-id":
                result.foregroundBundleIDs.insert(try value(after: argument))
            case "--foreground-name":
                result.foregroundNames.insert(try value(after: argument).lowercased())
            case "--help", "-h":
                print(usage)
                exit(0)
            default:
                throw CaptureError.arguments("unknown option '\(argument)'")
            }
            index += 1
        }
        if result.listDevices || result.diagnosePermissions { return result }
        if result.permissionRequest != nil {
            guard result.consent else {
                throw CaptureError.arguments("--request-permission requires --consent")
            }
            return result
        }
        guard result.consent else {
            throw CaptureError.arguments("capture requires the explicit --consent flag")
        }
        guard result.output != nil else {
            throw CaptureError.arguments("capture requires --output")
        }
        guard result.sessionLifecycle || result.durationSeconds != nil else {
            throw CaptureError.arguments("capture requires --session or --duration-seconds")
        }
        guard !(result.sessionLifecycle && result.durationSeconds != nil) else {
            throw CaptureError.arguments("--session and --duration-seconds are mutually exclusive")
        }
        guard !result.selectors.isEmpty || result.backend == .cgEvent else {
            throw CaptureError.arguments("capture requires at least one explicit --device VID:PID")
        }
        guard !result.foregroundBundleIDs.isEmpty || !result.foregroundNames.isEmpty else {
            throw CaptureError.arguments(
                "capture requires --foreground-bundle-id or --foreground-name gating"
            )
        }
        return result
    }
}

private enum CaptureError: Error, CustomStringConvertible {
    case arguments(String)
    case runtime(String)

    var description: String {
        switch self {
        case .arguments(let message), .runtime(let message): return message
        }
    }
}

private func property(_ device: IOHIDDevice, _ key: CFString) -> Any? {
    IOHIDDeviceGetProperty(device, key)
}

private func numberProperty(_ device: IOHIDDevice, _ key: CFString) -> UInt32? {
    (property(device, key) as? NSNumber)?.uint32Value
}

private func stringProperty(_ device: IOHIDDevice, _ key: CFString) -> String? {
    property(device, key) as? String
}

private func deviceIdentity(_ device: IOHIDDevice, includeDescriptor: Bool = false) -> [String: Any] {
    var identity: [String: Any] = [:]
    if let vendor = numberProperty(device, kIOHIDVendorIDKey as CFString) {
        identity["vendor_id"] = vendor
        identity["vendor_id_hex"] = String(format: "0x%04x", vendor)
    }
    if let product = numberProperty(device, kIOHIDProductIDKey as CFString) {
        identity["product_id"] = product
        identity["product_id_hex"] = String(format: "0x%04x", product)
    }
    if let value = stringProperty(device, kIOHIDManufacturerKey as CFString) {
        identity["manufacturer"] = value
    }
    if let value = stringProperty(device, kIOHIDProductKey as CFString) {
        identity["product"] = value
    }
    if let value = stringProperty(device, kIOHIDTransportKey as CFString) {
        identity["transport"] = value
    }
    // A hardware serial is unnecessary for this capture contract and would
    // turn a local input trace into a persistent device identifier. VID/PID,
    // product, transport, and location data are sufficient for provenance;
    // the operator still selects devices explicitly by VID/PID.
    if let value = numberProperty(device, kIOHIDLocationIDKey as CFString) {
        identity["location_id"] = value
    }
    if let value = numberProperty(device, kIOHIDPrimaryUsagePageKey as CFString) {
        identity["primary_usage_page"] = value
    }
    if let value = numberProperty(device, kIOHIDPrimaryUsageKey as CFString) {
        identity["primary_usage"] = value
    }
    if let value = numberProperty(device, kIOHIDMaxInputReportSizeKey as CFString) {
        identity["max_input_report_size"] = value
    }
    if let value = numberProperty(device, kIOHIDReportIntervalKey as CFString) {
        identity["report_interval_us_requested_or_reported"] = value
    }
    if includeDescriptor,
       let descriptor = property(device, kIOHIDReportDescriptorKey as CFString) as? Data {
        identity["report_descriptor_base64"] = descriptor.base64EncodedString()
        identity["report_descriptor_bytes"] = descriptor.count
    }
    return identity
}

private func allDevices(_ manager: IOHIDManager) -> [IOHIDDevice] {
    guard let set = IOHIDManagerCopyDevices(manager) as? Set<IOHIDDevice> else { return [] }
    return Array(set)
}

private func devicesMatching(
    _ selectors: Set<DeviceSelector>, from devices: [IOHIDDevice]
) -> [IOHIDDevice] {
    guard !selectors.isEmpty else { return devices }
    return devices.filter { device in selectors.contains(where: { $0.matches(device) }) }
}

private func selector(for device: IOHIDDevice) -> DeviceSelector? {
    guard let vendor = numberProperty(device, kIOHIDVendorIDKey as CFString),
          let product = numberProperty(device, kIOHIDProductIDKey as CFString) else {
        return nil
    }
    return DeviceSelector(vendorID: vendor, productID: product, locationID: nil)
}

private func monotonicNanoseconds() -> UInt64 {
    var value = timespec()
    clock_gettime(CLOCK_MONOTONIC_RAW, &value)
    return UInt64(value.tv_sec) * 1_000_000_000 + UInt64(value.tv_nsec)
}

private func wallNanoseconds() -> UInt64 {
    var value = timespec()
    clock_gettime(CLOCK_REALTIME, &value)
    return UInt64(value.tv_sec) * 1_000_000_000 + UInt64(value.tv_nsec)
}

private final class JSONLWriter {
    private let handle: FileHandle
    private var buffer = Data()
    private let flushThreshold = 256 * 1024
    private(set) var recordsWritten: UInt64 = 0
    private(set) var bytesWritten: UInt64 = 0

    init(newFileAt path: String) throws {
        let parent = URL(fileURLWithPath: path).deletingLastPathComponent().path
        try FileManager.default.createDirectory(
            atPath: parent,
            withIntermediateDirectories: true,
            attributes: [.posixPermissions: 0o700]
        )
        let descriptor = Darwin.open(path, O_WRONLY | O_CREAT | O_EXCL, S_IRUSR | S_IWUSR)
        guard descriptor >= 0 else {
            throw CaptureError.runtime("refusing unavailable/existing output '\(path)': \(String(cString: strerror(errno)))")
        }
        handle = FileHandle(fileDescriptor: descriptor, closeOnDealloc: true)
    }

    func write(_ object: [String: Any]) throws {
        var data = try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
        data.append(0x0a)
        buffer.append(data)
        recordsWritten += 1
        if buffer.count >= flushThreshold { try flush() }
    }

    func flush() throws {
        guard !buffer.isEmpty else { return }
        let pendingBytes = buffer.count
        try handle.write(contentsOf: buffer)
        bytesWritten += UInt64(pendingBytes)
        buffer.removeAll(keepingCapacity: true)
    }

    func close() throws {
        try flush()
        try handle.synchronize()
        try handle.close()
    }
}

private struct NotificationDelivery {
    let backend: String
    let succeeded: Bool
    let detail: String
}

// Prefer the UI belonging to the launching cmux workspace when available.
// Outside cmux, use macOS's own display-notification command. Both paths are
// one-way notices: they do not inspect, focus, or send input to the target.
private func deliverLoggingNotification(title: String, body: String) -> NotificationDelivery {
    let environment = ProcessInfo.processInfo.environment
    if let executable = environment["CMUX_BUNDLED_CLI_PATH"],
       FileManager.default.isExecutableFile(atPath: executable) {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: executable)
        var arguments = ["notify", "--title", title, "--body", body]
        if let workspace = environment["CMUX_WORKSPACE_ID"], !workspace.isEmpty {
            arguments += ["--workspace", workspace]
        }
        process.arguments = arguments
        process.standardOutput = FileHandle.nullDevice
        process.standardError = FileHandle.nullDevice
        do {
            try process.run()
            process.waitUntilExit()
            return NotificationDelivery(
                backend: "cmux_notify",
                succeeded: process.terminationStatus == 0,
                detail: "exit_\(process.terminationStatus)"
            )
        } catch {
            return NotificationDelivery(backend: "cmux_notify", succeeded: false, detail: "\(error)")
        }
    }

    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/usr/bin/osascript")
    process.arguments = [
        "-e", "on run argv",
        "-e", "display notification (item 2 of argv) with title (item 1 of argv)",
        "-e", "end run",
        "--", title, body,
    ]
    process.standardOutput = FileHandle.nullDevice
    process.standardError = FileHandle.nullDevice
    do {
        try process.run()
        process.waitUntilExit()
        return NotificationDelivery(
            backend: "macos_display_notification",
            succeeded: process.terminationStatus == 0,
            detail: "exit_\(process.terminationStatus)"
        )
    } catch {
        return NotificationDelivery(
            backend: "macos_display_notification", succeeded: false, detail: "\(error)"
        )
    }
}

// IOHID callbacks and timers are scheduled on the main run loop. Dispatch signal
// sources also target the main queue, so mutable recorder state is serialized.
private final class Recorder: @unchecked Sendable {
    let options: Options
    let effectiveBackend: CaptureBackend
    let writer: JSONLWriter
    let selectedDevices: [IOHIDDevice]
    private var sequence: UInt64 = 0
    private var sourceSequences: [String: UInt64] = [:]
    private var counts: [String: UInt64] = [:]
    private var callbackErrors: UInt64 = 0
    private var notificationFailures: UInt64 = 0
    private var notificationCount: UInt64 = 0
    var cgEventTap: CFMachPort?
    private var focusEligible = false
    private var focusSignature = ""
    private var boundTargetPID: pid_t?
    private var boundTargetName = ""
    private var targetBoundMonotonicNS: UInt64?
    private var targetEpoch: UInt64 = 0
    private var nextReminderMonotonicNS: UInt64?
    private var stopReason = "operator_stop"
    private var stopped = false

    init(options: Options, effectiveBackend: CaptureBackend, selectedDevices: [IOHIDDevice]) throws {
        self.options = options
        self.effectiveBackend = effectiveBackend
        self.selectedDevices = selectedDevices
        self.writer = try JSONLWriter(newFileAt: options.output!)
    }

    func begin() throws {
        var timebase = mach_timebase_info_data_t()
        mach_timebase_info(&timebase)
        let physicalAttribution = effectiveBackend.usesHID
        let attributionScope = effectiveBackend.usesHID
            ? (effectiveBackend.usesCGEvent ? "iohid_records_only" : "all_input_records")
            : "none"
        try writer.write([
            "type": "capture_header",
            "schema": schemaVersion,
            "game_variant": options.gameVariant,
            "capture_scope": "passive_host_input_observation",
            "consent": true,
            "non_seizing": true,
            "requested_backend": options.backend.rawValue,
            "effective_backend": effectiveBackend.rawValue,
            "capture_lifecycle": options.outerSessionOwnerPID != nil
                ? "successive_matching_processes_until_outer-monitor-termination"
                : options.sessionLifecycle
                ? "first_matching_foreground_pid_until_termination"
                : "finite_duration",
            "duration_limit_seconds": options.durationSeconds ?? NSNull(),
            "reminder_interval_minutes": options.sessionLifecycle
                ? options.reminderMinutes
                : NSNull(),
            "physical_device_attribution_scope": attributionScope,
            "requested_device_selectors": options.selectors.map(\.canonical).sorted(),
            "selected_devices": physicalAttribution
                ? selectedDevices.map { deviceIdentity($0, includeDescriptor: true) }
                : [],
            "unattributed_backend_device_inventory_metadata": physicalAttribution
                ? []
                : selectedDevices.map { deviceIdentity($0, includeDescriptor: true) },
            "foreground_bundle_ids": Array(options.foregroundBundleIDs).sorted(),
            "foreground_names": Array(options.foregroundNames).sorted(),
            "eligibility_control": options.eligibilityControlPath ?? NSNull(),
            "limitations": [
                "HID callback delivery timestamps are not game-engine-consumption timestamps.",
                "The recorder observes host input only and does not establish which event the selected game consumed.",
                "Demo-tick alignment requires independent anchors and is an estimate.",
                "CGEvent records are aggregate WindowServer events and cannot be attributed to a selected physical device.",
                "hid-report and hid-value records are parallel views of the same device traffic and are not merged or deduplicated."
            ]
        ])
        try writer.write([
            "type": "clock_anchor",
            "sequence": nextSequence(),
            "mach_continuous_ticks": mach_continuous_time(),
            "mach_absolute_ticks": mach_absolute_time(),
            "monotonic_raw_ns": monotonicNanoseconds(),
            "wall_unix_ns": wallNanoseconds(),
            "mach_timebase_numer": timebase.numer,
            "mach_timebase_denom": timebase.denom
        ])
    }

    func arm() {
        refreshFocus(force: true)
    }

    private func nextSequence() -> UInt64 {
        sequence += 1
        return sequence
    }

    private func nextSourceSequence(_ source: String) -> UInt64 {
        let next = (sourceSequences[source] ?? 0) + 1
        sourceSequences[source] = next
        counts[source, default: 0] += 1
        return next
    }

    private func matchesTargetSelector(_ app: NSRunningApplication?) -> Bool {
        let bundleID = app?.bundleIdentifier ?? ""
        let name = (app?.localizedName ?? "").lowercased()
        let executable = (app?.executableURL?.lastPathComponent ?? "").lowercased()
        let selectorMatches = options.foregroundBundleIDs.contains(bundleID)
            || options.foregroundNames.contains(name)
            || options.foregroundNames.contains(executable)
        guard selectorMatches, let app else { return false }
        guard options.eligibilityControlPath != nil else { return true }
        return allowedProcessEpoch(app.processIdentifier) != nil
    }

    private func allowedProcessEpoch(_ pid: pid_t) -> UInt64? {
        guard let path = options.eligibilityControlPath,
              let contents = try? String(contentsOfFile: path, encoding: .utf8) else { return nil }
        for line in contents.split(whereSeparator: \.isNewline) {
            let pieces = line.split(separator: ":", maxSplits: 1)
            if pieces.count == 2, Int32(pieces[0]) == pid, let epoch = UInt64(pieces[1]) {
                return epoch
            }
        }
        return nil
    }

    private func bindTarget(_ app: NSRunningApplication, now: UInt64) -> Bool {
        guard options.sessionLifecycle, boundTargetPID == nil else { return false }
        if let epoch = allowedProcessEpoch(app.processIdentifier) {
            targetEpoch = epoch
        } else {
            targetEpoch += 1
        }
        boundTargetPID = app.processIdentifier
        boundTargetName = app.localizedName ?? app.executableURL?.lastPathComponent ?? "target"
        targetBoundMonotonicNS = now
        let intervalNS = UInt64(options.reminderMinutes * 60 * 1_000_000_000)
        nextReminderMonotonicNS = now + intervalNS
        do {
            try writer.write([
                "type": "session_target_bound",
                "sequence": nextSequence(),
                "received_monotonic_raw_ns": now,
                "wall_unix_ns": wallNanoseconds(),
                "pid": app.processIdentifier,
                "process_epoch": targetEpoch,
                "bundle_id": app.bundleIdentifier ?? "",
                "application_name": boundTargetName,
                "lifecycle": options.outerSessionOwnerPID == nil
                    ? "record_until_this_pid_terminates"
                    : "record_this-target-until-transition;outer-session-continues"
            ])
        } catch {
            fputs("capture write failed while binding target: \(error)\n", stderr)
            requestStop("write_error")
            return false
        }
        return true
    }

    private func deliverReminder(reason: String, now: UInt64) {
        guard options.sessionLifecycle, let targetPID = boundTargetPID else { return }
        let elapsedMinutes = targetBoundMonotonicNS.map {
            Double(now - $0) / 60_000_000_000
        } ?? 0
        let persistent = options.outerSessionOwnerPID != nil
        let delivery = deliverLoggingNotification(
            title: "\(options.targetDisplayName) action recording active",
            body: persistent
                ? "Selected HID actions are recorded while process epoch \(targetEpoch) (PID \(targetPID)) is focused. This epoch ends with the process; the consented outer session remains armed until its coverage target or explicit close."
                : "Selected HID actions are being recorded while \(boundTargetName) is focused, at your request. Recording ends when PID \(targetPID) exits."
        )
        notificationCount += 1
        if !delivery.succeeded { notificationFailures += 1 }
        do {
            try writer.write([
                "type": "logging_notification",
                "sequence": nextSequence(),
                "received_monotonic_raw_ns": now,
                "wall_unix_ns": wallNanoseconds(),
                "reason": reason,
                "target_pid": targetPID,
                "process_epoch": targetEpoch,
                "outer_session_persistent": persistent,
                "elapsed_session_minutes": elapsedMinutes,
                "delivery_backend": delivery.backend,
                "delivery_succeeded": delivery.succeeded,
                "delivery_detail": delivery.detail
            ])
        } catch {
            fputs("capture write failed after logging notification: \(error)\n", stderr)
            requestStop("write_error")
        }
        fputs("[COUNTER-STRIKE EGO CAPTURE] visible logging reminder: \(delivery.backend) \(delivery.detail)\n", stderr)
    }

    private func deliverReminderIfDue(now: UInt64, reason: String) {
        guard options.sessionLifecycle, focusEligible,
              let deadline = nextReminderMonotonicNS, now >= deadline else { return }
        deliverReminder(reason: reason, now: now)
        let intervalNS = UInt64(options.reminderMinutes * 60 * 1_000_000_000)
        nextReminderMonotonicNS = now + intervalNS
    }

    func heartbeat() {
        let now = monotonicNanoseconds()
        if let ownerPID = options.outerSessionOwnerPID {
            if kill(ownerPID, 0) != 0 && errno == ESRCH {
                requestStop("outer_monitor_terminated")
                return
            }
        } else if options.sessionLifecycle, let targetPID = boundTargetPID {
            let application = NSRunningApplication(processIdentifier: targetPID)
            if application == nil || application?.isTerminated == true {
                requestStop("target_terminated")
                return
            }
        }
        deliverReminderIfDue(now: now, reason: "interval_elapsed_while_focused")
    }

    func applicationTerminated(_ app: NSRunningApplication) {
        guard options.sessionLifecycle, app.processIdentifier == boundTargetPID else { return }
        try? writer.write([
            "type": "session_target_terminated",
            "sequence": nextSequence(),
            "received_monotonic_raw_ns": monotonicNanoseconds(),
            "wall_unix_ns": wallNanoseconds(),
            "pid": app.processIdentifier,
            "process_epoch": targetEpoch,
            "application_name": boundTargetName
        ])
        if options.outerSessionOwnerPID != nil {
            boundTargetPID = nil
            boundTargetName = ""
            targetBoundMonotonicNS = nil
            nextReminderMonotonicNS = nil
            refreshFocus(force: true)
        } else {
            requestStop("target_terminated")
        }
    }

    func refreshFocus(force: Bool = false) {
        let app = NSWorkspace.shared.frontmostApplication
        let bundleID = app?.bundleIdentifier ?? ""
        let name = (app?.localizedName ?? "").lowercased()
        let selectorEligible = matchesTargetSelector(app)
        let now = monotonicNanoseconds()
        var newlyBound = false
        if selectorEligible, let app, boundTargetPID == nil {
            newlyBound = bindTarget(app, now: now)
        }
        let eligible: Bool
        if options.sessionLifecycle, let targetPID = boundTargetPID {
            eligible = app?.processIdentifier == targetPID
        } else {
            eligible = selectorEligible
        }
        let wasEligible = focusEligible
        let signature = "\(app?.processIdentifier ?? -1)|\(bundleID)|\(name)|\(eligible)"
        guard force || signature != focusSignature else { return }
        focusSignature = signature
        focusEligible = eligible
        try? writer.write([
            "type": "focus_transition",
            "sequence": nextSequence(),
            "received_continuous_ticks": mach_continuous_time(),
            "received_monotonic_raw_ns": monotonicNanoseconds(),
            "wall_unix_ns": wallNanoseconds(),
            "pid": app?.processIdentifier ?? -1,
            "process_epoch": targetEpoch,
            "bundle_id": bundleID,
            "application_name": app?.localizedName ?? "",
            "eligible": eligible
        ])
        if newlyBound {
            deliverReminder(reason: "session_started", now: now)
        } else if eligible && !wasEligible {
            deliverReminderIfDue(now: now, reason: "deferred_until_refocus")
        }
        let status = eligible ? "CAPTURING PASSIVE INPUT EVENTS" : "PAUSED (TARGET NOT FOREGROUND)"
        fputs("[COUNTER-STRIKE EGO CAPTURE] \(status)\n", stderr)
    }

    func receive(_ value: IOHIDValue) {
        guard focusEligible else { return }
        let element = IOHIDValueGetElement(value)
        let device = IOHIDElementGetDevice(element)
        guard options.selectors.contains(where: { $0.matches(device) }) else { return }
        let source = "hid_value"
        let record: [String: Any] = [
            "type": source,
            "sequence": nextSequence(),
            "source_sequence": nextSourceSequence(source),
            "received_continuous_ticks": mach_continuous_time(),
            "received_absolute_ticks": mach_absolute_time(),
            "received_monotonic_raw_ns": monotonicNanoseconds(),
            "hid_timestamp_ticks": IOHIDValueGetTimeStamp(value),
            "device": deviceIdentity(device),
            "usage_page": IOHIDElementGetUsagePage(element),
            "usage": IOHIDElementGetUsage(element),
            "integer_value": IOHIDValueGetIntegerValue(value)
            ,"process_epoch": targetEpoch
        ]
        do {
            try writer.write(record)
        } catch {
            fputs("capture write failed: \(error)\n", stderr)
            requestStop("write_error")
        }
    }

    func receiveReport(
        result: IOReturn,
        device: IOHIDDevice,
        reportType: IOHIDReportType,
        reportID: UInt32,
        report: UnsafeMutablePointer<UInt8>,
        reportLength: CFIndex,
        timestamp: UInt64
    ) {
        guard result == kIOReturnSuccess else {
            callbackErrors += 1
            return
        }
        guard focusEligible, reportLength >= 0 else { return }
        guard options.selectors.contains(where: { $0.matches(device) }) else { return }
        let source = "hid_report"
        let bytes = Data(bytes: report, count: Int(reportLength))
        do {
            try writer.write([
                "type": source,
                "sequence": nextSequence(),
                "source_sequence": nextSourceSequence(source),
                "received_continuous_ticks": mach_continuous_time(),
                "received_absolute_ticks": mach_absolute_time(),
                "received_monotonic_raw_ns": monotonicNanoseconds(),
                "hid_report_timestamp_ticks": timestamp,
                "device": deviceIdentity(device),
                "report_type": reportType.rawValue,
                "report_id": reportID,
                "report_length": reportLength,
                "report_base64": bytes.base64EncodedString()
                ,"process_epoch": targetEpoch
            ])
        } catch {
            fputs("capture write failed: \(error)\n", stderr)
            requestStop("write_error")
        }
    }

    func receiveCGEvent(type: CGEventType, event: CGEvent) {
        if type == .tapDisabledByTimeout || type == .tapDisabledByUserInput {
            callbackErrors += 1
            if let cgEventTap { CGEvent.tapEnable(tap: cgEventTap, enable: true) }
            return
        }
        guard focusEligible else { return }
        let source = "cg_event"
        var record: [String: Any] = [
            "type": source,
            "sequence": nextSequence(),
            "source_sequence": nextSourceSequence(source),
            "received_continuous_ticks": mach_continuous_time(),
            "received_absolute_ticks": mach_absolute_time(),
            "received_monotonic_raw_ns": monotonicNanoseconds(),
            "cg_event_timestamp_ns_since_startup": event.timestamp,
            "cg_event_type": type.rawValue,
            "cg_event_type_name": cgEventName(type),
            "flags_raw": event.flags.rawValue,
            "attribution": "host_aggregate_unattributed"
            ,"process_epoch": targetEpoch
        ]
        switch type {
        case .mouseMoved, .leftMouseDragged, .rightMouseDragged, .otherMouseDragged:
            record["delta_x"] = event.getIntegerValueField(.mouseEventDeltaX)
            record["delta_y"] = event.getIntegerValueField(.mouseEventDeltaY)
        case .leftMouseDown, .leftMouseUp, .rightMouseDown, .rightMouseUp,
             .otherMouseDown, .otherMouseUp:
            record["button_number"] = event.getIntegerValueField(.mouseEventButtonNumber)
            record["click_state"] = event.getIntegerValueField(.mouseEventClickState)
        case .scrollWheel:
            record["scroll_point_delta_axis_1"] = event.getIntegerValueField(.scrollWheelEventPointDeltaAxis1)
            record["scroll_point_delta_axis_2"] = event.getIntegerValueField(.scrollWheelEventPointDeltaAxis2)
            record["scroll_fixed_delta_axis_1"] = event.getIntegerValueField(.scrollWheelEventFixedPtDeltaAxis1)
            record["scroll_fixed_delta_axis_2"] = event.getIntegerValueField(.scrollWheelEventFixedPtDeltaAxis2)
        case .keyDown, .keyUp, .flagsChanged:
            record["virtual_keycode"] = event.getIntegerValueField(.keyboardEventKeycode)
            record["is_autorepeat"] = event.getIntegerValueField(.keyboardEventAutorepeat)
        default:
            break
        }
        do {
            try writer.write(record)
        } catch {
            fputs("capture write failed: \(error)\n", stderr)
            requestStop("write_error")
        }
    }

    func flush() {
        do { try writer.flush() }
        catch {
            fputs("capture flush failed: \(error)\n", stderr)
            requestStop("write_error")
        }
    }

    func banner() {
        let devices = effectiveBackend == .cgEvent
            ? "unattributed-host-aggregate"
            : options.selectors.map(\.canonical).sorted().joined(separator: ", ")
        let state = focusEligible ? "ACTIVE" : "FOREGROUND-GATED/PAUSED"
        let lifecycle = options.outerSessionOwnerPID != nil
            ? "until outer monitor terminates; successive target PIDs"
            : options.sessionLifecycle
            ? "until target PID terminates"
            : "finite probe"
        fputs("[COUNTER-STRIKE EGO CAPTURE: \(state)] variant=\(options.gameVariant) backend=\(effectiveBackend.rawValue) devices=\(devices); \(lifecycle); Ctrl-C stops\n", stderr)
    }

    func requestStop(_ reason: String) {
        guard !stopped else { return }
        stopReason = reason
        stopped = true
        CFRunLoopStop(CFRunLoopGetMain())
    }

    func finish() throws {
        var expected: [String] = []
        if effectiveBackend.usesHIDValue { expected.append("hid_value") }
        if effectiveBackend.usesHIDReport { expected.append("hid_report") }
        if effectiveBackend.usesCGEvent { expected.append("cg_event") }
        let zeroSources = expected.filter { counts[$0, default: 0] == 0 }
        try writer.write([
            "type": "capture_end",
            "sequence": nextSequence(),
            "received_continuous_ticks": mach_continuous_time(),
            "received_monotonic_raw_ns": monotonicNanoseconds(),
            "wall_unix_ns": wallNanoseconds(),
            "reason": stopReason,
            "session_lifecycle": options.sessionLifecycle,
            "bound_target_pid": boundTargetPID ?? -1,
            "bound_target_name": boundTargetName,
            "last_process_epoch": targetEpoch,
            "logging_notification_count": notificationCount,
            "logging_notification_failures": notificationFailures,
            "backend_record_counts": counts,
            "zero_event_sources": zeroSources,
            "observation_status": zeroSources.isEmpty ? "events_observed" : "inconclusive_zero_events",
            "callback_errors": callbackErrors,
            "internal_dropped_records": 0,
            "writer_records_before_end": writer.recordsWritten,
            "writer_bytes_flushed_before_end": writer.bytesWritten
        ])
        try writer.close()
    }
}

private final class ReportBinding: @unchecked Sendable {
    let recorder: Recorder
    let device: IOHIDDevice
    let buffer: UnsafeMutablePointer<UInt8>
    let capacity: Int

    init(recorder: Recorder, device: IOHIDDevice) {
        self.recorder = recorder
        self.device = device
        let advertised = Int(numberProperty(device, kIOHIDMaxInputReportSizeKey as CFString) ?? 4096)
        capacity = max(1, min(advertised, 65_536))
        buffer = .allocate(capacity: capacity)
        buffer.initialize(repeating: 0, count: capacity)
    }

    deinit {
        buffer.deinitialize(count: capacity)
        buffer.deallocate()
    }
}

private func inputCallback(
    context: UnsafeMutableRawPointer?,
    result: IOReturn,
    sender: UnsafeMutableRawPointer?,
    value: IOHIDValue
) {
    guard result == kIOReturnSuccess, let context else { return }
    Unmanaged<Recorder>.fromOpaque(context).takeUnretainedValue().receive(value)
}

private func reportCallback(
    context: UnsafeMutableRawPointer?,
    result: IOReturn,
    sender: UnsafeMutableRawPointer?,
    type: IOHIDReportType,
    reportID: UInt32,
    report: UnsafeMutablePointer<UInt8>,
    reportLength: CFIndex,
    timestamp: UInt64
) {
    guard let context else { return }
    let binding = Unmanaged<ReportBinding>.fromOpaque(context).takeUnretainedValue()
    binding.recorder.receiveReport(
        result: result,
        device: binding.device,
        reportType: type,
        reportID: reportID,
        report: report,
        reportLength: reportLength,
        timestamp: timestamp
    )
}

private func cgEventCallback(
    proxy: CGEventTapProxy,
    type: CGEventType,
    event: CGEvent,
    userInfo: UnsafeMutableRawPointer?
) -> Unmanaged<CGEvent>? {
    if let userInfo {
        Unmanaged<Recorder>.fromOpaque(userInfo).takeUnretainedValue().receiveCGEvent(
            type: type, event: event
        )
    }
    return Unmanaged.passUnretained(event)
}

private func cgEventName(_ type: CGEventType) -> String {
    switch type {
    case .leftMouseDown: return "left_mouse_down"
    case .leftMouseUp: return "left_mouse_up"
    case .rightMouseDown: return "right_mouse_down"
    case .rightMouseUp: return "right_mouse_up"
    case .mouseMoved: return "mouse_moved"
    case .leftMouseDragged: return "left_mouse_dragged"
    case .rightMouseDragged: return "right_mouse_dragged"
    case .otherMouseDown: return "other_mouse_down"
    case .otherMouseUp: return "other_mouse_up"
    case .otherMouseDragged: return "other_mouse_dragged"
    case .scrollWheel: return "scroll_wheel"
    case .keyDown: return "key_down"
    case .keyUp: return "key_up"
    case .flagsChanged: return "flags_changed"
    case .tapDisabledByTimeout: return "tap_disabled_by_timeout"
    case .tapDisabledByUserInput: return "tap_disabled_by_user_input"
    default: return "other_\(type.rawValue)"
    }
}

private func cgEventMask() -> CGEventMask {
    let types: [CGEventType] = [
        .leftMouseDown, .leftMouseUp, .rightMouseDown, .rightMouseUp,
        .mouseMoved, .leftMouseDragged, .rightMouseDragged,
        .otherMouseDown, .otherMouseUp, .otherMouseDragged,
        .scrollWheel, .keyDown, .keyUp, .flagsChanged
    ]
    return types.reduce(0) { $0 | (CGEventMask(1) << $1.rawValue) }
}

private func hidAccessName(_ access: IOHIDAccessType) -> String {
    switch access {
    case kIOHIDAccessTypeGranted: return "granted"
    case kIOHIDAccessTypeDenied: return "denied"
    default: return "unknown"
    }
}

private func permissionDiagnostics() -> [String: Any] {
    let hid = IOHIDCheckAccess(kIOHIDRequestTypeListenEvent)
    return [
        "type": "input_permission_diagnostics",
        "schema": schemaVersion,
        "iohid_listen_access": hidAccessName(hid),
        "iohid_listen_access_raw": hid.rawValue,
        "cgevent_listen_preflight": CGPreflightListenEventAccess(),
        "effective_uid": geteuid(),
        "permission_request_performed": false,
        "note": "IOHID and CGEvent access are diagnosed independently; neither status predicts the other backend."
    ]
}

private func printJSONObject(_ object: [String: Any]) throws {
    let data = try JSONSerialization.data(withJSONObject: object, options: [.prettyPrinted, .sortedKeys])
    FileHandle.standardOutput.write(data)
    FileHandle.standardOutput.write(Data([0x0a]))
}

private func requestPermissions(_ kind: String) throws {
    var before = permissionDiagnostics()
    before["type"] = "input_permission_request_result"
    before["requested"] = kind
    if kind == "iohid" || kind == "both" {
        before["iohid_request_returned"] = IOHIDRequestAccess(kIOHIDRequestTypeListenEvent)
    }
    if kind == "cgevent" || kind == "both" {
        before["cgevent_request_returned"] = CGRequestListenEventAccess()
    }
    let hid = IOHIDCheckAccess(kIOHIDRequestTypeListenEvent)
    before["iohid_listen_access_after"] = hidAccessName(hid)
    before["cgevent_listen_preflight_after"] = CGPreflightListenEventAccess()
    before["permission_request_performed"] = true
    try printJSONObject(before)
}

private func managerWithAllDevices(open: Bool = true) throws -> IOHIDManager {
    let manager = IOHIDManagerCreate(kCFAllocatorDefault, IOOptionBits(kIOHIDOptionsTypeNone))
    IOHIDManagerSetDeviceMatching(manager, nil)
    if open {
        let status = IOHIDManagerOpen(manager, IOOptionBits(kIOHIDOptionsTypeNone))
        guard status == kIOReturnSuccess else {
            throw CaptureError.runtime("IOHIDManagerOpen failed: \(status)")
        }
    }
    return manager
}

private func managerForSelectors(_ selectors: Set<DeviceSelector>) throws -> IOHIDManager {
    let access = IOHIDCheckAccess(kIOHIDRequestTypeListenEvent)
    guard access == kIOHIDAccessTypeGranted else {
        throw CaptureError.runtime(
            "IOHID listen access is \(hidAccessName(access)); no IOHID callback stream will be "
            + "claimed. Run --diagnose-permissions or explicitly use "
            + "--consent --request-permission iohid."
        )
    }
    let manager = IOHIDManagerCreate(kCFAllocatorDefault, IOOptionBits(kIOHIDOptionsTypeNone))
    let matching: [[String: Any]] = selectors.map { selector in
        var dictionary: [String: Any] = [
            kIOHIDVendorIDKey: selector.vendorID,
            kIOHIDProductIDKey: selector.productID
        ]
        if let locationID = selector.locationID { dictionary[kIOHIDLocationIDKey] = locationID }
        return dictionary
    }
    IOHIDManagerSetDeviceMatchingMultiple(manager, matching as CFArray)
    let status = IOHIDManagerOpen(manager, IOOptionBits(kIOHIDOptionsTypeNone))
    guard status == kIOReturnSuccess else {
        let access = IOHIDCheckAccess(kIOHIDRequestTypeListenEvent)
        throw CaptureError.runtime(
            "IOHIDManagerOpen failed: \(status); IOHID listen access=\(hidAccessName(access)). "
            + "Run --diagnose-permissions or explicitly use --consent --request-permission iohid."
        )
    }
    return manager
}

private func printInventory(selectors: Set<DeviceSelector> = []) throws {
    let manager = try managerWithAllDevices(open: false)
    let matchingDevices = devicesMatching(selectors, from: allDevices(manager))
    let devices = matchingDevices.map { deviceIdentity($0, includeDescriptor: true) }.sorted {
        String(describing: $0) < String(describing: $1)
    }
    let object: [String: Any] = [
        "type": "hid_device_inventory",
        "schema": schemaVersion,
        "selectors": selectors.map(\.canonical).sorted(),
        "devices": devices
    ]
    try printJSONObject(object)
}

private func runCapture(_ options: Options) throws {
    var effectiveBackend = options.backend
    var manager: IOHIDManager?
    if options.backend.usesHID || options.backend == .auto {
        do {
            manager = try managerForSelectors(options.selectors)
            if options.backend == .auto { effectiveBackend = .hidDual }
        } catch {
            if options.backend == .auto {
                fputs("IOHID unavailable (\(error)); trying listen-only CGEvent fallback\n", stderr)
                effectiveBackend = .cgEvent
            } else {
                throw error
            }
        }
    }
    defer {
        if let manager { IOHIDManagerClose(manager, IOOptionBits(kIOHIDOptionsTypeNone)) }
    }

    let selected: [IOHIDDevice]
    if effectiveBackend.usesHID, let manager {
        // These exact references belong to the manager that is opened and later
        // scheduled. Per-device report callbacks must never be registered on
        // lookalike IOHIDDevice objects copied from a separate inventory manager.
        selected = devicesMatching(options.selectors, from: allDevices(manager))
    } else {
        // CGEvent has no physical attribution. Inventory refs are metadata only
        // and are never registered for callbacks or scheduled.
        if options.selectors.isEmpty {
            selected = []
        } else {
            let inventoryManager = try managerWithAllDevices(open: false)
            selected = devicesMatching(options.selectors, from: allDevices(inventoryManager))
        }
    }
    if effectiveBackend.usesHID {
        let missing = options.selectors.filter { requested in
            !selected.contains(where: requested.matches)
        }
        guard missing.isEmpty else {
            let names = missing.map(\.canonical).sorted().joined(separator: ", ")
            throw CaptureError.runtime("selected HID device(s) are not present: \(names); use --list-devices")
        }
    }

    let recorder = try Recorder(
        options: options, effectiveBackend: effectiveBackend, selectedDevices: selected
    )
    try recorder.begin()

    let context = Unmanaged.passUnretained(recorder).toOpaque()
    var reportBindings: [ReportBinding] = []
    if let manager {
        if effectiveBackend.usesHIDValue {
            IOHIDManagerRegisterInputValueCallback(manager, inputCallback, context)
        }
        if effectiveBackend.usesHIDReport {
            reportBindings = selected.map { ReportBinding(recorder: recorder, device: $0) }
            for binding in reportBindings {
                IOHIDDeviceRegisterInputReportWithTimeStampCallback(
                    binding.device,
                    binding.buffer,
                    binding.capacity,
                    reportCallback,
                    Unmanaged.passUnretained(binding).toOpaque()
                )
            }
        }
        IOHIDManagerScheduleWithRunLoop(
            manager, CFRunLoopGetMain(), CFRunLoopMode.defaultMode.rawValue
        )
    }

    var tapSource: CFRunLoopSource?
    if effectiveBackend.usesCGEvent {
        guard let tap = CGEvent.tapCreate(
            tap: .cgSessionEventTap,
            place: .headInsertEventTap,
            options: .listenOnly,
            eventsOfInterest: cgEventMask(),
            callback: cgEventCallback,
            userInfo: context
        ) else {
            recorder.requestStop("backend_initialization_failed")
            try? recorder.finish()
            throw CaptureError.runtime(
                "listen-only CGEvent tap creation failed; cgevent preflight="
                + "\(CGPreflightListenEventAccess()). Run --diagnose-permissions or explicitly "
                + "use --consent --request-permission cgevent."
            )
        }
        recorder.cgEventTap = tap
        let source = CFMachPortCreateRunLoopSource(kCFAllocatorDefault, tap, 0)
        tapSource = source
        CFRunLoopAddSource(CFRunLoopGetMain(), source, .commonModes)
        CGEvent.tapEnable(tap: tap, enable: true)
    }
    // Arm foreground eligibility only after every requested callback source is
    // active. This avoids a target-bound interval before HID delivery exists.
    recorder.arm()
    recorder.banner()
    defer {
        if let manager {
            IOHIDManagerUnscheduleFromRunLoop(
                manager, CFRunLoopGetMain(), CFRunLoopMode.defaultMode.rawValue
            )
        }
        if let tapSource { CFRunLoopRemoveSource(CFRunLoopGetMain(), tapSource, .commonModes) }
    }

    let workspaceNotifications = NSWorkspace.shared.notificationCenter
    let activationObserver = workspaceNotifications.addObserver(
        forName: NSWorkspace.didActivateApplicationNotification,
        object: nil,
        queue: .main
    ) { _ in recorder.refreshFocus() }
    let terminationObserver = workspaceNotifications.addObserver(
        forName: NSWorkspace.didTerminateApplicationNotification,
        object: nil,
        queue: .main
    ) { notification in
        if let application = notification.userInfo?[NSWorkspace.applicationUserInfoKey]
            as? NSRunningApplication {
            recorder.applicationTerminated(application)
        }
    }
    defer {
        workspaceNotifications.removeObserver(activationObserver)
        workspaceNotifications.removeObserver(terminationObserver)
    }
    let focusTimer = Timer(timeInterval: 0.1, repeats: true) { _ in recorder.refreshFocus() }
    RunLoop.main.add(focusTimer, forMode: .common)
    let bannerTimer = Timer(timeInterval: 30, repeats: true) { _ in recorder.banner() }
    RunLoop.main.add(bannerTimer, forMode: .common)
    let flushTimer = Timer(timeInterval: 0.25, repeats: true) { _ in recorder.flush() }
    RunLoop.main.add(flushTimer, forMode: .common)
    let heartbeatTimer = Timer(timeInterval: 1, repeats: true) { _ in recorder.heartbeat() }
    RunLoop.main.add(heartbeatTimer, forMode: .common)
    var durationTimer: Timer?
    if let duration = options.durationSeconds {
        let timer = Timer(timeInterval: duration, repeats: false) { _ in
            recorder.requestStop("duration_limit")
        }
        durationTimer = timer
        RunLoop.main.add(timer, forMode: .common)
    }

    signal(SIGINT, SIG_IGN)
    signal(SIGTERM, SIG_IGN)
    let interruptSource = DispatchSource.makeSignalSource(signal: SIGINT, queue: .main)
    interruptSource.setEventHandler { recorder.requestStop("sigint") }
    interruptSource.resume()
    let terminateSource = DispatchSource.makeSignalSource(signal: SIGTERM, queue: .main)
    terminateSource.setEventHandler { recorder.requestStop("sigterm") }
    terminateSource.resume()

    withExtendedLifetime(reportBindings) { CFRunLoopRun() }
    focusTimer.invalidate()
    bannerTimer.invalidate()
    flushTimer.invalidate()
    heartbeatTimer.invalidate()
    durationTimer?.invalidate()
    interruptSource.cancel()
    terminateSource.cancel()
    try recorder.finish()
}

do {
    let options = try Options.parse(Array(CommandLine.arguments.dropFirst()))
    if options.listDevices {
        try printInventory(selectors: options.selectors)
    } else if options.diagnosePermissions {
        try printJSONObject(permissionDiagnostics())
    } else if let request = options.permissionRequest {
        try requestPermissions(request)
    } else {
        try runCapture(options)
    }
} catch let error as CaptureError {
    fputs("error: \(error)\n\n\(Options.usage)\n", stderr)
    exit(2)
} catch {
    fputs("error: \(error)\n", stderr)
    exit(1)
}
