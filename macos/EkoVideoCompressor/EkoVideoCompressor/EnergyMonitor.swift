import Foundation
import IOKit
import IOKit.ps

/// Tracks AC adapter state + battery percentage and surfaces a
/// pre-baked "should we let the user start a heavy job?" verdict.
///
/// The transcription pipeline is brutal on the SoC — the ``max``
/// preset can run multi-hour large-v3 passes that drain the battery
/// in under an hour. Without this monitor the user could kick off
/// such a job at 12 % battery and watch their laptop die mid-run,
/// leaving the workspace half-written. We block that by:
///
/// * Disabling the "Maximale" preset toggle when the laptop is on
///   battery only.
/// * Disabling the Run button below ``minBatteryToRunPercent`` (40 %
///   by default) so even ``balanced`` runs can't start on a near-dead
///   battery.
/// * Posting an interruption signal on the ``interruptionRequested``
///   stream when the user unplugs mid-run while in Max — the caller
///   listens to that and cancels the engine subprocess gracefully.
///
/// Read-only proxy over ``IOPSCopyPowerSourcesInfo``. We keep a
/// `CFRunLoopSource` alive for the whole app lifetime so we can
/// react to plug / unplug events without polling — IOKit fires the
/// callback in milliseconds.
@MainActor
final class EnergyMonitor: ObservableObject {
    /// Battery percentage threshold below which every transcription
    /// kick-off is blocked, not just Max. Tuned high enough that the
    /// 40-minute average run on an M1 Air will finish on residual
    /// charge if the adapter dies right after launch.
    static let minBatteryToRunPercent: Int = 40

    @Published private(set) var isOnAC: Bool = true
    /// ``nil`` for desktops with no battery (Mac mini / Studio) —
    /// the UI treats that as "always on AC", which mirrors macOS
    /// behaviour where there's no power-source UI either.
    @Published private(set) var batteryPercent: Int? = nil
    @Published private(set) var hasBattery: Bool = false

    /// Async stream the queue listener subscribes to. We post a value
    /// every time the laptop transitions from AC → battery, regardless
    /// of preset — the listener decides whether to act based on the
    /// active job's quality preset.
    let unplugSignal: AsyncStream<Void>
    private let unplugContinuation: AsyncStream<Void>.Continuation

    private var runLoopSource: CFRunLoopSource?
    private var lastKnownOnAC: Bool = true

    init() {
        var continuation: AsyncStream<Void>.Continuation!
        self.unplugSignal = AsyncStream { continuation = $0 }
        self.unplugContinuation = continuation
        refresh()
        lastKnownOnAC = isOnAC
        startObserving()
    }

    /// True when a fresh ``max`` preset run is allowed to start.
    var allowsMaxPreset: Bool {
        // Desktops with no battery always pass.
        return !hasBattery || isOnAC
    }

    /// True when a fresh transcription of any preset is allowed to
    /// start (independent of preset choice).
    var allowsTranscriptionStart: Bool {
        if !hasBattery { return true }
        if isOnAC { return true }
        // On battery: require the safety threshold so the user
        // can still kick off short fast-mode runs while travelling.
        return (batteryPercent ?? 0) >= Self.minBatteryToRunPercent
    }

    /// One-line caption for the disabled-state tooltip / banner.
    /// Empty when nothing is blocking.
    var blockingReason: String {
        if !hasBattery || isOnAC { return "" }
        let pct = batteryPercent ?? 0
        if pct < Self.minBatteryToRunPercent {
            return "Batterie à \(pct) %. Branchez l'alimentation pour lancer une transcription (seuil minimum \(Self.minBatteryToRunPercent) %)."
        }
        return ""
    }

    var maxPresetBlockedReason: String {
        if !hasBattery || isOnAC { return "" }
        return "Le mode Maximale nécessite l'alimentation secteur — il sature le SoC pendant des heures."
    }

    func refresh() {
        guard let snapshotRef = IOPSCopyPowerSourcesInfo()?.takeRetainedValue() else {
            // No IOKit data — best-effort assume desktop on AC so we
            // don't lock the user out of their app.
            isOnAC = true
            batteryPercent = nil
            hasBattery = false
            return
        }
        let snapshot = snapshotRef as CFTypeRef
        guard let listRef = IOPSCopyPowerSourcesList(snapshot)?.takeRetainedValue() else {
            isOnAC = true
            batteryPercent = nil
            hasBattery = false
            return
        }
        let sources = listRef as Array
        var foundBattery = false
        var onAC = true
        var pct: Int? = nil
        for source in sources {
            guard let descRef = IOPSGetPowerSourceDescription(snapshot, source as CFTypeRef)?.takeUnretainedValue() else {
                continue
            }
            guard let info = descRef as? [String: Any] else { continue }
            // Internal battery is the only source the user cares
            // about for "am I on adapter?" — external UPS et al are
            // not relevant to a laptop transcription workflow.
            let type = info[kIOPSTypeKey as String] as? String ?? ""
            if type == kIOPSInternalBatteryType as String {
                foundBattery = true
                if let current = info[kIOPSCurrentCapacityKey as String] as? Int,
                   let maxCap = info[kIOPSMaxCapacityKey as String] as? Int, maxCap > 0 {
                    pct = Int((Double(current) / Double(maxCap) * 100.0).rounded())
                }
                if let state = info[kIOPSPowerSourceStateKey as String] as? String {
                    onAC = (state == (kIOPSACPowerValue as String))
                }
            }
        }
        hasBattery = foundBattery
        isOnAC = foundBattery ? onAC : true
        batteryPercent = pct
        // Edge detect on AC → battery transition. Yields a single
        // value through the stream so the queue listener can decide
        // whether to interrupt based on the running job's preset.
        if lastKnownOnAC && !isOnAC {
            unplugContinuation.yield(())
        }
        lastKnownOnAC = isOnAC
    }

    private func startObserving() {
        let context = Unmanaged.passUnretained(self).toOpaque()
        let callback: IOPowerSourceCallbackType = { rawContext in
            guard let rawContext else { return }
            let monitor = Unmanaged<EnergyMonitor>.fromOpaque(rawContext).takeUnretainedValue()
            // The IOKit callback fires on a CF run loop thread that
            // isn't main. We hop back on the main actor before
            // touching @Published state.
            Task { @MainActor in
                monitor.refresh()
            }
        }
        guard let source = IOPSNotificationCreateRunLoopSource(callback, context)?.takeRetainedValue() else {
            return
        }
        runLoopSource = source
        CFRunLoopAddSource(CFRunLoopGetMain(), source, .defaultMode)
    }
}
