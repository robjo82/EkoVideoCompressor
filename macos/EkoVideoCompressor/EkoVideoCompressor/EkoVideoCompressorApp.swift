import SwiftUI

@main
struct EkoVideoCompressorApp: App {
    @StateObject private var engine = EngineProcess()
    @StateObject private var queue = QueueStore()
    @StateObject private var settings = SettingsStore()
    @StateObject private var library = LibraryStore()
    @StateObject private var models = ModelStore()
    @StateObject private var updater = UpdateStore()
    @StateObject private var odoo = OdooStore()
    @StateObject private var energy = EnergyMonitor()
    @StateObject private var pyannote = PyannoteStatusStore()
    @StateObject private var deps = DepsStore()
    @StateObject private var cloudUsage = CloudUsageStore()

    init() {
        let args = CommandLine.arguments
        if args.contains("--smoke-test") || args.contains("--startup-smoke-test") {
            print("EkoVideoCompressor SwiftUI smoke test ok")
            Foundation.exit(0)
        }
    }

    var body: some Scene {
        WindowGroup {
            RootView()
                .environmentObject(engine)
                .environmentObject(queue)
                .environmentObject(settings)
                .environmentObject(library)
                .environmentObject(models)
                .environmentObject(updater)
                .environmentObject(odoo)
                .environmentObject(energy)
                .environmentObject(pyannote)
                .environmentObject(deps)
                .environmentObject(cloudUsage)
                .frame(minWidth: 1180, minHeight: 760)
                .onAppear {
                    updater.setSettings(settings)
                    odoo.bind(settings)
                    pyannote.bind(settings)
                    Task {
                        await updater.checkUpdates(proactive: true)
                    }
                    // PR AT — keep the managed ML venv from rotting:
                    // enforce version floors at launch (only runs pip
                    // when something is actually below floor). Runs in
                    // the background so it never blocks the UI; progress
                    // surfaces in Réglages + a banner.
                    Task {
                        await deps.enforceFloors()
                    }
                }
        }
        .windowStyle(.titleBar)
        .commands {
            CommandGroup(replacing: .newItem) {}
        }
    }
}

/// Splash gate: shows a launch loader until the library has finished
/// its first refresh. Without this, the user lands on a blank Bibliothèque
/// for a beat or two while SQLite is cold — looked broken on a fresh open
/// after a reboot.
struct RootView: View {
    @EnvironmentObject private var library: LibraryStore
    @EnvironmentObject private var models: ModelStore
    @State private var isReady = false
    /// Once-only flag for the post-PR #30 heal pass. Bump the suffix
    /// when introducing a new migration that needs to run again.
    /// ``@AppStorage`` persists this across launches automatically.
    @AppStorage("librarySpeakerMapsRepaired_v1") private var speakerMapsRepaired = false
    /// PR AY — once-only "Poids" heal: sources freed on pre-PR-AS app
    /// versions left ``total_bytes`` frozen at the pre-deletion size.
    @AppStorage("libraryTotalBytesRepaired_v1") private var totalBytesRepaired = false

    var body: some View {
        ZStack {
            if isReady {
                ContentView()
                    .transition(.opacity)
            } else {
                LaunchSplashView()
                    .transition(.opacity)
            }
        }
        .animation(.easeInOut(duration: 0.25), value: isReady)
        .task {
            guard !isReady else { return }
            let started = Date()
            // Block on the library; models can keep loading in the
            // background since the Bibliothèque is what the user lands
            // on first.
            async let libraryLoad: Void = library.refresh()
            async let modelsLoad: Void = models.refresh()
            _ = await libraryLoad
            // Keep the splash up for a minimum beat so it doesn't
            // strobe on warm-cache launches — a 100ms flash reads as
            // a glitch, not a load screen.
            let elapsed = Date().timeIntervalSince(started)
            let minimumSplash = 0.4
            if elapsed < minimumSplash {
                try? await Task.sleep(nanoseconds: UInt64((minimumSplash - elapsed) * 1_000_000_000))
            }
            isReady = true
            // Don't block on the models refresh — but wait for it in
            // the background so any error surfaces in the Models tab
            // without the user seeing a half-loaded list mid-render.
            _ = await modelsLoad
            // One-shot heal pass for ``speaker_map_json`` drift that
            // PR #30 left on disk. Runs after the user has the UI
            // (so it never blocks launch), once per app version,
            // and refreshes the library when it touched anything so
            // the displayed speaker names reflect the cleanup
            // without a manual reload.
            if !speakerMapsRepaired {
                if let summary = await library.repairSpeakerMaps(),
                   (summary["repaired"] ?? 0) > 0 {
                    await library.refresh()
                }
                speakerMapsRepaired = true
            }
            // PR AY — refresh stale "Poids" snapshots (sources freed
            // on pre-PR-AS versions kept their pre-deletion size).
            // Runs after the UI is up; refreshes the list only when
            // something actually changed.
            if !totalBytesRepaired {
                if let summary = await library.recomputeSizes(),
                   (summary["updated"] ?? 0) > 0 {
                    await library.refresh()
                }
                totalBytesRepaired = true
            }
        }
    }
}

struct LaunchSplashView: View {
    var body: some View {
        VStack(spacing: 18) {
            Image(systemName: "waveform.circle.fill")
                .font(.system(size: 64, weight: .light))
                .foregroundStyle(.tint)
            Text("EkoVideoCompressor")
                .font(.title2.weight(.semibold))
            ProgressView()
                .progressViewStyle(.circular)
                .controlSize(.small)
            Text("Chargement de la bibliothèque…")
                .font(.callout)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color(nsColor: .windowBackgroundColor))
    }
}
