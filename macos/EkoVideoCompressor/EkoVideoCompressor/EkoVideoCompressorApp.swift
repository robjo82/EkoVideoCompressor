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

    init() {
        let args = CommandLine.arguments
        if args.contains("--smoke-test") || args.contains("--startup-smoke-test") {
            print("EkoVideoCompressor SwiftUI smoke test ok")
            Foundation.exit(0)
        }
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(engine)
                .environmentObject(queue)
                .environmentObject(settings)
                .environmentObject(library)
                .environmentObject(models)
                .environmentObject(updater)
                .environmentObject(odoo)
                .frame(minWidth: 1180, minHeight: 760)
                .onAppear {
                    updater.setSettings(settings)
                    odoo.bind(settings)
                    Task {
                        await updater.checkUpdates(proactive: true)
                    }
                }
        }
        .windowStyle(.titleBar)
        .commands {
            CommandGroup(replacing: .newItem) {}
        }
    }
}
