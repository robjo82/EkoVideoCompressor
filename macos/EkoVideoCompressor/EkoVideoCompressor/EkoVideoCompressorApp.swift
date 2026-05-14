import SwiftUI

@main
struct EkoVideoCompressorApp: App {
    @StateObject private var engine = EngineProcess()
    @StateObject private var queue = QueueStore()
    @StateObject private var settings = SettingsStore()

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
                .frame(minWidth: 1180, minHeight: 760)
        }
        .windowStyle(.titleBar)
        .commands {
            CommandGroup(replacing: .newItem) {}
        }
    }
}
