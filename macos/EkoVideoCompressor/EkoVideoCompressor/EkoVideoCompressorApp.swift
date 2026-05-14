import SwiftUI

@main
struct EkoVideoCompressorApp: App {
    @StateObject private var engine = EngineProcess()
    @StateObject private var queue = QueueStore()
    @StateObject private var settings = SettingsStore()

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
