import Foundation
import SwiftUI

struct QueueItem: Identifiable, Equatable {
    let id = UUID()
    var sourceURL: URL
    var status: String = "En attente"
}

@MainActor
final class QueueStore: ObservableObject {
    @Published var items: [QueueItem] = []

    func add(urls: [URL]) {
        let existing = Set(items.map(\.sourceURL))
        for url in urls where !existing.contains(url) {
            items.append(QueueItem(sourceURL: url))
        }
    }

    func move(from source: IndexSet, to destination: Int) {
        items.move(fromOffsets: source, toOffset: destination)
    }

    func remove(at offsets: IndexSet) {
        items.remove(atOffsets: offsets)
    }
}

@MainActor
final class SettingsStore: ObservableObject {
    @AppStorage("outputDir") var outputDir = "\(NSHomeDirectory())/Desktop"
    @AppStorage("glossary") var glossary = ""
    @AppStorage("hfToken") var hfToken = ""
    @AppStorage("whisperModel") var whisperModel = "mlx-community/whisper-large-v3-turbo"
    @AppStorage("audioRecheckEnabled") var audioRecheckEnabled = false
    @AppStorage("diarizationEnabled") var diarizationEnabled = false

    var glossaryTerms: [String] {
        glossary
            .split(whereSeparator: \.isNewline)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
    }
}

@MainActor
final class LibraryStore: ObservableObject {
    @Published var rows: [[String: JSONValue]] = []

    func refresh(using engine: EngineProcess) {
        engine.run(arguments: EngineProcess.defaultPythonArguments(["library-list"]))
    }
}

@MainActor
final class ModelStore: ObservableObject {
    @Published var models: [ModelRow] = []

    func load(using engine: EngineProcess) {
        engine.run(arguments: EngineProcess.defaultPythonArguments(["model-list", "--jsonl"]))
    }
}

struct ModelRow: Codable, Identifiable {
    var id: String
    var family: String
    var label: String
    var cached: Bool
    var cache_dir: String
}
