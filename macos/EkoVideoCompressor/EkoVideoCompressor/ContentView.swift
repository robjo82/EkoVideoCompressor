import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @EnvironmentObject private var engine: EngineProcess
    @EnvironmentObject private var queue: QueueStore
    @EnvironmentObject private var settings: SettingsStore
    @State private var selectedTab = "queue"
    @State private var showingSettings = false

    var body: some View {
        NavigationSplitView {
            List(selection: $selectedTab) {
                Label("File d'attente", systemImage: "text.line.first.and.arrowtriangle.forward")
                    .tag("queue")
                Label("Bibliotheque", systemImage: "tray.full")
                    .tag("library")
                Label("Modeles", systemImage: "shippingbox")
                    .tag("models")
            }
            .navigationSplitViewColumnWidth(min: 180, ideal: 210)
        } detail: {
            VStack(spacing: 0) {
                HeaderView(showingSettings: $showingSettings)
                Divider()
                Group {
                    switch selectedTab {
                    case "library":
                        LibraryView()
                    case "models":
                        ModelsView()
                    default:
                        QueueView()
                    }
                }
                Divider()
                StatusBarView()
            }
        }
        .sheet(isPresented: $showingSettings) {
            SettingsView()
                .environmentObject(settings)
        }
    }
}

struct HeaderView: View {
    @EnvironmentObject private var engine: EngineProcess
    @Binding var showingSettings: Bool

    var body: some View {
        HStack(spacing: 14) {
            Image(systemName: "link.circle.fill")
                .font(.system(size: 34))
                .foregroundStyle(.teal)
            VStack(alignment: .leading, spacing: 2) {
                Text("EkoVideo Compressor")
                    .font(.title2.bold())
                Text("Compression et transcription locale")
                    .foregroundStyle(.secondary)
            }
            Spacer()
            Button("Exporter logs") {
                engine.run(arguments: EngineProcess.defaultPythonArguments(["export-logs"]))
            }
            Button("Reglages") {
                showingSettings = true
            }
            .keyboardShortcut(",", modifiers: .command)
        }
        .padding(18)
    }
}

struct QueueView: View {
    @EnvironmentObject private var engine: EngineProcess
    @EnvironmentObject private var queue: QueueStore
    @EnvironmentObject private var settings: SettingsStore

    var body: some View {
        VStack(spacing: 12) {
            DropTargetView()
            List {
                ForEach(queue.items) { item in
                    HStack {
                        Image(systemName: "line.3.horizontal")
                            .foregroundStyle(.secondary)
                        VStack(alignment: .leading) {
                            Text(item.sourceURL.lastPathComponent)
                                .lineLimit(2)
                            Text(item.status)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                        Spacer()
                    }
                    .padding(.vertical, 4)
                }
                .onMove(perform: queue.move)
                .onDelete(perform: queue.remove)
            }
            .listStyle(.inset)
            HStack {
                Button("Ajouter") { chooseFiles() }
                Spacer()
                Button(engine.isRunning ? "Annuler" : "Lancer la file") {
                    engine.isRunning ? engine.cancel() : runFirstJob()
                }
                .buttonStyle(.borderedProminent)
                .disabled(queue.items.isEmpty && !engine.isRunning)
            }
        }
        .padding()
    }

    private func chooseFiles() {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = true
        panel.canChooseDirectories = false
        if panel.runModal() == .OK {
            queue.add(urls: panel.urls)
        }
    }

    private func runFirstJob() {
        guard let item = queue.items.first else { return }
        let request = JobRequest(
            source_path: item.sourceURL.path,
            workspace_dir: "",
            output_dir: settings.outputDir,
            mode: "compress_transcribe",
            profile: "Reunion equilibree",
            compression_settings: CompressionSettings(),
            transcription_settings: TranscriptionSettings(
                model: settings.whisperModel,
                diarization_enabled: settings.diarizationEnabled,
                hf_token: settings.hfToken,
                audio_recheck_enabled: settings.audioRecheckEnabled
            ),
            glossary_terms: settings.glossaryTerms,
            speaker_overrides: [:],
            technical_terms: [],
            rerun_steps: []
        )
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("ekovideo-job.json")
        do {
            let data = try JSONEncoder().encode(request)
            try data.write(to: url)
            engine.run(arguments: EngineProcess.defaultPythonArguments(["run-job", "--request", url.path]))
        } catch {
            engine.lastError = error.localizedDescription
        }
    }
}

struct DropTargetView: View {
    @EnvironmentObject private var queue: QueueStore
    @State private var isTargeted = false

    var body: some View {
        RoundedRectangle(cornerRadius: 8)
            .strokeBorder(style: StrokeStyle(lineWidth: 1.5, dash: [7]))
            .foregroundStyle(isTargeted ? .teal : .secondary)
            .frame(minHeight: 120)
            .overlay {
                VStack(spacing: 8) {
                    Image(systemName: "square.and.arrow.down")
                        .font(.title)
                    Text("Deposez vos videos ou audios")
                        .font(.headline)
                    Text("Vous pouvez continuer a ajouter des fichiers pendant un traitement.")
                        .foregroundStyle(.secondary)
                }
            }
            .onDrop(of: [.fileURL], isTargeted: $isTargeted) { providers in
                Task {
                    var urls: [URL] = []
                    for provider in providers {
                        if let data = try? await provider.loadItem(forTypeIdentifier: "public.file-url") as? Data,
                           let value = String(data: data, encoding: .utf8),
                           let url = URL(string: value) {
                            urls.append(url)
                        }
                    }
                    await MainActor.run { queue.add(urls: urls) }
                }
                return true
            }
    }
}

struct LibraryView: View {
    @EnvironmentObject private var engine: EngineProcess

    var body: some View {
        VStack(alignment: .leading) {
            HStack {
                Text("Bibliotheque")
                    .font(.title2.bold())
                Spacer()
                Button("Actualiser") {
                    engine.run(arguments: EngineProcess.defaultPythonArguments(["library-list"]))
                }
            }
            EventListView()
        }
        .padding()
    }
}

struct ModelsView: View {
    @EnvironmentObject private var engine: EngineProcess

    var body: some View {
        VStack(alignment: .leading) {
            HStack {
                Text("Modeles locaux")
                    .font(.title2.bold())
                Spacer()
                Button("Actualiser") {
                    engine.run(arguments: EngineProcess.defaultPythonArguments(["model-list", "--jsonl"]))
                }
            }
            EventListView()
        }
        .padding()
    }
}

struct SettingsView: View {
    @EnvironmentObject private var settings: SettingsStore
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        Form {
            TextField("Dossier sortie", text: $settings.outputDir)
            TextEditor(text: $settings.glossary)
                .frame(minHeight: 120)
            TextField("Modele Whisper", text: $settings.whisperModel)
            SecureField("Token Hugging Face", text: $settings.hfToken)
            Toggle("Detection des locuteurs", isOn: $settings.diarizationEnabled)
            Toggle("Reecoute IA multimodale", isOn: $settings.audioRecheckEnabled)
            HStack {
                Spacer()
                Button("Fermer") { dismiss() }
                    .keyboardShortcut(.defaultAction)
            }
        }
        .padding()
        .frame(minWidth: 560, minHeight: 420)
    }
}

struct StatusBarView: View {
    @EnvironmentObject private var engine: EngineProcess

    var body: some View {
        HStack {
            if engine.isRunning {
                ProgressView()
                    .controlSize(.small)
            }
            Text(engine.lastError ?? engine.events.last?.message ?? "Pret.")
                .lineLimit(1)
            Spacer()
        }
        .font(.callout)
        .padding(.horizontal, 18)
        .padding(.vertical, 10)
    }
}

struct EventListView: View {
    @EnvironmentObject private var engine: EngineProcess

    var body: some View {
        List(engine.events) { event in
            VStack(alignment: .leading) {
                Text(event.event.rawValue)
                    .font(.headline)
                Text(event.message ?? event.path ?? event.step ?? "")
                    .foregroundStyle(.secondary)
            }
        }
    }
}
