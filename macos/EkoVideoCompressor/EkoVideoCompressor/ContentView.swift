import SwiftUI
import AppKit
import UniformTypeIdentifiers

enum AppSection: String, CaseIterable, Hashable {
    case queue
    case library
    case models

    var title: String {
        switch self {
        case .queue: "Traitements"
        case .library: "Bibliotheque"
        case .models: "Modeles"
        }
    }

    var symbol: String {
        switch self {
        case .queue: "play.rectangle"
        case .library: "tray.full"
        case .models: "shippingbox"
        }
    }
}

struct ContentView: View {
    @EnvironmentObject private var engine: EngineProcess
    @EnvironmentObject private var queue: QueueStore
    @EnvironmentObject private var settings: SettingsStore
    @State private var selectedSection: AppSection = .queue
    @State private var showingSettings = false
    @State private var showingRunSetup = false

    var body: some View {
        NavigationSplitView {
            SidebarView(selection: $selectedSection)
        } detail: {
            Group {
                switch selectedSection {
                case .queue:
                    ProcessingWorkspaceView()
                case .library:
                    LibraryView()
                case .models:
                    ModelsView()
                }
            }
            .navigationTitle(selectedSection.title)
            .toolbar {
                ToolbarItemGroup(placement: .primaryAction) {
                    Button {
                        chooseFiles()
                    } label: {
                        Label("Ajouter", systemImage: "plus")
                    }

                    Button {
                        showingRunSetup = true
                    } label: {
                        Label(queue.isBatchRunning ? "En cours" : "Lancer la file", systemImage: "play.fill")
                    }
                    .disabled(queue.items.isEmpty || queue.isBatchRunning)

                    Button {
                        showingSettings = true
                    } label: {
                        Label("Reglages", systemImage: "gearshape")
                    }
                    .keyboardShortcut(",", modifiers: .command)
                }
            }
        }
        .sheet(isPresented: $showingSettings) {
            SettingsView()
                .environmentObject(settings)
        }
        .sheet(isPresented: $showingRunSetup) {
            RunSetupView {
                showingRunSetup = false
                Task { await runQueue() }
            }
            .environmentObject(settings)
            .environmentObject(engine)
        }
    }

    private func chooseFiles() {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = true
        panel.canChooseDirectories = false
        if panel.runModal() == .OK {
            queue.add(urls: panel.urls)
            selectedSection = .queue
        }
    }

    private func runQueue() async {
        guard !queue.items.isEmpty else { return }
        queue.isBatchRunning = true
        queue.resetPending()
        let items = queue.items

        for item in items {
            if Task.isCancelled { break }
            queue.update(item.id, status: "En cours", progress: 0)
            let exitCode = await runJob(item)
            if exitCode == 0 {
                queue.update(item.id, status: "Termine", progress: 100)
            } else {
                queue.update(item.id, status: "Erreur", progress: 0)
            }
        }
        queue.isBatchRunning = false
    }

    private func runJob(_ item: QueueItem) async -> Int32 {
        let request = JobRequest(
            source_path: item.sourceURL.path,
            workspace_dir: "",
            output_dir: settings.outputDir,
            mode: settings.processingMode,
            profile: "Reunion equilibree",
            compression_settings: CompressionSettings(),
            transcription_settings: TranscriptionSettings(
                model: settings.whisperModel,
                output_format: settings.outputFormat,
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
            return await engine.runAndWait(arguments: EngineProcess.defaultPythonArguments(["run-job", "--request", url.path]))
        } catch {
            engine.lastError = error.localizedDescription
            return -1
        }
    }
}

struct SidebarView: View {
    @Binding var selection: AppSection

    var body: some View {
        List(AppSection.allCases, id: \.self, selection: $selection) { section in
            Label(section.title, systemImage: section.symbol)
        }
        .listStyle(.sidebar)
        .navigationSplitViewColumnWidth(min: 180, ideal: 220)
    }
}

struct ProcessingWorkspaceView: View {
    @EnvironmentObject private var queue: QueueStore
    @EnvironmentObject private var engine: EngineProcess

    var body: some View {
        VStack(spacing: 0) {
            WorkflowHeaderView()
            Divider()
            QueueColumnView()
            Divider()
            StatusBarView()
        }
    }
}

struct WorkflowHeaderView: View {
    @EnvironmentObject private var queue: QueueStore
    @EnvironmentObject private var settings: SettingsStore

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(alignment: .firstTextBaseline) {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Preparer une transcription")
                        .font(.largeTitle.bold())
                    Text("Ajoutez les fichiers, renseignez le vocabulaire utile, puis lancez le traitement.")
                        .foregroundStyle(.secondary)
                }
                Spacer()
                SummaryBadge(title: "\(queue.items.count)", subtitle: "fichier(s)")
                SummaryBadge(title: queue.isBatchRunning ? "ON" : "OFF", subtitle: "file")
                SummaryBadge(title: settings.glossaryTerms.isEmpty ? "0" : "\(settings.glossaryTerms.count)", subtitle: "termes")
            }
            HStack(spacing: 10) {
                StepPill(index: 1, title: "Fichiers", isActive: true)
                StepConnector()
                StepPill(index: 2, title: "Contexte", isActive: !settings.glossaryTerms.isEmpty)
                StepConnector()
                StepPill(index: 3, title: "Traitement", isActive: !queue.items.isEmpty)
            }
        }
        .padding(24)
    }
}

struct QueueColumnView: View {
    @EnvironmentObject private var queue: QueueStore

    var body: some View {
        VStack(spacing: 16) {
            DropTargetView()
            if queue.items.isEmpty {
                EmptyStateView(
                    title: "Aucun fichier",
                    systemImage: "movie.stack",
                    message: "Deposez un enregistrement ou utilisez le bouton Ajouter dans la barre d'outils."
                )
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                List {
                    Section("File d'attente") {
                        ForEach(queue.items) { item in
                            QueueRowView(item: item)
                        }
                        .onMove(perform: queue.move)
                        .onDelete(perform: queue.remove)
                    }
                }
                .listStyle(.inset)
            }
        }
        .padding(20)
    }
}

struct QueueRowView: View {
    var item: QueueItem

    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: "line.3.horizontal")
                .foregroundStyle(.secondary)
            Image(systemName: "waveform.and.rectangle")
                .font(.title3)
                .foregroundStyle(.teal)
            VStack(alignment: .leading, spacing: 3) {
                Text(item.sourceURL.lastPathComponent)
                    .font(.headline)
                    .lineLimit(2)
                Text(item.sourceURL.deletingLastPathComponent().path)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }
            Spacer()
            VStack(alignment: .trailing, spacing: 5) {
                Text(item.status)
                    .font(.caption.weight(.medium))
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(.quaternary, in: Capsule())
                if item.status == "En cours" {
                    ProgressView(value: item.progress / 100.0)
                        .frame(width: 82)
                }
            }
        }
        .padding(.vertical, 6)
    }
}

struct ContextInspectorView: View {
    @EnvironmentObject private var settings: SettingsStore
    @EnvironmentObject private var engine: EngineProcess

    var body: some View {
        RunSettingsForm()
            .formStyle(.grouped)
            .padding(.vertical, 14)
            .padding(.trailing, 16)
    }
}

struct RunSettingsForm: View {
    @EnvironmentObject private var settings: SettingsStore
    @EnvironmentObject private var engine: EngineProcess

    var body: some View {
        Form {
            Section("Vocabulaire de la reunion") {
                TextEditor(text: $settings.glossary)
                    .font(.body.monospaced())
                    .frame(minHeight: 150)
                Text("\(settings.glossaryTerms.count) terme(s) transmis au moteur.")
                    .foregroundStyle(.secondary)
            }

            Section("Transcription") {
                Picker("Action", selection: $settings.processingMode) {
                    Text("Compresser + transcrire").tag("compress_transcribe")
                    Text("Transcrire seulement").tag("transcribe")
                    Text("Compresser seulement").tag("compress")
                }
                Picker("Format", selection: $settings.outputFormat) {
                    Text("Texte").tag("txt")
                    Text("SRT").tag("srt")
                    Text("VTT").tag("vtt")
                    Text("JSON").tag("json")
                }
                Picker("Modele", selection: $settings.whisperModel) {
                    Text("Whisper Large v3 Turbo").tag("mlx-community/whisper-large-v3-turbo")
                    Text("Whisper Large v3").tag("mlx-community/whisper-large-v3-mlx")
                    Text("Whisper Medium").tag("mlx-community/whisper-medium-mlx")
                }
                Toggle("Detection des locuteurs", isOn: $settings.diarizationEnabled)
                Toggle("Reecoute IA des passages douteux", isOn: $settings.audioRecheckEnabled)
            }

            Section("Sortie") {
                TextField("Dossier", text: $settings.outputDir)
                Button {
                    engine.run(arguments: EngineProcess.defaultPythonArguments(["export-logs"]))
                } label: {
                    Label("Exporter les logs", systemImage: "doc.zipper")
                }
            }
        }
    }
}

struct RunSetupView: View {
    @EnvironmentObject private var settings: SettingsStore
    @Environment(\.dismiss) private var dismiss
    var onStart: () -> Void

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Preparer le traitement")
                        .font(.title.bold())
                    Text("Ces informations sont appliquees a la file au lancement.")
                        .foregroundStyle(.secondary)
                }
                Spacer()
            }
            .padding(22)
            Divider()
            RunSettingsForm()
                .environmentObject(settings)
                .padding()
            Divider()
            HStack {
                Button("Annuler") { dismiss() }
                Spacer()
                Button("Lancer la file") { onStart() }
                    .buttonStyle(.borderedProminent)
                    .keyboardShortcut(.defaultAction)
            }
            .padding(18)
        }
        .frame(minWidth: 680, minHeight: 620)
    }
}

struct DropTargetView: View {
    @EnvironmentObject private var queue: QueueStore
    @State private var isTargeted = false

    var body: some View {
        RoundedRectangle(cornerRadius: 10)
            .fill(isTargeted ? Color.teal.opacity(0.10) : Color(nsColor: .controlBackgroundColor))
            .overlay {
                RoundedRectangle(cornerRadius: 10)
                    .strokeBorder(isTargeted ? Color.teal : Color.secondary.opacity(0.25), style: StrokeStyle(lineWidth: 1.5, dash: [7]))
            }
            .frame(minHeight: 138)
            .overlay {
                VStack(spacing: 10) {
                    Image(systemName: "square.and.arrow.down")
                        .font(.system(size: 30, weight: .medium))
                        .foregroundStyle(.teal)
                    Text("Deposez vos enregistrements")
                        .font(.title3.weight(.semibold))
                    Text("Videos et audios sont acceptes. La file reste modifiable pendant le traitement.")
                        .foregroundStyle(.secondary)
                }
            }
            .onDrop(of: [.fileURL], isTargeted: $isTargeted) { providers in
                Task {
                    var urls: [URL] = []
                    for provider in providers {
                        if let data = try? await provider.loadItem(forTypeIdentifier: UTType.fileURL.identifier) as? Data,
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
    @EnvironmentObject private var queue: QueueStore

    private var rows: [LibraryRow] {
        engine.outputLines.compactMap { line in
            try? JSONDecoder().decode(LibraryRow.self, from: Data(line.utf8))
        }
    }

    var body: some View {
        VStack(spacing: 0) {
            ListHeaderView(
                title: "Bibliotheque",
                subtitle: "Retrouvez les compressions, transcriptions et rapports produits.",
                actionTitle: "Actualiser",
                actionSystemImage: "arrow.clockwise"
            ) {
                engine.run(arguments: EngineProcess.defaultPythonArguments(["library-list", "--jsonl"]))
            }
            Divider()
            if rows.isEmpty {
                EmptyStateView(
                    title: "Aucun element charge",
                    systemImage: "tray",
                    message: "Actualisez la bibliotheque pour lire les traitements depuis le moteur."
                )
            } else {
                Table(rows) {
                    TableColumn("Fichier") { row in
                        Text(row.filename)
                            .lineLimit(2)
                    }
                    TableColumn("Statut") { row in
                        StatusText(row.status ?? "-")
                    }
                    TableColumn("Mis a jour") { row in
                        Text(row.updated_at ?? "-")
                            .foregroundStyle(.secondary)
                    }
                    TableColumn("Artefacts") { row in
                        ArtifactSummary(row: row)
                    }
                    TableColumn("Actions") { row in
                        LibraryActionsView(row: row)
                    }
                }
            }
        }
        .task {
            if engine.outputLines.isEmpty {
                engine.run(arguments: EngineProcess.defaultPythonArguments(["library-list", "--jsonl"]))
            }
        }
    }
}

struct ModelsView: View {
    @EnvironmentObject private var engine: EngineProcess

    private var rows: [ModelRow] {
        engine.outputLines.compactMap { line in
            try? JSONDecoder().decode(ModelRow.self, from: Data(line.utf8))
        }
    }

    var body: some View {
        VStack(spacing: 0) {
            ListHeaderView(
                title: "Modeles locaux",
                subtitle: "Telechargez les modeles avant une reunion pour eviter les surprises.",
                actionTitle: "Actualiser",
                actionSystemImage: "arrow.clockwise"
            ) {
                engine.run(arguments: EngineProcess.defaultPythonArguments(["model-list", "--jsonl"]))
            }
            Divider()
            if rows.isEmpty {
                EmptyStateView(
                    title: "Catalogue non charge",
                    systemImage: "shippingbox",
                    message: "Actualisez pour afficher les modeles Whisper, texte et audio."
                )
            } else {
                Table(rows) {
                    TableColumn("Modele") { row in
                        VStack(alignment: .leading) {
                            Text(row.label)
                                .font(.headline)
                            Text(row.id)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }
                    TableColumn("Famille", value: \.family)
                    TableColumn("Etat") { row in
                        StatusText(row.cached ? "Telecharge" : "A telecharger")
                    }
                    TableColumn("Actions") { row in
                        ModelActionsView(row: row)
                    }
                }
            }
        }
        .task {
            engine.run(arguments: EngineProcess.defaultPythonArguments(["model-list", "--jsonl"]))
        }
    }
}

struct SettingsView: View {
    @EnvironmentObject private var settings: SettingsStore
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Text("Reglages")
                    .font(.title.bold())
                Spacer()
                Button("OK") { dismiss() }
                    .keyboardShortcut(.defaultAction)
            }
            .padding(22)
            Divider()
            Form {
                Section("Sortie") {
                    TextField("Dossier", text: $settings.outputDir)
                }
                Section("Transcription") {
                    Picker("Action", selection: $settings.processingMode) {
                        Text("Compresser + transcrire").tag("compress_transcribe")
                        Text("Transcrire seulement").tag("transcribe")
                        Text("Compresser seulement").tag("compress")
                    }
                    Picker("Format", selection: $settings.outputFormat) {
                        Text("Texte").tag("txt")
                        Text("SRT").tag("srt")
                        Text("VTT").tag("vtt")
                        Text("JSON").tag("json")
                    }
                    TextField("Modele Whisper", text: $settings.whisperModel)
                    Toggle("Detection des locuteurs", isOn: $settings.diarizationEnabled)
                    Toggle("Reecoute IA multimodale", isOn: $settings.audioRecheckEnabled)
                }
                Section("Hugging Face") {
                    SecureField("Token", text: $settings.hfToken)
                }
                Section("Vocabulaire conserve") {
                    TextEditor(text: $settings.glossary)
                        .frame(minHeight: 130)
                }
            }
            .formStyle(.grouped)
            .padding()
        }
        .frame(minWidth: 620, minHeight: 520)
    }
}

struct StatusBarView: View {
    @EnvironmentObject private var engine: EngineProcess

    private var progress: Double? {
        engine.events.last(where: { $0.event == .progress })?.pct
    }

    var body: some View {
        HStack(spacing: 10) {
            if engine.isRunning {
                ProgressView(value: progress.map { $0 / 100.0 })
                    .frame(width: 120)
            }
            Text(engine.lastError ?? engine.events.last?.message ?? "Pret.")
                .lineLimit(1)
            Spacer()
            if engine.isRunning {
                Button("Annuler") { engine.cancel() }
            }
        }
        .font(.callout)
        .padding(.horizontal, 18)
        .padding(.vertical, 10)
    }
}

struct ListHeaderView: View {
    var title: String
    var subtitle: String
    var actionTitle: String
    var actionSystemImage: String
    var action: () -> Void

    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text(title)
                    .font(.largeTitle.bold())
                Text(subtitle)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            Button(action: action) {
                Label(actionTitle, systemImage: actionSystemImage)
            }
        }
        .padding(24)
    }
}

struct SummaryBadge: View {
    var title: String
    var subtitle: String

    var body: some View {
        VStack(spacing: 2) {
            Text(title)
                .font(.title2.bold())
            Text(subtitle)
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .frame(minWidth: 82)
        .padding(.vertical, 10)
        .background(.quaternary, in: RoundedRectangle(cornerRadius: 8))
    }
}

struct EmptyStateView: View {
    var title: String
    var systemImage: String
    var message: String

    var body: some View {
        VStack(spacing: 10) {
            Image(systemName: systemImage)
                .font(.system(size: 42, weight: .regular))
                .foregroundStyle(.secondary)
            Text(title)
                .font(.title3.weight(.semibold))
            Text(message)
                .multilineTextAlignment(.center)
                .foregroundStyle(.secondary)
                .frame(maxWidth: 360)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

struct StepPill: View {
    var index: Int
    var title: String
    var isActive: Bool

    var body: some View {
        HStack(spacing: 8) {
            Text("\(index)")
                .font(.caption.bold())
                .frame(width: 22, height: 22)
                .background(isActive ? Color.teal : Color.secondary.opacity(0.20), in: Circle())
                .foregroundStyle(isActive ? .white : .secondary)
            Text(title)
                .font(.callout.weight(.medium))
        }
    }
}

struct StepConnector: View {
    var body: some View {
        Rectangle()
            .fill(Color.secondary.opacity(0.25))
            .frame(width: 34, height: 1)
    }
}

struct StatusText: View {
    var value: String

    init(_ value: String) {
        self.value = value
    }

    var body: some View {
        Text(value)
            .font(.caption.weight(.medium))
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(.quaternary, in: Capsule())
    }
}

struct ArtifactSummary: View {
    var row: LibraryRow

    var body: some View {
        HStack {
            ArtifactDot(label: "C", isPresent: !(row.compressed_path ?? "").isEmpty)
            ArtifactDot(label: "T", isPresent: !(row.transcript_path ?? "").isEmpty)
            ArtifactDot(label: "A", isPresent: !(row.enhanced_transcript_path ?? "").isEmpty)
            ArtifactDot(label: "R", isPresent: !(row.review_path ?? "").isEmpty)
        }
    }
}

struct LibraryActionsView: View {
    @EnvironmentObject private var queue: QueueStore
    var row: LibraryRow

    var body: some View {
        HStack(spacing: 6) {
            if let source = row.source_path, !source.isEmpty {
                Button {
                    queue.add(urls: [URL(fileURLWithPath: source)])
                } label: {
                    Label("Relancer", systemImage: "arrow.clockwise")
                }
                .labelStyle(.iconOnly)
                .help("Ajouter ce fichier a la file")
            }

            Menu {
                ArtifactMenuButton(title: "Compresse", path: row.compressed_path)
                ArtifactMenuButton(title: "Transcription", path: row.transcript_path)
                ArtifactMenuButton(title: "Amelioree", path: row.enhanced_transcript_path)
                ArtifactMenuButton(title: "Rapport", path: row.review_path)
                Divider()
                if let source = row.source_path, !source.isEmpty {
                    Button("Afficher l'original dans le Finder") {
                        revealInFinder(source)
                    }
                }
            } label: {
                Image(systemName: "ellipsis.circle")
            }
            .menuStyle(.borderlessButton)
        }
    }
}

struct ArtifactMenuButton: View {
    var title: String
    var path: String?

    var body: some View {
        if let path, !path.isEmpty {
            Button(title) {
                openPath(path)
            }
        } else {
            Button(title) {}
                .disabled(true)
        }
    }
}

struct ModelActionsView: View {
    @EnvironmentObject private var engine: EngineProcess
    var row: ModelRow

    var body: some View {
        HStack(spacing: 6) {
            if row.cached {
                Button {
                    engine.run(arguments: EngineProcess.defaultPythonArguments(["model-delete", row.id]))
                } label: {
                    Label("Supprimer", systemImage: "trash")
                }
                .labelStyle(.iconOnly)
                .help("Supprimer le modele local")
            } else {
                Button {
                    engine.run(arguments: EngineProcess.defaultPythonArguments(["model-download", row.id]))
                } label: {
                    Label("Telecharger", systemImage: "arrow.down.circle")
                }
                .labelStyle(.iconOnly)
                .help("Pre-telecharger le modele")
            }
            Button {
                revealInFinder(row.cache_dir)
            } label: {
                Label("Cache", systemImage: "folder")
            }
            .labelStyle(.iconOnly)
            .help("Afficher le dossier de cache")
        }
        .disabled(engine.isRunning)
    }
}

struct ArtifactDot: View {
    var label: String
    var isPresent: Bool

    var body: some View {
        Text(label)
            .font(.caption2.bold())
            .frame(width: 20, height: 20)
            .background(isPresent ? Color.teal : Color.secondary.opacity(0.18), in: Circle())
            .foregroundStyle(isPresent ? .white : .secondary)
    }
}

func openPath(_ path: String) {
    NSWorkspace.shared.open(URL(fileURLWithPath: path))
}

func revealInFinder(_ path: String) {
    NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: path)])
}
