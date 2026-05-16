@preconcurrency import AVFoundation
@preconcurrency import Foundation
import SwiftUI
import AppKit
import UniformTypeIdentifiers

enum AppSection: String, CaseIterable, Hashable {
    case queue
    case library
    case models
    case vocabulary
    case speakers

    var title: String {
        switch self {
        case .queue: "Traitements"
        case .library: "Bibliothèque"
        case .models: "Modèles"
        case .vocabulary: "Vocabulaire"
        case .speakers: "Interlocuteurs"
        }
    }

    var symbol: String {
        switch self {
        case .queue: "play.rectangle"
        case .library: "tray.full"
        case .models: "shippingbox"
        case .vocabulary: "text.word.spacing"
        case .speakers: "person.text.rectangle"
        }
    }
}

struct ContentView: View {
    @EnvironmentObject private var engine: EngineProcess
    @EnvironmentObject private var queue: QueueStore
    @EnvironmentObject private var settings: SettingsStore
    @EnvironmentObject private var library: LibraryStore
    @EnvironmentObject private var models: ModelStore
    @EnvironmentObject private var odoo: OdooStore
    @State private var selectedSection: AppSection = .queue
    @State private var showingSettings = false
    @State private var showingRunSetup = false
    @State private var didPreloadSecondaryData = false
    @State private var lastLibraryRefreshAt = Date.distantPast

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
                case .vocabulary:
                    VocabularyView()
                case .speakers:
                    SpeakersView()
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
                        Label("Réglages", systemImage: "gearshape")
                    }
                    .keyboardShortcut(",", modifiers: .command)
                }
            }
        }
        .sheet(isPresented: $showingSettings) {
            SettingsView()
                .environmentObject(settings)
                .environmentObject(engine)
        }
        .sheet(isPresented: $showingRunSetup) {
            RunSetupView {
                showingRunSetup = false
                Task { await runQueue() }
            }
            .environmentObject(settings)
            .environmentObject(engine)
            .environmentObject(queue)
            .environmentObject(odoo)
            .environmentObject(library)
        }
        .task {
            guard !didPreloadSecondaryData else { return }
            didPreloadSecondaryData = true
            async let libraryLoad: Void = library.rows.isEmpty ? library.refresh() : ()
            async let modelsLoad: Void = models.models.isEmpty ? models.refresh() : ()
            _ = await (libraryLoad, modelsLoad)
        }
        .onChange(of: queue.autoRunRequestID) { _, requestID in
            guard requestID != nil else { return }
            queue.autoRunRequestID = nil
            guard !queue.isBatchRunning, !queue.items.isEmpty else { return }
            selectedSection = .queue
            Task { await runQueue() }
        }
        .onChange(of: engine.events.count) { _, _ in
            guard let event = engine.events.last else { return }
            guard event.event == .artifact || event.event == .done || event.event == .progress else { return }
            let now = Date()
            if event.event == .progress && now.timeIntervalSince(lastLibraryRefreshAt) < 2 {
                return
            }
            lastLibraryRefreshAt = now
            Task { await library.refresh() }
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
        let itemIDs = queue.items.map(\.id)

        for itemID in itemIDs {
            if Task.isCancelled { break }
            guard var currentItem = queue.items.first(where: { $0.id == itemID }) else {
                continue
            }
            queue.update(currentItem.id, status: "En cours", progress: 0)
            var exitCode = await runJob(currentItem)
            // Recovery loop: if the engine returned ``source_missing``,
            // give the user one shot to point us at the new location.
            // We only retry on explicit relocalisation — Cancel falls
            // through to the regular "Erreur" status so the batch
            // keeps moving instead of blocking on an unresolved item.
            while exitCode != 0 && engine.lastErrorCode == "source_missing" {
                if Task.isCancelled { break }
                guard let relocated = await promptRelocalize(missing: currentItem.sourceURL) else {
                    break
                }
                queue.replace(currentItem.id, with: relocated)
                currentItem.sourceURL = relocated
                queue.update(currentItem.id, status: "Reprise", progress: 0)
                exitCode = await runJob(currentItem)
            }
            if exitCode == 0 {
                queue.update(currentItem.id, status: "Terminé", progress: 100)
            } else {
                queue.update(currentItem.id, status: "Erreur", progress: 0)
            }
        }
        queue.isBatchRunning = false
    }

    /// Show an NSOpenPanel anchored on the queue window so the user
    /// can point us at the new location of a missing source. Returns
    /// the picked URL, or nil when the user cancels — the caller
    /// then marks the item as failed and moves on.
    @MainActor
    private func promptRelocalize(missing: URL) async -> URL? {
        let alert = NSAlert()
        alert.messageText = "Source introuvable"
        alert.informativeText = "\(missing.lastPathComponent) est introuvable à son emplacement d'origine. Voulez-vous sélectionner le nouveau fichier ?"
        alert.addButton(withTitle: "Sélectionner…")
        alert.addButton(withTitle: "Passer ce fichier")
        guard alert.runModal() == .alertFirstButtonReturn else { return nil }
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        panel.message = "Sélectionner le nouvel emplacement de \(missing.lastPathComponent)"
        panel.directoryURL = missing.deletingLastPathComponent()
        guard panel.runModal() == .OK, let chosen = panel.url else { return nil }
        return chosen
    }

    private func runJob(_ item: QueueItem) async -> Int32 {
        // Names suggested by an Odoo calendar event (or typed by
        // hand) ride in ``speaker_overrides`` keyed on themselves
        // — the engine's initial-prompt builder pulls them as
        // ``expected_speaker_names`` so Whisper biases toward those
        // first names without forcing a SPEAKER_NN assignment.
        var overrides: [String: String] = [:]
        for name in item.expectedSpeakerNames {
            let trimmed = name.trimmingCharacters(in: .whitespacesAndNewlines)
            if !trimmed.isEmpty {
                overrides[trimmed] = trimmed
            }
        }
        // Per-file Odoo meeting title becomes the ``profile`` — the
        // engine's ``_meeting_context`` helper reads that and
        // injects it as a one-line "Réunion sur X" prefix into the
        // Whisper initial prompt.
        let profile = item.odooMeetingTitle.trimmingCharacters(in: .whitespacesAndNewlines)
        // Build the optional Odoo context payloads — only when the
        // user actually picked a meeting in Run Setup. The credentials
        // ride alongside the model+id so the engine can fire the
        // chatter fetch on its own during the LLM step.
        var contextRef: OdooContextRef? = nil
        if let related = item.odooContextRef, settings.odooConfigured {
            contextRef = OdooContextRef(
                model: related.model,
                record_id: related.record_id,
                url: settings.odooUrl,
                database: settings.odooDatabase,
                login: settings.odooLogin,
                api_key: settings.odooApiKey
            )
        }
        let compressionSettings = compressionSettings(for: item)
        let termsForRun = item.selectedGlossaryTerms.map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        let request = JobRequest(
            source_path: item.sourceURL.path,
            workspace_dir: item.workspaceDir,
            output_dir: settings.outputDir,
            mode: settings.processingMode,
            profile: profile.isEmpty ? "Réunion équilibrée" : profile,
            compression_settings: compressionSettings,
            transcription_settings: TranscriptionSettings(
                model: settings.whisperModel,
                output_format: settings.outputFormat,
                diarization_enabled: settings.diarizationEnabled,
                hf_token: settings.hfToken,
                audio_recheck_enabled: settings.audioRecheckEnabled,
                quality_preset: TranscriptionQualityPreset(
                    rawValue: settings.qualityPreset
                )?.rawValue ?? TranscriptionQualityPreset.balanced.rawValue,
                expected_min_speakers: item.expectedSpeakerCount,
                expected_max_speakers: item.expectedSpeakerCount
            ),
            glossary_terms: termsForRun,
            speaker_overrides: overrides,
            technical_terms: item.focusNote.map { [$0] } ?? [],
            rerun_steps: [],
            library_job_id: item.libraryJobId,
            delete_source_after_copy: settings.deleteSourceAfterCopy && !item.isLibraryRerun,
            odoo_context_ref: contextRef,
            odoo_meeting_metadata: item.odooMeeting
        )
        settings.recordVocabularyUsage(termsForRun)
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

    private func compressionSettings(for item: QueueItem) -> CompressionSettings {
        var compression = CompressionSettings()
        let start = max(0, item.trimStartSeconds)
        let removedEnd = max(0, item.trimEndSeconds)
        compression.trim_enabled = start > 0 || removedEnd > 0
        compression.trim_start = formatHMS(start)
        if removedEnd > 0, item.mediaDurationSeconds > removedEnd {
            let absoluteEnd = max(start + 1, item.mediaDurationSeconds - removedEnd)
            compression.trim_end = formatHMS(absoluteEnd)
        }
        return compression
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
    @EnvironmentObject private var updater: UpdateStore

    var body: some View {
        VStack(spacing: 0) {
            if case .available(let info) = updater.state {
                UpdateBannerView(info: info)
            }
            WorkflowHeaderView()
            Divider()
            QueueColumnView()
            Divider()
            StatusBarView()
        }
    }
}

struct UpdateBannerView: View {
    @EnvironmentObject private var updater: UpdateStore
    var info: ReleaseInfo

    var body: some View {
        HStack {
            Image(systemName: "sparkles")
                .foregroundStyle(.yellow)
            VStack(alignment: .leading, spacing: 2) {
                Text("Une mise à jour est disponible : \(info.tag_name)")
                    .font(.headline)
                Text(info.name)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            Button("En savoir plus") {
                if let url = URL(string: info.html_url) {
                    NSWorkspace.shared.open(url)
                }
            }
            Button("Mettre à jour") {
                Task {
                    await updater.downloadAndPrepare(info: info)
                }
            }
            .buttonStyle(.borderedProminent)
        }
        .padding()
        .background(Color.yellow.opacity(0.1))
        .overlay(Divider(), alignment: .bottom)
    }
}

struct WorkflowHeaderView: View {
    @EnvironmentObject private var queue: QueueStore
    @EnvironmentObject private var settings: SettingsStore

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(alignment: .firstTextBaseline) {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Préparer une transcription")
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
                    message: "Déposez un enregistrement ou utilisez le bouton Ajouter dans la barre d'outils."
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
    @EnvironmentObject private var queue: QueueStore
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
                    .lineLimit(1)
                    .truncationMode(.tail)
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
            Button(role: .destructive) {
                queue.remove(id: item.id)
            } label: {
                Label("Retirer", systemImage: "xmark.circle")
            }
            .labelStyle(.iconOnly)
            .buttonStyle(.borderless)
            .help("Retirer ce fichier de la file d'attente")
            .disabled(queue.isBatchRunning && item.status == "En cours")
        }
        .padding(.vertical, 6)
    }
}

/// Batch-wide settings: action, format, model, quality preset,
/// diarisation toggle, audio recheck, glossary. Anything that
/// applies to every file in the queue lives here.
///
/// Hugging Face token + Exporter les logs deliberately stay out
/// of this form — they're application-wide preferences, not
/// per-run knobs, and surfaced in Réglages where they belong.
struct RunBatchSettingsForm: View {
    @EnvironmentObject private var settings: SettingsStore
    @EnvironmentObject private var queue: QueueStore

    private var canDeleteOriginalSources: Bool {
        queue.items.contains { !$0.isLibraryRerun }
    }

    var body: some View {
        Form {
            Section("Action") {
                Picker("Mode", selection: $settings.processingMode) {
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
            }

            Section("Moteur de transcription") {
                Picker("Modèle Whisper", selection: $settings.whisperModel) {
                    Text("Whisper Large v3 Turbo").tag("mlx-community/whisper-large-v3-turbo")
                    Text("Whisper Large v3").tag("mlx-community/whisper-large-v3-mlx")
                    Text("Whisper Medium").tag("mlx-community/whisper-medium-mlx")
                }
                Picker("Qualité", selection: $settings.qualityPreset) {
                    ForEach(TranscriptionQualityPreset.allCases) { preset in
                        Text(preset.displayName).tag(preset.rawValue)
                    }
                }
                if let preset = TranscriptionQualityPreset(rawValue: settings.qualityPreset) {
                    Text(preset.summary)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Toggle("Détection des locuteurs", isOn: $settings.diarizationEnabled)
                Toggle("Réécoute IA des passages douteux", isOn: $settings.audioRecheckEnabled)
            }

            Section("Vocabulaire") {
                Text("Choisissez les termes utiles dans chaque fichier. Le catalogue global sert de suggestions, mais il n'est plus envoyé automatiquement à toutes les réunions.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            if canDeleteOriginalSources {
                Section("Fichiers") {
                Toggle("Supprimer le fichier source après copie", isOn: $settings.deleteSourceAfterCopy)
                Text("Le moteur copie d'abord l'original dans le dossier de travail. Si cette option est active, seul le fichier à son emplacement d'origine est supprimé.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                }
            }
        }
        .formStyle(.grouped)
    }
}

/// One-file pre-flight panel: filename, audio preview, per-file
/// speaker count, per-file note. The user walks through the queue
/// with Précédent / Suivant so they can preview every recording
/// before launching the batch — particularly handy on a queue of
/// 5 meetings where remembering "wait who was in the second one
/// again?" used to mean opening Finder, hitting space, repeating.
struct PerFileSetupPanel: View {
    @EnvironmentObject private var settings: SettingsStore
    @EnvironmentObject private var odoo: OdooStore
    @Binding var item: QueueItem
    let index: Int
    let total: Int

    @StateObject private var player = AudioPreviewPlayer()
    @State private var suggestionsLoading = false
    @State private var suggestions: [OdooMeetingSuggestion] = []
    @State private var lastSuggestionsKey: String = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Fichier \(index + 1) / \(total)")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)
                Spacer()
            }
            HStack(spacing: 12) {
                Image(systemName: "waveform.and.rectangle")
                    .font(.title)
                    .foregroundStyle(.teal)
                VStack(alignment: .leading, spacing: 2) {
                    Text(item.sourceURL.lastPathComponent)
                        .font(.headline)
                        .lineLimit(2)
                    Text(item.sourceURL.deletingLastPathComponent().path)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                }
                Spacer()
                if player.isLoading {
                    ProgressView()
                        .controlSize(.small)
                        .help("Préparation de l'aperçu audio")
                }
                Button {
                    player.toggle(url: item.sourceURL)
                } label: {
                    Label(
                        player.isPlaying ? "Pause" : "Écouter",
                        systemImage: player.isPlaying ? "pause.circle.fill" : "play.circle.fill"
                    )
                }
                .controlSize(.large)
                .help("Aperçu audio — utile pour se rappeler des intervenants")
            }
            audioScrubber

            DisclosureGroup("Rogner le début ou la fin") {
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Stepper(
                            "Début ignoré : \(formatPreviewSeconds(item.trimStartSeconds))",
                            value: $item.trimStartSeconds,
                            in: 0...trimStartUpperBound,
                            step: 5
                        )
                        Spacer()
                        Stepper(
                            "Fin ignorée : \(formatPreviewSeconds(item.trimEndSeconds))",
                            value: $item.trimEndSeconds,
                            in: 0...trimEndUpperBound,
                            step: 5
                        )
                    }
                    Text("Le rognage est appliqué avant la compression et la transcription.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                .padding(.top, 4)
            }

            Picker("Nombre d'intervenants attendu", selection: $item.expectedSpeakerCount) {
                Text("Auto").tag(0)
                ForEach(1...12, id: \.self) { count in
                    Text("\(count)").tag(count)
                }
            }
            .pickerStyle(.menu)

            TokenListPicker(
                title: "Interlocuteurs attendus",
                placeholder: "Ajouter un nom",
                selected: $item.expectedSpeakerNames,
                suggestions: speakerSuggestions
            )

            VocabularyTokenPicker(selected: $item.selectedGlossaryTerms)
                .environmentObject(settings)

            VStack(alignment: .leading, spacing: 6) {
                Text("Notes pour ce fichier (optionnel)")
                    .font(.callout.weight(.medium))
                TextField(
                    "Termes à retenir, contexte particulier…",
                    text: Binding(
                        get: { item.focusNote ?? "" },
                        set: { item.focusNote = $0.isEmpty ? nil : $0 }
                    ),
                    axis: .vertical
                )
                .lineLimit(2...4)
                .textFieldStyle(.roundedBorder)
            }

            if settings.odooConfigured {
                OdooMeetingSuggestionsSection(
                    item: $item,
                    suggestions: suggestions,
                    loading: suggestionsLoading
                )
            }
        }
        .padding(.horizontal, 18)
        .padding(.vertical, 14)
        .background(Color(nsColor: .controlBackgroundColor))
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .task(id: item.sourceURL) {
            player.load(url: item.sourceURL)
            await loadMeetingSuggestions()
        }
        .onChange(of: player.totalSeconds) { _, duration in
            item.mediaDurationSeconds = duration
        }
        .onDisappear { player.stop() }
    }

    @ViewBuilder
    private var audioScrubber: some View {
        if player.totalSeconds > 0 {
            HStack(spacing: 8) {
                Button {
                    player.skip(seconds: -15)
                } label: {
                    Label("Reculer 15 s", systemImage: "gobackward.15")
                }
                .labelStyle(.iconOnly)
                .buttonStyle(.borderless)
                Slider(
                    value: Binding(
                        get: { player.currentSeconds },
                        set: { player.seek(to: $0) }
                    ),
                    in: 0...max(player.totalSeconds, 1)
                )
                Button {
                    player.skip(seconds: 15)
                } label: {
                    Label("Avancer 15 s", systemImage: "goforward.15")
                }
                .labelStyle(.iconOnly)
                .buttonStyle(.borderless)
                Text("\(formatPreviewSeconds(player.currentSeconds)) / \(formatPreviewSeconds(player.totalSeconds))")
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
                    .frame(width: 104, alignment: .trailing)
            }
        } else if player.isLoading {
            HStack(spacing: 8) {
                ProgressView()
                    .controlSize(.small)
                Text("Préparation de l'aperçu audio…")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }

    private var knownDurationOrFallback: Double {
        let known = max(player.totalSeconds, item.mediaDurationSeconds)
        return known
    }

    private var trimStartUpperBound: Double {
        if knownDurationOrFallback > 0 {
            return max(0, knownDurationOrFallback - item.trimEndSeconds - 1)
        }
        return 24 * 3600
    }

    private var trimEndUpperBound: Double {
        guard knownDurationOrFallback > 0 else { return 0 }
        return max(0, knownDurationOrFallback - item.trimStartSeconds - 1)
    }

    private var speakerSuggestions: [String] {
        let meetingNames = item.odooMeeting?.attendees.map(\.name) ?? []
        return meetingNames.filter { name in
            !name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            && !item.expectedSpeakerNames.contains(where: { $0.caseInsensitiveCompare(name) == .orderedSame })
        }
    }

    /// Bracket the file's modification time and ask Odoo for any
    /// ``calendar.event`` records that touch that window. The
    /// fetch is keyed on the URL so we only query once per file
    /// per sheet — flipping back to the same file with Précédent /
    /// Suivant doesn't re-hit the network.
    private func loadMeetingSuggestions() async {
        guard settings.odooConfigured else {
            suggestions = []
            return
        }
        let key = item.sourceURL.path
        if key == lastSuggestionsKey { return }
        lastSuggestionsKey = key
        suggestions = []
        suggestionsLoading = true
        defer { suggestionsLoading = false }
        // mtime of the recording typically ≈ end of the meeting.
        // We ask for a generous window (2 h before / 30 min after)
        // so a recording renamed slightly after the fact still
        // matches the event Odoo holds.
        let fileDate = (try? item.sourceURL.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate)
            ?? Date()
        let found = await odoo.searchMeetings(near: fileDate, windowHours: 2.5, limit: 8)
        await MainActor.run {
            suggestions = found
        }
    }
}

/// Renders the Odoo calendar suggestions inside ``PerFileSetupPanel``.
/// Hidden when nothing was found (no signal, no clutter); shows the
/// "appliquer" button per row so the user picks the meeting that
/// matches their recording.
struct OdooMeetingSuggestionsSection: View {
    @Binding var item: QueueItem
    let suggestions: [OdooMeetingSuggestion]
    let loading: Bool

    private var attachedMeetingId: Int? {
        // We don't persist the linkage on the QueueItem (the
        // attendees + title already capture the intent), but we
        // can still recognise "already applied" by checking
        // whether the speaker names match the suggestion's
        // attendees.
        for meeting in suggestions {
            let attendeeNames = Set(meeting.attendees.map(\.name))
            let pickedNames = Set(item.expectedSpeakerNames)
            if !attendeeNames.isEmpty && pickedNames == attendeeNames {
                return meeting.id
            }
        }
        return nil
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 6) {
                Image(systemName: "calendar")
                    .foregroundStyle(.teal)
                Text("Suggestions Odoo")
                    .font(.callout.weight(.medium))
                if loading {
                    ProgressView().controlSize(.small)
                }
                Spacer()
            }
            if !loading && suggestions.isEmpty {
                Text("Aucune réunion Odoo détectée autour de la date du fichier.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            ForEach(suggestions) { meeting in
                OdooMeetingSuggestionRow(
                    meeting: meeting,
                    attached: attachedMeetingId == meeting.id,
                    onApply: { apply(meeting) },
                    onClear: clear
                )
            }
        }
    }

    private func apply(_ meeting: OdooMeetingSuggestion) {
        item.expectedSpeakerNames = meeting.attendees.map(\.name)
        // Trust pyannote's clustering if the user picked a meeting
        // that doesn't list attendees; otherwise pin the count.
        if !meeting.attendees.isEmpty {
            item.expectedSpeakerCount = meeting.attendees.count
        }
        item.odooMeetingTitle = meeting.name
        item.odooMeeting = OdooMeetingMetadata(
            event_id: meeting.id,
            event_name: meeting.name,
            attendees: meeting.attendees,
            related: meeting.related_object
        )
        // ``related_object`` is what the engine's LLM step fetches
        // the chatter for. We forward just the model + id at this
        // stage; the credentials are stitched in later (in
        // ``runJob``) so they don't sit on the SwiftUI struct.
        if let related = meeting.related_object {
            item.odooContextRef = OdooContextRef(
                model: related.model,
                record_id: related.id,
                url: "",
                database: "",
                login: "",
                api_key: ""
            )
        } else {
            item.odooContextRef = nil
        }
    }

    private func clear() {
        item.expectedSpeakerNames = []
        item.odooMeetingTitle = ""
        item.odooMeeting = nil
        item.odooContextRef = nil
    }
}

struct OdooMeetingSuggestionRow: View {
    let meeting: OdooMeetingSuggestion
    let attached: Bool
    let onApply: () -> Void
    let onClear: () -> Void

    private var subtitle: String {
        var parts: [String] = []
        if !meeting.start.isEmpty {
            parts.append(meeting.start)
        }
        if meeting.attendee_count > 0 {
            parts.append("\(meeting.attendee_count) participant\(meeting.attendee_count > 1 ? "s" : "")")
        }
        if let related = meeting.related_object, !related.name.isEmpty {
            parts.append(related.name)
        }
        return parts.joined(separator: " · ")
    }

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            VStack(alignment: .leading, spacing: 2) {
                Text(meeting.name.isEmpty ? "(sans titre)" : meeting.name)
                    .font(.callout.weight(.medium))
                    .lineLimit(1)
                if !subtitle.isEmpty {
                    Text(subtitle)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(2)
                }
                if !meeting.attendees.isEmpty {
                    Text(meeting.attendees.map(\.name).joined(separator: ", "))
                        .font(.caption)
                        .foregroundStyle(.tertiary)
                        .lineLimit(2)
                }
            }
            Spacer()
            if attached {
                Button(role: .destructive, action: onClear) {
                    Label("Détacher", systemImage: "xmark.circle")
                }
                .controlSize(.small)
            } else {
                Button(action: onApply) {
                    Label("Utiliser", systemImage: "checkmark.circle")
                }
                .controlSize(.small)
                .buttonStyle(.borderedProminent)
            }
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 6))
    }
}

struct TokenListPicker: View {
    var title: String
    var placeholder: String
    @Binding var selected: [String]
    var suggestions: [String] = []
    @State private var draft = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 7) {
            Text(title)
                .font(.callout.weight(.medium))
            FlowLayout(spacing: 6) {
                ForEach(selected, id: \.self) { value in
                    RemovableToken(text: value) {
                        selected.removeAll { $0 == value }
                    }
                }
                TextField(placeholder, text: $draft)
                    .textFieldStyle(.roundedBorder)
                    .frame(minWidth: 180, maxWidth: 260)
                    .onSubmit { add(draft) }
            }
            if !suggestions.isEmpty {
                FlowLayout(spacing: 6) {
                    ForEach(suggestions.prefix(8), id: \.self) { suggestion in
                        Button {
                            add(suggestion)
                        } label: {
                            Label(suggestion, systemImage: "plus.circle")
                                .font(.caption)
                        }
                        .buttonStyle(.borderless)
                    }
                }
            }
        }
    }

    private func add(_ raw: String) {
        let value = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !value.isEmpty else { return }
        guard !selected.contains(where: { $0.caseInsensitiveCompare(value) == .orderedSame }) else {
            draft = ""
            return
        }
        selected.append(value)
        draft = ""
    }
}

struct VocabularyTokenPicker: View {
    @EnvironmentObject private var settings: SettingsStore
    @Binding var selected: [String]
    @State private var draft = ""

    private var suggestions: [String] {
        settings.suggestedVocabulary(matching: draft, excluding: selected)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 7) {
            Text("Vocabulaire de cette réunion")
                .font(.callout.weight(.medium))
            FlowLayout(spacing: 6) {
                ForEach(selected, id: \.self) { term in
                    RemovableToken(text: term) {
                        selected.removeAll { $0 == term }
                    }
                }
                TextField("Ajouter un terme", text: $draft)
                    .textFieldStyle(.roundedBorder)
                    .frame(minWidth: 190, maxWidth: 280)
                    .onSubmit { add(draft) }
            }
            if !suggestions.isEmpty {
                FlowLayout(spacing: 6) {
                    ForEach(suggestions.prefix(10), id: \.self) { suggestion in
                        Button {
                            add(suggestion)
                        } label: {
                            Label(suggestion, systemImage: "plus.circle")
                                .font(.caption)
                        }
                        .buttonStyle(.borderless)
                    }
                }
            } else if !draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                Text("Entrée crée le terme.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }

    private func add(_ raw: String) {
        let value = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !value.isEmpty else { return }
        guard !selected.contains(where: { $0.caseInsensitiveCompare(value) == .orderedSame }) else {
            draft = ""
            return
        }
        selected.append(value)
        settings.addVocabularyTerm(value)
        draft = ""
    }
}

struct RemovableToken: View {
    var text: String
    var onRemove: () -> Void

    var body: some View {
        HStack(spacing: 5) {
            Text(text)
                .lineLimit(1)
            Button(action: onRemove) {
                Image(systemName: "xmark.circle.fill")
                    .imageScale(.small)
            }
            .buttonStyle(.plain)
            .foregroundStyle(.secondary)
        }
        .font(.caption)
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(Color.teal.opacity(0.16), in: Capsule())
    }
}

struct FlowLayout: Layout {
    var spacing: CGFloat = 6

    func sizeThatFits(
        proposal: ProposedViewSize,
        subviews: Subviews,
        cache: inout Void
    ) -> CGSize {
        let maxWidth = proposal.width ?? 480
        var x: CGFloat = 0
        var y: CGFloat = 0
        var rowHeight: CGFloat = 0
        for subview in subviews {
            let size = subview.sizeThatFits(.unspecified)
            if x > 0 && x + size.width > maxWidth {
                x = 0
                y += rowHeight + spacing
                rowHeight = 0
            }
            x += size.width + spacing
            rowHeight = max(rowHeight, size.height)
        }
        return CGSize(width: maxWidth, height: y + rowHeight)
    }

    func placeSubviews(
        in bounds: CGRect,
        proposal: ProposedViewSize,
        subviews: Subviews,
        cache: inout Void
    ) {
        var x = bounds.minX
        var y = bounds.minY
        var rowHeight: CGFloat = 0
        for subview in subviews {
            let size = subview.sizeThatFits(.unspecified)
            if x > bounds.minX && x + size.width > bounds.maxX {
                x = bounds.minX
                y += rowHeight + spacing
                rowHeight = 0
            }
            subview.place(
                at: CGPoint(x: x, y: y),
                proposal: ProposedViewSize(size)
            )
            x += size.width + spacing
            rowHeight = max(rowHeight, size.height)
        }
    }
}

/// Minimal AVFoundation wrapper used by the per-file setup panel.
/// Exposes just the three knobs the UI binds against — toggle,
/// progress, formatted timestamps — so the heavy AVPlayer
/// surface stays contained.
@MainActor
final class AudioPreviewPlayer: ObservableObject {
    @Published private(set) var isPlaying = false
    @Published private(set) var progress: Double = 0
    @Published private(set) var isLoading = false
    @Published private(set) var currentSeconds: Double = 0
    @Published private(set) var totalSeconds: Double = 0

    private var player: AVPlayer?
    private var currentURL: URL?
    private var timeObserver: Any?

    // No ``deinit`` cleanup: Swift 6 makes deinit nonisolated and
    // ``timeObserver`` is ``Any?`` (non-Sendable), so removing it
    // there isn't allowed. ``stop()`` always runs before this
    // store is dropped (we call it from .onDisappear and on every
    // file swap inside Run Setup), and AVPlayer cleans up its
    // observers when the object itself deinitialises, so leaking
    // a stale token isn't a concern.

    var elapsedFormatted: String? {
        totalSeconds > 0 ? formatPreviewSeconds(currentSeconds) : nil
    }

    var durationFormatted: String? {
        totalSeconds > 0 ? formatPreviewSeconds(totalSeconds) : nil
    }

    func load(url: URL) {
        if currentURL == url, player != nil { return }
        stop()
        isLoading = true
        let asset = AVURLAsset(url: url)
        let item = AVPlayerItem(asset: asset)
        let player = AVPlayer(playerItem: item)
        self.player = player
        self.currentURL = url
        installObserver()
        Task { [weak self] in
            let duration = (try? await asset.load(.duration)) ?? .invalid
            let seconds = CMTimeGetSeconds(duration)
            await MainActor.run {
                guard let self, self.currentURL == url else { return }
                self.totalSeconds = seconds.isFinite && seconds > 0 ? seconds : 0
                self.isLoading = false
            }
        }
    }

    func toggle(url: URL) {
        if currentURL != url || player == nil {
            load(url: url)
        }
        guard let player else { return }
        if isPlaying {
            player.pause()
            isPlaying = false
        } else {
            player.play()
            isPlaying = true
        }
    }

    func stop() {
        if let player, let token = timeObserver {
            player.removeTimeObserver(token)
        }
        player?.pause()
        player = nil
        currentURL = nil
        isPlaying = false
        isLoading = false
        progress = 0
        currentSeconds = 0
        totalSeconds = 0
        timeObserver = nil
    }

    func seek(to seconds: Double) {
        let clamped = min(max(seconds, 0), max(totalSeconds, 0))
        currentSeconds = clamped
        progress = totalSeconds > 0 ? min(max(clamped / totalSeconds, 0), 1) : 0
        player?.seek(to: CMTime(seconds: clamped, preferredTimescale: 600))
    }

    func skip(seconds: Double) {
        seek(to: currentSeconds + seconds)
    }

    private var playerCurrentSeconds: Double? {
        guard let player else { return nil }
        let time = CMTimeGetSeconds(player.currentTime())
        return time.isFinite ? time : nil
    }

    private var playerTotalSeconds: Double? {
        guard let item = player?.currentItem else { return nil }
        let duration = CMTimeGetSeconds(item.duration)
        return duration.isFinite && duration > 0 ? duration : nil
    }

    private func installObserver() {
        guard let player else { return }
        let interval = CMTime(seconds: 0.25, preferredTimescale: 4)
        // ``addPeriodicTimeObserver`` invokes its closure on the
        // ``DispatchQueue`` we hand it, but Swift 6 strict
        // concurrency can't infer that ``.main`` aligns with the
        // ``@MainActor`` isolation of this store. Hopping through
        // ``MainActor.assumeIsolated`` lets the mutations stay
        // synchronous (no extra Task hop) while satisfying the
        // checker — we already know we're on the main queue.
        timeObserver = player.addPeriodicTimeObserver(
            forInterval: interval, queue: .main
        ) { [weak self] _ in
            MainActor.assumeIsolated {
                self?.tick()
            }
        }
    }

    private func tick() {
        guard let current = playerCurrentSeconds else { return }
        if let duration = playerTotalSeconds, duration > 0 {
            totalSeconds = duration
            isLoading = false
        }
        let total = totalSeconds
        guard total > 0 else { return }
        currentSeconds = current
        progress = min(max(current / total, 0), 1)
        // Auto-reset when playback finished — keeps the UI
        // honest about whether anything's currently playing.
        if current >= total - 0.05 {
            player?.seek(to: .zero)
            player?.pause()
            isPlaying = false
            progress = 0
        }
    }
}

private func formatPreviewSeconds(_ seconds: Double) -> String {
    let total = Int(seconds.rounded())
    let h = total / 3600
    let m = (total % 3600) / 60
    let s = total % 60
    if h > 0 {
        return String(format: "%d:%02d:%02d", h, m, s)
    }
    return String(format: "%d:%02d", m, s)
}

private func formatHMS(_ seconds: Double) -> String {
    let total = max(Int(seconds.rounded()), 0)
    let h = total / 3600
    let m = (total % 3600) / 60
    let s = total % 60
    return String(format: "%02d:%02d:%02d", h, m, s)
}

struct RunSetupView: View {
    @EnvironmentObject private var settings: SettingsStore
    @EnvironmentObject private var queue: QueueStore
    @Environment(\.dismiss) private var dismiss
    @State private var currentIndex = 0
    var onStart: () -> Void

    private var clampedIndex: Int {
        guard !queue.items.isEmpty else { return 0 }
        return min(max(currentIndex, 0), queue.items.count - 1)
    }

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Préparer le traitement")
                        .font(.title.bold())
                    Text(queue.items.count > 1
                         ? "Vérifiez chaque fichier puis lancez la file."
                         : "Vérifiez les paramètres puis lancez le traitement.")
                        .foregroundStyle(.secondary)
                }
                Spacer()
            }
            .padding(22)
            Divider()

            // ScrollView so the whole form (per-file panel + global
            // settings) fits even on a 13" laptop — the previous
            // layout grew without bounds and pushed the action bar
            // off-screen.
            ScrollView {
                VStack(spacing: 14) {
                    if !queue.items.isEmpty {
                        PerFileSetupPanel(
                            item: $queue.items[clampedIndex],
                            index: clampedIndex,
                            total: queue.items.count
                        )
                        .padding(.horizontal, 22)
                        .padding(.top, 14)
                    }
                    RunBatchSettingsForm()
                        .environmentObject(settings)
                }
                .padding(.bottom, 14)
            }
            .frame(minHeight: 360)

            Divider()
            HStack {
                Button("Annuler") { dismiss() }
                    .keyboardShortcut(.cancelAction)
                Spacer()
                if queue.items.count > 1 {
                    Button {
                        currentIndex = max(0, clampedIndex - 1)
                    } label: {
                        Label("Précédent", systemImage: "chevron.left")
                    }
                    .disabled(clampedIndex == 0)
                    Text("\(clampedIndex + 1) / \(queue.items.count)")
                        .font(.callout.monospacedDigit())
                        .foregroundStyle(.secondary)
                    if clampedIndex < queue.items.count - 1 {
                        Button {
                            currentIndex = min(queue.items.count - 1, clampedIndex + 1)
                        } label: {
                            Label("Suivant", systemImage: "chevron.right")
                        }
                    }
                }
                Button(queue.items.count > 1 ? "Lancer la file" : "Lancer") {
                    onStart()
                }
                .buttonStyle(.borderedProminent)
                .keyboardShortcut(.defaultAction)
                .disabled(queue.items.isEmpty)
            }
            .padding(18)
        }
        .frame(minWidth: 720, minHeight: 640)
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
                    Text("Déposez vos enregistrements")
                        .font(.title3.weight(.semibold))
                    Text("Vidéos et audios sont acceptés. La file reste modifiable pendant le traitement.")
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
    @EnvironmentObject private var library: LibraryStore
    @EnvironmentObject private var queue: QueueStore
    @State private var editingRow: LibraryRow?
    @State private var rowToDelete: LibraryRow?
    @State private var expandedJobIDs: Set<Int> = []
    @State private var queueNotice: String?
    // The "Fichier" + "Actions" columns are always on — they're
    // the table's reason to exist. Every other column is opt-in.
    // The first three (status / updated / artefacts) default ON
    // because they cover the 90 % case; the last two default OFF
    // so first-time users don't see two empty bonus columns.
    @AppStorage("libraryShowsStatusColumn") private var showsStatusColumn = true
    @AppStorage("libraryShowsUpdatedColumn") private var showsUpdatedColumn = true
    @AppStorage("libraryShowsArtefactsColumn") private var showsArtefactsColumn = true
    @AppStorage("libraryShowsSpeakersColumn") private var showsSpeakersColumn = false
    @AppStorage("libraryShowsProjectSizeColumn") private var showsProjectSizeColumn = false
    /// Sort order driving the table — defaults to most-recently
    /// updated first, mirroring what the engine returns from
    /// ``library_list``. Columns toggle between ascending and
    /// descending when the user clicks their header.
    @State private var sortOrder: [KeyPathComparator<LibraryDisplayRow>] = [
        .init(\.sortableUpdatedAt, order: .reverse),
    ]
    private var sortedRows: [LibraryRow] {
        // Pull the parent jobs and sort by whichever key the user
        // picked on the header. Children (artefact rows) follow
        // their parent in the flatMap below regardless of the
        // sort key, which preserves the visual hierarchy.
        let displayJobs = library.rows.map { LibraryDisplayRow(job: $0) }
        let sortedDisplay = displayJobs.sorted(using: sortOrder)
        return sortedDisplay.map(\.job)
    }

    private var displayRows: [LibraryDisplayRow] {
        sortedRows.flatMap { row in
            var rows = [LibraryDisplayRow(job: row)]
            if expandedJobIDs.contains(row.id) {
                rows.append(contentsOf: row.artifacts.map { LibraryDisplayRow(job: row, artifact: $0) })
            }
            return rows
        }
    }

    private var libraryHeader: some View {
        HStack(spacing: 8) {
            VStack(alignment: .leading, spacing: 4) {
                Text("Bibliothèque")
                    .font(.largeTitle.bold())
                Text("Retrouvez les compressions, transcriptions et rapports produits.")
                    .foregroundStyle(.secondary)
            }
            Spacer()
            Menu {
                Toggle("Statut", isOn: $showsStatusColumn)
                Toggle("Mis à jour", isOn: $showsUpdatedColumn)
                Toggle("Artefacts", isOn: $showsArtefactsColumn)
                Divider()
                Toggle("Interlocuteurs", isOn: $showsSpeakersColumn)
                Toggle("Poids du projet", isOn: $showsProjectSizeColumn)
            } label: {
                Label("Colonnes", systemImage: "tablecells")
                    .labelStyle(.iconOnly)
                    .frame(width: 24, height: 20)
            }
            .menuStyle(.button)
            .fixedSize()
            .help("Colonnes")
            Button {
                Task { await library.refresh() }
            } label: {
                Label("Actualiser", systemImage: "arrow.clockwise")
            }
        }
        .padding(24)
    }

    var body: some View {
        VStack(spacing: 0) {
            libraryHeader
            Divider()
            if library.isLoading && library.rows.isEmpty {
                ProgressView("Chargement de la bibliothèque…")
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else if library.rows.isEmpty {
                EmptyStateView(
                    title: "Aucun élément chargé",
                    systemImage: "tray",
                    message: "Actualisez la bibliothèque pour lire les traitements depuis le moteur."
                )
            } else {
                VStack(spacing: 0) {
                    Table(
                        displayRows,
                        sortOrder: $sortOrder
                    ) {
                        TableColumn("Fichier", value: \.sortableTitle) { displayRow in
                            if let artifact = displayRow.artifact {
                                ArtifactTreeNameCell(artifact: artifact)
                            } else {
                                HStack(spacing: 8) {
                                    Button {
                                        toggleExpanded(displayRow.job)
                                    } label: {
                                        Image(systemName: expandedJobIDs.contains(displayRow.job.id) ? "chevron.down" : "chevron.right")
                                            .font(.caption.weight(.semibold))
                                    }
                                    .buttonStyle(.plain)
                                    .frame(width: 18)
                                    Text(displayRow.job.customTitleOrFilename)
                                        .font(.headline)
                                        .lineLimit(1)
                                        .truncationMode(.tail)
                                }
                                .contentShape(Rectangle())
                                .onTapGesture {
                                    toggleExpanded(displayRow.job)
                                }
                            }
                        }
                        .width(min: 360, ideal: 520)

                        if showsStatusColumn {
                            TableColumn("Statut", value: \.sortableStatus) { displayRow in
                                if displayRow.artifact == nil {
                                    Image(systemName: statusIconName(displayRow.job.status))
                                        .foregroundStyle(statusColor(displayRow.job.status))
                                        .help(localizedStatus(displayRow.job.status))
                                        // Centre the glyph in the column;
                                        // a left-aligned icon hugged the
                                        // column edge and read like a list
                                        // bullet rather than a status.
                                        .frame(maxWidth: .infinity, alignment: .center)
                                }
                            }
                            .width(min: 54, ideal: 64, max: 80)
                        }

                        if showsUpdatedColumn {
                            TableColumn("Mis à jour", value: \.sortableUpdatedAt) { displayRow in
                                if displayRow.artifact == nil {
                                    Text(displayRow.job.updated_at ?? displayRow.job.created_at ?? "-")
                                        .foregroundStyle(.secondary)
                                        .lineLimit(2)
                                }
                            }
                            .width(min: 140, ideal: 170)
                        }

                        if showsArtefactsColumn {
                            TableColumn("Artefacts") { displayRow in
                                if let artifact = displayRow.artifact {
                                    Text(artifactSubtitle(artifact))
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                        .lineLimit(1)
                                } else {
                                    ArtifactDots(row: displayRow.job)
                                }
                            }
                            .width(min: 190, ideal: 240)
                        }

                        if showsSpeakersColumn {
                            TableColumn("Interlocuteurs", value: \.sortableSpeakerListing) { displayRow in
                                if displayRow.artifact == nil {
                                    SpeakerListingCell(names: displayRow.job.displayedSpeakerNames)
                                }
                            }
                            .width(min: 160, ideal: 220)
                        }

                        if showsProjectSizeColumn {
                            TableColumn("Poids", value: \.sortableTotalBytes) { displayRow in
                                if displayRow.artifact == nil {
                                    Text(displayRow.displayedTotalBytes)
                                        .font(.callout.monospacedDigit())
                                        .foregroundStyle(.secondary)
                                        .frame(maxWidth: .infinity, alignment: .trailing)
                                }
                            }
                            .width(min: 80, ideal: 100, max: 140)
                        }

                        TableColumn("Actions") { displayRow in
                            if let artifact = displayRow.artifact {
                                ArtifactInlineActions(row: displayRow.job, artifact: artifact) { path in
                                    if artifact.canRerun {
                                        enqueue(displayRow.job, sourcePath: path, label: "Relance ajoutée à la file d'attente")
                                    } else {
                                        enqueue(path, label: "Artefact ajouté à la file d'attente")
                                    }
                                }
                            } else {
                                LibraryTableActionsView(
                                    row: displayRow.job,
                                    onInspect: { toggleExpanded(displayRow.job) },
                                    onRerun: { enqueue(displayRow.job, label: "Relance ajoutée à la file d'attente") },
                                    onEditContext: { editingRow = displayRow.job },
                                    onDelete: { rowToDelete = displayRow.job }
                                )
                            }
                        }
                        .width(min: 120, ideal: 150)
                    }
                }
                .overlay(alignment: .bottomLeading) {
                    if let message = library.errorMessage, !message.isEmpty {
                        InlineErrorView(message: message)
                            .padding()
                    }
                }
                .overlay(alignment: .bottom) {
                    if let queueNotice {
                        Label(queueNotice, systemImage: "checkmark.circle.fill")
                            .font(.callout.weight(.medium))
                            .padding(.horizontal, 12)
                            .padding(.vertical, 8)
                            .background(.regularMaterial, in: Capsule())
                            .foregroundStyle(.green)
                            .padding(.bottom, 12)
                            .transition(.opacity.combined(with: .move(edge: .bottom)))
                    }
                }
            }
        }
        .sheet(item: $editingRow) { row in
            LibraryContextEditor(row: row)
                .environmentObject(library)
                .environmentObject(queue)
        }
        .sheet(item: $rowToDelete) { row in
            LibraryDeletionSheet(row: row)
                .environmentObject(library)
        }
        .task {
            if library.rows.isEmpty {
                await library.refresh()
            }
        }
        .onChange(of: library.rows) { _, rows in
            let validIDs = Set(rows.map(\.id))
            expandedJobIDs = expandedJobIDs.intersection(validIDs)
        }
    }

    private func toggleExpanded(_ row: LibraryRow) {
        if expandedJobIDs.contains(row.id) {
            expandedJobIDs.remove(row.id)
        } else {
            expandedJobIDs.insert(row.id)
        }
    }

    private func enqueue(_ path: String, label: String) {
        queue.add(urls: [URL(fileURLWithPath: path)])
        withAnimation(.easeInOut(duration: 0.16)) {
            queueNotice = label
        }
        Task {
            try? await Task.sleep(nanoseconds: 1_700_000_000)
            await MainActor.run {
                withAnimation(.easeInOut(duration: 0.16)) {
                    if queueNotice == label {
                        queueNotice = nil
                    }
                }
            }
        }
    }

    private func enqueue(_ row: LibraryRow, sourcePath: String? = nil, label: String) {
        queue.addRerun(row: row, sourcePath: sourcePath)
        withAnimation(.easeInOut(duration: 0.16)) {
            queueNotice = label
        }
        Task {
            try? await Task.sleep(nanoseconds: 1_700_000_000)
            await MainActor.run {
                withAnimation(.easeInOut(duration: 0.16)) {
                    if queueNotice == label {
                        queueNotice = nil
                    }
                }
            }
        }
    }

    private func artifactSubtitle(_ artifact: LibraryArtifact) -> String {
        guard let path = existingPath(for: artifact) else { return "Non généré" }
        let url = URL(fileURLWithPath: path)
        let size = fileSizeLabel(path)
        return size.isEmpty ? url.lastPathComponent : "\(url.lastPathComponent) · \(size)"
    }

    private func existingPath(for artifact: LibraryArtifact) -> String? {
        guard let path = artifact.path, !path.isEmpty else { return nil }
        return FileManager.default.fileExists(atPath: path) ? path : nil
    }
}

struct ModelsView: View {
    @EnvironmentObject private var models: ModelStore

    var body: some View {
        VStack(spacing: 0) {
            ListHeaderView(
                title: "Modèles locaux",
                subtitle: "Téléchargez les modèles avant une réunion pour éviter les surprises.",
                actionTitle: "Actualiser",
                actionSystemImage: "arrow.clockwise"
            ) {
                Task { await models.refresh() }
            }
            Divider()
            if models.isLoading && models.models.isEmpty {
                ProgressView("Chargement des modèles…")
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else if models.models.isEmpty {
                EmptyStateView(
                    title: "Catalogue non chargé",
                    systemImage: "shippingbox",
                    message: "Actualisez pour afficher les modèles Whisper, texte et audio."
                )
            } else {
                Table(models.models) {
                    TableColumn("Modèle") { row in
                        VStack(alignment: .leading) {
                            Text(row.label)
                                .font(.headline)
                            Text(row.id)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }
                    TableColumn("Famille", value: \.family)
                    TableColumn("État") { row in
                        StatusText(row.cached ? "Téléchargé" : "À télécharger")
                    }
                    TableColumn("Actions") { row in
                        ModelActionsView(row: row)
                    }
                }
                .overlay(alignment: .bottomLeading) {
                    if let message = models.errorMessage, !message.isEmpty {
                        InlineErrorView(message: message)
                            .padding()
                    }
                }
            }
        }
        .task {
            if models.models.isEmpty {
                await models.refresh()
            }
        }
    }
}

struct VocabularyView: View {
    @EnvironmentObject private var settings: SettingsStore
    @State private var draft = ""

    private var rows: [VocabularyDisplayRow] {
        let usage = settings.vocabularyUsage
        return settings.vocabularyCatalog.map {
            VocabularyDisplayRow(term: $0, usage: usage[$0] ?? 0)
        }
    }

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Vocabulaire")
                        .font(.largeTitle.bold())
                    Text("Conservez les noms propres et termes techniques, puis choisissez-les au lancement de chaque réunion.")
                        .foregroundStyle(.secondary)
                }
                Spacer()
            }
            .padding(24)
            Divider()
            HStack(spacing: 10) {
                TextField("Ajouter un mot ou une expression", text: $draft)
                    .textFieldStyle(.roundedBorder)
                    .onSubmit { addDraft() }
                Button {
                    addDraft()
                } label: {
                    Label("Ajouter", systemImage: "plus")
                }
                .disabled(draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
            .padding(.horizontal, 24)
            .padding(.vertical, 14)
            Divider()
            if rows.isEmpty {
                EmptyStateView(
                    title: "Aucun terme enregistré",
                    systemImage: "text.word.spacing",
                    message: "Ajoutez les noms de clients, outils, produits ou expressions métier que Whisper doit connaître."
                )
            } else {
                Table(rows) {
                    TableColumn("Terme") { row in
                        Text(row.term)
                            .font(.headline)
                    }
                    TableColumn("Utilisé") { row in
                        Text(row.usage > 0 ? "\(row.usage)" : "—")
                            .font(.callout.monospacedDigit())
                            .foregroundStyle(.secondary)
                    }
                    .width(min: 70, ideal: 90, max: 110)
                    TableColumn("Actions") { row in
                        Button(role: .destructive) {
                            settings.removeVocabularyTerm(row.term)
                        } label: {
                            Label("Supprimer", systemImage: "trash")
                        }
                        .labelStyle(.iconOnly)
                        .buttonStyle(.borderless)
                    }
                    .width(min: 80, ideal: 90, max: 120)
                }
            }
        }
    }

    private func addDraft() {
        settings.addVocabularyTerm(draft)
        draft = ""
    }
}

struct VocabularyDisplayRow: Identifiable {
    var term: String
    var usage: Int
    var id: String { term }
}

struct SettingsView: View {
    @EnvironmentObject private var settings: SettingsStore
    @EnvironmentObject private var updater: UpdateStore
    @EnvironmentObject private var engine: EngineProcess
    @Environment(\.dismiss) private var dismiss
    @State private var hfStatus = ""
    @State private var isCheckingHF = false

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Text("Réglages")
                    .font(.title.bold())
                Spacer()
                Button("OK") { dismiss() }
                    .keyboardShortcut(.defaultAction)
            }
            .padding(22)
            Divider()
            Form {
                Section("Mise à jour") {
                    HStack {
                        VStack(alignment: .leading) {
                            Text("Version actuelle : \(updater.currentVersion)")
                            switch updater.state {
                            case .idle:
                                Text("Vérifié à l'instant")
                                    .font(.caption).foregroundStyle(.secondary)
                            case .checking:
                                Text("Recherche en cours…")
                                    .font(.caption).foregroundStyle(.secondary)
                            case .available(let info):
                                Text("Version \(info.tag_name) disponible")
                                    .font(.caption).foregroundStyle(.teal)
                            case .upToDate:
                                Text("Vous êtes à jour")
                                    .font(.caption).foregroundStyle(.secondary)
                            case .downloading(let pct):
                                Text("Téléchargement… \(Int(pct * 100))%")
                                    .font(.caption).foregroundStyle(.teal)
                            case .readyToInstall:
                                Text("Prêt à installer")
                                    .font(.caption).foregroundStyle(.teal)
                            case .error(let msg):
                                Text(msg)
                                    .font(.caption).foregroundStyle(.red)
                            }
                        }
                        Spacer()
                        if case .readyToInstall(let url, _) = updater.state {
                            Button("Installer et redémarrer") {
                                updater.applyUpdate(zipURL: url)
                            }
                            .buttonStyle(.borderedProminent)
                        } else if case .available(let info) = updater.state {
                            Button("Télécharger") {
                                Task { await updater.downloadAndPrepare(info: info) }
                            }
                        } else {
                            Button("Vérifier") {
                                Task { await updater.checkUpdates() }
                            }
                            .disabled(updater.state == .checking)
                        }
                    }
                    SecureField("GitHub Token (optionnel)", text: $settings.githubToken)
                    Text("Requis seulement si le dépôt est privé.")
                        .font(.caption).foregroundStyle(.secondary)
                }

                Section("Sortie") {
                    TextField("Dossier", text: $settings.outputDir)
                    Toggle("Supprimer le fichier source après copie", isOn: $settings.deleteSourceAfterCopy)
                    Text("L'original est supprimé uniquement après sa copie dans le dossier de travail.")
                        .font(.caption)
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
                    // One picker replaces the previous patchwork of
                    // VAD/multipass/per-speaker/web toggles. The engine
                    // maps the preset string to the right combination.
                    // Power users can still drill down via "Avancé".
                    Picker("Qualité", selection: $settings.qualityPreset) {
                        ForEach(TranscriptionQualityPreset.allCases) { preset in
                            Text(preset.displayName).tag(preset.rawValue)
                        }
                    }
                    if let preset = TranscriptionQualityPreset(rawValue: settings.qualityPreset) {
                        Text(preset.summary)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    TextField("Modèle Whisper", text: $settings.whisperModel)
                }
                Section("Avancé") {
                    Toggle("Détection des locuteurs", isOn: $settings.diarizationEnabled)
                    Toggle("Réécoute IA multimodale", isOn: $settings.audioRecheckEnabled)
                    Text(
                        "Ces bascules complètent le réglage Qualité. La détection des locuteurs nécessite un token Hugging Face ; la réécoute IA est expérimentale et coûteuse en temps. Le nombre d'intervenants attendu se règle au lancement de chaque traitement."
                    )
                    .font(.caption)
                    .foregroundStyle(.secondary)
                }
                Section("Voix mémorisées") {
                    SpeakerProfilesSection()
                }
                Section("Connexion Odoo") {
                    OdooConnectionSection()
                }
                Section("Hugging Face") {
                    SecureField("Token Read", text: $settings.hfToken)
                    HStack {
                        Button {
                            openHuggingFaceTokens()
                        } label: {
                            Label("Créer ou gérer le token", systemImage: "person.crop.circle.badge.key")
                        }
                        Button {
                            Task { await checkHuggingFaceAccess() }
                        } label: {
                            if isCheckingHF {
                                ProgressView()
                                    .controlSize(.small)
                            } else {
                                Label("Vérifier l'accès", systemImage: "checkmark.shield")
                            }
                        }
                        .disabled(settings.hfToken.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || isCheckingHF)
                    }
                    if !hfStatus.isEmpty {
                        Text(hfStatus)
                            .font(.callout)
                            .foregroundStyle(hfStatus.hasPrefix("OK") ? .green : .secondary)
                    } else {
                        Text("Requis pour la détection des locuteurs pyannote. L'app vérifie le token et l'acceptation des conditions des modèles.")
                            .foregroundStyle(.secondary)
                    }
                }
                Section("Vocabulaire conservé") {
                    Text("\(settings.vocabularyCatalog.count) terme(s) dans le catalogue. L'onglet Vocabulaire permet d'ajouter, supprimer et prioriser les termes selon l'usage.")
                        .foregroundStyle(.secondary)
                }
                Section("Diagnostic") {
                    // Used to live in the Run Setup form. Moved
                    // here because exporting logs is a one-off
                    // troubleshooting action, not something the
                    // user does on every batch.
                    Button {
                        engine.run(arguments: EngineProcess.defaultPythonArguments(["export-logs"]))
                    } label: {
                        Label("Exporter les logs", systemImage: "doc.zipper")
                    }
                    Text("Une archive ZIP est déposée sur le Bureau pour partage avec le support.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            .formStyle(.grouped)
            .padding()
        }
        .frame(minWidth: 620, minHeight: 520)
    }

    private func checkHuggingFaceAccess() async {
        isCheckingHF = true
        defer { isCheckingHF = false }
        let result = await EngineProcess.runCommand(
            arguments: EngineProcess.defaultPythonArguments(["hf-check", "--token", settings.hfToken])
        )
        if result.status != 0 {
            hfStatus = result.events.last?.message ?? result.rawOutput
            return
        }
        guard let data = result.rawOutput.data(using: .utf8),
              let payload = try? JSONDecoder().decode(HuggingFaceCheckResponse.self, from: data) else {
            hfStatus = "Réponse Hugging Face illisible."
            return
        }
        let missing = payload.checks.filter { !$0.ok }
        if missing.isEmpty {
            let name = payload.account.name ?? payload.account.fullname ?? "compte connecté"
            hfStatus = "OK · \(name) · accès pyannote vérifié."
        } else {
            let labels = missing.map(\.label).joined(separator: ", ")
            hfStatus = "Accès incomplet : \(labels). Ouvrez Hugging Face et acceptez les conditions."
        }
    }
}

private struct HuggingFaceCheckResponse: Decodable {
    var account: HuggingFaceAccount
    var checks: [HuggingFaceModelCheck]
}

private struct HuggingFaceAccount: Decodable {
    var name: String?
    var fullname: String?
}

private struct HuggingFaceModelCheck: Decodable {
    var label: String
    var ok: Bool
}

struct StatusBarView: View {
    @EnvironmentObject private var engine: EngineProcess

    private var latestProgressEvent: EngineEvent? {
        engine.events.last(where: { $0.event == .progress })
    }

    private var progress: Double? {
        latestProgressEvent?.pct
    }

    private func statusText(now: Date) -> String {
        let message = engine.lastError ?? latestProgressEvent?.message ?? engine.events.last?.message ?? "Prêt."
        guard engine.isRunning, let startedAt = engine.runStartedAt else {
            return message
        }

        var parts = [message, "écoulé \(formatDuration(now.timeIntervalSince(startedAt)))"]
        if let eta = latestProgressEvent?.eta_seconds, eta.isFinite, eta > 0 {
            parts.append("reste ~\(formatDuration(eta))")
        }
        return parts.joined(separator: " · ")
    }

    private func formatDuration(_ seconds: TimeInterval) -> String {
        let total = max(Int(seconds.rounded()), 0)
        let hours = total / 3600
        let minutes = (total % 3600) / 60
        let seconds = total % 60
        if hours > 0 {
            return "\(hours):\(String(format: "%02d", minutes)):\(String(format: "%02d", seconds))"
        }
        return "\(minutes):\(String(format: "%02d", seconds))"
    }

    var body: some View {
        TimelineView(.periodic(from: .now, by: 1)) { timeline in
            HStack(spacing: 10) {
                if engine.isRunning {
                    ProgressView(value: progress.map { $0 / 100.0 })
                        .frame(width: 120)
                }
                Text(statusText(now: timeline.date))
                    .lineLimit(1)
                    .monospacedDigit()
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

struct InlineErrorView: View {
    var message: String

    var body: some View {
        Label(message, systemImage: "exclamationmark.triangle")
            .font(.callout)
            .lineLimit(3)
            .padding(10)
            .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 8))
            .foregroundStyle(.red)
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

struct LibraryArtifact: Identifiable, Equatable {
    var id: String { kind }
    var kind: String
    var title: String
    var path: String?
    var canRerun: Bool = false
}

struct LibraryDisplayRow: Identifiable, Equatable {
    var job: LibraryRow
    var artifact: LibraryArtifact?

    var id: String {
        if let artifact {
            return "\(job.id)-\(artifact.kind)"
        }
        return "\(job.id)"
    }
}

extension LibraryRow {
    var artifacts: [LibraryArtifact] {
        [
            LibraryArtifact(kind: "source", title: "Source copiée", path: copiedSourcePath, canRerun: true),
            LibraryArtifact(kind: "compressed", title: "Compressé", path: compressed_path),
            LibraryArtifact(kind: "transcript", title: "Transcription", path: transcript_path),
            LibraryArtifact(kind: "enhanced", title: "Améliorée", path: enhanced_transcript_path),
            LibraryArtifact(kind: "review", title: "Rapport", path: review_path),
        ]
    }

    /// Friendly speaker names for the optional Interlocuteurs
    /// column. Empty strings (placeholders the LLM didn't manage to
    /// name) are filtered out so the column stays readable. ID
    /// strings are sorted alphabetically so the listing is stable
    /// across refreshes.
    var displayedSpeakerNames: [String] {
        let map = speakerMap
        let names = map
            .compactMap { (key, value) -> String? in
                let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !trimmed.isEmpty else { return nil }
                return trimmed
            }
        // De-dupe — when the rename pass mapped the friendly name
        // to itself we'd otherwise see "Robin · Robin".
        var seen: Set<String> = []
        var deduped: [String] = []
        for name in names {
            if seen.insert(name).inserted {
                deduped.append(name)
            }
        }
        // Stable order across refreshes; otherwise the Table
        // recomputes diffs every time the dict iteration order
        // shifts.
        deduped.sort()
        // If we only have placeholders, expose them too — the column
        // becomes "SPEAKER_00 · SPEAKER_01" which is still useful
        // to the user even if no friendly names landed.
        if deduped.isEmpty {
            return map.keys.sorted()
        }
        return deduped
    }
}

private func formatFileBytes(_ bytes: Int64) -> String {
    ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)
}

extension LibraryDisplayRow {
    /// Stable, comparable values used by the SwiftUI Table sort
    /// machinery. The Table sorts ``LibraryDisplayRow`` items
    /// directly, so each sortable column needs a key path on this
    /// type. Artefact rows fall back to their parent job's value so
    /// they always sort alongside their parent.
    var sortableTitle: String { job.customTitleOrFilename.lowercased() }
    var sortableStatus: String {
        // Numeric prefix encodes the natural priority order
        // (RUNNING > QUEUED > COMPLETED > FAILED), matching what
        // the user reads off the icon.
        switch (job.status ?? "").uppercased() {
        case "RUNNING": return "0"
        case "QUEUED", "PENDING": return "1"
        case "COMPLETED": return "2"
        case "FAILED": return "3"
        default: return "4"
        }
    }
    /// ISO-8601 timestamps compare lexically, so this is good
    /// enough for sort purposes; we fall back to ``created_at``
    /// when ``updated_at`` is blank.
    var sortableUpdatedAt: String { job.updated_at ?? job.created_at ?? "" }
    /// 0 for legacy rows so they group at the bottom on descending
    /// sort, top on ascending. Either way they don't poison the
    /// real values.
    var sortableTotalBytes: Int64 { job.total_bytes ?? 0 }
    /// Sort by interlocuteur count first, then alphabetical.
    var sortableSpeakerListing: String {
        let names = job.displayedSpeakerNames
        return String(format: "%03d-%@", names.count, names.joined(separator: ","))
    }
    /// Human-readable rendering of ``total_bytes``. "—" for legacy
    /// rows; the byte formatter for everything else.
    var displayedTotalBytes: String {
        guard let bytes = job.total_bytes else { return "—" }
        return formatFileBytes(bytes)
    }
}

struct LibraryTableActionsView: View {
    var row: LibraryRow
    var onInspect: () -> Void
    var onRerun: () -> Void
    var onEditContext: () -> Void
    var onDelete: () -> Void

    var body: some View {
        HStack(spacing: 6) {
            Button(action: onInspect) {
                Label("Détails", systemImage: "sidebar.right")
            }
            .labelStyle(.iconOnly)
            .help("Afficher les artefacts")

            Button {
                onRerun()
            } label: {
                Label("Relancer", systemImage: "arrow.clockwise")
            }
            .labelStyle(.iconOnly)
            .help("Ajouter la source copiée à la file")

            Button(action: onEditContext) {
                Label("Contexte", systemImage: "person.2.badge.gearshape")
            }
            .labelStyle(.iconOnly)
            .help("Modifier les interlocuteurs et le vocabulaire")

            Button(role: .destructive, action: onDelete) {
                Label("Supprimer", systemImage: "trash")
            }
            .labelStyle(.iconOnly)
            .help("Supprimer de la bibliothèque")
        }
    }
}

struct ArtifactTreeNameCell: View {
    var artifact: LibraryArtifact

    var body: some View {
        HStack(spacing: 8) {
            Spacer()
                .frame(width: 26)
            ArtifactDot(label: String(artifact.title.prefix(1)), isPresent: pathExists(artifact.path))
            Text(artifact.title)
                .font(.callout)
                .foregroundStyle(pathExists(artifact.path) ? .primary : .secondary)
                .lineLimit(1)
                .truncationMode(.tail)
        }
    }
}

struct ArtifactInlineActions: View {
    var row: LibraryRow
    var artifact: LibraryArtifact
    var onRerun: (String) -> Void

    private var existingPath: String? {
        guard let path = artifact.path, !path.isEmpty else { return nil }
        return FileManager.default.fileExists(atPath: path) ? path : nil
    }

    var body: some View {
        HStack(spacing: 6) {
            if let path = existingPath {
                Button {
                    openPath(path)
                } label: {
                    Label("Ouvrir", systemImage: "doc.text")
                }
                .labelStyle(.iconOnly)
                .help("Ouvrir le fichier")

                Button {
                    revealInFinder(path)
                } label: {
                    Label("Finder", systemImage: "folder")
                }
                .labelStyle(.iconOnly)
                .help("Afficher dans le Finder")

                if artifact.canRerun {
                    Button {
                        onRerun(path)
                    } label: {
                        Label("Relancer", systemImage: "arrow.clockwise")
                    }
                    .labelStyle(.iconOnly)
                    .help("Relancer depuis cette source")
                }
            }
        }
    }
}

struct ArtifactDots: View {
    var row: LibraryRow

    var body: some View {
        HStack(spacing: 4) {
            ArtifactDot(label: "S", isPresent: pathExists(row.copiedSourcePath))
            ArtifactDot(label: "C", isPresent: pathExists(row.compressed_path))
            ArtifactDot(label: "T", isPresent: pathExists(row.transcript_path))
            ArtifactDot(label: "A", isPresent: pathExists(row.enhanced_transcript_path))
            ArtifactDot(label: "R", isPresent: pathExists(row.review_path))
        }
    }
}

/// Renders a list of speaker names as small capsules. Up to three
/// fit fully; the remainder collapses into a single "+N" capsule
/// with a tooltip listing the hidden names. Avoids the wall-of-text
/// problem when a meeting has 6+ participants and the column would
/// otherwise wrap or truncate ugly.
struct SpeakerListingCell: View {
    var names: [String]

    private var visibleNames: [String] {
        Array(names.prefix(3))
    }
    private var overflowCount: Int {
        max(0, names.count - visibleNames.count)
    }
    private var overflowTooltip: String {
        names.dropFirst(visibleNames.count).joined(separator: ", ")
    }

    var body: some View {
        if names.isEmpty {
            Text("—")
                .foregroundStyle(.secondary)
        } else {
            HStack(spacing: 4) {
                ForEach(visibleNames, id: \.self) { name in
                    SpeakerChip(text: name)
                }
                if overflowCount > 0 {
                    SpeakerChip(text: "+\(overflowCount)")
                        .help(overflowTooltip)
                }
            }
            .lineLimit(1)
        }
    }
}

struct SpeakerChip: View {
    var text: String

    var body: some View {
        Text(text)
            .font(.caption)
            .lineLimit(1)
            .padding(.horizontal, 7)
            .padding(.vertical, 2)
            .background(Color.teal.opacity(0.18), in: Capsule())
            .foregroundStyle(.primary)
    }
}

struct LibraryContextEditor: View {
    @EnvironmentObject private var library: LibraryStore
    @EnvironmentObject private var queue: QueueStore
    @Environment(\.dismiss) private var dismiss
    var row: LibraryRow
    @State private var speakers: [SpeakerEditRow]
    @State private var samples: [SpeakerSample] = []
    @State private var sampleIndexBySpeaker: [String: Int] = [:]
    @State private var loadingSamples = false
    @State private var currentSound: NSSound?
    @State private var reviewNotice: String?
    @State private var termsText: String

    init(row: LibraryRow) {
        self.row = row
        let speakerMap = row.speakerMap
        let mappedNames = Set(speakerMap.filter { $0.key != $0.value }.map(\.value))
        let initialSpeakers = speakerMap
            .filter { !(($0.key == $0.value) && mappedNames.contains($0.value)) }
            .sorted { $0.key < $1.key }
            .map { SpeakerEditRow(id: $0.key, name: $0.value) }
        _speakers = State(initialValue: initialSpeakers)
        _termsText = State(initialValue: row.technicalTerms.joined(separator: "\n"))
    }

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Contexte de transcription")
                        .font(.title.bold())
                    Text(row.filename)
                        .foregroundStyle(.secondary)
                }
                Spacer()
            }
            .padding(22)
            Divider()
            Form {
                Section("Interlocuteurs") {
                    if speakers.isEmpty {
                        Text("Aucun interlocuteur détecté pour cette transcription.")
                            .foregroundStyle(.secondary)
                    } else {
                        ForEach($speakers) { $speaker in
                            VStack(alignment: .leading, spacing: 5) {
                                HStack {
                                    if speaker.id != speaker.name {
                                        Text(speaker.id)
                                            .font(.body.monospaced())
                                            .foregroundStyle(.secondary)
                                            .frame(width: 120, alignment: .leading)
                                    }
                                    TextField(speaker.name.isEmpty ? speaker.id : speaker.name, text: $speaker.name)
                                    speakerSampleControls(for: speaker)
                                }
                                if let stats = speakerStatsText(for: speaker) {
                                    Text(stats)
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                        .padding(.leading, speaker.id != speaker.name ? 128 : 0)
                                }
                                if let attendees = unusedAttendees(for: speaker),
                                   !attendees.isEmpty {
                                    odooAttendeeChips(for: $speaker, attendees: attendees)
                                        .padding(.leading, speaker.id != speaker.name ? 128 : 0)
                                }
                            }
                        }
                    }
                    if let reviewNotice {
                        Label(reviewNotice, systemImage: "checkmark.circle.fill")
                            .font(.caption)
                            .foregroundStyle(.green)
                    }
                    Text("Saisissez uniquement le nom à afficher. Les fichiers texte existants sont réécrits quand un nom change.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Section("Termes techniques") {
                    TextEditor(text: $termsText)
                        .font(.body.monospaced())
                        .frame(minHeight: 140)
                    Text("Un terme par ligne. Conservé avec cette transcription et réutilisable lors d'une relance.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            .formStyle(.grouped)
            .padding()
            Divider()
            HStack {
                Button("Annuler") { dismiss() }
                Spacer()
                Button("Enregistrer") {
                    save()
                }
                .buttonStyle(.borderedProminent)
                .keyboardShortcut(.defaultAction)
            }
            .padding(18)
        }
        .frame(minWidth: 680, minHeight: 560)
        .task {
            await loadSamples()
        }
    }

    @ViewBuilder
    private func speakerSampleControls(for speaker: SpeakerEditRow) -> some View {
        let availableSamples = samples(for: speaker)
        if let sample = selectedSample(for: speaker, in: availableSamples) {
            HStack(spacing: 6) {
                Button {
                    play(sample)
                } label: {
                    Label("Écouter", systemImage: "play.circle")
                }
                .labelStyle(.iconOnly)
                .help("Écouter \(samplePositionText(for: speaker, in: availableSamples))")

                if availableSamples.count > 1 {
                    Button {
                        nextSample(for: speaker, count: availableSamples.count)
                    } label: {
                        Label("Extrait suivant", systemImage: "forward.end")
                    }
                    .labelStyle(.iconOnly)
                    .help("Passer à l'extrait suivant")
                }

                Button {
                    flagForReview(sample)
                } label: {
                    Label("À revoir", systemImage: "exclamationmark.bubble")
                }
                .labelStyle(.iconOnly)
                .help("Marquer cet extrait comme ambigu et relancer la source en priorité")
            }
        } else if loadingSamples {
            ProgressView()
                .controlSize(.small)
        }
    }

    private func save() {
        let updatedSpeakers = Dictionary(
            uniqueKeysWithValues: speakers.map {
                ($0.id, $0.name.trimmingCharacters(in: .whitespacesAndNewlines))
            }
        )
        var renameMapping = updatedSpeakers
        for (key, oldValue) in row.speakerMap {
            if let newValue = updatedSpeakers[key], !oldValue.isEmpty, oldValue != newValue {
                renameMapping[oldValue] = newValue
            }
        }
        let terms = termsText
            .split(whereSeparator: \.isNewline)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        dismiss()
        Task {
            if !renameMapping.isEmpty {
                await library.renameSpeakers(row, mapping: renameMapping)
            }
            await library.updateContext(row, speakers: updatedSpeakers, technicalTerms: terms)
        }
    }

    private func loadSamples() async {
        loadingSamples = true
        // For jobs that ran on the new pipeline the speaker list
        // came in via the LibraryRow's ``speakerMap``. For older
        // jobs (or jobs that ran before the pipeline persistence
        // fix) we ask the engine to backfill it by parsing the
        // artefact files on disk — that way the sheet never shows
        // "Aucun interlocuteur détecté" when the transcript clearly
        // contains them.
        var seededFromDiscovery = false
        if speakers.isEmpty {
            let discovered = await library.discoverSpeakers(row)
            if !discovered.isEmpty {
                await MainActor.run {
                    speakers = discovered
                        .sorted { $0.key < $1.key }
                        .map { SpeakerEditRow(id: $0.key, name: $0.value) }
                }
                seededFromDiscovery = true
            }
        }
        let loaded = await library.speakerSamples(row)
        await MainActor.run {
            samples = loaded
            for sample in loaded where !speakers.contains(where: { $0.id == sample.speaker }) {
                speakers.append(SpeakerEditRow(id: sample.speaker, name: row.speakerMap[sample.speaker] ?? ""))
            }
            loadingSamples = false
            // No need to surface the discovery to the user — the
            // sheet just shows speakers now. We keep
            // ``seededFromDiscovery`` around for a possible future
            // telemetry hook ("how often is the backfill needed?")
            // without bloating the UX right now.
            _ = seededFromDiscovery
        }
    }

    private func samples(for speaker: SpeakerEditRow) -> [SpeakerSample] {
        samples
            .filter { $0.speaker == speaker.id || $0.speaker == speaker.name }
            .sorted {
                if ($0.index ?? 0) != ($1.index ?? 0) {
                    return ($0.index ?? 0) < ($1.index ?? 0)
                }
                return $0.start < $1.start
            }
    }

    private func selectedSample(for speaker: SpeakerEditRow, in availableSamples: [SpeakerSample]) -> SpeakerSample? {
        guard !availableSamples.isEmpty else { return nil }
        let rawIndex = sampleIndexBySpeaker[speaker.id] ?? 0
        return availableSamples[rawIndex % availableSamples.count]
    }

    private func nextSample(for speaker: SpeakerEditRow, count: Int) {
        guard count > 0 else { return }
        let rawIndex = sampleIndexBySpeaker[speaker.id] ?? 0
        sampleIndexBySpeaker[speaker.id] = (rawIndex + 1) % count
    }

    private func samplePositionText(for speaker: SpeakerEditRow, in availableSamples: [SpeakerSample]) -> String {
        guard !availableSamples.isEmpty else { return "un extrait" }
        let index = (sampleIndexBySpeaker[speaker.id] ?? 0) % availableSamples.count
        if availableSamples.count == 1 {
            return "l'extrait"
        }
        return "l'extrait \(index + 1)/\(availableSamples.count)"
    }

    private func speakerStatsText(for speaker: SpeakerEditRow) -> String? {
        guard let sample = samples(for: speaker).first,
              let utterances = sample.utterance_count,
              let totalDuration = sample.total_duration else { return nil }
        return "\(utterances) prise\(utterances > 1 ? "s" : "") de parole · \(durationLabel(totalDuration)) de parole"
    }

    /// Attendees from the Odoo meeting paired with this job whose
    /// names aren't already used by another speaker row. Returned
    /// only for rows that still carry a SPEAKER_NN placeholder or
    /// an empty name — once the user has typed a name the chips
    /// disappear so the sheet doesn't keep pestering them.
    private func unusedAttendees(for speaker: SpeakerEditRow) -> [OdooMeetingAttendee]? {
        guard let meeting = row.odooMeeting, !meeting.attendees.isEmpty else { return nil }
        let typedName = speaker.name.trimmingCharacters(in: .whitespacesAndNewlines)
        // Skip rows already named — the user has decided. We only
        // want to surface hints when there's still a placeholder
        // staring at them.
        let placeholderLooking = typedName.isEmpty
            || typedName.uppercased().hasPrefix("SPEAKER_")
            || typedName == speaker.id
        guard placeholderLooking else { return nil }
        // Filter out attendees already attached to another row so
        // we don't suggest the same name on two clusters.
        let claimed: Set<String> = Set(
            speakers
                .filter { $0.id != speaker.id }
                .map { $0.name.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() }
                .filter { !$0.isEmpty }
        )
        return meeting.attendees.filter { attendee in
            let lowered = attendee.name.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            return !lowered.isEmpty && !claimed.contains(lowered)
        }
    }

    /// Renders one chip per remaining attendee. Click → fills the
    /// text field, so the user confirms with a single tap instead
    /// of typing the full name.
    @ViewBuilder
    private func odooAttendeeChips(
        for speaker: Binding<SpeakerEditRow>,
        attendees: [OdooMeetingAttendee]
    ) -> some View {
        HStack(spacing: 6) {
            Image(systemName: "calendar.badge.checkmark")
                .foregroundStyle(.teal)
                .font(.caption)
            Text("Réunion Odoo :")
                .font(.caption)
                .foregroundStyle(.secondary)
            ForEach(attendees) { attendee in
                Button {
                    speaker.wrappedValue.name = attendee.name
                } label: {
                    Text(attendee.name)
                        .font(.caption)
                }
                .buttonStyle(.borderless)
                .padding(.horizontal, 8)
                .padding(.vertical, 3)
                .background(.teal.opacity(0.15), in: Capsule())
                .help(attendee.company.isEmpty ? attendee.email : "\(attendee.company) · \(attendee.email)")
            }
            Spacer()
        }
    }

    private func durationLabel(_ seconds: Double) -> String {
        let totalSeconds = max(Int(seconds.rounded()), 0)
        let minutes = totalSeconds / 60
        let remainingSeconds = totalSeconds % 60
        if minutes <= 0 {
            return "\(remainingSeconds) s"
        }
        if remainingSeconds == 0 {
            return "\(minutes) min"
        }
        return "\(minutes) min \(remainingSeconds) s"
    }

    private func play(_ sample: SpeakerSample) {
        currentSound?.stop()
        let sound = NSSound(contentsOfFile: sample.path, byReference: true)
        currentSound = sound
        sound?.play()
    }

    private func flagForReview(_ sample: SpeakerSample) {
        Task {
            let marked = await library.flagSpeakerSampleForReview(row, sample: sample)
            guard marked else { return }
            await MainActor.run {
                if let path = pathExists(row.copiedSourcePath) ? row.copiedSourcePath : row.source_path {
                    queue.addRerun(
                        row: row,
                        sourcePath: path,
                        focusNote: focusNote(for: sample),
                        prioritize: true
                    )
                    queue.requestAutoRun()
                }
                reviewNotice = "Extrait marqué à revoir et relance prioritaire demandée."
            }
        }
    }

    private func focusNote(for sample: SpeakerSample) -> String {
        let end = sample.start + sample.duration
        return String(
            format: "Passage interlocuteur à revoir: %@ de %.1fs à %.1fs. Vérifier diarisation, attribution du locuteur et transcription; l'utilisateur a signalé que plusieurs voix peuvent être présentes.",
            sample.speaker,
            sample.start,
            end
        )
    }
}

struct SpeakerEditRow: Identifiable, Equatable {
    var id: String
    var name: String
}

/// Sheet shown when the user clicks "Supprimer" on a library row.
///
/// Two outcomes:
/// - Drop the library row only (legacy default): every artefact
///   stays where it is on disk.
/// - Also wipe the workspace directory: the sheet lists every file
///   we're about to delete + their cumulative size so the user sees
///   exactly the disk economy they're realising. Files are sorted
///   biggest first so the noisy parts (``audio.wav``, the compressed
///   video) jump out immediately.
/// Lists the voice profiles the engine has accumulated from past
/// renames. The user can drop one — useful when a colleague leaves
/// or when an early enrollment got the wrong audio. The list is
/// short on purpose; the engine handles the matching invisibly,
/// the panel only exists for the rare cases where the user wants
/// to override that.
struct SpeakerProfilesSection: View {
    @EnvironmentObject private var library: LibraryStore
    @State private var profiles: [SpeakerProfile] = []
    @State private var loading = false
    @State private var pendingDelete: SpeakerProfile?

    var body: some View {
        Group {
            if loading && profiles.isEmpty {
                HStack(spacing: 8) {
                    ProgressView().controlSize(.small)
                    Text("Chargement…").foregroundStyle(.secondary)
                }
            } else if profiles.isEmpty {
                Text("Aucune voix mémorisée pour l'instant. Renommer un locuteur après une transcription enregistre automatiquement sa voix pour la prochaine réunion.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else {
                ForEach(profiles) { profile in
                    HStack {
                        VStack(alignment: .leading, spacing: 1) {
                            Text(profile.name)
                                .font(.callout.weight(.medium))
                            Text("\(profile.sample_count) extrait(s)")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                        Spacer()
                        Button(role: .destructive) {
                            pendingDelete = profile
                        } label: {
                            Label("Supprimer", systemImage: "trash")
                        }
                        .labelStyle(.iconOnly)
                        .buttonStyle(.borderless)
                        .help("Oublier cette voix")
                    }
                }
            }
        }
        .task {
            await reload()
        }
        .confirmationDialog(
            "Oublier cette voix ?",
            isPresented: Binding(
                get: { pendingDelete != nil },
                set: { if !$0 { pendingDelete = nil } }
            ),
            presenting: pendingDelete
        ) { profile in
            Button("Supprimer", role: .destructive) {
                Task {
                    await library.deleteSpeakerProfile(profile)
                    await reload()
                }
            }
            Button("Annuler", role: .cancel) {}
        } message: { profile in
            Text("La prochaine fois que \(profile.name) parlera dans une réunion, l'app demandera à nouveau confirmation au lieu de pré-remplir le nom.")
        }
    }

    private func reload() async {
        loading = true
        profiles = await library.listSpeakerProfiles()
        loading = false
    }
}

struct LibraryDeletionSheet: View {
    @EnvironmentObject private var library: LibraryStore
    @Environment(\.dismiss) private var dismiss
    let row: LibraryRow

    @State private var usage: WorkspaceUsage?
    @State private var loading = true
    @State private var removeFiles = false
    @State private var working = false

    private static let byteFormatter: ByteCountFormatter = {
        let f = ByteCountFormatter()
        f.countStyle = .file
        f.includesUnit = true
        return f
    }()

    private var fileCount: Int { usage?.files.count ?? 0 }
    private var totalBytes: Int64 { usage?.total_bytes ?? 0 }
    private var hasWorkspace: Bool {
        guard let usage else { return false }
        return !usage.workspace_dir.isEmpty && !usage.files.isEmpty
    }

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Supprimer ce traitement")
                        .font(.title.bold())
                    Text(row.customTitleOrFilename)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                }
                Spacer()
            }
            .padding(22)
            Divider()

            Group {
                if loading {
                    HStack(spacing: 10) {
                        ProgressView()
                            .controlSize(.small)
                        Text("Inspection du dossier de travail…")
                            .foregroundStyle(.secondary)
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else {
                    deletionBody
                }
            }
            .padding(22)

            Divider()
            HStack {
                Button("Annuler") { dismiss() }
                    .keyboardShortcut(.cancelAction)
                Spacer()
                Button(role: .destructive) {
                    confirm()
                } label: {
                    Text(primaryButtonLabel)
                }
                .buttonStyle(.borderedProminent)
                .keyboardShortcut(.defaultAction)
                .disabled(working)
            }
            .padding(18)
        }
        .frame(minWidth: 560, idealWidth: 620, minHeight: 420)
        .task {
            usage = await library.workspaceUsage(row)
            loading = false
        }
    }

    private var primaryButtonLabel: String {
        if removeFiles && hasWorkspace {
            return "Supprimer + libérer \(formatted(totalBytes))"
        }
        return "Supprimer l'entrée"
    }

    @ViewBuilder
    private var deletionBody: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text(
                "L'entrée disparaît de la bibliothèque dans tous les cas. Vous pouvez aussi "
                + "libérer le dossier de travail si vous n'avez plus besoin des fichiers générés."
            )
            .foregroundStyle(.secondary)

            if hasWorkspace {
                Toggle(isOn: $removeFiles) {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Supprimer aussi le dossier de travail")
                            .fontWeight(.medium)
                        Text("\(fileCount) fichier\(fileCount > 1 ? "s" : "") · \(formatted(totalBytes)) à libérer")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
                .toggleStyle(.switch)

                if removeFiles {
                    workspaceFileList
                }
            } else {
                Text("Aucun fichier de travail détecté pour cette entrée (déjà nettoyé ou jamais produit).")
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    @ViewBuilder
    private var workspaceFileList: some View {
        let files = usage?.files ?? []
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("Fichiers concernés")
                    .font(.callout.weight(.medium))
                Spacer()
                if let path = usage?.workspace_dir, !path.isEmpty {
                    Button("Afficher le dossier") {
                        revealInFinder(path)
                    }
                    .controlSize(.small)
                    .buttonStyle(.borderless)
                }
            }
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 4) {
                    ForEach(files) { file in
                        HStack {
                            VStack(alignment: .leading, spacing: 2) {
                                Text(file.name)
                                    .font(.callout)
                                    .lineLimit(1)
                                    .truncationMode(.middle)
                                Text(file.label)
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                            Spacer()
                            Text(formatted(file.size))
                                .font(.callout.monospacedDigit())
                                .foregroundStyle(.secondary)
                        }
                        .padding(.vertical, 3)
                        Divider()
                    }
                }
            }
            .frame(minHeight: 140, maxHeight: 220)
            .background(Color(nsColor: .controlBackgroundColor))
            .clipShape(RoundedRectangle(cornerRadius: 6))
            .overlay {
                RoundedRectangle(cornerRadius: 6)
                    .strokeBorder(Color.secondary.opacity(0.2))
            }
        }
    }

    private func confirm() {
        working = true
        let shouldRemoveFiles = removeFiles && hasWorkspace
        Task {
            await library.delete(row, removeFiles: shouldRemoveFiles)
            await MainActor.run {
                working = false
                dismiss()
            }
        }
    }

    private func formatted(_ bytes: Int64) -> String {
        Self.byteFormatter.string(fromByteCount: bytes)
    }
}

struct ModelActionsView: View {
    @EnvironmentObject private var models: ModelStore
    var row: ModelRow

    var body: some View {
        HStack(spacing: 6) {
            if row.cached {
                Button {
                    Task { await models.delete(row) }
                } label: {
                    Label("Supprimer", systemImage: "trash")
                }
                .labelStyle(.iconOnly)
                .help("Supprimer le modèle local")
            } else {
                Button {
                    Task { await models.download(row) }
                } label: {
                    Label("Télécharger", systemImage: "arrow.down.circle")
                }
                .labelStyle(.iconOnly)
                .help("Pré-télécharger le modèle")
            }
            Button {
                revealInFinder(row.cache_dir)
            } label: {
                Label("Cache", systemImage: "folder")
            }
            .labelStyle(.iconOnly)
            .help("Afficher le dossier de cache")
        }
        .disabled(models.isLoading)
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

// ---------------------------------------------------------------------------
// Odoo connection panel (Réglages → Connexion Odoo)
// ---------------------------------------------------------------------------

/// Form section that lets the user paste their Odoo URL + database
/// + login + API key, then test the connection. Kept ultra-light:
/// four fields, one button, one status line. The user shouldn't
/// have to think about XML-RPC, JSON-RPC, sessions, etc — just
/// "where's your Odoo, what's your key".
struct OdooConnectionSection: View {
    @EnvironmentObject private var settings: SettingsStore
    @EnvironmentObject private var odoo: OdooStore

    var body: some View {
        TextField("URL Odoo", text: $settings.odooUrl, prompt: Text("https://erp.exemple.fr"))
        TextField("Base de données", text: $settings.odooDatabase, prompt: Text("acme_prod"))
        TextField("Email du compte", text: $settings.odooLogin, prompt: Text("vous@exemple.fr"))
        SecureField("Clé API", text: $settings.odooApiKey)
        HStack {
            Button {
                Task { await odoo.testConnection() }
            } label: {
                Label("Tester la connexion", systemImage: "checkmark.shield")
            }
            .disabled(!settings.odooConfigured)
            Spacer()
            statusLabel
        }
        Text("La clé API se génère depuis Odoo : Préférences utilisateur → Sécurité du compte → Nouvelle clé API. Aucune donnée Odoo n'est envoyée à un tiers.")
            .font(.caption)
            .foregroundStyle(.secondary)
    }

    @ViewBuilder
    private var statusLabel: some View {
        switch odoo.status {
        case .unknown:
            EmptyView()
        case .checking:
            HStack(spacing: 6) {
                ProgressView().controlSize(.small)
                Text("Vérification…").font(.caption).foregroundStyle(.secondary)
            }
        case .ok(let account, let count):
            Label(
                "Connecté · \(account) · \(count) contacts",
                systemImage: "checkmark.circle.fill"
            )
            .font(.caption)
            .foregroundStyle(.green)
        case .failed(let message):
            Label(message, systemImage: "exclamationmark.triangle.fill")
                .font(.caption)
                .foregroundStyle(.red)
                .lineLimit(2)
        }
    }
}

// ---------------------------------------------------------------------------
// Speakers view (4th sidebar item)
// ---------------------------------------------------------------------------

/// Top-level view of the local voice profiles, optionally grouped
/// by their linked Odoo company. The grouping kicks in only when
/// the user has linked at least one profile — otherwise we render
/// a flat list so first-time users don't see empty "Sans société"
/// brackets they don't understand.
struct SpeakersView: View {
    @EnvironmentObject private var library: LibraryStore
    @EnvironmentObject private var settings: SettingsStore
    @EnvironmentObject private var odoo: OdooStore
    @State private var profiles: [SpeakerProfile] = []
    @State private var loading = false
    @State private var profileToLink: SpeakerProfile?
    @State private var profileToDelete: SpeakerProfile?

    private var groupedProfiles: [(label: String, profiles: [SpeakerProfile])] {
        // Plain list when nothing's linked yet — avoids fake bucket
        // headers on a brand-new install.
        let anyLinked = profiles.contains(where: \.isLinkedToOdoo)
        if !anyLinked {
            return [("", profiles)]
        }
        let buckets = Dictionary(grouping: profiles, by: \.groupingLabel)
        return buckets
            .map { ($0.key, $0.value.sorted { $0.name.lowercased() < $1.name.lowercased() }) }
            .sorted { lhs, rhs in
                // "Sans société Odoo" goes last so the named
                // companies anchor the top of the list.
                if lhs.0 == "Sans société Odoo" { return false }
                if rhs.0 == "Sans société Odoo" { return true }
                return lhs.0.lowercased() < rhs.0.lowercased()
            }
    }

    var body: some View {
        VStack(spacing: 0) {
            ListHeaderView(
                title: "Interlocuteurs",
                subtitle: settings.odooConfigured
                    ? "Voix mémorisées, regroupées par société Odoo."
                    : "Voix mémorisées localement. Connectez Odoo dans Réglages pour les regrouper par entreprise.",
                actionTitle: "Actualiser",
                actionSystemImage: "arrow.clockwise"
            ) {
                Task { await reload() }
            }
            Divider()
            if loading && profiles.isEmpty {
                ProgressView("Chargement des voix…")
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else if profiles.isEmpty {
                EmptyStateView(
                    title: "Aucune voix mémorisée",
                    systemImage: "person.text.rectangle",
                    message: "Renommez les locuteurs après une transcription pour qu'ils apparaissent ici."
                )
            } else {
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 14) {
                        ForEach(groupedProfiles, id: \.label) { group in
                            VStack(alignment: .leading, spacing: 6) {
                                if !group.label.isEmpty {
                                    Text(group.label)
                                        .font(.headline)
                                        .foregroundStyle(.secondary)
                                        .padding(.horizontal, 4)
                                }
                                VStack(spacing: 4) {
                                    ForEach(group.profiles) { profile in
                                        SpeakerProfileRow(
                                            profile: profile,
                                            canLinkToOdoo: settings.odooConfigured,
                                            onLink: { profileToLink = profile },
                                            onUnlink: {
                                                Task {
                                                    if let updated = await library.unlinkSpeakerFromOdoo(profile) {
                                                        replace(profile, with: updated)
                                                    }
                                                }
                                            },
                                            onDelete: { profileToDelete = profile }
                                        )
                                    }
                                }
                            }
                        }
                    }
                    .padding(20)
                }
            }
        }
        .task { await reload() }
        .sheet(item: $profileToLink) { profile in
            OdooLinkSheet(profile: profile) { updated in
                replace(profile, with: updated)
            }
            .environmentObject(odoo)
            .environmentObject(library)
        }
        .confirmationDialog(
            "Oublier cette voix ?",
            isPresented: Binding(
                get: { profileToDelete != nil },
                set: { if !$0 { profileToDelete = nil } }
            ),
            presenting: profileToDelete
        ) { profile in
            Button("Supprimer", role: .destructive) {
                Task {
                    await library.deleteSpeakerProfile(profile)
                    await reload()
                }
            }
            Button("Annuler", role: .cancel) {}
        } message: { profile in
            Text("La prochaine fois que \(profile.name) parlera dans une réunion, l'app demandera à nouveau confirmation.")
        }
    }

    private func reload() async {
        loading = true
        profiles = await library.listSpeakerProfiles()
        loading = false
    }

    private func replace(_ old: SpeakerProfile, with new: SpeakerProfile) {
        if let idx = profiles.firstIndex(where: { $0.id == old.id }) {
            profiles[idx] = new
        }
    }
}

struct SpeakerProfileRow: View {
    var profile: SpeakerProfile
    var canLinkToOdoo: Bool
    var onLink: () -> Void
    var onUnlink: () -> Void
    var onDelete: () -> Void

    var body: some View {
        HStack(spacing: 10) {
            Image(systemName: profile.isLinkedToOdoo
                  ? "person.crop.circle.badge.checkmark"
                  : "person.crop.circle")
                .foregroundStyle(profile.isLinkedToOdoo ? .teal : .secondary)
                .font(.title3)
            VStack(alignment: .leading, spacing: 1) {
                Text(profile.odoo_partner_name ?? profile.name)
                    .font(.callout.weight(.medium))
                if let odooName = profile.odoo_partner_name,
                   odooName.lowercased() != profile.name.lowercased() {
                    Text("voix mémorisée : \(profile.name)")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Text("\(profile.sample_count) extrait(s)")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            if profile.isLinkedToOdoo {
                Button {
                    onUnlink()
                } label: {
                    Label("Délier", systemImage: "link.badge.minus")
                }
                .help("Délier ce contact Odoo")
            } else {
                Button {
                    onLink()
                } label: {
                    Label("Lier à Odoo", systemImage: "link")
                }
                .disabled(!canLinkToOdoo)
                .help(canLinkToOdoo
                      ? "Associer un contact Odoo à cette voix"
                      : "Configurez d'abord Odoo dans Réglages")
            }
            Button(role: .destructive) {
                onDelete()
            } label: {
                Label("Supprimer", systemImage: "trash")
            }
            .labelStyle(.iconOnly)
            .buttonStyle(.borderless)
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 6))
    }
}

/// Spotlight-style live search for an Odoo res.partner. The whole
/// linker UX collapses into a TextField + a results list — no
/// filters, no advanced options. The user types, we debounce
/// 250 ms, the engine returns matches, click = link.
struct OdooLinkSheet: View {
    @EnvironmentObject private var odoo: OdooStore
    @EnvironmentObject private var library: LibraryStore
    @Environment(\.dismiss) private var dismiss
    let profile: SpeakerProfile
    var onLinked: (SpeakerProfile) -> Void

    @State private var query: String = ""
    @State private var results: [OdooPartner] = []
    @State private var searching = false
    @State private var debounceTask: Task<Void, Never>?

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                VStack(alignment: .leading, spacing: 3) {
                    Text("Lier « \(profile.name) » à un contact Odoo")
                        .font(.title2.bold())
                    Text("Tapez le nom ou l'email d'un contact pour l'associer à cette voix.")
                        .foregroundStyle(.secondary)
                }
                Spacer()
            }
            .padding(20)
            Divider()

            HStack(spacing: 8) {
                Image(systemName: "magnifyingglass")
                    .foregroundStyle(.secondary)
                TextField("Rechercher dans les contacts Odoo…", text: $query)
                    .textFieldStyle(.plain)
                    .onChange(of: query) { _, newValue in
                        scheduleSearch(newValue)
                    }
                if searching {
                    ProgressView().controlSize(.small)
                }
            }
            .padding(12)
            .background(Color(nsColor: .controlBackgroundColor))

            Divider()

            if results.isEmpty {
                VStack(spacing: 6) {
                    Image(systemName: "person.crop.circle.badge.questionmark")
                        .font(.system(size: 36, weight: .light))
                        .foregroundStyle(.secondary)
                    Text(query.trimmingCharacters(in: .whitespaces).isEmpty
                         ? "Commencez à taper pour rechercher"
                         : "Aucun contact ne correspond")
                        .foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                List(results) { partner in
                    Button {
                        link(partner)
                    } label: {
                        HStack(spacing: 10) {
                            Image(systemName: partner.is_company
                                  ? "building.2"
                                  : "person")
                                .foregroundStyle(partner.is_company ? .teal : .primary)
                            VStack(alignment: .leading, spacing: 1) {
                                Text(partner.display_name)
                                    .font(.callout.weight(.medium))
                                HStack(spacing: 6) {
                                    if !partner.parent_name.isEmpty {
                                        Text(partner.parent_name)
                                            .font(.caption)
                                            .foregroundStyle(.secondary)
                                    }
                                    if !partner.function.isEmpty {
                                        Text("·")
                                            .font(.caption)
                                            .foregroundStyle(.secondary)
                                        Text(partner.function)
                                            .font(.caption)
                                            .foregroundStyle(.secondary)
                                    }
                                }
                                if !partner.email.isEmpty {
                                    Text(partner.email)
                                        .font(.caption)
                                        .foregroundStyle(.tertiary)
                                }
                            }
                            Spacer()
                            Image(systemName: "link")
                                .foregroundStyle(.secondary)
                        }
                        .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                }
                .listStyle(.inset)
            }

            Divider()
            HStack {
                Button("Annuler") { dismiss() }
                    .keyboardShortcut(.cancelAction)
                Spacer()
            }
            .padding(14)
        }
        .frame(minWidth: 620, minHeight: 460)
    }

    private func scheduleSearch(_ value: String) {
        debounceTask?.cancel()
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty {
            results = []
            return
        }
        debounceTask = Task {
            // Debounce so we don't spam Odoo for every keystroke.
            // 250 ms is short enough to feel instant, long enough
            // to skip mid-word noise.
            try? await Task.sleep(nanoseconds: 250_000_000)
            if Task.isCancelled { return }
            await MainActor.run { searching = true }
            let found = await odoo.searchPartners(trimmed)
            if Task.isCancelled { return }
            await MainActor.run {
                results = found
                searching = false
            }
        }
    }

    private func link(_ partner: OdooPartner) {
        // Top-level companies have no parent; we surface the
        // partner itself as the "company" so the grouping shelf
        // shows the company name rather than dumping it under
        // "Sans société".
        let companyId = partner.is_company ? partner.id : (partner.parent_id > 0 ? partner.parent_id : nil)
        let companyName = partner.is_company ? partner.name : partner.parent_name
        Task {
            if let updated = await library.linkSpeakerToOdoo(
                profile,
                partnerId: partner.id,
                partnerName: partner.display_name,
                companyId: companyId,
                companyName: companyName
            ) {
                onLinked(updated)
                await MainActor.run { dismiss() }
            }
        }
    }
}

func openHuggingFaceTokens() {
    guard let url = URL(string: "https://huggingface.co/settings/tokens") else { return }
    NSWorkspace.shared.open(url)
}

/// Builds the title shown next to the speaker-count Stepper.
/// Embedded in the title (rather than a sibling label View) so the
/// Form's column layout stays intact — wrapping a Spacer inside a
/// Stepper label collapses every other field on macOS.
func speakerCountStepperLabel(_ value: Int) -> String {
    let suffix = value == 0 ? "Auto" : "\(value)"
    return "Nombre d'intervenants attendu : \(suffix)"
}

func localizedStatus(_ raw: String?) -> String {
    switch (raw ?? "").uppercased() {
    case "COMPLETED": return "Terminé"
    case "RUNNING": return "En cours"
    case "FAILED": return "Échec"
    case "PENDING": return "En attente"
    default: return raw ?? "-"
    }
}

func statusIconName(_ raw: String?) -> String {
    switch (raw ?? "").uppercased() {
    case "COMPLETED": return "checkmark.circle.fill"
    case "RUNNING": return "clock.arrow.circlepath"
    case "FAILED": return "xmark.octagon.fill"
    default: return "circle"
    }
}

func statusColor(_ raw: String?) -> Color {
    switch (raw ?? "").uppercased() {
    case "COMPLETED": return .green
    case "RUNNING": return .teal
    case "FAILED": return .red
    default: return .secondary
    }
}

func fileSizeLabel(_ path: String) -> String {
    guard let attrs = try? FileManager.default.attributesOfItem(atPath: path),
          let size = attrs[.size] as? NSNumber else {
        return ""
    }
    return ByteCountFormatter.string(fromByteCount: size.int64Value, countStyle: .file)
}

func pathExists(_ path: String?) -> Bool {
    guard let path, !path.isEmpty else { return false }
    return FileManager.default.fileExists(atPath: path)
}
