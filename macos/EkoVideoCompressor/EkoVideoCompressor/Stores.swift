import Foundation
import SwiftUI

func sourceMeetingDate(for url: URL) -> Date {
    (try? url.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate)
        ?? Date()
}

func engineMeetingDateString(_ date: Date) -> String {
    let formatter = ISO8601DateFormatter()
    formatter.formatOptions = [.withInternetDateTime]
    return formatter.string(from: date)
}

func parseEngineMeetingDate(_ value: String?) -> Date? {
    guard let value, !value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
        return nil
    }
    let formatter = ISO8601DateFormatter()
    formatter.formatOptions = [.withInternetDateTime]
    if let date = formatter.date(from: value) {
        return date
    }
    formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
    return formatter.date(from: value)
}

func displayMeetingDate(_ value: String?) -> String {
    guard let date = parseEngineMeetingDate(value) else { return "—" }
    let formatter = DateFormatter()
    formatter.locale = Locale.autoupdatingCurrent
    formatter.dateStyle = .medium
    formatter.timeStyle = .short
    return formatter.string(from: date)
}

struct QueueItem: Identifiable, Equatable {
    let id = UUID()
    var sourceURL: URL
    var focusNote: String?
    /// Existing library row this queue item is reprocessing. Nil for a
    /// brand-new drop. When set, the engine updates that row instead of
    /// creating a second execution entry.
    var libraryJobId: Int?
    /// Existing job workspace reused for reruns. Empty on fresh drops;
    /// populated when the user relaunches from the library.
    var workspaceDir: String = ""
    /// Per-file vocabulary explicitly selected for this run. The global
    /// vocabulary catalog suggests entries, but nothing is sent to
    /// Whisper unless the user chose it here.
    var selectedGlossaryTerms: [String] = []
    /// User trim hints, in seconds. ``trimEndSeconds`` means "remove
    /// this much from the tail"; the final absolute ffmpeg `-to` value
    /// is computed from ``mediaDurationSeconds`` when launching.
    var trimStartSeconds: Double = 0
    var trimEndSeconds: Double = 0
    var mediaDurationSeconds: Double = 0
    /// Actual meeting date used for Odoo matching and artefact
    /// metadata. Nil means "use source file metadata"; the manual
    /// flag drives the Run Setup helper text.
    var meetingDate: Date?
    var meetingDateManuallyEdited: Bool = false
    /// Number of speakers the user expects on this specific
    /// recording. 0 means "let pyannote estimate". Per-file rather
    /// than per-batch because a 5-person standup followed by a
    /// 1-on-1 in the same queue need different bounds.
    var expectedSpeakerCount: Int = 0
    /// Names the user wants Whisper to be biased toward (typically
    /// pulled from an Odoo calendar.event the user picked in Run
    /// Setup). They land in ``JobRequest.speaker_overrides`` keyed
    /// on themselves so the engine's initial-prompt builder sees
    /// them as "expected participants" without forcing a SPEAKER_NN
    /// assignment.
    var expectedSpeakerNames: [String] = []
    /// Title of the Odoo meeting the user paired with this file —
    /// surfaced as the meeting_context line of the Whisper initial
    /// prompt. Optional: empty when no meeting was attached.
    var odooMeetingTitle: String = ""
    /// Full picked meeting, including its attendees + related CRM
    /// object. The runner persists this on the job row so the
    /// rename sheet can later show one-click attribution chips for
    /// each invitee. ``nil`` when no meeting was attached.
    var odooMeeting: OdooMeetingMetadata?
    /// Pointer to the related object whose chatter the pipeline
    /// fetches during the LLM step. ``nil`` when there's no
    /// related object (the meeting wasn't linked to a CRM lead or
    /// project task).
    var odooContextRef: OdooContextRef?
    var status: String = "En attente"
    var progress: Double = 0

    var isLibraryRerun: Bool {
        libraryJobId != nil || !workspaceDir.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }
}

@MainActor
final class QueueStore: ObservableObject {
    @Published var items: [QueueItem] = []
    @Published var isBatchRunning = false
    @Published var autoRunRequestID: UUID?

    func add(
        urls: [URL],
        focusNote: String? = nil,
        prioritize: Bool = false,
        libraryJobId: Int? = nil,
        workspaceDir: String = "",
        expectedSpeakerNames: [String] = [],
        selectedGlossaryTerms: [String] = [],
        odooMeetingTitle: String = "",
        odooMeeting: OdooMeetingMetadata? = nil,
        odooContextRef: OdooContextRef? = nil,
        meetingDate: Date? = nil,
        meetingDateManuallyEdited: Bool = false
    ) {
        for url in urls {
            if let index = items.firstIndex(where: { $0.sourceURL == url && $0.libraryJobId == libraryJobId }) {
                if let focusNote {
                    items[index].focusNote = focusNote
                }
                if libraryJobId != nil {
                    items[index].libraryJobId = libraryJobId
                    items[index].workspaceDir = workspaceDir
                }
                if !expectedSpeakerNames.isEmpty {
                    items[index].expectedSpeakerNames = expectedSpeakerNames
                    items[index].expectedSpeakerCount = expectedSpeakerNames.count
                }
                if !selectedGlossaryTerms.isEmpty {
                    items[index].selectedGlossaryTerms = selectedGlossaryTerms
                }
                if !odooMeetingTitle.isEmpty {
                    items[index].odooMeetingTitle = odooMeetingTitle
                }
                if let odooMeeting {
                    items[index].odooMeeting = odooMeeting
                }
                if let odooContextRef {
                    items[index].odooContextRef = odooContextRef
                }
                if let meetingDate {
                    items[index].meetingDate = meetingDate
                    items[index].meetingDateManuallyEdited = meetingDateManuallyEdited
                }
                if prioritize && index > 0 {
                    let item = items.remove(at: index)
                    items.insert(item, at: 0)
                }
                continue
            }
            var item = QueueItem(sourceURL: url, focusNote: focusNote)
            item.libraryJobId = libraryJobId
            item.workspaceDir = workspaceDir
            item.expectedSpeakerNames = expectedSpeakerNames
            item.expectedSpeakerCount = expectedSpeakerNames.count
            item.selectedGlossaryTerms = selectedGlossaryTerms
            item.odooMeetingTitle = odooMeetingTitle
            item.odooMeeting = odooMeeting
            item.odooContextRef = odooContextRef
            item.meetingDate = meetingDate
            item.meetingDateManuallyEdited = meetingDateManuallyEdited
            if prioritize {
                items.insert(item, at: 0)
            } else {
                items.append(item)
            }
        }
    }

    func addRerun(
        row: LibraryRow,
        sourcePath: String? = nil,
        focusNote: String? = nil,
        prioritize: Bool = false
    ) {
        let path = sourcePath
            ?? (pathExists(row.copiedSourcePath) ? row.copiedSourcePath : row.source_path)
        guard let path, !path.isEmpty else { return }
        let meeting = row.odooMeeting
        let attendeeNames = meeting?.attendees.map(\.name).filter { !$0.isEmpty } ?? []
        let existingTerms = row.technicalTerms
        let existingMeetingDate = parseEngineMeetingDate(row.meeting_date)
        add(
            urls: [URL(fileURLWithPath: path)],
            focusNote: focusNote,
            prioritize: prioritize,
            libraryJobId: row.id,
            workspaceDir: row.workspace_dir ?? "",
            expectedSpeakerNames: attendeeNames.isEmpty ? row.displayedSpeakerNames : attendeeNames,
            selectedGlossaryTerms: existingTerms,
            odooMeetingTitle: meeting?.event_name ?? "",
            odooMeeting: meeting,
            odooContextRef: meeting?.related.map {
                OdooContextRef(model: $0.model, record_id: $0.id, url: "", database: "", login: "", api_key: "")
            },
            meetingDate: existingMeetingDate,
            meetingDateManuallyEdited: existingMeetingDate != nil
        )
    }

    func requestAutoRun() {
        autoRunRequestID = UUID()
    }

    func move(from source: IndexSet, to destination: Int) {
        items.move(fromOffsets: source, toOffset: destination)
    }

    func remove(at offsets: IndexSet) {
        items.remove(atOffsets: offsets)
    }

    func remove(id: QueueItem.ID) {
        items.removeAll { $0.id == id }
    }

    func update(_ id: QueueItem.ID, status: String, progress: Double? = nil) {
        guard let index = items.firstIndex(where: { $0.id == id }) else { return }
        items[index].status = status
        if let progress {
            items[index].progress = progress
        }
    }

    /// Used by the source-relocalisation recovery flow: the user
    /// pointed us at a new file location, so the queue row updates
    /// in place (status reset, new URL) without losing its slot
    /// in the batch — the runner can retry immediately.
    func replace(_ id: QueueItem.ID, with newURL: URL) {
        guard let index = items.firstIndex(where: { $0.id == id }) else { return }
        items[index].sourceURL = newURL
        items[index].status = "En attente"
        items[index].progress = 0
    }

    func resetPending() {
        for index in items.indices {
            items[index].status = "En attente"
            items[index].progress = 0
        }
    }
}

@MainActor
final class SettingsStore: ObservableObject {
    @AppStorage("outputDir") var outputDir = "\(NSHomeDirectory())/EkoVideo Compressor"
    @AppStorage("glossary") var glossary = ""
    @AppStorage("glossaryUsageJSON") private var glossaryUsageJSON = "{}"
    @AppStorage("hfToken") var hfToken = ""
    @AppStorage("githubToken") var githubToken = ""
    @AppStorage("whisperModel") var whisperModel = "mlx-community/whisper-large-v3-turbo"
    // Active models for the other roles the Models tab now lists.
    // Defaults match the catalog's ``"default": True`` entries so a
    // first-run user still gets the recommended setup without
    // having to click "Activer" once.
    @AppStorage("multipassModel") var multipassModel = "mlx-community/whisper-large-v3-mlx"
    @AppStorage("textLlmModel") var textLlmModel = "mlx-community/Mistral-7B-Instruct-v0.3-4bit"
    @AppStorage("audioLlmModel") var audioLlmModel = "mlx-community/Qwen2-Audio-7B-Instruct-4bit"
    @AppStorage("processingMode") var processingMode = "compress_transcribe"
    @AppStorage("outputFormat") var outputFormat = "txt"
    @AppStorage("audioRecheckEnabled") var audioRecheckEnabled = false
    @AppStorage("diarizationEnabled") var diarizationEnabled = false
    @AppStorage("deleteSourceAfterCopy") var deleteSourceAfterCopy = false
    /// Single user-facing quality knob. Replaces the previous handful
    /// of toggles (VAD / multipass / per-speaker / web). The engine
    /// derives the real flags from the preset string at job time.
    @AppStorage("qualityPreset") var qualityPreset = TranscriptionQualityPreset.balanced.rawValue

    // Odoo connection. Stored alongside the other tokens via
    // @AppStorage for cohesion — moving everything to the macOS
    // Keychain is a separate, broader hardening pass. The API key
    // here is the API key a user creates under Account Security →
    // New API Key in Odoo 19+.
    @AppStorage("odooUrl") var odooUrl = ""
    @AppStorage("odooDatabase") var odooDatabase = ""
    @AppStorage("odooLogin") var odooLogin = ""
    @AppStorage("odooApiKey") var odooApiKey = ""

    var odooConfigured: Bool {
        !odooUrl.trimmingCharacters(in: .whitespaces).isEmpty
        && !odooDatabase.trimmingCharacters(in: .whitespaces).isEmpty
        && !odooLogin.trimmingCharacters(in: .whitespaces).isEmpty
        && !odooApiKey.trimmingCharacters(in: .whitespaces).isEmpty
    }

    var glossaryTerms: [String] {
        glossary
            .split(whereSeparator: \.isNewline)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
    }

    var vocabularyUsage: [String: Int] {
        guard let data = glossaryUsageJSON.data(using: .utf8),
              let decoded = try? JSONDecoder().decode([String: Int].self, from: data) else {
            return [:]
        }
        return decoded
    }

    var vocabularyCatalog: [String] {
        let usage = vocabularyUsage
        let terms = Set(glossaryTerms).union(usage.keys)
        return terms.sorted { left, right in
            let leftUsage = usage[left] ?? 0
            let rightUsage = usage[right] ?? 0
            if leftUsage != rightUsage {
                return leftUsage > rightUsage
            }
            return left.localizedCaseInsensitiveCompare(right) == .orderedAscending
        }
    }

    func suggestedVocabulary(matching query: String, excluding selected: [String]) -> [String] {
        let needle = query.trimmingCharacters(in: .whitespacesAndNewlines)
        let excluded = Set(selected.map { $0.lowercased() })
        return vocabularyCatalog.filter { term in
            guard !excluded.contains(term.lowercased()) else { return false }
            return needle.isEmpty || term.localizedCaseInsensitiveContains(needle)
        }
    }

    func addVocabularyTerm(_ value: String) {
        let term = value.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !term.isEmpty else { return }
        var terms = glossaryTerms
        guard !terms.contains(where: { $0.caseInsensitiveCompare(term) == .orderedSame }) else { return }
        terms.append(term)
        glossary = terms.joined(separator: "\n")
    }

    func removeVocabularyTerm(_ value: String) {
        let term = value.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !term.isEmpty else { return }
        let terms = glossaryTerms.filter { $0.caseInsensitiveCompare(term) != .orderedSame }
        glossary = terms.joined(separator: "\n")
        var usage = vocabularyUsage
        usage.removeValue(forKey: term)
        writeVocabularyUsage(usage)
    }

    func recordVocabularyUsage(_ terms: [String]) {
        var usage = vocabularyUsage
        for raw in terms {
            let term = raw.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !term.isEmpty else { continue }
            usage[term, default: 0] += 1
            addVocabularyTerm(term)
        }
        writeVocabularyUsage(usage)
    }

    private func writeVocabularyUsage(_ usage: [String: Int]) {
        guard let data = try? JSONEncoder().encode(usage),
              let text = String(data: data, encoding: .utf8) else { return }
        glossaryUsageJSON = text
    }
}

struct ReleaseInfo: Codable, Equatable {
    var tag_name: String
    var name: String
    var html_url: String
    var body: String
    var asset_name: String
    var asset_url: String
}

enum UpdateState: Equatable {
    case idle
    case checking
    case available(ReleaseInfo)
    case upToDate
    case downloading(Double)
    case readyToInstall(URL, ReleaseInfo)
    case error(String)
}

@MainActor
final class UpdateStore: ObservableObject {
    @Published var state: UpdateState = .idle
    private var settings: SettingsStore?

    func setSettings(_ settings: SettingsStore) {
        self.settings = settings
    }

    var currentVersion: String {
        Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "0.0.0"
    }

    func checkUpdates(proactive: Bool = false) async {
        if state == .checking || (caseUpdateStateAvailable(state) && proactive) { return }
        state = .checking

        let url = URL(string: "https://api.github.com/repos/robjo82/EkoVideoCompressor/releases/latest")!
        var request = URLRequest(url: url)
        request.setValue("application/vnd.github+json", forHTTPHeaderField: "Accept")
        request.setValue("EkoVideoCompressor/\(currentVersion)", forHTTPHeaderField: "User-Agent")
        if let token = settings?.githubToken, !token.isEmpty {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }

        do {
            let (data, response) = try await URLSession.shared.data(for: request)
            guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 else {
                if proactive { state = .idle; return }
                state = .error("Erreur HTTP: \((response as? HTTPURLResponse)?.statusCode ?? -1)")
                return
            }

            let payload = try JSONDecoder().decode(GitHubRelease.self, from: data)
            guard let chosenAsset = chooseAsset(payload.assets) else {
                if proactive { state = .idle; return }
                state = .error("Aucun asset compatible trouvé.")
                return
            }

            let info = ReleaseInfo(
                tag_name: payload.tag_name,
                name: payload.name,
                html_url: payload.html_url,
                body: payload.body,
                asset_name: chosenAsset.name,
                asset_url: chosenAsset.browser_download_url
            )

            if isNewer(payload.tag_name) {
                state = .available(info)
            } else {
                state = proactive ? .idle : .upToDate
            }
        } catch {
            if proactive { state = .idle; return }
            state = .error(error.localizedDescription)
        }
    }

    func downloadAndPrepare(info: ReleaseInfo) async {
        state = .downloading(0)
        
        do {
            let url = URL(string: info.asset_url)!
            var request = URLRequest(url: url)
            request.setValue("EkoVideoCompressor/\(currentVersion)", forHTTPHeaderField: "User-Agent")
            if let token = settings?.githubToken, !token.isEmpty {
                request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
            }

            let (localURL, response) = try await URLSession.shared.download(for: request)
            guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 else {
                state = .error("Erreur téléchargement: \((response as? HTTPURLResponse)?.statusCode ?? -1)")
                return
            }

            // Move to a more permanent temporary location
            let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
            try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
            let zipPath = tempDir.appendingPathComponent(info.asset_name)
            try FileManager.default.moveItem(at: localURL, to: zipPath)

            state = .readyToInstall(zipPath, info)
        } catch {
            state = .error(error.localizedDescription)
        }
    }

    func applyUpdate(zipURL: URL) {
        let tmpDir = zipURL.deletingLastPathComponent()
        let pid = ProcessInfo.processInfo.processIdentifier
        let targetAppPath = Bundle.main.bundlePath
        let fallbackApp = "\(NSHomeDirectory())/Applications/EkoVideoCompressor.app"
        let logDir = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first?.appendingPathComponent("EkoVideo Compressor")
        try? FileManager.default.createDirectory(at: logDir!, withIntermediateDirectories: true)
        let logPath = logDir!.appendingPathComponent("updater.log").path

        let scriptPath = tmpDir.appendingPathComponent("apply_update.sh")
        
        let scriptContent = """
#!/bin/bash
# EkoVideo Update Script
LOG_PATH=\(quote(logPath))
exec >> "$LOG_PATH" 2>&1

echo "=== EkoVideo Swift Updater $(date) ==="
PID=\(pid)
NEW_APP_ZIP=\(quote(zipURL.path))
TMP_DIR=\(quote(tmpDir.path))
TARGET_APP=\(quote(targetAppPath))
FALLBACK_APP=\(quote(fallbackApp))

echo "PID to wait for: $PID"
echo "Zip: $NEW_APP_ZIP"
echo "Target: $TARGET_APP"

# Wait for the app to exit
echo "Waiting for app (PID $PID) to exit..."
while kill -0 "$PID" 2>/dev/null; do
    sleep 0.5
done

echo "Extracting update..."
/usr/bin/ditto -x -k --noqtn "$NEW_APP_ZIP" "$TMP_DIR"
NEW_APP=$(find "$TMP_DIR" -name "EkoVideoCompressor.app" -type d | head -n 1)

if [ -z "$NEW_APP" ]; then
    echo "Error: EkoVideoCompressor.app not found in zip"
    exit 1
fi

echo "Installing to $TARGET_APP..."
if [ -w "$TARGET_APP" ] || [ ! -e "$TARGET_APP" ]; then
    rm -rf "$TARGET_APP"
    cp -R "$NEW_APP" "$TARGET_APP"
    INSTALL_PATH="$TARGET_APP"
elif [ -w "$(dirname "$FALLBACK_APP")" ]; then
    echo "Target app not writable, trying fallback: $FALLBACK_APP"
    rm -rf "$FALLBACK_APP"
    mkdir -p "$(dirname "$FALLBACK_APP")"
    cp -R "$NEW_APP" "$FALLBACK_APP"
    INSTALL_PATH="$FALLBACK_APP"
else
    echo "Error: No writable install path found."
    exit 1
fi

echo "Restarting app from $INSTALL_PATH..."
open "$INSTALL_PATH"
echo "Update complete."
exit 0
"""
        do {
            try scriptContent.write(to: scriptPath, atomically: true, encoding: .utf8)
            try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: scriptPath.path)
            
            let process = Process()
            process.executableURL = URL(fileURLWithPath: "/bin/bash")
            process.arguments = [scriptPath.path]
            
            // Start the process in a new session to ensure it survives app exit
            // On macOS, we can use 'nohup' or just trust that a backgrounded bash survives.
            // But better: use a separate process group if possible.
            // Simplified: launch via /usr/bin/nohup
            process.executableURL = URL(fileURLWithPath: "/usr/bin/nohup")
            process.arguments = ["/bin/bash", scriptPath.path]
            
            try process.run()
            Foundation.exit(0)
        } catch {
            state = .error("Échec du lancement du script de mise à jour: \\(error.localizedDescription)")
        }
    }

    private func quote(_ s: String) -> String {
        "'" + s.replacingOccurrences(of: "'", with: "'\\\\''") + "'"
    }

    private func isNewer(_ remoteTag: String) -> Bool {
        let remote = parseSemver(remoteTag)
        let local = parseSemver(currentVersion)
        if remote.0 > local.0 { return true }
        if remote.0 == local.0 && remote.1 > local.1 { return true }
        if remote.0 == local.0 && remote.1 == local.1 && remote.2 > local.2 { return true }
        return false
    }

    private func parseSemver(_ v: String) -> (Int, Int, Int) {
        let parts = v.trimmingCharacters(in: CharacterSet.decimalDigits.inverted)
            .split(separator: ".")
            .compactMap { Int($0) }
        return (parts.count > 0 ? parts[0] : 0,
                parts.count > 1 ? parts[1] : 0,
                parts.count > 2 ? parts[2] : 0)
    }

    private struct GitHubRelease: Codable {
        var tag_name: String
        var name: String
        var html_url: String
        var body: String
        var assets: [GitHubAsset]
    }

    private struct GitHubAsset: Codable {
        var name: String
        var browser_download_url: String
    }

    private func chooseAsset(_ assets: [GitHubAsset]) -> GitHubAsset? {
        let zips = assets.filter { $0.name.hasSuffix(".zip") }
        #if arch(arm64)
        return zips.first { $0.name.lowercased().contains("macos-arm64") } ?? zips.first
        #else
        return zips.first { $0.name.lowercased().contains("macos-x64") } ?? zips.first
        #endif
    }

    private func caseUpdateStateAvailable(_ state: UpdateState) -> Bool {
        if case .available = state { return true }
        return false
    }
}

@MainActor
final class LibraryStore: ObservableObject {
    @Published var rows: [LibraryRow] = []
    @Published var isLoading = false
    @Published var errorMessage: String?
    @Published var speakerProfiles: [SpeakerProfile] = []
    @Published var speakerProfilesLoading = false
    private var speakerProfilesLoaded = false

    func refresh() async {
        isLoading = true
        errorMessage = nil
        let result = await EngineProcess.runCommand(
            arguments: EngineProcess.defaultPythonArguments(["library-list", "--jsonl"])
        )
        if result.status != 0 {
            errorMessage = result.rawOutput
        }
        rows = result.lines.compactMap { line in
            try? JSONDecoder().decode(LibraryRow.self, from: Data(line.utf8))
        }
        isLoading = false
    }

    func delete(_ row: LibraryRow, removeFiles: Bool = false) async {
        let previousRows = rows
        rows.removeAll { $0.id == row.id }
        isLoading = true
        errorMessage = nil
        var args = ["library-delete", "\(row.id)"]
        if removeFiles {
            args.append("--remove-files")
        }
        let result = await EngineProcess.runCommand(
            arguments: EngineProcess.defaultPythonArguments(args)
        )
        if result.status != 0 {
            errorMessage = result.events.last?.message ?? result.rawOutput
            rows = previousRows
            isLoading = false
            return
        }
        isLoading = false
    }

    /// Preview what would be freed if the workspace got deleted.
    /// Used by the deletion sheet to show the file list + total size
    /// before the user confirms. Returns ``nil`` on engine error so
    /// the caller can fall back to a simpler "just drop the row"
    /// dialog instead of blocking on a stale path.
    func workspaceUsage(_ row: LibraryRow) async -> WorkspaceUsage? {
        let result = await EngineProcess.runCommand(
            arguments: EngineProcess.defaultPythonArguments([
                "library-workspace-usage",
                "\(row.id)",
            ])
        )
        if result.status != 0 {
            errorMessage = result.events.last?.message ?? result.rawOutput
            return nil
        }
        guard let data = result.rawOutput.data(using: .utf8) else { return nil }
        return try? JSONDecoder().decode(WorkspaceUsage.self, from: data)
    }

    func renameSpeakers(_ row: LibraryRow, mapping: [String: String]) async {
        guard let payload = jsonString(mapping) else {
            errorMessage = "Mapping interlocuteurs invalide."
            return
        }
        isLoading = true
        errorMessage = nil
        let result = await EngineProcess.runCommand(
            arguments: EngineProcess.defaultPythonArguments([
                "library-rename-speakers",
                "\(row.id)",
                "--mapping",
                payload,
            ])
        )
        if result.status != 0 {
            errorMessage = result.events.last?.message ?? result.rawOutput
            isLoading = false
            return
        }
        await refresh()
        await refreshSpeakerProfiles(force: true)
    }

    func updateContext(_ row: LibraryRow, speakers: [String: String], technicalTerms: [String]) async {
        guard let speakersPayload = jsonString(speakers),
              let termsPayload = jsonString(technicalTerms) else {
            errorMessage = "Contexte invalide."
            return
        }
        isLoading = true
        errorMessage = nil
        let result = await EngineProcess.runCommand(
            arguments: EngineProcess.defaultPythonArguments([
                "library-update-context",
                "\(row.id)",
                "--speakers",
                speakersPayload,
                "--technical-terms",
                termsPayload,
            ])
        )
        if result.status != 0 {
            errorMessage = result.events.last?.message ?? result.rawOutput
            isLoading = false
            return
        }
        await refresh()
    }

    func speakerSamples(_ row: LibraryRow) async -> [SpeakerSample] {
        let result = await EngineProcess.runCommand(
            arguments: EngineProcess.defaultPythonArguments([
                "library-speaker-samples",
                "\(row.id)",
                "--per-speaker",
                "4",
                "--jsonl",
            ])
        )
        if result.status != 0 {
            errorMessage = result.events.last?.message ?? result.rawOutput
            return []
        }
        return result.lines.compactMap { line in
            try? JSONDecoder().decode(SpeakerSample.self, from: Data(line.utf8))
        }
    }

    @discardableResult
    func flagSpeakerSampleForReview(_ row: LibraryRow, sample: SpeakerSample) async -> Bool {
        let result = await EngineProcess.runCommand(
            arguments: EngineProcess.defaultPythonArguments([
                "library-speaker-sample-review",
                "\(row.id)",
                "--speaker",
                sample.speaker,
                "--start",
                String(format: "%.3f", sample.start),
                "--duration",
                String(format: "%.3f", sample.duration),
                "--note",
                "Extrait signalé depuis l'éditeur d'interlocuteurs : plusieurs voix possibles.",
            ])
        )
        if result.status != 0 {
            errorMessage = result.events.last?.message ?? result.rawOutput
            return false
        }
        return true
    }

    /// Backfill the speaker map for jobs that completed before the
    /// pipeline started persisting it. The engine walks the artefact
    /// files on disk, extracts every bracketed prefix
    /// (``[SPEAKER_00]``, ``[Robin]``, etc.), and writes the merged
    /// list to ``speaker_map_json``. Returns the resulting map
    /// without touching ``rows`` — callers compose it with whatever
    /// they already have.
    func discoverSpeakers(_ row: LibraryRow) async -> [String: String] {
        let result = await EngineProcess.runCommand(
            arguments: EngineProcess.defaultPythonArguments([
                "library-discover-speakers",
                "\(row.id)",
            ])
        )
        if result.status != 0 {
            errorMessage = result.events.last?.message ?? result.rawOutput
            return [:]
        }
        guard let data = result.rawOutput.data(using: .utf8) else { return [:] }
        struct Payload: Decodable { var speakers: [String: String] }
        if let payload = try? JSONDecoder().decode(Payload.self, from: data) {
            return payload.speakers
        }
        return [:]
    }

    private func jsonString<T: Encodable>(_ value: T) -> String? {
        guard let data = try? JSONEncoder().encode(value) else { return nil }
        return String(data: data, encoding: .utf8)
    }

    /// Pulled out of the rename hot-path because ``LibraryStore``
    /// is the right home for read/write to the engine's library DB,
    /// and the speaker profiles surface (Settings) needs both list
    /// and delete without touching the rest of the library state.

    private func fetchSpeakerProfiles() async -> [SpeakerProfile]? {
        let result = await EngineProcess.runCommand(
            arguments: EngineProcess.defaultPythonArguments([
                "library-list-speaker-profiles",
                "--jsonl",
            ])
        )
        if result.status != 0 {
            errorMessage = result.events.last?.message ?? result.rawOutput
            return nil
        }
        return result.lines.compactMap { line in
            try? JSONDecoder().decode(SpeakerProfile.self, from: Data(line.utf8))
        }
    }

    func refreshSpeakerProfiles(force: Bool = false) async {
        if speakerProfilesLoaded && !force {
            return
        }
        speakerProfilesLoading = true
        defer { speakerProfilesLoading = false }
        guard let profiles = await fetchSpeakerProfiles() else { return }
        speakerProfiles = profiles
        speakerProfilesLoaded = true
    }

    func listSpeakerProfiles(force: Bool = false) async -> [SpeakerProfile] {
        if speakerProfilesLoaded && !force {
            return speakerProfiles
        }
        await refreshSpeakerProfiles(force: true)
        return speakerProfiles
    }

    func replaceSpeakerProfile(_ profile: SpeakerProfile) {
        if let index = speakerProfiles.firstIndex(where: { $0.id == profile.id }) {
            speakerProfiles[index] = profile
        } else {
            speakerProfiles.append(profile)
        }
        sortSpeakerProfiles()
        speakerProfilesLoaded = true
    }

    func removeSpeakerProfileLocally(_ profile: SpeakerProfile) {
        speakerProfiles.removeAll { $0.id == profile.id }
        speakerProfilesLoaded = true
    }

    func restoreSpeakerProfiles(_ snapshot: [SpeakerProfile]) {
        speakerProfiles = snapshot
        speakerProfilesLoaded = true
    }

    private func sortSpeakerProfiles() {
        speakerProfiles.sort { $0.name.localizedCaseInsensitiveCompare($1.name) == .orderedAscending }
    }

    @discardableResult
    func deleteSpeakerProfile(_ profile: SpeakerProfile) async -> Bool {
        let result = await EngineProcess.runCommand(
            arguments: EngineProcess.defaultPythonArguments([
                "library-delete-speaker-profile",
                "--id",
                "\(profile.id)",
            ])
        )
        if result.status != 0 {
            errorMessage = result.events.last?.message ?? result.rawOutput
            return false
        }
        removeSpeakerProfileLocally(profile)
        return true
    }

    /// Pair a local voice profile with an Odoo res.partner. The
    /// engine returns the updated profile so the caller can swap
    /// it into its in-memory list without a list-all round-trip.
    func linkSpeakerToOdoo(
        _ profile: SpeakerProfile,
        partnerId: Int,
        partnerName: String,
        companyId: Int?,
        companyName: String
    ) async -> SpeakerProfile? {
        var args: [String] = [
            "library-link-speaker-profile",
            "\(profile.id)",
            "--partner-id", "\(partnerId)",
            "--partner-name", partnerName,
            "--company-name", companyName,
        ]
        if let companyId, companyId > 0 {
            args.append(contentsOf: ["--company-id", "\(companyId)"])
        }
        let result = await EngineProcess.runCommand(
            arguments: EngineProcess.defaultPythonArguments(args)
        )
        if result.status != 0 {
            errorMessage = result.events.last?.message ?? result.rawOutput
            return nil
        }
        guard let data = result.rawOutput.data(using: .utf8),
              let updated = try? JSONDecoder().decode(SpeakerProfile.self, from: data)
        else { return nil }
        replaceSpeakerProfile(updated)
        return updated
    }

    func unlinkSpeakerFromOdoo(_ profile: SpeakerProfile) async -> SpeakerProfile? {
        let result = await EngineProcess.runCommand(
            arguments: EngineProcess.defaultPythonArguments([
                "library-unlink-speaker-profile",
                "\(profile.id)",
            ])
        )
        if result.status != 0 {
            errorMessage = result.events.last?.message ?? result.rawOutput
            return nil
        }
        guard let data = result.rawOutput.data(using: .utf8),
              let updated = try? JSONDecoder().decode(SpeakerProfile.self, from: data)
        else { return nil }
        replaceSpeakerProfile(updated)
        return updated
    }
}

/// Encapsulates everything Odoo-related the SwiftUI side calls. We
/// avoid stuffing this into ``LibraryStore`` because the Odoo
/// requests are network-bound and shouldn't share the library's
/// loading state — a slow Odoo server would otherwise spin the
/// library spinner forever.
@MainActor
final class OdooStore: ObservableObject {
    @Published var status: ConnectionStatus = .unknown
    @Published var lastError: String?
    private var partnerSearchCache: [String: [OdooPartner]] = [:]

    enum ConnectionStatus: Equatable {
        case unknown
        case checking
        case ok(account: String, partnerCount: Int)
        case failed(String)
    }

    private weak var settings: SettingsStore?

    func bind(_ settings: SettingsStore) {
        self.settings = settings
    }

    private func args(extra: [String]) -> [String] {
        guard let s = settings else { return [] }
        return [
            "--url", s.odooUrl,
            "--db", s.odooDatabase,
            "--login", s.odooLogin,
            "--api-key", s.odooApiKey,
        ] + extra
    }

    func testConnection() async {
        guard let settings, settings.odooConfigured else {
            status = .failed("Configuration Odoo incomplète.")
            return
        }
        status = .checking
        let result = await EngineProcess.runCommand(
            arguments: EngineProcess.defaultPythonArguments(["odoo-test"] + args(extra: []))
        )
        guard let data = result.rawOutput.data(using: .utf8) else {
            status = .failed("Réponse vide.")
            return
        }
        struct Payload: Decodable {
            var ok: Bool?
            var error: String?
            var login: String?
            var partner_count: Int?
        }
        guard let payload = try? JSONDecoder().decode(Payload.self, from: data) else {
            status = .failed("Réponse Odoo illisible.")
            return
        }
        if payload.ok == true {
            status = .ok(
                account: payload.login ?? settings.odooLogin,
                partnerCount: payload.partner_count ?? 0
            )
        } else {
            status = .failed(payload.error ?? "Connexion impossible.")
        }
    }

    /// Live-search res.partner records by name / email. Returns an
    /// empty list when the query is blank or when Odoo is offline —
    /// the linker UI shows that as "no results yet".
    func searchPartners(_ query: String, limit: Int = 25) async -> [OdooPartner] {
        guard let settings, settings.odooConfigured else { return [] }
        let trimmed = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return [] }
        let cacheKey = "\(settings.odooUrl)|\(settings.odooDatabase)|\(trimmed.lowercased())|\(limit)"
        if let cached = partnerSearchCache[cacheKey] {
            return cached
        }
        let result = await EngineProcess.runCommand(
            arguments: EngineProcess.defaultPythonArguments(
                ["odoo-search-partners", "--jsonl", "--query", trimmed, "--limit", "\(limit)"]
                + args(extra: [])
            )
        )
        if result.status != 0 {
            // Don't poison ``status`` — the user may still be typing.
            // The picker just stays empty.
            lastError = result.events.last?.message ?? result.rawOutput
            return []
        }
        let partners = result.lines.compactMap { line in
            try? JSONDecoder().decode(OdooPartner.self, from: Data(line.utf8))
        }
        partnerSearchCache[cacheKey] = partners
        return partners
    }

    /// Detect ``calendar.event`` records around ``moment`` so the
    /// Run Setup can offer "is this meeting one of those?" before
    /// the user types anything. Returns an empty list when Odoo
    /// isn't configured — the surface treats that as "no
    /// suggestions" without surfacing an error.
    func searchMeetings(
        near moment: Date,
        windowHours: Double = 2.0,
        limit: Int = 10
    ) async -> [OdooMeetingSuggestion] {
        guard let settings, settings.odooConfigured else { return [] }
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime]
        let nearString = formatter.string(from: moment)
        let result = await EngineProcess.runCommand(
            arguments: EngineProcess.defaultPythonArguments(
                [
                    "odoo-search-meetings",
                    "--jsonl",
                    "--near", nearString,
                    "--window-hours", String(format: "%.2f", windowHours),
                    "--limit", "\(limit)",
                ] + args(extra: [])
            )
        )
        if result.status != 0 {
            lastError = result.events.last?.message ?? result.rawOutput
            return []
        }
        return result.lines.compactMap { line in
            try? JSONDecoder().decode(OdooMeetingSuggestion.self, from: Data(line.utf8))
        }
    }
}

@MainActor
final class ModelStore: ObservableObject {
    @Published var models: [ModelRow] = []
    @Published var isLoading = false
    @Published var errorMessage: String?

    func refresh() async {
        isLoading = true
        errorMessage = nil
        let result = await EngineProcess.runCommand(
            arguments: EngineProcess.defaultPythonArguments(["model-list", "--jsonl"])
        )
        if result.status != 0 {
            errorMessage = result.rawOutput
        }
        models = result.lines.compactMap { line in
            try? JSONDecoder().decode(ModelRow.self, from: Data(line.utf8))
        }
        isLoading = false
    }

    func download(_ row: ModelRow) async {
        isLoading = true
        errorMessage = nil
        let result = await EngineProcess.runCommand(
            arguments: EngineProcess.defaultPythonArguments(["model-download", row.id])
        )
        if result.status != 0 {
            errorMessage = result.rawOutput
            isLoading = false
            return
        }
        await refresh()
    }

    func delete(_ row: ModelRow) async {
        isLoading = true
        errorMessage = nil
        let result = await EngineProcess.runCommand(
            arguments: EngineProcess.defaultPythonArguments(["model-delete", row.id])
        )
        if result.status != 0 {
            errorMessage = result.rawOutput
            isLoading = false
            return
        }
        await refresh()
    }
}

/// One catalogue entry as returned by the engine's
/// ``model_catalog`` CLI. Each row carries the metadata the role-
/// grouped Models tab needs: ``role`` decides which section the
/// row lives in, ``size_mb`` powers the download-confirmation
/// dialog, ``tier`` colours the recommendation badge.
struct ModelRow: Codable, Identifiable {
    var id: String
    var family: String
    var label: String
    var role: String
    var size_mb: Int
    var tier: String
    var language: [String]
    var `default`: Bool
    var gated: Bool
    var cached: Bool
    var cache_dir: String

    /// Composite identity: the same Whisper checkpoint can appear
    /// in both the ``transcription`` and ``multipass`` roles. The
    /// tab uses ``(id, role)`` everywhere so the "Activer" button
    /// targets the right slot without ambiguity.
    var compositeID: String { "\(role)|\(id)" }

    enum CodingKeys: String, CodingKey {
        case id, family, label, role, size_mb, tier, language, cached, cache_dir, gated
        case isDefault = "default"
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decode(String.self, forKey: .id)
        family = try c.decode(String.self, forKey: .family)
        label = try c.decode(String.self, forKey: .label)
        role = try c.decodeIfPresent(String.self, forKey: .role) ?? "transcription"
        size_mb = try c.decodeIfPresent(Int.self, forKey: .size_mb) ?? 0
        tier = try c.decodeIfPresent(String.self, forKey: .tier) ?? "balanced"
        language = try c.decodeIfPresent([String].self, forKey: .language) ?? ["multi"]
        `default` = try c.decodeIfPresent(Bool.self, forKey: .isDefault) ?? false
        gated = try c.decodeIfPresent(Bool.self, forKey: .gated) ?? false
        cached = try c.decode(Bool.self, forKey: .cached)
        cache_dir = try c.decode(String.self, forKey: .cache_dir)
    }

    func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(id, forKey: .id)
        try c.encode(family, forKey: .family)
        try c.encode(label, forKey: .label)
        try c.encode(role, forKey: .role)
        try c.encode(size_mb, forKey: .size_mb)
        try c.encode(tier, forKey: .tier)
        try c.encode(language, forKey: .language)
        try c.encode(self.default, forKey: .isDefault)
        try c.encode(gated, forKey: .gated)
        try c.encode(cached, forKey: .cached)
        try c.encode(cache_dir, forKey: .cache_dir)
    }
}
