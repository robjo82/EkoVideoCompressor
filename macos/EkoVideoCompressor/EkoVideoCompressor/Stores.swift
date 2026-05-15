import Foundation
import SwiftUI

struct QueueItem: Identifiable, Equatable {
    let id = UUID()
    var sourceURL: URL
    var status: String = "En attente"
    var progress: Double = 0
}

@MainActor
final class QueueStore: ObservableObject {
    @Published var items: [QueueItem] = []
    @Published var isBatchRunning = false

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

    func update(_ id: QueueItem.ID, status: String, progress: Double? = nil) {
        guard let index = items.firstIndex(where: { $0.id == id }) else { return }
        items[index].status = status
        if let progress {
            items[index].progress = progress
        }
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
    @AppStorage("hfToken") var hfToken = ""
    @AppStorage("githubToken") var githubToken = ""
    @AppStorage("whisperModel") var whisperModel = "mlx-community/whisper-large-v3-turbo"
    @AppStorage("processingMode") var processingMode = "compress_transcribe"
    @AppStorage("outputFormat") var outputFormat = "txt"
    @AppStorage("audioRecheckEnabled") var audioRecheckEnabled = false
    @AppStorage("diarizationEnabled") var diarizationEnabled = false
    /// User-declared expected speaker count for the *next* run.
    ///
    /// This one is deliberately **not** ``@AppStorage`` — the value
    /// is per-meeting, not a long-term preference. A 4-person
    /// meeting today shouldn't carry "4" into next week's 6-person
    /// meeting. The Run Setup sheet surfaces it; nothing else
    /// touches it. 0 means "let pyannote estimate".
    @Published var expectedSpeakerCount: Int = 0
    @AppStorage("deleteSourceAfterCopy") var deleteSourceAfterCopy = false
    /// Single user-facing quality knob. Replaces the previous handful
    /// of toggles (VAD / multipass / per-speaker / web). The engine
    /// derives the real flags from the preset string at job time.
    @AppStorage("qualityPreset") var qualityPreset = TranscriptionQualityPreset.balanced.rawValue

    var glossaryTerms: [String] {
        glossary
            .split(whereSeparator: \.isNewline)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
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

struct ModelRow: Codable, Identifiable {
    var id: String
    var family: String
    var label: String
    var cached: Bool
    var cache_dir: String
}
