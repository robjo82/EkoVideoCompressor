import Foundation

struct EngineCommandResult {
    var status: Int32
    var events: [EngineEvent]
    var lines: [String]
    var rawOutput: String
}

@MainActor
final class EngineProcess: ObservableObject {
    @Published private(set) var isRunning = false
    @Published private(set) var events: [EngineEvent] = []
    @Published private(set) var outputLines: [String] = []
    @Published private(set) var runStartedAt: Date?
    @Published private(set) var runFinishedAt: Date?
    @Published var lastError: String?

    private var process: Process?
    private let decoder = JSONDecoder()

    func run(arguments: [String], workingDirectory: URL? = nil) {
        Task {
            _ = await runAndWait(arguments: arguments, workingDirectory: workingDirectory)
        }
    }

    func runAndWait(arguments: [String], workingDirectory: URL? = nil) async -> Int32 {
        guard !isRunning else { return -1 }
        events.removeAll()
        outputLines.removeAll()
        runStartedAt = nil
        runFinishedAt = nil
        lastError = nil

        let process = Process()
        process.executableURL = Self.engineExecutableURL()
        process.arguments = arguments
        process.currentDirectoryURL = workingDirectory

        let output = Pipe()
        process.standardOutput = output
        process.standardError = output
        self.process = process
        isRunning = true
        runStartedAt = Date()

        output.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard !data.isEmpty else { return }
            let text = String(decoding: data, as: UTF8.self)
            guard let self else { return }
            Task { @MainActor [self] in
                self.consumeOutput(text)
            }
        }

        return await withCheckedContinuation { continuation in
            process.terminationHandler = { [weak self] process in
                output.fileHandleForReading.readabilityHandler = nil
                guard let self else {
                    continuation.resume(returning: process.terminationStatus)
                    return
                }
                Task { @MainActor [self] in
                    self.isRunning = false
                    self.runFinishedAt = Date()
                    self.process = nil
                    continuation.resume(returning: process.terminationStatus)
                }
            }

            do {
                try process.run()
            } catch {
                isRunning = false
                runFinishedAt = Date()
                self.process = nil
                lastError = error.localizedDescription
                continuation.resume(returning: -1)
            }
        }
    }

    func cancel() {
        process?.terminate()
    }

    static func runCommand(arguments: [String], workingDirectory: URL? = nil) async -> EngineCommandResult {
        let executableURL = await MainActor.run { engineExecutableURL() }
        let output = await Task.detached(priority: .userInitiated) {
            let process = Process()
            process.executableURL = executableURL
            process.arguments = arguments
            process.currentDirectoryURL = workingDirectory
            let pipe = Pipe()
            process.standardOutput = pipe
            process.standardError = pipe
            do {
                try process.run()
                let data = pipe.fileHandleForReading.readDataToEndOfFile()
                process.waitUntilExit()
                return (process.terminationStatus, String(decoding: data, as: UTF8.self))
            } catch {
                return (Int32(-1), error.localizedDescription)
            }
        }.value
        let parsed = parseOutput(output.1)
        return EngineCommandResult(
            status: output.0,
            events: parsed.events,
            lines: parsed.lines,
            rawOutput: output.1
        )
    }

    private func consumeOutput(_ text: String) {
        let parsed = Self.parseOutput(text)
        events.append(contentsOf: parsed.events)
        outputLines.append(contentsOf: parsed.lines)
        if let errorEvent = parsed.events.last(where: { $0.event == .error }) {
            lastError = errorEvent.message
        }
    }

    private static func parseOutput(_ text: String) -> (events: [EngineEvent], lines: [String]) {
        let decoder = JSONDecoder()
        var events: [EngineEvent] = []
        var lines: [String] = []
        for line in text.split(separator: "\n") {
            guard let data = String(line).data(using: .utf8) else { continue }
            if let event = try? decoder.decode(EngineEvent.self, from: data) {
                events.append(event)
            } else {
                lines.append(String(line))
            }
        }
        return (events, lines)
    }

    static func engineExecutableURL() -> URL {
        if let override = ProcessInfo.processInfo.environment["EKOVIDEO_ENGINE"] {
            return URL(fileURLWithPath: override)
        }
        let resourceURL = Bundle.main.resourceURL ?? URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        let bundled = resourceURL.appendingPathComponent("engine/ekovideo-engine")
        if FileManager.default.isExecutableFile(atPath: bundled.path) {
            return bundled
        }
        return URL(fileURLWithPath: "/usr/bin/python3")
    }

    static func defaultPythonArguments(_ args: [String]) -> [String] {
        let executable = engineExecutableURL().lastPathComponent
        if executable == "python3" || executable.hasPrefix("python") {
            return ["-m", "ekovideo_engine"] + args
        }
        return args
    }
}

extension EngineProcess: @unchecked Sendable {}
