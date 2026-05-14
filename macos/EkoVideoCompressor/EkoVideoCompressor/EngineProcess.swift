import Foundation

@MainActor
final class EngineProcess: ObservableObject {
    @Published private(set) var isRunning = false
    @Published private(set) var events: [EngineEvent] = []
    @Published private(set) var outputLines: [String] = []
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
                    self.process = nil
                    continuation.resume(returning: process.terminationStatus)
                }
            }

            do {
                try process.run()
            } catch {
                isRunning = false
                self.process = nil
                lastError = error.localizedDescription
                continuation.resume(returning: -1)
            }
        }
    }

    func cancel() {
        process?.terminate()
    }

    private func consumeOutput(_ text: String) {
        for line in text.split(separator: "\n") {
            guard let data = String(line).data(using: .utf8) else { continue }
            if let event = try? decoder.decode(EngineEvent.self, from: data) {
                events.append(event)
                if event.event == .error {
                    lastError = event.message
                }
            } else {
                outputLines.append(String(line))
            }
        }
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
