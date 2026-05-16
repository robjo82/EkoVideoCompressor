import Foundation

enum EngineEventKind: String, Codable {
    case progress
    case artifact
    case context
    case warning
    case error
    case done
}

struct EngineEvent: Codable, Identifiable {
    var id = UUID()
    let event: EngineEventKind
    let ts: String?
    let step: String?
    let pct: Double?
    let eta_seconds: Double?
    let message: String?
    let kind: String?
    let path: String?
    let model: String?
    let code: String?
    let speakers: [String: String]?
    let technical_terms: [String]?
    let summary: [String: JSONValue]?

    enum CodingKeys: String, CodingKey {
        case event
        case ts
        case step
        case pct
        case eta_seconds
        case message
        case kind
        case path
        case model
        case code
        case speakers
        case technical_terms
        case summary
    }
}

enum JSONValue: Codable, Equatable {
    case string(String)
    case number(Double)
    case bool(Bool)
    case object([String: JSONValue])
    case array([JSONValue])
    case null

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            self = .null
        } else if let value = try? container.decode(Bool.self) {
            self = .bool(value)
        } else if let value = try? container.decode(Double.self) {
            self = .number(value)
        } else if let value = try? container.decode(String.self) {
            self = .string(value)
        } else if let value = try? container.decode([String: JSONValue].self) {
            self = .object(value)
        } else {
            self = .array(try container.decode([JSONValue].self))
        }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .string(let value):
            try container.encode(value)
        case .number(let value):
            try container.encode(value)
        case .bool(let value):
            try container.encode(value)
        case .object(let value):
            try container.encode(value)
        case .array(let value):
            try container.encode(value)
        case .null:
            try container.encodeNil()
        }
    }
}

struct JobRequest: Codable {
    var source_path: String
    var workspace_dir: String
    var output_dir: String
    var mode: String
    var profile: String
    var compression_settings: CompressionSettings
    var transcription_settings: TranscriptionSettings
    var glossary_terms: [String]
    var speaker_overrides: [String: String]
    var technical_terms: [String]
    var rerun_steps: [String]
    var delete_source_after_copy: Bool
}

struct LibraryRow: Codable, Identifiable, Equatable {
    var id: Int
    var source_path: String?
    var workspace_dir: String?
    var output_path: String?
    var custom_title: String?
    var status: String?
    var error_message: String?
    var updated_at: String?
    var created_at: String?
    var compressed_path: String?
    var transcript_path: String?
    var enhanced_transcript_path: String?
    var review_path: String?
    var speaker_map_json: String?
    var technical_terms_json: String?
    var current_step: String?
    var progress_pct: Double?
    var eta_seconds: Double?
    /// Workspace size at completion. NULL on legacy rows that
    /// finished before the column existed — the library renders "—"
    /// for those instead of "0 octets".
    var total_bytes: Int64?

    var filename: String {
        URL(fileURLWithPath: source_path ?? "").lastPathComponent
    }

    var customTitleOrFilename: String {
        let title = (custom_title ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        return title.isEmpty ? filename : title
    }

    var copiedSourcePath: String? {
        guard let workspace_dir, !workspace_dir.isEmpty,
              let source_path, !source_path.isEmpty else { return nil }
        return URL(fileURLWithPath: workspace_dir)
            .appendingPathComponent(URL(fileURLWithPath: source_path).lastPathComponent)
            .path
    }

    var speakerMap: [String: String] {
        decodeJSONObject(speaker_map_json)
    }

    var technicalTerms: [String] {
        decodeJSONArray(technical_terms_json)
    }
}

struct SpeakerSample: Codable, Identifiable, Equatable {
    var speaker: String
    var path: String
    var start: Double
    var duration: Double

    var id: String { speaker }
}

/// Disk-usage preview shown in the deletion sheet so the user can
/// see exactly which files (and how many bytes) they're about to
/// free before confirming a "supprimer + dossier de travail".
struct WorkspaceUsage: Codable, Equatable {
    var workspace_dir: String
    var files: [WorkspaceFile]
    var total_bytes: Int64
}

struct WorkspaceFile: Codable, Identifiable, Equatable {
    var path: String
    var name: String
    var size: Int64
    var label: String

    /// SwiftUI ``Identifiable`` conformance. The absolute path is
    /// unique within a workspace so it doubles as the row key.
    var id: String { path }
}

private func decodeJSONObject(_ raw: String?) -> [String: String] {
    guard let raw, let data = raw.data(using: .utf8),
          let value = try? JSONDecoder().decode([String: String].self, from: data) else {
        return [:]
    }
    return value
}

private func decodeJSONArray(_ raw: String?) -> [String] {
    guard let raw, let data = raw.data(using: .utf8),
          let value = try? JSONDecoder().decode([String].self, from: data) else {
        return []
    }
    return value
}

struct CompressionSettings: Codable {
    var ffmpeg_path = Bundle.main.resourceURL?.appendingPathComponent("bin/ffmpeg").path ?? "ffmpeg"
    var ffprobe_path = Bundle.main.resourceURL?.appendingPathComponent("bin/ffprobe").path ?? "ffprobe"
    var resolution = "720p"
    var fps = 12
    var crf = 28
    var audio_bitrate = "128k"
    var preset = "medium"
    var speech_enhance = true
    var mono_audio = false
}

/// Single "quality" knob the user sees in Settings. The engine
/// derives every individual toggle (VAD, multipass, per-speaker, …)
/// from this preset, so the SwiftUI app only has to expose one
/// picker for the 95% case.
enum TranscriptionQualityPreset: String, Codable, CaseIterable, Identifiable {
    /// Whisper only, no quality phases. ~real-time on M1.
    case fast
    /// Default. VAD + multipass + LLM enhancement. ~1.5× real-time.
    case balanced
    /// Per-speaker Whisper + audio recheck + web enrichment.
    /// Reserved for strategic recordings; can take hours.
    case max
    /// Power user — keep the individual toggles as set, ignore the
    /// preset. Engine treats unknown values as ``custom`` too.
    case custom

    var id: String { rawValue }

    /// Human-readable name for the picker.
    var displayName: String {
        switch self {
        case .fast: return "Rapide"
        case .balanced: return "Équilibrée"
        case .max: return "Maximale"
        case .custom: return "Personnalisée"
        }
    }

    /// One-line caption shown under the picker.
    var summary: String {
        switch self {
        case .fast:
            return "Whisper seul, transcription quasi temps réel. Aucune correction automatique."
        case .balanced:
            return "VAD + repasse haute qualité + relecture LLM. Recommandé pour la plupart des réunions."
        case .max:
            // Honest description of what the orchestrator actually
            // wires today. Kept tight on purpose — adding "réécoute
            // IA" / "enrichissement web" promised features the
            // engine didn't deliver, and users noticed.
            return "Tout activer : VAD + repasse large-v3 sur les zones douteuses + diarisation + relecture LLM."
        case .custom:
            return "Conserve les bascules avancées telles que définies."
        }
    }
}

struct TranscriptionSettings: Codable {
    var mlx_whisper_path = "\(NSHomeDirectory())/Library/Application Support/EkoVideo Compressor/mlx-whisper-venv/bin/mlx_whisper"
    var model = "mlx-community/whisper-large-v3-turbo"
    var language = "fr"
    var output_format = "txt"
    var suffix = ""
    var enhance_audio = true
    var diarization_enabled = false
    var hf_token = ""
    var text_llm_model = "mlx-community/Mistral-7B-Instruct-v0.3-4bit"
    var audio_llm_model = "mlx-community/Qwen2-Audio-7B-Instruct-4bit"
    var audio_recheck_enabled = false
    var vad_enabled = true
    var multipass_enabled = true
    var per_speaker_enabled = false
    var web_enrichment_enabled = false
    /// Sent as a string so the engine's ``apply_quality_preset`` can
    /// override the individual flags from this single choice. Default
    /// matches the engine default (``custom``) so existing app
    /// versions keep their hand-tuned toggles working.
    var quality_preset: String = TranscriptionQualityPreset.balanced.rawValue
    /// Speaker-count hints forwarded to pyannote. 0 means "let the
    /// model decide" — but in practice that under-segments most
    /// meetings, merging two real voices into a single SPEAKER_NN
    /// cluster. When the user knows the meeting size, passing both
    /// bounds gives the cleanest diarisation we've measured.
    var expected_min_speakers: Int = 0
    var expected_max_speakers: Int = 0
}
