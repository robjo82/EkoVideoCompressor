import Foundation

enum EngineEventKind: String, Codable {
    case progress
    case artifact
    case context
    case warning
    case error
    case done
    /// Token + cost telemetry emitted by the cloud transcription
    /// engine after each remote API call.
    case usage
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
    // Usage-event payload (nil on every other event kind).
    let provider: String?
    let input_tokens: Int?
    let output_tokens: Int?
    let cost_usd: Double?

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
        case provider
        case input_tokens
        case output_tokens
        case cost_usd
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
    var library_job_id: Int?
    var delete_source_after_copy: Bool
    /// Actual meeting date used for Odoo matching and generated
    /// artefact metadata. ISO-8601 string; defaults to source file
    /// metadata in the SwiftUI setup screen.
    var meeting_date: String?
    /// Optional Odoo object whose chatter the engine fetches during
    /// the LLM step to enrich the correction prompt. ``nil`` when
    /// the user didn't pair the file with a meeting.
    var odoo_context_ref: OdooContextRef?
    /// Snapshot of the meeting the user paired in Run Setup. The
    /// runner persists this on the job row so the rename sheet can
    /// surface attendee hint chips after the engine exits.
    var odoo_meeting_metadata: OdooMeetingMetadata?
}

struct OdooContextRef: Codable, Equatable {
    var model: String
    var record_id: Int
    var url: String
    var database: String
    var login: String
    var api_key: String
}

/// JSON-shape the engine persists on ``jobs.odoo_meeting_json``.
/// Used both as a transport payload (Run Setup → runner) and as a
/// readback shape (library row → rename sheet).
struct OdooMeetingMetadata: Codable, Equatable {
    var event_id: Int
    var event_name: String
    var attendees: [OdooMeetingAttendee]
    var related: OdooRelatedObject?
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
    /// Actual meeting date captured at run setup, distinct from
    /// created/updated timestamps. Hidden by default in the library.
    var meeting_date: String?
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
    /// Raw JSON of the Odoo meeting metadata the user paired with
    /// this job in Run Setup. Decoded on demand by the rename
    /// sheet so it can show one-click attribution chips for each
    /// invitee.
    var odoo_meeting_json: String?
    /// Raw JSON array of previous-run snapshots created on rerun.
    /// Newest first. Each entry mirrors the four artefact paths
    /// (compressed / transcript / enhanced / review) that existed
    /// when the rerun started, moved into ``versions/<timestamp>/``
    /// so the current run is free to overwrite the originals
    /// without losing work.
    var previous_versions_json: String?
    /// Total remote-API spend of the job in USD, denormalised from
    /// the ``api_usage`` ledger. NULL on local-only jobs — the
    /// optional "Coût API" column renders those as "—".
    var cloud_cost_usd: Double?
    /// Remote model that produced the transcript (e.g.
    /// "gemini-3.5-flash"). NULL on local jobs.
    var cloud_model: String?
    /// Model + engine that actually produced the transcript, persisted
    /// at completion. ``transcription_engine`` is "local"/"cloud".
    /// NULL on jobs that finished before this column existed (cloud
    /// rows fall back to ``cloud_model``).
    var transcription_model: String?
    var transcription_engine: String?

    /// Raw model id used, with the cloud model as a fallback for rows
    /// written before ``transcription_model`` existed. Empty when
    /// unknown.
    var modelIdUsed: String {
        let id = (transcription_model ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        if !id.isEmpty { return id }
        return (cloud_model ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Whether this job ran on the cloud engine (for a badge).
    var ranOnCloud: Bool {
        if let engine = transcription_engine { return engine == "cloud" }
        return !(cloud_model ?? "").isEmpty
    }

    var filename: String {
        URL(fileURLWithPath: source_path ?? "").lastPathComponent
    }

    /// Decoded ``OdooMeetingMetadata`` if the job was paired with
    /// a calendar event in Run Setup. ``nil`` otherwise.
    var odooMeeting: OdooMeetingMetadata? {
        guard let raw = odoo_meeting_json, !raw.isEmpty,
              let data = raw.data(using: .utf8) else { return nil }
        return try? JSONDecoder().decode(OdooMeetingMetadata.self, from: data)
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

    /// Decoded list of previous-run snapshots. Newest first.
    /// Empty when the job has never been rerun, or when the
    /// stored JSON is malformed (we'd rather hide the section
    /// than crash the library on bad data).
    var previousVersions: [LibraryPreviousVersion] {
        guard let raw = previous_versions_json, !raw.isEmpty,
              let data = raw.data(using: .utf8) else { return [] }
        return (try? JSONDecoder().decode([LibraryPreviousVersion].self, from: data)) ?? []
    }
}

/// One archived snapshot of a previous run's artefacts. Lives in
/// ``workspace/versions/<label>/`` on disk; the four path fields
/// point to whichever of the user-facing outputs existed when
/// the rerun started (any can be empty when the prior run skipped
/// that step).
struct LibraryPreviousVersion: Codable, Identifiable, Equatable {
    var label: String
    var created_at: String
    var compressed_path: String?
    var transcript_path: String?
    var enhanced_transcript_path: String?
    var review_path: String?

    /// Stable identifier — the timestamp label is unique within a
    /// job's history (we generate one per rerun at second
    /// granularity, and the engine never re-runs twice in the same
    /// second on the same job).
    var id: String { label }

    /// Folder containing the snapshot files. Derived from any of
    /// the artefact paths since they all share the same parent.
    var folderPath: String? {
        let first = [compressed_path, transcript_path, enhanced_transcript_path, review_path]
            .compactMap { $0 }
            .first { !$0.isEmpty }
        guard let path = first else { return nil }
        return URL(fileURLWithPath: path).deletingLastPathComponent().path
    }

    /// Pretty French rendering of ``created_at`` for the detail
    /// section ("17 mai 2026 à 14:30"). Falls back to the raw
    /// timestamp when parsing fails so we never show nothing.
    var displayedTimestamp: String {
        let formatter = ISO8601DateFormatter()
        if let date = formatter.date(from: created_at) {
            let out = DateFormatter()
            out.locale = Locale(identifier: "fr_FR")
            out.dateStyle = .medium
            out.timeStyle = .short
            return out.string(from: date)
        }
        return created_at
    }

    /// Names of the artefacts kept in this snapshot, ready for a
    /// caption line ("compressé, transcription, améliorée").
    var artefactSummary: String {
        var parts: [String] = []
        if let p = compressed_path, !p.isEmpty { parts.append("compressé") }
        if let p = transcript_path, !p.isEmpty { parts.append("transcription") }
        if let p = enhanced_transcript_path, !p.isEmpty { parts.append("améliorée") }
        if let p = review_path, !p.isEmpty { parts.append("rapport") }
        return parts.joined(separator: " · ")
    }
}

struct SpeakerSample: Codable, Identifiable, Equatable {
    var speaker: String
    var path: String
    var start: Double
    var duration: Double
    var index: Int?
    var utterance_count: Int?
    var total_duration: Double?
    var text: String?

    var id: String { "\(speaker)-\(index ?? 1)-\(start)" }
}

/// One enrolled voice profile. The engine stores a 512-dim
/// embedding alongside; we don't ship it to the UI (5 KB per row,
/// useless for display), only the metadata.
struct SpeakerProfile: Codable, Identifiable, Equatable {
    var id: Int
    var name: String
    var name_key: String
    var sample_count: Int
    var created_at: String?
    var updated_at: String?
    // Optional Odoo linkage — present when the user paired the
    // voice profile with a ``res.partner`` record. Absent on any
    // profile that's still purely local.
    var odoo_partner_id: Int?
    var odoo_partner_name: String?
    var odoo_company_id: Int?
    var odoo_company_name: String?
    var linked_at: String?

    var isLinkedToOdoo: Bool {
        guard let pid = odoo_partner_id, pid > 0 else { return false }
        return true
    }

    /// Bucket key for the SwiftUI Interlocuteurs grouping.
    /// "Sans société" when no Odoo company is set; the company
    /// name otherwise (linked partner inherits its parent company).
    var groupingLabel: String {
        if let company = odoo_company_name, !company.isEmpty {
            return company
        }
        if isLinkedToOdoo {
            // Linked but the partner is itself a top-level company.
            return odoo_partner_name ?? name
        }
        return "Sans société Odoo"
    }
}

/// Minimal Odoo res.partner shape used by the search picker. The
/// engine flattens parent_id from Odoo's many2one [id, name] pair into
/// two scalar fields so the UI doesn't have to branch.
struct OdooPartner: Codable, Identifiable, Equatable {
    var id: Int
    var name: String
    var display_name: String
    var parent_id: Int
    var parent_name: String
    var is_company: Bool
    var email: String
    var phone: String
    var function: String
}

extension SpeakerProfile {
    var sampleSummary: String {
        if sample_count <= 0 {
            return "Nom enregistré · voix à apprendre"
        }
        return "\(sample_count) extrait\(sample_count > 1 ? "s" : "")"
    }

    /// PR AG — true when this row is just a "remembered name"
    /// (no embedding yet). The engine creates these shells from
    /// ``speaker_overrides`` at job start so the rename remains
    /// visible even if pyannote enrolment later fails, but they
    /// don't actually participate in voice matching (PR T filters
    /// them server-side). The SwiftUI row uses this to render a
    /// distinct icon + label so the user doesn't confuse a
    /// confirmed name with a memorised voice.
    var isUnvoicedShell: Bool {
        sample_count <= 0
    }

    func linked(to partner: OdooPartner, companyId: Int?, companyName: String) -> SpeakerProfile {
        var copy = self
        copy.odoo_partner_id = partner.id
        copy.odoo_partner_name = partner.display_name
        copy.odoo_company_id = companyId
        copy.odoo_company_name = companyName
        return copy
    }

    func unlinkedFromOdoo() -> SpeakerProfile {
        var copy = self
        copy.odoo_partner_id = nil
        copy.odoo_partner_name = nil
        copy.odoo_company_id = nil
        copy.odoo_company_name = nil
        copy.linked_at = nil
        return copy
    }
}

/// One Odoo ``calendar.event`` suggestion surfaced in Run Setup so
/// the user can click "this is the meeting" and pre-fill speaker
/// names + expected count without typing them by hand.
struct OdooMeetingSuggestion: Codable, Identifiable, Equatable {
    var id: Int
    var name: String
    var start: String
    var stop: String
    var duration_minutes: Double
    var allday: Bool
    var location: String
    var description: String
    var partner_ids: [Int]
    var attendee_count: Int
    var related_object: OdooRelatedObject?
    var attendees: [OdooMeetingAttendee]
}

struct OdooMeetingAttendee: Codable, Identifiable, Equatable {
    var id: Int
    var name: String
    var email: String
    var company: String
}

struct OdooRelatedObject: Codable, Equatable {
    var model: String
    var id: Int
    var name: String
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
    var trim_enabled = false
    var trim_start = "00:00:00"
    var trim_end = "00:00:00"
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
            // wires today. The list is kept in sync with
            // ``QUALITY_PRESETS["max"]`` in the engine (PR L). PR Y
            // removed "Whisper contexte croisé" from this preset
            // after the audit found it causing 70-minute decoder
            // loops on long meetings.
            return "Tout activer : VAD + repasses haute qualité (zones douteuses + frontières de locuteurs) + passe per-locuteur + relecture LLM + enrichissement web du glossaire + réécoute IA Qwen2-Audio des passages douteux."
        case .custom:
            return "Conserve les bascules avancées telles que définies."
        }
    }
}

/// Which transcription stack runs the queue: the historic on-device
/// MLX Whisper pipeline, or a remote API (Gemini) that returns
/// transcript + diarisation + titre in one call — better quality on
/// hard audio, but metered. Mirrors the engine's
/// ``TranscriptionSettings.transcription_engine`` values exactly.
enum TranscriptionEngineChoice: String, Codable, CaseIterable, Identifiable {
    case local
    case cloud

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .local: return "Local (Whisper)"
        case .cloud: return "Cloud (Gemini)"
        }
    }

    var summary: String {
        switch self {
        case .local:
            return "Tout reste sur ce Mac. Gratuit, plus lent, qualité selon le modèle Whisper choisi."
        case .cloud:
            return "L'audio est envoyé à l'API Gemini (transcription + locuteurs + titre en un appel). Facturé au token — le coût estimé s'affiche avant le lancement."
        }
    }
}

/// Client-side mirror of the engine's audio cost model, billing-aware:
///  * per-token (multimodal LLMs): Gemini tokenises audio at 32
///    tokens/s and a dense meeting renders to ~10 output tokens/s of
///    JSON;
///  * per-hour (dedicated STT): a flat duration rate.
/// Used for the pre-launch estimate only — the engine re-checks
/// against the real budget before any upload and bills from the API's
/// actual counters/duration afterwards.
func estimatedCloudCostUSD(
    durationSeconds: Double,
    billing: String,
    priceInPer1M: Double,
    priceOutPer1M: Double,
    pricePerHour: Double
) -> Double {
    guard durationSeconds > 0 else { return 0 }
    if billing == "per_hour" {
        return (durationSeconds / 3600.0) * pricePerHour
    }
    let inputTokens = durationSeconds * 32 + 400
    let outputTokens = durationSeconds * 10 + 300
    return (inputTokens * priceInPer1M + outputTokens * priceOutPer1M) / 1_000_000
}

func formatUSD(_ value: Double) -> String {
    String(format: "%.2f $US", value)
}

/// Human-readable label for a model id used in the library "Modèle"
/// column. Prefers the catalogue label (passed in by the caller from
/// ModelStore); otherwise cleans the raw id (drops the HF org prefix).
func friendlyModelLabel(_ modelID: String, catalogueLabel: String? = nil) -> String {
    let id = modelID.trimmingCharacters(in: .whitespacesAndNewlines)
    if id.isEmpty { return "—" }
    if let label = catalogueLabel, !label.isEmpty { return label }
    if let slash = id.lastIndex(of: "/") {
        return String(id[id.index(after: slash)...])
    }
    return id
}

/// Which provider owns a cloud model id. Derived from the id so the
/// SwiftUI side picks the right API key without needing the engine
/// catalogue loaded. Mirrors ``cloud_transcription.provider_for_model``.
func cloudProviderForModel(_ modelID: String) -> String {
    let id = modelID.lowercased()
    if id.hasPrefix("gpt-") || id.contains("openai") { return "openai" }
    if id.hasPrefix("assemblyai") { return "assemblyai" }
    if id.hasPrefix("gladia") { return "gladia" }
    if id.hasPrefix("deepgram") { return "deepgram" }
    return "gemini"
}

/// Display label + API-key console URL per provider, for the Réglages
/// key fields.
struct CloudProviderInfo: Identifiable {
    let id: String
    let label: String
    let consoleURL: String

    static let all: [CloudProviderInfo] = [
        .init(id: "gemini", label: "Google Gemini", consoleURL: "https://aistudio.google.com/apikey"),
        .init(id: "openai", label: "OpenAI", consoleURL: "https://platform.openai.com/api-keys"),
        .init(id: "assemblyai", label: "AssemblyAI", consoleURL: "https://www.assemblyai.com/app/account"),
        .init(id: "gladia", label: "Gladia", consoleURL: "https://app.gladia.io/account"),
        .init(id: "deepgram", label: "Deepgram", consoleURL: "https://console.deepgram.com/"),
    ]

    static func label(for provider: String) -> String {
        all.first { $0.id == provider }?.label ?? provider
    }
}

/// Decoded payload of the engine's ``usage-summary`` command.
struct CloudUsageSummary: Codable, Equatable {
    struct Month: Codable, Equatable, Identifiable {
        var month: String
        var calls: Int
        var input_tokens: Int?
        var output_tokens: Int?
        var cost_usd: Double?

        var id: String { month }
    }

    var current_month: String
    var current_month_cost_usd: Double
    var months: [Month]
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
    /// User-selected repass model. Empty string lets the engine fall
    /// back to the catalog default (Whisper Large v3).
    var multipass_model = ""
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
    /// Name the user designates as "themselves" in Réglages. The
    /// engine uses it to pre-attribute the cluster that speaks first
    /// to this name when no voiceprint match has been found — fixes
    /// the cold-start problem where every meeting starts with
    /// SPEAKER_00 unresolved.
    var current_user_name: String = ""
    /// Whisper ``--condition-on-previous-text``. When True, decoded
    /// text from window N is fed into the prompt of window N+1 to
    /// stabilise proper nouns across the meeting. The ``max`` preset
    /// enables it; balanced/fast keep it off because runaway decoder
    /// loops on long silences are harder to detect than they're
    /// worth in those modes. Engine clamps decoder loops > 2
    /// downstream so the worst case is bounded.
    var condition_on_previous_text: Bool = false
    /// When True, after the first Whisper pass the engine mines the
    /// transcript for repeated proper nouns and folds them into the
    /// glossary in-process. Cheaper alternative to chunked re-passes
    /// at 5-minute boundaries. ``max`` preset toggles this on.
    var hot_prompt_enrichment: Bool = false
    /// "local" (MLX Whisper on-device) or "cloud" (remote Gemini
    /// call). The engine routes the whole quality stack on this.
    var transcription_engine: String = TranscriptionEngineChoice.local.rawValue
    /// Remote model id; empty falls back to the engine's catalogue
    /// default.
    var cloud_model: String = ""
    /// Gemini API key. Travels on the job request like the Odoo
    /// credentials; the engine redacts it before persisting
    /// settings to the library DB.
    var cloud_api_key: String = ""
    /// Hard monthly spending cap in USD enforced engine-side
    /// (0 = no cap). The estimate is checked *before* any upload.
    var cloud_budget_monthly_usd: Double = 0
}
