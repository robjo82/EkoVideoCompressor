import Foundation
import SQLite3

// SQLite wants to know whether a bound blob/text is transient (copy it)
// or static. Our Swift strings are freed after the call returns, so we
// must ask SQLite to copy — that's what SQLITE_TRANSIENT means.
private let SQLITE_TRANSIENT = unsafeBitCast(-1, to: sqlite3_destructor_type.self)

/// Direct, in-process SQLite access to the engine's ``library.db`` for
/// the simple, high-frequency edits (speaker map, technical terms,
/// title).
///
/// Why this exists: every library edit used to spawn a cold-started
/// Python engine subprocess (~1-2 s) just to run a one-line UPDATE.
/// That cold start — stacked with the synchronous voice-enrolment the
/// rename path triggered — is what made renaming an interlocuteur feel
/// laggy and "not stick". The displayed state (what the library row and
/// the rename sheet read) is now written here, in-process and instantly,
/// committed to the very same database file the engine uses. The heavy,
/// genuinely-Python work (voice enrolment, rewriting the on-disk
/// transcript files with the new labels) stays in the engine and is
/// dispatched in the background to reconcile.
final class LibraryDatabase: @unchecked Sendable {
    static let shared = LibraryDatabase()

    /// All access is serialised on this queue: SQLite connections aren't
    /// safe to share across threads, and serialising keeps us off the
    /// main actor so a busy-timeout wait never freezes the UI.
    private let queue = DispatchQueue(label: "com.ekonum.ekovideo.library-db")

    private init() {}

    /// Same path the engine resolves (``app_support_dir()/library.db``
    /// = ``~/Library/Application Support/EkoVideo Compressor``). Built
    /// from ``NSHomeDirectory()`` to match the engine's ``Path.home()``
    /// byte-for-byte — and the rest of the app's @AppStorage paths —
    /// rather than ``.applicationSupportDirectory`` which would diverge
    /// to a container path under sandboxing.
    static func databaseURL() -> URL {
        URL(fileURLWithPath: NSHomeDirectory())
            .appendingPathComponent("Library/Application Support/EkoVideo Compressor/library.db")
    }

    /// Apply a simple edit to one job row. ``nil`` fields are left
    /// untouched. Returns ``true`` only when a row was actually updated,
    /// so callers can fall back to the engine if the direct write
    /// couldn't happen (DB missing on a fresh install, locked, etc.).
    func applyEdit(
        jobId: Int,
        title: String? = nil,
        technicalTermsJSON: String? = nil,
        speakerMapJSON: String? = nil
    ) async -> Bool {
        await withCheckedContinuation { continuation in
            queue.async {
                continuation.resume(returning: self.applyEditSync(
                    jobId: jobId,
                    title: title,
                    technicalTermsJSON: technicalTermsJSON,
                    speakerMapJSON: speakerMapJSON
                ))
            }
        }
    }

    private func applyEditSync(
        jobId: Int,
        title: String?,
        technicalTermsJSON: String?,
        speakerMapJSON: String?
    ) -> Bool {
        let path = Self.databaseURL().path
        guard FileManager.default.fileExists(atPath: path) else { return false }

        var db: OpaquePointer?
        guard sqlite3_open_v2(path, &db, SQLITE_OPEN_READWRITE, nil) == SQLITE_OK else {
            sqlite3_close(db)
            return false
        }
        defer { sqlite3_close(db) }
        // The engine may be writing concurrently (e.g. a background
        // rename reconcile). Wait briefly rather than failing instantly.
        sqlite3_busy_timeout(db, 4000)

        var columns: [String] = []
        var values: [String] = []
        if let title {
            columns.append("custom_title")
            values.append(title)
        }
        if let technicalTermsJSON {
            columns.append("technical_terms_json")
            values.append(technicalTermsJSON)
        }
        if let speakerMapJSON {
            columns.append("speaker_map_json")
            values.append(speakerMapJSON)
        }
        guard !columns.isEmpty else { return false }

        let assignments = columns.map { "\($0) = ?" }.joined(separator: ", ")
        let sql = "UPDATE jobs SET \(assignments), updated_at = CURRENT_TIMESTAMP WHERE id = ?;"
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { return false }
        defer { sqlite3_finalize(stmt) }

        var index: Int32 = 1
        for value in values {
            sqlite3_bind_text(stmt, index, value, -1, SQLITE_TRANSIENT)
            index += 1
        }
        sqlite3_bind_int64(stmt, index, Int64(jobId))

        guard sqlite3_step(stmt) == SQLITE_DONE else { return false }
        return sqlite3_changes(db) > 0
    }
}
