import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict

class DatabaseManager:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self):
        # SQLite disables foreign-key enforcement by default for backwards
        # compat — without this PRAGMA, `delete_job` would leave orphaned
        # transcription_segments rows even though the schema declares
        # ON DELETE CASCADE.
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_path TEXT,
                    workspace_dir TEXT,
                    output_path TEXT,
                    custom_title TEXT,
                    status TEXT DEFAULT 'PENDING',
                    error_message TEXT,
                    settings_json TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    duration_ffmpeg REAL,
                    duration_whisper REAL,
                    duration_diarization REAL,
                    duration_total REAL
                )
            """)
            self._ensure_jobs_columns(conn)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS transcription_segments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id INTEGER,
                    start_time REAL,
                    end_time REAL,
                    speaker TEXT,
                    text TEXT,
                    FOREIGN KEY (job_id) REFERENCES jobs (id) ON DELETE CASCADE
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_segments_job_id ON transcription_segments(job_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_segments_speaker ON transcription_segments(speaker)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_segments_text ON transcription_segments(text)")

    def _ensure_jobs_columns(self, conn):
        # The library now exposes each artefact as its own button in the
        # table view, so we track them as distinct columns instead of
        # squeezing everything into `output_path`. `current_step` /
        # `eta_seconds` power the live spinner during processing.
        existing = {
            row[1]
            for row in conn.execute("PRAGMA table_info(jobs)").fetchall()
        }
        columns = {
            "custom_title": "TEXT",
            "duration_ffmpeg": "REAL",
            "duration_whisper": "REAL",
            "duration_diarization": "REAL",
            "duration_total": "REAL",
            # Per-artefact paths so the library shows ✓ / — for each.
            "compressed_path": "TEXT",
            "transcript_path": "TEXT",
            "enhanced_transcript_path": "TEXT",
            "review_path": "TEXT",
            "speaker_map_json": "TEXT",
            "technical_terms_json": "TEXT",
            # Live progress for the in-flight job(s).
            "current_step": "TEXT",
            "progress_pct": "REAL",
            "eta_seconds": "REAL",
            # Total bytes consumed by the workspace dir, snapshotted
            # at job completion. NULL for legacy rows that finished
            # before this column existed; the SwiftUI library shows
            # "—" for those instead of "0".
            "total_bytes": "INTEGER",
        }
        for name, definition in columns.items():
            if name not in existing:
                conn.execute(f"ALTER TABLE jobs ADD COLUMN {name} {definition}")

    def create_job(self, source_path: str, workspace_dir: str, settings: dict) -> int:
        with self._get_connection() as conn:
            cursor = conn.execute(
                "INSERT INTO jobs (source_path, workspace_dir, settings_json) VALUES (?, ?, ?)",
                (source_path, workspace_dir, json.dumps(settings))
            )
            return cursor.lastrowid

    def update_job_status(self, job_id: int, status: str, error_message: Optional[str] = None):
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE jobs SET status = ?, error_message = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (status, error_message, job_id)
            )

    def update_job_output(self, job_id: int, output_path: str):
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE jobs SET output_path = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (output_path, job_id)
            )

    def update_job_durations(self, job_id: int, ffmpeg: float = 0, whisper: float = 0, diarization: float = 0):
        total = ffmpeg + whisper + diarization
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE jobs SET duration_ffmpeg = ?, duration_whisper = ?, duration_diarization = ?, duration_total = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (ffmpeg, whisper, diarization, total, job_id)
            )

    def update_job_total_bytes(self, job_id: int, total_bytes: int) -> None:
        """Snapshot the workspace size at the end of a successful job.

        The library's hidden "Poids" column reads from this. Rows
        from before the column existed stay at NULL — we only
        backfill on the next successful run for that job.
        """
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE jobs SET total_bytes = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (int(total_bytes), job_id),
            )

    def update_job_title(self, job_id: int, title: str):
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE jobs SET custom_title = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (title, job_id)
            )

    def update_job_context(
        self,
        job_id: int,
        speakers: Optional[Dict[str, str]] = None,
        technical_terms: Optional[List[str]] = None,
    ):
        sets = []
        params: list = []
        if speakers is not None:
            sets.append("speaker_map_json = ?")
            params.append(json.dumps(speakers, ensure_ascii=False))
        if technical_terms is not None:
            sets.append("technical_terms_json = ?")
            params.append(json.dumps(technical_terms, ensure_ascii=False))
        if not sets:
            return
        sets.append("updated_at = CURRENT_TIMESTAMP")
        params.append(job_id)
        with self._get_connection() as conn:
            conn.execute(
                f"UPDATE jobs SET {', '.join(sets)} WHERE id = ?",
                tuple(params),
            )

    def update_job_artefact(self, job_id: int, kind: str, path: str):
        """
        Record one of the artefact paths produced by a job. Allowed kinds:
        'compressed', 'transcript', 'enhanced_transcript', 'review'.
        Each call lights up one column in the library table and the
        corresponding "Ouvrir" button.
        """
        column = {
            "compressed": "compressed_path",
            "transcript": "transcript_path",
            "enhanced_transcript": "enhanced_transcript_path",
            "review": "review_path",
        }.get(kind)
        if not column:
            raise ValueError(f"unknown artefact kind: {kind!r}")
        with self._get_connection() as conn:
            conn.execute(
                f"UPDATE jobs SET {column} = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (path, job_id),
            )

    def update_job_progress(
        self,
        job_id: int,
        step: Optional[str] = None,
        progress_pct: Optional[float] = None,
        eta_seconds: Optional[float] = None,
    ):
        """
        Push the current step + progress so the library table can render
        a live status. Any field left as None keeps its previous value.
        """
        sets = []
        params: list = []
        if step is not None:
            sets.append("current_step = ?")
            params.append(step)
        if progress_pct is not None:
            sets.append("progress_pct = ?")
            params.append(float(progress_pct))
        if eta_seconds is not None:
            sets.append("eta_seconds = ?")
            params.append(float(eta_seconds))
        if not sets:
            return
        sets.append("updated_at = CURRENT_TIMESTAMP")
        params.append(job_id)
        with self._get_connection() as conn:
            conn.execute(
                f"UPDATE jobs SET {', '.join(sets)} WHERE id = ?",
                tuple(params),
            )

    def get_job(self, job_id: int) -> Optional[dict]:
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
            row = cursor.fetchone()
            if row:
                d = dict(row)
                d['settings'] = json.loads(d['settings_json']) if d['settings_json'] else {}
                return d
            return None

    def list_jobs(self, limit: int = 100, status: Optional[str] = None) -> List[dict]:
        query = "SELECT * FROM jobs"
        params = []
        if status:
            query += " WHERE status = ?"
            params.append(status)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, tuple(params))
            return [dict(row) for row in cursor.fetchall()]

    def add_segments(self, job_id: int, segments: List[dict]):
        with self._get_connection() as conn:
            conn.execute("DELETE FROM transcription_segments WHERE job_id = ?", (job_id,))
            conn.executemany(
                "INSERT INTO transcription_segments (job_id, start_time, end_time, speaker, text) VALUES (?, ?, ?, ?, ?)",
                [(job_id, s.get('start'), s.get('end'), s.get('speaker'), s.get('text')) for s in segments]
            )

    def get_segments(self, job_id: int) -> List[dict]:
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM transcription_segments WHERE job_id = ? ORDER BY start_time", (job_id,))
            return [dict(row) for row in cursor.fetchall()]

    def search_segments(self, query_text: str = "", speaker: str = "") -> List[dict]:
        sql = """
            SELECT s.*, j.source_path 
            FROM transcription_segments s
            JOIN jobs j ON s.job_id = j.id
            WHERE 1=1
        """
        params = []
        if query_text:
            sql += " AND s.text LIKE ?"
            params.append(f"%{query_text}%")
        if speaker:
            sql += " AND s.speaker = ?"
            params.append(speaker)
        
        sql += " ORDER BY j.created_at DESC, s.start_time LIMIT 200"
        
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(sql, tuple(params))
            return [dict(row) for row in cursor.fetchall()]

    def delete_job(self, job_id: int):
        with self._get_connection() as conn:
            conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
