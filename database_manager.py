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
                    status TEXT DEFAULT 'PENDING',
                    error_message TEXT,
                    settings_json TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
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
