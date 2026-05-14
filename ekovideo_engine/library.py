from __future__ import annotations

from pathlib import Path
from typing import Any

from database_manager import DatabaseManager

from .paths import library_db_path


def database(path: str | Path | None = None) -> DatabaseManager:
    db_path = Path(path) if path else library_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return DatabaseManager(db_path)


def library_list(limit: int = 1000, status: str | None = None) -> list[dict[str, Any]]:
    return database().list_jobs(limit=limit, status=status)


def library_delete(job_id: int) -> None:
    database().delete_job(job_id)


def library_update_context(
    job_id: int,
    speakers: dict[str, str] | None = None,
    technical_terms: list[str] | None = None,
) -> None:
    database().update_job_context(job_id, speakers=speakers, technical_terms=technical_terms)


def library_rename_speakers(job_id: int, mapping: dict[str, str]) -> int:
    db = database()
    segments = db.get_segments(job_id)
    changed = 0
    updated = []
    for segment in segments:
        segment = dict(segment)
        speaker = segment.get("speaker")
        if speaker in mapping:
            segment["speaker"] = mapping[speaker]
            changed += 1
        updated.append(segment)
    if changed:
        db.add_segments(job_id, updated)
    return changed
