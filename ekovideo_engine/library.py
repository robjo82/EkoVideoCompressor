from __future__ import annotations

import json
import re
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


def _job_artifact_paths(job: dict[str, Any]) -> list[Path]:
    paths: list[Path] = []
    for key in (
        "transcript_path",
        "enhanced_transcript_path",
        "review_path",
    ):
        value = (job.get(key) or "").strip()
        if value:
            paths.append(Path(value))
    return paths


def _replace_speaker_labels(text: str, mapping: dict[str, str]) -> tuple[str, int]:
    changed = 0
    output = text
    for old, new in mapping.items():
        old = old.strip()
        new = new.strip()
        if not old or not new or old == new:
            continue
        patterns = [
            (rf"\[{re.escape(old)}\]", f"[{new}]"),
            (rf"`{re.escape(old)}`", f"`{new}`"),
            (rf"(?m)^({re.escape(old)})(\s*:)", rf"{new}\2"),
        ]
        for pattern, replacement in patterns:
            output, count = re.subn(pattern, replacement, output)
            changed += count
    return output, changed


def _rewrite_speaker_artifacts(job: dict[str, Any], mapping: dict[str, str]) -> int:
    rewritten = 0
    for path in _job_artifact_paths(job):
        if not path.exists() or not path.is_file():
            continue
        try:
            original = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        updated, count = _replace_speaker_labels(original, mapping)
        if count:
            path.write_text(updated, encoding="utf-8")
            rewritten += 1
    return rewritten


def library_rename_speakers(job_id: int, mapping: dict[str, str]) -> dict[str, int]:
    db = database()
    job = db.get_job(job_id)
    if not job:
        raise ValueError(f"job not found: {job_id}")
    segments = db.get_segments(job_id)
    segments_changed = 0
    updated = []
    for segment in segments:
        segment = dict(segment)
        speaker = segment.get("speaker")
        if speaker in mapping:
            segment["speaker"] = mapping[speaker]
            segments_changed += 1
        segment["start"] = segment.get("start", segment.get("start_time"))
        segment["end"] = segment.get("end", segment.get("end_time"))
        updated.append(segment)
    if segments_changed:
        db.add_segments(job_id, updated)

    current_speakers: dict[str, str] = {}
    raw_speakers = job.get("speaker_map_json")
    if raw_speakers:
        try:
            current_speakers = json.loads(raw_speakers)
        except json.JSONDecodeError:
            current_speakers = {}
    current_speakers.update({k: v for k, v in mapping.items() if v.strip()})
    db.update_job_context(job_id, speakers=current_speakers)

    artifacts_rewritten = _rewrite_speaker_artifacts(job, mapping)
    return {
        "segments_changed": segments_changed,
        "artifacts_rewritten": artifacts_rewritten,
    }
