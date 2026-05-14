from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
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


def _safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("._") or "speaker"


def _bundled_ffmpeg_path() -> str:
    executable = Path(sys.executable).resolve()
    candidates = [
        executable.parent.parent / "bin" / "ffmpeg",
        Path(__file__).resolve().parent.parent / "bin" / "ffmpeg",
        Path.cwd() / "bin" / "ffmpeg",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return str(candidate)
    return shutil.which("ffmpeg") or "ffmpeg"


def _sample_audio_source(job: dict[str, Any]) -> Path | None:
    workspace = Path(job.get("workspace_dir") or "")
    candidates = [
        workspace / "audio.wav",
    ]
    source = job.get("source_path") or ""
    if source and workspace:
        candidates.append(workspace / Path(source).name)
    if source:
        candidates.append(Path(source))
    compressed = job.get("compressed_path") or ""
    if compressed:
        candidates.append(Path(compressed))
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def library_speaker_samples(job_id: int, seconds: float = 8.0) -> list[dict[str, Any]]:
    db = database()
    job = db.get_job(job_id)
    if not job:
        raise ValueError(f"job not found: {job_id}")
    source = _sample_audio_source(job)
    if source is None:
        return []

    segments_by_speaker: dict[str, list[dict[str, Any]]] = {}
    for segment in db.get_segments(job_id):
        speaker = str(segment.get("speaker") or "").strip()
        if not speaker:
            continue
        segments_by_speaker.setdefault(speaker, []).append(segment)

    sample_dir = Path(job.get("workspace_dir") or source.parent) / "speaker_samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg = _bundled_ffmpeg_path()
    samples: list[dict[str, Any]] = []
    for speaker, segments in sorted(segments_by_speaker.items()):
        segment = max(
            segments,
            key=lambda item: float(item.get("end_time") or 0) - float(item.get("start_time") or 0),
        )
        start = max(float(segment.get("start_time") or 0) - 0.2, 0)
        end = float(segment.get("end_time") or start + seconds)
        duration = max(min(seconds, end - start + 0.4), 1.0)
        out_path = sample_dir / f"{_safe_filename(speaker)}.wav"
        if not out_path.exists():
            cmd = [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{start:.3f}",
                "-i",
                str(source),
                "-t",
                f"{duration:.3f}",
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                str(out_path),
            ]
            subprocess.run(cmd, capture_output=True, text=True, check=False)
        if out_path.exists():
            samples.append(
                {
                    "speaker": speaker,
                    "path": str(out_path),
                    "start": start,
                    "duration": duration,
                }
            )
    return samples


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
