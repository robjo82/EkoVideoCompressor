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


def library_delete(job_id: int, *, remove_files: bool = False) -> dict[str, Any]:
    """Remove a job from the library DB and optionally wipe its
    on-disk workspace.

    ``remove_files=False`` (the default, and the legacy behaviour)
    only drops the row — every artefact stays on disk, which is what
    the user wants when they're cleaning up the library view but
    still need the transcripts.

    ``remove_files=True`` additionally deletes the workspace
    directory recursively, freeing the disk space the job
    consumed. We refuse to delete anything that isn't a child of the
    workspace path (so a malformed DB row can't trick us into
    rm -rf'ing the home folder).

    Returns a small audit dict describing what happened: the
    ``workspace_dir`` that was removed (or considered), the number
    of files dropped, and the cumulative byte count. The SwiftUI
    layer surfaces this so the user sees "Économie : 1,2 Go" after
    confirming.
    """
    db = database()
    summary: dict[str, Any] = {
        "job_id": job_id,
        "workspace_dir": "",
        "files_removed": 0,
        "bytes_removed": 0,
        "workspace_removed": False,
    }
    if remove_files:
        job = db.get_job(job_id)
        workspace = (job or {}).get("workspace_dir") or ""
        if workspace:
            summary["workspace_dir"] = workspace
            removed_files, removed_bytes = _safely_remove_workspace(workspace)
            summary["files_removed"] = removed_files
            summary["bytes_removed"] = removed_bytes
            summary["workspace_removed"] = removed_files > 0
    db.delete_job(job_id)
    return summary


def _safely_remove_workspace(workspace_dir: str) -> tuple[int, int]:
    """Delete ``workspace_dir`` recursively and return (file_count,
    bytes_freed).

    The workspace lives under the user-chosen output directory (the
    SwiftUI "Dossier" setting), not under our managed library data
    dir, so we can't enforce a strict ancestor check. Instead we
    apply three permissive but defensive rules:

    1. The path must exist and be a directory.
    2. It must sit at least 3 levels deep (so we never rm a root
       like ``/`` or ``/Users``).
    3. It must contain at least one engine-produced marker — ``audio.wav``,
       ``whisper.json``, a ``- à vérifier.md`` file, or a ``- améliorée``
       transcript. This protects against a malformed DB row pointing
       at, say, ``~/Documents``: the dir exists, sits deep enough,
       but doesn't look like an Eko workspace, so we refuse to touch
       it.

    Any failure returns (0, 0) and leaves disk alone.
    """
    path = Path(workspace_dir).expanduser()
    if not path.exists() or not path.is_dir():
        return (0, 0)
    if len(path.resolve().parts) < 4:
        return (0, 0)
    if not _looks_like_engine_workspace(path):
        return (0, 0)

    file_count = 0
    bytes_freed = 0
    for child in path.rglob("*"):
        try:
            if child.is_file() or child.is_symlink():
                file_count += 1
                try:
                    bytes_freed += child.stat().st_size
                except OSError:
                    pass
        except OSError:
            continue
    shutil.rmtree(path, ignore_errors=True)
    return (file_count, bytes_freed)


def _looks_like_engine_workspace(path: Path) -> bool:
    """Heuristic: refuse to ``rmtree`` something that doesn't carry
    the fingerprints of an engine run.

    The engine writes ``audio.wav`` first thing on every transcription
    job, ``whisper.json`` once Whisper finishes, and a ``- à vérifier.md``
    review file whenever the quality stack ran. Any of those proves
    we're looking at our own folder. ``speaker_samples/`` is another
    Eko-only directory we emit. If none of them exist the folder is
    almost certainly user content we'd rather leave alone.
    """
    if (path / "audio.wav").exists():
        return True
    if (path / "whisper.json").exists():
        return True
    if (path / "speaker_samples").is_dir():
        return True
    for child in path.iterdir():
        name = child.name
        if name.endswith(" - à vérifier.md"):
            return True
        if " améliorée" in name:
            return True
    return False


# Friendly labels shown alongside each file in the deletion sheet. The
# mapping is deliberately small and conservative — anything we don't
# recognise gets surfaced as "Autre" so the user still sees it and
# decides whether the size is worth keeping.
_WORKSPACE_FILE_LABELS: tuple[tuple[str, str], ...] = (
    ("audio.vad.wav", "Audio filtré (VAD)"),
    ("audio.wav", "Audio extrait"),
    ("whisper.json", "Sortie Whisper (JSON)"),
    ("transcript_for_local_analysis.txt", "Transcription pour le LLM"),
    ("speaker_samples", "Extraits audio par locuteur"),
)


def _label_for_workspace_path(path: Path, workspace: Path) -> str:
    name = path.name
    suffix = path.suffix.lower()
    for fragment, label in _WORKSPACE_FILE_LABELS:
        if name == fragment or fragment in path.relative_to(workspace).parts:
            return label
    if suffix in {".txt", ".srt", ".vtt", ".tsv"}:
        return "Transcription"
    if suffix == ".md":
        return "Rapport à vérifier"
    if suffix in {".mov", ".mp4", ".m4v", ".mkv"}:
        return "Vidéo compressée"
    if suffix in {".wav", ".m4a", ".mp3"}:
        return "Audio"
    if suffix == ".json":
        return "Données moteur (JSON)"
    return "Autre"


def library_workspace_usage(job_id: int) -> dict[str, Any]:
    """Describe what would be freed if the workspace got deleted.

    Returns:
      ``workspace_dir`` — the resolved path (empty string when no
      workspace is recorded or it doesn't exist).
      ``files`` — list of {path, name, size, label} dicts, sorted
      by descending size so the user immediately sees the heavy
      hitters (typically ``audio.wav`` or the compressed video).
      ``total_bytes`` — cumulative size of every file we'd remove.

    The SwiftUI deletion sheet uses this to render the "Économie
    réalisée" preview before the user confirms.
    """
    db = database()
    job = db.get_job(job_id)
    out: dict[str, Any] = {
        "workspace_dir": "",
        "files": [],
        "total_bytes": 0,
    }
    if not job:
        return out
    workspace_str = (job.get("workspace_dir") or "").strip()
    if not workspace_str:
        return out
    workspace = Path(workspace_str).expanduser()
    if not workspace.exists() or not workspace.is_dir():
        return out
    out["workspace_dir"] = str(workspace)

    entries: list[dict[str, Any]] = []
    total = 0
    for child in workspace.rglob("*"):
        try:
            if not (child.is_file() or child.is_symlink()):
                continue
            size = child.stat().st_size if child.is_file() else 0
        except OSError:
            continue
        total += size
        entries.append(
            {
                "path": str(child),
                "name": child.name,
                "size": size,
                "label": _label_for_workspace_path(child, workspace),
            }
        )
    entries.sort(key=lambda item: item["size"], reverse=True)
    out["files"] = entries
    out["total_bytes"] = total
    return out


def library_update_context(
    job_id: int,
    speakers: dict[str, str] | None = None,
    technical_terms: list[str] | None = None,
) -> None:
    database().update_job_context(job_id, speakers=speakers, technical_terms=technical_terms)


# Matches "[SPEAKER_00]", "[Robin]", "[Marie Dupont]" — a bracketed
# token at the start of a line followed by either a space or the end
# of the prefix. The renderer never produces any other shape, so
# this is sufficient on every artefact we emit. We still cap the
# inner text at 60 chars so a stray "[note: ...]" inside a paragraph
# can't masquerade as a speaker label.
_SPEAKER_PREFIX_RE = re.compile(r"^\s*\[(?P<label>[^\]\n]{1,60})\]", re.MULTILINE)


def _discover_speakers_from_text(text: str) -> list[str]:
    """Extract speaker labels from a rendered transcript file.

    Walks the text and returns each distinct bracket-prefixed token
    in first-seen order. ``SPEAKER_NN`` placeholders survive verbatim;
    friendly names (``Robin``, ``Marie``) survive too — the caller
    can then decide whether they should be treated as already-named
    or as a placeholder to re-edit.
    """
    seen: list[str] = []
    seen_set: set[str] = set()
    for match in _SPEAKER_PREFIX_RE.finditer(text or ""):
        label = match.group("label").strip()
        if not label or label.lower() in {"speaker", "intervenant"}:
            continue
        if label in seen_set:
            continue
        seen_set.add(label)
        seen.append(label)
    return seen


def library_discover_speakers(job_id: int) -> dict[str, str]:
    """Backfill ``speaker_map_json`` from artefact files.

    The new pipeline persists segments + the speaker map on every
    run, but jobs that completed before that fix have empty DB
    columns. The SwiftUI rename sheet then shows "Aucun
    interlocuteur détecté" — a frustrating dead-end given the
    speakers are literally visible in the transcript file.

    This helper walks every artefact path on disk, extracts the
    bracket-prefixed labels, merges them with whatever ``speaker_map_json``
    already contains, and writes the result back so the sheet has
    something to render.

    Returns the resulting map (placeholder → friendly name, "" when
    the label is still an opaque SPEAKER_NN).
    """
    db = database()
    job = db.get_job(job_id)
    if not job:
        raise ValueError(f"job not found: {job_id}")

    existing_raw = (job.get("speaker_map_json") or "").strip()
    existing: dict[str, str] = {}
    if existing_raw:
        try:
            payload = json.loads(existing_raw)
            if isinstance(payload, dict):
                existing = {str(k): str(v) for k, v in payload.items()}
        except json.JSONDecodeError:
            existing = {}

    discovered: list[str] = []
    for path in _job_artifact_paths(job):
        if not path.exists() or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for label in _discover_speakers_from_text(text):
            if label not in discovered:
                discovered.append(label)

    if not discovered and not existing:
        return {}

    merged: dict[str, str] = dict(existing)
    for label in discovered:
        if label in merged:
            continue
        # SPEAKER_NN placeholders get an empty value so the sheet
        # renders an editable field. Friendly names get themselves
        # so the user sees the current display name and can tweak.
        if label.upper().startswith("SPEAKER_"):
            merged[label] = ""
        else:
            merged[label] = label

    if merged != existing:
        db.update_job_context(job_id, speakers=merged)
    return merged


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
