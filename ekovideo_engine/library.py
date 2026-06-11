from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from database_manager import DatabaseManager
from speaker_recognition import (
    DEFAULT_MATCH_THRESHOLD,
    aggregate_embeddings,
    decode_embedding,
    encode_embedding,
    match_cluster_against_profiles,
    merge_centroids,
    merge_into_existing_centroid,
)
from transcription_utils import (
    build_embedding_extract_cmd,
    parse_embedding_output,
)

from .logging import append_app_log
from .paths import library_db_path, managed_venv_python_path


def database(path: str | Path | None = None) -> DatabaseManager:
    db_path = Path(path) if path else library_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return DatabaseManager(db_path)


def library_list(limit: int = 1000, status: str | None = None) -> list[dict[str, Any]]:
    return database().list_jobs(limit=limit, status=status)


def library_get(job_id: int) -> dict[str, Any] | None:
    """Return a single library row, or ``None`` if the job is gone.

    Added for the SwiftUI ``refreshOne`` path: rather than refetching
    the entire ``library_list`` (~1000 rows by default) every time the
    user saves the rename sheet, the app calls this for the specific
    row that changed and patches it into ``LibraryStore.rows`` in
    place. Cuts the post-Enregistrer beat from "noticeably laggy" to
    sub-100 ms on a warm DB.
    """
    return database().get_job(job_id)


def library_detach_odoo_meeting(job_id: int) -> dict[str, Any]:
    """Clear ``odoo_meeting_json`` on a job.

    Used by the library's hidden "Réunion Odoo" column to let the
    user break the link without going through a full re-run. Returns
    a small audit dict the CLI echoes back so SwiftUI knows the
    write succeeded.
    """
    db = database()
    db.update_job_odoo_meeting(job_id, None)
    return {"job_id": job_id, "detached": True}


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


def library_free_source(job_id: int) -> dict[str, Any]:
    """PR AP — delete a job's heavy source file(s) to reclaim disk,
    keeping the compressed version + the transcripts.

    Hard precondition (per the product rule): we ONLY free the
    source when a compressed version exists on disk. Compressing
    is the lossy substitute that makes the original safe to drop;
    without it, freeing the source would lose the only media for
    that meeting. When the precondition isn't met we no-op and
    return ``{"freed": False, "reason": "..."}``.

    What gets deleted:
      * the workspace copy ``<workspace>/<source basename>``
      * the original ``source_path`` file if it still exists

    What is preserved: the compressed file, the transcripts, the
    review markdown, everything else in the workspace. After this,
    the compressed file is the only media left — the SwiftUI side
    surfaces "relancer" on the compressed artefact (transcription
    only, since re-compression is impossible with no source).

    Never deletes the compressed file itself: its basename
    (``X_compressed.mp4`` / ``.m4a``) differs from the source's
    (``X.mov``), and we additionally guard on path-equality.

    Returns an audit dict the SwiftUI layer shows as
    "Source libérée · 1,2 Go récupérés".
    """
    db = database()
    summary: dict[str, Any] = {
        "job_id": job_id,
        "freed": False,
        "reason": "",
        "files_removed": 0,
        "bytes_removed": 0,
    }
    job = db.get_job(job_id)
    if not job:
        summary["reason"] = "job_not_found"
        return summary

    compressed = (job.get("compressed_path") or "").strip()
    if not compressed or not Path(compressed).expanduser().exists():
        # The whole safety contract: no compressed substitute → refuse.
        summary["reason"] = "no_compressed_version"
        return summary
    compressed_resolved = Path(compressed).expanduser().resolve()

    source_path = (job.get("source_path") or "").strip()
    workspace = (job.get("workspace_dir") or "").strip()

    # Build the candidate set: original source + workspace copy.
    candidates: list[Path] = []
    if source_path:
        candidates.append(Path(source_path).expanduser())
        if workspace:
            candidates.append(
                Path(workspace).expanduser() / Path(source_path).name
            )

    files_removed = 0
    bytes_removed = 0
    seen: set[str] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        # Never delete the compressed file, whatever the DB says.
        if resolved == compressed_resolved:
            continue
        if not candidate.exists() or not candidate.is_file():
            continue
        try:
            size = candidate.stat().st_size
        except OSError:
            size = 0
        try:
            candidate.unlink()
        except OSError as exc:
            append_app_log(
                f"engine_free_source_unlink_failed job_id={job_id} "
                f"path={str(candidate)!r} error={exc!r}"
            )
            continue
        files_removed += 1
        bytes_removed += size
        append_app_log(
            f"engine_free_source_removed job_id={job_id} "
            f"path={str(candidate)!r} bytes={size}"
        )

    summary["freed"] = files_removed > 0
    summary["files_removed"] = files_removed
    summary["bytes_removed"] = bytes_removed
    if files_removed == 0:
        summary["reason"] = "no_source_file_on_disk"

    # PR AS — the library's "Poids" column reads ``jobs.total_bytes``,
    # a snapshot taken once at job completion (see
    # runner._snapshot_workspace_size). Freeing the source shrinks the
    # workspace but left that snapshot stale, so the weight never
    # dropped. Re-walk the workspace now and persist the new total so
    # the column updates on the next list refresh. We also hand it back
    # in the summary so the SwiftUI side can update the row optimistically
    # (no full reload latency).
    if summary["freed"] and workspace:
        workspace_path = Path(workspace).expanduser()
        if workspace_path.is_dir():
            new_total = _directory_total_bytes(workspace_path)
            db.update_job_total_bytes(job_id, new_total)
            summary["total_bytes"] = new_total

    append_app_log(
        f"engine_free_source job_id={job_id} freed={summary['freed']} "
        f"files={files_removed} bytes={bytes_removed} "
        f"total_bytes={summary.get('total_bytes')}"
    )
    return summary


def library_recompute_total_bytes() -> dict[str, Any]:
    """PR AY — re-walk every job workspace and refresh the stored
    ``total_bytes`` snapshot.

    The "Poids" column reads a snapshot taken at job completion;
    PR AS refreshes it when "Libérer la source" runs — but sources
    freed on app versions BEFORE that fix left snapshots frozen at
    the pre-deletion size (the user's job showed 21,5 Go for a
    950 Mo workspace). One-shot heal pass driven by the SwiftUI
    launch sequence; cheap enough to re-run (a directory walk per
    job, ~dozens of folders).

    Jobs whose workspace no longer exists are skipped — NULLing the
    column would downgrade legacy rows to "—" for no benefit.
    """
    db = database()
    updated = 0
    skipped = 0
    for row in db.list_jobs(limit=10_000):
        workspace = (row.get("workspace_dir") or "").strip()
        if not workspace:
            skipped += 1
            continue
        workspace_path = Path(workspace).expanduser()
        if not workspace_path.is_dir():
            skipped += 1
            continue
        new_total = _directory_total_bytes(workspace_path)
        if new_total != (row.get("total_bytes") or 0):
            db.update_job_total_bytes(row["id"], new_total)
            updated += 1
    append_app_log(
        f"engine_recompute_total_bytes updated={updated} skipped={skipped}"
    )
    return {"updated": updated, "skipped": skipped}


def _directory_total_bytes(directory: Path) -> int:
    """Cumulative size of every regular file under ``directory``.

    Mirrors ``runner._snapshot_workspace_size`` (kept separate to
    avoid importing the heavy runner module here). Symlinks and
    unreadable entries are skipped rather than raising.
    """
    total = 0
    for child in directory.rglob("*"):
        try:
            if child.is_file() and not child.is_symlink():
                total += child.stat().st_size
        except OSError:
            continue
    return total


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


def _segment_start(segment: dict[str, Any]) -> float:
    return float(segment.get("start_time") or segment.get("start") or 0)


def _segment_end(segment: dict[str, Any]) -> float:
    return float(segment.get("end_time") or segment.get("end") or 0)


def _segment_duration(segment: dict[str, Any]) -> float:
    return max(0.0, _segment_end(segment) - _segment_start(segment))


def _speaker_stats(segments: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    stats: dict[str, dict[str, float | int]] = {}
    for segment in segments:
        speaker = str(segment.get("speaker") or "").strip()
        if not speaker:
            continue
        bucket = stats.setdefault(speaker, {"utterance_count": 0, "total_duration": 0.0})
        bucket["utterance_count"] = int(bucket["utterance_count"]) + 1
        bucket["total_duration"] = float(bucket["total_duration"]) + _segment_duration(segment)
    return stats


def _sample_isolation_score(
    segment: dict[str, Any],
    all_segments: list[dict[str, Any]],
) -> float:
    """Prefer long turns with silence or same-speaker context around them.

    Diarisation can be noisy: padding a clip into a neighbour from
    another speaker is exactly what makes the UX confusing. This score
    pushes isolated turns to the top and demotes segments that butt up
    against another speaker.
    """

    start = _segment_start(segment)
    end = _segment_end(segment)
    speaker = str(segment.get("speaker") or "")
    duration = max(0.0, end - start)
    nearest_other_gap = 30.0
    overlap_penalty = 0.0
    for other in all_segments:
        if other is segment:
            continue
        other_speaker = str(other.get("speaker") or "")
        if other_speaker == speaker:
            continue
        other_start = _segment_start(other)
        other_end = _segment_end(other)
        if other_start < end and other_end > start:
            overlap_penalty += 10.0
            nearest_other_gap = 0.0
        elif other_end <= start:
            nearest_other_gap = min(nearest_other_gap, start - other_end)
        elif other_start >= end:
            nearest_other_gap = min(nearest_other_gap, other_start - end)
    text_bonus = min(len(str(segment.get("text") or "")) / 120.0, 1.0)
    gap_bonus = min(nearest_other_gap, 3.0)
    return duration + gap_bonus + text_bonus - overlap_penalty


def _clip_window(segment: dict[str, Any], seconds: float) -> tuple[float, float]:
    """Pick a ``seconds``-long window inside the segment for sample
    extraction.

    Short segments (≤ ``seconds``) → use the whole thing.
    Long segments (> ``seconds``) → pick a window CENTERED on the
    segment instead of starting at the beginning. The beginning of
    a long segment is most often a boundary transition (Whisper's
    30 s window crossing a speaker change) where audio is more
    likely to contain the previous speaker; the middle is more
    likely to be the target speaker's central speech.
    """
    start = _segment_start(segment)
    end = _segment_end(segment)
    duration = max(0.0, end - start)
    if duration <= 0:
        return start, 0.0
    trim = 0.05 if duration > 1.2 else 0.0
    if duration <= seconds + 2 * trim:
        # Short enough to use whole — just trim the edges.
        clip_start = max(start + trim, 0)
        clip_duration = max(min(seconds, duration - (trim * 2)), min(duration, 1.0))
        return clip_start, clip_duration
    # Long segment: centre the clip window. Pulls the sample
    # away from likely-boundary edges.
    midpoint = start + duration / 2.0
    half_clip = seconds / 2.0
    clip_start = max(0.0, midpoint - half_clip)
    return clip_start, seconds


def library_speaker_samples(
    job_id: int,
    seconds: float = 8.0,
    per_speaker: int = 3,
) -> list[dict[str, Any]]:
    db = database()
    job = db.get_job(job_id)
    if not job:
        raise ValueError(f"job not found: {job_id}")
    source = _sample_audio_source(job)
    if source is None:
        return []

    all_segments = db.get_segments(job_id)
    stats = _speaker_stats(all_segments)
    segments_by_speaker: dict[str, list[dict[str, Any]]] = {}
    for segment in all_segments:
        speaker = str(segment.get("speaker") or "").strip()
        if not speaker:
            continue
        segments_by_speaker.setdefault(speaker, []).append(segment)

    sample_dir = Path(job.get("workspace_dir") or source.parent) / "speaker_samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg = _bundled_ffmpeg_path()
    samples: list[dict[str, Any]] = []
    for speaker, segments in sorted(segments_by_speaker.items()):
        candidates = sorted(
            (segment for segment in segments if _segment_duration(segment) >= 1.0),
            key=lambda item: _sample_isolation_score(item, all_segments),
            reverse=True,
        )
        if not candidates and segments:
            candidates = sorted(segments, key=_segment_duration, reverse=True)
        for index, segment in enumerate(candidates[: max(1, per_speaker)], start=1):
            start, duration = _clip_window(segment, seconds)
            if duration <= 0:
                continue
            out_path = sample_dir / f"{_safe_filename(speaker)}_{index}_{start:.2f}.wav"
            # Treat truncated / zero-byte WAVs as "needs re-extraction".
            # A 16 kHz mono WAV under ~2 KB is essentially empty —
            # ffmpeg failed silently on a previous run. The old check
            # ``not out_path.exists()`` accepted those files as-is and
            # the rename sheet ended up with empty audio samples.
            needs_extract = True
            if out_path.exists():
                try:
                    needs_extract = out_path.stat().st_size < 2048
                except OSError:
                    needs_extract = True
            if needs_extract:
                try:
                    out_path.unlink()
                except OSError:
                    pass
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
            # Skip the entry entirely if extraction left us with a
            # tiny/missing file — the UI would show an "empty" play
            # button otherwise.
            try:
                final_size = out_path.stat().st_size if out_path.exists() else 0
            except OSError:
                final_size = 0
            if final_size < 2048:
                continue
            if out_path.exists():
                speaker_stats = stats.get(speaker, {"utterance_count": 0, "total_duration": 0.0})
                samples.append(
                    {
                        "speaker": speaker,
                        "path": str(out_path),
                        "start": start,
                        "duration": duration,
                        "index": index,
                        "utterance_count": int(speaker_stats["utterance_count"]),
                        "total_duration": float(speaker_stats["total_duration"]),
                        "text": str(segment.get("text") or ""),
                    }
                )
    return samples


def library_flag_speaker_sample_review(
    job_id: int,
    *,
    speaker: str,
    start: float,
    duration: float,
    note: str = "",
) -> dict[str, Any]:
    """Persist a user signal that one speaker sample needs attention.

    The current pipeline cannot yet re-diarise only a 7-second span from
    the UI. Persisting a structured marker gives the next rerun a clear
    target and keeps the UX honest: the user can queue the meeting
    again while we retain exactly which passage sounded mixed.
    """

    db = database()
    job = db.get_job(job_id)
    if not job:
        raise ValueError(f"job not found: {job_id}")
    workspace_raw = (job.get("workspace_dir") or "").strip()
    if not workspace_raw:
        raise ValueError(f"job has no workspace: {job_id}")
    workspace = Path(workspace_raw)
    workspace.mkdir(parents=True, exist_ok=True)
    marker_path = workspace / "speaker_review_requests.jsonl"
    payload = {
        "job_id": job_id,
        "speaker": speaker,
        "start": float(start),
        "duration": float(duration),
        "note": note.strip() or "Extrait d'interlocuteur à revoir",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_path": job.get("source_path") or "",
    }
    with marker_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    return {
        "job_id": job_id,
        "review_path": str(marker_path),
        "source_path": job.get("source_path") or "",
        "copied_source_path": str(workspace / Path(job.get("source_path") or "").name)
        if job.get("source_path")
        else "",
        "marked": True,
    }


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


def library_rename_speakers(
    job_id: int,
    mapping: dict[str, str],
    *,
    enroll: bool = True,
    attendee_partner_map: dict[str, dict[str, Any]] | None = None,
) -> dict[str, int]:
    """Rename speakers in DB segments + artefacts and (optionally)
    enroll the renamed clusters as voice profiles.

    The enrollment loop fires when ``enroll=True`` (the default) and
    the placeholder being renamed actually looks like a SPEAKER_NN
    cluster from pyannote — the user telling us "SPEAKER_00 is
    Robin" is a strong signal we should remember Robin's voice for
    next time. We silently skip enrollment when the clustering audio
    isn't on disk anymore (old job whose workspace was wiped) or
    when the embedding venv isn't reachable.

    ``attendee_partner_map`` (optional) carries the Odoo attendee
    metadata the SwiftUI rename sheet already has on hand for a
    meeting-linked job. Shape: ``{name_lowercase: {partner_id,
    partner_name, company_id?, company_name?}}``. When a renamed
    speaker matches one of these names, the resulting voice profile
    is auto-linked to the corresponding ``res.partner`` — the user
    doesn't have to step into the Interlocuteurs tab and pair them
    by hand. We only link profiles that aren't already linked to
    *some* partner, to preserve a manual override.
    """
    db = database()
    job = db.get_job(job_id)
    if not job:
        raise ValueError(f"job not found: {job_id}")

    # Enrollment runs *before* the DB segments get renamed —
    # otherwise the labels we want to look up are already gone
    # from the segments table.
    #
    # We accept any "source label → real name" pair, not just
    # ``SPEAKER_NN → name``. The previous filter rejected the
    # common case where the LLM's title pass had already replaced
    # the placeholder with a wrong guess ("SPEAKER_00" → "Marie"
    # in the pipeline, then the user fixes "Marie" → "Sophie" in
    # the rename sheet). That second rename never enrolled
    # anything because "Marie" doesn't start with "SPEAKER_", so
    # the speaker_profiles table stayed empty across the user's
    # whole library.
    #
    # The only no-op case we still skip is ``X → X``: the user
    # confirmed the existing label is the right name (typically
    # via the auto-recognition pre-fill), which means the profile
    # is already enrolled.
    enrolled = 0
    if enroll:
        enrolment_targets: dict[str, str] = {
            source: name
            for source, name in mapping.items()
            if name.strip()
            and source.strip()
            and source.strip().lower() != name.strip().lower()
        }
        if enrolment_targets:
            enrolled = library_enroll_speakers_for_job(
                job_id, enrolment_targets, db=db, job=job
            )
    remembered = library_remember_speaker_names(
        [name for name in mapping.values() if (name or "").strip()],
        db=db,
    )

    # Auto-link the profiles created above (or pre-existing) to the
    # Odoo res.partner records carried by the meeting attendees.
    # Runs after ``remember`` so the lookup-by-name path always finds
    # a row, even when enrollment failed (no diarisation audio etc.).
    linked_to_odoo = 0
    if attendee_partner_map:
        for raw_name in mapping.values():
            name = str(raw_name or "").strip()
            if not name:
                continue
            attendee = attendee_partner_map.get(name.lower())
            if not attendee:
                continue
            partner_id = int(attendee.get("partner_id") or 0)
            if partner_id <= 0:
                continue
            profile = db.get_speaker_profile_by_name(name)
            if not profile:
                continue
            existing_partner = int(profile.get("odoo_partner_id") or 0)
            if existing_partner == partner_id:
                # Already linked to this exact partner — nothing to do.
                continue
            if existing_partner > 0:
                # Already linked to a *different* partner. Respect the
                # user's manual choice and don't silently overwrite.
                continue
            db.link_speaker_profile_to_odoo(
                int(profile["id"]),
                partner_id=partner_id,
                partner_name=str(attendee.get("partner_name") or name),
                company_id=int(attendee.get("company_id") or 0) or None,
                company_name=str(attendee.get("company_name") or ""),
            )
            linked_to_odoo += 1

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

    # Rebuild the speaker map *canonically* from whatever ended up
    # in the segments table — never merge historical rename pairs.
    # The previous behaviour (``current_speakers.update(mapping)``)
    # kept stale ``Marie → Sophie`` rows after Marie was already
    # rewritten in segments, which caused the rename sheet to
    # re-show "Sophie" twice on reopen (once from the historical
    # pair, once from the actual segment label). Treating the
    # segments table as the source of truth makes the map idempotent.
    db.update_job_context(
        job_id, speakers=_speaker_map_from_segments(db.get_segments(job_id))
    )

    artifacts_rewritten = _rewrite_speaker_artifacts(job, mapping)

    return {
        "segments_changed": segments_changed,
        "artifacts_rewritten": artifacts_rewritten,
        "speakers_enrolled": enrolled,
        "speakers_remembered": remembered,
        "speakers_linked_to_odoo": linked_to_odoo,
    }


def library_repair_all_speaker_maps() -> dict[str, int]:
    """Heal every library job's ``speaker_map_json`` from segments.

    PR #30 fixed the source of the drift (the rename sheet's
    ``updateContext(speakers=…)`` clobber over the canonical rebuild),
    but jobs that ran before the fix still carry the stale map in DB
    — the rename sheet now masks the duplicates visually, but the
    ``displayed_speaker_names`` accessor and the runner's history
    pull from the persisted map. Without this one-shot pass the user
    would have to re-Enregistrer every old job by hand to clean the
    column.

    Rules:

    * Skip jobs whose ``transcription_segments`` table is empty —
      they predate the persistence fix and have no source of truth
      to rebuild from. Leaving the existing map alone is safer than
      replacing it with ``{}``.
    * Skip jobs whose canonical rebuild equals what's already in
      DB (avoids needless writes / updated_at churn).
    * Always returns an audit dict so the caller can surface "5
      lignes nettoyées" if it wants to.
    """
    db = database()
    checked = 0
    repaired = 0
    skipped_no_segments = 0
    unchanged = 0
    for row in db.list_jobs(limit=10_000):
        checked += 1
        job_id = int(row.get("id") or 0)
        if job_id <= 0:
            continue
        segments = db.get_segments(job_id)
        if not segments:
            skipped_no_segments += 1
            continue
        canonical = _speaker_map_from_segments(segments)
        raw_existing = row.get("speaker_map_json") or ""
        try:
            existing = json.loads(raw_existing) if raw_existing else {}
        except (TypeError, ValueError):
            existing = {}
        if existing == canonical:
            unchanged += 1
            continue
        db.update_job_context(job_id, speakers=canonical)
        repaired += 1
    return {
        "checked": checked,
        "repaired": repaired,
        "skipped_no_segments": skipped_no_segments,
        "unchanged": unchanged,
    }


def _speaker_map_from_segments(segments: list[dict]) -> dict[str, str]:
    """Project the current ``transcription_segments.speaker`` column
    into the ``speaker_map_json`` shape the rename sheet consumes.

    Rules — kept identical to ``TranscriptionPipeline._build_speaker_map``
    so the post-rename map and the post-pipeline map are
    indistinguishable:

    * SPEAKER_NN placeholders map to an empty string (the sheet
      treats that as "user hasn't decided yet" and shows an
      editable field).
    * Friendly names map to themselves so the sheet can show the
      current display name pre-filled.
    * Empty / blank labels are skipped — they're a remnant of an
      unattributed segment, not a speaker to surface.
    """
    labels: list[str] = []
    seen: set[str] = set()
    for seg in segments:
        label = str(seg.get("speaker") or "").strip()
        if not label or label in seen:
            continue
        seen.add(label)
        labels.append(label)
    out: dict[str, str] = {}
    for label in labels:
        if label.upper().startswith("SPEAKER_"):
            out[label] = ""
        else:
            out[label] = label
    return out


# ---------------------------------------------------------------------------
# Speaker enrollment / recognition
# ---------------------------------------------------------------------------


# Up to this many of the longest turns per cluster get fed to the
# embedding model. More turns means a more stable centroid; past 3
# the marginal value drops off and the inference cost adds up.
_EMBEDDING_TURNS_PER_CLUSTER = 3
# Minimum turn duration we'll bother sending to the embedding model.
# Pyannote produces clean vectors from ~2 s of speech; anything
# shorter is mostly noise floor and pollutes the centroid.
_EMBEDDING_MIN_TURN_SECONDS = 2.0


def _venv_python(job: dict[str, Any] | None = None) -> str | None:
    """Resolve the venv python that runs pyannote.

    Reads ``settings_json.transcription_settings.venv_python_path``
    off the job row. Returns ``None`` when the path is missing or
    doesn't exist on disk — the caller then quietly skips the
    enrollment / recognition step.
    """
    if not job:
        return None
    settings_raw = job.get("settings_json") or "{}"
    try:
        settings = json.loads(settings_raw)
    except json.JSONDecodeError:
        return None
    transcription = settings.get("transcription_settings") or {}
    candidate = (transcription.get("venv_python_path") or "").strip()
    if candidate and Path(candidate).exists():
        return candidate
    managed_candidate = managed_venv_python_path()
    if managed_candidate.exists():
        return str(managed_candidate)
    return None


def library_remember_speaker_names(
    names: list[str] | tuple[str, ...] | set[str],
    *,
    db: DatabaseManager | None = None,
) -> int:
    """Persist user-confirmed speaker names before a voiceprint exists.

    The Interlocuteurs view is both an address book and the voice
    recognition store. A rename previously appeared there only when
    pyannote embedding extraction succeeded; any missing venv or old
    workspace made the save look lost. We now keep a 0-sample row
    immediately, then replace it with a real centroid on enrollment.
    """
    db = db or database()
    written = 0
    seen: set[str] = set()
    for raw_name in names:
        name = str(raw_name or "").strip()
        if not name:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        if db.get_speaker_profile_by_name(name):
            continue
        db.upsert_speaker_profile(name=name, embedding_json="[]", sample_count=0)
        written += 1
    return written


def _diarisation_audio_path(job: dict[str, Any]) -> Path | None:
    """Pick the on-disk audio that gives the cleanest embeddings.

    Preference order:
      1. ``audio.diar.wav`` — extracted without the ASR enhancement
         chain, which is exactly what pyannote was designed against.
      2. ``audio.wav`` — fall back when the diarisation-targeted
         file wasn't produced (job ran on an older engine version).
      3. The original source file — last resort, but pyannote will
         still happily accept it.
    """
    workspace = (job.get("workspace_dir") or "").strip()
    if workspace:
        for name in ("audio.diar.wav", "audio.wav"):
            candidate = Path(workspace) / name
            if candidate.exists():
                return candidate
    source = (job.get("source_path") or "").strip()
    if source:
        candidate = Path(source)
        if candidate.exists():
            return candidate
    return None


# Lower bound of clean speech we try to accumulate per cluster
# before stopping. Hitting this gives a stable centroid; cluster
# without that much speech still gets enrolled, just with fewer
# samples than ideal.
_EMBEDDING_TARGET_SECONDS = 30.0
# Upper bound on individual turns we'll feed to pyannote per
# cluster, even if we haven't hit the target. Each turn costs
# ~0.5 s of inference, and past 5 the centroid stops improving
# materially.
_EMBEDDING_MAX_TURNS = 5
# Two clusters' turns are considered "overlapping" if their
# timestamps interlock by more than this. Shorter overlaps are
# usually back-channel "ouais" / "hm" — harmless for the centroid.
_EMBEDDING_OVERLAP_TOLERANCE_SECONDS = 0.1


def _segments_per_cluster(
    segments: list[dict],
    labels: list[str],
) -> dict[str, list[dict]]:
    """Pick the cleanest turns per cluster for embedding extraction.

    PR J improvements over the previous "top-3 by duration":
    - **Overlap filter**: a turn that interlocks with another
      cluster's turn by > 100 ms is dropped. Otherwise pyannote
      computes a centroid that's polluted by the other speaker's
      timbre — destroys cross-meeting recognition accuracy.
    - **Target accumulation**: stop sampling once we have 30 s of
      clean speech per cluster (stable centroid) rather than
      taking exactly N turns regardless of total duration.
    - **Hard cap**: never feed more than 5 turns to pyannote — the
      centroid stops improving past that point and the inference
      cost adds up on a 10-speaker meeting.
    - Turns shorter than ``_EMBEDDING_MIN_TURN_SECONDS`` (2 s) are
      still dropped as before so back-channels don't poison the
      centroid.
    """
    # Build per-label turn lists with their numeric bounds.
    by_label_raw: dict[str, list[dict]] = {label: [] for label in labels}
    # Plus a flat list of ``(start, end, owner_label)`` for the
    # overlap check across all labels (NOT just the enrolment ones).
    flat_segments: list[tuple[float, float, str]] = []
    for segment in segments:
        label = (segment.get("speaker") or "").strip()
        try:
            start = float(segment.get("start_time") or segment.get("start") or 0)
            end = float(segment.get("end_time") or segment.get("end") or 0)
        except (TypeError, ValueError):
            continue
        if end <= start:
            continue
        flat_segments.append((start, end, label))
        if label not in by_label_raw:
            continue
        duration = end - start
        if duration < _EMBEDDING_MIN_TURN_SECONDS:
            continue
        by_label_raw[label].append({"start": start, "end": end, "duration": duration})

    def _overlaps_another_speaker(start: float, end: float, owner: str) -> bool:
        """True if any other-cluster segment interlocks by more
        than the tolerance with ``[start, end]``."""
        for s, e, label in flat_segments:
            if label == owner:
                continue
            overlap = min(end, e) - max(start, s)
            if overlap > _EMBEDDING_OVERLAP_TOLERANCE_SECONDS:
                return True
        return False

    out: dict[str, list[dict]] = {}
    for label, turns in by_label_raw.items():
        if not turns:
            continue
        clean = [
            t
            for t in turns
            if not _overlaps_another_speaker(t["start"], t["end"], label)
        ]
        # Fall back to the original turns if overlap filter wiped
        # everything — a single contaminated centroid is still
        # better than no centroid at all (the user can fix it later
        # by enroling on a cleaner job).
        candidates = clean or turns
        candidates.sort(key=lambda t: t["duration"], reverse=True)
        picked: list[dict] = []
        accumulated = 0.0
        for t in candidates:
            if len(picked) >= _EMBEDDING_MAX_TURNS:
                break
            picked.append({"start": t["start"], "end": t["end"]})
            accumulated += t["duration"]
            if accumulated >= _EMBEDDING_TARGET_SECONDS and len(picked) >= 2:
                break
        if picked:
            out[label] = picked
    return out


def _extract_cluster_embeddings(
    venv_python: str,
    audio_path: Path,
    cluster_segments: dict[str, list[dict]],
) -> dict[str, list[list[float]]]:
    """Shell out to pyannote-embedding for the requested clusters.

    Returns a mapping ``{label: [vector, ...]}``. Empty mapping on
    any failure — the caller treats that as "skip recognition for
    this job" and surfaces a warning rather than crashing.
    """
    if not cluster_segments:
        return {}
    payload = [
        {"label": label, "segments": segments}
        for label, segments in cluster_segments.items()
    ]
    cmd = build_embedding_extract_cmd(venv_python, str(audio_path), payload)
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if proc.returncode != 0:
        return {}
    try:
        return parse_embedding_output(proc.stdout)
    except RuntimeError:
        return {}


def library_enroll_speakers_for_job(
    job_id: int,
    enrolment_mapping: dict[str, str],
    *,
    db: DatabaseManager | None = None,
    job: dict[str, Any] | None = None,
) -> int:
    """Extract embeddings for the SPEAKER_NN clusters in ``enrolment_mapping``
    and persist them under the friendly names.

    ``enrolment_mapping`` is ``{SPEAKER_00: "Robin", SPEAKER_01: "Marie"}``.
    Returns the number of profiles successfully written.

    The function is intentionally lenient: it skips silently when
    the venv isn't reachable, the audio file is missing, or pyannote
    fails. Speaker rename remains valuable even when enrollment
    couldn't run.
    """
    db = db or database()
    job = job or db.get_job(job_id)
    if not job:
        return 0
    venv_python = _venv_python(job)
    if not venv_python:
        return 0
    audio_path = _diarisation_audio_path(job)
    if audio_path is None:
        return 0

    segments = db.get_segments(job_id)
    cluster_segments = _segments_per_cluster(
        segments, list(enrolment_mapping.keys())
    )
    if not cluster_segments:
        return 0

    embeddings = _extract_cluster_embeddings(venv_python, audio_path, cluster_segments)
    if not embeddings:
        return 0

    written = 0
    for placeholder, name in enrolment_mapping.items():
        vectors = embeddings.get(placeholder) or []
        if not vectors:
            continue
        existing = db.get_speaker_profile_by_name(name)
        if existing:
            existing_centroid = decode_embedding(existing.get("embedding_json") or "")
            existing_count = int(existing.get("sample_count") or 0)
            centroid, sample_count = merge_into_existing_centroid(
                existing_centroid=existing_centroid,
                existing_count=existing_count,
                new_embeddings=vectors,
            )
        else:
            centroid = aggregate_embeddings(vectors)
            sample_count = len(vectors)
        if not centroid:
            continue
        db.upsert_speaker_profile(
            name=name,
            embedding_json=encode_embedding(centroid),
            sample_count=sample_count,
        )
        written += 1
    return written


def library_recognize_speakers(
    job_id: int,
    *,
    threshold: float = DEFAULT_MATCH_THRESHOLD,
) -> dict[str, str]:
    """Match each SPEAKER_NN cluster against the profile store.

    Returns a mapping ``{SPEAKER_00: "Robin", ...}`` containing only
    the labels that crossed the cosine threshold. Used at pipeline
    completion to pre-fill the speaker map so the user opens the
    rename sheet to a recording where their team is already named
    correctly.
    """
    db = database()
    job = db.get_job(job_id)
    if not job:
        return {}
    profiles = db.list_speaker_profiles()
    if not profiles:
        return {}
    venv_python = _venv_python(job)
    if not venv_python:
        return {}
    audio_path = _diarisation_audio_path(job)
    if audio_path is None:
        return {}

    segments = db.get_segments(job_id)
    cluster_labels = sorted(
        {
            (segment.get("speaker") or "").strip()
            for segment in segments
            if (segment.get("speaker") or "").strip().upper().startswith("SPEAKER_")
        }
    )
    if not cluster_labels:
        return {}

    cluster_segments = _segments_per_cluster(segments, cluster_labels)
    if not cluster_segments:
        return {}

    embeddings = _extract_cluster_embeddings(venv_python, audio_path, cluster_segments)
    if not embeddings:
        return {}

    out: dict[str, str] = {}
    used_names: set[str] = set()
    for label in cluster_labels:
        vectors = embeddings.get(label) or []
        if not vectors:
            continue
        centroid = aggregate_embeddings(vectors)
        match = match_cluster_against_profiles(
            centroid, profiles, threshold=threshold
        )
        if not match or not match.profile_name:
            continue
        # Don't assign the same profile to two clusters in the same
        # meeting — pick the highest-confidence cluster for each
        # name. A simple "first wins" works because we processed
        # cluster_labels in sorted order; for a stricter rule we'd
        # need a Hungarian assignment, overkill for now.
        if match.profile_name in used_names:
            continue
        used_names.add(match.profile_name)
        out[label] = match.profile_name
    return out


def library_list_speaker_profiles() -> list[dict[str, Any]]:
    """Surface the stored profiles for the SwiftUI settings panel.

    The embedding blob is intentionally trimmed off the response —
    it's a 5 KB JSON array per row, useless to the UI, and noisy in
    the JSONL transport.
    """
    out: list[dict[str, Any]] = []
    for row in database().list_speaker_profiles():
        clean = dict(row)
        clean.pop("embedding_json", None)
        out.append(clean)
    return out


def library_link_speaker_profile_to_odoo(
    profile_id: int,
    *,
    partner_id: int,
    partner_name: str,
    company_id: int | None = None,
    company_name: str = "",
) -> dict[str, Any]:
    """Pair a local voice profile with an Odoo ``res.partner``.

    Returns the updated profile (without the embedding blob) so the
    SwiftUI side can refresh its in-memory list without a separate
    list-all round-trip.
    """
    db = database()
    db.link_speaker_profile_to_odoo(
        int(profile_id),
        partner_id=int(partner_id),
        partner_name=partner_name,
        company_id=company_id,
        company_name=company_name,
    )
    for row in db.list_speaker_profiles():
        if int(row.get("id") or 0) == int(profile_id):
            clean = dict(row)
            clean.pop("embedding_json", None)
            return clean
    return {}


def library_unlink_speaker_profile_from_odoo(profile_id: int) -> dict[str, Any]:
    db = database()
    db.unlink_speaker_profile_from_odoo(int(profile_id))
    for row in db.list_speaker_profiles():
        if int(row.get("id") or 0) == int(profile_id):
            clean = dict(row)
            clean.pop("embedding_json", None)
            return clean
    return {}


def library_delete_speaker_profile(profile_id: int | None = None, *, name: str | None = None) -> int:
    """Drop a stored profile by ``id`` or ``name``. Returns 1 when
    something was deleted, 0 otherwise."""
    db = database()
    if profile_id is not None:
        db.delete_speaker_profile(int(profile_id))
        return 1
    if name:
        existing = db.get_speaker_profile_by_name(name)
        if not existing:
            return 0
        db.delete_speaker_profile(int(existing["id"]))
        return 1
    return 0


def library_merge_speaker_profiles(
    survivor_id: int,
    absorbed_id: int,
    *,
    odoo_from: str = "survivor",
) -> dict[str, Any]:
    """PR AQ — merge two voice profiles into one.

    The user spotted duplicates (e.g. two "Mathilde Gérard" rows
    that differ only by an accent / Odoo link). This folds the
    ``absorbed`` profile into the ``survivor`` :

      * **Embedding** : weighted-average the two centroids by
        ``sample_count`` (``merge_centroids``), re-normalised. The
        merged profile then represents both voices' evidence.
      * **sample_count** : summed, so future enrolments keep
        accumulating against a profile that "knows" it has heard
        this person N+M times.
      * **Name** : the survivor's name is kept unchanged.
      * **Odoo link** : ``odoo_from`` decides which side's link
        wins — ``"survivor"`` (default) or ``"absorbed"``. The
        SwiftUI layer sets ``"absorbed"`` only when the user
        explicitly picks the absorbed profile's contact in the
        conflict prompt.
      * The absorbed row is deleted.

    Returns an audit dict. ``merged: False`` with a ``reason`` when
    a precondition fails (same id, missing profile).
    """
    summary: dict[str, Any] = {
        "merged": False,
        "reason": "",
        "survivor_id": survivor_id,
        "absorbed_id": absorbed_id,
        "survivor_name": "",
        "sample_count": 0,
    }
    if survivor_id == absorbed_id:
        summary["reason"] = "same_profile"
        return summary

    db = database()
    survivor = db.get_speaker_profile(int(survivor_id))
    absorbed = db.get_speaker_profile(int(absorbed_id))
    if not survivor:
        summary["reason"] = "survivor_not_found"
        return summary
    if not absorbed:
        summary["reason"] = "absorbed_not_found"
        return summary

    centroid_a = decode_embedding(survivor.get("embedding_json") or "")
    centroid_b = decode_embedding(absorbed.get("embedding_json") or "")
    count_a = int(survivor.get("sample_count") or 0)
    count_b = int(absorbed.get("sample_count") or 0)

    merged_centroid, merged_count = merge_centroids(
        centroid_a, count_a, centroid_b, count_b
    )
    merged_json = encode_embedding(merged_centroid) if merged_centroid else "[]"

    db.update_speaker_profile_embedding(
        int(survivor_id), merged_json, merged_count
    )

    # Odoo link resolution. Default keeps the survivor's existing
    # link (no-op). When the user chose the absorbed profile's
    # contact in the conflict prompt, re-point the survivor's link.
    if odoo_from == "absorbed" and absorbed.get("odoo_partner_id"):
        try:
            db.link_speaker_profile_to_odoo(
                int(survivor_id),
                partner_id=int(absorbed["odoo_partner_id"]),
                partner_name=str(absorbed.get("odoo_partner_name") or ""),
                company_id=absorbed.get("odoo_company_id"),
                company_name=str(absorbed.get("odoo_company_name") or ""),
            )
        except Exception as exc:  # pragma: no cover — defensive
            append_app_log(
                f"engine_merge_speakers_odoo_relink_failed "
                f"survivor={survivor_id} error={exc!r}"
            )
    elif (
        odoo_from == "survivor"
        and not survivor.get("odoo_partner_id")
        and absorbed.get("odoo_partner_id")
    ):
        # Survivor had no link but the absorbed did — inherit it so
        # we don't silently drop the only Odoo association. (No
        # conflict here: the survivor was unlinked.)
        try:
            db.link_speaker_profile_to_odoo(
                int(survivor_id),
                partner_id=int(absorbed["odoo_partner_id"]),
                partner_name=str(absorbed.get("odoo_partner_name") or ""),
                company_id=absorbed.get("odoo_company_id"),
                company_name=str(absorbed.get("odoo_company_name") or ""),
            )
        except Exception as exc:  # pragma: no cover — defensive
            append_app_log(
                f"engine_merge_speakers_odoo_inherit_failed "
                f"survivor={survivor_id} error={exc!r}"
            )

    db.delete_speaker_profile(int(absorbed_id))

    summary["merged"] = True
    summary["survivor_name"] = str(survivor.get("name") or "")
    summary["sample_count"] = merged_count
    append_app_log(
        f"engine_merge_speakers survivor={survivor_id} "
        f"absorbed={absorbed_id} merged_count={merged_count} "
        f"odoo_from={odoo_from}"
    )
    return summary


def library_reset_speaker_profiles() -> dict[str, int]:
    """PR X — purge ALL stored voice profiles in one call.

    Used by the SwiftUI "Réinitialiser la library vocale" button
    in Réglages, behind a confirmation modal. The motivating
    failure mode (CVR / Caste runs) was a feedback loop where:
    1. wrong attribution in run N → profile enrolled with wrong
       audio,
    2. run N+1 voice-matches the same wrong cluster against the
       polluted profile and reconfirms,
    3. user can't easily distinguish the cluster has been
       enrolled wrong because the per-profile delete only takes
       one row at a time.

    Returns ``{"removed": N}``. Always succeeds — when the table
    is empty, ``removed=0``. Does NOT touch the ``jobs`` table or
    the per-job speaker overrides; those carry useful history.
    """
    db = database()
    profiles = db.list_speaker_profiles()
    removed = 0
    for row in profiles:
        try:
            profile_id = int(row.get("id") or 0)
        except (TypeError, ValueError):
            continue
        if profile_id <= 0:
            continue
        db.delete_speaker_profile(profile_id)
        removed += 1
    return {"removed": removed}
