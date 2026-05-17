from __future__ import annotations

import json
import time
from pathlib import Path

from .events import EventSink
from .logging import append_app_log
from .models import (
    ArtifactEvent,
    DoneEvent,
    EngineEvent,
    ErrorEvent,
    JobRequest,
    ProgressEvent,
    WarningEvent,
)
from .library import database
from .pipeline import CompressionPipeline, StepResult, TranscriptionPipeline
from .pipeline import prepare_job_workspace


class EtaSmoothingSink:
    """Add a conservative, monotonic ETA to progress events.

    The engine progress percentages are step-local, so extrapolating
    from them makes the remaining time jump upward between phases. A
    historical full-job duration gives a less precise but much steadier
    estimate: it only counts down from the initial budget, and disappears
    when no history is available.
    """

    def __init__(self, sink: EventSink, estimated_total_seconds: float | None):
        self.sink = sink
        self.estimated_total_seconds = estimated_total_seconds
        self.started_at = time.monotonic()

    def __call__(self, event: EngineEvent) -> None:
        if (
            isinstance(event, ProgressEvent)
            and event.eta_seconds is None
            and self.estimated_total_seconds
            and (event.pct is None or event.pct < 100)
        ):
            elapsed = time.monotonic() - self.started_at
            eta = max(self.estimated_total_seconds - elapsed, 0)
            event = ProgressEvent(event.step, event.pct, event.message, eta_seconds=eta)
        self.sink(event)


def _historical_total_estimate_seconds(db, request: JobRequest) -> float | None:
    rows = db.list_jobs(limit=30, status="COMPLETED")
    durations: list[float] = []
    for row in rows:
        raw_duration = row.get("duration_total")
        if not raw_duration:
            continue
        try:
            duration = float(raw_duration)
        except (TypeError, ValueError):
            continue
        if duration < 10:
            continue
        raw_settings = row.get("settings_json") or "{}"
        try:
            settings = json.loads(raw_settings)
        except json.JSONDecodeError:
            settings = {}
        if settings.get("mode") and settings.get("mode") != request.mode:
            continue
        durations.append(duration)

    if len(durations) < 2:
        return None

    durations.sort()
    # A mild upper percentile is intentionally conservative: better to
    # count down a little slowly than to hit zero while work is still
    # running. The UI labels it as an estimate.
    index = min(len(durations) - 1, round((len(durations) - 1) * 0.65))
    return durations[index]


def _persist_step_durations(db, job_id: int, results: list[StepResult]) -> None:
    ffmpeg = sum(r.duration_seconds for r in results if r.name == "compression")
    diarization = sum(r.duration_seconds for r in results if r.name == "diarization")
    whisper = sum(
        r.duration_seconds
        for r in results
        if r.name not in {"compression", "diarization"}
    )
    db.update_job_durations(job_id, ffmpeg=ffmpeg, whisper=whisper, diarization=diarization)


def _resolve_source_path(request: JobRequest) -> Path | None:
    """Return the path the runner should actually feed to the
    pipeline, or ``None`` when nothing usable is on disk.

    Three lookup tiers, ordered by trust:

    1. ``request.source_path`` itself if it exists. Covers the
       fresh-run case (user just dropped a file).

    2. The workspace copy at ``request.workspace_dir / basename``.
       Covers reruns where ``delete_source_after_copy`` cleaned up
       the original. Without this the second run of any meeting
       would dead-lock on "source missing" — the canonical copy is
       in the workspace, not at the original drop path.

    3. The library row's stored ``workspace_dir`` when the SwiftUI
       caller didn't bother to forward one. Belt-and-braces for
       legacy callers that re-submit a job_id without a workspace.

    None of these touch the network, the DB, or anything stateful
    beyond ``Path.exists()`` — kept deliberately cheap because
    every job pays the cost on every launch.
    """
    if not request.source_path:
        append_app_log("engine_resolve_source skipped reason='empty_source_path'")
        return None
    candidate = Path(request.source_path).expanduser()
    if candidate.exists():
        append_app_log(f"engine_resolve_source hit='request_source' path={str(candidate)!r}")
        return candidate

    basename = candidate.name
    if not basename:
        append_app_log(f"engine_resolve_source skipped reason='empty_basename' source={request.source_path!r}")
        return None

    workspace_dirs: list[str] = []
    if request.workspace_dir:
        workspace_dirs.append(request.workspace_dir)
    if request.library_job_id is not None:
        try:
            row = database().get_job(request.library_job_id)
        except Exception:
            row = None
        if row:
            stored_workspace = (row.get("workspace_dir") or "").strip()
            if stored_workspace and stored_workspace not in workspace_dirs:
                workspace_dirs.append(stored_workspace)

    for workspace_dir in workspace_dirs:
        workspace_copy = Path(workspace_dir).expanduser() / basename
        if workspace_copy.exists():
            append_app_log(
                "engine_resolve_source hit='workspace_copy' "
                f"requested={str(candidate)!r} resolved={str(workspace_copy)!r}"
            )
            return workspace_copy
    append_app_log(
        "engine_resolve_source miss "
        f"requested={str(candidate)!r} workspaces={workspace_dirs!r}"
    )
    return None


def _auto_rename_job_from_transcript(
    db,
    job_id: int,
    transcript_path: str | None,
    source_path: str,
) -> str | None:
    """Pick a topical title from the transcript and store it on the job.

    Library rows fall back on the source filename when ``custom_title``
    is empty, which leaves the user staring at a wall of
    ``Enregistrement de l'écran 2026-05-16 à 14.42.30`` rows. Once the
    transcription is in, we have enough signal to do better — feed the
    text to :func:`suggest_transcript_stem` and persist the result.

    We never overwrite a title the user already set by hand: if a
    ``custom_title`` is present on the row before we run, we bail. The
    suggester also returns the fallback stem when it can't find a
    confident topic, in which case we still no-op to avoid promoting a
    raw filename to a "title".
    """
    if not transcript_path:
        return None
    try:
        row = db.get_job(job_id)
    except Exception:
        row = None
    if row and (row.get("custom_title") or "").strip():
        return None
    try:
        text = Path(transcript_path).read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    if not text.strip():
        return None
    # Lazy import to avoid pulling the full transcription_utils module
    # into the engine's startup path (it imports MLX/whisper bits).
    from transcription_utils import sanitize_filename_stem, suggest_transcript_stem

    fallback_stem = sanitize_filename_stem(Path(source_path).stem or "Transcription")
    suggestion = suggest_transcript_stem(text, fallback_stem)
    if not suggestion or suggestion == fallback_stem:
        return None
    db.update_job_title(job_id, suggestion)
    append_app_log(
        f"engine_auto_rename job_id={job_id} title={suggestion!r}"
    )
    return suggestion


def _snapshot_workspace_size(db, job_id: int, workspace: Path) -> None:
    """Walk the workspace and store the cumulative byte count.

    The library's optional "Poids" column reads this. We compute it
    once per successful job to avoid re-walking on every list
    refresh — the user has cheaper ways to ask for an update
    (delete + relaunch) and the alternative (lazy compute on
    library-list) would re-scan dozens of folders every refresh.
    """
    if not workspace.exists() or not workspace.is_dir():
        return
    total = 0
    for child in workspace.rglob("*"):
        try:
            if child.is_file() or child.is_symlink():
                total += child.stat().st_size if child.is_file() else 0
        except OSError:
            continue
    db.update_job_total_bytes(job_id, total)


class EngineRunner:
    def __init__(self, sink: EventSink):
        self.sink = sink

    def run_job(self, request: JobRequest) -> int:
        append_app_log(
            "engine_run_job "
            f"mode={request.mode!r} source={request.source_path!r} "
            f"workspace={request.workspace_dir!r} library_job_id={request.library_job_id!r}"
        )
        resolved = _resolve_source_path(request)
        if resolved is None:
            # ``source_missing`` is the trigger the SwiftUI layer
            # listens for to pop the "Relocaliser la source" sheet.
            # We surface the requested path AND the workspace we
            # tried so the dialog can offer both as one-click
            # actions.
            attempted_workspace = (request.workspace_dir or "").strip()
            append_app_log(
                "engine_source_missing "
                f"source={request.source_path!r} workspace={attempted_workspace!r}"
            )
            self.sink(
                ErrorEvent(
                    f"Source file not found: {request.source_path}",
                    code="source_missing",
                )
            )
            return 2
        # The resolver may have substituted a workspace copy or the
        # source. Update the request so downstream code (workspace
        # prep, DB row) sees the path that actually exists.
        request.source_path = str(resolved)
        Path(request.output_dir).mkdir(parents=True, exist_ok=True)
        db = database()
        estimated_total_seconds = _historical_total_estimate_seconds(db, request)
        sink = EtaSmoothingSink(self.sink, estimated_total_seconds)
        workspace, working_source = prepare_job_workspace(request, sink)
        request.workspace_dir = str(workspace)
        job_id = request.library_job_id
        if job_id is None:
            job_id = db.create_job(
                source_path=request.source_path,
                workspace_dir=str(workspace),
                settings=request.to_dict(),
            )
            # Nudge the SwiftUI library tab to refresh immediately. The
            # source ArtifactEvent from prepare_job_workspace fired
            # before the DB row existed, so its refresh saw nothing.
            # This second artifact event arrives after the INSERT, so
            # the list reflects the new job as soon as the user
            # switches tabs.
            sink(ArtifactEvent("library_row", str(workspace), model=str(job_id)))
        db.update_job_status(job_id, "RUNNING", "")
        db.update_job_progress(
            job_id,
            step="Démarrage…",
            progress_pct=0,
            eta_seconds=estimated_total_seconds,
        )
        # Persist the Odoo meeting metadata so the rename sheet can
        # surface attendee hint chips long after the engine exited.
        # Empty dict means "no meeting linked"; the helper takes care
        # of clearing the column on re-runs that detached the link.
        if request.odoo_meeting_metadata:
            db.update_job_odoo_meeting(job_id, request.odoo_meeting_metadata)
        elif request.library_job_id is None:
            # Fresh job, no meeting — explicitly null the column to
            # cover the unlikely case of stale data lingering on a
            # row reused via primary-key collision.
            db.update_job_odoo_meeting(job_id, None)

        results: list[StepResult] = []
        active_source = str(working_source)

        if request.mode in {"compress", "compress_transcribe"}:
            result = CompressionPipeline(request, sink).run()
            results.append(result)
            if not result.ok:
                _persist_step_durations(db, job_id, results)
                # Failed jobs still left partial files (the source
                # copy + maybe a half-written compressed clip). The
                # library "Poids" column wants a real number on
                # these, not "—", so the user can decide whether to
                # cleanup or rerun.
                _snapshot_workspace_size(db, job_id, workspace)
                db.update_job_status(job_id, "FAILED", result.error or "Compression failed")
                sink(ErrorEvent(result.error or "Compression failed", code="compression_failed"))
                return 1
            db.update_job_artefact(job_id, "compressed", result.artifact_path)
            db.update_job_output(job_id, result.artifact_path)
            db.update_job_progress(job_id, step="Compression terminée", progress_pct=50, eta_seconds=None)
            if request.mode == "compress_transcribe":
                active_source = result.artifact_path

        if request.mode in {"transcribe", "compress_transcribe"}:
            pipeline = TranscriptionPipeline(request, sink)
            tx_results = pipeline.run(active_source)
            results.extend(tx_results)
            failed = [r for r in tx_results if not r.ok and r.name in {"audio_extract", "whisper"}]
            # Quality steps (VAD, multipass, diarisation, llm_post) are
            # allowed to fail without sinking the whole job — they're
            # opt-in improvements. We only abort on the hard
            # prerequisites: audio extract and Whisper itself.
            if failed:
                _persist_step_durations(db, job_id, results)
                # Same rationale as the compression-failure branch:
                # the user wants to see how much disk this run
                # consumed before deciding whether to clean it up.
                _snapshot_workspace_size(db, job_id, workspace)
                db.update_job_status(job_id, "FAILED", failed[0].error or "Transcription failed")
                sink(ErrorEvent(failed[0].error or "Transcription failed", code="transcription_failed"))
                return 1
            transcript_for_title: str | None = None
            for r in tx_results:
                if r.name == "transcript" and r.artifact_path:
                    db.update_job_artefact(job_id, "transcript", r.artifact_path)
                    db.update_job_output(job_id, r.artifact_path)
                    transcript_for_title = r.artifact_path
                elif r.name == "enhanced_transcript" and r.artifact_path:
                    db.update_job_artefact(job_id, "enhanced_transcript", r.artifact_path)
                    db.update_job_output(job_id, r.artifact_path)
                    # The corrected transcript is a better title source
                    # — Whisper output sometimes mangles proper nouns
                    # that the LLM pass restores.
                    transcript_for_title = r.artifact_path
                elif r.name == "review" and r.artifact_path:
                    db.update_job_artefact(job_id, "review", r.artifact_path)
            # Promote a topical title onto the library row so the user
            # sees "Atelier RH 2026" rather than "Enregistrement de
            # l'écran 2026-05-16 à 14.42.30" — but only when no custom
            # title has been set yet (see the helper for the bail
            # rules).
            new_title = _auto_rename_job_from_transcript(
                db, job_id, transcript_for_title, request.source_path
            )
            if new_title:
                sink(ArtifactEvent("job_title", new_title, model=str(job_id)))
            # Persist segments + context so the library's rename
            # sheet has speakers to show and ``library_speaker_samples``
            # can cut audio extracts. Without these calls the
            # ``transcription_segments`` table and ``speaker_map_json``
            # column stay empty and the sheet renders "Aucun
            # interlocuteur détecté" on every run.
            if pipeline.final_segments:
                db.add_segments(job_id, pipeline.final_segments)
            if pipeline.final_speaker_map or pipeline.final_technical_terms:
                db.update_job_context(
                    job_id,
                    speakers=pipeline.final_speaker_map or None,
                    technical_terms=pipeline.final_technical_terms or None,
                )
            db.update_job_progress(job_id, step="Transcription terminée", progress_pct=100, eta_seconds=0)

        if request.mode in {"enhance", "review"}:
            sink(
                WarningEvent(
                    "Enhance/review-only mode is reserved for the next engine extraction step",
                    code="not_implemented_yet",
                )
            )

        _persist_step_durations(db, job_id, results)
        _snapshot_workspace_size(db, job_id, workspace)
        sink(ProgressEvent("done", 100, "Job complete", eta_seconds=0))
        db.update_job_status(job_id, "COMPLETED", "")
        sink(
            DoneEvent(
                {
                    "job_id": job_id,
                    "mode": request.mode,
                    "artifacts": [
                        {"kind": r.name, "path": r.artifact_path, "model": r.model}
                        for r in results
                        if r.artifact_path
                    ],
                    "steps": [
                        {
                            "name": r.name,
                            "ok": r.ok,
                            "duration_seconds": r.duration_seconds,
                            "metrics": r.metrics,
                        }
                        for r in results
                    ],
                }
            )
        )
        return 0
