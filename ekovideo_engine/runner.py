from __future__ import annotations

import json
import time
from pathlib import Path

from .events import EventSink
from .logging import append_app_log
from .models import DoneEvent, EngineEvent, ErrorEvent, JobRequest, ProgressEvent, WarningEvent
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


class EngineRunner:
    def __init__(self, sink: EventSink):
        self.sink = sink

    def run_job(self, request: JobRequest) -> int:
        append_app_log(f"engine_run_job mode={request.mode!r} source={request.source_path!r}")
        source = Path(request.source_path)
        if not source.exists():
            self.sink(ErrorEvent(f"Source file not found: {source}", code="source_missing"))
            return 2
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
        db.update_job_status(job_id, "RUNNING", "")
        db.update_job_progress(
            job_id,
            step="Démarrage…",
            progress_pct=0,
            eta_seconds=estimated_total_seconds,
        )

        results: list[StepResult] = []
        active_source = str(working_source)

        if request.mode in {"compress", "compress_transcribe"}:
            result = CompressionPipeline(request, sink).run()
            results.append(result)
            if not result.ok:
                _persist_step_durations(db, job_id, results)
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
                db.update_job_status(job_id, "FAILED", failed[0].error or "Transcription failed")
                sink(ErrorEvent(failed[0].error or "Transcription failed", code="transcription_failed"))
                return 1
            for r in tx_results:
                if r.name == "transcript" and r.artifact_path:
                    db.update_job_artefact(job_id, "transcript", r.artifact_path)
                    db.update_job_output(job_id, r.artifact_path)
                elif r.name == "enhanced_transcript" and r.artifact_path:
                    db.update_job_artefact(job_id, "enhanced_transcript", r.artifact_path)
                    db.update_job_output(job_id, r.artifact_path)
                elif r.name == "review" and r.artifact_path:
                    db.update_job_artefact(job_id, "review", r.artifact_path)
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
