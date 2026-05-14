from __future__ import annotations

from pathlib import Path

from .events import EventSink
from .logging import append_app_log
from .models import DoneEvent, ErrorEvent, JobRequest, ProgressEvent, WarningEvent
from .library import database
from .pipeline import CompressionPipeline, StepResult, TranscriptionPipeline
from .pipeline import prepare_job_workspace


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
        workspace, working_source = prepare_job_workspace(request, self.sink)
        request.workspace_dir = str(workspace)
        db = database()
        job_id = request.library_job_id
        if job_id is None:
            job_id = db.create_job(
                source_path=request.source_path,
                workspace_dir=str(workspace),
                settings=request.to_dict(),
            )
        db.update_job_status(job_id, "RUNNING", "")
        db.update_job_progress(job_id, step="Démarrage…", progress_pct=0, eta_seconds=None)

        results: list[StepResult] = []
        active_source = str(working_source)

        if request.mode in {"compress", "compress_transcribe"}:
            result = CompressionPipeline(request, self.sink).run()
            results.append(result)
            if not result.ok:
                db.update_job_status(job_id, "FAILED", result.error or "Compression failed")
                self.sink(ErrorEvent(result.error or "Compression failed", code="compression_failed"))
                return 1
            db.update_job_artefact(job_id, "compressed", result.artifact_path)
            db.update_job_output(job_id, result.artifact_path)
            db.update_job_progress(job_id, step="Compression terminée", progress_pct=50, eta_seconds=None)
            if request.mode == "compress_transcribe":
                active_source = result.artifact_path

        if request.mode in {"transcribe", "compress_transcribe"}:
            tx_results = TranscriptionPipeline(request, self.sink).run(active_source)
            results.extend(tx_results)
            failed = [r for r in tx_results if not r.ok and r.name in {"audio_extract", "whisper"}]
            # Quality steps (VAD, multipass, diarisation, llm_post) are
            # allowed to fail without sinking the whole job — they're
            # opt-in improvements. We only abort on the hard
            # prerequisites: audio extract and Whisper itself.
            if failed:
                db.update_job_status(job_id, "FAILED", failed[0].error or "Transcription failed")
                self.sink(ErrorEvent(failed[0].error or "Transcription failed", code="transcription_failed"))
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
            db.update_job_progress(job_id, step="Transcription terminée", progress_pct=100, eta_seconds=0)

        if request.mode in {"enhance", "review"}:
            self.sink(
                WarningEvent(
                    "Enhance/review-only mode is reserved for the next engine extraction step",
                    code="not_implemented_yet",
                )
            )

        self.sink(ProgressEvent("done", 100, "Job complete"))
        db.update_job_status(job_id, "COMPLETED", "")
        self.sink(
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
