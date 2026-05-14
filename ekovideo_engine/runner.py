from __future__ import annotations

from pathlib import Path

from .events import EventSink
from .logging import append_app_log
from .models import DoneEvent, ErrorEvent, JobRequest, ProgressEvent, WarningEvent
from .pipeline import CompressionPipeline, StepResult, TranscriptionPipeline


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
        if request.workspace_dir:
            Path(request.workspace_dir).mkdir(parents=True, exist_ok=True)

        results: list[StepResult] = []
        active_source = str(source)

        if request.mode in {"compress", "compress_transcribe"}:
            result = CompressionPipeline(request, self.sink).run()
            results.append(result)
            if not result.ok:
                self.sink(ErrorEvent(result.error or "Compression failed", code="compression_failed"))
                return 1
            if request.mode == "compress_transcribe":
                active_source = result.artifact_path

        if request.mode in {"transcribe", "compress_transcribe"}:
            tx_results = TranscriptionPipeline(request, self.sink).run(active_source)
            results.extend(tx_results)
            failed = [r for r in tx_results if not r.ok]
            if failed:
                self.sink(ErrorEvent(failed[0].error or "Transcription failed", code="transcription_failed"))
                return 1

        if request.mode in {"enhance", "review"}:
            self.sink(
                WarningEvent(
                    "Enhance/review-only mode is reserved for the next engine extraction step",
                    code="not_implemented_yet",
                )
            )

        self.sink(ProgressEvent("done", 100, "Job complete"))
        self.sink(
            DoneEvent(
                {
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
