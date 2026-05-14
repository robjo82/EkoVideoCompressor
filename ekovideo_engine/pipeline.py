from __future__ import annotations

import subprocess
import time
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from ffmpeg_utils import default_out_path, is_audio_only_path, build_ffmpeg_cmd
from transcription_utils import (
    build_audio_extract_cmd,
    build_mlx_whisper_cmd,
    default_transcript_path,
    parse_whisper_json_segments,
    render_segments_plain,
    structured_initial_prompt,
)

from .events import EventSink
from .logging import append_app_log, tail_text
from .models import ArtifactEvent, JobRequest, ProgressEvent


def _safe_stem(value: str) -> str:
    keep = []
    for char in value:
        if char.isalnum() or char in {" ", "-", "_", "."}:
            keep.append(char)
        else:
            keep.append(" ")
    cleaned = " ".join("".join(keep).split()).strip(" .-_")
    return (cleaned or "Transcription")[:80]


def job_workspace_dir(request: JobRequest) -> Path:
    if request.workspace_dir:
        return Path(request.workspace_dir)
    root = Path(request.output_dir).expanduser()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return root / f"{stamp} - {_safe_stem(Path(request.source_path).stem)}"


def prepare_job_workspace(request: JobRequest, sink: EventSink) -> tuple[Path, Path]:
    workspace = job_workspace_dir(request)
    workspace.mkdir(parents=True, exist_ok=True)
    source = Path(request.source_path).expanduser()
    copied_source = workspace / source.name
    if source.exists() and source.resolve() != copied_source.resolve() and not copied_source.exists():
        shutil.copy2(source, copied_source)
        sink(ArtifactEvent("source", str(copied_source)))
    return workspace, copied_source if copied_source.exists() else source


@dataclass(slots=True)
class StepResult:
    name: str
    ok: bool = True
    artifact_path: str = ""
    model: str = ""
    duration_seconds: float = 0.0
    metrics: dict[str, Any] = field(default_factory=dict)
    error: str = ""


class TranscriptionPipeline:
    """Headless transcription pipeline with the same command builders as the legacy UI."""

    def __init__(self, request: JobRequest, sink: EventSink):
        self.request = request
        self.sink = sink

    def run(self, source_path: str) -> list[StepResult]:
        results: list[StepResult] = []
        workspace = job_workspace_dir(self.request)
        workspace.mkdir(parents=True, exist_ok=True)

        wav_path = workspace / "audio.wav"
        results.append(self._extract_audio(source_path, wav_path))
        if not results[-1].ok:
            return results

        results.append(self._run_whisper(wav_path))
        return results

    def _extract_audio(self, source_path: str, wav_path: Path) -> StepResult:
        ts = time.monotonic()
        settings = self.request.transcription_settings
        cmd = build_audio_extract_cmd(
            self.request.compression_settings.ffmpeg_path or "ffmpeg",
            source_path,
            str(wav_path),
            speech_enhance=settings.enhance_audio,
        )
        self.sink(ProgressEvent("audio_extract", 0, "Extracting audio"))
        proc = subprocess.run(cmd, capture_output=True, text=True)
        duration = time.monotonic() - ts
        if proc.returncode != 0 or not wav_path.exists():
            message = tail_text(proc.stderr or proc.stdout)
            append_app_log(f"engine_audio_extract_failed rc={proc.returncode} error={message!r}")
            return StepResult("audio_extract", False, duration_seconds=duration, error=message)
        self.sink(ArtifactEvent("audio_wav", str(wav_path)))
        self.sink(ProgressEvent("audio_extract", 100, "Audio ready"))
        return StepResult("audio_extract", True, str(wav_path), duration_seconds=duration)

    def _run_whisper(self, wav_path: Path) -> StepResult:
        ts = time.monotonic()
        settings = self.request.transcription_settings
        out_dir = Path(self.request.output_dir)
        if self.request.workspace_dir:
            out_dir = Path(self.request.workspace_dir)
        elif Path(self.request.output_dir).name != Path(source_path).parent.name:
            out_dir = job_workspace_dir(self.request)
        out_dir.mkdir(parents=True, exist_ok=True)
        transcript_path = Path(
            default_transcript_path(
                self.request.source_path,
                str(out_dir),
                settings.suffix,
                settings.output_format,
            )
        )
        whisper_json = transcript_path.with_suffix(".whisper.json")
        glossary = "\n".join(
            [*self.request.glossary_terms, *self.request.technical_terms]
        )
        cmd = build_mlx_whisper_cmd(
            mlx_whisper_path=settings.mlx_whisper_path or "mlx_whisper",
            audio_path=str(wav_path),
            output_path=str(whisper_json),
            model=settings.model,
            language=settings.language,
            output_format="json",
            initial_prompt=structured_initial_prompt(glossary),
            condition_on_previous_text=False,
        )
        self.sink(ProgressEvent("whisper", 0, "Running Whisper"))
        proc = subprocess.run(cmd, capture_output=True, text=True)
        duration = time.monotonic() - ts
        if proc.returncode != 0 or not whisper_json.exists():
            message = tail_text(proc.stderr or proc.stdout)
            append_app_log(f"engine_whisper_failed rc={proc.returncode} error={message!r}")
            return StepResult(
                "whisper",
                False,
                model=settings.model,
                duration_seconds=duration,
                error=message,
            )

        segments = parse_whisper_json_segments(str(whisper_json))
        transcript_path.write_text(
            render_segments_plain(segments, settings.output_format),
            encoding="utf-8",
        )
        self.sink(ArtifactEvent("transcript", str(transcript_path), model=settings.model))
        self.sink(ProgressEvent("whisper", 100, "Transcript ready"))
        return StepResult(
            "whisper",
            True,
            str(transcript_path),
            model=settings.model,
            duration_seconds=duration,
            metrics={"segments": len(segments)},
        )


class CompressionPipeline:
    def __init__(self, request: JobRequest, sink: EventSink):
        self.request = request
        self.sink = sink

    def run(self) -> StepResult:
        ts = time.monotonic()
        settings = self.request.compression_settings
        out_dir = Path(self.request.output_dir)
        if self.request.workspace_dir:
            out_dir = Path(self.request.workspace_dir)
        else:
            out_dir = job_workspace_dir(self.request)
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = default_out_path(self.request.source_path, str(out_dir), "_compressed")
        cmd = build_ffmpeg_cmd(
            settings.ffmpeg_path or "ffmpeg",
            self.request.source_path,
            output_path,
            crf=settings.crf,
            resolution=settings.resolution,
            fps=settings.fps,
            audio_bitrate=settings.audio_bitrate,
            preset=settings.preset,
            speech_enhance=settings.speech_enhance,
            mono_audio=settings.mono_audio,
            ss=settings.trim_start if settings.trim_enabled else None,
            to=settings.trim_end if settings.trim_enabled else None,
            audio_only=is_audio_only_path(self.request.source_path),
        )
        self.sink(ProgressEvent("compression", 0, "Running FFmpeg"))
        proc = subprocess.run(cmd, capture_output=True, text=True)
        duration = time.monotonic() - ts
        if proc.returncode != 0 or not Path(output_path).exists():
            message = tail_text(proc.stderr or proc.stdout)
            append_app_log(f"engine_compress_failed rc={proc.returncode} error={message!r}")
            return StepResult("compression", False, duration_seconds=duration, error=message)
        self.sink(ArtifactEvent("compressed", output_path))
        self.sink(ProgressEvent("compression", 100, "Compression ready"))
        return StepResult("compression", True, output_path, duration_seconds=duration)
