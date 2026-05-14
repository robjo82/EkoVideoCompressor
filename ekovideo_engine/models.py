from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal


JobMode = Literal[
    "compress",
    "transcribe",
    "compress_transcribe",
    "enhance",
    "review",
]


@dataclass(slots=True)
class CompressionSettings:
    ffmpeg_path: str = "ffmpeg"
    ffprobe_path: str = "ffprobe"
    resolution: str = "720p"
    fps: int = 12
    crf: int = 28
    audio_bitrate: str = "128k"
    preset: str = "medium"
    speech_enhance: bool = True
    mono_audio: bool = False
    trim_enabled: bool = False
    trim_start: str = "00:00:00"
    trim_end: str = "00:00:00"

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "CompressionSettings":
        data = dict(raw or {})
        allowed = {field.name for field in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in allowed})


@dataclass(slots=True)
class TranscriptionSettings:
    mlx_whisper_path: str = ""
    model: str = "mlx-community/whisper-large-v3-turbo"
    language: str = "fr"
    output_format: str = "txt"
    suffix: str = ""
    enhance_audio: bool = True
    diarization_enabled: bool = False
    hf_token: str = ""
    venv_python_path: str = ""
    text_llm_model: str = "mlx-community/Mistral-7B-Instruct-v0.3-4bit"
    audio_llm_model: str = "mlx-community/Qwen2-Audio-7B-Instruct-4bit"
    audio_recheck_enabled: bool = False
    vad_enabled: bool = True
    multipass_enabled: bool = True
    per_speaker_enabled: bool = False
    web_enrichment_enabled: bool = False

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "TranscriptionSettings":
        data = dict(raw or {})
        allowed = {field.name for field in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in allowed})


@dataclass(slots=True)
class JobRequest:
    source_path: str
    output_dir: str
    mode: JobMode
    workspace_dir: str = ""
    profile: str = "Reunion equilibree"
    compression_settings: CompressionSettings = field(default_factory=CompressionSettings)
    transcription_settings: TranscriptionSettings = field(default_factory=TranscriptionSettings)
    glossary_terms: list[str] = field(default_factory=list)
    speaker_overrides: dict[str, str] = field(default_factory=dict)
    technical_terms: list[str] = field(default_factory=list)
    rerun_steps: list[str] = field(default_factory=list)
    library_job_id: int | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "JobRequest":
        if not isinstance(raw, dict):
            raise ValueError("JobRequest must be a JSON object")
        data = dict(raw)
        required = ["source_path", "output_dir", "mode"]
        missing = [key for key in required if not str(data.get(key) or "").strip()]
        if missing:
            raise ValueError(f"Missing required JobRequest field(s): {', '.join(missing)}")
        mode = str(data["mode"])
        if mode not in {"compress", "transcribe", "compress_transcribe", "enhance", "review"}:
            raise ValueError(f"Unsupported JobRequest mode: {mode}")
        return cls(
            source_path=str(data["source_path"]),
            output_dir=str(data["output_dir"]),
            mode=mode,  # type: ignore[arg-type]
            workspace_dir=str(data.get("workspace_dir") or ""),
            profile=str(data.get("profile") or "Reunion equilibree"),
            compression_settings=CompressionSettings.from_dict(
                data.get("compression_settings")
            ),
            transcription_settings=TranscriptionSettings.from_dict(
                data.get("transcription_settings")
            ),
            glossary_terms=[str(x) for x in data.get("glossary_terms") or []],
            speaker_overrides={
                str(k): str(v) for k, v in (data.get("speaker_overrides") or {}).items()
            },
            technical_terms=[str(x) for x in data.get("technical_terms") or []],
            rerun_steps=[str(x) for x in data.get("rerun_steps") or []],
            library_job_id=data.get("library_job_id"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EngineEvent:
    event: str
    ts: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _now_ts() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class ProgressEvent(EngineEvent):
    step: str = ""
    pct: float | None = None
    eta_seconds: float | None = None
    message: str = ""

    def __init__(
        self,
        step: str,
        pct: float | None = None,
        message: str = "",
        eta_seconds: float | None = None,
    ):
        self.event = "progress"
        self.ts = _now_ts()
        self.step = step
        self.pct = pct
        self.eta_seconds = eta_seconds
        self.message = message


@dataclass(slots=True)
class ArtifactEvent(EngineEvent):
    kind: str = ""
    path: str = ""
    model: str = ""

    def __init__(self, kind: str, path: str, model: str = ""):
        self.event = "artifact"
        self.ts = _now_ts()
        self.kind = kind
        self.path = path
        self.model = model


@dataclass(slots=True)
class ContextEvent(EngineEvent):
    speakers: dict[str, str] = field(default_factory=dict)
    technical_terms: list[str] = field(default_factory=list)

    def __init__(
        self,
        speakers: dict[str, str] | None = None,
        technical_terms: list[str] | None = None,
    ):
        self.event = "context"
        self.ts = _now_ts()
        self.speakers = dict(speakers or {})
        self.technical_terms = list(technical_terms or [])


@dataclass(slots=True)
class WarningEvent(EngineEvent):
    message: str = ""
    code: str = ""

    def __init__(self, message: str, code: str = ""):
        self.event = "warning"
        self.ts = _now_ts()
        self.message = message
        self.code = code


@dataclass(slots=True)
class ErrorEvent(EngineEvent):
    message: str = ""
    code: str = ""

    def __init__(self, message: str, code: str = ""):
        self.event = "error"
        self.ts = _now_ts()
        self.message = message
        self.code = code


@dataclass(slots=True)
class DoneEvent(EngineEvent):
    summary: dict[str, Any] = field(default_factory=dict)

    def __init__(self, summary: dict[str, Any] | None = None):
        self.event = "done"
        self.ts = _now_ts()
        self.summary = dict(summary or {})
