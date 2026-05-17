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


# Quality presets collapse "which boolean flags to enable" into a
# single choice that the SwiftUI app exposes in a Picker. Power users
# can still override individual flags by setting quality_preset to
# "custom" — that mode passes the values through untouched.
QualityPreset = Literal["fast", "balanced", "max", "custom"]


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
    # The repass model the pipeline reruns on Whisper's low-confidence
    # segments. Used to be hardcoded inside ``_run_multipass``; now
    # surfaced as a knob so users with the headroom can pin a
    # French-tuned distil checkpoint instead of the multilingual
    # large-v3. Empty string falls back to the catalog default
    # (``whisper-large-v3-mlx``) inside the pipeline.
    multipass_model: str = ""
    audio_recheck_enabled: bool = False
    vad_enabled: bool = True
    multipass_enabled: bool = True
    per_speaker_enabled: bool = False
    web_enrichment_enabled: bool = False
    # Default is "custom" so legacy callers that pass individual
    # flags keep their semantics. The SwiftUI app explicitly sends
    # "balanced" / "fast" / "max" to opt into the preset mapping.
    quality_preset: QualityPreset = "custom"
    # Speaker-count hints forwarded to pyannote. Left at 0 they fall
    # back to "let the model decide" — which under-segments most of
    # the time. The SwiftUI app surfaces these as an optional input
    # under the diarisation toggle ("nombre d'intervenants attendu").
    expected_min_speakers: int = 0
    expected_max_speakers: int = 0
    # Pyannote happily emits 100-300 ms turns when one participant
    # back-channels ("hm", "ouais") in the middle of another's
    # sentence. Anything shorter than this gets folded into the
    # surrounding turn so the transcript stops jumping speaker every
    # other word.
    min_speaker_turn_seconds: float = 0.4

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "TranscriptionSettings":
        data = dict(raw or {})
        allowed = {field.name for field in cls.__dataclass_fields__.values()}
        instance = cls(**{k: v for k, v in data.items() if k in allowed})
        return apply_quality_preset(instance)


def apply_quality_preset(settings: "TranscriptionSettings") -> "TranscriptionSettings":
    """Resolve the boolean flags from the ``quality_preset`` selector.

    Centralising the mapping here means the SwiftUI app only needs to
    send a single string (``"fast" / "balanced" / "max"``) and the
    engine derives the right knobs. Custom mode (or any unknown
    value) is a passthrough — power users keep full control.
    """
    preset = (settings.quality_preset or "balanced").strip().lower()
    if preset == "fast":
        # Pure baseline: Whisper only. No VAD, no multipass, no LLM,
        # no diarisation. Roughly real-time on M1.
        settings.vad_enabled = False
        settings.multipass_enabled = False
        settings.per_speaker_enabled = False
        settings.audio_recheck_enabled = False
        settings.web_enrichment_enabled = False
    elif preset == "balanced":
        # Default. VAD + multipass + LLM post-pass (when the venv
        # exists). No per-speaker (it doubles the runtime).
        settings.vad_enabled = True
        settings.multipass_enabled = True
        settings.per_speaker_enabled = False
        settings.audio_recheck_enabled = False
        settings.web_enrichment_enabled = False
    elif preset == "max":
        # Everything the engine actually wires today: VAD with the
        # safety-net fallback, multipass with the higher-accuracy
        # large-v3 repass, diarisation with word-level speaker
        # attribution, LLM post-process. ``per_speaker_enabled``,
        # ``audio_recheck_enabled``, ``web_enrichment_enabled`` are
        # *intentionally off* — the orchestrator doesn't run them
        # yet. Promising "Whisper par locuteur + réécoute IA +
        # enrichissement web" while doing none of those was a false
        # advertisement; once any of them ships we'll re-enable
        # them here.
        settings.vad_enabled = True
        settings.multipass_enabled = True
        settings.per_speaker_enabled = False
        settings.audio_recheck_enabled = False
        settings.web_enrichment_enabled = False
    # 'custom' (or any other value) is a passthrough — leave the
    # individual flags as the caller set them.
    return settings


@dataclass(slots=True)
class OdooContextRef:
    """Pointer to an Odoo object whose chatter the pipeline fetches
    during the LLM step to enrich the correction prompt.

    Lives outside ``TranscriptionSettings`` because the credentials
    needed to act on it live next to it on the same JSON envelope
    — keeps the LLM step from having to reach back into preferences
    every time.
    """

    model: str = ""
    record_id: int = 0
    url: str = ""
    database: str = ""
    login: str = ""
    api_key: str = ""

    def is_actionable(self) -> bool:
        return bool(
            self.model
            and self.record_id
            and self.url
            and self.database
            and self.api_key
        )

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "OdooContextRef":
        data = dict(raw or {})
        return cls(
            model=str(data.get("model") or ""),
            record_id=int(data.get("record_id") or data.get("id") or 0),
            url=str(data.get("url") or ""),
            database=str(data.get("database") or ""),
            login=str(data.get("login") or ""),
            api_key=str(data.get("api_key") or ""),
        )


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
    delete_source_after_copy: bool = False
    # ISO-8601 timestamp of the actual meeting. Defaults to the
    # source file's metadata on the SwiftUI side, but can be corrected
    # manually when a recording was copied or exported later than the
    # meeting itself.
    meeting_date: str = ""
    # Optional Odoo object whose chatter we'll fetch during the LLM
    # enhancement step. Empty when the user didn't link a meeting in
    # Run Setup; populated when they clicked "Utiliser" on a
    # ``calendar.event`` whose ``opportunity_id`` (or similar)
    # pointed at a useful CRM / project record.
    odoo_context_ref: OdooContextRef = field(default_factory=OdooContextRef)
    # Snapshot of the Odoo calendar event the user paired with this
    # job in Run Setup. Persisted on the ``jobs`` row so the rename
    # sheet later shows one-click attribution chips for each
    # invitee even after the engine has exited. Empty on jobs that
    # weren't paired with any meeting.
    odoo_meeting_metadata: dict[str, Any] = field(default_factory=dict)

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
            delete_source_after_copy=bool(data.get("delete_source_after_copy") or False),
            meeting_date=str(data.get("meeting_date") or ""),
            odoo_context_ref=OdooContextRef.from_dict(data.get("odoo_context_ref")),
            odoo_meeting_metadata=dict(data.get("odoo_meeting_metadata") or {}),
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
