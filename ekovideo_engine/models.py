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
    # PR AW — Gemma 4 12B Unified: mlx-vlm dropped ``qwen2_audio``
    # upstream; the 12B is the audio checkpoint verified end-to-end
    # (mlx_vlm.models.gemma4_unified).
    # Persisted Qwen2-Audio ids are remapped by
    # ``canonical_audio_llm_model_id`` at the pipeline's use sites.
    audio_llm_model: str = "mlx-community/gemma-4-12B-it-4bit"
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
    # Name the user designates as "themselves" in Réglages. When
    # set, the pipeline pre-attributes the cluster that speaks first
    # in the recording to this name — bypassing voice matching for
    # the user even before any voiceprint has been enrolled. Empty
    # string disables the heuristic.
    current_user_name: str = ""
    # Whisper's ``--condition-on-previous-text`` propagates the
    # decoded text of window N into the prompt of window N+1, which
    # boosts coherence and proper-noun stability across the meeting
    # (a name decoded right in minute 5 carries into minute 10).
    # Historically disabled because the decoder can latch onto a
    # hallucinated phrase and repeat it for minutes; we mitigate
    # downstream with ``clean_whisper_segments`` (drops decoder loops
    # of length > 2) and the multipass on weak segments. Off by
    # default for fast/balanced/custom — enabled by the ``max`` preset.
    condition_on_previous_text: bool = False
    # When True, after the first Whisper pass the engine scans the
    # transcript for repeated capitalised tokens not already in the
    # glossary and folds them into ``glossary_terms`` *in-process*
    # so the downstream passes (multipass, boundary multipass, LLM)
    # see an enriched prompt. Cheaper than re-running Whisper on
    # 5-minute chunks while still capturing the "name learned in
    # minute 5 helps minute 10" effect. Enabled by ``max`` preset.
    hot_prompt_enrichment: bool = False

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "TranscriptionSettings":
        data = dict(raw or {})
        allowed = {field.name for field in cls.__dataclass_fields__.values()}
        instance = cls(**{k: v for k, v in data.items() if k in allowed})
        return apply_quality_preset(instance)


# ---------------------------------------------------------------------
# Quality preset matrix (PR L)
# ---------------------------------------------------------------------
#
# Single source of truth for what each preset turns on. Centralising
# the matrix here lets us:
#   • Document the trade-off in one place (preset name → row).
#   • Sentinel-test that ``max`` enables every quality knob the
#     engine actually wires up (catches future PRs that add a flag
#     but forget to flip it on in ``max`` — which was the whole
#     point of the "Maximale" preset breaking down over time).
#   • Surface the matrix to docs / the SwiftUI summary without
#     copy-pasting strings.
#
# The keys are field names on ``TranscriptionSettings`` ; the values
# are booleans the preset writes when the user selects that preset.
# Fields not listed in a preset row are left untouched (so the user's
# advanced toggles in "custom" mode keep working). ``max`` is the
# only preset that the sentinel test enforces to be "true max" —
# explicitly listed exclusions (audio_recheck pending PR F) live in
# ``_MAX_PRESET_PENDING`` so we know exactly what's still missing.

QUALITY_PRESETS: dict[str, dict[str, bool]] = {
    "fast": {
        # Pure baseline: Whisper only. No VAD, no multipass, no LLM,
        # no diarisation, no context propagation. Roughly real-time
        # on M1.
        "vad_enabled": False,
        "multipass_enabled": False,
        "per_speaker_enabled": False,
        "audio_recheck_enabled": False,
        "web_enrichment_enabled": False,
        "condition_on_previous_text": False,
        "hot_prompt_enrichment": False,
    },
    "balanced": {
        # Default. VAD + multipass + LLM post-pass (when the venv
        # exists). No per-speaker (doubles the runtime), no context
        # propagation (extra repetition-loop surface for a marginal
        # gain when meetings are short).
        "vad_enabled": True,
        "multipass_enabled": True,
        "per_speaker_enabled": False,
        "audio_recheck_enabled": False,
        "web_enrichment_enabled": False,
        "condition_on_previous_text": False,
        "hot_prompt_enrichment": False,
    },
    "max": {
        # Everything wired in the engine today: VAD, multipass
        # (low-confidence + boundary), per-speaker Whisper pass
        # (PR E), web enrichment (PR H), hot prompt enrichment
        # (PR D), multimodal audio recheck (PR F — Qwen2-Audio via
        # ``mlx_vlm``). When the venv doesn't ship ``mlx_vlm`` the
        # recheck step degrades to a silent no-op with a warning,
        # so flipping it on here is safe even on machines that
        # haven't installed the audio LLM yet.
        #
        # PR Y — ``condition_on_previous_text`` deliberately stays
        # OFF in max. The audit on two 1-2 h meetings (CVR, Caste)
        # showed Whisper hallucinating a phrase (e.g. ``"On est sur
        # Zindoc pour la gestion des fichiers."``) and the
        # propagated context locking it in for 70+ minutes of audio.
        # ``clean_whisper_segments`` then drops the looped segments
        # (correct), but ~50 % of the recorded content vanishes from
        # the final transcript with no warning. The win on
        # proper-noun stability does not justify silently losing an
        # hour of meeting. Re-enabling requires a recovery mechanism
        # (re-Whisper the failed range without context).
        "vad_enabled": True,
        "multipass_enabled": True,
        "per_speaker_enabled": True,
        # PR AW — OFF until upstream mlx-vlm stabilises Gemma 4 audio
        # generation (qwen2_audio removed upstream; gemma edge fails
        # to load; gemma 12B loops). See pipeline.
        # _AUDIO_RECHECK_UPSTREAM_BLOCK for the full empirical dossier
        # and the single flag to flip when a fixed mlx-vlm ships.
        "audio_recheck_enabled": False,
        "web_enrichment_enabled": True,
        "condition_on_previous_text": False,
        "hot_prompt_enrichment": True,
    },
}

# Knobs that ``max`` deliberately leaves off. Each entry needs a
# comment block in the ``max`` row above explaining the engine work
# that re-enabling would require. The sentinel test in
# ``tests/test_quality_presets.py`` uses this set to allow the gap
# without losing the "max really is max" invariant for everything else.
_MAX_PRESET_PENDING: frozenset[str] = frozenset(
    {
        # PR Y — pending a re-Whisper recovery for loop ranges.
        "condition_on_previous_text",
        # PR AW — pending an upstream mlx-vlm fix for Gemma 4 audio
        # generation (qwen2_audio gone; gemma4 unstable). Gated by
        # pipeline._AUDIO_RECHECK_UPSTREAM_BLOCK.
        "audio_recheck_enabled",
    }
)


def apply_quality_preset(settings: "TranscriptionSettings") -> "TranscriptionSettings":
    """Resolve the boolean flags from the ``quality_preset`` selector.

    Centralising the mapping in ``QUALITY_PRESETS`` means the SwiftUI
    app only needs to send a single string (``"fast" / "balanced" /
    "max"``) and the engine derives the right knobs. ``custom`` (or
    any unknown value) is a passthrough — power users keep full
    control.
    """
    preset_key = (settings.quality_preset or "balanced").strip().lower()
    spec = QUALITY_PRESETS.get(preset_key)
    if spec is None:
        # 'custom' or anything else — leave the individual flags as
        # the caller set them.
        return settings
    for field_name, value in spec.items():
        setattr(settings, field_name, value)
    return settings


def quality_preset_levers(preset: str) -> dict[str, bool]:
    """Return the ``{field: bool}`` matrix the given preset writes.

    Returns an empty dict for ``custom`` / unknown presets. Exposed
    for tests and for docs that want to render the matrix without
    duplicating the strings.
    """
    return dict(QUALITY_PRESETS.get((preset or "").strip().lower(), {}))


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
