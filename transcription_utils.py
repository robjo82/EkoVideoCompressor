from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Iterable


TRANSCRIPTION_AUDIO_FILTERS = [
    "highpass=f=80",
    "lowpass=f=7600",
    "acompressor=threshold=-20dB:ratio=2.2:attack=5:release=160",
    "loudnorm=I=-16:TP=-1.5:LRA=11",
]

# Telephony chain: phone audio is narrowband (G.711, 8 kHz sample
# rate, 4 kHz Nyquist). The signal above 3400 Hz is dead spectrum.
# We swap the lowpass for a tighter one, add an FFT-based denoise
# pass tuned for narrowband background, and compress harder to
# bring up the quiet speech that's typical of distant callers.
# Empirically improves Whisper on phone recordings (the Caste job
# audio confirmed how much the default chain leaves on the table).
TRANSCRIPTION_AUDIO_FILTERS_TELEPHONY = [
    "highpass=f=80",
    # Above 3400 Hz is interpolation artefact from the resampler,
    # not signal. Trimming it stops Whisper from leaning on noise.
    "lowpass=f=3400",
    # FFT denoise. ``nr=18`` ≈ -18 dB attenuation on stationary
    # background; ``nf=-25`` sets the noise floor estimator.
    # Stronger than default because phone lines carry hum + hiss.
    "afftdn=nr=18:nf=-25",
    # Harder compression than the studio chain — caller may switch
    # between speakerphone and handset, big dynamic range to flatten.
    "acompressor=threshold=-22dB:ratio=3:attack=4:release=180",
    "loudnorm=I=-16:TP=-1.5:LRA=11",
]

# Audio profile names that ``select_transcription_audio_filters``
# returns. Pipeline + UI logging keys off these strings.
AUDIO_PROFILE_STANDARD = "standard"
AUDIO_PROFILE_TELEPHONY = "telephony"


def select_transcription_audio_filters(profile: str) -> list[str]:
    """Pick the right filter chain for ``profile``.

    Unknown values fall back to ``standard`` so a misconfigured
    caller still gets reasonable enhancement rather than nothing.
    """
    if profile == AUDIO_PROFILE_TELEPHONY:
        return list(TRANSCRIPTION_AUDIO_FILTERS_TELEPHONY)
    return list(TRANSCRIPTION_AUDIO_FILTERS)


def detect_audio_profile(
    in_path: str,
    *,
    ffprobe_path: str | None = None,
    bandwidth_threshold_hz: int = 12000,
) -> str:
    """Return ``"telephony"`` when the source audio is narrowband,
    ``"standard"`` otherwise.

    Heuristic :
    - Sample rate ≤ 11.025 kHz → telephony (G.711 is 8 kHz).
    - Codec in the narrowband family (``pcm_alaw``, ``pcm_mulaw``,
      ``g711``, ``g729``, ``opus`` with the narrowband ramp) → also
      telephony.
    - Anything else, or any probe error → ``standard`` (safe
      default: the existing chain still helps studio audio without
      the harsh telephony lowpass).

    Cheap probe (single ``ffprobe`` call, json output). Returns
    ``standard`` on missing ffprobe or any exception — never crashes
    the pipeline.
    """
    if not ffprobe_path:
        return AUDIO_PROFILE_STANDARD
    try:
        import subprocess
        out = subprocess.run(
            [
                ffprobe_path,
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=sample_rate,codec_name",
                "-of",
                "json",
                in_path,
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except Exception:
        return AUDIO_PROFILE_STANDARD
    if out.returncode != 0 or not out.stdout.strip():
        return AUDIO_PROFILE_STANDARD
    try:
        payload = json.loads(out.stdout)
    except json.JSONDecodeError:
        return AUDIO_PROFILE_STANDARD
    streams = payload.get("streams") or []
    if not streams:
        return AUDIO_PROFILE_STANDARD
    stream = streams[0]
    codec = str(stream.get("codec_name") or "").lower()
    try:
        sample_rate = int(stream.get("sample_rate") or 0)
    except (TypeError, ValueError):
        sample_rate = 0
    if codec in {"pcm_alaw", "pcm_mulaw", "g711", "g729"}:
        return AUDIO_PROFILE_TELEPHONY
    if 0 < sample_rate <= bandwidth_threshold_hz:
        return AUDIO_PROFILE_TELEPHONY
    return AUDIO_PROFILE_STANDARD

# Hugging Face model IDs used for diarisation. Both require accepting the
# license on huggingface.co before a token can download them.
PYANNOTE_DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
PYANNOTE_SEGMENTATION_MODEL = "pyannote/segmentation-3.0"
PYANNOTE_COMMUNITY_MODEL = "pyannote/speaker-diarization-community-1"


# ---------------------------------------------------------------------------
# Local LLM catalogues
#
# Two distinct families because they're loaded by two distinct mlx
# packages. Mixing them up (e.g. asking mlx_lm to load Qwen2-Audio) yields
# either a load failure or garbage output.
#
# We keep the lists short on purpose: 2-4 well-known checkpoints per
# family, ordered light → heavy, so users can pick a default for M1
# (3 B, 4-bit) and a stronger option for M4 Max (14 B, 4-bit).
# ---------------------------------------------------------------------------

#
# Each catalogue entry now carries:
#   - ``id``       canonical Hugging Face repo
#   - ``label``    short human-readable name (without the role)
#   - ``family``   model family (Whisper / Mistral / Qwen / …)
#   - ``role``     where the engine uses it
#                  (transcription / multipass / text_llm / audio_llm /
#                  diarisation / embedding)
#   - ``size_mb``  approximate on-disk weight in MB (post-quantisation)
#   - ``tier``     coarse hardware target (``light`` / ``balanced`` /
#                  ``heavy``) the SwiftUI tab uses to colour-code
#                  recommendations
#   - ``language`` optional ISO 639-1 list (defaults to ["multi"])
#
# Keep the lists short and curated — power users can override with
# any HF repo by typing the id directly in Settings, the catalogue
# is just what the Models tab surfaces.


_TRANSCRIPTION_ROLE = "transcription"
_MULTIPASS_ROLE = "multipass"
_TEXT_LLM_ROLE = "text_llm"
_AUDIO_LLM_ROLE = "audio_llm"
_DIARISATION_ROLE = "diarisation"
_EMBEDDING_ROLE = "embedding"


TEXT_LLM_MODELS: list[dict] = [
    {
        "id": "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
        "label": "Mistral 7B Instruct · 4-bit",
        "family": "Mistral",
        "role": _TEXT_LLM_ROLE,
        "size_mb": 4100,
        "tier": "balanced",
        "language": ["fr", "en", "multi"],
        "default": True,
    },
    {
        "id": "mlx-community/Llama-3.2-3B-Instruct-4bit",
        "label": "Llama 3.2 3B Instruct · 4-bit",
        "family": "Llama",
        "role": _TEXT_LLM_ROLE,
        "size_mb": 2000,
        "tier": "light",
        "language": ["fr", "en", "multi"],
    },
    {
        "id": "mlx-community/Qwen2.5-7B-Instruct-4bit",
        "label": "Qwen 2.5 7B Instruct · 4-bit",
        "family": "Qwen",
        "role": _TEXT_LLM_ROLE,
        "size_mb": 4400,
        "tier": "balanced",
        "language": ["fr", "en", "multi"],
    },
    {
        "id": "mlx-community/Qwen2.5-14B-Instruct-4bit",
        "label": "Qwen 2.5 14B Instruct · 4-bit",
        "family": "Qwen",
        "role": _TEXT_LLM_ROLE,
        "size_mb": 8200,
        "tier": "heavy",
        "language": ["fr", "en", "multi"],
    },
]

AUDIO_LLM_MODELS: list[dict] = [
    {
        "id": "mlx-community/Qwen2-Audio-7B-Instruct-4bit",
        "label": "Qwen2-Audio 7B Instruct · 4-bit",
        "family": "Qwen-Audio",
        "role": _AUDIO_LLM_ROLE,
        "size_mb": 4500,
        "tier": "balanced",
        "language": ["multi"],
        "default": True,
        # ``available=False`` because the new SwiftUI engine doesn't
        # wire the multimodal recheck pass yet — the orchestrator
        # in ``ekovideo_engine.pipeline`` has no audio-LLM step (it
        # only lives in the legacy ``video_compactor.py`` PySide
        # path). The Models tab surfaces this as "À venir" so the
        # user doesn't expect the toggle to do anything in v0.
        "available": False,
    },
]

DEFAULT_TEXT_LLM_MODEL = TEXT_LLM_MODELS[0]["id"]
DEFAULT_AUDIO_LLM_MODEL = AUDIO_LLM_MODELS[0]["id"]

LEGACY_AUDIO_LLM_MODEL_IDS: dict[str, str] = {
    # This repo id was listed by mistake in v0.17.0; mlx-community only
    # publishes the 4-bit Qwen2-Audio checkpoint.
    "mlx-community/Qwen2-Audio-7B-Instruct-8bit": "mlx-community/Qwen2-Audio-7B-Instruct-4bit",
}

LEGACY_WHISPER_MODEL_IDS: dict[str, str] = {
    "mlx-community/whisper-large-v3": "mlx-community/whisper-large-v3-mlx",
    "mlx-community/whisper-medium": "mlx-community/whisper-medium-mlx",
}

WHISPER_MODELS: list[dict] = [
    {
        "id": "mlx-community/whisper-large-v3-turbo",
        "label": "Whisper Large v3 Turbo",
        "family": "Whisper",
        "role": _TRANSCRIPTION_ROLE,
        "size_mb": 1600,
        "tier": "balanced",
        "language": ["multi"],
        "default": True,
    },
    {
        "id": "mlx-community/whisper-large-v3-mlx",
        "label": "Whisper Large v3",
        "family": "Whisper",
        "role": _TRANSCRIPTION_ROLE,
        "size_mb": 2900,
        "tier": "heavy",
        "language": ["multi"],
    },
    {
        "id": "mlx-community/distil-whisper-large-v3",
        "label": "Distil-Whisper Large v3 (rapide)",
        "family": "Whisper",
        "role": _TRANSCRIPTION_ROLE,
        "size_mb": 1500,
        "tier": "balanced",
        "language": ["en"],
    },
    {
        "id": "mlx-community/whisper-medium-mlx",
        "label": "Whisper Medium",
        "family": "Whisper",
        "role": _TRANSCRIPTION_ROLE,
        "size_mb": 1500,
        "tier": "light",
        "language": ["multi"],
    },
    {
        "id": "mlx-community/whisper-small-mlx",
        "label": "Whisper Small",
        "family": "Whisper",
        "role": _TRANSCRIPTION_ROLE,
        "size_mb": 970,
        "tier": "light",
        "language": ["multi"],
    },
    {
        "id": "mlx-community/whisper-base-mlx",
        "label": "Whisper Base",
        "family": "Whisper",
        "role": _TRANSCRIPTION_ROLE,
        "size_mb": 290,
        "tier": "light",
        "language": ["multi"],
    },
    {
        "id": "mlx-community/whisper-tiny-mlx",
        "label": "Whisper Tiny (temps réel)",
        "family": "Whisper",
        "role": _TRANSCRIPTION_ROLE,
        "size_mb": 140,
        "tier": "light",
        "language": ["multi"],
    },
    # French-fine-tuned distilled checkpoint published by bofenghuang.
    # Slightly stricter cadence on French than the multilingual large
    # but much faster than the full large-v3 model. Tagged ``balanced``
    # because it's a great default for French-only meetings.
    {
        "id": "bofenghuang/whisper-large-v3-french-distil-dec16",
        "label": "Whisper Large v3 French (distil)",
        "family": "Whisper",
        "role": _TRANSCRIPTION_ROLE,
        "size_mb": 2300,
        "tier": "balanced",
        "language": ["fr"],
    },
]


# Models the engine runs in the *multipass* repass slot — the
# higher-accuracy second pass triggered on low-confidence Whisper
# segments. Today the pipeline hardcoded ``whisper-large-v3-mlx``;
# the new Models tab exposes the alternatives so a user with the
# headroom can pin a stronger checkpoint.
MULTIPASS_MODELS: list[dict] = [
    {
        "id": "mlx-community/whisper-large-v3-mlx",
        "label": "Whisper Large v3",
        "family": "Whisper",
        "role": _MULTIPASS_ROLE,
        "size_mb": 2900,
        "tier": "heavy",
        "language": ["multi"],
        "default": True,
    },
    {
        "id": "bofenghuang/whisper-large-v3-french-distil-dec16",
        "label": "Whisper Large v3 French (distil)",
        "family": "Whisper",
        "role": _MULTIPASS_ROLE,
        "size_mb": 2300,
        "tier": "balanced",
        "language": ["fr"],
    },
    {
        "id": "mlx-community/whisper-large-v3-turbo",
        "label": "Whisper Large v3 Turbo (rapide)",
        "family": "Whisper",
        "role": _MULTIPASS_ROLE,
        "size_mb": 1600,
        "tier": "balanced",
        "language": ["multi"],
    },
]


# Pyannote pipelines the engine wires for diarisation + embedding.
# Surfaced read-only in the Models tab: not user-selectable today —
# the pipeline picks the first one the user's HF token can access.
# The Models tab uses these entries to render a "Diarisation"
# section with an explanation + a link to accept the gated repo
# licences when the model isn't yet downloaded.
DIARISATION_MODELS: list[dict] = [
    {
        "id": "pyannote/speaker-diarization-community-1",
        "label": "pyannote · speaker-diarization-community-1",
        "family": "Pyannote",
        "role": _DIARISATION_ROLE,
        "size_mb": 100,
        "tier": "balanced",
        "language": ["multi"],
        "default": True,
        "gated": True,
    },
    {
        "id": "pyannote/speaker-diarization-3.1",
        "label": "pyannote · speaker-diarization-3.1 (fallback)",
        "family": "Pyannote",
        "role": _DIARISATION_ROLE,
        "size_mb": 100,
        "tier": "balanced",
        "language": ["multi"],
        "gated": True,
    },
    {
        "id": "pyannote/embedding",
        "label": "pyannote · embedding (reconnaissance vocale)",
        "family": "Pyannote",
        "role": _EMBEDDING_ROLE,
        "size_mb": 90,
        "tier": "light",
        "language": ["multi"],
        "gated": True,
    },
]

DEFAULT_WHISPER_MODEL = WHISPER_MODELS[0]["id"]
DEFAULT_MULTIPASS_MODEL = MULTIPASS_MODELS[0]["id"]


def canonical_multipass_model_id(model_id: str) -> str:
    raw = (model_id or "").strip()
    if not raw:
        return DEFAULT_MULTIPASS_MODEL
    return LEGACY_WHISPER_MODEL_IDS.get(raw, raw)


def canonical_whisper_model_id(model_id: str) -> str:
    raw = (model_id or "").strip()
    if not raw:
        return DEFAULT_WHISPER_MODEL
    return LEGACY_WHISPER_MODEL_IDS.get(raw, raw)


def canonical_audio_llm_model_id(model_id: str) -> str:
    raw = (model_id or "").strip()
    if not raw:
        return DEFAULT_AUDIO_LLM_MODEL
    return LEGACY_AUDIO_LLM_MODEL_IDS.get(raw, raw)


def text_llm_label_for(model_id: str) -> str:
    for entry in TEXT_LLM_MODELS:
        if entry["id"] == model_id:
            return entry["label"]
    return model_id


def audio_llm_label_for(model_id: str) -> str:
    model_id = canonical_audio_llm_model_id(model_id)
    for entry in AUDIO_LLM_MODELS:
        if entry["id"] == model_id:
            return entry["label"]
    return model_id


_TRANSCRIPT_SPEAKER_LINE_RE = re.compile(
    r"^\[(?P<speaker>[^\]\n]+)\]\s+\((?P<timestamp>\d{1,2}:\d{2}(?::\d{2})?)\)\s*(?P<text>.*)$"
)


def _fold_for_matching(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value or "")
    normalized = "".join(c for c in normalized if not unicodedata.combining(c))
    return normalized.lower()


def _name_tokens(value: str) -> list[str]:
    return re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ'’-]{2,}", value or "")


def _timestamp_text_to_seconds(value: str) -> float:
    parts = [int(p) for p in (value or "0:00").split(":")]
    if len(parts) == 2:
        return float(parts[0] * 60 + parts[1])
    if len(parts) == 3:
        return float(parts[0] * 3600 + parts[1] * 60 + parts[2])
    return 0.0


def format_seconds_for_clip(seconds: float) -> str:
    """Render a duration as a short ``ffmpeg``-compatible string.

    ``-ss`` / ``-to`` accept plain decimal seconds; we trim trailing
    zeros so the audit log line shows ``"42"`` instead of
    ``"42.00"`` for the common integer case. Negative inputs clamp
    to 0 — every caller (clip extraction, repass timestamps) is
    upstream-protected against negative ranges but we belt-and-
    braces it here too.
    """
    seconds = max(0.0, float(seconds))
    return f"{seconds:.2f}".rstrip("0").rstrip(".")


def _person_aliases_from_terms(glossary_terms: list[str]) -> dict[str, set[str]]:
    """
    Extract likely person names from glossary terms.

    "Arnaud Maire" becomes {"arnaud": {"maire"}}. We intentionally
    only use multi-token terms so a bare glossary word like "Mollie"
    cannot affect speaker identity validation.
    """
    people: dict[str, set[str]] = {}
    for term in glossary_terms or []:
        tokens = _name_tokens(term)
        if len(tokens) < 2:
            continue
        first = _fold_for_matching(tokens[0])
        if not first:
            continue
        aliases = {_fold_for_matching(tok) for tok in tokens[1:] if len(tok) >= 3}
        if aliases:
            people.setdefault(first, set()).update(aliases)
    return people


def filter_speaker_names_by_context(
    transcript_text: str,
    speakers: dict[str, str],
    glossary_terms: list[str],
    *,
    neighbor_window_seconds: float = 10.0,
) -> dict[str, str]:
    """
    Drop risky global speaker renames when a diarization label appears
    to cover multiple people after a phone transfer.

    Real failure this guards against:
      - SPEAKER_01 says "Sainte-Ferrande, Philippe, bonjour" at reception.
      - The same diarization label is later assigned to the transferred
        contact, who answers after "Monsieur Maire".
      - A global SPEAKER_01 -> Philippe rename then mislabels Arnaud Maire.

    If the glossary contains a full person name such as "Arnaud Maire",
    and a speaker label mapped to another first name replies immediately
    after being addressed by that surname, we leave the label unrenamed
    instead of applying a wrong name everywhere.
    """
    if not transcript_text or not speakers:
        return dict(speakers or {})

    people = _person_aliases_from_terms(glossary_terms)
    if not people:
        return dict(speakers)

    rows: list[dict] = []
    for line in transcript_text.splitlines():
        match = _TRANSCRIPT_SPEAKER_LINE_RE.match(line.strip())
        if not match:
            continue
        rows.append(
            {
                "speaker": match.group("speaker").strip(),
                "time": _timestamp_text_to_seconds(match.group("timestamp")),
                "text": match.group("text") or "",
            }
        )
    if not rows:
        return dict(speakers)

    filtered = dict(speakers)
    for idx, row in enumerate(rows):
        speaker_id = row["speaker"]
        mapped_name = filtered.get(speaker_id)
        if not mapped_name:
            continue
        mapped_first_tokens = _name_tokens(mapped_name)
        if not mapped_first_tokens:
            continue
        mapped_first = _fold_for_matching(mapped_first_tokens[0])

        # Look at the immediately surrounding turns. If another speaker
        # addresses this turn with a surname from the glossary that belongs
        # to a different first name, the diarization label is ambiguous.
        for other in rows[max(0, idx - 2) : min(len(rows), idx + 3)]:
            if other is row or other["speaker"] == speaker_id:
                continue
            if abs(float(other["time"]) - float(row["time"])) > neighbor_window_seconds:
                continue
            folded_text = _fold_for_matching(other["text"])
            for first, aliases in people.items():
                if first == mapped_first:
                    continue
                for alias in aliases:
                    if re.search(rf"\b(?:m\.|monsieur)\s+(?:le\s+)?{re.escape(alias)}\b", folded_text):
                        filtered.pop(speaker_id, None)
                        break
                if speaker_id not in filtered:
                    break
            if speaker_id not in filtered:
                break
    return filtered

# Whisper's `--initial-prompt` is fed to the decoder as a fake prefix; the
# model truncates anything beyond ~224 tokens, so the prompt must be tight
# and front-load the most important vocabulary.
INITIAL_PROMPT_MAX_CHARS = 800


def transcript_output_ext(output_format: str) -> str:
    fmt = (output_format or "txt").strip().lower()
    if fmt == "all":
        return "txt"
    if fmt in {"txt", "srt", "vtt", "json", "tsv"}:
        return fmt
    return "txt"


def default_transcript_path(in_path: str, out_dir: str, suffix: str, output_format: str) -> str:
    source = Path(in_path)
    safe_suffix = suffix.strip()
    ext = transcript_output_ext(output_format)
    base = Path(out_dir) / f"{source.stem}{safe_suffix}.{ext}"
    out = base
    i = 1
    while out.exists():
        out = Path(out_dir) / f"{source.stem}{safe_suffix}_{i}.{ext}"
        i += 1
    return str(out)


_GENERIC_OPENING_RE = re.compile(
    r"^(bonjour|bonsoir|salut|merci|ok|alors|donc|du coup|euh|hum|oui|non|très bien|tres bien)[, .!?]*",
    re.IGNORECASE,
)
_TITLE_NOISE_RE = re.compile(
    r"^(on va parler de|je vais|on va|nous allons|aujourd'hui|aujourd’hui|là on va|c'est parti pour|c’est parti pour|le sujet de la réunion c'est|le sujet de la reunion c'est|cette réunion porte sur|cette reunion porte sur)\s+",
    re.IGNORECASE,
)
_BAD_TITLE_START_RE = re.compile(
    r"^(j['’ ]|je\b|tu\b|vous\b|nous\b|on\b|il faut\b|c'est\b|c’est\b|voilà\b|alors\b|du coup\b|là\b)",
    re.IGNORECASE,
)
_SPEAKER_PREFIX_RE = re.compile(r"^(\[[^\]]+\]|SPEAKER[_ -]?\d+|INTERVENANT[_ -]?\d+)\s*:?\s*", re.IGNORECASE)
_TIMESTAMP_RE = re.compile(r"^\[?\d{1,2}:\d{2}:\d{2}(?:[.,]\d+)?\s*-->|^\d+$")


def sanitize_filename_stem(stem: str, fallback: str = "Transcription") -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', " ", stem or "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .-_")
    if len(cleaned) > 80:
        cleaned = cleaned[:80].rsplit(" ", 1)[0].strip(" .-_")
    return cleaned or fallback


def is_useful_transcript_title(title: str, fallback_stem: str = "") -> bool:
    candidate = sanitize_filename_stem(title or "", "")
    if not candidate:
        return False
    fallback = sanitize_filename_stem(fallback_stem or "", "")
    if fallback and candidate.casefold() == fallback.casefold():
        return False
    words = candidate.split()
    if len(words) < 3 or len(words) > 12:
        return False
    if len(candidate) < 12 or len(candidate) > 80:
        return False
    lowered = candidate.casefold()
    if _BAD_TITLE_START_RE.search(lowered):
        return False
    # A library title should name the meeting topic, not quote a full
    # utterance from the transcript.
    if re.search(r"\b(j['’]ai|je suis|je vais|j'aimerais|j’aimerais|on va|nous allons)\b", lowered):
        return False
    return True


def _plain_transcript_lines(transcript_text: str) -> list[str]:
    text = transcript_text or ""
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            text = str(payload.get("text") or "")
    except Exception:
        pass

    lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or _TIMESTAMP_RE.search(line):
            continue
        line = _SPEAKER_PREFIX_RE.sub("", line).strip()
        line = re.sub(r"\[[0-9:. ,>\-]+\]", "", line).strip()
        if line:
            lines.append(line)
    return lines


def suggest_transcript_stem(transcript_text: str, fallback_stem: str) -> str:
    fallback = sanitize_filename_stem(fallback_stem or "Transcription")
    joined = " ".join(_plain_transcript_lines(transcript_text))[:5000]
    if not joined:
        return fallback

    chunks = [
        part.strip(" -–—:;,.!?")
        for part in re.split(r"[.!?\n]+", joined)
        if part.strip(" -–—:;,.!?")
    ]
    topic_words = {
        "présentation", "presentation", "outil", "outils", "rh", "module",
        "projet", "client", "atelier", "formation", "demo", "démo",
        "planning", "budget", "process", "workflow", "intégration", "integration",
        "facture", "factures", "fournisseur", "fournisseurs", "comptabilité",
        "comptable", "paie", "payfit", "odoo", "automatisation", "import",
    }

    best = ""
    best_score = -1
    for chunk in chunks[:30]:
        candidate = _GENERIC_OPENING_RE.sub("", chunk).strip(" ,.")
        while True:
            cleaned = _TITLE_NOISE_RE.sub("", candidate).strip(" ,.")
            if cleaned == candidate:
                break
            candidate = cleaned
        words = candidate.split()
        if len(words) < 3 or len(candidate) < 12:
            continue
        if len(words) > 14:
            candidate = " ".join(words[:14]).strip(" ,.")
            words = candidate.split()
        if not is_useful_transcript_title(candidate):
            continue

        lowered = {word.strip(" ,.;:!?()[]{}'\"").lower() for word in words}
        score = 0
        score += min(len(words), 10)
        score += 8 * len(lowered & topic_words)
        if any(word[:1].isupper() for word in words[1:]):
            score += 4
        if candidate.lower().startswith(("bonjour", "merci", "ok", "oui", "non")):
            score -= 8
        if score > best_score:
            best_score = score
            best = candidate

    if best_score < 10:
        return fallback
    return sanitize_filename_stem(best, fallback)


def build_audio_extract_cmd(
    ffmpeg_path: str,
    in_path: str,
    wav_path: str,
    speech_enhance: bool = True,
    ss: str | None = None,
    to: str | None = None,
    audio_profile: str = AUDIO_PROFILE_STANDARD,
) -> list[str]:
    cmd = [ffmpeg_path, "-y", "-hide_banner", "-loglevel", "error"]

    if ss is not None:
        cmd += ["-ss", ss]
    if to is not None:
        cmd += ["-to", to]

    cmd += ["-i", in_path, "-vn"]

    if speech_enhance:
        cmd += ["-af", ",".join(select_transcription_audio_filters(audio_profile))]

    cmd += [
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        "-progress",
        "pipe:1",
        "-nostats",
        wav_path,
    ]
    return cmd


def structured_initial_prompt(
    context: str,
    *,
    expected_speaker_names: list[str] | None = None,
    meeting_context: str = "",
) -> str:
    """
    Whisper interprets `--initial-prompt` as the imagined previous
    segment of the same audio. The decoder uses it to bias the
    language model — but only when the prompt reads like *actual
    French dialogue*. A bare list of names ("Ekonum, MAIA, RGPD")
    underperforms compared to a short conversational sentence that
    *uses* the terms inside their natural syntactic context.

    We split the user's glossary into individual terms, then weave
    them into a meeting-style monologue. The exact phrasing has been
    A/B tested against real Whisper outputs — verbs like "discute",
    "intègre", "rappelle" trigger the decoder to expect proper nouns
    in the immediately following slot, which is exactly where the
    glossary terms appear.

    Terms are quoted with French guillemets when they contain
    spaces (multi-word entities like "CVR Contrôles") so Whisper
    treats them as cohesive units.

    When ``expected_speaker_names`` is provided (typically the
    values of ``JobRequest.speaker_overrides``), a "Participants : …"
    clause is prepended so the decoder primes the same orthography
    every time the speaker introduces themselves. This is what saves
    "Adèle" from becoming "Adel" / "Adèle" / "Ali" across the meeting.

    ``meeting_context`` is an optional short sentence the caller can
    use to tell Whisper what the meeting is about ("Réunion sur Odoo
    et la migration Visiotech"). Whisper uses it as semantic prior;
    omit it if the glossary already conveys the topic.

    Length is still capped at INITIAL_PROMPT_MAX_CHARS so the prompt
    fits inside Whisper's ~224-token context window for the prompt.
    """
    raw = (context or "").strip()

    # Parse the glossary the same way the post-processor does, so
    # both passes see the same canonical term list.
    terms: list[str] = []
    if raw:
        try:
            # Local import — `glossary_postprocess` is a sibling module
            # but stays optional for callers that only use plain Whisper.
            from glossary_postprocess import parse_glossary_terms

            terms = parse_glossary_terms(raw)
        except Exception:
            terms = [t.strip() for t in re.split(r"[,\n;]+", raw) if t.strip()]

    # De-dupe + filter the speaker names. We don't enforce any
    # particular casing — the caller usually passes user-typed first
    # names ("Robin", "David"), and Whisper benefits from seeing the
    # exact orthography it should reproduce.
    speakers: list[str] = []
    seen_speakers: set[str] = set()
    for name in expected_speaker_names or []:
        cleaned = (name or "").strip()
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen_speakers:
            continue
        seen_speakers.add(key)
        speakers.append(cleaned)

    if not raw and not speakers and not meeting_context.strip():
        return ""

    if not terms and not speakers and not meeting_context.strip():
        flat = " ".join(raw.split())
        return f"Réunion en français. Termes attendus : {flat}."

    chunks: list[str] = []

    ctx_line = (meeting_context or "").strip()
    if ctx_line:
        # Add a trailing dot when the caller didn't bother — keeps
        # the rendered prompt looking like real prose.
        if not ctx_line.endswith((".", "!", "?")):
            ctx_line = ctx_line + "."
        chunks.append(ctx_line)
    else:
        chunks.append("Réunion professionnelle en français.")

    if speakers:
        chunks.append(
            "Participants : " + _join_fr(speakers[:6]) + "."
        )

    if terms:
        # Multi-word terms get guillemets so Whisper hears them as one
        # phrase ("CVR Contrôles" not "CVR" then "Contrôles" guessed
        # independently).
        def _quote(term: str) -> str:
            return f"« {term} »" if " " in term else term

        quoted = [_quote(t) for t in terms]
        # Distribute terms across a handful of natural-sounding clauses.
        # Each clause names ~2-3 terms, prefixed with a verb that the
        # decoder expects to precede a proper-noun cluster in French.
        head, tail = quoted[:12], quoted[12:]

        def _take(n: int) -> list[str]:
            return [head.pop(0) for _ in range(min(n, len(head)))]

        if head:
            chunks.append(
                "Aujourd'hui je vous présente " + _join_fr(_take(3)) + "."
            )
        if head:
            chunks.append(
                "Nous travaillons régulièrement avec " + _join_fr(_take(3)) + "."
            )
        if head:
            chunks.append(
                "Je vais aussi parler de " + _join_fr(_take(3)) + "."
            )
        if head:
            chunks.append(
                "On évoquera également " + _join_fr(_take(len(head))) + "."
            )

        if tail:
            chunks.append("D'autres noms à respecter : " + ", ".join(tail) + ".")

    out = " ".join(chunks).strip()
    if len(out) > INITIAL_PROMPT_MAX_CHARS:
        out = out[:INITIAL_PROMPT_MAX_CHARS].rsplit(" ", 1)[0]
    return out


def _join_fr(items: list[str]) -> str:
    """Comma-separated list with the French Oxford-style 'et'."""
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} et {items[1]}"
    return ", ".join(items[:-1]) + f" et {items[-1]}"


def build_mlx_whisper_cmd(
    mlx_whisper_path: str,
    audio_path: str,
    output_path: str,
    model: str,
    language: str = "fr",
    output_format: str = "txt",
    initial_prompt: str = "",
    condition_on_previous_text: bool = False,
    clip_timestamps: str = "",
    word_timestamps: bool = False,
) -> list[str]:
    out = Path(output_path)
    fmt = (output_format or "txt").strip().lower()
    cmd = [
        mlx_whisper_path,
        audio_path,
        "--model",
        canonical_whisper_model_id(model),
        "-f",
        fmt,
        "--output-dir",
        str(out.parent),
        "--output-name",
        out.stem,
    ]

    lang = (language or "fr").strip().lower()
    if lang and lang != "auto":
        cmd += ["--language", lang]

    prompt = (initial_prompt or "").strip()
    if prompt:
        cmd += ["--initial-prompt", prompt]

    # Long meetings often start with silence, room noise, or screen-share
    # sounds. If Whisper conditions each window on the previous one, a single
    # hallucinated "..." can poison the full recording.
    cmd += ["--condition-on-previous-text", "True" if condition_on_previous_text else "False"]

    if word_timestamps:
        # Word-level timestamps power per-word speaker attribution
        # downstream — when pyannote says the speaker switches mid-
        # segment we can split the segment on the actual word boundary
        # instead of guessing. Costs ~5 % runtime on top of the regular
        # Whisper pass.
        cmd += ["--word-timestamps", "True"]

    clips = (clip_timestamps or "").strip()
    if clips:
        cmd += ["--clip-timestamps", clips]

    return cmd


_TEXT_SIGNAL_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]")
_ONLY_PUNCTUATION_RE = re.compile(r"^[\s.…·!?;:,\\/_|()\\[\\]{}'\"`+-]+$")
_SPACE_RE = re.compile(r"\s+")
_PHONE_HOLD_MARKERS = (
    "merci de votre appel",
    "merci de rester en ligne",
    "patienter quelques instants",
    "un correspondant va vous repondre",
    "votre appel est en attente",
    "cours de transfert",
    "nous allons y donner suite",
    "bienvenue a",
)


def _normalize_segment_text(text: str) -> str:
    normalized = (text or "").strip().lower()
    normalized = normalized.replace("’", "'").replace("…", "...")
    normalized = re.sub(r"[^a-z0-9à-öø-ÿ']+", " ", normalized)
    return _SPACE_RE.sub(" ", normalized).strip()


def is_phone_hold_boilerplate_text(text: str) -> bool:
    """
    Detect stock IVR / hold-line speech.

    Silero VAD cannot remove these passages because they are spoken
    audio, not silence. For meeting/call notes they are almost always
    noise, and they poison downstream diarization + speaker naming.
    """
    normalized = _fold_for_matching(_normalize_segment_text(text))
    if not normalized:
        return False
    hits = sum(1 for marker in _PHONE_HOLD_MARKERS if marker in normalized)
    if hits >= 2:
        return True
    if normalized.count("merci de rester en ligne") >= 2:
        return True
    if normalized.count("un correspondant va vous repondre") >= 2:
        return True
    return False


def is_hallucinated_whisper_segment(segment: dict) -> bool:
    text = str(segment.get("text", "")).strip()
    if not text:
        return True
    if _ONLY_PUNCTUATION_RE.fullmatch(text):
        return True
    if not _TEXT_SIGNAL_RE.search(text):
        return True

    normalized = _normalize_segment_text(text)
    if is_phone_hold_boilerplate_text(normalized):
        return True
    if normalized in {
        "sous titrage st 501",
        "sous titres realises par la communaute d'amara org",
        "merci d'avoir regarde cette video",
    }:
        return True
    if normalized.startswith("sous titrage"):
        return True

    compression_ratio = float(segment.get("compression_ratio") or 0.0)
    if compression_ratio >= 2.4 and len(normalized) <= 4:
        return True
    return False


def clean_whisper_segments(segments: Iterable[dict]) -> list[dict]:
    """
    Drop obvious Whisper hallucinations without rewriting real speech.

    This targets the common local-Whisper failure mode on long recordings:
    silence or room noise produces "..." or stock subtitle artefacts, then
    `condition_on_previous_text=True` propagates that failure for minutes.
    """
    out: list[dict] = []
    last_norm = ""
    repeat_count = 0

    for seg in segments:
        if is_hallucinated_whisper_segment(seg):
            continue

        cleaned = dict(seg)
        cleaned["text"] = str(cleaned.get("text", "")).strip()
        normalized = _normalize_segment_text(cleaned["text"])
        if normalized and normalized == last_norm:
            repeat_count += 1
        else:
            last_norm = normalized
            repeat_count = 1

        # Keep the first repeated phrase because people do repeat themselves;
        # discard long decoder loops.
        if repeat_count > 2:
            continue
        out.append(cleaned)
    return out


# Short French function words / pronouns / forms that show up
# capitalised at sentence-start but never qualify as proper nouns.
# Kept folded (NFD-ascii lowercase) for the matcher.
_FR_SENTENCE_START_NOISE: frozenset[str] = frozenset(
    {
        "a", "ah", "alors", "apres", "as", "au", "aussi", "autre", "autres",
        "avec", "bah", "bien", "bon", "bonjour", "bonsoir", "ca", "car",
        "ce", "ces", "cet", "cette", "chez", "comme", "comment", "d",
        "dans", "de", "des", "deux", "dix", "donc", "donne", "donner",
        "dont", "du", "elle", "elles", "en", "encore", "enfin", "entre",
        "est", "et", "etre", "eu", "fait", "faire", "faut", "il", "ils",
        "j", "je", "juste", "l", "la", "le", "les", "leur", "leurs", "lui",
        "ma", "mais", "merci", "mes", "mois", "mon", "moi", "n", "ne",
        "non", "nos", "notre", "nous", "ok", "on", "ou", "oui", "par",
        "pas", "peu", "peut", "plus", "pour", "pourquoi", "quand", "que",
        "quel", "quelle", "qu", "qui", "quoi", "s", "sa", "sans", "se",
        "ses", "si", "soit", "son", "sont", "sous", "sur", "t", "ta",
        "tes", "ton", "tout", "tous", "toute", "toutes", "tres", "trois",
        "tu", "un", "une", "voici", "voila", "votre", "vos", "vous", "y",
        # Common verb forms that begin sentences after a clause break and
        # would otherwise sneak in (capitalised by Whisper after a period).
        "c", "c'est", "qu'est", "qu'on", "qu'il", "qu'elle",
    }
)


_PROPER_NOUN_TOKEN_RE = re.compile(
    # Capitalised word (with optional intra-word apostrophe / hyphen)
    # made of at least 2 letters. Accents accepted. We do not require
    # the next character to be lowercase — proper nouns sometimes
    # appear at end-of-sentence too.
    r"\b([A-ZÀ-ÖØ-Þ][a-zà-öø-ÿĀ-ſ]{1,}(?:[-’'][A-ZÀ-ÖØ-Þa-zà-öø-ÿ]+)*)\b"
)


def _fold_term(value: str) -> str:
    """Lowercased + accent-stripped form used as a dedupe / stopword key."""
    folded = _fold_for_matching(value or "").lower()
    return folded.strip()


def extract_new_proper_nouns_from_segments(
    segments: Iterable[dict],
    *,
    existing_terms: Iterable[str] = (),
    min_occurrences: int = 2,
    max_terms: int = 20,
    min_length: int = 3,
) -> list[str]:
    """
    Mine the first-pass Whisper transcript for proper-noun candidates
    we can fold back into the prompt of subsequent passes (multipass,
    boundary multipass, LLM).

    "Hot prompt cycling" in the PR D sense: instead of re-running
    Whisper on 5-minute chunks (which would 5× the wall time and
    blow out the model's prompt-cache), we let the first pass
    discover entities — repeated capitalised tokens — and feed them
    back into the *next* pass's glossary. Because the multipass /
    boundary / per-speaker passes all build their ``--initial-prompt``
    from ``self.request.glossary_terms``, enriching that list in
    place is enough to propagate the discovery.

    A token qualifies when:
      • It looks like a proper noun (initial capital, ≥ ``min_length``
        letters, optional intra-word apostrophe / hyphen).
      • It is *not* a French function word capitalised at sentence
        start (``_FR_SENTENCE_START_NOISE``).
      • It appears ``min_occurrences`` times or more across the
        transcript (one-off mentions are too noisy to risk).
      • It is not already present in ``existing_terms`` (case-
        insensitive, accent-insensitive — so adding "Castel" is
        skipped when the glossary already has "Castel").

    Returns up to ``max_terms`` candidates ordered by descending
    occurrence (then by first-occurrence order to keep results
    deterministic on ties).
    """
    existing_keys = {
        _fold_term(term)
        for term in (existing_terms or [])
        if _fold_term(term)
    }

    counts: dict[str, int] = {}
    first_seen: dict[str, int] = {}
    canonical: dict[str, str] = {}

    order_idx = 0
    for seg in segments or []:
        text = str(seg.get("text", "") or "")
        if not text:
            continue
        for match in _PROPER_NOUN_TOKEN_RE.finditer(text):
            token = match.group(1)
            if len(token) < min_length:
                continue
            key = _fold_term(token)
            if not key:
                continue
            # Strip the apostrophe-suffix forms ("l'Adèle" → "Adèle")
            # by re-running the matcher only on the head form. This
            # also covers the elision case "d'Adèle".
            if key in _FR_SENTENCE_START_NOISE:
                continue
            if key in existing_keys:
                continue
            counts[key] = counts.get(key, 0) + 1
            if key not in first_seen:
                first_seen[key] = order_idx
                # Keep the first surface form we saw (likely mid-
                # sentence, so capitalisation is meaningful rather
                # than an artefact of sentence-start).
                canonical[key] = token
            order_idx += 1

    qualifying = [
        (key, count)
        for key, count in counts.items()
        if count >= min_occurrences
    ]
    qualifying.sort(key=lambda kv: (-kv[1], first_seen[kv[0]]))

    return [canonical[key] for key, _ in qualifying[:max_terms]]


# The LLM post-process used to ask the model for {title, speakers,
# corrections, uncertain_passages} in a single 1800-token JSON blob.
# In practice mlx_lm + Mistral-7B-4bit drifts past ~700 tokens of JSON
# and emits a missing comma every other run, killing the whole pass.
# Splitting into two short calls makes each one reliable enough to
# parse, and lets the title/speakers pass succeed even if the
# corrections pass fails.

_LLM_TITLE_SPEAKERS_SCRIPT = '''
import json
import sys

try:
    from mlx_lm import load, generate
except ImportError:
    print(json.dumps({"error": "mlx-lm not installed"}))
    sys.exit(1)

model_path = sys.argv[1]
transcript_path = sys.argv[2]
glossary = sys.argv[3] if len(sys.argv) > 3 else ""

try:
    with open(transcript_path, "r", encoding="utf-8") as f:
        text = f.read()
    # Title/speakers can be guessed from the first 6-8 minutes of dialogue;
    # we cap the input so we always finish in a reasonable time and never
    # overflow the model's context.
    text = text[:18000]

    model, tokenizer = load(model_path)

    prompt = f"""[INST] Tu es un assistant qui résume de courtes transcriptions de réunions en français.

Tâche : à partir de la transcription ci-dessous, propose
- un titre court et descriptif (5 à 10 mots, sans guillemets)
- pour chaque SPEAKER_XX présent, son prénom probable s'il est mentionné explicitement dans la transcription
- les termes techniques / noms propres métier réellement présents ou très probablement présents

Règles strictes :
- Réponds UNIQUEMENT par un objet JSON valide, sans texte avant/après, sans markdown.
- Chaque clé "speakers" est un identifiant SPEAKER_XX et la valeur est un prénom (ou "" si tu n'es pas sûr).
- N'invente JAMAIS un prénom : laisse la chaîne vide en cas de doute.
- Si un même SPEAKER_XX semble contenir plusieurs personnes (standard téléphonique, transfert, mauvaise diarisation), laisse sa valeur vide.
- Ne renomme pas un SPEAKER_XX d'après une seule formule d'accueil si le même label répond ensuite à un autre nom.
- Ne mets que les SPEAKER_XX présents dans la transcription.
- "technical_terms" contient 0 à 20 termes, sans doublons, orthographiés proprement.
- Priorise les termes du vocabulaire métier quand ils apparaissent même phonétiquement.
- Le titre doit décrire le sujet global de toute la réunion, pas recopier une phrase locale.
- Le titre doit être nominal, sans première personne : pas de titre commençant par "J'ai", "Je", "On va", "Nous allons".
- Mauvais titre : "J'ai pris une facture fournisseur basique, c'est PayFit".
- Bon titre : "Traitement des factures fournisseurs avec PayFit".

Schéma exact attendu :
{{"title": "...", "speakers": {{"SPEAKER_00": "..."}}, "technical_terms": ["Odoo"]}}

Vocabulaire métier prioritaire (orthographe à respecter partout) :
{glossary or "(aucun)"}

Transcription :
{text}
[/INST]"""

    # Short cap — title + speakers + terms fit comfortably in 400 tokens.
    response = generate(model, tokenizer, prompt=prompt, max_tokens=450, verbose=False)
    print(response)

except Exception as e:
    print(json.dumps({"error": str(e)}))
    sys.exit(2)
'''


# The corrections pass is intentionally asked in markdown (not JSON) so a
# missing comma or quote doesn't burn the entire result. The Python-side
# parser walks the output line by line.
_LLM_CORRECTIONS_SCRIPT = '''
import json
import sys

try:
    from mlx_lm import load, generate
except ImportError:
    print(json.dumps({"error": "mlx-lm not installed"}))
    sys.exit(1)

model_path = sys.argv[1]
transcript_path = sys.argv[2]
glossary = sys.argv[3] if len(sys.argv) > 3 else ""
# Optional Odoo context blob (chatter summary of the linked CRM /
# project record). Surfaced to the LLM as a secondary priority
# below the explicit glossary, but above pure pattern guessing.
odoo_context = sys.argv[4] if len(sys.argv) > 4 else ""

try:
    with open(transcript_path, "r", encoding="utf-8") as f:
        text = f.read()
    text = text[:30000]

    model, tokenizer = load(model_path)

    context_section = ""
    if odoo_context.strip():
        # Cap the blob so a noisy chatter can't blow the prompt
        # budget. 1 500 chars leaves room for the glossary + the
        # transcript chunk.
        context_section = (
            "Contexte de la réunion (Odoo, à utiliser pour disambiguer les noms propres et les termes métier) :\\n"
            + odoo_context.strip()[:1500]
            + "\\n\\n"
        )

    prompt = f"""[INST] Tu es un relecteur expert de transcriptions de réunions professionnelles françaises.

Repère uniquement les erreurs de transcription audio, pas les maladresses orales :
1. Les passages où Whisper a probablement mal entendu un terme métier, une marque, un logiciel, un nom propre ou un mot français évident.
2. Les passages clairement douteux que tu signales SANS corriger.

Règles strictes :
- Ne corrige jamais la grammaire, le style, les répétitions, les tournures familières ou les hésitations.
- Ne reformule jamais une phrase correcte : corrige seulement une erreur phonétique locale.
- Une correction doit être courte, remplacer un extrait exact, et rester très proche phonétiquement.
- Le vocabulaire métier ci-dessous est prioritaire : si un passage ressemble phonétiquement à un terme listé, propose ce terme.
- Le contexte Odoo (s'il est fourni) est une aide secondaire pour disambiguer les noms propres et acronymes — ne corrige pas un terme uniquement parce qu'il est absent du contexte.
- Tu peux corriger un mot absurde vers un mot français évident si le contexte l'impose.
- Format de sortie : markdown, exactement le format ci-dessous, rien d'autre.
- Si rien à corriger : écris "Aucune correction." puis "Aucun doute." et stop.
- Maximum 10 corrections + 10 doutes.

Exemples de corrections attendues :
- "Infomaniac" -> "Infomaniak" si le contexte parle d'hébergement, mail ou nom de domaine.
- "glasculer" -> "basculer" si le contexte parle de changer de système ou de domaine.
- "Tchadjepté" -> "Chat GPT" si le contexte parle d'IA.

Exemples interdits :
- "Je n'ai pas de son." -> "Je n'ai pas de sonne." (faux)
- "t'as échangé avec lui ?" -> "t'as-tu échangé avec lui ?" (correction de style)
- "version brouillon" -> "version brouillonne" (correction grammaticale inutile)

Format exact (chaque entrée sur 2 lignes) :

# Corrections
- [00:12:34] "texte exact dans la transcription" -> "texte corrigé" (raison: …)
- [00:14:02] "..." -> "..." (raison: …)

# Doutes
- [00:18:30] "passage exact douteux" (raison: …)

Vocabulaire métier (priorité absolue) :
{glossary or "(aucun)"}

{context_section}Transcription :
{text}
[/INST]"""

    response = generate(model, tokenizer, prompt=prompt, max_tokens=900, verbose=False)
    print(response)

except Exception as e:
    print(json.dumps({"error": str(e)}))
    sys.exit(2)
'''


def build_llm_title_cmd(
    venv_python_path: str,
    model_path: str,
    transcript_path: str,
    glossary: str = "",
) -> list[str]:
    return [
        venv_python_path,
        "-c",
        _LLM_TITLE_SPEAKERS_SCRIPT,
        model_path,
        transcript_path,
        glossary,
    ]


def build_llm_corrections_cmd(
    venv_python_path: str,
    model_path: str,
    transcript_path: str,
    glossary: str = "",
    context: str = "",
) -> list[str]:
    """Build the corrections-LLM invocation.

    ``context`` carries the Odoo chatter summary (or any other
    meeting-level context blob the pipeline wants the model to see).
    Empty string is the default — the script's prompt simply omits
    the "Contexte de la réunion" section in that case.
    """
    return [
        venv_python_path,
        "-c",
        _LLM_CORRECTIONS_SCRIPT,
        model_path,
        transcript_path,
        glossary,
        context,
    ]


# Kept as an alias for callers that already use the old name; points at
# the title/speakers script which is what the worker calls first.
_LLM_POST_PROCESS_SCRIPT = _LLM_TITLE_SPEAKERS_SCRIPT


def build_llm_cmd(
    venv_python_path: str,
    model_path: str,
    transcript_path: str,
    glossary: str = "",
) -> list[str]:
    """Backwards-compatible alias for the title/speakers call."""
    return build_llm_title_cmd(venv_python_path, model_path, transcript_path, glossary)


def parse_llm_title_speakers(stdout: str) -> dict:
    """
    Parse a JSON {"title": ..., "speakers": {...}, "technical_terms": [...]}
    blob produced by the title/context script. Tolerates leading prose and
    missing trailing delimiters so a slightly drifting model still gives us
    something.
    """
    text = (stdout or "").strip()
    if not text:
        return {}
    # Direct parse first (the script prints raw JSON when it succeeds).
    try:
        payload = json.loads(text.splitlines()[-1])
        if isinstance(payload, dict) and "error" in payload:
            return {}
        if isinstance(payload, dict):
            return _coerce_title_speakers(payload)
    except Exception:
        pass

    # Fallback: extract the first balanced {...} substring.
    repaired = _extract_first_json_object(text)
    if repaired is None:
        return {}
    try:
        payload = json.loads(repaired)
    except Exception:
        # Best effort: trim trailing comma before closing brace.
        cleaned = re.sub(r",(\s*[}\]])", r"\1", repaired)
        try:
            payload = json.loads(cleaned)
        except Exception:
            return {}
    return _coerce_title_speakers(payload if isinstance(payload, dict) else {})


def _extract_first_json_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _coerce_title_speakers(payload: dict) -> dict:
    title = str(payload.get("title") or "").strip().strip('"').strip()
    speakers_raw = payload.get("speakers") or {}
    speakers: dict[str, str] = {}
    if isinstance(speakers_raw, dict):
        for key, value in speakers_raw.items():
            key_str = str(key).strip()
            value_str = str(value).strip().strip('"').strip()
            if key_str.startswith("SPEAKER_") and value_str:
                speakers[key_str] = value_str
    terms_raw = payload.get("technical_terms") or payload.get("terms") or []
    terms: list[str] = []
    if isinstance(terms_raw, list):
        for value in terms_raw:
            term = str(value).strip().strip('"').strip()
            if term and term not in terms:
                terms.append(term)
    return {"title": title, "speakers": speakers, "technical_terms": terms[:30]}


_CORRECTION_LINE = re.compile(
    r'^[-*]\s*\[?(?P<ts>\d{1,2}:\d{2}(?::\d{2})?)\]?\s*'
    r'"(?P<original>[^"]+)"\s*(?:->|→|=>)\s*"(?P<replacement>[^"]+)"\s*'
    r'(?:\(raison\s*:\s*(?P<reason>[^)]*)\))?',
    re.IGNORECASE,
)
_DOUBT_LINE = re.compile(
    r'^[-*]\s*\[?(?P<ts>\d{1,2}:\d{2}(?::\d{2})?)\]?\s*'
    r'"(?P<text>[^"]+)"\s*(?:\(raison\s*:\s*(?P<reason>[^)]*)\))?',
    re.IGNORECASE,
)


def parse_llm_corrections_markdown(stdout: str) -> dict:
    """
    Parse the markdown produced by the corrections pass. Tolerant: a
    drifting model that misses a quote on one line still leaves us
    with the lines that were well-formed.
    """
    text = (stdout or "").strip()
    if not text or "Aucune correction" in text and "Aucun doute" in text:
        return {"corrections": [], "uncertain_passages": []}

    corrections: list[dict] = []
    uncertain: list[dict] = []
    section: str | None = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        lowered = line.lower()
        if lowered.startswith("# correction"):
            section = "corrections"
            continue
        if lowered.startswith("# doute") or lowered.startswith("# uncertain"):
            section = "uncertain"
            continue
        if section == "corrections":
            match = _CORRECTION_LINE.match(line)
            if match and len(corrections) < 20:
                corrections.append({
                    "timestamp": match.group("ts"),
                    "original": match.group("original").strip(),
                    "replacement": match.group("replacement").strip(),
                    "confidence": 0.85,
                    "reason": (match.group("reason") or "").strip(),
                })
        elif section == "uncertain":
            match = _DOUBT_LINE.match(line)
            if match and len(uncertain) < 20:
                uncertain.append({
                    "timestamp": match.group("ts"),
                    "text": match.group("text").strip(),
                    "reason": (match.group("reason") or "").strip(),
                })

    return {"corrections": corrections, "uncertain_passages": uncertain}
#
# We run pyannote.audio inside the same managed venv that hosts mlx_whisper.
# The script below is invoked via `<venv>/bin/python -c <script>`. It loads
# the diarisation pipeline, runs it on the WAV, and prints a single JSON
# blob to stdout. Keeping it inline avoids shipping a separate Python file
# and keeps the venv self-contained.


_DIARIZATION_SCRIPT = '''
import json
import os
import sys

audio_path = sys.argv[1]
# Optional speaker-count hints. "" means "let pyannote decide", which
# matches the previous behaviour. Passing a tight (min, max) bracket
# dramatically improves attribution on under-segmented recordings —
# pyannote tends to merge speakers when left to estimate freely.
min_speakers_arg = sys.argv[2] if len(sys.argv) > 2 else ""
max_speakers_arg = sys.argv[3] if len(sys.argv) > 3 else ""

def _parse_int(value):
    try:
        v = int(str(value).strip())
        return v if v > 0 else None
    except (TypeError, ValueError):
        return None

min_speakers = _parse_int(min_speakers_arg)
max_speakers = _parse_int(max_speakers_arg)

hf_token = os.environ.get("HF_TOKEN", "")
if not hf_token:
    print(json.dumps({"error": "HF_TOKEN not set"}))
    sys.exit(2)

try:
    import torch
    from pyannote.audio import Pipeline
except Exception as exc:
    print(json.dumps({"error": f"import failed: {exc}"}))
    sys.exit(3)

try:
    def _clean_error(message):
        text = str(message)
        if "Cannot access gated repo" in text:
            return (
                "Accès Hugging Face refusé pour le modèle de diarisation. "
                "Ouvrez https://huggingface.co/pyannote/speaker-diarization-community-1 "
                "(et son fallback https://huggingface.co/pyannote/speaker-diarization-3.1), "
                "acceptez les conditions d'utilisation, vérifiez que votre token a le droit Read, "
                "puis relancez la transcription."
            )
        return text

    def _load(model_id):
        try:
            return Pipeline.from_pretrained(model_id, token=hf_token)
        except TypeError as exc:
            if "token" not in str(exc):
                raise
            return Pipeline.from_pretrained(model_id, use_auth_token=hf_token)

    # community-1 (released after 3.1) ships better speaker counting,
    # an exclusive-speaker output that's straightforward to align
    # with Whisper timestamps, and broader language coverage. We
    # try it first and fall back to 3.1 only when the user hasn't
    # accepted the community-1 license — that way existing tokens
    # keep working without a hard break.
    pipeline = None
    load_errors = []
    for candidate in (
        "pyannote/speaker-diarization-community-1",
        "pyannote/speaker-diarization-3.1",
    ):
        try:
            pipeline = _load(candidate)
            if pipeline is not None:
                break
        except Exception as exc:
            load_errors.append(f"{candidate}: {exc}")
    if pipeline is None:
        message = "; ".join(load_errors) or "no pyannote pipeline available"
        raise RuntimeError(f"pipeline unavailable: {message}")
except Exception as exc:
    print(json.dumps({"error": f"pipeline load failed: {_clean_error(exc)}"}))
    sys.exit(4)

# Apple Silicon: prefer MPS when available, fall back to CPU.
try:
    if torch.backends.mps.is_available():
        pipeline.to(torch.device("mps"))
except Exception:
    pass

try:
    # Pyannote's clustering tends to over-merge similar voices when
    # no count constraint is given. We pass the hints when available
    # and fall back to a hint-less call when the underlying pipeline
    # rejects the kwargs (older versions, or commercial SDK variants
    # that don't forward them).
    pipeline_kwargs = {}
    if min_speakers is not None:
        pipeline_kwargs["min_speakers"] = min_speakers
    if max_speakers is not None:
        pipeline_kwargs["max_speakers"] = max_speakers
    try:
        diar = pipeline(audio_path, **pipeline_kwargs) if pipeline_kwargs else pipeline(audio_path)
    except TypeError:
        # Hint kwargs not understood — replay without them.
        diar = pipeline(audio_path)
except Exception as exc:
    print(json.dumps({"error": f"diarization failed: {exc}"}))
    sys.exit(5)

turns = []
# pyannote-audio 3.x returns an Annotation object with itertracks.
# pyannote-audio 4.x (with pyannoteai-sdk) might return a DiarizeOutput object.

# If it's a DiarizeOutput (from pyannote-audio 4.x), extract the annotation.
if not hasattr(diar, "itertracks"):
    # Try commercial/new SDK structure first
    if hasattr(diar, "exclusive_speaker_diarization"):
        diar = diar.exclusive_speaker_diarization
    elif hasattr(diar, "speaker_diarization"):
        diar = diar.speaker_diarization
    elif hasattr(diar, "to_annotation"):
        try:
            diar = diar.to_annotation()
        except Exception:
            pass

if hasattr(diar, "itertracks"):
    for turn, _, speaker in diar.itertracks(yield_label=True):
        turns.append({
            "start": float(turn.start),
            "end": float(turn.end),
            "speaker": str(speaker),
        })
elif hasattr(diar, "segments"):
    # Probable structure for DiarizeOutput or similar commercial SDK outputs
    for segment in diar.segments:
        turns.append({
            "start": float(getattr(segment, "start", 0)),
            "end": float(getattr(segment, "end", 0)),
            "speaker": str(getattr(segment, "speaker", "UNKNOWN")),
        })
else:
    # Last resort: if it's iterable, maybe it's already a list of segments
    try:
        for segment in diar:
            if hasattr(segment, "start") and hasattr(segment, "speaker"):
                turns.append({
                    "start": float(segment.start),
                    "end": float(segment.end),
                    "speaker": str(segment.speaker),
                })
    except Exception:
        pass

if not turns:
    print(json.dumps({"error": f"Unexpected diarization output type: {type(diar)} or empty results"}))
    sys.exit(6)

print(json.dumps({"turns": turns}))
'''


def build_diarization_cmd(
    venv_python_path: str,
    wav_path: str,
    *,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> list[str]:
    """Assemble the python command that runs pyannote inside the
    managed venv.

    ``min_speakers`` / ``max_speakers`` constrain pyannote's
    clustering. They default to ``None`` (let the model decide). Pass
    them whenever the caller knows the meeting size — pyannote tends
    to under-segment when free to estimate alone, merging two real
    voices into a single SPEAKER_NN cluster. Tight bounds turn the
    sub-segmentation knob the other way.
    """
    return [
        venv_python_path,
        "-c",
        _DIARIZATION_SCRIPT,
        wav_path,
        "" if min_speakers is None else str(int(min_speakers)),
        "" if max_speakers is None else str(int(max_speakers)),
    ]


# ---------------------------------------------------------------------------
# Speaker embedding extraction (recognition pass)
# ---------------------------------------------------------------------------
#
# Diarisation tells us "this 8-second slice is speaker A". Embedding
# extraction turns those slices into 512-dim vectors that we can
# compare across meetings — the foundation of recognising "Robin"
# next week without the user retyping the name.
#
# We use ``pyannote/embedding`` (same family as the diarisation
# pipeline). The script reads a JSON list of {label, segments[]}
# from argv[2], extracts one vector per segment via
# ``Inference("pyannote/embedding", window="whole")``, and prints a
# JSON map ``{label: [[...512 floats...], ...]}`` so the caller can
# average them downstream.
#
# Loading the embedding model is a few seconds on Apple Silicon. The
# pipeline only invokes this when at least one speaker profile is
# stored — running it for users who haven't enrolled anyone wastes
# time.

_SPEAKER_EMBEDDING_SCRIPT = '''
import json
import os
import sys

audio_path = sys.argv[1]
clusters_json = sys.argv[2]
hf_token = os.environ.get("HF_TOKEN", "")
if not hf_token:
    print(json.dumps({"error": "HF_TOKEN not set"}))
    sys.exit(2)

try:
    import torch
    from pyannote.audio import Inference
    from pyannote.core import Segment
except Exception as exc:
    print(json.dumps({"error": f"import failed: {exc}"}))
    sys.exit(3)

try:
    clusters = json.loads(clusters_json)
    if not isinstance(clusters, list):
        raise ValueError("clusters payload must be a JSON list")
except Exception as exc:
    print(json.dumps({"error": f"clusters payload invalid: {exc}"}))
    sys.exit(4)

try:
    inference = Inference(
        "pyannote/embedding",
        token=hf_token,
        window="whole",
    )
except TypeError as exc:
    if "token" not in str(exc):
        print(json.dumps({"error": f"embedding load failed: {exc}"}))
        sys.exit(5)
    inference = Inference(
        "pyannote/embedding",
        use_auth_token=hf_token,
        window="whole",
    )
except Exception as exc:
    print(json.dumps({"error": f"embedding load failed: {exc}"}))
    sys.exit(5)

# Apple Silicon: prefer MPS when available, fall back to CPU.
try:
    if torch.backends.mps.is_available():
        try:
            inference.to(torch.device("mps"))
        except Exception:
            pass
except Exception:
    pass

out = {}
for cluster in clusters:
    label = str(cluster.get("label") or "").strip()
    segments = cluster.get("segments") or []
    if not label or not segments:
        continue
    vectors = []
    for entry in segments:
        try:
            start = float(entry["start"])
            end = float(entry["end"])
        except (TypeError, ValueError, KeyError):
            continue
        if end <= start:
            continue
        try:
            embedding = inference.crop(audio_path, Segment(start, end))
        except Exception as exc:
            # Skip the bad segment; the others may still produce a
            # usable centroid.
            continue
        # ``embedding`` is a numpy array of shape (D,) — turn it
        # into a plain Python list for JSON serialisation.
        vectors.append([float(v) for v in embedding.tolist()])
    if vectors:
        out[label] = vectors

print(json.dumps({"clusters": out}))
'''


def build_embedding_extract_cmd(
    venv_python_path: str,
    wav_path: str,
    clusters: list[dict],
) -> list[str]:
    """Build the command that extracts pyannote embeddings.

    ``clusters`` is a list of ``{label, segments: [{start, end}]}``
    — typically built by picking the longest 1-3 turns of each
    diarisation cluster. The embedding script returns one vector per
    segment, which the caller averages into a centroid.
    """
    return [
        venv_python_path,
        "-c",
        _SPEAKER_EMBEDDING_SCRIPT,
        wav_path,
        json.dumps(clusters or [], ensure_ascii=False),
    ]


def parse_embedding_output(stdout: str) -> dict[str, list[list[float]]]:
    """Counterpart to :func:`build_embedding_extract_cmd`.

    Returns a mapping ``{cluster_label: [[float, ...], ...]}``.
    Raises ``RuntimeError`` when the script reported an error so the
    caller can surface it in the UI.
    """
    text = (stdout or "").strip()
    if not text:
        raise RuntimeError("Embeddings: sortie vide.")
    last_line = text.splitlines()[-1].strip()
    try:
        payload = json.loads(last_line)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Embeddings: JSON invalide ({exc}).") from exc
    if "error" in payload:
        raise RuntimeError(f"Embeddings: {payload['error']}")
    raw = payload.get("clusters") or {}
    out: dict[str, list[list[float]]] = {}
    for label, vectors in raw.items():
        if not isinstance(vectors, list):
            continue
        cleaned: list[list[float]] = []
        for vector in vectors:
            if isinstance(vector, list):
                try:
                    cleaned.append([float(v) for v in vector])
                except (TypeError, ValueError):
                    continue
        if cleaned:
            out[str(label)] = cleaned
    return out


def fuse_micro_turns(
    turns: list[dict],
    *,
    min_duration: float = 0.4,
) -> list[dict]:
    """Merge speaker turns shorter than ``min_duration`` into the
    surrounding turn that's most likely the same speaker.

    Pyannote happily emits 100-300 ms turns when one participant
    nods in the middle of another's sentence. The downstream
    ``assign_speakers_to_segments`` then attributes a chunk of
    speech to that nodder, producing the classic
    "[Robin] Bah ouais. [David] OK." mid-sentence cut.

    Rules:
      - A turn shorter than ``min_duration`` is merged into the
        adjacent turn (before or after) belonging to the same
        speaker, when one exists within 1 s.
      - Otherwise it's absorbed by whichever neighbour is *closer*,
        extending that neighbour's end (or start) to cover the gap.
      - Turns are sorted by ``start`` before processing — pyannote
        usually returns them in order but we don't rely on it.
    """
    if not turns:
        return []
    if min_duration <= 0:
        return list(turns)

    items = sorted(
        (dict(t) for t in turns),
        key=lambda t: (float(t.get("start") or 0), float(t.get("end") or 0)),
    )

    # Pass 1: merge a micro-turn with a same-speaker neighbour when
    # one is nearby. Walk the list with explicit indices because we
    # mutate during iteration.
    i = 0
    while i < len(items):
        t = items[i]
        start = float(t.get("start") or 0)
        end = float(t.get("end") or 0)
        if end - start >= min_duration:
            i += 1
            continue
        speaker = t.get("speaker")
        # Look ±1 turn for the same speaker within a 1 s gap.
        prev_idx = i - 1 if i > 0 else None
        next_idx = i + 1 if i + 1 < len(items) else None
        merged_into: int | None = None
        if prev_idx is not None:
            p = items[prev_idx]
            if p.get("speaker") == speaker and start - float(p.get("end") or 0) <= 1.0:
                p["end"] = max(float(p.get("end") or 0), end)
                merged_into = prev_idx
        if merged_into is None and next_idx is not None:
            n = items[next_idx]
            if n.get("speaker") == speaker and float(n.get("start") or 0) - end <= 1.0:
                n["start"] = min(float(n.get("start") or 0), start)
                merged_into = next_idx
        if merged_into is not None:
            items.pop(i)
            # Don't advance i — the next item is now at our index.
            continue
        i += 1

    # Pass 2: absorb remaining micro-turns into the closer neighbour
    # so the time-axis stays gap-free.
    i = 0
    while i < len(items):
        t = items[i]
        start = float(t.get("start") or 0)
        end = float(t.get("end") or 0)
        if end - start >= min_duration:
            i += 1
            continue
        prev_idx = i - 1 if i > 0 else None
        next_idx = i + 1 if i + 1 < len(items) else None
        if prev_idx is None and next_idx is None:
            # Single tiny turn in the recording — keep it; better
            # than dropping the only label we have.
            i += 1
            continue
        prev_gap = (
            start - float(items[prev_idx].get("end") or 0)
            if prev_idx is not None
            else float("inf")
        )
        next_gap = (
            float(items[next_idx].get("start") or 0) - end
            if next_idx is not None
            else float("inf")
        )
        if prev_gap <= next_gap and prev_idx is not None:
            items[prev_idx]["end"] = max(
                float(items[prev_idx].get("end") or 0), end
            )
        elif next_idx is not None:
            items[next_idx]["start"] = min(
                float(items[next_idx].get("start") or 0), start
            )
        items.pop(i)
        # Same as pass 1: stay on this index.

    # Pass 3: collapse adjacent same-speaker turns. Pyannote
    # occasionally splits a continuous monologue into 3-5 turns when
    # the speaker pauses to breathe. Rendering all of them produces
    # ``[Robin] foo. [Robin] bar. [Robin] baz.`` instead of one
    # paragraph. We merge any pair separated by a gap of ≤ 1 s.
    j = 1
    while j < len(items):
        prev = items[j - 1]
        cur = items[j]
        prev_end = float(prev.get("end") or 0)
        cur_start = float(cur.get("start") or 0)
        same_speaker = prev.get("speaker") == cur.get("speaker")
        if same_speaker and cur_start - prev_end <= 1.0:
            prev["end"] = max(prev_end, float(cur.get("end") or 0))
            items.pop(j)
            continue
        j += 1

    return items


def parse_diarization_output(stdout: str) -> list[dict]:
    """
    Parses the JSON the diarisation script prints. Returns a list of
    {start, end, speaker} dicts. Raises RuntimeError on script-reported
    errors so the caller can surface them in the UI.
    """
    text = (stdout or "").strip()
    if not text:
        raise RuntimeError("Diarisation: sortie vide.")
    # The pyannote pipeline emits progress logs to stderr; stdout is reserved
    # for our JSON blob, but be defensive and pick the last line.
    last_line = text.splitlines()[-1].strip()
    try:
        payload = json.loads(last_line)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Diarisation: JSON invalide ({exc}).") from exc
    if "error" in payload:
        raise RuntimeError(f"Diarisation: {payload['error']}")
    return list(payload.get("turns", []))


def parse_whisper_json_segments(json_path: str) -> list[dict]:
    """
    MLX Whisper's --output-format json writes {"text": ..., "segments": [...],
    "language": ...}. We only care about the segments list.

    When ``--word-timestamps True`` ran, each segment also carries a
    ``words`` list of {start, end, word, probability}. We preserve
    those so the speaker-assignment pass downstream can split a
    segment on the actual word boundary when the diarisation turn
    changes mid-sentence.
    """
    raw = Path(json_path).read_text(encoding="utf-8")
    payload = json.loads(raw)
    segments = payload.get("segments") or []
    out = []
    for seg in segments:
        words_raw = seg.get("words") or []
        words: list[dict] = []
        for word in words_raw:
            try:
                words.append(
                    {
                        "start": float(word.get("start", 0.0)),
                        "end": float(word.get("end", 0.0)),
                        "word": str(word.get("word", "")),
                        "probability": word.get("probability"),
                    }
                )
            except (TypeError, ValueError):
                continue
        out.append({
            "start": float(seg.get("start", 0.0)),
            "end": float(seg.get("end", 0.0)),
            "text": str(seg.get("text", "")).strip(),
            "avg_logprob": seg.get("avg_logprob"),
            "compression_ratio": seg.get("compression_ratio"),
            "no_speech_prob": seg.get("no_speech_prob"),
            "words": words,
        })
    return clean_whisper_segments(out)


def render_segments_plain(segments: list[dict], output_format: str) -> str:
    """
    Render Whisper segments without speaker labels, after hallucination
    filtering. This keeps non-diarized transcripts clean.
    """
    fmt = (output_format or "txt").strip().lower()
    if fmt == "all":
        fmt = "txt"

    if fmt == "json":
        return json.dumps({"segments": segments}, ensure_ascii=False, indent=2)

    if fmt == "tsv":
        lines = ["start\tend\ttext"]
        for seg in segments:
            lines.append(f"{seg['start']:.3f}\t{seg['end']:.3f}\t{seg['text']}")
        return "\n".join(lines) + "\n"

    if fmt == "srt":
        lines = []
        for i, seg in enumerate(segments, start=1):
            lines.append(str(i))
            lines.append(
                f"{_format_timestamp_srt(seg['start'])} --> {_format_timestamp_srt(seg['end'])}"
            )
            lines.append(seg["text"])
            lines.append("")
        return "\n".join(lines)

    if fmt == "vtt":
        lines = ["WEBVTT", ""]
        for seg in segments:
            lines.append(
                f"{_format_timestamp_vtt(seg['start'])} --> {_format_timestamp_vtt(seg['end'])}"
            )
            lines.append(seg["text"])
            lines.append("")
        return "\n".join(lines)

    return "\n".join(seg["text"] for seg in segments if seg.get("text")).strip() + "\n"


def _smooth_word_assignments(
    word_speakers: list[str | None],
    word_starts: list[float],
    word_ends: list[float],
    *,
    min_run_words: int = 2,
    min_run_seconds: float = 0.4,
) -> list[str | None]:
    """Median-filter per-word speaker assignments to suppress
    1-word flickers between two stable runs of another speaker.

    The PR A motivating case: in the Caste transcript a single
    Whisper segment containing Manon's "des fois je vous dis oui"
    came out as three sub-segments — ``[?] des``, ``[SPEAKER_00]
    fois je vous`` (wrong attribution), ``[Manon] dis oui…``.
    pyannote's word-level boundaries flickered briefly at the
    start of Manon's turn. Smoothing collapses the 1-3 wrong words
    into the dominant neighbouring speaker.

    Algorithm:
    - Walk the list to compute runs of consecutive same-speaker words.
    - A run is "short" when it has < ``min_run_words`` words AND
      its total duration is < ``min_run_seconds`` seconds.
    - A short run sandwiched between two SAME-speaker neighbours
      gets absorbed by them.
    - A short run flanked by two DIFFERENT speakers goes to whichever
      neighbour the word's midpoint is closer to in time (no
      arbitrary preference for "next" vs "previous").
    - A short ``None`` run is treated like any other short run.
    """
    if not word_speakers:
        return word_speakers
    n = len(word_speakers)
    # Build (speaker, [word_idx, ...]) runs.
    runs: list[tuple[str | None, list[int]]] = []
    for idx, sp in enumerate(word_speakers):
        if runs and runs[-1][0] == sp:
            runs[-1][1].append(idx)
        else:
            runs.append((sp, [idx]))
    if len(runs) <= 2:
        return word_speakers
    # Mutable copy so we can rewrite assignments in place.
    out = list(word_speakers)
    for i in range(1, len(runs) - 1):
        sp, idxs = runs[i]
        run_duration = word_ends[idxs[-1]] - word_starts[idxs[0]]
        if len(idxs) >= min_run_words or run_duration >= min_run_seconds:
            continue
        prev_sp = runs[i - 1][0]
        next_sp = runs[i + 1][0]
        if prev_sp is None and next_sp is None:
            continue
        if prev_sp == next_sp and prev_sp is not None:
            target = prev_sp
        elif prev_sp is None:
            target = next_sp
        elif next_sp is None:
            target = prev_sp
        else:
            # Flanked by two distinct speakers: route to the closer
            # one in time. Tie-break favours the upcoming speaker
            # since the user's mental model is "the new speaker
            # started speaking around here".
            midpoint = (word_starts[idxs[0]] + word_ends[idxs[-1]]) / 2.0
            prev_end = word_ends[runs[i - 1][1][-1]]
            next_start = word_starts[runs[i + 1][1][0]]
            target = next_sp if abs(next_start - midpoint) <= abs(midpoint - prev_end) else prev_sp
        for idx in idxs:
            out[idx] = target
    return out


def merge_adjacent_same_speaker_segments(
    segments: list[dict],
    *,
    max_gap_seconds: float = 1.5,
) -> list[dict]:
    """Fuse consecutive segments by the same speaker when separated
    by at most ``max_gap_seconds``.

    Why this matters: even after the word-level smoothing, Whisper
    occasionally emits two adjacent segments for the same speaker
    (e.g. a brief pause). The textual rendering already groups them
    visually, but the LLM correction and multipass steps both see
    the raw list — running them on micro-fragments rather than full
    turns hurts both passes. Merging upfront gives them context.

    The merged segment carries:
    - the earliest ``start`` and latest ``end``
    - the concatenated ``text`` (single-space separator, dedup
      against double-leading-space artefacts from Whisper's word
      tokens)
    - the concatenated ``words`` list when both children had one
    - the WORST ``avg_logprob`` / highest ``no_speech_prob`` /
      highest ``compression_ratio`` of the merged children — so
      downstream confidence-triaged passes see the worst-case
      signal rather than an artificially-good average.
    """
    if not segments:
        return []
    out: list[dict] = []
    for seg in segments:
        if not out:
            out.append(dict(seg))
            continue
        prev = out[-1]
        prev_speaker = prev.get("speaker")
        cur_speaker = seg.get("speaker")
        if prev_speaker is None or cur_speaker is None or prev_speaker != cur_speaker:
            out.append(dict(seg))
            continue
        try:
            gap = float(seg["start"]) - float(prev["end"])
        except (KeyError, TypeError, ValueError):
            out.append(dict(seg))
            continue
        if gap > max_gap_seconds:
            out.append(dict(seg))
            continue
        # Merge into prev.
        prev["end"] = float(seg["end"])
        prev_text = (prev.get("text") or "").rstrip()
        cur_text = (seg.get("text") or "").lstrip()
        if prev_text and cur_text:
            prev["text"] = f"{prev_text} {cur_text}"
        elif cur_text:
            prev["text"] = cur_text
        prev_words = prev.get("words") or []
        cur_words = seg.get("words") or []
        if prev_words or cur_words:
            prev["words"] = list(prev_words) + list(cur_words)
        # Worst-case quality metrics so downstream passes don't miss
        # a hesitation that got absorbed into a longer turn.
        for key, reducer in (
            ("avg_logprob", min),
            ("no_speech_prob", max),
            ("compression_ratio", max),
        ):
            cur_val = seg.get(key)
            prev_val = prev.get(key)
            if cur_val is None:
                continue
            if prev_val is None:
                prev[key] = cur_val
            else:
                try:
                    prev[key] = reducer(prev_val, cur_val)
                except TypeError:
                    pass
    return out


def absorb_orphan_speaker_fragments(
    segments: list[dict],
    *,
    max_orphan_duration: float = 0.6,
    max_neighbor_gap: float = 2.0,
) -> list[dict]:
    """Drop the ``None`` (rendered as ``[?]``) label on very short
    segments flanked by known speakers.

    Concretely fixes the ``L4 [?] des`` pattern in the Caste
    transcript: a single-word segment of 0.2 s that pyannote
    couldn't confidently attribute, sandwiched between two known
    turns. The renderer's ``[?]`` label looks like a third unknown
    speaker and makes the transcript unreadable.

    Rules:
    - ``None`` speaker fragments shorter than ``max_orphan_duration``
      (and within ``max_neighbor_gap`` of at least one labelled
      neighbour) are absorbed.
    - Sandwiched between two same-speaker neighbours → absorbed into
      that speaker.
    - Otherwise absorbed into the **next** speaker (the one taking
      over) when both sides are within the neighbour gap, else the
      single available neighbour.
    """
    if not segments:
        return []
    out: list[dict] = []
    for index, raw in enumerate(segments):
        seg = dict(raw)
        speaker = seg.get("speaker")
        if speaker is not None:
            out.append(seg)
            continue
        try:
            duration = float(seg["end"]) - float(seg["start"])
        except (KeyError, TypeError, ValueError):
            duration = float("inf")
        if duration > max_orphan_duration:
            out.append(seg)
            continue
        prev_seg = out[-1] if out else None
        next_seg = segments[index + 1] if index + 1 < len(segments) else None
        prev_speaker = prev_seg.get("speaker") if prev_seg else None
        next_speaker = next_seg.get("speaker") if next_seg else None
        prev_end = float(prev_seg["end"]) if prev_seg else float("-inf")
        next_start = float(next_seg["start"]) if next_seg else float("inf")
        seg_start = float(seg.get("start") or 0)
        seg_end = float(seg.get("end") or 0)
        prev_gap = seg_start - prev_end if prev_seg else float("inf")
        next_gap = next_start - seg_end if next_seg else float("inf")
        target: str | None = None
        if prev_speaker and next_speaker and prev_speaker == next_speaker:
            # Same speaker on both sides — only a brief pause.
            target = prev_speaker
        elif prev_speaker and next_speaker:
            # Different neighbours: snap to the closer one. Tie goes
            # to the next speaker (turn-taking heuristic).
            target = next_speaker if next_gap <= prev_gap else prev_speaker
        elif next_speaker and next_gap <= max_neighbor_gap:
            target = next_speaker
        elif prev_speaker and prev_gap <= max_neighbor_gap:
            target = prev_speaker
        if target is not None:
            seg["speaker"] = target
        out.append(seg)
    return out


def assign_speakers_to_segments(
    whisper_segments: Iterable[dict],
    diarization_turns: Iterable[dict],
) -> list[dict]:
    """Pair each Whisper segment with a speaker, splitting on
    speaker boundaries when word-level timestamps are available.

    Two regimes:

    * **Word-level (preferred).** When the segment carries a non-
      empty ``words`` list, we look up the active speaker at each
      word's midpoint and group consecutive same-speaker words into
      sub-segments. A short Whisper segment that spans an
      interruption ("Bah ouais. — OK.") becomes two segments with
      the right names instead of one wrongly attributed line.

    * **Segment-level (fallback).** Without words we keep the
      original "max overlap wins" rule. This still kicks in for
      runs that didn't enable ``--word-timestamps`` and keeps the
      old contract for callers that haven't migrated.

    Returns a list of dicts. The same input segment may produce
    multiple output segments — order is preserved.
    """
    turns = [
        (float(t["start"]), float(t["end"]), str(t["speaker"]))
        for t in diarization_turns
    ]
    out: list[dict] = []
    for seg in whisper_segments:
        words = seg.get("words") or []
        if words and turns:
            sub_segments = _split_segment_on_speaker_changes(seg, words, turns)
            out.extend(sub_segments)
            continue

        s_start = float(seg["start"])
        s_end = float(seg["end"])
        per_speaker: dict[str, float] = {}
        for t_start, t_end, speaker in turns:
            overlap = max(0.0, min(s_end, t_end) - max(s_start, t_start))
            if overlap > 0:
                per_speaker[speaker] = per_speaker.get(speaker, 0.0) + overlap
        best_speaker = None
        if per_speaker:
            best_speaker = max(per_speaker.items(), key=lambda kv: kv[1])[0]
        new_seg = dict(seg)
        new_seg["speaker"] = best_speaker
        out.append(new_seg)
    # Two post-passes that turn a noisy diarisation projection into
    # a clean, readable speaker list:
    # 1. Absorb [?] fragments shorter than ~0.6 s into a neighbour
    #    when one's available (fixes ``[?] des`` orphans).
    # 2. Fuse adjacent same-speaker segments under a 1.5 s gap so
    #    downstream passes (LLM corrections, multipass) see full
    #    turns instead of micro-fragments.
    out = absorb_orphan_speaker_fragments(out)
    out = merge_adjacent_same_speaker_segments(out)
    return out


def _speaker_at(time: float, turns: list[tuple[float, float, str]]) -> str | None:
    """Return the speaker label active at ``time`` (or None when
    no turn covers it). Falls back to the closest turn within 0.5 s
    so a word that lands in the gap between two turns doesn't get
    orphaned."""
    for t_start, t_end, speaker in turns:
        if t_start <= time <= t_end:
            return speaker
    closest: tuple[float, str] | None = None
    for t_start, t_end, speaker in turns:
        gap = min(abs(t_start - time), abs(t_end - time))
        if gap <= 0.5 and (closest is None or gap < closest[0]):
            closest = (gap, speaker)
    return closest[1] if closest else None


def _split_segment_on_speaker_changes(
    segment: dict,
    words: list[dict],
    turns: list[tuple[float, float, str]],
) -> list[dict]:
    """Walk ``words``, group runs of same-speaker words, and emit
    one sub-segment per group.

    The sub-segments inherit Whisper's per-segment quality metrics
    (``avg_logprob``, ``compression_ratio``, ``no_speech_prob``) so
    downstream steps (multipass scoring, review markdown) still see
    the same signal — the metrics describe the audio chunk that
    *Whisper* processed, which we haven't re-decoded.
    """
    # First pass: per-word speaker assignment from diarisation turns.
    valid_words: list[dict] = []
    word_starts: list[float] = []
    word_ends: list[float] = []
    word_speakers: list[str | None] = []
    for word in words:
        try:
            wstart = float(word.get("start", 0.0))
            wend = float(word.get("end", 0.0))
        except (TypeError, ValueError):
            continue
        valid_words.append(word)
        word_starts.append(wstart)
        word_ends.append(wend)
        midpoint = (wstart + wend) / 2.0
        word_speakers.append(_speaker_at(midpoint, turns))
    # Second pass: smooth out 1-word "speaker flickers" so a brief
    # mis-attribution at a turn boundary doesn't produce a fragment.
    # See the Caste transcript L4-7 pattern in ``_smooth_word_assignments``.
    smoothed = _smooth_word_assignments(word_speakers, word_starts, word_ends)

    groups: list[tuple[str | None, list[dict]]] = []
    for word, speaker in zip(valid_words, smoothed):
        if groups and groups[-1][0] == speaker:
            groups[-1][1].append(word)
        else:
            groups.append((speaker, [word]))

    if not groups:
        # No usable word data — fall back to whole-segment max-overlap.
        s_start = float(segment["start"])
        s_end = float(segment["end"])
        per_speaker: dict[str, float] = {}
        for t_start, t_end, speaker in turns:
            overlap = max(0.0, min(s_end, t_end) - max(s_start, t_start))
            if overlap > 0:
                per_speaker[speaker] = per_speaker.get(speaker, 0.0) + overlap
        best_speaker = (
            max(per_speaker.items(), key=lambda kv: kv[1])[0] if per_speaker else None
        )
        new_seg = dict(segment)
        new_seg["speaker"] = best_speaker
        return [new_seg]

    sub_segments: list[dict] = []
    for speaker, group_words in groups:
        text = "".join(w.get("word", "") for w in group_words).strip()
        if not text:
            continue
        # Use word-level start/end so the timeline stays accurate at
        # the boundary we just split.
        sub = dict(segment)
        sub["start"] = float(group_words[0].get("start", segment["start"]))
        sub["end"] = float(group_words[-1].get("end", segment["end"]))
        sub["text"] = text
        sub["speaker"] = speaker
        sub["words"] = group_words
        sub_segments.append(sub)
    return sub_segments


def _format_timestamp_srt(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    ms_total = int(round(seconds * 1000))
    hours, rem = divmod(ms_total, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def _format_timestamp_vtt(seconds: float) -> str:
    return _format_timestamp_srt(seconds).replace(",", ".")


def _format_timestamp_short(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    total = int(seconds)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def render_segments_with_speakers(segments: list[dict], output_format: str) -> str:
    """
    Renders speaker-labeled segments to txt/srt/vtt/json/tsv. Speaker labels
    come from pyannote (e.g. SPEAKER_00); the user can rename them in the
    transcript afterwards. Unlabeled segments fall back to "?".
    """
    fmt = (output_format or "txt").strip().lower()
    if fmt == "all":
        fmt = "txt"

    def label(seg: dict) -> str:
        return seg.get("speaker") or "?"

    if fmt == "json":
        return json.dumps({"segments": segments}, ensure_ascii=False, indent=2)

    if fmt == "tsv":
        lines = ["start\tend\tspeaker\ttext"]
        for seg in segments:
            lines.append(
                f"{seg['start']:.3f}\t{seg['end']:.3f}\t{label(seg)}\t{seg['text']}"
            )
        return "\n".join(lines) + "\n"

    if fmt == "srt":
        lines = []
        for i, seg in enumerate(segments, start=1):
            lines.append(str(i))
            lines.append(
                f"{_format_timestamp_srt(seg['start'])} --> {_format_timestamp_srt(seg['end'])}"
            )
            lines.append(f"[{label(seg)}] {seg['text']}")
            lines.append("")
        return "\n".join(lines)

    if fmt == "vtt":
        lines = ["WEBVTT", ""]
        for seg in segments:
            lines.append(
                f"{_format_timestamp_vtt(seg['start'])} --> {_format_timestamp_vtt(seg['end'])}"
            )
            lines.append(f"[{label(seg)}] {seg['text']}")
            lines.append("")
        return "\n".join(lines)

    # txt: one block per consecutive run of the same speaker, prefixed
    # with the speaker tag and the start timestamp. Long monologues are
    # broken at internal pauses > 1.2 s or every ~25 s so a 5-minute
    # explanation doesn't end up on a single 800-character line.
    lines: list[str] = []
    current_speaker: str | None = None
    current_segs: list[dict] = []

    def flush():
        if not current_segs:
            return
        speaker_tag = f"[{current_speaker or '?'}]"
        for paragraph in _split_turn_into_paragraphs(current_segs):
            start = float(paragraph[0].get("start") or 0.0)
            text = " ".join(str(s.get("text") or "") for s in paragraph).strip()
            if not text:
                continue
            lines.append(
                f"{speaker_tag} ({_format_timestamp_short(start)}) {text}"
            )

    for seg in segments:
        speaker = label(seg)
        if speaker != current_speaker:
            flush()
            current_speaker = speaker
            current_segs = [seg]
        else:
            current_segs.append(seg)
    flush()
    return "\n".join(lines) + "\n"


def _split_turn_into_paragraphs(
    segs: list[dict],
    *,
    max_paragraph_seconds: float = 25.0,
    min_pause_seconds: float = 1.2,
) -> list[list[dict]]:
    """Split a same-speaker run into paragraph-sized chunks.

    Two break triggers:
    - A pause between consecutive segments ≥ ``min_pause_seconds``
      (matches the natural rhythm of a speaker taking a breath
      between thoughts).
    - A cumulative paragraph duration ≥ ``max_paragraph_seconds``
      (prevents 5-minute monologues from sitting on one line).

    Each paragraph is returned as a list of segments so the caller
    can pick the start timestamp of the first one for the line
    prefix. Empty input returns an empty list.
    """
    if not segs:
        return []
    paragraphs: list[list[dict]] = [[segs[0]]]
    for i in range(1, len(segs)):
        try:
            prev_end = float(segs[i - 1].get("end") or 0.0)
            cur_start = float(segs[i].get("start") or 0.0)
            paragraph_start = float(paragraphs[-1][0].get("start") or 0.0)
        except (TypeError, ValueError):
            paragraphs[-1].append(segs[i])
            continue
        gap = cur_start - prev_end
        paragraph_duration = cur_start - paragraph_start
        if (
            gap >= min_pause_seconds
            or paragraph_duration >= max_paragraph_seconds
        ):
            paragraphs.append([segs[i]])
        else:
            paragraphs[-1].append(segs[i])
    return paragraphs


_MULTIMODAL_AUDIO_SCRIPT = '''
import json
import sys
import os

try:
    from mlx_vlm import load, generate
except ImportError:
    print(json.dumps({"error": "mlx-vlm not installed"}))
    sys.exit(1)

model_path = sys.argv[1]
audio_path = sys.argv[2]
prompt_text = sys.argv[3]

try:
    model, processor = load(model_path)
    
    # Format prompt for Qwen2-Audio or generic Audio model
    if "qwen" in model_path.lower() and "audio" in model_path.lower():
        formatted_prompt = f"<|audio_bos|><|AUDIO|><|audio_eos|>{prompt_text}"
    else:
        formatted_prompt = prompt_text
        
    response = generate(model, processor, prompt=formatted_prompt, audio=audio_path, max_tokens=200, verbose=False)

    print(json.dumps({"suggestion": response.strip()}))

except Exception as e:
    print(json.dumps({"error": str(e)}))
    sys.exit(2)
'''

def build_multimodal_audio_cmd(
    venv_python_path: str,
    model_path: str,
    audio_path: str,
    prompt: str,
) -> list[str]:
    return [venv_python_path, "-c", _MULTIMODAL_AUDIO_SCRIPT, model_path, audio_path, prompt]


def parse_multimodal_audio_response(stdout: str) -> dict:
    """Decode the JSON line emitted by ``_MULTIMODAL_AUDIO_SCRIPT``.

    The embedded script prints exactly one JSON object on its last
    line — either ``{"suggestion": "..."}`` on success or
    ``{"error": "..."}`` on failure. Earlier lines may contain
    progress chatter from ``mlx_vlm`` itself, so we always parse
    the last non-empty line.

    Returns an empty dict on parse failure rather than raising —
    the caller treats "no suggestion" as "skip this passage" and
    keeps going.
    """
    if not stdout:
        return {}
    lines = [line for line in stdout.strip().splitlines() if line.strip()]
    if not lines:
        return {}
    last = lines[-1].strip()
    try:
        payload = json.loads(last)
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    return payload


def build_multimodal_recheck_prompt(
    *,
    whisper_text: str,
    reason: str,
    glossary: str = "",
) -> str:
    """Compose the French prompt sent to Qwen2-Audio.

    Ported verbatim from the legacy ``_run_clip_rechecks`` (with the
    glossary-priority sentence) so the new engine and the old app
    produce comparable suggestions on the same passage. Kept here
    rather than buried in the pipeline so we can unit-test the
    prompt construction without spinning up the orchestrator.
    """
    glossary_hint = ""
    if (glossary or "").strip():
        glossary_hint = (
            f"\nVocabulaire métier attendu (priorité absolue) : "
            f"{glossary.strip()}"
        )
    return (
        "Tu écoutes un court extrait d'une réunion en français.\n"
        f"Whisper hésite sur : « {whisper_text} ».\n"
        f"Raison du doute : {reason}.{glossary_hint}\n"
        "Compare l'audio aux termes du vocabulaire : les noms propres, "
        "marques, logiciels et acronymes doivent être orthographiés "
        "exactement comme dans ce vocabulaire si le son correspond. "
        "Corrige aussi les mots absurdes vers le mot français évident "
        "quand la proximité phonétique est forte.\n"
        "Que dit réellement la personne dans cet extrait ? "
        "Réponds par la phrase exacte, sans commentaire."
    )
