from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable


TRANSCRIPTION_AUDIO_FILTERS = [
    "highpass=f=80",
    "lowpass=f=7600",
    "acompressor=threshold=-20dB:ratio=2.2:attack=5:release=160",
    "loudnorm=I=-16:TP=-1.5:LRA=11",
]

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

TEXT_LLM_MODELS: list[dict] = [
    {
        "id": "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
        "label": "Mistral 7B Instruct · 4-bit (~4 Go) — recommandé",
        "family": "Mistral",
    },
    {
        "id": "mlx-community/Llama-3.2-3B-Instruct-4bit",
        "label": "Llama 3.2 3B Instruct · 4-bit (~2 Go) — léger (M1)",
        "family": "Llama",
    },
    {
        "id": "mlx-community/Qwen2.5-7B-Instruct-4bit",
        "label": "Qwen 2.5 7B Instruct · 4-bit (~4 Go)",
        "family": "Qwen",
    },
    {
        "id": "mlx-community/Qwen2.5-14B-Instruct-4bit",
        "label": "Qwen 2.5 14B Instruct · 4-bit (~8 Go) — qualité supérieure (M4 Max)",
        "family": "Qwen",
    },
]

AUDIO_LLM_MODELS: list[dict] = [
    {
        "id": "mlx-community/Qwen2-Audio-7B-Instruct-4bit",
        "label": "Qwen2-Audio 7B Instruct · 4-bit (~4 Go) — recommandé",
        "family": "Qwen-Audio",
    },
    {
        "id": "mlx-community/Qwen2-Audio-7B-Instruct-8bit",
        "label": "Qwen2-Audio 7B Instruct · 8-bit (~7 Go) — qualité supérieure",
        "family": "Qwen-Audio",
    },
]

DEFAULT_TEXT_LLM_MODEL = TEXT_LLM_MODELS[0]["id"]
DEFAULT_AUDIO_LLM_MODEL = AUDIO_LLM_MODELS[0]["id"]

LEGACY_WHISPER_MODEL_IDS: dict[str, str] = {
    "mlx-community/whisper-large-v3": "mlx-community/whisper-large-v3-mlx",
    "mlx-community/whisper-medium": "mlx-community/whisper-medium-mlx",
}

WHISPER_MODELS: list[dict] = [
    {
        "id": "mlx-community/whisper-large-v3-turbo",
        "label": "Whisper Large v3 Turbo · rapide — recommandé",
    },
    {
        "id": "mlx-community/whisper-large-v3-mlx",
        "label": "Whisper Large v3 · qualité maximale",
    },
    {
        "id": "mlx-community/whisper-medium-mlx",
        "label": "Whisper Medium · léger",
    },
]

DEFAULT_WHISPER_MODEL = WHISPER_MODELS[0]["id"]


def canonical_whisper_model_id(model_id: str) -> str:
    raw = (model_id or "").strip()
    if not raw:
        return DEFAULT_WHISPER_MODEL
    return LEGACY_WHISPER_MODEL_IDS.get(raw, raw)


def text_llm_label_for(model_id: str) -> str:
    for entry in TEXT_LLM_MODELS:
        if entry["id"] == model_id:
            return entry["label"]
    return model_id


def audio_llm_label_for(model_id: str) -> str:
    for entry in AUDIO_LLM_MODELS:
        if entry["id"] == model_id:
            return entry["label"]
    return model_id

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
    r"^(on va parler de|je vais|on va|nous allons|aujourd'hui|aujourd’hui|là on va|c'est parti pour|c’est parti pour)\s+",
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
) -> list[str]:
    cmd = [ffmpeg_path, "-y", "-hide_banner", "-loglevel", "error"]

    if ss is not None:
        cmd += ["-ss", ss]
    if to is not None:
        cmd += ["-to", to]

    cmd += ["-i", in_path, "-vn"]

    if speech_enhance:
        cmd += ["-af", ",".join(TRANSCRIPTION_AUDIO_FILTERS)]

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


def structured_initial_prompt(context: str) -> str:
    """
    Whisper interprets `--initial-prompt` as the imagined previous segment of
    the same audio. A bare list of names ("Ekonum, MAIA, RGPD") works less
    well than a short natural sentence that *uses* those terms — that primes
    the decoder's language model to expect them.
    """
    raw = (context or "").strip()
    if not raw:
        return ""
    flat = " ".join(raw.split())
    if len(flat) > INITIAL_PROMPT_MAX_CHARS:
        flat = flat[:INITIAL_PROMPT_MAX_CHARS].rsplit(" ", 1)[0]
    return f"Réunion en français. Termes attendus: {flat}."


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

    clips = (clip_timestamps or "").strip()
    if clips:
        cmd += ["--clip-timestamps", clips]

    return cmd


_TEXT_SIGNAL_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]")
_ONLY_PUNCTUATION_RE = re.compile(r"^[\s.…·!?;:,\\/_|()\\[\\]{}'\"`+-]+$")
_SPACE_RE = re.compile(r"\s+")


def _normalize_segment_text(text: str) -> str:
    normalized = (text or "").strip().lower()
    normalized = normalized.replace("’", "'").replace("…", "...")
    normalized = re.sub(r"[^a-z0-9à-öø-ÿ']+", " ", normalized)
    return _SPACE_RE.sub(" ", normalized).strip()


def is_hallucinated_whisper_segment(segment: dict) -> bool:
    text = str(segment.get("text", "")).strip()
    if not text:
        return True
    if _ONLY_PUNCTUATION_RE.fullmatch(text):
        return True
    if not _TEXT_SIGNAL_RE.search(text):
        return True

    normalized = _normalize_segment_text(text)
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

Règles strictes :
- Réponds UNIQUEMENT par un objet JSON valide, sans texte avant/après, sans markdown.
- Chaque clé "speakers" est un identifiant SPEAKER_XX et la valeur est un prénom (ou "" si tu n'es pas sûr).
- N'invente JAMAIS un prénom : laisse la chaîne vide en cas de doute.
- Ne mets que les SPEAKER_XX présents dans la transcription.

Schéma exact attendu :
{{"title": "...", "speakers": {{"SPEAKER_00": "..."}}}}

Vocabulaire métier (à respecter dans le titre) :
{glossary or "(aucun)"}

Transcription :
{text}
[/INST]"""

    # Short cap — title + speakers fit comfortably in 250 tokens.
    response = generate(model, tokenizer, prompt=prompt, max_tokens=300, verbose=False)
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

try:
    with open(transcript_path, "r", encoding="utf-8") as f:
        text = f.read()
    text = text[:30000]

    model, tokenizer = load(model_path)

    prompt = f"""[INST] Tu es un relecteur expert de transcriptions de réunions professionnelles françaises.

Repère uniquement :
1. Les passages où Whisper a probablement mal entendu un terme métier ou un nom propre listé dans le vocabulaire ci-dessous, et où tu peux proposer la bonne version sans inventer.
2. Les passages clairement douteux que tu signales SANS corriger.

Règles strictes :
- N'INVENTE JAMAIS de mots qui ne sont pas dans la transcription.
- Format de sortie : markdown, exactement le format ci-dessous, rien d'autre.
- Si rien à corriger : écris "Aucune correction." puis "Aucun doute." et stop.
- Maximum 5 corrections + 5 doutes.

Format exact (chaque entrée sur 2 lignes) :

# Corrections
- [00:12:34] "texte exact dans la transcription" -> "texte corrigé" (raison: …)
- [00:14:02] "..." -> "..." (raison: …)

# Doutes
- [00:18:30] "passage exact douteux" (raison: …)

Vocabulaire métier (priorité absolue) :
{glossary or "(aucun)"}

Transcription :
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
) -> list[str]:
    return [
        venv_python_path,
        "-c",
        _LLM_CORRECTIONS_SCRIPT,
        model_path,
        transcript_path,
        glossary,
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
    Parse a JSON {"title": ..., "speakers": {...}} blob produced by the
    title/speakers script. Tolerates leading prose and missing trailing
    delimiters so a slightly drifting model still gives us something.
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
    return {"title": title, "speakers": speakers}


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
        if "speaker-diarization-community-1" in text or "Cannot access gated repo" in text:
            return (
                "Accès Hugging Face refusé pour pyannote/speaker-diarization-community-1. "
                "Ouvrez https://huggingface.co/pyannote/speaker-diarization-community-1, "
                "acceptez les conditions d'utilisation, vérifiez que votre token a le droit Read, "
                "puis relancez la transcription."
            )
        return text

    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=hf_token,
        )
    except TypeError as exc:
        if "token" not in str(exc):
            raise
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token,
        )
    if pipeline is None:
        raise RuntimeError("pipeline unavailable; check Hugging Face token and accepted pyannote licenses")
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


def build_diarization_cmd(venv_python_path: str, wav_path: str) -> list[str]:
    return [venv_python_path, "-c", _DIARIZATION_SCRIPT, wav_path]


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
    """
    raw = Path(json_path).read_text(encoding="utf-8")
    payload = json.loads(raw)
    segments = payload.get("segments") or []
    out = []
    for seg in segments:
        out.append({
            "start": float(seg.get("start", 0.0)),
            "end": float(seg.get("end", 0.0)),
            "text": str(seg.get("text", "")).strip(),
            "avg_logprob": seg.get("avg_logprob"),
            "compression_ratio": seg.get("compression_ratio"),
            "no_speech_prob": seg.get("no_speech_prob"),
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


def assign_speakers_to_segments(
    whisper_segments: Iterable[dict],
    diarization_turns: Iterable[dict],
) -> list[dict]:
    """
    For each Whisper segment, assign the speaker whose turn(s) overlap
    most with the segment. Segments with zero overlap stay unlabeled.

    Returns a new list of dicts with an added "speaker" key (or None).
    """
    turns = [
        (float(t["start"]), float(t["end"]), str(t["speaker"]))
        for t in diarization_turns
    ]
    out = []
    for seg in whisper_segments:
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
    return out


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

    # txt: one block per consecutive run of the same speaker, prefixed with
    # the speaker tag and the start timestamp. Easier for humans to read
    # than a per-segment dump.
    lines = []
    current_speaker: str | None = None
    current_start: float | None = None
    current_text: list[str] = []

    def flush():
        if current_text:
            lines.append(
                f"[{current_speaker or '?'}] ({_format_timestamp_short(current_start or 0)}) "
                + " ".join(current_text).strip()
            )

    for seg in segments:
        speaker = label(seg)
        if speaker != current_speaker:
            flush()
            current_speaker = speaker
            current_start = seg["start"]
            current_text = [seg["text"]]
        else:
            current_text.append(seg["text"])
    flush()
    return "\n".join(lines) + "\n"


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
