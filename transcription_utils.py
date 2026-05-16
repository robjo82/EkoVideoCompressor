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
    groups: list[tuple[str | None, list[dict]]] = []
    for word in words:
        try:
            wstart = float(word.get("start", 0.0))
            wend = float(word.get("end", 0.0))
        except (TypeError, ValueError):
            continue
        midpoint = (wstart + wend) / 2.0
        speaker = _speaker_at(midpoint, turns)
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
