"""Cloud transcription via the Gemini API.

Counterpart of the local MLX Whisper stack: a single ``generateContent``
call returns the transcript with timestamps, speaker attribution, a
suggested title, the technical terms and the uncertain passages — work
the local pipeline spreads over five separate phases (Whisper +
multipass + diarisation + LLM title + LLM corrections).

Design constraints, in line with ``odoo_client.py``:

* No third-party SDK — pure ``urllib.request`` + ``certifi`` so the
  PyInstaller bundle doesn't grow and there's no dependency drift.
* Every user-facing error is raised as :class:`CloudTranscriptionError`
  with a French, actionable message.
* Cost is a first-class concern: the catalogue below carries the
  official per-token prices, :func:`estimate_cloud_cost` projects a
  job's cost from the audio duration *before* anything is uploaded,
  and :func:`compute_cost_usd` converts the API's real usage counters
  into dollars after each call.

Audio handling: Gemini tokenises audio at a flat 32 tokens/second and
internally resamples to 16 kbps mono, so we upload a compact MP3
(mono / 16 kHz / 64 kbps ≈ 28 MB per hour) instead of the 115 MB/h WAV
the local pipeline uses. Long meetings are split into fixed windows so
each response stays well under the model's output-token ceiling; the
speaker map discovered in earlier windows is fed back into later ones
to keep labels consistent across the whole meeting.
"""

from __future__ import annotations

import json
import re
import ssl
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable

try:
    import certifi
except ImportError:  # pragma: no cover - certifi ships with the bundle
    certifi = None  # type: ignore[assignment]


GEMINI_API_BASE = "https://generativelanguage.googleapis.com"
OPENAI_API_BASE = "https://api.openai.com"
ASSEMBLYAI_API_BASE = "https://api.assemblyai.com"
DEEPGRAM_API_BASE = "https://api.deepgram.com"
GLADIA_API_BASE = "https://api.gladia.io"

# Gemini bills audio input at a fixed rate regardless of content.
AUDIO_TOKENS_PER_SECOND = 32

# Empirical output budget: a dense one-hour French meeting renders to
# roughly 9-10k words; serialised as timestamped JSON segments that's
# ~600 tokens of JSON per minute of speech. Thinking tokens (billed as
# output on Gemini 3.x) add on top — covered by the safety margin in
# the estimate.
ESTIMATED_OUTPUT_TOKENS_PER_AUDIO_SECOND = 10

# Window size for long meetings. 30 minutes ≈ 57 600 audio tokens in
# and ≈ 18 000 JSON tokens out — comfortable for every catalogue model
# while keeping per-window progress visible in the UI.
CLOUD_CHUNK_SECONDS = 30 * 60

# Don't bother slicing when the overshoot is marginal: a 32-minute
# meeting is better served by one call than by a 30 + 2 split.
_CHUNK_TOLERANCE_SECONDS = 5 * 60


_CLOUD_ROLE = "cloud_transcription"

# Catalogue surfaced in the SwiftUI Models tab next to the local
# checkpoints. Two billing models live side by side:
#
#  * ``billing == "per_token"`` (the multimodal LLMs — Gemini): cost
#    is metered on the real ``usageMetadata`` counters, so the entry
#    carries ``price_in_per_1m`` (audio input) and ``price_out_per_1m``
#    (text output) in USD per million tokens.
#  * ``billing == "per_hour"`` (dedicated STT — AssemblyAI, Deepgram,
#    Gladia, OpenAI transcribe): providers bill on audio duration, so
#    the entry carries ``price_per_hour`` and we derive cost from the
#    chunk length.
#
# ``needs_enrichment`` marks the providers that return only a raw
# transcript (+ native diarisation) and therefore lean on the LLM
# post-pass for a title, speaker *names* and business-glossary
# corrections. The full-bundle providers (Gemini) return all of that
# in one call and set it to False.
#
# ``thinking`` (Gemini only) caps reasoning tokens — transcription is
# perception work, extended thinking only burns output budget.
# "level_low" is the Gemini 3.x dial, "budget_zero" the 2.5 one; the
# client retries without the knob if the API rejects it.
#
# Prices are June 2026 list rates from each vendor's pricing page.
# Per-hour figures pick the *conservative* (higher) tier when a
# vendor has volume discounts, so the budget guard never under-bills.
CLOUD_TRANSCRIPTION_MODELS: list[dict] = [
    {
        "id": "gemini-3.5-flash",
        "label": "Gemini 3.5 Flash",
        "family": "Gemini",
        "role": _CLOUD_ROLE,
        "kind": "cloud",
        "provider": "gemini",
        "tier": "balanced",
        "language": ["multi"],
        "default": True,
        "billing": "per_token",
        "price_in_per_1m": 1.50,
        "price_out_per_1m": 9.00,
        "needs_enrichment": False,
        "diarizes": True,
        "thinking": "level_low",
    },
    {
        "id": "gemini-3.1-pro-preview",
        "label": "Gemini 3.1 Pro (préversion)",
        "family": "Gemini",
        "role": _CLOUD_ROLE,
        "kind": "cloud",
        "provider": "gemini",
        "tier": "heavy",
        "language": ["multi"],
        "billing": "per_token",
        "price_in_per_1m": 4.00,
        "price_out_per_1m": 18.00,
        "needs_enrichment": False,
        "diarizes": True,
        "thinking": "level_low",
    },
    {
        "id": "gemini-2.5-flash",
        "label": "Gemini 2.5 Flash",
        "family": "Gemini",
        "role": _CLOUD_ROLE,
        "kind": "cloud",
        "provider": "gemini",
        "tier": "light",
        "language": ["multi"],
        "billing": "per_token",
        "price_in_per_1m": 1.00,
        "price_out_per_1m": 2.50,
        "needs_enrichment": False,
        "diarizes": True,
        "thinking": "budget_zero",
    },
    {
        "id": "gemini-3.1-flash-lite",
        "label": "Gemini 3.1 Flash-Lite",
        "family": "Gemini",
        "role": _CLOUD_ROLE,
        "kind": "cloud",
        "provider": "gemini",
        "tier": "light",
        "language": ["multi"],
        "billing": "per_token",
        "price_in_per_1m": 0.50,
        "price_out_per_1m": 1.50,
        "needs_enrichment": False,
        "diarizes": True,
        "thinking": "level_low",
    },
    # --- Dedicated STT providers (transcript + native diarisation) ---
    {
        "id": "assemblyai-universal-3",
        "label": "AssemblyAI Universal-3",
        "family": "AssemblyAI",
        "role": _CLOUD_ROLE,
        "kind": "cloud",
        "provider": "assemblyai",
        "tier": "balanced",
        "language": ["fr", "en", "multi"],
        "billing": "per_hour",
        "price_per_hour": 0.21,
        "needs_enrichment": True,
        "diarizes": True,
    },
    {
        "id": "gpt-4o-transcribe-diarize",
        "label": "OpenAI gpt-4o-transcribe (diarisation)",
        "family": "OpenAI",
        "role": _CLOUD_ROLE,
        "kind": "cloud",
        "provider": "openai",
        "tier": "balanced",
        "language": ["multi"],
        "billing": "per_hour",
        "price_per_hour": 0.36,
        "needs_enrichment": True,
        "diarizes": True,
        # OpenAI transcription endpoint model id sent as-is.
        "api_model": "gpt-4o-transcribe-diarize",
    },
    {
        "id": "gpt-4o-mini-transcribe",
        "label": "OpenAI gpt-4o-mini-transcribe",
        "family": "OpenAI",
        "role": _CLOUD_ROLE,
        "kind": "cloud",
        "provider": "openai",
        "tier": "light",
        "language": ["multi"],
        "billing": "per_hour",
        "price_per_hour": 0.18,
        "needs_enrichment": True,
        # No diarisation on the mini transcribe model — single speaker
        # track. Still useful for 1:1s and dictation at half the price.
        "diarizes": False,
        "api_model": "gpt-4o-mini-transcribe",
    },
    {
        "id": "gladia-solaria-3",
        "label": "Gladia (Solaria-3)",
        "family": "Gladia",
        "role": _CLOUD_ROLE,
        "kind": "cloud",
        "provider": "gladia",
        "tier": "balanced",
        # Solaria-3: best accuracy on real EU business audio, single
        # language (no code-switching), async only. Great for French
        # meetings. Sent as the ``model`` param — without it Gladia
        # falls back to Solaria-1.
        "language": ["fr", "en", "de", "es", "it"],
        "billing": "per_hour",
        # Starter async list price; drops to ~0.20 on the Growth plan.
        # We bill the conservative figure so the guard never under-counts.
        "price_per_hour": 0.61,
        "needs_enrichment": True,
        "diarizes": True,
        "api_model": "solaria-3",
    },
    {
        "id": "gladia-solaria-1",
        "label": "Gladia (Solaria-1, multilingue)",
        "family": "Gladia",
        "role": _CLOUD_ROLE,
        "kind": "cloud",
        "provider": "gladia",
        "tier": "balanced",
        # Solaria-1: generalist, 100+ languages with code-switching —
        # keep it for multilingual recordings where Solaria-3's EU-only
        # single-language focus doesn't fit.
        "language": ["multi"],
        "billing": "per_hour",
        "price_per_hour": 0.61,
        "needs_enrichment": True,
        "diarizes": True,
        "api_model": "solaria-1",
    },
    {
        "id": "deepgram-nova-3",
        "label": "Deepgram Nova-3",
        "family": "Deepgram",
        "role": _CLOUD_ROLE,
        "kind": "cloud",
        "provider": "deepgram",
        "tier": "light",
        "language": ["multi"],
        "billing": "per_hour",
        # Pay-as-you-go pre-recorded Nova rate (~$0.0043/min).
        "price_per_hour": 0.26,
        "needs_enrichment": True,
        "diarizes": True,
        "api_model": "nova-3",
    },
]

DEFAULT_CLOUD_MODEL = CLOUD_TRANSCRIPTION_MODELS[0]["id"]

# Providers the engine knows how to drive. Surfaced to the SwiftUI
# settings so it can render one API-key field per provider.
CLOUD_PROVIDERS: list[str] = ["gemini", "openai", "assemblyai", "gladia", "deepgram"]


def cloud_models_for_provider(provider: str) -> list[dict]:
    return [m for m in CLOUD_TRANSCRIPTION_MODELS if m["provider"] == provider]


def provider_for_model(model_id: str) -> str:
    return cloud_model_entry(model_id).get("provider", "gemini")


class CloudTranscriptionError(RuntimeError):
    """Raised on any cloud failure, with a French user-facing message.

    ``code`` mirrors the engine's error-event codes so the SwiftUI
    layer can branch (e.g. ``cloud_budget_exceeded`` aborts the job,
    ``cloud_api`` falls back to the local pipeline).
    """

    def __init__(self, message: str, code: str = "cloud_api"):
        super().__init__(message)
        self.code = code


def _entry_hourly_usd(entry: dict) -> float:
    """Estimated USD for one hour of audio on this model — the common
    yardstick used to compare per-token and per-hour models (e.g. to
    pick the most expensive when falling back on an unknown id)."""
    if entry.get("billing") == "per_hour":
        return float(entry.get("price_per_hour") or 0)
    seconds = 3600
    input_tokens = seconds * AUDIO_TOKENS_PER_SECOND + 400
    output_tokens = seconds * ESTIMATED_OUTPUT_TOKENS_PER_AUDIO_SECOND + 300
    return (
        input_tokens * float(entry.get("price_in_per_1m") or 0)
        + output_tokens * float(entry.get("price_out_per_1m") or 0)
    ) / 1_000_000


def cloud_model_entry(model_id: str) -> dict:
    raw = (model_id or "").strip() or DEFAULT_CLOUD_MODEL
    for entry in CLOUD_TRANSCRIPTION_MODELS:
        if entry["id"] == raw:
            return entry
    # Unknown id (user typed a fresh model before the catalogue knew
    # it): keep their choice, bill at the most expensive known rate
    # (compared on a per-hour basis across both billing models) so the
    # budget guard stays conservative.
    fallback = max(CLOUD_TRANSCRIPTION_MODELS, key=_entry_hourly_usd)
    return {**fallback, "id": raw, "label": raw, "default": False}


def canonical_cloud_model_id(model_id: str) -> str:
    return cloud_model_entry(model_id)["id"]


def compute_cost_usd(model_id: str, input_tokens: int, output_tokens: int) -> float:
    """Per-token cost from real usage counters (per_token models)."""
    entry = cloud_model_entry(model_id)
    cost = (
        max(input_tokens, 0) * float(entry.get("price_in_per_1m") or 0)
        + max(output_tokens, 0) * float(entry.get("price_out_per_1m") or 0)
    ) / 1_000_000
    return round(cost, 6)


def cost_for_duration(model_id: str, duration_seconds: float) -> float:
    """Duration-based cost (per_hour models). For per_token models this
    returns the *estimate* — handy as a fallback, but real per_token
    billing should use :func:`compute_cost_usd` on the API counters."""
    entry = cloud_model_entry(model_id)
    seconds = max(float(duration_seconds or 0), 0.0)
    if entry.get("billing") == "per_hour":
        return round((seconds / 3600.0) * float(entry.get("price_per_hour") or 0), 6)
    return estimate_cloud_cost(seconds, model_id)["cost_usd"]


def estimate_cloud_cost(duration_seconds: float, model_id: str) -> dict[str, Any]:
    """Project a job's cost from its audio duration.

    Used twice: by the SwiftUI Run Setup to display "≈ 0,35 $US" next
    to the cloud engine picker, and by the engine's budget guard
    before the first byte is uploaded. Deliberately rounds *up* (the
    prompt overhead and thinking tokens are folded into the output
    estimate) — an over-estimate that blocks a borderline job beats
    an under-estimate that overshoots the user's cap.
    """
    entry = cloud_model_entry(model_id)
    seconds = max(float(duration_seconds or 0), 0.0)
    if entry.get("billing") == "per_hour":
        return {
            "model": canonical_cloud_model_id(model_id),
            "duration_seconds": seconds,
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": round((seconds / 3600.0) * float(entry.get("price_per_hour") or 0), 6),
        }
    input_tokens = int(seconds * AUDIO_TOKENS_PER_SECOND) + 400  # + prompt
    output_tokens = int(seconds * ESTIMATED_OUTPUT_TOKENS_PER_AUDIO_SECOND) + 300
    return {
        "model": canonical_cloud_model_id(model_id),
        "duration_seconds": seconds,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": compute_cost_usd(model_id, input_tokens, output_tokens),
    }


def plan_audio_chunks(duration_seconds: float) -> list[tuple[float, float]]:
    """Split a meeting into upload windows.

    Returns ``[(start, end), ...]`` in seconds on the source timeline.
    Short meetings (≤ 35 min) stay whole; longer ones get equal-ish
    windows of at most :data:`CLOUD_CHUNK_SECONDS`. Equal windows
    avoid a degenerate last slice (a 61-minute meeting becomes
    2 × 30.5 min, not 30 + 30 + 1).
    """
    total = max(float(duration_seconds or 0), 0.0)
    if total <= 0:
        return [(0.0, 0.0)]
    if total <= CLOUD_CHUNK_SECONDS + _CHUNK_TOLERANCE_SECONDS:
        return [(0.0, total)]
    count = int(total // CLOUD_CHUNK_SECONDS) + (1 if total % CLOUD_CHUNK_SECONDS else 0)
    width = total / count
    chunks: list[tuple[float, float]] = []
    for index in range(count):
        start = index * width
        end = total if index == count - 1 else (index + 1) * width
        chunks.append((round(start, 2), round(end, 2)))
    return chunks


def build_cloud_audio_cmd(
    ffmpeg_path: str,
    source_path: str,
    output_path: str,
    *,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
) -> list[str]:
    """ffmpeg command producing the compact upload artefact.

    MP3 mono / 16 kHz / 64 kbps: Gemini resamples everything to
    16 kbps mono internally, so this is transparent for accuracy
    while dividing the upload size by ~4 versus WAV.
    """
    cmd = [ffmpeg_path, "-y"]
    if start_seconds is not None and start_seconds > 0:
        cmd += ["-ss", f"{start_seconds:.2f}"]
    if end_seconds is not None and end_seconds > 0:
        cmd += ["-to", f"{end_seconds:.2f}"]
    cmd += [
        "-i", source_path,
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-codec:a", "libmp3lame",
        "-b:a", "64k",
        output_path,
    ]
    return cmd


# ---------------------------------------------------------------------------
# Prompt + response contract
# ---------------------------------------------------------------------------

# responseSchema keeps the model glued to the shape the pipeline
# expects — no markdown fences, no chatty preamble to strip.
RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "title": {"type": "STRING"},
        "speakers": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "label": {"type": "STRING"},
                    "name": {"type": "STRING"},
                },
                "required": ["label"],
            },
        },
        "technical_terms": {"type": "ARRAY", "items": {"type": "STRING"}},
        "segments": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "start": {"type": "STRING"},
                    "end": {"type": "STRING"},
                    "speaker": {"type": "STRING"},
                    "text": {"type": "STRING"},
                },
                "required": ["start", "speaker", "text"],
            },
        },
        "uncertain": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "timestamp": {"type": "STRING"},
                    "text": {"type": "STRING"},
                    "reason": {"type": "STRING"},
                },
            },
        },
    },
    "required": ["segments"],
}


def build_cloud_prompt(
    *,
    language: str = "fr",
    glossary_terms: list[str] | None = None,
    expected_speaker_names: list[str] | None = None,
    meeting_context: str = "",
    odoo_context: str = "",
    known_speakers: dict[str, str] | None = None,
    chunk_index: int = 0,
    chunk_count: int = 1,
    chunk_offset_seconds: float = 0.0,
    previous_tail: str = "",
) -> str:
    """Assemble the transcription instruction for one audio window.

    Mirrors what ``structured_initial_prompt`` does for Whisper but
    leans on the cloud model's actual instruction-following: business
    vocabulary, expected participants and CRM context become explicit
    constraints instead of a decoder prior.
    """
    lang_label = "français" if (language or "fr").startswith("fr") else language
    lines = [
        "Tu es un transcripteur professionnel de réunions.",
        f"Transcris fidèlement et intégralement cet enregistrement audio en {lang_label}.",
        "Règles :",
        "- Transcription verbatim propre : garde chaque phrase, supprime "
        "uniquement les hésitations pures (euh, hum).",
        "- Découpe en segments de une à trois phrases, chacun avec son "
        "horodatage de début et de fin au format mm:ss (ou h:mm:ss).",
        "- Identifie qui parle. Utilise le prénom/nom réel quand le dialogue "
        "permet de l'identifier (salutations, interpellations), sinon "
        "« Intervenant 1 », « Intervenant 2 », etc., de manière stable.",
        "- Renseigne `speakers` avec chaque étiquette utilisée et, si connu, "
        "le nom réel correspondant.",
        "- Renseigne `technical_terms` avec les noms propres, clients, "
        "produits et termes métier entendus.",
        "- Renseigne `uncertain` avec les passages dont tu doutes "
        "(mot inaudible, terme ambigu) et la raison.",
        "- Propose dans `title` un titre court et professionnel de la réunion.",
    ]
    names = [n for n in (expected_speaker_names or []) if (n or "").strip()]
    if names:
        lines.append(
            "Participants attendus (utilise ces orthographes exactes) : "
            + ", ".join(names) + "."
        )
    terms = [t for t in (glossary_terms or []) if (t or "").strip()]
    if terms:
        lines.append(
            "Vocabulaire métier attendu (orthographes exactes) : "
            + ", ".join(terms) + "."
        )
    if (meeting_context or "").strip():
        lines.append(f"Contexte de la réunion : {meeting_context.strip()}")
    if (odoo_context or "").strip():
        lines.append(
            "Contexte CRM/projet associé (pour fiabiliser noms et enjeux) :\n"
            + odoo_context.strip()
        )
    if chunk_count > 1:
        minutes = int(chunk_offset_seconds // 60)
        lines.append(
            f"Cet extrait est la partie {chunk_index + 1} sur {chunk_count} "
            f"d'une réunion plus longue ; il commence à la minute {minutes} "
            "de la réunion. Les horodatages doivent être relatifs au DÉBUT "
            "DE CET EXTRAIT (commence à 00:00)."
        )
        known = {k: v for k, v in (known_speakers or {}).items() if (v or "").strip()}
        if known:
            mapping = ", ".join(f"{k} = {v}" for k, v in sorted(known.items()))
            lines.append(
                "Intervenants déjà identifiés dans les parties précédentes "
                f"(réutilise exactement les mêmes étiquettes) : {mapping}."
            )
        if (previous_tail or "").strip():
            lines.append(
                "Fin de la partie précédente (pour assurer la continuité) :\n"
                + previous_tail.strip()
            )
    return "\n".join(lines)


_TIMESTAMP_RE = re.compile(r"^\s*(?:(\d+):)?(\d{1,2}):(\d{2})(?:[.,](\d{1,3}))?\s*$")


def parse_cloud_timestamp(value: Any) -> float | None:
    """Parse ``mm:ss`` / ``h:mm:ss`` / bare seconds into float seconds."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return max(float(value), 0.0)
    raw = str(value).strip()
    if not raw:
        return None
    match = _TIMESTAMP_RE.match(raw)
    if match:
        hours = int(match.group(1) or 0)
        minutes = int(match.group(2))
        seconds = int(match.group(3))
        millis = int((match.group(4) or "0").ljust(3, "0"))
        return hours * 3600 + minutes * 60 + seconds + millis / 1000
    try:
        return max(float(raw.replace(",", ".")), 0.0)
    except ValueError:
        return None


@dataclass(slots=True)
class CloudUsage:
    """Token + cost counters for one API call (or an aggregate)."""

    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    def add(self, other: "CloudUsage") -> None:
        self.model = self.model or other.model
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cost_usd = round(self.cost_usd + other.cost_usd, 6)


@dataclass(slots=True)
class CloudChunkResult:
    segments: list[dict] = field(default_factory=list)
    speakers: dict[str, str] = field(default_factory=dict)
    technical_terms: list[str] = field(default_factory=list)
    title: str = ""
    uncertain: list[dict] = field(default_factory=list)
    # Glossary corrections — empty for the raw STT providers, filled by
    # the LLM enrichment pass (or by Gemini's full-bundle response).
    corrections: list[dict] = field(default_factory=list)
    usage: CloudUsage = field(default_factory=CloudUsage)


def parse_cloud_response(
    payload: dict,
    *,
    model_id: str,
    chunk_offset_seconds: float = 0.0,
) -> CloudChunkResult:
    """Convert a raw ``generateContent`` response into pipeline shapes.

    Segments come back on the chunk-local timeline; we shift them by
    ``chunk_offset_seconds`` so the merged transcript reads on the
    source timeline, exactly like the VAD remap does locally.
    """
    candidates = payload.get("candidates") or []
    if not candidates:
        block = ((payload.get("promptFeedback") or {}).get("blockReason") or "").strip()
        detail = f" (raison : {block})" if block else ""
        raise CloudTranscriptionError(
            "Gemini n'a renvoyé aucune transcription" + detail + "."
        )
    parts = ((candidates[0].get("content") or {}).get("parts")) or []
    text = "".join(str(part.get("text") or "") for part in parts).strip()
    if not text:
        finish = str(candidates[0].get("finishReason") or "").strip()
        raise CloudTranscriptionError(
            "Réponse Gemini vide"
            + (f" (finishReason : {finish})" if finish else "")
            + "."
        )
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise CloudTranscriptionError(
            f"Réponse Gemini illisible (JSON invalide) : {exc}"
        ) from exc

    result = CloudChunkResult()
    result.title = str(data.get("title") or "").strip()
    for entry in data.get("speakers") or []:
        if not isinstance(entry, dict):
            continue
        label = str(entry.get("label") or "").strip()
        if label:
            result.speakers[label] = str(entry.get("name") or "").strip()
    result.technical_terms = [
        str(t).strip() for t in (data.get("technical_terms") or []) if str(t).strip()
    ]
    for entry in data.get("uncertain") or []:
        if not isinstance(entry, dict):
            continue
        text_value = str(entry.get("text") or "").strip()
        if not text_value:
            continue
        result.uncertain.append(
            {
                "timestamp": str(entry.get("timestamp") or "").strip(),
                "text": text_value,
                "reason": str(entry.get("reason") or "").strip(),
            }
        )

    previous_start = 0.0
    for entry in data.get("segments") or []:
        if not isinstance(entry, dict):
            continue
        seg_text = str(entry.get("text") or "").strip()
        if not seg_text:
            continue
        start = parse_cloud_timestamp(entry.get("start"))
        end = parse_cloud_timestamp(entry.get("end"))
        if start is None:
            start = previous_start
        if end is None or end < start:
            # ~3 words/second of French speech keeps the synthetic end
            # plausible for SRT/VTT rendering.
            end = start + max(len(seg_text.split()) / 3.0, 1.0)
        previous_start = start
        result.segments.append(
            {
                "start": round(start + chunk_offset_seconds, 2),
                "end": round(end + chunk_offset_seconds, 2),
                "speaker": str(entry.get("speaker") or "").strip(),
                "text": seg_text,
            }
        )
    if not result.segments:
        raise CloudTranscriptionError(
            "Gemini a répondu mais sans aucun segment exploitable."
        )

    meta = payload.get("usageMetadata") or {}
    input_tokens = int(meta.get("promptTokenCount") or 0)
    # Gemini 3.x bills thinking tokens as output; fold them in so the
    # tracked cost matches the invoice, not just the visible text.
    output_tokens = int(meta.get("candidatesTokenCount") or 0) + int(
        meta.get("thoughtsTokenCount") or 0
    )
    result.usage = CloudUsage(
        model=model_id,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=compute_cost_usd(model_id, input_tokens, output_tokens),
    )
    return result


def merge_chunk_results(chunks: list[CloudChunkResult]) -> CloudChunkResult:
    """Stitch per-window results into one meeting-level result."""
    merged = CloudChunkResult()
    seen_terms: set[str] = set()
    for chunk in chunks:
        merged.segments.extend(chunk.segments)
        for label, name in chunk.speakers.items():
            if name or label not in merged.speakers:
                merged.speakers[label] = name
        for term in chunk.technical_terms:
            key = term.lower()
            if key not in seen_terms:
                seen_terms.add(key)
                merged.technical_terms.append(term)
        merged.uncertain.extend(chunk.uncertain)
        merged.corrections.extend(chunk.corrections)
        if not merged.title and chunk.title:
            merged.title = chunk.title
        merged.usage.add(chunk.usage)
    merged.segments.sort(key=lambda seg: float(seg.get("start") or 0))
    return merged


# ---------------------------------------------------------------------------
# HTTP client
# ---------------------------------------------------------------------------


def _ssl_context() -> ssl.SSLContext:
    if certifi is not None:
        return ssl.create_default_context(cafile=certifi.where())
    return ssl.create_default_context()


def _friendly_http_error(
    exc: urllib.error.HTTPError, provider_label: str = "Gemini"
) -> CloudTranscriptionError:
    try:
        body = exc.read().decode("utf-8", errors="replace")
    except Exception:
        body = ""
    detail = ""
    try:
        payload = json.loads(body)
        # Providers nest the message differently: Gemini/OpenAI use
        # {"error": {"message": ...}}, AssemblyAI {"error": "..."},
        # Deepgram {"err_msg": ...}, Gladia {"message": ...}.
        err = payload.get("error")
        if isinstance(err, dict):
            detail = str(err.get("message") or "").strip()
        elif isinstance(err, str):
            detail = err.strip()
        if not detail:
            detail = str(
                payload.get("message") or payload.get("err_msg") or ""
            ).strip()
    except Exception:
        detail = body[:300].strip()
    if exc.code in {401, 403}:
        return CloudTranscriptionError(
            f"Clé API {provider_label} refusée. Vérifiez la clé dans Réglages → "
            "Transcription Cloud." + (f" (Détail : {detail})" if detail else ""),
            code="cloud_auth",
        )
    if exc.code == 429:
        return CloudTranscriptionError(
            f"Quota {provider_label} atteint (HTTP 429). Réessayez dans quelques "
            "minutes ou vérifiez votre offre API."
            + (f" (Détail : {detail})" if detail else ""),
            code="cloud_quota",
        )
    return CloudTranscriptionError(
        f"Erreur API {provider_label} (HTTP {exc.code})"
        + (f" : {detail}" if detail else ".")
    )


class GeminiClient:
    """Thin wrapper over the Files + generateContent REST endpoints.

    ``opener`` is injectable for tests — same pattern as
    ``web_context.fetch_web_context``.
    """

    def __init__(
        self,
        api_key: str,
        *,
        opener: Callable[..., Any] | None = None,
        timeout: float = 600.0,
    ):
        key = (api_key or "").strip()
        if not key:
            raise CloudTranscriptionError(
                "Aucune clé API Gemini configurée. Ajoutez-la dans "
                "Réglages → Transcription Cloud.",
                code="cloud_auth",
            )
        self.api_key = key
        self.timeout = timeout
        self._opener = opener
        self._context = None if opener else _ssl_context()

    # -- low-level ----------------------------------------------------

    def _open(self, request: urllib.request.Request) -> Any:
        try:
            if self._opener is not None:
                return self._opener(request, timeout=self.timeout)
            return urllib.request.urlopen(
                request, timeout=self.timeout, context=self._context
            )
        except urllib.error.HTTPError as exc:
            raise _friendly_http_error(exc) from exc
        except urllib.error.URLError as exc:
            raise CloudTranscriptionError(
                f"Impossible de joindre l'API Gemini : {exc.reason}. "
                "Vérifiez la connexion internet.",
                code="cloud_network",
            ) from exc

    def _json_request(
        self,
        method: str,
        url: str,
        payload: dict | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[dict, dict[str, str]]:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(url, data=data, method=method)
        request.add_header("x-goog-api-key", self.api_key)
        if data is not None:
            request.add_header("Content-Type", "application/json")
        for key, value in (headers or {}).items():
            request.add_header(key, value)
        with self._open(request) as response:
            raw = response.read().decode("utf-8", errors="replace")
            response_headers = {k.lower(): v for k, v in response.headers.items()}
        body: dict = {}
        if raw.strip():
            try:
                body = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise CloudTranscriptionError(
                    f"Réponse Gemini illisible : {exc}"
                ) from exc
        return body, response_headers

    # -- public surface -------------------------------------------------

    def check_access(self) -> dict[str, Any]:
        """Cheap key validation: list the available models.

        Returns ``{"ok": True, "models": [ids…]}`` — the CLI surfaces
        this behind the Réglages « Vérifier la clé » button.
        """
        body, _ = self._json_request(
            "GET", f"{GEMINI_API_BASE}/v1beta/models?pageSize=50"
        )
        models = [
            str(entry.get("name") or "").removeprefix("models/")
            for entry in body.get("models") or []
        ]
        return {"ok": True, "models": [m for m in models if m]}

    def upload_audio(self, path: str, *, display_name: str = "") -> dict[str, Any]:
        """Resumable upload; returns the file resource (uri + name)."""
        from pathlib import Path as _Path

        file_path = _Path(path)
        if not file_path.exists():
            raise CloudTranscriptionError(
                f"Fichier audio à téléverser introuvable : {path}"
            )
        size = file_path.stat().st_size
        mime = "audio/mp3" if file_path.suffix.lower() == ".mp3" else "audio/wav"
        _, headers = self._json_request(
            "POST",
            f"{GEMINI_API_BASE}/upload/v1beta/files",
            payload={"file": {"display_name": display_name or file_path.name}},
            headers={
                "X-Goog-Upload-Protocol": "resumable",
                "X-Goog-Upload-Command": "start",
                "X-Goog-Upload-Header-Content-Length": str(size),
                "X-Goog-Upload-Header-Content-Type": mime,
            },
        )
        upload_url = headers.get("x-goog-upload-url", "")
        if not upload_url:
            raise CloudTranscriptionError(
                "L'API Gemini n'a pas renvoyé d'URL de téléversement."
            )
        request = urllib.request.Request(
            upload_url, data=file_path.read_bytes(), method="POST"
        )
        request.add_header("X-Goog-Upload-Offset", "0")
        request.add_header("X-Goog-Upload-Command", "upload, finalize")
        request.add_header("Content-Type", mime)
        with self._open(request) as response:
            raw = response.read().decode("utf-8", errors="replace")
        try:
            body = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise CloudTranscriptionError(
                f"Téléversement Gemini : réponse illisible : {exc}"
            ) from exc
        file_info = body.get("file") or {}
        if not file_info.get("uri"):
            raise CloudTranscriptionError(
                "Téléversement Gemini incomplet (pas d'URI de fichier)."
            )
        return file_info

    def wait_until_active(
        self,
        file_info: dict[str, Any],
        *,
        poll_seconds: float = 2.0,
        max_polls: int = 150,
        sleeper: Callable[[float], None] | None = None,
    ) -> dict[str, Any]:
        """Poll the file resource until Google finishes ingesting it."""
        import time as _time

        sleep = sleeper or _time.sleep
        info = dict(file_info)
        polls = 0
        while str(info.get("state") or "").upper() == "PROCESSING":
            if polls >= max_polls:
                raise CloudTranscriptionError(
                    "Le fichier audio est resté trop longtemps en cours de "
                    "traitement côté Gemini. Réessayez."
                )
            sleep(poll_seconds)
            polls += 1
            name = str(info.get("name") or "")
            info, _ = self._json_request(
                "GET", f"{GEMINI_API_BASE}/v1beta/{name}"
            )
        if str(info.get("state") or "").upper() == "FAILED":
            raise CloudTranscriptionError(
                "Gemini n'a pas pu ingérer le fichier audio (état FAILED)."
            )
        return info

    def delete_file(self, file_info: dict[str, Any]) -> None:
        """Best-effort remote cleanup — meeting audio shouldn't linger
        on Google's servers longer than the API call that needed it
        (they auto-expire after 48 h, but explicit is better)."""
        name = str(file_info.get("name") or "")
        if not name:
            return
        try:
            self._json_request(
                "DELETE", f"{GEMINI_API_BASE}/v1beta/{name}"
            )
        except CloudTranscriptionError:
            pass

    def generate_transcription(
        self,
        *,
        model_id: str,
        file_uri: str,
        mime_type: str,
        prompt: str,
    ) -> dict:
        """One ``generateContent`` call; returns the raw response."""
        entry = cloud_model_entry(model_id)
        generation_config: dict[str, Any] = {
            "responseMimeType": "application/json",
            "responseSchema": RESPONSE_SCHEMA,
            "temperature": 0.2,
        }
        thinking = entry.get("thinking")
        if thinking == "level_low":
            generation_config["thinkingConfig"] = {"thinkingLevel": "low"}
        elif thinking == "budget_zero":
            generation_config["thinkingConfig"] = {"thinkingBudget": 0}
        payload = {
            "contents": [
                {
                    "parts": [
                        {"fileData": {"mimeType": mime_type, "fileUri": file_uri}},
                        {"text": prompt},
                    ]
                }
            ],
            "generationConfig": generation_config,
        }
        url = (
            f"{GEMINI_API_BASE}/v1beta/models/"
            f"{canonical_cloud_model_id(model_id)}:generateContent"
        )
        try:
            body, _ = self._json_request("POST", url, payload=payload)
        except CloudTranscriptionError as exc:
            # The thinking knob is the most model-version-sensitive part
            # of the request; if a future model rejects it, retry plain
            # rather than failing the meeting.
            if "thinking" not in str(exc).lower() or "thinkingConfig" not in str(payload):
                raise
            generation_config.pop("thinkingConfig", None)
            body, _ = self._json_request("POST", url, payload=payload)
        return body


# ---------------------------------------------------------------------------
# Provider abstraction
#
# Two paradigms behind one interface:
#
#  * Multimodal LLM (Gemini): one call returns transcript + diarisation
#    + title + speaker names + glossary corrections.
#  * Dedicated STT (OpenAI transcribe, AssemblyAI, Deepgram, Gladia):
#    returns transcript + native diarisation only. The pipeline then
#    runs the existing LLM post-pass for title / names / corrections
#    (``needs_enrichment`` on the catalogue entry).
#
# Every provider implements ``transcribe(audio_path, model_id, context)
# -> CloudChunkResult`` for one audio window, and ``check_access()`` for
# the Réglages key check. STT providers bill per audio hour, so their
# usage carries cost derived from the window duration rather than from
# token counters.
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class CloudPromptContext:
    """Everything a provider needs to transcribe one window with the
    user's vocabulary, expected participants and cross-window
    continuity. Mirrors the arguments of :func:`build_cloud_prompt`."""

    language: str = "fr"
    glossary_terms: list[str] = field(default_factory=list)
    expected_speaker_names: list[str] = field(default_factory=list)
    meeting_context: str = ""
    odoo_context: str = ""
    known_speakers: dict[str, str] = field(default_factory=dict)
    chunk_index: int = 0
    chunk_count: int = 1
    chunk_offset_seconds: float = 0.0
    chunk_duration_seconds: float = 0.0
    previous_tail: str = ""


def _normalized_speaker(raw: Any, label_map: dict[str, str]) -> str:
    """Map a provider's raw speaker tag ("A", 0, "speaker_1") to a
    stable, user-friendly "Intervenant N" *within one window*. Empty
    tags (no diarisation) stay empty so the renderer omits the prefix.

    Cross-window identity is intentionally not reconciled here — STT
    diarisation labels aren't speaker IDs, so window 2's "A" may be a
    different person than window 1's. The LLM enrichment pass and the
    rename sheet reconcile names across the full transcript."""
    key = str(raw).strip()
    if not key:
        return ""
    if key not in label_map:
        label_map[key] = f"Intervenant {len(label_map) + 1}"
    return label_map[key]


def _segments_from_plain_text(text: str, duration_seconds: float) -> list[dict]:
    """Fallback segmentation when a provider returns only flat text
    (e.g. OpenAI transcribe without per-segment timestamps). Splits on
    sentence boundaries and distributes timestamps proportionally to
    word count across the window so the transcript stays navigable."""
    cleaned = (text or "").strip()
    if not cleaned:
        return []
    pieces = re.split(r"(?<=[.!?…])\s+", cleaned)
    pieces = [p.strip() for p in pieces if p.strip()]
    if not pieces:
        return []
    total_words = sum(len(p.split()) for p in pieces) or 1
    span = max(float(duration_seconds or 0), float(len(pieces)))
    cursor = 0.0
    out: list[dict] = []
    for piece in pieces:
        share = len(piece.split()) / total_words
        seg_len = max(span * share, 0.5)
        out.append(
            {"start": round(cursor, 2), "end": round(cursor + seg_len, 2),
             "speaker": "", "text": piece}
        )
        cursor += seg_len
    return out


def _result_from_utterances(
    utterances: list[dict],
    *,
    model_id: str,
    context: CloudPromptContext,
) -> CloudChunkResult:
    """Build a :class:`CloudChunkResult` from a provider's utterance
    list. Each utterance is ``{start, end, speaker, text}`` in
    *chunk-local seconds*; we offset to the source timeline and bill
    the window by duration (per-hour providers)."""
    offset = context.chunk_offset_seconds
    label_map: dict[str, str] = {}
    result = CloudChunkResult()
    for utt in utterances:
        text = str(utt.get("text") or "").strip()
        if not text:
            continue
        try:
            start = float(utt.get("start") or 0.0)
            end = float(utt.get("end") or 0.0)
        except (TypeError, ValueError):
            continue
        if end < start:
            end = start + max(len(text.split()) / 3.0, 1.0)
        result.segments.append(
            {
                "start": round(start + offset, 2),
                "end": round(end + offset, 2),
                "speaker": _normalized_speaker(utt.get("speaker"), label_map),
                "text": text,
            }
        )
    if not result.segments:
        raise CloudTranscriptionError(
            "Le fournisseur a répondu mais sans aucun segment exploitable."
        )
    result.usage = CloudUsage(
        model=model_id,
        input_tokens=0,
        output_tokens=0,
        cost_usd=cost_for_duration(model_id, context.chunk_duration_seconds),
    )
    return result


def _multipart_body(
    fields: dict[str, str],
    *,
    file_field: str,
    filename: str,
    file_bytes: bytes,
    file_content_type: str,
) -> tuple[bytes, str]:
    """Hand-rolled multipart/form-data (no third-party deps). The
    boundary is fixed and improbable — deterministic for tests and not
    present in audio payloads."""
    boundary = "----EkoVideoBoundary7MA4YWxkTrZu0gW"
    crlf = b"\r\n"
    bb = boundary.encode()
    parts: list[bytes] = []
    for name, value in fields.items():
        parts += [
            b"--" + bb,
            b'Content-Disposition: form-data; name="' + name.encode() + b'"',
            b"",
            str(value).encode("utf-8"),
        ]
    parts += [
        b"--" + bb,
        b'Content-Disposition: form-data; name="' + file_field.encode()
        + b'"; filename="' + filename.encode() + b'"',
        b"Content-Type: " + file_content_type.encode(),
        b"",
        file_bytes,
        b"--" + bb + b"--",
        b"",
    ]
    return crlf.join(parts), f"multipart/form-data; boundary={boundary}"


def _audio_mime(path: str) -> str:
    from pathlib import Path as _Path

    return "audio/mpeg" if _Path(path).suffix.lower() == ".mp3" else "audio/wav"


class CloudProvider:
    """Base class: HTTP plumbing + the two-method contract."""

    provider_id = ""
    provider_label = ""

    def __init__(
        self,
        api_key: str,
        *,
        opener: Callable[..., Any] | None = None,
        timeout: float = 600.0,
    ):
        key = (api_key or "").strip()
        if not key:
            raise CloudTranscriptionError(
                f"Aucune clé API {self.provider_label} configurée. Ajoutez-la "
                "dans Réglages → Transcription Cloud.",
                code="cloud_auth",
            )
        self.api_key = key
        self.timeout = timeout
        self._opener = opener
        self._context = None if opener else _ssl_context()

    # -- HTTP ---------------------------------------------------------

    def _open(self, request: urllib.request.Request) -> Any:
        try:
            if self._opener is not None:
                return self._opener(request, timeout=self.timeout)
            return urllib.request.urlopen(
                request, timeout=self.timeout, context=self._context
            )
        except urllib.error.HTTPError as exc:
            raise _friendly_http_error(exc, self.provider_label) from exc
        except urllib.error.URLError as exc:
            raise CloudTranscriptionError(
                f"Impossible de joindre l'API {self.provider_label} : {exc.reason}. "
                "Vérifiez la connexion internet.",
                code="cloud_network",
            ) from exc

    def _request(
        self,
        method: str,
        url: str,
        *,
        data: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[bytes, dict[str, str]]:
        request = urllib.request.Request(url, data=data, method=method)
        for key, value in (headers or {}).items():
            request.add_header(key, value)
        with self._open(request) as response:
            raw = response.read()
            resp_headers = {k.lower(): v for k, v in response.headers.items()}
        return raw, resp_headers

    def _json(
        self,
        method: str,
        url: str,
        *,
        json_body: dict | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[dict, dict[str, str]]:
        data = json.dumps(json_body).encode("utf-8") if json_body is not None else None
        merged = dict(headers or {})
        if data is not None:
            merged.setdefault("Content-Type", "application/json")
        raw, resp_headers = self._request(method, url, data=data, headers=merged)
        body: dict = {}
        text = raw.decode("utf-8", errors="replace") if raw else ""
        if text.strip():
            try:
                body = json.loads(text)
            except json.JSONDecodeError as exc:
                raise CloudTranscriptionError(
                    f"Réponse {self.provider_label} illisible : {exc}"
                ) from exc
        return body, resp_headers

    def _poll(
        self,
        url: str,
        *,
        headers: dict[str, str],
        is_done: Callable[[dict], bool],
        is_failed: Callable[[dict], bool],
        poll_seconds: float = 3.0,
        max_polls: int = 200,
        sleeper: Callable[[float], None] | None = None,
    ) -> dict:
        import time as _time

        sleep = sleeper or _time.sleep
        polls = 0
        while True:
            body, _ = self._json("GET", url, headers=headers)
            if is_failed(body):
                raise CloudTranscriptionError(
                    f"{self.provider_label} a échoué à transcrire l'audio."
                )
            if is_done(body):
                return body
            if polls >= max_polls:
                raise CloudTranscriptionError(
                    f"{self.provider_label} : délai dépassé en attendant la "
                    "transcription."
                )
            polls += 1
            sleep(poll_seconds)

    # -- contract -----------------------------------------------------

    def _catalogue_models(self) -> list[str]:
        """Model ids the app actually offers for this provider — what
        the key-check surfaces, so the count/names stay in sync with
        the catalogue instead of a stale hardcoded literal."""
        return [m["id"] for m in cloud_models_for_provider(self.provider_id)]

    def check_access(self) -> dict[str, Any]:
        raise NotImplementedError

    def transcribe(
        self, audio_path: str, *, model_id: str, context: CloudPromptContext
    ) -> CloudChunkResult:
        raise NotImplementedError


class GeminiProvider(CloudProvider):
    provider_id = "gemini"
    provider_label = "Gemini"

    def __init__(self, api_key: str, **kwargs):
        super().__init__(api_key, **kwargs)
        self._client = GeminiClient(api_key, opener=self._opener, timeout=self.timeout)

    def check_access(self) -> dict[str, Any]:
        # The live call validates the key (raises on failure); we then
        # report the models the app actually offers, not the full raw
        # API list.
        self._client.check_access()
        return {"ok": True, "models": self._catalogue_models()}

    def transcribe(
        self, audio_path: str, *, model_id: str, context: CloudPromptContext
    ) -> CloudChunkResult:
        prompt = build_cloud_prompt(
            language=context.language,
            glossary_terms=context.glossary_terms,
            expected_speaker_names=context.expected_speaker_names,
            meeting_context=context.meeting_context,
            odoo_context=context.odoo_context,
            known_speakers=context.known_speakers,
            chunk_index=context.chunk_index,
            chunk_count=context.chunk_count,
            chunk_offset_seconds=context.chunk_offset_seconds,
            previous_tail=context.previous_tail,
        )
        file_info: dict = {}
        try:
            file_info = self._client.upload_audio(audio_path)
            file_info = self._client.wait_until_active(file_info)
            raw = self._client.generate_transcription(
                model_id=model_id,
                file_uri=str(file_info.get("uri") or ""),
                mime_type=str(file_info.get("mimeType") or "audio/mp3"),
                prompt=prompt,
            )
            return parse_cloud_response(
                raw, model_id=model_id, chunk_offset_seconds=context.chunk_offset_seconds
            )
        finally:
            self._client.delete_file(file_info)


class OpenAITranscribeProvider(CloudProvider):
    provider_id = "openai"
    provider_label = "OpenAI"

    def _auth(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    def check_access(self) -> dict[str, Any]:
        # Validate the key against the live endpoint, then report the
        # transcription models the app offers (not OpenAI's full list).
        self._json("GET", f"{OPENAI_API_BASE}/v1/models", headers=self._auth())
        return {"ok": True, "models": self._catalogue_models()}

    def transcribe(
        self, audio_path: str, *, model_id: str, context: CloudPromptContext
    ) -> CloudChunkResult:
        from pathlib import Path as _Path

        entry = cloud_model_entry(model_id)
        api_model = str(entry.get("api_model") or model_id)
        prompt_terms = ", ".join(
            t for t in [*context.glossary_terms, *context.expected_speaker_names]
            if (t or "").strip()
        )
        fields: dict[str, str] = {"model": api_model, "response_format": "json"}
        if (context.language or "").strip():
            fields["language"] = context.language
        if prompt_terms:
            fields["prompt"] = (
                "Vocabulaire et participants attendus : " + prompt_terms
            )
        body_bytes = _Path(audio_path).read_bytes()
        multipart, content_type = _multipart_body(
            fields,
            file_field="file",
            filename=_Path(audio_path).name,
            file_bytes=body_bytes,
            file_content_type=_audio_mime(audio_path),
        )
        raw, _ = self._request(
            "POST",
            f"{OPENAI_API_BASE}/v1/audio/transcriptions",
            data=multipart,
            headers={**self._auth(), "Content-Type": content_type},
        )
        try:
            payload = json.loads(raw.decode("utf-8", errors="replace") or "{}")
        except json.JSONDecodeError as exc:
            raise CloudTranscriptionError(
                f"Réponse OpenAI illisible : {exc}"
            ) from exc
        # gpt-4o-transcribe-diarize returns per-speaker segments; the
        # plain/mini models return flat text. Handle both, plus the
        # legacy verbose_json "segments" shape, defensively.
        raw_segments = payload.get("segments") or payload.get("utterances") or []
        utterances: list[dict] = []
        for seg in raw_segments:
            if not isinstance(seg, dict):
                continue
            utterances.append(
                {
                    "start": parse_cloud_timestamp(seg.get("start")) or 0.0,
                    "end": parse_cloud_timestamp(seg.get("end")) or 0.0,
                    "speaker": seg.get("speaker") or seg.get("speaker_id") or "",
                    "text": seg.get("text") or seg.get("transcript") or "",
                }
            )
        if not utterances:
            text = str(payload.get("text") or "")
            utterances = _segments_from_plain_text(text, context.chunk_duration_seconds)
        return _result_from_utterances(utterances, model_id=model_id, context=context)


class AssemblyAIProvider(CloudProvider):
    provider_id = "assemblyai"
    provider_label = "AssemblyAI"

    def _auth(self) -> dict[str, str]:
        return {"authorization": self.api_key}

    def check_access(self) -> dict[str, Any]:
        # Listing transcripts is the cheapest authenticated GET.
        self._json(
            "GET",
            f"{ASSEMBLYAI_API_BASE}/v2/transcript?limit=1",
            headers=self._auth(),
        )
        return {"ok": True, "models": self._catalogue_models()}

    def transcribe(
        self, audio_path: str, *, model_id: str, context: CloudPromptContext
    ) -> CloudChunkResult:
        from pathlib import Path as _Path

        upload_body, _ = self._request(
            "POST",
            f"{ASSEMBLYAI_API_BASE}/v2/upload",
            data=_Path(audio_path).read_bytes(),
            headers={**self._auth(), "Content-Type": "application/octet-stream"},
        )
        try:
            upload_url = json.loads(upload_body.decode("utf-8")).get("upload_url")
        except (json.JSONDecodeError, AttributeError):
            upload_url = None
        if not upload_url:
            raise CloudTranscriptionError("Téléversement AssemblyAI sans URL.")
        config: dict[str, Any] = {
            "audio_url": upload_url,
            "speaker_labels": True,
            "punctuate": True,
            "format_text": True,
        }
        lang = (context.language or "").strip()
        if lang:
            config["language_code"] = lang
        else:
            config["language_detection"] = True
        boost = [t for t in context.glossary_terms if (t or "").strip()]
        if boost:
            config["word_boost"] = boost[:1000]
        created, _ = self._json(
            "POST",
            f"{ASSEMBLYAI_API_BASE}/v2/transcript",
            json_body=config,
            headers=self._auth(),
        )
        transcript_id = str(created.get("id") or "")
        if not transcript_id:
            raise CloudTranscriptionError("AssemblyAI n'a pas créé de transcription.")
        done = self._poll(
            f"{ASSEMBLYAI_API_BASE}/v2/transcript/{transcript_id}",
            headers=self._auth(),
            is_done=lambda b: str(b.get("status")) == "completed",
            is_failed=lambda b: str(b.get("status")) == "error",
        )
        utterances = [
            {
                "start": float(u.get("start") or 0) / 1000.0,
                "end": float(u.get("end") or 0) / 1000.0,
                "speaker": u.get("speaker") or "",
                "text": u.get("text") or "",
            }
            for u in (done.get("utterances") or [])
            if isinstance(u, dict)
        ]
        if not utterances and done.get("text"):
            utterances = _segments_from_plain_text(
                str(done.get("text")), context.chunk_duration_seconds
            )
        return _result_from_utterances(utterances, model_id=model_id, context=context)


class DeepgramProvider(CloudProvider):
    provider_id = "deepgram"
    provider_label = "Deepgram"

    def _auth(self) -> dict[str, str]:
        return {"Authorization": f"Token {self.api_key}"}

    def check_access(self) -> dict[str, Any]:
        self._json("GET", f"{DEEPGRAM_API_BASE}/v1/projects", headers=self._auth())
        return {"ok": True, "models": self._catalogue_models()}

    def transcribe(
        self, audio_path: str, *, model_id: str, context: CloudPromptContext
    ) -> CloudChunkResult:
        from pathlib import Path as _Path
        from urllib.parse import urlencode

        entry = cloud_model_entry(model_id)
        params = {
            "model": str(entry.get("api_model") or "nova-3"),
            "diarize": "true",
            "punctuate": "true",
            "utterances": "true",
            "smart_format": "true",
        }
        lang = (context.language or "").strip()
        if lang:
            params["language"] = lang
        else:
            params["detect_language"] = "true"
        query = urlencode(params)
        keyterms = [t for t in context.glossary_terms if (t or "").strip()]
        if keyterms:
            query += "".join(
                "&keyterm=" + urllib.parse.quote(t) for t in keyterms[:100]
            )
        raw, _ = self._request(
            "POST",
            f"{DEEPGRAM_API_BASE}/v1/listen?{query}",
            data=_Path(audio_path).read_bytes(),
            headers={**self._auth(), "Content-Type": _audio_mime(audio_path)},
        )
        try:
            payload = json.loads(raw.decode("utf-8", errors="replace") or "{}")
        except json.JSONDecodeError as exc:
            raise CloudTranscriptionError(f"Réponse Deepgram illisible : {exc}") from exc
        results = payload.get("results") or {}
        utterances = [
            {
                "start": float(u.get("start") or 0),
                "end": float(u.get("end") or 0),
                "speaker": f"spk{u.get('speaker')}" if u.get("speaker") is not None else "",
                "text": u.get("transcript") or "",
            }
            for u in (results.get("utterances") or [])
            if isinstance(u, dict)
        ]
        if not utterances:
            # Fall back to the channel transcript when utterances are off.
            channels = results.get("channels") or []
            if channels:
                alt = ((channels[0].get("alternatives") or [{}])[0])
                text = str(alt.get("transcript") or "")
                utterances = _segments_from_plain_text(
                    text, context.chunk_duration_seconds
                )
        return _result_from_utterances(utterances, model_id=model_id, context=context)


class GladiaProvider(CloudProvider):
    provider_id = "gladia"
    provider_label = "Gladia"

    def _auth(self) -> dict[str, str]:
        return {"x-gladia-key": self.api_key}

    def check_access(self) -> dict[str, Any]:
        self._json(
            "GET",
            f"{GLADIA_API_BASE}/v2/pre-recorded?limit=1",
            headers=self._auth(),
        )
        return {"ok": True, "models": self._catalogue_models()}

    def transcribe(
        self, audio_path: str, *, model_id: str, context: CloudPromptContext
    ) -> CloudChunkResult:
        from pathlib import Path as _Path

        multipart, content_type = _multipart_body(
            {},
            file_field="audio",
            filename=_Path(audio_path).name,
            file_bytes=_Path(audio_path).read_bytes(),
            file_content_type=_audio_mime(audio_path),
        )
        # _json only sends JSON; the multipart upload goes through the
        # raw _request path.
        raw, _ = self._request(
            "POST",
            f"{GLADIA_API_BASE}/v2/upload",
            data=multipart,
            headers={**self._auth(), "Content-Type": content_type},
        )
        try:
            audio_url = json.loads(raw.decode("utf-8")).get("audio_url")
        except (json.JSONDecodeError, AttributeError):
            audio_url = None
        if not audio_url:
            raise CloudTranscriptionError("Téléversement Gladia sans URL audio.")
        config: dict[str, Any] = {"audio_url": audio_url, "diarization": True}
        api_model = str(cloud_model_entry(model_id).get("api_model") or "").strip()
        if api_model:
            # Without an explicit model Gladia defaults to Solaria-1;
            # send it so Solaria-3 (and any future model) is honoured.
            config["model"] = api_model
        lang = (context.language or "").strip()
        if lang:
            config["language"] = lang
        else:
            config["detect_language"] = True
        vocab = [t for t in context.glossary_terms if (t or "").strip()]
        if vocab:
            config["custom_vocabulary"] = vocab[:1000]
        created, _ = self._json(
            "POST",
            f"{GLADIA_API_BASE}/v2/pre-recorded",
            json_body=config,
            headers=self._auth(),
        )
        result_url = str(created.get("result_url") or "")
        if not result_url:
            transcript_id = str(created.get("id") or "")
            if not transcript_id:
                raise CloudTranscriptionError("Gladia n'a pas créé de transcription.")
            result_url = f"{GLADIA_API_BASE}/v2/pre-recorded/{transcript_id}"
        done = self._poll(
            result_url,
            headers=self._auth(),
            is_done=lambda b: str(b.get("status")) == "done",
            is_failed=lambda b: str(b.get("status")) == "error",
        )
        transcription = (done.get("result") or {}).get("transcription") or {}
        utterances = [
            {
                "start": float(u.get("start") or 0),
                "end": float(u.get("end") or 0),
                "speaker": f"spk{u.get('speaker')}" if u.get("speaker") is not None else "",
                "text": u.get("text") or "",
            }
            for u in (transcription.get("utterances") or [])
            if isinstance(u, dict)
        ]
        if not utterances and transcription.get("full_transcript"):
            utterances = _segments_from_plain_text(
                str(transcription.get("full_transcript")),
                context.chunk_duration_seconds,
            )
        return _result_from_utterances(utterances, model_id=model_id, context=context)


_PROVIDER_CLASSES: dict[str, type[CloudProvider]] = {
    "gemini": GeminiProvider,
    "openai": OpenAITranscribeProvider,
    "assemblyai": AssemblyAIProvider,
    "deepgram": DeepgramProvider,
    "gladia": GladiaProvider,
}


def get_cloud_provider(
    provider_id: str,
    api_key: str,
    *,
    opener: Callable[..., Any] | None = None,
    timeout: float = 600.0,
) -> CloudProvider:
    cls = _PROVIDER_CLASSES.get((provider_id or "").strip().lower())
    if cls is None:
        raise CloudTranscriptionError(
            f"Fournisseur cloud inconnu : {provider_id!r}.", code="cloud_provider"
        )
    return cls(api_key, opener=opener, timeout=timeout)
