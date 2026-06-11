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
# checkpoints. ``price_in_per_1m`` is the *audio* input price (what a
# transcription job actually pays), ``price_out_per_1m`` the text
# output price; both in USD per million tokens, from
# https://ai.google.dev/gemini-api/docs/pricing (June 2026).
#
# ``thinking`` selects the request knob that caps reasoning tokens —
# transcription is perception work, extended thinking only burns
# output budget. "level_low" is the Gemini 3.x dial, "budget_zero"
# the 2.5 one; the client retries without the knob if the API
# rejects it, so a model migration can't brick the feature.
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
        "price_in_per_1m": 1.50,
        "price_out_per_1m": 9.00,
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
        "price_in_per_1m": 4.00,
        "price_out_per_1m": 18.00,
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
        "price_in_per_1m": 1.00,
        "price_out_per_1m": 2.50,
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
        "price_in_per_1m": 0.50,
        "price_out_per_1m": 1.50,
        "thinking": "level_low",
    },
]

DEFAULT_CLOUD_MODEL = CLOUD_TRANSCRIPTION_MODELS[0]["id"]


class CloudTranscriptionError(RuntimeError):
    """Raised on any cloud failure, with a French user-facing message.

    ``code`` mirrors the engine's error-event codes so the SwiftUI
    layer can branch (e.g. ``cloud_budget_exceeded`` aborts the job,
    ``cloud_api`` falls back to the local pipeline).
    """

    def __init__(self, message: str, code: str = "cloud_api"):
        super().__init__(message)
        self.code = code


def cloud_model_entry(model_id: str) -> dict:
    raw = (model_id or "").strip() or DEFAULT_CLOUD_MODEL
    for entry in CLOUD_TRANSCRIPTION_MODELS:
        if entry["id"] == raw:
            return entry
    # Unknown id (user typed a fresh model before the catalogue knew
    # it): keep their choice, bill at the most expensive known rates
    # so the budget guard stays conservative.
    fallback = max(
        CLOUD_TRANSCRIPTION_MODELS,
        key=lambda e: (e["price_in_per_1m"], e["price_out_per_1m"]),
    )
    return {**fallback, "id": raw, "label": raw, "default": False}


def canonical_cloud_model_id(model_id: str) -> str:
    return cloud_model_entry(model_id)["id"]


def compute_cost_usd(model_id: str, input_tokens: int, output_tokens: int) -> float:
    entry = cloud_model_entry(model_id)
    cost = (
        max(input_tokens, 0) * entry["price_in_per_1m"]
        + max(output_tokens, 0) * entry["price_out_per_1m"]
    ) / 1_000_000
    return round(cost, 6)


def estimate_cloud_cost(duration_seconds: float, model_id: str) -> dict[str, Any]:
    """Project a job's cost from its audio duration.

    Used twice: by the SwiftUI Run Setup to display "≈ 0,35 $US" next
    to the cloud engine picker, and by the engine's budget guard
    before the first byte is uploaded. Deliberately rounds *up* (the
    prompt overhead and thinking tokens are folded into the output
    estimate) — an over-estimate that blocks a borderline job beats
    an under-estimate that overshoots the user's cap.
    """
    seconds = max(float(duration_seconds or 0), 0.0)
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


def _friendly_http_error(exc: urllib.error.HTTPError) -> CloudTranscriptionError:
    try:
        body = exc.read().decode("utf-8", errors="replace")
    except Exception:
        body = ""
    detail = ""
    try:
        payload = json.loads(body)
        detail = str(((payload.get("error") or {}).get("message")) or "").strip()
    except Exception:
        detail = body[:300].strip()
    if exc.code in {401, 403}:
        return CloudTranscriptionError(
            "Clé API Gemini refusée. Vérifiez la clé dans Réglages → "
            "Transcription Cloud." + (f" (Détail : {detail})" if detail else ""),
            code="cloud_auth",
        )
    if exc.code == 429:
        return CloudTranscriptionError(
            "Quota Gemini atteint (HTTP 429). Réessayez dans quelques "
            "minutes ou vérifiez votre offre API."
            + (f" (Détail : {detail})" if detail else ""),
            code="cloud_quota",
        )
    return CloudTranscriptionError(
        f"Erreur API Gemini (HTTP {exc.code})"
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
