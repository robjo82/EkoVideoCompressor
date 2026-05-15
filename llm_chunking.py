"""Window a rendered transcript into LLM-friendly chunks.

The local LLM correction pass runs in a managed venv with a fixed
input cap (~30 000 chars in the embedded script). A two-hour meeting
easily produces 80 000+ chars of timestamped dialogue, so a single
call covers maybe a third of the recording — all the corrections
land in the opening minutes and the rest goes unreviewed.

This module slices the rendered transcript at line boundaries with
a configurable overlap, so each chunk respects the LLM input cap
while preserving enough trailing context that proper-noun references
spanning two windows still get caught.

We split at line boundaries (the renderer puts one segment per
line, prefixed with ``[HH:MM:SS]``), never mid-sentence. The
overlap is expressed in characters so callers can tune it against
the chosen model's prompt cap without thinking in tokens.
"""

from __future__ import annotations

from dataclasses import dataclass


__all__ = [
    "TranscriptChunk",
    "chunk_transcript_for_llm",
    "dedupe_corrections",
]


@dataclass(frozen=True)
class TranscriptChunk:
    """One window of a chunked transcript.

    ``index`` is 0-based; ``total`` is the full chunk count.
    ``text`` is the window content (already overlap-padded); the
    LLM sees this verbatim. Timestamps inside the text are global
    (e.g. ``[01:23:45]``) because the renderer never rewrites them.
    """

    index: int
    total: int
    text: str


def chunk_transcript_for_llm(
    text: str,
    *,
    max_chars: int = 22_000,
    overlap_chars: int = 1_500,
) -> list[TranscriptChunk]:
    """Split ``text`` into overlapping chunks at line boundaries.

    Defaults sized for a 7 B / 4-bit French LLM with a 32 k prompt
    cap: 22 000 chars of body + ~1 500 chars of overlap stays safely
    under the embedded script's 30 000-char cap, while leaving room
    for the system prompt + glossary + instructions.

    A short transcript (≤ ``max_chars``) returns a single chunk so
    callers can stay on the existing one-shot code path.

    The overlap is achieved by walking *back* from the next-chunk
    cursor: we re-emit the trailing ``overlap_chars`` characters of
    the previous chunk inside the next one, snapped to the nearest
    preceding newline so the LLM never sees a half-line that would
    confuse its line-by-line correction format.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be > 0")
    if overlap_chars < 0:
        raise ValueError("overlap_chars must be >= 0")
    if overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be < max_chars")
    if not text:
        return [TranscriptChunk(index=0, total=1, text="")]

    if len(text) <= max_chars:
        return [TranscriptChunk(index=0, total=1, text=text)]

    chunks_text: list[str] = []
    cursor = 0
    text_len = len(text)
    while cursor < text_len:
        end = min(cursor + max_chars, text_len)
        # Snap ``end`` to the nearest newline that still falls inside
        # the window. We don't want to cut a segment line in half —
        # the LLM relies on the ``[HH:MM:SS]`` prefix to anchor each
        # correction it emits, and a half-line breaks that.
        if end < text_len:
            nl = text.rfind("\n", cursor, end)
            if nl > cursor + max_chars // 2:
                end = nl
        chunks_text.append(text[cursor:end])
        if end >= text_len:
            break
        # Pull the next cursor back by ``overlap_chars`` so each
        # chunk re-sees the tail of the previous one. Snap that
        # rewind to the next newline forward so we keep clean line
        # boundaries — overlap is allowed to be slightly larger than
        # requested, never smaller.
        next_cursor = max(end - overlap_chars, cursor + 1)
        nl_forward = text.find("\n", next_cursor)
        if 0 <= nl_forward < end:
            next_cursor = nl_forward + 1
        cursor = next_cursor

    total = len(chunks_text)
    return [
        TranscriptChunk(index=i, total=total, text=c)
        for i, c in enumerate(chunks_text)
    ]


# ---------------------------------------------------------------------------
# Merging corrections across chunks
# ---------------------------------------------------------------------------


def _norm(value: str) -> str:
    """Loose normaliser for dedup: case-insensitive, whitespace-
    collapsed. We don't strip accents here — two corrections that
    differ only in their accents are likely the same proposal,
    rendered differently by the LLM on each pass.
    """
    return " ".join((value or "").split()).lower()


def dedupe_corrections(items: list[dict]) -> list[dict]:
    """Drop duplicate corrections emitted across overlapping chunks.

    A correction is identified by ``(timestamp, original, replacement)``
    after normalisation. The first occurrence wins so the ``reason``
    text from the earliest chunk is preserved.

    Order of the returned list matches the order of first occurrence,
    so applying the corrections downstream still walks the transcript
    in reading order.
    """
    seen: set[tuple[str, str, str]] = set()
    out: list[dict] = []
    for c in items or []:
        key = (
            _norm(str(c.get("timestamp") or "")),
            _norm(str(c.get("original") or "")),
            _norm(str(c.get("replacement") or "")),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out
