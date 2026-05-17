"""Apply LLM-suggested corrections to a transcript file, safely.

The LLM correction pass produces a list of {timestamp, original,
replacement, confidence, reason} entries describing fixes the model
recommends. Until now, the pipeline only *listed* them in the review
report and wrote the "améliorée" transcript byte-identical to the
raw one — so the user had to copy/paste corrections by hand.

This module finally closes that loop. It walks the list, validates
each correction against two guardrails, then rewrites the text.

Guardrails (both must pass — they cover different failure modes):

1. **Original must actually exist in the transcript.** The LLM
   sometimes hallucinates an `original` that paraphrases the audio.
   We refuse to substitute anything that isn't a literal (case-
   insensitive, whitespace-normalised) substring.

2. **Replacement must stay phonetically close to the original.** A
   correction that rewrites the meaning of a sentence is a style
   edit, not a transcription fix. We cap the normalised Levenshtein
   distance between the two surfaces — anything past that ratio is
   the LLM having an opinion, not catching an ASR error.

The function is intentionally pure-text: it takes a string and a
list of dicts, returns a new string and the audit lists (`applied`,
`rejected`). This keeps it testable without spinning up a model and
lets the caller decide where the output lands on disk.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable


__all__ = [
    "CorrectionOutcome",
    "AppliedCorrection",
    "RejectedCorrection",
    "apply_llm_corrections_to_text",
]


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class AppliedCorrection:
    """A correction that passed both guardrails and got substituted."""

    timestamp: str
    original: str
    replacement: str
    occurrences: int
    confidence: float
    reason: str


@dataclass
class RejectedCorrection:
    """A correction we refused to apply, with the reason why.

    The reason is a stable short string — UI surfaces show it as-is
    in the review report so the user can see which class of issue
    blocked the fix.
    """

    timestamp: str
    original: str
    replacement: str
    confidence: float
    reason: str  # one of: "low_confidence", "not_found", "too_distant", "empty"


@dataclass
class CorrectionOutcome:
    text: str
    applied: list[AppliedCorrection]
    rejected: list[RejectedCorrection]


# ---------------------------------------------------------------------------
# Distance helpers
# ---------------------------------------------------------------------------


def _normalize_for_match(value: str) -> str:
    """Lowercase + strip accents + collapse whitespace.

    Used for both substring lookup and Levenshtein distance, so the
    two guardrails see the same canonical form.
    """
    if not value:
        return ""
    stripped = unicodedata.normalize("NFKD", value)
    stripped = "".join(ch for ch in stripped if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", stripped).strip().lower()


def _levenshtein(a: str, b: str) -> int:
    """Plain Levenshtein. Inputs are short (handful of words) so
    the O(n*m) cost is negligible; we don't bother with the early-exit
    optimisation from glossary_postprocess.
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[len(b)]


def _normalised_edit_ratio(a: str, b: str) -> float:
    """Levenshtein normalised by the longer string. Returns 0.0 for
    identical strings, ~1.0 for completely different ones."""
    na = _normalize_for_match(a)
    nb = _normalize_for_match(b)
    if not na and not nb:
        return 0.0
    longest = max(len(na), len(nb))
    if longest == 0:
        return 0.0
    return _levenshtein(na, nb) / longest


# ---------------------------------------------------------------------------
# Substring lookup
# ---------------------------------------------------------------------------


def _find_all_normalised(haystack: str, needle: str) -> list[tuple[int, int]]:
    """Return (start, end) byte ranges in ``haystack`` whose normalised
    form matches ``needle`` normalised.

    Why we don't just lowercase + ``str.find``: the LLM frequently
    drops accents in its "original" quote ("Adele" instead of
    "Adèle"), or collapses double spaces from the transcript. A naive
    case-insensitive find would miss those — exactly the corrections
    that need to land most.

    Implementation: walk the haystack once, normalising one character
    at a time, and use a sliding window over a *parallel* index map
    so we can recover the byte range from the normalised match.
    """
    if not haystack or not needle:
        return []
    norm_needle = _normalize_for_match(needle)
    if not norm_needle:
        return []

    norm_chars: list[str] = []
    src_indices: list[int] = []  # for each char in norm_chars, the
                                 # byte offset in haystack where it
                                 # starts.
    last_was_space = True  # collapse runs of whitespace like the
                           # normaliser does, so the indices stay
                           # aligned with what we matched.
    for i, ch in enumerate(haystack):
        nc = unicodedata.normalize("NFKD", ch)
        nc = "".join(c for c in nc if not unicodedata.combining(c)).lower()
        for sub in nc:
            if sub.isspace():
                if last_was_space:
                    continue
                norm_chars.append(" ")
                src_indices.append(i)
                last_was_space = True
            else:
                norm_chars.append(sub)
                src_indices.append(i)
                last_was_space = False

    # Strip leading + trailing whitespace from the normalised stream
    # while keeping ``src_indices`` aligned with the surviving chars.
    lo = 0
    while lo < len(norm_chars) and norm_chars[lo] == " ":
        lo += 1
    hi = len(norm_chars)
    while hi > lo and norm_chars[hi - 1] == " ":
        hi -= 1
    norm_chars = norm_chars[lo:hi]
    src_indices = src_indices[lo:hi]
    norm_text = "".join(norm_chars)

    matches: list[tuple[int, int]] = []
    start = 0
    while True:
        pos = norm_text.find(norm_needle, start)
        if pos < 0:
            break
        end_in_norm = pos + len(norm_needle) - 1
        if end_in_norm >= len(src_indices):
            break
        src_start = src_indices[pos]
        src_end = src_indices[end_in_norm] + 1
        matches.append((src_start, src_end))
        start = pos + 1
    return matches


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def apply_llm_corrections_to_text(
    text: str,
    corrections: Iterable[dict],
    *,
    min_confidence: float = 0.7,
    max_edit_ratio: float = 0.45,
    max_replacements_per_correction: int = 4,
) -> CorrectionOutcome:
    """Rewrite ``text`` by applying every correction that clears both
    guardrails.

    Parameters
    ----------
    text:
        The full transcript content. Returned unchanged when no
        correction applies.
    corrections:
        Iterable of dicts with keys ``original``, ``replacement``,
        ``timestamp`` (optional), ``confidence`` (optional, defaults
        to 1.0), ``reason`` (optional). Anything else is ignored.
    min_confidence:
        Skip entries whose ``confidence`` is below this threshold.
        Default 0.7 — generous enough that an LLM stamping every
        correction at 0.85 (our current default) passes, but tight
        enough to reject a future model that emits 0.5.
    max_edit_ratio:
        Skip entries whose normalised Levenshtein distance between
        ``original`` and ``replacement`` exceeds this ratio. 0.45
        catches all the proper-noun rewrites we want ("Sudokiz" →
        "Sudokies") while blocking a rewritten sentence.
    max_replacements_per_correction:
        Hard cap on how many occurrences of ``original`` we'll
        replace per entry. Prevents a near-empty ``original`` like
        "ok" from rewriting the whole transcript.

    Returns
    -------
    CorrectionOutcome with the new text and audit lists. The order of
    ``applied`` reflects the order substitutions ran, which is also
    the order of the timestamps when they're present.
    """
    applied: list[AppliedCorrection] = []
    rejected: list[RejectedCorrection] = []
    items = list(corrections or [])
    if not text or not items:
        return CorrectionOutcome(text=text, applied=[], rejected=[])

    # Sort by timestamp when present, so we touch the text in
    # reading order. Stable sort keeps insertion order for entries
    # that share a timestamp (or have none).
    def _ts_key(c: dict) -> tuple[int, int, int]:
        ts = (c.get("timestamp") or "").strip()
        if not ts:
            return (10**9, 0, 0)
        m = re.match(r"(\d{1,2}):(\d{2})(?::(\d{2}))?", ts)
        if not m:
            return (10**9, 0, 0)
        return (
            int(m.group(1)),
            int(m.group(2)),
            int(m.group(3) or 0),
        )

    items_sorted = sorted(items, key=_ts_key)

    current = text
    for entry in items_sorted:
        original = str(entry.get("original") or "").strip()
        replacement = str(entry.get("replacement") or "").strip()
        ts = str(entry.get("timestamp") or "").strip()
        try:
            confidence = float(entry.get("confidence") or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        reason_in = str(entry.get("reason") or "").strip()

        if not original or not replacement:
            rejected.append(
                RejectedCorrection(
                    timestamp=ts,
                    original=original,
                    replacement=replacement,
                    confidence=confidence,
                    reason="empty",
                )
            )
            continue

        if confidence < min_confidence:
            rejected.append(
                RejectedCorrection(
                    timestamp=ts,
                    original=original,
                    replacement=replacement,
                    confidence=confidence,
                    reason="low_confidence",
                )
            )
            continue

        # Reject corrections whose normalised forms are identical.
        # Catches the LLM emitting cedilla- or accent-corrupted
        # words ("clique" → "çlique", "une" → "unee") that pass the
        # Levenshtein and substring checks because the surface
        # difference is tiny, but the normalised forms collide. The
        # "fix" would actually corrupt the transcript with mid-word
        # diacritics that weren't there in the original audio.
        if _normalize_for_match(original) == _normalize_for_match(replacement):
            rejected.append(
                RejectedCorrection(
                    timestamp=ts,
                    original=original,
                    replacement=replacement,
                    confidence=confidence,
                    reason="noop_after_normalization",
                )
            )
            continue

        # Guardrail 1: the LLM quoted an `original` that actually
        # exists in the transcript. If not, the model is paraphrasing
        # what it *heard* about the audio rather than catching a real
        # ASR error — that's the failure mode the user most needs to
        # see, so we surface it before the distance check.
        ranges = _find_all_normalised(current, original)
        if not ranges:
            rejected.append(
                RejectedCorrection(
                    timestamp=ts,
                    original=original,
                    replacement=replacement,
                    confidence=confidence,
                    reason="not_found",
                )
            )
            continue

        # Guardrail 2: stay phonetically close. A wide edit ratio
        # signals a rewrite ("On parle de X" -> "On parle ce matin de
        # notre amie X"), not a transcription fix. Refuse so we never
        # rewrite the user's style under the guise of correcting
        # Whisper.
        ratio = _normalised_edit_ratio(original, replacement)
        if ratio > max_edit_ratio:
            rejected.append(
                RejectedCorrection(
                    timestamp=ts,
                    original=original,
                    replacement=replacement,
                    confidence=confidence,
                    reason="too_distant",
                )
            )
            continue

        ranges = ranges[:max_replacements_per_correction]
        # Walk in reverse so earlier ranges aren't shifted by later
        # ones. Reversed-then-applied-reversed keeps every offset
        # valid until the substitution actually runs.
        new_text = current
        for start, end in reversed(ranges):
            new_text = new_text[:start] + replacement + new_text[end:]
        current = new_text

        applied.append(
            AppliedCorrection(
                timestamp=ts,
                original=original,
                replacement=replacement,
                occurrences=len(ranges),
                confidence=confidence,
                reason=reason_in,
            )
        )

    return CorrectionOutcome(text=current, applied=applied, rejected=rejected)
