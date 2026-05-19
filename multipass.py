"""
Confidence-based two-pass transcription helpers.

Whisper-large-v3-turbo is fast but sacrifices ~5-8% accuracy vs
whisper-large-v3 on French content. The trade-off is asymmetric:
the model is fine on clean, well-articulated passages, but it
silently degrades on segments that contain accented proper nouns,
overlapping speech, or quiet background voices.

Strategy:

  1. Run the fast model (turbo) once across the whole audio.
  2. For each segment, compute a quality score from Whisper's
     internal metadata (avg_logprob, no_speech_prob,
     compression_ratio) plus a "looks like the glossary"
     heuristic (phonetic-near-but-not-quite a glossary term).
  3. Group segments below the score threshold into contiguous
     clip ranges.
  4. Re-transcribe ONLY those ranges with whisper-large-v3
     (non-turbo) using `--clip-timestamps`. This is ~3× slower
     per second of audio but typically only 10-20% of the
     recording needs the second pass.
  5. Splice the new segments back into the timeline, replacing
     the originals.

The module is pure Python — the actual Whisper invocation stays
in the worker.
"""

from __future__ import annotations

from dataclasses import dataclass


__all__ = [
    "WeakSegment",
    "score_segment",
    "identify_weak_segments",
    "group_into_clip_ranges",
    "merge_repass_segments",
]


# Score thresholds. Below this we consider the segment "weak" and
# worth a second pass.
DEFAULT_WEAK_LOGPROB = -0.85
DEFAULT_WEAK_NO_SPEECH = 0.65
DEFAULT_WEAK_COMPRESSION = 2.0
# Score below this is considered weak — bottom 15-20% of typical
# Whisper output.
DEFAULT_WEAK_SCORE = 0.55


@dataclass
class WeakSegment:
    """A segment flagged as needing a second pass."""

    index: int          # position in the original list
    start: float        # source-timeline start (seconds)
    end: float          # source-timeline end (seconds)
    score: float        # 0-1, lower = worse
    reason: str         # human-readable hint for the review report


def score_segment(segment: dict) -> float:
    """
    Map Whisper's per-segment metadata to a normalised quality
    score in [0, 1]. 1.0 is "Whisper was very confident", 0.0 is
    "this segment is borderline noise".

    The score combines:
      - avg_logprob (higher = better; values typically -1.5..0)
      - no_speech_prob (higher = worse; 0..1)
      - compression_ratio (much higher = repetition/loop)
    """
    try:
        logprob = float(segment.get("avg_logprob") or -0.3)
    except (TypeError, ValueError):
        logprob = -0.3
    try:
        no_speech = float(segment.get("no_speech_prob") or 0.0)
    except (TypeError, ValueError):
        no_speech = 0.0
    try:
        compression = float(segment.get("compression_ratio") or 1.5)
    except (TypeError, ValueError):
        compression = 1.5

    # logprob is by far the strongest signal — Whisper's
    # confidence floor on clean French speech is around -0.2,
    # weak segments hit -0.8 to -1.2. We map a tighter window
    # so the second pass is triggered on actual uncertainty.
    # -1.5 → 0.0, -0.2 → 1.0 (clamped beyond).
    lp_score = max(0.0, min(1.0, (logprob + 1.5) / 1.3))
    # no_speech penalises: 0.0 → 1.0, 0.9 → 0.1
    ns_score = 1.0 - min(1.0, no_speech)
    # compression: 1.0–2.0 healthy → 1.0; 2.4 → 0.5; 3.5+ → 0.0
    if compression <= 2.0:
        cr_score = 1.0
    elif compression >= 3.5:
        cr_score = 0.0
    else:
        cr_score = max(0.0, (3.5 - compression) / 1.5)

    # logprob: 75% — it's the strongest predictor of whether a
    # second pass will rewrite the text.
    # no_speech: 15% — important for IVR/hold-music leftovers
    # that VAD missed.
    # compression: 10% — guards against decoder loops.
    return 0.75 * lp_score + 0.15 * ns_score + 0.10 * cr_score


def identify_boundary_segments(
    segments: list[dict],
    *,
    max_duration: float = 2.0,
    min_duration: float = 0.4,
) -> list[WeakSegment]:
    """Flag segments adjacent to a speaker change for re-transcription.

    The diarisation projection from PR A handles MIS-LABELLED words
    at boundaries (smoothing, orphan absorption, same-speaker merge).
    But Whisper sometimes produces the WRONG WORDS at those bounds —
    a single Whisper segment that spans an interruption gets the
    whole sentence wrong because the model conditioned on the wrong
    voice. Re-running Whisper on a tight clip around the boundary
    with the higher-quality multipass model usually recovers it.

    Heuristic :
    - Speaker changes from one segment to the next.
    - The shorter of the two adjacent segments is ≤ ``max_duration``.
    - Each candidate must be ≥ ``min_duration`` (no point repassing
      a 200 ms fragment — the model wouldn't even produce text).
    - We deduplicate by index so a single segment in the middle of
      a "ping-pong" exchange isn't repassed twice.
    """
    out: list[WeakSegment] = []
    seen_indices: set[int] = set()
    for i in range(len(segments)):
        seg = segments[i]
        try:
            s = float(seg.get("start") or 0.0)
            e = float(seg.get("end") or 0.0)
        except (TypeError, ValueError):
            continue
        duration = e - s
        if duration < min_duration or duration > max_duration:
            continue
        speaker = (seg.get("speaker") or "").strip() or None
        prev_speaker = (
            (segments[i - 1].get("speaker") or "").strip()
            if i > 0
            else None
        ) or None
        next_speaker = (
            (segments[i + 1].get("speaker") or "").strip()
            if i + 1 < len(segments)
            else None
        ) or None
        adjacent_change = (
            (prev_speaker is not None and speaker is not None and prev_speaker != speaker)
            or (next_speaker is not None and speaker is not None and next_speaker != speaker)
        )
        if not adjacent_change:
            continue
        if i in seen_indices:
            continue
        seen_indices.add(i)
        out.append(
            WeakSegment(
                index=i,
                start=s,
                end=e,
                score=0.5,
                reason="frontière de diarisation",
            )
        )
    return out


def identify_weak_segments(
    segments: list[dict],
    *,
    score_threshold: float = DEFAULT_WEAK_SCORE,
    min_duration: float = 0.4,
) -> list[WeakSegment]:
    """
    Walk Whisper segments and flag the ones likely to benefit from
    a higher-accuracy second pass. We skip segments shorter than
    ``min_duration`` because re-transcribing 200 ms of audio is
    rarely worth the model load time.
    """
    out: list[WeakSegment] = []
    for i, seg in enumerate(segments):
        try:
            s = float(seg.get("start") or 0.0)
            e = float(seg.get("end") or 0.0)
        except (TypeError, ValueError):
            continue
        if e - s < min_duration:
            continue
        score = score_segment(seg)
        if score < score_threshold:
            reasons = []
            try:
                if float(seg.get("avg_logprob") or 0) < DEFAULT_WEAK_LOGPROB:
                    reasons.append("logprob bas")
            except (TypeError, ValueError):
                pass
            try:
                if float(seg.get("no_speech_prob") or 0) > DEFAULT_WEAK_NO_SPEECH:
                    reasons.append("no_speech élevé")
            except (TypeError, ValueError):
                pass
            try:
                if float(seg.get("compression_ratio") or 0) > 2.2:
                    reasons.append("compression élevée")
            except (TypeError, ValueError):
                pass
            out.append(
                WeakSegment(
                    index=i,
                    start=s,
                    end=e,
                    score=score,
                    reason=", ".join(reasons) or "qualité globale faible",
                )
            )
    return out


def group_into_clip_ranges(
    weak: list[WeakSegment],
    *,
    pad_seconds: float = 1.0,
    max_gap_seconds: float = 4.0,
    max_segments_per_clip: int = 12,
) -> list[tuple[float, float]]:
    """
    Merge contiguous (or near-contiguous) weak segments into clip
    ranges suitable for Whisper's ``--clip-timestamps``. Each clip
    is padded by ``pad_seconds`` on both sides so the re-transcription
    has enough context to resolve cross-segment confusion.

    Two adjacent weak segments are merged when the gap between them
    is <= ``max_gap_seconds`` (which means an in-between confident
    segment would also be re-transcribed; that's fine because we
    splice on best-overlap and the cost is small).
    """
    if not weak:
        return []
    ranges: list[list[float]] = []
    current_start = max(0.0, weak[0].start - pad_seconds)
    current_end = weak[0].end + pad_seconds
    current_count = 1
    for seg in weak[1:]:
        gap = seg.start - (current_end - pad_seconds)
        if gap <= max_gap_seconds and current_count < max_segments_per_clip:
            current_end = seg.end + pad_seconds
            current_count += 1
        else:
            ranges.append([current_start, current_end])
            current_start = max(0.0, seg.start - pad_seconds)
            current_end = seg.end + pad_seconds
            current_count = 1
    ranges.append([current_start, current_end])
    return [(s, e) for s, e in ranges]


def merge_repass_segments(
    base_segments: list[dict],
    new_segments: list[dict],
    clip_ranges: list[tuple[float, float]],
) -> tuple[list[dict], int]:
    """
    Replace base segments overlapping any clip range with the
    matching new segments. Returns (merged_segments, replaced_count).

    Segments that didn't overlap a clip stay untouched. Within each
    clip range we drop all base segments that overlap it and
    insert the new ones in their place.
    """
    if not clip_ranges or not new_segments:
        return list(base_segments), 0

    # Tag base segments with whether they fall inside a clip range.
    def _in_any_range(start: float, end: float) -> bool:
        for cs, ce in clip_ranges:
            if start < ce and end > cs:
                return True
        return False

    kept: list[dict] = []
    replaced = 0
    for seg in base_segments:
        try:
            s = float(seg.get("start") or 0)
            e = float(seg.get("end") or 0)
        except (TypeError, ValueError):
            kept.append(seg)
            continue
        if _in_any_range(s, e):
            replaced += 1
            continue
        kept.append(seg)

    # Insert new segments at the right spots (sorted by start time).
    combined = kept + list(new_segments)
    combined.sort(key=lambda x: float(x.get("start") or 0))
    return combined, replaced
