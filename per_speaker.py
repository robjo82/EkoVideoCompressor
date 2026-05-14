"""
Per-speaker Whisper transcription helpers.

The default pipeline runs Whisper once on the whole audio and then
fuses speaker labels from pyannote turns. That's fast but fragile:
Whisper's decoder context window crosses speaker boundaries, so a
quiet voice answering after a loud one gets dragged along with the
previous speaker's lexicon, leading to cross-speaker contamination
(e.g. "uh ouais ouais ouais" from the DSI bleeding into the
integrator's measured "écoutez, dans ce cas-là…").

Alternative pipeline implemented here:
  1. Diarize first (already done by pyannote).
  2. For each speaker, pad+concatenate every turn they took into a
     single short WAV stream.
  3. Run Whisper independently on each speaker's WAV. The decoder
     never sees the other speakers' speech, so the prior on
     vocabulary stays consistent.
  4. Re-project each Whisper segment back to the source timeline
     via a per-speaker manifest (much like the VAD remapping).
  5. Sort all segments by start time and label them with the
     speaker — diarisation labels are now baked in, no fusion needed.

This module is pure Python — the actual ffmpeg + Whisper invocations
stay in the worker. We provide the manifest types and the splicing
logic; the worker drives the subprocesses.
"""

from __future__ import annotations

from dataclasses import dataclass


__all__ = [
    "SpeakerTurn",
    "SpeakerSlice",
    "build_speaker_slices",
    "remap_speaker_segments",
    "merge_speaker_segments",
]


@dataclass
class SpeakerTurn:
    """One pyannote turn for a given speaker."""

    speaker: str
    start: float
    end: float


@dataclass
class _SpeakerSpan:
    """Per-speaker mapping from trimmed-stream time to source time."""

    src_start: float
    src_end: float
    trim_start: float
    trim_end: float


@dataclass
class SpeakerSlice:
    """
    All the turns of one speaker, padded and concatenated into a
    single virtual stream. ``spans`` lets us shift Whisper's
    timestamps back to the source timeline.
    """

    speaker: str
    spans: list[_SpeakerSpan]

    @property
    def duration(self) -> float:
        return self.spans[-1].trim_end if self.spans else 0.0

    def to_ffmpeg_segments(self) -> list[tuple[float, float]]:
        """List of (src_start, src_end) pairs for ffmpeg `aselect`."""
        return [(s.src_start, s.src_end) for s in self.spans]


def build_speaker_slices(
    turns: list[dict] | list[SpeakerTurn],
    *,
    pad_seconds: float = 0.4,
    min_turn_duration: float = 0.4,
    min_speaker_total: float = 5.0,
) -> dict[str, SpeakerSlice]:
    """
    Group pyannote turns by speaker, drop the noise:
      - turns shorter than ``min_turn_duration`` are filtered out
        (single-word interjections rarely add useful context);
      - speakers whose total speech time is below
        ``min_speaker_total`` are dropped entirely (typically a
        passive bystander coughing into the mic that pyannote
        labelled as its own SPEAKER_XX).

    Each retained speaker becomes one ``SpeakerSlice`` with a span
    manifest mapping the per-speaker stream time back to the source
    timeline.
    """
    # Coerce to dataclasses if we got dicts.
    parsed: list[SpeakerTurn] = []
    for t in turns:
        if isinstance(t, SpeakerTurn):
            parsed.append(t)
        else:
            try:
                parsed.append(
                    SpeakerTurn(
                        speaker=str(t["speaker"]),
                        start=float(t["start"]),
                        end=float(t["end"]),
                    )
                )
            except Exception:
                continue
    if not parsed:
        return {}

    # Bucket by speaker and sort.
    buckets: dict[str, list[SpeakerTurn]] = {}
    for t in parsed:
        if t.end - t.start < min_turn_duration:
            continue
        buckets.setdefault(t.speaker, []).append(t)
    for sp in buckets:
        buckets[sp].sort(key=lambda x: x.start)

    out: dict[str, SpeakerSlice] = {}
    for speaker, segs in buckets.items():
        spans: list[_SpeakerSpan] = []
        cursor = 0.0
        for seg in segs:
            src_start = max(0.0, seg.start - pad_seconds)
            src_end = seg.end + pad_seconds
            duration = src_end - src_start
            spans.append(
                _SpeakerSpan(
                    src_start=src_start,
                    src_end=src_end,
                    trim_start=cursor,
                    trim_end=cursor + duration,
                )
            )
            cursor += duration
        total = cursor
        if total < min_speaker_total:
            continue
        out[speaker] = SpeakerSlice(speaker=speaker, spans=spans)
    return out


def remap_speaker_segments(
    segments: list[dict],
    speaker_slice: SpeakerSlice,
) -> list[dict]:
    """
    Map Whisper segments produced on a speaker's concatenated stream
    back to source-media timestamps. Each new segment also carries
    the ``speaker`` label so we don't have to re-fuse.
    """
    if not segments or not speaker_slice.spans:
        return []

    def _shift(t: float) -> float:
        for span in speaker_slice.spans:
            if span.trim_start <= t <= span.trim_end + 1e-3:
                return span.src_start + (t - span.trim_start)
        # Outside the manifest — clamp to the last source end.
        return speaker_slice.spans[-1].src_end

    out: list[dict] = []
    for seg in segments:
        try:
            s = float(seg.get("start") or 0.0)
            e = float(seg.get("end") or 0.0)
        except (TypeError, ValueError):
            continue
        new_seg = dict(seg)
        new_seg["start"] = _shift(s)
        new_seg["end"] = _shift(e)
        new_seg["speaker"] = speaker_slice.speaker
        out.append(new_seg)
    return out


def merge_speaker_segments(per_speaker: list[list[dict]]) -> list[dict]:
    """
    Flatten and sort segments from every speaker by start time.
    """
    flat: list[dict] = []
    for segs in per_speaker:
        flat.extend(segs)
    flat.sort(key=lambda x: float(x.get("start") or 0.0))
    return flat
