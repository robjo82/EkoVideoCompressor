"""Tests for the diarisation post-processing.

Pyannote happily emits 100-300 ms speaker turns whenever someone
back-channels in the middle of another's sentence. Left raw, these
micro-turns produce ``[Robin] Bah ouais. [David] OK.`` mid-thought
cuts that read like the transcript has a stutter.

``fuse_micro_turns`` collapses anything shorter than the configured
minimum into the surrounding turn, preferring a same-speaker
neighbour when one is close enough. These tests pin the strategy so
the heuristic doesn't drift.
"""

from __future__ import annotations

import unittest

from transcription_utils import fuse_micro_turns


class FuseMicroTurnsTest(unittest.TestCase):
    def test_short_turn_merges_into_same_speaker_neighbour(self):
        # Robin speaks, drops out for 200 ms, then keeps going.
        # Pyannote would emit two Robin turns split by nothing — we
        # want them merged so the rendered transcript reads as one.
        turns = [
            {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"},
            {"start": 5.1, "end": 5.3, "speaker": "SPEAKER_00"},  # 200 ms
            {"start": 5.4, "end": 10.0, "speaker": "SPEAKER_00"},
        ]
        out = fuse_micro_turns(turns, min_duration=0.4)
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(out[0]["start"], 0.0)
        self.assertGreaterEqual(out[0]["end"], 5.3)

    def test_back_channel_collapses_into_surrounding_speech(self):
        # A 150 ms "ouais" by Robin in the middle of David's
        # sentence should not break David's turn into two halves.
        # We absorb the micro-turn into whichever neighbour is
        # closer in time.
        turns = [
            {"start": 0.0, "end": 4.0, "speaker": "SPEAKER_00"},  # David
            {"start": 4.0, "end": 4.15, "speaker": "SPEAKER_01"},  # Robin "ouais"
            {"start": 4.15, "end": 9.0, "speaker": "SPEAKER_00"},  # David continues
        ]
        out = fuse_micro_turns(turns, min_duration=0.4)
        # No SPEAKER_01 turn survives — the micro-turn gets absorbed
        # by one of David's neighbours.
        speakers = {t["speaker"] for t in out}
        self.assertEqual(speakers, {"SPEAKER_00"})
        # The total time covered by SPEAKER_00 should still extend
        # over the whole [0, 9] window — no time hole left behind.
        total = sum(t["end"] - t["start"] for t in out)
        self.assertAlmostEqual(total, 9.0, delta=0.001)

    def test_long_turn_left_alone(self):
        turns = [
            {"start": 0.0, "end": 30.0, "speaker": "SPEAKER_00"},
            {"start": 30.0, "end": 60.0, "speaker": "SPEAKER_01"},
        ]
        out = fuse_micro_turns(turns, min_duration=0.4)
        self.assertEqual(out, turns)

    def test_unsorted_input_is_sorted_before_processing(self):
        # We don't trust pyannote to emit turns in order. The
        # micro-turn detection only makes sense once neighbours are
        # actually adjacent in time.
        turns = [
            {"start": 5.0, "end": 10.0, "speaker": "SPEAKER_00"},
            {"start": 4.85, "end": 5.0, "speaker": "SPEAKER_01"},
            {"start": 0.0, "end": 4.85, "speaker": "SPEAKER_00"},
        ]
        out = fuse_micro_turns(turns, min_duration=0.4)
        starts = [t["start"] for t in out]
        self.assertEqual(starts, sorted(starts))

    def test_single_short_turn_is_kept(self):
        # If the recording is a 200 ms phone tap, we'd rather keep
        # the one turn we have than drop the only speaker label.
        out = fuse_micro_turns(
            [{"start": 0.0, "end": 0.2, "speaker": "SPEAKER_00"}],
            min_duration=0.4,
        )
        self.assertEqual(len(out), 1)


if __name__ == "__main__":
    unittest.main()
