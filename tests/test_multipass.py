import unittest

from multipass import (
    DEFAULT_WEAK_SCORE,
    group_into_clip_ranges,
    identify_boundary_segments,
    identify_weak_segments,
    merge_repass_segments,
    score_segment,
)


class ScoreSegmentTest(unittest.TestCase):
    def test_clean_segment_scores_high(self):
        # Typical Whisper output on clean French speech.
        seg = {
            "start": 0.0, "end": 3.0, "text": "Bonjour à tous.",
            "avg_logprob": -0.15, "no_speech_prob": 0.02, "compression_ratio": 1.5,
        }
        self.assertGreater(score_segment(seg), 0.85)

    def test_silence_loop_scores_low(self):
        # Compression ratio 3.5+ → "...." or "Sous-titrage..." loops.
        seg = {
            "start": 0.0, "end": 4.0, "text": "...",
            "avg_logprob": -1.5, "no_speech_prob": 0.7, "compression_ratio": 3.6,
        }
        self.assertLess(score_segment(seg), 0.4)

    def test_uncertain_segment_scores_in_middle(self):
        seg = {
            "start": 0.0, "end": 2.0, "text": "MOLI",
            "avg_logprob": -1.0, "no_speech_prob": 0.1, "compression_ratio": 1.8,
        }
        score = score_segment(seg)
        self.assertGreater(score, 0.4)
        self.assertLess(score, 0.75)

    def test_handles_missing_fields(self):
        # Older transcripts may not have all keys.
        score = score_segment({"start": 0, "end": 1, "text": "x"})
        # Sane fallback values mean default score should be high.
        self.assertGreater(score, 0.7)


class IdentifyWeakSegmentsTest(unittest.TestCase):
    def test_flags_only_low_score_segments(self):
        segs = [
            {"start": 0, "end": 2, "text": "ok", "avg_logprob": -0.1,
             "no_speech_prob": 0, "compression_ratio": 1.5},
            {"start": 2, "end": 4, "text": "Sudokiz", "avg_logprob": -1.1,
             "no_speech_prob": 0.05, "compression_ratio": 2.3},
            {"start": 4, "end": 6, "text": "...", "avg_logprob": -1.5,
             "no_speech_prob": 0.8, "compression_ratio": 3.5},
        ]
        weak = identify_weak_segments(segs)
        self.assertEqual(len(weak), 2)
        self.assertEqual(weak[0].index, 1)
        self.assertEqual(weak[1].index, 2)
        self.assertLess(weak[1].score, weak[0].score)

    def test_skips_short_segments(self):
        segs = [
            # 200 ms long — even with bad logprob, not worth re-running.
            {"start": 0, "end": 0.2, "text": "uh", "avg_logprob": -1.5,
             "no_speech_prob": 0.3, "compression_ratio": 1.5},
        ]
        self.assertEqual(identify_weak_segments(segs), [])

    def test_carries_reason_for_each_flag(self):
        segs = [
            {"start": 0, "end": 3, "text": "huh", "avg_logprob": -1.2,
             "no_speech_prob": 0.05, "compression_ratio": 1.8},
        ]
        weak = identify_weak_segments(segs)
        self.assertEqual(len(weak), 1)
        self.assertIn("logprob", weak[0].reason)


class GroupIntoClipRangesTest(unittest.TestCase):
    def _w(self, start, end, score=0.4):
        return identify_weak_segments(
            [{"start": start, "end": end, "text": "x",
              "avg_logprob": -1.2, "no_speech_prob": 0.1,
              "compression_ratio": 2.0}]
        )[0]

    def test_merges_close_weak_segments(self):
        # Two weak segments 1 second apart should merge into one
        # padded clip range.
        weak = [self._w(10.0, 12.0), self._w(13.0, 15.0)]
        ranges = group_into_clip_ranges(weak, pad_seconds=0.5)
        self.assertEqual(len(ranges), 1)
        s, e = ranges[0]
        self.assertAlmostEqual(s, 9.5)
        self.assertAlmostEqual(e, 15.5)

    def test_splits_when_gap_too_large(self):
        weak = [self._w(10.0, 12.0), self._w(30.0, 32.0)]
        ranges = group_into_clip_ranges(weak, max_gap_seconds=4.0)
        self.assertEqual(len(ranges), 2)

    def test_caps_segments_per_clip(self):
        # 15 contiguous weak segments → split into multiple clips
        # to avoid sending Whisper a 5-minute slab to re-transcribe.
        weak = [self._w(i, i + 0.5) for i in range(15)]
        ranges = group_into_clip_ranges(weak, max_segments_per_clip=5, pad_seconds=0.2)
        self.assertGreaterEqual(len(ranges), 3)


class MergeRepassSegmentsTest(unittest.TestCase):
    def test_replaces_segments_inside_clip_range(self):
        base = [
            {"start": 0, "end": 2, "text": "clean"},
            {"start": 10, "end": 12, "text": "broken"},
            {"start": 12, "end": 14, "text": "alsobroken"},
            {"start": 20, "end": 22, "text": "clean2"},
        ]
        repass = [
            {"start": 10, "end": 14, "text": "fixed merged"},
        ]
        out, replaced = merge_repass_segments(base, repass, [(9.0, 14.5)])
        self.assertEqual(replaced, 2)
        self.assertEqual(len(out), 3)
        texts = [s["text"] for s in out]
        self.assertEqual(texts, ["clean", "fixed merged", "clean2"])

    def test_no_op_when_no_clip_range(self):
        base = [{"start": 0, "end": 2, "text": "x"}]
        out, replaced = merge_repass_segments(base, [], [])
        self.assertEqual(out, base)
        self.assertEqual(replaced, 0)


class IdentifyBoundarySegmentsTest(unittest.TestCase):
    """PR I — boundary multipass targets short segments at speaker
    changes, where Whisper's 30s context window often conditioned
    on the wrong voice."""

    def test_flags_short_segment_at_boundary(self):
        segments = [
            {"start": 0.0, "end": 5.0, "speaker": "Robin", "text": "Bonjour."},
            {"start": 5.0, "end": 6.2, "speaker": "Manon", "text": "Salut."},
            {"start": 6.2, "end": 10.0, "speaker": "Manon", "text": "Comment ça va."},
        ]
        out = identify_boundary_segments(segments)
        # Segment 0 (Robin → Manon change) AND segment 1 (Robin →
        # Manon, short) qualify. Segment 2 doesn't (no change after).
        labels = {(int(w.start), int(w.end)) for w in out}
        self.assertIn((5, 6), labels)  # segment 1, the short Manon turn

    def test_skips_long_segments(self):
        # Long monologues at boundaries don't qualify — repassing
        # 30s of audio for a wrong-speaker hint is expensive.
        segments = [
            {"start": 0.0, "end": 5.0, "speaker": "Robin", "text": "Bonjour."},
            {"start": 5.0, "end": 35.0, "speaker": "Manon", "text": "Long monologue."},
        ]
        out = identify_boundary_segments(segments)
        # Manon's 30s segment is too long for boundary repass.
        long_segments = [w for w in out if w.end - w.start > 2.5]
        self.assertEqual(long_segments, [])

    def test_skips_when_no_speaker_change(self):
        segments = [
            {"start": 0.0, "end": 1.5, "speaker": "Robin", "text": "Un."},
            {"start": 1.5, "end": 3.0, "speaker": "Robin", "text": "Deux."},
        ]
        self.assertEqual(identify_boundary_segments(segments), [])

    def test_skips_very_short_fragments(self):
        # Anything below ``min_duration`` is too short to repass
        # — model load alone would dwarf the savings.
        segments = [
            {"start": 0.0, "end": 5.0, "speaker": "Robin", "text": "Bonjour."},
            {"start": 5.0, "end": 5.1, "speaker": "Manon", "text": "Oui."},
            {"start": 5.1, "end": 8.0, "speaker": "Manon", "text": "Carrément."},
        ]
        out = identify_boundary_segments(segments)
        # The 100ms "Oui" fragment is excluded.
        for w in out:
            self.assertGreaterEqual(w.end - w.start, 0.4)


if __name__ == "__main__":
    unittest.main()
