import unittest

from multipass import (
    DEFAULT_WEAK_SCORE,
    group_into_clip_ranges,
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


if __name__ == "__main__":
    unittest.main()
