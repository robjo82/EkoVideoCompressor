"""
PR Y — decoder loop detection + WarningEvent surfacing.

The Caste/CVR audit caught a critical data-loss bug:
``condition_on_previous_text=True`` made Whisper hallucinate a
phrase and propagate it for 70+ minutes. ``clean_whisper_segments``
silently dropped the looped segments, so the user lost an hour of
meeting with no signal.

PR Y :
  1. Reverted ``condition_on_previous_text`` to ``False`` in the
     ``max`` preset.
  2. Wired a ``dropped_loops`` accumulator through
     ``clean_whisper_segments`` so the pipeline can surface a
     WarningEvent describing the lost ranges.

Tests :
  • ``DroppedLoop`` records are emitted on a run ≥
    ``EXTENDED_LOOP_MIN_DROPS`` (default 5) consecutive identical
    segments.
  • Short legitimate repetitions (≤ 2 consecutive) don't emit
    anything.
  • Multiple loops produce multiple records.
  • Timestamps span the full looped range.
  • ``parse_whisper_json_segments`` round-trips the loop info from
    the JSON file.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from transcription_utils import (
    EXTENDED_LOOP_MIN_DROPS,
    DroppedLoop,
    clean_whisper_segments,
    parse_whisper_json_segments,
)


def _seg(start: float, end: float, text: str) -> dict:
    return {"start": start, "end": end, "text": text}


class DroppedLoopDetectionTests(unittest.TestCase):
    def test_no_loop_no_record(self):
        # Three distinct segments — nothing dropped.
        segments = [
            _seg(0, 1, "Bonjour."),
            _seg(1, 2, "Comment allez-vous ?"),
            _seg(2, 3, "Bien merci."),
        ]
        loops: list[DroppedLoop] = []
        out = clean_whisper_segments(segments, dropped_loops=loops)
        self.assertEqual(len(out), 3)
        self.assertEqual(loops, [])

    def test_short_repeat_under_threshold_not_recorded(self):
        # "Oui." repeated 4 times: 1 kept, 1 kept (allowed repeat),
        # then 2 dropped — 2 drops is below EXTENDED_LOOP_MIN_DROPS=5
        # so no DroppedLoop record.
        segments = [
            _seg(0, 1, "Oui."),
            _seg(1, 2, "Oui."),
            _seg(2, 3, "Oui."),  # dropped (repeat_count=3 > 2)
            _seg(3, 4, "Oui."),  # dropped
        ]
        loops: list[DroppedLoop] = []
        clean_whisper_segments(segments, dropped_loops=loops)
        self.assertEqual(loops, [])

    def test_caste_zindoc_loop_pattern_emits_record(self):
        # The exact audit pattern: a single phrase repeated
        # 60 consecutive times. ``clean_whisper_segments`` drops
        # 58 of them (2 kept by the "real repetition" tolerance)
        # — well above the 5-drop threshold.
        phrase = "On est sur Zindoc pour la gestion des fichiers."
        segments = [_seg(t, t + 5, phrase) for t in range(2371, 2371 + 60 * 5, 5)]
        loops: list[DroppedLoop] = []
        out = clean_whisper_segments(segments, dropped_loops=loops)
        # Only 2 kept (repeat_count 1 and 2).
        self.assertEqual(len(out), 2)
        # One DroppedLoop record covering the rest.
        self.assertEqual(len(loops), 1)
        loop = loops[0]
        self.assertEqual(loop.dropped, 58)
        self.assertIn("zindoc", loop.text)
        # Range spans from the first dropped segment to the last.
        self.assertAlmostEqual(loop.start, 2381.0, places=1)  # 3rd segment
        self.assertGreater(loop.end, loop.start + 250)  # ~57 × 5s

    def test_multiple_loops_emit_multiple_records(self):
        # Two distinct loops separated by a real segment.
        loop_a = [_seg(0 + i, 1 + i, "Phrase A.") for i in range(10)]
        bridge = [_seg(10, 11, "Et voilà.")]
        loop_b = [_seg(20 + i, 21 + i, "Phrase B.") for i in range(10)]
        loops: list[DroppedLoop] = []
        clean_whisper_segments(loop_a + bridge + loop_b, dropped_loops=loops)
        self.assertEqual(len(loops), 2)
        self.assertIn("phrase a", loops[0].text)
        self.assertIn("phrase b", loops[1].text)

    def test_loops_param_is_optional_for_legacy_callers(self):
        # Calling without ``dropped_loops`` works the same as before
        # (no signal change for code that doesn't care).
        phrase = "Boucle."
        segments = [_seg(t, t + 1, phrase) for t in range(10)]
        out = clean_whisper_segments(segments)
        # Same drop behaviour — 2 kept, 8 dropped silently.
        self.assertEqual(len(out), 2)

    def test_trailing_loop_at_end_of_input_flushed(self):
        # The loop is the LAST thing in the input — verify the
        # in-flight loop state is flushed before returning.
        segments = [_seg(0, 1, "Intro.")] + [
            _seg(t, t + 1, "Boucle finale.") for t in range(1, 11)
        ]
        loops: list[DroppedLoop] = []
        clean_whisper_segments(segments, dropped_loops=loops)
        self.assertEqual(len(loops), 1)
        self.assertGreaterEqual(loops[0].dropped, EXTENDED_LOOP_MIN_DROPS)


class ParseWhisperJsonRoundTripTests(unittest.TestCase):
    def test_loops_threaded_through_parse(self):
        # End-to-end: write a whisper.json containing a loop, read
        # it back via parse_whisper_json_segments, verify the
        # ``dropped_loops`` list is populated.
        phrase = "Boucle JSON."
        segments = [
            {"start": float(t), "end": float(t + 1), "text": phrase}
            for t in range(20)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "whisper.json"
            path.write_text(json.dumps({"segments": segments}))
            loops: list[DroppedLoop] = []
            kept = parse_whisper_json_segments(
                str(path), dropped_loops=loops
            )
        self.assertEqual(len(kept), 2)  # repeat_count 1 and 2
        self.assertEqual(len(loops), 1)
        self.assertEqual(loops[0].dropped, 18)


if __name__ == "__main__":
    unittest.main()
