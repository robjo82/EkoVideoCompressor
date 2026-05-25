"""
PR V — intra-segment decoder loop collapse.

Covers:
  • ``collapse_intra_segment_loops`` finds repeated n-grams (1–6
    words) that fire ``_INTRA_SEGMENT_MIN_REPETITIONS`` times or
    more inside a single segment and collapses them to a single
    occurrence + "…".
  • Catches the canonical Caste-Power-BI failure mode:
    "de la Fondation de la Fondation … (×100)".
  • Preserves legitimate French repetitions ("oui oui oui",
    "très très bien") that stay below the threshold.
  • Punctuation attached to the last token of the n-gram is part
    of the n-gram (so ``Fondation. Fondation. Fondation.``
    collapses correctly).
  • ``clean_whisper_segments`` runs the collapse before cross-segment
    dedup so a looped segment no longer escapes the pipeline.
"""

from __future__ import annotations

import unittest

from transcription_utils import (
    clean_whisper_segments,
    collapse_intra_segment_loops,
)


class CollapseIntraSegmentLoopsTests(unittest.TestCase):
    def test_caste_powerbi_fondation_loop_collapses(self):
        # The canonical bug: "de la Fondation" repeats ~100 times.
        # Should collapse to one occurrence + "…".
        text = (
            "Je vous présente le groupe "
            + "de la Fondation " * 60
            + "fin."
        )
        result, collapsed = collapse_intra_segment_loops(text)
        # One collapse event for the whole run.
        self.assertGreaterEqual(collapsed, 1)
        # Result contains the prefix, ONE "de la Fondation", "…",
        # then "fin." — not 60 copies.
        self.assertIn("Je vous présente le groupe", result)
        self.assertIn("de la Fondation", result)
        self.assertIn("…", result)
        self.assertIn("fin.", result)
        # The result is much shorter than the input.
        self.assertLess(len(result), len(text) // 10)
        # Specifically: only ONE occurrence of "de la Fondation"
        # remains (the rest collapsed).
        self.assertEqual(result.count("de la Fondation"), 1)

    def test_single_word_loop_collapses(self):
        text = "et et et et et et et et bonjour"
        result, collapsed = collapse_intra_segment_loops(text)
        self.assertGreaterEqual(collapsed, 1)
        self.assertIn("…", result)
        # Only one "et " keeps, with the ellipsis marker.
        self.assertIn("et …", result)
        self.assertIn("bonjour", result)

    def test_two_word_loop_collapses(self):
        text = "on commence par par exemple par exemple par exemple par exemple voilà"
        result, collapsed = collapse_intra_segment_loops(text)
        self.assertGreaterEqual(collapsed, 1)
        self.assertIn("…", result)
        self.assertEqual(result.count("par exemple"), 1)

    def test_three_repetitions_kept_intact(self):
        # 3 reps is below the threshold (4) — legitimate French
        # emphasis pattern, must NOT collapse.
        text = "oui oui oui d'accord"
        result, collapsed = collapse_intra_segment_loops(text)
        self.assertEqual(collapsed, 0)
        self.assertEqual(result, text)

    def test_four_repetitions_collapse(self):
        # Exactly 4 reps is the threshold — should collapse.
        text = "oui oui oui oui d'accord"
        result, collapsed = collapse_intra_segment_loops(text)
        self.assertGreaterEqual(collapsed, 1)
        self.assertIn("…", result)
        # Only one "oui" left.
        self.assertEqual(result.count("oui"), 1)

    def test_non_repeating_text_passes_through(self):
        text = "Voilà, je vais vous présenter la suite du projet aujourd'hui."
        result, collapsed = collapse_intra_segment_loops(text)
        self.assertEqual(collapsed, 0)
        self.assertEqual(result, text)

    def test_empty_text_passes_through(self):
        self.assertEqual(collapse_intra_segment_loops(""), ("", 0))
        self.assertEqual(collapse_intra_segment_loops("   "), ("   ", 0))

    def test_short_text_below_min_words_no_collapse(self):
        # Less than 4 words total — can't possibly have 4 reps of
        # anything.
        text = "Bonjour."
        result, collapsed = collapse_intra_segment_loops(text)
        self.assertEqual(collapsed, 0)
        self.assertEqual(result, text)

    def test_punctuation_attached_to_last_token_collapses(self):
        # "Application. Application. Application. Application." — the
        # period is part of the token, so the 1-gram is "Application."
        # which still matches and collapses.
        text = "Application. Application. Application. Application. fin"
        result, collapsed = collapse_intra_segment_loops(text)
        self.assertGreaterEqual(collapsed, 1)
        self.assertIn("…", result)

    def test_punctuation_only_window_not_collapsed(self):
        # "... ... ... ..." — all-punctuation n-grams are rejected
        # to avoid eating ellipsis decorations the LLM may add.
        text = "... ... ... ... ouais"
        result, collapsed = collapse_intra_segment_loops(text)
        # The all-punctuation 1-gram should NOT be collapsed.
        self.assertEqual(collapsed, 0)

    def test_longest_ngram_wins(self):
        # ``de la Fondation`` is a 3-gram. The naive single-token
        # sweep could match ``de`` × N first. Verify we collapse
        # the 3-gram, leaving exactly one full phrase.
        text = "et de la Fondation de la Fondation de la Fondation de la Fondation et fin"
        result, _ = collapse_intra_segment_loops(text)
        self.assertEqual(result.count("de la Fondation"), 1)
        self.assertIn("…", result)


class CleanWhisperSegmentsIntegrationTests(unittest.TestCase):
    """The cleaner now collapses loops before the cross-segment dedup."""

    def test_looped_segment_collapses_in_pipeline(self):
        segments = [
            {
                "id": 1,
                "start": 0.0,
                "end": 30.0,
                "text": "Je vous présente le groupe " + "de la Fondation " * 50,
            },
            {
                "id": 2,
                "start": 30.0,
                "end": 32.0,
                "text": "Suite du discours.",
            },
        ]
        cleaned = clean_whisper_segments(segments)
        self.assertEqual(len(cleaned), 2)
        # First segment was collapsed.
        self.assertLess(len(cleaned[0]["text"]), len(segments[0]["text"]) // 5)
        self.assertIn("…", cleaned[0]["text"])
        # Second segment untouched.
        self.assertEqual(cleaned[1]["text"], "Suite du discours.")

    def test_normal_segments_pass_through(self):
        segments = [
            {"id": 1, "start": 0.0, "end": 5.0, "text": "Bonjour."},
            {"id": 2, "start": 5.0, "end": 10.0, "text": "Comment allez-vous ?"},
        ]
        cleaned = clean_whisper_segments(segments)
        self.assertEqual(len(cleaned), 2)
        self.assertEqual(cleaned[0]["text"], "Bonjour.")
        self.assertEqual(cleaned[1]["text"], "Comment allez-vous ?")


if __name__ == "__main__":
    unittest.main()
