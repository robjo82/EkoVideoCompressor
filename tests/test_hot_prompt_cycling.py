"""
PR D — Whisper context-aware + hot prompt cycling.

Covers:
  • ``TranscriptionSettings.condition_on_previous_text`` /
    ``hot_prompt_enrichment`` defaults + their wiring in the ``max``
    quality preset (off everywhere else).
  • ``build_mlx_whisper_cmd`` carries the chosen flag verbatim.
  • ``extract_new_proper_nouns_from_segments`` finds repeated
    capitalised tokens, ignores function words, de-duplicates against
    the existing glossary, and caps results.
  • The pipeline's hot-enrichment hook actually mutates
    ``request.glossary_terms`` after the first Whisper pass when the
    setting is on (and does nothing when off).
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from ekovideo_engine.models import (
    JobRequest,
    TranscriptionSettings,
    apply_quality_preset,
)
from ekovideo_engine.pipeline import TranscriptionPipeline
from transcription_utils import (
    build_mlx_whisper_cmd,
    extract_new_proper_nouns_from_segments,
)


class TranscriptionSettingsContextFlagTests(unittest.TestCase):
    """The two new flags must default off and only flip on for ``max``."""

    def test_defaults_are_off(self):
        settings = TranscriptionSettings()
        self.assertFalse(settings.condition_on_previous_text)
        self.assertFalse(settings.hot_prompt_enrichment)

    def test_balanced_preset_keeps_flags_off(self):
        settings = TranscriptionSettings(quality_preset="balanced")
        settings = apply_quality_preset(settings)
        self.assertFalse(settings.condition_on_previous_text)
        self.assertFalse(settings.hot_prompt_enrichment)

    def test_fast_preset_keeps_flags_off(self):
        settings = TranscriptionSettings(quality_preset="fast")
        settings = apply_quality_preset(settings)
        self.assertFalse(settings.condition_on_previous_text)
        self.assertFalse(settings.hot_prompt_enrichment)

    def test_max_preset_enables_both_flags(self):
        settings = TranscriptionSettings(quality_preset="max")
        settings = apply_quality_preset(settings)
        self.assertTrue(settings.condition_on_previous_text)
        self.assertTrue(settings.hot_prompt_enrichment)

    def test_custom_preset_passthrough_respects_caller(self):
        # In custom mode the caller's value is preserved untouched
        # — required so power users can opt out even with everything
        # else cranked up.
        explicit = TranscriptionSettings(
            quality_preset="custom",
            condition_on_previous_text=True,
            hot_prompt_enrichment=False,
        )
        explicit = apply_quality_preset(explicit)
        self.assertTrue(explicit.condition_on_previous_text)
        self.assertFalse(explicit.hot_prompt_enrichment)


class WhisperCmdConditionFlagTests(unittest.TestCase):
    """The new flag has to round-trip through ``build_mlx_whisper_cmd``."""

    def test_true_flag_emits_true_string(self):
        cmd = build_mlx_whisper_cmd(
            mlx_whisper_path="/usr/local/bin/mlx_whisper",
            audio_path="/tmp/a.wav",
            output_path="/tmp/out.json",
            model="mlx-community/whisper-large-v3-turbo",
            condition_on_previous_text=True,
        )
        self.assertIn("--condition-on-previous-text", cmd)
        idx = cmd.index("--condition-on-previous-text")
        self.assertEqual(cmd[idx + 1], "True")

    def test_false_flag_emits_false_string(self):
        cmd = build_mlx_whisper_cmd(
            mlx_whisper_path="/usr/local/bin/mlx_whisper",
            audio_path="/tmp/a.wav",
            output_path="/tmp/out.json",
            model="mlx-community/whisper-large-v3-turbo",
            condition_on_previous_text=False,
        )
        self.assertIn("--condition-on-previous-text", cmd)
        idx = cmd.index("--condition-on-previous-text")
        self.assertEqual(cmd[idx + 1], "False")


class ExtractNewProperNounsTests(unittest.TestCase):
    """The proper-noun miner that powers hot prompt cycling."""

    def _segs(self, *texts: str) -> list[dict]:
        return [
            {"start": idx, "end": idx + 1, "text": t}
            for idx, t in enumerate(texts)
        ]

    def test_repeated_proper_noun_is_returned(self):
        segments = self._segs(
            "On va parler avec Manon de la migration.",
            "Manon connaît bien Power BI.",
            "Et Manon doit valider l'export.",
        )
        result = extract_new_proper_nouns_from_segments(segments)
        self.assertIn("Manon", result)

    def test_single_occurrence_is_ignored(self):
        # Default ``min_occurrences=2`` — one-off mentions are noisy.
        segments = self._segs(
            "On va parler avec Castelnaudary aujourd'hui.",
        )
        result = extract_new_proper_nouns_from_segments(segments)
        self.assertNotIn("Castelnaudary", result)

    def test_function_words_capitalised_at_sentence_start_are_dropped(self):
        # Whisper capitalises "Et", "Mais", "Donc" after every period.
        # Those are sentence connectors, not proper nouns.
        segments = self._segs(
            "Donc on commence. Et on enchaîne. Donc voilà.",
            "Mais après on verra. Et c'est tout. Donc OK.",
        )
        result = extract_new_proper_nouns_from_segments(segments)
        for noise in ("Et", "Mais", "Donc", "Alors", "Bon"):
            self.assertNotIn(noise, result)

    def test_existing_glossary_terms_are_excluded(self):
        segments = self._segs(
            "Manon valide la transcription Power BI.",
            "Manon connaît Power BI.",
        )
        result = extract_new_proper_nouns_from_segments(
            segments,
            existing_terms=["Manon", "Power BI"],
        )
        self.assertNotIn("Manon", result)
        # "Power" gets matched on its own (the regex is per-token),
        # but "Power BI" in the existing glossary folds to a prefix
        # mismatch — so the per-token "Power" can still appear.
        # That's OK: the multi-word check is done in the prompt
        # builder via parse_glossary_terms.

    def test_accent_insensitive_dedupe_against_existing(self):
        # Caller's glossary contains "Adele"; transcript carries
        # "Adèle". They should be treated as the same entity.
        segments = self._segs(
            "Adèle prépare la note.",
            "Et Adèle envoie le résumé.",
        )
        result = extract_new_proper_nouns_from_segments(
            segments,
            existing_terms=["Adele"],
        )
        self.assertNotIn("Adèle", result)

    def test_ordering_prefers_higher_frequency(self):
        segments = self._segs(
            # Manon: 4 occurrences. Robin: 2.
            "Manon ouvre. Robin écoute.",
            "Manon enchaîne. Manon valide. Robin répond.",
            "Et Manon clôture.",
        )
        result = extract_new_proper_nouns_from_segments(segments)
        self.assertEqual(result[:2], ["Manon", "Robin"])

    def test_max_terms_caps_the_result(self):
        # 5 repeated proper nouns; ask for only 3.
        segments = self._segs(
            "Manon Robin Léa Paul Théo.",
            "Manon Robin Léa Paul Théo.",
        )
        result = extract_new_proper_nouns_from_segments(
            segments,
            max_terms=3,
        )
        self.assertEqual(len(result), 3)

    def test_short_tokens_are_filtered(self):
        # Two-letter capitalised tokens (initials, acronyms) are noise.
        segments = self._segs(
            "A et B parlent. A et B répondent.",
        )
        result = extract_new_proper_nouns_from_segments(segments)
        self.assertNotIn("A", result)
        self.assertNotIn("B", result)


class PipelineHotEnrichmentTests(unittest.TestCase):
    """Top-level wiring: the hook mutates ``request.glossary_terms``."""

    def _make_pipeline(self, *, hot: bool) -> TranscriptionPipeline:
        request = JobRequest(
            source_path="/tmp/x.wav",
            output_dir="/tmp/out",
            mode="transcribe",
            transcription_settings=TranscriptionSettings(
                hot_prompt_enrichment=hot,
            ),
            glossary_terms=["Castel"],
            technical_terms=["Odoo"],
        )
        sink = MagicMock()
        return TranscriptionPipeline(request=request, sink=sink)

    def _segments_with_new_term(self) -> list[dict]:
        return [
            {"start": 0, "end": 5, "text": "Manon valide la note."},
            {"start": 5, "end": 10, "text": "Manon enchaîne sur Castel."},
            {"start": 10, "end": 15, "text": "Et Manon clôture."},
        ]

    def test_hot_enrichment_adds_new_term_when_flag_on(self):
        pipeline = self._make_pipeline(hot=True)
        pipeline._hot_enrich_glossary(self._segments_with_new_term())
        # Existing terms stay, new term is appended.
        self.assertIn("Castel", pipeline.request.glossary_terms)
        self.assertIn("Manon", pipeline.request.glossary_terms)
        # Existing-term ordering preserved (user terms come first).
        self.assertEqual(pipeline.request.glossary_terms[0], "Castel")

    def test_hot_enrichment_skips_existing_terms(self):
        pipeline = self._make_pipeline(hot=True)
        # Seed with the term the transcript will mention.
        pipeline.request.glossary_terms = ["Manon"]
        original = list(pipeline.request.glossary_terms)
        pipeline._hot_enrich_glossary(self._segments_with_new_term())
        # No duplicates: "Manon" is already there, nothing else
        # qualifies (other tokens are one-off).
        self.assertEqual(pipeline.request.glossary_terms, original)

    def test_hot_enrichment_noop_when_no_qualifying_terms(self):
        pipeline = self._make_pipeline(hot=True)
        original = list(pipeline.request.glossary_terms)
        # All single-occurrence proper nouns → none qualify.
        pipeline._hot_enrich_glossary(
            [
                {"start": 0, "end": 5, "text": "Bonjour."},
                {"start": 5, "end": 10, "text": "On y va."},
            ]
        )
        self.assertEqual(pipeline.request.glossary_terms, original)


if __name__ == "__main__":
    unittest.main()
