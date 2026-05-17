"""Tests for the LLM-corrections-to-text apply path.

These exist because the previous pipeline listed corrections in the
review report but never actually applied them — every "améliorée"
file was byte-identical to the raw transcript. The tests pin the
guardrails so a future refactor can't silently regress to the
listing-only behaviour and so wild LLM proposals (rewrites, made-up
quotes, low confidence) stay rejected.
"""

from __future__ import annotations

import unittest

from llm_corrections import (
    AppliedCorrection,
    RejectedCorrection,
    apply_llm_corrections_to_text,
)


class ApplyCorrectionsTest(unittest.TestCase):
    def test_simple_replacement_is_applied(self):
        text = "Bonjour, ici Sudokiz."
        out = apply_llm_corrections_to_text(
            text,
            [{"original": "Sudokiz", "replacement": "Sudokies", "confidence": 0.85}],
        )
        self.assertEqual(out.text, "Bonjour, ici Sudokies.")
        self.assertEqual(len(out.applied), 1)
        self.assertEqual(out.applied[0].occurrences, 1)
        self.assertEqual(out.rejected, [])

    def test_accent_insensitive_match_lands_with_canonical_form(self):
        # The LLM frequently drops accents in the `original` it quotes
        # ("Adele Herbawi") but the transcript still has them.
        # The replacement must still find and substitute the accented
        # form — that's the whole reason this module exists.
        text = "On parle de Adèle Herbawi aujourd'hui."
        out = apply_llm_corrections_to_text(
            text,
            [
                {
                    "original": "Adele Herbawi",
                    "replacement": "Adèle Herbaoui",
                    "confidence": 0.85,
                }
            ],
        )
        self.assertIn("Adèle Herbaoui", out.text)
        self.assertNotIn("Adèle Herbawi", out.text)

    def test_rewrite_too_distant_is_rejected(self):
        # A correction that turns one phrase into a paraphrase is a
        # style edit, not a transcription fix. We refuse to let the
        # LLM rewrite the user's words under the guise of correcting
        # Whisper.
        text = "On parle de Adèle aujourd'hui."
        out = apply_llm_corrections_to_text(
            text,
            [
                {
                    "original": "On parle de Adèle",
                    "replacement": "On parle ce matin de notre amie",
                    "confidence": 0.85,
                }
            ],
        )
        self.assertEqual(out.text, text)
        self.assertEqual(len(out.rejected), 1)
        self.assertEqual(out.rejected[0].reason, "too_distant")

    def test_hallucinated_quote_is_rejected_with_not_found(self):
        # The LLM sometimes paraphrases what it *heard* about a
        # passage rather than quoting it. We catch this by refusing
        # to substitute anything that isn't actually present.
        text = "Bonjour, ici Sudokiz."
        out = apply_llm_corrections_to_text(
            text,
            [
                {
                    "original": "jamais dit",
                    "replacement": "jamais entendu",
                    "confidence": 0.85,
                }
            ],
        )
        self.assertEqual(out.text, text)
        self.assertEqual(out.rejected[0].reason, "not_found")

    def test_low_confidence_is_rejected_before_other_checks(self):
        # A model that flags its own correction at <0.7 confidence
        # is telling us not to trust it — honour that.
        text = "Bonjour, ici Sudokiz."
        out = apply_llm_corrections_to_text(
            text,
            [
                {
                    "original": "Sudokiz",
                    "replacement": "Sudokies",
                    "confidence": 0.4,
                }
            ],
        )
        self.assertEqual(out.text, text)
        self.assertEqual(out.rejected[0].reason, "low_confidence")

    def test_empty_fields_are_rejected_as_empty(self):
        text = "Bonjour."
        out = apply_llm_corrections_to_text(
            text,
            [
                {"original": "", "replacement": "Salut", "confidence": 0.9},
                {"original": "Bonjour", "replacement": "", "confidence": 0.9},
            ],
        )
        self.assertEqual(out.text, text)
        self.assertEqual([r.reason for r in out.rejected], ["empty", "empty"])

    def test_replacement_cap_limits_per_correction_substitutions(self):
        # A recurring proper noun appearing many times should still
        # only get capped substitutions — otherwise a single faulty
        # correction could rewrite half the transcript. The cap also
        # protects against the LLM hallucinating a tiny common word
        # as a fix target.
        text = "Sudokiz, Sudokiz, Sudokiz, Sudokiz, Sudokiz."
        out = apply_llm_corrections_to_text(
            text,
            [{"original": "Sudokiz", "replacement": "Sudokies", "confidence": 0.9}],
            max_replacements_per_correction=3,
        )
        self.assertEqual(out.applied[0].occurrences, 3)
        self.assertEqual(out.text.count("Sudokies"), 3)
        self.assertEqual(out.text.count("Sudokiz"), 2)

    def test_corrections_walk_in_timestamp_order(self):
        # The audit list preserves timestamp order regardless of
        # input shuffling — gives the downstream review report a
        # stable, readable timeline.
        text = "Alpha quelque part. Bravo ailleurs."
        out = apply_llm_corrections_to_text(
            text,
            [
                {
                    "timestamp": "00:00:10",
                    "original": "Bravo",
                    "replacement": "Bravoo",
                    "confidence": 0.9,
                },
                {
                    "timestamp": "00:00:05",
                    "original": "Alpha",
                    "replacement": "Alphaa",
                    "confidence": 0.9,
                },
            ],
        )
        self.assertEqual([a.original for a in out.applied], ["Alpha", "Bravo"])
        self.assertIn("Alphaa", out.text)
        self.assertIn("Bravoo", out.text)


class NoopAfterNormalisationTest(unittest.TestCase):
    def test_rejects_cedilla_corruption(self):
        # The LLM occasionally emits cedilla-corrupted French words
        # ("clique" → "çlique") that pass both the literal-presence
        # and the Levenshtein-distance checks. Their normalised forms
        # are identical to the originals, so the "correction" is a
        # pure no-op that just litters the transcript with mid-word
        # diacritics. Pin the new guardrail.
        text = "Quand je clique sur le bouton."
        out = apply_llm_corrections_to_text(
            text,
            [{"original": "clique", "replacement": "çlique", "confidence": 0.85}],
        )
        self.assertEqual(out.text, text)
        self.assertEqual(out.rejected[0].reason, "noop_after_normalization")

    def test_rejects_accent_only_noise(self):
        # "Et après" → "Et apres" (accent stripped) would also be a
        # no-op once normalised. Both directions of the
        # accent-corruption bug surface as the same rejection.
        text = "Et après que ça se passe."
        out = apply_llm_corrections_to_text(
            text,
            [{"original": "après", "replacement": "apres", "confidence": 0.85}],
        )
        self.assertEqual(out.text, text)
        self.assertEqual(out.rejected[0].reason, "noop_after_normalization")

    def test_keeps_real_correction_with_meaningful_change(self):
        # Guard: the new check must not over-reject. "Sudokiz" vs
        # "Sudokies" normalises differently (sudokiz vs sudokies),
        # so the correction lands.
        text = "Bonjour, ici Sudokiz."
        out = apply_llm_corrections_to_text(
            text,
            [{"original": "Sudokiz", "replacement": "Sudokies", "confidence": 0.85}],
        )
        self.assertEqual(out.text, "Bonjour, ici Sudokies.")
        self.assertEqual(out.rejected, [])


if __name__ == "__main__":
    unittest.main()
