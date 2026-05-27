"""
PR W — LLM speaker identification prompt hardening.

The CVR run produced ``SPEAKER_02 → Nicolas`` even though Nicolas
Lacombe is not in the conversation — he's mentioned by Robin in
the opening exchange. The LLM's previous prompt didn't make the
distinction between "this SPEAKER is X" and "this SPEAKER mentions X".

The prompt now contains a dedicated "IDENTITÉ vs MENTION" section
with explicit examples. These tests pin the prompt content so a
future refactor doesn't accidentally silence the guidance.

We can't easily test the LLM output (no model in CI), so we
verify the prompt contains the canonical anchors. That's the
contract the prompt is meant to honour.
"""

from __future__ import annotations

import unittest

import transcription_utils


class LlmSpeakerPromptContentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.script = transcription_utils._LLM_TITLE_SPEAKERS_SCRIPT

    def test_script_compiles_as_python(self):
        # Sanity check that the f-string + multiline string remains
        # valid Python after the PR W rewrite.
        compile(self.script, "<llm-title>", "exec")

    def test_prompt_warns_about_mention_vs_identity(self):
        # Anchor strings the future maintainer can search for.
        self.assertIn("IDENTITÉ vs MENTION", self.script)
        self.assertIn("n'est PAS le nom de ce SPEAKER", self.script)

    def test_prompt_lists_explicit_self_introduction_signals(self):
        # The prompt enumerates the patterns that DO justify
        # attribution. These are the explicit auto-introductions.
        for signal in (
            "je suis Robin",
            "Robin Joseph à l'appareil",
            "Moi c'est Manon",
            "Je m'appelle Vincent",
        ):
            self.assertIn(signal, self.script)

    def test_prompt_carries_negative_example_nicolas_lacombe(self):
        # The CVR-control regression: Nicolas Lacombe mentioned by
        # Robin, mis-attributed to SPEAKER_00 in the buggy run. The
        # negative example reproduces the exact pattern.
        self.assertIn("Nicolas Lacombe", self.script)
        self.assertIn("Mauvaise sortie", self.script)
        self.assertIn("Bonne sortie", self.script)

    def test_prompt_carries_addressing_example(self):
        # The "Manon, est-ce que tu peux nous montrer" example shows
        # that being addressed by name does NOT mean you are that
        # name — it means the OTHER speaker is. Important for
        # back-and-forth conversational patterns.
        self.assertIn("ADRESSE Manon", self.script)
        self.assertIn("ce n'est pas lui", self.script)

    def test_prompt_instructs_to_default_to_empty_on_doubt(self):
        # Failing closed is the conservative behaviour we want.
        self.assertIn("Quand tu hésites, laisse la chaîne vide", self.script)

    def test_prompt_still_carries_existing_constraints(self):
        # Sanity: the older constraints (no markdown, no inventing
        # names, JSON shape) must survive the PR W rewrite.
        self.assertIn("Réponds UNIQUEMENT par un objet JSON valide", self.script)
        self.assertIn("N'invente JAMAIS un prénom", self.script)
        self.assertIn('"title"', self.script)
        self.assertIn('"speakers"', self.script)
        self.assertIn('"technical_terms"', self.script)

    # -- PR AA: title format "Société - Sujet" -----------------------

    def test_prompt_specifies_company_dash_topic_format(self):
        # The new title format anchor.
        self.assertIn("Nom de société - Sujet court", self.script)
        self.assertIn("FORMAT DU TITRE", self.script)

    def test_prompt_excludes_ekonum_from_company_choice(self):
        # The user runs the app, so Ekonum is always one of the
        # parties — but the user wants the CLIENT's name, not their
        # own. The prompt makes this explicit.
        self.assertIn("non-Ekonum", self.script)

    def test_prompt_carries_company_title_examples(self):
        # Positive examples covering our real client cases.
        for example in (
            "CVR Contrôles - Configuration site web Odoo",
            "Caste - Audit système ERP et reporting",
            "Acritec - Migration facturation vers Odoo",
        ):
            self.assertIn(example, self.script)

    def test_prompt_allows_no_company_fallback(self):
        # When no client company is clear, the prompt allows a
        # bare topic — better than forcing a wrong company name.
        # PR AL: wording was tightened from "Acceptable sans
        # société" to "sans société = OK quand pas de client
        # identifiable" within the examples block. The
        # "omets le préfixe" instruction is still there (now used
        # in both the Ekonum ban and the general fallback rules).
        self.assertIn("omets le préfixe", self.script)
        self.assertIn("sans société = OK", self.script)

    # PR AL — explicit Ekonum ban anchors.

    def test_prompt_explicitly_bans_ekonum_as_company(self):
        # The CVR rerun produced ``Ekonum - Audit système ERP`` for a
        # Caste meeting. The prompt now contains an explicit
        # interdiction block + concrete bad-example bullets.
        self.assertIn("INTERDICTION ABSOLUE : EKONUM", self.script)
        self.assertIn("Ekonum est l'entreprise du locuteur", self.script)

    def test_prompt_lists_ekonum_variants_in_ban(self):
        # The ban enumerates the spelling variants Mistral has
        # produced (``Ekonum``, ``Econum``, ``EKONUM``) so a
        # case-fold or typo doesn't slip through.
        for variant in ("Ekonum", "EKONUM", "Econum"):
            self.assertIn(variant, self.script)

    def test_prompt_carries_ekonum_bad_examples(self):
        # Negative examples reproducing the exact CVR/Caste failure
        # mode so the LLM sees concrete patterns to avoid.
        self.assertIn("Ekonum - Audit système ERP", self.script)
        self.assertIn("INTERDIT", self.script)


if __name__ == "__main__":
    unittest.main()
