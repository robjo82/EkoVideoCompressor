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


if __name__ == "__main__":
    unittest.main()
