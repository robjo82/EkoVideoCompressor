import unittest

from glossary_postprocess import (  # noqa: E402
    apply_glossary_to_segments,
    apply_glossary_to_text,
    french_phonetic_key,
    parse_glossary_terms,
)


class FrenchPhoneticKeyTest(unittest.TestCase):
    """
    The keys themselves are an implementation detail, but the
    invariant matters: any two real-world Whisper variants of a
    known glossary word should produce the same key.
    """

    def test_mollie_variants_match(self):
        # We don't require IDENTICAL phonetic keys — that's too strict
        # given final-vowel asymmetry ("Mollie" keeps the trailing 'i',
        # "MOLLE" strips its mute 'e'). The contract we actually care
        # about is that they all get *matched* by apply_glossary_to_text.
        for variant in ("MOLI", "MOLLE", "Molli", "Mollys", "MOLIE"):
            with self.subTest(variant=variant):
                new, subs = apply_glossary_to_text(
                    f"Le module {variant} est utile.", ["Mollie"]
                )
                self.assertEqual(len(subs), 1, f"{variant!r} should match")

    def test_sudokies_variants_collide(self):
        canonical = french_phonetic_key("Sudokies")
        for variant in ("Sudokiz", "Sudokis", "SOUDOKIS"):
            with self.subTest(variant=variant):
                self.assertEqual(french_phonetic_key(variant), canonical)

    def test_symphonat_variants_collide(self):
        canonical = french_phonetic_key("Symphonat")
        for variant in ("Symphonate", "Simfonat", "Simphonate"):
            with self.subTest(variant=variant):
                self.assertEqual(french_phonetic_key(variant), canonical)

    def test_unrelated_words_do_not_collide(self):
        # Sanity: phonetic collisions shouldn't sweep in random vocabulary.
        for a, b in [
            ("Mollie", "Pierre"),
            ("Sudokies", "Mollie"),
            ("Symphonat", "Klarna"),
            ("Odoo", "Robin"),
            ("client", "bonjour"),
        ]:
            with self.subTest(a=a, b=b):
                self.assertNotEqual(french_phonetic_key(a), french_phonetic_key(b))

    def test_strips_accents_and_case(self):
        self.assertEqual(
            french_phonetic_key("Adèle"),
            french_phonetic_key("ADELE"),
        )
        self.assertEqual(
            french_phonetic_key("Ekonum"),
            french_phonetic_key("ekonum"),
        )


class ParseGlossaryTermsTest(unittest.TestCase):
    def test_splits_commas_and_newlines(self):
        out = parse_glossary_terms("Mollie, Klarna\nOdoo; Symphonat")
        self.assertEqual(out, ["Mollie", "Klarna", "Odoo", "Symphonat"])

    def test_strips_instruction_lines(self):
        raw = "Vocabulaire à respecter, noms propres:\nMollie\nKlarna"
        out = parse_glossary_terms(raw)
        self.assertEqual(out, ["Mollie", "Klarna"])

    def test_preserves_user_casing(self):
        out = parse_glossary_terms("Mollie, MGX Contrôles, iVerif")
        self.assertEqual(out, ["Mollie", "MGX Contrôles", "iVerif"])

    def test_dedupes_case_insensitive(self):
        out = parse_glossary_terms("Mollie, MOLLIE, mollie")
        self.assertEqual(out, ["Mollie"])

    def test_drops_tiny_tokens(self):
        # Single-character "M" tokens are usually leftover bullet
        # noise, not glossary entries.
        out = parse_glossary_terms("M, Mollie, X")
        self.assertEqual(out, ["Mollie"])


class ApplyGlossaryToTextTest(unittest.TestCase):
    """
    The four regression cases from the real Symphonat phone call.
    """

    def test_fixes_mollie_variants(self):
        # The same line — but with a mix of casings — should rewrite
        # to "Mollie" while echoing the original casing per-occurrence.
        text = (
            "Donc il y a un module qui s'appelle MOLI official, "
            "et le reste est dans le module Molli."
        )
        new, subs = apply_glossary_to_text(text, ["Mollie"])
        self.assertNotIn(" MOLI ", new)
        self.assertNotIn(" Molli.", new)
        self.assertEqual(len(subs), 2)
        # Casing echo: the ALL-CAPS occurrence becomes "MOLLIE",
        # the title-case occurrence becomes "Mollie".
        self.assertIn("MOLLIE", new)
        self.assertIn("Mollie", new)

    def test_fixes_sudokies(self):
        text = "On a travaillé avec Sudokiz à Montauban."
        new, subs = apply_glossary_to_text(text, ["Sudokies"])
        self.assertIn("Sudokies", new)
        self.assertNotIn("Sudokiz", new)
        self.assertEqual(len(subs), 1)

    def test_fixes_symphonat_within_sentence(self):
        text = "Bienvenue à Symphonate, toute l'équipe de Symphonate vous remercie."
        new, subs = apply_glossary_to_text(text, ["Symphonat"])
        # Both occurrences fixed.
        self.assertEqual(new.count("Symphonat"), 2)
        self.assertNotIn("Symphonate", new)
        self.assertEqual(len(subs), 2)

    def test_no_change_when_text_already_canonical(self):
        text = "On parle de Mollie et de Klarna."
        new, subs = apply_glossary_to_text(text, ["Mollie", "Klarna"])
        self.assertEqual(new, text)
        self.assertEqual(subs, [])

    def test_does_not_rewrite_unrelated_words(self):
        text = "Bonjour, je m'appelle Pierre et je travaille sur Odoo."
        new, subs = apply_glossary_to_text(text, ["Mollie", "Klarna"])
        self.assertEqual(new, text)
        self.assertEqual(subs, [])

    def test_does_not_inject_short_glossary_names_everywhere(self):
        text = (
            "Les commandes sont validées avec un paiement sécurisé. "
            "J'ai reçu sa demande par mail et Juliette a répondu à Philippe."
        )
        new, subs = apply_glossary_to_text(text, ["Romain", "Mollie"])
        self.assertEqual(new, text)
        self.assertEqual(subs, [])

    def test_does_not_rewrite_robin_to_romain(self):
        text = "Robin Joseph rappelle le client."
        new, subs = apply_glossary_to_text(text, ["Romain"])
        self.assertEqual(new, text)
        self.assertEqual(subs, [])

    def test_multi_word_terms(self):
        text = "On gère les CVR Controles depuis Albi."
        new, subs = apply_glossary_to_text(text, ["CVR Contrôles"])
        self.assertIn("CVR Contrôles", new)
        self.assertEqual(len(subs), 1)
        self.assertEqual(subs[0].replacement, "CVR Contrôles")

    def test_substitution_records_context(self):
        text = "On parle de MOLI dans la suite de l'appel."
        new, subs = apply_glossary_to_text(text, ["Mollie"])
        self.assertEqual(len(subs), 1)
        self.assertIn("On parle", subs[0].context_before)
        self.assertIn("dans la suite", subs[0].context_after)


class ApplyGlossaryToSegmentsTest(unittest.TestCase):
    def test_stamps_timestamp_on_each_substitution(self):
        segments = [
            {"start": 12.5, "end": 14.0, "text": "Bonjour à Sudokiz."},
            {"start": 30.0, "end": 32.0, "text": "MOLI est le module officiel."},
            {"start": 50.0, "end": 51.0, "text": "Rien à changer ici."},
        ]
        out, subs = apply_glossary_to_segments(segments, ["Sudokies", "Mollie"])
        self.assertEqual(out[0]["text"], "Bonjour à Sudokies.")
        self.assertEqual(out[1]["text"], "MOLLIE est le module officiel.")
        self.assertEqual(out[2]["text"], "Rien à changer ici.")
        self.assertEqual(len(subs), 2)
        self.assertEqual(subs[0].timestamp_seconds, 12.5)
        self.assertEqual(subs[1].timestamp_seconds, 30.0)

    def test_keeps_segments_when_terms_empty(self):
        segments = [{"start": 0.0, "end": 1.0, "text": "Bonjour"}]
        out, subs = apply_glossary_to_segments(segments, [])
        self.assertEqual(out, segments)
        self.assertEqual(subs, [])


class RobustnessTest(unittest.TestCase):
    def test_high_confidence_blocks_random_drift(self):
        # "bonjour" should NEVER be rewritten to "Mollie" even though
        # both are short French words. The phonetic keys differ.
        text = "Bonjour tout le monde."
        new, subs = apply_glossary_to_text(text, ["Mollie"])
        self.assertEqual(new, text)
        self.assertEqual(subs, [])

    def test_does_not_chain_substitutions(self):
        # If "MOLI" → "Mollie", the next pass must not consider
        # "Mollie" itself as a candidate again. This is enforced by
        # `consumed[]` and the case-insensitive canonical check.
        text = "MOLI et Mollie."
        new, subs = apply_glossary_to_text(text, ["Mollie"])
        # Only the first occurrence was wrong.
        self.assertEqual(len(subs), 1)
        self.assertIn("Mollie", new)


class MergedWindowMatchingTest(unittest.TestCase):
    """PR P — catch Whisper's multi-token hallucinations like
    ``pouvoir bien`` for ``Power BI``. The per-token matcher's
    surface guard rejects ``pouvoir`` vs ``power`` because Lev=4,
    but the joined phonetic key only differs by 1."""

    def test_pouvoir_bien_collapses_to_power_bi(self):
        text = "On utilise du pouvoir bien pour la data."
        new, subs = apply_glossary_to_text(text, ["Power BI"])
        self.assertIn("Power BI", new)
        self.assertEqual(len(subs), 1)
        self.assertEqual(subs[0].method, "merged_window")

    def test_short_glossary_term_does_not_overmatch(self):
        # ``Odoo`` (ADA key, 3 chars) must NOT match arbitrary
        # 2-token French. The min key length guard (≥ 4 on the
        # entry side) blocks ``Odoo`` from getting picked at all
        # by the merged-window pass — it's too phonetically risky.
        text = "Bonjour à tous, comment ça va ?"
        new, subs = apply_glossary_to_text(text, ["Odoo"])
        self.assertEqual(new, text)
        self.assertEqual(subs, [])

    def test_first_letter_safety_net(self):
        # The first letters of the joined window vs the entry's
        # joined surface must match. Stops cross-letter phonetic
        # collisions.
        text = "Le matin nous a surpris."
        new, subs = apply_glossary_to_text(text, ["Sudokies"])
        self.assertEqual(new, text)
        self.assertEqual(subs, [])

    def test_window_does_not_consume_already_substituted_tokens(self):
        # When the first pass already substituted a token, the
        # merged-window pass walks past it instead of re-attempting.
        text = "Sudokiz et pouvoir bien."
        new, subs = apply_glossary_to_text(text, ["Sudokies", "Power BI"])
        # Both substitutions applied independently.
        self.assertIn("Sudokies", new)
        self.assertIn("Power BI", new)


if __name__ == "__main__":
    unittest.main()
