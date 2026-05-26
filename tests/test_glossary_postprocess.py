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
    but the joined phonetic key only differs by 1.

    PR R — merged-window is gated behind ``merged_window_enabled``
    (default False). These tests opt in explicitly to exercise
    the path. Production callers in ``pipeline.py`` leave it off
    because the failure modes audited on the CVR run outweighed
    the win on ``pouvoir bien``-style hallucinations.
    """

    def test_pouvoir_bien_collapses_to_power_bi(self):
        text = "On utilise du pouvoir bien pour la data."
        new, subs = apply_glossary_to_text(
            text, ["Power BI"], merged_window_enabled=True
        )
        self.assertIn("Power BI", new)
        self.assertEqual(len(subs), 1)
        self.assertEqual(subs[0].method, "merged_window")

    def test_short_glossary_term_does_not_overmatch(self):
        # ``Odoo`` (ADA key, 3 chars) must NOT match arbitrary
        # 2-token French. PR R raised the minimum entry key length
        # to 5, so ``Odoo`` is rejected for merged-window outright
        # — the per-token matcher already handles single-token
        # glossary terms.
        text = "Bonjour à tous, comment ça va ?"
        new, subs = apply_glossary_to_text(
            text, ["Odoo"], merged_window_enabled=True
        )
        self.assertEqual(new, text)
        self.assertEqual(subs, [])

    def test_first_letter_safety_net(self):
        # The first letters of the joined window vs the entry's
        # joined surface must match. Stops cross-letter phonetic
        # collisions.
        text = "Le matin nous a surpris."
        new, subs = apply_glossary_to_text(
            text, ["Sudokies"], merged_window_enabled=True
        )
        self.assertEqual(new, text)
        self.assertEqual(subs, [])

    def test_window_does_not_consume_already_substituted_tokens(self):
        # When the first pass already substituted a token, the
        # merged-window pass walks past it instead of re-attempting.
        text = "Sudokiz et pouvoir bien."
        new, subs = apply_glossary_to_text(
            text,
            ["Sudokies", "Power BI"],
            merged_window_enabled=True,
        )
        # Both substitutions applied independently.
        self.assertIn("Sudokies", new)
        self.assertIn("Power BI", new)

    # -- PR R regression tests --------------------------------------

    def test_merged_window_disabled_by_default(self):
        # Same input as test_pouvoir_bien_collapses_to_power_bi but
        # without opting in. The merged-window pass stays silent.
        text = "On utilise du pouvoir bien pour la data."
        new, subs = apply_glossary_to_text(text, ["Power BI"])
        self.assertEqual(new, text)
        self.assertEqual(subs, [])

    def test_stoplist_rejects_function_word_window_par(self):
        # The canonical CVR-control failure: ``par`` is on the
        # stoplist, so even with merged_window on it never gets
        # rewritten to ``Parce``.
        text = "Donc par exemple on regarde."
        new, subs = apply_glossary_to_text(
            text, ["Parce"], merged_window_enabled=True
        )
        self.assertIn("par", new)
        self.assertNotIn("Parce", new)

    def test_stoplist_rejects_function_word_window_vient_ici(self):
        # ``vient ici`` got rewritten to ``Vincent`` (a glossary
        # speaker name) on the CVR audit. ``vient`` and ``ici``
        # are both on the stoplist → window refused.
        text = "Le commercial vient ici tous les jeudis."
        new, subs = apply_glossary_to_text(
            text, ["Vincent"], merged_window_enabled=True
        )
        self.assertNotIn("Vincent", new)
        self.assertEqual(subs, [])

    def test_stoplist_rejects_function_word_window_du_coup_sur(self):
        # ``Du coup, sur`` → ``Document`` was another offender.
        # ``du`` and ``sur`` are both stoplisted.
        text = "Du coup, sur la facture il y a un lien."
        new, subs = apply_glossary_to_text(
            text, ["Document"], merged_window_enabled=True
        )
        self.assertNotIn("Document", new)

    def test_two_letter_surface_guard(self):
        # ``le sont`` (LS surface) used to match ``Laurent`` (LRN
        # surface) because they share the first letter ``L``. PR R
        # raised the shared-prefix requirement to 2 letters → window
        # rejected. Plus ``le`` and ``sont`` are both stoplisted —
        # this test pins the surface guard layer specifically with
        # non-stoplisted tokens.
        text = "Lutter contre l'évasion fiscale."
        new, subs = apply_glossary_to_text(
            text, ["Laurent"], merged_window_enabled=True
        )
        self.assertNotIn("Laurent", new)


class PrAePhoneticEdgeCasesTest(unittest.TestCase):
    """PR AE — false positives caught on the CVR/Caste rerun:
    French words with diacritics rewritten to ASCII proper nouns
    (``Contrôle → Control``, ``tourné → Tarn``, ``allait → Allix``),
    plus the ``facturation électronique → facturation électronique``
    no-op that polluted the review report because of NFC/NFD
    mismatch."""

    def test_french_diacritic_source_protected_from_ascii_entry(self):
        # ``Contrôle`` (real FR word, has ``ô``) MUST NOT be
        # rewritten to ``Control`` (English brand on glossary).
        text = "Nous avons un Contrôle technique demain."
        new, subs = apply_glossary_to_text(text, ["Control"])
        self.assertEqual(new, text)
        self.assertEqual(subs, [])

    def test_tourne_not_rewritten_to_tarn(self):
        # ``tourné`` past participle vs ``Tarn`` region.
        text = "On a tourné une vidéo dans l'usine."
        new, subs = apply_glossary_to_text(text, ["Tarn"])
        self.assertEqual(new, text)
        self.assertEqual(subs, [])

    def test_allait_not_rewritten_to_allix(self):
        # ``allait`` imperfect of "aller" vs ``Allix`` software.
        text = "Il allait nous présenter le projet."
        new, subs = apply_glossary_to_text(text, ["Allix"])
        self.assertEqual(new, text)
        self.assertEqual(subs, [])

    def test_ascii_source_can_still_match_ascii_entry(self):
        # The guard only fires when the SOURCE has diacritics.
        # ``moli`` → ``Mollie`` is still legitimate (both ASCII).
        text = "On signe avec moli pour le paiement."
        new, subs = apply_glossary_to_text(text, ["Mollie"])
        self.assertIn("Mollie", new)
        self.assertEqual(len(subs), 1)

    def test_accented_entry_can_still_be_canonicalised(self):
        # The guard checks BOTH sides for diacritics. ``Castelnau``
        # (no accent) hearing ``Castelnaü`` (accented Whisper
        # rendering) — both have French character variants in the
        # comparison set, so the guard doesn't fire. The matcher
        # should still try.
        text = "Bonjour à Castelnaü ce matin."
        # ``Castelnaü`` and ``Castelnau`` differ only by accent.
        # The source has the accent, the entry has it too →
        # diacritic guard doesn't fire.
        new, subs = apply_glossary_to_text(text, ["Castelnaü"])
        # Already-canonical short-circuit kicks in.
        self.assertEqual(new, text)
        self.assertEqual(subs, [])

    def test_facturation_electronique_no_op_squelched_by_nfc(self):
        # Reproduce the macOS NFD vs NFC mismatch:
        # Whisper writes the word as NFD (``é``),
        # user glossary stores it as NFC (``é``). They render
        # identically but compare unequal as raw strings — the
        # canonical-form short-circuit missed it, then the
        # matcher "corrected" NFD → NFC with method=phonetic
        # and confidence=0.95. The fix: NFC both sides before
        # comparing.
        import unicodedata
        nfd_source = unicodedata.normalize(
            "NFD", "facturation électronique"
        )
        text = f"On utilise la {nfd_source} maintenant."
        new, subs = apply_glossary_to_text(
            text, ["facturation électronique"]
        )
        # No substitution should fire because it's already the
        # canonical form (modulo NFC).
        self.assertEqual(subs, [])
        # The text might have been NFC-normalised or not, but no
        # substitution was logged either way.


if __name__ == "__main__":
    unittest.main()
