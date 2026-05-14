import unittest

import web_context
from web_context import (
    enrich_glossary_via_web,
    extract_entity_candidates,
)


class ExtractEntityCandidatesTest(unittest.TestCase):
    def test_picks_up_multi_token_proper_nouns(self):
        text = (
            "Bonjour, je vous appelle de chez Symphonat. Nous travaillons "
            "avec Sud Odoo Montauban et avec un intégrateur Ekonum sur "
            "le module Mollie."
        )
        cands = extract_entity_candidates(text)
        self.assertIn("Symphonat", cands)
        self.assertIn("Sud Odoo Montauban", cands)
        self.assertIn("Ekonum", cands)
        self.assertIn("Mollie", cands)

    def test_drops_french_sentence_openers(self):
        # "Bonjour" / "Voilà" appear at the start of sentences but
        # aren't proper nouns. They should not become candidates.
        text = "Bonjour. Voilà. Allez. Tout va bien."
        self.assertEqual(extract_entity_candidates(text), [])

    def test_caps_candidate_count(self):
        # 50 truly distinct multi-token names — the cap should keep
        # us at 10 so the search step never gets flooded. We avoid
        # digits in the words themselves because our caps token
        # pattern only takes the leading letter chunk (which would
        # collapse "Beta0", "Beta1", … to "Beta").
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        text = ". ".join(
            f"Alpha {letters[i % 26]}{chr(97 + (i // 26 + 1))}rba"
            for i in range(50)
        )
        out = extract_entity_candidates(text, max_candidates=10)
        self.assertEqual(len(out), 10)


class EnrichGlossaryViaWebTest(unittest.TestCase):
    """
    We inject a fake search function so we never touch the
    network during tests.
    """

    def _fake_search(self, expected_hits):
        from web_context import _SearchHit

        def search(query, *, timeout=6):
            hits = expected_hits.get(query.lower())
            if hits is None:
                return []
            return [_SearchHit(title=h[0], snippet=h[1]) for h in hits]

        return search

    def test_confirms_candidates_with_matching_title(self):
        text = "On utilise Mollie sur Odoo, partenaire Sudokies."
        search = self._fake_search(
            {
                "mollie": [
                    ("Mollie — payments platform", "Online payments for businesses"),
                ],
                "odoo": [("Odoo: open source ERP", "ERP for SMBs")],
                "sudokies": [
                    ("Sudokies | Intégrateur Odoo", "Spécialiste à Montauban"),
                ],
            }
        )
        out = enrich_glossary_via_web(text, user_glossary=[], search_fn=search)
        terms = [r.candidate for r in out]
        self.assertIn("Mollie", terms)
        self.assertIn("Sudokies", terms)
        # Odoo wasn't extracted because it's only one token and at
        # a sentence-start position — that's fine, we just want the
        # ones we can confirm.

    def test_skips_user_glossary_entries(self):
        text = "On utilise Mollie."
        search = self._fake_search(
            {"mollie": [("Mollie", "Payments")]}
        )
        # User already has "Mollie" in their glossary — no need to
        # re-confirm via the web.
        out = enrich_glossary_via_web(text, user_glossary=["Mollie"], search_fn=search)
        self.assertEqual(out, [])

    def test_respects_max_queries_budget(self):
        # Each entity must be truly distinct so we hit the cap,
        # not the dedup.
        text = ". ".join(
            f"Société Alpha {chr(65 + i)}arna" for i in range(20)
        )
        calls = {"n": 0}

        def search(query, *, timeout=6):
            calls["n"] += 1
            return []

        enrich_glossary_via_web(text, [], max_queries=3, search_fn=search)
        self.assertEqual(calls["n"], 3)

    def test_drops_unconfirmed_candidates(self):
        # A candidate whose web search returns hits that don't
        # mention it is not added to the enriched glossary.
        text = "Discussion avec Synergie Cosmique."
        search = self._fake_search(
            {
                "synergie cosmique": [
                    ("Tout autre chose", "Aucun rapport ici."),
                ],
            }
        )
        out = enrich_glossary_via_web(text, [], search_fn=search)
        self.assertEqual(out, [])

    def test_skip_path_when_text_empty(self):
        self.assertEqual(enrich_glossary_via_web("", []), [])


class StubSearchFunctionTest(unittest.TestCase):
    """
    Verify our DuckDuckGo parser is robust to its expected HTML
    shape using a captured fixture. We don't hit the real endpoint.
    """

    SAMPLE_HTML = """
    <html><body>
    <div class="result"><a class="result__a" href="x">Mollie — payments</a></div>
    <a class="result__snippet">Online <b>payment</b> platform for businesses.</a>
    <div class="result"><a class="result__a" href="y">Mollie Wikipedia</a></div>
    <a class="result__snippet">Mollie is a Dutch fintech…</a>
    </body></html>
    """

    def test_parses_titles_and_snippets(self):
        # We poke the parser by stubbing urlopen.
        import io
        class _Resp:
            def __init__(self, body):
                self._body = body
            def __enter__(self): return self
            def __exit__(self, *_): return False
            def read(self): return self._body.encode("utf-8")

        def fake_urlopen(req, timeout=6):
            return _Resp(StubSearchFunctionTest.SAMPLE_HTML)

        hits = web_context.duckduckgo_search("Mollie", opener=fake_urlopen)
        self.assertEqual(len(hits), 2)
        self.assertIn("Mollie", hits[0].title)
        self.assertIn("payment", hits[0].snippet.lower())


if __name__ == "__main__":
    unittest.main()
