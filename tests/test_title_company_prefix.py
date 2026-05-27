"""
PR AF — Python fallback for the "Société - Sujet" title format.

PR AA added prompt guidance asking Mistral 7B for that format,
but the model routinely ignores it. PR AF post-processes the
LLM output to prepend the resolved company name when missing.

Resolution order :
  1. Odoo context pack → ``primary.raw.partner_id`` or
     ``primary.display_name``.
  2. ``odoo_meeting_metadata.partners`` — calendar invite.
  3. Speaker overrides values matching ``"Nom (Société)"``.

Tests :
  • ``extract_company_name_from_pack`` reads ``partner_id``,
    falls back to ``display_name``, returns "" on empty pack.
  • ``_apply_title_company_prefix`` prepends when missing,
    leaves untouched when the LLM already did the job.
  • Idempotent on rerun (no double prefix).
  • Doesn't double up when the LLM already produced a wrong
    company prefix — defers to the LLM in that case.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from ekovideo_engine.models import (
    JobRequest,
    OdooContextRef,
    TranscriptionSettings,
)
from ekovideo_engine.pipeline import TranscriptionPipeline
from odoo_client import extract_company_name_from_pack


class ExtractCompanyNameFromPackTests(unittest.TestCase):
    def test_partner_id_many2one_returns_name(self):
        pack = {
            "primary": {
                "raw": {"partner_id": [42, "Caste"]},
                "display_name": "Lead #42",
            }
        }
        self.assertEqual(extract_company_name_from_pack(pack), "Caste")

    def test_strips_contact_name_keeps_company_after_separator(self):
        # Odoo partner names sometimes carry the contact in front:
        # "Jean Dupont, Caste" or "Jean Dupont (Caste)".
        pack_comma = {
            "primary": {
                "raw": {"partner_id": [1, "Jean Dupont, Caste"]},
            }
        }
        self.assertEqual(
            extract_company_name_from_pack(pack_comma), "Caste"
        )
        pack_paren = {
            "primary": {
                "raw": {"partner_id": [1, "Jean Dupont (Caste)"]},
            }
        }
        self.assertEqual(
            extract_company_name_from_pack(pack_paren), "Caste"
        )

    def test_falls_back_to_display_name_when_no_partner(self):
        pack = {
            "primary": {
                "raw": {},
                "display_name": "Projet — Site web CVR",
            }
        }
        self.assertEqual(
            extract_company_name_from_pack(pack),
            "Projet — Site web CVR",
        )

    def test_empty_pack_returns_empty(self):
        self.assertEqual(extract_company_name_from_pack(None), "")
        self.assertEqual(extract_company_name_from_pack({}), "")
        self.assertEqual(
            extract_company_name_from_pack({"primary": {}}), ""
        )


def _make_pipeline(
    *,
    odoo_pack: dict | None = None,
    meeting_metadata: dict | None = None,
    speaker_overrides: dict | None = None,
) -> TranscriptionPipeline:
    request = JobRequest(
        source_path="/tmp/x.wav",
        output_dir="/tmp/out",
        mode="transcribe",
        transcription_settings=TranscriptionSettings(),
        speaker_overrides=dict(speaker_overrides or {}),
        odoo_meeting_metadata=dict(meeting_metadata or {}),
    )
    pipeline = TranscriptionPipeline(request=request, sink=MagicMock())
    # Inject the pack directly so we don't need a live Odoo.
    pipeline._odoo_pack_cache = odoo_pack or {}
    return pipeline


class ResolveCompanyNameTests(unittest.TestCase):
    def test_picks_odoo_pack_first(self):
        pipeline = _make_pipeline(
            odoo_pack={
                "primary": {"raw": {"partner_id": [1, "Caste"]}},
            },
            meeting_metadata={"partners": [{"name": "OtherCompany"}]},
        )
        self.assertEqual(pipeline._resolve_company_name_for_title(), "Caste")

    def test_falls_back_to_meeting_metadata(self):
        pipeline = _make_pipeline(
            odoo_pack={},
            meeting_metadata={
                "partners": [
                    {"name": "Ekonum"},       # skipped (= us)
                    {"name": "CVR Contrôles"},  # winner
                ],
            },
        )
        self.assertEqual(
            pipeline._resolve_company_name_for_title(),
            "CVR Contrôles",
        )

    def test_falls_back_to_speaker_overrides_paren_pattern(self):
        pipeline = _make_pipeline(
            odoo_pack={},
            meeting_metadata={},
            speaker_overrides={"SPEAKER_00": "Manon (Caste)"},
        )
        self.assertEqual(
            pipeline._resolve_company_name_for_title(), "Caste"
        )

    def test_empty_when_no_source_available(self):
        pipeline = _make_pipeline()
        self.assertEqual(pipeline._resolve_company_name_for_title(), "")


class ApplyTitleCompanyPrefixTests(unittest.TestCase):
    def test_prepends_when_missing(self):
        pipeline = _make_pipeline(
            odoo_pack={"primary": {"raw": {"partner_id": [1, "Caste"]}}},
        )
        out = pipeline._apply_title_company_prefix(
            "Discussion sur l'intégration Odoo"
        )
        self.assertEqual(out, "Caste - Discussion sur l'intégration Odoo")

    def test_idempotent_on_rerun(self):
        # Title already has the company prefix → no double up.
        pipeline = _make_pipeline(
            odoo_pack={"primary": {"raw": {"partner_id": [1, "Caste"]}}},
        )
        out = pipeline._apply_title_company_prefix(
            "Caste - Audit système ERP"
        )
        self.assertEqual(out, "Caste - Audit système ERP")

    def test_case_insensitive_prefix_detection(self):
        pipeline = _make_pipeline(
            odoo_pack={"primary": {"raw": {"partner_id": [1, "Caste"]}}},
        )
        # LLM produced "CASTE - …", we shouldn't double up.
        out = pipeline._apply_title_company_prefix(
            "CASTE - Audit ERP"
        )
        self.assertEqual(out, "CASTE - Audit ERP")

    def test_defers_to_llm_when_it_already_used_a_separator(self):
        # The LLM produced "Acme - Topic" but our pack says "Caste".
        # Don't double up — the LLM might be correct and we don't
        # want "Caste - Acme - Topic".
        pipeline = _make_pipeline(
            odoo_pack={"primary": {"raw": {"partner_id": [1, "Caste"]}}},
        )
        out = pipeline._apply_title_company_prefix(
            "Acme - Discussion technique"
        )
        self.assertEqual(out, "Acme - Discussion technique")

    def test_no_op_when_company_unresolved(self):
        pipeline = _make_pipeline()
        out = pipeline._apply_title_company_prefix(
            "Sujet sans contexte"
        )
        self.assertEqual(out, "Sujet sans contexte")

    def test_empty_title_returned_as_is(self):
        pipeline = _make_pipeline(
            odoo_pack={"primary": {"raw": {"partner_id": [1, "Caste"]}}},
        )
        self.assertEqual(pipeline._apply_title_company_prefix(""), "")
        self.assertEqual(pipeline._apply_title_company_prefix("   "), "   ")


class StripEkonumPrefixTests(unittest.TestCase):
    """PR AM — veto code-side pour ``Ekonum -`` prefix produit
    par Mistral malgré le ban de PR AL."""

    def test_strips_canonical_ekonum_dash(self):
        out = TranscriptionPipeline._strip_ekonum_prefix(
            "Ekonum - Audit système ERP et reporting"
        )
        self.assertEqual(out, "Audit système ERP et reporting")

    def test_strips_lowercase_variant(self):
        out = TranscriptionPipeline._strip_ekonum_prefix(
            "ekonum - Migration vers Odoo"
        )
        self.assertEqual(out, "Migration vers Odoo")

    def test_strips_uppercase_variant(self):
        out = TranscriptionPipeline._strip_ekonum_prefix(
            "EKONUM - Discussion sur Odoo"
        )
        self.assertEqual(out, "Discussion sur Odoo")

    def test_strips_typo_variants(self):
        # Mistral has produced these in the wild.
        for typo in ("Econum", "Ekonom", "Ekonun", "Ekanum"):
            out = TranscriptionPipeline._strip_ekonum_prefix(
                f"{typo} - Audit ERP"
            )
            self.assertEqual(out, "Audit ERP", f"failed for {typo}")

    def test_strips_accented_variant(self):
        out = TranscriptionPipeline._strip_ekonum_prefix(
            "Ékonum - Audit ERP"
        )
        self.assertEqual(out, "Audit ERP")

    def test_strips_em_dash_separator(self):
        out = TranscriptionPipeline._strip_ekonum_prefix(
            "Ekonum — Audit ERP"
        )
        self.assertEqual(out, "Audit ERP")

    def test_strips_colon_separator(self):
        out = TranscriptionPipeline._strip_ekonum_prefix(
            "Ekonum : Audit ERP"
        )
        self.assertEqual(out, "Audit ERP")

    def test_does_not_strip_legitimate_company_prefix(self):
        # Companies that contain "ekon" but aren't Ekonum stay.
        out = TranscriptionPipeline._strip_ekonum_prefix(
            "Ekonum-Suite - Audit ERP"  # hypothetical product name
        )
        # The folded comparison is "ekonum-suite" which is NOT in
        # the variants set → no strip.
        self.assertEqual(out, "Ekonum-Suite - Audit ERP")

    def test_does_not_strip_when_no_separator(self):
        # "Ekonum a fait un audit" has no " - " in the first 30
        # chars → keep the title intact.
        out = TranscriptionPipeline._strip_ekonum_prefix(
            "Ekonum a fait un audit chez Caste"
        )
        self.assertEqual(out, "Ekonum a fait un audit chez Caste")

    def test_does_not_strip_when_remainder_is_empty(self):
        # ``"Ekonum - "`` (no real topic after) — better keep the
        # original than produce an empty title.
        out = TranscriptionPipeline._strip_ekonum_prefix("Ekonum - ")
        self.assertEqual(out, "Ekonum - ")

    def test_does_not_strip_legitimate_title_starting_with_other_company(self):
        out = TranscriptionPipeline._strip_ekonum_prefix(
            "Caste - Audit système ERP"
        )
        self.assertEqual(out, "Caste - Audit système ERP")

    def test_separator_past_30_chars_not_a_prefix(self):
        # A title like "Ekonum a longuement travaillé - Audit ERP"
        # has the separator past position 30 — not a prefix, leave
        # it alone.
        out = TranscriptionPipeline._strip_ekonum_prefix(
            "Ekonum a longuement travaillé pour le client - Audit ERP"
        )
        self.assertTrue(out.startswith("Ekonum a longuement"))


class ApplyTitleEkonumPipelineTests(unittest.TestCase):
    """End-to-end : the Ekonum-strip runs INSIDE
    ``_apply_title_company_prefix`` before the legacy company
    logic, so a Mistral-emitted ``"Ekonum - Sujet"`` becomes
    ``"Caste - Sujet"`` when the Odoo pack carries Caste."""

    def test_ekonum_prefix_replaced_with_resolved_company(self):
        # The CVR/Caste regression scenario : LLM produced
        # "Ekonum - Audit ERP", Odoo pack has Caste.
        pipeline = _make_pipeline(
            odoo_pack={"primary": {"raw": {"partner_id": [1, "Caste"]}}},
        )
        out = pipeline._apply_title_company_prefix(
            "Ekonum - Audit système ERP et reporting"
        )
        self.assertEqual(
            out, "Caste - Audit système ERP et reporting"
        )

    def test_ekonum_prefix_stripped_when_no_company_resolved(self):
        # Same Mistral output but no Odoo / meeting / overrides
        # → at least we strip the Ekonum self-reference, even if
        # we can't replace it with a real company.
        pipeline = _make_pipeline()
        out = pipeline._apply_title_company_prefix(
            "Ekonum - Audit système ERP"
        )
        self.assertEqual(out, "Audit système ERP")


if __name__ == "__main__":
    unittest.main()
