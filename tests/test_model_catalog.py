"""Tests for the role-tagged model catalog.

Pins the contract the SwiftUI Models tab depends on: every entry
carries a role, a tier, a size, a default flag, and the gated bit
on pyannote rows. Also pins the new ``canonical_multipass_model_id``
fallback so the engine never breaks on an empty multipass setting.
"""

from __future__ import annotations

import unittest

from ekovideo_engine.model_cache import model_catalog
from transcription_utils import (
    DEFAULT_MULTIPASS_MODEL,
    DIARISATION_MODELS,
    MULTIPASS_MODELS,
    WHISPER_MODELS,
    canonical_multipass_model_id,
)


class CatalogRoleCoverageTest(unittest.TestCase):
    def test_catalog_carries_all_six_roles(self):
        rows = model_catalog()
        roles = {row["role"] for row in rows}
        self.assertEqual(
            roles,
            {
                "transcription",
                "multipass",
                "text_llm",
                "audio_llm",
                "diarisation",
                "embedding",
            },
        )

    def test_one_default_per_user_selectable_role(self):
        # The SwiftUI ``activeID`` resolution falls back on the
        # default flag, so each user-selectable role needs exactly
        # one entry marked ``default``.
        rows = model_catalog()
        for role in ("transcription", "multipass", "text_llm", "audio_llm"):
            defaults = [r for r in rows if r["role"] == role and r["default"]]
            self.assertEqual(
                len(defaults),
                1,
                f"role={role!r} has {len(defaults)} defaults",
            )

    def test_pyannote_rows_are_gated_and_under_diarisation_role(self):
        rows = model_catalog()
        pyannote = [r for r in rows if r["family"] == "Pyannote"]
        self.assertGreaterEqual(len(pyannote), 3)
        for row in pyannote:
            self.assertTrue(row["gated"], f"{row['id']} should be gated")
            self.assertIn(row["role"], {"diarisation", "embedding"})

    def test_french_distil_whisper_is_listed(self):
        rows = model_catalog()
        french = [r for r in rows if "fr" in r["language"] and "fr" == r["language"][0]]
        self.assertTrue(
            any("bofenghuang" in r["id"] for r in french),
            "Expected the bofenghuang French distil checkpoint to be surfaced",
        )

    def test_size_mb_is_set_on_every_row(self):
        # Size powers the download-confirmation dialog. A 0 here
        # would render "—" in the UI and look weirdly opaque.
        for row in model_catalog():
            self.assertGreater(row["size_mb"], 0, row["id"])

    def test_tiers_are_constrained_to_known_values(self):
        allowed = {"light", "balanced", "heavy"}
        for row in model_catalog():
            self.assertIn(row["tier"], allowed, row["id"])

    def test_available_flag_is_set_on_every_row(self):
        # Drives the "À venir" badge in the SwiftUI Models tab.
        # Every row carries the field — entries that haven't been
        # wired in the new engine (audio_llm today) flip it to
        # False so the UI doesn't promise a working toggle.
        for row in model_catalog():
            self.assertIn("available", row, row["id"])
            self.assertIsInstance(row["available"], bool, row["id"])

    def test_audio_llm_is_marked_unavailable(self):
        # The multimodal recheck pass lives in video_compactor.py
        # (legacy PySide path) but isn't ported to
        # ekovideo_engine.pipeline. Pin the explicit "not yet"
        # signal so SwiftUI keeps surfacing the warning until the
        # engine wires it.
        audio = [row for row in model_catalog() if row["role"] == "audio_llm"]
        self.assertTrue(audio, "expected at least one audio_llm row")
        for row in audio:
            self.assertFalse(row["available"], row["id"])

    def test_other_roles_stay_available_by_default(self):
        for row in model_catalog():
            if row["role"] == "audio_llm":
                continue
            self.assertTrue(row["available"], row["id"])


class CanonicalMultipassTest(unittest.TestCase):
    def test_blank_falls_back_to_catalog_default(self):
        self.assertEqual(canonical_multipass_model_id(""), DEFAULT_MULTIPASS_MODEL)
        self.assertEqual(canonical_multipass_model_id("   "), DEFAULT_MULTIPASS_MODEL)

    def test_legacy_repo_id_gets_rewritten(self):
        # ``LEGACY_WHISPER_MODEL_IDS`` (shared with the transcription
        # canonicaliser) covers the old short id ``whisper-large-v3``
        # → ``whisper-large-v3-mlx``. Pin the rewrite so a config
        # written before the rename still loads.
        self.assertEqual(
            canonical_multipass_model_id("mlx-community/whisper-large-v3"),
            "mlx-community/whisper-large-v3-mlx",
        )

    def test_unknown_id_passes_through(self):
        self.assertEqual(
            canonical_multipass_model_id("acme/whisper-fr-v0.1"),
            "acme/whisper-fr-v0.1",
        )


if __name__ == "__main__":
    unittest.main()
