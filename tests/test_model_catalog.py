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
    def test_catalog_carries_all_seven_roles(self):
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
                "cloud_transcription",
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

    def test_size_mb_is_set_on_every_local_row(self):
        # Size powers the download-confirmation dialog. A 0 here
        # would render "—" in the UI and look weirdly opaque. Cloud
        # rows have nothing on disk — they carry prices instead, in
        # whichever billing model the provider uses.
        for row in model_catalog():
            if row.get("kind") == "cloud":
                if row["billing"] == "per_hour":
                    self.assertGreater(row["price_per_hour"], 0, row["id"])
                else:
                    self.assertGreater(row["price_in_per_1m"], 0, row["id"])
                    self.assertGreater(row["price_out_per_1m"], 0, row["id"])
            else:
                self.assertGreater(row["size_mb"], 0, row["id"])

    def test_cloud_rows_have_one_default_and_no_download_surface(self):
        rows = [r for r in model_catalog() if r["role"] == "cloud_transcription"]
        self.assertGreaterEqual(len(rows), 2)
        defaults = [r for r in rows if r["default"]]
        self.assertEqual(len(defaults), 1)
        for row in rows:
            self.assertEqual(row["kind"], "cloud")
            self.assertTrue(row["cached"], row["id"])
            self.assertEqual(row["cache_dir"], "", row["id"])
            self.assertFalse(row["gated"], row["id"])
            self.assertIn(row["billing"], {"per_token", "per_hour"}, row["id"])

    def test_cloud_catalogue_spans_all_providers(self):
        # The Run Setup / Settings rely on every wired provider being
        # represented by at least one selectable model.
        from cloud_transcription import CLOUD_PROVIDERS

        providers = {
            r["provider"]
            for r in model_catalog()
            if r["role"] == "cloud_transcription"
        }
        self.assertEqual(providers, set(CLOUD_PROVIDERS))

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

    def test_audio_llm_availability(self):
        # PR AW — Gemma 4 E4B is the live audio model, selectable in
        # the Models tab (the recheck pass exists since PR F and the
        # gemma4 submodule ships in mlx-vlm >= 0.6). The legacy
        # Qwen2-Audio row is gone entirely: its id canonicalises to
        # the Gemma checkpoint, so keeping it would render a
        # duplicate (id, role) row.
        audio = [row for row in model_catalog() if row["role"] == "audio_llm"]
        self.assertEqual(len(audio), 1)
        row = audio[0]
        self.assertEqual(row["id"], "mlx-community/gemma-4-12B-it-4bit")
        # PR AW — "À venir" while upstream mlx-vlm Gemma 4 audio
        # generation is broken (see _AUDIO_RECHECK_UPSTREAM_BLOCK).
        self.assertFalse(row["available"])
        self.assertTrue(row["default"])

    def test_other_roles_stay_available_by_default(self):
        for row in model_catalog():
            if row["role"] == "audio_llm":
                continue  # gated by the PR AW upstream block
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
