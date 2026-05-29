"""
PR AQ — merge two voice profiles into one.

The user saw duplicate "Mathilde Gérard" rows (differing by accent
/ Odoo link) and wanted a way to fuse them. ``merge_centroids``
does the weighted-average maths; ``library_merge_speaker_profiles``
folds the absorbed profile into the survivor (embedding + count +
Odoo link), then deletes the absorbed row.
"""

from __future__ import annotations

import math
import unittest
from unittest.mock import MagicMock, patch

from speaker_recognition import merge_centroids, encode_embedding
from ekovideo_engine.library import library_merge_speaker_profiles


class MergeCentroidsTests(unittest.TestCase):
    def test_weighted_average_by_count(self):
        # Two unit vectors on orthogonal axes, counts 3 and 1.
        # Weighted mean leans toward A (3×) before re-normalisation.
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        merged, count = merge_centroids(a, 3, b, 1)
        self.assertEqual(count, 4)
        # Result is normalised (unit length).
        norm = math.sqrt(sum(x * x for x in merged))
        self.assertAlmostEqual(norm, 1.0, places=5)
        # A had 3× the weight → first component dominates.
        self.assertGreater(merged[0], merged[1])

    def test_empty_side_returns_other(self):
        merged, count = merge_centroids([], 0, [0.0, 1.0], 2)
        self.assertEqual(merged, [0.0, 1.0])
        self.assertEqual(count, 2)

    def test_both_empty_returns_empty(self):
        merged, count = merge_centroids([], 0, [], 0)
        self.assertEqual(merged, [])
        self.assertEqual(count, 0)

    def test_shape_mismatch_keeps_better_sampled(self):
        a = [1.0, 0.0, 0.0]  # count 5
        b = [0.0, 1.0]       # count 2, wrong dim
        merged, count = merge_centroids(a, 5, b, 2)
        self.assertEqual(merged, a)
        self.assertEqual(count, 7)


def _profile(pid, name, vec, count, partner_id=None, partner_name=""):
    return {
        "id": pid,
        "name": name,
        "embedding_json": encode_embedding(vec) if vec else "[]",
        "sample_count": count,
        "odoo_partner_id": partner_id,
        "odoo_partner_name": partner_name,
        "odoo_company_id": None,
        "odoo_company_name": "",
    }


class LibraryMergeSpeakerProfilesTests(unittest.TestCase):
    def _db_with(self, survivor, absorbed):
        db = MagicMock()
        def get(pid):
            if survivor and pid == survivor["id"]:
                return survivor
            if absorbed and pid == absorbed["id"]:
                return absorbed
            return None
        db.get_speaker_profile.side_effect = get
        return db

    def test_refuses_same_profile(self):
        db = MagicMock()
        with patch("ekovideo_engine.library.database", return_value=db):
            result = library_merge_speaker_profiles(5, 5)
        self.assertFalse(result["merged"])
        self.assertEqual(result["reason"], "same_profile")

    def test_refuses_missing_survivor(self):
        db = self._db_with(None, _profile(2, "B", [1.0, 0.0], 1))
        with patch("ekovideo_engine.library.database", return_value=db):
            result = library_merge_speaker_profiles(1, 2)
        self.assertFalse(result["merged"])
        self.assertEqual(result["reason"], "survivor_not_found")

    def test_merges_embedding_and_count_and_deletes_absorbed(self):
        survivor = _profile(1, "Mathilde Gérard", [1.0, 0.0], 3)
        absorbed = _profile(2, "Mathilde Gerard", [0.0, 1.0], 1)
        db = self._db_with(survivor, absorbed)
        with patch("ekovideo_engine.library.database", return_value=db):
            result = library_merge_speaker_profiles(1, 2)
        self.assertTrue(result["merged"])
        self.assertEqual(result["survivor_name"], "Mathilde Gérard")
        self.assertEqual(result["sample_count"], 4)
        # Survivor's embedding updated, absorbed deleted.
        db.update_speaker_profile_embedding.assert_called_once()
        update_args = db.update_speaker_profile_embedding.call_args
        self.assertEqual(update_args.args[0], 1)        # survivor id
        self.assertEqual(update_args.args[2], 4)        # merged count
        db.delete_speaker_profile.assert_called_once_with(2)

    def test_survivor_inherits_odoo_when_unlinked(self):
        # Survivor has no Odoo link, absorbed does → inherit it (no
        # conflict, don't drop the only association).
        survivor = _profile(1, "Mathilde", [1.0, 0.0], 2)
        absorbed = _profile(2, "Mathilde G", [0.0, 1.0], 1,
                            partner_id=42, partner_name="Mathilde Gérard")
        db = self._db_with(survivor, absorbed)
        with patch("ekovideo_engine.library.database", return_value=db):
            library_merge_speaker_profiles(1, 2, odoo_from="survivor")
        db.link_speaker_profile_to_odoo.assert_called_once()
        self.assertEqual(
            db.link_speaker_profile_to_odoo.call_args.kwargs["partner_id"], 42
        )

    def test_odoo_from_absorbed_relinks_survivor(self):
        # Conflict resolved by the user picking the absorbed contact.
        survivor = _profile(1, "Mathilde", [1.0, 0.0], 2,
                            partner_id=10, partner_name="Old")
        absorbed = _profile(2, "Mathilde G", [0.0, 1.0], 1,
                            partner_id=42, partner_name="Mathilde Gérard")
        db = self._db_with(survivor, absorbed)
        with patch("ekovideo_engine.library.database", return_value=db):
            library_merge_speaker_profiles(1, 2, odoo_from="absorbed")
        db.link_speaker_profile_to_odoo.assert_called_once()
        self.assertEqual(
            db.link_speaker_profile_to_odoo.call_args.kwargs["partner_id"], 42
        )

    def test_odoo_from_survivor_keeps_existing_link_no_relink(self):
        # Both linked, keep survivor's → no relink call (it already
        # points where we want).
        survivor = _profile(1, "Mathilde", [1.0, 0.0], 2,
                            partner_id=10, partner_name="Keep")
        absorbed = _profile(2, "Mathilde G", [0.0, 1.0], 1,
                            partner_id=42, partner_name="Drop")
        db = self._db_with(survivor, absorbed)
        with patch("ekovideo_engine.library.database", return_value=db):
            library_merge_speaker_profiles(1, 2, odoo_from="survivor")
        db.link_speaker_profile_to_odoo.assert_not_called()

    def test_merge_shell_into_real_keeps_real_embedding(self):
        # Absorbing a shell (count 0, empty embedding) keeps the
        # real centroid, count unchanged.
        survivor = _profile(1, "Mathilde", [1.0, 0.0], 5)
        absorbed = _profile(2, "Mathilde", None, 0)
        db = self._db_with(survivor, absorbed)
        with patch("ekovideo_engine.library.database", return_value=db):
            result = library_merge_speaker_profiles(1, 2)
        self.assertTrue(result["merged"])
        self.assertEqual(result["sample_count"], 5)


if __name__ == "__main__":
    unittest.main()
