"""
PR X — bulk reset of stored voice profiles.

Covers ``library_reset_speaker_profiles`` :
  • empty library → returns ``{"removed": 0}``
  • populated library → deletes every row, returns the count
  • partial-failure tolerance: a row with a bogus id doesn't
    abort the rest of the purge
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from ekovideo_engine.library import library_reset_speaker_profiles


class LibraryResetSpeakerProfilesTests(unittest.TestCase):
    def _mock_db(self, profile_rows: list[dict]) -> MagicMock:
        db = MagicMock()
        db.list_speaker_profiles.return_value = profile_rows
        return db

    def test_empty_library_returns_zero(self):
        db = self._mock_db([])
        with patch(
            "ekovideo_engine.library.database",
            return_value=db,
        ):
            result = library_reset_speaker_profiles()
        self.assertEqual(result, {"removed": 0})
        db.delete_speaker_profile.assert_not_called()

    def test_populated_library_deletes_every_row(self):
        rows = [
            {"id": 1, "name": "Robin"},
            {"id": 2, "name": "Vincent"},
            {"id": 3, "name": "Clothilde"},
        ]
        db = self._mock_db(rows)
        with patch(
            "ekovideo_engine.library.database",
            return_value=db,
        ):
            result = library_reset_speaker_profiles()
        self.assertEqual(result, {"removed": 3})
        # Every row's id was passed to delete_speaker_profile.
        ids_called = [
            call.args[0] for call in db.delete_speaker_profile.call_args_list
        ]
        self.assertEqual(sorted(ids_called), [1, 2, 3])

    def test_skips_bogus_ids(self):
        # A row with a missing / non-int id is silently skipped.
        # The other rows still get deleted.
        rows = [
            {"id": 1, "name": "Robin"},
            {"id": None, "name": "Mystery"},
            {"id": "abc", "name": "Junk"},
            {"id": 5, "name": "Vincent"},
        ]
        db = self._mock_db(rows)
        with patch(
            "ekovideo_engine.library.database",
            return_value=db,
        ):
            result = library_reset_speaker_profiles()
        self.assertEqual(result, {"removed": 2})
        ids_called = sorted(
            call.args[0] for call in db.delete_speaker_profile.call_args_list
        )
        self.assertEqual(ids_called, [1, 5])

    def test_skips_zero_or_negative_ids(self):
        # Defensive: SQLite autoincrement always yields positive ids,
        # but a manually-crafted JSON could carry 0 or -1.
        rows = [
            {"id": 0, "name": "Bogus"},
            {"id": -5, "name": "Negative"},
            {"id": 7, "name": "Good"},
        ]
        db = self._mock_db(rows)
        with patch(
            "ekovideo_engine.library.database",
            return_value=db,
        ):
            result = library_reset_speaker_profiles()
        self.assertEqual(result, {"removed": 1})


if __name__ == "__main__":
    unittest.main()
