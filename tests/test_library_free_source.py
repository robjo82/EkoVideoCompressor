"""
PR AP — library_free_source : delete the heavy source file(s) to
reclaim disk, but ONLY when a compressed version exists.

Product rule (from the user) :
  - "Libérer la source" only available if the compressed version
    was prepared.
  - After freeing, the source is gone everywhere; relaunch
    transcribes the compressed file (handled SwiftUI-side).

These tests pin the engine precondition + safety guards :
  - refuse when no compressed_path / compressed missing
  - delete the workspace copy + original source
  - NEVER delete the compressed file
  - report bytes freed

NOTE: every file-existence assertion stays INSIDE the
``with tempfile.TemporaryDirectory()`` block — otherwise the
tempdir is cleaned up before the check and ``exists()`` is
trivially False (a footgun that bit the first draft).
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from ekovideo_engine.library import library_free_source


def _job(workspace: str, source_path: str, compressed_path: str) -> dict:
    return {
        "id": 7,
        "workspace_dir": workspace,
        "source_path": source_path,
        "compressed_path": compressed_path,
    }


def _run_with_job(job: dict) -> dict:
    db = MagicMock()
    db.get_job.return_value = job
    with patch("ekovideo_engine.library.database", return_value=db):
        return library_free_source(7)


class LibraryFreeSourceTests(unittest.TestCase):
    def test_refuses_when_no_compressed_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "meeting.mov"
            src.write_bytes(b"x" * 1000)
            result = _run_with_job(_job(tmp, str(src), ""))
            self.assertFalse(result["freed"])
            self.assertEqual(result["reason"], "no_compressed_version")
            self.assertTrue(src.exists())  # source untouched

    def test_refuses_when_compressed_missing_on_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "meeting.mov"
            src.write_bytes(b"x" * 1000)
            result = _run_with_job(
                _job(tmp, str(src), f"{tmp}/meeting_compressed.mp4")
            )
            self.assertFalse(result["freed"])
            self.assertEqual(result["reason"], "no_compressed_version")
            self.assertTrue(src.exists())

    def test_frees_workspace_copy_and_original(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp) / "ws"
            ws.mkdir()
            desktop = Path(tmp) / "Desktop"
            desktop.mkdir()
            original = desktop / "meeting.mov"
            original.write_bytes(b"x" * 5000)
            copy = ws / "meeting.mov"
            copy.write_bytes(b"x" * 5000)
            compressed = ws / "meeting_compressed.mp4"
            compressed.write_bytes(b"y" * 500)

            result = _run_with_job(
                _job(str(ws), str(original), str(compressed))
            )

            self.assertTrue(result["freed"])
            self.assertEqual(result["files_removed"], 2)
            self.assertEqual(result["bytes_removed"], 10000)
            self.assertFalse(original.exists())
            self.assertFalse(copy.exists())
            self.assertTrue(compressed.exists())  # survives

    def test_never_deletes_compressed_even_if_db_aliases_it(self):
        # Defensive : malformed row with source_path == compressed_path.
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp) / "ws"
            ws.mkdir()
            compressed = ws / "meeting_compressed.mp4"
            compressed.write_bytes(b"y" * 500)
            result = _run_with_job(
                _job(str(ws), str(compressed), str(compressed))
            )
            self.assertFalse(result["freed"])
            self.assertTrue(compressed.exists())

    def test_no_op_when_source_already_gone(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp) / "ws"
            ws.mkdir()
            compressed = ws / "meeting_compressed.mp4"
            compressed.write_bytes(b"y" * 500)
            result = _run_with_job(
                _job(str(ws), f"{tmp}/gone/meeting.mov", str(compressed))
            )
            self.assertFalse(result["freed"])
            self.assertEqual(result["reason"], "no_source_file_on_disk")
            self.assertTrue(compressed.exists())

    def test_job_not_found(self):
        db = MagicMock()
        db.get_job.return_value = None
        with patch("ekovideo_engine.library.database", return_value=db):
            result = library_free_source(999)
        self.assertFalse(result["freed"])
        self.assertEqual(result["reason"], "job_not_found")

    def test_recomputes_total_bytes_after_freeing(self):
        # PR AS — after freeing, the workspace weight snapshot must be
        # refreshed so the library "Poids" column stops showing the
        # stale pre-deletion size. Here the workspace keeps only the
        # 500-byte compressed file once the 5000-byte copy is gone.
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp) / "ws"
            ws.mkdir()
            desktop = Path(tmp) / "Desktop"
            desktop.mkdir()
            original = desktop / "meeting.mov"
            original.write_bytes(b"x" * 5000)
            copy = ws / "meeting.mov"
            copy.write_bytes(b"x" * 5000)
            compressed = ws / "meeting_compressed.mp4"
            compressed.write_bytes(b"y" * 500)

            db = MagicMock()
            db.get_job.return_value = _job(str(ws), str(original), str(compressed))
            with patch("ekovideo_engine.library.database", return_value=db):
                result = library_free_source(7)

            self.assertTrue(result["freed"])
            # Only the compressed file remains in the workspace.
            self.assertEqual(result["total_bytes"], 500)
            db.update_job_total_bytes.assert_called_once_with(7, 500)

    def test_no_total_bytes_recompute_when_refused(self):
        # When freeing is refused (no compressed), the weight snapshot
        # must NOT be touched.
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "meeting.mov"
            src.write_bytes(b"x" * 1000)
            db = MagicMock()
            db.get_job.return_value = _job(tmp, str(src), "")
            with patch("ekovideo_engine.library.database", return_value=db):
                result = library_free_source(7)
            self.assertFalse(result["freed"])
            self.assertNotIn("total_bytes", result)
            db.update_job_total_bytes.assert_not_called()


class RecomputeTotalBytesTests(unittest.TestCase):
    """PR AY — heal stale "Poids" snapshots.

    Sources freed on pre-PR-AS app versions kept total_bytes frozen
    at the pre-deletion size (job 26 showed 21,5 Go for a 950 Mo
    workspace). The one-shot heal re-walks every workspace.
    """

    def test_recomputes_stale_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp) / "ws"
            ws.mkdir()
            (ws / "compressed.mp4").write_bytes(b"y" * 500)
            db = MagicMock()
            db.list_jobs.return_value = [
                {"id": 7, "workspace_dir": str(ws), "total_bytes": 21_000_000_000},
            ]
            from ekovideo_engine.library import library_recompute_total_bytes
            with patch("ekovideo_engine.library.database", return_value=db):
                result = library_recompute_total_bytes()
            self.assertEqual(result["updated"], 1)
            db.update_job_total_bytes.assert_called_once_with(7, 500)

    def test_skips_missing_workspace_and_current_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp) / "ws"
            ws.mkdir()
            (ws / "a.bin").write_bytes(b"x" * 100)
            db = MagicMock()
            db.list_jobs.return_value = [
                {"id": 1, "workspace_dir": str(ws), "total_bytes": 100},  # already right
                {"id": 2, "workspace_dir": f"{tmp}/gone", "total_bytes": 5},  # missing dir
                {"id": 3, "workspace_dir": "", "total_bytes": 5},  # no workspace
            ]
            from ekovideo_engine.library import library_recompute_total_bytes
            with patch("ekovideo_engine.library.database", return_value=db):
                result = library_recompute_total_bytes()
            self.assertEqual(result["updated"], 0)
            self.assertEqual(result["skipped"], 2)
            db.update_job_total_bytes.assert_not_called()


if __name__ == "__main__":
    unittest.main()
