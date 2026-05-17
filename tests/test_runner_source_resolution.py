"""Tests for the runner's smart source resolution.

The user reported a batch where every file failed because the
runner refused to start when ``source_path`` was missing — even
though the workspace folder still held the canonical copy from
the previous run. These tests pin the recovery path so the
regression can't slip back in.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ekovideo_engine.library import database
from ekovideo_engine.models import JobRequest
from ekovideo_engine.runner import (
    _auto_rename_job_from_transcript,
    _resolve_source_path,
)


def _make_request(
    *,
    source_path: str,
    workspace_dir: str = "",
    library_job_id: int | None = None,
) -> JobRequest:
    return JobRequest.from_dict(
        {
            "source_path": source_path,
            "output_dir": "/tmp/eko-tests",
            "mode": "transcribe",
            "workspace_dir": workspace_dir,
            "library_job_id": library_job_id,
        }
    )


class ResolveSourcePathTest(unittest.TestCase):
    def test_returns_source_when_it_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "meeting.mp4"
            src.write_bytes(b"fake")
            request = _make_request(source_path=str(src))
            self.assertEqual(_resolve_source_path(request), src)

    def test_falls_back_to_workspace_copy_when_source_moved(self):
        # The classic re-run case: original was deleted via
        # delete_source_after_copy, but the workspace still has
        # the copy.
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "ws"
            workspace.mkdir()
            (workspace / "meeting.mp4").write_bytes(b"fake")
            request = _make_request(
                source_path="/nowhere/meeting.mp4",
                workspace_dir=str(workspace),
            )
            self.assertEqual(
                _resolve_source_path(request), workspace / "meeting.mp4"
            )

    def test_falls_back_to_stored_workspace_when_request_lacks_one(self):
        # Legacy SwiftUI caller that re-submits a job_id without
        # forwarding workspace_dir. The runner picks the path off
        # the DB row.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "ws"
            workspace.mkdir()
            (workspace / "meeting.mp4").write_bytes(b"fake")
            with patch.dict(
                os.environ, {"EKO_APP_SUPPORT_DIR": str(root / "support")}
            ):
                db = database()
                job_id = db.create_job(
                    source_path="/nowhere/meeting.mp4",
                    workspace_dir=str(workspace),
                    settings={},
                )
                request = _make_request(
                    source_path="/nowhere/meeting.mp4",
                    library_job_id=job_id,
                )
                resolved = _resolve_source_path(request)
            self.assertEqual(resolved, workspace / "meeting.mp4")

    def test_returns_none_when_nothing_usable_exists(self):
        # Triggers the ``source_missing`` event the SwiftUI side
        # listens for to pop the relocalisation sheet.
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "ws"
            workspace.mkdir()
            # Workspace exists but has no copy of the source file.
            request = _make_request(
                source_path="/nowhere/meeting.mp4",
                workspace_dir=str(workspace),
            )
            self.assertIsNone(_resolve_source_path(request))

    def test_blank_source_path_returns_none(self):
        # JobRequest.from_dict refuses a blank source, so we
        # build the dataclass directly to mimic a future caller
        # that bypasses validation.
        from ekovideo_engine.models import (
            CompressionSettings,
            JobRequest as RawJobRequest,
            TranscriptionSettings,
        )

        request = RawJobRequest(
            source_path="",
            output_dir="/tmp",
            mode="transcribe",
            compression_settings=CompressionSettings(),
            transcription_settings=TranscriptionSettings(),
        )
        self.assertIsNone(_resolve_source_path(request))


class AutoRenameFromTranscriptTest(unittest.TestCase):
    """Pin the post-transcription job-title promotion.

    The library used to display ``Enregistrement de l'écran 2026-...``
    for every screen recording because the engine never set
    ``custom_title``. Now we feed the transcript through
    ``suggest_transcript_stem`` once the run completes — these tests
    pin that we (a) actually call the suggester, (b) skip when the
    user already typed a title, and (c) bail silently rather than
    overwriting with the fallback when the transcript has nothing
    topical to offer.
    """

    def _topical_transcript(self) -> str:
        # Long enough + structured enough that suggest_transcript_stem
        # passes its >=12 chars + score>=10 gates without us hand-tuning
        # the topic-word dictionary.
        return (
            "[SPEAKER_00] Bonjour, ravi de vous voir.\n"
            "[SPEAKER_01] Présentation du nouveau module RH "
            "pour la formation des managers.\n"
            "[SPEAKER_00] On va parler du planning et du budget "
            "associé au projet.\n"
        )

    def test_promotes_topical_title_when_custom_title_is_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            transcript = root / "transcript.txt"
            transcript.write_text(self._topical_transcript(), encoding="utf-8")
            with patch.dict(
                os.environ, {"EKO_APP_SUPPORT_DIR": str(root / "support")}
            ):
                db = database()
                job_id = db.create_job(
                    source_path=str(root / "Enregistrement de l'écran.mov"),
                    workspace_dir=str(root / "ws"),
                    settings={},
                )
                title = _auto_rename_job_from_transcript(
                    db,
                    job_id,
                    str(transcript),
                    str(root / "Enregistrement de l'écran.mov"),
                )
                row = db.get_job(job_id)

            self.assertIsNotNone(title)
            self.assertEqual(row["custom_title"], title)
            self.assertNotEqual(title, "Enregistrement de l'écran")

    def test_respects_existing_custom_title(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            transcript = root / "transcript.txt"
            transcript.write_text(self._topical_transcript(), encoding="utf-8")
            with patch.dict(
                os.environ, {"EKO_APP_SUPPORT_DIR": str(root / "support")}
            ):
                db = database()
                job_id = db.create_job(
                    source_path=str(root / "x.mov"),
                    workspace_dir=str(root / "ws"),
                    settings={},
                )
                db.update_job_title(job_id, "Mon titre manuel")
                title = _auto_rename_job_from_transcript(
                    db, job_id, str(transcript), str(root / "x.mov")
                )
                row = db.get_job(job_id)

            self.assertIsNone(title)
            self.assertEqual(row["custom_title"], "Mon titre manuel")

    def test_bails_when_transcript_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.dict(
                os.environ, {"EKO_APP_SUPPORT_DIR": str(root / "support")}
            ):
                db = database()
                job_id = db.create_job(
                    source_path=str(root / "x.mov"),
                    workspace_dir=str(root / "ws"),
                    settings={},
                )
                title = _auto_rename_job_from_transcript(
                    db, job_id, str(root / "missing.txt"), str(root / "x.mov")
                )
                row = db.get_job(job_id)

            self.assertIsNone(title)
            self.assertFalse((row["custom_title"] or "").strip())


if __name__ == "__main__":
    unittest.main()
