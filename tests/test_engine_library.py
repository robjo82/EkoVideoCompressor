from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ekovideo_engine.library import database, library_rename_speakers


class EngineLibraryActionsTest(unittest.TestCase):
    def test_rename_speakers_updates_segments_and_text_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            transcript = root / "transcript.txt"
            review = root / "review.md"
            transcript.write_text(
                "[SPEAKER_00] (00:00:01) Bonjour.\n[SPEAKER_01] (00:00:02) Salut.\n",
                encoding="utf-8",
            )
            review.write_text("- `SPEAKER_00` → **SPEAKER_00**\n", encoding="utf-8")

            with patch.dict(os.environ, {"EKO_APP_SUPPORT_DIR": str(root / "support")}):
                db = database()
                job_id = db.create_job(
                    source_path=str(root / "source.mov"),
                    workspace_dir=str(root / "work"),
                    settings={},
                )
                db.update_job_artefact(job_id, "transcript", str(transcript))
                db.update_job_artefact(job_id, "review", str(review))
                db.add_segments(
                    job_id,
                    [
                        {
                            "start": 1.0,
                            "end": 2.0,
                            "speaker": "SPEAKER_00",
                            "text": "Bonjour.",
                        },
                        {
                            "start": 2.0,
                            "end": 3.0,
                            "speaker": "SPEAKER_01",
                            "text": "Salut.",
                        },
                    ],
                )

                result = library_rename_speakers(job_id, {"SPEAKER_00": "Robin"})
                segments = db.get_segments(job_id)
                row = db.get_job(job_id)

            self.assertEqual(result["segments_changed"], 1)
            self.assertEqual(result["artifacts_rewritten"], 2)
            self.assertEqual(segments[0]["speaker"], "Robin")
            self.assertEqual(segments[0]["start_time"], 1.0)
            self.assertIn("[Robin]", transcript.read_text(encoding="utf-8"))
            self.assertIn("Robin", review.read_text(encoding="utf-8"))
            self.assertIn("Robin", row["speaker_map_json"])


if __name__ == "__main__":
    unittest.main()
