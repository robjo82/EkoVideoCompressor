from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ekovideo_engine.library import (
    _discover_speakers_from_text,
    database,
    library_discover_speakers,
    library_rename_speakers,
    library_speaker_samples,
)


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

    def test_speaker_samples_create_one_clip_per_speaker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "work"
            workspace.mkdir()
            audio = workspace / "audio.wav"
            audio.write_bytes(b"fake wav")

            with patch.dict(os.environ, {"EKO_APP_SUPPORT_DIR": str(root / "support")}):
                db = database()
                job_id = db.create_job(
                    source_path=str(root / "source.mov"),
                    workspace_dir=str(workspace),
                    settings={},
                )
                db.add_segments(
                    job_id,
                    [
                        {"start": 1.0, "end": 4.0, "speaker": "SPEAKER_00", "text": "Bonjour."},
                        {"start": 5.0, "end": 9.0, "speaker": "SPEAKER_01", "text": "Salut."},
                    ],
                )

                def fake_run(cmd, *args, **kwargs):
                    Path(cmd[-1]).write_bytes(b"sample")
                    return subprocess.CompletedProcess(cmd, 0, "", "")

                with patch("ekovideo_engine.library.subprocess.run", side_effect=fake_run):
                    samples = library_speaker_samples(job_id, seconds=3)

            self.assertEqual([sample["speaker"] for sample in samples], ["SPEAKER_00", "SPEAKER_01"])
            self.assertTrue(all(Path(sample["path"]).exists() for sample in samples))


class DiscoverSpeakersTest(unittest.TestCase):
    def test_extracts_bracketed_labels_in_first_seen_order(self):
        text = (
            "[SPEAKER_00] (00:00:01) Bonjour.\n"
            "[SPEAKER_01] (00:00:03) Salut.\n"
            "[SPEAKER_00] (00:00:05) Comment ça va ?\n"
            "[Robin] (00:00:08) Très bien.\n"
        )
        out = _discover_speakers_from_text(text)
        self.assertEqual(out, ["SPEAKER_00", "SPEAKER_01", "Robin"])

    def test_ignores_bracketed_metadata_lines(self):
        # "[note: ...]" mid-sentence isn't a speaker. The regex is
        # anchored on a line start so this never matches.
        text = "Cette phrase contient une [note: fragile] mais pas de locuteur."
        self.assertEqual(_discover_speakers_from_text(text), [])

    def test_backfills_empty_speaker_map_from_transcript_file(self):
        # Reproduce a job that ran on the legacy pipeline: the
        # transcript file is on disk, but the DB has no segments
        # and an empty speaker_map_json. The discover helper has to
        # rebuild the map from the file alone so the rename sheet
        # stops showing "Aucun interlocuteur détecté".
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            transcript = root / "transcript.txt"
            transcript.write_text(
                "[SPEAKER_00] (00:00:01) Bonjour.\n"
                "[SPEAKER_01] (00:00:05) Salut.\n"
                "[Robin] (00:00:09) Test.\n",
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"EKO_APP_SUPPORT_DIR": str(root / "support")}):
                db = database()
                job_id = db.create_job(
                    source_path=str(root / "source.mov"),
                    workspace_dir=str(root / "work"),
                    settings={},
                )
                db.update_job_artefact(job_id, "transcript", str(transcript))

                speakers = library_discover_speakers(job_id)
                row = db.get_job(job_id)

            # Placeholders get empty values (so the sheet renders an
            # editable field), real names get themselves.
            self.assertEqual(speakers["SPEAKER_00"], "")
            self.assertEqual(speakers["SPEAKER_01"], "")
            self.assertEqual(speakers["Robin"], "Robin")
            # Persisted to the DB so the next refresh sees it without
            # rerunning the discovery.
            self.assertIn("SPEAKER_00", row["speaker_map_json"])

    def test_preserves_existing_friendly_names_when_backfilling(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            transcript = root / "transcript.txt"
            # File already has the friendly name; the DB also has
            # the renaming pair. Backfill must not blow that away.
            transcript.write_text(
                "[Robin] (00:00:01) Salut.\n[SPEAKER_01] (00:00:05) Bonjour.\n",
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"EKO_APP_SUPPORT_DIR": str(root / "support")}):
                db = database()
                job_id = db.create_job(
                    source_path=str(root / "source.mov"),
                    workspace_dir=str(root / "work"),
                    settings={},
                )
                db.update_job_artefact(job_id, "transcript", str(transcript))
                db.update_job_context(job_id, speakers={"SPEAKER_00": "Robin"})

                merged = library_discover_speakers(job_id)

            # The pre-existing pair stays; the freshly-discovered
            # placeholder joins it.
            self.assertEqual(merged["SPEAKER_00"], "Robin")
            self.assertEqual(merged["SPEAKER_01"], "")
            # The friendly name was visible in the transcript so it
            # also gets a row, mapped to itself.
            self.assertEqual(merged["Robin"], "Robin")


if __name__ == "__main__":
    unittest.main()
