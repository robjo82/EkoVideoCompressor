from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ekovideo_engine.library import (
    _discover_speakers_from_text,
    _looks_like_engine_workspace,
    database,
    library_delete,
    library_discover_speakers,
    library_rename_speakers,
    library_speaker_samples,
    library_workspace_usage,
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


class WorkspaceUsageTest(unittest.TestCase):
    def test_lists_files_sorted_by_size_with_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "20260514 - Test"
            workspace.mkdir(parents=True)
            (workspace / "audio.wav").write_bytes(b"\x00" * 5000)
            (workspace / "whisper.json").write_text("{}", encoding="utf-8")
            (workspace / "Présentation.txt").write_text(
                "[SPEAKER_00] Bonjour.\n", encoding="utf-8"
            )

            with patch.dict(os.environ, {"EKO_APP_SUPPORT_DIR": str(root / "support")}):
                db = database()
                job_id = db.create_job(
                    source_path=str(root / "source.mov"),
                    workspace_dir=str(workspace),
                    settings={},
                )
                usage = library_workspace_usage(job_id)

            self.assertEqual(usage["workspace_dir"], str(workspace))
            # Biggest file first so the deletion sheet surfaces the
            # heavy hitters at the top.
            sizes = [item["size"] for item in usage["files"]]
            self.assertEqual(sizes, sorted(sizes, reverse=True))
            # Heuristic labels turn obscure filenames into something
            # the user can recognise without opening Finder.
            audio = next(item for item in usage["files"] if item["name"] == "audio.wav")
            self.assertEqual(audio["label"], "Audio extrait")
            text = next(item for item in usage["files"] if item["name"].endswith(".txt"))
            self.assertEqual(text["label"], "Transcription")
            # Total bytes matches the sum of every listed file.
            self.assertEqual(usage["total_bytes"], sum(sizes))

    def test_usage_returns_empty_when_workspace_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            missing = root / "never-created"
            with patch.dict(os.environ, {"EKO_APP_SUPPORT_DIR": str(root / "support")}):
                db = database()
                job_id = db.create_job(
                    source_path=str(root / "source.mov"),
                    workspace_dir=str(missing),
                    settings={},
                )
                usage = library_workspace_usage(job_id)
            self.assertEqual(usage["files"], [])
            self.assertEqual(usage["total_bytes"], 0)


class DeleteWithFilesTest(unittest.TestCase):
    def _make_workspace(self, parent: Path) -> Path:
        ws = parent / "20260514 - Test"
        ws.mkdir(parents=True)
        # Use real markers so ``_looks_like_engine_workspace`` says yes.
        (ws / "audio.wav").write_bytes(b"\x00" * 200)
        (ws / "whisper.json").write_text("{}", encoding="utf-8")
        (ws / "Présentation.txt").write_text("hi", encoding="utf-8")
        return ws

    def test_remove_files_wipes_workspace_and_returns_byte_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = self._make_workspace(root)
            with patch.dict(os.environ, {"EKO_APP_SUPPORT_DIR": str(root / "support")}):
                db = database()
                job_id = db.create_job(
                    source_path=str(root / "source.mov"),
                    workspace_dir=str(workspace),
                    settings={},
                )
                summary = library_delete(job_id, remove_files=True)

            self.assertFalse(workspace.exists())
            self.assertTrue(summary["workspace_removed"])
            self.assertGreater(summary["files_removed"], 0)
            self.assertGreater(summary["bytes_removed"], 0)

    def test_default_delete_keeps_workspace_on_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = self._make_workspace(root)
            with patch.dict(os.environ, {"EKO_APP_SUPPORT_DIR": str(root / "support")}):
                db = database()
                job_id = db.create_job(
                    source_path=str(root / "source.mov"),
                    workspace_dir=str(workspace),
                    settings={},
                )
                summary = library_delete(job_id)  # no remove_files

            self.assertTrue(workspace.exists())
            self.assertFalse(summary["workspace_removed"])

    def test_refuses_to_remove_dir_that_does_not_look_like_workspace(self):
        # Critical safety net: if the DB points at a folder full of
        # the user's own content (no audio.wav, no whisper.json, etc.)
        # we must refuse to nuke it even when ``remove_files=True``.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            user_folder = root / "Documents"
            user_folder.mkdir()
            (user_folder / "important.docx").write_bytes(b"x" * 500)

            with patch.dict(os.environ, {"EKO_APP_SUPPORT_DIR": str(root / "support")}):
                db = database()
                job_id = db.create_job(
                    source_path=str(root / "source.mov"),
                    workspace_dir=str(user_folder),
                    settings={},
                )
                summary = library_delete(job_id, remove_files=True)

            self.assertTrue(user_folder.exists())
            self.assertTrue((user_folder / "important.docx").exists())
            self.assertFalse(summary["workspace_removed"])

    def test_workspace_heuristic_matches_review_or_enhanced_files(self):
        # A workspace that only carries the review markdown (because
        # the LLM didn't run successfully, but the review file did
        # land) is still ours and should be removable.
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp) / "20260514 - Test"
            ws.mkdir()
            (ws / "Présentation - à vérifier.md").write_text(
                "report", encoding="utf-8"
            )
            self.assertTrue(_looks_like_engine_workspace(ws))


if __name__ == "__main__":
    unittest.main()
