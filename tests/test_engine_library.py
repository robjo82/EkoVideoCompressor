from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ekovideo_engine.library import (
    _clip_window,
    _discover_speakers_from_text,
    _looks_like_engine_workspace,
    _segments_per_cluster,
    database,
    library_delete,
    library_delete_speaker_profile,
    library_detach_odoo_meeting,
    library_discover_speakers,
    library_enroll_speakers_for_job,
    library_flag_speaker_sample_review,
    library_get,
    library_link_speaker_profile_to_odoo,
    library_list_speaker_profiles,
    library_recognize_speakers,
    library_remember_speaker_names,
    library_rename_speakers,
    library_repair_all_speaker_maps,
    library_speaker_samples,
    library_unlink_speaker_profile_from_odoo,
    library_workspace_usage,
)
from speaker_recognition import decode_embedding, encode_embedding


class EngineLibraryActionsTest(unittest.TestCase):
    def test_create_job_redacts_sensitive_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.dict(os.environ, {"EKO_APP_SUPPORT_DIR": str(root / "support")}):
                db = database()
                job_id = db.create_job(
                    source_path=str(root / "source.mov"),
                    workspace_dir=str(root / "work"),
                    settings={
                        "transcription_settings": {
                            "hf_token": "hf-secret",
                            "venv_python_path": "/usr/bin/python3",
                        },
                        "odoo_context_ref": {
                            "api_key": "odoo-secret",
                            "database": "prod",
                        },
                    },
                )
                row = db.get_job(job_id)

            stored = json.loads(row["settings_json"])
            self.assertEqual(
                stored["transcription_settings"]["hf_token"],
                "[redacted]",
            )
            self.assertEqual(
                stored["transcription_settings"]["venv_python_path"],
                "/usr/bin/python3",
            )
            self.assertEqual(stored["odoo_context_ref"]["api_key"], "[redacted]")
            self.assertEqual(stored["odoo_context_ref"]["database"], "prod")

    def test_job_meeting_date_is_persisted_for_library(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.dict(os.environ, {"EKO_APP_SUPPORT_DIR": str(root / "support")}):
                db = database()
                job_id = db.create_job(
                    source_path=str(root / "source.mov"),
                    workspace_dir=str(root / "work"),
                    settings={},
                )
                db.update_job_meeting_date(job_id, "2026-05-14T12:30:00Z")
                row = db.get_job(job_id)
                listed = db.list_jobs()

            self.assertEqual(row["meeting_date"], "2026-05-14T12:30:00Z")
            self.assertEqual(listed[0]["meeting_date"], "2026-05-14T12:30:00Z")

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
                    # PR Q: the sample extractor now refuses to
                    # surface < 2 KB files. Make the stub generate a
                    # bytes payload that clears the threshold.
                    Path(cmd[-1]).write_bytes(b"\0" * 4096)
                    return subprocess.CompletedProcess(cmd, 0, "", "")

                with patch("ekovideo_engine.library.subprocess.run", side_effect=fake_run):
                    samples = library_speaker_samples(job_id, seconds=3)

            self.assertEqual([sample["speaker"] for sample in samples], ["SPEAKER_00", "SPEAKER_01"])
            self.assertTrue(all(Path(sample["path"]).exists() for sample in samples))
            self.assertEqual(samples[0]["utterance_count"], 1)
            self.assertAlmostEqual(samples[0]["total_duration"], 3.0)
            self.assertEqual(samples[0]["index"], 1)

    def test_speaker_samples_can_return_multiple_clips_per_speaker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "work"
            workspace.mkdir()
            (workspace / "audio.wav").write_bytes(b"fake wav")

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
                        {"start": 1.0, "end": 4.0, "speaker": "SPEAKER_00", "text": "Premier."},
                        {"start": 8.0, "end": 12.0, "speaker": "SPEAKER_00", "text": "Deuxième."},
                        {"start": 15.0, "end": 21.0, "speaker": "SPEAKER_00", "text": "Troisième."},
                    ],
                )

                def fake_run(cmd, *args, **kwargs):
                    # PR Q: the sample extractor now refuses to
                    # surface < 2 KB files. Make the stub generate a
                    # bytes payload that clears the threshold.
                    Path(cmd[-1]).write_bytes(b"\0" * 4096)
                    return subprocess.CompletedProcess(cmd, 0, "", "")

                with patch("ekovideo_engine.library.subprocess.run", side_effect=fake_run):
                    samples = library_speaker_samples(job_id, seconds=3, per_speaker=2)

            self.assertEqual(len(samples), 2)
            self.assertEqual([sample["speaker"] for sample in samples], ["SPEAKER_00", "SPEAKER_00"])
            self.assertEqual([sample["index"] for sample in samples], [1, 2])
            self.assertTrue(all(sample["duration"] <= 3.0 for sample in samples))
            self.assertEqual(samples[0]["utterance_count"], 3)
            self.assertAlmostEqual(samples[0]["total_duration"], 13.0)

    def test_flag_speaker_sample_review_writes_workspace_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "work"
            workspace.mkdir()
            source = root / "source.mov"
            source.write_bytes(b"source")

            with patch.dict(os.environ, {"EKO_APP_SUPPORT_DIR": str(root / "support")}):
                db = database()
                job_id = db.create_job(
                    source_path=str(source),
                    workspace_dir=str(workspace),
                    settings={},
                )
                result = library_flag_speaker_sample_review(
                    job_id,
                    speaker="SPEAKER_00",
                    start=12.3,
                    duration=4.5,
                    note="Deux voix entendues.",
                )

            marker = workspace / "speaker_review_requests.jsonl"
            self.assertTrue(marker.exists())
            payload = json.loads(marker.read_text(encoding="utf-8").strip())
            self.assertEqual(payload["speaker"], "SPEAKER_00")
            self.assertEqual(payload["note"], "Deux voix entendues.")
            self.assertEqual(result["review_path"], str(marker))


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


class JobTotalBytesPersistenceTest(unittest.TestCase):
    def test_update_job_total_bytes_round_trips_via_list_jobs(self):
        # The library's optional "Poids" column reads ``total_bytes``
        # off each list_jobs row. Pin the round-trip so a future
        # schema change doesn't silently drop the value.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.dict(os.environ, {"EKO_APP_SUPPORT_DIR": str(root / "support")}):
                db = database()
                job_id = db.create_job(
                    source_path=str(root / "source.mov"),
                    workspace_dir=str(root / "work"),
                    settings={},
                )
                db.update_job_total_bytes(job_id, 1_234_567)
                rows = db.list_jobs(limit=10)
                self.assertEqual(rows[0]["total_bytes"], 1_234_567)
                # The runner's snapshot fires on every successful
                # job, so updating the same id twice must overwrite
                # rather than accumulate.
                db.update_job_total_bytes(job_id, 999)
                rows = db.list_jobs(limit=10)
                self.assertEqual(rows[0]["total_bytes"], 999)

    def test_legacy_rows_have_null_total_bytes(self):
        # Migrations are silent — a row created without ever calling
        # update_job_total_bytes must read back as None so the
        # SwiftUI side can render "—" instead of fabricating a 0.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.dict(os.environ, {"EKO_APP_SUPPORT_DIR": str(root / "support")}):
                db = database()
                job_id = db.create_job(
                    source_path=str(root / "source.mov"),
                    workspace_dir=str(root / "work"),
                    settings={},
                )
                rows = db.list_jobs(limit=10)
                self.assertEqual(rows[0]["id"], job_id)
                self.assertIsNone(rows[0]["total_bytes"])


class SpeakerEnrollmentTest(unittest.TestCase):
    """Cover the round-trip from rename → embedding → match.

    The pyannote subprocess is mocked because pulling 200 MB of
    weights into CI would be insane. We pin the contract on every
    moving part either side of the shell-out.
    """

    _job_counter = 0

    def _seed_job(
        self,
        root: Path,
        *,
        with_diar_audio: bool = True,
        venv_in_settings: bool = True,
    ) -> tuple[int, Path]:
        # Each call needs its own workspace dir, otherwise a second
        # _seed_job inside the same test trips ``mkdir(exist_ok=False)``.
        SpeakerEnrollmentTest._job_counter += 1
        workspace = root / f"work-{SpeakerEnrollmentTest._job_counter}"
        workspace.mkdir(parents=True)
        # ``_diarisation_audio_path`` looks for these markers in
        # order; touching either is enough for the helper to return
        # a real path.
        if with_diar_audio:
            (workspace / "audio.diar.wav").write_bytes(b"riff fake")
        else:
            (workspace / "audio.wav").write_bytes(b"riff fake")

        # The library's _venv_python helper can read the venv path off
        # settings_json. Point it at a fake interpreter so the
        # subprocess.run mock fires below.
        fake_python = root / "fake-python"
        fake_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        fake_python.chmod(0o755)
        settings = (
            {"transcription_settings": {"venv_python_path": str(fake_python)}}
            if venv_in_settings
            else {"transcription_settings": {"venv_python_path": ""}}
        )

        db = database()
        job_id = db.create_job(
            source_path=str(root / "source.mov"),
            workspace_dir=str(workspace),
            settings=settings,
        )
        # Two SPEAKER_NN clusters with several long enough turns
        # each, so the embedding picker accepts them.
        db.add_segments(
            job_id,
            [
                {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00", "text": "Bonjour."},
                {"start": 5.0, "end": 10.0, "speaker": "SPEAKER_01", "text": "Salut."},
                {"start": 10.0, "end": 15.0, "speaker": "SPEAKER_00", "text": "Hello."},
            ],
        )
        return job_id, workspace

    def test_rename_triggers_enrollment_and_persists_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.dict(os.environ, {"EKO_APP_SUPPORT_DIR": str(root / "support")}):
                job_id, _ = self._seed_job(root)

                def fake_run(cmd, *args, **kwargs):
                    # Embedding script returns one fake 4-dim vector
                    # per requested cluster. The pure-Python pipeline
                    # downstream averages + matches these.
                    payload = json.dumps(
                        {
                            "clusters": {
                                "SPEAKER_00": [[1.0, 0.0, 0.0, 0.0]],
                                "SPEAKER_01": [[0.0, 1.0, 0.0, 0.0]],
                            }
                        }
                    )
                    return subprocess.CompletedProcess(cmd, 0, payload, "")

                with patch(
                    "ekovideo_engine.library.subprocess.run",
                    side_effect=fake_run,
                ):
                    summary = library_rename_speakers(
                        job_id,
                        {"SPEAKER_00": "Robin", "SPEAKER_01": "David"},
                    )

                profiles = library_list_speaker_profiles()
                names = sorted(p["name"] for p in profiles)

            self.assertEqual(names, ["David", "Robin"])
            self.assertEqual(summary["speakers_enrolled"], 2)

    def test_rename_remembers_name_even_when_enrollment_cannot_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.dict(os.environ, {"EKO_APP_SUPPORT_DIR": str(root / "support")}):
                job_id, _ = self._seed_job(root, venv_in_settings=False)
                with patch(
                    "ekovideo_engine.library.managed_venv_python_path",
                    return_value=root / "missing-python",
                ), patch("ekovideo_engine.library.subprocess.run") as mock_run:
                    summary = library_rename_speakers(job_id, {"SPEAKER_00": "Robin"})
                    mock_run.assert_not_called()
                profiles = library_list_speaker_profiles()

            self.assertEqual(summary["speakers_enrolled"], 0)
            self.assertEqual(summary["speakers_remembered"], 1)
            self.assertEqual(profiles[0]["name"], "Robin")
            self.assertEqual(profiles[0]["sample_count"], 0)

    def test_enrollment_uses_managed_venv_when_settings_path_is_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            managed_python = root / "managed-python"
            managed_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            managed_python.chmod(0o755)
            with patch.dict(os.environ, {"EKO_APP_SUPPORT_DIR": str(root / "support")}):
                job_id, _ = self._seed_job(root, venv_in_settings=False)

                def fake_run(cmd, *args, **kwargs):
                    self.assertEqual(cmd[0], str(managed_python))
                    payload = json.dumps(
                        {"clusters": {"SPEAKER_00": [[1.0, 0.0, 0.0, 0.0]]}}
                    )
                    return subprocess.CompletedProcess(cmd, 0, payload, "")

                with patch(
                    "ekovideo_engine.library.managed_venv_python_path",
                    return_value=managed_python,
                ), patch(
                    "ekovideo_engine.library.subprocess.run",
                    side_effect=fake_run,
                ):
                    summary = library_rename_speakers(job_id, {"SPEAKER_00": "Robin"})

                profiles = library_list_speaker_profiles()

            self.assertEqual(summary["speakers_enrolled"], 1)
            self.assertEqual(profiles[0]["name"], "Robin")
            self.assertEqual(profiles[0]["sample_count"], 1)

    def test_remember_speaker_names_does_not_overwrite_voiceprint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.dict(os.environ, {"EKO_APP_SUPPORT_DIR": str(root / "support")}):
                db = database()
                db.upsert_speaker_profile(
                    name="Robin",
                    embedding_json=encode_embedding([1.0, 0.0]),
                    sample_count=3,
                )
                remembered = library_remember_speaker_names(["Robin", "Robin"], db=db)
                profile = library_list_speaker_profiles()[0]

            self.assertEqual(remembered, 0)
            self.assertEqual(profile["sample_count"], 3)

    def test_recognition_pre_fills_known_voices_on_new_job(self):
        # Enrol one profile, then run a fresh job whose SPEAKER_00
        # has the same fake embedding. Recognition should match it
        # back to the profile's name without the user typing
        # anything.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.dict(os.environ, {"EKO_APP_SUPPORT_DIR": str(root / "support")}):
                first_job_id, _ = self._seed_job(root)

                def first_run(cmd, *args, **kwargs):
                    payload = json.dumps(
                        {"clusters": {"SPEAKER_00": [[1.0, 0.0, 0.0, 0.0]]}}
                    )
                    return subprocess.CompletedProcess(cmd, 0, payload, "")

                with patch(
                    "ekovideo_engine.library.subprocess.run",
                    side_effect=first_run,
                ):
                    library_rename_speakers(first_job_id, {"SPEAKER_00": "Robin"})

                # Second meeting — same audio fingerprint.
                second_job_id, _ = self._seed_job(root)

                def second_run(cmd, *args, **kwargs):
                    payload = json.dumps(
                        {
                            "clusters": {
                                "SPEAKER_00": [[1.0, 0.0, 0.0, 0.0]],
                                "SPEAKER_01": [[0.0, 0.0, 1.0, 0.0]],
                            }
                        }
                    )
                    return subprocess.CompletedProcess(cmd, 0, payload, "")

                with patch(
                    "ekovideo_engine.library.subprocess.run",
                    side_effect=second_run,
                ):
                    recognised = library_recognize_speakers(second_job_id)

            # SPEAKER_00 matches Robin (same embedding); SPEAKER_01
            # is a new voice with no matching profile, so stays a
            # SPEAKER_NN placeholder for the user to confirm.
            self.assertEqual(recognised, {"SPEAKER_00": "Robin"})

    def test_recognition_skipped_when_no_profiles_exist(self):
        # First-run path: the user hasn't enrolled anyone yet, so
        # we don't even try to call pyannote — the cost is wasted
        # on a guaranteed empty result.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.dict(os.environ, {"EKO_APP_SUPPORT_DIR": str(root / "support")}):
                job_id, _ = self._seed_job(root)
                with patch(
                    "ekovideo_engine.library.subprocess.run"
                ) as mock_run:
                    out = library_recognize_speakers(job_id)
                    mock_run.assert_not_called()
            self.assertEqual(out, {})

    def test_enrollment_merges_into_existing_profile_centroid(self):
        # Enrol Robin twice. The stored centroid should be the
        # average of both samples and ``sample_count`` should bump
        # to 2 — not 1 + 1 separately, that would mean "two Robins"
        # in the store.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.dict(os.environ, {"EKO_APP_SUPPORT_DIR": str(root / "support")}):
                first_job_id, _ = self._seed_job(root)

                def first_run(cmd, *args, **kwargs):
                    payload = json.dumps(
                        {"clusters": {"SPEAKER_00": [[1.0, 0.0, 0.0, 0.0]]}}
                    )
                    return subprocess.CompletedProcess(cmd, 0, payload, "")

                with patch(
                    "ekovideo_engine.library.subprocess.run",
                    side_effect=first_run,
                ):
                    library_rename_speakers(first_job_id, {"SPEAKER_00": "Robin"})

                second_job_id, _ = self._seed_job(root)

                def second_run(cmd, *args, **kwargs):
                    payload = json.dumps(
                        {"clusters": {"SPEAKER_00": [[0.0, 1.0, 0.0, 0.0]]}}
                    )
                    return subprocess.CompletedProcess(cmd, 0, payload, "")

                with patch(
                    "ekovideo_engine.library.subprocess.run",
                    side_effect=second_run,
                ):
                    library_rename_speakers(second_job_id, {"SPEAKER_00": "Robin"})

                profiles = library_list_speaker_profiles()
                self.assertEqual(len(profiles), 1)
                self.assertEqual(profiles[0]["sample_count"], 2)

                db = database()
                stored = db.get_speaker_profile_by_name("Robin")
                centroid = decode_embedding(stored["embedding_json"])
            # Average of (1,0,0,0) and (0,1,0,0) → (0.5, 0.5, 0, 0)
            # → normalised (0.707, 0.707, 0, 0).
            self.assertAlmostEqual(centroid[0], math.sqrt(0.5), places=4)
            self.assertAlmostEqual(centroid[1], math.sqrt(0.5), places=4)

    def test_delete_speaker_profile_by_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.dict(os.environ, {"EKO_APP_SUPPORT_DIR": str(root / "support")}):
                db = database()
                db.upsert_speaker_profile(
                    name="Marie",
                    embedding_json=encode_embedding([1.0, 0.0]),
                    sample_count=1,
                )
                self.assertEqual(library_delete_speaker_profile(name="Marie"), 1)
                self.assertEqual(library_list_speaker_profiles(), [])
                # Idempotent — second delete returns 0 instead of
                # raising.
                self.assertEqual(library_delete_speaker_profile(name="Marie"), 0)

    def test_enrollment_fires_on_friendly_to_friendly_rename(self):
        # The pipeline's LLM title pass often replaces SPEAKER_NN
        # with a wrong guess ("Marie"). The user then corrects in
        # the rename sheet ("Marie" → "Sophie"). The previous
        # filter accepted only ``SPEAKER_NN → name`` pairs and
        # silently dropped this case, which is why the
        # speaker_profiles table stayed empty across the user's
        # whole library. Pin the fix.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.dict(os.environ, {"EKO_APP_SUPPORT_DIR": str(root / "support")}):
                job_id, _ = self._seed_job(root)
                # Pre-rename SPEAKER_00 → Marie so the segments
                # already carry friendly names — mimics what the
                # LLM title pass leaves behind.
                db = database()
                segments = db.get_segments(job_id)
                renamed = [
                    {**dict(s), "start": s.get("start_time"), "end": s.get("end_time"),
                     "speaker": "Marie" if s.get("speaker") == "SPEAKER_00" else "Paul"}
                    for s in segments
                ]
                db.add_segments(job_id, renamed)

                def fake_run(cmd, *args, **kwargs):
                    payload = json.dumps(
                        {"clusters": {"Marie": [[1.0, 0.0, 0.0, 0.0]]}}
                    )
                    return subprocess.CompletedProcess(cmd, 0, payload, "")

                with patch(
                    "ekovideo_engine.library.subprocess.run",
                    side_effect=fake_run,
                ):
                    summary = library_rename_speakers(
                        job_id, {"Marie": "Sophie"},
                    )

                profiles = library_list_speaker_profiles()
            self.assertEqual([p["name"] for p in profiles], ["Sophie"])
            self.assertEqual(summary["speakers_enrolled"], 1)

    def test_no_op_rename_does_not_re_enroll(self):
        # Pin the only "X → X" guard we kept: confirming the
        # existing label is correct must not trigger another
        # enrollment round-trip.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.dict(os.environ, {"EKO_APP_SUPPORT_DIR": str(root / "support")}):
                job_id, _ = self._seed_job(root)
                with patch(
                    "ekovideo_engine.library.subprocess.run"
                ) as mock_run:
                    summary = library_rename_speakers(
                        job_id, {"SPEAKER_00": "SPEAKER_00"},
                    )
                    mock_run.assert_not_called()
            self.assertEqual(summary["speakers_enrolled"], 0)

    def test_rename_rebuilds_canonical_speaker_map_from_segments(self):
        # The previous behaviour merged every historical rename
        # pair into ``speaker_map_json``, which caused the sheet
        # to show "Sophie" twice on reopen: once from the stale
        # ``Marie → Sophie`` pair, once from the actual ``Sophie``
        # cluster pulled off the samples list. Pin that the map
        # now mirrors the segments table 1:1.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.dict(os.environ, {"EKO_APP_SUPPORT_DIR": str(root / "support")}):
                job_id, _ = self._seed_job(root)
                db = database()
                # Pre-state: segments labelled Marie + Paul (LLM
                # guesses), map has them mapped to themselves.
                segments = db.get_segments(job_id)
                staged = [
                    {**dict(s), "start": s.get("start_time"), "end": s.get("end_time"),
                     "speaker": "Marie" if s.get("speaker") == "SPEAKER_00" else "Paul"}
                    for s in segments
                ]
                db.add_segments(job_id, staged)
                db.update_job_context(
                    job_id, speakers={"Marie": "Marie", "Paul": "Paul"}
                )

                # Stub the enrollment call so the test stays
                # focused on the map rebuild.
                with patch(
                    "ekovideo_engine.library.subprocess.run",
                    return_value=subprocess.CompletedProcess(
                        [], 0, json.dumps({"clusters": {}}), ""
                    ),
                ):
                    library_rename_speakers(
                        job_id, {"Marie": "Sophie", "Paul": "Robin"},
                    )

                row = db.get_job(job_id)
                final_map = json.loads(row.get("speaker_map_json") or "{}")
            # Map carries exactly the labels now in segments, each
            # pointing to itself — no stale Marie → Sophie pair.
            self.assertEqual(set(final_map.keys()), {"Sophie", "Robin"})
            self.assertEqual(final_map["Sophie"], "Sophie")
            self.assertEqual(final_map["Robin"], "Robin")

    def test_rename_with_enroll_disabled_skips_pyannote(self):
        # The CLI / Swift caller can opt out of enrollment when
        # they're just doing a string-only rename (e.g. fixing a
        # typo on an already-named speaker).
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.dict(os.environ, {"EKO_APP_SUPPORT_DIR": str(root / "support")}):
                job_id, _ = self._seed_job(root)
                with patch(
                    "ekovideo_engine.library.subprocess.run"
                ) as mock_run:
                    summary = library_rename_speakers(
                        job_id,
                        {"SPEAKER_00": "Robin"},
                        enroll=False,
                    )
                    mock_run.assert_not_called()
            self.assertEqual(summary["speakers_enrolled"], 0)


class OdooLinkageTest(unittest.TestCase):
    """Pin the contract of the link / unlink wrappers and the
    columns they hydrate on the speaker_profiles row."""

    def test_link_then_unlink_round_trips(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.dict(os.environ, {"EKO_APP_SUPPORT_DIR": str(root / "support")}):
                db = database()
                db.upsert_speaker_profile(
                    name="Robin",
                    embedding_json=encode_embedding([1.0, 0.0, 0.0]),
                    sample_count=1,
                )
                profile = library_list_speaker_profiles()[0]
                # Robin → res.partner #42 sitting under company #100 "Acme".
                updated = library_link_speaker_profile_to_odoo(
                    profile["id"],
                    partner_id=42,
                    partner_name="Robin Dupuy",
                    company_id=100,
                    company_name="Acme",
                )
                self.assertEqual(updated["odoo_partner_id"], 42)
                self.assertEqual(updated["odoo_partner_name"], "Robin Dupuy")
                self.assertEqual(updated["odoo_company_id"], 100)
                self.assertEqual(updated["odoo_company_name"], "Acme")
                self.assertIsNotNone(updated.get("linked_at"))

                # Unlink wipes everything Odoo-related but keeps
                # the voice profile (and its embedding) alive.
                cleared = library_unlink_speaker_profile_from_odoo(profile["id"])
                self.assertIsNone(cleared.get("odoo_partner_id"))
                self.assertIsNone(cleared.get("odoo_partner_name"))
                self.assertIsNone(cleared.get("linked_at"))

                self.assertEqual(len(library_list_speaker_profiles()), 1)

    def test_relink_overwrites_previous_partner(self):
        # The user picked the wrong contact, then re-runs the picker
        # and chooses the right one. Second link must replace the
        # first, not stack on top of it.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.dict(os.environ, {"EKO_APP_SUPPORT_DIR": str(root / "support")}):
                db = database()
                db.upsert_speaker_profile(
                    name="David",
                    embedding_json=encode_embedding([0.0, 1.0]),
                    sample_count=1,
                )
                profile_id = library_list_speaker_profiles()[0]["id"]
                library_link_speaker_profile_to_odoo(
                    profile_id, partner_id=10, partner_name="WRONG",
                )
                library_link_speaker_profile_to_odoo(
                    profile_id, partner_id=11, partner_name="RIGHT",
                    company_id=5, company_name="Acme",
                )
                only = library_list_speaker_profiles()[0]
                self.assertEqual(only["odoo_partner_id"], 11)
                self.assertEqual(only["odoo_partner_name"], "RIGHT")
                self.assertEqual(only["odoo_company_id"], 5)


class LibrarySingleRowFetchTest(unittest.TestCase):
    """The SwiftUI ``LibraryStore.refreshOne`` path swaps the
    1000-row ``library-list`` payload for a single ``library-get``
    SELECT after rename / context updates. Pin both the happy path
    and the "row already deleted" branch so the SwiftUI cache
    doesn't end up out of sync on either."""

    def test_returns_row_when_job_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.dict(os.environ, {"EKO_APP_SUPPORT_DIR": str(root / "support")}):
                db = database()
                job_id = db.create_job(
                    source_path=str(root / "x.mov"),
                    workspace_dir=str(root / "ws"),
                    settings={},
                )
                row = library_get(job_id)

            self.assertIsNotNone(row)
            assert row is not None
            self.assertEqual(row["id"], job_id)

    def test_returns_none_when_job_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.dict(os.environ, {"EKO_APP_SUPPORT_DIR": str(root / "support")}):
                self.assertIsNone(library_get(424242))


class LibraryRenameAutoLinkOdooTest(unittest.TestCase):
    """When the rename target matches an Odoo meeting attendee, the
    resulting speaker_profile should be auto-linked to the
    corresponding ``res.partner``. Pins the link-lookup, the
    case-insensitive match, and the "leave manual overrides alone"
    guard."""

    def _create_job_with_segments(self, db, root: Path) -> int:
        job_id = db.create_job(
            source_path=str(root / "source.mov"),
            workspace_dir=str(root / "work"),
            settings={},
        )
        db.add_segments(
            job_id,
            [
                {
                    "start": 1.0,
                    "end": 2.0,
                    "speaker": "SPEAKER_00",
                    "text": "Bonjour.",
                },
            ],
        )
        return job_id

    def test_links_new_profile_to_matching_attendee(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.dict(os.environ, {"EKO_APP_SUPPORT_DIR": str(root / "support")}):
                db = database()
                job_id = self._create_job_with_segments(db, root)
                attendee_map = {
                    "sophie martin": {
                        "partner_id": 42,
                        "partner_name": "Sophie Martin",
                        "company_name": "ACME",
                    }
                }
                result = library_rename_speakers(
                    job_id,
                    {"SPEAKER_00": "Sophie Martin"},
                    attendee_partner_map=attendee_map,
                )
                profile = db.get_speaker_profile_by_name("Sophie Martin")

            self.assertEqual(result["speakers_linked_to_odoo"], 1)
            assert profile is not None
            self.assertEqual(profile["odoo_partner_id"], 42)
            self.assertEqual(profile["odoo_partner_name"], "Sophie Martin")
            self.assertEqual(profile["odoo_company_name"], "ACME")

    def test_skips_when_name_not_in_attendee_map(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.dict(os.environ, {"EKO_APP_SUPPORT_DIR": str(root / "support")}):
                db = database()
                job_id = self._create_job_with_segments(db, root)
                result = library_rename_speakers(
                    job_id,
                    {"SPEAKER_00": "Inconnu"},
                    attendee_partner_map={
                        "sophie martin": {"partner_id": 42, "partner_name": "Sophie"}
                    },
                )
                profile = db.get_speaker_profile_by_name("Inconnu")

            self.assertEqual(result["speakers_linked_to_odoo"], 0)
            assert profile is not None
            self.assertFalse(profile.get("odoo_partner_id"))

    def test_respects_manual_link_to_different_partner(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.dict(os.environ, {"EKO_APP_SUPPORT_DIR": str(root / "support")}):
                db = database()
                job_id = self._create_job_with_segments(db, root)
                # Pre-existing profile already linked to partner 99.
                db.upsert_speaker_profile(name="Sophie", embedding_json="[]", sample_count=0)
                existing = db.get_speaker_profile_by_name("Sophie")
                assert existing is not None
                db.link_speaker_profile_to_odoo(
                    int(existing["id"]),
                    partner_id=99,
                    partner_name="Manual",
                    company_id=None,
                    company_name="",
                )

                result = library_rename_speakers(
                    job_id,
                    {"SPEAKER_00": "Sophie"},
                    attendee_partner_map={
                        "sophie": {
                            "partner_id": 42,
                            "partner_name": "Sophie From Odoo",
                            "company_name": "ACME",
                        }
                    },
                )
                profile = db.get_speaker_profile_by_name("Sophie")

            self.assertEqual(result["speakers_linked_to_odoo"], 0)
            assert profile is not None
            self.assertEqual(profile["odoo_partner_id"], 99)
            self.assertEqual(profile["odoo_partner_name"], "Manual")


class SnapshotExistingArtifactsTest(unittest.TestCase):
    """Pins the rerun safety net: previous outputs are moved into
    ``versions/<timestamp>/`` before the new run overwrites them."""

    def _seed_workspace(self, root: Path) -> tuple[Path, dict]:
        workspace = root / "ws"
        workspace.mkdir()
        compressed = workspace / "source.compressed.mp4"
        compressed.write_bytes(b"compressed-v1")
        transcript = workspace / "source.txt"
        transcript.write_text("transcript v1", encoding="utf-8")
        enhanced = workspace / "source - améliorée.txt"
        enhanced.write_text("enhanced v1", encoding="utf-8")
        review = workspace / "source - à vérifier.md"
        review.write_text("review v1", encoding="utf-8")
        job = {
            "id": 1,
            "compressed_path": str(compressed),
            "transcript_path": str(transcript),
            "enhanced_transcript_path": str(enhanced),
            "review_path": str(review),
        }
        return workspace, job

    def test_moves_existing_outputs_into_versioned_folder(self):
        from ekovideo_engine.pipeline import snapshot_existing_artifacts
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace, job = self._seed_workspace(root)
            summary = snapshot_existing_artifacts(workspace, job, lambda _: None)

            self.assertTrue(summary)
            versions_dir = workspace / "versions"
            snapshots = list(versions_dir.iterdir())
            self.assertEqual(len(snapshots), 1)
            snapshot = snapshots[0]
            # The four files moved out of the workspace root and
            # into the dated subfolder.
            self.assertTrue((snapshot / "source.compressed.mp4").exists())
            self.assertTrue((snapshot / "source.txt").exists())
            self.assertFalse((workspace / "source.txt").exists())
            # Summary carries the destination paths so the DB can
            # render Reveal-in-Finder later.
            self.assertEqual(summary["compressed_path"], str(snapshot / "source.compressed.mp4"))
            self.assertIn("label", summary)
            self.assertIn("created_at", summary)

    def test_fresh_workspace_no_ops(self):
        from ekovideo_engine.pipeline import snapshot_existing_artifacts
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "ws"
            workspace.mkdir()
            summary = snapshot_existing_artifacts(workspace, {}, lambda _: None)
            self.assertEqual(summary, {})
            self.assertFalse((workspace / "versions").exists())

    def test_protected_source_is_not_moved(self):
        # Regression: after "Libérer la source", the rerun's source IS
        # the compressed file. Snapshotting must leave it in place,
        # otherwise the pipeline can't read its own input.
        from ekovideo_engine.pipeline import (
            _normalized_realpath,
            snapshot_existing_artifacts,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace, job = self._seed_workspace(root)
            compressed = job["compressed_path"]
            summary = snapshot_existing_artifacts(
                workspace,
                job,
                lambda _: None,
                protected_paths={_normalized_realpath(compressed)},
            )
            # Compressed stayed put (it's the active source); the other
            # artefacts were still snapshotted.
            self.assertTrue(Path(compressed).exists())
            self.assertNotIn("compressed_path", summary)
            self.assertIn("transcript_path", summary)

    def test_skips_missing_files_silently(self):
        # A row whose paths point at deleted files (workspace was
        # wiped manually) must not crash — just skip the missing
        # entries and snapshot whatever survives.
        from ekovideo_engine.pipeline import snapshot_existing_artifacts
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "ws"
            workspace.mkdir()
            kept = workspace / "source.txt"
            kept.write_text("kept", encoding="utf-8")
            job = {
                "id": 1,
                "transcript_path": str(kept),
                "compressed_path": str(workspace / "missing.mp4"),
            }
            summary = snapshot_existing_artifacts(workspace, job, lambda _: None)
            self.assertIn("transcript_path", summary)
            self.assertNotIn("compressed_path", summary)


class PrependJobVersionTest(unittest.TestCase):
    """The new ``previous_versions_json`` column accumulates snapshot
    metadata across reruns. Pins: list grows newest-first and is
    capped at 10 entries so an aggressively rerun job doesn't bloat
    the DB."""

    def test_prepend_keeps_newest_first_and_caps_at_ten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.dict(os.environ, {"EKO_APP_SUPPORT_DIR": str(root / "support")}):
                db = database()
                job_id = db.create_job(
                    source_path=str(root / "x.mov"),
                    workspace_dir=str(root / "ws"),
                    settings={},
                )
                for i in range(12):
                    db.prepend_job_version(
                        job_id,
                        {
                            "label": f"20260517-{i:06d}",
                            "created_at": f"2026-05-17T14:00:{i:02d}Z",
                            "transcript_path": f"/ws/v{i}/t.txt",
                        },
                    )
                row = db.get_job(job_id)

            assert row is not None
            versions = json.loads(row["previous_versions_json"])
            self.assertEqual(len(versions), 10)
            # Newest first — the 11th prepend (i=11) lives at index 0;
            # the original two (i=0, i=1) got dropped off the tail.
            self.assertEqual(versions[0]["label"], "20260517-000011")
            self.assertEqual(versions[-1]["label"], "20260517-000002")


class LibraryRepairSpeakerMapsTest(unittest.TestCase):
    """One-shot heal pass triggered at app launch (post PR #30).
    Pins:

    * A drifted map gets rebuilt from segments.
    * A canonical map stays untouched (no needless writes).
    * Jobs without any segments are skipped (don't replace whatever
      the user might have entered with ``{}``).
    """

    def _seed_job(
        self,
        db,
        root: Path,
        *,
        segments: list[dict],
        stored_speaker_map: dict[str, str] | None,
    ) -> int:
        job_id = db.create_job(
            source_path=str(root / "x.mov"),
            workspace_dir=str(root / "ws"),
            settings={},
        )
        if segments:
            db.add_segments(job_id, segments)
        if stored_speaker_map is not None:
            db.update_job_context(job_id, speakers=stored_speaker_map)
        return job_id

    def test_rebuilds_drifted_map_from_segments(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.dict(os.environ, {"EKO_APP_SUPPORT_DIR": str(root / "support")}):
                db = database()
                job_id = self._seed_job(
                    db,
                    root,
                    segments=[
                        {"start": 0.0, "end": 1.0, "speaker": "Robin", "text": "Hi."},
                        {"start": 1.0, "end": 2.0, "speaker": "Sophie", "text": "Hi."},
                    ],
                    # Drift: cluster IDs that no longer exist in segments.
                    stored_speaker_map={"SPEAKER_01": "Robin", "SPEAKER_00": "Sophie"},
                )

                summary = library_repair_all_speaker_maps()
                row = db.get_job(job_id)

            self.assertEqual(summary["repaired"], 1)
            self.assertEqual(summary["unchanged"], 0)
            self.assertEqual(summary["skipped_no_segments"], 0)
            assert row is not None
            persisted = json.loads(row["speaker_map_json"])
            self.assertEqual(persisted, {"Robin": "Robin", "Sophie": "Sophie"})

    def test_canonical_map_is_left_alone(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.dict(os.environ, {"EKO_APP_SUPPORT_DIR": str(root / "support")}):
                db = database()
                self._seed_job(
                    db,
                    root,
                    segments=[
                        {"start": 0.0, "end": 1.0, "speaker": "Robin", "text": "Hi."},
                    ],
                    stored_speaker_map={"Robin": "Robin"},
                )

                summary = library_repair_all_speaker_maps()

            self.assertEqual(summary["repaired"], 0)
            self.assertEqual(summary["unchanged"], 1)

    def test_empty_segments_jobs_are_skipped(self):
        # Old jobs whose segments table was never populated — must
        # NOT be flushed to ``{}``: the existing map may carry hints
        # the user added by hand.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.dict(os.environ, {"EKO_APP_SUPPORT_DIR": str(root / "support")}):
                db = database()
                job_id = self._seed_job(
                    db,
                    root,
                    segments=[],
                    stored_speaker_map={"SPEAKER_00": "Marie"},
                )

                summary = library_repair_all_speaker_maps()
                row = db.get_job(job_id)

            self.assertEqual(summary["skipped_no_segments"], 1)
            self.assertEqual(summary["repaired"], 0)
            assert row is not None
            self.assertEqual(
                json.loads(row["speaker_map_json"]),
                {"SPEAKER_00": "Marie"},
            )


class LibraryDetachOdooMeetingTest(unittest.TestCase):
    """The hidden "Réunion Odoo" library column lets the user break
    a stale meeting link without a full rerun. The CLI helper writes
    ``odoo_meeting_json = NULL`` on the row — confirm both that the
    write hits the DB and that re-reading the row sees the cleared
    value."""

    def test_detach_clears_odoo_meeting_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.dict(os.environ, {"EKO_APP_SUPPORT_DIR": str(root / "support")}):
                db = database()
                job_id = db.create_job(
                    source_path=str(root / "x.mov"),
                    workspace_dir=str(root / "ws"),
                    settings={},
                )
                db.update_job_odoo_meeting(
                    job_id,
                    {
                        "event_id": 99,
                        "event_name": "Atelier RH",
                        "attendees": [],
                    },
                )

                summary = library_detach_odoo_meeting(job_id)

                self.assertEqual(summary["detached"], True)
                self.assertEqual(summary["job_id"], job_id)
                row = db.get_job(job_id)

            assert row is not None
            self.assertFalse((row.get("odoo_meeting_json") or "").strip())


class ClipWindowTest(unittest.TestCase):
    """PR Q — for long segments the sample window now CENTERS in
    the segment instead of starting at the beginning. Reduces the
    risk of capturing the previous speaker's audio when Whisper's
    30 s context window crossed a boundary."""

    def test_short_segment_uses_whole_audio(self):
        seg = {"start": 5.0, "end": 7.5}  # 2.5 s
        start, duration = _clip_window(seg, seconds=8.0)
        # Just trim the edges, return start ≈ 5.05.
        self.assertAlmostEqual(start, 5.05, places=2)
        self.assertAlmostEqual(duration, 2.4, places=1)

    def test_long_segment_centered(self):
        # 30 s segment, 8 s clip → centered around midpoint (start
        # + 15 s). Clip should start at 11 (midpoint 20 − half 4),
        # cover 11-19.
        seg = {"start": 5.0, "end": 35.0}
        start, duration = _clip_window(seg, seconds=8.0)
        self.assertAlmostEqual(start, 16.0, places=1)
        self.assertEqual(duration, 8.0)

    def test_zero_duration_segment_returns_zero(self):
        seg = {"start": 10.0, "end": 10.0}
        start, duration = _clip_window(seg, seconds=8.0)
        self.assertEqual(duration, 0.0)


class SampleExtractionRobustnessTest(unittest.TestCase):
    """PR Q — refuse < 2 KB sample files; re-extract instead of
    accepting a previous failed run's empty WAV."""

    def test_truncated_existing_sample_gets_reextracted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "work"
            workspace.mkdir()
            (workspace / "audio.wav").write_bytes(b"fake wav")
            sample_dir = workspace / "speaker_samples"
            sample_dir.mkdir()
            with patch.dict(os.environ, {"EKO_APP_SUPPORT_DIR": str(root / "support")}):
                db = database()
                job_id = db.create_job(
                    source_path=str(root / "source.mov"),
                    workspace_dir=str(workspace),
                    settings={},
                )
                db.add_segments(
                    job_id,
                    [{"start": 1.0, "end": 4.0, "speaker": "SPEAKER_00", "text": "Hi."}],
                )
                # Pre-place a "stale" truncated file at the path the
                # extractor would pick. Old behaviour: accepted as-is.
                # PR Q behaviour: detect tiny size and re-extract.
                pre_existing = sample_dir / "SPEAKER_00_1_1.05.wav"
                pre_existing.write_bytes(b"X")  # 1 byte → too small

                runs: list[list[str]] = []

                def fake_run(cmd, *args, **kwargs):
                    runs.append(list(cmd))
                    Path(cmd[-1]).write_bytes(b"\0" * 4096)
                    return subprocess.CompletedProcess(cmd, 0, "", "")

                with patch(
                    "ekovideo_engine.library.subprocess.run",
                    side_effect=fake_run,
                ):
                    samples = library_speaker_samples(job_id, seconds=3)

            self.assertEqual(len(samples), 1)
            # The truncated file was rewritten — ffmpeg fired once.
            self.assertEqual(len(runs), 1)

    def test_extraction_failure_skips_sample(self):
        # If ffmpeg returns successfully but the file stays under
        # threshold, drop the sample rather than surface an empty
        # play button in the rename sheet.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "work"
            workspace.mkdir()
            (workspace / "audio.wav").write_bytes(b"fake wav")
            with patch.dict(os.environ, {"EKO_APP_SUPPORT_DIR": str(root / "support")}):
                db = database()
                job_id = db.create_job(
                    source_path=str(root / "source.mov"),
                    workspace_dir=str(workspace),
                    settings={},
                )
                db.add_segments(
                    job_id,
                    [{"start": 1.0, "end": 4.0, "speaker": "SPEAKER_00", "text": "Hi."}],
                )

                def fake_run(cmd, *args, **kwargs):
                    # Simulate ffmpeg "succeeding" but writing junk.
                    Path(cmd[-1]).write_bytes(b"\0" * 100)
                    return subprocess.CompletedProcess(cmd, 0, "", "")

                with patch(
                    "ekovideo_engine.library.subprocess.run",
                    side_effect=fake_run,
                ):
                    samples = library_speaker_samples(job_id, seconds=3)

            self.assertEqual(samples, [])


class SegmentsPerClusterTest(unittest.TestCase):
    """PR J — tighter selection of enrolment samples. Should exclude
    turns that overlap another speaker (centroid contamination
    risk), accumulate up to ~30 s of clean speech per cluster,
    cap at 5 turns regardless."""

    def test_skips_turns_overlapping_other_speaker(self):
        # Robin's "Bonjour" overlaps Manon's "Salut" — that turn
        # would pollute Robin's centroid with Manon's voice.
        segments = [
            # Robin clean
            {"start": 0.0, "end": 5.0, "speaker": "Robin"},
            # Robin overlapping Manon → must be dropped
            {"start": 10.0, "end": 15.0, "speaker": "Robin"},
            {"start": 12.0, "end": 14.0, "speaker": "Manon"},
            # Manon clean
            {"start": 20.0, "end": 25.0, "speaker": "Manon"},
        ]
        out = _segments_per_cluster(segments, ["Robin", "Manon"])
        # Robin's overlapping turn (10-15) is excluded → only the
        # clean 0-5 turn remains.
        self.assertEqual(len(out["Robin"]), 1)
        self.assertAlmostEqual(out["Robin"][0]["start"], 0.0)
        # Manon's two turns: 12-14 overlaps Robin too, so dropped.
        # Only 20-25 remains.
        self.assertEqual(len(out["Manon"]), 1)
        self.assertAlmostEqual(out["Manon"][0]["start"], 20.0)

    def test_caps_at_five_turns(self):
        # 10 long clean turns for Robin — we should only feed
        # the top 5 to pyannote.
        segments = [
            {"start": float(i * 10), "end": float(i * 10 + 6), "speaker": "Robin"}
            for i in range(10)
        ]
        out = _segments_per_cluster(segments, ["Robin"])
        self.assertEqual(len(out["Robin"]), 5)

    def test_stops_early_once_target_seconds_hit(self):
        # Two ~20 s turns hit the 30 s target — no need to
        # accumulate more even if available.
        segments = [
            {"start": 0.0, "end": 20.0, "speaker": "Robin"},
            {"start": 30.0, "end": 50.0, "speaker": "Robin"},
            {"start": 60.0, "end": 80.0, "speaker": "Robin"},
            {"start": 90.0, "end": 100.0, "speaker": "Robin"},
        ]
        out = _segments_per_cluster(segments, ["Robin"])
        # 2 turns × 20 s = 40 s > 30 s target → stop.
        self.assertEqual(len(out["Robin"]), 2)

    def test_falls_back_to_contaminated_when_no_clean(self):
        # Every turn overlaps the other speaker — we still emit
        # SOME samples rather than 0 (a contaminated centroid is
        # still better than no centroid at all).
        segments = [
            {"start": 0.0, "end": 5.0, "speaker": "Robin"},
            {"start": 1.0, "end": 4.0, "speaker": "Manon"},
        ]
        out = _segments_per_cluster(segments, ["Robin", "Manon"])
        self.assertEqual(len(out.get("Robin", [])), 1)
        self.assertEqual(len(out.get("Manon", [])), 1)

    def test_drops_short_turns(self):
        # < 2 s turns are dropped (centroid would be unstable).
        segments = [
            {"start": 0.0, "end": 1.0, "speaker": "Robin"},  # 1s — dropped
            {"start": 5.0, "end": 8.0, "speaker": "Robin"},  # 3s — kept
        ]
        out = _segments_per_cluster(segments, ["Robin"])
        self.assertEqual(len(out["Robin"]), 1)
        self.assertAlmostEqual(out["Robin"][0]["start"], 5.0)


if __name__ == "__main__":
    unittest.main()
