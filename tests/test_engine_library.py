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
    _discover_speakers_from_text,
    _looks_like_engine_workspace,
    database,
    library_delete,
    library_delete_speaker_profile,
    library_discover_speakers,
    library_enroll_speakers_for_job,
    library_flag_speaker_sample_review,
    library_link_speaker_profile_to_odoo,
    library_list_speaker_profiles,
    library_recognize_speakers,
    library_rename_speakers,
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
                    Path(cmd[-1]).write_bytes(b"sample")
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

    def _seed_job(self, root: Path, *, with_diar_audio: bool = True) -> tuple[int, Path]:
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

        # The library's _venv_python helper reads the venv path off
        # settings_json. Point it at a fake interpreter so the
        # subprocess.run mock fires below.
        fake_python = root / "fake-python"
        fake_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        fake_python.chmod(0o755)

        db = database()
        job_id = db.create_job(
            source_path=str(root / "source.mov"),
            workspace_dir=str(workspace),
            settings={
                "transcription_settings": {
                    "venv_python_path": str(fake_python),
                }
            },
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


if __name__ == "__main__":
    unittest.main()
