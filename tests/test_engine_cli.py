from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from ekovideo_engine.events import event_to_json
from ekovideo_engine.events import collect_events
from ekovideo_engine.models import DoneEvent, JobRequest, ProgressEvent
from ekovideo_engine.pipeline import _friendly_ffmpeg_error, prepare_job_workspace
from transcription_eval.evaluate import evaluate_case


class EngineCliTest(unittest.TestCase):
    def _run(self, *args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
        full_env = os.environ.copy()
        if env:
            full_env.update(env)
        return subprocess.run(
            [sys.executable, "-m", "ekovideo_engine", *args],
            capture_output=True,
            text=True,
            env=full_env,
            check=False,
        )

    def test_smoke_test_emits_jsonl_done(self):
        proc = self._run("--smoke-test")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        lines = [json.loads(line) for line in proc.stdout.splitlines()]
        self.assertEqual(lines[-1]["event"], "done")
        self.assertTrue(lines[-1]["summary"]["ok"])

    def test_model_list_contains_audio_family(self):
        proc = self._run("model-list")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        rows = json.loads(proc.stdout)
        self.assertTrue(any(row["family"] == "audio_llm" for row in rows))

    def test_library_list_uses_configurable_support_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = self._run("library-list", env={"EKO_APP_SUPPORT_DIR": tmp})
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(json.loads(proc.stdout), [])


class EngineProtocolTest(unittest.TestCase):
    def test_event_serialization_is_stable_json(self):
        payload = json.loads(event_to_json(ProgressEvent("whisper", 42, "Running")))
        self.assertEqual(payload["event"], "progress")
        self.assertEqual(payload["step"], "whisper")
        self.assertEqual(payload["pct"], 42)
        self.assertEqual(payload["message"], "Running")

    def test_job_request_validates_required_fields(self):
        request = JobRequest.from_dict(
            {
                "source_path": "/tmp/in.mov",
                "output_dir": "/tmp/out",
                "mode": "compress_transcribe",
                "glossary_terms": ["Mollie"],
            }
        )
        self.assertEqual(request.mode, "compress_transcribe")
        self.assertEqual(request.glossary_terms, ["Mollie"])

        with self.assertRaises(ValueError):
            JobRequest.from_dict({"source_path": "/tmp/in.mov", "output_dir": "/tmp/out", "mode": "bad"})

    def test_prepare_job_workspace_copies_source_into_technical_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "meeting.mov"
            source.write_bytes(b"fake media")
            request = JobRequest.from_dict(
                {
                    "source_path": str(source),
                    "output_dir": str(root / "EkoVideo Compressor"),
                    "mode": "compress_transcribe",
                }
            )
            events, sink = collect_events()

            workspace, copied = prepare_job_workspace(request, sink)

            self.assertTrue(workspace.is_dir())
            self.assertEqual(copied.parent, workspace)
            self.assertEqual(copied.name, "meeting.mov")
            self.assertEqual(copied.read_bytes(), b"fake media")
            self.assertEqual(events[0]["event"], "artifact")
            self.assertEqual(events[0]["kind"], "source")


class TranscriptionEvalTest(unittest.TestCase):
    def test_seed_case_scores_perfectly(self):
        path = Path("transcription_eval/cases/symphonat_mollie.json")
        result = evaluate_case(path)
        self.assertEqual(result.missing_terms, [])
        self.assertEqual(result.forbidden_hits, [])
        self.assertEqual(result.missing_speakers, [])
        self.assertEqual(result.score, 1.0)


class FriendlyFfmpegErrorTest(unittest.TestCase):
    """The dyld error a broken Homebrew-linked bundle produced in
    v0.13.0+ was rendered verbatim in the SwiftUI UI, leaving users
    staring at a stack trace. The mapper substitutes a human-readable
    explanation while keeping the technical fingerprint in the log
    for debugging.
    """

    def test_translates_library_not_loaded(self):
        raw = (
            "dyld[12560]: Library not loaded: "
            "/opt/homebrew/Cellar/ffmpeg/8.1.1/lib/libavdevice.62.dylib\n"
            "  Referenced from: /Applications/EkoVideoCompressor.app/.../ffmpeg"
        )
        out = _friendly_ffmpeg_error(raw, "/Applications/.../ffmpeg")
        self.assertIn("binaire ffmpeg fourni", out)
        self.assertIn("Réinstallez", out)
        # The technical fingerprint survives so a support exchange has
        # something to grep for.
        self.assertIn("Library not loaded", out)

    def test_passes_through_real_ffmpeg_errors(self):
        raw = "Invalid data found when processing input"
        out = _friendly_ffmpeg_error(raw, "/usr/bin/ffmpeg")
        self.assertEqual(out, raw)

    def test_falls_back_when_stderr_is_empty(self):
        out = _friendly_ffmpeg_error("", "/usr/bin/ffmpeg")
        self.assertIn("/usr/bin/ffmpeg", out)


if __name__ == "__main__":
    unittest.main()
