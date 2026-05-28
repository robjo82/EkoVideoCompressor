"""
PR AN — CompressionPipeline must compress the WORKING source
(workspace copy), not ``request.source_path``.

Bug observed in app.log (2026-05-28) on a screen recording with
the "delete source after copy" toggle on :

  engine_prepare_workspace deleted_original source='.../Desktop/...mov'
  engine_compress_start source='.../Desktop/...mov'  ← already deleted!
  engine_compress_failed rc=254 error='No such file or directory'

``prepare_job_workspace`` copies the source into the workspace
then deletes the original. ``CompressionPipeline.run`` read
``request.source_path`` (the deleted original) instead of the
workspace copy, so ffmpeg failed.

Fix : ``run(source_path)`` takes the resolved working source (the
runner passes ``active_source``). These tests pin the contract.

NOTE: we do NOT patch ``Path.exists`` — ``default_out_path``
contains a ``while out.exists()`` uniqueness loop, and a global
``Path.exists → True`` patch turns it into an infinite loop.
Instead the fake ``subprocess.run`` creates the real output file
so the genuine ``.exists()`` checks pass.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from ekovideo_engine.models import JobRequest
from ekovideo_engine.pipeline import CompressionPipeline


def _make_pipeline(workspace: str, source_path: str) -> CompressionPipeline:
    request = JobRequest(
        source_path=source_path,
        output_dir=workspace,
        workspace_dir=workspace,
        mode="compress",
    )
    return CompressionPipeline(request=request, sink=MagicMock())


def _fake_ffmpeg_creating_output():
    """subprocess.run replacement that 'compresses' by touching the
    output file (last positional arg of the ffmpeg cmd) so the
    pipeline's genuine ``Path(output_path).exists()`` check passes."""
    captured: dict = {}

    def fake_run(cmd, *args, **kwargs):  # noqa: ARG001
        captured["cmd"] = cmd
        # Last arg is the output path. Create it for real.
        out = Path(cmd[-1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"\x00")
        return MagicMock(returncode=0, stdout="", stderr="")

    return fake_run, captured


class CompressionWorkingSourceTests(unittest.TestCase):
    def _ffmpeg_input_for(self, pipeline, run_arg) -> str:
        fake_run, captured = _fake_ffmpeg_creating_output()
        with patch(
            "ekovideo_engine.pipeline.subprocess.run", side_effect=fake_run
        ):
            if run_arg is None:
                pipeline.run()
            else:
                pipeline.run(run_arg)
        cmd = captured["cmd"]
        idx = cmd.index("-i")
        return cmd[idx + 1]

    def test_uses_working_source_when_provided(self):
        # The canonical bug : original deleted, workspace copy is
        # the only file left. run(working_source) must feed ffmpeg
        # the workspace copy.
        with tempfile.TemporaryDirectory() as tmp:
            original = "/Users/robin/Desktop/meeting.mov"  # deleted
            working = f"{tmp}/meeting.mov"  # the workspace copy
            pipeline = _make_pipeline(tmp, original)
            ffmpeg_input = self._ffmpeg_input_for(pipeline, working)
        self.assertEqual(ffmpeg_input, working)
        self.assertNotEqual(ffmpeg_input, original)

    def test_falls_back_to_request_source_when_no_arg(self):
        # Legacy / programmatic caller that doesn't pass a source —
        # behaviour unchanged, uses request.source_path.
        with tempfile.TemporaryDirectory() as tmp:
            source = f"{tmp}/meeting.mov"
            pipeline = _make_pipeline(tmp, source)
            ffmpeg_input = self._ffmpeg_input_for(pipeline, None)
        self.assertEqual(ffmpeg_input, source)

    def test_output_name_derives_from_original_source(self):
        # Even when compressing the workspace copy, the output
        # artefact keeps the original meeting's basename.
        with tempfile.TemporaryDirectory() as tmp:
            original = "/Users/robin/Desktop/Réunion CVR.mov"
            working = f"{tmp}/Réunion CVR.mov"
            pipeline = _make_pipeline(tmp, original)
            fake_run, captured = _fake_ffmpeg_creating_output()
            with patch(
                "ekovideo_engine.pipeline.subprocess.run", side_effect=fake_run
            ):
                result = pipeline.run(working)
            output_path = captured["cmd"][-1]
        self.assertIn("Réunion CVR_compressed", output_path)
        self.assertTrue(result.ok)

    def test_run_succeeds_when_only_working_source_exists(self):
        # End-to-end : the StepResult is ok because ffmpeg read the
        # workspace copy (which exists) — reproduces the fixed bug.
        with tempfile.TemporaryDirectory() as tmp:
            original = "/nonexistent/deleted.mov"
            working = f"{tmp}/deleted.mov"
            pipeline = _make_pipeline(tmp, original)
            fake_run, _ = _fake_ffmpeg_creating_output()
            with patch(
                "ekovideo_engine.pipeline.subprocess.run", side_effect=fake_run
            ):
                result = pipeline.run(working)
        self.assertTrue(result.ok)
        self.assertEqual(result.name, "compression")


if __name__ == "__main__":
    unittest.main()
