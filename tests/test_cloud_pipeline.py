"""Pipeline-level tests for the cloud transcription engine.

No real network, no real ffmpeg: ``subprocess.run`` is stubbed for
the ffprobe/ffmpeg shell-outs and :class:`GeminiClient` calls are
intercepted. Pins the behaviours the SwiftUI app depends on:

* budget guard fails the job *before* any upload, with the
  ``cloud_transcription`` step carrying the user-facing message;
* a successful run writes the transcript, the usage records, and the
  final speaker/title context the runner persists;
* transient cloud failures fall back to the local pipeline
  (``_run_cloud_transcription`` returns ``None`` + a warning).
"""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

from cloud_transcription import CloudTranscriptionError
from ekovideo_engine.events import collect_events
from ekovideo_engine.models import JobRequest
from ekovideo_engine.pipeline import TranscriptionPipeline


@dataclass
class _StubResult:
    returncode: int = 0
    stdout: str = ""
    stderr: str = ""


def _make_cloud_request(workspace: Path, source: Path, **tx_overrides) -> JobRequest:
    return JobRequest.from_dict(
        {
            "source_path": str(source),
            "output_dir": str(workspace),
            "mode": "transcribe",
            "workspace_dir": str(workspace),
            "transcription_settings": {
                "transcription_engine": "cloud",
                "cloud_model": "gemini-2.5-flash",
                "cloud_api_key": "test-key",
                "vad_enabled": False,
                "multipass_enabled": False,
                "diarization_enabled": False,
                "output_format": "txt",
                "language": "fr",
                **tx_overrides,
            },
        }
    )


def _gemini_payload() -> dict:
    body = {
        "title": "Comité produit",
        "speakers": [{"label": "Intervenant 1", "name": "Jean Dupont"}],
        "technical_terms": ["Odoo"],
        "segments": [
            {
                "start": "00:01",
                "end": "00:06",
                "speaker": "Intervenant 1",
                "text": "Bonjour à tous, on démarre.",
            },
            {
                "start": "00:06",
                "end": "00:11",
                "speaker": "Intervenant 2",
                "text": "Bonjour Jean.",
            },
        ],
        "uncertain": [
            {"timestamp": "00:09", "text": "Bonjour Jean.", "reason": "voix faible"}
        ],
    }
    return {
        "candidates": [{"content": {"parts": [{"text": json.dumps(body)}]}}],
        "usageMetadata": {
            "promptTokenCount": 9_000,
            "candidatesTokenCount": 1_500,
            "thoughtsTokenCount": 100,
        },
    }


def _fake_subprocess_run(workspace: Path, duration_seconds: float = 600.0):
    """Stub ffprobe (duration JSON) + ffmpeg (touch the mp3)."""

    def runner(cmd, **kwargs):
        binary = Path(cmd[0]).name
        if binary == "ffprobe":
            return _StubResult(
                stdout=json.dumps({"format": {"duration": str(duration_seconds)}})
            )
        if binary == "ffmpeg":
            # Last positional argument is the output path.
            Path(cmd[-1]).write_bytes(b"mp3")
            return _StubResult()
        raise AssertionError(f"unexpected subprocess: {cmd!r}")

    return runner


class _FakeClient:
    """GeminiClient stand-in. Class-level hooks let each test tune
    behaviour without re-plumbing the constructor patch."""

    response: dict = {}
    fail_with: CloudTranscriptionError | None = None
    deleted: list = []

    def __init__(self, api_key: str, **kwargs):
        if not (api_key or "").strip():
            raise CloudTranscriptionError("clé vide", code="cloud_auth")

    def upload_audio(self, path, display_name=""):
        if self.fail_with is not None:
            raise self.fail_with
        return {"uri": "files/abc", "name": "files/abc", "state": "ACTIVE", "mimeType": "audio/mp3"}

    def wait_until_active(self, info, **kwargs):
        return info

    def generate_transcription(self, **kwargs):
        return self.response

    def delete_file(self, info):
        type(self).deleted.append(info)


class CloudPipelineTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.workspace = Path(self._tmp.name)
        self.source = self.workspace / "meeting.mp4"
        self.source.write_bytes(b"fake video")
        _FakeClient.response = _gemini_payload()
        _FakeClient.fail_with = None
        _FakeClient.deleted = []

    def tearDown(self):
        self._tmp.cleanup()

    def _run(self, request: JobRequest):
        events, sink = collect_events()
        pipeline = TranscriptionPipeline(request, sink)
        with patch(
            "ekovideo_engine.pipeline.subprocess.run",
            side_effect=_fake_subprocess_run(self.workspace),
        ), patch("ekovideo_engine.pipeline.GeminiClient", _FakeClient):
            results = pipeline.run(str(self.source))
        return pipeline, results, events

    def test_successful_cloud_run_writes_transcript_and_usage(self):
        request = _make_cloud_request(self.workspace, self.source)
        pipeline, results, events = self._run(request)

        by_name = {r.name: r for r in results}
        self.assertIn("cloud_transcription", by_name)
        self.assertTrue(by_name["cloud_transcription"].ok)
        self.assertIn("transcript", by_name)
        transcript = Path(by_name["transcript"].artifact_path)
        self.assertTrue(transcript.exists())
        text = transcript.read_text(encoding="utf-8")
        # The label → name mapping is applied to the written file.
        self.assertIn("Jean Dupont", text)
        self.assertNotIn("Intervenant 1", text)

        # Usage tracked for the runner to persist, thinking tokens
        # folded into output.
        self.assertEqual(len(pipeline.cloud_usage_records), 1)
        record = pipeline.cloud_usage_records[0]
        self.assertEqual(record["input_tokens"], 9_000)
        self.assertEqual(record["output_tokens"], 1_600)
        self.assertGreater(record["cost_usd"], 0)
        self.assertTrue(any(e["event"] == "usage" for e in events))

        # Context for the rename sheet.
        self.assertEqual(pipeline.final_title, "Comité produit")
        self.assertIn("Jean Dupont", pipeline.final_speaker_map)
        self.assertIn("Odoo", pipeline.final_technical_terms)

        # Review markdown surfaces the uncertain passage.
        self.assertIn("review", by_name)
        review_text = Path(by_name["review"].artifact_path).read_text(encoding="utf-8")
        self.assertIn("voix faible", review_text)

        # Remote file cleaned up.
        self.assertEqual(len(_FakeClient.deleted), 1)

    def test_budget_guard_blocks_before_upload(self):
        request = _make_cloud_request(
            self.workspace, self.source, cloud_budget_monthly_usd=0.05
        )
        events, sink = collect_events()
        pipeline = TranscriptionPipeline(request, sink)

        class _FakeDb:
            def month_api_spend_usd(self):
                return 0.049  # almost exhausted: any estimate crosses

        with patch(
            "ekovideo_engine.pipeline.subprocess.run",
            side_effect=_fake_subprocess_run(self.workspace),
        ), patch("ekovideo_engine.pipeline.GeminiClient", _FakeClient), patch(
            "ekovideo_engine.library.database", return_value=_FakeDb()
        ):
            results = pipeline.run(str(self.source))

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].name, "cloud_transcription")
        self.assertFalse(results[0].ok)
        self.assertIn("Budget cloud mensuel atteint", results[0].error)
        # Nothing was uploaded, nothing was spent.
        self.assertEqual(pipeline.cloud_usage_records, [])
        self.assertEqual(_FakeClient.deleted, [])

    def test_transient_cloud_failure_falls_back_to_local(self):
        _FakeClient.fail_with = CloudTranscriptionError(
            "réseau coupé", code="cloud_network"
        )
        request = _make_cloud_request(self.workspace, self.source)
        events, sink = collect_events()
        pipeline = TranscriptionPipeline(request, sink)
        with patch(
            "ekovideo_engine.pipeline.subprocess.run",
            side_effect=_fake_subprocess_run(self.workspace),
        ), patch("ekovideo_engine.pipeline.GeminiClient", _FakeClient):
            fallback = pipeline._run_cloud_transcription(
                str(self.source), self.workspace
            )
        self.assertIsNone(fallback)
        warnings = [e for e in events if e["event"] == "warning"]
        self.assertTrue(
            any("bascule" in (w.get("message") or "") for w in warnings),
            [w.get("message") for w in warnings],
        )

    def test_auth_failure_is_fatal_not_fallback(self):
        _FakeClient.fail_with = CloudTranscriptionError(
            "clé refusée", code="cloud_auth"
        )
        request = _make_cloud_request(self.workspace, self.source)
        _, sink = collect_events()
        pipeline = TranscriptionPipeline(request, sink)
        with patch(
            "ekovideo_engine.pipeline.subprocess.run",
            side_effect=_fake_subprocess_run(self.workspace),
        ), patch("ekovideo_engine.pipeline.GeminiClient", _FakeClient):
            results = pipeline._run_cloud_transcription(
                str(self.source), self.workspace
            )
        self.assertIsNotNone(results)
        self.assertFalse(results[0].ok)
        self.assertIn("clé refusée", results[0].error)


if __name__ == "__main__":
    unittest.main()
