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


def _gemini_payload(title: str = "Comité produit") -> dict:
    body = {
        "title": title,
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


class _FakeProvider:
    """CloudProvider stand-in returned by a patched ``get_cloud_provider``.
    Class-level hooks let each test tune behaviour without re-plumbing
    the factory patch."""

    response: dict = {}
    fail_with: CloudTranscriptionError | None = None
    fail_times: int = 0  # transient (retryable) failures before succeeding
    fail_for_model: str | None = None  # always 503 for this model id
    fail_indices: set = set()  # always 503 for these chunk indices
    models_seen: list[str] = []
    indices_seen: list[int] = []
    calls: int = 0

    def __init__(self, *args, **kwargs):
        pass

    def transcribe(self, audio_path, *, model_id, context):
        type(self).calls += 1
        type(self).models_seen.append(model_id)
        type(self).indices_seen.append(context.chunk_index)
        if context.chunk_index in type(self).fail_indices:
            raise CloudTranscriptionError(
                "Erreur API Gemini (HTTP 503) : high demand", status=503
            )
        if type(self).fail_for_model and model_id == type(self).fail_for_model:
            raise CloudTranscriptionError(
                "Erreur API Gemini (HTTP 503) : high demand", status=503
            )
        if type(self).fail_times > 0:
            type(self).fail_times -= 1
            raise CloudTranscriptionError(
                "Erreur API Gemini (HTTP 503) : high demand", status=503
            )
        if self.fail_with is not None:
            raise self.fail_with
        from cloud_transcription import parse_cloud_response

        return parse_cloud_response(
            type(self).response,
            model_id=model_id,
            chunk_offset_seconds=context.chunk_offset_seconds,
        )


def _fake_factory(provider_id, api_key, **kwargs):
    return _FakeProvider()


class CloudPipelineTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.workspace = Path(self._tmp.name)
        self.source = self.workspace / "meeting.mp4"
        self.source.write_bytes(b"fake video")
        _FakeProvider.response = _gemini_payload()
        _FakeProvider.fail_with = None
        _FakeProvider.fail_times = 0
        _FakeProvider.fail_for_model = None
        _FakeProvider.fail_indices = set()
        _FakeProvider.models_seen = []
        _FakeProvider.indices_seen = []
        _FakeProvider.calls = 0

    def tearDown(self):
        self._tmp.cleanup()

    def _run_cloud(self, request: JobRequest, duration_seconds: float = 600.0):
        """Drive just the cloud path with a chosen media duration (to
        control the chunk count). Returns (pipeline, results, events)."""
        events, sink = collect_events()
        pipeline = TranscriptionPipeline(request, sink)
        with patch(
            "ekovideo_engine.pipeline.subprocess.run",
            side_effect=_fake_subprocess_run(self.workspace, duration_seconds),
        ), patch("ekovideo_engine.pipeline.get_cloud_provider", _fake_factory), patch(
            "ekovideo_engine.pipeline.time.sleep"
        ):
            results = pipeline._run_cloud_transcription(
                str(self.source), self.workspace
            )
        return pipeline, results, events

    def _run(self, request: JobRequest):
        events, sink = collect_events()
        pipeline = TranscriptionPipeline(request, sink)
        with patch(
            "ekovideo_engine.pipeline.subprocess.run",
            side_effect=_fake_subprocess_run(self.workspace),
        ), patch("ekovideo_engine.pipeline.get_cloud_provider", _fake_factory):
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
        ), patch("ekovideo_engine.pipeline.get_cloud_provider", _fake_factory), patch(
            "ekovideo_engine.library.database", return_value=_FakeDb()
        ):
            results = pipeline.run(str(self.source))

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].name, "cloud_transcription")
        self.assertFalse(results[0].ok)
        self.assertIn("Budget cloud mensuel atteint", results[0].error)
        # Nothing was uploaded, nothing was spent.
        self.assertEqual(pipeline.cloud_usage_records, [])

    def test_transient_cloud_failure_falls_back_to_local(self):
        # A *persistent* retryable failure exhausts the retries, then
        # falls back to local so the user still gets a transcript.
        _FakeProvider.fail_with = CloudTranscriptionError(
            "réseau coupé", code="cloud_network"
        )
        request = _make_cloud_request(self.workspace, self.source)
        events, sink = collect_events()
        pipeline = TranscriptionPipeline(request, sink)
        with patch(
            "ekovideo_engine.pipeline.subprocess.run",
            side_effect=_fake_subprocess_run(self.workspace),
        ), patch("ekovideo_engine.pipeline.get_cloud_provider", _fake_factory), patch(
            "ekovideo_engine.pipeline.time.sleep"
        ) as sleeper:
            fallback = pipeline._run_cloud_transcription(
                str(self.source), self.workspace
            )
        self.assertIsNone(fallback)
        # It retried (with backoff) before giving up.
        self.assertEqual(sleeper.call_count, 3)
        warnings = [e for e in events if e["event"] == "warning"]
        self.assertTrue(
            any("bascule" in (w.get("message") or "") for w in warnings),
            [w.get("message") for w in warnings],
        )

    def test_retryable_failure_retries_then_succeeds_on_cloud(self):
        # The reported bug: a momentary HTTP 503 must NOT drop the whole
        # meeting to local. Two transient failures, then success — stays
        # on cloud, no fallback.
        _FakeProvider.fail_times = 2
        request = _make_cloud_request(self.workspace, self.source)
        events, sink = collect_events()
        pipeline = TranscriptionPipeline(request, sink)
        with patch(
            "ekovideo_engine.pipeline.subprocess.run",
            side_effect=_fake_subprocess_run(self.workspace),
        ), patch("ekovideo_engine.pipeline.get_cloud_provider", _fake_factory), patch(
            "ekovideo_engine.pipeline.time.sleep"
        ) as sleeper:
            results = pipeline._run_cloud_transcription(
                str(self.source), self.workspace
            )
        self.assertIsNotNone(results)  # stayed on cloud
        self.assertEqual(_FakeProvider.calls, 3)  # 2 failures + 1 success
        self.assertEqual(sleeper.call_count, 2)
        retries = [e for e in events if e.get("code") == "cloud_retry"]
        self.assertEqual(len(retries), 2)
        # No "bascule sur le moteur local" warning was emitted.
        self.assertFalse(
            any("bascule" in (e.get("message") or "") for e in events)
        )

    def test_capacity_rationed_model_fails_over_to_ga_model(self):
        # The reported case: Gemini 3.5 Flash (preview) is persistently
        # 503'd. Instead of dropping to local, the engine fails over to a
        # GA model (2.5 Flash) and stays on cloud.
        _FakeProvider.fail_for_model = "gemini-3.5-flash"
        request = _make_cloud_request(
            self.workspace, self.source, cloud_model="gemini-3.5-flash"
        )
        events, sink = collect_events()
        pipeline = TranscriptionPipeline(request, sink)
        with patch(
            "ekovideo_engine.pipeline.subprocess.run",
            side_effect=_fake_subprocess_run(self.workspace),
        ), patch("ekovideo_engine.pipeline.get_cloud_provider", _fake_factory), patch(
            "ekovideo_engine.pipeline.time.sleep"
        ):
            results = pipeline._run_cloud_transcription(
                str(self.source), self.workspace
            )
        self.assertIsNotNone(results)  # stayed on cloud, no local fallback
        # The GA fallback model actually ran.
        self.assertIn("gemini-2.5-flash", _FakeProvider.models_seen)
        cloud_step = next(r for r in results if r.name == "cloud_transcription")
        self.assertEqual(cloud_step.model, "gemini-2.5-flash")
        self.assertTrue(
            any(e.get("code") == "cloud_model_fallback" for e in events)
        )
        self.assertFalse(
            any("bascule sur le moteur local" in (e.get("message") or "") for e in events)
        )

    def test_all_cloud_models_down_falls_back_to_local(self):
        # Both the chosen preview model and its GA fallback are 503'd →
        # only then do we drop to local.
        _FakeProvider.fail_with = CloudTranscriptionError(
            "Erreur API Gemini (HTTP 503)", status=503
        )
        request = _make_cloud_request(
            self.workspace, self.source, cloud_model="gemini-3.5-flash"
        )
        events, sink = collect_events()
        pipeline = TranscriptionPipeline(request, sink)
        with patch(
            "ekovideo_engine.pipeline.subprocess.run",
            side_effect=_fake_subprocess_run(self.workspace),
        ), patch("ekovideo_engine.pipeline.get_cloud_provider", _fake_factory), patch(
            "ekovideo_engine.pipeline.time.sleep"
        ):
            fallback = pipeline._run_cloud_transcription(
                str(self.source), self.workspace
            )
        self.assertIsNone(fallback)
        # Both models were attempted before giving up.
        self.assertIn("gemini-3.5-flash", _FakeProvider.models_seen)
        self.assertIn("gemini-2.5-flash", _FakeProvider.models_seen)
        self.assertTrue(
            any("bascule sur le moteur local" in (e.get("message") or "") for e in events)
        )

    def test_cloud_title_uses_client_company_not_ekonum(self):
        # Gemini titled the meeting after our own company; the pipeline
        # rewrites it to "Client - Sujet" using the calendar partner.
        _FakeProvider.response = _gemini_payload("Ekonum - Présentation des modules RH")
        request = _make_cloud_request(self.workspace, self.source)
        request.odoo_meeting_metadata = {
            "partners": [{"name": "Ekonum"}, {"name": "Acritec"}]
        }
        pipeline, _results, _events = self._run(request)
        self.assertEqual(
            pipeline.final_title, "Acritec - Présentation des modules RH"
        )

    def test_cloud_title_strips_ekonum_when_no_client_resolved(self):
        # No client resolvable → at least drop the useless "Ekonum -"
        # self-prefix instead of shipping it.
        _FakeProvider.response = _gemini_payload("Ekonum - Audit ERP")
        request = _make_cloud_request(self.workspace, self.source)
        pipeline, _results, _events = self._run(request)
        self.assertEqual(pipeline.final_title, "Audit ERP")

    def test_stay_cloud_policy_never_leaves_the_chosen_model(self):
        # "stay_cloud": a persistent 503 must NOT fail over to another
        # model nor drop to local — the job fails on the chosen model.
        _FakeProvider.fail_for_model = "gemini-3.5-flash"
        request = _make_cloud_request(
            self.workspace,
            self.source,
            cloud_model="gemini-3.5-flash",
            cloud_unavailable_policy="stay_cloud",
        )
        events, sink = collect_events()
        pipeline = TranscriptionPipeline(request, sink)
        with patch(
            "ekovideo_engine.pipeline.subprocess.run",
            side_effect=_fake_subprocess_run(self.workspace),
        ), patch("ekovideo_engine.pipeline.get_cloud_provider", _fake_factory), patch(
            "ekovideo_engine.pipeline.time.sleep"
        ):
            results = pipeline._run_cloud_transcription(
                str(self.source), self.workspace
            )
        # Hard failure (not None → no local), and only the chosen model
        # was ever contacted (no fail-over to 2.5-flash).
        self.assertIsNotNone(results)
        cloud_step = next(r for r in results if r.name == "cloud_transcription")
        self.assertFalse(cloud_step.ok)
        self.assertEqual(set(_FakeProvider.models_seen), {"gemini-3.5-flash"})
        self.assertNotIn("gemini-2.5-flash", _FakeProvider.models_seen)
        self.assertFalse(
            any("bascule sur le moteur local" in (e.get("message") or "") for e in events)
        )

    def test_stay_cloud_uses_the_expanding_backoff(self):
        _FakeProvider.fail_for_model = "gemini-3.5-flash"
        request = _make_cloud_request(
            self.workspace,
            self.source,
            cloud_model="gemini-3.5-flash",
            cloud_unavailable_policy="stay_cloud",
        )
        events, sink = collect_events()
        pipeline = TranscriptionPipeline(request, sink)
        with patch(
            "ekovideo_engine.pipeline.subprocess.run",
            side_effect=_fake_subprocess_run(self.workspace),
        ), patch("ekovideo_engine.pipeline.get_cloud_provider", _fake_factory), patch(
            "ekovideo_engine.pipeline.time.sleep"
        ) as sleeper:
            pipeline._run_cloud_transcription(str(self.source), self.workspace)
        from ekovideo_engine.pipeline import _CLOUD_STAY_BACKOFF_SECONDS

        waits = [c.args[0] for c in sleeper.call_args_list]
        self.assertEqual(waits, list(_CLOUD_STAY_BACKOFF_SECONDS))
        self.assertEqual(waits[:3], [10, 20, 40])  # expanding, as asked

    def test_truncated_json_response_is_retryable(self):
        # The real 2.5-flash failure: Gemini returned a truncated JSON
        # payload ("Unterminated string"). That error is now retryable so
        # the engine gives it another go instead of bailing to local.
        from cloud_transcription import CloudTranscriptionError, parse_cloud_response

        with self.assertRaises(CloudTranscriptionError) as ctx:
            parse_cloud_response(
                {"candidates": [{"content": {"parts": [{"text": '{"segments": ['}]}}]},
                model_id="gemini-2.5-flash",
                chunk_offset_seconds=0,
            )
        self.assertTrue(ctx.exception.retryable)

    def test_failed_chunks_resume_from_cache_on_rerun(self):
        # 3-chunk meeting, chunk 2 fails → chunks 0/1 cached, run drops to
        # local. Rerun reuses 0/1 and only redoes chunk 2.
        request = _make_cloud_request(self.workspace, self.source)
        _FakeProvider.fail_indices = {2}
        _p1, r1, _e1 = self._run_cloud(request, duration_seconds=5400)
        self.assertIsNone(r1)  # chunk 2 failed → local fallback
        cache = self.workspace / "cloud_chunks"
        self.assertTrue((cache / "chunk_00.json").exists())
        self.assertTrue((cache / "chunk_01.json").exists())
        self.assertFalse((cache / "chunk_02.json").exists())

        _FakeProvider.fail_indices = set()
        _FakeProvider.indices_seen = []
        _FakeProvider.calls = 0
        _p2, r2, _e2 = self._run_cloud(request, duration_seconds=5400)
        self.assertIsNotNone(r2)  # completed on resume
        self.assertEqual(_FakeProvider.indices_seen, [2])  # only the failed one
        self.assertEqual(_FakeProvider.calls, 1)
        self.assertFalse(cache.exists())  # cleared on full success

    def test_partial_run_persists_speakers_terms_and_status(self):
        # Even when a later chunk fails, the speakers, vocabulary and
        # per-chunk status from the successful windows are saved to the
        # job — not lost until the (never-reached) final merge.
        from ekovideo_engine.library import database
        db = database()
        job_id = db.create_job(str(self.source), str(self.workspace), {})
        request = _make_cloud_request(self.workspace, self.source)
        request.library_job_id = job_id
        _FakeProvider.fail_indices = {2}
        self._run_cloud(request, duration_seconds=5400)

        row = db.get_job(job_id)
        speakers = json.loads(row.get("speaker_map_json") or "{}")
        terms = json.loads(row.get("technical_terms_json") or "[]")
        chunks = json.loads(row.get("cloud_chunks_json") or "[]")
        self.assertTrue(
            any("Jean Dupont" in k or "Jean Dupont" in v for k, v in speakers.items()),
            speakers,
        )
        self.assertIn("Odoo", terms)
        self.assertEqual([c["ok"] for c in chunks], [True, True, False])
        # A partial transcript was written too.
        self.assertTrue(row.get("transcript_path"))

    def test_cloud_chunk_status_records_failed_windows(self):
        # After a partial run the pipeline exposes per-chunk state so the
        # library can list which windows failed.
        request = _make_cloud_request(self.workspace, self.source)
        _FakeProvider.fail_indices = {2}
        pipeline, r, _e = self._run_cloud(request, duration_seconds=5400)
        self.assertIsNone(r)
        status = pipeline.cloud_chunk_status
        self.assertEqual([c["index"] for c in status], [0, 1, 2])
        self.assertEqual([c["ok"] for c in status], [True, True, False])

    def test_cloud_chunk_status_all_ok_on_full_success(self):
        request = _make_cloud_request(self.workspace, self.source)
        pipeline, r, _e = self._run_cloud(request, duration_seconds=5400)
        self.assertIsNotNone(r)
        self.assertTrue(all(c["ok"] for c in pipeline.cloud_chunk_status))
        self.assertEqual(len(pipeline.cloud_chunk_status), 3)

    def test_cloud_redo_chunks_forces_retranscription(self):
        # Seed cache for chunks 0/1 (chunk 2 fails), then rerun forcing
        # chunk 0 to redo — chunk 1 stays cached, chunks 0 and 2 run.
        request = _make_cloud_request(self.workspace, self.source)
        _FakeProvider.fail_indices = {2}
        self._run_cloud(request, duration_seconds=5400)

        _FakeProvider.fail_indices = set()
        _FakeProvider.indices_seen = []
        request.cloud_redo_chunks = [0]
        _p, r, _e = self._run_cloud(request, duration_seconds=5400)
        self.assertIsNotNone(r)
        self.assertEqual(sorted(_FakeProvider.indices_seen), [0, 2])

    def test_cache_cleared_after_full_success_so_rerun_is_fresh(self):
        # A fully successful run leaves no cache — a later rerun (e.g. to
        # apply new vocabulary) re-transcribes everything.
        request = _make_cloud_request(self.workspace, self.source)
        _p, r, _e = self._run_cloud(request, duration_seconds=5400)
        self.assertIsNotNone(r)
        self.assertFalse((self.workspace / "cloud_chunks").exists())

        _FakeProvider.indices_seen = []
        self._run_cloud(request, duration_seconds=5400)
        self.assertEqual(sorted(_FakeProvider.indices_seen), [0, 1, 2])

    def test_non_retryable_failure_falls_back_without_retrying(self):
        # A 4xx (e.g. bad request) is not transient — fall back at once,
        # don't waste time retrying.
        _FakeProvider.fail_with = CloudTranscriptionError(
            "Erreur API Gemini (HTTP 400)", status=400
        )
        request = _make_cloud_request(self.workspace, self.source)
        events, sink = collect_events()
        pipeline = TranscriptionPipeline(request, sink)
        with patch(
            "ekovideo_engine.pipeline.subprocess.run",
            side_effect=_fake_subprocess_run(self.workspace),
        ), patch("ekovideo_engine.pipeline.get_cloud_provider", _fake_factory), patch(
            "ekovideo_engine.pipeline.time.sleep"
        ) as sleeper:
            fallback = pipeline._run_cloud_transcription(
                str(self.source), self.workspace
            )
        self.assertIsNone(fallback)
        sleeper.assert_not_called()

    def test_stt_provider_records_per_hour_usage(self):
        # A dedicated STT provider (per-hour billing, no full bundle):
        # transcript is written, usage is attributed to the provider and
        # billed by duration, and the missing-venv enrichment is skipped
        # gracefully (no crash, transcript still ships).
        from cloud_transcription import CloudChunkResult, CloudUsage, cost_for_duration

        model = "assemblyai-universal-3"

        class _STTProvider:
            def __init__(self, *a, **k):
                pass

            def transcribe(self, audio_path, *, model_id, context):
                result = CloudChunkResult(
                    segments=[
                        {"start": 1.0 + context.chunk_offset_seconds,
                         "end": 4.0 + context.chunk_offset_seconds,
                         "speaker": "Intervenant 1", "text": "Bonjour."},
                    ],
                    usage=CloudUsage(
                        model=model_id,
                        cost_usd=cost_for_duration(model_id, context.chunk_duration_seconds),
                    ),
                )
                return result

        request = _make_cloud_request(self.workspace, self.source, cloud_model=model)
        events, sink = collect_events()
        pipeline = TranscriptionPipeline(request, sink)
        with patch(
            "ekovideo_engine.pipeline.subprocess.run",
            side_effect=_fake_subprocess_run(self.workspace, duration_seconds=600.0),
        ), patch(
            "ekovideo_engine.pipeline.get_cloud_provider",
            lambda *a, **k: _STTProvider(),
        ):
            results = pipeline.run(str(self.source))

        by_name = {r.name: r for r in results}
        self.assertTrue(by_name["cloud_transcription"].ok)
        self.assertTrue(Path(by_name["transcript"].artifact_path).exists())
        self.assertEqual(len(pipeline.cloud_usage_records), 1)
        record = pipeline.cloud_usage_records[0]
        self.assertEqual(record["provider"], "assemblyai")
        # 600 s of 0.21 $/h.
        self.assertAlmostEqual(record["cost_usd"], 0.035, places=4)
        self.assertEqual(record["input_tokens"], 0)
        # No venv in the test env → enrichment skipped, transcript intact.
        self.assertTrue(pipeline.final_segments)

    def test_stt_cloud_enrichment_adds_title_and_records_usage(self):
        # STT provider + a configured enrich key → the cheap cloud text
        # pass fills title/speakers and its (small) cost is tracked as a
        # separate "enrich" usage record.
        from cloud_transcription import CloudChunkResult, CloudUsage, cost_for_duration

        model = "assemblyai-universal-3"

        class _STTProvider:
            def __init__(self, *a, **k):
                pass

            def transcribe(self, audio_path, *, model_id, context):
                return CloudChunkResult(
                    segments=[{"start": 1.0, "end": 4.0, "speaker": "Intervenant 1", "text": "Bonjour."}],
                    usage=CloudUsage(
                        model=model_id,
                        cost_usd=cost_for_duration(model_id, context.chunk_duration_seconds),
                    ),
                )

        def fake_enrich(api_key, model_id, transcript_text, **kwargs):
            payload = {
                "title": "Comité produit",
                "speakers": {"Intervenant 1": "Jean Dupont"},
                "technical_terms": ["Odoo"],
                "corrections": [],
                "uncertain_passages": [],
            }
            return payload, CloudUsage(model=model_id, input_tokens=8000, output_tokens=300, cost_usd=0.0045)

        request = _make_cloud_request(
            self.workspace, self.source,
            cloud_model=model,
            cloud_enrich_model="gemini-3.1-flash-lite",
            cloud_enrich_api_key="gem-key",
        )
        events, sink = collect_events()
        pipeline = TranscriptionPipeline(request, sink)
        with patch(
            "ekovideo_engine.pipeline.subprocess.run",
            side_effect=_fake_subprocess_run(self.workspace, duration_seconds=600.0),
        ), patch(
            "ekovideo_engine.pipeline.get_cloud_provider", lambda *a, **k: _STTProvider()
        ), patch(
            "ekovideo_engine.pipeline.enrich_transcript_via_gemini", side_effect=fake_enrich
        ):
            results = pipeline.run(str(self.source))

        by_name = {r.name: r for r in results}
        self.assertTrue(by_name["cloud_transcription"].ok)
        self.assertEqual(pipeline.final_title, "Comité produit")
        self.assertIn("Jean Dupont", pipeline.final_speaker_map)
        # Two usage records: the STT chunk + the enrichment.
        steps = {r["step"] for r in pipeline.cloud_usage_records}
        self.assertIn("enrich", steps)
        transcript = Path(by_name["transcript"].artifact_path).read_text(encoding="utf-8")
        self.assertIn("Jean Dupont", transcript)

    def test_auth_failure_is_fatal_not_fallback(self):
        _FakeProvider.fail_with = CloudTranscriptionError(
            "clé refusée", code="cloud_auth"
        )
        request = _make_cloud_request(self.workspace, self.source)
        _, sink = collect_events()
        pipeline = TranscriptionPipeline(request, sink)
        with patch(
            "ekovideo_engine.pipeline.subprocess.run",
            side_effect=_fake_subprocess_run(self.workspace),
        ), patch("ekovideo_engine.pipeline.get_cloud_provider", _fake_factory):
            results = pipeline._run_cloud_transcription(
                str(self.source), self.workspace
            )
        self.assertIsNotNone(results)
        self.assertFalse(results[0].ok)
        self.assertIn("clé refusée", results[0].error)


if __name__ == "__main__":
    unittest.main()
