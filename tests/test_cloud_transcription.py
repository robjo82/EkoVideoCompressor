"""Tests for the cloud (Gemini) transcription path.

Pins the contracts the rest of the app builds on:

* cost math — the budget guard and the SwiftUI estimate both rely on
  :func:`estimate_cloud_cost` / :func:`compute_cost_usd` being
  deterministic and conservative;
* response parsing — timestamps, speaker maps, usage counters
  (thinking tokens billed as output);
* chunk planning and merge — long meetings must keep a monotonic
  timeline and a consistent speaker map;
* the ``api_usage`` ledger and the settings redaction that keeps the
  API key out of ``settings_json``.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from cloud_transcription import (
    CLOUD_PROVIDERS,
    CLOUD_TRANSCRIPTION_MODELS,
    CLOUD_CHUNK_SECONDS,
    CloudPromptContext,
    CloudTranscriptionError,
    CloudChunkResult,
    CloudUsage,
    DEFAULT_CLOUD_MODEL,
    GeminiClient,
    build_cloud_audio_cmd,
    build_cloud_prompt,
    canonical_cloud_model_id,
    cloud_model_entry,
    compute_cost_usd,
    cost_for_duration,
    estimate_cloud_cost,
    get_cloud_provider,
    merge_chunk_results,
    parse_cloud_response,
    parse_cloud_timestamp,
    plan_audio_chunks,
    provider_for_model,
)
from database_manager import DatabaseManager, _redact_settings


class CostMathTest(unittest.TestCase):
    def test_known_model_prices_apply(self):
        # 1M audio-input tokens + 1M output tokens on 2.5 Flash:
        # $1.00 + $2.50 exactly.
        self.assertAlmostEqual(
            compute_cost_usd("gemini-2.5-flash", 1_000_000, 1_000_000), 3.50
        )

    def test_unknown_model_bills_at_most_expensive_known_rate(self):
        unknown = compute_cost_usd("gemini-9-ultra", 1_000_000, 1_000_000)
        most_expensive = max(
            compute_cost_usd(entry["id"], 1_000_000, 1_000_000)
            for entry in CLOUD_TRANSCRIPTION_MODELS
        )
        self.assertAlmostEqual(unknown, most_expensive)

    def test_estimate_uses_audio_token_rate(self):
        estimate = estimate_cloud_cost(3600, "gemini-2.5-flash")
        # 32 tokens/s × 3600 s = 115 200, plus the prompt overhead.
        self.assertEqual(estimate["input_tokens"], 115_200 + 400)
        self.assertGreater(estimate["cost_usd"], 0)
        self.assertEqual(estimate["model"], "gemini-2.5-flash")

    def test_blank_model_falls_back_to_default(self):
        self.assertEqual(canonical_cloud_model_id(""), DEFAULT_CLOUD_MODEL)
        self.assertEqual(canonical_cloud_model_id("  "), DEFAULT_CLOUD_MODEL)

    def test_unknown_model_keeps_its_id(self):
        entry = cloud_model_entry("gemini-9-ultra")
        self.assertEqual(entry["id"], "gemini-9-ultra")


class TimestampTest(unittest.TestCase):
    def test_minute_second(self):
        self.assertEqual(parse_cloud_timestamp("12:34"), 12 * 60 + 34)

    def test_hour_minute_second(self):
        self.assertEqual(parse_cloud_timestamp("1:02:03"), 3723)

    def test_bare_seconds_and_garbage(self):
        self.assertEqual(parse_cloud_timestamp(42), 42.0)
        self.assertEqual(parse_cloud_timestamp("7.5"), 7.5)
        self.assertIsNone(parse_cloud_timestamp("n/a"))
        self.assertIsNone(parse_cloud_timestamp(""))


class PerModelChunkingTest(unittest.TestCase):
    """Chunking is a per-provider technical ceiling, engine-decided:
    dedicated STT send the whole meeting (coherent diarisation); the
    LLMs window at their token/size limit."""

    def test_chunk_seconds_per_provider(self):
        from cloud_transcription import chunk_seconds_for_model

        self.assertEqual(chunk_seconds_for_model("gemini-3.5-flash"), 30 * 60)
        self.assertEqual(chunk_seconds_for_model("gpt-4o-transcribe-diarize"), 40 * 60)
        self.assertEqual(chunk_seconds_for_model("gladia-solaria-3"), 120 * 60)
        self.assertEqual(chunk_seconds_for_model("assemblyai-universal-3"), 180 * 60)
        self.assertEqual(chunk_seconds_for_model("deepgram-nova-3"), 120 * 60)

    def test_stt_keeps_normal_meeting_whole(self):
        from cloud_transcription import chunk_seconds_for_model

        # 90-min meeting on Gladia → a single job (vs 3 windows on Gemini).
        gladia = plan_audio_chunks(90 * 60, chunk_seconds_for_model("gladia-solaria-3"))
        self.assertEqual(gladia, [(0.0, 90 * 60)])
        gemini = plan_audio_chunks(90 * 60, chunk_seconds_for_model("gemini-3.5-flash"))
        self.assertEqual(len(gemini), 3)

    def test_stt_splits_only_beyond_provider_limit(self):
        from cloud_transcription import chunk_seconds_for_model

        # 150 min > Gladia's 135-min ceiling → 2 windows, each < limit.
        chunks = plan_audio_chunks(150 * 60, chunk_seconds_for_model("gladia-solaria-3"))
        self.assertEqual(len(chunks), 2)
        for start, end in chunks:
            self.assertLessEqual(end - start, 135 * 60)


class ChunkPlanTest(unittest.TestCase):
    def test_short_meeting_stays_whole(self):
        self.assertEqual(plan_audio_chunks(1200), [(0.0, 1200)])

    def test_slightly_over_threshold_stays_whole(self):
        # 32 minutes: not worth a 30 + 2 split.
        self.assertEqual(plan_audio_chunks(32 * 60), [(0.0, 32 * 60)])

    def test_long_meeting_splits_evenly(self):
        chunks = plan_audio_chunks(2 * 3600)  # 2 h
        self.assertEqual(len(chunks), 4)
        self.assertEqual(chunks[0][0], 0.0)
        self.assertEqual(chunks[-1][1], 7200)
        for start, end in chunks:
            self.assertLessEqual(end - start, CLOUD_CHUNK_SECONDS + 1)
        # Contiguous coverage, no gaps.
        for (_, prev_end), (next_start, _) in zip(chunks, chunks[1:]):
            self.assertAlmostEqual(prev_end, next_start)


class CloudAudioCmdTest(unittest.TestCase):
    def test_mono_16k_mp3(self):
        cmd = build_cloud_audio_cmd("ffmpeg", "in.mp4", "out.mp3")
        self.assertIn("libmp3lame", cmd)
        self.assertIn("16000", cmd)
        self.assertNotIn("-ss", cmd)

    def test_window_bounds(self):
        cmd = build_cloud_audio_cmd(
            "ffmpeg", "in.mp4", "out.mp3", start_seconds=1800, end_seconds=3600
        )
        self.assertIn("-ss", cmd)
        self.assertIn("1800.00", cmd)
        self.assertIn("-to", cmd)
        self.assertIn("3600.00", cmd)


class PromptTest(unittest.TestCase):
    def test_includes_glossary_and_speakers(self):
        prompt = build_cloud_prompt(
            glossary_terms=["Odoo", "EkoVidéo"],
            expected_speaker_names=["Robin Joseph"],
            meeting_context="Comité produit",
        )
        self.assertIn("Odoo", prompt)
        self.assertIn("Robin Joseph", prompt)
        self.assertIn("Comité produit", prompt)

    def test_chunk_context_propagates_known_speakers(self):
        prompt = build_cloud_prompt(
            chunk_index=1,
            chunk_count=3,
            chunk_offset_seconds=1800,
            known_speakers={"Intervenant 1": "Jean Dupont"},
            previous_tail="[Jean Dupont] On reprend après la pause.",
        )
        self.assertIn("partie 2 sur 3", prompt)
        self.assertIn("Intervenant 1 = Jean Dupont", prompt)
        self.assertIn("On reprend après la pause.", prompt)
        self.assertIn("minute 30", prompt)


def _gemini_payload(segments: list[dict], **extra) -> dict:
    body = {
        "title": extra.get("title", "Réunion produit"),
        "speakers": extra.get(
            "speakers", [{"label": "Intervenant 1", "name": "Jean Dupont"}]
        ),
        "technical_terms": extra.get("technical_terms", ["Odoo"]),
        "segments": segments,
        "uncertain": extra.get("uncertain", []),
    }
    return {
        "candidates": [{"content": {"parts": [{"text": json.dumps(body)}]}}],
        "usageMetadata": {
            "promptTokenCount": extra.get("input_tokens", 10_000),
            "candidatesTokenCount": extra.get("output_tokens", 2_000),
            "thoughtsTokenCount": extra.get("thinking_tokens", 500),
        },
    }


class ParseResponseTest(unittest.TestCase):
    def test_parses_segments_and_offsets_timeline(self):
        payload = _gemini_payload(
            [
                {"start": "00:05", "end": "00:12", "speaker": "Intervenant 1", "text": "Bonjour à tous."},
                {"start": "00:12", "end": "00:20", "speaker": "Intervenant 2", "text": "Bonjour Jean."},
            ]
        )
        result = parse_cloud_response(
            payload, model_id="gemini-2.5-flash", chunk_offset_seconds=1800
        )
        self.assertEqual(len(result.segments), 2)
        self.assertEqual(result.segments[0]["start"], 1805.0)
        self.assertEqual(result.segments[0]["speaker"], "Intervenant 1")
        self.assertEqual(result.title, "Réunion produit")
        self.assertEqual(result.speakers["Intervenant 1"], "Jean Dupont")

    def test_thinking_tokens_billed_as_output(self):
        payload = _gemini_payload(
            [{"start": "00:01", "speaker": "A", "text": "Oui."}],
            input_tokens=1000,
            output_tokens=200,
            thinking_tokens=300,
        )
        result = parse_cloud_response(payload, model_id="gemini-2.5-flash")
        self.assertEqual(result.usage.input_tokens, 1000)
        self.assertEqual(result.usage.output_tokens, 500)
        self.assertAlmostEqual(
            result.usage.cost_usd,
            compute_cost_usd("gemini-2.5-flash", 1000, 500),
        )

    def test_missing_end_gets_synthesised(self):
        payload = _gemini_payload(
            [{"start": "00:10", "speaker": "A", "text": "Une phrase de six mots environ."}]
        )
        result = parse_cloud_response(payload, model_id="gemini-2.5-flash")
        self.assertGreater(result.segments[0]["end"], result.segments[0]["start"])

    def test_empty_candidates_raises(self):
        with self.assertRaises(CloudTranscriptionError):
            parse_cloud_response(
                {"candidates": [], "promptFeedback": {"blockReason": "SAFETY"}},
                model_id="gemini-2.5-flash",
            )

    def test_invalid_json_raises(self):
        payload = {
            "candidates": [{"content": {"parts": [{"text": "pas du json"}]}}]
        }
        with self.assertRaises(CloudTranscriptionError):
            parse_cloud_response(payload, model_id="gemini-2.5-flash")


class MergeChunksTest(unittest.TestCase):
    def test_merge_keeps_timeline_and_dedupes_terms(self):
        first = CloudChunkResult(
            segments=[{"start": 0.0, "end": 5.0, "speaker": "A", "text": "Un."}],
            speakers={"Intervenant 1": ""},
            technical_terms=["Odoo", "EkoVidéo"],
            title="Titre A",
            usage=CloudUsage("m", 100, 10, 0.001),
        )
        second = CloudChunkResult(
            segments=[{"start": 1800.0, "end": 1805.0, "speaker": "A", "text": "Deux."}],
            speakers={"Intervenant 1": "Jean Dupont"},
            technical_terms=["odoo", "JSON-2"],
            usage=CloudUsage("m", 200, 20, 0.002),
        )
        merged = merge_chunk_results([first, second])
        self.assertEqual([s["text"] for s in merged.segments], ["Un.", "Deux."])
        # The named mapping from a later chunk wins over the earlier
        # anonymous one.
        self.assertEqual(merged.speakers["Intervenant 1"], "Jean Dupont")
        self.assertEqual(merged.technical_terms, ["Odoo", "EkoVidéo", "JSON-2"])
        self.assertEqual(merged.title, "Titre A")
        self.assertEqual(merged.usage.input_tokens, 300)
        self.assertAlmostEqual(merged.usage.cost_usd, 0.003)


class EnrichmentTest(unittest.TestCase):
    """Cheap cloud text-enrichment of dedicated-STT transcripts."""

    def test_parse_enrich_response(self):
        from cloud_transcription import parse_enrich_response

        body = {
            "title": "Comité produit",
            "speakers": [{"label": "Intervenant 1", "name": "Jean Dupont"}],
            "technical_terms": ["Odoo"],
            "corrections": [{"original": "Odo", "replacement": "Odoo"}],
            "uncertain": [{"timestamp": "00:30", "text": "?", "reason": "bruit"}],
        }
        payload = {
            "candidates": [{"content": {"parts": [{"text": json.dumps(body)}]}}],
            "usageMetadata": {
                "promptTokenCount": 9000,
                "candidatesTokenCount": 300,
                "thoughtsTokenCount": 100,
            },
        }
        result, usage = parse_enrich_response(payload, model_id="gemini-3.1-flash-lite")
        self.assertEqual(result["title"], "Comité produit")
        self.assertEqual(result["speakers"], {"Intervenant 1": "Jean Dupont"})
        self.assertEqual(result["technical_terms"], ["Odoo"])
        self.assertEqual(len(result["corrections"]), 1)
        self.assertEqual(len(result["uncertain_passages"]), 1)
        # Enrichment is text-only and cheap: 400 output tokens on
        # Flash-Lite ($1.50/M) ≈ fractions of a cent.
        self.assertLess(usage.cost_usd, 0.02)
        self.assertEqual(usage.output_tokens, 400)

    def test_enrich_via_gemini_uses_text_endpoint(self):
        from cloud_transcription import enrich_transcript_via_gemini

        captured: dict = {}

        def opener(request, timeout=None):
            captured["url"] = request.full_url
            captured["body"] = json.loads(request.data.decode("utf-8"))

            class _R:
                headers = {}
                def read(self_):
                    return json.dumps({
                        "candidates": [{"content": {"parts": [{"text": json.dumps({"title": "T"})}]}}],
                        "usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 10},
                    }).encode()
                def __enter__(self_): return self_
                def __exit__(self_, *a): return False
            return _R()

        result, _ = enrich_transcript_via_gemini(
            "key", "gemini-3.1-flash-lite", "[Intervenant 1] (00:01) Bonjour.",
            glossary_terms=["Odoo"], opener=opener,
        )
        self.assertEqual(result["title"], "T")
        # Text-only: no audio fileData part, just the prompt text.
        parts = captured["body"]["contents"][0]["parts"]
        self.assertTrue(all("fileData" not in p for p in parts))
        self.assertIn("Bonjour", parts[0]["text"])


class GeminiClientTest(unittest.TestCase):
    def test_blank_key_refused_upfront(self):
        with self.assertRaises(CloudTranscriptionError) as ctx:
            GeminiClient("   ")
        self.assertEqual(ctx.exception.code, "cloud_auth")

    def test_check_access_lists_models(self):
        class _Response:
            headers = {}

            def read(self):
                return json.dumps(
                    {"models": [{"name": "models/gemini-2.5-flash"}]}
                ).encode("utf-8")

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        captured: dict = {}

        def opener(request, timeout=0):
            captured["url"] = request.full_url
            captured["api_key"] = request.get_header("X-goog-api-key")
            return _Response()

        client = GeminiClient("test-key", opener=opener)
        payload = client.check_access()
        self.assertTrue(payload["ok"])
        self.assertIn("gemini-2.5-flash", payload["models"])
        self.assertEqual(captured["api_key"], "test-key")
        self.assertIn("/v1beta/models", captured["url"])


class UsageLedgerTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db = DatabaseManager(Path(self._tmp.name) / "library.db")

    def tearDown(self):
        self._tmp.cleanup()

    def test_month_spend_aggregates_rows(self):
        self.db.add_api_usage(
            provider="gemini",
            model="gemini-2.5-flash",
            input_tokens=100_000,
            output_tokens=10_000,
            cost_usd=0.125,
            job_id=1,
            step="chunk_1/1",
        )
        self.db.add_api_usage(
            provider="gemini",
            model="gemini-2.5-flash",
            input_tokens=50_000,
            output_tokens=5_000,
            cost_usd=0.0625,
        )
        self.assertAlmostEqual(self.db.month_api_spend_usd(), 0.1875)
        summary = self.db.api_usage_summary()
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["calls"], 2)
        self.assertEqual(summary[0]["input_tokens"], 150_000)

    def test_other_month_does_not_count(self):
        self.db.add_api_usage(
            provider="gemini",
            model="m",
            input_tokens=1,
            output_tokens=1,
            cost_usd=42.0,
        )
        self.assertAlmostEqual(self.db.month_api_spend_usd("1999-01"), 0.0)

    def test_cloud_cost_denormalised_on_job_row(self):
        job_id = self.db.create_job("/tmp/a.mp4", "/tmp/ws", {})
        self.db.update_job_cloud_cost(job_id, 0.21, "gemini-3.5-flash")
        row = self.db.get_job(job_id)
        self.assertAlmostEqual(row["cloud_cost_usd"], 0.21)
        self.assertEqual(row["cloud_model"], "gemini-3.5-flash")

    def test_transcription_model_persisted_for_history(self):
        job_id = self.db.create_job("/tmp/a.mp4", "/tmp/ws", {})
        self.db.update_job_transcription_model(job_id, "assemblyai-universal-3", "cloud")
        row = self.db.get_job(job_id)
        self.assertEqual(row["transcription_model"], "assemblyai-universal-3")
        self.assertEqual(row["transcription_engine"], "cloud")
        # Local jobs record the Whisper model + "local" engine.
        local_id = self.db.create_job("/tmp/b.mp4", "/tmp/ws", {})
        self.db.update_job_transcription_model(
            local_id, "mlx-community/whisper-large-v3-turbo", "local"
        )
        local_row = self.db.get_job(local_id)
        self.assertEqual(local_row["transcription_engine"], "local")

    def test_usage_survives_job_deletion(self):
        job_id = self.db.create_job("/tmp/a.mp4", "/tmp/ws", {})
        self.db.add_api_usage(
            provider="gemini",
            model="m",
            input_tokens=10,
            output_tokens=5,
            cost_usd=0.5,
            job_id=job_id,
        )
        self.db.delete_job(job_id)
        self.assertAlmostEqual(self.db.month_api_spend_usd(), 0.5)


class SettingsRedactionTest(unittest.TestCase):
    def test_cloud_api_key_is_redacted(self):
        settings = {
            "transcription_settings": {
                "cloud_api_key": "AIzaSecret",
                "cloud_model": "gemini-3.5-flash",
            }
        }
        redacted = _redact_settings(settings)
        self.assertEqual(
            redacted["transcription_settings"]["cloud_api_key"], "[redacted]"
        )
        self.assertEqual(
            redacted["transcription_settings"]["cloud_model"], "gemini-3.5-flash"
        )


class PerHourBillingTest(unittest.TestCase):
    def test_cost_for_duration_per_hour(self):
        # AssemblyAI at 0.21 $/h → half an hour = 0.105.
        self.assertAlmostEqual(
            cost_for_duration("assemblyai-universal-3", 1800), 0.105, places=4
        )

    def test_estimate_per_hour_model(self):
        est = estimate_cloud_cost(3600, "gpt-4o-mini-transcribe")
        self.assertEqual(est["input_tokens"], 0)
        self.assertEqual(est["output_tokens"], 0)
        self.assertAlmostEqual(est["cost_usd"], 0.18, places=4)

    def test_unknown_model_fallback_is_conservative_across_billing(self):
        # The most expensive known model on a per-hour basis is the
        # Gemini Pro preview; an unknown id should bill at least that.
        unknown = estimate_cloud_cost(3600, "brand-new-model")["cost_usd"]
        known_max = max(
            estimate_cloud_cost(3600, m["id"])["cost_usd"]
            for m in CLOUD_TRANSCRIPTION_MODELS
        )
        self.assertAlmostEqual(unknown, known_max, places=4)

    def test_provider_for_model(self):
        self.assertEqual(provider_for_model("assemblyai-universal-3"), "assemblyai")
        self.assertEqual(provider_for_model("gpt-4o-mini-transcribe"), "openai")
        self.assertEqual(provider_for_model("gemini-3.5-flash"), "gemini")


class ProviderFactoryTest(unittest.TestCase):
    def test_factory_dispatches_by_provider(self):
        from cloud_transcription import (
            AssemblyAIProvider,
            DeepgramProvider,
            GeminiProvider,
            GladiaProvider,
            OpenAITranscribeProvider,
        )

        opener = lambda *a, **k: None  # noqa: E731 - never called by ctor
        self.assertIsInstance(get_cloud_provider("gemini", "k", opener=opener), GeminiProvider)
        self.assertIsInstance(get_cloud_provider("openai", "k", opener=opener), OpenAITranscribeProvider)
        self.assertIsInstance(get_cloud_provider("assemblyai", "k", opener=opener), AssemblyAIProvider)
        self.assertIsInstance(get_cloud_provider("deepgram", "k", opener=opener), DeepgramProvider)
        self.assertIsInstance(get_cloud_provider("gladia", "k", opener=opener), GladiaProvider)

    def test_unknown_provider_raises(self):
        with self.assertRaises(CloudTranscriptionError) as ctx:
            get_cloud_provider("nope", "k")
        self.assertEqual(ctx.exception.code, "cloud_provider")

    def test_blank_key_refused(self):
        with self.assertRaises(CloudTranscriptionError) as ctx:
            get_cloud_provider("assemblyai", "  ")
        self.assertEqual(ctx.exception.code, "cloud_auth")

    def test_every_listed_provider_is_constructible(self):
        opener = lambda *a, **k: None  # noqa: E731
        for provider in CLOUD_PROVIDERS:
            self.assertIsNotNone(get_cloud_provider(provider, "k", opener=opener))


class MultipartTest(unittest.TestCase):
    def test_multipart_carries_fields_and_file(self):
        from cloud_transcription import _multipart_body

        body, content_type = _multipart_body(
            {"model": "gpt-4o-transcribe-diarize", "language": "fr"},
            file_field="file",
            filename="a.mp3",
            file_bytes=b"\x00\x01AUDIO",
            file_content_type="audio/mpeg",
        )
        self.assertIn("multipart/form-data; boundary=", content_type)
        self.assertIn(b'name="model"', body)
        self.assertIn(b"gpt-4o-transcribe-diarize", body)
        self.assertIn(b'filename="a.mp3"', body)
        self.assertIn(b"\x00\x01AUDIO", body)


class _FakeResponse:
    def __init__(self, body, headers=None):
        self._body = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.headers = headers or {}

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _router(routes):
    """Build a urlopen-style opener that picks a response by
    (method, url-substring). ``routes`` is a list of
    ``(method, substring, body)``; first match wins."""

    def opener(request, timeout=None):
        method = request.get_method()
        url = request.full_url
        for want_method, substr, body in routes:
            if method == want_method and substr in url:
                return _FakeResponse(body)
        raise AssertionError(f"no route for {method} {url}")

    return opener


def _ctx(duration=600.0, **overrides):
    base = dict(
        language="fr",
        glossary_terms=["Odoo"],
        chunk_offset_seconds=0.0,
        chunk_duration_seconds=duration,
    )
    base.update(overrides)
    return CloudPromptContext(**base)


class _CapturingRouter:
    """Opener that records each request's parsed JSON body / query so
    tests can assert what context was forwarded to the provider."""

    def __init__(self, routes):
        self.routes = routes
        self.bodies: list[dict] = []
        self.urls: list[str] = []

    def __call__(self, request, timeout=None):
        method = request.get_method()
        url = request.full_url
        self.urls.append(url)
        if request.data:
            try:
                self.bodies.append(json.loads(request.data.decode("utf-8")))
            except (json.JSONDecodeError, UnicodeDecodeError):
                self.bodies.append({"_raw": True})
        for want_method, substr, body in self.routes:
            if method == want_method and substr in url:
                return _FakeResponse(body)
        raise AssertionError(f"no route for {method} {url}")

    def body_matching(self, key: str) -> dict | None:
        for body in self.bodies:
            if key in body:
                return body
        return None


class ContextEnrichmentTest(unittest.TestCase):
    """The signals EVC already has — expected speaker count, names,
    business vocabulary — must reach each provider's native params."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.audio = Path(self._tmp.name) / "chunk.mp3"
        self.audio.write_bytes(b"fake-mp3")

    def tearDown(self):
        self._tmp.cleanup()

    def test_gladia_forwards_vocab_names_and_speaker_count(self):
        router = _CapturingRouter([
            ("POST", "/v2/upload", {"audio_url": "https://x/up"}),
            ("POST", "/v2/pre-recorded", {"result_url": "https://api.gladia.io/v2/pre-recorded/g1"}),
            ("GET", "/v2/pre-recorded/g1", {
                "status": "done",
                "result": {"transcription": {"utterances": [
                    {"speaker": 0, "start": 0.0, "end": 2.0, "text": "Oui."},
                ]}},
            }),
        ])
        provider = get_cloud_provider("gladia", "key", opener=router)
        provider.transcribe(
            str(self.audio),
            model_id="gladia-solaria-3",
            context=_ctx(
                glossary_terms=["Odoo", "Ekonum"],
                expected_speaker_names=["Robin Joseph"],
                expected_min_speakers=3,
                expected_max_speakers=3,
            ),
        )
        body = router.body_matching("custom_vocabulary")
        self.assertIsNotNone(body)
        self.assertIn("Odoo", body["custom_vocabulary"])
        self.assertIn("Robin Joseph", body["custom_vocabulary"])  # name added
        self.assertEqual(body["diarization_config"]["number_of_speakers"], 3)
        self.assertEqual(body["model"], "solaria-3")

    def test_assemblyai_forwards_speaker_range_not_word_boost(self):
        router = _CapturingRouter([
            ("POST", "/v2/upload", {"upload_url": "https://x/up"}),
            ("POST", "/v2/transcript", {"id": "t1"}),
            ("GET", "/v2/transcript/t1", {
                "status": "completed",
                "utterances": [{"speaker": "A", "start": 0, "end": 2000, "text": "Oui."}],
            }),
        ])
        provider = get_cloud_provider("assemblyai", "key", opener=router)
        provider.transcribe(
            str(self.audio),
            model_id="assemblyai-universal-3",
            context=_ctx(expected_min_speakers=2, expected_max_speakers=4),
        )
        body = router.body_matching("speaker_options")
        self.assertIsNotNone(body)
        self.assertEqual(body["speaker_options"]["min_speakers_expected"], 2)
        self.assertEqual(body["speaker_options"]["max_speakers_expected"], 4)
        # word_boost must not be sent (absent from the current schema).
        self.assertNotIn("word_boost", body)

    def test_deepgram_keyterms_include_names(self):
        router = _CapturingRouter([
            ("POST", "/v1/listen", {"results": {"utterances": [
                {"speaker": 0, "start": 0.0, "end": 2.0, "transcript": "Oui."},
            ]}}),
        ])
        provider = get_cloud_provider("deepgram", "key", opener=router)
        provider.transcribe(
            str(self.audio),
            model_id="deepgram-nova-3",
            context=_ctx(glossary_terms=["Ekonum"], expected_speaker_names=["Lùka"]),
        )
        from urllib.parse import unquote

        listen_url = next(u for u in router.urls if "/v1/listen" in u)
        self.assertIn("keyterm=Ekonum", listen_url)
        # The expected name is added too (URL-encoded in the query).
        self.assertIn("Lùka", unquote(listen_url))

    def test_bias_terms_dedupes_and_caps(self):
        ctx = _ctx(
            glossary_terms=["Odoo", "odoo", "ERP"],
            expected_speaker_names=["Robin", "ERP"],
        )
        terms = ctx.bias_terms(limit=10)
        # "odoo"/"Odoo" dedup case-insensitively; "ERP" appears once.
        self.assertEqual(terms.count("Odoo"), 1)
        self.assertEqual([t.lower() for t in terms].count("erp"), 1)
        self.assertIn("Robin", terms)


class STTProviderParsingTest(unittest.TestCase):
    """Each STT provider must turn its native response shape into the
    common segment/usage form, on a per-hour cost basis. HTTP is faked
    via an injected opener — no network, no real audio decode."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.audio = Path(self._tmp.name) / "chunk.mp3"
        self.audio.write_bytes(b"fake-mp3-bytes")

    def tearDown(self):
        self._tmp.cleanup()

    def test_assemblyai(self):
        opener = _router([
            ("POST", "/v2/upload", {"upload_url": "https://x/up"}),
            ("POST", "/v2/transcript", {"id": "t1", "status": "queued"}),
            ("GET", "/v2/transcript/t1", {
                "status": "completed",
                "utterances": [
                    {"speaker": "A", "start": 1000, "end": 6000, "text": "Bonjour."},
                    {"speaker": "B", "start": 6000, "end": 9000, "text": "Salut."},
                ],
            }),
        ])
        provider = get_cloud_provider("assemblyai", "key", opener=opener)
        result = provider.transcribe(
            str(self.audio), model_id="assemblyai-universal-3", context=_ctx(600)
        )
        self.assertEqual(len(result.segments), 2)
        self.assertEqual(result.segments[0]["start"], 1.0)  # ms → s
        self.assertEqual(result.segments[0]["speaker"], "Intervenant 1")
        self.assertEqual(result.segments[1]["speaker"], "Intervenant 2")
        # Per-hour billing: 600 s of 0.21 $/h.
        self.assertAlmostEqual(result.usage.cost_usd, 0.035, places=4)
        self.assertEqual(result.usage.input_tokens, 0)

    def test_deepgram(self):
        opener = _router([
            ("POST", "/v1/listen", {
                "results": {
                    "utterances": [
                        {"speaker": 0, "start": 0.5, "end": 4.0, "transcript": "Bonjour."},
                        {"speaker": 1, "start": 4.0, "end": 7.0, "transcript": "Salut."},
                    ]
                }
            }),
        ])
        provider = get_cloud_provider("deepgram", "key", opener=opener)
        result = provider.transcribe(
            str(self.audio), model_id="deepgram-nova-3", context=_ctx(3600)
        )
        self.assertEqual([s["text"] for s in result.segments], ["Bonjour.", "Salut."])
        self.assertEqual(result.segments[0]["speaker"], "Intervenant 1")
        self.assertAlmostEqual(result.usage.cost_usd, 0.26, places=4)

    def test_gladia(self):
        opener = _router([
            ("POST", "/v2/upload", {"audio_url": "https://x/up"}),
            ("POST", "/v2/pre-recorded", {"id": "g1", "result_url": "https://api.gladia.io/v2/pre-recorded/g1"}),
            ("GET", "/v2/pre-recorded/g1", {
                "status": "done",
                "result": {"transcription": {"utterances": [
                    {"speaker": 0, "start": 0.0, "end": 3.0, "text": "Bonjour."},
                ]}},
            }),
        ])
        provider = get_cloud_provider("gladia", "key", opener=opener)
        result = provider.transcribe(
            str(self.audio), model_id="gladia-solaria-1", context=_ctx(1800)
        )
        self.assertEqual(result.segments[0]["text"], "Bonjour.")
        self.assertAlmostEqual(result.usage.cost_usd, 0.305, places=4)

    def test_gladia_sends_solaria_3_model_param(self):
        # Solaria-3 must be requested explicitly — Gladia defaults to
        # Solaria-1 otherwise.
        captured: dict = {}

        def opener(request, timeout=None):
            url = request.full_url
            method = request.get_method()
            if method == "POST" and "/v2/pre-recorded" in url:
                captured["body"] = json.loads(request.data.decode("utf-8"))
                return _FakeResponse({"id": "g1", "result_url": "https://api.gladia.io/v2/pre-recorded/g1"})
            if method == "POST" and "/v2/upload" in url:
                return _FakeResponse({"audio_url": "https://x/up"})
            if method == "GET" and "/v2/pre-recorded/g1" in url:
                return _FakeResponse({
                    "status": "done",
                    "result": {"transcription": {"utterances": [
                        {"speaker": 0, "start": 0.0, "end": 2.0, "text": "Oui."},
                    ]}},
                })
            raise AssertionError(f"no route for {method} {url}")

        provider = get_cloud_provider("gladia", "key", opener=opener)
        provider.transcribe(
            str(self.audio), model_id="gladia-solaria-3", context=_ctx(600)
        )
        self.assertEqual(captured["body"]["model"], "solaria-3")

    def test_check_access_reports_catalogue_models(self):
        # The key check must surface the models the app offers for the
        # provider, not a stale single hardcoded id.
        from cloud_transcription import cloud_models_for_provider

        opener = _router([("GET", "/v2/pre-recorded", {"items": []})])
        provider = get_cloud_provider("gladia", "key", opener=opener)
        payload = provider.check_access()
        expected = [m["id"] for m in cloud_models_for_provider("gladia")]
        self.assertEqual(payload["models"], expected)
        self.assertIn("gladia-solaria-3", payload["models"])

    def test_openai_plain_text_is_segmented(self):
        opener = _router([
            ("POST", "/v1/audio/transcriptions", {"text": "Bonjour à tous. On démarre la réunion."}),
        ])
        provider = get_cloud_provider("openai", "key", opener=opener)
        result = provider.transcribe(
            str(self.audio), model_id="gpt-4o-mini-transcribe", context=_ctx(120)
        )
        # Flat text split into sentence segments with proportional times.
        self.assertEqual(len(result.segments), 2)
        self.assertEqual(result.segments[0]["text"], "Bonjour à tous.")
        self.assertEqual(result.segments[0]["speaker"], "")  # no diarisation
        self.assertAlmostEqual(result.usage.cost_usd, 0.006, places=4)  # 120s @0.18/h

    def test_openai_diarized_segments(self):
        opener = _router([
            ("POST", "/v1/audio/transcriptions", {
                "segments": [
                    {"start": 0.0, "end": 3.0, "speaker": "speaker_1", "text": "Bonjour."},
                    {"start": 3.0, "end": 5.0, "speaker": "speaker_2", "text": "Salut."},
                ]
            }),
        ])
        provider = get_cloud_provider("openai", "key", opener=opener)
        result = provider.transcribe(
            str(self.audio), model_id="gpt-4o-transcribe-diarize", context=_ctx(60)
        )
        self.assertEqual(result.segments[0]["speaker"], "Intervenant 1")
        self.assertEqual(result.segments[1]["speaker"], "Intervenant 2")

    def test_chunk_offset_applied(self):
        opener = _router([
            ("POST", "/v1/listen", {"results": {"utterances": [
                {"speaker": 0, "start": 1.0, "end": 4.0, "transcript": "Suite."},
            ]}}),
        ])
        provider = get_cloud_provider("deepgram", "key", opener=opener)
        ctx = CloudPromptContext(chunk_offset_seconds=1800.0, chunk_duration_seconds=600.0)
        result = provider.transcribe(
            str(self.audio), model_id="deepgram-nova-3", context=ctx
        )
        self.assertEqual(result.segments[0]["start"], 1801.0)

    def test_empty_response_raises(self):
        opener = _router([
            ("POST", "/v1/listen", {"results": {"utterances": []}}),
        ])
        provider = get_cloud_provider("deepgram", "key", opener=opener)
        with self.assertRaises(CloudTranscriptionError):
            provider.transcribe(
                str(self.audio), model_id="deepgram-nova-3", context=_ctx(600)
            )

    def test_check_access_smoke(self):
        for provider_id, route in [
            ("openai", ("GET", "/v1/models", {"data": [{"id": "gpt-4o-transcribe-diarize"}]})),
            ("assemblyai", ("GET", "/v2/transcript", {"transcripts": []})),
            ("deepgram", ("GET", "/v1/projects", {"projects": []})),
            ("gladia", ("GET", "/v2/pre-recorded", {"items": []})),
        ]:
            provider = get_cloud_provider(provider_id, "key", opener=_router([route]))
            payload = provider.check_access()
            self.assertTrue(payload["ok"], provider_id)


if __name__ == "__main__":
    unittest.main()
