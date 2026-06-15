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
    CLOUD_TRANSCRIPTION_MODELS,
    CLOUD_CHUNK_SECONDS,
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
    estimate_cloud_cost,
    merge_chunk_results,
    parse_cloud_response,
    parse_cloud_timestamp,
    plan_audio_chunks,
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


if __name__ == "__main__":
    unittest.main()
