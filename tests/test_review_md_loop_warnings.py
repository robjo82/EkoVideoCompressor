"""
PR AB — surface decoder-loop drops in the review markdown.

PR Y added the in-memory ``DroppedLoop`` record + a live
WarningEvent, but the user reads the ``- à vérifier.md`` long
after the run is gone from the live stream. This PR persists the
loops in the markdown so the user has a permanent paper trail
of what content was lost — including a coverage badge.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from ekovideo_engine.models import JobRequest, TranscriptionSettings
from ekovideo_engine.pipeline import TranscriptionPipeline
from transcription_utils import DroppedLoop


def _make_pipeline() -> TranscriptionPipeline:
    request = JobRequest(
        source_path="/tmp/x.wav",
        output_dir="/tmp/out",
        mode="transcribe",
        transcription_settings=TranscriptionSettings(),
    )
    return TranscriptionPipeline(request=request, sink=MagicMock())


def _write_review(
    pipeline: TranscriptionPipeline, tmpdir: Path
) -> str:
    transcript_path = tmpdir / "réunion.txt"
    transcript_path.write_text("dummy")
    out_path = pipeline._write_review_markdown(transcript_path, segments=[])
    assert out_path is not None, "expected review markdown to be written"
    return out_path.read_text(encoding="utf-8")


class ReviewMarkdownCoverageTests(unittest.TestCase):
    def test_no_loops_full_coverage_friendly_line(self):
        # When the markdown is being written for some other reason
        # (here: a non-empty LLM title) AND no loops were dropped,
        # the report includes a friendly coverage line — gives the
        # user a positive confirmation that nothing was lost.
        pipeline = _make_pipeline()
        pipeline._dropped_loops = []
        pipeline._audio_seconds = 1800.0  # 30 min
        # Force the .md to be written by setting a non-empty LLM
        # payload — otherwise the writer short-circuits on
        # ``_has_quality_output`` because there's nothing to report.
        pipeline._llm_payload = {"title": "Test - Sujet"}
        with tempfile.TemporaryDirectory() as tmp:
            md = _write_review(pipeline, Path(tmp))
        self.assertIn("## Couverture audio", md)
        self.assertIn("✓ Tout l'audio", md)
        self.assertIn("1800", md)
        # No alert framing.
        self.assertNotIn("⚠️", md)
        self.assertNotIn("Zones perdues", md)

    def test_minor_loss_softer_header(self):
        # 10 % loss → "Zones perdues" but no red-alert header.
        pipeline = _make_pipeline()
        pipeline._dropped_loops = [
            DroppedLoop(start=100.0, end=200.0, text="bla bla", dropped=20),
        ]
        pipeline._audio_seconds = 1000.0  # 100s lost / 1000s = 10%
        with tempfile.TemporaryDirectory() as tmp:
            md = _write_review(pipeline, Path(tmp))
        self.assertIn("## Zones perdues", md)
        self.assertNotIn("⚠️ Zones perdues", md)
        # 90% coverage rendered.
        self.assertIn("90 %", md)

    def test_severe_loss_triggers_alert_header(self):
        # The Caste-21mai pattern : 80 min lost on 132 min audio
        # (60 % coverage) → red alert.
        pipeline = _make_pipeline()
        pipeline._dropped_loops = [
            DroppedLoop(start=600.0, end=4800.0, text="loop A", dropped=400),
            DroppedLoop(start=5000.0, end=5400.0, text="loop B", dropped=60),
        ]
        pipeline._audio_seconds = 7920.0  # 132 min
        with tempfile.TemporaryDirectory() as tmp:
            md = _write_review(pipeline, Path(tmp))
        self.assertIn("## ⚠️ Zones perdues", md)
        self.assertIn("relecture impossible", md)
        # Both loops surfaced.
        self.assertIn("loop A", md)
        self.assertIn("loop B", md)

    def test_loop_entries_sorted_worst_first(self):
        # Three loops with different durations — the longest must
        # be listed first.
        pipeline = _make_pipeline()
        pipeline._dropped_loops = [
            DroppedLoop(start=0, end=60, text="court", dropped=10),
            DroppedLoop(start=200, end=800, text="long", dropped=100),
            DroppedLoop(start=900, end=1000, text="moyen", dropped=20),
        ]
        pipeline._audio_seconds = 2000.0
        with tempfile.TemporaryDirectory() as tmp:
            md = _write_review(pipeline, Path(tmp))
        # "long" appears before "moyen" appears before "court".
        idx_long = md.find("long")
        idx_moyen = md.find("moyen")
        idx_court = md.find("court")
        self.assertGreater(idx_long, 0)
        self.assertGreater(idx_moyen, idx_long)
        self.assertGreater(idx_court, idx_moyen)

    def test_timestamps_rendered_mm_ss(self):
        # The Caste loop at 2371s should render as 39:31.
        pipeline = _make_pipeline()
        pipeline._dropped_loops = [
            DroppedLoop(start=2371.0, end=4200.0, text="Zindoc", dropped=200),
        ]
        pipeline._audio_seconds = 7920.0
        with tempfile.TemporaryDirectory() as tmp:
            md = _write_review(pipeline, Path(tmp))
        self.assertIn("`39:31`", md)
        self.assertIn("`70:00`", md)  # end at 4200s = 70:00

    def test_caps_at_10_loops_with_more_indicator(self):
        # 15 loops — only 10 listed, then "et 5 autre(s)".
        pipeline = _make_pipeline()
        pipeline._dropped_loops = [
            DroppedLoop(
                start=i * 100, end=i * 100 + 50, text=f"l{i}", dropped=10
            )
            for i in range(15)
        ]
        pipeline._audio_seconds = 2000.0
        with tempfile.TemporaryDirectory() as tmp:
            md = _write_review(pipeline, Path(tmp))
        self.assertIn("et 5 autre(s)", md)

    def test_loops_alone_trigger_review_md(self):
        # A run with NO LLM output and NO glossary subs should
        # still produce the .md if loops were detected — that's
        # exactly when the user needs the alert most.
        pipeline = _make_pipeline()
        pipeline._dropped_loops = [
            DroppedLoop(start=0, end=600, text="seul signal", dropped=120),
        ]
        pipeline._audio_seconds = 1000.0
        # No glossary, no LLM, no VAD, no nothing.
        with tempfile.TemporaryDirectory() as tmp:
            md = _write_review(pipeline, Path(tmp))
        self.assertIn("seul signal", md)
        self.assertIn("Zones perdues", md)


if __name__ == "__main__":
    unittest.main()
