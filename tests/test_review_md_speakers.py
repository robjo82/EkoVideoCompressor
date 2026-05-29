"""
PR AO — the review markdown's "Interlocuteurs identifiés" must
list pre-attributed / voice-matched speakers, not only the LLM's.

Bug (Caste 21mai rerun) : the transcript showed ``[Robin]`` (176
lines, via the "Vous êtes" pre-attribution) but the review .md
"Interlocuteurs identifiés" section was EMPTY — because it only
read ``_llm_payload["speakers"]`` and the LLM (conservative since
PR W) named nobody.

Fix : ``_write_review_markdown`` merges ``_llm_payload["speakers"]``
with ``self._recognized_speakers`` (voice-match + pre-attribution),
annotating the source of each name.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from ekovideo_engine.models import JobRequest, TranscriptionSettings
from ekovideo_engine.pipeline import TranscriptionPipeline


def _make_pipeline() -> TranscriptionPipeline:
    request = JobRequest(
        source_path="/tmp/x.wav",
        output_dir="/tmp/out",
        mode="transcribe",
        transcription_settings=TranscriptionSettings(),
    )
    return TranscriptionPipeline(request=request, sink=MagicMock())


def _write(pipeline: TranscriptionPipeline, tmp: Path) -> str:
    transcript = tmp / "réunion.txt"
    transcript.write_text("dummy")
    out = pipeline._write_review_markdown(transcript, segments=[])
    assert out is not None, "expected review markdown to be written"
    return out.read_text(encoding="utf-8")


class ReviewMarkdownSpeakersTests(unittest.TestCase):
    def test_preattributed_speaker_listed_when_llm_named_nobody(self):
        # The Caste regression : pre-attribution found Robin, LLM
        # named nobody. The .md must still show Robin.
        pipeline = _make_pipeline()
        pipeline._recognized_speakers = {"SPEAKER_01": "Robin"}
        pipeline._llm_payload = {}  # LLM named nobody
        with tempfile.TemporaryDirectory() as tmp:
            md = _write(pipeline, Path(tmp))
        self.assertIn("## Interlocuteurs identifiés", md)
        self.assertIn("SPEAKER_01", md)
        self.assertIn("Robin", md)
        self.assertIn("voix mémorisée / réglage", md)

    def test_llm_and_recognised_merged(self):
        # LLM named SPEAKER_00, pre-attribution named SPEAKER_01.
        # Both appear, with distinct source annotations.
        pipeline = _make_pipeline()
        pipeline._recognized_speakers = {"SPEAKER_01": "Robin"}
        pipeline._llm_payload = {"speakers": {"SPEAKER_00": "Manon"}}
        with tempfile.TemporaryDirectory() as tmp:
            md = _write(pipeline, Path(tmp))
        self.assertIn("Manon", md)
        self.assertIn("déduit du texte", md)
        self.assertIn("Robin", md)
        self.assertIn("voix mémorisée / réglage", md)

    def test_recognised_takes_precedence_over_llm(self):
        # If both name the SAME cluster, the recognised (voice /
        # setting) name wins over the LLM guess.
        pipeline = _make_pipeline()
        pipeline._recognized_speakers = {"SPEAKER_00": "Robin"}
        pipeline._llm_payload = {"speakers": {"SPEAKER_00": "Nicolas"}}
        with tempfile.TemporaryDirectory() as tmp:
            md = _write(pipeline, Path(tmp))
        self.assertIn("SPEAKER_00", md)
        self.assertIn("Robin", md)
        self.assertNotIn("Nicolas", md)

    def test_recognised_speakers_alone_trigger_review_md(self):
        # A run whose ONLY signal is the pre-attribution still
        # produces the .md (so the user sees who was identified).
        pipeline = _make_pipeline()
        pipeline._recognized_speakers = {"SPEAKER_00": "Robin"}
        pipeline._llm_payload = {}
        # No glossary, no VAD, no loops, nothing else.
        self.assertTrue(pipeline._has_quality_output())
        with tempfile.TemporaryDirectory() as tmp:
            md = _write(pipeline, Path(tmp))
        self.assertIn("Robin", md)

    def test_empty_recognised_and_llm_no_section(self):
        # Neither source named anyone → no Interlocuteurs section,
        # but the .md may still be written for another reason.
        pipeline = _make_pipeline()
        pipeline._recognized_speakers = {}
        pipeline._llm_payload = {"title": "Un sujet"}
        with tempfile.TemporaryDirectory() as tmp:
            md = _write(pipeline, Path(tmp))
        self.assertNotIn("## Interlocuteurs identifiés", md)

    def test_blank_recognised_values_ignored(self):
        # A recognised entry with an empty value (shouldn't happen,
        # but defensive) is filtered out.
        pipeline = _make_pipeline()
        pipeline._recognized_speakers = {"SPEAKER_00": "", "SPEAKER_01": "Robin"}
        pipeline._llm_payload = {}
        with tempfile.TemporaryDirectory() as tmp:
            md = _write(pipeline, Path(tmp))
        self.assertIn("SPEAKER_01", md)
        self.assertNotIn("`SPEAKER_00`", md)


if __name__ == "__main__":
    unittest.main()
