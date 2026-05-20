"""
End-to-end tests for the engine quality pipeline.

We don't run real Whisper / pyannote / mlx_lm — they're 1-4 Gb
models we never want in CI. Instead we stub ``subprocess.run`` so
each shell-out returns a canned response, then assert that the
pipeline:

  1. Wires the steps in the right order (audio → VAD → Whisper →
     multipass → glossary post → LLM → write).
  2. Applies the phonetic glossary post-processor (deterministic,
     does NOT need subprocess stubbing — it's pure Python).
  3. Falls back gracefully when an optional step fails.
  4. Writes the review markdown with the steps that did fire.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from ekovideo_engine.events import collect_events
from ekovideo_engine.models import JobRequest
from ekovideo_engine.pipeline import (
    StepResult,
    TranscriptionPipeline,
    apply_meeting_date_to_artifact,
    apply_spoken_punctuation_in_email_contexts,
    normalize_spoken_clock_times,
    reconstruct_letter_spellings,
    reconstruct_spelled_text,
    _apply_glossary_capitalization,
    _filter_tautological_doubts,
    _friendly_ffmpeg_error,
)


@dataclass
class _StubResult:
    """Mimic the slice of CompletedProcess our pipeline reads."""

    returncode: int = 0
    stdout: str = ""
    stderr: str = ""


def _make_request(workspace: Path, source: Path, **overrides) -> JobRequest:
    tx_overrides = overrides.pop("transcription_settings", {})
    return JobRequest.from_dict(
        {
            "source_path": str(source),
            "output_dir": str(workspace),
            "mode": "transcribe",
            "workspace_dir": str(workspace),
            "glossary_terms": overrides.pop("glossary_terms", []),
            "technical_terms": overrides.pop("technical_terms", []),
            "transcription_settings": {
                # Disable everything that needs external models by default;
                # individual tests opt in by overriding here.
                "vad_enabled": False,
                "multipass_enabled": False,
                "diarization_enabled": False,
                "venv_python_path": "",
                "mlx_whisper_path": "/usr/local/bin/mlx_whisper",
                "model": "mlx-community/whisper-large-v3-turbo",
                "output_format": "txt",
                "language": "fr",
                **tx_overrides,
            },
        }
    )


def _write_canned_whisper(target: Path, segments: list[dict]) -> None:
    target.write_text(
        json.dumps({"segments": segments, "text": " ".join(s.get("text", "") for s in segments)}),
        encoding="utf-8",
    )


class TranscriptionPipelinePortTest(unittest.TestCase):
    """The engine used to do `audio_extract` then a single Whisper run
    and nothing else. The port adds VAD, multipass, glossary post-pass,
    diarisation, LLM. These tests pin the order and the wiring.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.workspace = Path(self._tmp.name)
        self.source = self.workspace / "meeting.mov"
        self.source.write_bytes(b"fake media")

    def tearDown(self):
        self._tmp.cleanup()

    def test_meeting_date_is_applied_to_generated_artifact_mtime(self):
        artifact = self.workspace / "transcription.txt"
        artifact.write_text("Bonjour", encoding="utf-8")
        request = JobRequest.from_dict(
            {
                "source_path": str(self.source),
                "output_dir": str(self.workspace),
                "mode": "transcribe",
                "meeting_date": "2026-05-14T12:30:00Z",
            }
        )

        apply_meeting_date_to_artifact(request, artifact)

        expected = int(datetime(2026, 5, 14, 12, 30, tzinfo=timezone.utc).timestamp())
        self.assertEqual(int(artifact.stat().st_mtime), expected)

    def test_basic_flow_writes_transcript_and_emits_artifact(self):
        """Plumbing test: with every quality phase off and Whisper
        stubbed, the pipeline should still produce a transcript file
        and a 'transcript' StepResult that the runner can wire to the
        library DB.
        """
        request = _make_request(self.workspace, self.source)
        events, sink = collect_events()
        pipeline = TranscriptionPipeline(request, sink)

        def fake_run(cmd, *args, **kwargs):
            # Audio extract emits the wav, Whisper emits the json.
            if "ffmpeg" in (cmd[0] if cmd else ""):
                wav_index = cmd.index("-progress") - 1 if "-progress" in cmd else -1
                # Find the explicit output path (last arg) since the
                # extract command varies.
                wav_path = Path(cmd[-1])
                wav_path.write_bytes(b"riff fake")
                return _StubResult(returncode=0)
            if "mlx_whisper" in (cmd[0] if cmd else ""):
                # Output dir + name combine into the JSON path.
                out_dir = Path(cmd[cmd.index("--output-dir") + 1])
                stem = cmd[cmd.index("--output-name") + 1]
                target = out_dir / f"{stem}.json"
                target.parent.mkdir(parents=True, exist_ok=True)
                _write_canned_whisper(
                    target,
                    [
                        {
                            "start": 0.0, "end": 2.0, "text": "Bonjour.",
                            "avg_logprob": -0.1, "no_speech_prob": 0.0,
                            "compression_ratio": 1.5,
                        }
                    ],
                )
                return _StubResult(returncode=0)
            raise AssertionError(f"unexpected subprocess: {cmd}")

        with patch.object(subprocess, "run", side_effect=fake_run):
            results = pipeline.run(str(self.source))

        names = [r.name for r in results]
        self.assertIn("audio_extract", names)
        self.assertIn("whisper", names)
        self.assertIn("transcript", names)
        self.assertNotIn("enhanced_transcript", names)
        # All hard steps succeeded.
        for r in results:
            self.assertTrue(r.ok, f"step {r.name} unexpectedly failed: {r.error}")

        transcript = next(r for r in results if r.name == "transcript")
        self.assertTrue(Path(transcript.artifact_path).exists())
        text = Path(transcript.artifact_path).read_text(encoding="utf-8")
        self.assertIn("Bonjour", text)

    def test_phonetic_glossary_rewrites_segments(self):
        """When the user provides a glossary, mis-spellings Whisper
        emits should be corrected before the transcript hits disk —
        no model dependency, fully deterministic.
        """
        request = _make_request(
            self.workspace,
            self.source,
            glossary_terms=["Mollie", "Sudokies"],
        )
        events, sink = collect_events()
        pipeline = TranscriptionPipeline(request, sink)

        def fake_run(cmd, *args, **kwargs):
            if "ffmpeg" in (cmd[0] if cmd else ""):
                Path(cmd[-1]).write_bytes(b"riff fake")
                return _StubResult(returncode=0)
            if "mlx_whisper" in (cmd[0] if cmd else ""):
                out_dir = Path(cmd[cmd.index("--output-dir") + 1])
                stem = cmd[cmd.index("--output-name") + 1]
                target = out_dir / f"{stem}.json"
                _write_canned_whisper(
                    target,
                    [
                        {
                            "start": 0.0, "end": 3.0,
                            "text": "On utilise MOLI avec Sudokiz.",
                            "avg_logprob": -0.1, "no_speech_prob": 0.0,
                            "compression_ratio": 1.5,
                        }
                    ],
                )
                return _StubResult(returncode=0)
            raise AssertionError(f"unexpected subprocess: {cmd}")

        with patch.object(subprocess, "run", side_effect=fake_run):
            results = pipeline.run(str(self.source))

        transcript = next(r for r in results if r.name == "transcript")
        text = Path(transcript.artifact_path).read_text(encoding="utf-8")
        # Phonetic post-processor should have fixed both terms.
        self.assertIn("Sudokies", text)
        self.assertNotIn("Sudokiz", text)
        # MOLI was ALL-CAPS so casing echo gives MOLLIE; the canonical
        # term shows up in either casing.
        self.assertTrue("Mollie" in text or "MOLLIE" in text)
        enhanced = next(r for r in results if r.name == "enhanced_transcript")
        self.assertTrue(Path(enhanced.artifact_path).exists())

    def test_pipeline_forwards_expected_speakers_to_pyannote(self):
        """Pin that the SwiftUI ``expected_speaker_count`` field
        actually reaches the diarisation subprocess. Pyannote
        under-segments aggressively when left to estimate on its own,
        so the user's hint must survive the JSON round-trip into the
        pipeline command builder.
        """
        fake_python = tempfile.NamedTemporaryFile(
            suffix="-fake-python", delete=False
        )
        fake_python.write(b"#!/bin/sh\nexit 0\n")
        fake_python.close()
        Path(fake_python.name).chmod(0o755)

        request = _make_request(
            self.workspace,
            self.source,
            transcription_settings={
                "venv_python_path": fake_python.name,
                "vad_enabled": False,
                "multipass_enabled": False,
                "diarization_enabled": True,
                "hf_token": "fake-token",
                "expected_min_speakers": 4,
                "expected_max_speakers": 4,
            },
        )
        events, sink = collect_events()
        pipeline = TranscriptionPipeline(request, sink)
        diarisation_cmds: list[list[str]] = []

        def fake_run(cmd, *args, **kwargs):
            if "ffmpeg" in (cmd[0] if cmd else ""):
                Path(cmd[-1]).write_bytes(b"riff fake")
                return _StubResult(returncode=0)
            if "mlx_whisper" in (cmd[0] if cmd else ""):
                target = (
                    Path(cmd[cmd.index("--output-dir") + 1])
                    / f"{cmd[cmd.index('--output-name') + 1]}.json"
                )
                _write_canned_whisper(
                    target,
                    [
                        {
                            "start": 0.0, "end": 2.0, "text": "Bonjour.",
                            "avg_logprob": -0.1, "no_speech_prob": 0.0,
                            "compression_ratio": 1.5,
                        }
                    ],
                )
                return _StubResult(returncode=0)
            if cmd and cmd[0] == fake_python.name:
                script = cmd[2] if len(cmd) > 2 else ""
                if "speaker-diarization" in script:
                    diarisation_cmds.append(list(cmd))
                    # Return an empty-turn JSON so the downstream
                    # speaker-assignment path is exercised without
                    # producing noise.
                    return _StubResult(
                        returncode=0, stdout='{"turns": []}'
                    )
                # Title + corrections LLM calls — return empty.
                if "title" in script.lower() or "speakers" in script.lower():
                    return _StubResult(
                        returncode=0,
                        stdout='{"title":"","speakers":{},"technical_terms":[]}',
                    )
                return _StubResult(returncode=0, stdout="")
            raise AssertionError(f"unexpected subprocess: {cmd}")

        with patch.object(subprocess, "run", side_effect=fake_run):
            pipeline.run(str(self.source))

        self.assertEqual(len(diarisation_cmds), 1)
        cmd = diarisation_cmds[0]
        # The last two args are the speaker hints we pinned.
        self.assertEqual(cmd[-2], "4")
        self.assertEqual(cmd[-1], "4")

    def test_llm_corrections_actually_land_in_enhanced_file(self):
        """The previous pipeline listed LLM corrections in the
        review report but wrote the ``améliorée`` transcript byte-
        identical to the raw one. Pin the new behaviour: the LLM
        substitutions land in ``améliorée``, the base file stays
        untouched (so the user can diff), and the review section
        shifts from "proposées" to "appliquées".
        """
        fake_python = tempfile.NamedTemporaryFile(
            suffix="-fake-python", delete=False
        )
        fake_python.write(b"#!/bin/sh\nexit 0\n")
        fake_python.close()
        Path(fake_python.name).chmod(0o755)

        request = _make_request(
            self.workspace,
            self.source,
            transcription_settings={
                # LLM gate keys on venv_python_path existing.
                "venv_python_path": fake_python.name,
                "vad_enabled": False,
                "multipass_enabled": False,
            },
        )
        events, sink = collect_events()
        pipeline = TranscriptionPipeline(request, sink)

        def fake_run(cmd, *args, **kwargs):
            if "ffmpeg" in (cmd[0] if cmd else ""):
                Path(cmd[-1]).write_bytes(b"riff fake")
                return _StubResult(returncode=0)
            if "mlx_whisper" in (cmd[0] if cmd else ""):
                target = (
                    Path(cmd[cmd.index("--output-dir") + 1])
                    / f"{cmd[cmd.index('--output-name') + 1]}.json"
                )
                _write_canned_whisper(
                    target,
                    [
                        {
                            "start": 0.0,
                            "end": 3.0,
                            "text": "Bonjour, ici Sudokiz.",
                            "avg_logprob": -0.1,
                            "no_speech_prob": 0.0,
                            "compression_ratio": 1.5,
                        }
                    ],
                )
                return _StubResult(returncode=0)
            # The LLM is called via the fake python interpreter; we
            # branch on the script body to know which call this is.
            if cmd and cmd[0] == fake_python.name:
                script = cmd[2] if len(cmd) > 2 else ""
                if "title" in script.lower() or "speakers" in script.lower():
                    return _StubResult(
                        returncode=0,
                        stdout='{"title":"Test","speakers":{},"technical_terms":[]}',
                    )
                # Otherwise it's the corrections call.
                stdout = (
                    "# Corrections\n"
                    '- [00:00:01] "Sudokiz" -> "Sudokies" (raison: glossaire)\n'
                    "# Doutes\n"
                )
                return _StubResult(returncode=0, stdout=stdout)
            raise AssertionError(f"unexpected subprocess: {cmd}")

        with patch.object(subprocess, "run", side_effect=fake_run):
            results = pipeline.run(str(self.source))

        base = next(r for r in results if r.name == "transcript")
        enhanced = next(r for r in results if r.name == "enhanced_transcript")
        base_text = Path(base.artifact_path).read_text(encoding="utf-8")
        enhanced_text = Path(enhanced.artifact_path).read_text(encoding="utf-8")
        # Base transcript stays as Whisper produced it.
        self.assertIn("Sudokiz", base_text)
        # Enhanced transcript carries the LLM substitution.
        self.assertIn("Sudokies", enhanced_text)
        self.assertNotIn("Sudokiz", enhanced_text)

        review = next(r for r in results if r.name == "review")
        review_text = Path(review.artifact_path).read_text(encoding="utf-8")
        self.assertIn("## Corrections LLM appliquées", review_text)

    def test_audio_extract_produces_separate_wav_for_diarisation(self):
        """Pyannote's clustering relies on timbre cues that the
        speech-enhancement compressor smooths away. Pin that the
        pipeline asks ffmpeg twice: once with filters (audio.wav for
        Whisper) and once without (audio.diar.wav for pyannote).
        """
        request = _make_request(self.workspace, self.source)
        events, sink = collect_events()
        pipeline = TranscriptionPipeline(request, sink)
        ffmpeg_calls: list[list[str]] = []

        def fake_run(cmd, *args, **kwargs):
            if "ffmpeg" in (cmd[0] if cmd else ""):
                ffmpeg_calls.append(list(cmd))
                Path(cmd[-1]).write_bytes(b"riff fake")
                return _StubResult(returncode=0)
            if "mlx_whisper" in (cmd[0] if cmd else ""):
                target = (
                    Path(cmd[cmd.index("--output-dir") + 1])
                    / f"{cmd[cmd.index('--output-name') + 1]}.json"
                )
                _write_canned_whisper(
                    target,
                    [
                        {
                            "start": 0.0, "end": 2.0, "text": "Bonjour.",
                            "avg_logprob": -0.1, "no_speech_prob": 0.0,
                            "compression_ratio": 1.5,
                        }
                    ],
                )
                return _StubResult(returncode=0)
            raise AssertionError(f"unexpected subprocess: {cmd}")

        with patch.object(subprocess, "run", side_effect=fake_run):
            pipeline.run(str(self.source))

        # Two ffmpeg invocations: one for audio.wav (with filters),
        # one for audio.diar.wav (without). The filtered call carries
        # the ``-af`` flag; the diarisation call does not.
        wav_calls = [c for c in ffmpeg_calls if c[-1].endswith("audio.wav")]
        diar_calls = [c for c in ffmpeg_calls if c[-1].endswith("audio.diar.wav")]
        self.assertEqual(len(wav_calls), 1)
        self.assertEqual(len(diar_calls), 1)
        self.assertIn("-af", wav_calls[0])
        self.assertNotIn("-af", diar_calls[0])

    def test_pipeline_exposes_final_segments_and_speakers(self):
        """The runner needs the pipeline's final segment list +
        speaker map to populate the library DB. Pin those exposed
        attributes — without them the rename-speakers sheet shows
        "Aucun interlocuteur détecté" on every run.
        """
        request = _make_request(self.workspace, self.source)
        events, sink = collect_events()
        pipeline = TranscriptionPipeline(request, sink)

        def fake_run(cmd, *args, **kwargs):
            if "ffmpeg" in (cmd[0] if cmd else ""):
                Path(cmd[-1]).write_bytes(b"riff fake")
                return _StubResult(returncode=0)
            if "mlx_whisper" in (cmd[0] if cmd else ""):
                target = (
                    Path(cmd[cmd.index("--output-dir") + 1])
                    / f"{cmd[cmd.index('--output-name') + 1]}.json"
                )
                _write_canned_whisper(
                    target,
                    [
                        {
                            "start": 0.0, "end": 3.0, "text": "Bonjour.",
                            "avg_logprob": -0.1, "no_speech_prob": 0.0,
                            "compression_ratio": 1.5,
                        }
                    ],
                )
                return _StubResult(returncode=0)
            raise AssertionError(f"unexpected subprocess: {cmd}")

        with patch.object(subprocess, "run", side_effect=fake_run):
            pipeline.run(str(self.source))

        # Whisper alone (no diarisation) means no speaker labels.
        # The segments are still surfaced so the runner can persist
        # them; the speaker map is empty, which the runner then
        # leaves untouched.
        self.assertEqual(len(pipeline.final_segments), 1)
        self.assertEqual(pipeline.final_segments[0]["text"], "Bonjour.")
        self.assertEqual(pipeline.final_speaker_map, {})

    def test_review_markdown_written_when_glossary_corrections_apply(self):
        request = _make_request(
            self.workspace, self.source, glossary_terms=["Mollie"]
        )
        events, sink = collect_events()
        pipeline = TranscriptionPipeline(request, sink)

        def fake_run(cmd, *args, **kwargs):
            if "ffmpeg" in (cmd[0] if cmd else ""):
                Path(cmd[-1]).write_bytes(b"riff fake")
                return _StubResult(returncode=0)
            if "mlx_whisper" in (cmd[0] if cmd else ""):
                target = (
                    Path(cmd[cmd.index("--output-dir") + 1])
                    / f"{cmd[cmd.index('--output-name') + 1]}.json"
                )
                _write_canned_whisper(
                    target,
                    [
                        {
                            "start": 0.0, "end": 3.0, "text": "Le module MOLI.",
                            "avg_logprob": -0.1, "no_speech_prob": 0.0,
                            "compression_ratio": 1.5,
                        }
                    ],
                )
                return _StubResult(returncode=0)
            raise AssertionError(f"unexpected subprocess: {cmd}")

        with patch.object(subprocess, "run", side_effect=fake_run):
            results = pipeline.run(str(self.source))

        review = next((r for r in results if r.name == "review"), None)
        self.assertIsNotNone(review)
        body = Path(review.artifact_path).read_text(encoding="utf-8")
        self.assertIn("Vocabulaire métier", body)
        # Casing echo: "MOLI" was ALL-CAPS so the replacement becomes
        # "MOLLIE". Both spellings are acceptable evidence that the
        # glossary substitution made it into the report.
        self.assertTrue("Mollie" in body or "MOLLIE" in body, body)

    def test_vad_falls_back_to_full_audio_when_too_aggressive(self):
        """If VAD claims more than 60 % of the audio is non-speech we
        almost certainly lost real content. Fall back to the full
        audio so the transcript covers the whole meeting instead of
        a sliced-out half.
        """
        fake_python = tempfile.NamedTemporaryFile(
            suffix="-fake-python", delete=False
        )
        fake_python.write(b"#!/bin/sh\nexit 0\n")
        fake_python.close()
        Path(fake_python.name).chmod(0o755)

        request = _make_request(
            self.workspace,
            self.source,
            transcription_settings={
                "vad_enabled": True,
                "venv_python_path": fake_python.name,
            },
        )
        events, sink = collect_events()
        pipeline = TranscriptionPipeline(request, sink)

        def fake_run(cmd, *args, **kwargs):
            if cmd and cmd[0] == fake_python.name:
                # The pipeline calls the same fake python for both
                # VAD and the LLM. The VAD invocation has the
                # trimmed-WAV path at argv[2] (after the script body
                # at argv[1]); the LLM ones don't. Match on the
                # script content to know which is which.
                script = cmd[2] if len(cmd) > 2 else ""
                if "silero" in script.lower() or "speech_timestamps" in script.lower():
                    stdout = json.dumps(
                        {
                            "spans": [
                                {
                                    "trim_start": 0,
                                    "trim_end": 30,
                                    "src_start": 0,
                                    "src_end": 100,
                                }
                            ],
                            "trimmed_seconds": 30,
                            "total_seconds": 100,
                        }
                    )
                    # ``build_vad_cmd`` puts the output WAV at argv[4]
                    # (cmd[4] when you count [python, -c, script,
                    # in_path, out_path, ...]).
                    trimmed = Path(cmd[4])
                    trimmed.write_bytes(b"riff fake")
                    return _StubResult(returncode=0, stdout=stdout)
                # LLM (title or corrections) — return empty.
                return _StubResult(
                    returncode=0,
                    stdout='{"title":"","speakers":{},"technical_terms":[]}',
                )
            if "ffmpeg" in (cmd[0] if cmd else ""):
                Path(cmd[-1]).write_bytes(b"riff fake")
                return _StubResult(returncode=0)
            if "mlx_whisper" in (cmd[0] if cmd else ""):
                target = (
                    Path(cmd[cmd.index("--output-dir") + 1])
                    / f"{cmd[cmd.index('--output-name') + 1]}.json"
                )
                _write_canned_whisper(
                    target,
                    [
                        {
                            "start": 0.0, "end": 2.0, "text": "OK.",
                            "avg_logprob": -0.1, "no_speech_prob": 0.0,
                            "compression_ratio": 1.5,
                        }
                    ],
                )
                return _StubResult(returncode=0)
            raise AssertionError(f"unexpected subprocess: {cmd}")

        with patch.object(subprocess, "run", side_effect=fake_run):
            results = pipeline.run(str(self.source))

        # VAD step succeeded but flagged the fallback in metrics.
        vad = next(r for r in results if r.name == "vad")
        self.assertTrue(vad.ok)
        self.assertTrue(vad.metrics.get("fallback"))
        # The artifact path is the *original* WAV, not the trimmed
        # one — the audio Whisper saw is the full meeting.
        self.assertTrue(vad.artifact_path.endswith("audio.wav"))
        # No manifest survives, so downstream remap is a no-op.
        self.assertEqual(pipeline._vad_manifest, [])

    def test_vad_failure_does_not_kill_the_job(self):
        """VAD is an optimisation. If silero-vad isn't installed we
        warn and fall back to the original audio, but the job still
        produces a transcript.
        """
        with tempfile.NamedTemporaryFile(suffix="-fake-python", delete=False) as fake_python:
            fake_python.write(b"#!/bin/sh\nexit 1\n")
        Path(fake_python.name).chmod(0o755)

        request = _make_request(
            self.workspace,
            self.source,
            transcription_settings={
                "vad_enabled": True,
                "venv_python_path": fake_python.name,
            },
        )
        events, sink = collect_events()
        pipeline = TranscriptionPipeline(request, sink)

        def fake_run(cmd, *args, **kwargs):
            if cmd and cmd[0] == fake_python.name:
                # Simulate VAD command failing.
                return _StubResult(
                    returncode=1, stderr="ModuleNotFoundError: silero_vad"
                )
            if "ffmpeg" in (cmd[0] if cmd else ""):
                Path(cmd[-1]).write_bytes(b"riff fake")
                return _StubResult(returncode=0)
            if "mlx_whisper" in (cmd[0] if cmd else ""):
                target = (
                    Path(cmd[cmd.index("--output-dir") + 1])
                    / f"{cmd[cmd.index('--output-name') + 1]}.json"
                )
                _write_canned_whisper(
                    target,
                    [
                        {
                            "start": 0.0, "end": 2.0, "text": "OK.",
                            "avg_logprob": -0.1, "no_speech_prob": 0.0,
                            "compression_ratio": 1.5,
                        }
                    ],
                )
                return _StubResult(returncode=0)
            raise AssertionError(f"unexpected subprocess: {cmd}")

        with patch.object(subprocess, "run", side_effect=fake_run):
            results = pipeline.run(str(self.source))

        # VAD step is recorded as a failure but doesn't sink the job.
        vad = next(r for r in results if r.name == "vad")
        self.assertFalse(vad.ok)
        transcript = next(r for r in results if r.name == "transcript")
        self.assertTrue(transcript.ok)
        self.assertTrue(Path(transcript.artifact_path).exists())

    def test_managed_venv_is_used_when_swiftui_does_not_send_python_path(self):
        with tempfile.TemporaryDirectory() as support_dir:
            managed_python = (
                Path(support_dir)
                / "mlx-whisper-venv"
                / "bin"
                / "python"
            )
            managed_python.parent.mkdir(parents=True)
            managed_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            managed_python.chmod(0o755)

            request = _make_request(
                self.workspace,
                self.source,
                transcription_settings={
                    "vad_enabled": True,
                    "venv_python_path": "",
                    "quality_preset": "balanced",
                },
            )
            events, sink = collect_events()
            with patch.dict(os.environ, {"EKO_APP_SUPPORT_DIR": support_dir}):
                TranscriptionPipeline(request, sink)

            self.assertEqual(
                request.transcription_settings.venv_python_path,
                str(managed_python),
            )

    def test_audio_extract_failure_aborts(self):
        request = _make_request(self.workspace, self.source)
        events, sink = collect_events()
        pipeline = TranscriptionPipeline(request, sink)

        def fake_run(cmd, *args, **kwargs):
            return _StubResult(returncode=1, stderr="dyld: Library not loaded: foo")

        with patch.object(subprocess, "run", side_effect=fake_run):
            results = pipeline.run(str(self.source))
        # Just audio_extract — nothing further should have happened.
        self.assertEqual([r.name for r in results], ["audio_extract"])
        self.assertFalse(results[0].ok)
        self.assertIn("binaire ffmpeg fourni", results[0].error)


class SubprocessEnvAndErrorMapperTest(unittest.TestCase):
    """Pin the two fixes that landed when a user reported the
    'Fetching 4 files: 0%|...' dialog: the engine wasn't enriching
    PATH for venv-side commands (so mlx_whisper crashed looking for
    ffmpeg) and it was surfacing the raw HF download progress
    interleaved with the actual exception, making the dialog look
    like a stack trace.
    """

    def _make_request(self) -> JobRequest:
        return JobRequest.from_dict(
            {
                "source_path": "/tmp/x.mov",
                "output_dir": "/tmp",
                "mode": "transcribe",
                "compression_settings": {
                    "ffmpeg_path": "/Applications/Eko.app/Contents/Resources/bin/ffmpeg",
                    "ffprobe_path": "/Applications/Eko.app/Contents/Resources/bin/ffprobe",
                },
            }
        )

    def test_subprocess_env_prepends_bundled_bin_to_path(self):
        from ekovideo_engine.pipeline import subprocess_env_for_request

        env = subprocess_env_for_request(self._make_request())
        self.assertIn(
            "/Applications/Eko.app/Contents/Resources/bin",
            env["PATH"],
        )
        # System PATH must still come AFTER the bundle so we never
        # accidentally pick up a stale Homebrew binary.
        bundled_index = env["PATH"].index(
            "/Applications/Eko.app/Contents/Resources/bin"
        )
        rest = env["PATH"][bundled_index:]
        self.assertTrue(
            rest.startswith("/Applications/Eko.app/Contents/Resources/bin"),
            "bundle dir must be at the front of PATH",
        )

    def test_subprocess_env_keeps_existing_path(self):
        from ekovideo_engine.pipeline import subprocess_env_for_request

        env = subprocess_env_for_request(self._make_request())
        # Whatever the user's $PATH was, it should still be in the
        # composed value — we prepend, never replace.
        for shell_dir in ("/usr/bin", "/bin"):
            if shell_dir in os.environ.get("PATH", ""):
                self.assertIn(shell_dir, env["PATH"])
                break

    def test_tqdm_progress_lines_stripped_from_error(self):
        from ekovideo_engine.pipeline import _clean_subprocess_stderr

        raw = (
            "Fetching 4 files:   0%|          | 0/4 [00:00<?, ?it/s]\n"
            "Fetching 4 files: 100%|██████████| 4/4 [00:00<00:00, 85163.53it/s]\n"
            "Traceback (most recent call last):\n"
            "  File 'mlx_whisper/audio.py', line 59, in load_audio\n"
            "FileNotFoundError: [Errno 2] No such file or directory: 'ffmpeg'"
        )
        cleaned = _clean_subprocess_stderr(raw)
        self.assertNotIn("Fetching", cleaned)
        self.assertIn("FileNotFoundError", cleaned)
        self.assertIn("ffmpeg", cleaned)

    def test_friendly_error_maps_mlx_ffmpeg_not_found(self):
        from ekovideo_engine.pipeline import _friendly_ffmpeg_error

        raw = (
            "Fetching 4 files: 100%|██████████| 4/4 [00:00<00:00, 85163.53it/s]\n"
            "FileNotFoundError: [Errno 2] No such file or directory: 'ffmpeg'"
        )
        out = _friendly_ffmpeg_error(raw, "/usr/local/bin/mlx_whisper")
        # The cleaned message should mention what failed (ffmpeg
        # missing in PATH), not just leak the raw stack trace.
        self.assertIn("ffmpeg", out.lower())
        self.assertNotIn("Fetching", out)


class QualityPresetTest(unittest.TestCase):
    """The SwiftUI app sends a single ``quality_preset`` string; the
    engine derives the right toggles. Power users keep full control
    via the ``"custom"`` preset (which is also the legacy default
    so older callers don't have their config flipped from under
    them).
    """

    def test_fast_preset_disables_every_quality_phase(self):
        settings = JobRequest.from_dict(
            {
                "source_path": "/tmp/x.mov",
                "output_dir": "/tmp",
                "mode": "transcribe",
                "transcription_settings": {
                    "quality_preset": "fast",
                    "vad_enabled": True,
                    "multipass_enabled": True,
                    "per_speaker_enabled": True,
                },
            }
        ).transcription_settings
        self.assertFalse(settings.vad_enabled)
        self.assertFalse(settings.multipass_enabled)
        self.assertFalse(settings.per_speaker_enabled)
        self.assertFalse(settings.audio_recheck_enabled)

    def test_balanced_preset_picks_safe_quality_wins(self):
        settings = JobRequest.from_dict(
            {
                "source_path": "/tmp/x.mov",
                "output_dir": "/tmp",
                "mode": "transcribe",
                "transcription_settings": {"quality_preset": "balanced"},
            }
        ).transcription_settings
        self.assertTrue(settings.vad_enabled)
        self.assertTrue(settings.multipass_enabled)
        self.assertFalse(settings.per_speaker_enabled)

    def test_max_preset_enables_only_wired_phases(self):
        # Audit: PR E wired per_speaker, PR H wired web_enrichment,
        # PR F wired audio_recheck (Qwen2-Audio). All on by default
        # in the max preset; the audio_recheck step degrades to a
        # silent no-op when ``mlx_vlm`` isn't installed in the venv.
        settings = JobRequest.from_dict(
            {
                "source_path": "/tmp/x.mov",
                "output_dir": "/tmp",
                "mode": "transcribe",
                "transcription_settings": {"quality_preset": "max"},
            }
        ).transcription_settings
        self.assertTrue(settings.vad_enabled)
        self.assertTrue(settings.multipass_enabled)
        self.assertTrue(settings.per_speaker_enabled)
        self.assertTrue(settings.audio_recheck_enabled)
        self.assertTrue(settings.web_enrichment_enabled)

    def test_custom_preset_passes_individual_flags_through(self):
        settings = JobRequest.from_dict(
            {
                "source_path": "/tmp/x.mov",
                "output_dir": "/tmp",
                "mode": "transcribe",
                "transcription_settings": {
                    "quality_preset": "custom",
                    "vad_enabled": False,
                    "multipass_enabled": True,
                    "per_speaker_enabled": True,
                },
            }
        ).transcription_settings
        self.assertFalse(settings.vad_enabled)
        self.assertTrue(settings.multipass_enabled)
        self.assertTrue(settings.per_speaker_enabled)


class SpokenClockTimeNormalizationTest(unittest.TestCase):
    """PR O — turn ``neuf heures`` into ``9h``. The Caste job had
    ``Neuveur à Mirandol`` (Whisper hallucination from "neuf heures")
    — out of scope here, but the standard ``neuf heures`` pattern
    is fixable and very common in French dictation."""

    def test_simple_hour(self):
        self.assertEqual(
            normalize_spoken_clock_times("On se voit à neuf heures."),
            "On se voit à 9h.",
        )

    def test_hour_with_demie(self):
        self.assertEqual(
            normalize_spoken_clock_times("Rendez-vous à dix heures et demie."),
            "Rendez-vous à 10h30.",
        )

    def test_hour_with_quart(self):
        self.assertEqual(
            normalize_spoken_clock_times("À huit heures et quart."),
            "À 8h15.",
        )

    def test_hour_moins_le_quart(self):
        self.assertEqual(
            normalize_spoken_clock_times("Démarrage à neuf heures moins le quart."),
            "Démarrage à 8h45.",
        )

    def test_hour_with_digit_minutes(self):
        self.assertEqual(
            normalize_spoken_clock_times("Le RDV est à treize heures 45."),
            "Le RDV est à 13h45.",
        )

    def test_hour_pile(self):
        self.assertEqual(
            normalize_spoken_clock_times("À quinze heures pile."),
            "À 15h.",
        )

    def test_does_not_touch_durations_with_articles(self):
        # ``deux heures de réunion`` IS a duration but ``2h de
        # réunion`` reads fine either way — we don't try to
        # disambiguate. Pin that ``deux heures`` becomes ``2h``
        # consistently (acceptable behaviour for transcripts).
        out = normalize_spoken_clock_times("On a deux heures de réunion.")
        self.assertEqual(out, "On a 2h de réunion.")

    def test_does_not_touch_other_number_uses(self):
        # ``deux mille`` shouldn't become ``2000h`` or anything
        # weird — the regex requires ``heure(s)`` to fire.
        text = "On a vendu deux mille unités."
        self.assertEqual(normalize_spoken_clock_times(text), text)

    def test_case_insensitive(self):
        self.assertEqual(
            normalize_spoken_clock_times("À NEUF HEURES précises."),
            "À 9h précises.",
        )

    def test_idempotent_on_already_normalized(self):
        text = "RDV à 9h30."
        self.assertEqual(normalize_spoken_clock_times(text), text)


class LetterSpellingReconstructionTest(unittest.TestCase):
    """PR M — collapse ``N O U V I A L E`` / ``n-o-u-v-i-a-l-e`` /
    ``c-a-s-t-e.fr`` sequences back into single tokens. The Caste
    job had ``n-o-u-v-i-a-l-e`` and ``c-a-s-t-e.fr`` in the
    transcript — unusable as a livrable."""

    def test_collapse_hyphenated_lower(self):
        text = "C'est manon point n-o-u-v-i-a-l-e arobase caste."
        out = reconstruct_letter_spellings(text)
        self.assertIn("nouviale", out)
        self.assertNotIn("n-o-u-v-i-a-l-e", out)

    def test_collapse_space_separated_caps(self):
        text = "On épelle N O U V I A L E."
        out = reconstruct_letter_spellings(text)
        self.assertIn("nouviale", out)

    def test_short_acronym_kept_uppercase(self):
        # ≤ 3 letters → API stays API, SQL stays SQL, not "api".
        text = "Une A P I REST et du S Q L."
        out = reconstruct_letter_spellings(text)
        self.assertIn("API", out)
        self.assertIn("SQL", out)
        self.assertNotIn("a-p-i", out.lower())

    def test_tld_suffix_preserved(self):
        text = "c-a-s-t-e.fr"
        out = reconstruct_letter_spellings(text)
        self.assertEqual(out, "caste.fr")

    def test_does_not_break_normal_french(self):
        text = "Il y a des règles, c'est-à-dire des conventions."
        out = reconstruct_letter_spellings(text)
        # Hyphens inside compound words shouldn't trigger collapse
        # (less than 3 single-letter tokens between hyphens).
        self.assertIn("c'est-à-dire", out)


class SpokenPunctuationTest(unittest.TestCase):
    """PR M — ``arobase`` → ``@`` and ``point`` → ``.`` ONLY in
    email-like contexts so we don't rewrite ``point d'attention``."""

    def test_email_context_substitutes_arobase_and_point(self):
        # The ``.fr`` already in the text triggers the email context
        # detector. The ``arobase`` between two tokens becomes ``@``.
        text = "C'est manon arobase caste.fr"
        out = apply_spoken_punctuation_in_email_contexts(text)
        self.assertIn("manon@caste.fr", out)

    def test_no_substitution_when_no_email_context(self):
        # No TLD nearby, no `@` — leave ``point`` alone as a word.
        text = "Le point d'attention principal."
        out = apply_spoken_punctuation_in_email_contexts(text)
        self.assertEqual(out, text)

    def test_arobas_variants_also_substitute(self):
        # PR P — Whisper transcribes "arobase" as ``Arrobas`` /
        # ``arobas`` on the Caste call. All these variants now
        # map to ``@`` in email context.
        for variant in ("Arrobas", "Arobas", "arobaze"):
            text = f"C'est manon {variant} caste.fr"
            out = apply_spoken_punctuation_in_email_contexts(text)
            self.assertIn("manon@caste.fr", out, variant)


class ReconstructSpelledTextEndToEndTest(unittest.TestCase):
    """End-to-end wrapper — pin the Caste-style input gets cleaned up."""

    def test_caste_email_pattern(self):
        # Whisper output approximation: spelling then verbalised
        # punctuation, all jumbled in the email context.
        text = "C'est manon point n-o-u-v-i-a-l-e arobase c-a-s-t-e.fr."
        out = reconstruct_spelled_text(text)
        self.assertIn("nouviale", out)
        self.assertIn("caste.fr", out)
        self.assertIn("@", out)

    def test_idempotent(self):
        text = "C'est manon point n-o-u-v-i-a-l-e arobase c-a-s-t-e.fr."
        once = reconstruct_spelled_text(text)
        twice = reconstruct_spelled_text(once)
        self.assertEqual(once, twice)

    def test_preserves_speaker_brackets(self):
        # Speaker prefix shouldn't be touched even when the body
        # carries a TLD that triggers the context detector.
        text = "[SPEAKER_00] (00:11:15) c-a-s-t-e.fr"
        out = reconstruct_spelled_text(text)
        self.assertTrue(out.startswith("[SPEAKER_00] (00:11:15)"))


class TautologicalDoubtFilterTest(unittest.TestCase):
    """PR C — drop the auto-referential "X qui pourrait être X ou X"
    entries that polluted the Cozynergy review file (5 out of 10
    uncertain passages were tautologies)."""

    def test_drops_X_ou_X_pattern(self):
        doubts = [
            {
                "timestamp": "00:03:01",
                "text": "nous on peut retraiter la donnée",
                "reason": 'doute sur le "retraiter" qui pourrait être "retraiter" ou "retraiter"',
            },
            {
                "timestamp": "00:00:00",
                "text": "Mathilde Gérard, c'est Cozynergy. Ah",
                "reason": 'doute sur le "Ah" final, possible coupure',
            },
        ]
        out = _filter_tautological_doubts(doubts)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["timestamp"], "00:00:00")

    def test_drops_when_both_alternatives_already_in_source(self):
        # The LLM hallucinates "uncertainty" between two words that
        # both appear in the original text — pointless to surface.
        doubts = [
            {
                "timestamp": "00:01:00",
                "text": "Pour passer commande on regarde le système",
                "reason": 'doute sur "passer" qui pourrait être "passer" ou "système"',
            }
        ]
        self.assertEqual(_filter_tautological_doubts(doubts), [])

    def test_keeps_real_alternatives(self):
        doubts = [
            {
                "timestamp": "00:02:35",
                "text": "On peut sortir un fake",
                "reason": 'doute sur le mot "fake" qui pourrait être "FEC"',
            }
        ]
        out = _filter_tautological_doubts(doubts)
        self.assertEqual(len(out), 1)

    def test_drops_very_short_reasons(self):
        # Garbage fragments from a misbehaving LLM — < 8 chars
        # carries no actionable signal.
        doubts = [
            {"timestamp": "00:00:01", "text": "Allô", "reason": "doute"},
            {"timestamp": "00:00:02", "text": "Allô", "reason": "X"},
        ]
        self.assertEqual(_filter_tautological_doubts(doubts), [])


class GlossaryCapitalizationTest(unittest.TestCase):
    """PR C — enforce canonical case for glossary terms in the final
    rendered transcript. The user added ``Quadra`` / ``Excel`` /
    ``Odoo`` to the glossary specifically to get those spellings;
    Whisper + LLM steps don't always honor them.
    """

    def test_replaces_lowercase_with_canonical(self):
        text = "On utilise quadra et excel."
        out = _apply_glossary_capitalization(text, ["Quadra", "Excel"])
        self.assertEqual(out, "On utilise Quadra et Excel.")

    def test_respects_word_boundaries(self):
        # ``quadra`` in ``quadragénaire`` is NOT the glossary term —
        # word-boundary regex must protect it.
        text = "C'est un quadragénaire qui utilise quadra."
        out = _apply_glossary_capitalization(text, ["Quadra"])
        self.assertEqual(
            out, "C'est un quadragénaire qui utilise Quadra."
        )

    def test_preserves_speaker_brackets(self):
        # Speaker names live inside ``[...]`` — never rewrite them
        # via glossary substitution (could collide with a partner
        # name like "Audoo" hypothetically matching "Odoo").
        text = "[Odoo] On parle de odoo."
        out = _apply_glossary_capitalization(text, ["Odoo"])
        self.assertEqual(out, "[Odoo] On parle de Odoo.")

    def test_skips_multi_word_terms(self):
        # Multi-word glossary entries can have legitimate variants
        # ("Power BI" vs "Power-BI"). Skip them rather than do
        # brittle multi-token matching.
        text = "On utilise power bi."
        out = _apply_glossary_capitalization(text, ["Power BI"])
        self.assertEqual(out, "On utilise power bi.")

    def test_handles_empty_inputs_safely(self):
        self.assertEqual(_apply_glossary_capitalization("", ["Odoo"]), "")
        self.assertEqual(_apply_glossary_capitalization("text", []), "text")
        self.assertEqual(_apply_glossary_capitalization("text", ["a"]), "text")


class WebEnrichmentPipelineHookTest(unittest.TestCase):
    """PR H — pipeline mutates ``request.glossary_terms`` in place
    when ``web_enrichment_enabled`` is True and the enrichment
    function returns confirmed entities."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.workspace = Path(self._tmp.name)
        self.source = self.workspace / "meeting.mov"
        self.source.write_bytes(b"fake")

    def tearDown(self):
        self._tmp.cleanup()

    def _pipeline(self, web_enabled: bool) -> TranscriptionPipeline:
        request = _make_request(
            self.workspace,
            self.source,
            transcription_settings={
                "web_enrichment_enabled": web_enabled,
                "quality_preset": "custom",  # don't let preset force-off
            },
            glossary_terms=["Odoo"],
        )
        return TranscriptionPipeline(request, lambda _event: None)

    def test_no_op_when_flag_disabled(self):
        pipeline = self._pipeline(web_enabled=False)
        with patch("web_context.enrich_glossary_via_web") as mock_enrich:
            pipeline._maybe_enrich_glossary_from_web("Cozynergy is a company.")
            mock_enrich.assert_not_called()
        # Glossary unchanged.
        self.assertEqual(list(pipeline.request.glossary_terms), ["Odoo"])

    def test_adds_confirmed_terms_to_glossary(self):
        pipeline = self._pipeline(web_enabled=True)
        # Stub the returned WebEnrichmentResult.
        from web_context import WebEnrichmentResult
        stub_results = [
            WebEnrichmentResult(
                candidate="Cozynergy",
                confirmed_term="Cozynergy",
                citation="Cozynergy — company",
                snippet="...",
            )
        ]
        with patch("web_context.enrich_glossary_via_web", return_value=stub_results):
            pipeline._maybe_enrich_glossary_from_web("Mathilde Gérard de Cozynergy.")
        self.assertIn("Cozynergy", pipeline.request.glossary_terms)
        # Existing term preserved.
        self.assertIn("Odoo", pipeline.request.glossary_terms)

    def test_dedups_against_existing_glossary(self):
        pipeline = self._pipeline(web_enabled=True)
        from web_context import WebEnrichmentResult
        stub_results = [
            WebEnrichmentResult(
                candidate="odoo",  # different case than existing "Odoo"
                confirmed_term="odoo",
                citation="...",
                snippet="...",
            )
        ]
        with patch("web_context.enrich_glossary_via_web", return_value=stub_results):
            pipeline._maybe_enrich_glossary_from_web("Some text about odoo.")
        # ``odoo`` wasn't added — ``Odoo`` is already in the glossary
        # (case-insensitive dedup).
        self.assertEqual(list(pipeline.request.glossary_terms), ["Odoo"])

    def test_swallows_exception(self):
        # Network errors must NOT sink the LLM step.
        pipeline = self._pipeline(web_enabled=True)
        with patch("web_context.enrich_glossary_via_web", side_effect=RuntimeError("network down")):
            # Should not raise.
            pipeline._maybe_enrich_glossary_from_web("Some text.")
        self.assertEqual(list(pipeline.request.glossary_terms), ["Odoo"])


class CurrentUserPreAttributionTest(unittest.TestCase):
    """PR B — pin the cold-start heuristic that attributes the first
    cluster to ``current_user_name`` when no voiceprint match was
    found. Without it, every meeting's SPEAKER_00 lingered as a
    placeholder because voice profiles started empty (recognition
    only fires when sample_count > 0)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.workspace = Path(self._tmp.name)
        self.source = self.workspace / "meeting.mov"
        self.source.write_bytes(b"fake")

    def tearDown(self):
        self._tmp.cleanup()

    def _pipeline(self, current_user: str) -> TranscriptionPipeline:
        request = _make_request(
            self.workspace,
            self.source,
            transcription_settings={"current_user_name": current_user},
        )
        return TranscriptionPipeline(request, lambda _event: None)

    def test_attributes_first_cluster_to_current_user(self):
        pipeline = self._pipeline("Robin")
        segments = [
            {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00", "text": "Allô"},
            {"start": 1.0, "end": 2.0, "speaker": "SPEAKER_01", "text": "Bonjour"},
        ]
        out = pipeline._pre_attribute_current_user(segments, already_recognized={})
        self.assertEqual(out, {"SPEAKER_00": "Robin"})

    def test_no_op_when_current_user_blank(self):
        pipeline = self._pipeline("")
        segments = [
            {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00", "text": "Allô"},
        ]
        self.assertEqual(
            pipeline._pre_attribute_current_user(segments, already_recognized={}),
            {},
        )

    def test_picks_first_unclaimed_cluster_when_voice_match_filled_others(self):
        # On a 2-person call, if voice match already attributed
        # Mathilde to SPEAKER_00, the user (Robin) must be the
        # other cluster. The heuristic walks past claimed clusters
        # rather than giving up — useful in the 80 % case.
        pipeline = self._pipeline("Robin")
        segments = [
            {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00", "text": "Allô"},
            {"start": 1.0, "end": 2.0, "speaker": "SPEAKER_01", "text": "Bonjour"},
        ]
        out = pipeline._pre_attribute_current_user(
            segments,
            already_recognized={"SPEAKER_00": "Mathilde"},
        )
        self.assertEqual(out, {"SPEAKER_01": "Robin"})

    def test_skips_when_current_user_already_recognised_elsewhere(self):
        # Voice match found Robin on SPEAKER_01. Heuristic should
        # NOT also tag SPEAKER_00 as Robin — would create two
        # Robins in the same conversation.
        pipeline = self._pipeline("Robin")
        segments = [
            {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00", "text": "Allô"},
            {"start": 1.0, "end": 2.0, "speaker": "SPEAKER_01", "text": "Bonjour"},
        ]
        out = pipeline._pre_attribute_current_user(
            segments,
            already_recognized={"SPEAKER_01": "Robin"},
        )
        self.assertEqual(out, {})

    def test_picks_lowest_start_time_not_first_segment(self):
        # Out-of-order Whisper output: SPEAKER_01 has the lowest
        # actual start time — heuristic must respect that.
        pipeline = self._pipeline("Robin")
        segments = [
            {"start": 5.0, "end": 6.0, "speaker": "SPEAKER_00", "text": "Tard"},
            {"start": 0.5, "end": 1.5, "speaker": "SPEAKER_01", "text": "Tôt"},
        ]
        out = pipeline._pre_attribute_current_user(segments, already_recognized={})
        self.assertEqual(out, {"SPEAKER_01": "Robin"})


if __name__ == "__main__":
    unittest.main()
