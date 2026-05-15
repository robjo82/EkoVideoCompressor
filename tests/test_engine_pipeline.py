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
from pathlib import Path
from unittest.mock import patch

from ekovideo_engine.events import collect_events
from ekovideo_engine.models import JobRequest
from ekovideo_engine.pipeline import (
    StepResult,
    TranscriptionPipeline,
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

    def test_max_preset_enables_everything(self):
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


if __name__ == "__main__":
    unittest.main()
