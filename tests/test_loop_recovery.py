"""
PR AD — re-Whisper recovery of decoder-loop zones.

The Caste 21mai audit showed 95 min of 132 min audio lost to
Whisper decoder loops on quiet zones. PR Y stopped silent loss
+ PR AB made it visible; PR AD actually claws back the lost
content by re-Whispering each loop range in isolation with
``--condition-on-previous-text False``, breaking the runaway
context.

Tests:
  • No loops → no-op (returns None, segments untouched).
  • Single loop recovered → segments spliced in, ``_dropped_loops``
    cleared, ``_recovered_loops`` populated, StepResult metrics
    correct.
  • Sub-loop in the retry → mark zone as still-lost, don't pollute
    output.
  • Subprocess failure → mark zone as still-lost.
  • Cap at ``_LOOP_RECOVERY_MAX_RANGES``.
  • Spliced segments sorted by start time.
  • Empty whisper output (silence) → counts as recovered, no
    segments injected, gap left as legitimate silence.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from ekovideo_engine.models import JobRequest, TranscriptionSettings
from ekovideo_engine.pipeline import TranscriptionPipeline
from transcription_utils import DroppedLoop


def _make_pipeline(loops: list[DroppedLoop]) -> TranscriptionPipeline:
    request = JobRequest(
        source_path="/tmp/x.wav",
        output_dir="/tmp/out",
        mode="transcribe",
        transcription_settings=TranscriptionSettings(
            mlx_whisper_path="/usr/local/bin/mlx_whisper",
            model="mlx-community/whisper-large-v3-turbo",
        ),
    )
    pipeline = TranscriptionPipeline(request=request, sink=MagicMock())
    pipeline._dropped_loops = list(loops)
    return pipeline


class LoopRecoveryGuardTests(unittest.TestCase):
    def test_no_loops_returns_none(self):
        pipeline = _make_pipeline([])
        result, segs = pipeline._run_loop_recovery(
            Path("/tmp/a.wav"), Path("/tmp/work"), []
        )
        self.assertIsNone(result)
        self.assertEqual(segs, [])

    def test_missing_whisper_wav_returns_none(self):
        pipeline = _make_pipeline([
            DroppedLoop(start=10.0, end=60.0, text="loop", dropped=20),
        ])
        # Path doesn't exist by default.
        result, _ = pipeline._run_loop_recovery(
            Path("/tmp/nonexistent.wav"), Path("/tmp/work"), []
        )
        self.assertIsNone(result)


class LoopRecoveryHappyPathTests(unittest.TestCase):
    """Mock subprocess + filesystem to exercise the splicing."""

    def _patch(
        self,
        tmpdir: Path,
        recovered_text: str = "Contenu récupéré.",
    ):
        """Build a side_effect that fakes mlx_whisper writing a
        JSON containing one recovered segment per call."""

        recovery_counter = {"i": 0}

        def side_effect(cmd, *args, **kwargs):  # noqa: ARG001
            # The recovery cmd has ``--clip-timestamps`` and an
            # output_path matching the loop_recovery_N.json pattern.
            # Find the output dir + name to write a stub JSON.
            try:
                output_dir_idx = cmd.index("--output-dir") + 1
                output_name_idx = cmd.index("--output-name") + 1
                output_path = (
                    Path(cmd[output_dir_idx])
                    / f"{cmd[output_name_idx]}.json"
                )
                # Pull the clip start so we can shift back.
                clip_idx = cmd.index("--clip-timestamps") + 1
                clip_start = float(cmd[clip_idx].split(",")[0])
            except (ValueError, IndexError):
                return MagicMock(returncode=1, stderr="malformed cmd")

            recovery_counter["i"] += 1
            # Write a single segment (relative to clip start = 0).
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps({
                "segments": [{
                    "start": 0.0,
                    "end": 5.0,
                    "text": f"{recovered_text} #{recovery_counter['i']}",
                }],
            }))
            return MagicMock(returncode=0, stdout="", stderr="")

        return side_effect

    def test_single_loop_recovered_and_spliced(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            whisper_wav = tmp_path / "audio.wav"
            whisper_wav.touch()
            pipeline = _make_pipeline([
                DroppedLoop(
                    start=100.0, end=160.0, text="loop A", dropped=15,
                ),
            ])
            existing = [
                {"start": 50.0, "end": 55.0, "text": "Avant."},
                {"start": 200.0, "end": 205.0, "text": "Après."},
            ]
            with patch(
                "ekovideo_engine.pipeline.subprocess.run",
                side_effect=self._patch(tmp_path),
            ):
                result, segs = pipeline._run_loop_recovery(
                    whisper_wav, tmp_path, existing
                )

        self.assertIsNotNone(result)
        self.assertTrue(result.ok)
        self.assertEqual(result.metrics["attempted"], 1)
        self.assertEqual(result.metrics["recovered"], 1)
        self.assertEqual(result.metrics["still_lost"], 0)

        # Spliced + sorted: Avant, recovered, Après.
        starts = [float(s["start"]) for s in segs]
        self.assertEqual(starts, sorted(starts))
        recovered = [
            s for s in segs if "Contenu récupéré" in s.get("text", "")
        ]
        self.assertEqual(len(recovered), 1)
        # Shifted back to the whisper_wav timeline.
        # clip_start = max(0, 100 - padding=1.0) = 99.0
        self.assertAlmostEqual(recovered[0]["start"], 99.0, places=1)

        # State reflects success.
        self.assertEqual(len(pipeline._recovered_loops), 1)
        self.assertEqual(len(pipeline._dropped_loops), 0)

    def test_subprocess_failure_marks_zone_as_still_lost(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            whisper_wav = tmp_path / "audio.wav"
            whisper_wav.touch()
            pipeline = _make_pipeline([
                DroppedLoop(
                    start=10.0, end=70.0, text="loop", dropped=12,
                ),
            ])
            with patch(
                "ekovideo_engine.pipeline.subprocess.run"
            ) as mock_run:
                mock_run.return_value = MagicMock(
                    returncode=1, stderr="mlx_whisper crashed"
                )
                result, segs = pipeline._run_loop_recovery(
                    whisper_wav, tmp_path, []
                )

        self.assertIsNotNone(result)
        self.assertEqual(result.metrics["recovered"], 0)
        self.assertEqual(result.metrics["still_lost"], 1)
        # Loop stayed in ``_dropped_loops`` so the markdown still
        # alerts about it.
        self.assertEqual(len(pipeline._dropped_loops), 1)
        self.assertEqual(len(pipeline._recovered_loops), 0)
        # No segments injected.
        self.assertEqual(segs, [])

    def test_retry_reloops_marks_zone_as_still_lost(self):
        """If the retry produces ITS OWN big sub-loop, we don't
        want to inject those segments. The cleaner will return
        them as dropped; if > 50 % of the recovered duration was
        dropped, we admit defeat."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            whisper_wav = tmp_path / "audio.wav"
            whisper_wav.touch()
            pipeline = _make_pipeline([
                DroppedLoop(
                    start=10.0, end=70.0, text="loop", dropped=12,
                ),
            ])

            def side_effect(cmd, *args, **kwargs):  # noqa: ARG001
                # Write a JSON that the cleaner will treat as a loop.
                output_dir_idx = cmd.index("--output-dir") + 1
                output_name_idx = cmd.index("--output-name") + 1
                output_path = (
                    Path(cmd[output_dir_idx])
                    / f"{cmd[output_name_idx]}.json"
                )
                output_path.parent.mkdir(parents=True, exist_ok=True)
                # 50 consecutive identical segments → cleaner drops
                # them and reports DroppedLoop covering most of the
                # range. ratio = 50/60 > 0.5 → still_lost.
                loop_phrase = "Toujours bouclé."
                output_path.write_text(json.dumps({
                    "segments": [
                        {"start": float(i), "end": float(i + 1), "text": loop_phrase}
                        for i in range(60)
                    ],
                }))
                return MagicMock(returncode=0, stdout="", stderr="")

            with patch(
                "ekovideo_engine.pipeline.subprocess.run",
                side_effect=side_effect,
            ):
                result, segs = pipeline._run_loop_recovery(
                    whisper_wav, tmp_path, []
                )

        self.assertEqual(result.metrics["recovered"], 0)
        self.assertEqual(result.metrics["still_lost"], 1)
        # No looped segments injected.
        self.assertEqual(
            [s for s in segs if "Toujours bouclé" in s.get("text", "")],
            [],
        )

    def test_cap_at_max_ranges(self):
        # More loops than the cap → only the first
        # ``_LOOP_RECOVERY_MAX_RANGES`` get attempted, the rest
        # stay in still_lost untouched.
        cap = TranscriptionPipeline._LOOP_RECOVERY_MAX_RANGES
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            whisper_wav = tmp_path / "audio.wav"
            whisper_wav.touch()
            many_loops = [
                DroppedLoop(
                    start=float(i * 200),
                    end=float(i * 200 + 100),
                    text=f"l{i}",
                    dropped=20,
                )
                for i in range(cap + 3)
            ]
            pipeline = _make_pipeline(many_loops)
            with patch(
                "ekovideo_engine.pipeline.subprocess.run",
                side_effect=self._patch(tmp_path),
            ):
                result, _ = pipeline._run_loop_recovery(
                    whisper_wav, tmp_path, []
                )
        self.assertEqual(result.metrics["attempted"], cap)
        # 3 leftovers untouched, all in still_lost.
        self.assertEqual(len(pipeline._dropped_loops), 3)

    def test_too_short_loop_skipped(self):
        # A loop < ``_LOOP_RECOVERY_MIN_DURATION_SECONDS`` (default
        # 2s) is too short to be worth re-Whispering. Skip and
        # keep as still-lost.
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            whisper_wav = tmp_path / "audio.wav"
            whisper_wav.touch()
            pipeline = _make_pipeline([
                DroppedLoop(start=10.0, end=10.5, text="tiny", dropped=5),
            ])
            with patch(
                "ekovideo_engine.pipeline.subprocess.run",
                side_effect=self._patch(tmp_path),
            ) as mock_run:
                result, _ = pipeline._run_loop_recovery(
                    whisper_wav, tmp_path, []
                )
            # No subprocess call for the tiny loop.
            self.assertEqual(mock_run.call_count, 0)
        self.assertEqual(result.metrics["recovered"], 0)
        self.assertEqual(result.metrics["still_lost"], 1)

    def test_recovered_seconds_metric_correct(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            whisper_wav = tmp_path / "audio.wav"
            whisper_wav.touch()
            loops = [
                DroppedLoop(start=10.0, end=70.0, text="loop A", dropped=12),
                DroppedLoop(start=200.0, end=330.0, text="loop B", dropped=25),
            ]
            pipeline = _make_pipeline(loops)
            with patch(
                "ekovideo_engine.pipeline.subprocess.run",
                side_effect=self._patch(tmp_path),
            ):
                result, _ = pipeline._run_loop_recovery(
                    whisper_wav, tmp_path, []
                )
        # 60 + 130 = 190 seconds recovered.
        self.assertEqual(result.metrics["recovered_seconds"], 190)
        self.assertEqual(result.metrics["recovered"], 2)


class LoopRecoveryReviewMarkdownTests(unittest.TestCase):
    """The markdown shows recovered + still-lost in separate sections."""

    def test_recovered_section_present_when_any_recovered(self):
        request = JobRequest(
            source_path="/tmp/x.wav",
            output_dir="/tmp/out",
            mode="transcribe",
            transcription_settings=TranscriptionSettings(),
        )
        pipeline = TranscriptionPipeline(request=request, sink=MagicMock())
        pipeline._recovered_loops = [
            DroppedLoop(start=10.0, end=70.0, text="recouvré", dropped=12),
        ]
        pipeline._audio_seconds = 1000.0
        with tempfile.TemporaryDirectory() as tmp:
            transcript_path = Path(tmp) / "x.txt"
            transcript_path.write_text("dummy")
            out = pipeline._write_review_markdown(transcript_path, [])
            md = out.read_text(encoding="utf-8")
        self.assertIn("## ✓ Zones récupérées", md)
        self.assertIn("60s", md)


if __name__ == "__main__":
    unittest.main()
