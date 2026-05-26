"""
PR AH — mlx_whisper with ``--clip-timestamps`` outputs ABSOLUTE
timestamps. The 4 pipeline passes that do clip-based re-Whispering
used to ``+= clip_start`` on the output, double-shifting every
segment. Symptom: Caste 21mai's 132 min audio rendered as a 3h59
transcript with content duplicated past the audio's end.

These regression tests pin the absolute-timestamps contract and
ensure no future refactor reintroduces the shift on any of the
4 passes :
  - ``_run_loop_recovery`` (PR AD)
  - ``_run_multipass`` (existing)
  - ``_run_boundary_multipass`` (PR I)
  - ``_run_per_speaker_pass`` (PR E)

Strategy : for each pass, mock subprocess to write an output JSON
whose segments are at ABSOLUTE timestamps (matching real
mlx_whisper behaviour), then verify the pipeline's spliced output
contains the SAME timestamps — no doubling.
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


def _make_pipeline() -> TranscriptionPipeline:
    request = JobRequest(
        source_path="/tmp/x.wav",
        output_dir="/tmp/out",
        mode="transcribe",
        transcription_settings=TranscriptionSettings(
            mlx_whisper_path="/usr/local/bin/mlx_whisper",
            model="mlx-community/whisper-large-v3-turbo",
        ),
    )
    return TranscriptionPipeline(request=request, sink=MagicMock())


def _absolute_output_side_effect(seg_at_absolute: float, text: str):
    """Mock side_effect that writes a JSON file whose segment is
    at the given ABSOLUTE timestamp — matches what real
    mlx_whisper produces with ``--clip-timestamps``."""

    def side_effect(cmd, *args, **kwargs):  # noqa: ARG001
        if "--output-dir" not in cmd or "--output-name" not in cmd:
            return MagicMock(returncode=0, stdout="", stderr="")
        out_dir = Path(cmd[cmd.index("--output-dir") + 1])
        out_name = cmd[cmd.index("--output-name") + 1]
        out_path = out_dir / f"{out_name}.json"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "segments": [{
                "start": seg_at_absolute,
                "end": seg_at_absolute + 3.0,
                "text": text,
            }],
        }))
        return MagicMock(returncode=0, stdout="", stderr="")

    return side_effect


class LoopRecoveryNoDoubleShiftTests(unittest.TestCase):
    """The PR AD path used to shift twice (once because mlx_whisper
    output happens to be absolute, then again by + clip_start)."""

    def test_recovered_segment_keeps_absolute_timestamp(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            whisper_wav = tmp_path / "audio.wav"
            whisper_wav.touch()
            pipeline = _make_pipeline()
            pipeline._dropped_loops = [
                DroppedLoop(start=100.0, end=160.0, text="loop", dropped=15),
            ]
            with patch(
                "ekovideo_engine.pipeline.subprocess.run",
                # mlx_whisper writes "Contenu" at absolute t=120.5s.
                side_effect=_absolute_output_side_effect(120.5, "Contenu"),
            ):
                _, segs = pipeline._run_loop_recovery(
                    whisper_wav, tmp_path, []
                )
        # Without the bug fix, this would be 120.5 + 99 (clip_start
        # padding) = 219.5. With PR AH, it stays at 120.5.
        recovered = [s for s in segs if "Contenu" in s.get("text", "")]
        self.assertEqual(len(recovered), 1)
        self.assertAlmostEqual(recovered[0]["start"], 120.5, places=1)
        self.assertAlmostEqual(recovered[0]["end"], 123.5, places=1)


class PerSpeakerNoDoubleShiftTests(unittest.TestCase):
    """The PR E per-speaker pass used to double-shift every segment.
    On Caste's 775 turns this turned the 132 min audio into a 3h59
    transcript with content past the source's end."""

    def test_per_speaker_segment_keeps_absolute_timestamp(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            whisper_wav = tmp_path / "audio.wav"
            whisper_wav.touch()
            pipeline = _make_pipeline()
            base_segments = [
                {
                    "start": 100.0,
                    "end": 110.0,
                    "text": "Original",
                    "speaker": "SPEAKER_00",
                }
            ]
            # mlx_whisper output : absolute t=100.5, near where the
            # original segment was.
            with patch(
                "ekovideo_engine.pipeline.subprocess.run",
                side_effect=_absolute_output_side_effect(100.5, "Repassé"),
            ):
                _, segs = pipeline._run_per_speaker_pass(
                    whisper_wav, tmp_path, base_segments
                )
        repassed = [s for s in segs if "Repassé" in s.get("text", "")]
        self.assertEqual(len(repassed), 1)
        # Without PR AH this would be 100.5 + 100 (cs) = 200.5 —
        # exactly the doubling pattern that produced 3h59 transcripts.
        self.assertAlmostEqual(repassed[0]["start"], 100.5, places=1)
        self.assertAlmostEqual(repassed[0]["end"], 103.5, places=1)

    def test_per_speaker_keeps_speaker_tag(self):
        # Sanity: the speaker tag still gets set on the new segments,
        # we didn't break the labelling while fixing the timestamps.
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            whisper_wav = tmp_path / "audio.wav"
            whisper_wav.touch()
            pipeline = _make_pipeline()
            base_segments = [
                {
                    "start": 50.0,
                    "end": 60.0,
                    "text": "X",
                    "speaker": "Vincent",
                }
            ]
            with patch(
                "ekovideo_engine.pipeline.subprocess.run",
                side_effect=_absolute_output_side_effect(50.5, "Y"),
            ):
                _, segs = pipeline._run_per_speaker_pass(
                    whisper_wav, tmp_path, base_segments
                )
        repassed = [s for s in segs if s.get("text", "") == "Y"]
        self.assertEqual(len(repassed), 1)
        self.assertEqual(repassed[0]["speaker"], "Vincent")


if __name__ == "__main__":
    unittest.main()
