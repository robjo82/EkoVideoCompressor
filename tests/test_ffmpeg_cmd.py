import unittest

from ffmpeg_utils import build_ffmpeg_cmd
from transcription_utils import build_audio_extract_cmd, build_mlx_whisper_cmd, default_transcript_path


class BuildFfmpegCmdTest(unittest.TestCase):
    def test_build_command_with_trim_and_audio_options(self):
        cmd = build_ffmpeg_cmd(
            ffmpeg_path="/usr/local/bin/ffmpeg",
            in_path="/tmp/in.mp4",
            out_path="/tmp/out.mp4",
            crf=30,
            resolution="480p",
            fps=10,
            audio_bitrate="96k",
            preset="veryfast",
            speech_enhance=True,
            mono_audio=True,
            ss="00:00:05",
            to="00:01:00",
        )

        self.assertIn("-ss", cmd)
        self.assertIn("00:00:05", cmd)
        self.assertIn("-to", cmd)
        self.assertIn("00:01:00", cmd)
        self.assertIn("-af", cmd)
        self.assertIn("-ac", cmd)
        self.assertIn("1", cmd)
        self.assertIn("-preset", cmd)
        self.assertIn("veryfast", cmd)
        self.assertIn("-crf", cmd)
        self.assertIn("30", cmd)
        self.assertIn("scale=-2:480,fps=10", cmd)
        self.assertEqual(cmd[-1], "/tmp/out.mp4")


class TranscriptionCommandTest(unittest.TestCase):
    def test_build_audio_extract_command_uses_original_audio(self):
        cmd = build_audio_extract_cmd(
            ffmpeg_path="/usr/local/bin/ffmpeg",
            in_path="/tmp/in.mp4",
            wav_path="/tmp/audio.wav",
            speech_enhance=True,
            ss="00:00:05",
            to="00:01:00",
        )

        self.assertIn("-vn", cmd)
        self.assertIn("-af", cmd)
        self.assertIn("pcm_s16le", cmd)
        self.assertIn("16000", cmd)
        self.assertIn("-ac", cmd)
        self.assertIn("1", cmd)
        self.assertEqual(cmd[-1], "/tmp/audio.wav")

    def test_build_mlx_whisper_command_uses_context_prompt(self):
        cmd = build_mlx_whisper_cmd(
            mlx_whisper_path="/opt/homebrew/bin/mlx_whisper",
            audio_path="/tmp/audio.wav",
            output_path="/tmp/reunion_transcription.srt",
            model="mlx-community/whisper-large-v3-turbo",
            language="fr",
            output_format="srt",
            initial_prompt="Noms propres: Ekonum, Maia.",
        )

        self.assertEqual(cmd[0], "/opt/homebrew/bin/mlx_whisper")
        self.assertIn("--model", cmd)
        self.assertIn("mlx-community/whisper-large-v3-turbo", cmd)
        self.assertIn("-f", cmd)
        self.assertIn("srt", cmd)
        self.assertIn("--language", cmd)
        self.assertIn("fr", cmd)
        self.assertIn("--initial-prompt", cmd)

    def test_default_transcript_path_uses_format_extension(self):
        path = default_transcript_path("/tmp/reunion.mp4", "/tmp", "_notes", "json")
        self.assertEqual(path, "/tmp/reunion_notes.json")


if __name__ == "__main__":
    unittest.main()
