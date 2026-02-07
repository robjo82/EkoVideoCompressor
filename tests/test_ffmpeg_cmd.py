import unittest

from ffmpeg_utils import build_ffmpeg_cmd


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


if __name__ == "__main__":
    unittest.main()
