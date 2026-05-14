import json
import unittest

import vad_silero
from vad_silero import (
    build_vad_cmd,
    parse_vad_manifest,
    remap_segment_to_source,
    remap_segments_to_source,
)


class BuildVadCmdTest(unittest.TestCase):
    def test_carries_paths_and_tunables(self):
        cmd = build_vad_cmd(
            "/v/bin/python",
            "/tmp/in.wav",
            "/tmp/trim.wav",
            min_speech_ms=300,
            min_silence_ms=600,
            pad_ms=150,
            threshold=0.4,
        )
        self.assertEqual(cmd[0], "/v/bin/python")
        self.assertEqual(cmd[1], "-c")
        self.assertEqual(cmd[3], "/tmp/in.wav")
        self.assertEqual(cmd[4], "/tmp/trim.wav")
        self.assertEqual(cmd[5], "300")
        self.assertEqual(cmd[6], "600")
        self.assertEqual(cmd[7], "150")
        self.assertEqual(cmd[8], "0.400")
        # The inline script must use silero_vad — that's our contract
        # with the managed venv installer.
        self.assertIn("silero_vad", cmd[2])

    def test_inline_script_is_valid_python(self):
        compile(vad_silero._VAD_SCRIPT, "<vad>", "exec")


class ParseVadManifestTest(unittest.TestCase):
    def test_parses_well_formed_manifest(self):
        payload = {
            "spans": [
                {"src_start": 12.0, "src_end": 18.0, "trim_start": 0.0, "trim_end": 6.0},
                {"src_start": 60.0, "src_end": 65.0, "trim_start": 6.0, "trim_end": 11.0},
            ],
            "total_seconds": 120.0,
            "trimmed_seconds": 11.0,
        }
        out = parse_vad_manifest(json.dumps(payload))
        self.assertEqual(out["spans"], payload["spans"])
        self.assertEqual(out["total_seconds"], 120.0)

    def test_picks_last_line_when_torch_hub_prints_progress(self):
        # Torch.hub prints "Downloading: …" the first time the model
        # is fetched. The manifest is always the last line.
        text = (
            "Downloading: https://github.com/snakers4/silero-vad/...\n"
            "Loading model…\n"
            + json.dumps({"spans": [], "total_seconds": 5.0})
        )
        out = parse_vad_manifest(text)
        self.assertEqual(out["spans"], [])

    def test_raises_on_error_payload(self):
        with self.assertRaises(RuntimeError):
            parse_vad_manifest(json.dumps({"error": "torch missing"}))

    def test_raises_on_empty_output(self):
        with self.assertRaises(RuntimeError):
            parse_vad_manifest("")


class RemapTimestampsTest(unittest.TestCase):
    """
    The whole point of the manifest is that Whisper sees a trimmed
    timeline but the user must see source timestamps. Mismapping
    by even a few seconds means segments line up with the wrong
    audio when re-played.
    """

    SAMPLE_MANIFEST = [
        # 12s of speech from 12.0–24.0, then 8s from 60.0–68.0.
        {"src_start": 12.0, "src_end": 24.0, "trim_start": 0.0, "trim_end": 12.0},
        {"src_start": 60.0, "src_end": 68.0, "trim_start": 12.0, "trim_end": 20.0},
    ]

    def test_first_span_offset(self):
        # Whisper's "0.0 → 6.0" actually means 12.0 → 18.0 in source.
        s, e = remap_segment_to_source(0.0, 6.0, self.SAMPLE_MANIFEST)
        self.assertAlmostEqual(s, 12.0)
        self.assertAlmostEqual(e, 18.0)

    def test_second_span_offset(self):
        # Trim 14.0 → 18.0 falls in the second kept span (12.0–20.0
        # on trim = 60.0–68.0 on source). Offset 2.0 into that span.
        s, e = remap_segment_to_source(14.0, 18.0, self.SAMPLE_MANIFEST)
        self.assertAlmostEqual(s, 62.0)
        self.assertAlmostEqual(e, 66.0)

    def test_endpoint_at_span_boundary(self):
        # Boundary at 12.0 (trim) should map cleanly to either end of
        # span 1 or start of span 2 — both happen to be different on
        # the source side, so the implementation has to pick one.
        # We accept the first containing span.
        s, _ = remap_segment_to_source(12.0, 12.0, self.SAMPLE_MANIFEST)
        # Either 24.0 (end of span 1) or 60.0 (start of span 2) is
        # acceptable; we don't pin a choice in the contract.
        self.assertIn(s, {24.0, 60.0})

    def test_remap_segments_preserves_extras(self):
        segments = [
            {"start": 1.0, "end": 4.0, "text": "Bonjour", "speaker": "S00"},
            {"start": 14.0, "end": 15.5, "text": "Merci", "speaker": "S01"},
        ]
        out = remap_segments_to_source(segments, self.SAMPLE_MANIFEST)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]["text"], "Bonjour")
        self.assertEqual(out[0]["speaker"], "S00")
        self.assertAlmostEqual(out[0]["start"], 13.0)
        self.assertAlmostEqual(out[0]["end"], 16.0)
        self.assertAlmostEqual(out[1]["start"], 62.0)
        self.assertAlmostEqual(out[1]["end"], 63.5)

    def test_empty_manifest_passes_through(self):
        segments = [{"start": 1.0, "end": 2.0, "text": "X"}]
        out = remap_segments_to_source(segments, [])
        self.assertEqual(out, segments)
        self.assertIsNot(out[0], segments[0])  # cloned


if __name__ == "__main__":
    unittest.main()
