import unittest

from per_speaker import (
    SpeakerTurn,
    build_speaker_slices,
    merge_speaker_segments,
    remap_speaker_segments,
)


class BuildSpeakerSlicesTest(unittest.TestCase):
    def test_groups_turns_by_speaker(self):
        turns = [
            {"speaker": "S0", "start": 0.0, "end": 5.0},
            {"speaker": "S1", "start": 5.0, "end": 9.0},
            {"speaker": "S0", "start": 10.0, "end": 14.0},
            {"speaker": "S1", "start": 15.0, "end": 18.0},
        ]
        slices = build_speaker_slices(turns, pad_seconds=0.0, min_speaker_total=2.0)
        self.assertIn("S0", slices)
        self.assertIn("S1", slices)
        # S0 has 5 + 4 = 9 s of speech.
        self.assertAlmostEqual(slices["S0"].duration, 9.0)
        # S1 has 4 + 3 = 7 s.
        self.assertAlmostEqual(slices["S1"].duration, 7.0)

    def test_drops_speakers_below_total(self):
        # S1 only has 1.5 s of speech total — pyannote often labels
        # background cough as its own speaker. Drop it.
        turns = [
            {"speaker": "S0", "start": 0.0, "end": 60.0},
            {"speaker": "S1", "start": 30.0, "end": 31.5},
        ]
        slices = build_speaker_slices(turns, min_speaker_total=5.0)
        self.assertIn("S0", slices)
        self.assertNotIn("S1", slices)

    def test_drops_very_short_turns(self):
        # 200 ms turns are noise — should not contribute.
        turns = [
            {"speaker": "S0", "start": 0.0, "end": 30.0},
            {"speaker": "S0", "start": 30.2, "end": 30.3},  # 100 ms
            {"speaker": "S0", "start": 40.0, "end": 50.0},
        ]
        slices = build_speaker_slices(
            turns, pad_seconds=0.0, min_turn_duration=0.4, min_speaker_total=5.0
        )
        # Only the two meaningful turns are kept (~30 + 10 = 40 s).
        self.assertAlmostEqual(slices["S0"].duration, 40.0)

    def test_handles_dataclass_input(self):
        turns = [
            SpeakerTurn(speaker="S0", start=0.0, end=10.0),
            SpeakerTurn(speaker="S0", start=20.0, end=30.0),
        ]
        slices = build_speaker_slices(turns, pad_seconds=0.0, min_speaker_total=5.0)
        self.assertEqual(slices["S0"].duration, 20.0)


class RemapSpeakerSegmentsTest(unittest.TestCase):
    def test_shifts_segments_to_source_timeline(self):
        # Speaker had two turns: 10–15s and 50–55s on source.
        # On the per-speaker stream they're 0–5s and 5–10s.
        turns = [
            {"speaker": "S0", "start": 10.0, "end": 15.0},
            {"speaker": "S0", "start": 50.0, "end": 55.0},
        ]
        slices = build_speaker_slices(
            turns, pad_seconds=0.0, min_speaker_total=2.0
        )
        whisper_out = [
            {"start": 0.0, "end": 4.0, "text": "Bonjour"},   # in span 1
            {"start": 6.0, "end": 9.0, "text": "Au revoir"},  # in span 2
        ]
        remapped = remap_speaker_segments(whisper_out, slices["S0"])
        self.assertEqual(len(remapped), 2)
        self.assertAlmostEqual(remapped[0]["start"], 10.0)
        self.assertAlmostEqual(remapped[0]["end"], 14.0)
        self.assertEqual(remapped[0]["speaker"], "S0")
        self.assertAlmostEqual(remapped[1]["start"], 51.0)
        self.assertAlmostEqual(remapped[1]["end"], 54.0)


class MergeSpeakerSegmentsTest(unittest.TestCase):
    def test_sorts_by_start_time_across_speakers(self):
        s0 = [
            {"start": 0.0, "end": 5.0, "text": "Bonjour", "speaker": "S0"},
            {"start": 20.0, "end": 22.0, "text": "Merci", "speaker": "S0"},
        ]
        s1 = [
            {"start": 5.0, "end": 10.0, "text": "Salut", "speaker": "S1"},
            {"start": 15.0, "end": 18.0, "text": "OK", "speaker": "S1"},
        ]
        merged = merge_speaker_segments([s0, s1])
        speakers_order = [seg["speaker"] for seg in merged]
        self.assertEqual(speakers_order, ["S0", "S1", "S1", "S0"])


if __name__ == "__main__":
    unittest.main()
