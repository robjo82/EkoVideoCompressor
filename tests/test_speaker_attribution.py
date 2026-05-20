"""
PR T — speaker attribution diagnostics + dominant-window heuristic.

Covers:
  • ``filter_usable_profiles`` drops rows with empty embeddings /
    zero sample_count (stale profiles from prior failed enrolments).
  • ``score_cluster_against_all_profiles`` returns every candidate
    with its cosine score, sorted descending — feeds the audit log.
  • ``_pre_attribute_current_user`` now picks the cluster
    *dominating* the first 60 seconds, not the first to speak.
    This regression test is the CVR-control failure mode: Robin
    says "Bonjour" (3 s), Vincent struggles for 12 s of
    silence-prefixed audio that Whisper timestamps slightly
    earlier, then Robin takes over for the rest of the meeting.
    Old heuristic latched onto Vincent; new heuristic correctly
    picks Robin.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from ekovideo_engine.models import JobRequest, TranscriptionSettings
from ekovideo_engine.pipeline import TranscriptionPipeline
from speaker_recognition import (
    DEFAULT_MATCH_THRESHOLD,
    encode_embedding,
    filter_usable_profiles,
    score_cluster_against_all_profiles,
)


# ---------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------


def _profile(
    name: str,
    embedding: list[float] | None = None,
    *,
    sample_count: int | None = 1,
) -> dict:
    """Shape that mirrors what ``DatabaseManager.list_speaker_profiles``
    yields. ``embedding=None`` produces an empty embedding_json so we
    can test the stale-row filter."""
    out: dict = {
        "name": name,
        "embedding_json": encode_embedding(embedding) if embedding else "",
    }
    if sample_count is not None:
        out["sample_count"] = sample_count
    return out


class FilterUsableProfilesTests(unittest.TestCase):
    def test_drops_profile_with_empty_embedding(self):
        profiles = [
            _profile("Robin", embedding=[0.1, 0.2, 0.3]),
            _profile("Benjamin", embedding=None),  # stale, no embedding
        ]
        usable = filter_usable_profiles(profiles)
        self.assertEqual([p["name"] for p in usable], ["Robin"])

    def test_drops_profile_with_zero_sample_count(self):
        profiles = [
            _profile("Robin", embedding=[0.1, 0.2, 0.3], sample_count=2),
            _profile("Laurent", embedding=[0.4, 0.5, 0.6], sample_count=0),
        ]
        usable = filter_usable_profiles(profiles)
        self.assertEqual([p["name"] for p in usable], ["Robin"])

    def test_keeps_profile_when_sample_count_missing(self):
        # Legacy rows without the ``sample_count`` column don't get
        # filtered on that criterion — only the embedding check
        # applies.
        legacy = {
            "name": "Manon",
            "embedding_json": encode_embedding([0.1, 0.2, 0.3]),
            # no sample_count key
        }
        usable = filter_usable_profiles([legacy])
        self.assertEqual([p["name"] for p in usable], ["Manon"])

    def test_empty_input_yields_empty(self):
        self.assertEqual(filter_usable_profiles([]), [])
        self.assertEqual(filter_usable_profiles(None), [])


class ScoreClusterAgainstAllProfilesTests(unittest.TestCase):
    def test_returns_sorted_descending_by_score(self):
        # Build three profiles where ``Robin`` is closest to the
        # query centroid.
        query = [1.0, 0.0]
        profiles = [
            _profile("Vincent", embedding=[-1.0, 0.0]),   # cos = -1
            _profile("Robin", embedding=[1.0, 0.0]),      # cos = +1
            _profile("Clothilde", embedding=[0.0, 1.0]),  # cos = 0
        ]
        scored = score_cluster_against_all_profiles(query, profiles)
        names = [name for name, _ in scored]
        self.assertEqual(names, ["Robin", "Clothilde", "Vincent"])
        # Scores roughly in (-1, 0, +1) — we don't assert exact
        # floats because cosine_similarity normalises.
        self.assertGreater(scored[0][1], scored[1][1])
        self.assertGreater(scored[1][1], scored[2][1])

    def test_skips_profiles_with_empty_embedding(self):
        query = [1.0, 0.0]
        profiles = [
            _profile("Robin", embedding=[1.0, 0.0]),
            _profile("Benjamin", embedding=None),
        ]
        scored = score_cluster_against_all_profiles(query, profiles)
        self.assertEqual([name for name, _ in scored], ["Robin"])

    def test_empty_inputs_return_empty(self):
        self.assertEqual(score_cluster_against_all_profiles([], []), [])
        self.assertEqual(score_cluster_against_all_profiles([1.0], []), [])
        self.assertEqual(
            score_cluster_against_all_profiles([], [_profile("X", [1.0])]),
            [],
        )


# ---------------------------------------------------------------------
# Pre-attribution heuristic
# ---------------------------------------------------------------------


def _pipeline(current_user: str) -> TranscriptionPipeline:
    request = JobRequest(
        source_path="/tmp/x.mov",
        output_dir="/tmp",
        mode="transcribe",
        transcription_settings=TranscriptionSettings(
            current_user_name=current_user,
        ),
    )
    return TranscriptionPipeline(request=request, sink=MagicMock())


class PreAttributionWindowTests(unittest.TestCase):
    """Replaces the old 'first-to-speak' test with the dominant-
    speaker-in-first-60-seconds contract."""

    def test_dominant_speaker_in_first_minute_wins(self):
        # CVR-control regression: Robin says "Bonjour" briefly,
        # Vincent answers with a longer turn, but Robin then
        # dominates the rest of the minute. With "dominant in
        # 60 s", Robin (= SPEAKER_00) wins.
        pipeline = _pipeline("Robin")
        segments = [
            {"start": 0.0, "end": 3.0, "speaker": "SPEAKER_00", "text": "Bonjour"},
            {"start": 3.0, "end": 15.0, "speaker": "SPEAKER_01", "text": "...pas de son..."},
            {"start": 15.0, "end": 55.0, "speaker": "SPEAKER_00", "text": "Reprise"},
        ]
        out = pipeline._pre_attribute_current_user(segments, already_recognized={})
        self.assertEqual(out, {"SPEAKER_00": "Robin"})

    def test_first_to_speak_no_longer_auto_wins(self):
        # Old behaviour: SPEAKER_01 (5 s segment, started at t=0)
        # would have won because it spoke first. New behaviour:
        # SPEAKER_00 dominates 40 s vs 5 s in the window.
        pipeline = _pipeline("Robin")
        segments = [
            {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_01", "text": "Court"},
            {"start": 5.0, "end": 45.0, "speaker": "SPEAKER_00", "text": "Long"},
        ]
        out = pipeline._pre_attribute_current_user(segments, already_recognized={})
        self.assertEqual(out, {"SPEAKER_00": "Robin"})

    def test_tie_breaks_to_earlier_start(self):
        # When two clusters speak the exact same amount, the one
        # that started earlier wins — preserves the deterministic
        # behaviour the previous "lowest-start-time" test pinned.
        pipeline = _pipeline("Robin")
        segments = [
            {"start": 5.0, "end": 6.0, "speaker": "SPEAKER_00", "text": "Tard"},
            {"start": 0.5, "end": 1.5, "speaker": "SPEAKER_01", "text": "Tôt"},
        ]
        out = pipeline._pre_attribute_current_user(segments, already_recognized={})
        self.assertEqual(out, {"SPEAKER_01": "Robin"})

    def test_segments_past_window_are_ignored(self):
        # SPEAKER_01 dominates the full meeting but only after the
        # 60-second window. SPEAKER_00 wins the opening, so
        # attribution still picks them.
        pipeline = _pipeline("Robin")
        segments = [
            {"start": 0.0, "end": 30.0, "speaker": "SPEAKER_00", "text": "Open"},
            {"start": 65.0, "end": 600.0, "speaker": "SPEAKER_01", "text": "Long"},
        ]
        out = pipeline._pre_attribute_current_user(segments, already_recognized={})
        self.assertEqual(out, {"SPEAKER_00": "Robin"})

    def test_skips_when_signal_too_weak(self):
        # All speech outside the window → no signal → no attribution.
        pipeline = _pipeline("Robin")
        segments = [
            {"start": 70.0, "end": 80.0, "speaker": "SPEAKER_00", "text": "Tard"},
        ]
        out = pipeline._pre_attribute_current_user(segments, already_recognized={})
        self.assertEqual(out, {})

    def test_skips_when_user_already_recognised(self):
        # Voice match already claimed Robin → heuristic stays out
        # of the way.
        pipeline = _pipeline("Robin")
        segments = [
            {"start": 0.0, "end": 30.0, "speaker": "SPEAKER_00", "text": "Open"},
        ]
        out = pipeline._pre_attribute_current_user(
            segments, already_recognized={"SPEAKER_01": "Robin"}
        )
        self.assertEqual(out, {})

    def test_skips_clusters_already_voice_matched(self):
        # A cluster that voice match named (even with a different
        # name) is locked. Heuristic only assigns to the
        # remaining unnamed clusters.
        pipeline = _pipeline("Robin")
        segments = [
            {"start": 0.0, "end": 30.0, "speaker": "SPEAKER_00", "text": "Long"},
            {"start": 30.0, "end": 35.0, "speaker": "SPEAKER_01", "text": "Court"},
        ]
        out = pipeline._pre_attribute_current_user(
            segments, already_recognized={"SPEAKER_00": "Clothilde"}
        )
        # SPEAKER_00 is locked → SPEAKER_01 wins by default.
        self.assertEqual(out, {"SPEAKER_01": "Robin"})

    def test_no_attribution_when_name_blank(self):
        pipeline = _pipeline("")
        segments = [
            {"start": 0.0, "end": 30.0, "speaker": "SPEAKER_00", "text": "X"},
        ]
        out = pipeline._pre_attribute_current_user(segments, already_recognized={})
        self.assertEqual(out, {})


if __name__ == "__main__":
    unittest.main()
