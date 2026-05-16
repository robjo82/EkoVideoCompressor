"""Tests for the speaker recognition primitives.

These cover the pure-Python side: cosine similarity, embedding
aggregation, incremental centroid merging, threshold matching. The
inline pyannote subprocess is a separate concern — we mock the
shell-out at the integration boundary in test_engine_library.
"""

from __future__ import annotations

import math
import unittest

from speaker_recognition import (
    DEFAULT_MATCH_THRESHOLD,
    EmbeddingMismatchError,
    aggregate_embeddings,
    cosine_similarity,
    decode_embedding,
    encode_embedding,
    match_cluster_against_profiles,
    merge_into_existing_centroid,
)


def _unit_vector(length: int, hot_index: int) -> list[float]:
    """Build a one-hot vector of length ``length`` so we can build
    handpicked similarities in tests (cosine of two one-hot vectors
    is 1.0 if they share an index, 0.0 otherwise)."""
    out = [0.0] * length
    out[hot_index] = 1.0
    return out


class CosineSimilarityTest(unittest.TestCase):
    def test_identical_vectors_score_one(self):
        v = [1.0, 2.0, 3.0]
        self.assertAlmostEqual(cosine_similarity(v, v), 1.0)

    def test_orthogonal_vectors_score_zero(self):
        a = _unit_vector(8, 0)
        b = _unit_vector(8, 1)
        self.assertAlmostEqual(cosine_similarity(a, b), 0.0)

    def test_opposite_vectors_score_minus_one(self):
        a = [1.0, 0.0, 0.0]
        b = [-1.0, 0.0, 0.0]
        self.assertAlmostEqual(cosine_similarity(a, b), -1.0)

    def test_zero_norm_returns_zero_silently(self):
        # Zero-norm input would make the standard formula divide by
        # zero. We return 0.0 instead so the threshold check below
        # cleanly rejects.
        self.assertEqual(cosine_similarity([0.0, 0.0], [1.0, 1.0]), 0.0)

    def test_length_mismatch_raises(self):
        with self.assertRaises(EmbeddingMismatchError):
            cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0])


class AggregateEmbeddingsTest(unittest.TestCase):
    def test_returns_normalised_mean(self):
        v1 = [1.0, 0.0]
        v2 = [0.0, 1.0]
        out = aggregate_embeddings([v1, v2])
        # Mean is (0.5, 0.5); normalised becomes (0.707, 0.707).
        self.assertAlmostEqual(out[0], math.sqrt(0.5))
        self.assertAlmostEqual(out[1], math.sqrt(0.5))

    def test_empty_input_returns_empty(self):
        self.assertEqual(aggregate_embeddings([]), [])

    def test_mismatched_lengths_raise(self):
        with self.assertRaises(EmbeddingMismatchError):
            aggregate_embeddings([[1.0, 2.0], [1.0, 2.0, 3.0]])


class MergeCentroidTest(unittest.TestCase):
    def test_incremental_average_matches_recomputed_average(self):
        # Walking three vectors one at a time should land on the
        # same centroid as aggregating them in one shot. This is
        # the contract that lets us not store every raw embedding.
        v1 = [1.0, 0.0, 0.0]
        v2 = [0.0, 1.0, 0.0]
        v3 = [0.0, 0.0, 1.0]
        full = aggregate_embeddings([v1, v2, v3])

        centroid, count = merge_into_existing_centroid(
            existing_centroid=[],
            existing_count=0,
            new_embeddings=[v1],
        )
        centroid, count = merge_into_existing_centroid(
            existing_centroid=centroid,
            existing_count=count,
            new_embeddings=[v2, v3],
        )
        for a, b in zip(centroid, full):
            self.assertAlmostEqual(a, b, places=4)
        self.assertEqual(count, 3)

    def test_dimension_mismatch_resets_to_fresh_centroid(self):
        # A profile from an old engine version that emitted 256-dim
        # vectors shouldn't poison a new run that emits 512-dim.
        # Falling back to a fresh aggregate heals on first re-enrol.
        old_centroid = [1.0, 0.0, 0.0]
        new_vectors = [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
        centroid, count = merge_into_existing_centroid(
            existing_centroid=old_centroid,
            existing_count=10,
            new_embeddings=new_vectors,
        )
        self.assertEqual(len(centroid), 4)
        self.assertEqual(count, 2)


class MatchClusterTest(unittest.TestCase):
    def _profile(self, name: str, vector: list[float], count: int = 1) -> dict:
        return {
            "name": name,
            "embedding_json": encode_embedding(vector),
            "sample_count": count,
        }

    def test_returns_best_match_above_threshold(self):
        cluster = [1.0, 0.0, 0.0]
        profiles = [
            self._profile("Robin", [1.0, 0.0, 0.0]),
            self._profile("David", [0.0, 1.0, 0.0]),
        ]
        match = match_cluster_against_profiles(cluster, profiles)
        self.assertIsNotNone(match)
        self.assertEqual(match.profile_name, "Robin")
        self.assertAlmostEqual(match.similarity, 1.0)

    def test_no_match_below_threshold(self):
        cluster = [1.0, 0.0, 0.0]
        # All profiles orthogonal to the cluster — score 0, well
        # below the 0.75 threshold.
        profiles = [
            self._profile("Robin", [0.0, 1.0, 0.0]),
            self._profile("David", [0.0, 0.0, 1.0]),
        ]
        self.assertIsNone(match_cluster_against_profiles(cluster, profiles))

    def test_threshold_can_be_relaxed_per_call(self):
        # Two near-orthogonal vectors don't pass the default
        # threshold but do pass a lenient one. Useful for power-user
        # tooling that wants to see weaker suggestions.
        cluster = [0.7, 0.7, 0.0]
        cluster_norm = [v / math.sqrt(sum(x * x for x in cluster)) for v in cluster]
        profiles = [self._profile("Robin", [1.0, 0.0, 0.0])]
        # Cosine ~ 0.707 — below 0.75 but above 0.5.
        self.assertIsNone(match_cluster_against_profiles(cluster_norm, profiles))
        match = match_cluster_against_profiles(
            cluster_norm, profiles, threshold=0.5
        )
        self.assertIsNotNone(match)

    def test_corrupt_profile_skipped_silently(self):
        # A profile row with bad JSON shouldn't kill the lookup —
        # the rest of the store still gets evaluated.
        cluster = [1.0, 0.0, 0.0]
        profiles = [
            {"name": "Corrupt", "embedding_json": "{not json", "sample_count": 1},
            self._profile("Robin", [1.0, 0.0, 0.0]),
        ]
        match = match_cluster_against_profiles(cluster, profiles)
        self.assertIsNotNone(match)
        self.assertEqual(match.profile_name, "Robin")


class EncodingRoundTripTest(unittest.TestCase):
    def test_encode_decode_preserves_values(self):
        v = [0.1, -0.2, 1e-9, 3.14]
        out = decode_embedding(encode_embedding(v))
        self.assertEqual(len(out), len(v))
        for a, b in zip(out, v):
            self.assertAlmostEqual(a, b, places=12)

    def test_decode_handles_garbage_gracefully(self):
        self.assertEqual(decode_embedding(""), [])
        self.assertEqual(decode_embedding("not json"), [])
        self.assertEqual(decode_embedding('{"foo": 1}'), [])
        self.assertEqual(decode_embedding('["a", "b"]'), [])


if __name__ == "__main__":
    unittest.main()
