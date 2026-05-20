"""Pure-Python speaker recognition primitives.

The heavy lifting (loading the pyannote embedding model, running
inference on audio) is shelled out to the managed venv via
``transcription_utils.build_embedding_extract_cmd``. Once we have the
512-dim vectors back, everything else is plain numpy-style maths
that fits in this module — and stays trivially testable without a
GPU or HuggingFace token.

Two responsibilities:

1. **Aggregate** several embeddings for the same speaker (one per
   sampled segment) into a single centroid. Pyannote embeddings are
   already L2-normalised, so a plain mean works well; we re-normalise
   afterwards to keep cosine similarity bounded by [-1, 1].

2. **Match** a freshly-extracted cluster centroid against every
   stored profile and surface the best fit *if* it clears the
   confidence threshold. We're conservative on purpose — silently
   pre-filling the wrong name is more annoying than asking the user
   to type it themselves.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass


__all__ = [
    "SpeakerMatch",
    "EmbeddingMismatchError",
    "aggregate_embeddings",
    "cosine_similarity",
    "match_cluster_against_profiles",
    "merge_into_existing_centroid",
    "encode_embedding",
    "decode_embedding",
]


# Cosine similarity threshold above which we treat a cluster as
# "definitely the same person as the stored profile". 0.75 is the
# canonical value cited in pyannote's own README for verification —
# tighter than the diarisation clustering threshold because we'd
# rather pre-fill nothing than the wrong name.
DEFAULT_MATCH_THRESHOLD = 0.75


class EmbeddingMismatchError(ValueError):
    """Raised when two embeddings have incompatible shapes.

    Pyannote always returns 512-dim vectors but a corrupt JSON file
    or a future model swap could produce something different. Better
    to fail loudly than to compute a meaningless cosine.
    """


@dataclass(frozen=True)
class SpeakerMatch:
    """Result of looking up a cluster centroid against the store.

    ``profile_name`` is empty when nothing crossed the threshold —
    the caller leaves the SPEAKER_NN placeholder alone in that case.
    """

    cluster_label: str
    profile_name: str
    similarity: float


def encode_embedding(vector: list[float]) -> str:
    """Serialise an embedding vector as compact JSON.

    Stored as a TEXT column because SQLite's BLOB is awkward to
    inspect from sqlite3 CLI when debugging. JSON of 512 floats is
    ~5 KB per profile — negligible.
    """
    return json.dumps(list(vector), separators=(",", ":"))


def decode_embedding(blob: str) -> list[float]:
    """Counterpart to :func:`encode_embedding`. Returns a plain list
    of floats. Empty / malformed input yields an empty list so the
    caller can detect "no embedding" via ``not vector``."""
    if not blob:
        return []
    try:
        data = json.loads(blob)
    except (TypeError, ValueError):
        return []
    if not isinstance(data, list):
        return []
    out: list[float] = []
    for value in data:
        try:
            out.append(float(value))
        except (TypeError, ValueError):
            return []
    return out


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Standard cosine similarity. Returns 0.0 when either vector is
    empty or zero-norm — silently — because the only callers care
    about a single yes/no decision against a threshold and a 0.0
    score is unambiguously below.
    """
    if not a or not b:
        return 0.0
    if len(a) != len(b):
        raise EmbeddingMismatchError(
            f"embedding length mismatch: {len(a)} vs {len(b)}"
        )
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / math.sqrt(norm_a * norm_b)


def _normalise(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0:
        return list(vector)
    return [v / norm for v in vector]


def aggregate_embeddings(embeddings: list[list[float]]) -> list[float]:
    """Mean of L2-normalised embeddings, re-normalised.

    Pyannote already L2-normalises each output, so a straight mean
    is a sensible centroid. We re-normalise so subsequent cosine
    similarity stays in [-1, 1] and the threshold maths is honest.
    """
    if not embeddings:
        return []
    dim = len(embeddings[0])
    if any(len(v) != dim for v in embeddings):
        raise EmbeddingMismatchError("aggregate: vectors have different lengths")
    summed = [0.0] * dim
    count = 0
    for vector in embeddings:
        for i, v in enumerate(vector):
            summed[i] += v
        count += 1
    if count == 0:
        return []
    averaged = [v / count for v in summed]
    return _normalise(averaged)


def merge_into_existing_centroid(
    *,
    existing_centroid: list[float],
    existing_count: int,
    new_embeddings: list[list[float]],
) -> tuple[list[float], int]:
    """Incrementally average a stored centroid with fresh embeddings.

    Used when the user re-confirms a name on a second meeting: we
    don't keep every raw embedding, just the running mean and the
    sample count, so adding new evidence is ``(c*old + new) / (c+1)``
    repeated for each new vector. Re-normalised at the end for the
    same reason as :func:`aggregate_embeddings`.

    Returns ``(centroid, sample_count)``. When ``existing_centroid``
    is empty / shape-incompatible we fall back to a fresh aggregate
    so a corrupted profile heals on next enrollment instead of
    rejecting forever.
    """
    if not new_embeddings:
        return list(existing_centroid), int(existing_count)
    new_dim = len(new_embeddings[0])
    if not existing_centroid or len(existing_centroid) != new_dim:
        return aggregate_embeddings(new_embeddings), len(new_embeddings)

    accumulator = [v * existing_count for v in existing_centroid]
    count = int(existing_count)
    for vector in new_embeddings:
        if len(vector) != new_dim:
            raise EmbeddingMismatchError("merge: vectors have different lengths")
        for i, v in enumerate(vector):
            accumulator[i] += v
        count += 1
    averaged = [v / count for v in accumulator]
    return _normalise(averaged), count


def match_cluster_against_profiles(
    cluster_centroid: list[float],
    profiles: list[dict],
    *,
    threshold: float = DEFAULT_MATCH_THRESHOLD,
) -> SpeakerMatch | None:
    """Return the best-matching profile, or None when no profile
    crosses ``threshold``.

    ``profiles`` is the shape returned by
    ``DatabaseManager.list_speaker_profiles`` — each dict carries an
    ``embedding_json`` we decode lazily, plus a ``name`` for the
    surface-level result.
    """
    if not cluster_centroid or not profiles:
        return None
    best_name = ""
    best_score = -1.0
    for profile in profiles:
        embedding = decode_embedding(profile.get("embedding_json") or "")
        if not embedding:
            continue
        try:
            score = cosine_similarity(cluster_centroid, embedding)
        except EmbeddingMismatchError:
            continue
        if score > best_score:
            best_score = score
            best_name = str(profile.get("name") or "")
    if best_score < threshold or not best_name:
        return None
    return SpeakerMatch(
        cluster_label="",  # caller fills this in (we don't know it here)
        profile_name=best_name,
        similarity=best_score,
    )


def score_cluster_against_all_profiles(
    cluster_centroid: list[float],
    profiles: list[dict],
) -> list[tuple[str, float]]:
    """Return ``[(profile_name, score), ...]`` sorted descending.

    PR T diagnostic helper: lets the pipeline log the TOP-K profiles
    for every cluster, not just the winning match. Without this,
    the only signal in app_log is "best match was Clothilde" — and
    when that turns out to be the WRONG speaker (the case we
    audited on the CVR run), there's no way to tell whether
    Clothilde was a marginal-pass or whether Robin's profile
    scored almost as high.

    Profiles with empty embeddings (``embedding_json`` blank or
    invalid) are skipped silently — they would be dead weight in
    the diagnostic too.
    """
    if not cluster_centroid or not profiles:
        return []
    scored: list[tuple[str, float]] = []
    for profile in profiles:
        embedding = decode_embedding(profile.get("embedding_json") or "")
        if not embedding:
            continue
        name = str(profile.get("name") or "").strip()
        if not name:
            continue
        try:
            score = cosine_similarity(cluster_centroid, embedding)
        except EmbeddingMismatchError:
            continue
        scored.append((name, score))
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored


def filter_usable_profiles(profiles: list[dict]) -> list[dict]:
    """Drop profiles that have no usable embedding.

    PR T defense in depth: a stale ``speaker_profiles`` row with an
    empty ``embedding_json`` (or one that decodes to an empty list)
    should not even reach the matcher. The audit on the CVR run
    showed the library had accumulated rows like
    ``Benjamin`` whose samples were 78-byte WAV stubs from a
    previous failed enrollment — their embeddings are unusable but
    the row is still listed.

    Also requires ``sample_count`` ≥ 1 when the column is present:
    a profile that has never been confirmed by the user is a guess,
    not a match anchor.
    """
    out: list[dict] = []
    for profile in profiles or []:
        if not decode_embedding(profile.get("embedding_json") or ""):
            continue
        sample_count = profile.get("sample_count")
        if sample_count is not None:
            try:
                if int(sample_count) < 1:
                    continue
            except (TypeError, ValueError):
                # Unknown column type → don't filter on it.
                pass
        out.append(profile)
    return out
