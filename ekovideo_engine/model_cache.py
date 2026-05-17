from __future__ import annotations

import shutil
from pathlib import Path

from transcription_utils import (
    AUDIO_LLM_MODELS,
    DIARISATION_MODELS,
    MULTIPASS_MODELS,
    TEXT_LLM_MODELS,
    WHISPER_MODELS,
    canonical_audio_llm_model_id,
    canonical_multipass_model_id,
    canonical_whisper_model_id,
)


def hf_cache_root() -> Path:
    return Path.home() / ".cache" / "huggingface" / "hub"


def model_cache_dir(repo_id: str) -> Path:
    safe = repo_id.replace("/", "--")
    return hf_cache_root() / f"models--{safe}"


def is_model_cached(repo_id: str) -> bool:
    path = model_cache_dir(repo_id)
    return path.exists() and any(path.iterdir())


def model_catalog() -> list[dict]:
    """Flat catalogue used by the SwiftUI Models tab.

    Each row carries the metadata the new role-grouped UI needs:
    ``role`` (transcription / multipass / text_llm / audio_llm /
    diarisation / embedding), ``family`` (Whisper / Mistral / Qwen /
    Pyannote), ``size_mb`` (approximate on-disk weight after the
    snapshot finishes), ``tier`` (light / balanced / heavy), and a
    ``language`` list so a French-only user can spot the
    French-tuned forks at a glance.

    The same model id can legitimately appear in two roles
    (e.g. ``whisper-large-v3-mlx`` is both a top-tier transcription
    pick *and* the default multipass model). The Models tab uses
    ``(id, role)`` as the row identity so each role's "Activer"
    button targets the right slot.
    """
    rows: list[dict] = []
    for entry in WHISPER_MODELS:
        rows.append(_build_row(entry, canonical_whisper_model_id))
    for entry in MULTIPASS_MODELS:
        rows.append(_build_row(entry, canonical_multipass_model_id))
    for entry in TEXT_LLM_MODELS:
        rows.append(_build_row(entry, lambda x: x))
    for entry in AUDIO_LLM_MODELS:
        rows.append(_build_row(entry, canonical_audio_llm_model_id))
    for entry in DIARISATION_MODELS:
        rows.append(_build_row(entry, lambda x: x))
    return rows


def _build_row(entry: dict, canonicalise) -> dict:
    repo_id = canonicalise(str(entry["id"]))
    cached = is_model_cached(repo_id)
    return {
        "id": repo_id,
        "label": entry.get("label", repo_id),
        "family": entry.get("family", ""),
        "role": entry.get("role", "transcription"),
        "size_mb": int(entry.get("size_mb") or 0),
        "tier": entry.get("tier", "balanced"),
        "language": list(entry.get("language") or ["multi"]),
        "default": bool(entry.get("default") or False),
        "gated": bool(entry.get("gated") or False),
        "cached": cached,
        "cache_dir": str(model_cache_dir(repo_id)),
    }


def delete_model(repo_id: str) -> Path:
    path = model_cache_dir(repo_id)
    if path.exists():
        shutil.rmtree(path)
    return path


def download_model(repo_id: str, token: str = "") -> Path:
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise RuntimeError(
            "huggingface_hub is not installed in the current Python environment"
        ) from exc
    snapshot_download(repo_id=repo_id, token=token or None)
    return model_cache_dir(repo_id)
