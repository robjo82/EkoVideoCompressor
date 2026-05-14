from __future__ import annotations

import shutil
from pathlib import Path

from transcription_utils import (
    AUDIO_LLM_MODELS,
    TEXT_LLM_MODELS,
    WHISPER_MODELS,
    canonical_audio_llm_model_id,
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
    rows: list[dict] = []
    for family, entries in (
        ("whisper", WHISPER_MODELS),
        ("text_llm", TEXT_LLM_MODELS),
        ("audio_llm", AUDIO_LLM_MODELS),
    ):
        for entry in entries:
            repo_id = str(entry["id"])
            if family == "whisper":
                repo_id = canonical_whisper_model_id(repo_id)
            if family == "audio_llm":
                repo_id = canonical_audio_llm_model_id(repo_id)
            rows.append(
                {
                    "family": family,
                    "id": repo_id,
                    "label": entry.get("label", repo_id),
                    "cached": is_model_cached(repo_id),
                    "cache_dir": str(model_cache_dir(repo_id)),
                }
            )
    return rows


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
