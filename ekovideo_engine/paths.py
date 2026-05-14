from __future__ import annotations

import os
import sys
from pathlib import Path


APP_NAME = "EkoVideo Compressor"
BUNDLE_IDENTIFIER = "com.ekonum.ekovideocompressor"


def app_support_dir() -> Path:
    override = os.getenv("EKO_APP_SUPPORT_DIR", "").strip()
    if override:
        return Path(override)
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / APP_NAME
    return Path.home() / f".{APP_NAME.replace(' ', '').lower()}"


def app_log_path() -> Path:
    return app_support_dir() / "app.log"


def workspace_root() -> Path:
    return app_support_dir() / "Workspace"


def library_db_path() -> Path:
    return app_support_dir() / "library.db"


def managed_transcription_venv_dir() -> Path:
    return app_support_dir() / "mlx-whisper-venv"


def managed_venv_python_path() -> Path:
    if sys.platform.startswith("win"):
        return managed_transcription_venv_dir() / "Scripts" / "python.exe"
    return managed_transcription_venv_dir() / "bin" / "python"


def managed_mlx_whisper_path() -> Path:
    if sys.platform.startswith("win"):
        return managed_transcription_venv_dir() / "Scripts" / "mlx_whisper.exe"
    return managed_transcription_venv_dir() / "bin" / "mlx_whisper"
