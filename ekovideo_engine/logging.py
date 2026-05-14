from __future__ import annotations

from datetime import datetime
from pathlib import Path

from .paths import app_log_path


def append_app_log(message: str) -> None:
    try:
        path = app_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().isoformat(timespec="seconds")
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"[{timestamp}] {message}\n")
    except Exception:
        pass


def tail_text(text: str | None, limit: int = 4000) -> str:
    value = (text or "").strip()
    if len(value) <= limit:
        return value
    return f"...{value[-limit:]}"


def export_logs_archive(destination: Path) -> Path:
    import json
    import platform
    import zipfile

    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        manifest = {
            "app": "EkoVideo Compressor",
            "platform": platform.platform(),
        }
        archive.writestr("manifest.json", json.dumps(manifest, indent=2))
        log_path = app_log_path()
        if log_path.exists():
            archive.write(log_path, "app.log")
    append_app_log(f"logs_exported path={str(destination)!r}")
    return destination
