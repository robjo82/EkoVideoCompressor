import json
import os
import re
import shlex
import shutil
import ssl
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import webbrowser
import zipfile
from dataclasses import dataclass, replace, asdict
from datetime import datetime
from pathlib import Path

import certifi
from ffmpeg_utils import (
    AUDIO_EXTENSIONS,
    MEDIA_EXTENSIONS,
    MEDIA_FILTER,
    VIDEO_EXTENSIONS,
    build_ffmpeg_cmd,
    default_out_path,
    is_audio_only_path,
)
from transcription_utils import (
    AUDIO_LLM_MODELS,
    DEFAULT_AUDIO_LLM_MODEL,
    DEFAULT_TEXT_LLM_MODEL,
    DEFAULT_WHISPER_MODEL,
    TEXT_LLM_MODELS,
    WHISPER_MODELS,
    assign_speakers_to_segments,
    audio_llm_label_for,
    build_audio_extract_cmd,
    build_diarization_cmd,
    build_llm_corrections_cmd,
    build_llm_title_cmd,
    build_multimodal_audio_cmd,
    build_mlx_whisper_cmd,
    canonical_audio_llm_model_id,
    canonical_whisper_model_id,
    default_transcript_path,
    parse_diarization_output,
    parse_llm_corrections_markdown,
    parse_llm_title_speakers,
    parse_whisper_json_segments,
    render_segments_plain,
    render_segments_with_speakers,
    structured_initial_prompt,
    suggest_transcript_stem,
    text_llm_label_for,
    transcript_output_ext,
)
from PySide6.QtCore import QObject, QSettings, QThread, QTime, QTimer, Qt, Signal
from PySide6.QtGui import QIcon, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListView,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QScrollArea,
    QSplitter,
    QSlider,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QTimeEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

APP_NAME = "EkoVideo Compressor"

try:
    from _build_version import APP_VERSION as BUILT_APP_VERSION
except Exception:
    BUILT_APP_VERSION = ""


def normalize_app_version(version_text: str | None) -> str:
    value = (version_text or "").strip()
    if not value:
        return "dev"
    if value.startswith("v"):
        value = value[1:]
    if value.startswith("."):
        value = value[1:]
    return value or "dev"


APP_VERSION = normalize_app_version(BUILT_APP_VERSION or os.getenv("APP_VERSION", "dev"))
ORG_NAME = "Ekonum"
ORG_DOMAIN = "ekonum"

# Audio/video extension catalogues + MEDIA_FILTER live in ffmpeg_utils so
# they can be unit-tested without pulling in the Qt stack.
APP_ICON_FILE = "ekovideo_icon.png"
APP_LOGO_FILE = "ekovideo_logo.png"
BUNDLE_IDENTIFIER = "com.ekonum.ekovideocompressor"
GITHUB_OWNER = "robjo82"
GITHUB_REPO = "EkoVideoCompressor"
GITHUB_LATEST_RELEASE_API = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/releases/latest"
MLX_WHISPER_PACKAGE = "mlx-whisper"
MANAGED_TRANSCRIPTION_ENV = "mlx-whisper-venv"
# pyannote 3.1 needs torch>=2.0; on Apple Silicon we use the standard wheels
# (MPS support is built in). torchaudio is required for audio I/O.
DIARIZATION_PIP_PACKAGES = ["torch", "torchaudio", "pyannote.audio>=3.1"]
# mlx-lm is small (~200 Mo) and powers the text LLM step (title, speakers,
# corrections, doubt flagging). We install it eagerly so the first
# transcription doesn't block on a pip install. mlx-vlm is heavier and only
# pulled in if the user opts into the multimodal re-listen step.
TEXT_LLM_PIP_PACKAGES = ["mlx-lm"]
MULTIMODAL_LLM_PIP_PACKAGES = ["mlx-vlm"]

PROFILE_PRESETS = {
    "Réunion rapide": {
        "resolution": "480p",
        "fps": 10,
        "crf": 32,
        "audio_bitrate": "96k",
        "x265_preset": "veryfast",
        "speech_enhance": False,
        "mono_audio": True,
    },
    "Réunion équilibrée": {
        "resolution": "720p",
        "fps": 12,
        "crf": 28,
        "audio_bitrate": "128k",
        "x265_preset": "medium",
        "speech_enhance": True,
        "mono_audio": False,
    },
    "Archive qualité": {
        "resolution": "1080p",
        "fps": 20,
        "crf": 24,
        "audio_bitrate": "160k",
        "x265_preset": "slow",
        "speech_enhance": True,
        "mono_audio": False,
    },
}

PROFILE_SUFFIX = {
    "Réunion rapide": "[Rapide]",
    "Réunion équilibrée": "[Équilibré]",
    "Archive qualité": "[Archive]",
    "Personnalisé": "[Custom]",
}

DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")
WHISPER_SEGMENT_RE = re.compile(r"\[(\d{1,2}):(\d{2}):(\d{2}(?:[.,]\d+)?)\s*-->")
SEMVER_RE = re.compile(r"^v?\.?(\d+)\.(\d+)\.(\d+)$")


from database_manager import DatabaseManager

@dataclass
class QueueJob:
    input_path: str
    db_id: int | None = None
    workspace_dir: str = ""
    output_path: str = ""
    custom_title: str = ""
    duration_ffmpeg: float = 0
    duration_whisper: float = 0
    duration_diarization: float = 0
    duration_total: float = 0
    profile_name: str = "Réunion équilibrée"
    resolution: str = "720p"
    fps: int = 12
    crf: int = 28
    audio_bitrate: str = "128k"
    x265_preset: str = "medium"
    speech_enhance: bool = True
    mono_audio: bool = False
    trim_enabled: bool = False
    trim_start: str = "00:00:00"
    trim_end: str = "00:00:00"
    transcript_path: str = ""
    status: str = "pending"
    error_message: str = ""


def resource_path(rel_name: str) -> str:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return str(Path(sys._MEIPASS) / rel_name)
    return str(Path(__file__).resolve().parent / rel_name)


def app_base_dir() -> Path:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent


def app_support_dir() -> Path:
    override = os.getenv("EKO_APP_SUPPORT_DIR", "").strip()
    if override:
        return Path(override)
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / APP_NAME
    return Path.home() / f".{APP_NAME.replace(' ', '').lower()}"


def app_log_path() -> Path:
    return app_support_dir() / "app.log"


def updater_log_path() -> Path:
    return app_support_dir() / "updater.log"


def _tail_for_log(text: str | None, limit: int = 4000) -> str:
    value = (text or "").strip()
    if len(value) <= limit:
        return value
    return f"...{value[-limit:]}"


def _command_for_log(cmd: list[str]) -> str:
    redacted: list[str] = []
    redact_next = False
    inline_python_next = False
    for arg in cmd:
        if redact_next:
            redacted.append("<redacted>")
            redact_next = False
            continue
        if inline_python_next:
            redacted.append("<inline-python>")
            inline_python_next = False
            continue

        redacted.append(arg)
        if arg in {"--initial-prompt"}:
            redact_next = True
        elif arg == "-c":
            inline_python_next = True

    return shlex.join(redacted)


def append_app_log(message: str):
    try:
        path = app_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().isoformat(timespec="seconds")
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"[{timestamp}] {message}\n")
    except Exception:
        pass


HF_API_BASE = "https://huggingface.co"
HF_TOKEN_URL = "https://huggingface.co/settings/tokens"
HF_GATED_MODEL_CHECKS: list[tuple[str, str, str]] = [
    ("pyannote/segmentation-3.0", "config.yaml", "Segmentation pyannote 3.0"),
    ("pyannote/speaker-diarization-3.1", "config.yaml", "Diarisation pyannote 3.1"),
    ("pyannote/speaker-diarization-community-1", "config.yaml", "Diarisation Community-1"),
]


def _open_url(url: str):
    try:
        webbrowser.open(url)
    except Exception:
        pass


def _hf_request(url: str, token: str = "", method: str = "GET") -> urllib.request.Request:
    headers = {"User-Agent": f"{APP_NAME}/{APP_VERSION}"}
    if token.strip():
        headers["Authorization"] = f"Bearer {token.strip()}"
    return urllib.request.Request(url, headers=headers, method=method)


def hf_whoami(token: str) -> dict:
    with urllib.request.urlopen(
        _hf_request(f"{HF_API_BASE}/api/whoami-v2", token),
        timeout=12,
        context=ssl.create_default_context(cafile=certifi.where()),
    ) as response:
        return json.loads(response.read().decode("utf-8"))


def hf_file_access_status(token: str, repo_id: str, filename: str) -> tuple[bool, str]:
    url = f"{HF_API_BASE}/{repo_id}/resolve/main/{filename}"
    try:
        with urllib.request.urlopen(
            _hf_request(url, token, method="HEAD"),
            timeout=12,
            context=ssl.create_default_context(cafile=certifi.where()),
        ) as response:
            return 200 <= response.status < 400, f"HTTP {response.status}"
    except urllib.error.HTTPError as exc:
        if exc.code in {401, 403}:
            return False, "conditions à accepter ou token sans accès"
        if exc.code == 404:
            return False, "fichier de contrôle introuvable"
        return False, f"HTTP {exc.code}"
    except Exception as exc:
        return False, str(exc)


def export_logs_archive(parent: QWidget | None = None) -> Path | None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_path = Path.home() / "Desktop" / f"ekovideo-logs-{timestamp}.zip"
    chosen, _ = QFileDialog.getSaveFileName(
        parent,
        "Exporter les logs",
        str(default_path),
        "Archive ZIP (*.zip)",
    )
    if not chosen:
        return None

    out_path = Path(chosen)
    if out_path.suffix.lower() != ".zip":
        out_path = out_path.with_suffix(".zip")

    support_dir = app_support_dir()
    summary = {
        "app": APP_NAME,
        "version": APP_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "support_dir": str(support_dir),
        "python": sys.version,
        "platform": sys.platform,
        "ffmpeg_in_path": bool(find_binary("ffmpeg")),
        "ffprobe_in_path": bool(find_binary("ffprobe")),
        "mlx_whisper_in_path": bool(find_binary("mlx_whisper")),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path, arcname in (
            (app_log_path(), "app.log"),
            (updater_log_path(), "updater.log"),
        ):
            if path.exists():
                archive.write(path, arcname)
        archive.writestr("diagnostic.json", json.dumps(summary, indent=2, ensure_ascii=False))

    append_app_log(f"logs_exported path={str(out_path)!r}")
    return out_path


def managed_transcription_venv_dir() -> Path:
    return app_support_dir() / MANAGED_TRANSCRIPTION_ENV


def managed_mlx_whisper_path() -> Path:
    if sys.platform.startswith("win"):
        return managed_transcription_venv_dir() / "Scripts" / "mlx_whisper.exe"
    return managed_transcription_venv_dir() / "bin" / "mlx_whisper"


def managed_venv_python_path() -> Path:
    """Path to the Python interpreter inside the managed transcription venv."""
    if sys.platform.startswith("win"):
        return managed_transcription_venv_dir() / "Scripts" / "python.exe"
    return managed_transcription_venv_dir() / "bin" / "python"


def candidate_bin_paths() -> list[Path]:
    base = app_base_dir()
    return [
        managed_mlx_whisper_path(),
        base / "ffmpeg",
        base / "ffprobe",
        base / "mlx_whisper",
        base / "whisper-cli",
        base / "bin" / "ffmpeg",
        base / "bin" / "ffprobe",
        base / "bin" / "mlx_whisper",
        base / "bin" / "whisper-cli",
        base / "Resources" / "ffmpeg",
        base / "Resources" / "ffprobe",
        base / "Resources" / "mlx_whisper",
        base / "Resources" / "whisper-cli",
        base / "Resources" / "bin" / "ffmpeg",
        base / "Resources" / "bin" / "ffprobe",
        base / "Resources" / "bin" / "mlx_whisper",
        base / "Resources" / "bin" / "whisper-cli",
        Path("/opt/homebrew/bin/ffmpeg"),
        Path("/opt/homebrew/bin/ffprobe"),
        Path("/opt/homebrew/bin/mlx_whisper"),
        Path("/opt/homebrew/bin/whisper-cli"),
        Path("/usr/local/bin/ffmpeg"),
        Path("/usr/local/bin/ffprobe"),
        Path("/usr/local/bin/mlx_whisper"),
        Path("/usr/local/bin/whisper-cli"),
    ]


def is_executable(p: Path) -> bool:
    return p.exists() and os.access(str(p), os.X_OK) and p.is_file()


def find_binary(name: str) -> str | None:
    in_path = shutil.which(name)
    if in_path:
        return in_path

    for c in candidate_bin_paths():
        if c.name == name and is_executable(c):
            return str(c)

    if sys.platform.startswith("win"):
        in_path = shutil.which(name + ".exe")
        if in_path:
            return in_path

    return None


def find_compatible_python() -> str | None:
    candidates = [
        os.getenv("EKO_TRANSCRIPTION_PYTHON", "").strip(),
        "/opt/homebrew/bin/python3.12",
        "/opt/homebrew/bin/python3.13",
        "/opt/homebrew/bin/python3.11",
        "/usr/local/bin/python3.12",
        "/usr/local/bin/python3.13",
        "/usr/local/bin/python3.11",
        "python3.12",
        "python3.13",
        "python3.11",
        "python3",
    ]
    seen = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        path = shutil.which(candidate) if not Path(candidate).is_absolute() else candidate
        if not path or not Path(path).exists():
            continue
        try:
            proc = subprocess.run(
                [
                    path,
                    "-c",
                    "import sys; raise SystemExit(0 if (3, 9) <= sys.version_info[:2] < (3, 14) else 1)",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if proc.returncode == 0:
                return str(path)
        except Exception:
            continue
    return None


def parse_duration_hms(text: str) -> float | None:
    m = DURATION_RE.search(text)
    if not m:
        return None
    hh = int(m.group(1))
    mm = int(m.group(2))
    ss = float(m.group(3))
    return hh * 3600 + mm * 60 + ss


def parse_whisper_segment_seconds(line: str) -> float | None:
    m = WHISPER_SEGMENT_RE.search(line)
    if not m:
        return None
    hh = int(m.group(1))
    mm = int(m.group(2))
    ss = float(m.group(3).replace(",", "."))
    return hh * 3600 + mm * 60 + ss


def probe_duration_seconds(in_path: str, ffprobe_path: str | None, ffmpeg_path: str) -> float | None:
    if ffprobe_path and Path(ffprobe_path).exists():
        try:
            probe_cmd = [
                ffprobe_path,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                in_path,
            ]
            p = subprocess.run(probe_cmd, capture_output=True, text=True)
            if p.returncode == 0 and p.stdout.strip():
                duration = float(p.stdout.strip())
                if duration > 0:
                    return duration
        except Exception:
            pass

    try:
        ffmpeg_cmd = [ffmpeg_path, "-hide_banner", "-i", in_path]
        p = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        combined = f"{p.stderr or ''}\n{p.stdout or ''}"
        duration = parse_duration_hms(combined)
        if duration and duration > 0:
            return duration
    except Exception:
        pass

    return None


def _parse_fraction(value: str) -> float:
    if not value:
        return 0.0
    if "/" in value:
        a, b = value.split("/", 1)
        try:
            af = float(a)
            bf = float(b)
            return af / bf if bf else 0.0
        except Exception:
            return 0.0
    try:
        return float(value)
    except Exception:
        return 0.0


def probe_video_metadata(in_path: str, ffprobe_path: str | None) -> dict:
    fallback = {
        "duration": None,
        "width": None,
        "height": None,
        "fps": None,
        "size_bytes": Path(in_path).stat().st_size if Path(in_path).exists() else None,
    }
    if not ffprobe_path or not Path(ffprobe_path).exists() or not Path(in_path).exists():
        return fallback

    probe_cmd = [
        ffprobe_path,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        in_path,
    ]
    try:
        p = subprocess.run(probe_cmd, capture_output=True, text=True)
        if p.returncode != 0 or not p.stdout.strip():
            return fallback

        data = json.loads(p.stdout)
        fmt = data.get("format", {})
        streams = data.get("streams", [])
        vstream = next((s for s in streams if s.get("codec_type") == "video"), {})

        duration_raw = fmt.get("duration")
        size_raw = fmt.get("size")
        return {
            "duration": float(duration_raw) if duration_raw not in (None, "") else None,
            "width": vstream.get("width"),
            "height": vstream.get("height"),
            "fps": _parse_fraction(vstream.get("r_frame_rate", "")),
            "size_bytes": int(size_raw) if size_raw not in (None, "") else fallback["size_bytes"],
        }
    except Exception:
        return fallback


def format_seconds(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    total = int(seconds)
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"


def format_compact_seconds(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    total = max(0, int(seconds))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    if hours:
        return f"{hours}h{minutes:02d}"
    if minutes:
        return f"{minutes}m{secs:02d}"
    return f"{secs}s"


def format_size(size_bytes: int | None) -> str:
    if not size_bytes:
        return "-"
    val = float(size_bytes)
    units = ["B", "KB", "MB", "GB", "TB"]
    idx = 0
    while val >= 1024 and idx < len(units) - 1:
        val /= 1024
        idx += 1
    return f"{val:.1f} {units[idx]}"


def infer_profile_name(job: QueueJob) -> str:
    for preset_name, preset in PROFILE_PRESETS.items():
        if (
            job.resolution == preset["resolution"]
            and job.fps == preset["fps"]
            and job.crf == preset["crf"]
            and job.audio_bitrate == preset["audio_bitrate"]
            and job.x265_preset == preset["x265_preset"]
            and job.speech_enhance == preset["speech_enhance"]
            and job.mono_audio == preset["mono_audio"]
        ):
            return preset_name
    return "Personnalisé"


def apply_profile_to_job(job: QueueJob, profile_name: str) -> QueueJob:
    preset = PROFILE_PRESETS.get(profile_name)
    if not preset:
        return replace(job, profile_name="Personnalisé")
    return replace(
        job,
        profile_name=profile_name,
        resolution=preset["resolution"],
        fps=preset["fps"],
        crf=preset["crf"],
        audio_bitrate=preset["audio_bitrate"],
        x265_preset=preset["x265_preset"],
        speech_enhance=preset["speech_enhance"],
        mono_audio=preset["mono_audio"],
    )


def parse_semver(version_text: str) -> tuple[int, int, int] | None:
    match = SEMVER_RE.match((version_text or "").strip())
    if not match:
        return None
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def choose_release_asset(assets: list[dict]) -> dict | None:
    zip_assets = [a for a in assets if str(a.get("name", "")).endswith(".zip")]
    if not zip_assets:
        return None

    arm_assets = [a for a in zip_assets if "macos-arm64" in str(a.get("name", "")).lower()]
    return arm_assets[0] if arm_assets else zip_assets[0]


def is_hf_model_cached(repo_id: str) -> bool:
    """
    True if a Hugging Face repo has at least one snapshot under the local
    cache. We use this in Settings to show the user which models are already
    downloaded vs need a fresh pull (multi-Go).
    """
    repo_dir = hf_model_cache_dir(repo_id)
    if not repo_dir.exists():
        return False
    snapshots_dir = repo_dir / "snapshots"
    if snapshots_dir.exists() and any(snapshots_dir.iterdir()):
        return True
    return False


def hf_model_cache_dir(repo_id: str) -> Path:
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    return cache_dir / f"models--{repo_id.replace('/', '--')}"


class ModelCacheWorker(QThread):
    progress = Signal(int, str)
    finished_ok = Signal(str, str)
    failed = Signal(str)

    def __init__(self, mode: str, repo_id: str, token: str = "", parent: QObject | None = None):
        super().__init__(parent)
        self.mode = mode
        self.repo_id = repo_id
        self.token = token

    def run(self):
        try:
            if self.mode == "delete":
                cache_dir = hf_model_cache_dir(self.repo_id)
                if cache_dir.exists():
                    shutil.rmtree(cache_dir)
                self.finished_ok.emit(self.mode, self.repo_id)
                return

            python_path = managed_venv_python_path()
            python = str(python_path) if python_path.exists() else sys.executable
            script = r'''
import json
import sys

try:
    from huggingface_hub import snapshot_download
    from tqdm.auto import tqdm

    repo_id = sys.argv[1]
    token = sys.argv[2] or None

    class JsonTqdm(tqdm):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._last_pct = -1
            self._emit_progress(force=True)

        def update(self, n=1):
            result = super().update(n)
            self._emit_progress()
            return result

        def close(self):
            self._emit_progress(force=True)
            return super().close()

        def _emit_progress(self, force=False):
            total = self.total or 0
            current = self.n or 0
            if total > 0:
                pct = max(0, min(100, int((current / total) * 100)))
            else:
                pct = 0
            if force or pct != self._last_pct:
                self._last_pct = pct
                print(
                    json.dumps(
                        {
                            "event": "progress",
                            "pct": pct,
                            "current": current,
                            "total": total,
                            "label": str(self.desc or ""),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    print(json.dumps({"event": "progress", "pct": 0, "label": "Préparation…"}, ensure_ascii=False), flush=True)
    snapshot_download(repo_id=repo_id, token=token, tqdm_class=JsonTqdm)
    print(json.dumps({"event": "progress", "pct": 100, "label": "Terminé"}, ensure_ascii=False), flush=True)
except Exception as exc:
    print(json.dumps({"event": "error", "message": str(exc)}, ensure_ascii=False), flush=True)
    sys.exit(2)
'''
            proc = subprocess.Popen(
                [python, "-u", "-c", script, self.repo_id, self.token],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            output_lines: list[str] = []
            if proc.stdout:
                for line in proc.stdout:
                    output_lines.append(line)
                    stripped = line.strip()
                    if not stripped.startswith("{"):
                        continue
                    try:
                        event = json.loads(stripped)
                    except Exception:
                        continue
                    if event.get("event") == "progress":
                        self.progress.emit(
                            int(event.get("pct") or 0),
                            str(event.get("label") or ""),
                        )
                    elif event.get("event") == "error":
                        output_lines.append(str(event.get("message") or ""))
            proc.wait()
            if proc.returncode != 0:
                detail = self._clean_download_error(output_lines)
                raise RuntimeError(detail or f"Téléchargement impossible pour {self.repo_id}.")
            self.finished_ok.emit(self.mode, self.repo_id)
        except Exception as exc:
            self.failed.emit(str(exc))

    @staticmethod
    def _clean_download_error(lines: list[str]) -> str:
        messages: list[str] = []
        raw_lines: list[str] = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            raw_lines.append(stripped)
            if stripped.startswith("{"):
                try:
                    event = json.loads(stripped)
                except Exception:
                    continue
                if event.get("event") == "error" and event.get("message"):
                    messages.append(str(event["message"]))
        if messages:
            return "\n".join(dict.fromkeys(messages))
        return "\n".join(line for line in raw_lines if not line.startswith("{"))


class HuggingFaceAuthDialog(QDialog):
    def __init__(self, token: str = "", parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Connexion Hugging Face")
        self.setModal(True)

        root = QVBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(12)

        intro = QLabel(
            "Connectez un compte Hugging Face pour télécharger les modèles gated "
            "utilisés par la détection des locuteurs. Hugging Face impose que "
            "l'utilisateur accepte les conditions dans le navigateur ; l'app vérifie "
            "ensuite automatiquement l'accès."
        )
        intro.setWordWrap(True)
        root.addWidget(intro)

        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.token_edit = QLineEdit(token or "")
        self.token_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.token_edit.setPlaceholderText("Token Hugging Face avec droit Read")
        form.addRow("Token", self.token_edit)
        root.addLayout(form)

        actions = QHBoxLayout()
        self.btn_token = QPushButton("Créer / gérer le token")
        self.btn_token.clicked.connect(lambda: _open_url(HF_TOKEN_URL))
        self.btn_terms = QPushButton("Ouvrir les conditions")
        self.btn_terms.clicked.connect(self.open_terms_pages)
        self.btn_verify = QPushButton("Vérifier l'accès")
        self.btn_verify.setObjectName("primaryButton")
        self.btn_verify.clicked.connect(self.verify_access)
        actions.addWidget(self.btn_token)
        actions.addWidget(self.btn_terms)
        actions.addWidget(self.btn_verify)
        root.addLayout(actions)

        self.status = QTextEdit()
        self.status.setReadOnly(True)
        self.status.setAcceptRichText(False)
        self.status.setMinimumHeight(170)
        self.status.setPlainText(
            "1. Ouvrez les conditions et acceptez les modèles nécessaires.\n"
            "2. Créez un token Hugging Face avec le droit Read.\n"
            "3. Collez-le ici puis vérifiez l'accès."
        )
        root.addWidget(self.status)

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        self.setMinimumWidth(680)
        self.resize(740, 430)

    def token(self) -> str:
        return self.token_edit.text().strip()

    def open_terms_pages(self):
        _open_url(f"{HF_API_BASE}/login")
        for repo_id, _filename, _label in HF_GATED_MODEL_CHECKS:
            _open_url(f"{HF_API_BASE}/{repo_id}")

    def verify_access(self):
        token = self.token()
        if not token:
            self.status.setPlainText("Collez d'abord un token Hugging Face.")
            return

        lines: list[str] = []
        try:
            user = hf_whoami(token)
            username = (
                user.get("name")
                or user.get("preferred_username")
                or user.get("sub")
                or "compte HF"
            )
            lines.append(f"Compte connecté: {username}")
        except urllib.error.HTTPError as exc:
            self.status.setPlainText(
                f"Token refusé par Hugging Face (HTTP {exc.code}). Vérifiez le token et son droit Read."
            )
            return
        except Exception as exc:
            self.status.setPlainText(f"Vérification du compte impossible: {exc}")
            return

        all_ok = True
        lines.append("")
        for repo_id, filename, label in HF_GATED_MODEL_CHECKS:
            ok, detail = hf_file_access_status(token, repo_id, filename)
            all_ok = all_ok and ok
            marker = "OK" if ok else "À accepter"
            lines.append(f"{marker} · {label} ({repo_id}) — {detail}")

        if all_ok:
            lines.append("\nTous les accès nécessaires à la détection des locuteurs sont validés.")
        else:
            lines.append(
                "\nOuvrez les conditions, acceptez les modèles manquants dans le navigateur, "
                "puis relancez cette vérification."
            )
        self.status.setPlainText("\n".join(lines))


class SettingsDialog(QDialog):
    def __init__(
        self,
        parent: QWidget,
        ffmpeg_path: str,
        ffprobe_path: str,
        github_token: str,
        transcription_settings: dict[str, str | bool],
    ):
        super().__init__(parent)
        self.setWindowTitle("Paramètres")
        self.setModal(True)
        self._model_cache_worker: ModelCacheWorker | None = None
        self._model_controls: list[tuple[QComboBox, QPushButton, QPushButton]] = []
        self._model_busy_repo_id = ""

        root = QVBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(14)

        tabs = QTabWidget()
        tabs.setObjectName("settingsTabs")
        root.addWidget(tabs)

        tools_tab = QWidget()
        tools_form = QFormLayout(tools_tab)
        tools_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        tools_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        tools_form.setHorizontalSpacing(12)
        tools_form.setVerticalSpacing(12)

        self.ffmpeg_edit = QLineEdit(ffmpeg_path or "")
        self.ffmpeg_edit.setPlaceholderText("Chemin vers ffmpeg")
        btn_ffmpeg = QPushButton("Parcourir…")
        btn_ffmpeg.clicked.connect(self.pick_ffmpeg)
        row1 = QHBoxLayout()
        row1.addWidget(self.ffmpeg_edit)
        row1.addWidget(btn_ffmpeg)

        self.ffprobe_edit = QLineEdit(ffprobe_path or "")
        self.ffprobe_edit.setPlaceholderText("Chemin vers ffprobe")
        btn_ffprobe = QPushButton("Parcourir…")
        btn_ffprobe.clicked.connect(self.pick_ffprobe)
        row2 = QHBoxLayout()
        row2.addWidget(self.ffprobe_edit)
        row2.addWidget(btn_ffprobe)

        tools_form.addRow("ffmpeg", row1)
        tools_form.addRow("ffprobe", row2)

        self.token_edit = QLineEdit(github_token or "")
        self.token_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.token_edit.setPlaceholderText("GitHub token (optionnel, requis si repo privé)")
        tools_form.addRow("Token update", self.token_edit)

        hint = QLabel(
            "Si ffmpeg est bloqué par macOS, ouvrez Terminal puis:\n"
            "xattr -dr com.apple.quarantine /Applications/EkoVideoCompressor.app"
        )
        hint.setWordWrap(True)
        tools_form.addRow("", hint)

        btn_export_logs = QPushButton("Exporter les logs…")
        btn_export_logs.setObjectName("secondaryButton")
        btn_export_logs.clicked.connect(self.export_logs)
        tools_form.addRow("Diagnostic", btn_export_logs)

        transcription_tab = QWidget()
        transcription_form = QFormLayout(transcription_tab)
        transcription_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        transcription_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        transcription_form.setHorizontalSpacing(12)
        transcription_form.setVerticalSpacing(12)

        self.mlx_whisper_edit = QLineEdit(str(transcription_settings.get("mlx_whisper_path", "")))
        self.mlx_whisper_edit.setPlaceholderText("Chemin vers mlx_whisper")
        btn_mlx_whisper = QPushButton("Parcourir…")
        btn_mlx_whisper.clicked.connect(self.pick_mlx_whisper)
        mlx_row = QHBoxLayout()
        mlx_row.addWidget(self.mlx_whisper_edit)
        mlx_row.addWidget(btn_mlx_whisper)
        transcription_form.addRow("Commande", mlx_row)

        current_whisper = canonical_whisper_model_id(
            str(transcription_settings.get("model") or DEFAULT_WHISPER_MODEL)
        )
        self.transcription_model_combo = self._build_model_combo(WHISPER_MODELS, current_whisper)
        transcription_form.addRow(
            "Modèle Whisper",
            self._build_model_cache_row(self.transcription_model_combo),
        )

        # --- Text LLM (analyse, titre, locuteurs, corrections) ---------------
        # Run via mlx_lm. Default Mistral 7B is the sweet spot on M1 16 Go.
        # Heavier variants are listed for users with M4 Max / more RAM.
        current_text_llm = str(
            transcription_settings.get("text_llm_model")
            or transcription_settings.get("llm_model")
            or DEFAULT_TEXT_LLM_MODEL
        )
        self.transcription_text_llm_combo = self._build_llm_combo(TEXT_LLM_MODELS, current_text_llm)
        transcription_form.addRow(
            "Modèle IA texte",
            self._build_model_cache_row(self.transcription_text_llm_combo),
        )

        text_llm_hint = QLabel(
            "Utilisé après transcription pour proposer un titre, identifier les "
            "interlocuteurs, signaler les passages douteux et corriger les erreurs "
            "manifestes. Tourne 100% en local. Le modèle est téléchargé au premier usage."
        )
        text_llm_hint.setWordWrap(True)
        text_llm_hint.setStyleSheet("color: #6e6e73; font-size: 12px;")
        transcription_form.addRow("", text_llm_hint)

        # --- Multimodal audio recheck (re-écoute IA) -------------------------
        self.transcription_audio_recheck_check = QCheckBox(
            "Réécoute IA des passages douteux (multimodal, expérimental)"
        )
        self.transcription_audio_recheck_check.setChecked(
            bool(transcription_settings.get("audio_recheck_enabled", False))
        )
        transcription_form.addRow("", self.transcription_audio_recheck_check)

        current_audio_llm = canonical_audio_llm_model_id(
            str(transcription_settings.get("audio_llm_model") or DEFAULT_AUDIO_LLM_MODEL)
        )
        self.transcription_audio_llm_combo = self._build_llm_combo(AUDIO_LLM_MODELS, current_audio_llm)
        transcription_form.addRow(
            "Modèle IA audio",
            self._build_model_cache_row(self.transcription_audio_llm_combo),
        )

        # The checkbox controls whether the audio model is used during a
        # transcription. The selector stays active so users can pre-download
        # or remove the model without enabling the recheck step.
        self.transcription_audio_recheck_check.toggled.connect(
            lambda _=False: self._refresh_model_cache_buttons()
        )

        audio_llm_hint = QLabel(
            "Optionnel. Quand activé, l'IA réécoute des extraits courts (5-15 s) autour "
            "des passages signalés douteux pour proposer une transcription plus précise. "
            "Téléchargement supplémentaire ~4-7 Go au premier usage."
        )
        audio_llm_hint.setWordWrap(True)
        audio_llm_hint.setStyleSheet("color: #6e6e73; font-size: 12px;")
        transcription_form.addRow("", audio_llm_hint)

        self.model_cache_progress = QProgressBar()
        self.model_cache_progress.setObjectName("modelCacheProgress")
        self.model_cache_progress.setRange(0, 100)
        self.model_cache_progress.setValue(0)
        self.model_cache_progress.setTextVisible(True)
        self.model_cache_progress.setFormat("Prêt")
        self.model_cache_progress.setVisible(False)
        transcription_form.addRow("", self.model_cache_progress)

        self.transcription_language_combo = QComboBox()
        self.transcription_language_combo.addItems(["fr", "auto", "en", "es", "de", "it"])
        self.transcription_language_combo.setCurrentText(str(transcription_settings.get("language", "fr")))
        transcription_form.addRow("Langue", self.transcription_language_combo)

        self.transcription_format_combo = QComboBox()
        self.transcription_format_combo.addItems(["txt", "srt", "vtt", "json", "all"])
        self.transcription_format_combo.setCurrentText(str(transcription_settings.get("format", "txt")))
        transcription_form.addRow("Format", self.transcription_format_combo)

        self.transcription_suffix_value = str(transcription_settings.get("suffix", "")).strip()
        if self.transcription_suffix_value == "_transcription":
            self.transcription_suffix_value = ""

        self.transcription_enhance_check = QCheckBox("Nettoyer la voix avant transcription")
        self.transcription_enhance_check.setChecked(bool(transcription_settings.get("enhance_audio", True)))
        transcription_form.addRow("", self.transcription_enhance_check)

        self.transcription_diarization_check = QCheckBox(
            "Détection des locuteurs (pyannote)"
        )
        self.transcription_diarization_check.setChecked(
            bool(transcription_settings.get("diarization_enabled", False))
        )
        transcription_form.addRow("", self.transcription_diarization_check)

        self.transcription_hf_token_edit = QLineEdit(str(transcription_settings.get("hf_token", "")))
        self.transcription_hf_token_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.transcription_hf_token_edit.setPlaceholderText(
            "hf_xxxxxxxx (Hugging Face, requis pour la détection des locuteurs)"
        )
        hf_token_row = QHBoxLayout()
        hf_token_row.addWidget(self.transcription_hf_token_edit, 1)
        btn_hf_auth = QPushButton("Connecter…")
        btn_hf_auth.setObjectName("secondaryButton")
        btn_hf_auth.clicked.connect(self.open_huggingface_auth)
        hf_token_row.addWidget(btn_hf_auth, 0)
        transcription_form.addRow("Hugging Face", hf_token_row)

        hf_hint = QLabel(
            "Connexion requise pour la détection des locuteurs. Hugging Face impose "
            "l'acceptation des conditions dans le navigateur ; l'app vérifie ensuite "
            "que le token Read peut accéder aux modèles :\n"
            "• huggingface.co/pyannote/segmentation-3.0\n"
            "• huggingface.co/pyannote/speaker-diarization-3.1\n"
            "• huggingface.co/pyannote/speaker-diarization-community-1"
        )
        hf_hint.setWordWrap(True)
        hf_hint.setStyleSheet("color: #6e6e73; font-size: 12px;")
        transcription_form.addRow("", hf_hint)

        tabs.addTab(tools_tab, "Outils")
        tabs.addTab(transcription_tab, "Transcription")

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)
        self._refresh_model_cache_buttons()

        self.setMinimumWidth(720)
        self.resize(820, 600)
        chevron_url = Path(resource_path("assets/chevron-down.svg")).as_posix()
        check_url = Path(resource_path("assets/check.svg")).as_posix()
        self.setStyleSheet(
            ("""
        QDialog {
            background: #f5f5f7;
            color: #1d1d1f;
            font-family: ".AppleSystemUIFont", "Helvetica Neue", "Arial", sans-serif;
            font-size: 14px;
        }
        QLabel { background: transparent; color: #1d1d1f; }
        QLineEdit {
            background: #ffffff;
            border: 1px solid #d2d2d7;
            border-radius: 9px;
            padding: 8px 10px;
            min-height: 24px;
        }
        QLineEdit:focus { border-color: #007aff; }
        /* QComboBox needs its own block: padded right side so the
           displayed text never overlaps the dropdown chevron. */
        QComboBox {
            background: #ffffff;
            border: 1px solid #d2d2d7;
            border-radius: 9px;
            padding: 7px 30px 7px 10px;
            min-height: 24px;
        }
        QComboBox:hover { border-color: #b8b8c0; }
        QComboBox:focus { border-color: #007aff; }
        QComboBox::drop-down {
            subcontrol-origin: padding;
            subcontrol-position: top right;
            width: 26px;
            border-left: 1px solid #ececef;
            border-top-right-radius: 9px;
            border-bottom-right-radius: 9px;
            background: #fbfbfd;
        }
        QComboBox::drop-down:hover { background: #f2f2f7; }
        QComboBox::down-arrow {
            image: url(__CHEVRON_URL__);
            width: 12px;
            height: 12px;
        }
        QComboBox QAbstractItemView {
            background: #ffffff;
            alternate-background-color: #ffffff;
            border: 1px solid #d2d2d7;
            border-radius: 8px;
            padding: 4px;
            color: #1d1d1f;
            selection-background-color: #e8f2ff;
            selection-color: #1d1d1f;
            outline: none;
        }
        QComboBox QAbstractItemView::item {
            background: #ffffff;
            padding: 6px 10px;
            border-radius: 5px;
            min-height: 22px;
        }
        QComboBox QAbstractItemView::item:hover,
        QComboBox QAbstractItemView::item:selected {
            background: #e8f2ff;
            color: #1d1d1f;
        }
        QCheckBox { color: #1d1d1f; spacing: 8px; }
        QCheckBox::indicator {
            width: 16px; height: 16px;
            border-radius: 4px;
            border: 1px solid #c7c7cc;
            background: #ffffff;
        }
        QCheckBox::indicator:checked {
            background: #007aff;
            border-color: #007aff;
            image: url(__CHECK_URL__);
        }
        QPushButton {
            background: #ffffff;
            border: 1px solid #d2d2d7;
            border-radius: 10px;
            padding: 8px 14px;
            font-weight: 600;
            color: #1d1d1f;
        }
        QPushButton:hover { background: #f2f2f7; }
        QPushButton#modelCacheButton {
            padding: 8px 12px;
            min-width: 92px;
        }
        QPushButton#modelDeleteButton {
            padding: 8px 12px;
            min-width: 82px;
        }
        QProgressBar#modelCacheProgress {
            background: #ffffff;
            border: 1px solid #d2d2d7;
            border-radius: 8px;
            min-height: 18px;
            text-align: center;
            color: #1d1d1f;
            font-size: 12px;
        }
        QProgressBar#modelCacheProgress::chunk {
            background: #007aff;
            border-radius: 7px;
        }
        QTabWidget::pane {
            border: none;
            margin-top: 10px;
        }
        QTabBar::tab {
            background: #f2f2f7;
            border: 1px solid #d2d2d7;
            border-radius: 9px;
            color: #3a3a3c;
            padding: 8px 18px;
            margin-right: 6px;
            font-weight: 600;
        }
        QTabBar::tab:selected {
            background: #007aff;
            color: #ffffff;
            border-color: #007aff;
        }
        """)
            .replace("__CHEVRON_URL__", chevron_url)
            .replace("__CHECK_URL__", check_url)
        )

    def _build_model_combo(self, catalog: list[dict], current_id: str) -> QComboBox:
        combo = QComboBox()
        combo.setEditable(False)
        combo.setView(self._model_combo_view(combo))
        combo._model_catalog = catalog  # type: ignore[attr-defined]
        combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        combo.setMinimumContentsLength(34)
        combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        for entry in catalog:
            combo.addItem(self._model_combo_label(entry), entry["id"])

        idx = combo.findData(current_id)
        if idx >= 0:
            combo.setCurrentIndex(idx)
        elif current_id:
            combo.addItem(f"{current_id} — modèle actuel · cache inconnu", current_id)
            combo.setCurrentIndex(combo.count() - 1)
        return combo

    def _build_llm_combo(self, catalog: list[dict], current_id: str) -> QComboBox:
        """
        Build a QComboBox from a model catalog (TEXT_LLM_MODELS / AUDIO_LLM_MODELS).
        Each entry is shown with its human label + cache status. The
        underlying model id is stored as Qt user-data so we read it back via
        currentData() instead of parsing the visible text.
        """
        combo = QComboBox()
        combo.setEditable(False)
        combo.setView(self._model_combo_view(combo))
        combo._model_catalog = catalog  # type: ignore[attr-defined]
        combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        combo.setMinimumContentsLength(30)
        combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        for entry in catalog:
            combo.addItem(self._model_combo_label(entry), entry["id"])

        # Pre-select the current model (or add it if it's a custom override).
        idx = combo.findData(current_id)
        if idx >= 0:
            combo.setCurrentIndex(idx)
        elif current_id:
            combo.addItem(f"{current_id} — modèle actuel · cache inconnu", current_id)
            combo.setCurrentIndex(combo.count() - 1)
        return combo

    def _model_combo_label(self, entry: dict) -> str:
        status = "téléchargé" if is_hf_model_cached(str(entry["id"])) else "à télécharger"
        return f"{entry['label']} — {status}"

    def _model_combo_view(self, combo: QComboBox) -> QListView:
        view = QListView(combo)
        view.setAlternatingRowColors(False)
        view.setUniformItemSizes(True)
        view.setStyleSheet(
            """
            QListView {
                background: #ffffff;
                alternate-background-color: #ffffff;
                color: #1d1d1f;
                selection-background-color: #e8f2ff;
                selection-color: #1d1d1f;
                outline: 0;
            }
            QListView::item {
                background: #ffffff;
                color: #1d1d1f;
                padding: 6px 10px;
                min-height: 22px;
            }
            QListView::item:hover,
            QListView::item:selected,
            QListView::item:selected:active,
            QListView::item:selected:!active {
                background: #e8f2ff;
                color: #1d1d1f;
            }
            """
        )
        return view

    def _build_model_cache_row(self, combo: QComboBox) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(8)
        row.addWidget(combo, 1)

        btn_download = QPushButton("Télécharger")
        btn_download.setObjectName("modelCacheButton")
        btn_download.clicked.connect(lambda _=False, c=combo: self.download_selected_model(c))
        row.addWidget(btn_download, 0)

        btn_delete = QPushButton("Supprimer")
        btn_delete.setObjectName("modelDeleteButton")
        btn_delete.clicked.connect(lambda _=False, c=combo: self.delete_selected_model(c))
        row.addWidget(btn_delete, 0)

        combo.currentIndexChanged.connect(lambda _=0: self._refresh_model_cache_buttons())
        self._model_controls.append((combo, btn_download, btn_delete))
        return row

    def _refresh_model_combo_labels(self, combo: QComboBox):
        current_id = str(combo.currentData() or "")
        catalog = getattr(combo, "_model_catalog", [])
        combo.blockSignals(True)
        combo.clear()
        for entry in catalog:
            combo.addItem(self._model_combo_label(entry), entry["id"])
        idx = combo.findData(current_id)
        if idx >= 0:
            combo.setCurrentIndex(idx)
        elif current_id:
            combo.addItem(f"{current_id} — modèle actuel · cache inconnu", current_id)
            combo.setCurrentIndex(combo.count() - 1)
        combo.blockSignals(False)

    def _refresh_model_cache_buttons(self):
        busy = self._model_cache_worker is not None and self._model_cache_worker.isRunning()
        for combo, btn_download, btn_delete in self._model_controls:
            repo_id = str(combo.currentData() or "")
            cached = bool(repo_id and is_hf_model_cached(repo_id))
            active = combo.isEnabled() and not busy and bool(repo_id)
            btn_download.setEnabled(active and not cached)
            btn_delete.setEnabled(active and cached)
            if busy and repo_id == self._model_busy_repo_id:
                btn_download.setEnabled(False)
                btn_delete.setEnabled(False)
                btn_download.setText("Téléchargement…" if self._model_cache_worker.mode == "download" else "En cours…")
            else:
                btn_download.setText("Télécharger")
                btn_delete.setText("Supprimer")

    def _selected_model_id(self, combo: QComboBox) -> str:
        return str(combo.currentData() or "").strip()

    def download_selected_model(self, combo: QComboBox):
        repo_id = self._selected_model_id(combo)
        if not repo_id:
            return
        if is_hf_model_cached(repo_id):
            self._refresh_model_cache_buttons()
            return
        self._run_model_cache_worker("download", repo_id)

    def delete_selected_model(self, combo: QComboBox):
        repo_id = self._selected_model_id(combo)
        if not repo_id:
            return
        confirm = QMessageBox.question(
            self,
            "Supprimer le modèle",
            f"Supprimer le cache local du modèle ?\n\n{repo_id}\n\nIl sera retéléchargé au prochain usage.",
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        self._run_model_cache_worker("delete", repo_id)

    def _run_model_cache_worker(self, mode: str, repo_id: str):
        if self._model_cache_worker and self._model_cache_worker.isRunning():
            return
        token = self.transcription_hf_token_edit.text().strip()
        self._model_busy_repo_id = repo_id
        self._model_cache_worker = ModelCacheWorker(mode, repo_id, token, self)
        self._model_cache_worker.progress.connect(self._on_model_cache_progress)
        self._model_cache_worker.finished_ok.connect(self._on_model_cache_finished)
        self._model_cache_worker.failed.connect(self._on_model_cache_failed)
        if mode == "download":
            self.model_cache_progress.setRange(0, 100)
            self.model_cache_progress.setValue(0)
            self.model_cache_progress.setFormat(f"Téléchargement de {repo_id}… 0%")
        else:
            self.model_cache_progress.setRange(0, 0)
            self.model_cache_progress.setFormat(f"Suppression de {repo_id}…")
        self.model_cache_progress.setVisible(True)
        self._refresh_model_cache_buttons()
        self._model_cache_worker.start()

    def _on_model_cache_progress(self, pct: int, label: str):
        pct = max(0, min(100, int(pct)))
        self.model_cache_progress.setRange(0, 100)
        self.model_cache_progress.setValue(pct)
        suffix = f" · {label}" if label and label not in {"None", "null"} else ""
        self.model_cache_progress.setFormat(f"Téléchargement… {pct}%{suffix}")

    def _on_model_cache_finished(self, mode: str, repo_id: str):
        action = "téléchargé" if mode == "download" else "supprimé"
        for combo, _, _ in self._model_controls:
            self._refresh_model_combo_labels(combo)
        self._model_cache_worker = None
        self._model_busy_repo_id = ""
        self.model_cache_progress.setRange(0, 100)
        self.model_cache_progress.setValue(100 if mode == "download" else 0)
        self.model_cache_progress.setFormat(f"Modèle {action}")
        self._refresh_model_cache_buttons()
        QMessageBox.information(self, "Modèles", f"Modèle {action} :\n{repo_id}")
        self.model_cache_progress.setVisible(False)

    def _on_model_cache_failed(self, message: str):
        self._model_cache_worker = None
        self._model_busy_repo_id = ""
        self.model_cache_progress.setRange(0, 100)
        self.model_cache_progress.setValue(0)
        self.model_cache_progress.setFormat("Échec")
        self.model_cache_progress.setVisible(False)
        self._refresh_model_cache_buttons()
        QMessageBox.warning(self, "Modèles", f"Opération impossible :\n{message}")

    def pick_ffmpeg(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choisir ffmpeg", str(Path.home()), "Tous (*.*)")
        if path:
            self.ffmpeg_edit.setText(path)

    def pick_ffprobe(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choisir ffprobe", str(Path.home()), "Tous (*.*)")
        if path:
            self.ffprobe_edit.setText(path)

    def pick_mlx_whisper(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choisir mlx_whisper", str(Path.home()), "Tous (*.*)")
        if path:
            self.mlx_whisper_edit.setText(path)

    def export_logs(self):
        try:
            archive = export_logs_archive(self)
        except Exception as exc:
            QMessageBox.warning(self, "Exporter les logs", f"Export impossible: {exc}")
            return
        if archive:
            QMessageBox.information(self, "Logs exportés", f"Archive créée:\n{archive}")

    def open_huggingface_auth(self):
        dlg = HuggingFaceAuthDialog(self.transcription_hf_token_edit.text().strip(), self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self.transcription_hf_token_edit.setText(dlg.token())

    def _combo_model_id(self, combo: QComboBox) -> str:
        # currentData() is the structured model id. The visible text includes
        # human labels and cache badges, so only use it as a defensive fallback.
        data = combo.currentData()
        if isinstance(data, str) and data.strip():
            return data.strip()
        return combo.currentText().strip()

    def values(self) -> tuple[str, str, str, dict[str, str | bool]]:
        return (
            self.ffmpeg_edit.text().strip(),
            self.ffprobe_edit.text().strip(),
            self.token_edit.text().strip(),
            {
                "mlx_whisper_path": self.mlx_whisper_edit.text().strip(),
                "model": canonical_whisper_model_id(
                    str(self.transcription_model_combo.currentData() or DEFAULT_WHISPER_MODEL)
                ),
                "text_llm_model": self._combo_model_id(self.transcription_text_llm_combo),
                "audio_llm_model": canonical_audio_llm_model_id(
                    self._combo_model_id(self.transcription_audio_llm_combo)
                ),
                "audio_recheck_enabled": self.transcription_audio_recheck_check.isChecked(),
                "language": self.transcription_language_combo.currentText(),
                "format": self.transcription_format_combo.currentText(),
                "suffix": self.transcription_suffix_value,
                "enhance_audio": self.transcription_enhance_check.isChecked(),
                "diarization_enabled": self.transcription_diarization_check.isChecked(),
                "hf_token": self.transcription_hf_token_edit.text().strip(),
            },
        )


class EncodeWorker(QThread):
    progress = Signal(int)
    status = Signal(str)
    finished_ok = Signal(str)
    failed = Signal(str)
    duration_unknown = Signal(bool)

    def __init__(
        self,
        ffmpeg_path: str,
        ffprobe_path: str | None,
        job: QueueJob,
    ):
        super().__init__()
        self.ffmpeg_path = ffmpeg_path
        self.ffprobe_path = ffprobe_path
        self.job = job
        self._stop_requested = False

    def request_stop(self):
        self._stop_requested = True

    def run(self):
        if not self.ffmpeg_path or not Path(self.ffmpeg_path).exists():
            self.failed.emit("ffmpeg introuvable. Vérifiez les paramètres.")
            return
        if not Path(self.job.input_path).exists():
            self.failed.emit("Fichier d'entrée introuvable.")
            return

        duration = probe_duration_seconds(self.job.input_path, self.ffprobe_path, self.ffmpeg_path)
        self.duration_unknown.emit(duration is None)

        # Audio-only inputs (.mp3 / .m4a / .wav…) shouldn't be re-encoded
        # as H.265 video. We trust the extension first; ffprobe will catch
        # anything weird when it actually runs.
        audio_only = is_audio_only_path(self.job.input_path) or is_audio_only_path(self.job.output_path)

        ss = self.job.trim_start if self.job.trim_enabled else None
        to = self.job.trim_end if self.job.trim_enabled else None
        cmd = build_ffmpeg_cmd(
            ffmpeg_path=self.ffmpeg_path,
            in_path=self.job.input_path,
            out_path=self.job.output_path,
            crf=self.job.crf,
            resolution=self.job.resolution,
            fps=self.job.fps,
            audio_bitrate=self.job.audio_bitrate,
            preset=self.job.x265_preset,
            speech_enhance=self.job.speech_enhance,
            mono_audio=self.job.mono_audio,
            ss=ss,
            to=to,
            audio_only=audio_only,
        )

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )

            last_pct = -1
            while True:
                if self._stop_requested:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                    self.failed.emit("Annulé.")
                    return

                line = proc.stdout.readline() if proc.stdout else ""
                if not line:
                    break
                line = line.strip()

                if line.startswith("out_time_ms="):
                    try:
                        out_time_ms = int(line.split("=", 1)[1])
                    except Exception:
                        continue

                    if duration and duration > 0:
                        current_sec = out_time_ms / 1_000_000.0
                        pct = int(min(100, max(0, (current_sec / duration) * 100)))
                        if pct != last_pct:
                            last_pct = pct
                            self.progress.emit(pct)
                            self.status.emit("Compression…")
                    else:
                        self.status.emit("Compression…")

                elif line.startswith("progress=end"):
                    break

            rc = proc.wait()
            if rc != 0:
                err = ""
                try:
                    err = (proc.stderr.read() or "").strip()
                except Exception:
                    pass
                self.failed.emit(err or f"ffmpeg a échoué (code {rc}).")
                return

            self.progress.emit(100)
            self.status.emit("Terminé.")
            self.finished_ok.emit(self.job.output_path)
        except Exception as exc:
            self.failed.emit(str(exc))


class TranscribeWorker(QThread):
    progress = Signal(int)
    status = Signal(str)
    finished_ok = Signal(str)
    failed = Signal(str)
    duration_unknown = Signal(bool)

    def __init__(
        self,
        ffmpeg_path: str,
        ffprobe_path: str | None,
        job: QueueJob,
        mlx_whisper_path: str,
        model: str,
        language: str,
        output_format: str,
        initial_prompt: str,
        enhance_audio: bool,
        db: DatabaseManager,
        diarization_enabled: bool = False,
        hf_token: str = "",
        venv_python_path: str = "",
        text_llm_model: str = DEFAULT_TEXT_LLM_MODEL,
        audio_llm_model: str = DEFAULT_AUDIO_LLM_MODEL,
        audio_recheck_enabled: bool = False,
    ):
        super().__init__()
        self.ffmpeg_path = ffmpeg_path
        self.ffprobe_path = ffprobe_path
        self.job = job
        self.mlx_whisper_path = mlx_whisper_path
        self.model = model
        self.language = language
        self.output_format = output_format
        self.db = db
        # Two distinct models: the text LLM (mlx_lm) is required for the
        # title/speakers/corrections pass; the audio multimodal model
        # (mlx_vlm) is only loaded when the optional re-listen step runs.
        self.text_llm_model = text_llm_model or DEFAULT_TEXT_LLM_MODEL
        self.audio_llm_model = audio_llm_model or DEFAULT_AUDIO_LLM_MODEL
        self.audio_recheck_enabled = bool(audio_recheck_enabled)
        # Wrap the user's "Contexte" in a French priming sentence so Whisper
        # treats the listed terms as expected vocabulary, not a token salad.
        self.initial_prompt = structured_initial_prompt(initial_prompt)
        self.enhance_audio = enhance_audio
        self.diarization_enabled = diarization_enabled
        self.hf_token = hf_token
        self.venv_python_path = venv_python_path
        self._stop_requested = False
        self._proc: subprocess.Popen | None = None
        self._mlx_vlm_available: bool | None = None

    def _log(self, message: str):
        append_app_log(f"transcription file={Path(self.job.input_path).name!r} {message}")

    def _update_db_status(self, status: str, error: str = ""):
        if self.job.db_id:
            self.db.update_job_status(self.job.db_id, status, error)
            self.job.status = status

    def _log_process_result(
        self,
        label: str,
        returncode: int | None,
        stdout: str | None = None,
        stderr: str | None = None,
    ):
        details = [f"{label} returncode={returncode}"]
        stdout_tail = _tail_for_log(stdout)
        stderr_tail = _tail_for_log(stderr)
        if stdout_tail:
            details.append(f"stdout={stdout_tail!r}")
        if stderr_tail:
            details.append(f"stderr={stderr_tail!r}")
        self._log(" ".join(details))

    def _subprocess_env(self) -> dict[str, str]:
        env = os.environ.copy()
        ffmpeg_dir = str(Path(self.ffmpeg_path).parent)
        if ffmpeg_dir:
            env["PATH"] = ffmpeg_dir + os.pathsep + env.get("PATH", "")
        return env

    def _ensure_mlx_lm_available(self) -> bool:
        """
        Probe / install mlx_lm in the managed venv. Used by the text LLM step
        (title/speakers/corrections). Cheap install (~200 Mo).
        """
        if not self.venv_python_path or not Path(self.venv_python_path).exists():
            return False
        probe = subprocess.run(
            [self.venv_python_path, "-c", "import mlx_lm"],
            capture_output=True,
            text=True,
            env=self._subprocess_env(),
        )
        if probe.returncode == 0:
            return True

        self.status.emit("Installation de l'IA texte locale (MLX LM)…")
        install = subprocess.run(
            [self.venv_python_path, "-m", "pip", "install", "--upgrade", "mlx-lm"],
            capture_output=True,
            text=True,
            env=self._subprocess_env(),
        )
        if install.returncode != 0:
            self._log_process_result("mlx_lm_install_failed", install.returncode, install.stdout, install.stderr)
            return False
        return True

    def _ensure_mlx_vlm_available(self) -> bool:
        """
        Probe / install mlx_vlm. Heavier than mlx_lm (~500 Mo) and only
        needed if the user opted into the audio re-listen step. We cache
        the probe result so we don't pay for it on every clip.
        """
        if self._mlx_vlm_available is not None:
            return self._mlx_vlm_available
        if not self.venv_python_path or not Path(self.venv_python_path).exists():
            self._mlx_vlm_available = False
            return False
        probe = subprocess.run(
            [self.venv_python_path, "-c", "import mlx_vlm"],
            capture_output=True,
            text=True,
            env=self._subprocess_env(),
        )
        if probe.returncode == 0:
            self._mlx_vlm_available = True
            return True

        self.status.emit("Installation de l'IA multimodale (MLX VLM, ~500 Mo)…")
        install = subprocess.run(
            [self.venv_python_path, "-m", "pip", "install", "--upgrade", "mlx-vlm"],
            capture_output=True,
            text=True,
            env=self._subprocess_env(),
        )
        if install.returncode != 0:
            self._log_process_result("mlx_vlm_install_failed", install.returncode, install.stdout, install.stderr)
            self._mlx_vlm_available = False
            return False
        self._mlx_vlm_available = True
        return True

    def _run_llm_post_process(self, analysis_path: Path, glossary: str) -> dict:
        """
        Two-step LLM analysis. Asking a 7B-4bit model for {title, speakers,
        corrections, uncertain_passages} in one shot consistently broke the
        JSON past ~700 output tokens. We now do:

        1. Short JSON call for title + speakers + technical terms — small
           enough to be reliable.
        2. Markdown call (max 900 tokens) for corrections + doubts —
           tolerant parser, a malformed line drops one entry instead of
           wrecking the whole pass.

        Either step may fail independently; the worker keeps going with
        whatever partial data it got.
        """
        if not analysis_path.exists() or not self._ensure_mlx_lm_available():
            return {}

        # --- Step 1: title + speakers + technical terms ----------------
        self.status.emit("Analyse: titre, interlocuteurs et termes…")
        title_cmd = build_llm_title_cmd(
            self.venv_python_path, self.text_llm_model, str(analysis_path), glossary
        )
        self._log(f"llm_title_start model={self.text_llm_model!r}")
        title_proc = subprocess.run(
            title_cmd,
            capture_output=True,
            text=True,
            env=self._subprocess_env(),
            timeout=900,
        )
        title_speakers: dict = {}
        if title_proc.returncode != 0:
            self._log_process_result(
                "llm_title_failed", title_proc.returncode, title_proc.stdout, title_proc.stderr
            )
        else:
            self._log_process_result(
                "llm_title_ok", title_proc.returncode, title_proc.stdout, title_proc.stderr
            )
            title_speakers = parse_llm_title_speakers(title_proc.stdout)
            if not title_speakers:
                self._log("llm_title_parse_empty")

        # --- Step 2: corrections + doubts ------------------------------
        self.status.emit("Analyse: corrections et passages douteux…")
        corrections_cmd = build_llm_corrections_cmd(
            self.venv_python_path, self.text_llm_model, str(analysis_path), glossary
        )
        self._log(f"llm_corrections_start model={self.text_llm_model!r}")
        corrections_proc = subprocess.run(
            corrections_cmd,
            capture_output=True,
            text=True,
            env=self._subprocess_env(),
            timeout=1800,
        )
        corrections_payload: dict = {"corrections": [], "uncertain_passages": []}
        if corrections_proc.returncode != 0:
            self._log_process_result(
                "llm_corrections_failed",
                corrections_proc.returncode,
                corrections_proc.stdout,
                corrections_proc.stderr,
            )
        else:
            self._log_process_result(
                "llm_corrections_ok",
                corrections_proc.returncode,
                corrections_proc.stdout,
                corrections_proc.stderr,
            )
            corrections_payload = parse_llm_corrections_markdown(corrections_proc.stdout)

        # Merge both responses into the legacy payload shape so downstream
        # code (_create_enhanced_transcript, _write_review_file…) stays
        # untouched.
        return {
            "title": title_speakers.get("title", ""),
            "speakers": title_speakers.get("speakers", {}),
            "technical_terms": title_speakers.get("technical_terms", []),
            "corrections": corrections_payload.get("corrections", []),
            "uncertain_passages": corrections_payload.get("uncertain_passages", []),
        }

    def _speaker_names_from_payload(self, payload: dict) -> dict[str, str]:
        speakers_raw = payload.get("speakers") or {}
        if not isinstance(speakers_raw, dict):
            return {}
        return {
            str(key).strip(): str(value).strip()
            for key, value in speakers_raw.items()
            if str(key).strip().startswith("SPEAKER_") and str(value).strip()
        }

    def _technical_terms_from_payload(self, payload: dict) -> list[str]:
        terms_raw = payload.get("technical_terms") or []
        terms: list[str] = []
        if isinstance(terms_raw, list):
            for value in terms_raw:
                term = str(value).strip()
                if term and term not in terms:
                    terms.append(term)
        return terms

    def _apply_speaker_names(self, out_path: Path, speakers: dict[str, str]):
        if not speakers or not out_path.exists():
            return
        try:
            if out_path.suffix.lower() == ".json":
                payload = json.loads(out_path.read_text(encoding="utf-8"))
                for segment in payload.get("segments", []):
                    speaker = str(segment.get("speaker") or "")
                    if speaker in speakers:
                        segment["speaker"] = speakers[speaker]
                out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                return

            text = out_path.read_text(encoding="utf-8")
            for speaker_id, speaker_name in speakers.items():
                text = text.replace(f"[{speaker_id}]", f"[{speaker_name}]")
                text = re.sub(rf"\b{re.escape(speaker_id)}\b", speaker_name, text)
            out_path.write_text(text, encoding="utf-8")
        except Exception as exc:
            self._log(f"speaker_rename_failed error={exc!r}")

    def _enhanced_transcript_path(self, out_path: Path) -> Path:
        return out_path.with_name(f"{out_path.stem} - améliorée{out_path.suffix}")

    def _review_transcript_path(self, out_path: Path) -> Path:
        return out_path.with_name(f"{out_path.stem} - à vérifier.md")

    def _confidence_value(self, value) -> float:
        try:
            return float(value)
        except Exception:
            return 0.0

    def _apply_text_corrections(self, text: str, corrections: list[dict]) -> tuple[str, list[dict]]:
        applied: list[dict] = []
        improved = text
        for raw in corrections:
            if not isinstance(raw, dict):
                continue
            original = str(raw.get("original") or "").strip()
            replacement = str(raw.get("replacement") or "").strip()
            confidence = self._confidence_value(raw.get("confidence"))
            if len(original) < 3 or not replacement or original == replacement:
                continue
            if confidence < 0.75 or original not in improved:
                continue
            improved = improved.replace(original, replacement)
            applied.append({
                "timestamp": str(raw.get("timestamp") or "").strip(),
                "original": original,
                "replacement": replacement,
                "confidence": confidence,
                "reason": str(raw.get("reason") or "").strip(),
            })
        return improved, applied

    def _timestamp_to_seconds(self, value: str) -> float | None:
        text = (value or "").strip()
        match = re.search(r"(\d{1,2}):(\d{2})(?::(\d{2}))?", text)
        if not match:
            return None
        first = int(match.group(1))
        second = int(match.group(2))
        third = int(match.group(3) or 0)
        if match.group(3) is None:
            return float(first * 60 + second)
        return float(first * 3600 + second * 60 + third)

    def _format_seconds_for_clip(self, seconds: float) -> str:
        seconds = max(0.0, seconds)
        return f"{seconds:.2f}".rstrip("0").rstrip(".")

    def _run_clip_rechecks(
        self,
        wav_path: Path,
        work_dir: Path,
        uncertain_passages: list[dict],
    ) -> list[dict]:
        # The multimodal step is opt-in. Even if the text LLM flagged doubts,
        # we don't load mlx_vlm + Qwen2-Audio (~5 Go RAM, slow first launch)
        # unless the user explicitly enabled it in Settings.
        if not self.audio_recheck_enabled:
            return []
        if not uncertain_passages or not wav_path.exists() or not self.venv_python_path:
            return []
        if not self._ensure_mlx_vlm_available():
            self.status.emit("Réécoute IA ignorée (mlx-vlm indisponible).")
            self._log("clip_recheck_skipped reason=mlx_vlm_unavailable")
            return []

        checks: list[dict] = []
        self.status.emit("Réécoute IA des passages douteux (multimodal)…")
        for index, passage in enumerate(uncertain_passages[:10], start=1):
            if self._stop_requested:
                break
            if not isinstance(passage, dict):
                continue
            center = self._timestamp_to_seconds(str(passage.get("timestamp") or ""))
            if center is None:
                continue
            start = max(0.0, center - 5.0)
            end = center + 10.0
            
            # 1. Extract specific audio clip
            clip_wav = work_dir / f"clip_{index}.wav"
            extract_cmd = build_audio_extract_cmd(
                self.ffmpeg_path,
                str(wav_path),
                str(clip_wav),
                speech_enhance=False,
                ss=self._format_seconds_for_clip(start),
                to=self._format_seconds_for_clip(end)
            )
            subprocess.run(extract_cmd, capture_output=True, env=self._subprocess_env())
            if not clip_wav.exists():
                continue
                
            passage["clip_path"] = str(clip_wav)
            
            # 2. Run multimodal AI check.
            # Pass the meeting glossary so Qwen2-Audio knows the
            # business terminology (proper nouns, project names, client
            # acronyms) — without it the model has zero context for
            # "Ekonum" / "Adèle Herbaoui" / "Captivea" and just guesses
            # phonetically.
            question = str(passage.get("text") or "")
            reason = str(passage.get("reason") or "")
            glossary_block = self.initial_prompt or ""
            glossary_hint = (
                f"\nVocabulaire métier attendu (priorité absolue) : {glossary_block}"
                if glossary_block.strip()
                else ""
            )
            prompt = (
                "Tu écoutes un court extrait d'une réunion en français.\n"
                f"Whisper hésite sur : « {question} ».\n"
                f"Raison du doute : {reason}.{glossary_hint}\n"
                "Compare l'audio aux termes du vocabulaire : les noms propres, marques, "
                "logiciels et acronymes doivent être orthographiés exactement comme dans ce vocabulaire "
                "si le son correspond. Corrige aussi les mots absurdes vers le mot français évident "
                "quand la proximité phonétique est forte.\n"
                "Que dit réellement la personne dans cet extrait ? "
                "Réponds par la phrase exacte, sans commentaire."
            )
            
            cmd = build_multimodal_audio_cmd(
                venv_python_path=self.venv_python_path,
                model_path=self.audio_llm_model,
                audio_path=str(clip_wav),
                prompt=prompt
            )
            self._log(
                "multimodal_check_start "
                f"timestamp={str(passage.get('timestamp') or '')!r} cmd={_command_for_log(cmd)!r}"
            )
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=self._subprocess_env(),
                timeout=600,
            )
            if proc.returncode != 0:
                self._log_process_result("multimodal_check_failed", proc.returncode, proc.stdout, proc.stderr)
                continue
                
            try:
                result = json.loads((proc.stdout or "").strip().splitlines()[-1])
                suggestion = result.get("suggestion")
                if suggestion:
                    passage["suggestion"] = suggestion
            except Exception as exc:
                self._log(f"multimodal_check_parse_failed error={exc!r}")
                continue
                
            checks.append({
                "timestamp": str(passage.get("timestamp") or ""),
                "question": question,
                "suggestion": str(passage.get("suggestion") or ""),
                "audio_transcript": "(multimodal évalué)",
            })
        return checks

    def _write_review_file(
        self,
        review_path: Path,
        payload: dict,
        applied_corrections: list[dict],
        clip_checks: list[dict],
    ):
        lines = [
            "# Vérification de transcription",
            "",
            "Ce fichier liste les corrections appliquées automatiquement et les passages qui restent à vérifier.",
            "",
        ]

        speakers = self._speaker_names_from_payload(payload)
        terms = self._technical_terms_from_payload(payload)
        if speakers or terms:
            lines += ["## Contexte détecté", ""]
            if speakers:
                lines.append("Interlocuteurs :")
                for speaker_id, speaker_name in sorted(speakers.items()):
                    lines.append(f"- `{speaker_id}` : {speaker_name}")
                lines.append("")
            if terms:
                lines.append("Termes techniques :")
                for term in terms:
                    lines.append(f"- {term}")
                lines.append("")

        if applied_corrections:
            lines += ["## Corrections appliquées", ""]
            for corr in applied_corrections:
                ts = corr.get("timestamp") or "sans timestamp"
                reason = corr.get("reason") or "contexte"
                lines.append(
                    f"- `{ts}` `{corr['original']}` -> `{corr['replacement']}` "
                    f"(confiance {corr['confidence']:.2f}) : {reason}"
                )
            lines.append("")

        uncertain = payload.get("uncertain_passages") or []
        if isinstance(uncertain, list) and uncertain:
            lines += ["## Passages à vérifier", ""]
            for item in uncertain[:20]:
                if not isinstance(item, dict):
                    continue
                lines.append(f"- `{item.get('timestamp') or 'sans timestamp'}` {item.get('text') or ''}")
                if item.get("reason"):
                    lines.append(f"  Raison : {item.get('reason')}")
                if item.get("suggestion"):
                    lines.append(f"  Hypothèse : {item.get('suggestion')}")
            lines.append("")

        if clip_checks:
            lines += ["## Réécoute ciblée", ""]
            for check in clip_checks:
                lines.append(f"- `{check['timestamp']}` {check['question']}")
                if check.get("suggestion"):
                    lines.append(f"  Hypothèse IA : {check['suggestion']}")
                lines.append(f"  Réécoute Whisper : {check.get('audio_transcript') or '(vide)'}")
            lines.append("")

        if len(lines) <= 4:
            lines.append("Aucune correction ni vérification signalée.")
        review_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    def _create_enhanced_transcript(
        self,
        out_path: Path,
        payload: dict,
        wav_path: Path,
        work_dir: Path,
    ) -> Path | None:
        if not payload or not out_path.exists():
            return None
        try:
            base_text = out_path.read_text(encoding="utf-8")
        except Exception as exc:
            self._log(f"enhancement_read_failed error={exc!r}")
            return None

        speakers = self._speaker_names_from_payload(payload)
        technical_terms = self._technical_terms_from_payload(payload)
        if self.job.db_id and (speakers or technical_terms):
            try:
                self.db.update_job_context(self.job.db_id, speakers, technical_terms)
            except Exception as exc:
                self._log(f"context_persist_failed error={exc!r}")
        enhanced_text = base_text
        for speaker_id, speaker_name in speakers.items():
            enhanced_text = enhanced_text.replace(f"[{speaker_id}]", f"[{speaker_name}]")
            enhanced_text = re.sub(rf"\b{re.escape(speaker_id)}\b", speaker_name, enhanced_text)

        corrections_raw = payload.get("corrections") or []
        corrections = corrections_raw if isinstance(corrections_raw, list) else []
        enhanced_text, applied = self._apply_text_corrections(enhanced_text, corrections)

        uncertain_raw = payload.get("uncertain_passages") or []
        uncertain = uncertain_raw if isinstance(uncertain_raw, list) else []
        clip_checks = self._run_clip_rechecks(wav_path, work_dir, uncertain)

        has_changes = enhanced_text != base_text or applied or speakers
        has_review = applied or uncertain or clip_checks or speakers or technical_terms
        if has_review:
            self._write_review_file(self._review_transcript_path(out_path), payload, applied, clip_checks)

        if not has_changes:
            return None

        enhanced_path = self._enhanced_transcript_path(out_path)
        index = 1
        while enhanced_path.exists():
            enhanced_path = out_path.with_name(f"{out_path.stem} - améliorée_{index}{out_path.suffix}")
            index += 1
        enhanced_path.write_text(enhanced_text, encoding="utf-8")
        self._log(
            "enhanced_transcript_written "
            f"output={str(enhanced_path)!r} corrections={len(applied)} speakers={len(speakers)} "
            f"review={has_review}"
        )
        return enhanced_path

    def _rename_transcript_file(self, out_path: Path, title: str) -> Path:
        if not out_path.exists():
            return out_path
        try:
            if not title:
                title = suggest_transcript_stem(out_path.read_text(encoding="utf-8"), Path(self.job.input_path).stem)
            safe_stem = suggest_transcript_stem(title, Path(self.job.input_path).stem)
            candidate = out_path.with_name(f"{safe_stem}{out_path.suffix}")
            if candidate == out_path:
                return out_path
            index = 1
            while candidate.exists():
                candidate = out_path.with_name(f"{safe_stem}_{index}{out_path.suffix}")
                index += 1
            out_path.rename(candidate)
            self._log(f"transcription_auto_rename from={str(out_path)!r} to={str(candidate)!r}")
            return candidate
        except Exception as exc:
            self._log(f"transcription_auto_rename_failed path={str(out_path)!r} error={exc!r}")
            return out_path

    def request_stop(self):
        self._stop_requested = True
        if self._proc:
            try:
                self._proc.kill()
            except Exception:
                pass

    def run(self):
        if not self.ffmpeg_path or not Path(self.ffmpeg_path).exists():
            self._log(f"blocked reason=missing_ffmpeg path={self.ffmpeg_path!r}")
            self.failed.emit("ffmpeg introuvable. Vérifiez les paramètres.")
            return
        if not self.mlx_whisper_path or not Path(self.mlx_whisper_path).exists():
            self._log(f"blocked reason=missing_mlx_whisper path={self.mlx_whisper_path!r}")
            self.failed.emit(
                "mlx_whisper introuvable. Utilisez le bouton Installer MLX Whisper dans l'onglet Transcrire."
            )
            return
        if not Path(self.job.input_path).exists():
            self._log(f"blocked reason=missing_input path={self.job.input_path!r}")
            self.failed.emit("Fichier d'entrée introuvable.")
            return
        if not self.job.transcript_path:
            self._log("blocked reason=missing_transcript_path")
            self.failed.emit("Chemin de transcription manquant.")
            return

        self._log(
            "start "
            f"output={self.job.transcript_path!r} model={self.model!r} language={self.language!r} "
            f"format={self.output_format!r} enhance_audio={self.enhance_audio} "
            f"diarization_enabled={self.diarization_enabled} "
            f"hf_token_configured={bool(self.hf_token)} venv_python_configured={bool(self.venv_python_path)} "
            f"prompt_chars={len(self.initial_prompt)} db_status={self.job.status}"
        )
        duration = probe_duration_seconds(self.job.input_path, self.ffprobe_path, self.ffmpeg_path)
        self.duration_unknown.emit(duration is None)

        # Resumption logic: use workspace if it exists, otherwise use temp
        tmp_dir = None
        if self.job.workspace_dir and Path(self.job.workspace_dir).exists():
            work_dir = Path(self.job.workspace_dir)
        else:
            tmp_dir = Path(tempfile.mkdtemp(prefix="ekovideo-transcribe-"))
            work_dir = tmp_dir
            
        wav_path = work_dir / "audio.wav"
        
        t_start = time.monotonic()
        d_ffmpeg = 0
        d_whisper = 0
        d_diarization = 0

        try:
            # --- STEP 1: audio extraction ---------------------------------
            t_step = time.monotonic()
            skip_audio = (self.job.status in {"AUDIO_READY", "WHISPER_DONE", "COMPLETED"} 
                         and wav_path.exists())
            
            if not skip_audio:
                self.status.emit("Préparation audio original…")
                ss = self.job.trim_start if self.job.trim_enabled else None
                to = self.job.trim_end if self.job.trim_enabled else None
                extract_cmd = build_audio_extract_cmd(
                    ffmpeg_path=self.ffmpeg_path,
                    in_path=self.job.input_path,
                    wav_path=str(wav_path),
                    speech_enhance=self.enhance_audio,
                    ss=ss,
                    to=to,
                )
                self._log(f"audio_extract_start cmd={_command_for_log(extract_cmd)!r}")
                self._proc = subprocess.Popen(
                    extract_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                    universal_newlines=True,
                    env=self._subprocess_env(),
                )

                last_pct = -1
                while True:
                    if self._stop_requested:
                        self.failed.emit("Annulé.")
                        return

                    line = self._proc.stdout.readline() if self._proc.stdout else ""
                    if not line:
                        break
                    line = line.strip()

                    if line.startswith("out_time_ms="):
                        try:
                            out_time_ms = int(line.split("=", 1)[1])
                        except Exception:
                            continue

                        if duration and duration > 0:
                            current_sec = out_time_ms / 1_000_000.0
                            pct = int(min(25, max(0, (current_sec / duration) * 25)))
                            if pct != last_pct:
                                last_pct = pct
                                self.progress.emit(pct)
                                self.status.emit("Préparation audio…")
                        else:
                            self.status.emit("Préparation audio…")

                    elif line.startswith("progress=end"):
                        break

                rc = self._proc.wait()
                d_ffmpeg = time.monotonic() - t_step
                if rc != 0:
                    err = ""
                    try:
                        err = (self._proc.stderr.read() or "").strip()
                    except Exception:
                        pass
                    self._log_process_result("ffmpeg_extract_failed", rc, stderr=err)
                    self._update_db_status("FAILED", err or f"ffmpeg a échoué (code {rc}).")
                    self.failed.emit(err or f"ffmpeg a échoué (code {rc}).")
                    return
                self._update_db_status("AUDIO_READY")
            else:
                self._log("resumption skip=audio_extract")
                self.progress.emit(25)

            if self._stop_requested:
                self.failed.emit("Annulé.")
                return

            out_path = Path(self.job.transcript_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)

            run_diarization = bool(self.diarization_enabled and self.hf_token and self.venv_python_path)
            if self.diarization_enabled and not run_diarization:
                self._log(
                    "diarization_skipped "
                    f"hf_token_configured={bool(self.hf_token)} venv_python_configured={bool(self.venv_python_path)}"
                )
                self.status.emit("Détection des locuteurs ignorée (token Hugging Face ou installation manquante).")

            # Always keep a JSON Whisper intermediate so we can validate and
            # filter hallucinated segments before rendering the user-facing
            # transcript. This is essential for long recordings with noisy
            # leading silence, where Whisper can otherwise output only "...".
            whisper_target_dir = work_dir if run_diarization else out_path.parent
            whisper_target_stem = "whisper" if run_diarization else f"{out_path.stem}.whisper"
            whisper_format = "json"
            whisper_target = whisper_target_dir / f"{whisper_target_stem}.{transcript_output_ext(whisper_format)}"

            # --- STEP 2: MLX Whisper --------------------------------------
            t_step = time.monotonic()
            skip_whisper = (self.job.status in {"WHISPER_DONE", "COMPLETED"} 
                           and whisper_target.exists())
            if skip_whisper:
                try:
                    if not parse_whisper_json_segments(str(whisper_target)):
                        whisper_target.unlink(missing_ok=True)
                        skip_whisper = False
                        self._log("resumption discard=invalid_whisper_json")
                except Exception:
                    whisper_target.unlink(missing_ok=True)
                    skip_whisper = False
                    self._log("resumption discard=unreadable_whisper_json")
            
            if not skip_whisper:
                cmd = build_mlx_whisper_cmd(
                    mlx_whisper_path=self.mlx_whisper_path,
                    audio_path=str(wav_path),
                    output_path=str(whisper_target),
                    model=self.model,
                    language=self.language,
                    output_format=whisper_format,
                    initial_prompt=self.initial_prompt,
                )

                self.duration_unknown.emit(duration is None)
                self.progress.emit(25)
                self.status.emit("Transcription locale MLX…")
                self._log(
                    "mlx_whisper_start "
                    f"target={str(whisper_target)!r} target_format={whisper_format!r} "
                    f"cmd={_command_for_log(cmd)!r}"
                )
                self._proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True,
                    env=self._subprocess_env(),
                )

                output_lines: list[str] = []
                last_transcribe_pct = 25
                while True:
                    if self._stop_requested:
                        self._proc.kill()
                        self.failed.emit("Annulé.")
                        return

                    line = self._proc.stdout.readline() if self._proc.stdout else ""
                    if not line:
                        if self._proc.poll() is not None:
                            break
                        continue

                    output_lines.append(line)
                    if duration and duration > 0:
                        segment_seconds = parse_whisper_segment_seconds(line)
                        if segment_seconds is not None:
                            pct = 25 + int(min(50, max(0, (segment_seconds / duration) * 50)))
                            if pct > last_transcribe_pct:
                                last_transcribe_pct = pct
                                self.progress.emit(pct)

                self._proc.wait()
                d_whisper = time.monotonic() - t_step
                stdout = "".join(output_lines)
                stderr = ""
                if self._proc.returncode != 0:
                    detail = (stderr or stdout or "").strip()
                    self._log_process_result("mlx_whisper_failed", self._proc.returncode, stdout, stderr)
                    self._update_db_status("FAILED", detail or f"mlx_whisper a échoué (code {self._proc.returncode}).")
                    self.failed.emit(detail or f"mlx_whisper a échoué (code {self._proc.returncode}).")
                    return
                if not whisper_target.exists():
                    detail = _tail_for_log(stdout)
                    self._log_process_result("mlx_whisper_missing_output", self._proc.returncode, stdout, stderr)
                    self._update_db_status("FAILED", "mlx_whisper n'a pas produit le fichier attendu.")
                    self.failed.emit(
                        "mlx_whisper n'a pas produit le fichier de transcription attendu. "
                        f"{detail}".strip()
                    )
                    return
                try:
                    whisper_segments = parse_whisper_json_segments(str(whisper_target))
                except Exception as exc:
                    self._log(f"mlx_whisper_invalid_json error={exc!r}")
                    self._update_db_status("FAILED", f"Lecture de la transcription Whisper impossible: {exc}")
                    self.failed.emit(f"Lecture de la transcription Whisper impossible: {exc}")
                    return
                if not whisper_segments:
                    self._log_process_result("mlx_whisper_no_speech_after_filter", self._proc.returncode, stdout, stderr)
                    self._update_db_status(
                        "FAILED",
                        "Whisper n'a produit que des segments vides ou hallucines.",
                    )
                    self.failed.emit(
                        "Whisper n'a produit que des segments vides ou hallucines. "
                        "Essayez sans nettoyage audio, ou avec la vidéo originale si vous transcriviez une version compressée."
                    )
                    return
                self._log_process_result("mlx_whisper_ok", self._proc.returncode, stdout, stderr)
                self._update_db_status("WHISPER_DONE")
            else:
                self._log("resumption skip=whisper")
                self.progress.emit(75)

            try:
                whisper_segments = parse_whisper_json_segments(str(whisper_target))
            except Exception as exc:
                self._log(f"mlx_whisper_parse_failed error={exc!r}")
                self._update_db_status("FAILED", f"Lecture de la transcription Whisper impossible: {exc}")
                self.failed.emit(f"Lecture de la transcription Whisper impossible: {exc}")
                return
            if not whisper_segments:
                self._log("mlx_whisper_empty_after_filter")
                self._update_db_status("FAILED", "Whisper n'a produit aucun segment vocal exploitable.")
                self.failed.emit(
                    "Whisper n'a produit aucun segment vocal exploitable. "
                    "La sortie brute ressemblait à une hallucination de silence."
                )
                return

            # --- STEP 3: diarization and fusion ---------------------------
            analysis_segments = whisper_segments
            if run_diarization:
                t_step = time.monotonic()
                self.progress.emit(75)
                self.status.emit("Détection des locuteurs (pyannote)…")
                diar_cmd = build_diarization_cmd(self.venv_python_path, str(wav_path))
                env = self._subprocess_env()
                env["HF_TOKEN"] = self.hf_token
                self._log(f"diarization_start cmd={_command_for_log(diar_cmd)!r}")
                self._proc = subprocess.Popen(
                    diar_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=env,
                )
                d_stdout, d_stderr = self._proc.communicate()
                d_diarization = time.monotonic() - t_step

                if self._stop_requested:
                    self.failed.emit("Annulé.")
                    return

                if self._proc.returncode != 0:
                    detail = (d_stderr or d_stdout or "").strip()
                    try:
                        parse_diarization_output(d_stdout)
                    except Exception as exc:
                        detail = str(exc).replace("Diarisation: ", "", 1)
                    self._log_process_result("diarization_failed", self._proc.returncode, d_stdout, d_stderr)
                    self._update_db_status("FAILED", f"Détection des locuteurs échouée: {detail}")
                    self.failed.emit(
                        f"Détection des locuteurs échouée (code {self._proc.returncode}). {detail}"
                    )
                    return
                self._log_process_result("diarization_ok", self._proc.returncode, d_stdout, d_stderr)

                try:
                    turns = parse_diarization_output(d_stdout)
                    fused = assign_speakers_to_segments(whisper_segments, turns)
                    analysis_segments = fused
                    rendered = render_segments_with_speakers(fused, self.output_format)
                except Exception as exc:
                    self._log(f"fusion_failed error={exc!r}")
                    self._update_db_status("FAILED", f"Fusion transcription/locuteurs échouée: {exc}")
                    self.failed.emit(f"Fusion transcription/locuteurs échouée: {exc}")
                    return

                out_path.write_text(rendered, encoding="utf-8")
                if self.job.db_id:
                    self.db.add_segments(self.job.db_id, fused)
            else:
                rendered = render_segments_plain(whisper_segments, self.output_format)
                out_path.write_text(rendered, encoding="utf-8")
                if self.job.db_id:
                    self.db.add_segments(self.job.db_id, whisper_segments)

            # The base transcript is now usable on disk. Surface it
            # immediately in the library so the user can open it while
            # the LLM enhancement keeps running.
            if self.job.db_id:
                self.db.update_job_artefact(self.job.db_id, "transcript", str(out_path))
                self.db.update_job_progress(
                    self.job.db_id,
                    step="Transcription disponible · amélioration en cours…",
                    progress_pct=90,
                )

            if self.job.db_id:
                self.db.update_job_durations(self.job.db_id, d_ffmpeg, d_whisper, d_diarization)

            # --- STEP 4: optional local LLM post-processing ---------------
            self._log(f"base_transcript_ready output={str(out_path)!r}")
            self.progress.emit(90)
            self.status.emit("Transcription disponible. Amélioration locale en cours…")
            analysis_path = work_dir / "transcript_for_local_analysis.txt"
            analysis_path.write_text(
                render_segments_with_speakers(analysis_segments, "txt"),
                encoding="utf-8",
            )
            payload = self._run_llm_post_process(analysis_path, self.initial_prompt)
            title = str(payload.get("title") or "").strip() if payload else ""
            enhanced_out_path = self._create_enhanced_transcript(out_path, payload, wav_path, work_dir)
            if enhanced_out_path:
                final_out_path = enhanced_out_path
                final_out_path = self._rename_transcript_file(enhanced_out_path, f"{title} améliorée")
                if self.job.db_id:
                    self.db.update_job_artefact(
                        self.job.db_id, "enhanced_transcript", str(final_out_path)
                    )
            else:
                final_out_path = self._rename_transcript_file(out_path, title)
                # The base transcript may have been renamed — update the
                # column so the "Ouvrir" button stays in sync.
                if self.job.db_id and str(final_out_path) != str(out_path):
                    self.db.update_job_artefact(
                        self.job.db_id, "transcript", str(final_out_path)
                    )

            # The .md "à vérifier" report only exists when the LLM had
            # something to flag, so we resolve the path lazily and
            # record it only when present.
            review_candidate = self._review_transcript_path(out_path)
            if review_candidate.exists() and self.job.db_id:
                self.db.update_job_artefact(
                    self.job.db_id, "review", str(review_candidate)
                )

            if self.job.db_id and title:
                self.db.update_job_title(self.job.db_id, title)

            self.duration_unknown.emit(False)
            self.progress.emit(100)
            self.status.emit("Transcription terminée.")
            self._update_db_status("COMPLETED")
            if self.job.db_id:
                self.db.update_job_output(self.job.db_id, str(final_out_path))
                self.db.update_job_progress(
                    self.job.db_id,
                    step="Terminé",
                    progress_pct=100,
                    eta_seconds=0,
                )
            self._log(f"success output={str(final_out_path)!r}")
            self.finished_ok.emit(str(final_out_path))
        except Exception as exc:
            self._log(f"unexpected_error error={exc!r}")
            self._update_db_status("FAILED", str(exc))
            self.failed.emit(str(exc))
        finally:
            self._proc = None
            if tmp_dir:
                shutil.rmtree(tmp_dir, ignore_errors=True)


class MlxWhisperInstallWorker(QThread):
    status = Signal(str)
    finished_ok = Signal(str)
    failed = Signal(str)

    def __init__(
        self,
        python_path: str,
        venv_dir: Path,
        install_diarization: bool = True,
        install_multimodal: bool = False,
    ):
        super().__init__()
        self.python_path = python_path
        self.venv_dir = venv_dir
        self.install_diarization = install_diarization
        # Multimodal audio (mlx-vlm + Qwen2-Audio) is heavier and rarely used,
        # so we keep it opt-in. The user can enable it later via Settings.
        self.install_multimodal = install_multimodal

    def _run_cmd(self, cmd: list[str], label: str):
        self.status.emit(label)
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip()
            raise RuntimeError(detail or f"Commande échouée: {' '.join(cmd)}")

    def run(self):
        try:
            self.venv_dir.parent.mkdir(parents=True, exist_ok=True)
            self._run_cmd(
                [self.python_path, "-m", "venv", str(self.venv_dir)],
                "Création de l'environnement local…",
            )

            venv_python = self.venv_dir / "bin" / "python"
            if sys.platform.startswith("win"):
                venv_python = self.venv_dir / "Scripts" / "python.exe"

            self._run_cmd(
                [str(venv_python), "-m", "pip", "install", "--upgrade", "pip"],
                "Préparation de pip…",
            )
            self._run_cmd(
                [str(venv_python), "-m", "pip", "install", "--upgrade", MLX_WHISPER_PACKAGE],
                "Installation de MLX Whisper…",
            )

            mlx_path = managed_mlx_whisper_path()
            if not is_executable(mlx_path):
                raise RuntimeError("Installation terminée, mais la commande mlx_whisper est introuvable.")

            # mlx-lm is the workhorse for the text post-processing step and is
            # cheap to install — bundle it with every fresh install so the
            # "améliorée" transcript works out of the box.
            self._run_cmd(
                [str(venv_python), "-m", "pip", "install", "--upgrade", *TEXT_LLM_PIP_PACKAGES],
                "Installation de l'IA texte locale (MLX LM, ~200 Mo)…",
            )

            if self.install_diarization:
                self._run_cmd(
                    [str(venv_python), "-m", "pip", "install", "--upgrade", *DIARIZATION_PIP_PACKAGES],
                    "Installation détection des locuteurs (~2 Go, peut prendre 5-10 min)…",
                )
                # Probe the import chain so the user discovers a broken install
                # here, not the first time they hit "Transcrire".
                probe = subprocess.run(
                    [str(venv_python), "-c", "import torch, pyannote.audio"],
                    capture_output=True,
                    text=True,
                )
                if probe.returncode != 0:
                    raise RuntimeError(
                        "Installation détection des locuteurs OK mais l'import échoue: "
                        + (probe.stderr or probe.stdout or "").strip()
                    )

            if self.install_multimodal:
                self._run_cmd(
                    [str(venv_python), "-m", "pip", "install", "--upgrade", *MULTIMODAL_LLM_PIP_PACKAGES],
                    "Installation IA multimodale (MLX VLM, ~500 Mo)…",
                )

            self.status.emit("MLX Whisper installé.")
            self.finished_ok.emit(str(mlx_path))
        except Exception as exc:
            self.failed.emit(str(exc))


class UpdateWorker(QThread):
    check_finished = Signal(object)
    download_progress = Signal(int)
    download_finished = Signal(str, object)
    failed = Signal(str)

    def __init__(
        self,
        mode: str,
        current_version: str = "",
        release_info: dict | None = None,
        github_token: str = "",
    ):
        super().__init__()
        self.mode = mode
        self.current_version = current_version
        self.release_info = release_info or {}
        self.github_token = github_token

    def run(self):
        try:
            if self.mode == "check":
                self._run_check()
            elif self.mode == "download":
                self._run_download()
            else:
                self.failed.emit("Mode de mise à jour invalide.")
        except Exception as exc:
            self.failed.emit(str(exc))

    def _run_check(self):
        try:
            context = ssl.create_default_context(cafile=certifi.where())
            headers = {
                "Accept": "application/vnd.github+json",
                "User-Agent": f"{APP_NAME}/{APP_VERSION}",
            }
            if self.github_token:
                headers["Authorization"] = f"Bearer {self.github_token}"
            request = urllib.request.Request(GITHUB_LATEST_RELEASE_API, headers=headers)
            with urllib.request.urlopen(request, timeout=20, context=context) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as http_err:
            if http_err.code == 404:
                raise RuntimeError(
                    "Release introuvable (404). "
                    "Si le dépôt est privé, renseignez un GitHub token dans Paramètres."
                ) from http_err
            raise RuntimeError(f"Erreur HTTP update: {http_err.code}") from http_err

        assets = payload.get("assets") or []
        chosen = choose_release_asset(assets)
        if not chosen:
            self.check_finished.emit({"state": "no_asset"})
            return

        info = {
            "tag_name": str(payload.get("tag_name", "")),
            "name": str(payload.get("name", "")),
            "html_url": str(payload.get("html_url", "")),
            "body": str(payload.get("body", "")),
            "asset_name": str(chosen.get("name", "")),
            "asset_url": str(chosen.get("browser_download_url", "")),
        }

        latest = parse_semver(info["tag_name"])
        current = parse_semver(self.current_version)

        if latest is None:
            self.check_finished.emit({"state": "unknown_remote", "info": info})
            return

        if current is None:
            self.check_finished.emit({"state": "available", "info": info, "current_unknown": True})
            return

        if latest > current:
            self.check_finished.emit({"state": "available", "info": info, "current_unknown": False})
            return

        self.check_finished.emit({"state": "up_to_date", "info": info})

    def _run_download(self):
        if not self.release_info.get("asset_url"):
            self.failed.emit("URL de téléchargement manquante.")
            return

        try:
            context = ssl.create_default_context(cafile=certifi.where())
            headers = {"User-Agent": f"{APP_NAME}/{APP_VERSION}"}
            if self.github_token:
                headers["Authorization"] = f"Bearer {self.github_token}"
            request = urllib.request.Request(self.release_info["asset_url"], headers=headers)
            with urllib.request.urlopen(request, timeout=60, context=context) as response:
                total = int(response.headers.get("Content-Length") or 0)
                fd, zip_path = tempfile.mkstemp(prefix="ekovideo-update-", suffix=".zip")
                os.close(fd)
                read_size = 0
                with open(zip_path, "wb") as out_file:
                    while True:
                        chunk = response.read(1024 * 128)
                        if not chunk:
                            break
                        out_file.write(chunk)
                        read_size += len(chunk)
                        if total > 0:
                            pct = int(min(100, (read_size / total) * 100))
                            self.download_progress.emit(pct)
        except urllib.error.HTTPError as http_err:
            if http_err.code == 404:
                raise RuntimeError(
                    "Asset introuvable (404). "
                    "Si le dépôt est privé, renseignez un GitHub token dans Paramètres."
                ) from http_err
            raise RuntimeError(f"Erreur HTTP téléchargement: {http_err.code}") from http_err

        self.download_progress.emit(100)
        self.download_finished.emit(zip_path, self.release_info)


class DropZone(QFrame):
    files_dropped = Signal(list)
    clicked = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setObjectName("dropZone")
        # Pinned compact height — the drop zone is a hint, not the main
        # canvas. Used to be a 4-line vertical block that hogged the
        # queue's vertical room; the queue list itself is the real
        # workspace.
        self.setFixedHeight(120)

        # Horizontal layout: icon left, two-line text centered next to it.
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(14)

        icon = QLabel("⌘")
        icon.setObjectName("dropIcon")
        icon.setAlignment(Qt.AlignCenter)
        icon.setFixedWidth(56)

        text_box = QVBoxLayout()
        text_box.setSpacing(2)

        title = QLabel("Déposez vos vidéos ou audios ici")
        title.setObjectName("dropTitle")
        title.setWordWrap(True)
        title.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)

        details = QLabel(
            "Vidéos (mp4, mov, mkv…) ou audios (mp3, m4a, wav…) · "
            "compression par lots, transcription locale en option"
        )
        details.setObjectName("dropDetails")
        details.setWordWrap(True)
        details.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)

        text_box.addStretch(1)
        text_box.addWidget(title)
        text_box.addWidget(details)
        text_box.addStretch(1)

        layout.addWidget(icon)
        layout.addLayout(text_box, 1)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if not urls:
            return
        paths = [u.toLocalFile() for u in urls if u.toLocalFile()]
        if paths:
            self.files_dropped.emit(paths)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
            event.accept()
            return
        super().mousePressEvent(event)


class CollapsibleSection(QFrame):
    """
    Header row + collapsible body. The header is a QToolButton showing a
    chevron that rotates when expanded. Used to fold per-job advanced
    overrides (resolution, bitrate, trim…) so the main view stays clean
    for the 95% case.
    """

    def __init__(self, title: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("collapsibleSection")
        self._toggle = QToolButton()
        self._toggle.setObjectName("collapsibleToggle")
        self._toggle.setText(title)
        self._toggle.setCheckable(True)
        self._toggle.setChecked(False)
        self._toggle.setArrowType(Qt.ArrowType.RightArrow)
        self._toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._toggle.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._toggle.toggled.connect(self._on_toggled)

        self._body = QFrame()
        self._body.setObjectName("collapsibleBody")
        self._body_layout = QVBoxLayout(self._body)
        self._body_layout.setContentsMargins(0, 8, 0, 0)
        self._body_layout.setSpacing(10)
        self._body.setVisible(False)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self._toggle)
        outer.addWidget(self._body)

    def add_widget(self, widget: QWidget):
        self._body_layout.addWidget(widget)

    def set_expanded(self, expanded: bool):
        self._toggle.setChecked(expanded)

    def _on_toggled(self, checked: bool):
        self._toggle.setArrowType(
            Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow
        )
        self._body.setVisible(checked)


def _open_path(path: str | Path) -> None:
    """Cross-platform "open this file in its default app"."""
    p = Path(path)
    if sys.platform == "darwin":
        subprocess.Popen(["open", str(p)])
    elif sys.platform.startswith("win"):
        os.startfile(str(p))
    else:
        subprocess.Popen(["xdg-open", str(p)])


def _reveal_in_finder(path: str | Path) -> None:
    p = Path(path)
    if sys.platform == "darwin":
        subprocess.Popen(["open", "-R", str(p)] if p.is_file() else ["open", str(p)])
    elif sys.platform.startswith("win"):
        if p.is_file():
            subprocess.Popen(["explorer", "/select,", str(p)])
        else:
            os.startfile(str(p))
    else:
        target = p if p.is_dir() else p.parent
        subprocess.Popen(["xdg-open", str(target)])


# Status enums used across DB / Library / workers. Kept loose (str) to
# stay backwards-compatible with the older PENDING/COMPLETED/FAILED
# values already stored in users' SQLite databases.
JOB_RUNNING_STATES = {"RUNNING", "AUDIO_READY", "WHISPER_DONE", "IMPORTING"}
JOB_DONE_STATES = {"COMPLETED"}
JOB_FAILED_STATES = {"FAILED", "CANCELLED"}


class LibraryView(QWidget):
    """
    The library is the canonical job dashboard: every job that's ever
    been queued is here, with one row per file and one column per
    artefact (compressed video, transcript, enhanced transcript, review
    report). Running jobs show a spinner + step + ETA; completed jobs
    expose their files via "Ouvrir" buttons that disable themselves
    when the artefact isn't (yet) available.
    """

    COL_STATUS = 0
    COL_FILE = 1
    COL_COMPRESSED = 2
    COL_TRANSCRIPT = 3
    COL_ENHANCED = 4
    COL_REVIEW = 5
    COL_ACTIONS = 6
    HEADERS = [
        "Statut",
        "Fichier",
        "Compressé",
        "Transcription",
        "Améliorée",
        "Rapport",
        "Actions",
    ]
    SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, db: DatabaseManager, parent: QWidget | None = None):
        super().__init__(parent)
        self.db = db
        self._job_ids: list[int] = []
        # Map db_id → row index so live status updates can repaint a
        # single row without the cost of a full rebuild.
        self._row_by_id: dict[int, int] = {}
        self._spinner_step = 0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 10, 0, 0)
        layout.setSpacing(8)

        search_box = QHBoxLayout()
        search_box.setSpacing(6)
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Rechercher un mot-clé ou un locuteur…")
        self.search_input.returnPressed.connect(self.refresh)
        btn_search = QPushButton("Chercher")
        btn_search.setObjectName("secondaryButton")
        btn_search.clicked.connect(self.refresh)
        search_box.addWidget(self.search_input, 1)
        search_box.addWidget(btn_search, 0)
        layout.addLayout(search_box)

        self.table = QTableWidget(0, len(self.HEADERS))
        self.table.setObjectName("libraryTable")
        self.table.setHorizontalHeaderLabels(self.HEADERS)
        self.table.verticalHeader().setVisible(False)
        # No row-selection visual: the artefact buttons + kebab menu
        # provide all the actions per-row, and a translucent selection
        # halo bleeding behind a "Ouvrir" button looked like a colour
        # bug. Double-click still works for the default "open the most
        # relevant artefact" behaviour.
        self.table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.table.setShowGrid(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setWordWrap(True)
        self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        header = self.table.horizontalHeader()
        header.setStretchLastSection(False)
        # We fit the columns ourselves on resize so the last column never
        # slips under the right border of the table.
        header.setSectionResizeMode(self.COL_STATUS, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(self.COL_FILE, QHeaderView.ResizeMode.Fixed)
        for col in (self.COL_COMPRESSED, self.COL_TRANSCRIPT, self.COL_ENHANCED, self.COL_REVIEW):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(self.COL_ACTIONS, QHeaderView.ResizeMode.Fixed)
        self.table.verticalHeader().setDefaultSectionSize(74)
        self.table.setColumnHidden(self.COL_FILE, False)
        # A minimum so the filename never disappears entirely on very
        # small windows; if we run out of horizontal room the table will
        # scroll horizontally instead of clipping.
        self.table.horizontalHeader().setMinimumSectionSize(80)
        self.table.itemDoubleClicked.connect(self._on_double_click)
        layout.addWidget(self.table, 1)
        QTimer.singleShot(0, self._fit_columns_to_viewport)

        # Spinner ticker — repaints just the running rows so we don't
        # rebuild the whole table every second.
        self._spinner_timer = QTimer(self)
        self._spinner_timer.setInterval(120)
        self._spinner_timer.timeout.connect(self._tick_spinners)

        self.refresh()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        QTimer.singleShot(0, self._fit_columns_to_viewport)

    def _fit_columns_to_viewport(self):
        viewport_width = max(0, self.table.viewport().width() - 8)
        if viewport_width <= 0:
            return

        status_w = 136
        action_w = 74
        artefact_w = 116
        fixed = status_w + action_w + (artefact_w * 4)
        file_w = viewport_width - fixed
        if file_w < 260:
            artefact_w = 108
            status_w = 126
            action_w = 68
            fixed = status_w + action_w + (artefact_w * 4)
            file_w = max(180, viewport_width - fixed)

        widths = {
            self.COL_STATUS: status_w,
            self.COL_FILE: file_w,
            self.COL_COMPRESSED: artefact_w,
            self.COL_TRANSCRIPT: artefact_w,
            self.COL_ENHANCED: artefact_w,
            self.COL_REVIEW: artefact_w,
            self.COL_ACTIONS: action_w,
        }
        for col, width in widths.items():
            self.table.setColumnWidth(col, max(1, width))

    # ------------------------------------------------------------------
    # Building the table
    # ------------------------------------------------------------------

    def refresh(self):
        query = self.search_input.text().strip()
        if query:
            jobs = self._search_jobs(query)
        else:
            jobs = self.db.list_jobs(limit=200)

        self._job_ids = [int(job["id"]) for job in jobs]
        self._row_by_id = {jid: i for i, jid in enumerate(self._job_ids)}

        self.table.setRowCount(0)
        self.table.setRowCount(len(jobs))
        for row, job in enumerate(jobs):
            self._fill_row(row, job)

        # Only run the spinner timer while at least one job is in flight.
        if any(self._is_running(j) for j in jobs):
            if not self._spinner_timer.isActive():
                self._spinner_timer.start()
        else:
            self._spinner_timer.stop()

    def _search_jobs(self, query: str) -> list[dict]:
        """
        Filter the full job listing by query. We match against:
        - the transcript segments (full-text), so the user can find
          a meeting by something said in it,
        - the source filename and the LLM-derived custom_title, so
          they can also find a meeting by its file or topic.

        Previously we only hit segments, which silently returned an
        empty list when the user searched by file name or speaker.
        """
        q = query.lower()

        # Bag of job ids that match the segment search.
        segment_results = self.db.search_segments(query_text=query)
        seg_hits = {int(s["job_id"]) for s in segment_results}

        # Walk the full job list once and keep anything that matches
        # via segments OR via file path / custom_title (case-insensitive).
        out: list[dict] = []
        for job in self.db.list_jobs(limit=500):
            jid = int(job["id"])
            if jid in seg_hits:
                out.append(job)
                continue
            haystacks = [
                str(job.get("source_path") or ""),
                str(job.get("custom_title") or ""),
                str(Path(job.get("source_path") or "").name),
            ]
            if any(q in h.lower() for h in haystacks if h):
                out.append(job)
        return out

    def _fill_row(self, row: int, job: dict):
        # --- Statut column ---------------------------------------------
        status_widget = self._build_status_cell(job)
        self.table.setCellWidget(row, self.COL_STATUS, status_widget)
        # The hidden item carries the job id so currentItem() works for
        # actions that don't need a widget.
        anchor = QTableWidgetItem("")
        anchor.setData(Qt.ItemDataRole.UserRole, int(job["id"]))
        self.table.setItem(row, self.COL_STATUS, anchor)

        # --- Source column --------------------------------------------
        title = job.get("custom_title") or Path(job["source_path"]).name
        file_item = QTableWidgetItem(title)
        file_item.setToolTip(str(job["source_path"]))
        file_item.setData(Qt.ItemDataRole.UserRole, int(job["id"]))
        file_item.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.table.setItem(row, self.COL_FILE, file_item)

        # --- Artefact columns -----------------------------------------
        running = self._is_running(job)
        jid = int(job["id"])
        self._set_artefact_cell(
            row, self.COL_COMPRESSED, job.get("compressed_path"), running, "Compressé", jid
        )
        self._set_artefact_cell(
            row, self.COL_TRANSCRIPT, job.get("transcript_path") or job.get("output_path"),
            running, "Transcription", jid
        )
        self._set_artefact_cell(
            row, self.COL_ENHANCED, job.get("enhanced_transcript_path"), running, "Améliorée", jid
        )
        self._set_artefact_cell(
            row, self.COL_REVIEW, job.get("review_path"), running, "Rapport", jid
        )

        # --- Actions kebab --------------------------------------------
        actions = self._build_actions_cell(jid)
        self.table.setCellWidget(row, self.COL_ACTIONS, actions)

    def _build_status_cell(self, job: dict) -> QWidget:
        running = self._is_running(job)
        completed = job.get("status") in JOB_DONE_STATES
        failed = job.get("status") in JOB_FAILED_STATES

        cell = QFrame()
        cell.setObjectName("libraryStatusCell")
        layout = QVBoxLayout(cell)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(2)

        if running:
            head = QLabel(f"{self.SPINNER_FRAMES[0]} En cours")
            head.setObjectName("libraryStatusRunning")
        elif completed:
            head = QLabel("✅ Terminé")
            head.setObjectName("libraryStatusDone")
        elif failed:
            head = QLabel("❌ Échec")
            head.setObjectName("libraryStatusFail")
        else:
            head = QLabel("⏳ En attente")
            head.setObjectName("libraryStatusPending")
        head.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(head)

        # Sub-line wraps inside the status column so a long step label
        # like "Compression… · reste ~10m" is readable instead of
        # silently truncated by the cell border.
        if running:
            step = (job.get("current_step") or "").strip() or "Préparation…"
            eta = self._format_eta(job.get("eta_seconds"))
            sub = QLabel(f"{step}{(' · ' + eta) if eta else ''}")
            sub.setObjectName("libraryStatusSub")
            sub.setWordWrap(True)
            sub.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
            sub.setToolTip(sub.text())
            layout.addWidget(sub)
        elif failed and job.get("error_message"):
            sub = QLabel(str(job["error_message"])[:120])
            sub.setObjectName("libraryStatusSub")
            sub.setWordWrap(True)
            sub.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
            sub.setToolTip(str(job["error_message"]))
            layout.addWidget(sub)

        return cell

    # Map of column → ("step kind", "human label").
    # The "step kind" is the value passed to the smart-relaunch popup
    # so we know which step the user wants to redo from this cell.
    _COL_TO_STEP = {
        2: ("compression", "Compression"),
        3: ("transcription", "Transcription"),
        4: ("enhanced", "Améliorée"),
        5: ("review", "Rapport"),
    }

    def _set_artefact_cell(
        self,
        row: int,
        col: int,
        path: str | None,
        running: bool,
        label: str,
        job_id: int,
    ):
        wrapper = QWidget()
        wrapper.setObjectName("artefactCell")
        wrapper.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        wrap_layout = QHBoxLayout(wrapper)
        wrap_layout.setContentsMargins(2, 4, 2, 4)
        wrap_layout.setSpacing(0)
        wrap_layout.addStretch(1)

        if path and Path(path).exists():
            mtime = self._format_mtime(path)
            btn = QPushButton("Ouvrir")
            btn.setObjectName("artefactButton")
            btn.setFixedWidth(70)
            btn.setToolTip(f"{path}\nProduit le {mtime}" if mtime else str(path))
            btn.clicked.connect(lambda _=False, p=str(path): _open_path(p))
            wrap_layout.addWidget(btn, 0)
            # Re-run icon. Single click triggers a one-step relaunch
            # popup pre-filled with this column's step ticked.
            kind = self._COL_TO_STEP.get(col, (None, None))[0]
            if kind:
                rerun = QToolButton()
                rerun.setObjectName("artefactReplay")
                rerun.setText("↻")
                rerun.setFixedWidth(30)
                rerun.setToolTip(f"Relancer cette étape ({label})")
                rerun.clicked.connect(
                    lambda _=False, jid=job_id, k=kind: self._action_restart(jid, preselect=[k])
                )
                wrap_layout.addWidget(rerun, 0)
        elif running:
            btn = QPushButton("⏳")
            btn.setObjectName("artefactButton")
            btn.setFixedWidth(100)
            btn.setToolTip(f"{label} en cours…")
            btn.setEnabled(False)
            wrap_layout.addWidget(btn, 0)
        else:
            btn = QPushButton("—")
            btn.setObjectName("artefactButton")
            btn.setFixedWidth(100)
            btn.setToolTip(f"Pas de {label.lower()} disponible")
            btn.setEnabled(False)
            wrap_layout.addWidget(btn, 0)

        wrap_layout.addStretch(1)

        self.table.setCellWidget(row, col, wrapper)

    @staticmethod
    def _format_mtime(path: str) -> str:
        try:
            ts = Path(path).stat().st_mtime
        except Exception:
            return ""
        return datetime.fromtimestamp(ts).strftime("%d/%m/%Y à %H:%M")

    def _build_actions_cell(self, job_id: int) -> QWidget:
        btn = QToolButton()
        btn.setObjectName("artefactButton")
        btn.setText("⋯")
        btn.setFixedWidth(46)
        btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QMenu(btn)
        menu.addAction("Ouvrir le dossier", lambda jid=job_id: self._action_show_folder(jid))
        menu.addAction("Contexte transcription…", lambda jid=job_id: self._action_edit_context(jid))
        menu.addAction("Renommer les locuteurs…", lambda jid=job_id: self._action_rename_speakers(jid))
        menu.addAction("Relancer / Réparer", lambda jid=job_id: self._action_restart(jid))
        menu.addSeparator()
        menu.addAction("Supprimer de la bibliothèque", lambda jid=job_id: self._action_delete(jid))
        btn.setMenu(menu)

        wrapper = QWidget()
        wrapper.setObjectName("artefactCell")
        wrapper.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        wrap_layout = QHBoxLayout(wrapper)
        wrap_layout.setContentsMargins(2, 4, 2, 4)
        wrap_layout.addStretch(1)
        wrap_layout.addWidget(btn, 0)
        wrap_layout.addStretch(1)
        return wrapper

    # ------------------------------------------------------------------
    # Live updates while a job is running
    # ------------------------------------------------------------------

    def tick_running_row(self, job_id: int):
        """
        Called by MainWindow when the current job's progress/step
        changes — repaints just the status cell instead of rebuilding
        the whole table.
        """
        row = self._row_by_id.get(int(job_id))
        if row is None:
            return
        job = self.db.get_job(int(job_id))
        if not job:
            return
        self.table.setCellWidget(row, self.COL_STATUS, self._build_status_cell(job))
        # Update artefact cells too — the transcript becomes available
        # mid-run, so the "Ouvrir" button must light up without waiting
        # for a full refresh.
        running = self._is_running(job)
        jid = int(job_id)
        self._set_artefact_cell(
            row, self.COL_COMPRESSED, job.get("compressed_path"), running, "Compressé", jid
        )
        self._set_artefact_cell(
            row, self.COL_TRANSCRIPT, job.get("transcript_path") or job.get("output_path"),
            running, "Transcription", jid
        )
        self._set_artefact_cell(
            row, self.COL_ENHANCED, job.get("enhanced_transcript_path"), running, "Améliorée", jid
        )
        self._set_artefact_cell(
            row, self.COL_REVIEW, job.get("review_path"), running, "Rapport", jid
        )
        if not self._spinner_timer.isActive() and running:
            self._spinner_timer.start()

    def _tick_spinners(self):
        self._spinner_step = (self._spinner_step + 1) % len(self.SPINNER_FRAMES)
        any_running = False
        for job_id, row in self._row_by_id.items():
            job = self.db.get_job(int(job_id))
            if not job or not self._is_running(job):
                continue
            any_running = True
            cell = self.table.cellWidget(row, self.COL_STATUS)
            if cell is None:
                continue
            head = cell.findChild(QLabel, "libraryStatusRunning")
            if head:
                head.setText(f"{self.SPINNER_FRAMES[self._spinner_step]} En cours")
        if not any_running:
            self._spinner_timer.stop()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_running(job: dict) -> bool:
        return job.get("status") in JOB_RUNNING_STATES

    @staticmethod
    def _format_eta(seconds) -> str:
        try:
            sec = float(seconds)
        except (TypeError, ValueError):
            return ""
        if sec <= 0 or sec != sec:  # NaN
            return ""
        return f"reste ~{format_compact_seconds(sec)}"

    def _on_double_click(self, item: QTableWidgetItem):
        # Default action when the user double-clicks a row: open the
        # most relevant artefact (enhanced transcript > transcript >
        # compressed > source folder).
        if item is None:
            return
        job_id = item.data(Qt.ItemDataRole.UserRole)
        if job_id is None:
            return
        job = self.db.get_job(int(job_id))
        if not job:
            return
        for key in ("enhanced_transcript_path", "transcript_path", "compressed_path", "output_path"):
            path = job.get(key)
            if path and Path(path).exists():
                _open_path(path)
                return
        QMessageBox.information(self, "Infos", "Aucun fichier disponible pour cette entrée.")

    # ------------------------------------------------------------------
    # Action handlers (used by the kebab menu)
    # ------------------------------------------------------------------

    def _reload_main_queue(self):
        main_win = self.window()
        if hasattr(main_win, "queue_jobs"):
            main_win.queue_jobs.clear()
            main_win.queue_list.clear()
            main_win.load_jobs_from_db()

    def _action_restart(self, job_id: int, preselect: list[str] | None = None):
        """
        Open the smart-relaunch popup: the user picks which steps to
        redo (compression / transcription / améliorée / rapport). On
        confirm we ask the MainWindow to restart the job — and we drive
        the worker ourselves so it actually runs even when nothing
        else is currently in the queue.
        """
        job = self.db.get_job(job_id)
        if not job:
            return
        dlg = RelaunchStepsDialog(job, preselect=preselect or [], parent=self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        steps = dlg.selected_steps()
        if not steps:
            return

        main_win = self.window()
        if hasattr(main_win, "relaunch_job_with_steps"):
            main_win.relaunch_job_with_steps(job_id, steps)
        else:
            # Defensive fallback: keep the previous behaviour if we're
            # somehow detached from the main window.
            self.db.update_job_status(job_id, "PENDING", "")
            self.refresh()

    def _action_delete(self, job_id: int):
        confirm = QMessageBox.question(
            self,
            "Confirmer",
            "Supprimer cette entrée de la bibliothèque ? Les fichiers de sortie ne seront pas effacés.",
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        job = self.db.get_job(job_id)
        if job and job.get("workspace_dir") and Path(job["workspace_dir"]).exists():
            try:
                shutil.rmtree(job["workspace_dir"])
            except Exception:
                pass
        self.db.delete_job(job_id)
        self.refresh()
        self._reload_main_queue()

    def _action_show_folder(self, job_id: int):
        job = self.db.get_job(job_id)
        if not job:
            return
        for key in ("transcript_path", "compressed_path", "output_path", "source_path"):
            path = job.get(key)
            if path and Path(path).exists():
                _reveal_in_finder(path)
                return
            if path and Path(path).parent.exists():
                _reveal_in_finder(Path(path).parent)
                return
        QMessageBox.information(self, "Infos", "Pas de dossier disponible pour cette entrée.")

    def _action_rename_speakers(self, job_id: int):
        dlg = SpeakerRenameDialog(self.db, job_id, parent=self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        mapping = dlg.values()
        if not mapping:
            return

        record = self.db.get_job(job_id) or {}
        previous_speakers: dict[str, str] = {}
        try:
            loaded = json.loads(record.get("speaker_map_json") or "{}")
            if isinstance(loaded, dict):
                previous_speakers = {
                    str(key).strip(): str(value).strip()
                    for key, value in loaded.items()
                    if str(key).strip() and str(value).strip()
                }
        except Exception:
            previous_speakers = {}

        replacement_map = dict(mapping)
        stored_speakers = dict(previous_speakers)
        for old_label, new_name in mapping.items():
            if old_label.startswith("SPEAKER_"):
                stored_speakers[old_label] = new_name
                continue
            matched_existing_id = False
            for speaker_id, current_name in previous_speakers.items():
                if current_name == old_label:
                    replacement_map[speaker_id] = new_name
                    stored_speakers[speaker_id] = new_name
                    matched_existing_id = True
            if not matched_existing_id:
                stored_speakers[old_label] = new_name

        if stored_speakers:
            try:
                self.db.update_job_context(job_id, stored_speakers, None)
            except Exception:
                pass

        applied_files, applied_segments = self._apply_speaker_rename(job_id, replacement_map)
        self.refresh()
        QMessageBox.information(
            self,
            "Renommage appliqué",
            (
                f"{len(mapping)} locuteur(s) renommé(s).\n"
                f"Fichiers mis à jour : {applied_files}\n"
                f"Segments DB mis à jour : {applied_segments}"
            ),
        )

    def _action_edit_context(self, job_id: int):
        dlg = TranscriptContextDialog(self.db, job_id, parent=self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        record = self.db.get_job(job_id) or {}
        previous_speakers: dict[str, str] = {}
        try:
            loaded = json.loads(record.get("speaker_map_json") or "{}")
            if isinstance(loaded, dict):
                previous_speakers = {
                    str(key).strip(): str(value).strip()
                    for key, value in loaded.items()
                    if str(key).strip() and str(value).strip()
                }
        except Exception:
            previous_speakers = {}

        speakers, terms = dlg.values()
        try:
            self.db.update_job_context(job_id, speakers, terms)
        except Exception as exc:
            QMessageBox.warning(self, "Contexte transcription", f"Enregistrement impossible: {exc}")
            return

        replacement_map = dict(speakers)
        for speaker_id, previous_name in previous_speakers.items():
            new_name = speakers.get(speaker_id, "")
            if new_name and previous_name and previous_name != new_name:
                replacement_map[previous_name] = new_name

        applied_files, applied_segments = self._apply_speaker_rename(job_id, replacement_map)
        self.refresh()
        QMessageBox.information(
            self,
            "Contexte enregistré",
            (
                f"{len(speakers)} interlocuteur(s) et {len(terms)} terme(s) enregistrés.\n"
                f"Fichiers mis à jour : {applied_files}\n"
                f"Segments DB mis à jour : {applied_segments}"
            ),
        )

    def _apply_speaker_rename(self, job_id: int, mapping: dict[str, str]) -> tuple[int, int]:
        """
        Apply the SPEAKER_XX → real-name mapping to every artefact on
        disk and to the DB segments. Returns (files_touched, segments_touched).
        """
        record = self.db.get_job(job_id) or {}

        def _rewrite(text: str) -> str:
            for old, new in mapping.items():
                # Bracketed form (most common in our render).
                text = text.replace(f"[{old}]", f"[{new}]")
                # Bare token, only when surrounded by word boundaries
                # so we don't accidentally chew a substring of something.
                text = re.sub(rf"\b{re.escape(old)}\b", new, text)
            return text

        files_touched = 0
        for key in ("transcript_path", "enhanced_transcript_path", "review_path"):
            path = record.get(key)
            if not path or not Path(path).exists():
                continue
            try:
                p = Path(path)
                old = p.read_text(encoding="utf-8", errors="ignore")
                new_text = _rewrite(old)
                if new_text != old:
                    p.write_text(new_text, encoding="utf-8")
                    files_touched += 1
            except Exception:
                continue

        # Update DB segments so the search index also reflects the
        # renames. We load, rewrite, and re-insert via add_segments
        # which already handles the "delete + bulk insert" pattern.
        segments = self.db.get_segments(job_id) or []
        seg_changed = 0
        if segments:
            updated = []
            for seg in segments:
                seg_copy = dict(seg)
                speaker = (seg_copy.get("speaker") or "").strip()
                if speaker in mapping:
                    seg_copy["speaker"] = mapping[speaker]
                    seg_changed += 1
                # `add_segments` expects start/end/text/speaker keys.
                seg_copy["start"] = seg_copy.get("start_time", seg_copy.get("start", 0.0))
                seg_copy["end"] = seg_copy.get("end_time", seg_copy.get("end", 0.0))
                updated.append(seg_copy)
            if seg_changed:
                self.db.add_segments(job_id, updated)
        return files_touched, seg_changed

class SpeakerRenameDialog(QDialog):
    """
    Lets the user replace SPEAKER_XX placeholders by real names.
    Loads existing names from the DB, shows one editable field per
    detected speaker, and on save rewrites every artefact + the DB
    segments so the new names propagate everywhere.
    """

    SPEAKER_RE = re.compile(r"SPEAKER_\d{2,}")
    BRACKETED_LABEL_RE = re.compile(r"(?m)^\[([^\]\n]{1,80})\]\s")

    def __init__(self, db, job_id: int, parent: QWidget | None = None):
        super().__init__(parent)
        self.db = db
        self.job_id = job_id
        self.setWindowTitle("Renommer les locuteurs")
        self.setModal(True)
        self.setMinimumWidth(480)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 16)
        layout.setSpacing(12)

        head = QLabel(
            "Indiquez le prénom (ou nom) de chaque interlocuteur. "
            "Les remplacements seront appliqués à toute la transcription, "
            "à la version améliorée et au rapport."
        )
        head.setWordWrap(True)
        layout.addWidget(head)

        speakers = self._detect_speakers(job_id)
        self._fields: dict[str, QLineEdit] = {}

        if not speakers:
            empty = QLabel(
                "Aucun locuteur détecté pour cette transcription. "
                "Active la diarisation dans Paramètres pour étiqueter les voix."
            )
            empty.setObjectName("inlineHint")
            empty.setWordWrap(True)
            layout.addWidget(empty)
        else:
            form = QFormLayout()
            form.setHorizontalSpacing(10)
            form.setVerticalSpacing(10)
            form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
            for label, current in sorted(speakers.items()):
                edit = QLineEdit(current)
                edit.setPlaceholderText("Prénom (laisser vide pour ne pas renommer)")
                form.addRow(label, edit)
                self._fields[label] = edit
            layout.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Appliquer")
        buttons.button(QDialogButtonBox.Ok).setEnabled(bool(speakers))
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _detect_speakers(self, job_id: int) -> dict[str, str]:
        """
        Prefer the speaker labels visible in the enhanced transcript.
        The DB segments are the first-pass diarization labels, so they
        are only a fallback when no produced artefact has speaker tags.
        """
        record = self.db.get_job(job_id) or {}
        stored = self._stored_speaker_map(record)

        for key in ("enhanced_transcript_path", "review_path", "transcript_path"):
            labels = self._speaker_labels_from_file(record.get(key))
            if labels:
                return self._names_for_labels(labels, stored)

        seen_speakers: list[str] = []
        for seg in self.db.get_segments(job_id) or []:
            sp = (seg.get("speaker") or "").strip()
            if sp and sp not in seen_speakers:
                seen_speakers.append(sp)
        return self._names_for_labels(seen_speakers, stored)

    def _stored_speaker_map(self, record: dict) -> dict[str, str]:
        try:
            stored = json.loads(record.get("speaker_map_json") or "{}")
            if isinstance(stored, dict):
                return {
                    str(key).strip(): str(value).strip()
                    for key, value in stored.items()
                    if str(key).strip() and str(value).strip()
                }
        except Exception:
            pass
        return {}

    def _speaker_labels_from_file(self, path: str | None) -> list[str]:
        if not path:
            return []
        try:
            text = Path(path).read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return []
        labels: list[str] = []
        for raw in self.BRACKETED_LABEL_RE.findall(text):
            label = raw.strip()
            if label and label != "?" and label not in labels:
                labels.append(label)
        if labels:
            return labels
        for raw in self.SPEAKER_RE.findall(text):
            label = raw.strip()
            if label and label not in labels:
                labels.append(label)
        return labels

    def _names_for_labels(self, labels: list[str], stored: dict[str, str]) -> dict[str, str]:
        names: dict[str, str] = {}
        for label in labels:
            if label.startswith("SPEAKER_"):
                names[label] = stored.get(label, "")
            else:
                names[label] = stored.get(label, label)
        return names

    def values(self) -> dict[str, str]:
        values: dict[str, str] = {}
        for label, edit in self._fields.items():
            value = edit.text().strip()
            if not value:
                continue
            if not label.startswith("SPEAKER_") and value == label:
                continue
            values[label] = value
        return values


class TranscriptContextDialog(QDialog):
    SPEAKER_RE = re.compile(r"SPEAKER_\d{2,}")
    BRACKETED_LABEL_RE = re.compile(r"(?m)^\[([^\]\n]{1,80})\]\s")

    def __init__(self, db, job_id: int, parent: QWidget | None = None):
        super().__init__(parent)
        self.db = db
        self.job_id = job_id
        self.setWindowTitle("Contexte transcription")
        self.setModal(True)
        self.setMinimumWidth(620)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 16)
        layout.setSpacing(12)

        head = QLabel(
            "Vérifiez les interlocuteurs et les termes techniques relevés. "
            "Ces éléments servent à corriger la transcription et à mieux orienter les relances."
        )
        head.setWordWrap(True)
        layout.addWidget(head)

        speakers = self._detect_speakers()
        self._speaker_fields: dict[str, QLineEdit] = {}
        speaker_group = QGroupBox("Interlocuteurs")
        speaker_form = QFormLayout(speaker_group)
        speaker_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        speaker_form.setHorizontalSpacing(10)
        speaker_form.setVerticalSpacing(10)
        if speakers:
            for label, current in sorted(speakers.items()):
                edit = QLineEdit(current)
                edit.setPlaceholderText("Nom ou prénom")
                speaker_form.addRow(label, edit)
                self._speaker_fields[label] = edit
        else:
            empty = QLabel("Aucun locuteur détecté pour l'instant.")
            empty.setObjectName("inlineHint")
            empty.setWordWrap(True)
            speaker_form.addRow(empty)
        layout.addWidget(speaker_group)

        terms_group = QGroupBox("Termes techniques")
        terms_layout = QVBoxLayout(terms_group)
        terms_layout.setContentsMargins(12, 16, 12, 12)
        self.terms_edit = QTextEdit()
        self.terms_edit.setAcceptRichText(False)
        self.terms_edit.setPlaceholderText("Un terme par ligne : Odoo, Infomaniak, Chat GPT…")
        self.terms_edit.setMinimumHeight(130)
        self.terms_edit.setPlainText("\n".join(self._detect_terms()))
        terms_layout.addWidget(self.terms_edit)
        layout.addWidget(terms_group)

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _record(self) -> dict:
        return self.db.get_job(self.job_id) or {}

    def _detect_speakers(self) -> dict[str, str]:
        record = self._record()
        stored = self._stored_speaker_map(record)

        for key in ("enhanced_transcript_path", "review_path", "transcript_path"):
            labels = self._speaker_labels_from_file(record.get(key))
            if labels:
                return self._names_for_labels(labels, stored)

        names: dict[str, str] = {}
        for seg in self.db.get_segments(self.job_id) or []:
            speaker = str(seg.get("speaker") or "").strip()
            if speaker.startswith("SPEAKER_"):
                names.setdefault(speaker, stored.get(speaker, ""))
        return names

    def _stored_speaker_map(self, record: dict) -> dict[str, str]:
        try:
            stored = json.loads(record.get("speaker_map_json") or "{}")
            if isinstance(stored, dict):
                return {
                    str(key).strip(): str(value).strip()
                    for key, value in stored.items()
                    if str(key).strip() and str(value).strip()
                }
        except Exception:
            pass
        return {}

    def _speaker_labels_from_file(self, path: str | None) -> list[str]:
        if not path:
            return []
        try:
            text = Path(path).read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return []
        labels: list[str] = []
        for raw in self.BRACKETED_LABEL_RE.findall(text):
            label = raw.strip()
            if label and label != "?" and label not in labels:
                labels.append(label)
        if labels:
            return labels
        for raw in self.SPEAKER_RE.findall(text):
            label = raw.strip()
            if label and label not in labels:
                labels.append(label)
        return labels

    def _names_for_labels(self, labels: list[str], stored: dict[str, str]) -> dict[str, str]:
        names: dict[str, str] = {}
        for label in labels:
            if label.startswith("SPEAKER_"):
                names[label] = stored.get(label, "")
            else:
                names[label] = stored.get(label, label)
        return names

    def _detect_terms(self) -> list[str]:
        record = self._record()
        terms: list[str] = []

        def add(term: str):
            clean = term.strip().strip("-•` ")
            if clean and clean not in terms:
                terms.append(clean)

        try:
            stored = json.loads(record.get("technical_terms_json") or "[]")
            if isinstance(stored, list):
                for term in stored:
                    add(str(term))
        except Exception:
            pass

        review_path = record.get("review_path")
        if review_path and Path(review_path).exists():
            try:
                lines = Path(review_path).read_text(encoding="utf-8", errors="ignore").splitlines()
            except Exception:
                lines = []
            in_terms = False
            for line in lines:
                stripped = line.strip()
                if stripped.startswith("## "):
                    in_terms = False
                if stripped == "Termes techniques :":
                    in_terms = True
                    continue
                if in_terms and stripped.startswith("- "):
                    add(stripped[2:])
        return terms

    def values(self) -> tuple[dict[str, str], list[str]]:
        speakers = {
            label: edit.text().strip()
            for label, edit in self._speaker_fields.items()
            if edit.text().strip()
        }
        terms: list[str] = []
        for line in self.terms_edit.toPlainText().splitlines():
            term = line.strip().strip("-•")
            if term and term not in terms:
                terms.append(term)
        return speakers, terms


class RelaunchStepsDialog(QDialog):
    """
    "Relancer / Réparer" popup: lets the user pick exactly which steps
    to redo for an existing job, with each option pre-checked when the
    artefact is missing (so the obvious "just fix the bits that didn't
    finish" case is one click away).
    """

    # (key, label, helper text, predicate(job) -> True if artefact exists)
    STEPS = [
        ("compression", "Compression", "Re-encoder la vidéo / l'audio source.", "compressed_path"),
        ("transcription", "Transcription Whisper",
         "Re-transcrire le média (passe Whisper + diarisation).", "transcript_path"),
        ("enhanced", "Transcription améliorée (LLM)",
         "Re-faire l'analyse titre + interlocuteurs + corrections.", "enhanced_transcript_path"),
        ("review", "Réécoute IA multimodale",
         "Re-lancer la vérification audio des passages douteux.", "review_path"),
    ]

    def __init__(self, job: dict, preselect: list[str], parent: QWidget | None = None):
        super().__init__(parent)
        title = job.get("custom_title") or Path(job.get("source_path") or "").name or "ce fichier"
        self.setWindowTitle("Relancer / Réparer")
        self.setModal(True)
        self.setMinimumWidth(520)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 16)
        layout.setSpacing(12)

        head = QLabel(f"Quelles étapes voulez-vous relancer pour <b>{title}</b> ?")
        head.setWordWrap(True)
        layout.addWidget(head)

        sub = QLabel(
            "Les étapes manquantes sont cochées par défaut. "
            "Une étape relancée écrase son fichier de sortie."
        )
        sub.setObjectName("inlineHint")
        sub.setWordWrap(True)
        layout.addWidget(sub)

        self._checks: dict[str, QCheckBox] = {}
        preselect_set = set(preselect or [])
        for key, label, helper, artefact_col in self.STEPS:
            row = QFrame()
            row_layout = QVBoxLayout(row)
            row_layout.setContentsMargins(0, 4, 0, 4)
            row_layout.setSpacing(2)
            cb = QCheckBox(label)
            already = bool(job.get(artefact_col))
            should_check = (key in preselect_set) or (not already)
            cb.setChecked(should_check)
            row_layout.addWidget(cb)
            note_text = helper
            if already:
                note_text += "  ·  déjà produit"
            note = QLabel(note_text)
            note.setObjectName("inlineHint")
            note.setIndent(24)
            note.setWordWrap(True)
            row_layout.addWidget(note)
            self._checks[key] = cb
            layout.addWidget(row)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Relancer")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def selected_steps(self) -> list[str]:
        return [key for key, cb in self._checks.items() if cb.isChecked()]


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        # Below ~1140 the right-side settings column gets squashed and
        # the splitter starts cropping group titles. 1180×760 fits a
        # 13" MacBook screen while leaving room for the library table.
        self.setMinimumSize(1180, 760)

        self.settings = QSettings(ORG_NAME, APP_NAME)
        
        # Initialize Database and Workspace
        app_data = app_support_dir()
        app_data.mkdir(parents=True, exist_ok=True)
        self.db = DatabaseManager(app_data / "library.db")
        self.workspace_root = app_data / "Workspace"
        self.workspace_root.mkdir(parents=True, exist_ok=True)

        icon_path = resource_path(APP_ICON_FILE)
        if Path(icon_path).exists():
            self.setWindowIcon(QIcon(icon_path))

        self.ffmpeg_path = self.settings.value("ffmpeg_path", "", type=str).strip() or find_binary("ffmpeg") or ""
        self.ffprobe_path = self.settings.value("ffprobe_path", "", type=str).strip() or find_binary("ffprobe") or ""
        self.github_token = (
            self.settings.value("github_token", "", type=str).strip()
            or os.getenv("EKO_UPDATER_GITHUB_TOKEN", "").strip()
        )
        self.mlx_whisper_path = (
            self.settings.value("mlx_whisper_path", "", type=str).strip()
            or find_binary("mlx_whisper")
            or ""
        )
        self.transcription_model = canonical_whisper_model_id(
            str(self.settings.value("transcription_model", DEFAULT_WHISPER_MODEL, type=str))
        )
        # The legacy "transcription_llm_model" key was a single field used for
        # both text and audio analyses, even though those need different
        # families. We split it into two and migrate any old value into the
        # text slot — the audio one always starts on its sane default.
        legacy_llm_model = str(self.settings.value("transcription_llm_model", "", type=str)).strip()
        self.transcription_text_llm_model = str(
            self.settings.value("transcription_text_llm_model", "", type=str)
        ).strip() or legacy_llm_model or DEFAULT_TEXT_LLM_MODEL
        self.transcription_audio_llm_model = canonical_audio_llm_model_id(
            str(self.settings.value("transcription_audio_llm_model", DEFAULT_AUDIO_LLM_MODEL, type=str))
        )
        self.transcription_audio_recheck_enabled = self.settings.value(
            "transcription_audio_recheck_enabled", False, type=bool
        )
        self.transcription_language = str(self.settings.value("transcription_language", "fr", type=str)).strip() or "fr"
        self.transcription_format = str(self.settings.value("transcription_format", "txt", type=str)).strip() or "txt"
        self.transcription_suffix = str(self.settings.value("transcription_suffix", "", type=str)).strip()
        if self.transcription_suffix == "_transcription":
            self.transcription_suffix = ""
        self.transcription_enhance_audio = self.settings.value("transcription_enhance_audio", True, type=bool)
        # Diarisation (séparation des locuteurs) — opt-in, requires HF token.
        self.transcription_diarization_enabled = self.settings.value(
            "transcription_diarization_enabled", False, type=bool
        )
        self.transcription_hf_token = (
            self.settings.value("transcription_hf_token", "", type=str).strip()
            or os.getenv("HF_TOKEN", "").strip()
        )

        self.queue_jobs: list[QueueJob] = []
        self.pending_indices: list[int] = []
        self.completed_count = 0
        self.failed_count = 0
        self.running_index: int | None = None
        self.current_index: int = -1
        self.worker: EncodeWorker | TranscribeWorker | None = None
        self._active_workers: list[QThread] = []
        self.mlx_install_worker: MlxWhisperInstallWorker | None = None
        self.batch_mode = "compress"
        self.job_run_modes: dict[int, str] = {}
        self.current_job_mode = "compress"
        self.current_job_progress_offset = 0
        self.current_job_progress_scale = 100
        self.current_job_started_at: float | None = None
        self.current_job_status_text = ""
        self.current_job_display_pct = 0
        self.update_worker: UpdateWorker | None = None
        self._last_update_info: dict | None = None
        self._update_phase: str = ""
        self._update_check_payload: dict | None = None
        self._update_download_payload: tuple[str, dict] | None = None
        self._update_error_message: str | None = None
        self.is_batch_running = False
        self._syncing_job = False

        self.status_timer = QTimer(self)
        self.status_timer.timeout.connect(self._set_status_with_progress_context)
        self.status_timer.start(1000)

        self._build_ui(icon_path)
        self.apply_style()
        self.load_jobs_from_db()

        if not self.ffmpeg_path:
            self.status.setText("ffmpeg non détecté. Ouvrez Paramètres (⚙).")
        if not self.transcription_hf_token and not self.settings.value(
            "hf_onboarding_seen", False, type=bool
        ):
            QTimer.singleShot(800, self._maybe_prompt_hf_onboarding)

    def _maybe_prompt_hf_onboarding(self):
        if self.transcription_hf_token:
            return
        self.settings.setValue("hf_onboarding_seen", True)
        answer = QMessageBox.question(
            self,
            "Connexion Hugging Face",
            (
                "La détection des locuteurs utilise des modèles Hugging Face qui "
                "demandent une connexion et l'acceptation de conditions.\n\n"
                "Voulez-vous configurer Hugging Face maintenant ?"
            ),
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        dlg = HuggingFaceAuthDialog("", self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self.transcription_hf_token = dlg.token()
            self.settings.setValue("transcription_hf_token", self.transcription_hf_token)
            self._refresh_transcription_summary()

    def _build_ui(self, icon_path: str):
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 18)
        root.setSpacing(14)

        header = QFrame()
        header.setObjectName("header")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(18, 14, 18, 14)
        header_layout.setSpacing(14)

        logo = QLabel()
        logo.setObjectName("logoMark")
        if Path(icon_path).exists():
            logo.setPixmap(QIcon(icon_path).pixmap(38, 38))
        else:
            logo.setText("∞")
            logo.setAlignment(Qt.AlignmentFlag.AlignCenter)

        title_box = QVBoxLayout()
        title_box.setSpacing(1)
        self.h1 = QLabel(APP_NAME)
        self.h1.setObjectName("h1")
        # Avoid "vdev" when APP_VERSION isn't a real semver; only prefix
        # "v" when the value is a numeric version string.
        version_label = (
            f"v{APP_VERSION}" if APP_VERSION and APP_VERSION[:1].isdigit() else APP_VERSION
        )
        self.h2 = QLabel(f"Compression et transcription locale · macOS · {version_label}")
        self.h2.setObjectName("h2")
        title_box.addWidget(self.h1)
        title_box.addWidget(self.h2)

        self.btn_settings = QToolButton()
        self.btn_settings.setObjectName("gear")
        self.btn_settings.setText("Réglages")
        self.btn_settings.clicked.connect(self.open_settings)
        self.btn_update = QPushButton("Mise à jour")
        self.btn_update.setObjectName("secondaryButton")
        self.btn_update.clicked.connect(self.check_updates)

        header_layout.addWidget(logo)
        header_layout.addLayout(title_box, 1)
        header_layout.addWidget(self.btn_update)
        header_layout.addWidget(self.btn_settings)
        root.addWidget(header)

        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter = self.main_splitter
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(8)

        self.left_panel = QWidget()
        left_panel = self.left_panel
        left_panel.setMinimumWidth(0)
        left_panel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        left_col = QVBoxLayout(left_panel)
        left_col.setContentsMargins(0, 0, 0, 0)
        left_col.setSpacing(10)

        self.tabs = QTabWidget()
        self.tabs.setObjectName("mainTabs")
        self.tabs.setMinimumWidth(0)
        self.tabs.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        # TAB 1: Queue
        self.queue_tab = QWidget()
        queue_tab = self.queue_tab
        queue_tab.setMinimumWidth(0)
        queue_layout = QVBoxLayout(queue_tab)
        queue_layout.setContentsMargins(0, 10, 0, 0)
        queue_layout.setSpacing(10)

        self.drop = DropZone()
        self.drop.files_dropped.connect(self.add_input_files)
        self.drop.clicked.connect(self.pick_files)
        # Stretch 0 — the queue list (added below) gets all the extra
        # vertical room; the drop zone keeps its fixed 120 px.
        queue_layout.addWidget(self.drop, 0)

        queue_actions = QHBoxLayout()
        queue_actions.setSpacing(8)
        self.btn_pick = QPushButton("Ajouter des fichiers…")
        self.btn_pick.setObjectName("primaryButton")
        self.btn_pick.clicked.connect(self.pick_files)
        self.btn_remove = QPushButton("Retirer")
        self.btn_remove.setObjectName("secondaryButton")
        self.btn_remove.clicked.connect(self.remove_selected_files)
        self.btn_clear = QPushButton("Vider")
        self.btn_clear.setObjectName("secondaryButton")
        self.btn_clear.clicked.connect(self.clear_queue)
        queue_actions.addWidget(self.btn_pick, 1)
        queue_actions.addWidget(self.btn_remove, 0)
        queue_actions.addWidget(self.btn_clear, 0)
        queue_layout.addLayout(queue_actions)

        self.queue_list = QListWidget()
        self.queue_list.setObjectName("queueList")
        self.queue_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        # Internal-move drag-and-drop so the user can reorder pending
        # files before starting a batch. We keep the underlying
        # queue_jobs list in sync via the rowsMoved signal below.
        self.queue_list.setDragDropMode(QListWidget.DragDropMode.InternalMove)
        self.queue_list.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.queue_list.setMovement(QListWidget.Movement.Snap)
        self.queue_list.model().rowsMoved.connect(self._on_queue_rows_moved)
        self.queue_list.currentRowChanged.connect(self.on_queue_selection_changed)
        queue_layout.addWidget(self.queue_list, 1)
        
        self.tabs.addTab(queue_tab, "File d'attente")
        
        # TAB 2: Library
        self.library_view = LibraryView(self.db)
        self.tabs.addTab(self.library_view, "Bibliothèque")
        
        left_col.addWidget(self.tabs, 1)

        self.lbl_input_meta = QLabel("Aucun fichier sélectionné.")
        self.lbl_input_meta.setObjectName("metaLabel")
        self.lbl_input_meta.setWordWrap(True)
        left_col.addWidget(self.lbl_input_meta)

        # Right panel: single scrollable column instead of 4 tabs.
        # Per-job essentials are visible by default; technical overrides
        # (resolution, bitrate, trim) are folded into a collapsible section
        # so the team's day-to-day flow stays uncluttered.
        self.right_panel = QFrame()
        right_col = self.right_panel
        right_col.setObjectName("settingsPanel")
        # 380 guarantees the longest checkbox label, the dropdown
        # chevron and the QGroupBox titles all fit without cropping.
        # 520 is a soft cap so on a 27" screen the right panel doesn't
        # become wider than the queue/library tabs.
        right_col.setMinimumWidth(380)
        right_col.setMaximumWidth(520)
        right_col.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        right_layout = QVBoxLayout(right_col)
        right_layout.setContentsMargins(12, 12, 12, 12)
        right_layout.setSpacing(10)

        self.settings_scroll = QScrollArea()
        scroll = self.settings_scroll
        scroll.setObjectName("settingsScroll")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.settings_scroll_body = QFrame()
        scroll_body = self.settings_scroll_body
        scroll_body.setObjectName("settingsScrollBody")
        scroll_layout = QVBoxLayout(scroll_body)
        scroll_layout.setContentsMargins(4, 4, 4, 4)
        scroll_layout.setSpacing(12)

        # --- Profil + sortie -----------------------------------------
        self.main_settings_group = QGroupBox("Réglages de cette vidéo")
        main_group = self.main_settings_group
        main_form = QFormLayout(main_group)
        main_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        main_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        main_form.setHorizontalSpacing(10)
        main_form.setVerticalSpacing(12)

        self.combo_profile = QComboBox()
        self.combo_profile.addItems(["Personnalisé", *PROFILE_PRESETS.keys()])
        self.combo_profile.currentTextChanged.connect(self.on_profile_changed)
        main_form.addRow("Profil", self.combo_profile)

        out_row = QHBoxLayout()
        self.edit_output_dir = QLineEdit(str(self.settings.value("output_dir", str(Path.home() / "Desktop"), type=str)))
        self.btn_output_dir = QPushButton("…")
        self.btn_output_dir.setObjectName("secondaryButton")
        self.btn_output_dir.clicked.connect(self.pick_output_dir)
        out_row.addWidget(self.edit_output_dir)
        out_row.addWidget(self.btn_output_dir)
        main_form.addRow("Dossier sortie", out_row)

        # Kept as an internal setting for existing users, but hidden from the
        # main flow: the default output naming is enough for day-to-day use.
        self.edit_suffix = QLineEdit(str(self.settings.value("suffix", "_compressed", type=str)))
        self.edit_suffix.setVisible(False)

        # The checkboxes used to ride the QFormLayout's field column,
        # which reserved the leftmost ~150px for the (empty) label and
        # cropped the longer label "Continuer en cas d'erreur" to "…d'erreu".
        # We span both columns by using the single-widget overload.
        self.check_continue_on_error = QCheckBox("Continuer en cas d'erreur")
        self.check_continue_on_error.setChecked(self.settings.value("continue_on_error", True, type=bool))
        main_form.addRow(self.check_continue_on_error)

        self.check_transcribe_after_encode = QCheckBox("Transcrire après compression")
        self.check_transcribe_after_encode.setChecked(self.settings.value("transcribe_after_encode", False, type=bool))
        main_form.addRow(self.check_transcribe_after_encode)

        scroll_layout.addWidget(main_group)

        # --- Glossaire transcription ---------------------------------
        glossary_group = QGroupBox("Vocabulaire de la réunion")
        glossary_layout = QVBoxLayout(glossary_group)
        glossary_layout.setContentsMargins(12, 16, 12, 12)
        glossary_layout.setSpacing(8)

        self.edit_transcription_prompt = QTextEdit()
        self.edit_transcription_prompt.setAcceptRichText(False)
        self.edit_transcription_prompt.setPlaceholderText(
            "Noms propres, clients, projets, acronymes, vocabulaire métier…"
        )
        self.edit_transcription_prompt.setPlainText(
            str(
                self.settings.value(
                    "transcription_glossary",
                    self.settings.value("transcription_prompt", "", type=str),
                    type=str,
                )
            )
        )

        # Compact by default; the field grows up to 140 px if the user
        # has a long glossary, but won't bury the rest of the panel.
        self.edit_transcription_prompt.setMinimumHeight(70)
        self.edit_transcription_prompt.setMaximumHeight(140)
        glossary_layout.addWidget(self.edit_transcription_prompt)

        # Subtle inline note rather than the previous "metaLabel" boxed
        # card-within-a-card.
        self.lbl_transcription_hint = QLabel(
            "Conservé entre les réunions · transmis à Whisper comme vocabulaire attendu."
        )
        self.lbl_transcription_hint.setObjectName("inlineHint")
        self.lbl_transcription_hint.setWordWrap(True)
        glossary_layout.addWidget(self.lbl_transcription_hint)

        scroll_layout.addWidget(glossary_group)

        # --- État de la transcription + installer --------------------
        transcription_status_group = QGroupBox("Transcription locale")
        transcription_status_layout = QVBoxLayout(transcription_status_group)
        transcription_status_layout.setContentsMargins(12, 16, 12, 12)
        transcription_status_layout.setSpacing(8)

        self.lbl_transcription_config = QLabel()
        self.lbl_transcription_config.setObjectName("metaLabel")
        self.lbl_transcription_config.setWordWrap(True)
        transcription_status_layout.addWidget(self.lbl_transcription_config)

        self.btn_install_mlx = QPushButton("Installer MLX Whisper")
        self.btn_install_mlx.setObjectName("secondaryButton")
        self.btn_install_mlx.clicked.connect(self.install_mlx_whisper)
        transcription_status_layout.addWidget(self.btn_install_mlx)

        scroll_layout.addWidget(transcription_status_group)

        # --- Avancé (collapsible): compression / audio / rognage ----
        self.advanced_section = CollapsibleSection("▸ Réglages avancés (compression, audio, rognage)")

        comp_group = QGroupBox("Compression vidéo")
        comp_form = QFormLayout(comp_group)
        comp_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        comp_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        comp_form.setHorizontalSpacing(10)
        comp_form.setVerticalSpacing(12)

        self.combo_res = QComboBox()
        self.combo_res.addItems(["Original", "1080p", "720p", "480p"])
        self.combo_res.currentTextChanged.connect(self.on_control_changed)
        comp_form.addRow("Résolution", self.combo_res)

        self.spin_fps = QSpinBox()
        self.spin_fps.setRange(1, 60)
        self.spin_fps.setSuffix(" fps")
        self.spin_fps.valueChanged.connect(self.on_control_changed)
        comp_form.addRow("Fluidité", self.spin_fps)

        self.slider_quality = QSlider(Qt.Orientation.Horizontal)
        self.slider_quality.setRange(18, 35)
        self.slider_quality.valueChanged.connect(self.on_crf_changed)
        self.lbl_quality = QLabel("Qualité (CRF 28)")
        comp_form.addRow(self.lbl_quality, self.slider_quality)

        self.combo_preset = QComboBox()
        self.combo_preset.addItems(["ultrafast", "veryfast", "faster", "fast", "medium", "slow"])
        self.combo_preset.currentTextChanged.connect(self.on_control_changed)
        comp_form.addRow("Vitesse encodage", self.combo_preset)

        self.advanced_section.add_widget(comp_group)

        audio_group = QGroupBox("Audio / voix")
        audio_form = QFormLayout(audio_group)
        audio_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        audio_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        audio_form.setHorizontalSpacing(10)
        audio_form.setVerticalSpacing(12)

        self.combo_audio_bitrate = QComboBox()
        self.combo_audio_bitrate.addItems(["64k", "96k", "128k", "160k", "192k"])
        self.combo_audio_bitrate.currentTextChanged.connect(self.on_control_changed)
        audio_form.addRow("Bitrate", self.combo_audio_bitrate)

        self.check_speech_enhance = QCheckBox("Améliorer l'intelligibilité de la parole")
        self.check_speech_enhance.toggled.connect(self.on_control_changed)
        audio_form.addRow("", self.check_speech_enhance)

        self.check_mono = QCheckBox("Forcer mono")
        self.check_mono.toggled.connect(self.on_control_changed)
        audio_form.addRow("", self.check_mono)

        self.advanced_section.add_widget(audio_group)

        trim_group = QGroupBox("Rognage")
        trim_form = QFormLayout(trim_group)
        trim_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        trim_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        trim_form.setHorizontalSpacing(10)
        trim_form.setVerticalSpacing(12)

        self.check_trim = QPushButton("Activer le rognage")
        self.check_trim.setObjectName("trimToggle")
        self.check_trim.setCheckable(True)
        self.check_trim.toggled.connect(self.on_trim_toggled)
        trim_form.addRow(self.check_trim)

        self.time_start = QTimeEdit()
        self.time_start.setDisplayFormat("HH:mm:ss")
        self.time_start.timeChanged.connect(self.on_control_changed)
        trim_form.addRow("Début", self.time_start)

        self.time_end = QTimeEdit()
        self.time_end.setDisplayFormat("HH:mm:ss")
        self.time_end.timeChanged.connect(self.on_control_changed)
        trim_form.addRow("Fin", self.time_end)

        self.advanced_section.add_widget(trim_group)

        per_job_btns = QHBoxLayout()
        self.btn_apply_to_all = QPushButton("Appliquer à toute la file")
        self.btn_apply_to_all.setObjectName("secondaryButton")
        self.btn_apply_to_all.clicked.connect(self.apply_current_settings_to_all)
        self.btn_reset_preset = QPushButton("Réinitialiser au profil")
        self.btn_reset_preset.setObjectName("secondaryButton")
        self.btn_reset_preset.clicked.connect(self.reset_current_job_to_preset)
        per_job_btns.addWidget(self.btn_apply_to_all)
        per_job_btns.addWidget(self.btn_reset_preset)
        per_job_btns_widget = QWidget()
        per_job_btns_widget.setLayout(per_job_btns)
        self.advanced_section.add_widget(per_job_btns_widget)

        scroll_layout.addWidget(self.advanced_section)

        # --- Estimation (toujours visible en bas) --------------------
        self.lbl_estimation = QLabel("Estimation: sélectionnez une vidéo.")
        self.lbl_estimation.setObjectName("metaLabel")
        self.lbl_estimation.setWordWrap(True)
        scroll_layout.addWidget(self.lbl_estimation)

        scroll_layout.addStretch(1)
        scroll.setWidget(scroll_body)
        right_layout.addWidget(scroll, 1)

        splitter.addWidget(left_panel)
        splitter.addWidget(right_col)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([800, 420])
        splitter.setHandleWidth(1)

        root.addWidget(splitter, 1)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setTextVisible(False)
        self.progress.setVisible(False)
        root.addWidget(self.progress)

        self.progress_global = QProgressBar()
        self.progress_global.setRange(0, 100)
        self.progress_global.setTextVisible(False)
        self.progress_global.setVisible(False)
        root.addWidget(self.progress_global)

        self.status = QLabel("Prêt.")
        self.status.setAlignment(Qt.AlignCenter)
        root.addWidget(self.status)

        actions = QHBoxLayout()
        actions.setSpacing(12)
        self.btn_open_output = QPushButton("Ouvrir dossier sortie")
        self.btn_open_output.setObjectName("secondaryButton")
        self.btn_open_output.clicked.connect(self.open_output_dir)

        self.btn_start = QPushButton("Lancer la file")
        self.btn_start.setObjectName("accentButton")
        self.btn_start.clicked.connect(self.start_encode)
        self.btn_start.setEnabled(False)

        self.btn_transcribe = QPushButton("Transcrire")
        self.btn_transcribe.setObjectName("secondaryButton")
        self.btn_transcribe.clicked.connect(self.start_transcription)
        self.btn_transcribe.setEnabled(False)

        self.btn_cancel = QPushButton("Annuler")
        self.btn_cancel.setObjectName("secondaryButton")
        self.btn_cancel.clicked.connect(self.cancel_encode)
        self.btn_cancel.setEnabled(False)

        actions.addWidget(self.btn_open_output)
        actions.addStretch(1)
        actions.addWidget(self.btn_transcribe)
        actions.addWidget(self.btn_start)
        actions.addWidget(self.btn_cancel)
        root.addLayout(actions)

        # Cmd+, opens Réglages (macOS standard preferences shortcut).
        self._settings_shortcut = QShortcut(QKeySequence("Ctrl+,"), self)
        self._settings_shortcut.activated.connect(self.open_settings)

        self._refresh_transcription_summary()
        self.on_trim_toggled(False)

    def apply_style(self):
        chevron_url = Path(resource_path("assets/chevron-down.svg")).as_posix()
        check_url = Path(resource_path("assets/check.svg")).as_posix()
        self.setStyleSheet(
            ("""
        QWidget {
            background: #f5f5f7;
            color: #1d1d1f;
            font-family: ".AppleSystemUIFont", "Helvetica Neue", "Arial", sans-serif;
            font-size: 13px;
        }
        QLabel {
            color: #1d1d1f;
            background: transparent;
        }
        QFrame#header {
            background: #ffffff;
            border: 1px solid #dedee3;
            border-radius: 14px;
        }
        QLabel#logoMark {
            min-width: 48px;
            max-width: 48px;
            min-height: 48px;
            max-height: 48px;
            background: #f5f5f7;
            border: 1px solid #e1e1e6;
            border-radius: 12px;
            padding: 5px;
            color: #007aff;
            font-size: 24px;
            font-weight: 700;
        }
        QLabel#h1 {
            font-size: 24px;
            font-weight: 700;
            color: #1d1d1f;
            letter-spacing: 0px;
        }
        QLabel#h2 {
            color: #6e6e73;
            font-size: 14px;
            font-weight: 400;
        }
        QToolButton#gear {
            background: #f5f5f7;
            border: 1px solid #d2d2d7;
            border-radius: 10px;
            padding: 8px 12px;
            color: #1d1d1f;
            font-weight: 600;
        }
        QToolButton#gear:hover {
            background: #ececf0;
        }
        QTabWidget#mainTabs::pane {
            border: 1px solid #d2d2d7;
            border-radius: 12px;
            background: #fbfbfd;
            top: -1px;
        }
        QTabBar::tab {
            background: #f5f5f7;
            border: 1px solid #d2d2d7;
            border-bottom: none;
            border-top-left-radius: 8px;
            border-top-right-radius: 8px;
            padding: 6px 16px;
            margin-right: 4px;
            color: #1d1d1f;
            font-weight: 500;
        }
        QTabBar::tab:selected {
            background: #ffffff;
            border-color: #d2d2d7;
            font-weight: 600;
        }
        QListWidget#libraryList {
            background: #ffffff;
            border: 1px solid #d2d2d7;
            border-radius: 10px;
            outline: none;
        }
        QListWidget#libraryList::item {
            padding: 10px;
            border-bottom: 1px solid #f2f2f7;
        }
        QListWidget#libraryList::item:selected {
            background: #e8f2ff;
            color: #007aff;
        }
        QTableWidget#libraryTable {
            background: #ffffff;
            border: 1px solid #d2d2d7;
            border-radius: 10px;
            gridline-color: transparent;
            outline: none;
            selection-background-color: #e8f2ff;
            selection-color: #1d1d1f;
        }
        QTableWidget#libraryTable::item {
            padding: 8px 10px;
            border: none;
            border-bottom: 1px solid #f2f2f7;
        }
        QTableWidget#libraryTable QHeaderView::section {
            background: #fbfbfd;
            border: none;
            border-bottom: 1px solid #e5e5ea;
            padding: 8px 10px;
            font-weight: 600;
            color: #3a3a3c;
        }
        QFrame#libraryStatusCell {
            background: transparent;
        }
        QLabel#libraryStatusRunning {
            color: #007aff;
            font-weight: 600;
        }
        QLabel#libraryStatusDone {
            color: #2e8b57;
            font-weight: 600;
        }
        QLabel#libraryStatusFail {
            color: #d93025;
            font-weight: 600;
        }
        QLabel#libraryStatusPending {
            color: #6e6e73;
            font-weight: 600;
        }
        QLabel#libraryStatusSub {
            color: #6e6e73;
            font-size: 11px;
        }
        QWidget#artefactCell {
            background: transparent;
        }
        QPushButton#artefactButton, QToolButton#artefactButton {
            background: #ffffff;
            border: 1px solid #d2d2d7;
            border-radius: 8px;
            padding: 7px 0;
            font-size: 12px;
            color: #1d1d1f;
        }
        QPushButton#artefactButton:hover, QToolButton#artefactButton:hover {
            background: #f2f2f7;
        }
        QPushButton#artefactButton:disabled {
            background: #fbfbfd;
            color: #b0b0b8;
            border-color: #ececef;
        }
        QToolButton#artefactButton::menu-indicator { image: none; }
        /* Re-run icon next to "Ouvrir" — visible but quieter so the
           main "Ouvrir" stays the affordance the eye lands on. */
        QToolButton#artefactReplay {
            background: #ffffff;
            border: 1px solid #d2d2d7;
            border-top-left-radius: 0;
            border-bottom-left-radius: 0;
            border-top-right-radius: 8px;
            border-bottom-right-radius: 8px;
            margin-left: -1px;
            padding: 7px 0;
            color: #6e6e73;
            font-size: 13px;
        }
        QToolButton#artefactReplay:hover {
            background: #f2f2f7;
            border-color: #d2d2d7;
            color: #1d1d1f;
        }
        QFrame#dropZone {
            background: #ffffff;
            border: 1.5px dashed #c7c7cc;
            border-radius: 14px;
        }
        QFrame#dropZone:hover {
            border-color: #007aff;
            background: #f8fbff;
        }
        QLabel#dropIcon {
            color: #007aff;
            font-size: 28px;
            font-weight: 700;
        }
        QLabel#dropTitle {
            font-size: 19px;
            font-weight: 700;
            color: #1d1d1f;
        }
        QLabel#dropSubtitle, QLabel#dropDetails {
            color: #6e6e73;
            font-size: 14px;
        }
        QFrame#settingsPanel {
            background: #ffffff;
            border: 1px solid #dedee3;
            border-radius: 14px;
        }
        QListWidget#queueList {
            background: #ffffff;
            border: 1px solid #dedee3;
            border-radius: 10px;
            padding: 8px;
        }
        QListWidget#queueList::item {
            padding: 8px 10px;
            margin: 2px 0;
            border-radius: 8px;
            color: #1d1d1f;
        }
        QListWidget#queueList::item:selected {
            background: #e8f1ff;
            color: #0057b8;
        }
        QGroupBox {
            border: 1px solid #e5e5ea;
            border-radius: 12px;
            margin-top: 16px;
            padding: 16px 12px 12px 12px;
            background: #ffffff;
            font-weight: 700;
            color: #1d1d1f;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 12px;
            padding: 0 7px;
            color: #1d1d1f;
            background: #ffffff;
        }
        QScrollArea#settingsScroll {
            background: transparent;
            border: none;
        }
        QFrame#settingsScrollBody {
            background: transparent;
        }
        QFrame#collapsibleSection {
            background: transparent;
            border: none;
        }
        QToolButton#collapsibleToggle {
            background: #f2f2f7;
            border: 1px solid #e1e1e6;
            border-radius: 9px;
            padding: 8px 12px;
            text-align: left;
            color: #1d1d1f;
            font-weight: 600;
        }
        QToolButton#collapsibleToggle:hover {
            background: #e9e9ee;
        }
        QToolButton#collapsibleToggle:checked {
            background: #e9e9ee;
        }
        QFrame#collapsibleBody {
            background: transparent;
            padding-left: 4px;
        }
        QPushButton#primaryButton {
            background: #007aff;
            color: #ffffff;
            border: none;
            border-radius: 10px;
            padding: 10px 16px;
            font-size: 14px;
            font-weight: 700;
        }
        QPushButton#primaryButton:hover { background: #006ee6; }
        QPushButton#accentButton {
            background: #007aff;
            color: #ffffff;
            border: none;
            border-radius: 12px;
            padding: 12px 24px;
            font-size: 15px;
            font-weight: 700;
        }
        QPushButton#accentButton:hover {
            background: #006ee6;
        }
        QPushButton#accentButton:disabled {
            background: #e5e5ea;
            color: #8e8e93;
        }
        QPushButton#secondaryButton {
            background: #ffffff;
            border: 1px solid #d2d2d7;
            border-radius: 10px;
            padding: 9px 12px;
            color: #1d1d1f;
            font-weight: 600;
        }
        QPushButton#secondaryButton:hover {
            background: #f2f2f7;
        }
        QPushButton#trimToggle {
            background: #ffffff;
            border: 1px solid #d2d2d7;
            color: #1d1d1f;
            border-radius: 9px;
            padding: 8px;
            font-weight: 600;
        }
        QPushButton#trimToggle:checked {
            background: #007aff;
            border-color: #007aff;
            color: #ffffff;
        }
        QSpinBox, QTimeEdit, QLineEdit, QTextEdit {
            background: #ffffff;
            border: 1px solid #d2d2d7;
            border-radius: 9px;
            padding: 7px 10px;
            min-height: 24px;
            color: #1d1d1f;
        }
        /* QComboBox needs extra right padding so the displayed text
           never overlaps the dropdown chevron area. */
        QComboBox {
            background: #ffffff;
            border: 1px solid #d2d2d7;
            border-radius: 9px;
            padding: 6px 30px 6px 10px;
            min-height: 24px;
            color: #1d1d1f;
        }
        QComboBox:hover { border-color: #b8b8c0; }
        QTextEdit {
            selection-background-color: #b8d7ff;
        }
        QComboBox::drop-down {
            subcontrol-origin: padding;
            subcontrol-position: top right;
            width: 26px;
            border-left: 1px solid #ececef;
            border-top-right-radius: 9px;
            border-bottom-right-radius: 9px;
            background: #fbfbfd;
        }
        QComboBox::drop-down:hover { background: #f2f2f7; }
        QComboBox::down-arrow {
            image: url(__CHEVRON_URL__);
            width: 12px;
            height: 12px;
        }
        QComboBox QAbstractItemView {
            background: #ffffff;
            border: 1px solid #d2d2d7;
            border-radius: 8px;
            padding: 4px;
            selection-background-color: #007aff;
            selection-color: #ffffff;
            outline: none;
        }
        QComboBox QAbstractItemView::item {
            padding: 6px 10px;
            border-radius: 5px;
            min-height: 22px;
        }
        QCheckBox { color: #1d1d1f; spacing: 8px; }
        QCheckBox::indicator {
            width: 16px;
            height: 16px;
            border-radius: 4px;
            border: 1px solid #c7c7cc;
            background: #ffffff;
        }
        QCheckBox::indicator:checked {
            background: #007aff;
            border-color: #007aff;
            image: url(__CHECK_URL__);
        }
        QCheckBox::indicator:hover { border-color: #007aff; }
        QLineEdit:focus, QSpinBox:focus, QTimeEdit:focus, QTextEdit:focus,
        QComboBox:focus { border-color: #007aff; }
        QSlider::groove:horizontal {
            height: 6px;
            background: #e5e5ea;
            border-radius: 3px;
        }
        QSlider::handle:horizontal {
            background: #007aff;
            border: 2px solid #ffffff;
            width: 16px;
            margin: -6px 0;
            border-radius: 8px;
        }
        QSlider::sub-page:horizontal {
            background: #007aff;
            border-radius: 3px;
        }
        QProgressBar {
            background: #ececf0;
            border: none;
            border-radius: 5px;
            text-align: center;
            min-height: 10px;
            max-height: 10px;
            color: transparent;
        }
        QProgressBar::chunk {
            background: #007aff;
            border-radius: 5px;
        }
        QLabel#metaLabel {
            color: #3a3a3c;
            background: #ffffff;
            border: 1px solid #e5e5ea;
            border-radius: 10px;
            padding: 10px;
            font-size: 13px;
        }
        QLabel#inlineHint {
            color: #6e6e73;
            background: transparent;
            font-size: 12px;
            padding: 2px 0;
        }
        QSplitter::handle {
            background: #e5e5ea;
            width: 1px;
            margin: 8px 10px;
        }
        QScrollBar:vertical {
            width: 9px;
            background: transparent;
            margin: 2px;
        }
        QScrollBar::handle:vertical {
            border-radius: 4px;
            background: #c7c7cc;
            min-height: 30px;
        }
        QScrollBar::handle:vertical:hover {
            background: #aeaeb2;
        }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            height: 0px;
        }
        """)
            .replace("__CHEVRON_URL__", chevron_url)
            .replace("__CHECK_URL__", check_url)
        )

    def _transcription_settings_payload(self) -> dict[str, str | bool]:
        return {
            "mlx_whisper_path": self.mlx_whisper_path,
            "model": self.transcription_model,
            "text_llm_model": self.transcription_text_llm_model,
            "audio_llm_model": self.transcription_audio_llm_model,
            "audio_recheck_enabled": self.transcription_audio_recheck_enabled,
            "language": self.transcription_language,
            "format": self.transcription_format,
            "suffix": self.transcription_suffix,
            "enhance_audio": self.transcription_enhance_audio,
            "diarization_enabled": self.transcription_diarization_enabled,
            "hf_token": self.transcription_hf_token,
        }

    def _refresh_transcription_summary(self):
        mlx_installed = bool(self._mlx_whisper_path() and Path(self._mlx_whisper_path()).exists())
        mlx_status = "installé" if mlx_installed else "à installer"
        if self.transcription_diarization_enabled:
            if self.transcription_hf_token:
                diar_status = "ON"
            else:
                diar_status = "ON (token HF manquant)"
        else:
            diar_status = "OFF"

        text_llm_short = text_llm_label_for(self.transcription_text_llm_model).split(" · ")[0]
        if self.transcription_audio_recheck_enabled:
            audio_llm_short = audio_llm_label_for(self.transcription_audio_llm_model).split(" · ")[0]
            recheck_line = f"Réécoute IA: ON · {audio_llm_short}"
        else:
            recheck_line = "Réécoute IA: OFF"

        self.lbl_transcription_config.setText(
            f"MLX Whisper: {mlx_status}\n"
            f"Modèle Whisper: {self.transcription_model or DEFAULT_WHISPER_MODEL}\n"
            f"Langue: {self.transcription_language} · Sortie: {self.transcription_format}\n"
            f"Détection des locuteurs: {diar_status}\n"
            f"IA texte: {text_llm_short}\n"
            f"{recheck_line}"
        )
        if hasattr(self, "btn_install_mlx"):
            self.btn_install_mlx.setVisible(not mlx_installed or self.mlx_install_worker is not None)
            self.btn_install_mlx.setText("Installation en cours…" if self.mlx_install_worker else "Installer MLX Whisper")

    def _transcription_glossary(self, job: QueueJob | None = None) -> str:
        glossary_parts: list[str] = []
        base_glossary = self.edit_transcription_prompt.toPlainText().strip()
        if base_glossary:
            glossary_parts.append(base_glossary)

        if job and job.db_id:
            record = self.db.get_job(job.db_id) or {}
            try:
                stored_terms = json.loads(record.get("technical_terms_json") or "[]")
                if isinstance(stored_terms, list):
                    glossary_parts.extend(str(term).strip() for term in stored_terms if str(term).strip())
            except Exception:
                pass
            try:
                stored_speakers = json.loads(record.get("speaker_map_json") or "{}")
                if isinstance(stored_speakers, dict):
                    glossary_parts.extend(str(name).strip() for name in stored_speakers.values() if str(name).strip())
            except Exception:
                pass

        glossary_lines: list[str] = []
        for part in glossary_parts:
            for line in str(part).splitlines():
                clean = line.strip()
                if clean and clean not in glossary_lines:
                    glossary_lines.append(clean)
        glossary = "\n".join(glossary_lines)
        if not glossary:
            return ""
        return "Vocabulaire à respecter, noms propres, clients et projets:\n" + glossary

    def open_settings(self):
        dlg = SettingsDialog(
            self,
            self.ffmpeg_path,
            self.ffprobe_path,
            self.github_token,
            self._transcription_settings_payload(),
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        ffmpeg_path, ffprobe_path, github_token, transcription_settings = dlg.values()
        self.ffmpeg_path = ffmpeg_path or find_binary("ffmpeg") or ""
        self.ffprobe_path = ffprobe_path or find_binary("ffprobe") or ""
        self.github_token = github_token
        self.mlx_whisper_path = str(transcription_settings.get("mlx_whisper_path", "")).strip()
        self.transcription_model = canonical_whisper_model_id(
            str(transcription_settings.get("model", ""))
        )
        self.transcription_text_llm_model = (
            str(transcription_settings.get("text_llm_model", "")).strip() or DEFAULT_TEXT_LLM_MODEL
        )
        self.transcription_audio_llm_model = canonical_audio_llm_model_id(
            str(transcription_settings.get("audio_llm_model", "")).strip() or DEFAULT_AUDIO_LLM_MODEL
        )
        self.transcription_audio_recheck_enabled = bool(
            transcription_settings.get("audio_recheck_enabled", False)
        )
        self.transcription_language = str(transcription_settings.get("language", "fr")).strip() or "fr"
        self.transcription_format = str(transcription_settings.get("format", "txt")).strip() or "txt"
        self.transcription_suffix = str(transcription_settings.get("suffix", "")).strip()
        if self.transcription_suffix == "_transcription":
            self.transcription_suffix = ""
        self.transcription_enhance_audio = bool(transcription_settings.get("enhance_audio", True))
        self.transcription_diarization_enabled = bool(
            transcription_settings.get("diarization_enabled", False)
        )
        self.transcription_hf_token = str(transcription_settings.get("hf_token", "")).strip()
        self.settings.setValue("ffmpeg_path", self.ffmpeg_path)
        self.settings.setValue("ffprobe_path", self.ffprobe_path)
        self.settings.setValue("github_token", self.github_token)
        self.settings.setValue("mlx_whisper_path", self.mlx_whisper_path)
        self.settings.setValue("transcription_model", self.transcription_model)
        self.settings.setValue("transcription_text_llm_model", self.transcription_text_llm_model)
        self.settings.setValue("transcription_audio_llm_model", self.transcription_audio_llm_model)
        self.settings.setValue(
            "transcription_audio_recheck_enabled",
            self.transcription_audio_recheck_enabled,
        )
        # Drop the legacy single-key so it doesn't shadow the new ones on
        # next startup. Safe even if the key isn't there.
        self.settings.remove("transcription_llm_model")
        self.settings.setValue("transcription_language", self.transcription_language)
        self.settings.setValue("transcription_format", self.transcription_format)
        self.settings.setValue("transcription_suffix", self.transcription_suffix)
        self.settings.setValue("transcription_enhance_audio", self.transcription_enhance_audio)
        self.settings.setValue(
            "transcription_diarization_enabled", self.transcription_diarization_enabled
        )
        self.settings.setValue("transcription_hf_token", self.transcription_hf_token)
        self._refresh_transcription_summary()

        if self.ffmpeg_path:
            self.status.setText("Paramètres enregistrés.")
        else:
            self.status.setText("ffmpeg non détecté. Installez ffmpeg ou bundlez-le.")

    def save_global_settings(self):
        self.settings.setValue("output_dir", self.edit_output_dir.text().strip())
        self.settings.setValue("suffix", self.edit_suffix.text().strip())
        self.settings.setValue("continue_on_error", self.check_continue_on_error.isChecked())
        self.settings.setValue("transcribe_after_encode", self.check_transcribe_after_encode.isChecked())
        self.settings.setValue("mlx_whisper_path", self.mlx_whisper_path)
        self.settings.setValue("transcription_model", self.transcription_model)
        self.settings.setValue("transcription_language", self.transcription_language)
        self.settings.setValue("transcription_format", self.transcription_format)
        self.settings.setValue("transcription_suffix", self.transcription_suffix)
        self.settings.setValue("transcription_enhance_audio", self.transcription_enhance_audio)
        self.settings.setValue("transcription_glossary", self.edit_transcription_prompt.toPlainText().strip())

    def check_updates(self):
        if self.update_worker is not None:
            return
        self.btn_update.setEnabled(False)
        self.status.setText("Recherche de mise à jour…")
        self._start_update_worker(mode="check", current_version=APP_VERSION)

    def _start_update_worker(self, mode: str, current_version: str = "", release_info: dict | None = None):
        self._update_phase = mode
        self._update_check_payload = None
        self._update_download_payload = None
        self._update_error_message = None

        worker = UpdateWorker(
            mode=mode,
            current_version=current_version,
            release_info=release_info,
            github_token=self.github_token,
        )
        self.update_worker = worker
        worker.failed.connect(self.on_update_failed)
        if mode == "check":
            worker.check_finished.connect(self.on_update_check_finished)
        else:
            worker.download_progress.connect(self.on_update_download_progress)
            worker.download_finished.connect(self.on_update_download_finished)
        worker.finished.connect(self.on_update_worker_finished)
        worker.start()

    def on_update_check_finished(self, payload: dict):
        self._update_check_payload = payload

    def on_update_download_finished(self, zip_path: str, info: dict):
        self._update_download_payload = (zip_path, info)

    def on_update_worker_finished(self):
        phase = self._update_phase
        self.update_worker = None
        self._update_phase = ""
        if phase == "check":
            self._handle_update_check_completion()
            return
        if phase == "download":
            self._handle_update_download_completion()
            return
        self.btn_update.setEnabled(not self.is_batch_running)

    def _handle_update_check_completion(self):
        self.btn_update.setEnabled(not self.is_batch_running)
        if self._update_error_message:
            self._show_update_error(self._update_error_message)
            return

        payload = self._update_check_payload or {}
        state = payload.get("state")
        info = payload.get("info", {})
        self._last_update_info = info if info else None

        if state == "up_to_date":
            self.status.setText("Aucune mise à jour disponible.")
            QMessageBox.information(self, "Mise à jour", "Vous utilisez déjà la dernière version.")
            return

        if state in {"no_asset", "unknown_remote"}:
            self.status.setText("Impossible de déterminer une mise à jour compatible.")
            QMessageBox.warning(
                self,
                "Mise à jour",
                "Aucune archive macOS compatible trouvée dans la dernière release.",
            )
            return

        if state != "available" or not info.get("asset_url"):
            self.status.setText("Vérification de mise à jour incomplète.")
            return

        current_label = APP_VERSION if parse_semver(APP_VERSION) else "dev/local"
        target_label = info.get("tag_name", "nouvelle version")
        answer = QMessageBox.question(
            self,
            "Mise à jour disponible",
            (
                f"Version actuelle: {current_label}\n"
                f"Dernière version: {target_label}\n\n"
                "Télécharger et installer automatiquement maintenant ?"
            ),
        )
        if answer != QMessageBox.StandardButton.Yes:
            self.status.setText("Mise à jour reportée.")
            return

        self.status.setText("Téléchargement de la mise à jour…")
        self.progress_global.setRange(0, 100)
        self.progress_global.setValue(0)
        self.btn_update.setEnabled(False)
        self._start_update_worker(mode="download", release_info=info)

    def on_update_download_progress(self, pct: int):
        self.progress_global.setRange(0, 100)
        self.progress_global.setValue(pct)
        self.status.setText(f"Téléchargement mise à jour… {pct}%")

    def _handle_update_download_completion(self):
        self.btn_update.setEnabled(not self.is_batch_running)
        if self._update_error_message:
            self._show_update_error(self._update_error_message)
            return

        if not self._update_download_payload:
            self.status.setText("Mise à jour échouée.")
            QMessageBox.warning(self, "Mise à jour", "Téléchargement de mise à jour incomplet.")
            return

        zip_path, info = self._update_download_payload
        try:
            tmp_dir = Path(tempfile.mkdtemp(prefix="ekovideo-update-"))
            ditto = shutil.which("ditto")
            if ditto:
                subprocess.run([ditto, "-x", "-k", "--noqtn", zip_path, str(tmp_dir)], check=True)
            else:
                with zipfile.ZipFile(zip_path, "r") as archive:
                    archive.extractall(tmp_dir)

            candidates = list(tmp_dir.rglob("EkoVideoCompressor.app"))
            if not candidates:
                candidates = list(tmp_dir.rglob("*.app"))
            if not candidates:
                raise RuntimeError("Archive de mise à jour invalide: app introuvable.")

            new_app_path = candidates[0].resolve()
            self._restore_app_executable_permissions(new_app_path)
            target_app_path = self._target_app_bundle_path()
            script_path = (tmp_dir / "apply_update.sh").resolve()
            pid = os.getpid()
            fallback_app = (Path.home() / "Applications" / "EkoVideoCompressor.app").resolve()
            log_dir = app_support_dir()
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = (log_dir / "updater.log").resolve()

            lsregister_path = (
                "/System/Library/Frameworks/CoreServices.framework"
                "/Frameworks/LaunchServices.framework/Support/lsregister"
            )
            script_content = "\n".join(
                [
                    "#!/bin/bash",
                    "set -u",
                    f"LOG_PATH={shlex.quote(str(log_path))}",
                    "exec >> \"$LOG_PATH\" 2>&1",
                    "echo \"=== EkoVideo updater $(date) ===\"",
                    f"PID={pid}",
                    f"NEW_APP={shlex.quote(str(new_app_path))}",
                    f"TARGET_APP={shlex.quote(str(target_app_path))}",
                    f"FALLBACK_APP={shlex.quote(str(fallback_app))}",
                    f"LSREGISTER={shlex.quote(lsregister_path)}",
                    "echo \"New app: $NEW_APP\"",
                    "echo \"Target app: $TARGET_APP\"",
                    "echo \"Fallback app: $FALLBACK_APP\"",
                    "restore_exec_bits() {",
                    "  local app=\"$1\"",
                    "  if [ -d \"$app/Contents/MacOS\" ]; then",
                    "    chmod +x \"$app/Contents/MacOS\"/* 2>/dev/null || true",
                    "  fi",
                    "  if [ -d \"$app/Contents/Resources/bin\" ]; then",
                    "    chmod +x \"$app/Contents/Resources/bin\"/* 2>/dev/null || true",
                    "  fi",
                    "  if [ -d \"$app/Contents/Frameworks/bin\" ]; then",
                    "    chmod +x \"$app/Contents/Frameworks/bin\"/* 2>/dev/null || true",
                    "  fi",
                    "}",
                    "verify_app() {",
                    "  local app=\"$1\"",
                    "  local exe=\"$app/Contents/MacOS/EkoVideoCompressor\"",
                    "  restore_exec_bits \"$app\"",
                    "  if [ ! -x \"$exe\" ]; then",
                    "    echo \"Executable missing or not executable: $exe\"",
                    "    return 1",
                    "  fi",
                    "  \"$exe\" --smoke-test >/dev/null 2>&1",
                    "}",
                    # Re-sign ad-hoc and refresh LaunchServices so macOS treats the
                    # replaced bundle as the same identity it knew before. Without
                    # this, Gatekeeper's first-launch check on the moved bundle
                    # races with `open` and the app fails to relaunch — and even
                    # manual reopen can fail until reinstall.
                    "finalize_app() {",
                    "  local dst=\"$1\"",
                    "  restore_exec_bits \"$dst\"",
                    "  xattr -cr \"$dst\" || true",
                    "  xattr -dr com.apple.quarantine \"$dst\" || true",
                    "  codesign --force --deep --sign - --timestamp=none \"$dst\" >/dev/null 2>&1 || true",
                    "  if [ -x \"$LSREGISTER\" ]; then",
                    "    \"$LSREGISTER\" -f -R \"$dst\" >/dev/null 2>&1 || true",
                    "  fi",
                    "}",
                    "install_app() {",
                    "  local src=\"$1\"",
                    "  local dst=\"$2\"",
                    "  local parent",
                    "  local staged",
                    "  local backup",
                    "  parent=\"$(dirname \"$dst\")\"",
                    "  staged=\"${parent}/.EkoVideoCompressor.update.$$.app\"",
                    "  backup=\"${parent}/.EkoVideoCompressor.backup.$(date +%s).app\"",
                    "  mkdir -p \"$parent\" || return 1",
                    "  echo \"Installing from $src to $dst via $staged\"",
                    "  verify_app \"$src\" || { echo \"Source verification failed: $src\"; return 1; }",
                    "  rm -rf \"$staged\" || { echo \"Cannot remove old staged app: $staged\"; return 1; }",
                    "  ditto --noqtn \"$src\" \"$staged\" || { echo \"ditto failed: $src -> $staged\"; return 1; }",
                    "  restore_exec_bits \"$staged\"",
                    "  xattr -cr \"$staged\" || true",
                    "  xattr -dr com.apple.quarantine \"$staged\" || true",
                    "  verify_app \"$staged\" || { echo \"Staged verification failed: $staged\"; rm -rf \"$staged\"; return 1; }",
                    "  if [ -d \"$dst\" ]; then",
                    "    mv \"$dst\" \"$backup\" || { echo \"Cannot move existing app to backup: $dst -> $backup\"; rm -rf \"$staged\"; return 1; }",
                    "  fi",
                    "  if ! mv \"$staged\" \"$dst\"; then",
                    "    echo \"Cannot move staged app into place: $staged -> $dst\"",
                    "    [ -d \"$backup\" ] && mv \"$backup\" \"$dst\"",
                    "    return 1",
                    "  fi",
                    "  finalize_app \"$dst\"",
                    "  rm -rf \"$backup\" || true",
                    "  return 0",
                    "}",
                    "for i in {1..120}; do",
                    "  if ! kill -0 \"$PID\" 2>/dev/null; then",
                    "    break",
                    "  fi",
                    "  sleep 0.25",
                    "done",
                    # Settle the filesystem so LaunchServices sees a stable bundle
                    # before `open -n` requests a fresh instance.
                    "relaunch() {",
                    "  local app=\"$1\"",
                    "  sleep 0.5",
                    "  open -n \"$app\" && return 0",
                    "  sleep 1",
                    "  open -n \"$app\" && return 0",
                    "  open \"$app\" && return 0",
                    "  return 1",
                    "}",
                    "if install_app \"$NEW_APP\" \"$TARGET_APP\"; then",
                    "  echo \"Installed update to: $TARGET_APP\"",
                    "  relaunch \"$TARGET_APP\" && exit 0",
                    "  echo \"Relaunch failed for: $TARGET_APP\"",
                    "fi",
                    "echo \"Primary target failed, trying fallback: $FALLBACK_APP\"",
                    "if install_app \"$NEW_APP\" \"$FALLBACK_APP\"; then",
                    "  echo \"Installed update to fallback: $FALLBACK_APP\"",
                    "  relaunch \"$FALLBACK_APP\" && exit 0",
                    "  echo \"Relaunch failed for: $FALLBACK_APP\"",
                    "fi",
                    "echo \"Install failed, opening extracted app directly\"",
                    "finalize_app \"$NEW_APP\"",
                    "relaunch \"$NEW_APP\" && exit 0",
                    "echo \"Updater failed to relaunch app\"",
                    "",
                ]
            )
            script_path.write_text(script_content, encoding="utf-8")
            script_path.chmod(0o755)

            release_label = str(info.get("tag_name", "nouvelle version"))
            QMessageBox.information(
                self,
                "Installation de mise à jour",
                f"La version {release_label} va s'installer puis l'app sera relancée.",
            )
            subprocess.Popen(["/bin/bash", str(script_path)], start_new_session=True)
            QApplication.quit()
        except Exception as exc:
            self.status.setText("Mise à jour échouée.")
            QMessageBox.warning(
                self,
                "Mise à jour",
                (
                    "Échec de l'installation automatique.\n"
                    f"Détail: {exc}\n\n"
                    "Vous pouvez installer manuellement depuis la page Releases."
                ),
            )

    def on_update_failed(self, message: str):
        self._update_error_message = message

    def _restore_app_executable_permissions(self, app_path: Path):
        for rel_dir in (
            Path("Contents/MacOS"),
            Path("Contents/Resources/bin"),
            Path("Contents/Frameworks/bin"),
        ):
            directory = app_path / rel_dir
            if not directory.is_dir():
                continue
            for item in directory.iterdir():
                if item.is_file():
                    try:
                        item.chmod(item.stat().st_mode | 0o111)
                    except Exception:
                        pass

    def _show_update_error(self, message: str):
        self.btn_update.setEnabled(not self.is_batch_running)
        self.status.setText("Mise à jour indisponible.")
        hint = ""
        if "CERTIFICATE_VERIFY_FAILED" in (message or ""):
            hint = (
                "\n\nConseil: vérifiez la date/heure Mac et les certificats système."
                "\nSi votre réseau d'entreprise inspecte SSL, autorisez github.com"
                " ou testez depuis un autre réseau."
            )
        QMessageBox.warning(
            self,
            "Mise à jour",
            (
                "Impossible de vérifier/télécharger la mise à jour.\n"
                f"Détail: {message}{hint}"
            ),
        )

    def _target_app_bundle_path(self) -> Path:
        exe = Path(sys.executable).resolve()
        for parent in [exe, *exe.parents]:
            if parent.name.endswith(".app"):
                p = str(parent)
                # App translocation paths are read-only/ephemeral; install to stable location instead.
                if "/AppTranslocation/" in p or p.startswith("/private/var/folders/"):
                    break
                return parent
        user_apps = Path.home() / "Applications"
        user_apps.mkdir(parents=True, exist_ok=True)
        return user_apps / "EkoVideoCompressor.app"

    def pick_output_dir(self):
        start = self.edit_output_dir.text().strip() or str(Path.home())
        directory = QFileDialog.getExistingDirectory(self, "Choisir dossier de sortie", start)
        if directory:
            self.edit_output_dir.setText(directory)

    def pick_mlx_whisper(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choisir mlx_whisper", str(Path.home()), "Tous (*.*)")
        if path:
            self.mlx_whisper_path = path
            self.settings.setValue("mlx_whisper_path", self.mlx_whisper_path)
            self._refresh_transcription_summary()

    def install_mlx_whisper(self):
        if self.mlx_install_worker is not None:
            return

        python_path = find_compatible_python()
        if not python_path:
            QMessageBox.warning(
                self,
                "Python compatible introuvable",
                (
                    "L'installation automatique requiert Python 3.11, 3.12 ou 3.13.\n\n"
                    "Installez-en un avec Homebrew, puis relancez l'installation:\n"
                    "brew install python@3.12"
                ),
            )
            return

        answer = QMessageBox.question(
            self,
            "Installer MLX Whisper",
            (
                "L'app va installer MLX Whisper dans un environnement isolé:\n"
                f"{managed_transcription_venv_dir()}\n\n"
                "Cette installation peut prendre quelques minutes et télécharger plusieurs paquets."
            ),
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        self.btn_install_mlx.setEnabled(False)
        self.btn_transcribe.setEnabled(False)
        self.status.setText("Installation de MLX Whisper…")
        self._set_progress_visible(True)
        self.progress.setRange(0, 0)

        worker = MlxWhisperInstallWorker(
            python_path,
            managed_transcription_venv_dir(),
            install_diarization=self.transcription_diarization_enabled,
            install_multimodal=self.transcription_audio_recheck_enabled,
        )
        self.mlx_install_worker = worker
        worker.status.connect(self.on_mlx_install_status)
        worker.finished_ok.connect(self.on_mlx_install_done)
        worker.failed.connect(self.on_mlx_install_failed)
        worker.finished.connect(self.on_mlx_install_finished)
        worker.start()

    def on_mlx_install_status(self, text: str):
        self.status.setText(text)

    def on_mlx_install_done(self, mlx_path: str):
        self.mlx_whisper_path = mlx_path
        self.settings.setValue("mlx_whisper_path", mlx_path)
        self._refresh_transcription_summary()
        self.status.setText("MLX Whisper installé.")
        QMessageBox.information(
            self,
            "MLX Whisper installé",
            "La transcription locale est prête. Le modèle sera téléchargé automatiquement au premier lancement.",
        )

    def on_mlx_install_failed(self, message: str):
        self.status.setText("Installation MLX Whisper échouée.")
        QMessageBox.warning(
            self,
            "Installation MLX Whisper",
            (
                "Impossible d'installer MLX Whisper dans l'environnement isolé.\n"
                f"Détail: {message}"
            ),
        )

    def on_mlx_install_finished(self):
        self.mlx_install_worker = None
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self._set_progress_visible(False)
        self._refresh_transcription_summary()
        self.btn_install_mlx.setEnabled(True)
        self.btn_transcribe.setEnabled(bool(self.queue_jobs) and not self.is_batch_running)

    def open_output_dir(self):
        out_dir = Path(self.edit_output_dir.text().strip() or str(Path.home()))
        out_dir.mkdir(parents=True, exist_ok=True)
        if sys.platform == "darwin":
            subprocess.Popen(["open", str(out_dir)])
        elif sys.platform.startswith("win"):
            subprocess.Popen(["explorer", str(out_dir)])
        else:
            subprocess.Popen(["xdg-open", str(out_dir)])

    def pick_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Choisir des vidéos ou audios", str(Path.home()), MEDIA_FILTER
        )
        if paths:
            self.add_input_files(paths)

    def load_jobs_from_db(self):
        rows = self.db.list_jobs(limit=1000)
        for row in rows:
            settings = json.loads(row['settings_json']) if row['settings_json'] else {}
            # Restore status and other fields from DB row
            job = QueueJob(**settings)
            job.db_id = row['id']
            job.status = row['status']
            job.error_message = row['error_message'] or ""
            job.workspace_dir = row['workspace_dir'] or ""
            job.output_path = row['output_path'] or ""

            self.queue_jobs.append(job)
            self.queue_list.addItem(QListWidgetItem())
            self.refresh_queue_item(len(self.queue_jobs) - 1)

        if self.queue_jobs:
            self.queue_list.setCurrentRow(0)

    def _new_job(self, input_path: str) -> QueueJob:
        base_job = QueueJob(input_path=input_path)
        base_job = apply_profile_to_job(base_job, "Réunion équilibrée")
        meta = probe_video_metadata(input_path, self.ffprobe_path)
        if meta.get("duration"):
            base_job.trim_end = format_seconds(meta["duration"])
        return base_job

    def add_input_files(self, paths: list[str]):
        existing = {job.input_path for job in self.queue_jobs}
        added = 0
        rejected = 0
        for p in paths:
            resolved = str(Path(p).resolve())
            if not resolved or resolved in existing or not Path(resolved).exists():
                continue
            # Reject anything that isn't a known media file. ffmpeg accepts
            # almost anything, but silently encoding (say) a `.txt` into an
            # `.mp4` confuses users — better to surface it here.
            if Path(resolved).suffix.lower() not in MEDIA_EXTENSIONS:
                rejected += 1
                continue

            job = self._new_job(resolved)
            
            # Create persistent workspace and DB record
            job_slug = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{Path(resolved).stem}"[:50]
            job_ws = self.workspace_root / job_slug
            job_ws.mkdir(parents=True, exist_ok=True)
            job.workspace_dir = str(job_ws)
            
            db_id = self.db.create_job(
                source_path=resolved,
                workspace_dir=str(job_ws),
                settings=asdict(job)
            )
            job.db_id = db_id

            if self.is_batch_running:
                self._prepare_job_for_active_batch(job, self.batch_mode)
            
            self.queue_jobs.append(job)
            self.queue_list.addItem(QListWidgetItem())
            new_idx = len(self.queue_jobs) - 1
            if self.is_batch_running:
                self.job_run_modes[id(job)] = self.batch_mode
                self.pending_indices.append(new_idx)
            self.refresh_queue_item(new_idx)
            existing.add(resolved)
            added += 1

        if added and self.queue_list.currentRow() < 0:
            self.queue_list.setCurrentRow(0)

        self.btn_start.setEnabled(bool(self.queue_jobs) and not self.is_batch_running)
        self.btn_transcribe.setEnabled(bool(self.queue_jobs) and not self.is_batch_running)
        if added and rejected:
            self.status.setText(
                f"{added} fichier(s) ajouté(s) · {rejected} ignoré(s) (extension non supportée)."
            )
        elif added:
            suffix = " à la file en cours." if self.is_batch_running else "."
            self.status.setText(f"{added} fichier(s) ajouté(s){suffix}")
        elif rejected:
            self.status.setText(
                f"{rejected} fichier(s) ignoré(s) — extensions acceptées: vidéo (mp4, mov…) ou audio (mp3, m4a, wav…)."
            )

    def _prepare_job_for_active_batch(self, job: QueueJob, mode: str):
        """
        Allow users to keep dropping files while a batch is running. New
        items inherit the current batch mode and are appended after all
        already-pending items; the running job itself is untouched.
        """
        out_dir = Path(self.edit_output_dir.text().strip() or str(Path.home() / "Desktop"))
        out_dir.mkdir(parents=True, exist_ok=True)
        if mode in {"compress", "compress_transcribe"}:
            job.output_path = default_out_path(job.input_path, str(out_dir), self.edit_suffix.text().strip())
        if mode in {"transcribe", "compress_transcribe"}:
            job.transcript_path = self._transcription_output_path(job, out_dir)
        job.status = "pending"
        job.error_message = ""
        if job.db_id:
            self.db.update_job_status(job.db_id, "PENDING", "")
            if job.output_path:
                self.db.update_job_output(job.db_id, job.output_path)
            self.db.update_job_progress(job.db_id, step="En attente…", progress_pct=0, eta_seconds=None)

    def remove_selected_files(self):
        rows = sorted({idx.row() for idx in self.queue_list.selectedIndexes()}, reverse=True)
        if not rows:
            return
        for row in rows:
            if 0 <= row < len(self.queue_jobs):
                self.queue_jobs.pop(row)
                self.queue_list.takeItem(row)

        self.current_index = -1
        if self.queue_jobs:
            self.queue_list.setCurrentRow(min(rows[-1], len(self.queue_jobs) - 1))
        else:
            self.lbl_input_meta.setText("Aucun fichier sélectionné.")
            self.update_estimation(None)

        self.btn_start.setEnabled(bool(self.queue_jobs) and not self.is_batch_running)
        self.btn_transcribe.setEnabled(bool(self.queue_jobs) and not self.is_batch_running)

    def clear_queue(self):
        if self.is_batch_running:
            return
        self.queue_jobs.clear()
        self.queue_list.clear()
        self.current_index = -1
        self.lbl_input_meta.setText("Aucun fichier sélectionné.")
        self.update_estimation(None)
        self.progress.setValue(0)
        self.progress_global.setValue(0)
        self._set_progress_visible(False)
        self.btn_start.setEnabled(False)
        self.btn_transcribe.setEnabled(False)

    def status_prefix(self, job: QueueJob) -> str:
        if job.status == "done":
            return "✅"
        if job.status == "failed":
            return "❌"
        if job.status == "running":
            return "⏳"
        return "⏳"

    def refresh_queue_item(self, idx: int):
        if idx < 0 or idx >= len(self.queue_jobs):
            return
        job = self.queue_jobs[idx]
        suffix = PROFILE_SUFFIX.get(job.profile_name, "[Custom]")
        # The 🎧 marker tells the user at a glance that this entry skips
        # the video re-encode pipeline and outputs an .m4a — useful when
        # a queue mixes meeting screen-shares and audio-only Zoom dumps.
        media_marker = "🎧 " if is_audio_only_path(job.input_path) else ""
        title = f"{self.status_prefix(job)} {media_marker}{Path(job.input_path).name} {suffix}"
        item = self.queue_list.item(idx)
        if item:
            item.setText(title)
            tooltip = job.input_path
            if is_audio_only_path(job.input_path):
                tooltip += "\nType: audio (sortie en .m4a)"
            if job.output_path:
                tooltip += f"\nSortie: {job.output_path}"
            if job.transcript_path:
                tooltip += f"\nTranscription: {job.transcript_path}"
            if job.error_message:
                tooltip += f"\nErreur: {job.error_message}"
            item.setToolTip(tooltip)
            # The queue is staging only — once a job is running or done
            # it lives in the Library tab, not here. We hide rather than
            # remove so running_index / queue_jobs indices stay valid
            # for the existing worker plumbing.
            item.setHidden(job.status in {"running", "done", "failed"})

    def refresh_all_queue_items(self):
        for idx in range(len(self.queue_jobs)):
            self.refresh_queue_item(idx)

    def _on_queue_rows_moved(self, parent, start: int, end: int, dest, dest_row: int):
        """
        Keep self.queue_jobs in sync when the user drags an item
        somewhere else in the queue list. Qt has already moved the
        QListWidgetItem; we mirror the same move on the parallel
        queue_jobs array. While a batch is running, only still-pending
        items can move; the running/done/failed rows stay anchored so
        worker indices keep pointing at the right job.
        """
        if start < 0 or start >= len(self.queue_jobs):
            return
        if end != start:
            self.refresh_all_queue_items()
            return
        # Qt's contract for rowsMoved: dest_row is the index BEFORE the
        # move, in the destination, where the item lands. When moving
        # downward inside the same parent, the effective index is
        # dest_row - 1.
        target = dest_row
        if dest_row > start:
            target -= 1
        if target < 0 or target >= len(self.queue_jobs) or target == start:
            return
        if self.is_batch_running and not self._can_move_queue_row_while_running(start, target):
            self.refresh_all_queue_items()
            return
        item = self.queue_jobs.pop(start)
        self.queue_jobs.insert(target, item)
        if self.is_batch_running:
            self._rebuild_pending_indices()
        # The current_index should track the moved item if it was
        # selected, otherwise stay on whatever's now at that row.
        if self.current_index == start:
            self.current_index = target
        elif start < self.current_index <= target:
            self.current_index -= 1
        elif target <= self.current_index < start:
            self.current_index += 1
        self.refresh_all_queue_items()

    def _can_move_queue_row_while_running(self, start: int, target: int) -> bool:
        moving = self.queue_jobs[start]
        if moving.status != "pending":
            return False
        moved_ids = {id(moving)}
        low = min(start, target)
        high = max(start, target)
        for idx in range(low, high + 1):
            if idx == start:
                continue
            job = self.queue_jobs[idx]
            if job.status != "pending" and id(job) not in moved_ids:
                return False
        return True

    def _rebuild_pending_indices(self):
        self.pending_indices = [
            idx
            for idx, job in enumerate(self.queue_jobs)
            if job.status == "pending"
        ]

    def _job_run_mode(self, job: QueueJob) -> str:
        return self.job_run_modes.get(id(job), self.batch_mode)

    def _set_all_job_run_modes(self, mode: str):
        self.job_run_modes = {id(job): mode for job in self.queue_jobs}

    def on_queue_selection_changed(self, row: int):
        self.current_index = row
        if row < 0 or row >= len(self.queue_jobs):
            self.lbl_input_meta.setText("Aucun fichier sélectionné.")
            self.update_estimation(None)
            return

        job = self.queue_jobs[row]
        self._syncing_job = True
        self.combo_profile.setCurrentText(job.profile_name if job.profile_name in ["Personnalisé", *PROFILE_PRESETS.keys()] else "Personnalisé")
        self.combo_res.setCurrentText(job.resolution)
        self.spin_fps.setValue(job.fps)
        self.slider_quality.setValue(job.crf)
        self.combo_preset.setCurrentText(job.x265_preset)
        self.combo_audio_bitrate.setCurrentText(job.audio_bitrate)
        self.check_speech_enhance.setChecked(job.speech_enhance)
        self.check_mono.setChecked(job.mono_audio)
        self.check_trim.setChecked(job.trim_enabled)
        self.time_start.setTime(QTime.fromString(job.trim_start, "HH:mm:ss"))
        self.time_end.setTime(QTime.fromString(job.trim_end, "HH:mm:ss"))
        self._syncing_job = False
        self.on_trim_toggled(job.trim_enabled)

        meta = probe_video_metadata(job.input_path, self.ffprobe_path)
        if is_audio_only_path(job.input_path):
            # Audio-only files have no resolution/FPS — showing "-" everywhere
            # is noisy, so we render a dedicated one-liner.
            self.lbl_input_meta.setText(
                f"Fichier audio: {Path(job.input_path).name}\n"
                f"Durée: {format_seconds(meta.get('duration'))} | Sortie: .m4a (AAC)\n"
                f"Poids actuel: {format_size(meta.get('size_bytes'))}"
            )
        else:
            dims = "-"
            if meta.get("width") and meta.get("height"):
                dims = f"{meta['width']}x{meta['height']}"
            fps = f"{meta['fps']:.1f}" if meta.get("fps") else "-"
            self.lbl_input_meta.setText(
                f"Fichier: {Path(job.input_path).name}\n"
                f"Durée: {format_seconds(meta.get('duration'))} | Résolution: {dims} | FPS: {fps}\n"
                f"Poids actuel: {format_size(meta.get('size_bytes'))}"
            )
        self.update_estimation(meta)

    def on_profile_changed(self, profile_name: str):
        if self._syncing_job or self.current_index < 0 or self.current_index >= len(self.queue_jobs):
            return
        if profile_name == "Personnalisé":
            self.save_current_job_from_controls(force_custom=True)
            return

        self.queue_jobs[self.current_index] = apply_profile_to_job(self.queue_jobs[self.current_index], profile_name)
        self.on_queue_selection_changed(self.current_index)
        self.refresh_queue_item(self.current_index)

    def on_crf_changed(self, value: int):
        self.lbl_quality.setText(f"Qualité (CRF {value})")
        self.on_control_changed()

    def on_trim_toggled(self, checked: bool):
        self.time_start.setEnabled(checked)
        self.time_end.setEnabled(checked)
        self.on_control_changed()

    def on_control_changed(self):
        if self._syncing_job:
            return
        self.save_current_job_from_controls(force_custom=False)

    def save_current_job_from_controls(self, force_custom: bool):
        if self.current_index < 0 or self.current_index >= len(self.queue_jobs):
            return

        job = self.queue_jobs[self.current_index]
        job.resolution = self.combo_res.currentText()
        job.fps = self.spin_fps.value()
        job.crf = self.slider_quality.value()
        job.x265_preset = self.combo_preset.currentText()
        job.audio_bitrate = self.combo_audio_bitrate.currentText()
        job.speech_enhance = self.check_speech_enhance.isChecked()
        job.mono_audio = self.check_mono.isChecked()
        job.trim_enabled = self.check_trim.isChecked()
        job.trim_start = self.time_start.time().toString("HH:mm:ss")
        job.trim_end = self.time_end.time().toString("HH:mm:ss")

        if force_custom:
            job.profile_name = "Personnalisé"
        else:
            job.profile_name = infer_profile_name(job)

        self.refresh_queue_item(self.current_index)
        self.update_estimation()

    def apply_current_settings_to_all(self):
        if self.current_index < 0 or self.current_index >= len(self.queue_jobs):
            return

        self.save_current_job_from_controls(force_custom=False)
        source = self.queue_jobs[self.current_index]
        for idx, job in enumerate(self.queue_jobs):
            if idx == self.current_index:
                continue
            self.queue_jobs[idx] = replace(
                job,
                profile_name=source.profile_name,
                resolution=source.resolution,
                fps=source.fps,
                crf=source.crf,
                audio_bitrate=source.audio_bitrate,
                x265_preset=source.x265_preset,
                speech_enhance=source.speech_enhance,
                mono_audio=source.mono_audio,
                trim_enabled=source.trim_enabled,
                trim_start=source.trim_start,
                trim_end=source.trim_end,
            )
        self.refresh_all_queue_items()
        self.status.setText("Réglages appliqués à tous les fichiers.")

    def reset_current_job_to_preset(self):
        if self.current_index < 0 or self.current_index >= len(self.queue_jobs):
            return
        job = self.queue_jobs[self.current_index]
        preset_name = job.profile_name if job.profile_name in PROFILE_PRESETS else "Réunion équilibrée"
        self.queue_jobs[self.current_index] = apply_profile_to_job(job, preset_name)
        self.on_queue_selection_changed(self.current_index)
        self.refresh_queue_item(self.current_index)

    def _estimate_compression_ratio(self, job: QueueJob) -> tuple[int, int]:
        base_min, base_max = 35, 55
        if job.crf >= 30:
            base_min, base_max = 60, 80
        elif job.crf >= 26:
            base_min, base_max = 45, 65

        if job.fps <= 12:
            base_min += 8
            base_max += 8
        if job.resolution == "480p":
            base_min += 8
            base_max += 8
        elif job.resolution == "1080p":
            base_min -= 5
            base_max -= 5

        if job.mono_audio:
            base_min += 3
            base_max += 3

        return max(20, base_min), min(92, base_max)

    def update_estimation(self, meta: dict | None = None):
        if self.current_index < 0 or self.current_index >= len(self.queue_jobs):
            self.lbl_estimation.setText("Estimation: sélectionnez une vidéo.")
            return

        job = self.queue_jobs[self.current_index]
        if meta is None:
            meta = probe_video_metadata(job.input_path, self.ffprobe_path)

        min_gain, max_gain = self._estimate_compression_ratio(job)
        text = f"Gain estimé: {min_gain}% à {max_gain}%"
        if meta and meta.get("size_bytes"):
            src = int(meta["size_bytes"])
            min_out = int(src * (1 - max_gain / 100.0))
            max_out = int(src * (1 - min_gain / 100.0))
            text += f" | Sortie approx: {format_size(min_out)} à {format_size(max_out)}"
        self.lbl_estimation.setText(text)

    def _validate_trim(self, job: QueueJob) -> bool:
        if not job.trim_enabled:
            return True
        start = QTime.fromString(job.trim_start, "HH:mm:ss")
        end = QTime.fromString(job.trim_end, "HH:mm:ss")
        return start.isValid() and end.isValid() and start < end

    def _transcription_output_path(self, job: QueueJob, out_dir: Path) -> str:
        return default_transcript_path(
            job.input_path,
            str(out_dir),
            self.transcription_suffix,
            self.transcription_format,
        )

    def _auto_rename_transcript(self, transcript_path: str, job: QueueJob) -> str:
        path = Path(transcript_path)
        if not path.exists() or path.suffix.lower() not in {".txt", ".srt", ".vtt", ".json"}:
            return transcript_path
        try:
            raw = path.read_text(encoding="utf-8")
        except Exception:
            return transcript_path

        suggested_stem = suggest_transcript_stem(raw, Path(job.input_path).stem)
        suggested_path = path.with_name(f"{suggested_stem}{path.suffix}")
        if suggested_path == path:
            return transcript_path

        candidate = suggested_path
        index = 1
        while candidate.exists():
            candidate = suggested_path.with_name(f"{suggested_path.stem}_{index}{suggested_path.suffix}")
            index += 1

        try:
            path.rename(candidate)
            append_app_log(
                f"transcription_auto_rename from={str(path)!r} to={str(candidate)!r}"
            )
            return str(candidate)
        except Exception as exc:
            append_app_log(f"transcription_auto_rename_failed path={str(path)!r} error={exc!r}")
            return transcript_path

    def _mlx_whisper_path(self) -> str:
        return self.mlx_whisper_path.strip() or find_binary("mlx_whisper") or ""

    def _prompt_install_mlx_whisper(self, message: str):
        # Make sure the install button is on screen — the advanced section
        # is collapsed by default but the install button is always visible.
        answer = QMessageBox.question(
            self,
            "MLX Whisper requis",
            (
                f"{message}\n\n"
                "L'app peut l'installer automatiquement dans un environnement isolé, sans modifier Python ou Homebrew."
            ),
        )
        if answer == QMessageBox.StandardButton.Yes:
            self.install_mlx_whisper()

    def _set_progress_visible(self, visible: bool):
        self.progress.setVisible(visible)
        self.progress_global.setVisible(False)

    def _set_running_ui(self, running: bool):
        self.is_batch_running = running
        # A fresh batch should re-trigger the "auto-switch to library"
        # behaviour for its first job. Otherwise users who run several
        # batches in a row only see the switch once.
        if running:
            self._did_auto_switch_to_library = False
        self.btn_pick.setEnabled(True)
        self.btn_remove.setEnabled(not running)
        self.btn_clear.setEnabled(not running)
        self.btn_start.setEnabled(bool(self.queue_jobs) and not running)
        self.btn_transcribe.setEnabled(bool(self.queue_jobs) and not running)
        self.btn_cancel.setEnabled(running)
        self.btn_settings.setEnabled(True)
        self.btn_update.setEnabled(not running and self.update_worker is None)
        self.btn_apply_to_all.setEnabled(not running)
        self.btn_reset_preset.setEnabled(not running)
        self.btn_install_mlx.setEnabled(not running and self.mlx_install_worker is None)

    def start_encode(self):
        if not self.queue_jobs:
            return

        if not self.ffmpeg_path or not Path(self.ffmpeg_path).exists():
            QMessageBox.critical(self, "ffmpeg introuvable", "ffmpeg est requis. Ouvrez Paramètres (⚙).")
            return

        if self.current_index >= 0:
            self.save_current_job_from_controls(force_custom=False)
        self.save_global_settings()
        self.batch_mode = "compress_transcribe" if self.check_transcribe_after_encode.isChecked() else "compress"
        if self.batch_mode == "compress_transcribe":
            mlx_path = self._mlx_whisper_path()
            if not mlx_path or not Path(mlx_path).exists():
                self._prompt_install_mlx_whisper(
                    "La transcription après compression est activée, mais MLX Whisper n'est pas encore installé."
                )
                return
            self.mlx_whisper_path = mlx_path
            self._refresh_transcription_summary()

        out_dir = Path(self.edit_output_dir.text().strip() or str(Path.home() / "Desktop"))
        out_dir.mkdir(parents=True, exist_ok=True)
        self.edit_output_dir.setText(str(out_dir))

        for idx, job in enumerate(self.queue_jobs):
            if not self._validate_trim(job):
                QMessageBox.warning(self, "Rognage invalide", f"Le rognage est invalide pour {Path(job.input_path).name}.")
                self.queue_list.setCurrentRow(idx)
                return

            self.queue_jobs[idx].output_path = default_out_path(job.input_path, str(out_dir), self.edit_suffix.text().strip())
            self.queue_jobs[idx].transcript_path = (
                self._transcription_output_path(job, out_dir) if self.batch_mode == "compress_transcribe" else ""
            )
            self.queue_jobs[idx].status = "pending"
            self.queue_jobs[idx].error_message = ""

        self.pending_indices = list(range(len(self.queue_jobs)))
        self._set_all_job_run_modes(self.batch_mode)
        self.completed_count = 0
        self.failed_count = 0
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress_global.setRange(0, 100)
        self.progress_global.setValue(0)
        self._set_progress_visible(True)
        self.refresh_all_queue_items()
        self._set_running_ui(True)
        self._start_next_job()

    def start_transcription(self):
        if not self.queue_jobs:
            return

        if not self.ffmpeg_path or not Path(self.ffmpeg_path).exists():
            QMessageBox.critical(self, "ffmpeg introuvable", "ffmpeg est requis pour préparer l'audio.")
            return

        mlx_path = self._mlx_whisper_path()
        if not mlx_path or not Path(mlx_path).exists():
            self._prompt_install_mlx_whisper(
                "MLX Whisper n'est pas encore installé pour la transcription locale."
            )
            return
        self.mlx_whisper_path = mlx_path
        self._refresh_transcription_summary()

        if self.current_index >= 0:
            self.save_current_job_from_controls(force_custom=False)
        self.save_global_settings()
        self.batch_mode = "transcribe"

        out_dir = Path(self.edit_output_dir.text().strip() or str(Path.home() / "Desktop"))
        out_dir.mkdir(parents=True, exist_ok=True)
        self.edit_output_dir.setText(str(out_dir))

        for idx, job in enumerate(self.queue_jobs):
            if not self._validate_trim(job):
                QMessageBox.warning(self, "Rognage invalide", f"Le rognage est invalide pour {Path(job.input_path).name}.")
                self.queue_list.setCurrentRow(idx)
                return

            self.queue_jobs[idx].transcript_path = self._transcription_output_path(job, out_dir)
            self.queue_jobs[idx].status = "pending"
            self.queue_jobs[idx].error_message = ""

        self.pending_indices = list(range(len(self.queue_jobs)))
        self._set_all_job_run_modes(self.batch_mode)
        self.completed_count = 0
        self.failed_count = 0
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress_global.setRange(0, 100)
        self.progress_global.setValue(0)
        self._set_progress_visible(True)
        self.refresh_all_queue_items()
        self._set_running_ui(True)
        self._start_next_job()

    def relaunch_job_with_steps(self, job_id: int, steps: list[str]):
        """
        Take an existing library job, reset whatever artefacts the user
        wants to redo, and start a single-job batch on it. Driven from
        LibraryView's "Relancer / Réparer" popup so a relaunch actually
        runs even when the queue is otherwise empty. If another batch is
        already running, append the repaired job to the pending queue.
        """
        record = self.db.get_job(job_id)
        if not record:
            return

        # Reconstruct a QueueJob from what the DB already knows about
        # this entry. Settings are preserved (so the user's custom
        # profile survives a relaunch); paths are reset for the
        # specific steps they ticked.
        settings = record.get("settings") or {}
        # The settings_json was written from asdict(QueueJob(...)) at
        # creation; coerce missing fields to defaults via the dataclass.
        try:
            job = QueueJob(**settings)
        except TypeError:
            job = QueueJob(input_path=str(record.get("source_path") or ""))
        job.input_path = str(record.get("source_path") or job.input_path)
        job.workspace_dir = str(record.get("workspace_dir") or "")
        job.db_id = job_id
        job.status = "pending"
        job.error_message = ""

        # Plan the run: which worker to dispatch and what status to
        # leave on the DB so the resumption logic in TranscribeWorker
        # picks up at the right step.
        wants_compress = "compression" in steps
        wants_transcript = "transcription" in steps
        wants_enhanced = "enhanced" in steps
        wants_review = "review" in steps

        if wants_compress and (wants_transcript or wants_enhanced or wants_review):
            mode = "compress_transcribe"
            db_status = "PENDING"
        elif wants_compress:
            mode = "compress"
            db_status = "PENDING"
        elif wants_transcript:
            mode = "transcribe"
            db_status = "PENDING"
        elif wants_enhanced or wants_review:
            # Skip audio extraction + Whisper, reuse the existing
            # artefacts and only redo the LLM passes.
            mode = "transcribe"
            db_status = "WHISPER_DONE"
            # Override the output path to point at the existing
            # transcript so TranscribeWorker doesn't write a new file.
            existing = record.get("transcript_path") or record.get("output_path")
            if existing:
                job.transcript_path = existing
        else:
            return

        # Forget the previous artefact paths the user asked to redo so
        # the library "Ouvrir" buttons go back to "—" until the new
        # files land.
        if wants_compress:
            self.db.update_job_artefact(job_id, "compressed", "")
        if wants_transcript:
            for kind in ("transcript", "enhanced_transcript", "review"):
                self.db.update_job_artefact(job_id, kind, "")
        elif wants_enhanced:
            self.db.update_job_artefact(job_id, "enhanced_transcript", "")
        if wants_review:
            self.db.update_job_artefact(job_id, "review", "")

        self.db.update_job_status(job_id, db_status, "")
        self.db.update_job_progress(job_id, step="Relance demandée…", progress_pct=0, eta_seconds=None)

        # When the user opted into the multimodal recheck via the popup,
        # honour it for this run regardless of the saved setting.
        prev_recheck = self.transcription_audio_recheck_enabled
        if wants_review:
            self.transcription_audio_recheck_enabled = True

        out_dir = Path(self.edit_output_dir.text().strip() or str(Path.home() / "Desktop"))
        out_dir.mkdir(parents=True, exist_ok=True)
        if mode in ("compress", "compress_transcribe") and not job.output_path:
            job.output_path = default_out_path(job.input_path, str(out_dir), self.edit_suffix.text().strip())
        if mode in ("transcribe", "compress_transcribe") and not job.transcript_path:
            job.transcript_path = self._transcription_output_path(job, out_dir)

        if self.is_batch_running:
            self.queue_jobs.append(job)
            self.queue_list.addItem(QListWidgetItem())
            new_idx = len(self.queue_jobs) - 1
            self.job_run_modes[id(job)] = mode
            self.pending_indices.append(new_idx)
            self.refresh_queue_item(new_idx)
            self.library_view.refresh()
            self.status.setText(f"Relance ajoutée à la file: {Path(job.input_path).name}")
            if wants_review and prev_recheck is not None:
                self.transcription_audio_recheck_enabled = prev_recheck
            return

        # Replace the queue with this single job and kick off the
        # appropriate batch.
        self.queue_jobs = [job]
        self.queue_list.clear()
        self.queue_list.addItem(QListWidgetItem())
        self.refresh_queue_item(0)
        self.batch_mode = mode
        self.job_run_modes = {id(job): mode}

        self.pending_indices = [0]
        self.completed_count = 0
        self.failed_count = 0
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress_global.setRange(0, 100)
        self.progress_global.setValue(0)
        self._set_progress_visible(True)
        self._set_running_ui(True)
        # Restore the user's recheck preference once the worker has
        # captured the current value via _start_transcribe_worker.
        self._pending_recheck_restore = (prev_recheck if wants_review else None)
        self._start_next_job()
        if wants_review and prev_recheck is not None:
            # Restore the original recheck preference now that the
            # worker has been spawned with the override value.
            self.transcription_audio_recheck_enabled = prev_recheck

        # Library refresh so the row shows "En cours…" immediately.
        self.library_view.refresh()

    def _start_next_job(self):
        if not self.pending_indices:
            self._finish_batch(cancelled=False)
            return

        idx = self.pending_indices.pop(0)
        self.running_index = idx
        job = self.queue_jobs[idx]
        self.current_job_mode = self._job_run_mode(job)
        job.status = "running"
        job.error_message = ""
        self.current_job_started_at = time.monotonic()
        self.current_job_status_text = ""
        self.current_job_display_pct = 0
        self.refresh_queue_item(idx)
        self.queue_list.setCurrentRow(idx)

        # The library is now the real workspace: as soon as a job starts,
        # mark it RUNNING in the DB so the table can show a spinner and
        # disable the artefact buttons until they're produced.
        if job.db_id:
            self.db.update_job_status(job.db_id, "RUNNING")
            self.db.update_job_progress(
                job.db_id,
                step="Démarrage…",
                progress_pct=0,
                eta_seconds=None,
            )
        self.library_view.refresh()
        # Once a job is RUNNING the user expects to watch it in the
        # library, not stare at the queue. Auto-switch on the first job.
        if not getattr(self, "_did_auto_switch_to_library", False):
            try:
                self.tabs.setCurrentWidget(self.library_view)
            except Exception:
                pass
            self._did_auto_switch_to_library = True

        current = self.completed_count + self.failed_count + 1
        total = len(self.queue_jobs)
        self.status.setText(f"Traitement {current}/{total}: {Path(job.input_path).name}")

        if self.current_job_mode == "transcribe":
            self.current_job_progress_offset = 0
            self.current_job_progress_scale = 100
            self._start_transcribe_worker(idx)
            return

        self.current_job_progress_offset = 0
        self.current_job_progress_scale = 50 if self.current_job_mode == "compress_transcribe" else 100
        worker = EncodeWorker(self.ffmpeg_path, self.ffprobe_path, replace(job))
        self.worker = worker
        self._track_worker(worker)
        worker.progress.connect(self.on_progress)
        worker.status.connect(self.on_status)
        worker.duration_unknown.connect(self.on_duration_unknown)
        worker.finished_ok.connect(self.on_done)
        worker.failed.connect(self.on_fail)
        worker.start()

    def _start_transcribe_worker(self, idx: int):
        job = self.queue_jobs[idx]
        venv_python = managed_venv_python_path()
        worker = TranscribeWorker(
            self.ffmpeg_path,
            self.ffprobe_path,
            replace(job),
            self._mlx_whisper_path(),
            self.transcription_model,
            self.transcription_language,
            self.transcription_format,
            self._transcription_glossary(job),
            self.transcription_enhance_audio,
            self.db,
            diarization_enabled=self.transcription_diarization_enabled,
            hf_token=self.transcription_hf_token,
            venv_python_path=str(venv_python) if venv_python.exists() else "",
            text_llm_model=self.transcription_text_llm_model,
            audio_llm_model=self.transcription_audio_llm_model,
            audio_recheck_enabled=self.transcription_audio_recheck_enabled,
        )
        self.worker = worker
        self._track_worker(worker)
        worker.progress.connect(self.on_progress)
        worker.status.connect(self.on_status)
        worker.duration_unknown.connect(self.on_duration_unknown)
        worker.finished_ok.connect(self.on_transcription_done)
        worker.failed.connect(self.on_fail)
        worker.start()

    def _track_worker(self, worker: QThread):
        self._active_workers.append(worker)
        worker.finished.connect(lambda w=worker: self._release_worker(w))

    def _release_worker(self, worker: QThread):
        if self.worker is worker:
            self.worker = None
        try:
            self._active_workers.remove(worker)
        except ValueError:
            pass

    def on_duration_unknown(self, unknown: bool):
        if unknown:
            self.progress.setRange(0, 0)

    def _status_timing_suffix(self) -> str:
        if self.current_job_started_at is None:
            return ""
        elapsed = time.monotonic() - self.current_job_started_at
        parts = [f"écoulé {format_compact_seconds(elapsed)}"]
        if self.progress.maximum() > 0 and 2 <= self.current_job_display_pct < 100:
            remaining = elapsed * (100 / self.current_job_display_pct - 1)
            parts.append(f"reste ~{format_compact_seconds(remaining)}")
        return f" ({' · '.join(parts)})"

    def _set_status_with_progress_context(self):
        if not self.is_batch_running or self.running_index is None:
            return
        current = self.completed_count + self.failed_count + (1 if self.running_index is not None else 0)
        total = max(1, len(self.queue_jobs))
        text = self.current_job_status_text or "Traitement…"
        self.status.setText(f"[{current}/{total}] {text}{self._status_timing_suffix()}")

    def on_status(self, text: str):
        self.current_job_status_text = text
        self._set_status_with_progress_context()
        # Forward the human-readable step to the DB so the library
        # column reflects exactly what the worker is doing right now.
        if self.running_index is not None:
            job = self.queue_jobs[self.running_index]
            if job.db_id:
                self.db.update_job_progress(job.db_id, step=text)
                self.library_view.tick_running_row(job.db_id)

    def on_progress(self, pct: int):
        if self.progress.maximum() == 0:
            self.progress.setRange(0, 100)
        display_pct = int(
            min(100, max(0, self.current_job_progress_offset + (pct * self.current_job_progress_scale / 100.0)))
        )
        self.current_job_display_pct = display_pct
        self.progress.setValue(display_pct)

        total = max(1, len(self.queue_jobs))
        done = self.completed_count + self.failed_count
        global_pct = int(min(100, ((done + display_pct / 100.0) / total) * 100))
        self.progress_global.setValue(global_pct)
        self._set_status_with_progress_context()

        # Same idea as on_status: push the percentage + ETA so the live
        # library row stays in sync with the progress bar at the bottom.
        if self.running_index is not None:
            job = self.queue_jobs[self.running_index]
            if job.db_id:
                eta = None
                if self.current_job_started_at is not None and 2 <= display_pct < 100:
                    elapsed = time.monotonic() - self.current_job_started_at
                    eta = elapsed * (100 / display_pct - 1)
                self.db.update_job_progress(
                    job.db_id, progress_pct=display_pct, eta_seconds=eta
                )
                self.library_view.tick_running_row(job.db_id)

    def cancel_encode(self):
        self.pending_indices.clear()
        if self.worker:
            self.worker.request_stop()
            self.btn_cancel.setEnabled(False)

    def _cleanup_workspace_wav(self, job: QueueJob):
        # The audio.wav we extract for transcription is the bulky leftover
        # in the workspace once the job is done. Drop it; keep everything
        # else so a re-run with a different model is still possible.
        if not job.workspace_dir:
            return
        wav_path = Path(job.workspace_dir) / "audio.wav"
        if wav_path.exists():
            try:
                wav_path.unlink()
            except Exception:
                pass

    def on_done(self, out_path: str):
        if self.running_index is None:
            return
        idx = self.running_index
        job = self.queue_jobs[idx]
        mode = self._job_run_mode(job)
        if mode == "compress_transcribe":
            job.output_path = out_path
            self.current_job_progress_offset = 50
            self.current_job_progress_scale = 50
            self.progress.setRange(0, 100)
            self.progress.setValue(50)
            self.status.setText(f"Compression terminée, transcription: {Path(job.input_path).name}")
            self._start_transcribe_worker(idx)
            return

        job.status = "done"
        job.error_message = ""
        job.output_path = out_path
        self.completed_count += 1
        self.running_index = None

        # EncodeWorker doesn't talk to the DB itself, so update from here.
        # TranscribeWorker updates the DB during run(); this is the
        # compression-only path.
        if job.db_id:
            self.db.update_job_status(job.db_id, "COMPLETED")
            self.db.update_job_output(job.db_id, out_path)
            self.db.update_job_artefact(job.db_id, "compressed", out_path)
            self.db.update_job_progress(
                job.db_id, step="Compression terminée", progress_pct=100, eta_seconds=0
            )

        self.refresh_queue_item(idx)
        self.progress.setRange(0, 100)
        self.progress.setValue(100)
        self.library_view.refresh()

        self._cleanup_workspace_wav(job)

        total = max(1, len(self.queue_jobs))
        self.progress_global.setValue(int(((self.completed_count + self.failed_count) / total) * 100))
        self._start_next_job()

    def on_transcription_done(self, transcript_path: str):
        if self.running_index is None:
            return
        idx = self.running_index
        job = self.queue_jobs[idx]
        job.status = "done"
        job.error_message = ""
        # TranscribeWorker owns final naming and DB persistence. Renaming
        # again here would desynchronise the path stored in the library.
        job.transcript_path = transcript_path
        job.output_path = transcript_path
        self.completed_count += 1
        self.running_index = None

        if job.db_id:
            self.db.update_job_status(job.db_id, "COMPLETED")
            self.db.update_job_output(job.db_id, transcript_path)

        self.refresh_queue_item(idx)
        self.progress.setRange(0, 100)
        self.progress.setValue(100)
        self.library_view.refresh()
        
        self._cleanup_workspace_wav(job)

        total = max(1, len(self.queue_jobs))
        self.progress_global.setValue(int(((self.completed_count + self.failed_count) / total) * 100))
        self._start_next_job()

    def on_fail(self, msg: str):
        if self.running_index is None:
            return

        idx = self.running_index
        job = self.queue_jobs[idx]
        self.running_index = None
        append_app_log(f"queue_failure file={Path(job.input_path).name!r} message={_tail_for_log(msg)!r}")

        if msg == "Annulé.":
            job.status = "failed"
            job.error_message = "Annulé"
            if job.db_id:
                self.db.update_job_status(job.db_id, "CANCELLED", "Annulé")
            self.refresh_queue_item(idx)
            self.progress.setRange(0, 100)
            self._finish_batch(cancelled=True)
            return

        job.status = "failed"
        job.error_message = msg
        self.failed_count += 1
        # TranscribeWorker may already have set FAILED with a richer message,
        # but EncodeWorker never touches the DB. Update unconditionally —
        # update_job_status overwrites with the latest, so this is safe.
        if job.db_id:
            self.db.update_job_status(job.db_id, "FAILED", msg)
        self.refresh_queue_item(idx)
        self.progress.setRange(0, 100)
        self.library_view.refresh()

        if not self.check_continue_on_error.isChecked():
            self.pending_indices.clear()

        self._start_next_job()

    def _finish_batch(self, cancelled: bool):
        self._set_running_ui(False)

        total = len(self.queue_jobs)
        done = self.completed_count
        failed = self.failed_count

        if cancelled:
            self.status.setText("Traitement annulé.")
            self._set_progress_visible(False)
            QMessageBox.warning(self, "Annulé", f"Traitement annulé. Réussis: {done}, Erreurs: {failed}.")
            return

        self.progress.setRange(0, 100)
        self.progress.setValue(100 if done else 0)
        self.progress_global.setValue(100 if total else 0)
        self._set_progress_visible(False)

        if failed == 0:
            self.status.setText("Terminé.")
            QMessageBox.information(
                self,
                "Terminé",
                f"{done}/{total} vidéo(s) traitée(s) avec succès.\nDossier: {self.edit_output_dir.text().strip()}",
            )
            return

        gatekeeper_hint = ""
        if any(
            keyword in (job.error_message or "").lower()
            for keyword in ["operation not permitted", "permission denied", "not authorized"]
            for job in self.queue_jobs
        ):
            gatekeeper_hint = (
                "\n\nAide macOS: si ffmpeg est bloqué, exécutez:\n"
                "xattr -dr com.apple.quarantine /Applications/EkoVideoCompressor.app"
            )

        preview = "\n".join(
            f"- {Path(job.input_path).name}: {job.error_message[:120]}" for job in self.queue_jobs if job.status == "failed"
        )
        self.status.setText("Terminé avec erreurs.")
        QMessageBox.warning(
            self,
            "Terminé avec erreurs",
            f"Réussis: {done}/{total}\nErreurs: {failed}\n\n{preview}{gatekeeper_hint}",
        )


def main():
    if "--version" in sys.argv:
        print(APP_VERSION)
        return
    if "--smoke-test" in sys.argv:
        print(f"{APP_NAME} {APP_VERSION} ok")
        return
    if "--startup-smoke-test" in sys.argv:
        with tempfile.TemporaryDirectory(prefix="ekovideo-startup-") as tmp_dir:
            os.environ["EKO_APP_SUPPORT_DIR"] = tmp_dir
            DatabaseManager(Path(tmp_dir) / "library.db").create_job(
                source_path="/tmp/ekovideo-startup-smoke.mp4",
                workspace_dir=str(Path(tmp_dir) / "Workspace"),
                settings=asdict(QueueJob(input_path="/tmp/ekovideo-startup-smoke.mp4")),
            )
            app = QApplication([sys.argv[0]])
            app.setOrganizationName(ORG_NAME)
            app.setOrganizationDomain(ORG_DOMAIN)
            app.setApplicationName(APP_NAME)
            window = MainWindow()
            window.close()
            print(f"{APP_NAME} {APP_VERSION} startup ok")
        return

    app = QApplication(sys.argv)
    app.setOrganizationName(ORG_NAME)
    app.setOrganizationDomain(ORG_DOMAIN)
    app.setApplicationName(APP_NAME)

    window = MainWindow()
    # Default to maximized on first launch — the library table + right
    # settings panel breathe much better with the full screen, and the
    # team's flow is to leave the app open for the duration of a batch.
    # showMaximized() is honoured on macOS / Windows / X11; on Wayland
    # it falls back to the stored geometry if the WM rejects it.
    window.showMaximized()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
