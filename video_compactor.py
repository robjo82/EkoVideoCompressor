import json
import os
import re
import shlex
import shutil
import ssl
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass, replace
from pathlib import Path

import certifi
from ffmpeg_utils import build_ffmpeg_cmd
from transcription_utils import (
    assign_speakers_to_segments,
    build_audio_extract_cmd,
    build_diarization_cmd,
    build_mlx_whisper_cmd,
    default_transcript_path,
    parse_diarization_output,
    parse_whisper_json_segments,
    render_segments_with_speakers,
    structured_initial_prompt,
    transcript_output_ext,
)
from PySide6.QtCore import QSettings, QThread, QTime, Qt, Signal
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
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QScrollArea,
    QSplitter,
    QSlider,
    QSpinBox,
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

VIDEO_FILTER = "Vidéos (*.mp4 *.mov *.mkv *.m4v *.avi *.webm);;Tous les fichiers (*.*)"
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
SEMVER_RE = re.compile(r"^v?\.?(\d+)\.(\d+)\.(\d+)$")


@dataclass
class QueueJob:
    input_path: str
    output_path: str = ""
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
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / APP_NAME
    return Path.home() / f".{APP_NAME.replace(' ', '').lower()}"


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


def default_out_path(in_path: str, out_dir: str, suffix: str) -> str:
    source = Path(in_path)
    safe_suffix = suffix.strip() or "_compressed"
    base = Path(out_dir) / f"{source.stem}{safe_suffix}.mp4"
    out = base
    i = 1
    while out.exists():
        out = Path(out_dir) / f"{source.stem}{safe_suffix}_{i}.mp4"
        i += 1
    return str(out)


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

        self.transcription_model_edit = QLineEdit(str(transcription_settings.get("model", "")))
        transcription_form.addRow("Modèle", self.transcription_model_edit)

        self.transcription_language_combo = QComboBox()
        self.transcription_language_combo.addItems(["fr", "auto", "en", "es", "de", "it"])
        self.transcription_language_combo.setCurrentText(str(transcription_settings.get("language", "fr")))
        transcription_form.addRow("Langue", self.transcription_language_combo)

        self.transcription_format_combo = QComboBox()
        self.transcription_format_combo.addItems(["txt", "srt", "vtt", "json", "all"])
        self.transcription_format_combo.setCurrentText(str(transcription_settings.get("format", "txt")))
        transcription_form.addRow("Format", self.transcription_format_combo)

        self.transcription_suffix_edit = QLineEdit(str(transcription_settings.get("suffix", "_transcription")))
        transcription_form.addRow("Suffixe", self.transcription_suffix_edit)

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

        self.transcription_hf_token_edit = QLineEdit(
            str(transcription_settings.get("hf_token", ""))
        )
        self.transcription_hf_token_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.transcription_hf_token_edit.setPlaceholderText(
            "hf_xxxxxxxx (Hugging Face, requis pour la détection des locuteurs)"
        )
        transcription_form.addRow("Token Hugging Face", self.transcription_hf_token_edit)

        hf_hint = QLabel(
            "Token requis pour la détection des locuteurs. Créez-le sur "
            "huggingface.co/settings/tokens (Read access) APRÈS avoir accepté "
            "les licences :\n"
            "• huggingface.co/pyannote/segmentation-3.0\n"
            "• huggingface.co/pyannote/speaker-diarization-3.1"
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

        self.setMinimumWidth(660)
        self.setStyleSheet(
            """
        QDialog {
            background: #f5f5f7;
            color: #1d1d1f;
            font-family: ".AppleSystemUIFont", "Helvetica Neue", "Arial", sans-serif;
            font-size: 14px;
        }
        QLabel { background: transparent; color: #1d1d1f; }
        QLineEdit, QComboBox {
            background: #ffffff;
            border: 1px solid #d2d2d7;
            border-radius: 9px;
            padding: 8px 10px;
            min-height: 24px;
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
        """
        )

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

    def values(self) -> tuple[str, str, str, dict[str, str | bool]]:
        return (
            self.ffmpeg_edit.text().strip(),
            self.ffprobe_edit.text().strip(),
            self.token_edit.text().strip(),
            {
                "mlx_whisper_path": self.mlx_whisper_edit.text().strip(),
                "model": self.transcription_model_edit.text().strip(),
                "language": self.transcription_language_combo.currentText(),
                "format": self.transcription_format_combo.currentText(),
                "suffix": self.transcription_suffix_edit.text().strip(),
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
                            self.status.emit(f"Compression… {pct}%")
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
        diarization_enabled: bool = False,
        hf_token: str = "",
        venv_python_path: str = "",
    ):
        super().__init__()
        self.ffmpeg_path = ffmpeg_path
        self.ffprobe_path = ffprobe_path
        self.job = job
        self.mlx_whisper_path = mlx_whisper_path
        self.model = model
        self.language = language
        self.output_format = output_format
        # Wrap the user's "Contexte" in a French priming sentence so Whisper
        # treats the listed terms as expected vocabulary, not a token salad.
        self.initial_prompt = structured_initial_prompt(initial_prompt)
        self.enhance_audio = enhance_audio
        self.diarization_enabled = diarization_enabled
        self.hf_token = hf_token
        self.venv_python_path = venv_python_path
        self._stop_requested = False
        self._proc: subprocess.Popen | None = None

    def request_stop(self):
        self._stop_requested = True
        if self._proc:
            try:
                self._proc.kill()
            except Exception:
                pass

    def run(self):
        if not self.ffmpeg_path or not Path(self.ffmpeg_path).exists():
            self.failed.emit("ffmpeg introuvable. Vérifiez les paramètres.")
            return
        if not self.mlx_whisper_path or not Path(self.mlx_whisper_path).exists():
            self.failed.emit(
                "mlx_whisper introuvable. Utilisez le bouton Installer MLX Whisper dans l'onglet Transcrire."
            )
            return
        if not Path(self.job.input_path).exists():
            self.failed.emit("Fichier d'entrée introuvable.")
            return
        if not self.job.transcript_path:
            self.failed.emit("Chemin de transcription manquant.")
            return

        duration = probe_duration_seconds(self.job.input_path, self.ffprobe_path, self.ffmpeg_path)
        self.duration_unknown.emit(duration is None)

        tmp_dir = Path(tempfile.mkdtemp(prefix="ekovideo-transcribe-"))
        wav_path = tmp_dir / "audio.wav"
        try:
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
            self._proc = subprocess.Popen(
                extract_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True,
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
                            self.status.emit(f"Préparation audio… {pct}%")
                    else:
                        self.status.emit("Préparation audio…")

                elif line.startswith("progress=end"):
                    break

            rc = self._proc.wait()
            if rc != 0:
                err = ""
                try:
                    err = (self._proc.stderr.read() or "").strip()
                except Exception:
                    pass
                self.failed.emit(err or f"ffmpeg a échoué (code {rc}).")
                return

            if self._stop_requested:
                self.failed.emit("Annulé.")
                return

            out_path = Path(self.job.transcript_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)

            # When diarisation is on, we always need Whisper's JSON output
            # (segments with timestamps) — we'll fuse it with pyannote turns
            # and re-render to the user's chosen format ourselves.
            run_diarization = bool(self.diarization_enabled and self.hf_token and self.venv_python_path)
            if self.diarization_enabled and not run_diarization:
                self.status.emit(
                    "Détection des locuteurs ignorée (token Hugging Face ou installation manquante)."
                )

            whisper_target_dir = tmp_dir if run_diarization else out_path.parent
            whisper_target_stem = "whisper" if run_diarization else out_path.stem
            whisper_format = "json" if run_diarization else self.output_format
            whisper_target = whisper_target_dir / f"{whisper_target_stem}.{transcript_output_ext(whisper_format)}"
            cmd = build_mlx_whisper_cmd(
                mlx_whisper_path=self.mlx_whisper_path,
                audio_path=str(wav_path),
                output_path=str(whisper_target),
                model=self.model,
                language=self.language,
                output_format=whisper_format,
                initial_prompt=self.initial_prompt,
            )

            self.duration_unknown.emit(True)
            self.progress.emit(25)
            self.status.emit("Transcription locale MLX…")
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            stdout, stderr = self._proc.communicate()

            if self._stop_requested:
                self.failed.emit("Annulé.")
                return

            if self._proc.returncode != 0:
                detail = (stderr or stdout or "").strip()
                self.failed.emit(detail or f"mlx_whisper a échoué (code {self._proc.returncode}).")
                return

            if run_diarization:
                self.progress.emit(75)
                self.status.emit("Détection des locuteurs (pyannote)…")
                diar_cmd = build_diarization_cmd(self.venv_python_path, str(wav_path))
                env = os.environ.copy()
                env["HF_TOKEN"] = self.hf_token
                self._proc = subprocess.Popen(
                    diar_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=env,
                )
                d_stdout, d_stderr = self._proc.communicate()

                if self._stop_requested:
                    self.failed.emit("Annulé.")
                    return

                if self._proc.returncode != 0:
                    detail = (d_stderr or d_stdout or "").strip()
                    self.failed.emit(
                        f"Détection des locuteurs échouée (code {self._proc.returncode}). {detail}"
                    )
                    return

                try:
                    turns = parse_diarization_output(d_stdout)
                    whisper_segments = parse_whisper_json_segments(str(whisper_target))
                    fused = assign_speakers_to_segments(whisper_segments, turns)
                    rendered = render_segments_with_speakers(fused, self.output_format)
                except Exception as exc:
                    self.failed.emit(f"Fusion transcription/locuteurs échouée: {exc}")
                    return

                out_path.write_text(rendered, encoding="utf-8")

            self.duration_unknown.emit(False)
            self.progress.emit(100)
            self.status.emit("Transcription terminée.")
            self.finished_ok.emit(str(out_path))
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            self._proc = None
            shutil.rmtree(tmp_dir, ignore_errors=True)


class MlxWhisperInstallWorker(QThread):
    status = Signal(str)
    finished_ok = Signal(str)
    failed = Signal(str)

    def __init__(self, python_path: str, venv_dir: Path, install_diarization: bool = True):
        super().__init__()
        self.python_path = python_path
        self.venv_dir = venv_dir
        self.install_diarization = install_diarization

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

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)

        icon = QLabel("⌘")
        icon.setObjectName("dropIcon")
        icon.setAlignment(Qt.AlignCenter)

        title = QLabel("Déposez vos vidéos")
        title.setObjectName("dropTitle")
        title.setAlignment(Qt.AlignCenter)

        subtitle = QLabel('ou utilisez "Ajouter des vidéos"')
        subtitle.setObjectName("dropSubtitle")
        subtitle.setAlignment(Qt.AlignCenter)

        details = QLabel("Compression par lots, transcription locale en option")
        details.setObjectName("dropDetails")
        details.setAlignment(Qt.AlignCenter)

        layout.addWidget(icon)
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addWidget(details)

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


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        self.setMinimumSize(1080, 700)

        self.settings = QSettings(ORG_NAME, APP_NAME)
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
        self.transcription_model = str(
            self.settings.value("transcription_model", "mlx-community/whisper-large-v3-turbo", type=str)
        ).strip()
        self.transcription_language = str(self.settings.value("transcription_language", "fr", type=str)).strip() or "fr"
        self.transcription_format = str(self.settings.value("transcription_format", "txt", type=str)).strip() or "txt"
        self.transcription_suffix = (
            str(self.settings.value("transcription_suffix", "_transcription", type=str)).strip()
            or "_transcription"
        )
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
        self.current_job_progress_offset = 0
        self.current_job_progress_scale = 100
        self.update_worker: UpdateWorker | None = None
        self._last_update_info: dict | None = None
        self._update_phase: str = ""
        self._update_check_payload: dict | None = None
        self._update_download_payload: tuple[str, dict] | None = None
        self._update_error_message: str | None = None
        self.is_batch_running = False
        self._syncing_job = False

        self._build_ui(icon_path)
        self.apply_style()

        if not self.ffmpeg_path:
            self.status.setText("ffmpeg non détecté. Ouvrez Paramètres (⚙).")

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
        self.h2 = QLabel(f"Compression réunion macOS · version {APP_VERSION}")
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

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)

        left_panel = QWidget()
        left_col = QVBoxLayout(left_panel)
        left_col.setContentsMargins(0, 0, 0, 0)
        left_col.setSpacing(10)

        self.drop = DropZone()
        self.drop.files_dropped.connect(self.add_input_files)
        self.drop.clicked.connect(self.pick_files)
        left_col.addWidget(self.drop, 1)

        queue_actions = QHBoxLayout()
        queue_actions.setSpacing(8)
        self.btn_pick = QPushButton("Ajouter des vidéos…")
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
        left_col.addLayout(queue_actions)

        self.queue_list = QListWidget()
        self.queue_list.setObjectName("queueList")
        self.queue_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.queue_list.currentRowChanged.connect(self.on_queue_selection_changed)
        left_col.addWidget(self.queue_list, 1)

        self.lbl_input_meta = QLabel("Aucune vidéo sélectionnée.")
        self.lbl_input_meta.setObjectName("metaLabel")
        self.lbl_input_meta.setWordWrap(True)
        left_col.addWidget(self.lbl_input_meta)

        # Right panel: single scrollable column instead of 4 tabs.
        # Per-job essentials are visible by default; technical overrides
        # (resolution, bitrate, trim) are folded into a collapsible section
        # so the team's day-to-day flow stays uncluttered.
        right_col = QFrame()
        right_col.setObjectName("settingsPanel")
        right_col.setMinimumWidth(430)
        right_layout = QVBoxLayout(right_col)
        right_layout.setContentsMargins(12, 12, 12, 12)
        right_layout.setSpacing(10)

        scroll = QScrollArea()
        scroll.setObjectName("settingsScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll_body = QFrame()
        scroll_body.setObjectName("settingsScrollBody")
        scroll_layout = QVBoxLayout(scroll_body)
        scroll_layout.setContentsMargins(4, 4, 4, 4)
        scroll_layout.setSpacing(12)

        # --- Profil + sortie -----------------------------------------
        main_group = QGroupBox("Réglages de cette vidéo")
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

        self.edit_suffix = QLineEdit(str(self.settings.value("suffix", "_compressed", type=str)))
        main_form.addRow("Suffixe", self.edit_suffix)

        self.check_continue_on_error = QCheckBox("Continuer en cas d'erreur")
        self.check_continue_on_error.setChecked(self.settings.value("continue_on_error", True, type=bool))
        main_form.addRow("", self.check_continue_on_error)

        self.check_transcribe_after_encode = QCheckBox("Transcrire après compression")
        self.check_transcribe_after_encode.setChecked(self.settings.value("transcribe_after_encode", False, type=bool))
        main_form.addRow("", self.check_transcribe_after_encode)

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
        self.edit_transcription_prompt.setMinimumHeight(110)
        glossary_layout.addWidget(self.edit_transcription_prompt)

        self.lbl_transcription_hint = QLabel(
            "Conservé entre les réunions. Sera transmis à Whisper comme vocabulaire attendu."
        )
        self.lbl_transcription_hint.setObjectName("metaLabel")
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
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([640, 440])

        root.addWidget(splitter, 1)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setVisible(False)
        root.addWidget(self.progress)

        self.progress_global = QProgressBar()
        self.progress_global.setRange(0, 100)
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
        self.setStyleSheet(
            """
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
        QComboBox, QSpinBox, QTimeEdit, QLineEdit, QTextEdit {
            background: #ffffff;
            border: 1px solid #d2d2d7;
            border-radius: 9px;
            padding: 7px 10px;
            min-height: 24px;
            color: #1d1d1f;
        }
        QTextEdit {
            selection-background-color: #b8d7ff;
        }
        QComboBox::drop-down {
            border: none;
            width: 22px;
        }
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
        """
        )

    def _transcription_settings_payload(self) -> dict[str, str | bool]:
        return {
            "mlx_whisper_path": self.mlx_whisper_path,
            "model": self.transcription_model,
            "language": self.transcription_language,
            "format": self.transcription_format,
            "suffix": self.transcription_suffix,
            "enhance_audio": self.transcription_enhance_audio,
            "diarization_enabled": self.transcription_diarization_enabled,
            "hf_token": self.transcription_hf_token,
        }

    def _refresh_transcription_summary(self):
        mlx_status = "installé" if self._mlx_whisper_path() and Path(self._mlx_whisper_path()).exists() else "à installer"
        if self.transcription_diarization_enabled:
            if self.transcription_hf_token:
                diar_status = "ON"
            else:
                diar_status = "ON (token HF manquant)"
        else:
            diar_status = "OFF"
        self.lbl_transcription_config.setText(
            f"MLX Whisper: {mlx_status}\n"
            f"Modèle: {self.transcription_model or 'mlx-community/whisper-large-v3-turbo'}\n"
            f"Langue: {self.transcription_language} · Sortie: {self.transcription_format}\n"
            f"Détection des locuteurs: {diar_status}"
        )

    def _transcription_glossary(self) -> str:
        glossary = self.edit_transcription_prompt.toPlainText().strip()
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
        self.transcription_model = (
            str(transcription_settings.get("model", "")).strip() or "mlx-community/whisper-large-v3-turbo"
        )
        self.transcription_language = str(transcription_settings.get("language", "fr")).strip() or "fr"
        self.transcription_format = str(transcription_settings.get("format", "txt")).strip() or "txt"
        self.transcription_suffix = (
            str(transcription_settings.get("suffix", "_transcription")).strip() or "_transcription"
        )
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
            with zipfile.ZipFile(zip_path, "r") as archive:
                archive.extractall(tmp_dir)

            candidates = list(tmp_dir.rglob("EkoVideoCompressor.app"))
            if not candidates:
                candidates = list(tmp_dir.rglob("*.app"))
            if not candidates:
                raise RuntimeError("Archive de mise à jour invalide: app introuvable.")

            new_app_path = candidates[0].resolve()
            target_app_path = self._target_app_bundle_path()
            script_path = (tmp_dir / "apply_update.sh").resolve()
            pid = os.getpid()
            fallback_app = (Path.home() / "Applications" / "EkoVideoCompressor.app").resolve()
            log_path = (Path(tempfile.gettempdir()) / "ekovideo-updater.log").resolve()

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
                    "verify_app() {",
                    "  local app=\"$1\"",
                    "  local exe=\"$app/Contents/MacOS/EkoVideoCompressor\"",
                    "  if [ ! -x \"$exe\" ]; then",
                    "    echo \"Executable missing: $exe\"",
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
                    "  staged=\"${dst}.new\"",
                    "  backup=\"${dst}.backup.$(date +%s)\"",
                    "  mkdir -p \"$parent\" || return 1",
                    "  verify_app \"$src\" || return 1",
                    "  rm -rf \"$staged\" || return 1",
                    "  ditto --noqtn \"$src\" \"$staged\" || return 1",
                    "  xattr -cr \"$staged\" || true",
                    "  xattr -dr com.apple.quarantine \"$staged\" || true",
                    "  verify_app \"$staged\" || { rm -rf \"$staged\"; return 1; }",
                    "  if [ -d \"$dst\" ]; then",
                    "    mv \"$dst\" \"$backup\" || { rm -rf \"$staged\"; return 1; }",
                    "  fi",
                    "  if ! mv \"$staged\" \"$dst\"; then",
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

        worker = MlxWhisperInstallWorker(python_path, managed_transcription_venv_dir())
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
        paths, _ = QFileDialog.getOpenFileNames(self, "Choisir des vidéos", str(Path.home()), VIDEO_FILTER)
        if paths:
            self.add_input_files(paths)

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
        for p in paths:
            resolved = str(Path(p))
            if not resolved or resolved in existing or not Path(resolved).exists():
                continue
            job = self._new_job(resolved)
            self.queue_jobs.append(job)
            self.queue_list.addItem(QListWidgetItem())
            self.refresh_queue_item(len(self.queue_jobs) - 1)
            existing.add(resolved)
            added += 1

        if added and self.queue_list.currentRow() < 0:
            self.queue_list.setCurrentRow(0)

        self.btn_start.setEnabled(bool(self.queue_jobs) and not self.is_batch_running)
        self.btn_transcribe.setEnabled(bool(self.queue_jobs) and not self.is_batch_running)
        if added:
            self.status.setText(f"{added} fichier(s) ajouté(s).")

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
            self.lbl_input_meta.setText("Aucune vidéo sélectionnée.")
            self.update_estimation(None)

        self.btn_start.setEnabled(bool(self.queue_jobs) and not self.is_batch_running)
        self.btn_transcribe.setEnabled(bool(self.queue_jobs) and not self.is_batch_running)

    def clear_queue(self):
        if self.is_batch_running:
            return
        self.queue_jobs.clear()
        self.queue_list.clear()
        self.current_index = -1
        self.lbl_input_meta.setText("Aucune vidéo sélectionnée.")
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
        title = f"{self.status_prefix(job)} {Path(job.input_path).name} {suffix}"
        item = self.queue_list.item(idx)
        if item:
            item.setText(title)
            tooltip = job.input_path
            if job.output_path:
                tooltip += f"\nVidéo: {job.output_path}"
            if job.transcript_path:
                tooltip += f"\nTranscription: {job.transcript_path}"
            if job.error_message:
                tooltip += f"\nErreur: {job.error_message}"
            item.setToolTip(tooltip)

    def refresh_all_queue_items(self):
        for idx in range(len(self.queue_jobs)):
            self.refresh_queue_item(idx)

    def on_queue_selection_changed(self, row: int):
        self.current_index = row
        if row < 0 or row >= len(self.queue_jobs):
            self.lbl_input_meta.setText("Aucune vidéo sélectionnée.")
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
        self.btn_pick.setEnabled(not running)
        self.btn_remove.setEnabled(not running)
        self.btn_clear.setEnabled(not running)
        self.btn_start.setEnabled(bool(self.queue_jobs) and not running)
        self.btn_transcribe.setEnabled(bool(self.queue_jobs) and not running)
        self.btn_cancel.setEnabled(running)
        self.btn_settings.setEnabled(not running)
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

    def _start_next_job(self):
        if not self.pending_indices:
            self._finish_batch(cancelled=False)
            return

        idx = self.pending_indices.pop(0)
        self.running_index = idx
        job = self.queue_jobs[idx]
        job.status = "running"
        job.error_message = ""
        self.refresh_queue_item(idx)
        self.queue_list.setCurrentRow(idx)

        current = self.completed_count + self.failed_count + 1
        total = len(self.queue_jobs)
        self.status.setText(f"Traitement {current}/{total}: {Path(job.input_path).name}")

        if self.batch_mode == "transcribe":
            self.current_job_progress_offset = 0
            self.current_job_progress_scale = 100
            self._start_transcribe_worker(idx)
            return

        self.current_job_progress_offset = 0
        self.current_job_progress_scale = 50 if self.batch_mode == "compress_transcribe" else 100
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
            self._transcription_glossary(),
            self.transcription_enhance_audio,
            diarization_enabled=self.transcription_diarization_enabled,
            hf_token=self.transcription_hf_token,
            venv_python_path=str(venv_python) if venv_python.exists() else "",
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

    def on_status(self, text: str):
        current = self.completed_count + self.failed_count + (1 if self.running_index is not None else 0)
        total = max(1, len(self.queue_jobs))
        self.status.setText(f"[{current}/{total}] {text}")

    def on_progress(self, pct: int):
        if self.progress.maximum() == 0:
            self.progress.setRange(0, 100)
        display_pct = int(
            min(100, max(0, self.current_job_progress_offset + (pct * self.current_job_progress_scale / 100.0)))
        )
        self.progress.setValue(display_pct)

        total = max(1, len(self.queue_jobs))
        done = self.completed_count + self.failed_count
        global_pct = int(min(100, ((done + display_pct / 100.0) / total) * 100))
        self.progress_global.setValue(global_pct)

    def cancel_encode(self):
        self.pending_indices.clear()
        if self.worker:
            self.worker.request_stop()
            self.btn_cancel.setEnabled(False)

    def on_done(self, out_path: str):
        if self.running_index is None:
            return
        idx = self.running_index
        job = self.queue_jobs[idx]
        if self.batch_mode == "compress_transcribe":
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

        self.refresh_queue_item(idx)
        self.progress.setRange(0, 100)
        self.progress.setValue(100)

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
        job.transcript_path = transcript_path
        self.completed_count += 1
        self.running_index = None

        self.refresh_queue_item(idx)
        self.progress.setRange(0, 100)
        self.progress.setValue(100)

        total = max(1, len(self.queue_jobs))
        self.progress_global.setValue(int(((self.completed_count + self.failed_count) / total) * 100))
        self._start_next_job()

    def on_fail(self, msg: str):
        if self.running_index is None:
            return

        idx = self.running_index
        job = self.queue_jobs[idx]
        self.running_index = None

        if msg == "Annulé.":
            job.status = "failed"
            job.error_message = "Annulé"
            self.refresh_queue_item(idx)
            self.progress.setRange(0, 100)
            self._finish_batch(cancelled=True)
            return

        job.status = "failed"
        job.error_message = msg
        self.failed_count += 1
        self.refresh_queue_item(idx)
        self.progress.setRange(0, 100)

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

    app = QApplication(sys.argv)
    app.setOrganizationName(ORG_NAME)
    app.setOrganizationDomain(ORG_DOMAIN)
    app.setApplicationName(APP_NAME)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
