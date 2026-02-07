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
from PySide6.QtCore import QSettings, QThread, QTime, Qt, Signal
from PySide6.QtGui import QIcon
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
    QSplitter,
    QSlider,
    QSpinBox,
    QTabWidget,
    QTimeEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

APP_NAME = "EkoVideo Compressor"
APP_VERSION = os.getenv("APP_VERSION", "dev")
ORG_NAME = "Ekonum"
ORG_DOMAIN = "ekonum"

VIDEO_FILTER = "Vidéos (*.mp4 *.mov *.mkv *.m4v *.avi *.webm);;Tous les fichiers (*.*)"
APP_ICON_FILE = "ekovideo_icon.png"
APP_LOGO_FILE = "ekovideo_logo.png"
BUNDLE_IDENTIFIER = "com.ekonum.ekovideocompressor"
GITHUB_OWNER = "robjo82"
GITHUB_REPO = "EkoVideoCompressor"
GITHUB_LATEST_RELEASE_API = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/releases/latest"

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
SEMVER_RE = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)$")


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


def candidate_bin_paths() -> list[Path]:
    base = app_base_dir()
    return [
        base / "ffmpeg",
        base / "ffprobe",
        base / "bin" / "ffmpeg",
        base / "bin" / "ffprobe",
        base / "Resources" / "ffmpeg",
        base / "Resources" / "ffprobe",
        base / "Resources" / "bin" / "ffmpeg",
        base / "Resources" / "bin" / "ffprobe",
        Path("/opt/homebrew/bin/ffmpeg"),
        Path("/opt/homebrew/bin/ffprobe"),
        Path("/usr/local/bin/ffmpeg"),
        Path("/usr/local/bin/ffprobe"),
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
    def __init__(self, parent: QWidget, ffmpeg_path: str, ffprobe_path: str, github_token: str):
        super().__init__(parent)
        self.setWindowTitle("Paramètres")
        self.setModal(True)

        form = QFormLayout(self)

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

        form.addRow("ffmpeg", row1)
        form.addRow("ffprobe", row2)

        self.token_edit = QLineEdit(github_token or "")
        self.token_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.token_edit.setPlaceholderText("GitHub token (optionnel, requis si repo privé)")
        form.addRow("Token update", self.token_edit)

        hint = QLabel(
            "Si ffmpeg est bloqué par macOS, ouvrez Terminal puis:\n"
            "xattr -dr com.apple.quarantine /Applications/EkoVideoCompressor.app"
        )
        hint.setWordWrap(True)
        form.addRow("", hint)

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

        self.setMinimumWidth(580)
        self.setStyleSheet(
            """
        QDialog {
            background: #f4faf6;
            color: #1f3529;
            font-family: "SF Pro Text", "Avenir Next", "Trebuchet MS", sans-serif;
            font-size: 14px;
        }
        QLineEdit {
            background: #ffffff;
            border: 1px solid #bdd6c7;
            border-radius: 9px;
            padding: 8px 10px;
            min-height: 24px;
        }
        QPushButton {
            background: #ffffff;
            border: 1px solid #bcd6c8;
            border-radius: 10px;
            padding: 8px 14px;
            font-weight: 600;
            color: #244238;
        }
        QPushButton:hover { background: #eef7f1; }
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

    def values(self) -> tuple[str, str, str]:
        return (
            self.ffmpeg_edit.text().strip(),
            self.ffprobe_edit.text().strip(),
            self.token_edit.text().strip(),
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

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setObjectName("dropZone")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)

        icon = QLabel("🎬")
        icon.setObjectName("dropIcon")
        icon.setAlignment(Qt.AlignCenter)

        title = QLabel("Glissez-déposez une ou plusieurs vidéos")
        title.setObjectName("dropTitle")
        title.setAlignment(Qt.AlignCenter)

        subtitle = QLabel('ou cliquez sur "Ajouter des vidéos"')
        subtitle.setObjectName("dropSubtitle")
        subtitle.setAlignment(Qt.AlignCenter)

        details = QLabel("Preset par fichier + compression lot")
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


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        self.setMinimumSize(980, 640)

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

        self.queue_jobs: list[QueueJob] = []
        self.pending_indices: list[int] = []
        self.completed_count = 0
        self.failed_count = 0
        self.running_index: int | None = None
        self.current_index: int = -1
        self.worker: EncodeWorker | None = None
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
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        header = QFrame()
        header.setObjectName("header")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(16, 14, 16, 14)

        logo = QLabel()
        logo.setObjectName("logoMark")
        if Path(icon_path).exists():
            logo.setPixmap(QIcon(icon_path).pixmap(44, 44))
        else:
            logo.setText("🎬")
            logo.setAlignment(Qt.AlignmentFlag.AlignCenter)

        title_box = QVBoxLayout()
        self.h1 = QLabel(APP_NAME)
        self.h1.setObjectName("h1")
        self.h2 = QLabel(f"Compression réunion macOS • version {APP_VERSION}")
        self.h2.setObjectName("h2")
        title_box.addWidget(self.h1)
        title_box.addWidget(self.h2)

        self.btn_settings = QToolButton()
        self.btn_settings.setObjectName("gear")
        self.btn_settings.setText("⚙")
        self.btn_settings.clicked.connect(self.open_settings)
        self.btn_update = QPushButton("Vérifier mise à jour")
        self.btn_update.setObjectName("secondaryButton")
        self.btn_update.clicked.connect(self.check_updates)

        header_layout.addWidget(logo)
        header_layout.addLayout(title_box, 1)
        header_layout.addWidget(self.btn_update)
        header_layout.addWidget(self.btn_settings)
        root.addWidget(header)

        main_content = QHBoxLayout()
        main_content.setSpacing(18)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)

        left_panel = QWidget()
        left_col = QVBoxLayout(left_panel)
        left_col.setContentsMargins(0, 0, 0, 0)
        left_col.setSpacing(10)

        self.drop = DropZone()
        self.drop.files_dropped.connect(self.add_input_files)
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

        right_col = QFrame()
        right_col.setObjectName("settingsPanel")
        right_layout = QVBoxLayout(right_col)
        right_layout.setContentsMargins(8, 8, 8, 8)
        self.tabs = QTabWidget()
        self.tabs.setObjectName("controlTabs")
        right_layout.addWidget(self.tabs, 1)

        workflow_tab = QWidget()
        workflow_layout = QVBoxLayout(workflow_tab)
        workflow_layout.setContentsMargins(6, 6, 6, 6)
        workflow_layout.setSpacing(8)

        video_tab = QWidget()
        video_layout = QVBoxLayout(video_tab)
        video_layout.setContentsMargins(6, 6, 6, 6)
        video_layout.setSpacing(8)

        audio_tab = QWidget()
        audio_layout = QVBoxLayout(audio_tab)
        audio_layout.setContentsMargins(6, 6, 6, 6)
        audio_layout.setSpacing(8)

        workflow_group = QGroupBox("Workflow")
        workflow_form = QFormLayout(workflow_group)
        workflow_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        workflow_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self.combo_profile = QComboBox()
        self.combo_profile.addItems(["Personnalisé", *PROFILE_PRESETS.keys()])
        self.combo_profile.currentTextChanged.connect(self.on_profile_changed)
        workflow_form.addRow("Preset", self.combo_profile)

        out_row = QHBoxLayout()
        self.edit_output_dir = QLineEdit(str(self.settings.value("output_dir", str(Path.home() / "Desktop"), type=str)))
        self.btn_output_dir = QPushButton("…")
        self.btn_output_dir.setObjectName("secondaryButton")
        self.btn_output_dir.clicked.connect(self.pick_output_dir)
        out_row.addWidget(self.edit_output_dir)
        out_row.addWidget(self.btn_output_dir)
        workflow_form.addRow("Dossier sortie", out_row)

        self.edit_suffix = QLineEdit(str(self.settings.value("suffix", "_compressed", type=str)))
        workflow_form.addRow("Suffixe nom", self.edit_suffix)

        self.check_continue_on_error = QCheckBox("Continuer en cas d'erreur")
        self.check_continue_on_error.setChecked(self.settings.value("continue_on_error", True, type=bool))
        workflow_form.addRow("", self.check_continue_on_error)

        btn_row = QVBoxLayout()
        self.btn_apply_to_all = QPushButton("Appliquer ces réglages à tous")
        self.btn_apply_to_all.setObjectName("secondaryButton")
        self.btn_apply_to_all.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.btn_apply_to_all.clicked.connect(self.apply_current_settings_to_all)
        self.btn_reset_preset = QPushButton("Réinitialiser au preset")
        self.btn_reset_preset.setObjectName("secondaryButton")
        self.btn_reset_preset.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.btn_reset_preset.clicked.connect(self.reset_current_job_to_preset)
        btn_row.addWidget(self.btn_apply_to_all)
        btn_row.addWidget(self.btn_reset_preset)
        workflow_form.addRow("", btn_row)

        workflow_layout.addWidget(workflow_group)

        comp_group = QGroupBox("Compression vidéo")
        comp_form = QFormLayout(comp_group)
        comp_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        comp_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

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

        video_layout.addWidget(comp_group)

        audio_group = QGroupBox("Audio / voix")
        audio_form = QFormLayout(audio_group)
        audio_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        audio_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

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

        audio_layout.addWidget(audio_group)

        trim_group = QGroupBox("Rognage")
        trim_form = QFormLayout(trim_group)
        trim_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        trim_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

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

        video_layout.addWidget(trim_group)

        self.lbl_estimation = QLabel("Estimation: sélectionnez une vidéo.")
        self.lbl_estimation.setObjectName("metaLabel")
        self.lbl_estimation.setWordWrap(True)
        workflow_layout.addWidget(self.lbl_estimation)
        workflow_layout.addStretch(1)
        video_layout.addStretch(1)
        audio_layout.addStretch(1)

        self.tabs.addTab(workflow_tab, "Workflow")
        self.tabs.addTab(video_tab, "Vidéo")
        self.tabs.addTab(audio_tab, "Audio")

        splitter.addWidget(left_panel)
        splitter.addWidget(right_col)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([740, 440])
        main_content.addWidget(splitter)

        root.addLayout(main_content, 1)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        root.addWidget(self.progress)

        self.progress_global = QProgressBar()
        self.progress_global.setRange(0, 100)
        root.addWidget(self.progress_global)

        self.status = QLabel("Prêt.")
        self.status.setAlignment(Qt.AlignCenter)
        root.addWidget(self.status)

        actions = QHBoxLayout()
        self.btn_open_output = QPushButton("Ouvrir dossier sortie")
        self.btn_open_output.setObjectName("secondaryButton")
        self.btn_open_output.clicked.connect(self.open_output_dir)

        self.btn_start = QPushButton("Lancer la file")
        self.btn_start.setObjectName("accentButton")
        self.btn_start.clicked.connect(self.start_encode)
        self.btn_start.setEnabled(False)

        self.btn_cancel = QPushButton("Annuler")
        self.btn_cancel.setObjectName("secondaryButton")
        self.btn_cancel.clicked.connect(self.cancel_encode)
        self.btn_cancel.setEnabled(False)

        actions.addWidget(self.btn_open_output)
        actions.addStretch(1)
        actions.addWidget(self.btn_start)
        actions.addWidget(self.btn_cancel)
        root.addLayout(actions)

        self.on_trim_toggled(False)

    def apply_style(self):
        self.setStyleSheet(
            """
        QWidget {
            background: #f3f8f4;
            color: #1d2c23;
            font-family: "SF Pro Text", "Avenir Next", "Trebuchet MS", sans-serif;
            font-size: 13px;
        }
        QLabel { color: #20382c; }
        QFrame#header {
            background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #1f6b4f, stop:1 #2f8f69);
            border-radius: 16px;
            border: 1px solid rgba(255,255,255,0.25);
        }
        QLabel#logoMark {
            min-width: 52px;
            max-width: 52px;
            min-height: 52px;
            max-height: 52px;
            background: rgba(255,255,255,0.12);
            border: 1px solid rgba(255,255,255,0.22);
            border-radius: 12px;
            padding: 4px;
        }
        QLabel#h1 {
            font-size: 30px;
            font-weight: 800;
            color: #ffffff;
            letter-spacing: 0.2px;
        }
        QLabel#h2 {
            color: #d6f4e3;
            font-size: 14px;
            font-weight: 500;
        }
        QToolButton#gear {
            background: rgba(255,255,255,0.16);
            border: 1px solid rgba(255,255,255,0.38);
            border-radius: 11px;
            padding: 8px;
            color: #ffffff;
        }
        QToolButton#gear:hover {
            background: rgba(255,255,255,0.25);
        }
        QFrame#dropZone {
            background: #ffffff;
            border: 2px dashed #84c4a0;
            border-radius: 18px;
        }
        QFrame#dropZone:hover {
            border-color: #2f8f69;
            background: #f7fdf9;
        }
        QLabel#dropTitle {
            font-size: 20px;
            font-weight: 800;
            color: #205842;
        }
        QLabel#dropSubtitle, QLabel#dropDetails {
            color: #4d6a5a;
            font-size: 14px;
        }
        QFrame#settingsPanel {
            background: #ffffff;
            border: 1px solid #d5e8dc;
            border-radius: 16px;
        }
        QListWidget#queueList {
            background: #ffffff;
            border: 1px solid #d3e5d9;
            border-radius: 10px;
            padding: 8px;
        }
        QListWidget#queueList::item {
            padding: 7px 8px;
            margin: 2px 0;
            border-radius: 7px;
            color: #254237;
        }
        QListWidget#queueList::item:selected {
            background: #ddf1e6;
            color: #1c5a42;
        }
        QGroupBox {
            border: 1px solid #dcebe1;
            border-radius: 12px;
            margin-top: 10px;
            padding: 8px 8px 8px 8px;
            background: #fbfefc;
            font-weight: 800;
            color: #2a664d;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 12px;
            padding: 0 6px;
            color: #2a664d;
            background: #fbfefc;
        }
        QTabWidget#controlTabs::pane {
            border: none;
            background: transparent;
            margin-top: 2px;
        }
        QTabBar::tab {
            background: #e7f2eb;
            border: 1px solid #c8ded1;
            border-bottom: none;
            border-top-left-radius: 8px;
            border-top-right-radius: 8px;
            color: #2f5444;
            padding: 7px 14px;
            margin-right: 6px;
            font-weight: 700;
        }
        QTabBar::tab:selected {
            background: #ffffff;
            color: #215c43;
            border-color: #b3d2bf;
        }
        QTabBar::tab:!selected {
            margin-top: 3px;
        }
        QPushButton#primaryButton {
            background: #2f8f69;
            color: #ffffff;
            border: none;
            border-radius: 11px;
            padding: 11px 16px;
            font-size: 16px;
            font-weight: 800;
        }
        QPushButton#primaryButton:hover { background: #277d5b; }
        QPushButton#accentButton {
            background: #3fa77a;
            color: #ffffff;
            border: none;
            border-radius: 13px;
            padding: 12px 24px;
            font-size: 18px;
            font-weight: 900;
        }
        QPushButton#accentButton:hover {
            background: #37956d;
        }
        QPushButton#accentButton:disabled {
            background: #d5e1d9;
            color: #6d7e74;
        }
        QPushButton#secondaryButton {
            background: #ffffff;
            border: 1px solid #bcd6c8;
            border-radius: 10px;
            padding: 9px 12px;
            color: #2c4338;
            font-weight: 600;
        }
        QPushButton#secondaryButton:hover {
            background: #eef7f1;
        }
        QPushButton#trimToggle {
            background: #edf8f1;
            border: 1px solid #74b895;
            color: #2a664d;
            border-radius: 9px;
            padding: 8px;
            font-weight: 700;
        }
        QPushButton#trimToggle:checked {
            background: #2f8f69;
            color: #ffffff;
        }
        QComboBox, QSpinBox, QTimeEdit, QLineEdit {
            background: #ffffff;
            border: 1px solid #bdd6c7;
            border-radius: 9px;
            padding: 6px 9px;
            min-height: 20px;
        }
        QComboBox::drop-down {
            border: none;
            width: 18px;
        }
        QSlider::groove:horizontal {
            height: 6px;
            background: #d8e9de;
            border-radius: 3px;
        }
        QSlider::handle:horizontal {
            background: #2f8f69;
            border: 2px solid #ffffff;
            width: 15px;
            margin: -6px 0;
            border-radius: 7px;
        }
        QSlider::sub-page:horizontal {
            background: #2f8f69;
            border-radius: 3px;
        }
        QProgressBar {
            background: #eaf2ed;
            border: 1px solid #cfddd5;
            border-radius: 8px;
            text-align: center;
            min-height: 18px;
            color: #305144;
            font-weight: 700;
        }
        QProgressBar::chunk {
            background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #2f8f69, stop:1 #66b990);
            border-radius: 8px;
        }
        QLabel#metaLabel {
            color: #3b5d4d;
            background: #f6fbf8;
            border: 1px solid #d4e7dc;
            border-radius: 10px;
            padding: 10px;
            font-size: 14px;
        }
        QSplitter::handle {
            background: transparent;
            width: 8px;
        }
        QScrollBar:vertical {
            width: 9px;
            background: transparent;
            margin: 2px;
        }
        QScrollBar::handle:vertical {
            border-radius: 4px;
            background: #b9d9c7;
            min-height: 30px;
        }
        QScrollBar::handle:vertical:hover {
            background: #a3ccb6;
        }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            height: 0px;
        }
        """
        )

    def open_settings(self):
        dlg = SettingsDialog(self, self.ffmpeg_path, self.ffprobe_path, self.github_token)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        ffmpeg_path, ffprobe_path, github_token = dlg.values()
        self.ffmpeg_path = ffmpeg_path or find_binary("ffmpeg") or ""
        self.ffprobe_path = ffprobe_path or find_binary("ffprobe") or ""
        self.github_token = github_token
        self.settings.setValue("ffmpeg_path", self.ffmpeg_path)
        self.settings.setValue("ffprobe_path", self.ffprobe_path)
        self.settings.setValue("github_token", self.github_token)

        if self.ffmpeg_path:
            self.status.setText("Paramètres enregistrés.")
        else:
            self.status.setText("ffmpeg non détecté. Installez ffmpeg ou bundlez-le.")

    def save_global_settings(self):
        self.settings.setValue("output_dir", self.edit_output_dir.text().strip())
        self.settings.setValue("suffix", self.edit_suffix.text().strip())
        self.settings.setValue("continue_on_error", self.check_continue_on_error.isChecked())

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

            script_content = "\n".join(
                [
                    "#!/bin/bash",
                    "set -e",
                    f"PID={pid}",
                    f"NEW_APP={shlex.quote(str(new_app_path))}",
                    f"TARGET_APP={shlex.quote(str(target_app_path))}",
                    "for i in {1..120}; do",
                    "  if ! kill -0 \"$PID\" 2>/dev/null; then",
                    "    break",
                    "  fi",
                    "  sleep 0.25",
                    "done",
                    "rm -rf \"${TARGET_APP}.new\"",
                    "cp -R \"$NEW_APP\" \"${TARGET_APP}.new\"",
                    "rm -rf \"$TARGET_APP\"",
                    "mv \"${TARGET_APP}.new\" \"$TARGET_APP\"",
                    "xattr -dr com.apple.quarantine \"$TARGET_APP\" || true",
                    "open \"$TARGET_APP\"",
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
                return parent
        user_apps = Path.home() / "Applications"
        user_apps.mkdir(parents=True, exist_ok=True)
        return user_apps / "EkoVideoCompressor.app"

    def pick_output_dir(self):
        start = self.edit_output_dir.text().strip() or str(Path.home())
        directory = QFileDialog.getExistingDirectory(self, "Choisir dossier de sortie", start)
        if directory:
            self.edit_output_dir.setText(directory)

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
        self.btn_start.setEnabled(False)

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

    def _set_running_ui(self, running: bool):
        self.is_batch_running = running
        self.btn_pick.setEnabled(not running)
        self.btn_remove.setEnabled(not running)
        self.btn_clear.setEnabled(not running)
        self.btn_start.setEnabled(bool(self.queue_jobs) and not running)
        self.btn_cancel.setEnabled(running)
        self.btn_settings.setEnabled(not running)
        self.btn_update.setEnabled(not running and self.update_worker is None)
        self.btn_apply_to_all.setEnabled(not running)
        self.btn_reset_preset.setEnabled(not running)

    def start_encode(self):
        if not self.queue_jobs:
            return

        if not self.ffmpeg_path or not Path(self.ffmpeg_path).exists():
            QMessageBox.critical(self, "ffmpeg introuvable", "ffmpeg est requis. Ouvrez Paramètres (⚙).")
            return

        if self.current_index >= 0:
            self.save_current_job_from_controls(force_custom=False)
        self.save_global_settings()

        out_dir = Path(self.edit_output_dir.text().strip() or str(Path.home() / "Desktop"))
        out_dir.mkdir(parents=True, exist_ok=True)
        self.edit_output_dir.setText(str(out_dir))

        for idx, job in enumerate(self.queue_jobs):
            if not self._validate_trim(job):
                QMessageBox.warning(self, "Rognage invalide", f"Le rognage est invalide pour {Path(job.input_path).name}.")
                self.queue_list.setCurrentRow(idx)
                return

            self.queue_jobs[idx].output_path = default_out_path(job.input_path, str(out_dir), self.edit_suffix.text().strip())
            self.queue_jobs[idx].status = "pending"
            self.queue_jobs[idx].error_message = ""

        self.pending_indices = list(range(len(self.queue_jobs)))
        self.completed_count = 0
        self.failed_count = 0
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress_global.setRange(0, 100)
        self.progress_global.setValue(0)
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

        self.worker = EncodeWorker(self.ffmpeg_path, self.ffprobe_path, replace(job))
        self.worker.progress.connect(self.on_progress)
        self.worker.status.connect(self.on_status)
        self.worker.duration_unknown.connect(self.on_duration_unknown)
        self.worker.finished_ok.connect(self.on_done)
        self.worker.failed.connect(self.on_fail)
        self.worker.start()

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
        self.progress.setValue(pct)

        total = max(1, len(self.queue_jobs))
        done = self.completed_count + self.failed_count
        global_pct = int(min(100, ((done + pct / 100.0) / total) * 100))
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
        job.status = "done"
        job.error_message = ""
        job.output_path = out_path
        self.completed_count += 1
        self.running_index = None
        self.worker = None

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
        self.worker = None

        if msg == "Annulé.":
            job.status = "failed"
            job.error_message = "Annulé"
            self.refresh_queue_item(idx)
            self._finish_batch(cancelled=True)
            return

        job.status = "failed"
        job.error_message = msg
        self.failed_count += 1
        self.refresh_queue_item(idx)

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
            QMessageBox.warning(self, "Annulé", f"Traitement annulé. Réussis: {done}, Erreurs: {failed}.")
            return

        self.progress.setRange(0, 100)
        self.progress.setValue(100 if done else 0)
        self.progress_global.setValue(100 if total else 0)

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
    app = QApplication(sys.argv)
    app.setOrganizationName(ORG_NAME)
    app.setOrganizationDomain(ORG_DOMAIN)
    app.setApplicationName(APP_NAME)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
