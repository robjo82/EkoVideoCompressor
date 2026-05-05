#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VERSION="${1:-${APP_VERSION:-dev}}"
ARTIFACT_NAME="EkoVideoCompressor-macos-arm64-v${VERSION}.zip"
VENV_DIR=".venv-build"
PYTHON_BIN="${PYTHON_BIN:-}"

if [[ -z "$PYTHON_BIN" ]]; then
  for candidate in python3.11 python3.12 python3.13 python3; do
    if command -v "$candidate" >/dev/null 2>&1 && "$candidate" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if (3, 9) <= sys.version_info[:2] < (3, 14) else 1)
PY
    then
      PYTHON_BIN="$(command -v "$candidate")"
      break
    fi
  done
fi

if [[ -z "$PYTHON_BIN" ]]; then
  echo "Python compatible introuvable. Installez Python 3.11, 3.12 ou 3.13." >&2
  exit 1
fi

rm -rf "$VENV_DIR"
"$PYTHON_BIN" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
pip install -r requirements.txt

mkdir -p bin
if command -v ffmpeg >/dev/null 2>&1; then
  rm -f bin/ffmpeg
  cp "$(command -v ffmpeg)" bin/ffmpeg
else
  echo "ffmpeg introuvable dans PATH" >&2
  exit 1
fi

if command -v ffprobe >/dev/null 2>&1; then
  rm -f bin/ffprobe
  cp "$(command -v ffprobe)" bin/ffprobe
else
  echo "ffprobe introuvable dans PATH" >&2
  exit 1
fi

chmod +x bin/ffmpeg bin/ffprobe

export APP_VERSION="$VERSION"
cleanup() {
  rm -f _build_version.py
}
trap cleanup EXIT

cat > _build_version.py <<EOF
APP_VERSION = "${VERSION}"
EOF
pyinstaller --noconfirm --clean video_compactor.spec

# Ad-hoc sign so macOS treats the bundle as a stable identity across replacements.
# Without this, an updated bundle is treated as a brand-new unsigned app and
# Gatekeeper may refuse to relaunch — even on manual reopen — until reinstall.
APP_BUNDLE="dist/EkoVideoCompressor.app"
xattr -cr "$APP_BUNDLE" || true
codesign --force --deep --sign - --timestamp=none "$APP_BUNDLE"
codesign --verify --deep --strict "$APP_BUNDLE"

mkdir -p dist/release
rm -f "dist/release/${ARTIFACT_NAME}"
(
  cd dist
  zip -r "release/${ARTIFACT_NAME}" "EkoVideoCompressor.app"
)

echo "Artefact créé: dist/release/${ARTIFACT_NAME}"
