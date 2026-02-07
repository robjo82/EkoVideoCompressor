#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VERSION="${1:-${APP_VERSION:-dev}}"
ARTIFACT_NAME="EkoVideoCompressor-macos-arm64-v${VERSION}.zip"
VENV_DIR=".venv-build"

python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
pip install -r requirements.txt

mkdir -p bin
if command -v ffmpeg >/dev/null 2>&1; then
  cp "$(command -v ffmpeg)" bin/ffmpeg
else
  echo "ffmpeg introuvable dans PATH" >&2
  exit 1
fi

if command -v ffprobe >/dev/null 2>&1; then
  cp "$(command -v ffprobe)" bin/ffprobe
else
  echo "ffprobe introuvable dans PATH" >&2
  exit 1
fi

chmod +x bin/ffmpeg bin/ffprobe

export APP_VERSION="$VERSION"
pyinstaller --noconfirm --clean video_compactor.spec

mkdir -p dist/release
rm -f "dist/release/${ARTIFACT_NAME}"
(
  cd dist
  zip -r "release/${ARTIFACT_NAME}" "EkoVideoCompressor.app"
)

echo "Artefact créé: dist/release/${ARTIFACT_NAME}"
