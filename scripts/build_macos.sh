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

# Fetch static, notarised ffmpeg + ffprobe from evermeet.cx. The old
# code did `cp $(command -v ffmpeg) bin/ffmpeg`, which on a Homebrew
# install grabs a binary that dynamically references
# /opt/homebrew/Cellar/ffmpeg/<version>/lib/*. End users never have
# the exact same dylib paths, so they hit dyld errors like:
#   Library not loaded: libavdevice.62.dylib
# The static evermeet.cx builds only depend on macOS system frameworks,
# so the bundle becomes truly self-contained. Verifier inside the
# script aborts the build if any forbidden dylib reference slips in.
FORCE=1 scripts/fetch_static_ffmpeg.sh

export APP_VERSION="$VERSION"
cleanup() {
  rm -f _build_version.py
}
trap cleanup EXIT

cat > _build_version.py <<EOF
APP_VERSION = "${VERSION}"
EOF
QT_QPA_PLATFORM=offscreen python video_compactor.py --startup-smoke-test
python -m ekovideo_engine --startup-smoke-test
pyinstaller --noconfirm --clean ekovideo_engine.spec
swift build -c release --package-path macos/EkoVideoCompressor

# Ad-hoc sign so macOS treats the bundle as a stable identity across replacements.
# Without this, an updated bundle is treated as a brand-new unsigned app and
# Gatekeeper may refuse to relaunch — even on manual reopen — until reinstall.
APP_BUNDLE="dist/EkoVideoCompressor.app"
rm -rf "$APP_BUNDLE"
mkdir -p "$APP_BUNDLE/Contents/MacOS" "$APP_BUNDLE/Contents/Resources/bin"

SWIFT_BIN_DIR="$(swift build -c release --package-path macos/EkoVideoCompressor --show-bin-path)"
cp "$SWIFT_BIN_DIR/EkoVideoCompressor" "$APP_BUNDLE/Contents/MacOS/EkoVideoCompressor"
chmod +x "$APP_BUNDLE/Contents/MacOS/EkoVideoCompressor"

cat > "$APP_BUNDLE/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleDevelopmentRegion</key>
  <string>fr</string>
  <key>CFBundleDisplayName</key>
  <string>EkoVideoCompressor</string>
  <key>CFBundleExecutable</key>
  <string>EkoVideoCompressor</string>
  <key>CFBundleIconFile</key>
  <string>EkoVideoCompressor</string>
  <key>CFBundleIdentifier</key>
  <string>com.ekonum.ekovideocompressor</string>
  <key>CFBundleInfoDictionaryVersion</key>
  <string>6.0</string>
  <key>CFBundleName</key>
  <string>EkoVideoCompressor</string>
  <key>CFBundlePackageType</key>
  <string>APPL</string>
  <key>CFBundleShortVersionString</key>
  <string>${VERSION}</string>
  <key>CFBundleVersion</key>
  <string>${VERSION}</string>
  <key>LSMinimumSystemVersion</key>
  <string>13.0</string>
  <key>NSHighResolutionCapable</key>
  <true/>
</dict>
</plist>
EOF

cp EkoVideoCompressor.icns "$APP_BUNDLE/Contents/Resources/"
cp bin/ffmpeg bin/ffprobe "$APP_BUNDLE/Contents/Resources/bin/"
chmod u+rw,go+r "$APP_BUNDLE/Contents/Resources/bin/ffmpeg" "$APP_BUNDLE/Contents/Resources/bin/ffprobe"
chmod +x "$APP_BUNDLE/Contents/Resources/bin/ffmpeg" "$APP_BUNDLE/Contents/Resources/bin/ffprobe"
ENGINE_DIR="$APP_BUNDLE/Contents/Resources/engine"
mkdir -p "$ENGINE_DIR"
cp "dist/ekovideo-engine" "$ENGINE_DIR/ekovideo-engine"
chmod +x "$ENGINE_DIR/ekovideo-engine"

"$APP_BUNDLE/Contents/MacOS/EkoVideoCompressor" --smoke-test

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
