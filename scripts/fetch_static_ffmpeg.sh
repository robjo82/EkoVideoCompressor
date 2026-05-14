#!/usr/bin/env bash
# Fetch static, notarised ffmpeg + ffprobe binaries for macOS arm64.
#
# Why this script exists: we used to just `cp $(command -v ffmpeg)`
# inside the build, which on a Homebrew install grabs a binary that
# dynamically links into /opt/homebrew/Cellar/ffmpeg/<version>/lib/*.
# That works on the build machine but blows up the moment we ship the
# bundle to a user who has no Homebrew at all — or has a different
# ffmpeg version — with the dyld error:
#
#   Library not loaded: /opt/homebrew/Cellar/ffmpeg/8.1.1/lib/libavdevice.62.dylib
#
# evermeet.cx ships statically-linked, Apple-notarised ffmpeg + ffprobe
# binaries for macOS. They depend only on system frameworks, so the
# bundle is fully self-contained.
#
# The script is idempotent: it skips the download when bin/ffmpeg and
# bin/ffprobe already exist and are statically linked.
#
# Exit code 0 on success, non-zero with a diagnostic message on
# failure (network down, otool found a forbidden dylib, etc.).
#
# Usage:
#   scripts/fetch_static_ffmpeg.sh           # writes bin/ffmpeg + bin/ffprobe
#   FORCE=1 scripts/fetch_static_ffmpeg.sh   # re-download even if static binaries already exist

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
mkdir -p bin

FORCE="${FORCE:-0}"
EVERMEET_FFMPEG_URL="${EVERMEET_FFMPEG_URL:-https://evermeet.cx/ffmpeg/getrelease/ffmpeg/zip}"
EVERMEET_FFPROBE_URL="${EVERMEET_FFPROBE_URL:-https://evermeet.cx/ffmpeg/getrelease/ffprobe/zip}"

# A binary is "self-contained" when otool -L returns only system
# frameworks. Anything pointing at /opt/homebrew, /usr/local/Cellar,
# @rpath leftovers from Homebrew, or any /private/tmp path is a fail.
is_self_contained() {
  local path="$1"
  if [[ ! -x "$path" ]]; then
    return 1
  fi
  # System libs allowed: /usr/lib/*, /System/*. Any other absolute or
  # @rpath reference is suspicious. We pipe through tail to skip the
  # binary's own headline line that otool prints first.
  local refs
  refs="$(otool -L "$path" 2>/dev/null | tail -n +2 | awk '{print $1}' || true)"
  if [[ -z "$refs" ]]; then
    # otool can fail on a missing binary or wrong arch.
    return 1
  fi
  while IFS= read -r line; do
    case "$line" in
      /usr/lib/*|/System/*) ;;
      "") ;;
      *)
        echo "  forbidden dylib reference: $line" >&2
        return 1
        ;;
    esac
  done <<< "$refs"
  return 0
}

needs_download() {
  local path="$1"
  if [[ "$FORCE" == "1" ]]; then
    return 0
  fi
  if [[ ! -x "$path" ]]; then
    return 0
  fi
  if ! is_self_contained "$path" 2>/dev/null; then
    return 0
  fi
  return 1
}

download_and_extract() {
  local url="$1"
  local target="$2"
  local stem
  stem="$(basename "$target")"
  echo "Downloading static $stem from $url"
  local tmp_dir
  tmp_dir="$(mktemp -d)"
  trap 'rm -rf "$tmp_dir"' RETURN
  local zip_path="$tmp_dir/$stem.zip"
  # `-fL` so we follow redirects (evermeet.cx redirects to the
  # actual versioned URL) and fail on HTTP errors.
  if ! curl -fL --retry 3 --retry-delay 2 --connect-timeout 30 -o "$zip_path" "$url"; then
    echo "Failed to download $url" >&2
    return 1
  fi
  unzip -q -o "$zip_path" -d "$tmp_dir"
  if [[ ! -f "$tmp_dir/$stem" ]]; then
    echo "Archive from $url did not contain $stem" >&2
    return 1
  fi
  install -m 0755 "$tmp_dir/$stem" "$target"
}

verify_self_contained() {
  local path="$1"
  echo "Verifying $path is statically linked"
  if ! is_self_contained "$path"; then
    echo "$path still references non-system libraries; refusing to ship" >&2
    otool -L "$path" >&2 || true
    return 1
  fi
  # `--version` is a cheap end-to-end check — confirms the binary
  # actually runs on this machine (right arch, not corrupt).
  if ! "$path" -version >/dev/null 2>&1; then
    echo "$path failed --version smoke test" >&2
    return 1
  fi
}

if needs_download "bin/ffmpeg"; then
  download_and_extract "$EVERMEET_FFMPEG_URL" "bin/ffmpeg"
else
  echo "bin/ffmpeg already statically linked, skipping download (set FORCE=1 to override)"
fi
verify_self_contained "bin/ffmpeg"

if needs_download "bin/ffprobe"; then
  download_and_extract "$EVERMEET_FFPROBE_URL" "bin/ffprobe"
else
  echo "bin/ffprobe already statically linked, skipping download (set FORCE=1 to override)"
fi
verify_self_contained "bin/ffprobe"

echo "Static ffmpeg + ffprobe ready in bin/"
