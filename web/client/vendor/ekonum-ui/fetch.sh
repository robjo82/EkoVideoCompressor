#!/bin/sh
# Fetches the ekonum-ui version pinned in an application's manifest.
#
#   fetch.sh [--manifest ekonum.yaml] [--target static] [--version X.Y.Z] [--no-fonts]
#
# Copy it into the application's repository (scripts/fetch-ekonum-ui.sh). The version
# comes from `design_system:` in ekonum.yaml, not from a constant: that is the value the
# portal reads to measure how far the application lags behind the brand.
#
# Access: the ekonum-ui repository is private (font licence). It needs either an
# authenticated `gh` (workstation) or GH_TOKEN / GITHUB_TOKEN with curl (CI); jq in both cases.
# Another application's own GITHUB_TOKEN is not enough: it only opens its own repository.
#
# --no-fonts: installs only the publishable part (CSS, Tailwind theme, tokens), for an
# application that receives the fonts some other way.
set -eu

MANIFEST=ekonum.yaml
TARGET=static
VERSION=
FONTS=yes
while [ $# -gt 0 ]; do
  case "$1" in
    --manifest) MANIFEST=$2; shift 2 ;;
    --target) TARGET=$2; shift 2 ;;
    --version) VERSION=$2; shift 2 ;;
    --no-fonts) FONTS=no; shift ;;
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    *) echo "✗ unknown option: $1" >&2; exit 2 ;;
  esac
done

if [ -z "$VERSION" ]; then
  [ -f "$MANIFEST" ] || { echo "✗ manifest not found: $MANIFEST" >&2; exit 1; }
  VERSION="$(sed -n 's/^design_system: *["'\'']*\([0-9][0-9.]*\).*/\1/p' "$MANIFEST")"
  [ -n "$VERSION" ] || { echo "✗ design_system: missing from $MANIFEST" >&2; exit 1; }
fi

REPO=Ekonum/ekonum-ui
API=https://api.github.com
TOKEN="${GH_TOKEN:-${GITHUB_TOKEN:-}}"

# Two transports, one contract: GET an API path, as JSON or as binary.
command -v jq >/dev/null || { echo "✗ jq is required to read GitHub API responses" >&2; exit 1; }
json() { jq -r "$1"; }
if [ -n "$TOKEN" ] && command -v curl >/dev/null; then
  api()  { curl -fsSL -H "Authorization: Bearer $TOKEN" -H "Accept: application/vnd.github+json" "$API/$1"; }
  raw()  { curl -fsSL -H "Authorization: Bearer $TOKEN" -H "Accept: application/octet-stream" "$API/$1"; }
elif command -v gh >/dev/null; then
  api()  { gh api "$1"; }
  raw()  { gh api -H "Accept: application/octet-stream" "$1"; }
else
  echo "✗ neither gh nor GH_TOKEN with curl: cannot read the private repository $REPO" >&2; exit 1
fi

TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT

# No `gh release download`: it reads the release's aggregated asset list, which GitHub
# may serve stale right after publication (seen on v1.0.1: "no assets to download"
# while the files were attached). The dedicated assets endpoint is up to date.
# In a pipe, jq's exit code is what counts: a 404 would yield "null" without an error.
RELEASE_ID="$(api "repos/$REPO/releases/tags/v$VERSION" 2>/dev/null | json '.id // empty' 2>/dev/null || true)"
case "$RELEASE_ID" in
  ''|*[!0-9]*) echo "✗ version $VERSION not found in $REPO, or access denied (token without rights on this repository?)" >&2; exit 1 ;;
esac
api "repos/$REPO/releases/$RELEASE_ID/assets" | json '.[]|"\(.id) \(.name)"' > "$TMP/assets"
while read -r ID NAME; do
  case "$NAME" in "ekonum-ui-$VERSION.tar.gz"|"ekonum-ui-$VERSION.tar.gz.sha256") ;; *) continue ;; esac
  raw "repos/$REPO/releases/assets/$ID" > "$TMP/$NAME"
done < "$TMP/assets"
[ -f "$TMP/ekonum-ui-$VERSION.tar.gz" ] || { echo "✗ ekonum-ui $VERSION archive missing from the release" >&2; exit 1; }

if command -v sha256sum >/dev/null; then CHECK="sha256sum -c"; else CHECK="shasum -a 256 -c"; fi
( cd "$TMP" && $CHECK "ekonum-ui-$VERSION.tar.gz.sha256" >/dev/null ) ||
  { echo "✗ wrong SHA-256 checksum: corrupted or tampered archive" >&2; exit 1; }

tar -xzf "$TMP/ekonum-ui-$VERSION.tar.gz" -C "$TMP"
[ "$(cat "$TMP/ekonum-ui/VERSION")" = "$VERSION" ] || { echo "✗ the archive does not contain version $VERSION" >&2; exit 1; }
# fonts/ since 3.0.0, polices/ before: this script also fetches 2.x versions.
[ "$FONTS" = yes ] || rm -rf "$TMP/ekonum-ui/fonts" "$TMP/ekonum-ui/polices"

mkdir -p "$TARGET" && rm -rf "$TARGET/ekonum-ui" && mv "$TMP/ekonum-ui" "$TARGET/ekonum-ui"
echo "✓ ekonum-ui $VERSION → $TARGET/ekonum-ui/$( [ "$FONTS" = yes ] || echo ' (without fonts)')"
