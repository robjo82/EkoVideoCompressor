# -*- mode: python ; coding: utf-8 -*-

import os
from pathlib import Path

import certifi

# `__file__` is not defined when PyInstaller executes spec files in CI.
# The build script/workflow runs from repository root, so cwd is reliable here.
project_dir = Path.cwd().resolve()
app_version = os.environ.get("APP_VERSION", "0.0.0").strip().lstrip("v") or "0.0.0"

binaries = []
for name in ("ffmpeg", "ffprobe"):
    bin_path = project_dir / "bin" / name
    if bin_path.exists():
        binaries.append((str(bin_path), "bin"))

datas = []
for asset in ("ekovideo_icon.png", "ekovideo_logo.png"):
    asset_path = project_dir / asset
    if asset_path.exists():
        datas.append((str(asset_path), "."))

# Ship the SVG assets folder used by the stylesheet (chevron etc).
assets_dir = project_dir / "assets"
if assets_dir.exists():
    for asset_path in sorted(assets_dir.iterdir()):
        if asset_path.is_file():
            datas.append((str(asset_path), "assets"))

datas.append((certifi.where(), "certifi"))

a = Analysis(
    ['video_compactor.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=["certifi"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    name='EkoVideoCompressor',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch='arm64',
    codesign_identity=None,
    entitlements_file=None,
    exclude_binaries=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='EkoVideoCompressor',
)

app = BUNDLE(
    coll,
    name='EkoVideoCompressor.app',
    icon='EkoVideoCompressor.icns',
    bundle_identifier='com.ekonum.ekovideocompressor',
    info_plist={
        "CFBundleDisplayName": "EkoVideoCompressor",
        "CFBundleShortVersionString": app_version,
        "CFBundleVersion": app_version,
    },
)
