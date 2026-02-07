# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

project_dir = Path(__file__).resolve().parent

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

a = Analysis(
    ['video_compactor.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=[],
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
    a.binaries,
    a.datas,
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
)

app = BUNDLE(
    exe,
    name='EkoVideoCompressor.app',
    icon=None,
    bundle_identifier='com.ekonum.ekovideocompressor',
)
