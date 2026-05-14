# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

import certifi

project_dir = Path.cwd().resolve()

datas = [(certifi.where(), "certifi")]
for rel in (
    "ffmpeg_utils.py",
    "transcription_utils.py",
    "glossary_postprocess.py",
    "database_manager.py",
    "multipass.py",
    "per_speaker.py",
    "vad_silero.py",
    "web_context.py",
):
    path = project_dir / rel
    if path.exists():
        datas.append((str(path), "."))

a = Analysis(
    ["ekovideo_engine_cli.py"],
    pathex=[str(project_dir)],
    binaries=[],
    datas=datas,
    hiddenimports=["certifi", "ekovideo_engine.cli"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["PySide6"],
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
    name="ekovideo-engine",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch="arm64",
    codesign_identity=None,
    entitlements_file=None,
)
