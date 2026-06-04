"""PR AT — keep the managed transcription venv's ML libraries fresh.

Why this exists
---------------
The venv was historically provisioned *once* (legacy ``video_compactor``
``_ensure_*`` helpers + the engine's own probes) with a "probe ``import
X`` → if it imports, do nothing" pattern. The ``--upgrade`` flag on
those installs only ever ran when a package was *missing*, so an
existing venv silently froze at whatever versions were latest the day
it was created. That left users stuck on e.g. ``mlx-vlm 0.4.4`` — too
old for reliable Gemma 4 audio (audio fixes landed in 0.5.0, the 12B
"Unified" path in 0.6.1) — with no way to move forward short of nuking
the venv by hand.

What this module does
---------------------
Defines a per-package **version floor** (the minimum we require for the
features we ship) and the plumbing to:
  * read the versions actually installed in a given venv,
  * decide which packages sit below their floor,
  * upgrade them (to the floor, or all the way to latest).

It is deliberately dependency-free and the comparison logic is pure so
it can be unit-tested without a real venv or network. The CLI
(``deps-check`` / ``deps-upgrade``) and the SwiftUI app drive it.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from typing import Callable, Optional

from .paths import managed_venv_python_path


@dataclass(frozen=True)
class Requirement:
    """One managed package.

    ``pip_name`` is what ``pip install`` wants (``mlx-vlm``);
    ``import_name`` is what Python imports (``mlx_vlm``) — they differ
    often enough (dashes vs underscores, ``pyannote.audio`` vs the
    ``pyannote-audio`` distribution) that we track both. ``min_version``
    is the floor we enforce automatically.
    """

    pip_name: str
    import_name: str
    min_version: str


# The managed ML stack. Floors reflect the 2026-06 audit: the core
# (mlx / mlx-lm / whisper / pyannote / silero) was already current, so
# their floors pin that; the laggards we deliberately raise — chiefly
# ``mlx-vlm`` 0.4.4 → 0.6.1 (the Gemma 4 audio blocker), plus
# transformers / huggingface-hub / torch which had drifted a release or
# two behind. ``torchaudio`` is kept lockstep with ``torch``.
REQUIREMENTS: tuple[Requirement, ...] = (
    Requirement("mlx", "mlx", "0.31.2"),
    Requirement("mlx-lm", "mlx_lm", "0.31.3"),
    Requirement("mlx-vlm", "mlx_vlm", "0.6.1"),
    Requirement("mlx-whisper", "mlx_whisper", "0.4.3"),
    Requirement("pyannote.audio", "pyannote.audio", "4.0.4"),
    Requirement("silero-vad", "silero_vad", "6.2.1"),
    Requirement("transformers", "transformers", "5.10.1"),
    Requirement("huggingface-hub", "huggingface_hub", "1.17.0"),
    Requirement("torch", "torch", "2.12.0"),
    Requirement("torchaudio", "torchaudio", "2.12.0"),
)


# Status constants for a single package — kept as plain strings so they
# serialise straight to JSON for the SwiftUI layer.
STATUS_OK = "ok"            # installed >= floor
STATUS_OUTDATED = "outdated"  # installed < floor
STATUS_MISSING = "missing"    # not installed at all
STATUS_UNKNOWN = "unknown"    # couldn't parse the installed version


def parse_version(value: str) -> tuple[int, ...]:
    """Parse a version string into a comparable tuple of ints.

    Tolerant by design: ``"0.31.3"`` → ``(0, 31, 3)``; pre/post/dev
    suffixes are dropped component-wise (``"2.12.0.dev1"`` → the leading
    integer of each dotted part, non-numeric tails ignored). Empty /
    unparseable input yields ``()`` which compares as the lowest
    possible version. We don't need full PEP 440 — only "is A below B"
    for our own floor strings, which are plain ``x.y.z``.
    """
    parts: list[int] = []
    for chunk in (value or "").strip().split("."):
        num = ""
        for ch in chunk:
            if ch.isdigit():
                num += ch
            else:
                break
        if num == "":
            break
        parts.append(int(num))
    return tuple(parts)


def version_lt(a: str, b: str) -> bool:
    """True when version ``a`` is strictly older than ``b``.

    Pads the shorter tuple with zeros so ``"2.12" < "2.12.0"`` is
    False (they're equal) and ``"2.11.9" < "2.12"`` is True.
    """
    ta, tb = parse_version(a), parse_version(b)
    length = max(len(ta), len(tb))
    ta = ta + (0,) * (length - len(ta))
    tb = tb + (0,) * (length - len(tb))
    return ta < tb


def needs_upgrade(installed: Optional[str], minimum: str) -> bool:
    """True when ``installed`` is missing or below ``minimum``."""
    if not installed:
        return True
    return version_lt(installed, minimum)


def read_installed_versions(venv_python: str) -> dict[str, str]:
    """Map ``pip_name`` → installed version for the given interpreter.

    Uses ``pip list --format=json`` (one subprocess, exact versions)
    rather than importing each package — cheaper and side-effect-free.
    Names are normalised to lowercase so ``Pyannote.Audio`` and
    ``pyannote-audio`` collapse to the same key. Returns an empty dict
    on any failure (missing venv, pip error) so callers treat every
    package as "missing" and surface that cleanly.
    """
    try:
        proc = subprocess.run(
            [venv_python, "-m", "pip", "list", "--format=json"],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    if proc.returncode != 0:
        return {}
    try:
        rows = json.loads(proc.stdout)
    except (TypeError, ValueError):
        return {}
    out: dict[str, str] = {}
    for row in rows if isinstance(rows, list) else []:
        name = _normalise(str(row.get("name", "")))
        version = str(row.get("version", "")).strip()
        if name:
            out[name] = version
    return out


def _normalise(name: str) -> str:
    """PEP 503 distribution-name normalisation.

    pip reports the same package under inconsistent punctuation —
    ``pyannote-audio`` for the ``pyannote.audio`` spec,
    ``huggingface_hub`` for ``huggingface-hub``. Collapsing every run of
    ``-``/``_``/``.`` to a single ``-`` (lowercased) makes both sides of
    the comparison agree.
    """
    return re.sub(r"[-_.]+", "-", name.strip().lower())


def check(venv_python: Optional[str] = None) -> dict:
    """Inspect the venv and report each managed package's status.

    Returns ``{"venv_python", "any_outdated", "packages": [...]}`` where
    each package carries ``pip_name``, ``installed`` (or None),
    ``minimum`` and ``status``.
    """
    target = venv_python or str(managed_venv_python_path())
    installed = read_installed_versions(target)
    packages: list[dict] = []
    any_outdated = False
    for req in REQUIREMENTS:
        current = installed.get(_normalise(req.pip_name))
        if not current:
            status = STATUS_MISSING
        elif not parse_version(current):
            status = STATUS_UNKNOWN
        elif version_lt(current, req.min_version):
            status = STATUS_OUTDATED
        else:
            status = STATUS_OK
        if status in (STATUS_MISSING, STATUS_OUTDATED):
            any_outdated = True
        packages.append(
            {
                "pip_name": req.pip_name,
                "import_name": req.import_name,
                "installed": current,
                "minimum": req.min_version,
                "status": status,
            }
        )
    return {
        "venv_python": target,
        "any_outdated": any_outdated,
        "packages": packages,
    }


def _pip_specs(to_latest: bool) -> list[str]:
    """Build the ``pip install --upgrade`` target specs.

    ``to_latest`` (the manual "update everything" button) drops the
    version constraint so pip resolves to the newest release. Otherwise
    we pin ``pkg>=floor`` so an automatic run only ever moves a package
    *up to* the floor we require — never surprises the user with a
    bleeding-edge bump they didn't ask for.
    """
    specs: list[str] = []
    for req in REQUIREMENTS:
        if to_latest:
            specs.append(req.pip_name)
        else:
            specs.append(f"{req.pip_name}>={req.min_version}")
    return specs


def upgrade(
    venv_python: Optional[str] = None,
    *,
    to_latest: bool = False,
    only_outdated: bool = True,
    progress: Optional[Callable[[str], None]] = None,
) -> dict:
    """Upgrade managed packages in the venv.

    * ``to_latest`` — install the newest release of every package (the
      manual "Mettre à jour les moteurs IA" action). When False, only
      enforce the floors (``pkg>=min``), the automatic startup path.
    * ``only_outdated`` — when True (and not ``to_latest``), restrict
      the install to packages currently below their floor, so a healthy
      venv runs no pip at all.
    * ``progress`` — optional sink for human-readable status lines.

    Returns ``{"upgraded": bool, "packages": [...], "returncode",
    "stdout_tail", "stderr_tail"}``. A no-op (nothing to do) is a
    success with ``upgraded=False``.
    """
    target = venv_python or str(managed_venv_python_path())

    if to_latest:
        specs = _pip_specs(to_latest=True)
        names = [req.pip_name for req in REQUIREMENTS]
    else:
        status = check(target)
        outdated = {
            p["pip_name"]
            for p in status["packages"]
            if p["status"] in (STATUS_MISSING, STATUS_OUTDATED)
        }
        if only_outdated:
            specs = [
                f"{req.pip_name}>={req.min_version}"
                for req in REQUIREMENTS
                if req.pip_name in outdated
            ]
            names = [req.pip_name for req in REQUIREMENTS if req.pip_name in outdated]
        else:
            specs = _pip_specs(to_latest=False)
            names = [req.pip_name for req in REQUIREMENTS]

    if not specs:
        if progress:
            progress("Les moteurs IA sont déjà à jour.")
        return {
            "upgraded": False,
            "packages": [],
            "returncode": 0,
            "stdout_tail": "",
            "stderr_tail": "",
        }

    if progress:
        progress(f"Mise à jour de {len(names)} paquet(s) : {', '.join(names)}…")

    try:
        proc = subprocess.run(
            [target, "-m", "pip", "install", "--upgrade", *specs],
            capture_output=True,
            text=True,
            timeout=3600,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        if progress:
            progress(f"Échec de la mise à jour : {exc!r}")
        return {
            "upgraded": False,
            "packages": names,
            "returncode": -1,
            "stdout_tail": "",
            "stderr_tail": repr(exc),
        }

    ok = proc.returncode == 0
    if progress:
        progress("Mise à jour terminée." if ok else "Échec de la mise à jour (voir les logs).")
    return {
        "upgraded": ok,
        "packages": names,
        "returncode": proc.returncode,
        "stdout_tail": (proc.stdout or "")[-2000:],
        "stderr_tail": (proc.stderr or "")[-2000:],
    }
