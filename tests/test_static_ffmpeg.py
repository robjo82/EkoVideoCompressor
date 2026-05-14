"""
Guard that the binary we're about to ship to users doesn't drag
in Homebrew-only dylibs.

On a developer machine you can run ``scripts/fetch_static_ffmpeg.sh``
and this test will pass. In CI the build workflow runs that script
first; if it ever silently swaps the static binary for a Homebrew
one we'd ship dyld errors to every team member without noticing,
which is exactly what bit us in v0.13.0+.
"""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
BIN_DIR = REPO_ROOT / "bin"


def _otool_refs(binary: Path) -> list[str]:
    """Return the dylib install names referenced by ``binary``.

    Skips the first line (otool repeats the binary path) and the
    leading whitespace each dependency carries.
    """
    proc = subprocess.run(
        ["otool", "-L", str(binary)],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"otool -L {binary} failed: {proc.stderr.strip()}")
    refs: list[str] = []
    for line in proc.stdout.splitlines()[1:]:
        token = line.strip().split(" ", 1)[0]
        if token:
            refs.append(token)
    return refs


def _is_self_contained(binary: Path) -> tuple[bool, list[str]]:
    """A binary is self-contained when every install name points at a
    system library (``/usr/lib`` or ``/System``). Anything else means
    we'd need to also ship that path on the user's machine, which is
    the bug we're guarding against.
    """
    forbidden: list[str] = []
    for ref in _otool_refs(binary):
        if ref.startswith("/usr/lib/") or ref.startswith("/System/"):
            continue
        forbidden.append(ref)
    return (not forbidden, forbidden)


@unittest.skipUnless(sys.platform == "darwin", "macOS-only guard")
class BundledFfmpegIsSelfContainedTest(unittest.TestCase):
    """Test runs only when the static binary has already been fetched.

    The CI build workflow runs ``scripts/fetch_static_ffmpeg.sh`` before
    the test suite, so this is a hard gate in CI. Locally, the test
    is a no-op until the developer downloads the binaries.
    """

    def _assert_self_contained(self, name: str) -> None:
        path = BIN_DIR / name
        if not path.exists():
            self.skipTest(
                f"bin/{name} not present — run scripts/fetch_static_ffmpeg.sh "
                "to populate it before relying on this test."
            )
        if not path.is_file() or not path.stat().st_mode & 0o100:
            self.skipTest(f"bin/{name} is not an executable file on this filesystem")
        ok, forbidden = _is_self_contained(path)
        self.assertTrue(
            ok,
            (
                f"bin/{name} references non-system dylibs that won't exist on a "
                f"user's machine — this is the dyld error we're trying to prevent. "
                f"Run scripts/fetch_static_ffmpeg.sh.\n"
                f"Forbidden references: {forbidden}"
            ),
        )

    def test_ffmpeg_links_only_to_system_libs(self):
        self._assert_self_contained("ffmpeg")

    def test_ffprobe_links_only_to_system_libs(self):
        self._assert_self_contained("ffprobe")


if __name__ == "__main__":
    unittest.main()
