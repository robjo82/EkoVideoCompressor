"""
PR AT — managed-venv dependency freshness.

The venv used to freeze at install-time versions (probe-then-skip,
never upgrade), stranding users on e.g. mlx-vlm 0.4.4. These tests pin
the version-floor logic and the check/upgrade plumbing (subprocess
mocked — no real venv or network).
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from ekovideo_engine import deps


class VersionMathTests(unittest.TestCase):
    def test_parse_version_basic(self):
        self.assertEqual(deps.parse_version("0.31.3"), (0, 31, 3))
        self.assertEqual(deps.parse_version("2.12.0"), (2, 12, 0))

    def test_parse_version_tolerant_of_suffixes(self):
        self.assertEqual(deps.parse_version("2.12.0.dev1"), (2, 12, 0))
        self.assertEqual(deps.parse_version("1.17.0rc2"), (1, 17, 0))
        self.assertEqual(deps.parse_version(""), ())
        self.assertEqual(deps.parse_version("garbage"), ())

    def test_version_lt(self):
        self.assertTrue(deps.version_lt("0.4.4", "0.6.1"))
        self.assertTrue(deps.version_lt("2.11.0", "2.12.0"))
        self.assertFalse(deps.version_lt("0.6.1", "0.6.1"))
        self.assertFalse(deps.version_lt("0.6.2", "0.6.1"))
        # Padding: 2.12 == 2.12.0
        self.assertFalse(deps.version_lt("2.12", "2.12.0"))
        self.assertTrue(deps.version_lt("2.11.9", "2.12"))

    def test_needs_upgrade(self):
        self.assertTrue(deps.needs_upgrade(None, "0.6.1"))
        self.assertTrue(deps.needs_upgrade("", "0.6.1"))
        self.assertTrue(deps.needs_upgrade("0.4.4", "0.6.1"))
        self.assertFalse(deps.needs_upgrade("0.6.1", "0.6.1"))
        self.assertFalse(deps.needs_upgrade("0.7.0", "0.6.1"))


class CheckTests(unittest.TestCase):
    def _fake_installed(self, mapping):
        return patch.object(deps, "read_installed_versions", return_value=mapping)

    def test_flags_outdated_mlx_vlm(self):
        # Mirror the user's real venv: everything current except mlx-vlm.
        installed = {
            "mlx": "0.31.2",
            "mlx-lm": "0.31.3",
            "mlx-vlm": "0.4.4",            # below 0.6.1 floor
            "mlx-whisper": "0.4.3",
            "pyannote.audio": "4.0.4",
            "silero-vad": "6.2.1",
            "transformers": "5.10.1",
            "huggingface-hub": "1.17.0",
            "torch": "2.12.0",
            "torchaudio": "2.12.0",
        }
        with self._fake_installed(installed):
            result = deps.check("/fake/python")
        self.assertTrue(result["any_outdated"])
        by_name = {p["pip_name"]: p for p in result["packages"]}
        self.assertEqual(by_name["mlx-vlm"]["status"], deps.STATUS_OUTDATED)
        self.assertEqual(by_name["mlx-lm"]["status"], deps.STATUS_OK)
        self.assertEqual(by_name["torch"]["status"], deps.STATUS_OK)

    def test_missing_package_flagged(self):
        with self._fake_installed({}):
            result = deps.check("/fake/python")
        self.assertTrue(result["any_outdated"])
        self.assertTrue(
            all(p["status"] == deps.STATUS_MISSING for p in result["packages"])
        )

    def test_all_current_is_clean(self):
        installed = {
            deps._normalise(r.pip_name): r.min_version for r in deps.REQUIREMENTS
        }
        with self._fake_installed(installed):
            result = deps.check("/fake/python")
        self.assertFalse(result["any_outdated"])
        self.assertTrue(all(p["status"] == deps.STATUS_OK for p in result["packages"]))

    def test_pep503_name_normalisation(self):
        # pip reports pyannote.audio as "pyannote-audio" and
        # huggingface-hub as "huggingface_hub" — both must still match.
        self.assertEqual(deps._normalise("pyannote.audio"), "pyannote-audio")
        self.assertEqual(deps._normalise("pyannote-audio"), "pyannote-audio")
        self.assertEqual(deps._normalise("huggingface-hub"), "huggingface-hub")
        self.assertEqual(deps._normalise("huggingface_hub"), "huggingface-hub")
        installed = {
            "pyannote-audio": "4.0.4",   # dash form, as pip emits it
            "huggingface_hub": "1.17.0",  # underscore form
        }
        full = {deps._normalise(r.pip_name): r.min_version for r in deps.REQUIREMENTS}
        full.update({deps._normalise(k): v for k, v in installed.items()})
        with self._fake_installed(full):
            result = deps.check("/fake/python")
        by_name = {p["pip_name"]: p for p in result["packages"]}
        self.assertEqual(by_name["pyannote.audio"]["status"], deps.STATUS_OK)
        self.assertEqual(by_name["huggingface-hub"]["status"], deps.STATUS_OK)

    def test_case_insensitive_name_match(self):
        installed = {"mlx-vlm": "0.6.1"}  # already lowercased by reader
        with self._fake_installed({**{deps._normalise(r.pip_name): r.min_version for r in deps.REQUIREMENTS}, **installed}):
            result = deps.check("/fake/python")
        by_name = {p["pip_name"]: p for p in result["packages"]}
        self.assertEqual(by_name["mlx-vlm"]["status"], deps.STATUS_OK)


class UpgradeTests(unittest.TestCase):
    def test_floor_upgrade_only_touches_outdated(self):
        # Only mlx-vlm is below floor → only it should be in the specs.
        installed = {deps._normalise(r.pip_name): r.min_version for r in deps.REQUIREMENTS}
        installed["mlx-vlm"] = "0.4.4"
        captured = {}

        class FakeProc:
            returncode = 0
            stdout = "ok"
            stderr = ""

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            return FakeProc()

        with patch.object(deps, "read_installed_versions", return_value=installed), \
             patch.object(deps.subprocess, "run", side_effect=fake_run):
            result = deps.upgrade("/fake/python")
        self.assertTrue(result["upgraded"])
        self.assertEqual(result["packages"], ["mlx-vlm"])
        self.assertIn("mlx-vlm>=0.6.1", captured["cmd"])
        # A current package must NOT be in the install line.
        self.assertFalse(any("mlx-whisper" in part for part in captured["cmd"]))

    def test_noop_when_all_current(self):
        installed = {deps._normalise(r.pip_name): r.min_version for r in deps.REQUIREMENTS}
        with patch.object(deps, "read_installed_versions", return_value=installed), \
             patch.object(deps.subprocess, "run") as run:
            result = deps.upgrade("/fake/python")
        self.assertFalse(result["upgraded"])
        self.assertEqual(result["packages"], [])
        run.assert_not_called()  # healthy venv → no pip at all

    def test_all_latest_upgrades_everything_without_floor_pin(self):
        captured = {}

        class FakeProc:
            returncode = 0
            stdout = ""
            stderr = ""

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            return FakeProc()

        with patch.object(deps.subprocess, "run", side_effect=fake_run):
            result = deps.upgrade("/fake/python", to_latest=True)
        self.assertTrue(result["upgraded"])
        self.assertEqual(len(result["packages"]), len(deps.REQUIREMENTS))
        # No ">=" pins in the latest mode — pip resolves newest.
        self.assertIn("mlx-vlm", captured["cmd"])
        self.assertFalse(any(">=" in part for part in captured["cmd"]))

    def test_upgrade_reports_failure(self):
        installed = {deps._normalise(r.pip_name): r.min_version for r in deps.REQUIREMENTS}
        installed["mlx-vlm"] = "0.4.4"

        class FakeProc:
            returncode = 1
            stdout = ""
            stderr = "boom"

        with patch.object(deps, "read_installed_versions", return_value=installed), \
             patch.object(deps.subprocess, "run", return_value=FakeProc()):
            result = deps.upgrade("/fake/python")
        self.assertFalse(result["upgraded"])
        self.assertEqual(result["returncode"], 1)
        self.assertIn("boom", result["stderr_tail"])

    def test_torchaudio_floor_exists_on_pypi(self):
        # Regression for the user's first failed upgrade: a floor that
        # references a NON-EXISTENT version (torchaudio>=2.12.0 — that
        # release doesn't exist, torchaudio stops at 2.11.x) makes pip
        # abort the entire atomic install. Pin the pair coherent.
        by_name = {r.pip_name: r for r in deps.REQUIREMENTS}
        self.assertEqual(
            by_name["torch"].min_version,
            by_name["torchaudio"].min_version,
            "torch et torchaudio doivent partager le même plancher",
        )
        self.assertTrue(deps.version_lt(by_name["torchaudio"].min_version, "2.12.0"))

    def test_partial_failure_falls_back_per_package(self):
        # Combined pip call fails (one bad spec) → per-package retry
        # upgrades what it can and reports only the straggler. This is
        # exactly the failure mode that froze the user's first upgrade:
        # one unresolvable spec must no longer block the others.
        installed = {deps._normalise(r.pip_name): r.min_version for r in deps.REQUIREMENTS}
        installed["mlx-vlm"] = "0.4.4"          # below floor
        installed["transformers"] = "5.8.0"      # below floor

        class FakeProc:
            def __init__(self, rc, err=""):
                self.returncode = rc
                self.stdout = ""
                self.stderr = err

        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            if len(calls) == 1:
                return FakeProc(1, "combined boom")     # combined fails
            if len(calls) == 2:
                return FakeProc(0)                       # mlx-vlm ok
            return FakeProc(1, "still boom")             # transformers fails

        with patch.object(deps, "read_installed_versions", return_value=installed), \
             patch.object(deps.subprocess, "run", side_effect=fake_run):
            result = deps.upgrade("/fake/python")

        self.assertTrue(result["upgraded"])              # something moved
        self.assertEqual(result["packages"], ["mlx-vlm"])
        self.assertEqual(len(result["failed"]), 1)
        self.assertEqual(result["failed"][0]["package"], "transformers")
        self.assertEqual(result["returncode"], 1)        # partial → non-zero
        self.assertEqual(len(calls), 3)                  # combined + 2 retries


if __name__ == "__main__":
    unittest.main()
