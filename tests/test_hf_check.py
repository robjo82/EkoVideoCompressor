"""Tests for the Hugging Face access pre-flight.

The SwiftUI Pyannote setup section renders one row per checked
model, with a one-click "Accepter la licence" button driven by the
``license_url`` field. Pin that the engine emits the field on
every entry and that the URL points at the model card the user
needs to visit.
"""

from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from ekovideo_engine.hf import (
    HF_API_BASE,
    HF_GATED_MODEL_CHECKS,
    hf_check,
)


def _stub_urlopen(payload: dict, head_status: int = 200):
    """Cheap test double for ``urllib.request.urlopen``.

    ``GET`` requests (the whoami call) return ``payload`` as JSON.
    ``HEAD`` requests (the per-model access checks) return a bare
    response with the requested status code.
    """

    class _GETResponse:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return json.dumps(payload).encode("utf-8")

    class _HEADResponse:
        status = head_status

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake(request, timeout=None, context=None):
        if request.get_method() == "HEAD":
            return _HEADResponse()
        return _GETResponse()

    return fake


class HfCheckShapeTest(unittest.TestCase):
    def test_emits_license_url_for_every_check(self):
        with patch(
            "ekovideo_engine.hf.urllib.request.urlopen",
            side_effect=_stub_urlopen({"name": "robin", "fullname": "Robin"}),
        ):
            result = hf_check("hf_fake_token")

        self.assertEqual(len(result["checks"]), len(HF_GATED_MODEL_CHECKS))
        for entry, (repo_id, _, label) in zip(result["checks"], HF_GATED_MODEL_CHECKS):
            self.assertEqual(entry["repo_id"], repo_id)
            self.assertEqual(entry["label"], label)
            self.assertTrue(entry["ok"], entry["detail"])
            self.assertEqual(entry["license_url"], f"{HF_API_BASE}/{repo_id}")
            # Sanity check the URL points at the model card root
            # (no leading slash, no trailing path) — that's where
            # the "Agree and access repository" gate lives.
            self.assertTrue(entry["license_url"].startswith("https://huggingface.co/"))


if __name__ == "__main__":
    unittest.main()
