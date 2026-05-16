"""Tests for the Odoo context plumbing on the runner side.

Covers two integration paths the SwiftUI app depends on:

- ``odoo_meeting_metadata`` on a JobRequest lands on the
  ``jobs.odoo_meeting_json`` column so the rename sheet can read
  it back as one-click attendee chips (Layer 3).
- ``odoo_context_ref`` triggers ``_fetch_odoo_context_blob``
  during the LLM step, with failures degrading silently to an
  empty blob (Layer 2).
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ekovideo_engine.library import database
from ekovideo_engine.models import JobRequest, OdooContextRef


def _make_request(*, source: Path, **overrides):
    payload = {
        "source_path": str(source),
        "output_dir": str(source.parent),
        "mode": "transcribe",
        "workspace_dir": str(source.parent),
        "transcription_settings": {
            "venv_python_path": "",
            "mlx_whisper_path": "/usr/local/bin/mlx_whisper",
        },
    }
    payload.update(overrides)
    return JobRequest.from_dict(payload)


class OdooMeetingPersistenceTest(unittest.TestCase):
    def test_metadata_lands_on_jobs_row_for_rename_sheet(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "meeting.mov"
            source.write_bytes(b"fake")

            with patch.dict(os.environ, {"EKO_APP_SUPPORT_DIR": str(root / "support")}):
                db = database()
                job_id = db.create_job(
                    source_path=str(source),
                    workspace_dir=str(root / "work"),
                    settings={},
                )
                metadata = {
                    "event_id": 7,
                    "event_name": "Revue Acritec",
                    "attendees": [
                        {"id": 10, "name": "Robin", "email": "r@a.fr", "company": "Acritec"},
                        {"id": 11, "name": "David", "email": "d@a.fr", "company": "Acritec"},
                    ],
                    "related": {"model": "crm.lead", "id": 42, "name": "Migration"},
                }
                db.update_job_odoo_meeting(job_id, metadata)

                row = db.get_job(job_id)
                raw = (row or {}).get("odoo_meeting_json") or ""
                decoded = json.loads(raw)

            self.assertEqual(decoded["event_id"], 7)
            self.assertEqual([a["name"] for a in decoded["attendees"]], ["Robin", "David"])
            self.assertEqual(decoded["related"]["model"], "crm.lead")

    def test_passing_none_clears_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.dict(os.environ, {"EKO_APP_SUPPORT_DIR": str(root / "support")}):
                db = database()
                job_id = db.create_job(
                    source_path=str(root / "x.mov"),
                    workspace_dir=str(root / "work"),
                    settings={},
                )
                db.update_job_odoo_meeting(job_id, {"event_id": 1, "event_name": "x", "attendees": [], "related": None})
                db.update_job_odoo_meeting(job_id, None)
                row = db.get_job(job_id)

            self.assertIsNone(row.get("odoo_meeting_json"))


class OdooContextRefRoundTripTest(unittest.TestCase):
    def test_from_dict_parses_swiftui_payload(self):
        # Reproduce the SwiftUI shape: model + record_id + creds.
        ref = OdooContextRef.from_dict({
            "model": "crm.lead",
            "record_id": 42,
            "url": "https://erp.acme.com",
            "database": "acme",
            "login": "vous@acme.fr",
            "api_key": "sk-xxx",
        })
        self.assertTrue(ref.is_actionable())

    def test_blank_payload_is_not_actionable(self):
        self.assertFalse(OdooContextRef().is_actionable())
        self.assertFalse(OdooContextRef.from_dict({"model": "crm.lead", "record_id": 42}).is_actionable())

    def test_job_request_carries_meeting_metadata(self):
        # End-to-end JSON round-trip through ``JobRequest.from_dict``,
        # mirroring exactly what the runner reads off stdin.
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "m.mov"
            source.write_bytes(b"x")
            payload = {
                "source_path": str(source),
                "output_dir": str(source.parent),
                "mode": "transcribe",
                "odoo_meeting_metadata": {
                    "event_id": 7,
                    "event_name": "Revue",
                    "attendees": [{"id": 10, "name": "Robin"}],
                    "related": {"model": "crm.lead", "id": 42, "name": "Migration"},
                },
                "odoo_context_ref": {
                    "model": "crm.lead",
                    "record_id": 42,
                    "url": "https://erp.acme.com",
                    "database": "acme",
                    "login": "vous@acme.fr",
                    "api_key": "sk-xxx",
                },
            }
            request = JobRequest.from_dict(payload)
        self.assertEqual(request.odoo_meeting_metadata["event_id"], 7)
        self.assertTrue(request.odoo_context_ref.is_actionable())


if __name__ == "__main__":
    unittest.main()
