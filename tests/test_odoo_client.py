"""Tests for the Odoo JSON-2 client.

We never hit a real Odoo. ``urllib.request.urlopen`` is replaced with
a tiny stub so tests pin the endpoint, headers, payload shape and error
mapping for the Odoo 19 external JSON-2 API.
"""

from __future__ import annotations

import io
import json
import ssl
import unittest
import urllib.error
from unittest.mock import patch

from datetime import datetime, timezone

from odoo_client import (
    OdooAuthError,
    OdooConfig,
    OdooConnectionError,
    _build_chatter_summary,
    _connection_error_message,
    _exception_chain,
    _format_odoo_datetime,
    _html_to_text,
    _is_certificate_error,
    _json2_call,
    _normalise_url,
    _strip_meeting_record,
    _strip_partner_record,
    extract_odoo_glossary_candidates,
    fetch_object_chatter,
    fetch_partner,
    fetch_related_context_pack,
    search_meeting_events,
    search_partners,
    test_connection,
)


class _FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return json.dumps(self.payload, ensure_ascii=False).encode("utf-8")


def _urlopen_replayer(responses, calls):
    queue = list(responses)

    def fake_urlopen(request, timeout=None, context=None):
        calls.append(
            {
                "url": request.full_url,
                "timeout": timeout,
                "context": context,
                "headers": dict(request.header_items()),
                "body": json.loads((request.data or b"{}").decode("utf-8")),
            }
        )
        response = queue.pop(0)
        if isinstance(response, Exception):
            raise response
        return _FakeResponse(response)

    return fake_urlopen


class NormaliseUrlTest(unittest.TestCase):
    def test_blank_raises(self):
        with self.assertRaises(OdooConnectionError):
            _normalise_url("")
        with self.assertRaises(OdooConnectionError):
            _normalise_url("   ")

    def test_adds_https_scheme_when_missing(self):
        self.assertEqual(_normalise_url("erp.acme.com"), "https://erp.acme.com")
        self.assertEqual(_normalise_url("https://erp.acme.com"), "https://erp.acme.com")

    def test_strips_trailing_slash(self):
        self.assertEqual(_normalise_url("https://erp.acme.com/"), "https://erp.acme.com")


class StripPartnerRecordTest(unittest.TestCase):
    def test_unpacks_parent_id_pair(self):
        row = _strip_partner_record({
            "id": 7,
            "name": "Robin Dupuy",
            "parent_id": [42, "Acme"],
            "is_company": False,
            "email": "r@a.com",
        })
        self.assertEqual(row["parent_id"], 42)
        self.assertEqual(row["parent_name"], "Acme")
        self.assertEqual(row["email"], "r@a.com")

    def test_handles_false_parent_id(self):
        row = _strip_partner_record({"id": 7, "name": "Acme", "parent_id": False, "is_company": True})
        self.assertEqual(row["parent_id"], 0)
        self.assertEqual(row["parent_name"], "")
        self.assertTrue(row["is_company"])

    def test_falls_back_to_name_when_display_name_absent(self):
        row = _strip_partner_record({"id": 1, "name": "Solo"})
        self.assertEqual(row["display_name"], "Solo")


class Json2CallTest(unittest.TestCase):
    def _config(self):
        return OdooConfig(
            url="https://erp.acme.com",
            database="acme",
            login="vous@acme.fr",
            api_key="sk-xxx",
        )

    def test_posts_to_json2_with_bearer_key_and_database_header(self):
        calls = []
        with patch(
            "odoo_client.urllib.request.urlopen",
            side_effect=_urlopen_replayer([{"ok": True}], calls),
        ):
            out = _json2_call(
                self._config(),
                "res.partner",
                "search_read",
                {"domain": [["name", "ilike", "Robin"]]},
            )

        self.assertEqual(out, {"ok": True})
        self.assertEqual(calls[0]["url"], "https://erp.acme.com/json/2/res.partner/search_read")
        self.assertEqual(calls[0]["headers"]["Authorization"], "bearer sk-xxx")
        self.assertEqual(calls[0]["headers"]["X-odoo-database"], "acme")
        self.assertEqual(calls[0]["body"]["domain"], [["name", "ilike", "Robin"]])
        self.assertIsNotNone(calls[0]["context"])

    def test_401_maps_to_auth_error(self):
        body = json.dumps({"message": "Invalid apikey"}).encode("utf-8")
        error = urllib.error.HTTPError(
            "https://erp.acme.com/json/2/res.partner/search_count",
            401,
            "Unauthorized",
            {},
            io.BytesIO(body),
        )
        with patch(
            "odoo_client.urllib.request.urlopen",
            side_effect=_urlopen_replayer([error], []),
        ):
            with self.assertRaises(OdooAuthError):
                _json2_call(self._config(), "res.partner", "search_count", {"domain": []})

    def test_url_error_maps_to_connection_error(self):
        with patch(
            "odoo_client.urllib.request.urlopen",
            side_effect=_urlopen_replayer([urllib.error.URLError("offline")], []),
        ):
            with self.assertRaises(OdooConnectionError):
                _json2_call(self._config(), "res.partner", "search_count", {"domain": []})

    def test_certificate_error_gets_actionable_message_and_log(self):
        cert_error = ssl.SSLCertVerificationError(
            "certificate verify failed: self-signed certificate"
        )
        wrapped = urllib.error.URLError(cert_error)

        with patch("odoo_client.append_app_log") as log, patch(
            "odoo_client.urllib.request.urlopen",
            side_effect=_urlopen_replayer([wrapped], []),
        ):
            with self.assertRaises(OdooConnectionError) as ctx:
                _json2_call(self._config(), "res.partner", "search_count", {"domain": []})

        self.assertIn("Certificat TLS Odoo invalide", str(ctx.exception))
        self.assertTrue(any("certificate_error=True" in call.args[0] for call in log.call_args_list))
        self.assertTrue(any("tls=" in call.args[0] for call in log.call_args_list))

    def test_certificate_detection_walks_urlerror_reason(self):
        cert_error = urllib.error.URLError(
            ssl.SSLCertVerificationError("certificate verify failed")
        )

        self.assertTrue(_is_certificate_error(cert_error))
        self.assertIn("SSLCertVerificationError", _exception_chain(cert_error))
        self.assertIn("Certificat TLS Odoo invalide", _connection_error_message(cert_error))


class TestConnectionTest(unittest.TestCase):
    def _config(self):
        return OdooConfig(
            url="https://erp.acme.com",
            database="acme",
            login="vous@acme.fr",
            api_key="sk-xxx",
        )

    def test_returns_summary_on_success(self):
        calls = []
        with patch(
            "odoo_client.urllib.request.urlopen",
            side_effect=_urlopen_replayer(
                [
                    1234,
                    [{"id": 4, "name": "Robin", "login": "vous@acme.fr"}],
                ],
                calls,
            ),
        ):
            summary = test_connection(self._config())

        self.assertTrue(summary["ok"])
        self.assertEqual(summary["uid"], 0)
        self.assertEqual(summary["login"], "Robin")
        self.assertEqual(summary["partner_count"], 1234)
        self.assertEqual(summary["server_version"], "Odoo 19+ JSON-2")
        self.assertEqual(calls[0]["url"], "https://erp.acme.com/json/2/res.partner/search_count")
        self.assertEqual(calls[1]["url"], "https://erp.acme.com/json/2/res.users/search_read")


class SearchPartnersTest(unittest.TestCase):
    def _config(self):
        return OdooConfig(
            url="https://erp.acme.com",
            database="acme",
            login="vous@acme.fr",
            api_key="sk-xxx",
        )

    def test_blank_query_returns_empty_without_calling_odoo(self):
        with patch("odoo_client.urllib.request.urlopen") as urlopen:
            self.assertEqual(search_partners(self._config(), ""), [])
            self.assertEqual(search_partners(self._config(), "   "), [])
            urlopen.assert_not_called()

    def test_returns_normalised_records(self):
        partners = [
            {
                "id": 1, "name": "Robin Dupuy", "display_name": "Robin Dupuy (Acme)",
                "parent_id": [42, "Acme"], "is_company": False,
                "email": "robin@acme.com", "phone": "", "function": "CTO",
            },
            {
                "id": 42, "name": "Acme", "display_name": "Acme",
                "parent_id": False, "is_company": True,
                "email": False, "phone": False, "function": False,
            },
        ]
        calls = []
        with patch(
            "odoo_client.urllib.request.urlopen",
            side_effect=_urlopen_replayer([partners], calls),
        ):
            out = search_partners(self._config(), "robin")
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]["parent_name"], "Acme")
        self.assertEqual(out[0]["function"], "CTO")
        self.assertEqual(out[1]["email"], "")
        self.assertEqual(out[1]["phone"], "")
        self.assertEqual(
            calls[0]["body"]["domain"],
            ["|", ["name", "ilike", "robin"], ["email", "ilike", "robin"]],
        )


class FetchPartnerTest(unittest.TestCase):
    def test_returns_none_when_partner_does_not_exist(self):
        with patch(
            "odoo_client.urllib.request.urlopen",
            side_effect=_urlopen_replayer([[]], []),
        ):
            self.assertIsNone(fetch_partner(
                OdooConfig("https://x", "y", "z@a.b", "k"), 99
            ))


class FormatOdooDatetimeTest(unittest.TestCase):
    def test_naive_datetime_passes_through(self):
        out = _format_odoo_datetime(datetime(2026, 5, 14, 19, 4, 12))
        self.assertEqual(out, "2026-05-14 19:04:12")

    def test_aware_datetime_converted_to_utc_naive(self):
        # Paris is UTC+2 in May, so 19:04 local becomes 17:04 UTC.
        from datetime import timedelta as _td
        paris = datetime(2026, 5, 14, 19, 4, 12, tzinfo=timezone(_td(hours=2)))
        self.assertEqual(_format_odoo_datetime(paris), "2026-05-14 17:04:12")


class StripMeetingRecordTest(unittest.TestCase):
    def test_unpacks_opportunity_id_and_counts_partners(self):
        record = _strip_meeting_record({
            "id": 7,
            "name": "Revue Acritec",
            "start": "2026-05-14 17:00:00",
            "stop": "2026-05-14 18:00:00",
            "duration": 1.0,
            "partner_ids": [1, 2, 3],
            "opportunity_id": [42, "Migration Odoo"],
            "location": "Visio",
        })
        self.assertEqual(record["id"], 7)
        self.assertEqual(record["attendee_count"], 3)
        self.assertEqual(record["duration_minutes"], 60.0)
        self.assertEqual(record["related_object"], {
            "model": "crm.lead", "id": 42, "name": "Migration Odoo",
        })
        self.assertEqual(record["location"], "Visio")

    def test_missing_opportunity_returns_none(self):
        record = _strip_meeting_record({
            "id": 7, "name": "Standup", "partner_ids": [],
            "opportunity_id": False, "duration": 0.5,
        })
        self.assertIsNone(record["related_object"])
        self.assertEqual(record["duration_minutes"], 30.0)
        self.assertEqual(record["attendee_count"], 0)


class SearchMeetingEventsTest(unittest.TestCase):
    def _config(self):
        return OdooConfig(
            url="https://erp.acme.com",
            database="acme",
            login="vous@acme.fr",
            api_key="sk-xxx",
        )

    def test_returns_meetings_with_attendees_expanded(self):
        # First call: calendar.event/search_read returns 1 meeting
        # with partner_ids [10, 11]. Second call: res.partner/read
        # returns the two attendees. The wrapper threads both
        # responses transparently.
        calls = []
        with patch(
            "odoo_client.urllib.request.urlopen",
            side_effect=_urlopen_replayer(
                [
                    [
                        {
                            "id": 7, "name": "Revue Acritec",
                            "start": "2026-05-14 17:00:00",
                            "stop": "2026-05-14 18:00:00",
                            "duration": 1.0, "partner_ids": [10, 11],
                            "opportunity_id": [42, "Migration Odoo"],
                        }
                    ],
                    [
                        {"id": 10, "name": "Robin", "email": "r@acme.fr",
                         "parent_id": [99, "Acritec"], "is_company": False},
                        {"id": 11, "name": "David", "email": "d@acme.fr",
                         "parent_id": [99, "Acritec"], "is_company": False},
                    ],
                ],
                calls,
            ),
        ):
            meetings = search_meeting_events(
                self._config(),
                near=datetime(2026, 5, 14, 19, 0, 0, tzinfo=timezone.utc),
                window_hours=2.0,
            )

        self.assertEqual(len(meetings), 1)
        meeting = meetings[0]
        self.assertEqual([a["name"] for a in meeting["attendees"]], ["Robin", "David"])
        self.assertEqual(meeting["attendees"][0]["company"], "Acritec")
        # The search domain includes the bracket bounds.
        domain = calls[0]["body"]["domain"]
        self.assertEqual(domain[0][0], "start")
        self.assertEqual(domain[1][0], "stop")
        # And the partner expansion uses a single batched read.
        self.assertEqual(calls[1]["url"].split("/")[-2:], ["res.partner", "read"])
        self.assertEqual(sorted(calls[1]["body"]["ids"]), [10, 11])

    def test_empty_calendar_returns_empty_without_partner_lookup(self):
        # No meetings → no partner expansion round-trip wasted.
        calls = []
        with patch(
            "odoo_client.urllib.request.urlopen",
            side_effect=_urlopen_replayer([[]], calls),
        ):
            meetings = search_meeting_events(
                self._config(),
                near=datetime(2026, 5, 14, 19, 0, 0, tzinfo=timezone.utc),
            )
        self.assertEqual(meetings, [])
        # Only one network call (the search_read), no partner read.
        self.assertEqual(len(calls), 1)

    def test_single_guest_calendar_entries_are_ignored(self):
        calls = []
        with patch(
            "odoo_client.urllib.request.urlopen",
            side_effect=_urlopen_replayer(
                [
                    [
                        {
                            "id": 8, "name": "Rappel solo",
                            "start": "2026-05-14 17:00:00",
                            "stop": "2026-05-14 17:30:00",
                            "duration": 0.5, "partner_ids": [10],
                            "opportunity_id": False,
                        },
                        {
                            "id": 9, "name": "Vraie réunion",
                            "start": "2026-05-14 18:00:00",
                            "stop": "2026-05-14 19:00:00",
                            "duration": 1.0, "partner_ids": [10, 11],
                            "opportunity_id": False,
                        },
                    ],
                    [
                        {"id": 10, "name": "Robin", "email": "r@acme.fr",
                         "parent_id": False, "is_company": False},
                        {"id": 11, "name": "David", "email": "d@acme.fr",
                         "parent_id": False, "is_company": False},
                    ],
                ],
                calls,
            ),
        ):
            meetings = search_meeting_events(
                self._config(),
                near=datetime(2026, 5, 14, 19, 0, 0, tzinfo=timezone.utc),
                window_hours=2.0,
            )

        self.assertEqual([meeting["id"] for meeting in meetings], [9])
        self.assertEqual(sorted(calls[1]["body"]["ids"]), [10, 11])


class ChatterHelpersTest(unittest.TestCase):
    def test_html_to_text_strips_tags_and_entities(self):
        out = _html_to_text(
            "<p>Bonjour <a href='x'>Robin</a>,&nbsp;la migration est validée.</p>"
        )
        self.assertEqual(out, "Bonjour Robin, la migration est validée.")

    def test_html_to_text_handles_none_and_blank(self):
        self.assertEqual(_html_to_text(""), "")
        self.assertEqual(_html_to_text(None), "")  # type: ignore[arg-type]

    def test_chatter_summary_concatenates_recent_messages(self):
        summary = _build_chatter_summary(
            "Migration Odoo",
            [
                {"date": "2026-05-13", "author": "Florence",
                 "body": "On lance le go le 14."},
                {"date": "2026-05-12", "author": "Antoine",
                 "body": "Validé côté technique."},
            ],
        )
        self.assertIn("Migration Odoo", summary)
        self.assertIn("Florence", summary)
        self.assertIn("Validé", summary)

    def test_chatter_summary_caps_at_prompt_budget(self):
        # 50 long bodies should not blow through the 1800-char cap
        # — the formatter walks until budget is exhausted then
        # stops without truncating mid-message.
        messages = [
            {"date": f"2026-05-{i:02d}", "author": "Spam",
             "body": "x" * 400}
            for i in range(1, 50)
        ]
        summary = _build_chatter_summary("Sujet", messages)
        self.assertLessEqual(len(summary), 1900)


class FetchObjectChatterTest(unittest.TestCase):
    def _config(self):
        return OdooConfig(
            url="https://erp.acme.com",
            database="acme",
            login="vous@acme.fr",
            api_key="sk-xxx",
        )

    def test_returns_summary_and_messages(self):
        calls = []
        with patch(
            "odoo_client.urllib.request.urlopen",
            side_effect=_urlopen_replayer(
                [
                    # 1st: read(crm.lead, 42) → display_name + message_ids
                    [
                        {"id": 42, "display_name": "Migration Odoo Acritec",
                         "message_ids": [101, 102, 103]}
                    ],
                    # 2nd: read(mail.message, [101,102,103]) → bodies
                    [
                        {"id": 101, "date": "2026-05-13 14:30:00",
                         "author_id": [7, "Florence"],
                         "body": "<p>On lance le go le 14.</p>",
                         "message_type": "comment"},
                        {"id": 102, "date": "2026-05-12 09:10:00",
                         "author_id": [8, "Antoine"],
                         "body": "Validé côté technique.",
                         "message_type": "comment"},
                        {"id": 103, "date": "2026-05-11 17:00:00",
                         "author_id": [9, "Robin"],
                         "body": "Plan revu.",
                         "message_type": "comment"},
                    ],
                ],
                calls,
            ),
        ):
            payload = fetch_object_chatter(
                self._config(), "crm.lead", 42, limit=3,
            )
        self.assertEqual(payload["display_name"], "Migration Odoo Acritec")
        self.assertEqual(len(payload["messages"]), 3)
        # Messages sorted newest-first.
        self.assertEqual(payload["messages"][0]["author"], "Florence")
        # Summary carries the display name + the most recent body.
        self.assertIn("Migration Odoo Acritec", payload["summary"])
        self.assertIn("Florence", payload["summary"])

    def test_missing_record_returns_empty_payload(self):
        with patch(
            "odoo_client.urllib.request.urlopen",
            side_effect=_urlopen_replayer([[]], []),
        ):
            payload = fetch_object_chatter(
                self._config(), "crm.lead", 99,
            )
        self.assertEqual(payload["messages"], [])
        self.assertEqual(payload["summary"], "")

    def test_blank_model_returns_empty_without_network(self):
        with patch("odoo_client.urllib.request.urlopen") as mock:
            payload = fetch_object_chatter(
                self._config(), "", 0,
            )
            mock.assert_not_called()
        self.assertEqual(payload["messages"], [])


class FetchRelatedContextPackTest(unittest.TestCase):
    """The pack drives the LLM correction pass and the Whisper
    glossary boost. Pin three concerns:

    1. A ``crm.lead`` recurses into its linked ``sale.order``
       quotations.
    2. A ``project.task`` recurses into its parent
       ``project.project``.
    3. Compression respects ``max_total_chars`` even when every
       record carries huge chatter.
    """

    def _config(self):
        return OdooConfig(
            url="https://erp.acme.com",
            database="acme",
            login="vous@acme.fr",
            api_key="sk-xxx",
        )

    def test_crm_lead_recurses_into_sale_orders(self):
        calls: list[dict] = []
        responses = [
            # read crm.lead 42
            [{
                "id": 42,
                "display_name": "Migration Acritec",
                "description": "<p>Refonte de l'ERP pour Acritec</p>",
                "partner_id": [99, "Acritec SAS"],
            }],
            # fetch_object_chatter → read message_ids
            [{"id": 42, "display_name": "Migration Acritec", "message_ids": [1]}],
            # fetch_object_chatter → read messages
            [{"id": 1, "date": "2026-05-10 09:00:00",
              "author_id": [7, "Florence"], "subject": "",
              "body": "<p>Kickoff prévu le 15.</p>",
              "message_type": "comment"}],
            # search_read sale.order where opportunity_id=42
            [{"id": 7, "display_name": "SO/2026/001",
              "name": "SO/2026/001", "state": "sent",
              "partner_id": [99, "Acritec SAS"],
              "amount_total": 12000}],
            # read sale.order 7
            [{"id": 7, "display_name": "SO/2026/001",
              "name": "SO/2026/001", "state": "sent",
              "partner_id": [99, "Acritec SAS"],
              "amount_total": 12000, "opportunity_id": [42, "Migration"]}],
            # fetch_object_chatter on the quote → message_ids
            [{"id": 7, "display_name": "SO/2026/001", "message_ids": [11]}],
            # fetch_object_chatter on the quote → messages
            [{"id": 11, "date": "2026-05-12 11:00:00",
              "author_id": [9, "Robin"], "subject": "",
              "body": "Validé par le client.",
              "message_type": "comment"}],
        ]
        with patch(
            "odoo_client.urllib.request.urlopen",
            side_effect=_urlopen_replayer(responses, calls),
        ):
            pack = fetch_related_context_pack(
                self._config(), "crm.lead", 42, max_total_chars=4000,
            )
        self.assertEqual(pack["primary"]["model"], "crm.lead")
        self.assertEqual(pack["primary"]["display_name"], "Migration Acritec")
        # The lead's quote was discovered.
        self.assertEqual(len(pack["related"]), 1)
        self.assertEqual(pack["related"][0]["model"], "sale.order")
        # Summary carries both sections.
        self.assertIn("[Opportunité] Migration Acritec", pack["summary"])
        self.assertIn("[Devis] SO/2026/001", pack["summary"])
        # Glossary candidates include the customer and the quote ref.
        self.assertIn("Acritec SAS", pack["terms"])

    def test_project_task_recurses_into_parent_project(self):
        calls: list[dict] = []
        responses = [
            # read project.task 5
            [{
                "id": 5,
                "display_name": "Onboarding RH",
                "description": "<p>Configurer les badges</p>",
                "project_id": [3, "Acritec — RH"],
            }],
            # task chatter (no messages)
            [{"id": 5, "display_name": "Onboarding RH", "message_ids": []}],
            # read project.project 3
            [{
                "id": 3,
                "display_name": "Acritec — RH",
                "description": "<p>Projet RH Acritec</p>",
                "partner_id": [99, "Acritec SAS"],
            }],
            # project chatter (no messages)
            [{"id": 3, "display_name": "Acritec — RH", "message_ids": []}],
        ]
        with patch(
            "odoo_client.urllib.request.urlopen",
            side_effect=_urlopen_replayer(responses, calls),
        ):
            pack = fetch_related_context_pack(
                self._config(), "project.task", 5,
            )
        self.assertEqual(pack["primary"]["model"], "project.task")
        self.assertEqual(len(pack["related"]), 1)
        self.assertEqual(pack["related"][0]["model"], "project.project")
        self.assertIn("[Tâche] Onboarding RH", pack["summary"])
        self.assertIn("[Projet] Acritec — RH", pack["summary"])

    def test_blank_inputs_short_circuit_without_network(self):
        with patch("odoo_client.urllib.request.urlopen") as mock:
            pack = fetch_related_context_pack(self._config(), "", 0)
            mock.assert_not_called()
        self.assertEqual(pack["primary"], {})
        self.assertEqual(pack["related"], [])
        self.assertEqual(pack["summary"], "")
        self.assertEqual(pack["terms"], [])

    def test_compression_caps_summary_at_budget(self):
        # Single primary record with a huge body — verify
        # ``max_total_chars`` actually bites. The recursion call
        # for sale.order returns an empty list (no quotes), so we
        # supply a 3rd response for that search.
        huge_body = "Lorem ipsum dolor sit amet, " * 200  # ~5600 chars
        calls: list[dict] = []
        responses = [
            [{"id": 1, "display_name": "Big record",
              "description": f"<p>{huge_body}</p>", "partner_id": False}],
            [{"id": 1, "display_name": "Big record", "message_ids": []}],
            [],  # _safe_search_read(sale.order, opportunity_id=1) — no quotes
        ]
        with patch(
            "odoo_client.urllib.request.urlopen",
            side_effect=_urlopen_replayer(responses, calls),
        ):
            pack = fetch_related_context_pack(
                self._config(), "crm.lead", 1, max_total_chars=1000,
            )
        # The 400-char per-model excerpt + the header ~ 420 chars,
        # comfortably under the 1000 budget. The point: nothing
        # exploded and the summary stays bounded.
        self.assertLessEqual(len(pack["summary"]), 1000)
        self.assertTrue(pack["summary"].startswith("[Opportunité]"))


class ExtractOdooGlossaryCandidatesTest(unittest.TestCase):
    """The Whisper initial prompt gets a glossary boost from these
    candidates. Pin the dedupe, the stopword filter, and the
    explicit promotion of customer / project / task names so they
    still surface when the regex would have missed them."""

    def test_promotes_explicit_partner_and_record_names(self):
        primary = {
            "model": "crm.lead",
            "id": 1,
            "display_name": "Migration Acritec",
            "raw": {"partner_id": [99, "Acritec SAS"]},
            "body": "",
            "chatter": [],
        }
        terms = extract_odoo_glossary_candidates(primary, [])
        self.assertIn("Migration Acritec", terms)
        self.assertIn("Acritec SAS", terms)

    def test_drops_french_stopwords(self):
        primary = {
            "model": "crm.lead",
            "id": 1,
            "display_name": "",
            "raw": {},
            "body": "Bonjour, ravi de vous voir. Cordialement, Robin.",
            "chatter": [],
        }
        terms = extract_odoo_glossary_candidates(primary, [])
        for stopword in ("Bonjour", "Cordialement"):
            self.assertNotIn(stopword, terms)
        self.assertIn("Robin", terms)

    def test_caps_at_max_terms(self):
        body = " ".join(f"Personne{i}" for i in range(200))
        primary = {
            "model": "crm.lead", "id": 1, "display_name": "",
            "raw": {}, "body": body, "chatter": [],
        }
        terms = extract_odoo_glossary_candidates(primary, [], max_terms=8)
        self.assertEqual(len(terms), 8)


if __name__ == "__main__":
    unittest.main()
