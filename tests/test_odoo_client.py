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

from odoo_client import (
    OdooAuthError,
    OdooConfig,
    OdooConnectionError,
    _connection_error_message,
    _exception_chain,
    _is_certificate_error,
    _json2_call,
    _normalise_url,
    _strip_partner_record,
    fetch_partner,
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


if __name__ == "__main__":
    unittest.main()
