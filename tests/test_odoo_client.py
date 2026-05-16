"""Tests for the Odoo XML-RPC client.

We never actually hit a real Odoo. Every call goes through a
``ServerProxy`` we replace with a stub at module level — that
isolates the parsing / error-mapping logic, which is the only
thing worth pinning here. The XML-RPC transport is stdlib code
we trust.
"""

from __future__ import annotations

import unittest
import xmlrpc.client
from unittest.mock import patch

from odoo_client import (
    OdooAuthError,
    OdooConfig,
    OdooConnectionError,
    OdooError,
    _normalise_url,
    _strip_partner_record,
    fetch_partner,
    search_partners,
    test_connection,
)


class _StubCommonProxy:
    def __init__(self, *, uid: int = 0, fault: Exception | None = None,
                 version_info: dict | None = None):
        self._uid = uid
        self._fault = fault
        self._version_info = version_info or {"server_version": "17.0"}

    def authenticate(self, db, login, key, ctx):  # noqa: D401 — XML-RPC sig
        if self._fault:
            raise self._fault
        return self._uid

    def version(self):
        return self._version_info


class _StubObjectProxy:
    """Records calls and replays canned responses keyed on
    (model, method)."""

    def __init__(self, responses: dict[tuple[str, str], object] | None = None,
                 fault: Exception | None = None):
        self.responses = responses or {}
        self.fault = fault
        self.calls: list[tuple] = []

    def execute_kw(self, db, uid, key, model, method, args, kwargs=None):
        self.calls.append((db, uid, model, method, args, kwargs or {}))
        if self.fault:
            raise self.fault
        return self.responses.get((model, method), [])


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


class TestConnectionTest(unittest.TestCase):
    def _config(self):
        return OdooConfig(
            url="https://erp.acme.com",
            database="acme",
            login="vous@acme.fr",
            api_key="sk-xxx",
        )

    def test_returns_summary_on_success(self):
        with patch(
            "odoo_client._common_proxy",
            return_value=_StubCommonProxy(uid=4, version_info={"server_version": "17.0"}),
        ), patch(
            "odoo_client._object_proxy",
            return_value=_StubObjectProxy(responses={("res.partner", "search_count"): 1234}),
        ):
            summary = test_connection(self._config())
        self.assertTrue(summary["ok"])
        self.assertEqual(summary["uid"], 4)
        self.assertEqual(summary["partner_count"], 1234)
        self.assertEqual(summary["server_version"], "17.0")

    def test_authentication_failure_raises_auth_error(self):
        # Odoo returns ``False`` for bad credentials. Our wrapper
        # promotes that to ``OdooAuthError`` so the SwiftUI status
        # banner can show "vérifiez la clé API".
        with patch(
            "odoo_client._common_proxy", return_value=_StubCommonProxy(uid=False)
        ):
            with self.assertRaises(OdooAuthError):
                test_connection(self._config())

    def test_xmlrpc_protocol_error_maps_to_connection_error(self):
        proto_err = xmlrpc.client.ProtocolError("url", 502, "Bad Gateway", {})
        with patch(
            "odoo_client._common_proxy",
            return_value=_StubCommonProxy(fault=proto_err),
        ):
            with self.assertRaises(OdooConnectionError):
                test_connection(self._config())

    def test_xmlrpc_fault_maps_to_auth_error(self):
        fault = xmlrpc.client.Fault(1, "AccessDenied: invalid api key")
        with patch(
            "odoo_client._common_proxy",
            return_value=_StubCommonProxy(fault=fault),
        ):
            with self.assertRaises(OdooAuthError):
                test_connection(self._config())


class SearchPartnersTest(unittest.TestCase):
    def _config(self):
        return OdooConfig(
            url="https://erp.acme.com",
            database="acme",
            login="vous@acme.fr",
            api_key="sk-xxx",
        )

    def test_blank_query_returns_empty_without_calling_odoo(self):
        with patch("odoo_client._common_proxy") as common, patch(
            "odoo_client._object_proxy"
        ) as obj:
            self.assertEqual(search_partners(self._config(), ""), [])
            self.assertEqual(search_partners(self._config(), "   "), [])
            common.assert_not_called()
            obj.assert_not_called()

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
        with patch(
            "odoo_client._common_proxy", return_value=_StubCommonProxy(uid=4)
        ), patch(
            "odoo_client._object_proxy",
            return_value=_StubObjectProxy(responses={("res.partner", "search_read"): partners}),
        ):
            out = search_partners(self._config(), "robin")
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]["parent_name"], "Acme")
        self.assertEqual(out[0]["function"], "CTO")
        # False email / phone become empty strings, not literal "False".
        self.assertEqual(out[1]["email"], "")
        self.assertEqual(out[1]["phone"], "")


class FetchPartnerTest(unittest.TestCase):
    def test_returns_none_when_partner_does_not_exist(self):
        with patch(
            "odoo_client._common_proxy", return_value=_StubCommonProxy(uid=4)
        ), patch(
            "odoo_client._object_proxy",
            return_value=_StubObjectProxy(responses={("res.partner", "read"): []}),
        ):
            self.assertIsNone(fetch_partner(
                OdooConfig("https://x", "y", "z@a.b", "k"), 99
            ))


if __name__ == "__main__":
    unittest.main()
