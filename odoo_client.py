"""Minimal Odoo 19 JSON-2 client for speaker-profile linkage.

The app only needs to link a locally-enrolled voice profile to an
Odoo ``res.partner``. Odoo 19 introduced the external JSON-2 API as
the successor to XML-RPC / JSON-RPC for model calls, so this module
uses plain HTTP POST requests against ``/json/2/<model>/<method>``.

The SwiftUI side stores an Odoo URL, database, login hint and API key.
JSON-2 authenticates with the API key as a bearer token; the login is
kept as a human label and, when possible, checked against ``res.users``
for a friendlier connection status.
"""

from __future__ import annotations

import json
import socket
import ssl
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


__all__ = [
    "OdooConfig",
    "OdooError",
    "OdooConnectionError",
    "OdooAuthError",
    "test_connection",
    "search_partners",
    "fetch_partner",
]


DEFAULT_TIMEOUT = 12


_PARTNER_FIELDS = (
    "id",
    "name",
    "display_name",
    "parent_id",
    "is_company",
    "email",
    "phone",
    "function",
)


class OdooError(RuntimeError):
    """Anything the Odoo client can't recover from."""


class OdooConnectionError(OdooError):
    """Network / TLS / DNS / unreachable host."""


class OdooAuthError(OdooError):
    """Wrong database / API key / access rights combination."""


@dataclass(frozen=True)
class OdooConfig:
    url: str
    database: str
    login: str
    api_key: str

    def is_configured(self) -> bool:
        return all(
            (
                (self.url or "").strip(),
                self.database.strip(),
                self.login.strip(),
                self.api_key.strip(),
            )
        )


def _normalise_url(raw: str) -> str:
    url = (raw or "").strip().rstrip("/")
    if not url:
        raise OdooConnectionError("L'URL Odoo n'est pas renseignée.")
    if "://" not in url:
        url = f"https://{url}"
    return url


def _json2_base_url(config: OdooConfig) -> str:
    return _normalise_url(config.url) + "/json/2"


def _error_message_from_payload(payload: Any) -> str:
    if isinstance(payload, dict):
        data = payload.get("data")
        for value in (
            payload.get("message"),
            data.get("message") if isinstance(data, dict) else None,
            data.get("debug") if isinstance(data, dict) else None,
            payload.get("error"),
        ):
            if value:
                return str(value)
    return ""


def _json2_call(
    config: OdooConfig,
    model: str,
    method: str,
    payload: dict[str, Any] | None = None,
) -> Any:
    url = f"{_json2_base_url(config)}/{model}/{method}"
    body = json.dumps(payload or {}, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Authorization": f"bearer {config.api_key.strip()}",
            "Content-Type": "application/json; charset=utf-8",
            "X-Odoo-Database": config.database.strip(),
            "User-Agent": "EkoVideoCompressor/odoo-json2",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=DEFAULT_TIMEOUT) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raw_error = exc.read().decode("utf-8", errors="replace")
        try:
            payload_error = json.loads(raw_error) if raw_error else {}
        except json.JSONDecodeError:
            payload_error = {}
        message = _error_message_from_payload(payload_error) or raw_error or str(exc)
        if exc.code in {401, 403}:
            raise OdooAuthError(
                f"Accès Odoo refusé par l'API JSON-2 : {message}"
            ) from exc
        raise OdooError(f"Erreur Odoo HTTP {exc.code} : {message}") from exc
    except (urllib.error.URLError, TimeoutError, OSError, ssl.SSLError, socket.timeout) as exc:
        raise OdooConnectionError(f"Connexion à Odoo impossible : {exc}") from exc

    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise OdooError(f"Réponse Odoo JSON-2 illisible : {exc}") from exc


def _strip_partner_record(record: dict) -> dict:
    """Normalise an Odoo partner dict for shipment to SwiftUI."""

    parent = record.get("parent_id")
    parent_id = 0
    parent_name = ""
    if isinstance(parent, (list, tuple)) and len(parent) >= 2:
        try:
            parent_id = int(parent[0])
        except (TypeError, ValueError):
            parent_id = 0
        parent_name = str(parent[1])
    return {
        "id": int(record.get("id") or 0),
        "name": str(record.get("name") or ""),
        "display_name": str(
            record.get("display_name") or record.get("name") or ""
        ),
        "parent_id": parent_id,
        "parent_name": parent_name,
        "is_company": bool(record.get("is_company") or False),
        "email": str(record.get("email") or "") if record.get("email") else "",
        "phone": str(record.get("phone") or "") if record.get("phone") else "",
        "function": str(record.get("function") or "") if record.get("function") else "",
    }


def _current_user_label(config: OdooConfig) -> str:
    """Best-effort label for the Settings status line.

    JSON-2 does not expose the old XML-RPC ``common.authenticate`` UID
    flow. The API key already authenticates the request, so this helper
    only tries to resolve the configured login for nicer UI text.
    """

    try:
        users = _json2_call(
            config,
            "res.users",
            "search_read",
            {
                "domain": [["login", "=", config.login.strip()]],
                "fields": ["name", "login"],
                "limit": 1,
            },
        )
    except OdooError:
        return config.login
    if isinstance(users, list) and users:
        first = users[0]
        if isinstance(first, dict):
            return str(first.get("name") or first.get("login") or config.login)
    return config.login


def test_connection(config: OdooConfig) -> dict:
    """Return ``{ok: True, partner_count, login, server_version}``.

    ``server_version`` is kept for the existing SwiftUI/test payload
    shape, but JSON-2's model endpoint does not expose the old common
    service version call. We therefore report the transport explicitly.
    """

    partner_count = _json2_call(
        config,
        "res.partner",
        "search_count",
        {"domain": []},
    )
    return {
        "ok": True,
        "uid": 0,
        "login": _current_user_label(config),
        "server_version": "Odoo 19+ JSON-2",
        "partner_count": int(partner_count or 0),
    }


def search_partners(
    config: OdooConfig,
    query: str,
    *,
    limit: int = 25,
) -> list[dict]:
    """Free-text search for partners by name / email."""

    text = (query or "").strip()
    if not text:
        return []
    domain = ["|", ["name", "ilike", text], ["email", "ilike", text]]
    records = _json2_call(
        config,
        "res.partner",
        "search_read",
        {
            "domain": domain,
            "fields": list(_PARTNER_FIELDS),
            "limit": int(limit),
        },
    )
    if not isinstance(records, list):
        return []
    return [_strip_partner_record(rec) for rec in records if isinstance(rec, dict)]


def fetch_partner(config: OdooConfig, partner_id: int) -> dict | None:
    if not partner_id:
        return None
    records = _json2_call(
        config,
        "res.partner",
        "read",
        {
            "ids": [int(partner_id)],
            "fields": list(_PARTNER_FIELDS),
        },
    )
    if not isinstance(records, list) or not records:
        return None
    record = records[0]
    if not isinstance(record, dict):
        return None
    return _strip_partner_record(record)
