"""Minimal Odoo XML-RPC client for the speaker-profile linkage.

The app stays single-purpose: link a locally-enrolled voice
profile to an Odoo ``res.partner`` so the user sees real people
grouped by company instead of a flat list of voice memos. We
deliberately don't pull in odoorpc / openerplib — we only need
three calls (auth, search_read, name_get) and the stdlib ships
``xmlrpc.client`` for free.

API key auth (settings → Account Security → New API Key in Odoo
17+) is treated as a long-lived password: the user pastes it once
in the SwiftUI Settings panel, the engine forwards it on every
call. We never hit ``/web/session/authenticate`` so a leaked key
is no worse than a leaked DB password.

Every public function returns plain dicts / lists so the CLI layer
can json.dump them straight to stdout for the SwiftUI side.
"""

from __future__ import annotations

import socket
import ssl
import xmlrpc.client
from dataclasses import dataclass


__all__ = [
    "OdooConfig",
    "OdooError",
    "OdooConnectionError",
    "OdooAuthError",
    "test_connection",
    "search_partners",
    "fetch_partner",
]


# Network-level timeout for any single XML-RPC round-trip. Long
# enough that a slow VPN won't trip it on a cold cache; short
# enough that a wedged Odoo doesn't lock up the SwiftUI app.
DEFAULT_TIMEOUT = 12


# Fields we always pull on res.partner. Trim on purpose: anything
# the linker UI doesn't render is just bandwidth + JSON we'd then
# have to throw away on the SwiftUI side.
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
    """Network / TLS / DNS / unreachable host. Distinct from
    OdooAuthError so the SwiftUI status banner can give actionable
    advice ("vérifier l'URL" vs "vérifier la clé API")."""


class OdooAuthError(OdooError):
    """Wrong DB / login / API key combination."""


@dataclass(frozen=True)
class OdooConfig:
    """Bundle of credentials. Built from the SwiftUI settings then
    passed around as a single argument; tests construct one directly.
    """

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


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _normalise_url(raw: str) -> str:
    url = (raw or "").strip().rstrip("/")
    if not url:
        raise OdooConnectionError("L'URL Odoo n'est pas renseignée.")
    if "://" not in url:
        # Default to https when the user pasted just "erp.acme.com".
        url = f"https://{url}"
    return url


def _common_proxy(config: OdooConfig) -> xmlrpc.client.ServerProxy:
    url = _normalise_url(config.url) + "/xmlrpc/2/common"
    # ``allow_none=True`` is needed because Odoo happily returns
    # null for missing fields (e.g. ``parent_id`` on top-level
    # companies); without the flag xmlrpc.client raises on the
    # marshalling.
    return xmlrpc.client.ServerProxy(url, allow_none=True)


def _object_proxy(config: OdooConfig) -> xmlrpc.client.ServerProxy:
    url = _normalise_url(config.url) + "/xmlrpc/2/object"
    return xmlrpc.client.ServerProxy(url, allow_none=True)


def _authenticate(config: OdooConfig) -> int:
    """Resolve the user UID. Raises on any failure with a French
    message the SwiftUI status field can show as-is.
    """
    common = _common_proxy(config)
    socket.setdefaulttimeout(DEFAULT_TIMEOUT)
    try:
        uid = common.authenticate(
            config.database, config.login, config.api_key, {}
        )
    except (xmlrpc.client.ProtocolError, OSError, ssl.SSLError) as exc:
        raise OdooConnectionError(
            f"Connexion à Odoo impossible : {exc}"
        ) from exc
    except xmlrpc.client.Fault as exc:
        raise OdooAuthError(
            f"Identifiants Odoo refusés : {exc.faultString}"
        ) from exc
    if not uid:
        raise OdooAuthError(
            "Identifiants Odoo refusés. Vérifiez la base, le login et la clé API."
        )
    return int(uid)


def _execute(
    config: OdooConfig,
    uid: int,
    model: str,
    method: str,
    args: list,
    kwargs: dict | None = None,
) -> object:
    """Wrapper around ``execute_kw`` that converts XML-RPC's noisy
    failure modes into our two-class hierarchy (connection vs
    auth)."""
    proxy = _object_proxy(config)
    socket.setdefaulttimeout(DEFAULT_TIMEOUT)
    try:
        return proxy.execute_kw(
            config.database, uid, config.api_key, model, method, args, kwargs or {}
        )
    except (xmlrpc.client.ProtocolError, OSError, ssl.SSLError) as exc:
        raise OdooConnectionError(
            f"Connexion Odoo perdue : {exc}"
        ) from exc
    except xmlrpc.client.Fault as exc:
        # Odoo returns Faults for both auth (AccessDenied) and
        # logic (XYZNotFound) errors. Anything that mentions
        # access / token / api key gets bucketed as auth so the UI
        # surfaces the right hint.
        text = str(exc.faultString or "")
        lowered = text.lower()
        if any(token in lowered for token in (
            "access denied", "accessdenied", "session expired", "api key", "invalid token"
        )):
            raise OdooAuthError(text) from exc
        raise OdooError(text) from exc


def _strip_partner_record(record: dict) -> dict:
    """Normalise an Odoo partner dict for shipment to SwiftUI.

    XML-RPC encodes Many2one fields as ``[id, display_name]`` lists
    (or ``False`` when null). We unpack them into ``parent_id`` and
    ``parent_name`` so the SwiftUI side gets a flat object it can
    decode without any "is it a list or false?" branches.
    """
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def test_connection(config: OdooConfig) -> dict:
    """Return ``{ok: True, partner_count, login, server_version}`` on
    success. Raises ``OdooError`` on failure so the CLI layer can
    surface the precise reason in the SwiftUI status field.
    """
    uid = _authenticate(config)
    common = _common_proxy(config)
    try:
        info = common.version() or {}
    except (xmlrpc.client.ProtocolError, OSError, ssl.SSLError) as exc:
        raise OdooConnectionError(
            f"Connexion à Odoo impossible : {exc}"
        ) from exc
    except xmlrpc.client.Fault as exc:
        raise OdooError(str(exc.faultString or exc)) from exc

    # Cheap sanity probe — we know the credentials work, but the
    # API key might be tied to a user that lacks res.partner read
    # access. Catching it here gives a nicer error than a fault on
    # the first real search.
    try:
        partner_count = _execute(
            config, uid, "res.partner", "search_count", [[]], {}
        )
    except OdooError:
        partner_count = -1
    return {
        "ok": True,
        "uid": int(uid),
        "login": config.login,
        "server_version": str(info.get("server_version") or ""),
        "partner_count": int(partner_count or 0),
    }


def search_partners(
    config: OdooConfig,
    query: str,
    *,
    limit: int = 25,
) -> list[dict]:
    """Free-text search for partners by name (case-insensitive).

    Returns the ``_strip_partner_record`` shape, one dict per match.
    An empty query returns an empty list — the linker UI uses that
    as the "show nothing yet" affordance until the user types.
    """
    text = (query or "").strip()
    if not text:
        return []
    uid = _authenticate(config)
    domain = ["|", ("name", "ilike", text), ("email", "ilike", text)]
    records = _execute(
        config,
        uid,
        "res.partner",
        "search_read",
        [domain],
        {"fields": list(_PARTNER_FIELDS), "limit": int(limit)},
    )
    if not isinstance(records, list):
        return []
    return [_strip_partner_record(rec) for rec in records if isinstance(rec, dict)]


def fetch_partner(config: OdooConfig, partner_id: int) -> dict | None:
    """Re-fetch a single partner by id. Used to refresh the cached
    company name on the SwiftUI side when the user wants to
    re-verify a stale link. Returns ``None`` when the id no longer
    exists (deleted / archived in Odoo)."""
    if not partner_id:
        return None
    uid = _authenticate(config)
    records = _execute(
        config,
        uid,
        "res.partner",
        "read",
        [[int(partner_id)]],
        {"fields": list(_PARTNER_FIELDS)},
    )
    if not isinstance(records, list) or not records:
        return None
    record = records[0]
    if not isinstance(record, dict):
        return None
    return _strip_partner_record(record)
