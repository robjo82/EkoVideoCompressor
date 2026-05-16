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
import os
import re
import socket
import ssl
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlparse

from ekovideo_engine.logging import append_app_log, tail_text


__all__ = [
    "OdooConfig",
    "OdooError",
    "OdooConnectionError",
    "OdooAuthError",
    "test_connection",
    "fetch_object_chatter",
    "search_meeting_events",
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

# Fields the meeting-event lookup pulls. ``opportunity_id`` and
# ``task_id`` are the two canonical "related object" links Odoo
# 17+ exposes on ``calendar.event``; we read them defensively so a
# DB without those columns (e.g. base Community without CRM)
# doesn't trip the request.
_MEETING_FIELDS = (
    "id",
    "name",
    "start",
    "stop",
    "duration",
    "allday",
    "partner_ids",
    "attendee_ids",
    "description",
    "location",
    "opportunity_id",
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


def _safe_host(value: str) -> str:
    try:
        parsed = urlparse(_normalise_url(value))
    except OdooConnectionError:
        return "<invalid-url>"
    return parsed.netloc or parsed.path or "<unknown-host>"


def _exception_chain(exc: BaseException) -> str:
    parts: list[str] = []
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        parts.append(f"{current.__class__.__name__}: {current}")
        if isinstance(current, urllib.error.URLError) and isinstance(current.reason, BaseException):
            current = current.reason
        else:
            current = current.__cause__ or current.__context__
    return " <- ".join(parts)


def _is_certificate_error(exc: BaseException) -> bool:
    text = _exception_chain(exc).lower()
    return any(
        token in text
        for token in (
            "certificate verify failed",
            "self-signed certificate",
            "hostname",
            "certificat",
            "sslcertverificationerror",
        )
    )


def _connection_error_message(exc: BaseException) -> str:
    if _is_certificate_error(exc):
        return (
            "Certificat TLS Odoo invalide ou non approuvé par macOS. "
            "Vérifiez le certificat HTTPS du serveur Odoo, sa chaîne "
            "intermédiaire et le nom de domaine utilisé."
        )
    return f"Connexion à Odoo impossible : {exc}"


def _certifi_cafile() -> str | None:
    try:
        import certifi  # type: ignore
    except Exception:
        return None
    try:
        path = certifi.where()
    except Exception:
        return None
    if path and os.path.exists(path):
        return path
    return None


def _ssl_context() -> tuple[ssl.SSLContext, str]:
    cafile = _certifi_cafile()
    if cafile:
        return ssl.create_default_context(cafile=cafile), f"certifi:{cafile}"
    return ssl.create_default_context(), "system-default"


def _tls_diagnostics(active_ca: str) -> str:
    paths = ssl.get_default_verify_paths()
    diagnostics = {
        "active_ca": active_ca,
        "openssl": ssl.OPENSSL_VERSION,
        "certifi_cafile": _certifi_cafile(),
        "cafile": paths.cafile,
        "capath": paths.capath,
        "openssl_cafile_env": paths.openssl_cafile_env,
        "openssl_cafile": paths.openssl_cafile,
        "openssl_capath_env": paths.openssl_capath_env,
        "openssl_capath": paths.openssl_capath,
    }
    return " ".join(f"{key}={value!r}" for key, value in diagnostics.items())


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
    host = _safe_host(config.url)
    context, active_ca = _ssl_context()
    append_app_log(
        "odoo_json2_request "
        f"host={host!r} db={config.database.strip()!r} "
        f"model={model!r} method={method!r} tls_ca={active_ca!r}"
    )
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
        with urllib.request.urlopen(request, timeout=DEFAULT_TIMEOUT, context=context) as response:
            raw = response.read().decode("utf-8")
            status = getattr(response, "status", None) or getattr(response, "code", None)
            append_app_log(
                "odoo_json2_response "
                f"host={host!r} model={model!r} method={method!r} "
                f"status={status!r} bytes={len(raw.encode('utf-8'))}"
            )
    except urllib.error.HTTPError as exc:
        raw_error = exc.read().decode("utf-8", errors="replace")
        try:
            payload_error = json.loads(raw_error) if raw_error else {}
        except json.JSONDecodeError:
            payload_error = {}
        message = _error_message_from_payload(payload_error) or raw_error or str(exc)
        append_app_log(
            "odoo_json2_http_error "
            f"host={host!r} model={model!r} method={method!r} "
            f"status={exc.code} error={tail_text(message, 1200)!r}"
        )
        if exc.code in {401, 403}:
            raise OdooAuthError(
                f"Accès Odoo refusé par l'API JSON-2 : {message}"
            ) from exc
        raise OdooError(f"Erreur Odoo HTTP {exc.code} : {message}") from exc
    except (urllib.error.URLError, TimeoutError, OSError, ssl.SSLError, socket.timeout) as exc:
        is_certificate_error = _is_certificate_error(exc)
        extra = f" tls={tail_text(_tls_diagnostics(active_ca), 1200)!r}" if is_certificate_error else ""
        append_app_log(
            "odoo_json2_connection_error "
            f"host={host!r} model={model!r} method={method!r} "
            f"certificate_error={is_certificate_error} "
            f"error_chain={tail_text(_exception_chain(exc), 2000)!r}"
            f"{extra}"
        )
        raise OdooConnectionError(_connection_error_message(exc)) from exc

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


# ---------------------------------------------------------------------------
# Calendar event discovery (Run Setup "Suggestions Odoo" section)
# ---------------------------------------------------------------------------


def _format_odoo_datetime(value: datetime) -> str:
    """Odoo's JSON-2 expects naive UTC strings (``YYYY-MM-DD HH:MM:SS``).

    SwiftUI passes the file's modification time as an ISO 8601 string
    we parse on the way in; the formatter normalises back to that
    shape so domain comparisons stay timezone-aligned.
    """
    if value.tzinfo is not None:
        value = value.astimezone(timezone.utc).replace(tzinfo=None)
    return value.strftime("%Y-%m-%d %H:%M:%S")


def _strip_meeting_record(record: dict) -> dict:
    """Flatten an Odoo ``calendar.event`` for SwiftUI consumption.

    Many2one fields (``opportunity_id``) come back as ``[id, name]``
    pairs or ``False`` — unpacked into a plain
    ``related_object: {model, id, name}`` dict so the Swift side
    doesn't have to branch.
    """
    related: dict | None = None
    opportunity = record.get("opportunity_id")
    if isinstance(opportunity, (list, tuple)) and len(opportunity) >= 2:
        try:
            related = {
                "model": "crm.lead",
                "id": int(opportunity[0]),
                "name": str(opportunity[1]),
            }
        except (TypeError, ValueError):
            related = None

    # Attendee identities live in ``partner_ids`` as a list of ids.
    # We don't expand them here — the caller can do a second
    # batched ``read`` if they want richer info. Surface raw ids so
    # the SwiftUI side knows how many people were invited.
    raw_partners = record.get("partner_ids") or []
    partner_ids: list[int] = []
    if isinstance(raw_partners, list):
        for value in raw_partners:
            try:
                partner_ids.append(int(value))
            except (TypeError, ValueError):
                continue

    duration = record.get("duration")
    try:
        duration_minutes = float(duration) * 60.0 if duration else 0.0
    except (TypeError, ValueError):
        duration_minutes = 0.0

    return {
        "id": int(record.get("id") or 0),
        "name": str(record.get("name") or ""),
        "start": str(record.get("start") or ""),
        "stop": str(record.get("stop") or ""),
        "duration_minutes": duration_minutes,
        "allday": bool(record.get("allday") or False),
        "location": str(record.get("location") or "") if record.get("location") else "",
        "description": str(record.get("description") or "") if record.get("description") else "",
        "partner_ids": partner_ids,
        "attendee_count": len(partner_ids),
        "related_object": related,
    }


def _attendees_for_meetings(
    config: OdooConfig,
    partner_ids: list[int],
) -> dict[int, dict]:
    """Batched ``res.partner.read`` for every partner referenced by
    the meetings we just fetched. Returns ``{id: stripped_record}``.

    Done in one round-trip rather than N to keep the suggestion
    list snappy even when several meetings invite the same crowd.
    """
    if not partner_ids:
        return {}
    unique = sorted({int(p) for p in partner_ids if p})
    if not unique:
        return {}
    records = _json2_call(
        config,
        "res.partner",
        "read",
        {
            "ids": unique,
            "fields": list(_PARTNER_FIELDS),
        },
    )
    if not isinstance(records, list):
        return {}
    out: dict[int, dict] = {}
    for raw in records:
        if not isinstance(raw, dict):
            continue
        clean = _strip_partner_record(raw)
        out[int(clean.get("id") or 0)] = clean
    return out


def search_meeting_events(
    config: OdooConfig,
    *,
    near: datetime | None = None,
    window_hours: float = 2.0,
    limit: int = 10,
) -> list[dict]:
    """Return the ``calendar.event`` records that bracket ``near``.

    The domain looks for events whose [start, stop] window overlaps
    [near - window_hours, near + window_hours]. ``near`` defaults to
    "right now", which is what an `Enregistrer dans le moment`
    workflow expects.

    Each returned dict is augmented with an ``attendees`` list of
    flat ``{id, name, email}`` records (resolved via a single batched
    ``res.partner.read``). Returned in ascending start order.
    """
    moment = near or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    window = timedelta(hours=max(0.25, window_hours))
    earliest = _format_odoo_datetime(moment - window)
    latest = _format_odoo_datetime(moment + window)

    # Overlap test: start <= latest AND stop >= earliest.
    domain: list = [
        ("start", "<=", latest),
        ("stop", ">=", earliest),
    ]
    records = _json2_call(
        config,
        "calendar.event",
        "search_read",
        {
            "domain": domain,
            "fields": list(_MEETING_FIELDS),
            "limit": int(limit),
            "order": "start asc",
        },
    )
    if not isinstance(records, list):
        return []

    stripped = [
        meeting
        for meeting in (_strip_meeting_record(r) for r in records if isinstance(r, dict))
        if len(meeting.get("partner_ids") or []) > 1
    ]

    # Expand attendees once for the whole batch.
    every_partner_id: list[int] = []
    for meeting in stripped:
        every_partner_id.extend(meeting.get("partner_ids") or [])
    attendee_map = _attendees_for_meetings(config, every_partner_id)
    for meeting in stripped:
        attendees: list[dict] = []
        for pid in meeting.get("partner_ids") or []:
            partner = attendee_map.get(int(pid))
            if not partner:
                continue
            attendees.append(
                {
                    "id": int(partner.get("id") or 0),
                    "name": str(partner.get("name") or partner.get("display_name") or ""),
                    "email": str(partner.get("email") or ""),
                    "company": str(partner.get("parent_name") or ""),
                }
            )
        meeting["attendees"] = attendees
    return stripped


# ---------------------------------------------------------------------------
# Object chatter (Layer 2 — semantic context for the LLM)
# ---------------------------------------------------------------------------


# Strip every HTML tag the Odoo chatter ships with. ``mail.message.body``
# is rich HTML (``<p>``, ``<a>``, etc.) which adds noise to the LLM
# prompt budget for zero benefit; the LLM works better on plain text.
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_HTML_ENTITY_RE = re.compile(r"&(?:nbsp|amp|lt|gt|quot|#\d+);")
_WS_RUN_RE = re.compile(r"[ \t]+")
_NL_RUN_RE = re.compile(r"\n{3,}")


def _html_to_text(raw: str) -> str:
    if not raw:
        return ""
    text = _HTML_TAG_RE.sub("", str(raw))
    # Translate the handful of HTML entities that crop up in Odoo
    # bodies. Anything beyond ``&nbsp; &amp; &lt; &gt; &quot;`` and
    # numeric refs is rare enough to ignore — the LLM tolerates it.
    text = text.replace("&nbsp;", " ").replace("&amp;", "&").replace(
        "&lt;", "<"
    ).replace("&gt;", ">").replace("&quot;", '"')
    text = _HTML_ENTITY_RE.sub("", text)
    text = _WS_RUN_RE.sub(" ", text)
    text = _NL_RUN_RE.sub("\n\n", text)
    return text.strip()


# Fields we ask ``mail.message.read`` for. Author is a Many2one we
# unpack via the same shape as ``parent_id`` elsewhere.
_MESSAGE_FIELDS = (
    "id",
    "date",
    "author_id",
    "subject",
    "body",
    "message_type",
)


def fetch_object_chatter(
    config: OdooConfig,
    model: str,
    record_id: int,
    *,
    limit: int = 20,
) -> dict:
    """Pull a slim snapshot of an Odoo object's chatter.

    Returns
    -------
    ``{display_name, model, id, fetched_at, summary, messages}`` —
    or an empty payload (``messages=[]``) when the record doesn't
    exist or the user's API key lacks read access.

    ``summary`` is a 1-2 sentence flattening of the most recent
    messages, ready to drop straight into the Whisper / LLM prompt
    (≤ 2 000 chars) without further processing on the SwiftUI side.
    """
    clean_model = (model or "").strip()
    if not clean_model or not record_id:
        return {
            "display_name": "",
            "model": clean_model,
            "id": int(record_id or 0),
            "summary": "",
            "messages": [],
            "fetched_at": _format_odoo_datetime(datetime.now(timezone.utc)),
        }

    # 1. Resolve display_name + message_ids on the record itself.
    try:
        records = _json2_call(
            config,
            clean_model,
            "read",
            {
                "ids": [int(record_id)],
                "fields": ["id", "display_name", "message_ids"],
            },
        )
    except OdooError:
        records = []
    if not isinstance(records, list) or not records:
        return {
            "display_name": "",
            "model": clean_model,
            "id": int(record_id),
            "summary": "",
            "messages": [],
            "fetched_at": _format_odoo_datetime(datetime.now(timezone.utc)),
        }
    head = records[0]
    display_name = str(head.get("display_name") or "")
    message_ids = head.get("message_ids") or []
    if not isinstance(message_ids, list):
        message_ids = []

    if not message_ids:
        return {
            "display_name": display_name,
            "model": clean_model,
            "id": int(record_id),
            "summary": "",
            "messages": [],
            "fetched_at": _format_odoo_datetime(datetime.now(timezone.utc)),
        }

    # 2. Pull the N most-recent messages. Odoo gives us message_ids
    #    in chronological order so we slice the tail and reverse for
    #    a "newest first" reading shape.
    tail_ids = [int(m) for m in message_ids[-max(1, int(limit)):]]
    raw_messages: list[dict] = []
    try:
        records = _json2_call(
            config,
            "mail.message",
            "read",
            {
                "ids": tail_ids,
                "fields": list(_MESSAGE_FIELDS),
            },
        )
        if isinstance(records, list):
            raw_messages = [r for r in records if isinstance(r, dict)]
    except OdooError:
        raw_messages = []

    cleaned: list[dict] = []
    for msg in sorted(raw_messages, key=lambda r: str(r.get("date") or ""), reverse=True):
        author_name = ""
        author = msg.get("author_id")
        if isinstance(author, (list, tuple)) and len(author) >= 2:
            author_name = str(author[1])
        cleaned.append(
            {
                "id": int(msg.get("id") or 0),
                "date": str(msg.get("date") or ""),
                "author": author_name,
                "subject": str(msg.get("subject") or "") if msg.get("subject") else "",
                "body": _html_to_text(msg.get("body") or ""),
                "type": str(msg.get("message_type") or ""),
            }
        )

    summary = _build_chatter_summary(display_name, cleaned)

    return {
        "display_name": display_name,
        "model": clean_model,
        "id": int(record_id),
        "summary": summary,
        "messages": cleaned,
        "fetched_at": _format_odoo_datetime(datetime.now(timezone.utc)),
    }


def _build_chatter_summary(display_name: str, messages: list[dict]) -> str:
    """Compose a compact, prompt-ready paragraph the LLM can read.

    Format: ``<display_name> — <author1> (date) : <first 220 chars>``
    repeated for each message, capped at ~1 800 chars so a long
    chatter doesn't blow the prompt budget.
    """
    if not messages:
        return ""
    header = display_name.strip()
    chunks: list[str] = []
    budget = 1800
    used = len(header) + 4  # for ``", ".join`` + trailing dot
    for msg in messages:
        body = (msg.get("body") or "").strip()
        if not body:
            continue
        snippet = body[:220].rsplit(" ", 1)[0] if len(body) > 220 else body
        author = (msg.get("author") or "Anonyme").strip()
        date = (msg.get("date") or "").strip()
        pieces: list[str] = []
        if date:
            pieces.append(date)
        pieces.append(author)
        prefix = " — ".join(pieces)
        rendered = f"{prefix} : {snippet}"
        if used + len(rendered) + 2 > budget:
            break
        chunks.append(rendered)
        used += len(rendered) + 2
    if not chunks:
        return ""
    return (header + ". " if header else "") + " ".join(chunks)
