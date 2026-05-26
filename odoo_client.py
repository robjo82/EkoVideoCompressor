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
import time
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
    "fetch_related_context_pack",
    "extract_odoo_glossary_candidates",
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
    started = time.monotonic()
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
        append_app_log(
            "odoo_partner_search_done "
            f"host={_safe_host(config.url)!r} query={tail_text(text, 80)!r} "
            f"limit={int(limit)} count=0 duration_ms={(time.monotonic() - started) * 1000:.0f} "
            "unexpected_payload=True"
        )
        return []
    rows = [_strip_partner_record(rec) for rec in records if isinstance(rec, dict)]
    append_app_log(
        "odoo_partner_search_done "
        f"host={_safe_host(config.url)!r} query={tail_text(text, 80)!r} "
        f"limit={int(limit)} count={len(rows)} duration_ms={(time.monotonic() - started) * 1000:.0f}"
    )
    return rows


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
    min_attendees: int = 2,
) -> list[dict]:
    """Return the ``calendar.event`` records that bracket ``near``.

    The domain looks for events whose [start, stop] window overlaps
    [near - window_hours, near + window_hours]. ``near`` defaults to
    "right now", which is what an `Enregistrer dans le moment`
    workflow expects.

    ``min_attendees`` filters out personal blockers / focus slots —
    we default to 2 because a meeting recording always implies at
    least two people. Pass 0 to disable the filter.

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

    min_required = max(0, int(min_attendees))
    stripped = [
        meeting
        for meeting in (_strip_meeting_record(r) for r in records if isinstance(r, dict))
        if len(meeting.get("partner_ids") or []) >= min_required
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


# ----------------------------------------------------------------------
# Recursive context pack
# ----------------------------------------------------------------------
#
# The single ``fetch_object_chatter`` call gives the LLM enough to
# correct a meeting transcript only when the primary record (the lead,
# the task) carries the whole story in its chatter. In practice the
# interesting context spans multiple records:
#
# * A ``crm.lead`` opportunity sits next to the ``sale.order``
#   quotation(s) sent for it — sometimes only the quote describes
#   the actual offering.
# * A ``project.task`` lives under a ``project.project`` whose
#   description and recent chatter set the broader frame.
# * Both can themselves point back to a ``res.partner`` whose name
#   and parent_name are useful biases for Whisper.
#
# ``fetch_related_context_pack`` does that traversal in one round,
# applies a token-budget-aware truncation so local LLMs (Mistral
# 7B, Llama 3 8B) don't get overrun, and returns both:
#   * a prompt-ready ``summary`` string for the LLM corrections pass
#   * a deduplicated ``terms`` list of proper-noun candidates to
#     splice into the glossary feeding Whisper's initial prompt.
#
# Failures along the way are silent — recursion just stops at the
# broken edge — so the LLM step always gets a best-effort blob
# instead of an exception.


# Fields we read from a sale.order to extract a meaningful one-liner
# in addition to the chatter. ``name`` is the quote ref, ``partner_id``
# the customer.
_SALE_ORDER_FIELDS = (
    "id",
    "name",
    "display_name",
    "state",
    "amount_total",
    "currency_id",
    "partner_id",
    "opportunity_id",
)

# Fields we pull off a project.task / project.project to enrich the
# context blob with a topical sentence beyond just the message log.
_TASK_FIELDS = (
    "id",
    "name",
    "display_name",
    "stage_id",
    "user_ids",
    "description",
    "project_id",
)

_PROJECT_FIELDS = (
    "id",
    "name",
    "display_name",
    "description",
    "partner_id",
)

# ``crm.lead`` is the model we recurse from. We pull the linked
# project_id (Enterprise CRM ↔ Project bridge) AND the partner so
# the resulting pack carries the customer name as a glossary boost.
_LEAD_FIELDS = (
    "id",
    "name",
    "display_name",
    "description",
    "partner_id",
    "partner_name",
    "user_id",
    "team_id",
    "stage_id",
)


def _safe_read(
    config: OdooConfig,
    model: str,
    ids: list[int],
    fields: tuple[str, ...],
) -> list[dict]:
    """Defensive ``read`` that returns ``[]`` on any error rather
    than propagating. Used everywhere in the recursion path so a
    permission glitch on one model never sinks the whole pack."""
    if not ids:
        return []
    try:
        out = _json2_call(
            config, model, "read", {"ids": [int(i) for i in ids], "fields": list(fields)}
        )
    except OdooError:
        return []
    return [r for r in (out or []) if isinstance(r, dict)]


def _safe_search_read(
    config: OdooConfig,
    model: str,
    domain: list,
    fields: tuple[str, ...],
    *,
    limit: int = 10,
    order: str = "id desc",
) -> list[dict]:
    if not domain:
        return []
    try:
        out = _json2_call(
            config,
            model,
            "search_read",
            {
                "domain": domain,
                "fields": list(fields),
                "limit": int(limit),
                "order": order,
            },
        )
    except OdooError:
        return []
    return [r for r in (out or []) if isinstance(r, dict)]


def _scalar_from_many2one(value: Any) -> tuple[int, str]:
    """Odoo many2one fields come back as ``[id, display_name]`` —
    flatten to ``(id, name)`` with ``(0, "")`` on garbage input."""
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            return int(value[0]), str(value[1] or "")
        except (TypeError, ValueError):
            return 0, ""
    return 0, ""


def fetch_related_context_pack(
    config: OdooConfig,
    model: str,
    record_id: int,
    *,
    chatter_limit: int = 8,
    max_total_chars: int = 4000,
) -> dict:
    """Build a recursive context pack rooted at ``(model, record_id)``.

    Returns ``{primary, related, summary, terms, fetched_at}``.

    * ``primary`` and ``related`` carry per-record dicts of the form
      ``{model, id, display_name, body, chatter}`` (the most
      informative single text snippet + an ordered chatter list).
    * ``summary`` is a prompt-ready compact blob respecting
      ``max_total_chars`` — local LLMs choke past 4-5 K extra chars
      with 4-8 K windows.
    * ``terms`` is the deduplicated list of proper-noun candidates
      the pipeline can splice into ``glossary_terms`` for Whisper.

    Traversal rules:

    * ``crm.lead`` → linked ``sale.order`` (via ``opportunity_id``)
      and any ``project.project`` referenced by ``project_ids``.
    * ``project.task`` → its parent ``project.project``.
    * ``calendar.event`` → ``opportunity_id`` (delegated to the
      ``crm.lead`` branch).
    * Anything else → no recursion; we still return the primary
      record's chatter so old callers don't regress.
    """
    clean_model = (model or "").strip()
    record_id = int(record_id or 0)
    fetched_at = _format_odoo_datetime(datetime.now(timezone.utc))
    if not clean_model or record_id <= 0:
        return {
            "primary": {},
            "related": [],
            "summary": "",
            "terms": [],
            "fetched_at": fetched_at,
        }

    primary = _fetch_one_record(config, clean_model, record_id, chatter_limit=chatter_limit)
    related: list[dict] = []
    if primary:
        related = _expand_related_records(
            config, primary, chatter_limit=chatter_limit
        )

    summary = _compose_pack_summary(
        primary, related, budget=max(1000, int(max_total_chars))
    )
    terms = extract_odoo_glossary_candidates(primary, related)
    return {
        "primary": primary,
        "related": related,
        "summary": summary,
        "terms": terms,
        "fetched_at": fetched_at,
    }


def _fetch_one_record(
    config: OdooConfig,
    model: str,
    record_id: int,
    *,
    chatter_limit: int,
) -> dict:
    """Fetch a single Odoo record + its chatter + a per-model body
    excerpt. Returns ``{}`` on failure rather than raising."""
    if model == "crm.lead":
        fields = _LEAD_FIELDS
    elif model == "project.task":
        fields = _TASK_FIELDS
    elif model == "project.project":
        fields = _PROJECT_FIELDS
    elif model == "sale.order":
        fields = _SALE_ORDER_FIELDS
    else:
        fields = ("id", "display_name")

    records = _safe_read(config, model, [record_id], fields)
    if not records:
        return {}
    head = records[0]
    display_name = str(head.get("display_name") or head.get("name") or "")

    chatter_payload: dict = {}
    try:
        chatter_payload = fetch_object_chatter(
            config, model, record_id, limit=chatter_limit
        )
    except OdooError:
        chatter_payload = {}

    return {
        "model": model,
        "id": record_id,
        "display_name": display_name,
        "raw": head,
        "body": _per_model_excerpt(model, head),
        "chatter": chatter_payload.get("messages") or [],
        "chatter_summary": str(chatter_payload.get("summary") or ""),
    }


def _per_model_excerpt(model: str, record: dict) -> str:
    """Pick the single most useful free-text snippet for each model.
    A short body excerpt (≤ 400 chars) beats the chatter for the
    primary topic since it's the static "what is this record".
    """
    if model in {"crm.lead", "project.task", "project.project"}:
        body = _html_to_text(record.get("description") or "")
        return body[:400].strip()
    if model == "sale.order":
        bits: list[str] = []
        if record.get("name"):
            bits.append(str(record["name"]))
        _, partner_name = _scalar_from_many2one(record.get("partner_id"))
        if partner_name:
            bits.append(f"client: {partner_name}")
        if record.get("amount_total"):
            bits.append(f"montant: {record['amount_total']}")
        return " — ".join(bits)
    return ""


def extract_company_name_from_pack(pack: dict | None) -> str:
    """PR AF: pull the client/partner name out of an Odoo context
    pack so the pipeline can prefix titles with "Company - Topic".

    Looks at the primary record's ``partner_id`` (for sale.order,
    crm.lead, calendar.event…) and falls back to ``display_name``
    when no partner is attached. Returns "" when nothing useful
    is available — the caller treats that as "no prefix".

    Kept here rather than in the pipeline because the Odoo
    schema knowledge (which model has which field) belongs with
    the Odoo client. The pipeline only needs a plain string.
    """
    if not pack or not isinstance(pack, dict):
        return ""
    primary = pack.get("primary") or {}
    raw = primary.get("raw") or {}

    # Primary: ``partner_id`` on the record. Sale orders, leads,
    # calendar events all use this many2one.
    partner_value = raw.get("partner_id")
    if partner_value:
        _, partner_name = _scalar_from_many2one(partner_value)
        if partner_name:
            cleaned = partner_name.strip()
            # Many partners in Odoo include the company in
            # parentheses or after a comma: "Jean Dupont, Caste"
            # or "Jean Dupont (Caste)". Prefer the company part
            # because the title needs the org, not the contact.
            for sep in (", ", " (", ":"):
                if sep in cleaned:
                    candidate = cleaned.split(sep, 1)[1].rstrip(")")
                    if candidate.strip():
                        return candidate.strip()
            return cleaned

    # Fallback: the display_name. Useful for project.project / etc.
    display = str(primary.get("display_name") or "").strip()
    if display:
        return display

    return ""


def _expand_related_records(
    config: OdooConfig,
    primary: dict,
    *,
    chatter_limit: int,
) -> list[dict]:
    """Discover and fetch records the primary points to."""
    related: list[dict] = []
    model = primary.get("model")
    head = primary.get("raw") or {}
    primary_id = int(primary.get("id") or 0)

    if model == "crm.lead":
        # Quotations linked back via opportunity_id. Cap at 5 to
        # avoid blowing the budget when a hot lead has 30 quotes.
        quotes = _safe_search_read(
            config,
            "sale.order",
            [("opportunity_id", "=", primary_id)],
            _SALE_ORDER_FIELDS,
            limit=5,
            order="date_order desc",
        )
        for q in quotes:
            qid = int(q.get("id") or 0)
            if qid <= 0:
                continue
            related.append(_fetch_one_record(
                config, "sale.order", qid, chatter_limit=chatter_limit
            ))

    elif model == "project.task":
        project_id, _ = _scalar_from_many2one(head.get("project_id"))
        if project_id:
            project = _fetch_one_record(
                config, "project.project", project_id, chatter_limit=chatter_limit
            )
            if project:
                related.append(project)

    elif model == "calendar.event":
        opportunity_id, _ = _scalar_from_many2one(head.get("opportunity_id"))
        if opportunity_id:
            lead = _fetch_one_record(
                config, "crm.lead", opportunity_id, chatter_limit=chatter_limit
            )
            if lead:
                related.append(lead)
                # And recurse one more level into the lead's quotes.
                related.extend(_expand_related_records(
                    config, lead, chatter_limit=chatter_limit
                ))

    return [r for r in related if r]


_SECTION_LABELS = {
    "crm.lead": "Opportunité",
    "project.task": "Tâche",
    "project.project": "Projet",
    "sale.order": "Devis",
    "calendar.event": "Réunion",
    "res.partner": "Contact",
}


def _section_label(model: str) -> str:
    return _SECTION_LABELS.get(model, model)


def _compose_pack_summary(
    primary: dict,
    related: list[dict],
    *,
    budget: int,
) -> str:
    """Glue the records into a single prompt-ready blob.

    Priority order — what we'd surface even on a tight budget:

    1. Primary record header + body excerpt.
    2. 1-2 most recent chatter messages on the primary.
    3. Each related record's header + body excerpt.
    4. 1 most recent chatter message per related record.

    Anything still over budget gets dropped from the bottom up.
    """
    if not primary:
        return ""
    chunks: list[str] = []
    used = 0

    def append(text: str) -> bool:
        nonlocal used
        if not text:
            return True
        if used + len(text) + 2 > budget:
            return False
        chunks.append(text)
        used += len(text) + 2
        return True

    # --- primary ---
    label = _section_label(primary.get("model", ""))
    name = (primary.get("display_name") or "").strip()
    body = (primary.get("body") or "").strip()
    if name:
        append(f"[{label}] {name}")
    if body:
        append(body)
    primary_messages = primary.get("chatter") or []
    for msg in primary_messages[:2]:
        snippet = _format_message(msg)
        if snippet and not append(snippet):
            break

    # --- related ---
    for rec in related:
        rlabel = _section_label(rec.get("model", ""))
        rname = (rec.get("display_name") or "").strip()
        rbody = (rec.get("body") or "").strip()
        if rname and not append(f"[{rlabel}] {rname}"):
            break
        if rbody and not append(rbody):
            break
        rmessages = rec.get("chatter") or []
        if rmessages:
            snippet = _format_message(rmessages[0])
            if snippet and not append(snippet):
                break

    return "\n".join(chunks).strip()


def _format_message(msg: dict) -> str:
    body = (msg.get("body") or "").strip()
    if not body:
        return ""
    body = body.replace("\n", " ").strip()
    snippet = body[:220].rsplit(" ", 1)[0] if len(body) > 220 else body
    author = (msg.get("author") or "Anonyme").strip()
    date = (msg.get("date") or "").strip()
    bits: list[str] = []
    if date:
        bits.append(date.split(" ")[0])  # drop the hour, keep the day
    bits.append(author)
    return f"  · {' — '.join(bits)} : {snippet}"


# Tokens that look like proper nouns but are uninformative in
# French and just bloat the glossary. Stripped before deduplication.
_GLOSSARY_STOPWORDS = {
    "Bonjour", "Merci", "Cordialement", "Salutations", "Madame",
    "Monsieur", "Mme", "Mr", "Anonyme", "Odoo", "Réunion",
}

# Match capitalised tokens (incl. é à ç etc.) + multi-token proper
# nouns ("Sophie Martin"). Restricted to 2-4 token sequences so
# arbitrary capitalised sentence openers don't sneak in.
_ENTITY_RE = re.compile(
    r"\b([A-ZÉÈÊÀÂÎÔÛÇ][\wÉÈÊÀÂÎÔÛÇéèêàâîôûç'\-]+(?:\s+[A-ZÉÈÊÀÂÎÔÛÇ][\wÉÈÊÀÂÎÔÛÇéèêàâîôûç'\-]+){0,3})\b"
)


def extract_odoo_glossary_candidates(
    primary: dict,
    related: list[dict],
    *,
    max_terms: int = 30,
) -> list[str]:
    """Pull proper-noun candidates from the pack to splice into the
    glossary that feeds Whisper's initial prompt + the LLM
    correction pass.

    Keeps things simple: regex over the combined free text, dedupe
    case-insensitively, drop French stopwords. Returns at most
    ``max_terms`` candidates so a chatty opportunity doesn't dilute
    the glossary with 200 sentence openers.

    Customer / partner / project / task display names are added
    explicitly even when the regex would have missed them (e.g. a
    single-token customer name like "PayFit") since we know they're
    structural, not just headline tokens.
    """
    text_parts: list[str] = []
    explicit_names: list[str] = []
    for record in [primary, *related]:
        if not record:
            continue
        name = (record.get("display_name") or "").strip()
        if name:
            explicit_names.append(name)
        head = record.get("raw") or {}
        partner_name = ""
        if isinstance(head.get("partner_id"), (list, tuple)) and len(head["partner_id"]) >= 2:
            partner_name = str(head["partner_id"][1] or "")
        if partner_name:
            explicit_names.append(partner_name)
        if record.get("body"):
            text_parts.append(str(record["body"]))
        for msg in record.get("chatter") or []:
            body = (msg.get("body") or "").strip()
            if body:
                text_parts.append(body)
            subject = (msg.get("subject") or "").strip()
            if subject:
                text_parts.append(subject)

    combined = " ".join(text_parts)
    candidates: list[str] = list(explicit_names)
    for match in _ENTITY_RE.finditer(combined):
        candidate = match.group(1).strip(" ,;:.!?'\"")
        if candidate and candidate not in candidates:
            candidates.append(candidate)

    seen: set[str] = set()
    unique: list[str] = []
    for raw in candidates:
        cleaned = raw.strip(" ,;:.!?'\"")
        if not cleaned or len(cleaned) < 2:
            continue
        if cleaned in _GLOSSARY_STOPWORDS:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(cleaned)
        if len(unique) >= max_terms:
            break
    return unique


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
