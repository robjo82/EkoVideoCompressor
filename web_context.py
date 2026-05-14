"""
Web context enrichment for transcription.

The local Whisper pass + post-processor + LLM corrections all
benefit hugely from a complete glossary. But the user can only
type so many terms before a meeting; proper nouns specific to the
participants' world (their employer, their products, their towns)
are routinely missed.

This module reads the first few minutes of a transcript, extracts
candidate entities (companies, people, product names), looks them
up on the open web via DuckDuckGo's HTML endpoint, and produces
a list of enriched glossary terms that get merged with whatever
the user typed. The next LLM correction pass receives the
enriched glossary, and the phonetic post-processor uses it for
substitutions.

Strict opt-in: defaults OFF, the network access is gated behind
a Settings checkbox. Privacy stance is documented in the hint
text — the user's team is the only consumer of the app, so we
trust them with that trade-off.

Implementation notes:
  * No third-party SDK — pure ``urllib.request`` so we don't
    introduce a runtime dependency.
  * Per-request timeout + a hard cap on total round-trips, so a
    flaky network can't stall the whole transcription pipeline.
  * Conservative entity extraction: we only consider capitalised
    multi-token phrases that look like proper nouns. The local
    Mistral can later be plugged in to extract more aggressively;
    for now we go regex-only so the path is robust without
    requiring mlx_lm to be available.
"""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass, field


__all__ = [
    "extract_entity_candidates",
    "duckduckgo_search",
    "enrich_glossary_via_web",
    "WebEnrichmentResult",
]


# --- Entity extraction ---------------------------------------------------


_STOPWORDS = {
    "Bonjour", "Voilà", "Allô", "Merci", "Allez", "Donc", "Alors", "OK",
    "Oui", "Non", "Bien", "Très", "Salut", "Bon", "Aujourd'hui",
    # English / generic
    "The", "It", "And", "But", "However",
}

# Multi-token capitalised sequence. We require at least one of:
#   - two consecutive capitalised tokens, OR
#   - one capitalised token of >=4 letters that doesn't sit at
#     sentence start (heuristic: preceded by a lowercase word).
_CAPS_TOKEN = r"(?:[A-ZÀ-Ý][a-zà-ÿ'’\-]+|[A-Z][A-Z0-9'\-]+)"
_MULTI_CAPS_RE = re.compile(rf"(?:{_CAPS_TOKEN}\s+){{1,3}}{_CAPS_TOKEN}")
_INLINE_PROPER_RE = re.compile(
    rf"(?<=[a-z])[ ’](?P<noun>{_CAPS_TOKEN})\b"
)


def _normalise_candidate(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().strip("'.,;:!?")


def extract_entity_candidates(text: str, *, max_candidates: int = 20) -> list[str]:
    """
    Pull proper-noun-looking phrases from a transcript snippet.

    The goal is to feed the web-search step a clean list of
    candidate names. We accept some recall loss for higher
    precision: single-token names get through only when they
    appear mid-sentence (lower false-positive rate).
    """
    if not text:
        return []

    seen: set[str] = set()
    out: list[str] = []

    for match in _MULTI_CAPS_RE.finditer(text):
        candidate = _normalise_candidate(match.group(0))
        if not candidate:
            continue
        # Filter sentence-start single-token captures.
        if candidate in _STOPWORDS:
            continue
        if len(candidate) < 3:
            continue
        key = candidate.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(candidate)
        if len(out) >= max_candidates:
            return out

    for match in _INLINE_PROPER_RE.finditer(text):
        candidate = _normalise_candidate(match.group("noun"))
        if not candidate or candidate in _STOPWORDS:
            continue
        if len(candidate) < 4:
            continue
        key = candidate.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(candidate)
        if len(out) >= max_candidates:
            break
    return out


# --- Web search ----------------------------------------------------------


# Strict total cost ceiling: even if the user has a long
# transcript, we don't want a runaway loop spending minutes on
# search latency.
DEFAULT_MAX_QUERIES = 12
DEFAULT_TIMEOUT_S = 6


@dataclass
class _SearchHit:
    title: str
    snippet: str


_DUCK_TITLE_RE = re.compile(
    r'<a[^>]*class="result__a"[^>]*>(?P<title>.*?)</a>', re.DOTALL
)
_DUCK_SNIPPET_RE = re.compile(
    r'<a[^>]*class="result__snippet"[^>]*>(?P<snippet>.*?)</a>', re.DOTALL
)
_TAG_RE = re.compile(r"<[^>]+>")


def _strip_html(blob: str) -> str:
    return re.sub(r"\s+", " ", _TAG_RE.sub("", blob)).strip()


def duckduckgo_search(
    query: str,
    *,
    timeout: float = DEFAULT_TIMEOUT_S,
    user_agent: str = "EkoVideoCompressor/1.0",
    opener=None,
) -> list[_SearchHit]:
    """
    Issue a single search against ``html.duckduckgo.com``. The
    response is server-rendered HTML (no JS), so we can parse it
    with a couple of regexes. Returns at most ~5 hits.

    The ``opener`` parameter is for tests — pass a stub callable
    that takes a Request and returns a context-manager file-like.
    """
    url = "https://html.duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query})
    request = urllib.request.Request(url, headers={"User-Agent": user_agent})
    opener_fn = opener or urllib.request.urlopen
    try:
        with opener_fn(request, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="ignore")
    except Exception:
        return []
    titles = [_strip_html(m.group("title")) for m in _DUCK_TITLE_RE.finditer(body)]
    snippets = [_strip_html(m.group("snippet")) for m in _DUCK_SNIPPET_RE.finditer(body)]
    hits: list[_SearchHit] = []
    for i in range(min(len(titles), 5)):
        title = titles[i] if i < len(titles) else ""
        snippet = snippets[i] if i < len(snippets) else ""
        if title or snippet:
            hits.append(_SearchHit(title=title, snippet=snippet))
    return hits


@dataclass
class WebEnrichmentResult:
    """One per accepted enriched glossary term."""

    candidate: str
    confirmed_term: str
    citation: str = ""              # title or URL fragment the answer came from
    snippet: str = ""

    def to_log(self) -> str:
        return f"{self.candidate!r} → {self.confirmed_term!r} ({self.citation})"


# A candidate is "confirmed" by a web search when at least one
# returned title/snippet contains the same token (case-insensitive,
# accent-folded). That's enough to say "this is a real proper noun
# worth biasing Whisper towards"; the actual semantic match would
# need an LLM step which we leave for the existing post-pass.

_ACCENT_FOLD = str.maketrans(
    "àáâäãåèéêëìíîïòóôöõùúûüýÿñç",
    "aaaaaaeeeeiiiiooooouuuuyync",
)


def _fold(s: str) -> str:
    return s.lower().translate(_ACCENT_FOLD)


def enrich_glossary_via_web(
    transcript_snippet: str,
    user_glossary: list[str],
    *,
    max_queries: int = DEFAULT_MAX_QUERIES,
    timeout: float = DEFAULT_TIMEOUT_S,
    search_fn=None,
) -> list[WebEnrichmentResult]:
    """
    Top-level glue: extract candidates from the snippet, search each
    against DuckDuckGo, keep only those confirmed by the search
    results. ``search_fn`` is injectable for tests.
    """
    if not transcript_snippet:
        return []
    candidates = extract_entity_candidates(transcript_snippet)
    if not candidates:
        return []

    known = {_fold(term) for term in user_glossary or []}
    confirmed: list[WebEnrichmentResult] = []
    queries = 0
    search = search_fn or duckduckgo_search

    for candidate in candidates:
        if queries >= max_queries:
            break
        folded = _fold(candidate)
        if folded in known:
            # The user already typed this — no need to confirm.
            continue
        try:
            hits = search(candidate, timeout=timeout)
        except Exception:
            continue
        queries += 1
        if not hits:
            continue
        # Confirmation = the candidate (or a 1-char variant) appears
        # in at least one title.
        confirmed_in_title = None
        for hit in hits:
            haystack = _fold(hit.title + " " + hit.snippet)
            if folded in haystack:
                confirmed_in_title = hit
                break
        if not confirmed_in_title:
            continue
        # Stay with the user's casing — they may have typed it
        # already in a structured form ("CVR Contrôles"); the
        # candidate we extracted is what Whisper *said*, which is
        # what we want to compare against in the post-processor.
        confirmed.append(
            WebEnrichmentResult(
                candidate=candidate,
                confirmed_term=candidate,
                citation=confirmed_in_title.title[:120],
                snippet=confirmed_in_title.snippet[:240],
            )
        )

    return confirmed
