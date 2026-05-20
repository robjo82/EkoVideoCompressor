"""
Phonetic glossary post-processor.

Whisper's `--initial-prompt` is a soft prior, not a hard constraint:
the model can — and routinely does — emit "MOLI" for "Mollie" or
"Sudokiz" for "Sudokies" even when the glossary names are in the
prompt. The fallback is a deterministic post-pass that walks the
transcript word by word, fuzzy/phonetic-matches each token against
the user-provided glossary, and substitutes when the match score
is strong enough.

Design choices:
  * No new heavy dependency. We implement a French-friendly Soundex/
    Metaphone variant in pure Python, plus normalized edit distance.
  * Multi-word glossary terms ("MGX Contrôles", "CVR Contrôles") are
    matched against rolling N-grams of the transcript.
  * Every substitution is logged so it can be surfaced in the review
    markdown file — never a silent rewrite.

The module is intentionally Qt-free so it lives alongside the
existing `transcription_utils.py` and is unit-testable without a
display server.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field


__all__ = [
    "parse_glossary_terms",
    "french_phonetic_key",
    "apply_glossary_to_segments",
    "apply_glossary_to_text",
    "GlossarySubstitution",
]


# ---------------------------------------------------------------------------
# Phonetic encoding
# ---------------------------------------------------------------------------

# A pragmatic French Soundex / Metaphone-lite. It's not the canonical
# Phonex algorithm but it's tuned for the failure modes we actually see
# (Mollie/MOLI/Molli, Sudokies/Sudokiz, Symphonat/Symphonate). The goal
# is "do these two strings sound the same when said aloud in French?",
# not "produce a paper-correct phonetic transcription".

_FR_DIGRAPHS = [
    # (regex, replacement)  — order matters, longest first
    (re.compile(r"sch", re.IGNORECASE), "S"),
    (re.compile(r"ch", re.IGNORECASE), "S"),
    (re.compile(r"ph", re.IGNORECASE), "F"),
    (re.compile(r"th", re.IGNORECASE), "T"),
    (re.compile(r"gh", re.IGNORECASE), "G"),
    (re.compile(r"qu", re.IGNORECASE), "K"),
    (re.compile(r"gn", re.IGNORECASE), "N"),  # cognac → kognak phonetically; "n" is close
    (re.compile(r"ll", re.IGNORECASE), "L"),  # collapse double letters
    (re.compile(r"mm", re.IGNORECASE), "M"),
    (re.compile(r"nn", re.IGNORECASE), "N"),
    (re.compile(r"tt", re.IGNORECASE), "T"),
    (re.compile(r"ss", re.IGNORECASE), "S"),
    (re.compile(r"rr", re.IGNORECASE), "R"),
    (re.compile(r"pp", re.IGNORECASE), "P"),
    (re.compile(r"ck", re.IGNORECASE), "K"),
    (re.compile(r"x", re.IGNORECASE), "KS"),
    (re.compile(r"z$", re.IGNORECASE), "S"),  # "Sudokiz" → "Sudokis"
    (re.compile(r"c(?=[eiy])", re.IGNORECASE), "S"),  # ci/ce → si/se
    (re.compile(r"g(?=[eiy])", re.IGNORECASE), "J"),  # gi/ge → ji/je
    (re.compile(r"c", re.IGNORECASE), "K"),
    (re.compile(r"q", re.IGNORECASE), "K"),
    (re.compile(r"w", re.IGNORECASE), "V"),
]

# Silent letter endings common in French.
_SILENT_END = re.compile(r"(s|t|d|x|z|p)$", re.IGNORECASE)

# Letters we throw away entirely — Whisper drops h's, vowel tone is
# unreliable, etc.
_DROP_LETTERS = set("h ")

# Vowel folding: we don't need to distinguish é/è/a/o etc. for
# fuzzy matching of proper nouns. Keep one vowel slot for each cluster.
_VOWEL_RE = re.compile(r"[aeiouy]+", re.IGNORECASE)


def french_phonetic_key(word: str) -> str:
    """
    Return a coarse phonetic key for a French word. Two words that
    sound similar in French share the same key (or one of distance 1).

    Examples (real-world):
        "Mollie", "MOLI", "Molli", "Moly"   → same key
        "Sudokies", "Sudokiz", "Sudokis"    → same key
        "Symphonat", "Symphonate"           → same key
        "Klarna", "Clarna"                  → same key
    """
    if not word:
        return ""

    # 1. Strip accents.
    s = unicodedata.normalize("NFKD", word)
    s = "".join(c for c in s if not unicodedata.combining(c))
    # 2. Lowercase, drop punctuation.
    s = re.sub(r"[^a-zA-Z]", "", s).lower()
    if not s:
        return ""

    # 3. Apply French digraph rules.
    for pat, repl in _FR_DIGRAPHS:
        s = pat.sub(repl, s)

    # 4. Drop letters we never care about.
    s = "".join(c for c in s if c not in _DROP_LETTERS)

    # 5. Collapse repeated consonants (after digraphs ran).
    s = re.sub(r"(.)\1+", r"\1", s)

    # 6. Strip silent endings iteratively. French mute "e" + silent
    #    consonants like "t/s/d/x/z/p" can stack: "Symphonate" needs
    #    to lose both the trailing 'e' and the now-trailing 't' to
    #    collide with "Symphonat".
    while True:
        new_s = _SILENT_END.sub("", s)
        # Also peel off a trailing mute "e" (always silent in French
        # at word-end), but keep at least 2 letters so we don't
        # erase tiny words like "le".
        if len(new_s) > 2 and new_s.endswith("e"):
            new_s = new_s[:-1]
        if new_s == s:
            break
        s = new_s

    # 7. Fold vowel clusters into a single 'A'.
    s = _VOWEL_RE.sub("A", s)

    return s.upper()


# ---------------------------------------------------------------------------
# Edit distance
# ---------------------------------------------------------------------------


def _levenshtein(a: str, b: str, limit: int = 4) -> int:
    """
    Standard Levenshtein with an early-exit when the running min row
    is above ``limit``. We use it on phonetic keys (4–10 chars) so a
    full O(n*m) is fine — the limit just lets us bail out fast on
    obviously-different candidates.
    """
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if abs(la - lb) > limit:
        return limit + 1
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i] + [0] * lb
        best_in_row = i
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            best_in_row = min(best_in_row, cur[j])
        if best_in_row > limit:
            return limit + 1
        prev = cur
    return prev[lb]


# ---------------------------------------------------------------------------
# Glossary parsing
# ---------------------------------------------------------------------------

_GLOSSARY_SPLIT = re.compile(r"[,\n;·•|/]+")
_INSTRUCTION_LINE = re.compile(
    r"^(vocabulaire|noms?\s*propres?|termes?|priorité|priority|expected|attendu)",
    re.IGNORECASE,
)


def parse_glossary_terms(raw: str) -> list[str]:
    """
    Split a free-form glossary into the individual terms we want to
    enforce. Tolerant of commas, newlines, bullets, instruction lines
    like "Vocabulaire à respecter:".

    Returns a stable-ordered, de-duplicated list, preserving each
    term's exact casing as the user typed it (since that's what we'll
    use as the replacement value).
    """
    if not raw:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for line in raw.splitlines():
        clean = line.strip().lstrip("-*•·").strip()
        if not clean:
            continue
        if _INSTRUCTION_LINE.match(clean):
            # "Vocabulaire à respecter, noms propres, clients:" — strip
            # everything up to the colon if there is one.
            if ":" in clean:
                clean = clean.split(":", 1)[1].strip()
            else:
                continue
        for part in _GLOSSARY_SPLIT.split(clean):
            term = part.strip().strip('"').strip("'").strip(".")
            if not term:
                continue
            # A single short letter ("M", "X") is almost certainly a
            # bullet we missed, not a glossary entry — skip.
            if len(term) < 2:
                continue
            key = term.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(term)
    return out


# ---------------------------------------------------------------------------
# Substitution
# ---------------------------------------------------------------------------


@dataclass
class GlossarySubstitution:
    """One applied substitution — surfaced in the review report."""

    original: str
    replacement: str
    timestamp_seconds: float | None = None
    confidence: float = 1.0
    method: str = ""  # "exact" | "phonetic" | "edit"
    context_before: str = ""
    context_after: str = ""


@dataclass
class _GlossaryEntry:
    term: str                    # canonical replacement (user's casing)
    tokens: list[str]            # whitespace-split tokens
    keys: list[str]              # phonetic key per token
    n: int                       # number of tokens (1, 2, 3...)
    canonical_lower: str = ""    # exact-match shortcut

    def __post_init__(self):
        self.canonical_lower = " ".join(self.tokens).lower()


def _build_entries(terms: list[str]) -> list[_GlossaryEntry]:
    entries: list[_GlossaryEntry] = []
    for term in terms:
        tokens = term.split()
        if not tokens:
            continue
        keys = [french_phonetic_key(t) for t in tokens]
        # Skip entries that phoneticize to nothing (e.g. all-numeric).
        if not any(keys):
            continue
        entries.append(_GlossaryEntry(term=term, tokens=tokens, keys=keys, n=len(tokens)))
    # Sort by descending token count so multi-word terms get a chance
    # to match before their individual words do.
    entries.sort(key=lambda e: -e.n)
    return entries


# Tokenization that PRESERVES punctuation around words so we can
# splice replacements back into the original text without losing
# spacing or commas.

_TOKEN_RE = re.compile(r"(\s+|[^\w\sÀ-ÖØ-öø-ÿ'’-]+|[\w'’-]+)", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return [m.group(0) for m in _TOKEN_RE.finditer(text)]


def _is_word(tok: str) -> bool:
    return bool(tok) and tok[0].isalnum()


def _word_indices(tokens: list[str]) -> list[int]:
    return [i for i, t in enumerate(tokens) if _is_word(t)]


def _phonetic_distance(key_a: str, key_b: str) -> int:
    if not key_a or not key_b:
        return 99
    return _levenshtein(key_a, key_b, limit=3)


def _surface_letters(value: str) -> str:
    return re.sub(r"[^a-z]", "", value.lower())


def _surface_close_enough(a: str, b: str, *, max_ratio: float = 0.34) -> bool:
    """
    Guardrail for the fuzzy phonetic tier. French phonetic keys collide
    surprisingly easily after vowel folding: "commande" and "Romain" are
    one phonetic edit apart, but the written words are clearly unrelated.

    We therefore require both:
      - same first 3 letters for medium/long words, or same first 2 for
        short words;
      - a small surface edit distance.

    This still allows real ASR variants like "MOLLE" -> "Mollie", while
    blocking glossary terms from invading ordinary French words.
    """
    surface_a = _surface_letters(a)
    surface_b = _surface_letters(b)
    if not surface_a or not surface_b:
        return False
    prefix_len = 2 if min(len(surface_a), len(surface_b)) <= 4 else 3
    if surface_a[:prefix_len] != surface_b[:prefix_len]:
        return False
    max_len = max(len(surface_a), len(surface_b))
    budget = max(1, int(max_len * max_ratio))
    return _levenshtein(surface_a, surface_b, limit=budget) <= budget


def _match_score(
    transcript_word: str,
    entry_surface: str,
    entry_key: str,
) -> tuple[bool, float, str]:
    """
    Returns (matched, confidence, method).

    Tiers (strongest → weakest):
      1. Exact surface (case-insensitive) match. Confidence 1.0.
      2. Same phonetic key + surface forms within a reasonable edit
         distance of each other. Confidence 0.95.
      3. Phonetic keys differ by exactly 1 character. Confidence 0.8.
    Otherwise no match.
    """
    surface = transcript_word.strip("'’-")
    if not surface or not entry_key:
        return (False, 0.0, "")

    # Tier 0: nothing to do if the surface is already the canonical
    # word — short-circuit so we don't even produce a "substitution".
    if surface.lower() == entry_surface.lower():
        return (True, 1.0, "exact")

    # PR R: per-token stoplist. Common French function words are
    # the dominant source of phonetic false positives we audited
    # on the CVR-control run (``par`` → ``Parce``, ``ou`` →
    # ``Audoo``, etc.). Refuse phonetic / edit tiers when the
    # transcript word is on the stoplist — those tokens have a
    # frequency floor (they appear every sentence) and Whisper's
    # priors keep them stable. The only safe match is tier 0
    # exact, handled above.
    normalised = surface.lower().replace("’", "'")
    if normalised in _MERGED_WINDOW_STOPLIST:
        return (False, 0.0, "")

    transcript_key = french_phonetic_key(surface)
    if not transcript_key:
        return (False, 0.0, "")

    # The minimum effective key length we trust. Below 3 the keys
    # collide too easily (any 2-letter word phoneticizes to a tiny
    # cluster), so we refuse anything weaker than tier-1.
    min_key_len = min(len(entry_key), len(transcript_key))
    if min_key_len < 3:
        return (False, 0.0, "")

    # Tier 1: same phonetic key. Confirm with a coarse surface check
    # that the words have at least *some* letters in common — this
    # guards against very different surfaces colliding via aggressive
    # folding.
    if transcript_key == entry_key:
        surface_a = _surface_letters(surface)
        surface_b = _surface_letters(entry_surface)
        budget = max(2, max(len(surface_a), len(surface_b)) // 2)
        if _levenshtein(surface_a, surface_b, limit=budget) <= budget:
            return (True, 0.95, "phonetic")
        return (False, 0.0, "")

    # Tier 2: keys 1 edit apart and entry key not super short.
    if (
        len(entry_key) >= 4
        and _phonetic_distance(transcript_key, entry_key) == 1
        and _surface_close_enough(surface, entry_surface)
    ):
        return (True, 0.8, "edit")

    return (False, 0.0, "")


def _replace_token_preserve_case(replacement: str, original_token: str) -> str:
    """
    Echo casing from the transcript token into the replacement if the
    replacement is single-word and the original used a distinctive
    case pattern. We keep it conservative: ALL_CAPS, Title_Case,
    lowercase are echoed; anything else uses the replacement as-is.
    """
    if " " in replacement or "-" in replacement:
        return replacement
    if not original_token:
        return replacement
    if original_token.isupper() and len(original_token) > 1:
        return replacement.upper()
    if original_token[0].isupper() and original_token[1:].islower():
        return replacement[:1].upper() + replacement[1:]
    return replacement


# PR R: tighter guards on merged-window matching.
#
# The PR P matcher caught a real failure mode (``pouvoir bien`` →
# ``Power BI``), but the loose Levenshtein-on-phonetic-keys logic
# generates catastrophic false positives on production audio. The
# canonical example from the CVR run was a flood of:
#   - ``par`` → ``Parce``
#   - ``vient ici`` → ``Vincent``
#   - ``Du coup, sur`` → ``Document``
#   - ``Odoo`` → ``Audoo``  (when ``Audoo`` was a hallucination PR D
#     had injected into the glossary)
#
# A common pattern: any 2-token window that contains a French
# function word (``par``, ``de``, ``une``, ``ici``) collapses to a
# glossary entry whose phonetic key happens to be short.
#
# This stoplist rejects the window outright when ANY of its tokens
# is a common French word. We trade off recall on phonetic
# corrections that legitimately involve function words (rare) for
# precision on the everyday "par/voir/vient" failure mode (common
# enough to make the whole feature net-negative in PR R's audit).
_MERGED_WINDOW_STOPLIST: frozenset[str] = frozenset(
    {
        # Pronouns / articles / determiners.
        "a", "à", "ai", "au", "aux", "ça", "ce", "ces", "cet", "cette",
        "c'est", "ci", "d", "de", "des", "du", "elle", "elles", "en",
        "es", "est", "et", "eu", "il", "ils", "j", "je", "l", "la", "le",
        "les", "leur", "leurs", "lui", "m", "ma", "me", "mes", "moi",
        "mon", "n", "ne", "nos", "notre", "nous", "on", "ou", "où",
        "par", "pas", "qu", "que", "qui", "quoi", "s", "sa", "se",
        "ses", "si", "son", "sont", "sur", "t", "ta", "te", "tes",
        "toi", "ton", "tout", "tous", "toute", "toutes", "tu", "un",
        "une", "vos", "votre", "vous", "y",
        # Prepositions + common adverbs. ``bien`` is intentionally
        # NOT included: ``pouvoir bien`` → ``Power BI`` is the
        # canonical merged-window win we want to preserve.
        "alors", "après", "aussi", "avec", "avant", "bon",
        "car", "chez", "comme", "comment", "dans", "déjà", "donc",
        "encore", "ensuite", "ici", "jamais", "juste", "là", "mais",
        "même", "moins", "non", "oui", "peu", "plus", "pour",
        "pourquoi", "puis", "quand", "rien", "sans", "selon", "seul",
        "seule", "seules", "seuls", "sinon", "sous", "très", "trop",
        "voici", "voilà",
        # High-frequency verbs (3rd person, imperative, common forms).
        "fait", "fais", "faut", "faire", "peut", "peux", "veut", "veux",
        "doit", "dois", "voir", "vient", "vais", "vas", "va", "as",
        "avais", "avait", "avez", "avons", "ai", "aurai", "aurez",
        "était", "étaient", "étant", "été", "étiez", "êtes", "êtes",
        "soit", "suis", "sait", "sais", "savoir",
    }
)


def _joined_phonetic_key(words: list[str]) -> str:
    """Concatenate per-word phonetic keys.

    Used by the merged-window matcher to compare a multi-token
    transcript window against any glossary entry (single or multi-
    token). Lets us catch ``pouvoir bien`` → ``Power BI`` and similar
    cases where Whisper hallucinated a multi-word surface for a
    single glossary token (the existing per-token matcher's surface
    guard rejects ``pouvoir`` vs ``power`` because their orthographic
    distance is 4 — but the joined phonetic key only differs by 1).
    """
    return "".join(french_phonetic_key(w) for w in words if w)


def _window_contains_stopword(words: list[str]) -> bool:
    """True if any token in the window is on
    ``_MERGED_WINDOW_STOPLIST``.

    The stoplist matches **after** lowercasing and stripping French
    apostrophes (``c'est`` ↔ ``c'est``). When any function word
    appears in the window, the matcher refuses to fire — those
    windows are real sentences, not glossary collisions.
    """
    for token in words:
        normalised = (token or "").strip().lower().replace("’", "'")
        if normalised in _MERGED_WINDOW_STOPLIST:
            return True
    return False


def _try_merged_window_match(
    tokens: list[str],
    word_idx: list[int],
    pos: int,
    entries: list[_GlossaryEntry],
    consumed: list[bool],
) -> tuple[_GlossaryEntry, int, int] | None:
    """Match a 2-3 token transcript window against any glossary
    entry's joined phonetic key.

    Returns ``(entry, start_idx, end_idx)`` of the best match, or
    ``None`` when no entry crosses the threshold. PR R: tightened
    guards so a 3-char key like ``Odoo``'s ADA can no longer sweep
    half the French language into "matches".
    """
    best: tuple[_GlossaryEntry, int, int, int] | None = None
    for window_size in (2, 3):
        if pos + window_size > len(word_idx):
            continue
        window_indices = word_idx[pos : pos + window_size]
        if any(consumed[i] for i in window_indices):
            continue
        window_words = [tokens[i] for i in window_indices]
        # PR R guard #1: reject windows that contain any common
        # French function word. This is the single biggest source
        # of false positives we audited (``par`` → ``Parce``,
        # ``vient ici`` → ``Vincent``, ``Du coup, sur`` → ``Document``).
        if _window_contains_stopword(window_words):
            continue
        window_key = _joined_phonetic_key(window_words)
        # PR R guard #2: raise the minimum joined-key length from 5
        # to 6. Below 6 the search space is too small — random French
        # bigrams collide with 4-5 char glossary keys.
        if len(window_key) < 6:
            continue
        for entry in entries:
            entry_key = "".join(entry.keys)
            # PR R guard #3: raise the minimum entry key length
            # from 4 to 5. ``Odoo`` (key ADA, 3 chars) no longer
            # qualifies for merged-window — the per-token matcher
            # already handles single-token glossary terms.
            if len(entry_key) < 5:
                continue
            # PR R guard #4: tighter Levenshtein scaling. Old
            # values ({≤5:1, ≤8:2, else:3}) were lenient enough
            # that "Du coup l'idée" (DK-PL-T) matched "Document"
            # (DKMT) with dist=2. New values: {≤5:0, ≤8:1, else:2}.
            if len(entry_key) <= 5:
                threshold = 0
            elif len(entry_key) <= 8:
                threshold = 1
            else:
                threshold = 2
            dist = _levenshtein(window_key, entry_key, limit=threshold + 1)
            if dist > threshold:
                continue
            # PR R guard #5: surface safety net upgraded from
            # 1-letter to 2-letter shared prefix. Stops
            # ``le sont`` (LS) from matching ``Laurent`` (LRN).
            window_surface = _surface_letters("".join(window_words))
            entry_surface = _surface_letters("".join(entry.tokens))
            if len(window_surface) < 2 or len(entry_surface) < 2:
                continue
            if window_surface[:2] != entry_surface[:2]:
                continue
            # Anti-collapse guard (unchanged from PR P): if the
            # window already contains the canonical glossary form
            # as one of its tokens, this is a real sentence
            # (``Mollie et Klarna``), not a Whisper hallucination —
            # never collapse it.
            entry_canonical_lower = entry.term.lower()
            if any(
                w.lower() == entry_canonical_lower for w in window_words
            ):
                continue
            score = -dist  # negative so smaller distance is "better"
            if best is None or score > best[3]:
                best = (entry, window_indices[0], window_indices[-1], score)
    if best is None:
        return None
    entry, start, end, _ = best
    return entry, start, end


def apply_glossary_to_text(
    text: str,
    terms: list[str],
    *,
    min_confidence: float = 0.8,
    merged_window_enabled: bool = False,
) -> tuple[str, list[GlossarySubstitution]]:
    """
    Pure-text version (no segments). Returns (new_text, substitutions).

    ``merged_window_enabled`` (PR R, default False) controls the
    multi-token phonetic matcher that catches cases like
    ``pouvoir bien`` → ``Power BI``. Off by default because the
    PR P feature, even with the PR R guards in place, can still
    fire on edge cases that aren't worth the false-positive risk
    in production. The single-token phonetic matcher (which fixes
    the bulk of Whisper's surface errors) stays active either way.
    Tests opt in explicitly to exercise the multi-token path.
    """
    if not text or not terms:
        return text, []

    entries = _build_entries(terms)
    if not entries:
        return text, []

    tokens = _tokenize(text)
    word_idx = _word_indices(tokens)
    substitutions: list[GlossarySubstitution] = []
    consumed = [False] * len(tokens)

    # Walk through every starting position; for each, try entries
    # largest-first so multi-word terms win over their single-word
    # alternatives.
    pos = 0
    while pos < len(word_idx):
        start = word_idx[pos]
        if consumed[start]:
            pos += 1
            continue

        best: tuple[_GlossaryEntry, float, str, int] | None = None
        for entry in entries:
            if pos + entry.n > len(word_idx):
                continue
            sub_idx = word_idx[pos : pos + entry.n]
            confidences: list[float] = []
            methods: list[str] = []
            for ti, ei in enumerate(sub_idx):
                ok, conf, method = _match_score(
                    tokens[ei], entry.tokens[ti], entry.keys[ti]
                )
                if not ok:
                    confidences = []
                    break
                confidences.append(conf)
                methods.append(method)
            if not confidences:
                continue
            score = min(confidences)
            if score < min_confidence:
                continue
            # Refuse to "correct" a token that's already the canonical
            # form — that's a no-op that would just inflate the report.
            joined = "".join(tokens[sub_idx[0] : sub_idx[-1] + 1]).strip().lower()
            if joined == entry.canonical_lower:
                continue
            primary_method = methods[0] if len(set(methods)) == 1 else "phonetic"
            if best is None or score > best[1] or entry.n > best[0].n:
                best = (entry, score, primary_method, sub_idx[-1])

        if best is None:
            pos += 1
            continue

        entry, score, method, end_idx = best
        # Build a small context window for the report.
        window_start = max(0, start - 6)
        window_end = min(len(tokens), end_idx + 7)
        before_ctx = "".join(tokens[window_start:start]).strip()
        after_ctx = "".join(tokens[end_idx + 1 : window_end]).strip()
        original_span = "".join(tokens[start : end_idx + 1])

        # Replace the first token in the span; clear the rest.
        first_replacement = _replace_token_preserve_case(entry.term, tokens[start])
        tokens[start] = first_replacement
        for k in range(start + 1, end_idx + 1):
            tokens[k] = ""
            consumed[k] = True
        consumed[start] = True

        substitutions.append(
            GlossarySubstitution(
                original=original_span,
                replacement=first_replacement,
                confidence=score,
                method=method,
                context_before=before_ctx,
                context_after=after_ctx,
            )
        )
        # Advance past the matched run.
        while pos < len(word_idx) and word_idx[pos] <= end_idx:
            pos += 1

    # Second pass: merged-window matching. Catches cases where
    # Whisper hallucinated a multi-token surface for a glossary
    # term (e.g. ``pouvoir bien`` → ``Power BI`` — the surface
    # distance per-token is too high for tier 1 but the joined
    # phonetic key is within 1 edit). Walks the still-unconsumed
    # word positions only.
    #
    # PR R: gated behind ``merged_window_enabled`` (default False).
    # The CVR-control audit showed this pass producing catastrophic
    # rewrites (``par`` → ``Parce``, ``vient ici`` → ``Vincent``,
    # ``Odoo`` → ``Audoo``) even with the original guards. The
    # tighter guards above + the opt-in default together stop the
    # bleeding.
    if merged_window_enabled:
        pos = 0
        while pos < len(word_idx):
            idx = word_idx[pos]
            if consumed[idx]:
                pos += 1
                continue
            match = _try_merged_window_match(
                tokens, word_idx, pos, entries, consumed
            )
            if match is None:
                pos += 1
                continue
            entry, start, end = match
            # Replace the first token; clear the rest in the window.
            window_start = max(0, start - 6)
            window_end = min(len(tokens), end + 7)
            before_ctx = "".join(tokens[window_start:start]).strip()
            after_ctx = "".join(tokens[end + 1 : window_end]).strip()
            original_span = "".join(tokens[start : end + 1])
            first_replacement = _replace_token_preserve_case(
                entry.term, tokens[start]
            )
            tokens[start] = first_replacement
            for k in range(start + 1, end + 1):
                tokens[k] = ""
                consumed[k] = True
            consumed[start] = True
            # PR R bump: confidence 0.78 → 0.85. Honest signal that
            # this path is harder to verify than per-token matching;
            # also lets callers filter merged-window subs with a
            # ``confidence >= 0.85`` predicate when needed.
            substitutions.append(
                GlossarySubstitution(
                    original=original_span,
                    replacement=first_replacement,
                    confidence=0.85,
                    method="merged_window",
                    context_before=before_ctx,
                    context_after=after_ctx,
                )
            )
            while pos < len(word_idx) and word_idx[pos] <= end:
                pos += 1

    return "".join(tokens), substitutions


def apply_glossary_to_segments(
    segments: list[dict],
    terms: list[str],
    *,
    min_confidence: float = 0.8,
    merged_window_enabled: bool = False,
) -> tuple[list[dict], list[GlossarySubstitution]]:
    """
    Walk Whisper segments, rewrite each `text` in place, and stamp
    each substitution with the timestamp of the segment it was found
    in. Returns the new segments list + the flat substitution list.
    """
    if not segments or not terms:
        return list(segments), []

    out: list[dict] = []
    all_subs: list[GlossarySubstitution] = []
    for seg in segments:
        new_seg = dict(seg)
        original_text = str(seg.get("text") or "")
        new_text, subs = apply_glossary_to_text(
            original_text,
            terms,
            min_confidence=min_confidence,
            merged_window_enabled=merged_window_enabled,
        )
        new_seg["text"] = new_text
        for sub in subs:
            sub.timestamp_seconds = float(seg.get("start") or 0.0)
        out.append(new_seg)
        all_subs.extend(subs)
    return out, all_subs
