from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ffmpeg_utils import build_ffmpeg_cmd, default_out_path, is_audio_only_path
from glossary_postprocess import (
    GlossarySubstitution,
    apply_glossary_to_segments,
    parse_glossary_terms,
)
from llm_chunking import chunk_transcript_for_llm, dedupe_corrections
from llm_corrections import (
    AppliedCorrection,
    RejectedCorrection,
    apply_llm_corrections_to_text,
)
from speaker_recognition import (
    DEFAULT_MATCH_THRESHOLD,
    aggregate_embeddings,
    filter_usable_profiles,
    match_cluster_against_profiles,
    score_cluster_against_all_profiles,
)
from multipass import (
    WeakSegment,
    group_into_clip_ranges,
    identify_boundary_segments,
    identify_weak_segments,
    merge_repass_segments,
)
from transcription_utils import (
    AUDIO_PROFILE_STANDARD,
    AUDIO_PROFILE_TELEPHONY,
    assign_speakers_to_segments,
    build_audio_extract_cmd,
    build_diarization_cmd,
    detect_audio_profile,
    build_embedding_extract_cmd,
    build_llm_corrections_cmd,
    build_llm_title_cmd,
    build_mlx_whisper_cmd,
    canonical_multipass_model_id,
    default_transcript_path,
    extract_new_proper_nouns_from_segments,
    build_multimodal_audio_cmd,
    build_multimodal_recheck_prompt,
    format_seconds_for_clip,
    fuse_micro_turns,
    parse_multimodal_audio_response,
    _timestamp_text_to_seconds as timestamp_text_to_seconds,
    parse_diarization_output,
    parse_embedding_output,
    parse_llm_corrections_markdown,
    parse_llm_title_speakers,
    parse_whisper_json_segments,
    render_segments_plain,
    render_segments_with_speakers,
    structured_initial_prompt,
)
from vad_silero import build_vad_cmd, parse_vad_manifest, remap_segments_to_source

from .events import EventSink
from .logging import append_app_log, tail_text
from .models import (
    ArtifactEvent,
    ContextEvent,
    JobRequest,
    ProgressEvent,
    WarningEvent,
)
from .paths import managed_venv_python_path


# Tqdm progress lines that ``huggingface_hub`` writes to stderr while
# downloading model weights. They're not errors, but they end up in
# the same stderr buffer the engine surfaces to the UI when the
# subsequent command actually fails — making the dialog read like a
# wall of meaningless progress bars.
_TQDM_FETCHING_RE = re.compile(
    r"^Fetching\s+\d+\s+files?:.*$", re.MULTILINE
)


def _clean_subprocess_stderr(raw: str) -> str:
    """Strip tqdm-style progress lines + collapse runs of blank lines.

    Anything the model loader prints during a normal HF download is
    noise from the user's point of view; we surface the actual
    Python exception that follows.
    """
    if not raw:
        return raw
    cleaned = _TQDM_FETCHING_RE.sub("", raw)
    # Drop runs of blank lines that the regex leaves behind.
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _friendly_ffmpeg_error(raw: str, binary_path: str) -> str:
    """Translate cryptic ffmpeg failure modes into actionable French.

    Two failure modes show up in the wild:

    1. Bundled binary dynamically linked against Homebrew dylibs that
       don't exist on a user's machine — the original v0.13.0 dyld
       error.
    2. ``mlx_whisper`` itself fork-execs ``ffmpeg`` from ``$PATH`` to
       decode the input audio. When the user's PATH has no ffmpeg
       (we ship our own under Contents/Resources/bin) the venv-side
       Python raises ``FileNotFoundError: 'ffmpeg'`` deep inside
       ``mlx_whisper/audio.py``. The fix at the call site is to
       pass an enriched ``env``; the message here helps when the
       diagnosis path is still useful.
    """
    cleaned = _clean_subprocess_stderr(raw)
    if not cleaned:
        return f"ffmpeg ({binary_path}) a échoué sans message d'erreur."
    if "Library not loaded" in cleaned or "dyld" in cleaned.lower():
        return (
            "Le binaire ffmpeg fourni avec l'application est cassé : il "
            "dépend de bibliothèques absentes de votre machine. "
            "Réinstallez la dernière version d'EkoVideoCompressor pour "
            "corriger le problème. "
            f"(Détail technique : {cleaned.splitlines()[0]})"
        )
    if "No such file or directory: 'ffmpeg'" in cleaned or (
        "FileNotFoundError" in cleaned and "ffmpeg" in cleaned
    ):
        return (
            "Whisper a tenté d'appeler ffmpeg mais ne l'a pas trouvé. "
            "Cela ne devrait plus arriver depuis la version courante "
            "— signalez-le si vous le voyez. "
            "(Détail technique : FileNotFoundError sur ffmpeg dans mlx_whisper.)"
        )
    return cleaned


def subprocess_env_for_request(request: JobRequest) -> dict[str, str]:
    """Build a PATH that includes the bundle's ``bin/`` directory.

    Critical for ``mlx_whisper`` and friends: the Python package
    internally fork-execs ``ffmpeg`` to decode the input audio
    (``mlx_whisper/audio.py::load_audio``). If ``$PATH`` doesn't
    carry ffmpeg, we crash deep inside the venv with a
    ``FileNotFoundError: 'ffmpeg'`` even though we ship the binary
    alongside the engine. Prepending the bundled ``bin/`` directory
    makes the lookup succeed without polluting the user's
    environment.

    Same logic applies to pyannote (via torchaudio) and to any
    future tool that shells out to ffprobe.
    """
    env = os.environ.copy()
    candidates: list[str] = []
    for path in (
        request.compression_settings.ffmpeg_path,
        request.compression_settings.ffprobe_path,
    ):
        if path:
            parent = str(Path(path).parent)
            if parent and parent not in candidates:
                candidates.append(parent)
    if candidates:
        existing = env.get("PATH", "")
        env["PATH"] = (
            os.pathsep.join([*candidates, existing]) if existing else os.pathsep.join(candidates)
        )
    return env


def meeting_datetime_from_request(request: JobRequest) -> datetime | None:
    """Return the user-facing meeting date as an aware UTC datetime.

    SwiftUI sends ISO-8601 strings with a timezone. Headless callers
    may still pass a date without one; in that case we treat it as UTC
    rather than guessing a local timezone inside the engine process.
    """
    raw = (request.meeting_date or "").strip()
    if not raw:
        return None
    normalized = raw.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        append_app_log(f"engine_meeting_date_invalid value={raw!r}")
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def ffmpeg_creation_time_from_request(request: JobRequest) -> str | None:
    meeting_dt = meeting_datetime_from_request(request)
    if meeting_dt is None:
        return None
    return meeting_dt.isoformat().replace("+00:00", "Z")


def apply_meeting_date_to_artifact(request: JobRequest, path: str | Path | None) -> None:
    """Apply the meeting date to lightweight filesystem metadata.

    Text/Markdown files do not carry a portable internal creation
    timestamp, so we at least align their access/modification times.
    Media containers additionally receive FFmpeg's ``creation_time``
    metadata through ``build_ffmpeg_cmd``.
    """
    if not path:
        return
    meeting_dt = meeting_datetime_from_request(request)
    if meeting_dt is None:
        return
    artifact = Path(path)
    if not artifact.exists():
        return
    timestamp = meeting_dt.timestamp()
    try:
        os.utime(artifact, (timestamp, timestamp))
    except OSError as exc:
        append_app_log(
            "engine_meeting_date_utime_failed "
            f"path={str(artifact)!r} error={exc!r}"
        )


def _safe_stem(value: str) -> str:
    keep = []
    for char in value:
        if char.isalnum() or char in {" ", "-", "_", "."}:
            keep.append(char)
        else:
            keep.append(" ")
    cleaned = " ".join("".join(keep).split()).strip(" .-_")
    return (cleaned or "Transcription")[:80]


# Matches "X qui pourrait être Y ou Z" patterns the LLM emits on
# uncertain passages — captures the two proposed alternatives so we
# can reject tautological cases (Y == Z, or Y already in the source).
_DOUBT_ALTERNATIVES_RE = re.compile(
    r"qui pourrait\s+(?:\s*être)?\s*[\"«']?([^\"»'(),.;]+?)[\"»']?\s+ou\s+[\"«']?([^\"»'(),.;]+?)[\"»'.]?\s*(?:\(|$|\))",
    re.IGNORECASE | re.DOTALL,
)


def _normalize_word(value: str) -> str:
    """Lowercase + strip punctuation/whitespace for tautology comparison."""
    return re.sub(r"[\s.,;:!?'\"«»()]+", "", (value or "")).lower()


# Number words 1-24 — covers every realistic clock-time mention
# without taking on the full French numeric parsing surface.
_HOUR_NUMBER_WORDS = {
    "zéro": 0,
    "une": 1, "un": 1,
    "deux": 2,
    "trois": 3,
    "quatre": 4,
    "cinq": 5,
    "six": 6,
    "sept": 7,
    "huit": 8,
    "neuf": 9,
    "dix": 10,
    "onze": 11,
    "douze": 12,
    "treize": 13,
    "quatorze": 14,
    "quinze": 15,
    "seize": 16,
    "dix-sept": 17,
    "dix sept": 17,
    "dix-huit": 18,
    "dix huit": 18,
    "dix-neuf": 19,
    "dix neuf": 19,
    "vingt": 20,
    "vingt et un": 21, "vingt-et-un": 21, "vingt-et-une": 21,
    "vingt-deux": 22, "vingt deux": 22,
    "vingt-trois": 23, "vingt trois": 23,
    "vingt-quatre": 24, "vingt quatre": 24,
}

# Minute fragments that follow ``<N> heures``.
_HOUR_MINUTE_FRAGMENTS = {
    "et demie": 30,
    "et demi": 30,
    "et quart": 15,
    "moins le quart": -15,
    "et quinze": 15,
    "et trente": 30,
    "et quarante-cinq": 45,
    "et quarante cinq": 45,
    "pile": 0,
}

# Pattern: ``<number_word> heure(s) [<minute fragment or digits>]``
# Greedy on the optional suffix so ``neuf heures et demie`` matches
# ``9h30`` rather than ``9h`` + dangling tail.
_HOURS_PATTERN_RE = re.compile(
    r"(?<![\w])"
    r"(?P<hour>"
    + "|".join(
        sorted((re.escape(w) for w in _HOUR_NUMBER_WORDS), key=len, reverse=True)
    )
    + r")"
    r"\s+heure[s]?"
    r"(?:\s+(?P<suffix>"
    + "|".join(
        sorted((re.escape(f) for f in _HOUR_MINUTE_FRAGMENTS), key=len, reverse=True)
    )
    + r"|\d{1,2}))?"
    r"(?![\w])",
    re.IGNORECASE | re.UNICODE,
)


def normalize_spoken_clock_times(text: str) -> str:
    """Convert ``neuf heures`` / ``neuf heures et demie`` /
    ``neuf heures trente`` into ``9h`` / ``9h30``.

    Whisper transcribes spoken times verbatim ("neuf heures") rather
    than as digits. Readability suffers; downstream tools (calendar,
    Notion, search) can't index them. We convert the recognised
    patterns deterministically, leaving unrecognised number words
    intact so a "deux mille tonnes" doesn't accidentally become
    "2000h" or similar.

    Scope :
    - Hour 0-24 from a fixed table of number words.
    - Optional minute fragment (``et demie``, ``et quart``, ``moins
      le quart``, ``pile``, ``et trente``, ``et quarante-cinq``).
    - Or a bare digit suffix (``neuf heures 30``).

    Anything beyond hour-of-day is intentionally NOT touched — a
    general French → digits converter would be its own project.
    """
    def _convert(match: re.Match) -> str:
        hour_word = match.group("hour").lower()
        hour_value = _HOUR_NUMBER_WORDS.get(hour_word)
        if hour_value is None:
            return match.group(0)
        suffix = (match.group("suffix") or "").lower().strip()
        minutes = 0
        if suffix:
            if suffix in _HOUR_MINUTE_FRAGMENTS:
                offset = _HOUR_MINUTE_FRAGMENTS[suffix]
                if offset < 0:
                    # ``neuf heures moins le quart`` → 8h45
                    hour_value = (hour_value - 1) % 24
                    minutes = 60 + offset
                else:
                    minutes = offset
            elif suffix.isdigit():
                m = int(suffix)
                if 0 <= m < 60:
                    minutes = m
        if minutes:
            return f"{hour_value}h{minutes:02d}"
        return f"{hour_value}h"

    return _HOURS_PATTERN_RE.sub(_convert, text)


# Sequence of single letters separated by hyphens or whitespace,
# optionally with a final TLD-like suffix attached (``C A S T E.fr``).
# Min 3 letters so we don't catch "y a" (= "il y a" abbreviation).
_SPELLING_SEQUENCE_RE = re.compile(
    r"(?<![\w])"
    r"(?P<letters>(?:[A-Za-zÀ-ÖØ-öø-ÿ][\s\.\-]+){2,}[A-Za-zÀ-ÖØ-öø-ÿ])"
    r"(?P<tld>\.(?:fr|com|org|net|io|app|tech|eu))?"
    r"(?![\w])",
    re.IGNORECASE | re.UNICODE,
)

# Map common spoken-punctuation tokens to their symbolic form.
# Applied case-insensitively, surrounded by whole-word boundaries.
# Variants like ``arrobas`` / ``arobas`` cover the typical Whisper
# transcriptions of ``arobase`` said aloud (the actual Caste call
# came out as ``Arrobas``).
_SPOKEN_PUNCTUATION = {
    "arobase": "@",
    "arrobase": "@",
    "arrobas": "@",
    "arobas": "@",
    "arobaze": "@",
    "at": "@",
    "point": ".",
    "dot": ".",
    "tiret": "-",
    "trait d'union": "-",
    "underscore": "_",
    "souligné": "_",
    "barre oblique": "/",
    "slash": "/",
}

# Detector for "email-likely" context. If a span contains an existing
# ``@`` or a TLD like ``.fr``, we'll apply spoken-punctuation
# substitutions there. Outside such contexts, ``point`` should
# remain the French word ``point``.
_EMAIL_CONTEXT_RE = re.compile(
    r"[@]|\.(?:fr|com|org|net|io|app|tech|eu)\b",
    re.IGNORECASE,
)


def reconstruct_letter_spellings(text: str) -> str:
    """Collapse ``N O U V I A L E`` / ``n-o-u-v-i-a-l-e`` /
    ``c-a-s-t-e.fr`` style spellings into a single token.

    Whisper struggles with letter-by-letter spelling: when a speaker
    spells a word aloud, Whisper inserts hyphens or extra spaces
    between letters. The output is unreadable, particularly for
    email addresses (``C A S T E . F R`` should be ``caste.fr``).

    Heuristics:
    - ``<= 3`` letters → preserve uppercase (``API``, ``SQL``, ``RH``).
    - ``> 3`` letters → lowercase (``nouviale``, ``caste``).
    - A TLD suffix like ``.fr`` attached to the spelling is preserved
      and joined directly.
    """
    def _collapse(match: re.Match) -> str:
        letters = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]", match.group("letters"))
        joined = "".join(letters)
        if not joined:
            return match.group(0)
        tld = match.group("tld") or ""
        word = joined.upper() if len(joined) <= 3 else joined.lower()
        return word + tld.lower()

    return _SPELLING_SEQUENCE_RE.sub(_collapse, text)


def apply_spoken_punctuation_in_email_contexts(text: str) -> str:
    """Replace ``arobase`` / ``point`` / ``tiret`` etc. with their
    symbolic form when the surrounding text reads like an email or
    URL (carries an ``@`` already, or a TLD suffix).

    Conservative on purpose: ``point`` is a normal French word and
    we never want to replace it in ``point d'attention``. The
    context detector restricts substitutions to spans where an
    email/URL is being dictated.
    """
    # Replace at the line level rather than globally so the
    # email-context detection stays local.
    out_lines: list[str] = []
    for line in text.splitlines(keepends=True):
        # Only consider the rest of the line beyond the speaker tag
        # (the `[Speaker] (timestamp)` prefix isn't part of the
        # email context — strip it for the search but keep it in
        # the output).
        bracket_match = re.match(r"^(\s*\[[^\]]*\]\s*(?:\([^)]*\)\s*)?)?(.*)$", line)
        prefix = bracket_match.group(1) if bracket_match and bracket_match.group(1) else ""
        body = bracket_match.group(2) if bracket_match else line
        if not _EMAIL_CONTEXT_RE.search(body):
            out_lines.append(line)
            continue
        transformed = body
        for spoken, symbol in _SPOKEN_PUNCTUATION.items():
            transformed = re.sub(
                rf"(?<!\w){re.escape(spoken)}(?!\w)",
                symbol,
                transformed,
                flags=re.IGNORECASE,
            )
        # Tidy up "word @ word" → "word@word", "word . fr" → "word.fr"
        # while preserving inner sentences.
        transformed = re.sub(r"\s*@\s*", "@", transformed)
        transformed = re.sub(r"(?<=\w)\s*\.(?=\s*[a-zA-Z]{2,4}\b)", ".", transformed)
        transformed = re.sub(r"(?<=\w)\.\s+(?=[a-zA-Z]{2,4}\b)", ".", transformed)
        out_lines.append(prefix + transformed)
    return "".join(out_lines)


def reconstruct_spelled_text(text: str) -> str:
    """Pipeline-friendly wrapper: collapse letter spellings, apply
    spoken-punctuation substitutions in email/URL contexts, and
    convert spoken clock times to digits.

    All three passes are conservative and idempotent — calling
    twice on the same text returns the same result.
    """
    text = reconstruct_letter_spellings(text)
    text = apply_spoken_punctuation_in_email_contexts(text)
    text = normalize_spoken_clock_times(text)
    return text


def _apply_glossary_capitalization(text: str, glossary_terms: list[str]) -> str:
    """Replace every case-insensitive whole-word occurrence of a
    glossary term with its canonical form.

    Why: Whisper / LLM corrections sometimes leave glossary terms
    in inconsistent case ("Quadra" then "quadra", "Excel" then
    "excel"). The user added these to the glossary precisely
    because they wanted a canonical spelling — we honor that
    spelling end-to-end.

    Rules:
    - Word boundaries (``\\b``) so ``quadra`` matches in ``Quadra``
      and at the start of a sentence but NOT inside ``quadragénaire``.
    - Skips empty terms, terms with non-word characters that would
      defeat ``\\b``, and terms shorter than 2 chars (too risky).
    - Preserves the text inside ``[SPEAKER]`` brackets — speaker
      names shouldn't be rewritten by glossary substitutions.
    """
    if not text or not glossary_terms:
        return text
    # Pull bracketed speaker tags out, restore after substitution.
    placeholders: list[str] = []

    def _stash(match: re.Match) -> str:
        placeholders.append(match.group(0))
        return f"\x00SPK{len(placeholders) - 1}\x00"

    protected = re.sub(r"\[[^\]\n]*\]", _stash, text)
    for raw in glossary_terms:
        term = (raw or "").strip()
        if len(term) < 2:
            continue
        # Only apply when the term is a single word (or hyphenated
        # multi-token like "Power-BI") — multi-word glossary
        # entries can have legitimate orthographic variants.
        if not re.fullmatch(r"[\w\-'.&]+", term, re.UNICODE):
            continue
        pattern = re.compile(
            rf"(?<![\w]){re.escape(term)}(?![\w])",
            re.IGNORECASE,
        )
        protected = pattern.sub(term, protected)
    return re.sub(
        r"\x00SPK(\d+)\x00",
        lambda m: placeholders[int(m.group(1))],
        protected,
    )


def _filter_tautological_doubts(doubts: list[dict]) -> list[dict]:
    """Drop uncertain-passage entries whose ``reason`` is auto-
    referential — patterns where the LLM hallucinated alternatives
    that are identical to the original or to each other.

    Real-world examples observed on the Cozynergy job:
        - ``"retraiter" qui pourrait être "retraiter" ou "retraiter"``
        - ``"mélangeait" qui pourrait être "mélangeait" ou "mélangeait"``
        - ``"proposait" qui pourrait être "proposait" ou "proposait"``
    Five out of ten entries in that review file were of this shape.
    They erode user trust in the entire "Passages douteux" section.

    Strategy:
    1. Parse ``"qui pourrait être X ou Y"`` from the reason.
    2. Drop if X == Y (case-insensitive after punctuation strip).
    3. Drop if X appears verbatim in the original ``text``.
    4. Drop when the reason is so short it carries no signal (< 8
       chars, almost always a model fragment).
    """
    out: list[dict] = []
    for entry in doubts:
        reason = (entry.get("reason") or "").strip()
        if len(reason) < 8:
            continue
        match = _DOUBT_ALTERNATIVES_RE.search(reason)
        if match:
            alt1_norm = _normalize_word(match.group(1))
            alt2_norm = _normalize_word(match.group(2))
            if alt1_norm and alt2_norm and alt1_norm == alt2_norm:
                # "X ou X" — pure tautology.
                continue
            text_norm = _normalize_word(entry.get("text") or "")
            if alt1_norm and alt1_norm in text_norm and alt2_norm in text_norm:
                # Both alternatives are literally in the source — the
                # LLM is "uncertain" about what's already there.
                continue
        out.append(entry)
    return out


def job_workspace_dir(request: JobRequest) -> Path:
    if request.workspace_dir:
        return Path(request.workspace_dir)
    root = Path(request.output_dir).expanduser()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return root / f"{stamp} - {_safe_stem(Path(request.source_path).stem)}"


def snapshot_existing_artifacts(
    workspace: Path,
    job: dict[str, Any],
    sink: EventSink,
) -> dict[str, Any]:
    """Move the user-facing outputs of the previous run into a dated
    ``versions/`` subfolder so the rerun about to start can't clobber
    them.

    Returns a metadata dict suitable for
    ``DatabaseManager.prepend_job_version`` — empty when there was
    nothing to snapshot (fresh job, or workspace already wiped).

    What we snapshot:

    * ``compressed_path``
    * ``transcript_path``
    * ``enhanced_transcript_path``
    * ``review_path``

    We deliberately skip intermediates (``audio.wav``, ``whisper.json``,
    ``speaker_samples/``) because (a) they're regenerated anyway and
    (b) keeping them inflates the workspace footprint for negligible
    user value. The compressed file is included because the
    compression preset might change between runs and the user could
    reasonably want to A/B compare.

    Files are MOVED rather than copied — disk is cheap but not free
    for hour-long meetings, and the snapshot is meant as a safety
    net, not a permanent archive.
    """
    candidates = {
        "compressed_path": (job.get("compressed_path") or "").strip(),
        "transcript_path": (job.get("transcript_path") or "").strip(),
        "enhanced_transcript_path": (job.get("enhanced_transcript_path") or "").strip(),
        "review_path": (job.get("review_path") or "").strip(),
    }
    existing = {
        column: path
        for column, path in candidates.items()
        if path and Path(path).exists()
    }
    if not existing:
        return {}

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    versions_dir = workspace / "versions" / timestamp
    versions_dir.mkdir(parents=True, exist_ok=True)
    moved: dict[str, str] = {}
    for column, source_path in existing.items():
        src = Path(source_path)
        dest = versions_dir / src.name
        # Disambiguate when two artefacts share a filename — unlikely
        # given the engine's naming, but cheaper than a debugging
        # session if the unlikely happens.
        if dest.exists():
            counter = 1
            while dest.exists():
                dest = versions_dir / f"{src.stem}_{counter}{src.suffix}"
                counter += 1
        try:
            shutil.move(str(src), str(dest))
        except OSError as exc:
            append_app_log(
                f"engine_snapshot_move_failed src={source_path!r} "
                f"dest={str(dest)!r} error={exc!r}"
            )
            # Best-effort: copy when move fails (e.g. cross-device).
            try:
                shutil.copy2(src, dest)
                src.unlink()
            except OSError:
                continue
        moved[column] = str(dest)
        sink(
            ArtifactEvent(
                "previous_version",
                str(dest),
                model=column,
            )
        )

    if not moved:
        return {}

    summary = {
        "label": timestamp,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        **moved,
    }
    append_app_log(
        "engine_snapshot_previous_run "
        f"workspace={str(workspace)!r} files={len(moved)} label={timestamp!r}"
    )
    return summary


def prepare_job_workspace(request: JobRequest, sink: EventSink) -> tuple[Path, Path]:
    workspace = job_workspace_dir(request)
    workspace.mkdir(parents=True, exist_ok=True)
    source = Path(request.source_path).expanduser()
    copied_source = workspace / source.name
    append_app_log(
        "engine_prepare_workspace "
        f"workspace={str(workspace)!r} source={str(source)!r} "
        f"copied_source={str(copied_source)!r} delete_source={request.delete_source_after_copy!r}"
    )
    if (
        source.exists()
        and source.resolve() != copied_source.resolve()
        and not copied_source.exists()
    ):
        shutil.copy2(source, copied_source)
        append_app_log(
            "engine_prepare_workspace copied "
            f"source={str(source)!r} copied_source={str(copied_source)!r}"
        )
        sink(ArtifactEvent("source", str(copied_source)))
        if request.delete_source_after_copy:
            try:
                source.unlink()
                append_app_log(
                    "engine_prepare_workspace deleted_original "
                    f"source={str(source)!r}"
                )
                sink(
                    ProgressEvent(
                        "source_cleanup",
                        100,
                        "Fichier source supprimé de son emplacement d'origine",
                    )
                )
            except OSError as exc:
                append_app_log(
                    "engine_prepare_workspace delete_original_failed "
                    f"source={str(source)!r} error={exc!r}"
                )
                sink(
                    WarningEvent(
                        f"Le fichier source a été copié, mais n'a pas pu être supprimé : {exc}",
                        code="source_delete_failed",
                    )
                )
    return workspace, copied_source if copied_source.exists() else source


@dataclass(slots=True)
class StepResult:
    name: str
    ok: bool = True
    artifact_path: str = ""
    model: str = ""
    duration_seconds: float = 0.0
    metrics: dict[str, Any] = field(default_factory=dict)
    error: str = ""


class TranscriptionPipeline:
    """Headless transcription pipeline with the full quality stack.

    The first version of this module only ran Whisper once and wrote
    the transcript. None of the quality phases the legacy worker
    spent a PR building actually shipped through the new SwiftUI app
    — VAD pre-filter, phonetic glossary post-processor, confidence-
    triaged second pass, diarisation, LLM post-pass — all were dead
    code from the engine's point of view. This rewrite plumbs them
    in, in order, so users get the same quality the legacy PySide
    UI offered.

    Each phase is gated by a flag on ``TranscriptionSettings`` so a
    fast preset can skip everything below the basic Whisper pass.
    """

    def __init__(self, request: JobRequest, sink: EventSink):
        self.request = request
        self.sink = sink
        self._resolve_managed_python()
        # Accumulators surfaced in the review markdown.
        self._glossary_subs: list[GlossarySubstitution] = []
        self._repass_replaced: int = 0
        self._vad_manifest: list[dict] = []
        self._llm_payload: dict[str, Any] = {}
        # PR AB: dropped decoder loops surfaced from the cleaner so
        # the review markdown can render a "⚠️ Zones perdues" section.
        # Without this the user has no written signal that 60-80 %
        # of a 2 h meeting was lost to Whisper hallucinations.
        self._dropped_loops: list[Any] = []
        # PR AD: loops that the recovery step (re-Whisper with
        # ``--clip-timestamps`` + ``--condition-on-previous-text
        # False``) successfully clawed back. Used by the review
        # markdown to render a "✓ Récupérées" section alongside the
        # still-lost ones.
        self._recovered_loops: list[Any] = []
        # PR AB: total audio duration (Whisper-visible end timestamp,
        # in seconds). Lets the review markdown compute coverage =
        # kept / total. Empty when the Whisper step never ran or
        # failed before populating segments.
        self._audio_seconds: float = 0.0
        # LLM corrections actually applied vs rejected, captured so
        # the review report can show "12 corrections appliquées, 3
        # refusées (raison: not_found / too_distant / low_confidence)".
        self._llm_corrections_applied: list[AppliedCorrection] = []
        self._llm_corrections_rejected: list[RejectedCorrection] = []
        # End-of-run artefacts the runner picks up to persist into
        # the library DB. Exposed as attributes (not StepResult
        # payloads) so we can carry richer data without bloating the
        # public step contract:
        #
        # ``final_segments`` — the segment list as written to disk,
        #   timestamped on the source timeline, with speaker labels
        #   when diarisation ran. The runner inserts them into
        #   ``transcription_segments`` so the rename-speakers sheet
        #   can iterate over the speakers and cut audio samples.
        #
        # ``final_speaker_map`` — every speaker label discovered in
        #   ``final_segments``, mapped to the friendly name when the
        #   LLM identified one (otherwise an empty string). Persisted
        #   to ``speaker_map_json``; the sheet renders one row per
        #   key so the user can fill in the names manually even when
        #   the LLM stayed silent.
        #
        # ``final_technical_terms`` — same idea, persisted to
        #   ``technical_terms_json``.
        self.final_segments: list[dict] = []
        self.final_speaker_map: dict[str, str] = {}
        self.final_technical_terms: list[str] = []
        self.final_title: str = ""
        # ``mlx_vlm`` probe is cached because importing it can take
        # ~1 s and the audio recheck step runs N subprocesses in
        # sequence — probing once per pipeline is plenty. ``None``
        # means "not probed yet", True / False are the resolved
        # values.
        self._mlx_vlm_available: bool | None = None

    def _resolve_managed_python(self) -> None:
        settings = self.request.transcription_settings
        if settings.venv_python_path and Path(settings.venv_python_path).exists():
            return
        if (settings.quality_preset or "custom").strip().lower() not in {
            "balanced",
            "max",
        }:
            return
        candidate = managed_venv_python_path()
        if candidate.exists():
            settings.venv_python_path = str(candidate)

    def _subprocess_env(self) -> dict[str, str]:
        return subprocess_env_for_request(self.request)

    def _expected_speaker_names(self) -> list[str]:
        """Names the caller declared up front (in ``speaker_overrides``).

        Whisper benefits from seeing them in the ``initial-prompt`` so
        the decoder produces a consistent orthography every time the
        speaker introduces themselves.

        ``speaker_overrides`` maps SPEAKER_NN → real name, so we want
        the values. Empty/blank entries are filtered out — they mean
        "the user hasn't decided yet", not "force an empty name".
        """
        overrides = self.request.speaker_overrides or {}
        return [name for name in overrides.values() if (name or "").strip()]

    def _ensure_odoo_pack(self) -> dict:
        """Lazily fetch (and cache) the recursive Odoo context pack.

        Behavioural difference vs the old single-record chatter
        fetch this replaced:

        * We recurse one level into the linked record's neighbours
          (opportunity → quotations, task → project) so the LLM
          sees the surrounding business context, not just the
          calendar invite chatter.
        * The pack is fetched **once per pipeline run** and the
          extracted glossary terms are spliced into
          ``request.glossary_terms`` in place. That mutation is
          intentional — it lets every downstream consumer (Whisper
          initial prompt, LLM corrections, per-speaker pass) read
          the enriched list without each one having to merge
          again.
        * Failures stay silent: a broken Odoo just returns
          ``{summary: "", terms: []}`` and the LLM falls back on
          the user-typed glossary.

        Returns the pack dict (possibly empty) so callers can read
        ``summary`` for the LLM corrections prompt.
        """
        if hasattr(self, "_odoo_pack_cache"):
            return self._odoo_pack_cache  # type: ignore[has-type]

        ref = self.request.odoo_context_ref
        if not ref or not ref.is_actionable():
            self._odoo_pack_cache: dict = {}
            return self._odoo_pack_cache

        # Lazy imports: keeps a missing-credentials run from paying
        # the import cost on the hot path.
        from odoo_client import (
            OdooConfig,
            OdooError,
            fetch_related_context_pack,
        )

        self.sink(
            ProgressEvent(
                "odoo_context", 0,
                f"Récupération du contexte Odoo ({ref.model})",
            )
        )
        config = OdooConfig(
            url=ref.url,
            database=ref.database,
            login=ref.login,
            api_key=ref.api_key,
        )
        try:
            pack = fetch_related_context_pack(
                config, ref.model, ref.record_id
            )
        except OdooError as exc:
            self.sink(
                WarningEvent(
                    f"Contexte Odoo indisponible : {exc}",
                    code="odoo_context_failed",
                )
            )
            append_app_log(
                f"engine_odoo_context_failed model={ref.model!r} "
                f"id={ref.record_id} error={exc!r}"
            )
            self._odoo_pack_cache = {}
            return self._odoo_pack_cache

        related_count = len(pack.get("related") or [])
        term_count = len(pack.get("terms") or [])
        if not pack.get("summary"):
            self.sink(
                ProgressEvent(
                    "odoo_context", 100,
                    "Contexte Odoo: aucun message à exploiter",
                )
            )
        else:
            self.sink(
                ProgressEvent(
                    "odoo_context", 100,
                    f"Contexte Odoo récupéré (+{related_count} record(s) lié(s), "
                    f"{term_count} terme(s) glossaire)",
                )
            )

        # Splice the recovered entity names into the glossary the
        # initial prompt feeds Whisper. Done case-insensitively to
        # avoid stacking ``Sophie`` and ``sophie`` and surprising
        # the user with double mentions in the rename sheet.
        new_terms = [str(t) for t in (pack.get("terms") or []) if str(t).strip()]
        if new_terms:
            existing_keys = {
                str(t).strip().lower()
                for t in self.request.glossary_terms
                if str(t).strip()
            }
            appended = [
                t for t in new_terms if t.strip().lower() not in existing_keys
            ]
            if appended:
                self.request.glossary_terms = [
                    *self.request.glossary_terms, *appended
                ]
                append_app_log(
                    "engine_odoo_glossary_boost "
                    f"added={len(appended)} from_pack={term_count}"
                )

        self._odoo_pack_cache = pack
        return pack

    def _fetch_odoo_context_blob(self) -> str:
        """Backwards-compatible wrapper. Returns the recursive pack's
        ``summary`` so the LLM corrections path keeps its call shape
        even after the recursion rewrite."""
        return str(self._ensure_odoo_pack().get("summary") or "")

    def _hot_enrich_glossary(self, segments: list[dict]) -> None:
        """
        Fold proper-noun candidates discovered in the first Whisper
        pass into ``self.request.glossary_terms`` so the multipass,
        boundary multipass, per-speaker pass and LLM correction step
        all see them in their ``--initial-prompt``.

        Bounded to 20 new terms by default to keep the prompt under
        ``INITIAL_PROMPT_MAX_CHARS`` (the prompt builder truncates
        but truncation is silent — better to cap upstream).
        """
        try:
            existing = [
                *(self.request.glossary_terms or []),
                *(self.request.technical_terms or []),
                *(self._expected_speaker_names() or []),
            ]
            new_terms = extract_new_proper_nouns_from_segments(
                segments,
                existing_terms=existing,
                min_occurrences=2,
                max_terms=20,
            )
        except Exception as exc:  # pragma: no cover — guard rail
            append_app_log(f"engine_hot_prompt_enrichment_failed err={exc!r}")
            return

        if not new_terms:
            return

        # Mutate the request in place so the downstream passes (which
        # rebuild their initial_prompt from these lists every call)
        # see the enriched vocabulary. Append rather than prepend so
        # the user-provided / Odoo-derived glossary keeps priority in
        # the prompt slots that fit before INITIAL_PROMPT_MAX_CHARS.
        self.request.glossary_terms = [
            *self.request.glossary_terms,
            *new_terms,
        ]
        append_app_log(
            "engine_hot_prompt_enrichment added="
            + ",".join(new_terms[:10])
            + (f" (+{len(new_terms) - 10} more)" if len(new_terms) > 10 else "")
        )
        # Surface the discovery to the UI as a ContextEvent so the
        # rename sheet / glossary chips reflect the boosted vocabulary
        # without the user having to inspect the JSONL stream.
        try:
            self.sink(
                ContextEvent(
                    technical_terms=list(new_terms),
                )
            )
        except Exception:
            # ContextEvent emission is decorative — never let it
            # break the pipeline.
            pass

    # ------------------------------------------------------------------
    # PR AF — title company prefix fallback
    # ------------------------------------------------------------------

    def _resolve_company_name_for_title(self) -> str:
        """Find the client/partner company name for the title prefix.

        Strategy (first-match-wins, per user's PR AF answer):
          1. Odoo context pack (``primary.raw.partner_id`` or
             ``primary.display_name``). Most reliable when the user
             paired a CRM lead / sale order / calendar event in Run
             Setup.
          2. ``odoo_meeting_metadata.partners`` — the calendar
             invite's partner list, captured at Run Setup time.
          3. Speaker overrides values — sometimes the user typed
             "Nom (Société)" or similar.

        Returns "" when nothing useful — caller leaves the LLM
        title alone.
        """
        # Source 1: Odoo context pack (lazily fetched, already
        # cached if the LLM step has run).
        try:
            from odoo_client import extract_company_name_from_pack

            pack = self._ensure_odoo_pack()
            name = extract_company_name_from_pack(pack)
            if name:
                return name
        except Exception:  # pragma: no cover — defensive
            pass

        # Source 2: meeting metadata snapshot (calendar invite).
        meta = self.request.odoo_meeting_metadata or {}
        partners = meta.get("partners") if isinstance(meta, dict) else None
        if isinstance(partners, list):
            for partner in partners:
                if not isinstance(partner, dict):
                    continue
                name = str(partner.get("name") or "").strip()
                if name and "ekonum" not in name.lower():
                    # First non-Ekonum partner is the client.
                    return name

        # Source 3: speaker overrides — sometimes ``"Nom (Société)"``.
        for raw_name in (self.request.speaker_overrides or {}).values():
            name = str(raw_name or "").strip()
            if "(" in name and ")" in name:
                inside = name[name.find("(") + 1 : name.find(")")]
                if inside.strip() and "ekonum" not in inside.lower():
                    return inside.strip()

        return ""

    def _apply_title_company_prefix(self, title: str) -> str:
        """Ensure the title is rendered as ``"Société - Sujet"`` when
        we can resolve a company name. Idempotent : if the title
        already starts with the resolved company (case-insensitive)
        or the LLM already produced a ``" - "`` separator that
        looks like a company prefix, do nothing.
        """
        company = self._resolve_company_name_for_title()
        if not company:
            return title
        normalised_title = title.strip()
        if not normalised_title:
            return title
        # Already prefixed? Check both exact and case-insensitive
        # so "caste - sujet" or "CASTE - Sujet" both count.
        company_lower = company.lower()
        if normalised_title.lower().startswith(company_lower + " -"):
            return normalised_title
        if normalised_title.lower().startswith(company_lower + " :"):
            return normalised_title
        # The LLM may have produced its OWN "Company - Topic" with
        # a different (wrong) company. We don't second-guess — if
        # any " - " separator is present in the first ~30 chars,
        # leave the LLM's choice alone rather than risk doubling up
        # an incorrect prefix.
        first_30 = normalised_title[:30]
        if " - " in first_30:
            return normalised_title
        return f"{company} - {normalised_title}"

    def _meeting_context(self) -> str:
        """Short topic line surfaced to Whisper as semantic prior.

        We currently derive it from ``JobRequest.profile`` ("Réunion
        équilibrée", etc.). Empty / generic profiles fall through to
        the default "Réunion professionnelle en français." that the
        prompt builder emits when nothing was provided.
        """
        profile = (self.request.profile or "").strip()
        generic = {"", "default", "reunion equilibree", "Reunion equilibree"}
        if profile in generic or profile.lower() in {p.lower() for p in generic}:
            return ""
        return profile

    # ------------------------------------------------------------------
    # Public orchestrator
    # ------------------------------------------------------------------

    def run(self, source_path: str) -> list[StepResult]:
        results: list[StepResult] = []
        workspace = job_workspace_dir(self.request)
        workspace.mkdir(parents=True, exist_ok=True)
        wav_path = workspace / "audio.wav"
        # Pyannote runs on a separate WAV that skips the ASR
        # filtering chain (compressor + loudnorm). Those filters help
        # Whisper but degrade speaker embeddings — the compressor
        # smooths out the timbre cues pyannote uses to tell two voices
        # apart. Keep the ASR-targeted file for Whisper, give pyannote
        # an unfiltered mono 16 kHz stream.
        diar_wav_path = workspace / "audio.diar.wav"

        # --- step 0: warm the Odoo context pack before Whisper --------
        # The pack mutates ``request.glossary_terms`` with the
        # entity names mined from the linked record's neighbours
        # (opportunity → quotations, task → project, etc.). Doing it
        # here — before Whisper builds its initial prompt — is what
        # lets the boost actually bias the transcription rather than
        # only landing in the LLM correction pass. Cheap no-op when
        # no Odoo ref is wired.
        self._ensure_odoo_pack()

        # --- step 1: audio extract -------------------------------------
        results.append(self._extract_audio(source_path, wav_path, diar_wav_path))
        if not results[-1].ok:
            return results

        # --- step 1bis: VAD pre-filter --------------------------------
        whisper_wav = wav_path
        if self.request.transcription_settings.vad_enabled:
            vad_result = self._run_vad(wav_path, workspace)
            results.append(vad_result)
            if vad_result.ok and vad_result.artifact_path:
                whisper_wav = Path(vad_result.artifact_path)
            # VAD failures are never fatal: we fall back to the
            # full WAV and surface a WarningEvent.

        # --- step 2: Whisper first pass --------------------------------
        whisper_result, segments = self._run_whisper(whisper_wav, workspace)
        results.append(whisper_result)
        if not whisper_result.ok or not segments:
            return results

        # --- step 2½: decoder-loop recovery (PR AD) -------------------
        # If Whisper looped on quiet/ambiguous zones, ``_dropped_loops``
        # carries the lost ranges. Re-Whisper each range in isolation
        # with ``--clip-timestamps`` and a fresh decoder state — that
        # often breaks the loop because Whisper's repetition penalty
        # gets a clean slate. Recovered segments get spliced back
        # into the main list. The audit on Caste 21mai showed 95 min
        # of 132 min audio lost to this failure mode; recovery
        # claws back the majority on retry.
        if self._dropped_loops:
            recovery_result, segments = self._run_loop_recovery(
                whisper_wav, workspace, segments
            )
            if recovery_result is not None:
                results.append(recovery_result)

        # --- step 2*: hot prompt enrichment (PR D) --------------------
        # Mine the first-pass transcript for repeated capitalised
        # tokens that weren't in the glossary and fold them into
        # ``self.request.glossary_terms`` *in place*. The downstream
        # passes (multipass, boundary multipass, LLM, phonetic
        # post-process) all read from that list when building their
        # initial_prompt, so a single mutation propagates the
        # discovery without re-Whispering the file in chunks.
        if getattr(
            self.request.transcription_settings,
            "hot_prompt_enrichment",
            False,
        ):
            self._hot_enrich_glossary(segments)

        # --- step 2bis: confidence-triaged second pass ----------------
        if self.request.transcription_settings.multipass_enabled:
            multipass_result, segments = self._run_multipass(
                whisper_wav, workspace, segments
            )
            if multipass_result is not None:
                results.append(multipass_result)

        # --- step 2ter: remap timestamps back to source if VAD ran -----
        if self._vad_manifest:
            segments = remap_segments_to_source(segments, self._vad_manifest)

        # --- step 2quater: phonetic glossary post-processor -----------
        segments = self._run_phonetic_postprocess(segments)

        # --- step 3: diarisation (optional) ----------------------------
        analysis_segments = segments
        recognized_speakers: dict[str, str] = {}
        if self._should_run_diarisation():
            # Pyannote sees the unfiltered audio (no compressor /
            # loudnorm). Whisper still sees the enhanced WAV.
            diar_source = diar_wav_path if diar_wav_path.exists() else wav_path
            diar_result, turns = self._run_diarisation(diar_source)
            if diar_result is not None:
                results.append(diar_result)
            if turns:
                segments = assign_speakers_to_segments(segments, turns)
                analysis_segments = segments
                # --- step 3bis: speaker recognition --------------------
                # Match each SPEAKER_NN cluster against the stored
                # voice profiles so the rename sheet opens with the
                # right names already filled in. No-op when no profiles
                # exist yet — the user trains the system implicitly by
                # confirming names on each meeting.
                recognized_speakers = self._run_speaker_recognition(
                    diar_source, segments
                )
                # If voice matching didn't catch the user themselves
                # (typical on first runs: empty voice profile, so no
                # centroid match), attribute the first cluster that
                # speaks to ``current_user_name``. The phone-call
                # / meeting heuristic is reliable enough for the
                # 80 % case (user initiates / picks up the call,
                # speaks first) and gets confirmed at end of job via
                # auto-enrolment so future runs use voice matching.
                pre_attribution = self._pre_attribute_current_user(
                    segments, already_recognized=recognized_speakers
                )
                if pre_attribution:
                    recognized_speakers = {**recognized_speakers, **pre_attribution}
                if recognized_speakers:
                    segments = self._apply_speaker_renames(
                        segments, recognized_speakers
                    )
                    analysis_segments = segments
                # --- step 3ter: boundary multipass ----------------------
                # Re-Whisper short segments adjacent to a speaker
                # change. The first multipass (step 2bis) only
                # targets low-confidence avg_logprob segments — but
                # Whisper also fails at clean-confidence-but-wrong
                # words when its 30s context spans a turn boundary
                # (Caste pattern). The boundary pass uses the same
                # high-quality multipass model and is cost-bounded.
                if (
                    self.request.transcription_settings.multipass_enabled
                    and self._should_run_diarisation()
                ):
                    boundary_result, segments = self._run_boundary_multipass(
                        whisper_wav, workspace, segments
                    )
                    if boundary_result is not None:
                        results.append(boundary_result)
                    analysis_segments = segments
                # --- step 3quater: per-speaker Whisper pass --------------
                # Re-transcribe each speaker's segments separately so
                # Whisper conditions on a homogeneous voice — fixes
                # the failure mode where the model leaks one speaker's
                # vocabulary / accent into another's segments. Heavy
                # (multiplies Whisper invocations by N speakers) so
                # gated on the ``per_speaker_enabled`` flag.
                if (
                    self.request.transcription_settings.per_speaker_enabled
                    and self._should_run_diarisation()
                ):
                    per_speaker_result, segments = self._run_per_speaker_pass(
                        whisper_wav, workspace, segments
                    )
                    if per_speaker_result is not None:
                        results.append(per_speaker_result)
                    analysis_segments = segments

        # --- step 4: write the base transcript -----------------------
        transcript_path = self._write_transcript(segments, workspace)
        # Surface the user-facing transcript as a dedicated step so
        # the runner can wire it into the library DB without having
        # to introspect every Whisper intermediate.
        results.append(
            StepResult("transcript", True, str(transcript_path))
        )

        # --- step 5: LLM post-process (title, speakers, corrections) --
        llm_result = self._run_llm_post(analysis_segments, transcript_path)
        if llm_result is not None:
            results.append(llm_result)

        # --- step 5ter: multimodal audio recheck (PR F) ---------------
        # Re-listen to every passage the text LLM flagged as uncertain
        # with Qwen2-Audio via ``mlx_vlm``. Mutates the doubt entries
        # in place so ``_write_review_markdown`` can surface the
        # multimodal suggestions next to the original passage. Opt-in
        # via ``audio_recheck_enabled`` (Max preset turns it on once
        # the venv ships ``mlx_vlm``).
        audio_recheck_result = self._run_audio_recheck(whisper_wav, workspace)
        if audio_recheck_result is not None:
            results.append(audio_recheck_result)

        # --- step 5bis: apply LLM speaker renames if any --------------
        speakers = self._llm_payload.get("speakers") or {}
        if speakers and any(speakers.values()):
            segments = self._apply_speaker_renames(segments, speakers)
            # Rewrite the transcript with the friendly speaker names.
            transcript_path = self._write_transcript(
                segments, workspace, force_path=transcript_path
            )

        enhanced_path = self._write_enhanced_transcript(transcript_path, segments)
        if enhanced_path is not None:
            results.append(
                StepResult(
                    "enhanced_transcript",
                    True,
                    str(enhanced_path),
                    model=self.request.transcription_settings.text_llm_model,
                )
            )

        # --- step 6: produce the review markdown --------------------
        review_path = self._write_review_markdown(transcript_path, segments)
        if review_path is not None:
            results.append(StepResult("review", True, str(review_path)))

        # --- step 7: expose end-of-run context to the runner ----------
        # The runner persists these into the library DB so the
        # SwiftUI rename-speakers sheet finds the speaker list even
        # when the LLM didn't identify a single name. Building the
        # map from the final segments instead of from the LLM payload
        # guarantees the placeholders survive every code path: pure
        # diarisation, LLM-renamed, LLM-skipped, LLM-failed.
        self.final_segments = segments
        # Recognised names take precedence over LLM-inferred ones —
        # they came from a real voice match against a confirmed
        # profile, not a probabilistic guess off the transcript.
        merged_speakers = dict(speakers or {})
        for placeholder, name in recognized_speakers.items():
            if not name:
                continue
            merged_speakers[placeholder] = name
        self.final_speaker_map = self._build_speaker_map(
            segments, llm_speakers=merged_speakers
        )
        self.final_technical_terms = list(
            self._llm_payload.get("technical_terms") or []
        )
        self.final_title = str(self._llm_payload.get("title") or "").strip()
        # Always emit a ContextEvent — the SwiftUI app uses it to
        # surface live speaker hints in the queue view. Empty values
        # are fine: the sheet renders one row per placeholder either
        # way.
        if self.final_speaker_map or self.final_technical_terms:
            self.sink(
                ContextEvent(
                    speakers=self.final_speaker_map,
                    technical_terms=self.final_technical_terms,
                )
            )
        return results

    def _build_speaker_map(
        self,
        segments: list[dict],
        *,
        llm_speakers: dict[str, str],
    ) -> dict[str, str]:
        """Collect every speaker label appearing in ``segments`` and
        pair it with the friendly name from ``llm_speakers`` when one
        is available. Otherwise the value is "" — that's the signal
        the SwiftUI sheet needs to render an editable row.

        ``llm_speakers`` may also already have been *applied* to the
        segments (the rename pass rewrites the ``speaker`` field in
        place). In that case we map the friendly name to itself, so
        the rename sheet still shows a row the user can tweak.
        """
        labels: list[str] = []
        seen: set[str] = set()
        for seg in segments:
            label = str(seg.get("speaker") or "").strip()
            if not label or label in seen:
                continue
            seen.add(label)
            labels.append(label)
        if not labels:
            return {}
        names: dict[str, str] = {}
        for label in labels:
            # Prefer the LLM mapping when available; fall back to the
            # label itself if it's clearly a real name (no SPEAKER_NN
            # prefix) so the sheet shows the friendly value pre-filled.
            mapped = (llm_speakers.get(label) or "").strip()
            if mapped:
                names[label] = mapped
            elif label.upper().startswith("SPEAKER_"):
                names[label] = ""
            else:
                names[label] = label
        return names

    # ------------------------------------------------------------------
    # Step 1 — audio extract
    # ------------------------------------------------------------------

    def _extract_audio(
        self,
        source_path: str,
        wav_path: Path,
        diar_wav_path: Path | None = None,
    ) -> StepResult:
        """Extract the audio stream(s) needed for downstream stages.

        Two files are produced:

        * ``wav_path`` — mono 16 kHz, plus the speech-enhancement
          chain (highpass / lowpass / compressor / loudnorm) when the
          user enabled it. This is the file Whisper sees.
        * ``diar_wav_path`` — same mono 16 kHz, **without** the
          enhancement chain. Pyannote's clustering relies on timbre
          cues that the compressor smooths away; feeding it the raw
          stream measurably improves speaker-counting accuracy on
          recordings with both close-mic and lavalier voices.

        Failure to produce the diarisation WAV is non-fatal — the
        diarisation step falls back to the ASR-targeted WAV.
        """
        ts = time.monotonic()
        settings = self.request.transcription_settings
        ffmpeg = self.request.compression_settings.ffmpeg_path or "ffmpeg"
        ffprobe = self.request.compression_settings.ffprobe_path or ""
        # Detect telephony audio so the extraction step swaps in the
        # narrowband-tuned filter chain (tighter lowpass + FFT
        # denoise + heavier compression). Falls back to standard on
        # any probe error so the pipeline never blocks on detection.
        audio_profile = AUDIO_PROFILE_STANDARD
        if settings.enhance_audio and ffprobe:
            audio_profile = detect_audio_profile(source_path, ffprobe_path=ffprobe)
            if audio_profile != AUDIO_PROFILE_STANDARD:
                append_app_log(
                    f"engine_audio_profile detected={audio_profile!r} "
                    f"source={source_path!r}"
                )
                self.sink(
                    ProgressEvent(
                        "audio_extract",
                        0,
                        "Détection : audio téléphonique — filtre adaptatif activé",
                    )
                )
        cmd = build_audio_extract_cmd(
            ffmpeg,
            source_path,
            str(wav_path),
            speech_enhance=settings.enhance_audio,
            ss=self.request.compression_settings.trim_start
            if self.request.compression_settings.trim_enabled
            else None,
            to=self.request.compression_settings.trim_end
            if self.request.compression_settings.trim_enabled
            and self.request.compression_settings.trim_end != "00:00:00"
            else None,
            audio_profile=audio_profile,
        )
        self.sink(ProgressEvent("audio_extract", 0, "Extracting audio"))
        proc = subprocess.run(
            cmd, capture_output=True, text=True, env=self._subprocess_env()
        )
        duration = time.monotonic() - ts
        if proc.returncode != 0 or not wav_path.exists():
            raw = tail_text(proc.stderr or proc.stdout)
            message = _friendly_ffmpeg_error(raw, cmd[0])
            append_app_log(f"engine_audio_extract_failed rc={proc.returncode} error={raw!r}")
            return StepResult("audio_extract", False, duration_seconds=duration, error=message)

        if diar_wav_path is not None:
            diar_cmd = build_audio_extract_cmd(
                ffmpeg,
                source_path,
                str(diar_wav_path),
                speech_enhance=False,
                ss=self.request.compression_settings.trim_start
                if self.request.compression_settings.trim_enabled
                else None,
                to=self.request.compression_settings.trim_end
                if self.request.compression_settings.trim_enabled
                and self.request.compression_settings.trim_end != "00:00:00"
                else None,
            )
            diar_proc = subprocess.run(
                diar_cmd,
                capture_output=True,
                text=True,
                env=self._subprocess_env(),
            )
            if diar_proc.returncode != 0 or not diar_wav_path.exists():
                # Don't fail the whole job — diarisation will reuse
                # the ASR WAV when this one is missing.
                append_app_log(
                    f"engine_diar_audio_extract_failed rc={diar_proc.returncode} "
                    f"stderr={tail_text(diar_proc.stderr)!r}"
                )
            else:
                self.sink(ArtifactEvent("audio_diar_wav", str(diar_wav_path)))

        self.sink(ArtifactEvent("audio_wav", str(wav_path)))
        self.sink(ProgressEvent("audio_extract", 100, "Audio ready"))
        return StepResult("audio_extract", True, str(wav_path), duration_seconds=duration)

    # ------------------------------------------------------------------
    # Step 1bis — VAD pre-filter
    # ------------------------------------------------------------------

    def _run_vad(self, wav_path: Path, workspace: Path) -> StepResult:
        """Trim non-speech regions before Whisper. On phone-call
        recordings this kills ~30% of the compute (no IVR transcription)
        and prevents hallucinated 'Merci de rester en ligne' loops.

        Falls back to the original WAV if anything goes wrong — VAD is
        an optimisation, not a hard requirement.
        """
        ts = time.monotonic()
        settings = self.request.transcription_settings
        venv_python = settings.venv_python_path
        if not venv_python or not Path(venv_python).exists():
            self.sink(
                WarningEvent(
                    "VAD ignoré : environnement Python introuvable.",
                    code="vad_no_venv",
                )
            )
            return StepResult(
                "vad",
                ok=False,
                duration_seconds=time.monotonic() - ts,
                error="venv missing",
            )

        trimmed = workspace / "audio.vad.wav"
        cmd = build_vad_cmd(venv_python, str(wav_path), str(trimmed))
        self.sink(ProgressEvent("vad", 0, "Filtering non-speech regions"))
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=900,
            env=self._subprocess_env(),
        )
        duration = time.monotonic() - ts
        if proc.returncode != 0:
            self.sink(
                WarningEvent(
                    "VAD désactivé pour ce job : "
                    + tail_text(proc.stderr or proc.stdout),
                    code="vad_failed",
                )
            )
            append_app_log(f"engine_vad_failed rc={proc.returncode} stderr={tail_text(proc.stderr)!r}")
            return StepResult("vad", False, duration_seconds=duration, error="vad failed")
        try:
            payload = parse_vad_manifest(proc.stdout)
        except Exception as exc:
            self.sink(WarningEvent(f"VAD output illisible : {exc}", code="vad_parse"))
            return StepResult("vad", False, duration_seconds=duration, error=str(exc))

        spans = payload.get("spans") or []
        if not spans or not trimmed.exists():
            # Nothing detected as speech — keep the original WAV.
            self.sink(ProgressEvent("vad", 100, "VAD found no speech, using original audio"))
            return StepResult("vad", True, str(wav_path), duration_seconds=duration)

        kept = payload.get("trimmed_seconds") or 0
        total = payload.get("total_seconds") or 0
        ratio = (kept / total) if total else 0

        # Safety net: if VAD claims more than 60 % of the audio is
        # non-speech we almost certainly lost real content (faint
        # voices, overlapping speech, noisy mics). Fall back to the
        # full audio so the user gets *something* over a transcript
        # missing half the meeting.
        if total >= 30 and ratio < 0.4:
            self.sink(
                WarningEvent(
                    f"VAD trop agressif ({ratio*100:.0f}% du signal conservé) — "
                    "passage en transcription complète pour préserver les voix faibles.",
                    code="vad_fallback",
                )
            )
            self._vad_manifest = []
            return StepResult(
                "vad",
                True,
                str(wav_path),
                duration_seconds=duration,
                metrics={
                    "kept_seconds": kept,
                    "total_seconds": total,
                    "spans": len(spans),
                    "fallback": True,
                },
            )

        self._vad_manifest = spans
        self.sink(
            ProgressEvent(
                "vad",
                100,
                f"Voice-only stream ready ({kept:.0f}s / {total:.0f}s, "
                f"{ratio*100:.0f}% of audio)",
            )
        )
        return StepResult(
            "vad",
            True,
            str(trimmed),
            duration_seconds=duration,
            metrics={
                "kept_seconds": kept,
                "total_seconds": total,
                "spans": len(spans),
            },
        )

    # ------------------------------------------------------------------
    # Step 2 — Whisper
    # ------------------------------------------------------------------

    def _run_whisper(
        self, wav_path: Path, workspace: Path
    ) -> tuple[StepResult, list[dict]]:
        ts = time.monotonic()
        settings = self.request.transcription_settings
        whisper_json = workspace / "whisper.json"
        glossary = "\n".join(
            [*self.request.glossary_terms, *self.request.technical_terms]
        )
        # Word timestamps drive per-word speaker attribution in the
        # diarisation step (see ``assign_speakers_to_segments``).
        # Skip them when diarisation is disabled — they cost roughly
        # 5 % of the runtime for no downstream benefit.
        request_word_timestamps = self._should_run_diarisation()
        cmd = build_mlx_whisper_cmd(
            mlx_whisper_path=settings.mlx_whisper_path or "mlx_whisper",
            audio_path=str(wav_path),
            output_path=str(whisper_json),
            model=settings.model,
            language=settings.language,
            output_format="json",
            initial_prompt=structured_initial_prompt(
                glossary,
                expected_speaker_names=self._expected_speaker_names(),
                meeting_context=self._meeting_context(),
            ),
            # Context-aware decoding (PR D): off by default — the
            # ``max`` preset turns it on. ``clean_whisper_segments``
            # downstream drops decoder loops of length > 2 so a
            # rogue hallucinated phrase can't poison more than two
            # consecutive windows.
            condition_on_previous_text=bool(
                getattr(settings, "condition_on_previous_text", False)
            ),
            word_timestamps=request_word_timestamps,
        )
        self.sink(ProgressEvent("whisper", 0, f"Running Whisper ({settings.model})"))
        # env= prepends the bundle's bin/ to PATH so mlx_whisper's
        # internal subprocess.run(["ffmpeg", ...]) call from
        # mlx_whisper/audio.py:load_audio() actually finds the
        # binary. Without this, the venv-side Python raises
        # FileNotFoundError('ffmpeg'), mlx_whisper's cli.py main()
        # catches it and exits cleanly (rc=0) — and our pipeline
        # only notices because whisper.json never appears.
        proc = subprocess.run(
            cmd, capture_output=True, text=True, env=self._subprocess_env()
        )
        duration = time.monotonic() - ts
        if proc.returncode != 0 or not whisper_json.exists():
            raw = tail_text(proc.stderr or proc.stdout)
            message = _friendly_ffmpeg_error(raw, cmd[0])
            append_app_log(f"engine_whisper_failed rc={proc.returncode} error={raw!r}")
            return (
                StepResult(
                    "whisper",
                    False,
                    model=settings.model,
                    duration_seconds=duration,
                    error=message,
                ),
                [],
            )

        # PR Y: capture decoder-loop ranges so we can surface a
        # WarningEvent when the cleaner drops minutes of looped
        # output. The audit on the Caste run found a 70-minute
        # stretch of repeated ``"On est sur Zindoc"`` dropped
        # silently — the user lost an hour of meeting with no signal.
        # PR AB: persist the list on self so ``_write_review_markdown``
        # can render a "Zones perdues" section. Live WarningEvent
        # alone wasn't enough — the user reads the .md file long
        # after the run, when the live events are gone.
        dropped_loops: list[Any] = []
        segments = parse_whisper_json_segments(
            str(whisper_json), dropped_loops=dropped_loops
        )
        self._dropped_loops = list(dropped_loops)
        # PR AB: cache the audio duration Whisper saw (last segment
        # end before cleanup) so the markdown can compute coverage.
        # We use the post-clean segments here as a floor — the real
        # audio length might be slightly longer if the trailing
        # tail was itself dropped, but that's an acceptable
        # under-estimate (coverage shown will be conservative).
        try:
            self._audio_seconds = max(
                float(seg.get("end") or 0.0) for seg in segments
            ) if segments else 0.0
        except (TypeError, ValueError):
            self._audio_seconds = 0.0
        # If loops were dropped, their END timestamps are also
        # audio our microphone captured — bump the cached duration
        # so coverage doesn't ignore the lost regions.
        for loop in dropped_loops:
            try:
                self._audio_seconds = max(self._audio_seconds, float(loop.end))
            except (TypeError, ValueError):
                continue
        if dropped_loops:
            total_dropped = sum(loop.dropped for loop in dropped_loops)
            total_seconds = sum(
                max(0.0, loop.end - loop.start) for loop in dropped_loops
            )
            # Most-impactful loop first so the user sees the worst
            # range in the warning text.
            worst = max(dropped_loops, key=lambda l: l.end - l.start)
            self.sink(
                WarningEvent(
                    f"Whisper a bouclé sur {len(dropped_loops)} zone(s) "
                    f"({total_dropped} segments, ~{total_seconds:.0f}s d'audio). "
                    f"La pire : {worst.start:.0f}s → {worst.end:.0f}s "
                    f"(\"{worst.text[:60]}…\"). "
                    f"Le contenu de ces zones n'est pas dans la transcription.",
                    code="whisper_decoder_loop_dropped",
                )
            )
            append_app_log(
                "engine_whisper_decoder_loops "
                f"count={len(dropped_loops)} dropped_segments={total_dropped} "
                f"dropped_seconds={total_seconds:.1f}"
            )
        self.sink(
            ProgressEvent("whisper", 100, f"Transcript ready ({len(segments)} segments)")
        )
        return (
            StepResult(
                "whisper",
                True,
                str(whisper_json),
                model=settings.model,
                duration_seconds=duration,
                metrics={
                    "segments": len(segments),
                    "decoder_loops": len(dropped_loops),
                },
            ),
            segments,
        )

    # ------------------------------------------------------------------
    # Step 2½ — decoder-loop recovery (PR AD)
    # ------------------------------------------------------------------

    # Cost / quality knobs. Caps tuned on the Caste 21mai audit:
    # 5 loops detected, total 95 min of lost audio. With these
    # caps recovery costs roughly 10-15 min extra wall time and
    # claws back most of the loss.
    _LOOP_RECOVERY_MAX_RANGES = 8
    _LOOP_RECOVERY_PADDING_SECONDS = 1.0
    _LOOP_RECOVERY_MIN_DURATION_SECONDS = 2.0

    def _run_loop_recovery(
        self,
        whisper_wav: Path,
        workspace: Path,
        segments: list[dict],
    ) -> tuple[StepResult | None, list[dict]]:
        """Re-Whisper the ranges that ``_run_whisper`` dropped as
        decoder loops, splice the recovered text back into the
        segment list.

        Strategy:
          1. Pick up to ``_LOOP_RECOVERY_MAX_RANGES`` lost ranges
             from ``self._dropped_loops``, worst-first.
          2. For each range, run ``mlx_whisper --clip-timestamps``
             with ``--condition-on-previous-text False`` so the
             decoder gets a fresh state. The loop ALMOST always
             breaks on retry because the runaway phrase is no
             longer in the priming context.
          3. Run the cleaner on the new output (with its own
             ``dropped_loops`` collector) — if the retry STILL
             loops, mark the zone as unrecoverable and keep it in
             ``self._dropped_loops`` for the markdown to surface.
          4. Splice recovered segments into ``segments`` and sort
             by start time. ``self._dropped_loops`` is filtered to
             contain only the still-unrecoverable zones; recovered
             ones go to ``self._recovered_loops`` for the metric.

        Returns ``(StepResult, segments)``. StepResult is ``None``
        when nothing was attempted (no loops to recover).
        """
        if not self._dropped_loops or not whisper_wav.exists():
            return None, segments
        settings = self.request.transcription_settings
        mlx_path = settings.mlx_whisper_path or "mlx_whisper"
        glossary = "\n".join(
            [*self.request.glossary_terms, *self.request.technical_terms]
        )

        # Order worst-first so we recover the most painful losses
        # before hitting the per-run cap.
        ordered = sorted(
            self._dropped_loops,
            key=lambda loop: (loop.end - loop.start),
            reverse=True,
        )
        targets = ordered[: self._LOOP_RECOVERY_MAX_RANGES]
        # Anything past the cap stays in ``_dropped_loops`` for the
        # markdown alert.
        leftovers = ordered[self._LOOP_RECOVERY_MAX_RANGES :]

        ts = time.monotonic()
        recovered_segments: list[dict] = []
        still_lost: list[Any] = list(leftovers)  # type: list[DroppedLoop]
        recovered_count = 0

        self.sink(
            ProgressEvent(
                "loop_recovery",
                0,
                f"Récupération de {len(targets)} zone(s) perdue(s) "
                f"par re-Whisper",
            )
        )

        for idx, loop in enumerate(targets):
            duration_s = float(loop.end - loop.start)
            if duration_s < self._LOOP_RECOVERY_MIN_DURATION_SECONDS:
                # Too short to bother — keep as lost.
                still_lost.append(loop)
                continue
            clip_start = max(0.0, float(loop.start) - self._LOOP_RECOVERY_PADDING_SECONDS)
            clip_end = float(loop.end) + self._LOOP_RECOVERY_PADDING_SECONDS
            target_json = workspace / f"loop_recovery_{idx}.json"

            cmd = build_mlx_whisper_cmd(
                mlx_whisper_path=mlx_path,
                audio_path=str(whisper_wav),
                output_path=str(target_json),
                model=settings.model,
                language=settings.language,
                output_format="json",
                initial_prompt=structured_initial_prompt(
                    glossary,
                    expected_speaker_names=self._expected_speaker_names(),
                    meeting_context=self._meeting_context(),
                ),
                # The whole POINT — escape the looped context.
                condition_on_previous_text=False,
                clip_timestamps=f"{clip_start:.2f},{clip_end:.2f}",
                word_timestamps=self._should_run_diarisation(),
            )
            self.sink(
                ProgressEvent(
                    "loop_recovery",
                    int(100 * idx / max(len(targets), 1)),
                    f"Recovery {idx + 1}/{len(targets)} "
                    f"({clip_start:.0f}s → {clip_end:.0f}s)",
                )
            )

            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=1800,
                    env=self._subprocess_env(),
                )
            except Exception as exc:  # pragma: no cover — defensive
                append_app_log(
                    f"engine_loop_recovery_failed loop_idx={idx} "
                    f"err={exc!r}"
                )
                still_lost.append(loop)
                continue

            if proc.returncode != 0 or not target_json.exists():
                append_app_log(
                    f"engine_loop_recovery_clip_failed loop_idx={idx} "
                    f"rc={proc.returncode} stderr={tail_text(proc.stderr)!r}"
                )
                still_lost.append(loop)
                continue

            # Run the cleaner on the recovered output — if Whisper
            # looped AGAIN even without context propagation, we
            # don't want to inject those segments. Track its own
            # dropped_loops so we can detect that case.
            sub_dropped: list[Any] = []
            try:
                clip_segments = parse_whisper_json_segments(
                    str(target_json), dropped_loops=sub_dropped
                )
            except Exception as exc:
                append_app_log(
                    f"engine_loop_recovery_parse_failed loop_idx={idx} "
                    f"err={exc!r}"
                )
                still_lost.append(loop)
                continue

            # Did the retry loop again? If a meaningful share of
            # the recovered audio is itself dropped as a sub-loop,
            # admit defeat for this zone.
            sub_dropped_seconds = sum(
                max(0.0, l.end - l.start) for l in sub_dropped
            )
            if sub_dropped_seconds > duration_s * 0.5:
                append_app_log(
                    f"engine_loop_recovery_re_looped loop_idx={idx} "
                    f"sub_dropped={sub_dropped_seconds:.0f}s "
                    f"original={duration_s:.0f}s"
                )
                still_lost.append(loop)
                continue

            if not clip_segments:
                # Clean ran fine but nothing came out — the zone
                # was probably pure silence, mark recovered with
                # zero content (the gap is now legitimate silence).
                recovered_count += 1
                continue

            # Whisper's ``--clip-timestamps`` reports times
            # relative to the clip start, so shift back to the
            # whisper_wav timeline.
            for seg in clip_segments:
                try:
                    seg["start"] = float(seg.get("start") or 0.0) + clip_start
                    seg["end"] = float(seg.get("end") or 0.0) + clip_start
                except (TypeError, ValueError):
                    continue
            recovered_segments.extend(clip_segments)
            recovered_count += 1

        # Splice: insert recovered segments and re-sort by start time.
        # Simpler than tracking insertion points individually, and
        # cheap at segment-list scale (typical: < 5 000 entries).
        merged_segments = list(segments) + recovered_segments
        merged_segments.sort(key=lambda s: float(s.get("start") or 0.0))

        # Update the persistent state so the markdown reflects the
        # post-recovery reality. ``_recovered_loops`` = ``targets``
        # minus the ones that ended up in ``still_lost`` (we use
        # ``id()`` because ``DroppedLoop`` is frozen and we want
        # identity, not equality, to dedupe).
        still_lost_ids = {id(loop) for loop in still_lost}
        self._recovered_loops = [
            loop for loop in targets if id(loop) not in still_lost_ids
        ]
        self._dropped_loops = still_lost

        duration = time.monotonic() - ts
        recovered_seconds = sum(
            max(0.0, l.end - l.start) for l in self._recovered_loops
        )
        still_lost_seconds = sum(
            max(0.0, l.end - l.start) for l in still_lost
        )
        self.sink(
            ProgressEvent(
                "loop_recovery",
                100,
                f"Recovered {recovered_count}/{len(targets)} zone(s) "
                f"(~{recovered_seconds:.0f}s récupérées)",
            )
        )
        append_app_log(
            f"engine_loop_recovery attempted={len(targets)} "
            f"recovered={recovered_count} still_lost={len(still_lost)} "
            f"recovered_seconds={recovered_seconds:.1f} "
            f"still_lost_seconds={still_lost_seconds:.1f}"
        )

        return (
            StepResult(
                "loop_recovery",
                True,
                duration_seconds=duration,
                metrics={
                    "attempted": len(targets),
                    "recovered": recovered_count,
                    "still_lost": len(still_lost),
                    "recovered_seconds": int(recovered_seconds),
                    "still_lost_seconds": int(still_lost_seconds),
                },
            ),
            merged_segments,
        )

    # ------------------------------------------------------------------
    # Step 2bis — confidence-triaged second pass
    # ------------------------------------------------------------------

    def _run_multipass(
        self,
        whisper_wav: Path,
        workspace: Path,
        segments: list[dict],
    ) -> tuple[StepResult | None, list[dict]]:
        """Re-transcribe the bottom 10-20% of segments with the
        higher-quality whisper-large-v3 (non-turbo). Cost guard: skip
        when more than 30% of the audio is weak; that's a hint to
        switch the first-pass model rather than burning 4× the time.
        """
        ts = time.monotonic()
        settings = self.request.transcription_settings
        weak = identify_weak_segments(segments)
        if not weak:
            return None, segments
        clip_ranges = group_into_clip_ranges(weak)
        if not clip_ranges:
            return None, segments

        total_weak = sum(e - s for s, e in clip_ranges)
        try:
            total_audio = sum(
                float(s.get("end") or 0) - float(s.get("start") or 0) for s in segments
            )
        except Exception:
            total_audio = 0.0
        if total_audio > 0 and total_weak / total_audio > 0.3:
            append_app_log(
                f"engine_multipass_skipped reason=too_much_weak "
                f"weak_s={total_weak:.1f} total_s={total_audio:.1f}"
            )
            return None, segments

        # User-selectable in Settings (Models tab) since the
        # role-based refactor. ``canonical_multipass_model_id``
        # falls back to ``whisper-large-v3-mlx`` when the user
        # hasn't picked anything, preserving the previous default.
        repass_model = canonical_multipass_model_id(settings.multipass_model)
        self.sink(
            ProgressEvent(
                "multipass",
                0,
                f"High-quality repass on {len(clip_ranges)} weak zone(s)",
            )
        )
        new_segments: list[dict] = []
        glossary = "\n".join([*self.request.glossary_terms, *self.request.technical_terms])
        for idx, (cs, ce) in enumerate(clip_ranges):
            target = workspace / f"repass_{idx}.json"
            cmd = build_mlx_whisper_cmd(
                mlx_whisper_path=settings.mlx_whisper_path or "mlx_whisper",
                audio_path=str(whisper_wav),
                output_path=str(target),
                model=repass_model,
                language=settings.language,
                output_format="json",
                initial_prompt=structured_initial_prompt(
                    glossary,
                    expected_speaker_names=self._expected_speaker_names(),
                    meeting_context=self._meeting_context(),
                ),
                condition_on_previous_text=False,
                clip_timestamps=f"{cs:.2f},{ce:.2f}",
                # Repassed segments need to carry word timestamps too
                # so the post-multipass diarisation step can split
                # them on speaker boundaries — same reason as the
                # primary pass.
                word_timestamps=self._should_run_diarisation(),
            )
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800,
                env=self._subprocess_env(),
            )
            if proc.returncode != 0 or not target.exists():
                append_app_log(
                    f"engine_multipass_clip_failed rc={proc.returncode} "
                    f"stderr={tail_text(proc.stderr)!r}"
                )
                continue
            try:
                clip_segments = parse_whisper_json_segments(str(target))
            except Exception as exc:
                append_app_log(f"engine_multipass_clip_parse_failed error={exc!r}")
                continue
            # Whisper's `--clip-timestamps` reports times relative to
            # the clip start, so shift back to wav-stream time.
            for seg in clip_segments:
                try:
                    seg["start"] = float(seg.get("start") or 0.0) + cs
                    seg["end"] = float(seg.get("end") or 0.0) + cs
                except (TypeError, ValueError):
                    continue
            new_segments.extend(clip_segments)

        if not new_segments:
            return None, segments

        merged, replaced = merge_repass_segments(segments, new_segments, clip_ranges)
        self._repass_replaced = replaced
        duration = time.monotonic() - ts
        self.sink(
            ProgressEvent(
                "multipass",
                100,
                f"Repass replaced {replaced} weak segment(s)",
            )
        )
        return (
            StepResult(
                "multipass",
                True,
                model=repass_model,
                duration_seconds=duration,
                metrics={"replaced": replaced, "clip_ranges": len(clip_ranges)},
            ),
            merged,
        )

    # ------------------------------------------------------------------
    # Step 3ter — boundary multipass (post-diarisation)
    # ------------------------------------------------------------------

    def _run_boundary_multipass(
        self,
        whisper_wav: Path,
        workspace: Path,
        segments: list[dict],
    ) -> tuple[StepResult | None, list[dict]]:
        """Re-Whisper short segments adjacent to a speaker change.

        The first multipass (``_run_multipass``) only targets
        avg_logprob-weak segments. But Whisper also produces wrong
        words at clean-confidence-but-misaligned-boundary positions:
        its 30s context window crosses a turn boundary and conditions
        on the wrong voice. This pass catches that failure mode after
        diarisation has shown us where the boundaries actually are.

        Cost guard: cap at 8 clip ranges per job. Beyond that the
        recording is too conversational for a per-boundary repass to
        be cost-effective — the user should rerun on the Max preset
        which will eventually carry the per-speaker pass (PR E).
        """
        ts = time.monotonic()
        settings = self.request.transcription_settings
        boundary = identify_boundary_segments(segments)
        if not boundary:
            return None, segments
        # max_segments_per_clip=1 keeps each repass clip bounded to a
        # SINGLE original segment. Groups of 4 used to span multiple
        # speakers' turns; the new Whisper output for that range got
        # one speaker label via majority overlap, but its audio range
        # then mixed multiple voices — visible in the rename sheet
        # samples as "voices mélangées".
        clip_ranges = group_into_clip_ranges(boundary, max_segments_per_clip=1)
        if not clip_ranges or len(clip_ranges) > 8:
            if clip_ranges:
                append_app_log(
                    f"engine_boundary_multipass_skipped reason=too_many "
                    f"ranges={len(clip_ranges)}"
                )
            return None, segments

        repass_model = canonical_multipass_model_id(settings.multipass_model)
        self.sink(
            ProgressEvent(
                "multipass_boundary",
                0,
                f"Repass on {len(clip_ranges)} speaker-boundary zone(s)",
            )
        )
        new_segments: list[dict] = []
        glossary = "\n".join([*self.request.glossary_terms, *self.request.technical_terms])
        for idx, (cs, ce) in enumerate(clip_ranges):
            target = workspace / f"boundary_repass_{idx}.json"
            cmd = build_mlx_whisper_cmd(
                mlx_whisper_path=settings.mlx_whisper_path or "mlx_whisper",
                audio_path=str(whisper_wav),
                output_path=str(target),
                model=repass_model,
                language=settings.language,
                output_format="json",
                initial_prompt=structured_initial_prompt(
                    glossary,
                    expected_speaker_names=self._expected_speaker_names(),
                    meeting_context=self._meeting_context(),
                ),
                condition_on_previous_text=False,
                clip_timestamps=f"{cs:.2f},{ce:.2f}",
                word_timestamps=True,
            )
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800,
                env=self._subprocess_env(),
            )
            if proc.returncode != 0 or not target.exists():
                append_app_log(
                    f"engine_boundary_multipass_clip_failed rc={proc.returncode} "
                    f"stderr={tail_text(proc.stderr)!r}"
                )
                continue
            try:
                clip_segments = parse_whisper_json_segments(str(target))
            except Exception as exc:
                append_app_log(
                    f"engine_boundary_multipass_clip_parse_failed error={exc!r}"
                )
                continue
            for seg in clip_segments:
                try:
                    seg["start"] = float(seg.get("start") or 0.0) + cs
                    seg["end"] = float(seg.get("end") or 0.0) + cs
                except (TypeError, ValueError):
                    continue
            new_segments.extend(clip_segments)

        if not new_segments:
            return None, segments
        # Reuse the merge helper from the regular multipass — it
        # already knows how to splice repassed clips into the
        # original timeline. Speaker labels are dropped during merge
        # (the new clip has none), so we run the diarisation
        # projection again on the merged segments.
        merged, replaced = merge_repass_segments(segments, new_segments, clip_ranges)
        # Re-project speakers onto the freshly-transcribed clip text.
        # We don't have the diarisation turns object here, so reuse
        # the existing labels in the original segments where they
        # cover the same timestamps.
        merged = self._fill_speakers_from_neighbours(merged, segments)
        duration = time.monotonic() - ts
        self.sink(
            ProgressEvent(
                "multipass_boundary",
                100,
                f"Boundary repass replaced {replaced} segment(s)",
            )
        )
        return (
            StepResult(
                "multipass_boundary",
                True,
                model=repass_model,
                duration_seconds=duration,
                metrics={"replaced": replaced, "clip_ranges": len(clip_ranges)},
            ),
            merged,
        )

    def _run_per_speaker_pass(
        self,
        whisper_wav: Path,
        workspace: Path,
        segments: list[dict],
    ) -> tuple[StepResult | None, list[dict]]:
        """Re-Whisper each speaker's segments separately.

        The Whisper context window (30 s) crosses speaker boundaries
        on conversational recordings. When two voices have distinct
        timbres / accents the model can "leak" vocabulary between
        them ("Sudokiz" once leaked into the other speaker's line).
        Running Whisper on a per-speaker chunk eliminates that
        cross-contamination at the cost of one extra invocation per
        speaker.

        Implementation:
        - Group segments by speaker (skip ``None`` and short
          fragments < 1 s).
        - For each speaker, group consecutive segments into clip
          ranges (same helper as multipass uses).
        - Run Whisper on each clip range with the higher-quality
          multipass model, then merge the resulting text back in.

        Cost guard: skip clusters with < 6 s total speech (centroid
        wouldn't benefit, and short clusters are usually the user's
        "merci / oui" channel that doesn't need a quality boost).
        """
        ts = time.monotonic()
        settings = self.request.transcription_settings
        # Group eligible segments by speaker.
        per_speaker: dict[str, list[dict]] = {}
        for idx, seg in enumerate(segments):
            speaker = (seg.get("speaker") or "").strip()
            if not speaker:
                continue
            try:
                s = float(seg.get("start") or 0)
                e = float(seg.get("end") or 0)
            except (TypeError, ValueError):
                continue
            if e - s < 0.4:
                continue
            per_speaker.setdefault(speaker, []).append({"index": idx, "start": s, "end": e})

        # Discard speakers with too little speech to repass usefully.
        eligible: dict[str, list[dict]] = {}
        for speaker, segs in per_speaker.items():
            total = sum(s["end"] - s["start"] for s in segs)
            if total >= 6.0:
                eligible[speaker] = segs

        if not eligible:
            return None, segments

        repass_model = canonical_multipass_model_id(settings.multipass_model)
        self.sink(
            ProgressEvent(
                "multipass_per_speaker",
                0,
                f"Per-speaker repass on {len(eligible)} cluster(s)",
            )
        )

        replacements: list[tuple[int, dict]] = []
        for sp_index, (speaker, segs) in enumerate(sorted(eligible.items())):
            # Build clip ranges from this speaker's segments. The
            # ``group_into_clip_ranges`` helper merges close-together
            # entries with a 1 s pad, exactly what we want here.
            weak = [
                WeakSegment(
                    index=s["index"], start=s["start"], end=s["end"],
                    score=0.0, reason="per_speaker",
                )
                for s in segs
            ]
            # Same constraint as boundary multipass: keep each
            # Whisper repass bounded to a SINGLE original segment so
            # the per-speaker audio range never accidentally covers
            # an adjacent other-speaker turn.
            clip_ranges = group_into_clip_ranges(
                weak, max_segments_per_clip=1
            )
            if not clip_ranges:
                continue
            new_clip_segments: list[dict] = []
            for idx, (cs, ce) in enumerate(clip_ranges):
                target = workspace / f"per_speaker_{sp_index}_{idx}.json"
                cmd = build_mlx_whisper_cmd(
                    mlx_whisper_path=settings.mlx_whisper_path or "mlx_whisper",
                    audio_path=str(whisper_wav),
                    output_path=str(target),
                    model=repass_model,
                    language=settings.language,
                    output_format="json",
                    initial_prompt=structured_initial_prompt(
                        "\n".join(
                            [*self.request.glossary_terms, *self.request.technical_terms]
                        ),
                        expected_speaker_names=self._expected_speaker_names(),
                        meeting_context=self._meeting_context(),
                    ),
                    condition_on_previous_text=False,
                    clip_timestamps=f"{cs:.2f},{ce:.2f}",
                    word_timestamps=True,
                )
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=1800,
                    env=self._subprocess_env(),
                )
                if proc.returncode != 0 or not target.exists():
                    append_app_log(
                        f"engine_per_speaker_clip_failed speaker={speaker!r} "
                        f"rc={proc.returncode}"
                    )
                    continue
                try:
                    clip_segments = parse_whisper_json_segments(str(target))
                except Exception:
                    continue
                for seg in clip_segments:
                    try:
                        seg["start"] = float(seg.get("start") or 0.0) + cs
                        seg["end"] = float(seg.get("end") or 0.0) + cs
                        # Tag with the speaker we know — bypass
                        # the diarisation re-projection step since
                        # we extracted by speaker upfront.
                        seg["speaker"] = speaker
                    except (TypeError, ValueError):
                        continue
                    new_clip_segments.append(seg)
            if not new_clip_segments:
                continue
            replacements.append((sp_index, {
                "speaker": speaker,
                "clip_ranges": clip_ranges,
                "segments": new_clip_segments,
            }))

        if not replacements:
            return None, segments

        # Splice the per-speaker results into the original timeline.
        # For each speaker we run ``merge_repass_segments`` with the
        # clip ranges as targets; same logic as the global multipass.
        merged_segments = segments
        replaced_total = 0
        for _, payload in replacements:
            merged_segments, replaced = merge_repass_segments(
                merged_segments,
                payload["segments"],
                payload["clip_ranges"],
            )
            replaced_total += replaced

        duration = time.monotonic() - ts
        self.sink(
            ProgressEvent(
                "multipass_per_speaker",
                100,
                f"Per-speaker repass: {replaced_total} segment(s) replaced",
            )
        )
        return (
            StepResult(
                "multipass_per_speaker",
                True,
                model=repass_model,
                duration_seconds=duration,
                metrics={
                    "speakers": len(eligible),
                    "replaced": replaced_total,
                },
            ),
            merged_segments,
        )

    def _fill_speakers_from_neighbours(
        self,
        merged: list[dict],
        original: list[dict],
    ) -> list[dict]:
        """Restore speaker labels on segments that lost them through
        the boundary repass merge.

        The merge helper concatenates new clips into the original
        timeline; the new clips have no ``speaker`` field. We rebuild
        it by looking at the overlap with the original (pre-repass)
        segments — whichever speaker dominated the timespan keeps
        that label.
        """
        if not merged or not original:
            return merged
        for seg in merged:
            if (seg.get("speaker") or "").strip():
                continue
            try:
                s = float(seg.get("start") or 0)
                e = float(seg.get("end") or 0)
            except (TypeError, ValueError):
                continue
            per_speaker: dict[str, float] = {}
            for o in original:
                speaker = (o.get("speaker") or "").strip()
                if not speaker:
                    continue
                try:
                    os_ = float(o.get("start") or 0)
                    oe_ = float(o.get("end") or 0)
                except (TypeError, ValueError):
                    continue
                overlap = max(0.0, min(e, oe_) - max(s, os_))
                if overlap > 0:
                    per_speaker[speaker] = per_speaker.get(speaker, 0.0) + overlap
            if per_speaker:
                seg["speaker"] = max(per_speaker.items(), key=lambda kv: kv[1])[0]
        return merged

    # ------------------------------------------------------------------
    # Step 2quater — phonetic glossary post-processor
    # ------------------------------------------------------------------

    def _run_phonetic_postprocess(self, segments: list[dict]) -> list[dict]:
        terms = list(self.request.glossary_terms) + list(self.request.technical_terms)
        # Some callers paste the glossary as a single multi-line blob;
        # parse_glossary_terms handles both shapes.
        if len(terms) == 1 and ("\n" in terms[0] or "," in terms[0]):
            terms = parse_glossary_terms(terms[0])
        if not terms:
            return segments
        new_segments, subs = apply_glossary_to_segments(segments, terms)
        if subs:
            self._glossary_subs.extend(subs)
            append_app_log(
                f"engine_phonetic_postprocess applied={len(subs)} terms={len(terms)}"
            )
        return new_segments

    # ------------------------------------------------------------------
    # Step 3 — diarisation
    # ------------------------------------------------------------------

    def _should_run_diarisation(self) -> bool:
        settings = self.request.transcription_settings
        return bool(
            settings.diarization_enabled
            and settings.hf_token
            and settings.venv_python_path
            and Path(settings.venv_python_path).exists()
        )

    def _run_diarisation(self, wav_path: Path) -> tuple[StepResult | None, list[dict]]:
        ts = time.monotonic()
        settings = self.request.transcription_settings
        cmd = build_diarization_cmd(
            settings.venv_python_path,
            str(wav_path),
            min_speakers=(
                settings.expected_min_speakers
                if settings.expected_min_speakers > 0
                else None
            ),
            max_speakers=(
                settings.expected_max_speakers
                if settings.expected_max_speakers > 0
                else None
            ),
        )
        # Build on top of the bundle's PATH so pyannote / torchaudio
        # can shell out to ffmpeg without finding nothing on $PATH.
        env = self._subprocess_env()
        env["HF_TOKEN"] = settings.hf_token
        self.sink(ProgressEvent("diarisation", 0, "Detecting speakers"))
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=3600)
        duration = time.monotonic() - ts
        if proc.returncode != 0:
            message = tail_text(proc.stderr or proc.stdout)
            self.sink(
                WarningEvent(
                    f"Diarisation indisponible : {message}", code="diarisation_failed"
                )
            )
            append_app_log(f"engine_diarisation_failed rc={proc.returncode} error={message!r}")
            return (
                StepResult(
                    "diarisation",
                    False,
                    duration_seconds=duration,
                    error=message,
                ),
                [],
            )
        try:
            turns = parse_diarization_output(proc.stdout)
        except Exception as exc:
            self.sink(WarningEvent(f"Diarisation: sortie illisible : {exc}", code="diarisation_parse"))
            return (
                StepResult(
                    "diarisation",
                    False,
                    duration_seconds=duration,
                    error=str(exc),
                ),
                [],
            )
        # Pyannote emits sub-second turns on back-channels ("hm",
        # "ouais") that otherwise produce nonsensical speaker
        # switches mid-sentence. Fuse them into the surrounding turn
        # before propagating downstream.
        raw_turn_count = len(turns)
        turns = fuse_micro_turns(
            turns,
            min_duration=settings.min_speaker_turn_seconds,
        )
        fused_dropped = raw_turn_count - len(turns)
        speakers_seen = sorted({turn["speaker"] for turn in turns})
        self.sink(
            ProgressEvent(
                "diarisation",
                100,
                f"{len(speakers_seen)} speaker(s) detected",
            )
        )
        return (
            StepResult(
                "diarisation",
                True,
                duration_seconds=duration,
                metrics={
                    "speakers": len(speakers_seen),
                    "turns": len(turns),
                    "micro_turns_fused": fused_dropped,
                },
            ),
            turns,
        )

    # ------------------------------------------------------------------
    # Step 3bis — speaker recognition (voice profile match)
    # ------------------------------------------------------------------

    def _run_speaker_recognition(
        self,
        diar_audio_path: Path,
        segments: list[dict],
    ) -> dict[str, str]:
        """Match each SPEAKER_NN cluster against the stored voice
        profiles. Returns a mapping ``{cluster_label: friendly_name}``
        the orchestrator then applies as a rename so the rename
        sheet opens with the right names already filled in.

        Bails out cleanly when:
          * no profiles are enrolled yet (most common path on first
            run for a new user),
          * the embedding venv isn't reachable,
          * pyannote fails / segments don't yield enough audio.

        We surface a WarningEvent when something explicit went
        wrong; silent skips on "no profiles yet" so we don't spam
        the SwiftUI status bar with non-actionable noise.
        """
        # Lazy import — touching the DB here would force every test
        # of TranscriptionPipeline to provide a stub library DB.
        from .library import database

        try:
            db = database()
            raw_profiles = db.list_speaker_profiles()
        except Exception as exc:
            append_app_log(f"engine_speaker_recog_no_db error={exc!r}")
            return {}
        if not raw_profiles:
            return {}
        # PR T: drop stale rows (no embedding, sample_count=0) so they
        # can't pollute the matcher. The CVR-control library had
        # leftover ``Benjamin`` / ``Laurent`` rows from prior failed
        # enrolments — their embeddings were unusable but the rows
        # were still listed.
        profiles = filter_usable_profiles(raw_profiles)
        dropped = len(raw_profiles) - len(profiles)
        if dropped:
            append_app_log(
                f"engine_speaker_recog_filtered_stale "
                f"raw={len(raw_profiles)} usable={len(profiles)} "
                f"dropped={dropped}"
            )
        if not profiles:
            append_app_log(
                "engine_speaker_recog_no_usable_profiles "
                f"raw_count={len(raw_profiles)}"
            )
            return {}

        settings = self.request.transcription_settings
        venv_python = settings.venv_python_path
        if not venv_python or not Path(venv_python).exists():
            self.sink(
                WarningEvent(
                    "Reconnaissance des locuteurs ignorée : environnement Python introuvable.",
                    code="speaker_recog_no_venv",
                )
            )
            return {}

        cluster_segments = self._build_recognition_clusters(segments)
        if not cluster_segments:
            return {}

        clusters_payload = [
            {"label": label, "segments": segs}
            for label, segs in cluster_segments.items()
        ]
        ts = time.monotonic()
        self.sink(ProgressEvent("speaker_recog", 0, "Matching voices against known profiles"))
        cmd = build_embedding_extract_cmd(
            venv_python, str(diar_audio_path), clusters_payload
        )
        env = self._subprocess_env()
        env["HF_TOKEN"] = settings.hf_token
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,
            env=env,
        )
        if proc.returncode != 0:
            self.sink(
                WarningEvent(
                    "Reconnaissance des locuteurs indisponible : "
                    + tail_text(proc.stderr or proc.stdout),
                    code="speaker_recog_failed",
                )
            )
            append_app_log(
                f"engine_speaker_recog_failed rc={proc.returncode} "
                f"stderr={tail_text(proc.stderr)!r}"
            )
            return {}
        try:
            embeddings = parse_embedding_output(proc.stdout)
        except RuntimeError as exc:
            self.sink(
                WarningEvent(
                    f"Reconnaissance des locuteurs : sortie illisible : {exc}",
                    code="speaker_recog_parse",
                )
            )
            return {}

        recognized: dict[str, str] = {}
        used_names: set[str] = set()
        # Iterate in deterministic order so the "first match wins"
        # tie-break against duplicate-profile assignments stays
        # stable across runs.
        for label in sorted(cluster_segments.keys()):
            vectors = embeddings.get(label) or []
            if not vectors:
                continue
            centroid = aggregate_embeddings(vectors)
            # PR T diagnostic: log the top-3 profile candidates for
            # this cluster, regardless of whether any of them passes
            # the threshold. Lets us tell the difference between
            # "Clothilde at 0.79, Robin at 0.78" (a marginal pass we
            # should second-guess) and "Clothilde at 0.91, all others
            # below 0.4" (a confident match). The CVR audit had no
            # signal at all here — the log only said "best=Clothilde".
            all_scores = score_cluster_against_all_profiles(centroid, profiles)
            top_three = all_scores[:3]
            top_summary = ", ".join(
                f"{name}={score:.3f}" for name, score in top_three
            ) or "none"
            match = match_cluster_against_profiles(
                centroid, profiles, threshold=DEFAULT_MATCH_THRESHOLD
            )
            if not match or not match.profile_name:
                append_app_log(
                    f"engine_speaker_recog_cluster_unmatched "
                    f"cluster={label!r} threshold={DEFAULT_MATCH_THRESHOLD:.2f} "
                    f"top={top_summary}"
                )
                continue
            if match.profile_name in used_names:
                append_app_log(
                    f"engine_speaker_recog_cluster_collision "
                    f"cluster={label!r} winner={match.profile_name!r} "
                    f"score={match.similarity:.3f} already_claimed=true "
                    f"top={top_summary}"
                )
                continue
            used_names.add(match.profile_name)
            recognized[label] = match.profile_name
            append_app_log(
                f"engine_speaker_recog_cluster_matched "
                f"cluster={label!r} winner={match.profile_name!r} "
                f"score={match.similarity:.3f} top={top_summary}"
            )
        duration = time.monotonic() - ts
        self.sink(
            ProgressEvent(
                "speaker_recog",
                100,
                f"{len(recognized)} locuteur(s) reconnu(s) "
                f"({duration:.1f}s)",
            )
        )
        return recognized

    # PR T: window over which we weigh "who's dominating the
    # conversation start". The user usually opens the call ("Bonjour,
    # vous m'entendez ?") then lets the others speak — so the
    # *dominant* speaker in the first minute is a far more reliable
    # signal than the literal first-to-speak.
    _PRE_ATTRIBUTION_WINDOW_SECONDS = 60.0

    def _pre_attribute_current_user(
        self,
        segments: list[dict],
        *,
        already_recognized: dict[str, str],
    ) -> dict[str, str]:
        """Attribute the cluster dominating the first 60 seconds to
        the user named in ``current_user_name``.

        Used when voice matching couldn't pick the user out of the
        stored profiles (typical on a first run: ``sample_count=0``
        profile means no centroid, so the recognition step skips
        them). Without this, ``SPEAKER_00`` lingers as a placeholder
        in the transcript even when the user has explicitly told
        Réglages "I am Robin".

        PR T heuristic upgrade — was: "first cluster to emit any
        segment". The CVR-control run exposed the failure mode:
        Robin says ``Bonjour, vous m'entendez ?`` (3 s), then
        Vincent answers ``...pas de son...`` (12 s of struggling
        audio), then Robin resumes for the rest of the meeting. The
        old heuristic latched onto Vincent's cluster because his
        very-first segment started a fraction earlier when Whisper
        timestamped the silence-prefixed ``...pas de son...``
        — even though Robin overwhelmingly dominates speech time.
        Switching to "most cumulative speech time in [0, 60 s]"
        fixes this: the cluster that talks longest in the opening
        window is overwhelmingly the user on every call we audit.

        Rules:
        - Skip when ``current_user_name`` is blank.
        - Skip when no SPEAKER_NN clusters remain (everything's
          already named).
        - Skip when the user's name is already in
          ``already_recognized.values()`` — voice match wins, the
          heuristic just confirms.
        - Otherwise attribute the SPEAKER_NN cluster with the most
          cumulative speech time in [0, _PRE_ATTRIBUTION_WINDOW_SECONDS].

        Returns ``{cluster_label: current_user_name}`` to merge into
        the recognised map. Empty dict means no attribution.
        """
        name = (self.request.transcription_settings.current_user_name or "").strip()
        if not name:
            return {}
        existing_names = {v.strip().lower() for v in already_recognized.values() if v}
        if name.lower() in existing_names:
            return {}

        window_end = self._PRE_ATTRIBUTION_WINDOW_SECONDS
        speech_time_by_cluster: dict[str, float] = {}
        first_start_by_cluster: dict[str, float] = {}
        for seg in segments:
            label = (seg.get("speaker") or "").strip()
            if not label or not label.upper().startswith("SPEAKER_"):
                continue
            if label in already_recognized:
                # Voice match has already claimed this cluster — never
                # overwrite a centroid match with a heuristic guess.
                continue
            try:
                start = float(seg.get("start") or 0)
                end = float(seg.get("end") or 0)
            except (TypeError, ValueError):
                continue
            if end <= 0 or end <= start:
                continue
            # Clip the segment to the [0, window_end] window so a
            # 10-minute monologue counts only what falls inside the
            # opening minute.
            clipped_start = max(0.0, start)
            clipped_end = min(window_end, end)
            if clipped_end <= clipped_start:
                continue
            speech_time_by_cluster[label] = (
                speech_time_by_cluster.get(label, 0.0)
                + (clipped_end - clipped_start)
            )
            previous = first_start_by_cluster.get(label)
            if previous is None or clipped_start < previous:
                first_start_by_cluster[label] = clipped_start

        if not speech_time_by_cluster:
            return {}

        # Tie-break: prefer the cluster that started earlier in the
        # window. Keeps deterministic behaviour on short fixtures
        # where speech time is identical (the test we audited had
        # two 1-second segments — symmetric). Negate the
        # ``first_start`` so ``max()`` ranks "earlier" higher.
        winning_cluster, winning_seconds = max(
            speech_time_by_cluster.items(),
            key=lambda kv: (
                kv[1],
                -first_start_by_cluster.get(kv[0], float("inf")),
            ),
        )
        # Sanity floor: when ALL clusters speak < 1 s in the window
        # (e.g. mostly-silent meeting opening), the signal is too
        # weak to attribute confidently. Falls back to "no
        # attribution" rather than guessing wrong.
        if winning_seconds < 1.0:
            append_app_log(
                "engine_current_user_preattribution_skipped "
                f"reason=signal_too_weak winner={winning_cluster!r} "
                f"seconds={winning_seconds:.2f}"
            )
            return {}

        append_app_log(
            "engine_current_user_preattributed "
            f"cluster={winning_cluster!r} name={name!r} "
            f"window_seconds={window_end:.0f} "
            f"speech_in_window={winning_seconds:.2f} "
            f"distribution={dict(sorted(speech_time_by_cluster.items()))!r}"
        )
        return {winning_cluster: name}

    # Embedding-extraction tuning constants kept identical to the
    # ones the library uses for explicit re-recognition, so a job
    # run-time match and a post-hoc one converge on the same answer.
    _RECOGNITION_TURNS_PER_CLUSTER = 3
    _RECOGNITION_MIN_TURN_SECONDS = 2.0

    def _build_recognition_clusters(
        self, segments: list[dict]
    ) -> dict[str, list[dict]]:
        """Pick the longest 1-3 turns per SPEAKER_NN cluster — the
        embedding model wants ≥ 2 s of clean speech per sample, more
        is better, past 3 the centroid stops improving.
        """
        candidates: dict[str, list[dict]] = {}
        for segment in segments:
            label = (segment.get("speaker") or "").strip()
            if not label or not label.upper().startswith("SPEAKER_"):
                continue
            try:
                start = float(segment.get("start") or 0)
                end = float(segment.get("end") or 0)
            except (TypeError, ValueError):
                continue
            duration = end - start
            if duration < self._RECOGNITION_MIN_TURN_SECONDS:
                continue
            candidates.setdefault(label, []).append(
                {"start": start, "end": end, "duration": duration}
            )
        out: dict[str, list[dict]] = {}
        for label, turns in candidates.items():
            turns.sort(key=lambda t: t["duration"], reverse=True)
            out[label] = [
                {"start": t["start"], "end": t["end"]}
                for t in turns[: self._RECOGNITION_TURNS_PER_CLUSTER]
            ]
        return out

    # ------------------------------------------------------------------
    # Step 5 — LLM post-process
    # ------------------------------------------------------------------

    def _maybe_enrich_glossary_from_web(self, transcript_text: str) -> None:
        """Optional web enrichment pass that confirms proper-noun
        candidates against DuckDuckGo and adds the confirmed ones
        to the glossary.

        Gated by ``transcription_settings.web_enrichment_enabled``
        (forced off in every preset today; only the ``custom``
        preset honours the user-set value). Best-effort: any
        exception is swallowed and the pipeline continues with the
        unenriched glossary.

        The Odoo recursive context pack already mutates
        ``request.glossary_terms`` once per job — this pass appends
        on top, so the two enrichment sources compose naturally.
        """
        settings = self.request.transcription_settings
        if not settings.web_enrichment_enabled:
            return
        if not transcript_text:
            return
        try:
            from web_context import enrich_glossary_via_web
        except Exception:
            return
        ts = time.monotonic()
        self.sink(
            ProgressEvent(
                "web_enrichment",
                0,
                "Confirmation web des entités candidates",
            )
        )
        try:
            results = enrich_glossary_via_web(
                transcript_text,
                list(self.request.glossary_terms),
            )
        except Exception as exc:
            append_app_log(f"engine_web_enrichment_failed error={exc!r}")
            self.sink(
                WarningEvent(
                    f"Enrichissement web : indisponible ({exc})",
                    code="web_enrichment_failed",
                )
            )
            return
        if not results:
            self.sink(
                ProgressEvent(
                    "web_enrichment",
                    100,
                    "Aucune entité supplémentaire confirmée",
                )
            )
            return
        # Append to the glossary in place, dedup against existing
        # terms (case-insensitive).
        existing = {t.strip().lower() for t in self.request.glossary_terms if t.strip()}
        added = 0
        for result in results:
            term = (result.confirmed_term or result.candidate or "").strip()
            if not term or term.lower() in existing:
                continue
            self.request.glossary_terms = [*self.request.glossary_terms, term]
            existing.add(term.lower())
            added += 1
        append_app_log(
            f"engine_web_enrichment added={added} elapsed={time.monotonic() - ts:.1f}s"
        )
        self.sink(
            ProgressEvent(
                "web_enrichment",
                100,
                f"Enrichissement web : +{added} terme(s) confirmé(s)",
            )
        )

    def _run_llm_post(
        self,
        analysis_segments: list[dict],
        transcript_path: Path,
    ) -> StepResult | None:
        """Title, speakers, technical terms (short JSON call) +
        corrections / doubts (markdown call). Both are tolerant of
        partial output — if either fails we still keep whatever the
        other produced.
        """
        settings = self.request.transcription_settings
        venv_python = settings.venv_python_path
        if not venv_python or not Path(venv_python).exists():
            return None

        # Render to plain text on the analysis timeline so the LLM
        # sees timestamps + speaker tags.
        analysis_text = render_segments_with_speakers(analysis_segments, "txt")
        analysis_path = transcript_path.parent / "transcript_for_local_analysis.txt"
        analysis_path.write_text(analysis_text, encoding="utf-8")
        # Web enrichment fires HERE rather than earlier so it sees
        # the cleanest transcript text we have (post-multipass,
        # post-diarisation, post-PR-A smoothing). Mutates
        # ``request.glossary_terms`` in place — same pattern as the
        # Odoo context pack — so the LLM correction pass picks up
        # the newly-confirmed entities automatically.
        self._maybe_enrich_glossary_from_web(analysis_text)
        glossary_text = "\n".join(
            [*self.request.glossary_terms, *self.request.technical_terms]
        )

        # Optional Odoo chatter context. Fetched once per job, here
        # rather than at Run Setup time so the user's UI never
        # blocks on a slow Odoo. Failures are silent — the LLM step
        # then proceeds without the bonus context.
        odoo_context_blob = self._fetch_odoo_context_blob()

        ts = time.monotonic()
        payload: dict[str, Any] = {}

        # ---- title + speakers (short JSON) ---------------------------
        title_cmd = build_llm_title_cmd(
            venv_python, settings.text_llm_model, str(analysis_path), glossary_text
        )
        self.sink(ProgressEvent("llm_title", 0, "LLM: title + speakers"))
        title_proc = subprocess.run(
            title_cmd,
            capture_output=True,
            text=True,
            timeout=900,
            env=self._subprocess_env(),
        )
        if title_proc.returncode == 0:
            try:
                title_payload = parse_llm_title_speakers(title_proc.stdout)
                payload.update(title_payload)
            except Exception as exc:
                append_app_log(f"engine_llm_title_parse_failed error={exc!r}")
        else:
            append_app_log(
                f"engine_llm_title_failed rc={title_proc.returncode} "
                f"stderr={tail_text(title_proc.stderr)!r}"
            )

        # PR AF: Mistral 7B local routinely ignores the
        # "Société - Sujet" instruction from PR AA. As a fallback,
        # resolve the company name from richer sources (Odoo
        # context, meeting metadata, speaker overrides) and prepend
        # it ourselves if missing. The LLM keeps its job (deciding
        # the topic part) but we no longer rely on it for the
        # structural prefix.
        existing_title = str(payload.get("title") or "").strip()
        if existing_title:
            prefixed = self._apply_title_company_prefix(existing_title)
            if prefixed != existing_title:
                payload["title"] = prefixed
                append_app_log(
                    f"engine_llm_title_company_prefix "
                    f"original={existing_title!r} prefixed={prefixed!r}"
                )

        # ---- corrections + doubts (markdown) -------------------------
        # The embedded LLM script truncates its input at 30 000 chars,
        # which silently drops the second half of any meeting beyond
        # ~30 minutes. We chunk the rendered transcript ourselves with
        # an overlap so every minute of the meeting gets a correction
        # pass, and we dedup across overlapping chunks so the user
        # never sees the same fix twice.
        chunks = chunk_transcript_for_llm(analysis_text)
        all_corrections: list[dict] = []
        all_uncertain: list[dict] = []
        for chunk in chunks:
            chunk_path = (
                analysis_path
                if chunk.total == 1
                else analysis_path.with_name(
                    f"{analysis_path.stem}.chunk{chunk.index + 1:02d}.txt"
                )
            )
            if chunk.total > 1:
                chunk_path.write_text(chunk.text, encoding="utf-8")
            corr_cmd = build_llm_corrections_cmd(
                venv_python,
                settings.text_llm_model,
                str(chunk_path),
                glossary_text,
                odoo_context_blob,
            )
            # Linear progress across the chunks so the SwiftUI bar
            # moves on long meetings instead of sitting at 50 %.
            pct = 50 + int(50 * chunk.index / max(chunk.total, 1))
            self.sink(
                ProgressEvent(
                    "llm_corrections",
                    pct,
                    f"LLM: corrections + doubts ({chunk.index + 1}/{chunk.total})",
                )
            )
            corr_proc = subprocess.run(
                corr_cmd,
                capture_output=True,
                text=True,
                timeout=1800,
                env=self._subprocess_env(),
            )
            if corr_proc.returncode == 0:
                try:
                    corr_payload = parse_llm_corrections_markdown(corr_proc.stdout)
                    all_corrections.extend(corr_payload.get("corrections") or [])
                    all_uncertain.extend(corr_payload.get("uncertain_passages") or [])
                except Exception as exc:
                    append_app_log(
                        f"engine_llm_corrections_parse_failed chunk={chunk.index} "
                        f"error={exc!r}"
                    )
            else:
                append_app_log(
                    f"engine_llm_corrections_failed chunk={chunk.index} "
                    f"rc={corr_proc.returncode} "
                    f"stderr={tail_text(corr_proc.stderr)!r}"
                )

        payload["corrections"] = dedupe_corrections(all_corrections)
        payload["uncertain_passages"] = _filter_tautological_doubts(
            dedupe_corrections(all_uncertain)
        )

        self.sink(ProgressEvent("llm_corrections", 100, "LLM enhancement done"))
        self._llm_payload = payload
        duration = time.monotonic() - ts
        if not payload:
            return None
        return StepResult(
            "llm_post",
            True,
            model=settings.text_llm_model,
            duration_seconds=duration,
            metrics={
                "corrections": len(payload.get("corrections") or []),
                "uncertain": len(payload.get("uncertain_passages") or []),
                "speakers": sum(
                    1 for v in (payload.get("speakers") or {}).values() if v
                ),
            },
        )

    # ------------------------------------------------------------------
    # Step 5ter — multimodal audio recheck (Qwen2-Audio via mlx_vlm)
    # ------------------------------------------------------------------

    _AUDIO_RECHECK_MAX_PASSAGES = 10
    _AUDIO_RECHECK_PRE_SECONDS = 5.0
    _AUDIO_RECHECK_POST_SECONDS = 10.0
    _AUDIO_RECHECK_TIMEOUT_SECONDS = 600

    # PR AC: map an ``audio_llm_model`` model path to the
    # ``mlx_vlm.models`` submodule that must be importable for the
    # actual ``load(model_path)`` call to succeed. Without this map
    # the old probe was useless — it OK'd ``import mlx_vlm`` then
    # the embedded multimodal script crashed with "Model type
    # qwen2_audio not supported" on every uncertain passage,
    # leaving a trail of ``engine_audio_recheck_failed`` in app_log.
    _MLX_VLM_MODEL_SLUG_HINTS = (
        ("qwen2-audio", "qwen2_audio"),
        ("qwen2_audio", "qwen2_audio"),
    )

    @classmethod
    def _mlx_vlm_submodule_for(cls, model_path: str) -> str:
        """Return the ``mlx_vlm.models.<slug>`` submodule the given
        model path requires. Empty string when we don't have a hint —
        the caller skips the submodule check in that case (we
        degrade to the legacy "just ``import mlx_vlm``" probe).
        """
        lowered = (model_path or "").lower()
        for needle, slug in cls._MLX_VLM_MODEL_SLUG_HINTS:
            if needle in lowered:
                return slug
        return ""

    def _ensure_mlx_vlm_available(self) -> bool:
        """Probe whether ``mlx_vlm`` AND the specific model submodule
        for the configured ``audio_llm_model`` are importable.

        Cached: the probe runs at most once per pipeline. Unlike the
        legacy ``video_compactor`` we *don't* auto-install — the new
        engine pins its dependencies via ``managed_venv`` and a
        surprise ``pip install`` from inside the runner has historically
        deadlocked the SwiftUI progress bar. If the probe fails we
        emit a single actionable WarningEvent (telling the user
        whether to install ``mlx-vlm`` or upgrade it) and skip the
        recheck step for the whole pipeline.

        PR AC: previously the probe only checked ``import mlx_vlm``
        which succeeded on the bundled venv. The actual
        ``load(model_path)`` then failed with "Model type qwen2_audio
        not supported. Error: No module named
        'mlx_vlm.models.qwen2_audio'" on every recheck attempt.
        Now we probe the specific submodule the model needs.
        """
        if self._mlx_vlm_available is not None:
            return self._mlx_vlm_available
        venv_python = self.request.transcription_settings.venv_python_path
        if not venv_python or not Path(venv_python).exists():
            self._mlx_vlm_available = False
            return False

        model_path = self.request.transcription_settings.audio_llm_model or ""
        submodule_slug = self._mlx_vlm_submodule_for(model_path)
        # Build a single probe script that returns:
        #   exit 0 → fully available
        #   exit 1 → ``mlx_vlm`` itself missing
        #   exit 2 → ``mlx_vlm`` present but the model submodule
        #            missing (the canonical CVR/Caste failure mode)
        probe_script = (
            "import importlib, sys\n"
            "try:\n"
            "    import mlx_vlm\n"
            "except Exception:\n"
            "    sys.exit(1)\n"
        )
        if submodule_slug:
            probe_script += (
                f"try:\n"
                f"    importlib.import_module('mlx_vlm.models.{submodule_slug}')\n"
                f"except Exception:\n"
                f"    sys.exit(2)\n"
            )
        probe_script += "sys.exit(0)\n"

        try:
            probe = subprocess.run(
                [venv_python, "-c", probe_script],
                capture_output=True,
                text=True,
                timeout=30,
                env=self._subprocess_env(),
            )
        except Exception as exc:  # pragma: no cover — defensive
            append_app_log(f"engine_mlx_vlm_probe_error err={exc!r}")
            self._mlx_vlm_available = False
            return False

        rc = probe.returncode
        if rc == 0:
            self._mlx_vlm_available = True
            return True

        # Distinguish the two failure modes so the WarningEvent
        # gives the user something actionable to do.
        self._mlx_vlm_available = False
        if rc == 1:
            message = (
                "Réécoute IA ignorée : ``mlx-vlm`` n'est pas installé "
                "dans l'environnement Python géré. "
                "Désactive la réécoute IA dans Réglages ou installe "
                "``mlx-vlm`` dans la venv pour activer Qwen2-Audio."
            )
            log_reason = "mlx_vlm_missing"
        elif rc == 2:
            message = (
                f"Réécoute IA ignorée : ``mlx-vlm`` est installé mais "
                f"ne contient pas le support du modèle "
                f"``{submodule_slug}`` ({model_path}). "
                f"Mets à jour ``mlx-vlm`` dans la venv "
                f"(``pip install -U mlx-vlm``) ou choisis un modèle "
                f"audio LLM supporté dans Réglages."
            )
            log_reason = f"model_submodule_missing slug={submodule_slug}"
        else:
            message = (
                "Réécoute IA ignorée : la sonde mlx-vlm a échoué "
                f"(rc={rc}). Voir app_log pour le détail."
            )
            log_reason = f"probe_failed_unknown_rc rc={rc}"

        self.sink(
            WarningEvent(
                message,
                code="audio_recheck_mlx_vlm_unavailable",
            )
        )
        append_app_log(
            f"engine_mlx_vlm_unavailable reason={log_reason} "
            f"rc={rc} model_path={model_path!r} "
            f"stderr={tail_text(probe.stderr)!r}"
        )
        return False

    def _run_audio_recheck(
        self, whisper_wav: Path, workspace: Path
    ) -> StepResult | None:
        """
        Run a short multimodal pass (Qwen2-Audio via ``mlx_vlm``) on
        every passage the text LLM flagged as uncertain. Each
        passage is re-listened to in isolation (clip window = pre/post
        seconds around the timestamp) so the audio model can suggest
        a transcription that lines up with the meeting's vocabulary.

        Returns ``None`` (no StepResult appended) when the feature is
        off, gated by a missing dependency, or there's nothing to
        recheck — so the SwiftUI step list only grows when actual
        work was done.

        Mutates ``self._llm_payload["uncertain_passages"]`` in place:
        each processed passage gains a ``suggestion`` (and a
        ``clip_path`` for the review markdown to optionally render
        an audio link). This is the same contract the legacy
        ``_run_clip_rechecks`` produced.
        """
        settings = self.request.transcription_settings
        if not getattr(settings, "audio_recheck_enabled", False):
            return None
        passages = list(self._llm_payload.get("uncertain_passages") or [])
        if not passages:
            return None
        venv_python = settings.venv_python_path
        if not venv_python or not Path(venv_python).exists():
            append_app_log(
                "engine_audio_recheck_skipped reason=no_venv_python"
            )
            return None
        if not self._ensure_mlx_vlm_available():
            self.sink(
                WarningEvent(
                    "Réécoute IA ignorée (mlx-vlm indisponible).",
                    code="audio_recheck_mlx_vlm_missing",
                )
            )
            return None
        if not whisper_wav.exists():
            return None

        clip_dir = workspace / "audio_recheck_clips"
        clip_dir.mkdir(parents=True, exist_ok=True)

        glossary_text = "\n".join(
            [
                *(self.request.glossary_terms or []),
                *(self.request.technical_terms or []),
            ]
        )

        ts = time.monotonic()
        total = min(len(passages), self._AUDIO_RECHECK_MAX_PASSAGES)
        self.sink(
            ProgressEvent(
                "audio_recheck",
                0,
                f"Réécoute IA des passages douteux ({total}× Qwen2-Audio)",
            )
        )

        suggestions = 0
        failures = 0
        ffmpeg_path = self.request.compression_settings.ffmpeg_path or "ffmpeg"

        for index, passage in enumerate(passages[:total], start=1):
            if not isinstance(passage, dict):
                continue
            timestamp = str(passage.get("timestamp") or "").strip()
            if not timestamp:
                continue
            try:
                center = timestamp_text_to_seconds(timestamp)
            except Exception:
                continue
            if center <= 0:
                continue
            start = max(0.0, center - self._AUDIO_RECHECK_PRE_SECONDS)
            end = center + self._AUDIO_RECHECK_POST_SECONDS

            clip_wav = clip_dir / f"clip_{index:02d}.wav"
            extract_cmd = build_audio_extract_cmd(
                ffmpeg_path,
                str(whisper_wav),
                str(clip_wav),
                # The recheck clip is short (≤ 15 s); the heavier
                # enhancement filters add little here and can mask the
                # very phoneme cue the user wants Qwen2-Audio to hear.
                speech_enhance=False,
                ss=format_seconds_for_clip(start),
                to=format_seconds_for_clip(end),
            )
            try:
                subprocess.run(
                    extract_cmd,
                    capture_output=True,
                    timeout=60,
                    env=self._subprocess_env(),
                )
            except Exception as exc:  # pragma: no cover — defensive
                append_app_log(
                    f"engine_audio_recheck_clip_extract_failed "
                    f"timestamp={timestamp!r} err={exc!r}"
                )
                failures += 1
                continue
            if not clip_wav.exists():
                failures += 1
                continue

            passage["clip_path"] = str(clip_wav)

            prompt = build_multimodal_recheck_prompt(
                whisper_text=str(passage.get("text") or ""),
                reason=str(passage.get("reason") or ""),
                glossary=glossary_text,
            )

            cmd = build_multimodal_audio_cmd(
                venv_python_path=venv_python,
                model_path=settings.audio_llm_model,
                audio_path=str(clip_wav),
                prompt=prompt,
            )
            self.sink(
                ProgressEvent(
                    "audio_recheck",
                    int(100 * index / max(total, 1)),
                    f"Réécoute IA {index}/{total} ({timestamp})",
                )
            )
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self._AUDIO_RECHECK_TIMEOUT_SECONDS,
                    env=self._subprocess_env(),
                )
            except subprocess.TimeoutExpired:
                append_app_log(
                    f"engine_audio_recheck_timeout timestamp={timestamp!r}"
                )
                failures += 1
                continue
            if proc.returncode != 0:
                append_app_log(
                    f"engine_audio_recheck_failed timestamp={timestamp!r} "
                    f"rc={proc.returncode} stderr={tail_text(proc.stderr)!r}"
                )
                failures += 1
                continue

            payload = parse_multimodal_audio_response(proc.stdout)
            suggestion = str(payload.get("suggestion") or "").strip()
            if not suggestion:
                # ``mlx_vlm`` returned either an error JSON or
                # something we couldn't parse — keep going but log.
                err = str(payload.get("error") or "").strip()
                append_app_log(
                    f"engine_audio_recheck_no_suggestion "
                    f"timestamp={timestamp!r} err={err!r}"
                )
                failures += 1
                continue

            passage["suggestion"] = suggestion
            suggestions += 1

        self.sink(ProgressEvent("audio_recheck", 100, "Réécoute IA terminée"))
        duration = time.monotonic() - ts
        return StepResult(
            "audio_recheck",
            True,
            model=settings.audio_llm_model,
            duration_seconds=duration,
            metrics={
                "rechecked": total,
                "suggestions": suggestions,
                "failures": failures,
            },
        )

    # ------------------------------------------------------------------
    # Speaker rename + transcript writers
    # ------------------------------------------------------------------

    def _apply_speaker_renames(
        self, segments: list[dict], speakers: dict[str, str]
    ) -> list[dict]:
        if not speakers:
            return segments
        out: list[dict] = []
        for seg in segments:
            new_seg = dict(seg)
            spk = (new_seg.get("speaker") or "").strip()
            if spk in speakers and speakers[spk]:
                new_seg["speaker"] = speakers[spk]
            out.append(new_seg)
        return out

    def _write_transcript(
        self,
        segments: list[dict],
        workspace: Path,
        force_path: Path | None = None,
    ) -> Path:
        settings = self.request.transcription_settings
        if force_path is not None:
            transcript_path = force_path
        else:
            transcript_path = Path(
                default_transcript_path(
                    self.request.source_path,
                    str(workspace),
                    settings.suffix,
                    settings.output_format,
                )
            )
        # Pick the renderer based on whether the segments have speaker
        # tags. After diarisation they all do.
        has_speakers = any(seg.get("speaker") for seg in segments)
        if has_speakers:
            rendered = render_segments_with_speakers(segments, settings.output_format)
        else:
            rendered = render_segments_plain(segments, settings.output_format)
        # Reconstruct letter-by-letter spellings (``N O U V I A L E``
        # → ``nouviale``) and verbalised punctuation in email/URL
        # contexts (``arobase`` → ``@``, ``point`` → ``.``). Whisper
        # transcribed the Caste call's email as ``n-o-u-v-i-a-l-e
        # a-v-a-z-cast.fr c-a-s-t-e.fr`` — the new helpers turn that
        # noise back into ``nouviale@caste.fr``-shaped text.
        rendered = reconstruct_spelled_text(rendered)
        # Final capitalization sweep: enforce the canonical
        # spelling of every glossary term across the transcript so
        # ``Quadra`` doesn't co-exist with ``quadra``, ``Excel``
        # with ``excel``. The mutation observed on the Cozynergy
        # and Caste outputs.
        rendered = _apply_glossary_capitalization(
            rendered, list(self.request.glossary_terms)
        )
        transcript_path.write_text(rendered, encoding="utf-8")
        apply_meeting_date_to_artifact(self.request, transcript_path)
        self.sink(
            ArtifactEvent("transcript", str(transcript_path), model=settings.model)
        )
        return transcript_path

    def _has_quality_output(self) -> bool:
        return bool(
            self._glossary_subs
            or self._repass_replaced
            or self._vad_manifest
            or self._llm_payload.get("corrections")
            or self._llm_payload.get("uncertain_passages")
            or self._llm_payload.get("speakers")
            or self._llm_payload.get("technical_terms")
            or self._llm_payload.get("title")
            # PR AB: a run that ONLY produced decoder loops (no LLM,
            # no VAD, nothing else worth surfacing) still deserves
            # the review markdown — that's exactly when the user
            # most needs to see the "Zones perdues" alert.
            or self._dropped_loops
            # PR AD: recovered loops are also worth surfacing as a
            # positive signal (engine fixed an issue silently).
            or self._recovered_loops
        )

    def _write_enhanced_transcript(
        self, transcript_path: Path, segments: list[dict]
    ) -> Path | None:
        """Render an ``- améliorée`` file that actually carries the
        LLM corrections.

        Before this change, ``améliorée.txt`` was byte-identical to
        the base transcript: the review report listed corrections,
        but no file ever applied them, forcing the user to copy/paste
        edits manually. The substitutions are now applied here through
        the safe ``apply_llm_corrections_to_text`` (confidence +
        Levenshtein + literal-presence guardrails). The base file
        stays untouched so the user can always diff the two and audit
        what the LLM changed.
        """
        if not self._has_quality_output():
            return None
        settings = self.request.transcription_settings
        enhanced_path = transcript_path.with_name(
            f"{transcript_path.stem} améliorée{transcript_path.suffix}"
        )
        has_speakers = any(seg.get("speaker") for seg in segments)
        if has_speakers:
            rendered = render_segments_with_speakers(segments, settings.output_format)
        else:
            rendered = render_segments_plain(segments, settings.output_format)

        # Apply the LLM corrections to the rendered text. We skip the
        # heavier substitution path when there's nothing to do — the
        # function would short-circuit anyway, but bailing here keeps
        # the audit list clean ("no corrections were even attempted").
        corrections = self._llm_payload.get("corrections") or []
        if corrections:
            # Pass the glossary so the ``betrays_glossary`` guardrail
            # can refuse corrections that move the text away from a
            # term the user explicitly added (e.g. ``au doubs`` →
            # ``au doute`` when the glossary has ``Odoo``).
            glossary_for_correction = list(self.request.glossary_terms) + list(
                self.request.technical_terms
            )
            outcome = apply_llm_corrections_to_text(
                rendered,
                corrections,
                glossary_terms=glossary_for_correction,
            )
            rendered = outcome.text
            self._llm_corrections_applied = outcome.applied
            self._llm_corrections_rejected = outcome.rejected

        # Same letter-spelling + spoken-punctuation reconstruction
        # as the base transcript. Idempotent — re-running over an
        # already-reconstructed text is a no-op.
        rendered = reconstruct_spelled_text(rendered)
        # Final capitalization sweep, same as the base transcript.
        # The LLM corrections sometimes touch case (e.g. fixing
        # ``au doute`` would leave a lowercase remnant); enforcing
        # glossary canonical case here covers that path too.
        rendered = _apply_glossary_capitalization(
            rendered, list(self.request.glossary_terms)
        )

        enhanced_path.write_text(rendered, encoding="utf-8")
        apply_meeting_date_to_artifact(self.request, enhanced_path)
        self.sink(
            ArtifactEvent(
                "enhanced_transcript",
                str(enhanced_path),
                model=settings.text_llm_model,
            )
        )
        return enhanced_path

    def _write_review_markdown(
        self, transcript_path: Path, segments: list[dict]
    ) -> Path | None:
        """Produce ``<stem> - à vérifier.md`` summarising every
        automatic edit so the user can audit the pipeline.

        We only write the file when there's *something* to surface.
        Returns the written path, or None when nothing was worth a
        report.
        """
        if not self._has_quality_output():
            return None

        review_path = transcript_path.with_name(f"{transcript_path.stem} - à vérifier.md")
        lines = [
            "# Vérification de transcription",
            "",
            "Synthèse automatique des étapes appliquées et des points à relire.",
            "",
        ]

        if self._llm_payload.get("title"):
            lines += [
                "## Titre proposé",
                "",
                f"> {self._llm_payload['title']}",
                "",
            ]

        speakers = {k: v for k, v in (self._llm_payload.get("speakers") or {}).items() if v}
        if speakers:
            lines += ["## Interlocuteurs identifiés", ""]
            for sp, name in sorted(speakers.items()):
                lines.append(f"- `{sp}` → **{name}**")
            lines.append("")

        terms = self._llm_payload.get("technical_terms") or []
        if terms:
            lines += ["## Termes techniques détectés", ""]
            for t in terms[:30]:
                lines.append(f"- {t}")
            lines.append("")

        if self._vad_manifest:
            kept = sum(
                float(s.get("trim_end") or 0) - float(s.get("trim_start") or 0)
                for s in self._vad_manifest
            )
            lines += [
                "## VAD",
                "",
                f"{len(self._vad_manifest)} zones de parole conservées ({kept:.0f} s).",
                "",
            ]

        # PR AD: positive signal first — when the recovery step
        # clawed back lost zones, show that win before any
        # remaining alert. Builds confidence that the engine is
        # actively repairing the rough edges.
        if self._recovered_loops:
            recovered_seconds = sum(
                max(0.0, l.end - l.start) for l in self._recovered_loops
            )
            lines += [
                "## ✓ Zones récupérées",
                "",
                f"**{len(self._recovered_loops)} zone(s) ({recovered_seconds:.0f}s)** "
                f"où Whisper avait initialement bouclé ont été ré-transcrites "
                f"avec succès en seconde passe. Leur contenu est dans la "
                f"transcription.",
                "",
            ]

        # PR AB: surface decoder-loop drops AT THE TOP of the
        # report (right after VAD, before any text edits) so the
        # user sees the gravest signal first. Long meetings can
        # silently lose 60-80 % of content to Whisper hallucination
        # loops — when that happens, the user needs to know BEFORE
        # they trust the rest of the transcript.
        if self._dropped_loops or self._audio_seconds > 0:
            total_dropped = sum(loop.dropped for loop in self._dropped_loops)
            total_lost_s = sum(
                max(0.0, loop.end - loop.start) for loop in self._dropped_loops
            )
            kept_s = max(0.0, self._audio_seconds - total_lost_s)
            coverage_pct = (
                100.0 * kept_s / self._audio_seconds
                if self._audio_seconds > 0
                else 0.0
            )

            # The header text depends on the severity. ≥ 20 % loss
            # gets a red-alert framing; < 20 % gets a softer warning;
            # nothing dropped gets a friendly "✓" coverage line.
            if not self._dropped_loops:
                lines += [
                    "## Couverture audio",
                    "",
                    f"✓ Tout l'audio de la réunion est dans la transcription "
                    f"({self._audio_seconds:.0f} s).",
                    "",
                ]
            else:
                if coverage_pct < 80.0:
                    header = "## ⚠️ Zones perdues — relecture impossible"
                else:
                    header = "## Zones perdues"
                lines += [
                    header,
                    "",
                    f"**{total_lost_s:.0f} s d'audio perdues sur {self._audio_seconds:.0f} s** "
                    f"({100.0 - coverage_pct:.0f}% du fichier). "
                    f"La transcription ne couvre que **{coverage_pct:.0f} %** de la réunion.",
                    "",
                    "Cause : Whisper s'est mis à répéter une même phrase pendant "
                    "plusieurs minutes (boucle de décodage). Le moteur a supprimé "
                    "ces segments pour ne pas polluer le texte, mais le contenu "
                    "réel parlé pendant ces zones n'est PAS récupérable depuis "
                    "cette transcription.",
                    "",
                    f"**{len(self._dropped_loops)} zone(s) perdue(s) "
                    f"({total_dropped} segments droppés au total) :**",
                    "",
                ]
                # Show worst-first so the most impactful loss is at the top.
                ordered = sorted(
                    self._dropped_loops,
                    key=lambda l: (l.end - l.start),
                    reverse=True,
                )
                for loop in ordered[:10]:
                    start_min = int(loop.start // 60)
                    start_sec = int(loop.start % 60)
                    end_min = int(loop.end // 60)
                    end_sec = int(loop.end % 60)
                    duration_s = max(0.0, loop.end - loop.start)
                    snippet = (loop.text or "").strip()
                    if len(snippet) > 70:
                        snippet = snippet[:70] + "…"
                    lines.append(
                        f"- `{start_min:02d}:{start_sec:02d}` → "
                        f"`{end_min:02d}:{end_sec:02d}` "
                        f"({duration_s:.0f}s, {loop.dropped} segments) "
                        f": « {snippet} »"
                    )
                if len(self._dropped_loops) > 10:
                    lines.append(
                        f"- … et {len(self._dropped_loops) - 10} autre(s)."
                    )
                lines += [
                    "",
                    "Pour récupérer ces zones, ré-enregistrez la conversation "
                    "ou re-Whisperez manuellement le range audio en question.",
                    "",
                ]

        if self._repass_replaced:
            lines += [
                "## Repasse qualité maximale",
                "",
                f"{self._repass_replaced} segment(s) re-transcrit(s) avec "
                "`whisper-large-v3` après détection de faible confiance.",
                "",
            ]

        if self._glossary_subs:
            lines += ["## Vocabulaire métier (corrigé automatiquement)", ""]
            for sub in self._glossary_subs[:40]:
                ts = (
                    f"{int(sub.timestamp_seconds // 60):02d}:"
                    f"{int(sub.timestamp_seconds % 60):02d}"
                    if sub.timestamp_seconds is not None
                    else "—"
                )
                lines.append(
                    f"- `{ts}` `{sub.original}` → `{sub.replacement}` "
                    f"({sub.method}, confiance {sub.confidence:.2f})"
                )
            lines.append("")

        # Report applied corrections vs rejected ones in two sections
        # so the user can see (a) what landed in ``améliorée``, and
        # (b) what we deliberately refused, with the reason. Falling
        # back to the raw ``corrections`` list keeps backwards
        # compatibility for older runs that pre-date the apply path.
        applied = self._llm_corrections_applied
        rejected = self._llm_corrections_rejected
        if applied:
            lines += ["## Corrections LLM appliquées", ""]
            for c in applied[:40]:
                occ = f" ×{c.occurrences}" if c.occurrences > 1 else ""
                reason = f" — {c.reason}" if c.reason else ""
                lines.append(
                    f"- `{c.timestamp or '—'}` `{c.original}` → "
                    f"`{c.replacement}`{occ}{reason}"
                )
            lines.append("")
        if rejected:
            lines += [
                "## Corrections LLM refusées",
                "",
                "Refusées par garde-fou pour ne pas dénaturer le texte ou "
                "rejouer un quote inexistant.",
                "",
            ]
            reason_label = {
                "not_found": "quote absente du texte",
                "too_distant": "écart trop large (réécriture)",
                "low_confidence": "confiance trop basse",
                "empty": "champ vide",
            }
            for r in rejected[:40]:
                label = reason_label.get(r.reason, r.reason)
                lines.append(
                    f"- `{r.timestamp or '—'}` `{r.original}` → "
                    f"`{r.replacement}` ({label})"
                )
            lines.append("")
        if not applied and not rejected:
            raw_corrections = self._llm_payload.get("corrections") or []
            if raw_corrections:
                lines += ["## Corrections proposées par le LLM", ""]
                for c in raw_corrections[:25]:
                    lines.append(
                        f"- `{c.get('timestamp', '—')}` `{c.get('original')}` → "
                        f"`{c.get('replacement')}` "
                        f"(confiance {float(c.get('confidence', 0)):.2f}) — "
                        f"{c.get('reason', 'contexte')}"
                    )
                lines.append("")

        doubts = self._llm_payload.get("uncertain_passages") or []
        if doubts:
            lines += ["## Passages douteux", ""]
            for d in doubts[:25]:
                lines.append(
                    f"- `{d.get('timestamp', '—')}` {d.get('text', '')} "
                    f"_(raison : {d.get('reason', '—')})_"
                )
                # PR F: surface the multimodal recheck suggestion when
                # the audio model produced one. Rendered as an indented
                # bullet so the relationship to the parent doubt stays
                # visible at a glance — the user can scan the column
                # for "📣" markers and judge in seconds whether
                # Qwen2-Audio caught what Whisper missed.
                suggestion = str(d.get("suggestion") or "").strip()
                if suggestion:
                    lines.append(
                        f"  - 📣 Réécoute IA suggère : {suggestion}"
                    )
            lines.append("")

        review_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        apply_meeting_date_to_artifact(self.request, review_path)
        self.sink(ArtifactEvent("review", str(review_path)))
        return review_path


# ----------------------------------------------------------------------
# Compression pipeline (unchanged logic, friendlier error)
# ----------------------------------------------------------------------


class CompressionPipeline:
    def __init__(self, request: JobRequest, sink: EventSink):
        self.request = request
        self.sink = sink

    def run(self) -> StepResult:
        ts = time.monotonic()
        settings = self.request.compression_settings
        if self.request.workspace_dir:
            out_dir = Path(self.request.workspace_dir)
        else:
            out_dir = job_workspace_dir(self.request)
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = default_out_path(self.request.source_path, str(out_dir), "_compressed")
        append_app_log(
            "engine_compress_start "
            f"source={self.request.source_path!r} output={output_path!r} "
            f"trim_enabled={settings.trim_enabled!r} "
            f"trim_start={settings.trim_start!r} trim_end={settings.trim_end!r}"
        )
        cmd = build_ffmpeg_cmd(
            settings.ffmpeg_path or "ffmpeg",
            self.request.source_path,
            output_path,
            crf=settings.crf,
            resolution=settings.resolution,
            fps=settings.fps,
            audio_bitrate=settings.audio_bitrate,
            preset=settings.preset,
            speech_enhance=settings.speech_enhance,
            mono_audio=settings.mono_audio,
            ss=settings.trim_start if settings.trim_enabled else None,
            to=settings.trim_end
            if settings.trim_enabled and settings.trim_end != "00:00:00"
            else None,
            audio_only=is_audio_only_path(self.request.source_path),
            creation_time=ffmpeg_creation_time_from_request(self.request),
        )
        self.sink(ProgressEvent("compression", 0, "Running FFmpeg"))
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=subprocess_env_for_request(self.request),
        )
        duration = time.monotonic() - ts
        if proc.returncode != 0 or not Path(output_path).exists():
            raw = tail_text(proc.stderr or proc.stdout)
            message = _friendly_ffmpeg_error(raw, cmd[0])
            append_app_log(f"engine_compress_failed rc={proc.returncode} error={raw!r}")
            return StepResult("compression", False, duration_seconds=duration, error=message)
        append_app_log(
            "engine_compress_done "
            f"output={output_path!r} duration_seconds={duration:.2f}"
        )
        apply_meeting_date_to_artifact(self.request, output_path)
        self.sink(ArtifactEvent("compressed", output_path))
        self.sink(ProgressEvent("compression", 100, "Compression ready"))
        return StepResult("compression", True, output_path, duration_seconds=duration)
