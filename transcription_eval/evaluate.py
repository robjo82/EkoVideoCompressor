from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class EvalResult:
    case: str
    score: float
    missing_terms: list[str]
    forbidden_hits: list[str]
    missing_speakers: list[str]
    # PR K additions — fragmentation telemetry to track the
    # qualitative improvements PR A + I delivered on top of the
    # existing term-based criteria.
    fragment_count: int = 0
    orphan_speaker_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "case": self.case,
            "score": self.score,
            "missing_terms": self.missing_terms,
            "forbidden_hits": self.forbidden_hits,
            "missing_speakers": self.missing_speakers,
            "fragment_count": self.fragment_count,
            "orphan_speaker_count": self.orphan_speaker_count,
        }


def count_fragmentation_indicators(transcript: str) -> tuple[int, int]:
    """Surface the two fragmentation signals PR A targeted.

    - ``orphan_speaker_count`` : lines whose speaker tag is ``[?]``
      (pyannote couldn't attribute). PR A's
      ``absorb_orphan_speaker_fragments`` should drive this to 0.
    - ``fragment_count`` : lines that look like a Whisper mis-split
      (less than 5 words after the timestamp). Doesn't catch every
      issue but is a cheap proxy that decreases as PR A's smoothing
      + same-speaker merge get applied.
    """
    if not transcript:
        return 0, 0
    orphan = 0
    fragments = 0
    for line in transcript.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("[?]"):
            orphan += 1
            continue
        if not stripped.startswith("["):
            continue
        # Strip leading ``[Speaker] (timestamp)`` then count words.
        # Format: ``[Name] (HH:MM:SS) text body``
        match = re.match(r"^\[[^\]]+\]\s*(?:\([^)]*\)\s*)?(.*)$", stripped)
        if not match:
            continue
        body = match.group(1).strip()
        words = [w for w in re.split(r"\s+", body) if w]
        if 0 < len(words) <= 4:
            fragments += 1
    return fragments, orphan


def _contains_term(text: str, term: str) -> bool:
    pattern = r"(?<!\w)" + re.escape(term) + r"(?!\w)"
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def evaluate_case(path: Path) -> EvalResult:
    """Score a JSON case against the embedded transcript.

    PR K extension: ``transcript_path`` can override the embedded
    text so the same criteria can be re-checked on a freshly-run
    Eko output without editing the case file.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    embedded = str(payload.get("transcript") or "")
    transcript_path = payload.get("transcript_path") or ""
    if transcript_path:
        candidate = Path(transcript_path).expanduser()
        if candidate.is_file():
            try:
                embedded = candidate.read_text(encoding="utf-8")
            except OSError:
                pass
    transcript = embedded
    expected_terms = [str(x) for x in payload.get("expected_terms") or []]
    forbidden = [str(x) for x in payload.get("forbidden_substrings") or []]
    expected_speakers = [str(x) for x in payload.get("expected_speakers") or []]

    missing_terms = [term for term in expected_terms if not _contains_term(transcript, term)]
    forbidden_hits = [term for term in forbidden if term.lower() in transcript.lower()]
    missing_speakers = [
        speaker for speaker in expected_speakers if f"[{speaker}]" not in transcript
    ]
    fragments, orphans = count_fragmentation_indicators(transcript)

    # Fragmentation contributes to the error count when above a
    # tolerance (3 orphans + 5 short lines = noise; more = signal).
    fragment_penalty = max(0, fragments - 5) + orphans
    total = max(
        1,
        len(expected_terms)
        + len(forbidden)
        + len(expected_speakers)
        + fragment_penalty,
    )
    errors = (
        len(missing_terms)
        + len(forbidden_hits)
        + len(missing_speakers)
        + fragment_penalty
    )
    score = max(0.0, 1.0 - (errors / total))
    return EvalResult(
        case=str(payload.get("name") or path.stem),
        score=score,
        missing_terms=missing_terms,
        forbidden_hits=forbidden_hits,
        missing_speakers=missing_speakers,
        fragment_count=fragments,
        orphan_speaker_count=orphans,
    )


def evaluate_dir(case_dir: Path) -> list[EvalResult]:
    return [evaluate_case(path) for path in sorted(case_dir.glob("*.json"))]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default=str(Path(__file__).parent / "cases"))
    parser.add_argument("--min-score", type=float, default=0.95)
    args = parser.parse_args(argv)

    results = evaluate_dir(Path(args.cases))
    payload = {
        "score": sum(r.score for r in results) / max(1, len(results)),
        "results": [r.to_dict() for r in results],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["score"] >= args.min_score else 1


if __name__ == "__main__":
    raise SystemExit(main())
