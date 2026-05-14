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

    def to_dict(self) -> dict[str, Any]:
        return {
            "case": self.case,
            "score": self.score,
            "missing_terms": self.missing_terms,
            "forbidden_hits": self.forbidden_hits,
            "missing_speakers": self.missing_speakers,
        }


def _contains_term(text: str, term: str) -> bool:
    pattern = r"(?<!\w)" + re.escape(term) + r"(?!\w)"
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def evaluate_case(path: Path) -> EvalResult:
    payload = json.loads(path.read_text(encoding="utf-8"))
    transcript = str(payload.get("transcript") or "")
    expected_terms = [str(x) for x in payload.get("expected_terms") or []]
    forbidden = [str(x) for x in payload.get("forbidden_substrings") or []]
    expected_speakers = [str(x) for x in payload.get("expected_speakers") or []]

    missing_terms = [term for term in expected_terms if not _contains_term(transcript, term)]
    forbidden_hits = [term for term in forbidden if term.lower() in transcript.lower()]
    missing_speakers = [
        speaker for speaker in expected_speakers if f"[{speaker}]" not in transcript
    ]

    total = max(1, len(expected_terms) + len(forbidden) + len(expected_speakers))
    errors = len(missing_terms) + len(forbidden_hits) + len(missing_speakers)
    score = max(0.0, 1.0 - (errors / total))
    return EvalResult(
        case=str(payload.get("name") or path.stem),
        score=score,
        missing_terms=missing_terms,
        forbidden_hits=forbidden_hits,
        missing_speakers=missing_speakers,
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
