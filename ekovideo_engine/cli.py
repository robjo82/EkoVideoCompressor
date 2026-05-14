from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from .events import stdout_event_sink
from .hf import hf_check
from .library import (
    library_delete,
    library_list,
    library_rename_speakers,
    library_update_context,
)
from .logging import export_logs_archive
from .model_cache import delete_model, download_model, model_catalog
from .models import DoneEvent, ErrorEvent, JobRequest, ProgressEvent
from .runner import EngineRunner


def _print_json(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _load_json_arg(path_or_json: str) -> dict:
    candidate = Path(path_or_json)
    if candidate.exists():
        return json.loads(candidate.read_text(encoding="utf-8"))
    return json.loads(path_or_json)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ekovideo-engine")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--startup-smoke-test", action="store_true")
    sub = parser.add_subparsers(dest="command")

    run_job = sub.add_parser("run-job")
    run_job.add_argument("--request", required=True, help="Path to a JobRequest JSON file")

    sub.add_parser("library-list")
    library_delete_parser = sub.add_parser("library-delete")
    library_delete_parser.add_argument("job_id", type=int)

    rename = sub.add_parser("library-rename-speakers")
    rename.add_argument("job_id", type=int)
    rename.add_argument("--mapping", required=True, help="JSON object or JSON file")

    context = sub.add_parser("library-update-context")
    context.add_argument("job_id", type=int)
    context.add_argument("--speakers", default="{}", help="JSON object or JSON file")
    context.add_argument("--technical-terms", default="[]", help="JSON array or JSON file")

    model_list = sub.add_parser("model-list")
    model_list.add_argument("--jsonl", action="store_true")
    model_download = sub.add_parser("model-download")
    model_download.add_argument("repo_id")
    model_download.add_argument("--token", default="")
    model_delete = sub.add_parser("model-delete")
    model_delete.add_argument("repo_id")

    hf = sub.add_parser("hf-check")
    hf.add_argument("--token", required=True)

    logs = sub.add_parser("export-logs")
    logs.add_argument("--output", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.smoke_test:
        stdout_event_sink(ProgressEvent("smoke", 100, "Engine smoke test ok"))
        stdout_event_sink(DoneEvent({"ok": True}))
        return 0
    if args.startup_smoke_test:
        stdout_event_sink(ProgressEvent("startup", 100, "Engine startup ok"))
        stdout_event_sink(DoneEvent({"ok": True}))
        return 0

    try:
        if args.command == "run-job":
            payload = json.loads(Path(args.request).read_text(encoding="utf-8"))
            request = JobRequest.from_dict(payload)
            return EngineRunner(stdout_event_sink).run_job(request)

        if args.command == "library-list":
            _print_json(library_list())
            return 0

        if args.command == "library-delete":
            library_delete(args.job_id)
            _print_json({"deleted": args.job_id})
            return 0

        if args.command == "library-rename-speakers":
            mapping = _load_json_arg(args.mapping)
            changed = library_rename_speakers(args.job_id, {str(k): str(v) for k, v in mapping.items()})
            _print_json({"job_id": args.job_id, "segments_changed": changed})
            return 0

        if args.command == "library-update-context":
            speakers = _load_json_arg(args.speakers)
            technical_terms = _load_json_arg(args.technical_terms)
            library_update_context(args.job_id, speakers=speakers, technical_terms=technical_terms)
            _print_json({"job_id": args.job_id, "updated": True})
            return 0

        if args.command == "model-list":
            rows = model_catalog()
            if args.jsonl:
                for row in rows:
                    print(json.dumps(row, ensure_ascii=False, sort_keys=True))
            else:
                _print_json(rows)
            return 0

        if args.command == "model-download":
            path = download_model(args.repo_id, token=args.token)
            _print_json({"repo_id": args.repo_id, "cache_dir": str(path)})
            return 0

        if args.command == "model-delete":
            path = delete_model(args.repo_id)
            _print_json({"repo_id": args.repo_id, "cache_dir": str(path), "deleted": True})
            return 0

        if args.command == "hf-check":
            _print_json(hf_check(args.token))
            return 0

        if args.command == "export-logs":
            output = args.output or str(
                Path.home()
                / "Desktop"
                / f"ekovideo-logs-{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
            )
            path = export_logs_archive(Path(output))
            _print_json({"path": str(path)})
            return 0

        build_parser().print_help()
        return 2
    except Exception as exc:
        stdout_event_sink(ErrorEvent(str(exc), code="cli_error"))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
