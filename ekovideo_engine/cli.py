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
    library_delete_speaker_profile,
    library_discover_speakers,
    library_flag_speaker_sample_review,
    library_link_speaker_profile_to_odoo,
    library_list,
    library_list_speaker_profiles,
    library_recognize_speakers,
    library_rename_speakers,
    library_speaker_samples,
    library_unlink_speaker_profile_from_odoo,
    library_update_context,
    library_workspace_usage,
)
from .logging import export_logs_archive
from .model_cache import delete_model, download_model, model_catalog
from .models import DoneEvent, ErrorEvent, JobRequest, ProgressEvent
from .runner import EngineRunner
from odoo_client import (
    OdooConfig,
    OdooError,
    fetch_partner,
    search_meeting_events,
    search_partners,
    test_connection as odoo_test_connection,
)


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

    library_list_parser = sub.add_parser("library-list")
    library_list_parser.add_argument("--jsonl", action="store_true")
    library_delete_parser = sub.add_parser("library-delete")
    library_delete_parser.add_argument("job_id", type=int)
    # Opt-in flag: also wipe the workspace dir on disk. Without this
    # we only drop the DB row (legacy behaviour). With it, the
    # engine returns an audit summary listing files removed + bytes
    # freed, which the SwiftUI sheet surfaces as "Économie : X Mo".
    library_delete_parser.add_argument(
        "--remove-files",
        action="store_true",
        help="Also delete the workspace directory on disk.",
    )

    # Quick disk-usage preview so the SwiftUI sheet can show "what
    # will be freed" before the user clicks Supprimer.
    usage = sub.add_parser("library-workspace-usage")
    usage.add_argument("job_id", type=int)

    rename = sub.add_parser("library-rename-speakers")
    rename.add_argument("job_id", type=int)
    rename.add_argument("--mapping", required=True, help="JSON object or JSON file")

    samples = sub.add_parser("library-speaker-samples")
    samples.add_argument("job_id", type=int)
    samples.add_argument("--seconds", type=float, default=8.0)
    samples.add_argument("--per-speaker", type=int, default=3)
    samples.add_argument("--jsonl", action="store_true")

    sample_review = sub.add_parser("library-speaker-sample-review")
    sample_review.add_argument("job_id", type=int)
    sample_review.add_argument("--speaker", required=True)
    sample_review.add_argument("--start", type=float, required=True)
    sample_review.add_argument("--duration", type=float, required=True)
    sample_review.add_argument("--note", default="")

    # Backfill the speaker list for old jobs whose pipeline didn't
    # persist segments or speaker_map_json. The SwiftUI rename sheet
    # calls this when its locally-known list is empty, so users can
    # still edit speakers on jobs that completed before the
    # persistence fix.
    discover = sub.add_parser("library-discover-speakers")
    discover.add_argument("job_id", type=int)

    # Speaker enrollment store. The pipeline matches new clusters
    # against this store automatically, but the user can also list /
    # delete profiles via the SwiftUI settings panel.
    list_profiles = sub.add_parser("library-list-speaker-profiles")
    list_profiles.add_argument("--jsonl", action="store_true")

    delete_profile = sub.add_parser("library-delete-speaker-profile")
    profile_group = delete_profile.add_mutually_exclusive_group(required=True)
    profile_group.add_argument("--id", type=int, dest="profile_id")
    profile_group.add_argument("--name", type=str)

    # Re-run recognition on an existing job (e.g. after the user
    # added a new profile and wants to back-fill an older meeting).
    recognise = sub.add_parser("library-recognize-speakers")
    recognise.add_argument("job_id", type=int)

    # ----- Odoo integration ---------------------------------------
    # JSON-2 commands. ``odoo-test`` lives behind the SwiftUI
    # "Tester la connexion" button; ``odoo-search-partners`` powers
    # the link sheet's live search; ``library-link-speaker-profile``
    # / ``library-unlink-speaker-profile`` persist the user's choice.
    def _add_odoo_args(parser):
        parser.add_argument("--url", required=True)
        parser.add_argument("--db", required=True)
        parser.add_argument("--login", required=True)
        parser.add_argument("--api-key", required=True)

    odoo_test = sub.add_parser("odoo-test")
    _add_odoo_args(odoo_test)

    odoo_search = sub.add_parser("odoo-search-partners")
    _add_odoo_args(odoo_search)
    odoo_search.add_argument("--query", required=True)
    odoo_search.add_argument("--limit", type=int, default=25)
    odoo_search.add_argument("--jsonl", action="store_true")

    # Discovers calendar.event records around a given moment so the
    # SwiftUI Run Setup can suggest "is this meeting one of those?"
    # without forcing the user to type the title from scratch.
    odoo_meetings = sub.add_parser("odoo-search-meetings")
    _add_odoo_args(odoo_meetings)
    odoo_meetings.add_argument(
        "--near",
        help="ISO 8601 datetime to bracket the search around. Defaults to now (UTC).",
        default="",
    )
    odoo_meetings.add_argument(
        "--window-hours",
        type=float,
        default=2.0,
        help="Half-width of the search bracket (default 2 h).",
    )
    odoo_meetings.add_argument("--limit", type=int, default=10)
    odoo_meetings.add_argument("--jsonl", action="store_true")

    link_profile = sub.add_parser("library-link-speaker-profile")
    link_profile.add_argument("profile_id", type=int)
    link_profile.add_argument("--partner-id", type=int, required=True)
    link_profile.add_argument("--partner-name", required=True)
    link_profile.add_argument("--company-id", type=int, default=0)
    link_profile.add_argument("--company-name", default="")

    unlink_profile = sub.add_parser("library-unlink-speaker-profile")
    unlink_profile.add_argument("profile_id", type=int)

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
            rows = library_list()
            if args.jsonl:
                for row in rows:
                    print(json.dumps(row, ensure_ascii=False, sort_keys=True))
            else:
                _print_json(rows)
            return 0

        if args.command == "library-delete":
            summary = library_delete(args.job_id, remove_files=args.remove_files)
            _print_json({"deleted": args.job_id, **summary})
            return 0

        if args.command == "library-workspace-usage":
            _print_json(library_workspace_usage(args.job_id))
            return 0

        if args.command == "library-rename-speakers":
            mapping = _load_json_arg(args.mapping)
            result = library_rename_speakers(args.job_id, {str(k): str(v) for k, v in mapping.items()})
            _print_json({"job_id": args.job_id, **result})
            return 0

        if args.command == "library-speaker-samples":
            rows = library_speaker_samples(
                args.job_id,
                seconds=args.seconds,
                per_speaker=args.per_speaker,
            )
            if args.jsonl:
                for row in rows:
                    print(json.dumps(row, ensure_ascii=False, sort_keys=True))
            else:
                _print_json(rows)
            return 0

        if args.command == "library-speaker-sample-review":
            _print_json(
                library_flag_speaker_sample_review(
                    args.job_id,
                    speaker=args.speaker,
                    start=args.start,
                    duration=args.duration,
                    note=args.note,
                )
            )
            return 0

        if args.command == "library-update-context":
            speakers = _load_json_arg(args.speakers)
            technical_terms = _load_json_arg(args.technical_terms)
            library_update_context(args.job_id, speakers=speakers, technical_terms=technical_terms)
            _print_json({"job_id": args.job_id, "updated": True})
            return 0

        if args.command == "library-discover-speakers":
            speakers = library_discover_speakers(args.job_id)
            _print_json({"job_id": args.job_id, "speakers": speakers})
            return 0

        if args.command == "library-list-speaker-profiles":
            rows = library_list_speaker_profiles()
            if args.jsonl:
                for row in rows:
                    print(json.dumps(row, ensure_ascii=False, sort_keys=True))
            else:
                _print_json(rows)
            return 0

        if args.command == "library-delete-speaker-profile":
            removed = library_delete_speaker_profile(
                profile_id=args.profile_id, name=args.name
            )
            _print_json(
                {
                    "deleted": bool(removed),
                    "profile_id": args.profile_id,
                    "name": args.name,
                }
            )
            return 0

        if args.command == "library-recognize-speakers":
            speakers = library_recognize_speakers(args.job_id)
            _print_json({"job_id": args.job_id, "recognized": speakers})
            return 0

        if args.command in {
            "odoo-test", "odoo-search-partners", "odoo-search-meetings",
        }:
            config = OdooConfig(
                url=args.url,
                database=args.db,
                login=args.login,
                api_key=args.api_key,
            )
            try:
                if args.command == "odoo-test":
                    payload = odoo_test_connection(config)
                    _print_json(payload)
                    return 0
                if args.command == "odoo-search-meetings":
                    near = None
                    raw_near = (args.near or "").strip()
                    if raw_near:
                        # Tolerate the trailing "Z" SwiftUI emits.
                        try:
                            near = datetime.fromisoformat(raw_near.replace("Z", "+00:00"))
                        except ValueError as exc:
                            _print_json(
                                {
                                    "ok": False,
                                    "error": f"Date '--near' invalide : {exc}",
                                }
                            )
                            return 1
                    meetings = search_meeting_events(
                        config,
                        near=near,
                        window_hours=args.window_hours,
                        limit=args.limit,
                    )
                    if args.jsonl:
                        for row in meetings:
                            print(json.dumps(row, ensure_ascii=False, sort_keys=True))
                    else:
                        _print_json(meetings)
                    return 0
                # odoo-search-partners
                rows = search_partners(config, args.query, limit=args.limit)
                if args.jsonl:
                    for row in rows:
                        print(json.dumps(row, ensure_ascii=False, sort_keys=True))
                else:
                    _print_json(rows)
                return 0
            except OdooError as exc:
                # Odoo errors carry a French human-readable message —
                # surface them as a structured failure the SwiftUI
                # status banner can render verbatim.
                _print_json({"ok": False, "error": str(exc)})
                return 1

        if args.command == "library-link-speaker-profile":
            updated = library_link_speaker_profile_to_odoo(
                args.profile_id,
                partner_id=args.partner_id,
                partner_name=args.partner_name,
                company_id=args.company_id or None,
                company_name=args.company_name,
            )
            _print_json(updated)
            return 0

        if args.command == "library-unlink-speaker-profile":
            updated = library_unlink_speaker_profile_from_odoo(args.profile_id)
            _print_json(updated)
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
