from __future__ import annotations

import json
import time
from pathlib import Path

from .events import EventSink
from .logging import append_app_log
from .models import (
    ArtifactEvent,
    DoneEvent,
    EngineEvent,
    ErrorEvent,
    JobRequest,
    ProgressEvent,
    WarningEvent,
)
from .library import database, library_enroll_speakers_for_job, library_remember_speaker_names
from .pipeline import (
    CompressionPipeline,
    StepResult,
    TranscriptionPipeline,
    apply_meeting_date_to_artifact,
)
from .pipeline import prepare_job_workspace, snapshot_existing_artifacts


class EtaSmoothingSink:
    """Add a conservative, monotonic ETA to progress events.

    The engine progress percentages are step-local, so extrapolating
    from them makes the remaining time jump upward between phases. A
    historical full-job duration gives a less precise but much steadier
    estimate: it only counts down from the initial budget, and disappears
    when no history is available.
    """

    def __init__(self, sink: EventSink, estimated_total_seconds: float | None):
        self.sink = sink
        self.estimated_total_seconds = estimated_total_seconds
        self.started_at = time.monotonic()

    def __call__(self, event: EngineEvent) -> None:
        if (
            isinstance(event, ProgressEvent)
            and event.eta_seconds is None
            and self.estimated_total_seconds
            and (event.pct is None or event.pct < 100)
        ):
            elapsed = time.monotonic() - self.started_at
            eta = max(self.estimated_total_seconds - elapsed, 0)
            event = ProgressEvent(event.step, event.pct, event.message, eta_seconds=eta)
        self.sink(event)


def _historical_total_estimate_seconds(db, request: JobRequest) -> float | None:
    rows = db.list_jobs(limit=30, status="COMPLETED")
    durations: list[float] = []
    for row in rows:
        raw_duration = row.get("duration_total")
        if not raw_duration:
            continue
        try:
            duration = float(raw_duration)
        except (TypeError, ValueError):
            continue
        if duration < 10:
            continue
        raw_settings = row.get("settings_json") or "{}"
        try:
            settings = json.loads(raw_settings)
        except json.JSONDecodeError:
            settings = {}
        if settings.get("mode") and settings.get("mode") != request.mode:
            continue
        durations.append(duration)

    if len(durations) < 2:
        return None

    durations.sort()
    # A mild upper percentile is intentionally conservative: better to
    # count down a little slowly than to hit zero while work is still
    # running. The UI labels it as an estimate.
    index = min(len(durations) - 1, round((len(durations) - 1) * 0.65))
    return durations[index]


def _persist_step_durations(db, job_id: int, results: list[StepResult]) -> None:
    ffmpeg = sum(r.duration_seconds for r in results if r.name == "compression")
    diarization = sum(r.duration_seconds for r in results if r.name == "diarization")
    whisper = sum(
        r.duration_seconds
        for r in results
        if r.name not in {"compression", "diarization"}
    )
    db.update_job_durations(job_id, ffmpeg=ffmpeg, whisper=whisper, diarization=diarization)


def _resolve_source_path(request: JobRequest) -> tuple[Path | None, str]:
    """Return ``(path, tier)`` — the path the runner should actually
    feed to the pipeline (or ``None`` when nothing usable is on disk)
    and which lookup tier produced it.

    Four lookup tiers, ordered by trust:

    1. ``request.source_path`` itself if it exists. Covers the
       fresh-run case (user just dropped a file). Tier
       ``"request_source"``.

    2. The workspace copy at ``request.workspace_dir / basename``.
       Covers reruns where ``delete_source_after_copy`` cleaned up
       the original. Without this the second run of any meeting
       would dead-lock on "source missing" — the canonical copy is
       in the workspace, not at the original drop path. Tier
       ``"workspace_copy"``.

    3. The library row's stored ``workspace_dir`` when the SwiftUI
       caller didn't bother to forward one. Belt-and-braces for
       legacy callers that re-submit a job_id without a workspace.
       (Same tier label as 2.)

    4. PR AY — the job's COMPRESSED file when both the original and
       the workspace copy are gone. This is the "Libérer la source"
       aftermath (PR AP): the product rule says relaunching such a
       project transcribes the compressed version. PR AP only wired
       that on the artifact-level button; every other rerun path
       (row-level Relancer, context editor, à-revoir priority rerun)
       still submitted the original path and died on
       ``source_missing``. Resolving it here fixes them all at once.
       Sources: the DB row's ``compressed_path``, then a
       ``<stem>_compressed.*`` glob in the known workspaces for
       callers without a job id. Tier ``"compressed"`` — the caller
       degrades compression modes accordingly.

    None of these touch the network beyond one DB read — kept
    deliberately cheap because every job pays the cost on launch.
    """
    if not request.source_path:
        append_app_log("engine_resolve_source skipped reason='empty_source_path'")
        return None, ""
    candidate = Path(request.source_path).expanduser()
    if candidate.exists():
        append_app_log(f"engine_resolve_source hit='request_source' path={str(candidate)!r}")
        return candidate, "request_source"

    basename = candidate.name
    if not basename:
        append_app_log(f"engine_resolve_source skipped reason='empty_basename' source={request.source_path!r}")
        return None, ""

    workspace_dirs: list[str] = []
    if request.workspace_dir:
        workspace_dirs.append(request.workspace_dir)
    row = None
    if request.library_job_id is not None:
        try:
            row = database().get_job(request.library_job_id)
        except Exception:
            row = None
        if row:
            stored_workspace = (row.get("workspace_dir") or "").strip()
            if stored_workspace and stored_workspace not in workspace_dirs:
                workspace_dirs.append(stored_workspace)

    for workspace_dir in workspace_dirs:
        workspace_copy = Path(workspace_dir).expanduser() / basename
        if workspace_copy.exists():
            append_app_log(
                "engine_resolve_source hit='workspace_copy' "
                f"requested={str(candidate)!r} resolved={str(workspace_copy)!r}"
            )
            return workspace_copy, "workspace_copy"

    # Tier 4 — compressed fallback (PR AY).
    compressed_candidates: list[Path] = []
    if row:
        stored_compressed = (row.get("compressed_path") or "").strip()
        if stored_compressed:
            compressed_candidates.append(Path(stored_compressed).expanduser())
        # Recovery for jobs hit by the pre-fix snapshot bug: the only
        # surviving compressed file may sit in ``versions/<ts>/`` and
        # the live ``compressed_path`` column was cleared. The snapshot
        # location is recorded in ``previous_versions_json`` (newest
        # first) — trust it before falling back to a glob.
        raw_versions = (row.get("previous_versions_json") or "").strip()
        if raw_versions:
            try:
                versions = json.loads(raw_versions)
            except (json.JSONDecodeError, TypeError):
                versions = []
            for entry in versions if isinstance(versions, list) else []:
                if not isinstance(entry, dict):
                    continue
                versioned = (entry.get("compressed_path") or "").strip()
                if versioned:
                    compressed_candidates.append(Path(versioned).expanduser())
    stem = candidate.stem
    if stem:
        for workspace_dir in workspace_dirs:
            ws = Path(workspace_dir).expanduser()
            compressed_candidates.extend(sorted(ws.glob(f"{stem}_compressed.*")))
            # Snapshotted compressed files live one level down under
            # versions/<ts>/ — newest timestamp first.
            compressed_candidates.extend(
                sorted(ws.glob(f"versions/*/{stem}_compressed.*"), reverse=True)
            )
    for compressed in compressed_candidates:
        if compressed.exists() and compressed.is_file():
            append_app_log(
                "engine_resolve_source hit='compressed' "
                f"requested={str(candidate)!r} resolved={str(compressed)!r}"
            )
            return compressed, "compressed"

    append_app_log(
        "engine_resolve_source miss "
        f"requested={str(candidate)!r} workspaces={workspace_dirs!r}"
    )
    return None, ""


def _degrade_mode_for_compressed_source(mode: str) -> str:
    """PR AY — adjust the job mode when the only media left is the
    compressed file. Re-compressing a compressed file is pointless
    (and the product rule from PR AP forbids it: once freed, no
    re-compression), so compression modes degrade to a plain
    transcription. Non-compression modes pass through untouched.
    """
    if mode in ("compress", "compress_transcribe"):
        return "transcribe"
    return mode


def _auto_rename_job_from_transcript(
    db,
    job_id: int,
    transcript_path: str | None,
    source_path: str,
    suggested_title: str | None = None,
) -> str | None:
    """Pick a topical title from the transcript and store it on the job.

    Library rows fall back on the source filename when ``custom_title``
    is empty, which leaves the user staring at a wall of
    ``Enregistrement de l'écran 2026-05-16 à 14.42.30`` rows. Once the
    transcription is in, we have enough signal to do better — feed the
    text to :func:`suggest_transcript_stem` and persist the result.

    We never overwrite a title the user already set by hand: if a
    ``custom_title`` is present on the row before we run, we bail. The
    suggester also returns the fallback stem when it can't find a
    confident topic, in which case we still no-op to avoid promoting a
    raw filename to a "title".
    """
    if not transcript_path:
        return None
    try:
        row = db.get_job(job_id)
    except Exception:
        row = None
    if row and (row.get("custom_title") or "").strip():
        return None
    try:
        text = Path(transcript_path).read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    if not text.strip():
        return None
    # Lazy import to avoid pulling the full transcription_utils module
    # into the engine's startup path (it imports MLX/whisper bits).
    from transcription_utils import (
        is_useful_transcript_title,
        sanitize_filename_stem,
        suggest_transcript_stem,
    )

    fallback_stem = sanitize_filename_stem(Path(source_path).stem or "Transcription")
    llm_title = sanitize_filename_stem(suggested_title or "", "")
    if is_useful_transcript_title(llm_title, fallback_stem):
        suggestion = llm_title
    else:
        suggestion = suggest_transcript_stem(text, fallback_stem)
    if not suggestion or suggestion == fallback_stem:
        return None
    db.update_job_title(job_id, suggestion)
    append_app_log(
        f"engine_auto_rename job_id={job_id} title={suggestion!r}"
    )
    return suggestion


def _merge_declared_speaker_context(
    detected: dict[str, str] | None,
    declared_names: list[str] | tuple[str, ...] | set[str],
) -> dict[str, str]:
    """Keep user-declared participants in the job speaker context.

    The final pipeline map is built from the labels that actually
    appear in segments. That is the right source for diarisation rows,
    but it can drop a participant the user declared at launch when the
    LLM only identified one speaker by name. The library context should
    preserve explicit user input: those names bias Whisper and should
    stay available in the row/editor even before a segment has been
    confidently assigned to them.
    """
    merged = dict(detected or {})
    present: set[str] = set()
    for key, value in merged.items():
        key_norm = str(key or "").strip().lower()
        value_norm = str(value or "").strip().lower()
        if key_norm:
            present.add(key_norm)
        if value_norm:
            present.add(value_norm)

    for raw_name in declared_names:
        name = str(raw_name or "").strip()
        if not name:
            continue
        norm = name.lower()
        if norm in present:
            if name in merged and not str(merged.get(name) or "").strip():
                merged[name] = name
            continue
        merged[name] = name
        present.add(norm)
    return merged


def _snapshot_workspace_size(db, job_id: int, workspace: Path) -> None:
    """Walk the workspace and store the cumulative byte count.

    The library's optional "Poids" column reads this. We compute it
    once per successful job to avoid re-walking on every list
    refresh — the user has cheaper ways to ask for an update
    (delete + relaunch) and the alternative (lazy compute on
    library-list) would re-scan dozens of folders every refresh.
    """
    if not workspace.exists() or not workspace.is_dir():
        return
    total = 0
    for child in workspace.rglob("*"):
        try:
            if child.is_file() or child.is_symlink():
                total += child.stat().st_size if child.is_file() else 0
        except OSError:
            continue
    db.update_job_total_bytes(job_id, total)


class EngineRunner:
    def __init__(self, sink: EventSink):
        self.sink = sink

    def run_job(self, request: JobRequest) -> int:
        append_app_log(
            "engine_run_job "
            f"mode={request.mode!r} source={request.source_path!r} "
            f"workspace={request.workspace_dir!r} library_job_id={request.library_job_id!r}"
        )
        resolved, resolve_tier = _resolve_source_path(request)
        if resolved is None:
            # ``source_missing`` is the trigger the SwiftUI layer
            # listens for to pop the "Relocaliser la source" sheet.
            # We surface the requested path AND the workspace we
            # tried so the dialog can offer both as one-click
            # actions.
            attempted_workspace = (request.workspace_dir or "").strip()
            append_app_log(
                "engine_source_missing "
                f"source={request.source_path!r} workspace={attempted_workspace!r}"
            )
            self.sink(
                ErrorEvent(
                    f"Source file not found: {request.source_path}",
                    code="source_missing",
                )
            )
            return 2
        # The resolver may have substituted a workspace copy or the
        # source. Update the request so downstream code (workspace
        # prep, DB row) sees the path that actually exists.
        request.source_path = str(resolved)
        if resolve_tier == "compressed":
            # PR AY — the source was freed (PR AP) and only the
            # compressed file remains. Degrade compression modes and
            # tell the user what's happening instead of failing with
            # ``source_missing`` like before.
            degraded = _degrade_mode_for_compressed_source(request.mode)
            if degraded != request.mode:
                append_app_log(
                    f"engine_mode_degraded_for_compressed from={request.mode!r} "
                    f"to={degraded!r}"
                )
                request.mode = degraded
            self.sink(
                WarningEvent(
                    "La source originale a été libérée — la version "
                    "compressée est utilisée pour cette relance "
                    "(transcription uniquement, re-compression impossible).",
                    code="source_freed_using_compressed",
                )
            )
        Path(request.output_dir).mkdir(parents=True, exist_ok=True)
        db = database()
        estimated_total_seconds = _historical_total_estimate_seconds(db, request)
        sink = EtaSmoothingSink(self.sink, estimated_total_seconds)
        workspace, working_source = prepare_job_workspace(request, sink)
        request.workspace_dir = str(workspace)
        job_id = request.library_job_id
        if job_id is None:
            job_id = db.create_job(
                source_path=request.source_path,
                workspace_dir=str(workspace),
                settings=request.to_dict(),
            )
            # Nudge the SwiftUI library tab to refresh immediately. The
            # source ArtifactEvent from prepare_job_workspace fired
            # before the DB row existed, so its refresh saw nothing.
            # This second artifact event arrives after the INSERT, so
            # the list reflects the new job as soon as the user
            # switches tabs.
            sink(ArtifactEvent("library_row", str(workspace), model=str(job_id)))
        db.update_job_status(job_id, "RUNNING", "")
        db.update_job_meeting_date(job_id, request.meeting_date or None)
        # Rerun safety: before any pipeline step writes new artefacts,
        # snapshot the previous outputs into ``versions/<timestamp>/``.
        # Only fires when the job already has artefacts on disk —
        # fresh jobs no-op silently. The snapshot helper logs and
        # emits ``ArtifactEvent("previous_version", …)`` for each
        # file moved so the SwiftUI library refresh picks them up.
        if request.library_job_id is not None:
            existing_job = db.get_job(job_id)
            if existing_job:
                # Protect the active source: when the rerun runs on the
                # compressed file (original freed), it lives among the
                # snapshot candidates and must stay in place.
                from .pipeline import _normalized_realpath, produced_artifact_columns

                # Only archive what THIS run will overwrite — so a
                # "compress only" rerun after a "transcribe only" fills
                # the empty compressed slot and leaves the transcript
                # active, instead of pushing it to history.
                snapshot = snapshot_existing_artifacts(
                    workspace,
                    existing_job,
                    sink,
                    protected_paths={_normalized_realpath(working_source)},
                    kinds=produced_artifact_columns(request.mode),
                )
                if snapshot:
                    db.prepend_job_version(job_id, snapshot)
                    # Clear the current artefact paths — the pipeline
                    # will repopulate them as it runs. Without this
                    # the library would briefly show the moved paths
                    # as "current" until the new outputs land.
                    for column in (
                        "compressed_path",
                        "transcript_path",
                        "enhanced_transcript_path",
                        "review_path",
                    ):
                        if column in snapshot:
                            kind = {
                                "compressed_path": "compressed",
                                "transcript_path": "transcript",
                                "enhanced_transcript_path": "enhanced_transcript",
                                "review_path": "review",
                            }[column]
                            db.update_job_artefact(job_id, kind, "")
        db.update_job_progress(
            job_id,
            step="Démarrage…",
            progress_pct=0,
            eta_seconds=estimated_total_seconds,
        )
        # PR AW — the old "réécoute pas encore branchée" warning is
        # gone: the multimodal recheck has been wired since PR F and
        # now runs on Gemma 4 (12B Unified). When the venv or the
        # model genuinely can't serve it, ``_ensure_mlx_vlm_available``
        # emits its own actionable warning instead.
        # Persist the Odoo meeting metadata so the rename sheet can
        # surface attendee hint chips long after the engine exited.
        # Empty dict means "no meeting linked"; the helper takes care
        # of clearing the column on re-runs that detached the link.
        if request.odoo_meeting_metadata:
            db.update_job_odoo_meeting(job_id, request.odoo_meeting_metadata)
        elif request.library_job_id is None:
            # Fresh job, no meeting — explicitly null the column to
            # cover the unlikely case of stale data lingering on a
            # row reused via primary-key collision.
            db.update_job_odoo_meeting(job_id, None)
        declared_speakers = [
            name
            for name in (request.speaker_overrides or {}).values()
            if (name or "").strip()
        ]
        if declared_speakers:
            remembered = library_remember_speaker_names(declared_speakers, db=db)
            if remembered:
                append_app_log(
                    "engine_speakers_remembered "
                    f"job_id={job_id} count={remembered}"
                )

        results: list[StepResult] = []
        active_source = str(working_source)

        if request.mode in {"compress", "compress_transcribe"}:
            # PR AN: pass the resolved working source so compression
            # reads the workspace copy, not request.source_path which
            # may have been deleted by prepare_job_workspace when
            # delete_source_after_copy is set.
            result = CompressionPipeline(request, sink).run(active_source)
            results.append(result)
            if not result.ok:
                _persist_step_durations(db, job_id, results)
                # Failed jobs still left partial files (the source
                # copy + maybe a half-written compressed clip). The
                # library "Poids" column wants a real number on
                # these, not "—", so the user can decide whether to
                # cleanup or rerun.
                _snapshot_workspace_size(db, job_id, workspace)
                db.update_job_status(job_id, "FAILED", result.error or "Compression failed")
                sink(ErrorEvent(result.error or "Compression failed", code="compression_failed"))
                return 1
            apply_meeting_date_to_artifact(request, result.artifact_path)
            db.update_job_artefact(job_id, "compressed", result.artifact_path)
            db.update_job_output(job_id, result.artifact_path)
            db.update_job_progress(job_id, step="Compression terminée", progress_pct=50, eta_seconds=None)
            if request.mode == "compress_transcribe":
                active_source = result.artifact_path

        if request.mode in {"transcribe", "compress_transcribe"}:
            pipeline = TranscriptionPipeline(request, sink)
            tx_results = pipeline.run(active_source)
            results.extend(tx_results)
            # Money was spent the moment the API answered — persist the
            # ledger rows before any failure branch can return, and
            # denormalise the total onto the job row for the library.
            if pipeline.cloud_usage_records:
                cloud_total = 0.0
                cloud_model = ""
                for record in pipeline.cloud_usage_records:
                    db.add_api_usage(job_id=job_id, **record)
                    cloud_total += float(record.get("cost_usd") or 0)
                    cloud_model = record.get("model") or cloud_model
                db.update_job_cloud_cost(job_id, round(cloud_total, 6), cloud_model)
            # Per-chunk state for the library's "relancer les chunks
            # échoués" panel. Persisted on any cloud attempt (success or
            # partial) so the UI can list which windows failed.
            if pipeline.cloud_chunk_status:
                db.update_job_cloud_chunks(job_id, pipeline.cloud_chunk_status)
            failed = [
                r
                for r in tx_results
                if not r.ok
                and r.name in {"audio_extract", "whisper", "cloud_transcription"}
            ]
            # Quality steps (VAD, multipass, diarisation, llm_post) are
            # allowed to fail without sinking the whole job — they're
            # opt-in improvements. We only abort on the hard
            # prerequisites: audio extract, Whisper itself, and the
            # cloud engine when the user explicitly selected it
            # (budget cap or rejected API key).
            if failed:
                _persist_step_durations(db, job_id, results)
                # Same rationale as the compression-failure branch:
                # the user wants to see how much disk this run
                # consumed before deciding whether to clean it up.
                _snapshot_workspace_size(db, job_id, workspace)
                db.update_job_status(job_id, "FAILED", failed[0].error or "Transcription failed")
                sink(ErrorEvent(failed[0].error or "Transcription failed", code="transcription_failed"))
                return 1
            # Record which model + engine produced this transcript so
            # the library can show "fait avec X". Cloud step wins when
            # present; otherwise the Whisper pass model.
            tx_engine = (
                request.transcription_settings.transcription_engine or "local"
            ).strip().lower()
            tx_model = ""
            for r in tx_results:
                if r.name == "cloud_transcription" and r.model:
                    tx_model = r.model
                    break
            if not tx_model:
                for r in tx_results:
                    if r.name == "whisper" and r.model:
                        tx_model = r.model
                        break
            db.update_job_transcription_model(job_id, tx_model or None, tx_engine)
            transcript_for_title: str | None = None
            for r in tx_results:
                if r.artifact_path:
                    apply_meeting_date_to_artifact(request, r.artifact_path)
                if r.name == "transcript" and r.artifact_path:
                    db.update_job_artefact(job_id, "transcript", r.artifact_path)
                    db.update_job_output(job_id, r.artifact_path)
                    transcript_for_title = r.artifact_path
                elif r.name == "enhanced_transcript" and r.artifact_path:
                    db.update_job_artefact(job_id, "enhanced_transcript", r.artifact_path)
                    db.update_job_output(job_id, r.artifact_path)
                    # The corrected transcript is a better title source
                    # — Whisper output sometimes mangles proper nouns
                    # that the LLM pass restores.
                    transcript_for_title = r.artifact_path
                elif r.name == "review" and r.artifact_path:
                    db.update_job_artefact(job_id, "review", r.artifact_path)
            # Promote a topical title onto the library row so the user
            # sees "Atelier RH 2026" rather than "Enregistrement de
            # l'écran 2026-05-16 à 14.42.30" — but only when no custom
            # title has been set yet (see the helper for the bail
            # rules).
            new_title = _auto_rename_job_from_transcript(
                db,
                job_id,
                transcript_for_title,
                request.source_path,
                suggested_title=pipeline.final_title,
            )
            if new_title:
                sink(ArtifactEvent("job_title", new_title, model=str(job_id)))
            # Persist segments + context so the library's rename
            # sheet has speakers to show and ``library_speaker_samples``
            # can cut audio extracts. Without these calls the
            # ``transcription_segments`` table and ``speaker_map_json``
            # column stay empty and the sheet renders "Aucun
            # interlocuteur détecté" on every run.
            if pipeline.final_segments:
                db.add_segments(job_id, pipeline.final_segments)
            speaker_context = _merge_declared_speaker_context(
                pipeline.final_speaker_map, declared_speakers
            )
            if speaker_context or pipeline.final_technical_terms:
                db.update_job_context(
                    job_id,
                    speakers=speaker_context or None,
                    technical_terms=pipeline.final_technical_terms or None,
                )
            # Auto-enrol the friendly-named clusters into the voice
            # profile store so the NEXT job recognises them without
            # the user having to open the rename sheet. Previously
            # ``library_enroll_speakers_for_job`` only fired from the
            # rename flow — which left every voice profile at
            # ``sample_count=0`` for users who confirmed the LLM-
            # detected names by simply *not* editing them. The
            # ``current_user_name`` heuristic from
            # ``_pre_attribute_current_user`` lands here too, so
            # Robin's voice gets enrolled the first time he runs
            # any job with his name in Réglages.
            if pipeline.final_speaker_map:
                enrolment = {
                    label: name
                    for label, name in pipeline.final_speaker_map.items()
                    if name
                    and label
                    and not label.upper().startswith("SPEAKER_")
                    and name.strip().lower() == label.strip().lower()
                }
                if enrolment:
                    try:
                        enrolled_count = library_enroll_speakers_for_job(
                            job_id, enrolment, db=db
                        )
                        append_app_log(
                            f"engine_auto_enrol job_id={job_id} count={enrolled_count}"
                        )
                    except Exception as exc:
                        # Enrolment is best-effort. Logging the
                        # failure is enough — never sink a successful
                        # transcription on a downstream embedding
                        # glitch.
                        append_app_log(
                            f"engine_auto_enrol_failed job_id={job_id} error={exc!r}"
                        )
            db.update_job_progress(job_id, step="Transcription terminée", progress_pct=100, eta_seconds=0)

        if request.mode in {"enhance", "review"}:
            sink(
                WarningEvent(
                    "Enhance/review-only mode is reserved for the next engine extraction step",
                    code="not_implemented_yet",
                )
            )

        _persist_step_durations(db, job_id, results)
        _snapshot_workspace_size(db, job_id, workspace)
        sink(ProgressEvent("done", 100, "Job complete", eta_seconds=0))
        db.update_job_status(job_id, "COMPLETED", "")
        sink(
            DoneEvent(
                {
                    "job_id": job_id,
                    "mode": request.mode,
                    "artifacts": [
                        {"kind": r.name, "path": r.artifact_path, "model": r.model}
                        for r in results
                        if r.artifact_path
                    ],
                    "steps": [
                        {
                            "name": r.name,
                            "ok": r.ok,
                            "duration_seconds": r.duration_seconds,
                            "metrics": r.metrics,
                        }
                        for r in results
                    ],
                }
            )
        )
        return 0
