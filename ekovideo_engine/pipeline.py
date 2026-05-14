from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from ffmpeg_utils import build_ffmpeg_cmd, default_out_path, is_audio_only_path
from glossary_postprocess import (
    GlossarySubstitution,
    apply_glossary_to_segments,
    parse_glossary_terms,
)
from multipass import (
    group_into_clip_ranges,
    identify_weak_segments,
    merge_repass_segments,
)
from transcription_utils import (
    assign_speakers_to_segments,
    build_audio_extract_cmd,
    build_diarization_cmd,
    build_llm_corrections_cmd,
    build_llm_title_cmd,
    build_mlx_whisper_cmd,
    default_transcript_path,
    parse_diarization_output,
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


def _safe_stem(value: str) -> str:
    keep = []
    for char in value:
        if char.isalnum() or char in {" ", "-", "_", "."}:
            keep.append(char)
        else:
            keep.append(" ")
    cleaned = " ".join("".join(keep).split()).strip(" .-_")
    return (cleaned or "Transcription")[:80]


def job_workspace_dir(request: JobRequest) -> Path:
    if request.workspace_dir:
        return Path(request.workspace_dir)
    root = Path(request.output_dir).expanduser()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return root / f"{stamp} - {_safe_stem(Path(request.source_path).stem)}"


def prepare_job_workspace(request: JobRequest, sink: EventSink) -> tuple[Path, Path]:
    workspace = job_workspace_dir(request)
    workspace.mkdir(parents=True, exist_ok=True)
    source = Path(request.source_path).expanduser()
    copied_source = workspace / source.name
    if (
        source.exists()
        and source.resolve() != copied_source.resolve()
        and not copied_source.exists()
    ):
        shutil.copy2(source, copied_source)
        sink(ArtifactEvent("source", str(copied_source)))
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

    # ------------------------------------------------------------------
    # Public orchestrator
    # ------------------------------------------------------------------

    def run(self, source_path: str) -> list[StepResult]:
        results: list[StepResult] = []
        workspace = job_workspace_dir(self.request)
        workspace.mkdir(parents=True, exist_ok=True)
        wav_path = workspace / "audio.wav"

        # --- step 1: audio extract -------------------------------------
        results.append(self._extract_audio(source_path, wav_path))
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
        if self._should_run_diarisation():
            diar_result, turns = self._run_diarisation(wav_path)
            if diar_result is not None:
                results.append(diar_result)
            if turns:
                segments = assign_speakers_to_segments(segments, turns)
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

        # --- step 5bis: apply LLM speaker renames if any --------------
        speakers = self._llm_payload.get("speakers") or {}
        if speakers and any(speakers.values()):
            segments = self._apply_speaker_renames(segments, speakers)
            # Rewrite the transcript with the friendly speaker names.
            transcript_path = self._write_transcript(
                segments, workspace, force_path=transcript_path
            )
            self.sink(
                ContextEvent(
                    speakers=speakers,
                    technical_terms=self._llm_payload.get("technical_terms") or [],
                )
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
        return results

    # ------------------------------------------------------------------
    # Step 1 — audio extract
    # ------------------------------------------------------------------

    def _extract_audio(self, source_path: str, wav_path: Path) -> StepResult:
        ts = time.monotonic()
        settings = self.request.transcription_settings
        cmd = build_audio_extract_cmd(
            self.request.compression_settings.ffmpeg_path or "ffmpeg",
            source_path,
            str(wav_path),
            speech_enhance=settings.enhance_audio,
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

        self._vad_manifest = spans
        kept = payload.get("trimmed_seconds") or 0
        total = payload.get("total_seconds") or 0
        ratio = (kept / total) if total else 0
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
        cmd = build_mlx_whisper_cmd(
            mlx_whisper_path=settings.mlx_whisper_path or "mlx_whisper",
            audio_path=str(wav_path),
            output_path=str(whisper_json),
            model=settings.model,
            language=settings.language,
            output_format="json",
            initial_prompt=structured_initial_prompt(glossary),
            condition_on_previous_text=False,
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

        segments = parse_whisper_json_segments(str(whisper_json))
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
                metrics={"segments": len(segments)},
            ),
            segments,
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

        repass_model = "mlx-community/whisper-large-v3"
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
                initial_prompt=structured_initial_prompt(glossary),
                condition_on_previous_text=False,
                clip_timestamps=f"{cs:.2f},{ce:.2f}",
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
        cmd = build_diarization_cmd(settings.venv_python_path, str(wav_path))
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
                metrics={"speakers": len(speakers_seen), "turns": len(turns)},
            ),
            turns,
        )

    # ------------------------------------------------------------------
    # Step 5 — LLM post-process
    # ------------------------------------------------------------------

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
        glossary_text = "\n".join(
            [*self.request.glossary_terms, *self.request.technical_terms]
        )

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

        # ---- corrections + doubts (markdown) -------------------------
        corr_cmd = build_llm_corrections_cmd(
            venv_python, settings.text_llm_model, str(analysis_path), glossary_text
        )
        self.sink(ProgressEvent("llm_corrections", 50, "LLM: corrections + doubts"))
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
                payload["corrections"] = corr_payload.get("corrections") or []
                payload["uncertain_passages"] = (
                    corr_payload.get("uncertain_passages") or []
                )
            except Exception as exc:
                append_app_log(f"engine_llm_corrections_parse_failed error={exc!r}")
        else:
            append_app_log(
                f"engine_llm_corrections_failed rc={corr_proc.returncode} "
                f"stderr={tail_text(corr_proc.stderr)!r}"
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
        transcript_path.write_text(rendered, encoding="utf-8")
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
        )

    def _write_enhanced_transcript(
        self, transcript_path: Path, segments: list[dict]
    ) -> Path | None:
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
        enhanced_path.write_text(rendered, encoding="utf-8")
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

        corrections = self._llm_payload.get("corrections") or []
        if corrections:
            lines += ["## Corrections proposées par le LLM", ""]
            for c in corrections[:25]:
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
            lines.append("")

        review_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
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
            to=settings.trim_end if settings.trim_enabled else None,
            audio_only=is_audio_only_path(self.request.source_path),
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
        self.sink(ArtifactEvent("compressed", output_path))
        self.sink(ProgressEvent("compression", 100, "Compression ready"))
        return StepResult("compression", True, output_path, duration_seconds=duration)
