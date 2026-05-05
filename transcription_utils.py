from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


TRANSCRIPTION_AUDIO_FILTERS = [
    "highpass=f=80",
    "lowpass=f=7600",
    "acompressor=threshold=-20dB:ratio=2.2:attack=5:release=160",
    "loudnorm=I=-16:TP=-1.5:LRA=11",
]

# Hugging Face model IDs used for diarisation. Both require accepting the
# license on huggingface.co before a token can download them.
PYANNOTE_DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
PYANNOTE_SEGMENTATION_MODEL = "pyannote/segmentation-3.0"

# Whisper's `--initial-prompt` is fed to the decoder as a fake prefix; the
# model truncates anything beyond ~224 tokens, so the prompt must be tight
# and front-load the most important vocabulary.
INITIAL_PROMPT_MAX_CHARS = 800


def transcript_output_ext(output_format: str) -> str:
    fmt = (output_format or "txt").strip().lower()
    if fmt == "all":
        return "txt"
    if fmt in {"txt", "srt", "vtt", "json", "tsv"}:
        return fmt
    return "txt"


def default_transcript_path(in_path: str, out_dir: str, suffix: str, output_format: str) -> str:
    source = Path(in_path)
    safe_suffix = suffix.strip() or "_transcription"
    ext = transcript_output_ext(output_format)
    base = Path(out_dir) / f"{source.stem}{safe_suffix}.{ext}"
    out = base
    i = 1
    while out.exists():
        out = Path(out_dir) / f"{source.stem}{safe_suffix}_{i}.{ext}"
        i += 1
    return str(out)


def build_audio_extract_cmd(
    ffmpeg_path: str,
    in_path: str,
    wav_path: str,
    speech_enhance: bool = True,
    ss: str | None = None,
    to: str | None = None,
) -> list[str]:
    cmd = [ffmpeg_path, "-y", "-hide_banner", "-loglevel", "error"]

    if ss is not None:
        cmd += ["-ss", ss]
    if to is not None:
        cmd += ["-to", to]

    cmd += ["-i", in_path, "-vn"]

    if speech_enhance:
        cmd += ["-af", ",".join(TRANSCRIPTION_AUDIO_FILTERS)]

    cmd += [
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        "-progress",
        "pipe:1",
        "-nostats",
        wav_path,
    ]
    return cmd


def structured_initial_prompt(context: str) -> str:
    """
    Whisper interprets `--initial-prompt` as the imagined previous segment of
    the same audio. A bare list of names ("Ekonum, MAIA, RGPD") works less
    well than a short natural sentence that *uses* those terms — that primes
    the decoder's language model to expect them.
    """
    raw = (context or "").strip()
    if not raw:
        return ""
    flat = " ".join(raw.split())
    if len(flat) > INITIAL_PROMPT_MAX_CHARS:
        flat = flat[:INITIAL_PROMPT_MAX_CHARS].rsplit(" ", 1)[0]
    return f"Réunion en français. Termes attendus: {flat}."


def build_mlx_whisper_cmd(
    mlx_whisper_path: str,
    audio_path: str,
    output_path: str,
    model: str,
    language: str = "fr",
    output_format: str = "txt",
    initial_prompt: str = "",
) -> list[str]:
    out = Path(output_path)
    fmt = (output_format or "txt").strip().lower()
    cmd = [
        mlx_whisper_path,
        audio_path,
        "--model",
        model.strip() or "mlx-community/whisper-large-v3-turbo",
        "-f",
        fmt,
        "--output-dir",
        str(out.parent),
        "--output-name",
        out.stem,
    ]

    lang = (language or "fr").strip().lower()
    if lang and lang != "auto":
        cmd += ["--language", lang]

    prompt = (initial_prompt or "").strip()
    if prompt:
        cmd += ["--initial-prompt", prompt]

    return cmd


# --- Diarisation ----------------------------------------------------------
#
# We run pyannote.audio inside the same managed venv that hosts mlx_whisper.
# The script below is invoked via `<venv>/bin/python -c <script>`. It loads
# the diarisation pipeline, runs it on the WAV, and prints a single JSON
# blob to stdout. Keeping it inline avoids shipping a separate Python file
# and keeps the venv self-contained.


_DIARIZATION_SCRIPT = '''
import json
import os
import sys

audio_path = sys.argv[1]
hf_token = os.environ.get("HF_TOKEN", "")
if not hf_token:
    print(json.dumps({"error": "HF_TOKEN not set"}))
    sys.exit(2)

try:
    import torch
    from pyannote.audio import Pipeline
except Exception as exc:
    print(json.dumps({"error": f"import failed: {exc}"}))
    sys.exit(3)

try:
    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=hf_token,
        )
    except TypeError as exc:
        if "token" not in str(exc):
            raise
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token,
        )
    if pipeline is None:
        raise RuntimeError("pipeline unavailable; check Hugging Face token and accepted pyannote licenses")
except Exception as exc:
    print(json.dumps({"error": f"pipeline load failed: {exc}"}))
    sys.exit(4)

# Apple Silicon: prefer MPS when available, fall back to CPU.
try:
    if torch.backends.mps.is_available():
        pipeline.to(torch.device("mps"))
except Exception:
    pass

try:
    diar = pipeline(audio_path)
except Exception as exc:
    print(json.dumps({"error": f"diarization failed: {exc}"}))
    sys.exit(5)

turns = []
for turn, _, speaker in diar.itertracks(yield_label=True):
    turns.append({
        "start": float(turn.start),
        "end": float(turn.end),
        "speaker": str(speaker),
    })

print(json.dumps({"turns": turns}))
'''


def build_diarization_cmd(venv_python_path: str, wav_path: str) -> list[str]:
    return [venv_python_path, "-c", _DIARIZATION_SCRIPT, wav_path]


def parse_diarization_output(stdout: str) -> list[dict]:
    """
    Parses the JSON the diarisation script prints. Returns a list of
    {start, end, speaker} dicts. Raises RuntimeError on script-reported
    errors so the caller can surface them in the UI.
    """
    text = (stdout or "").strip()
    if not text:
        raise RuntimeError("Diarisation: sortie vide.")
    # The pyannote pipeline emits progress logs to stderr; stdout is reserved
    # for our JSON blob, but be defensive and pick the last line.
    last_line = text.splitlines()[-1].strip()
    try:
        payload = json.loads(last_line)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Diarisation: JSON invalide ({exc}).") from exc
    if "error" in payload:
        raise RuntimeError(f"Diarisation: {payload['error']}")
    return list(payload.get("turns", []))


def parse_whisper_json_segments(json_path: str) -> list[dict]:
    """
    MLX Whisper's --output-format json writes {"text": ..., "segments": [...],
    "language": ...}. We only care about the segments list.
    """
    raw = Path(json_path).read_text(encoding="utf-8")
    payload = json.loads(raw)
    segments = payload.get("segments") or []
    out = []
    for seg in segments:
        out.append({
            "start": float(seg.get("start", 0.0)),
            "end": float(seg.get("end", 0.0)),
            "text": str(seg.get("text", "")).strip(),
        })
    return out


def assign_speakers_to_segments(
    whisper_segments: Iterable[dict],
    diarization_turns: Iterable[dict],
) -> list[dict]:
    """
    For each Whisper segment, assign the speaker whose turn(s) overlap
    most with the segment. Segments with zero overlap stay unlabeled.

    Returns a new list of dicts with an added "speaker" key (or None).
    """
    turns = [
        (float(t["start"]), float(t["end"]), str(t["speaker"]))
        for t in diarization_turns
    ]
    out = []
    for seg in whisper_segments:
        s_start = float(seg["start"])
        s_end = float(seg["end"])
        per_speaker: dict[str, float] = {}
        for t_start, t_end, speaker in turns:
            overlap = max(0.0, min(s_end, t_end) - max(s_start, t_start))
            if overlap > 0:
                per_speaker[speaker] = per_speaker.get(speaker, 0.0) + overlap
        best_speaker = None
        if per_speaker:
            best_speaker = max(per_speaker.items(), key=lambda kv: kv[1])[0]
        new_seg = dict(seg)
        new_seg["speaker"] = best_speaker
        out.append(new_seg)
    return out


def _format_timestamp_srt(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    ms_total = int(round(seconds * 1000))
    hours, rem = divmod(ms_total, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def _format_timestamp_vtt(seconds: float) -> str:
    return _format_timestamp_srt(seconds).replace(",", ".")


def _format_timestamp_short(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    total = int(seconds)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def render_segments_with_speakers(segments: list[dict], output_format: str) -> str:
    """
    Renders speaker-labeled segments to txt/srt/vtt/json/tsv. Speaker labels
    come from pyannote (e.g. SPEAKER_00); the user can rename them in the
    transcript afterwards. Unlabeled segments fall back to "?".
    """
    fmt = (output_format or "txt").strip().lower()
    if fmt == "all":
        fmt = "txt"

    def label(seg: dict) -> str:
        return seg.get("speaker") or "?"

    if fmt == "json":
        return json.dumps({"segments": segments}, ensure_ascii=False, indent=2)

    if fmt == "tsv":
        lines = ["start\tend\tspeaker\ttext"]
        for seg in segments:
            lines.append(
                f"{seg['start']:.3f}\t{seg['end']:.3f}\t{label(seg)}\t{seg['text']}"
            )
        return "\n".join(lines) + "\n"

    if fmt == "srt":
        lines = []
        for i, seg in enumerate(segments, start=1):
            lines.append(str(i))
            lines.append(
                f"{_format_timestamp_srt(seg['start'])} --> {_format_timestamp_srt(seg['end'])}"
            )
            lines.append(f"[{label(seg)}] {seg['text']}")
            lines.append("")
        return "\n".join(lines)

    if fmt == "vtt":
        lines = ["WEBVTT", ""]
        for seg in segments:
            lines.append(
                f"{_format_timestamp_vtt(seg['start'])} --> {_format_timestamp_vtt(seg['end'])}"
            )
            lines.append(f"[{label(seg)}] {seg['text']}")
            lines.append("")
        return "\n".join(lines)

    # txt: one block per consecutive run of the same speaker, prefixed with
    # the speaker tag and the start timestamp. Easier for humans to read
    # than a per-segment dump.
    lines = []
    current_speaker: str | None = None
    current_start: float | None = None
    current_text: list[str] = []

    def flush():
        if current_text:
            lines.append(
                f"[{current_speaker or '?'}] ({_format_timestamp_short(current_start or 0)}) "
                + " ".join(current_text).strip()
            )

    for seg in segments:
        speaker = label(seg)
        if speaker != current_speaker:
            flush()
            current_speaker = speaker
            current_start = seg["start"]
            current_text = [seg["text"]]
        else:
            current_text.append(seg["text"])
    flush()
    return "\n".join(lines) + "\n"
