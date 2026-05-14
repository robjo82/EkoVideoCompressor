"""
Silero VAD wrapper that runs inside the managed venv.

We never want to ship torch's bytes into our `transcription_utils.py`
module (a Qt-free import surface) or into the main process. Instead,
the wrapper here builds a subprocess command that runs an inline
Python script in the same venv we already use for pyannote +
mlx_lm. That script:

  1. Loads the user's WAV (already 16 kHz mono PCM thanks to our
     ffmpeg extract step).
  2. Runs Silero VAD to detect speech regions.
  3. Writes a *trimmed* WAV that concatenates only the speech.
  4. Prints a JSON manifest mapping the original timeline to the
     trimmed timeline, so the caller can shift Whisper's timestamps
     back to the source media's timeline.

Why this matters:
  * On phone calls, the first ~2 minutes is IVR / hold music. Whisper
    spends compute on it and tends to hallucinate "Merci de rester
    en ligne…" loops. Pre-VAD cuts that to near-zero compute on the
    non-speech audio.
  * Skipping pure silence shaves ~20-30% of the transcription time
    on enterprise recordings and removes a whole class of
    hallucinations on long inter-segment gaps.
"""

from __future__ import annotations

import json
from pathlib import Path


__all__ = [
    "build_vad_cmd",
    "parse_vad_manifest",
    "remap_segment_to_source",
    "remap_segments_to_source",
]


# Inline subprocess script. Kept here (not as a separate .py inside
# the venv) so the wrapper stays self-contained and survives the
# PyInstaller bundle.
_VAD_SCRIPT = '''
import json
import sys
import wave

try:
    import torch
    from silero_vad import load_silero_vad, get_speech_timestamps, read_audio
except Exception as exc:  # pragma: no cover — venv probe path
    print(json.dumps({"error": f"vad import failed: {exc}"}))
    sys.exit(2)


in_path = sys.argv[1]
out_path = sys.argv[2]
# Soft tunables that we let the caller override via argv if needed.
min_speech_ms = int(sys.argv[3]) if len(sys.argv) > 3 else 250
min_silence_ms = int(sys.argv[4]) if len(sys.argv) > 4 else 500
pad_ms = int(sys.argv[5]) if len(sys.argv) > 5 else 200
threshold = float(sys.argv[6]) if len(sys.argv) > 6 else 0.5
sample_rate = 16000

try:
    audio = read_audio(in_path, sampling_rate=sample_rate)
except Exception as exc:
    print(json.dumps({"error": f"vad read_audio failed: {exc}"}))
    sys.exit(3)

try:
    model = load_silero_vad()
except Exception as exc:
    print(json.dumps({"error": f"vad load_silero_vad failed: {exc}"}))
    sys.exit(4)

try:
    spans = get_speech_timestamps(
        audio,
        model,
        sampling_rate=sample_rate,
        min_speech_duration_ms=min_speech_ms,
        min_silence_duration_ms=min_silence_ms,
        speech_pad_ms=pad_ms,
        threshold=threshold,
        return_seconds=True,
    )
except Exception as exc:
    print(json.dumps({"error": f"vad inference failed: {exc}"}))
    sys.exit(5)

if not spans:
    # Nothing detected as speech — pass the original audio through
    # so the caller can still try Whisper on the raw WAV.
    print(json.dumps({"spans": [], "total_seconds": float(len(audio)) / sample_rate}))
    sys.exit(0)

# Concatenate the speech regions into a new WAV file. Each kept span
# is written contiguously; the manifest records (orig_start, length)
# so the caller can map any timestamp inside the trimmed file back to
# its source position.
manifest = []
kept_samples = []
cursor_seconds = 0.0
for span in spans:
    start_sec = float(span["start"])
    end_sec = float(span["end"])
    s = int(start_sec * sample_rate)
    e = int(end_sec * sample_rate)
    e = min(e, len(audio))
    if e <= s:
        continue
    chunk = audio[s:e]
    kept_samples.append(chunk)
    duration = (e - s) / sample_rate
    manifest.append({
        "src_start": start_sec,
        "src_end": end_sec,
        "trim_start": cursor_seconds,
        "trim_end": cursor_seconds + duration,
    })
    cursor_seconds += duration

if not kept_samples:
    print(json.dumps({"spans": [], "total_seconds": float(len(audio)) / sample_rate}))
    sys.exit(0)

trimmed = torch.cat(kept_samples)
# Silero gives a float32 tensor in [-1, 1]; write 16-bit PCM.
pcm = (trimmed.clamp(-1, 1) * 32767).to(torch.int16).numpy().tobytes()
with wave.open(out_path, "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(sample_rate)
    wf.writeframes(pcm)

print(json.dumps({
    "spans": manifest,
    "total_seconds": float(len(audio)) / sample_rate,
    "trimmed_seconds": cursor_seconds,
}))
'''


def build_vad_cmd(
    venv_python_path: str,
    in_path: str,
    out_path: str,
    *,
    min_speech_ms: int = 250,
    min_silence_ms: int = 500,
    pad_ms: int = 200,
    threshold: float = 0.5,
) -> list[str]:
    """
    Subprocess command that runs Silero VAD on ``in_path`` and writes
    the speech-only trimmed WAV to ``out_path``. The script's stdout
    carries the JSON manifest the caller uses to remap timestamps.
    """
    return [
        venv_python_path,
        "-c",
        _VAD_SCRIPT,
        in_path,
        out_path,
        str(min_speech_ms),
        str(min_silence_ms),
        str(pad_ms),
        f"{threshold:.3f}",
    ]


def parse_vad_manifest(stdout: str) -> dict:
    """
    Parse the JSON manifest printed by the VAD script. Tolerant of
    leading progress logs (e.g. torch.hub messages on first load) —
    we keep the last JSON line.
    """
    text = (stdout or "").strip()
    if not text:
        raise RuntimeError("VAD: sortie vide.")
    last = text.splitlines()[-1].strip()
    try:
        payload = json.loads(last)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"VAD: JSON invalide ({exc}).") from exc
    if "error" in payload:
        raise RuntimeError(f"VAD: {payload['error']}")
    return payload


def remap_segment_to_source(
    trim_start: float,
    trim_end: float,
    manifest: list[dict],
) -> tuple[float, float]:
    """
    Given a (start, end) pair on the *trimmed* timeline (i.e. what
    Whisper sees), return the corresponding (start, end) on the
    *source* media timeline. If the span straddles multiple manifest
    entries we map endpoints independently — Whisper segments rarely
    cross VAD boundaries because the trimmed audio is continuous
    speech.
    """
    if not manifest:
        return trim_start, trim_end

    def _shift(t: float) -> float:
        # Walk the manifest to find the entry containing ``t`` on the
        # trimmed timeline; offset by (src_start - trim_start).
        for span in manifest:
            ts_start = float(span["trim_start"])
            ts_end = float(span["trim_end"])
            if ts_start <= t <= ts_end + 1e-3:
                return float(span["src_start"]) + (t - ts_start)
        # Past the last span — clamp to the last source end so we
        # never produce a timestamp outside the file.
        last = manifest[-1]
        return float(last["src_end"])

    return _shift(trim_start), _shift(trim_end)


def remap_segments_to_source(
    segments: list[dict],
    manifest: list[dict],
) -> list[dict]:
    """
    Bulk version: clones each segment and rewrites its ``start`` /
    ``end`` to the source timeline. Returns the new list. Segments
    without ``start`` / ``end`` (e.g. plain text passes) are returned
    unchanged.
    """
    if not segments or not manifest:
        return [dict(s) for s in segments]
    out: list[dict] = []
    for seg in segments:
        try:
            s = float(seg.get("start"))
            e = float(seg.get("end"))
        except (TypeError, ValueError):
            out.append(dict(seg))
            continue
        new_s, new_e = remap_segment_to_source(s, e, manifest)
        new_seg = dict(seg)
        new_seg["start"] = new_s
        new_seg["end"] = new_e
        out.append(new_seg)
    return out
