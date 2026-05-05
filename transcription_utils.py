from pathlib import Path


TRANSCRIPTION_AUDIO_FILTERS = [
    "highpass=f=80",
    "lowpass=f=7600",
    "acompressor=threshold=-20dB:ratio=2.2:attack=5:release=160",
    "loudnorm=I=-16:TP=-1.5:LRA=11",
]


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
