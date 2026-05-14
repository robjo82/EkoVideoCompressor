from pathlib import Path


# Inputs accepted by the file picker / drag-drop. We keep these as the
# single source of truth so the GUI filter, the queue rejector and the
# audio-detection helper agree on what's a media file.
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".m4v", ".avi", ".webm"}
AUDIO_EXTENSIONS = {
    ".mp3", ".m4a", ".aac", ".wav", ".flac", ".ogg", ".oga",
    ".opus", ".wma", ".aiff", ".aif",
}
MEDIA_EXTENSIONS = VIDEO_EXTENSIONS | AUDIO_EXTENSIONS

MEDIA_FILTER = (
    "Médias ("
    + " ".join(f"*{ext}" for ext in sorted(MEDIA_EXTENSIONS))
    + ");;Vidéos ("
    + " ".join(f"*{ext}" for ext in sorted(VIDEO_EXTENSIONS))
    + ");;Audios ("
    + " ".join(f"*{ext}" for ext in sorted(AUDIO_EXTENSIONS))
    + ");;Tous les fichiers (*.*)"
)


def is_audio_only_path(path: str) -> bool:
    """
    True if the file's extension is one of our known audio-only formats.
    This is the cheap pre-filter the EncodeWorker uses to skip the H.265
    pipeline; ffmpeg will catch a misnamed `.m4a` that's actually video.
    """
    return Path(path).suffix.lower() in AUDIO_EXTENSIONS


def default_out_path(in_path: str, out_dir: str, suffix: str) -> str:
    """
    Build a unique output path next to `out_dir` based on the input
    stem + a user suffix (defaults to "_compressed"). Audio-only inputs
    come back as `.m4a` (AAC container) so the user doesn't end up
    with a 0-byte video stream wrapped in `.mp4`.
    """
    source = Path(in_path)
    safe_suffix = suffix.strip() or "_compressed"
    ext = ".m4a" if is_audio_only_path(in_path) else ".mp4"
    base = Path(out_dir) / f"{source.stem}{safe_suffix}{ext}"
    out = base
    i = 1
    while out.exists():
        out = Path(out_dir) / f"{source.stem}{safe_suffix}_{i}{ext}"
        i += 1
    return str(out)


def build_speaker_concat_cmd(
    ffmpeg_path: str,
    in_wav: str,
    out_wav: str,
    spans: list[tuple[float, float]],
) -> list[str]:
    """
    Build an ffmpeg command that selects + concatenates a list of
    ``(start, end)`` time ranges from ``in_wav`` into a single
    ``out_wav``. Used by the per-speaker pipeline: for each
    diarised speaker we hand Whisper a stream that contains *only*
    their voice, so the decoder context doesn't cross speaker
    boundaries.

    The ``aselect`` filter chains range expressions joined with `+`
    so ffmpeg keeps samples whose timestamp falls inside any span,
    then ``asetpts=N/SR/TB`` resets the timeline so the output is a
    contiguous WAV starting at t=0.
    """
    if not spans:
        raise ValueError("build_speaker_concat_cmd needs at least one span")
    parts = [
        f"between(t,{start:.3f},{end:.3f})"
        for start, end in spans
        if end > start
    ]
    if not parts:
        raise ValueError("build_speaker_concat_cmd needs at least one valid span")
    expr = "+".join(parts)
    afilter = f"aselect='{expr}',asetpts=N/SR/TB"
    return [
        ffmpeg_path,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        in_wav,
        "-af",
        afilter,
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        out_wav,
    ]


def build_ffmpeg_cmd(
    ffmpeg_path: str,
    in_path: str,
    out_path: str,
    crf: int = 28,
    resolution: str = "720p",
    fps: int = 12,
    audio_bitrate: str = "128k",
    preset: str = "medium",
    speech_enhance: bool = False,
    mono_audio: bool = False,
    ss: str | None = None,
    to: str | None = None,
    audio_only: bool = False,
) -> list[str]:
    cmd = [ffmpeg_path, "-y", "-hide_banner"]

    if ss is not None:
        cmd += ["-ss", ss]
    if to is not None:
        cmd += ["-to", to]

    cmd += ["-i", in_path]

    af = []
    if speech_enhance:
        af.extend(
            [
                "highpass=f=120",
                "lowpass=f=7000",
                "acompressor=threshold=-18dB:ratio=2.5:attack=5:release=120",
                "loudnorm=I=-16:TP=-1.5:LRA=11",
            ]
        )

    # Audio-only path: drop every video flag and write a clean .m4a.
    # Keeping the same audio filters / bitrate / mono toggle means the
    # user's chosen profile still applies, just without video re-encode.
    if audio_only:
        cmd += ["-vn"]
        if af:
            cmd += ["-af", ",".join(af)]
        cmd += ["-c:a", "aac", "-b:a", audio_bitrate]
        if mono_audio:
            cmd += ["-ac", "1"]
        cmd += ["-movflags", "+faststart", "-progress", "pipe:1", "-nostats", out_path]
        return cmd

    vf = []
    if resolution == "1080p":
        vf.append("scale=-2:1080")
    elif resolution == "720p":
        vf.append("scale=-2:720")
    elif resolution == "480p":
        vf.append("scale=-2:480")
    vf.append(f"fps={fps}")
    cmd += ["-vf", ",".join(vf)]

    if af:
        cmd += ["-af", ",".join(af)]

    cmd += [
        "-c:v",
        "libx265",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-tag:v",
        "hvc1",
        "-c:a",
        "aac",
        "-b:a",
        audio_bitrate,
        "-movflags",
        "+faststart",
    ]
    if mono_audio:
        cmd += ["-ac", "1"]

    cmd += ["-progress", "pipe:1", "-nostats", out_path]
    return cmd
