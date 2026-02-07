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
) -> list[str]:
    cmd = [ffmpeg_path, "-y", "-hide_banner"]

    if ss is not None:
        cmd += ["-ss", ss]
    if to is not None:
        cmd += ["-to", to]

    cmd += ["-i", in_path]

    vf = []
    if resolution == "1080p":
        vf.append("scale=-2:1080")
    elif resolution == "720p":
        vf.append("scale=-2:720")
    elif resolution == "480p":
        vf.append("scale=-2:480")
    vf.append(f"fps={fps}")
    cmd += ["-vf", ",".join(vf)]

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
