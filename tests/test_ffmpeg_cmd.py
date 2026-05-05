import json
import tempfile
import unittest
from pathlib import Path

import transcription_utils
from ffmpeg_utils import build_ffmpeg_cmd
from transcription_utils import (
    assign_speakers_to_segments,
    build_audio_extract_cmd,
    build_diarization_cmd,
    build_llm_cmd,
    build_mlx_whisper_cmd,
    clean_whisper_segments,
    default_transcript_path,
    parse_diarization_output,
    parse_whisper_json_segments,
    render_segments_plain,
    render_segments_with_speakers,
    structured_initial_prompt,
    suggest_transcript_stem,
)


class BuildFfmpegCmdTest(unittest.TestCase):
    def test_build_command_with_trim_and_audio_options(self):
        cmd = build_ffmpeg_cmd(
            ffmpeg_path="/usr/local/bin/ffmpeg",
            in_path="/tmp/in.mp4",
            out_path="/tmp/out.mp4",
            crf=30,
            resolution="480p",
            fps=10,
            audio_bitrate="96k",
            preset="veryfast",
            speech_enhance=True,
            mono_audio=True,
            ss="00:00:05",
            to="00:01:00",
        )

        self.assertIn("-ss", cmd)
        self.assertIn("00:00:05", cmd)
        self.assertIn("-to", cmd)
        self.assertIn("00:01:00", cmd)
        self.assertIn("-af", cmd)
        self.assertIn("-ac", cmd)
        self.assertIn("1", cmd)
        self.assertIn("-preset", cmd)
        self.assertIn("veryfast", cmd)
        self.assertIn("-crf", cmd)
        self.assertIn("30", cmd)
        self.assertIn("scale=-2:480,fps=10", cmd)
        self.assertEqual(cmd[-1], "/tmp/out.mp4")


class TranscriptionCommandTest(unittest.TestCase):
    def test_build_audio_extract_command_uses_original_audio(self):
        cmd = build_audio_extract_cmd(
            ffmpeg_path="/usr/local/bin/ffmpeg",
            in_path="/tmp/in.mp4",
            wav_path="/tmp/audio.wav",
            speech_enhance=True,
            ss="00:00:05",
            to="00:01:00",
        )

        self.assertIn("-vn", cmd)
        self.assertIn("-af", cmd)
        self.assertIn("pcm_s16le", cmd)
        self.assertIn("16000", cmd)
        self.assertIn("-ac", cmd)
        self.assertIn("1", cmd)
        self.assertEqual(cmd[-1], "/tmp/audio.wav")

    def test_build_mlx_whisper_command_uses_context_prompt(self):
        cmd = build_mlx_whisper_cmd(
            mlx_whisper_path="/opt/homebrew/bin/mlx_whisper",
            audio_path="/tmp/audio.wav",
            output_path="/tmp/reunion_transcription.srt",
            model="mlx-community/whisper-large-v3-turbo",
            language="fr",
            output_format="srt",
            initial_prompt="Noms propres: Ekonum, Maia.",
        )

        self.assertEqual(cmd[0], "/opt/homebrew/bin/mlx_whisper")
        self.assertIn("--model", cmd)
        self.assertIn("mlx-community/whisper-large-v3-turbo", cmd)
        self.assertIn("-f", cmd)
        self.assertIn("srt", cmd)
        self.assertIn("--language", cmd)
        self.assertIn("fr", cmd)
        self.assertIn("--initial-prompt", cmd)
        self.assertIn("--condition-on-previous-text", cmd)
        self.assertIn("False", cmd)

    def test_build_mlx_whisper_command_can_clip_audio(self):
        cmd = build_mlx_whisper_cmd(
            mlx_whisper_path="/opt/homebrew/bin/mlx_whisper",
            audio_path="/tmp/audio.wav",
            output_path="/tmp/reunion.json",
            model="mlx-community/whisper-large-v3-turbo",
            clip_timestamps="600,900",
        )

        self.assertIn("--clip-timestamps", cmd)
        self.assertIn("600,900", cmd)

    def test_build_llm_command_passes_glossary(self):
        cmd = build_llm_cmd(
            "/tmp/venv/bin/python",
            "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
            "/tmp/transcript.txt",
            "DKV, Odoo, CVR Contrôles",
        )

        self.assertEqual(cmd[0], "/tmp/venv/bin/python")
        self.assertEqual(cmd[-1], "DKV, Odoo, CVR Contrôles")
        self.assertIn("corrections", cmd[2])

    def test_inline_llm_script_is_valid_python(self):
        compile(transcription_utils._LLM_POST_PROCESS_SCRIPT, "<llm-post-process>", "exec")

    def test_default_transcript_path_uses_format_extension(self):
        path = default_transcript_path("/tmp/reunion.mp4", "/tmp", "_notes", "json")
        self.assertEqual(path, "/tmp/reunion_notes.json")

    def test_default_transcript_path_allows_no_suffix(self):
        path = default_transcript_path("/tmp/reunion.mp4", "/tmp", "", "txt")
        self.assertEqual(path, "/tmp/reunion.txt")


class StructuredInitialPromptTest(unittest.TestCase):
    def test_empty_context_returns_empty(self):
        self.assertEqual(structured_initial_prompt(""), "")
        self.assertEqual(structured_initial_prompt("   "), "")

    def test_wraps_terms_in_french_priming_sentence(self):
        prompt = structured_initial_prompt("Ekonum, MAIA, RGPD")
        self.assertTrue(prompt.startswith("Réunion en français"))
        self.assertIn("Ekonum", prompt)
        self.assertIn("MAIA", prompt)
        self.assertIn("RGPD", prompt)

    def test_collapses_whitespace(self):
        prompt = structured_initial_prompt("Ekonum,   MAIA\n\nRGPD")
        self.assertNotIn("\n", prompt)
        self.assertNotIn("  ", prompt)


class TranscriptTitleTest(unittest.TestCase):
    def test_suggest_transcript_stem_uses_topic_sentence(self):
        text = "Bonjour à tous. Aujourd'hui on va parler de Présentation des outils RH pour les équipes."
        self.assertEqual(
            suggest_transcript_stem(text, "Capture ecran"),
            "Présentation des outils RH pour les équipes",
        )

    def test_suggest_transcript_stem_falls_back_for_generic_opening(self):
        self.assertEqual(suggest_transcript_stem("Bonjour. Merci.", "Réunion client"), "Réunion client")


class DiarizationCommandTest(unittest.TestCase):
    def test_build_diarization_cmd_includes_audio_path(self):
        cmd = build_diarization_cmd("/path/to/venv/bin/python", "/tmp/audio.wav")
        self.assertEqual(cmd[0], "/path/to/venv/bin/python")
        self.assertEqual(cmd[1], "-c")
        self.assertEqual(cmd[-1], "/tmp/audio.wav")
        # The inline script must reference the diarisation pipeline by ID.
        self.assertIn("speaker-diarization-3.1", cmd[2])

    def test_parse_diarization_output_handles_well_formed_json(self):
        payload = json.dumps({"turns": [
            {"start": 0.0, "end": 1.5, "speaker": "SPEAKER_00"},
            {"start": 1.5, "end": 3.0, "speaker": "SPEAKER_01"},
        ]})
        turns = parse_diarization_output(payload)
        self.assertEqual(len(turns), 2)
        self.assertEqual(turns[0]["speaker"], "SPEAKER_00")

    def test_parse_diarization_output_picks_last_line(self):
        # pyannote can leak progress lines to stdout; we only want the JSON.
        text = "loading model...\nrunning inference...\n" + json.dumps({"turns": []})
        self.assertEqual(parse_diarization_output(text), [])

    def test_parse_diarization_output_raises_on_script_error(self):
        with self.assertRaises(RuntimeError):
            parse_diarization_output(json.dumps({"error": "HF_TOKEN not set"}))

    def test_parse_diarization_output_raises_on_empty(self):
        with self.assertRaises(RuntimeError):
            parse_diarization_output("")


class FuseAndRenderTest(unittest.TestCase):
    def test_assign_speakers_picks_max_overlap(self):
        whisper = [
            {"start": 0.0, "end": 2.0, "text": "Bonjour à tous"},
            {"start": 2.0, "end": 4.0, "text": "Merci de me recevoir"},
            {"start": 10.0, "end": 11.0, "text": "Aucun chevauchement"},
        ]
        diar = [
            {"start": 0.0, "end": 1.5, "speaker": "SPEAKER_00"},
            {"start": 1.5, "end": 4.0, "speaker": "SPEAKER_01"},
        ]
        out = assign_speakers_to_segments(whisper, diar)
        self.assertEqual(out[0]["speaker"], "SPEAKER_00")  # 1.5s vs 0.5s
        self.assertEqual(out[1]["speaker"], "SPEAKER_01")  # full overlap
        self.assertIsNone(out[2]["speaker"])  # no diarisation turn

    def test_render_txt_groups_consecutive_same_speaker(self):
        segs = [
            {"start": 0.0, "end": 2.0, "text": "Bonjour", "speaker": "SPEAKER_00"},
            {"start": 2.0, "end": 4.0, "text": "à tous", "speaker": "SPEAKER_00"},
            {"start": 4.0, "end": 6.0, "text": "Salut", "speaker": "SPEAKER_01"},
        ]
        out = render_segments_with_speakers(segs, "txt")
        # Two blocks total, not three — the first two segments share a speaker.
        self.assertEqual(out.count("[SPEAKER_00]"), 1)
        self.assertEqual(out.count("[SPEAKER_01]"), 1)
        self.assertIn("Bonjour à tous", out)

    def test_render_srt_emits_speaker_prefix_and_timestamps(self):
        segs = [
            {"start": 0.0, "end": 1.5, "text": "Salut", "speaker": "SPEAKER_00"},
            {"start": 1.5, "end": 3.0, "text": "Bonjour", "speaker": "SPEAKER_01"},
        ]
        out = render_segments_with_speakers(segs, "srt")
        self.assertIn("00:00:00,000 --> 00:00:01,500", out)
        self.assertIn("[SPEAKER_00] Salut", out)
        self.assertIn("[SPEAKER_01] Bonjour", out)

    def test_render_unlabeled_segments_fall_back_to_question_mark(self):
        segs = [{"start": 0.0, "end": 1.0, "text": "Hum", "speaker": None}]
        out = render_segments_with_speakers(segs, "srt")
        self.assertIn("[?] Hum", out)

    def test_parse_whisper_json_segments_round_trips(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "out.json"
            p.write_text(json.dumps({
                "text": "Bonjour à tous. Merci.",
                "segments": [
                    {"start": 0.0, "end": 2.0, "text": " Bonjour à tous."},
                    {"start": 2.0, "end": 3.5, "text": " Merci."},
                ],
                "language": "fr",
            }), encoding="utf-8")
            segs = parse_whisper_json_segments(str(p))
        self.assertEqual(len(segs), 2)
        self.assertEqual(segs[0]["text"], "Bonjour à tous.")  # stripped
        self.assertEqual(segs[1]["start"], 2.0)

    def test_clean_whisper_segments_drops_ellipsis_hallucinations(self):
        segs = clean_whisper_segments([
            {"start": 0.0, "end": 2.0, "text": " ...", "compression_ratio": 3.6},
            {"start": 30.0, "end": 32.0, "text": "Sous-titrage ST' 501"},
            {"start": 117.0, "end": 119.0, "text": "Bonjour tous les deux."},
        ])

        self.assertEqual(len(segs), 1)
        self.assertEqual(segs[0]["text"], "Bonjour tous les deux.")

    def test_clean_whisper_segments_limits_decoder_loops(self):
        repeated = [
            {"start": float(i), "end": float(i + 1), "text": "C'est pour tout ce qui est avant-vente."}
            for i in range(5)
        ]

        self.assertEqual(len(clean_whisper_segments(repeated)), 2)

    def test_render_plain_segments_has_no_speaker_placeholder(self):
        out = render_segments_plain([
            {"start": 0.0, "end": 2.0, "text": "Bonjour"},
            {"start": 2.0, "end": 4.0, "text": "à tous"},
        ], "txt")

        self.assertEqual(out, "Bonjour\nà tous\n")


if __name__ == "__main__":
    unittest.main()
