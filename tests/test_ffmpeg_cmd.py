import json
import tempfile
import unittest
from pathlib import Path

import transcription_utils
from ffmpeg_utils import (
    AUDIO_EXTENSIONS,
    MEDIA_EXTENSIONS,
    MEDIA_FILTER,
    VIDEO_EXTENSIONS,
    build_ffmpeg_cmd,
    build_speaker_concat_cmd,
    default_out_path,
    is_audio_only_path,
)
from transcription_utils import (
    AUDIO_LLM_MODELS,
    DEFAULT_AUDIO_LLM_MODEL,
    DEFAULT_TEXT_LLM_MODEL,
    DEFAULT_WHISPER_MODEL,
    TEXT_LLM_MODELS,
    WHISPER_MODELS,
    assign_speakers_to_segments,
    audio_llm_label_for,
    build_audio_extract_cmd,
    build_diarization_cmd,
    build_llm_cmd,
    build_llm_corrections_cmd,
    build_llm_title_cmd,
    build_mlx_whisper_cmd,
    build_multimodal_audio_cmd,
    canonical_audio_llm_model_id,
    canonical_whisper_model_id,
    clean_whisper_segments,
    default_transcript_path,
    filter_speaker_names_by_context,
    is_phone_hold_boilerplate_text,
    parse_diarization_output,
    parse_llm_corrections_markdown,
    parse_llm_title_speakers,
    parse_whisper_json_segments,
    render_segments_plain,
    render_segments_with_speakers,
    structured_initial_prompt,
    suggest_transcript_stem,
    text_llm_label_for,
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

    def test_audio_only_command_skips_video_options(self):
        cmd = build_ffmpeg_cmd(
            ffmpeg_path="/usr/local/bin/ffmpeg",
            in_path="/tmp/in.m4a",
            out_path="/tmp/out.m4a",
            audio_bitrate="96k",
            speech_enhance=True,
            mono_audio=True,
            audio_only=True,
        )

        # No video stream → -vn must appear and the H.265/scale knobs
        # must NOT be in the command. Otherwise ffmpeg fails immediately.
        self.assertIn("-vn", cmd)
        self.assertNotIn("libx265", cmd)
        self.assertNotIn("-c:v", cmd)
        self.assertNotIn("-vf", cmd)
        # Audio knobs are still respected.
        self.assertIn("-c:a", cmd)
        self.assertIn("aac", cmd)
        self.assertIn("-b:a", cmd)
        self.assertIn("96k", cmd)
        self.assertIn("-ac", cmd)
        self.assertEqual(cmd[-1], "/tmp/out.m4a")

    def test_audio_only_command_respects_trim_window(self):
        cmd = build_ffmpeg_cmd(
            ffmpeg_path="/usr/local/bin/ffmpeg",
            in_path="/tmp/in.mp3",
            out_path="/tmp/out.m4a",
            ss="00:01:00",
            to="00:02:00",
            audio_only=True,
        )

        self.assertIn("-ss", cmd)
        self.assertIn("00:01:00", cmd)
        self.assertIn("-to", cmd)
        self.assertIn("00:02:00", cmd)


class BuildSpeakerConcatCmdTest(unittest.TestCase):
    def test_aselect_filter_carries_every_span(self):
        cmd = build_speaker_concat_cmd(
            "/usr/local/bin/ffmpeg",
            "/tmp/in.wav",
            "/tmp/speaker_S0.wav",
            [(10.0, 15.0), (50.5, 55.0)],
        )
        # The aselect filter must reference both ranges, joined with
        # a `+` (logical OR). asetpts resets the timeline.
        af = cmd[cmd.index("-af") + 1]
        self.assertIn("between(t,10.000,15.000)", af)
        self.assertIn("between(t,50.500,55.000)", af)
        self.assertIn("+", af)
        self.assertIn("asetpts=N/SR/TB", af)
        self.assertIn("pcm_s16le", cmd)
        self.assertIn("16000", cmd)
        self.assertEqual(cmd[-1], "/tmp/speaker_S0.wav")

    def test_raises_on_empty_spans(self):
        with self.assertRaises(ValueError):
            build_speaker_concat_cmd("ffmpeg", "/i.wav", "/o.wav", [])

    def test_drops_invalid_spans(self):
        # end <= start should be skipped, not crash.
        cmd = build_speaker_concat_cmd(
            "ffmpeg", "/i.wav", "/o.wav", [(0.0, 0.0), (5.0, 10.0), (20.0, 19.0)]
        )
        af = cmd[cmd.index("-af") + 1]
        self.assertIn("between(t,5.000,10.000)", af)
        self.assertNotIn("between(t,0.000,0.000)", af)
        self.assertNotIn("between(t,20.000,19.000)", af)


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
        # The legacy alias now points at the title/speakers script.
        self.assertIn("speakers", cmd[2])

    def test_inline_llm_scripts_are_valid_python(self):
        compile(transcription_utils._LLM_TITLE_SPEAKERS_SCRIPT, "<llm-title>", "exec")
        compile(transcription_utils._LLM_CORRECTIONS_SCRIPT, "<llm-corrections>", "exec")

    def test_build_multimodal_audio_command_carries_model_and_clip(self):
        cmd = build_multimodal_audio_cmd(
            venv_python_path="/tmp/venv/bin/python",
            model_path="mlx-community/Qwen2-Audio-7B-Instruct-4bit",
            audio_path="/tmp/clip.wav",
            prompt="Que dit la personne ?",
        )

        self.assertEqual(cmd[0], "/tmp/venv/bin/python")
        self.assertEqual(cmd[1], "-c")
        # The script must reference the audio entry-point so the venv knows
        # to pull in mlx_vlm rather than mlx_lm.
        self.assertIn("mlx_vlm", cmd[2])
        self.assertEqual(cmd[3], "mlx-community/Qwen2-Audio-7B-Instruct-4bit")
        self.assertEqual(cmd[4], "/tmp/clip.wav")
        self.assertEqual(cmd[5], "Que dit la personne ?")

    def test_inline_multimodal_script_is_valid_python(self):
        compile(transcription_utils._MULTIMODAL_AUDIO_SCRIPT, "<multimodal-audio>", "exec")

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

    def test_wraps_terms_in_contextual_french_sentence(self):
        # The new prompt opens with a meeting greeting so the Whisper
        # decoder treats what follows as plausible French dialogue —
        # the LM bias is much stronger than with a bare term list.
        prompt = structured_initial_prompt("Ekonum, MAIA, RGPD")
        self.assertTrue(prompt.startswith("Bonjour."))
        self.assertIn("Ekonum", prompt)
        self.assertIn("MAIA", prompt)
        self.assertIn("RGPD", prompt)

    def test_quotes_multi_word_terms_with_guillemets(self):
        # Multi-word terms must reach Whisper as a single phonetic
        # unit, not as two separately-decoded words.
        prompt = structured_initial_prompt("Mollie, CVR Contrôles")
        self.assertIn("« CVR Contrôles »", prompt)
        self.assertIn("Mollie", prompt)

    def test_handles_long_glossary_via_trailing_section(self):
        # 15 terms — more than the 4 main clauses can absorb (each
        # takes ~3). The tail goes into a final "D'autres noms…".
        terms = ", ".join(f"Term{i}" for i in range(15))
        prompt = structured_initial_prompt(terms)
        self.assertIn("D'autres noms à respecter", prompt)
        for i in range(15):
            self.assertIn(f"Term{i}", prompt)


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

    def test_clean_whisper_segments_drops_phone_hold_boilerplate(self):
        text = (
            "Bienvenue à Symphonat. Toute l'équipe vous remercie de votre appel "
            "et vous invite à patienter quelques instants. "
            "Un correspondant va vous répondre. Merci de rester en ligne."
        )
        self.assertTrue(is_phone_hold_boilerplate_text(text))
        self.assertEqual(clean_whisper_segments([{"text": text, "compression_ratio": 1.0}]), [])

    def test_phone_hold_filter_keeps_real_business_speech(self):
        text = (
            "Oui bonjour Monsieur Maire, je vous rappelais pour savoir "
            "dans quelle mesure vous utilisez Odoo avec Mollie."
        )
        self.assertFalse(is_phone_hold_boilerplate_text(text))
        self.assertEqual(len(clean_whisper_segments([{"text": text, "compression_ratio": 1.0}])), 1)

    def test_render_plain_segments_has_no_speaker_placeholder(self):
        out = render_segments_plain([
            {"start": 0.0, "end": 2.0, "text": "Bonjour"},
            {"start": 2.0, "end": 4.0, "text": "à tous"},
        ], "txt")

        self.assertEqual(out, "Bonjour\nà tous\n")


class AudioOnlyDetectionTest(unittest.TestCase):
    def test_video_extensions_are_recognised(self):
        for ext in VIDEO_EXTENSIONS:
            self.assertFalse(is_audio_only_path(f"/tmp/file{ext}"), ext)

    def test_audio_extensions_are_recognised(self):
        for ext in AUDIO_EXTENSIONS:
            self.assertTrue(is_audio_only_path(f"/tmp/file{ext}"), ext)

    def test_audio_detection_is_case_insensitive(self):
        self.assertTrue(is_audio_only_path("/tmp/PODCAST.MP3"))
        self.assertTrue(is_audio_only_path("/tmp/Meeting.M4A"))

    def test_extensions_sets_are_disjoint(self):
        self.assertFalse(VIDEO_EXTENSIONS & AUDIO_EXTENSIONS)
        self.assertEqual(VIDEO_EXTENSIONS | AUDIO_EXTENSIONS, MEDIA_EXTENSIONS)

    def test_media_filter_advertises_both_categories(self):
        self.assertIn("Vidéos", MEDIA_FILTER)
        self.assertIn("Audios", MEDIA_FILTER)
        self.assertIn("*.mp4", MEDIA_FILTER)
        self.assertIn("*.mp3", MEDIA_FILTER)

    def test_default_out_path_keeps_mp4_for_video(self):
        with tempfile.TemporaryDirectory() as d:
            out = default_out_path("/tmp/in.mov", d, "_compressed")
        self.assertTrue(out.endswith("_compressed.mp4"))

    def test_default_out_path_uses_m4a_for_audio(self):
        with tempfile.TemporaryDirectory() as d:
            out = default_out_path("/tmp/dictaphone.wav", d, "_compressed")
        self.assertTrue(out.endswith("_compressed.m4a"), out)

    def test_default_out_path_increments_when_target_exists(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "meeting_compressed.m4a").touch()
            out = default_out_path("/tmp/meeting.mp3", d, "_compressed")
        self.assertTrue(out.endswith("_compressed_1.m4a"), out)


class LlmCatalogTest(unittest.TestCase):
    """
    The two catalogues power the Settings dropdowns. The data needs to stay
    well-structured: an unparseable label or a missing id silently breaks
    the UI without raising at import time.
    """

    def test_text_catalog_has_required_keys(self):
        self.assertGreaterEqual(len(TEXT_LLM_MODELS), 2)
        for entry in TEXT_LLM_MODELS:
            self.assertIn("id", entry)
            self.assertIn("label", entry)
            self.assertIn("family", entry)
            self.assertTrue(entry["id"].startswith("mlx-community/"))

    def test_whisper_catalog_has_default_model(self):
        ids = {entry["id"] for entry in WHISPER_MODELS}
        self.assertIn(DEFAULT_WHISPER_MODEL, ids)
        self.assertIn("turbo", DEFAULT_WHISPER_MODEL)
        self.assertIn("mlx-community/whisper-large-v3-mlx", ids)
        self.assertNotIn("mlx-community/whisper-large-v3", ids)

    def test_whisper_legacy_model_ids_are_canonicalized(self):
        self.assertEqual(
            canonical_whisper_model_id("mlx-community/whisper-large-v3"),
            "mlx-community/whisper-large-v3-mlx",
        )
        self.assertEqual(
            canonical_whisper_model_id("mlx-community/whisper-medium"),
            "mlx-community/whisper-medium-mlx",
        )

    def test_audio_legacy_model_ids_are_canonicalized(self):
        self.assertEqual(
            canonical_audio_llm_model_id("mlx-community/Qwen2-Audio-7B-Instruct-8bit"),
            "mlx-community/Qwen2-Audio-7B-Instruct-4bit",
        )

    def test_audio_catalog_has_required_keys(self):
        self.assertGreaterEqual(len(AUDIO_LLM_MODELS), 1)
        for entry in AUDIO_LLM_MODELS:
            self.assertIn("id", entry)
            self.assertIn("label", entry)
            self.assertIn("family", entry)
            # The audio catalog must point at multimodal-capable repos.
            self.assertIn("Audio", entry["id"])
            self.assertNotIn("8bit", entry["id"])

    def test_text_and_audio_catalogs_are_disjoint(self):
        text_ids = {e["id"] for e in TEXT_LLM_MODELS}
        audio_ids = {e["id"] for e in AUDIO_LLM_MODELS}
        # Mixing them up is exactly the bug we're guarding against.
        self.assertFalse(text_ids & audio_ids)

    def test_default_models_are_present_in_their_catalogs(self):
        text_ids = {e["id"] for e in TEXT_LLM_MODELS}
        audio_ids = {e["id"] for e in AUDIO_LLM_MODELS}
        self.assertIn(DEFAULT_TEXT_LLM_MODEL, text_ids)
        self.assertIn(DEFAULT_AUDIO_LLM_MODEL, audio_ids)

    def test_label_helpers_return_humanised_label_for_known_id(self):
        self.assertIn("Mistral", text_llm_label_for(DEFAULT_TEXT_LLM_MODEL))
        self.assertIn("Qwen", audio_llm_label_for(DEFAULT_AUDIO_LLM_MODEL))

    def test_label_helpers_fall_back_to_raw_id_for_custom_model(self):
        custom = "user-org/Some-Custom-Model"
        self.assertEqual(text_llm_label_for(custom), custom)
        self.assertEqual(audio_llm_label_for(custom), custom)


class SpeakerNameContextFilterTest(unittest.TestCase):
    def test_drops_phone_transfer_name_conflict(self):
        transcript = "\n".join(
            [
                "[SPEAKER_01] (00:01:00) Sainte-Ferrande, Philippe, bonjour.",
                "[Robin] (00:02:52) Oui, bonjour Monsieur Maire.",
                "[SPEAKER_01] (00:02:53) Bonjour, oui.",
            ]
        )
        out = filter_speaker_names_by_context(
            transcript,
            {"SPEAKER_01": "Philippe"},
            ["Arnaud Maire", "Robin Joseph"],
        )
        self.assertEqual(out, {})

    def test_keeps_compatible_phone_transfer_name(self):
        transcript = "\n".join(
            [
                "[Robin] (00:02:52) Oui, bonjour Monsieur Maire.",
                "[SPEAKER_01] (00:02:53) Bonjour, oui.",
            ]
        )
        out = filter_speaker_names_by_context(
            transcript,
            {"SPEAKER_01": "Arnaud"},
            ["Arnaud Maire", "Robin Joseph"],
        )
        self.assertEqual(out, {"SPEAKER_01": "Arnaud"})


class LlmPostProcessParsingTest(unittest.TestCase):
    """
    These tests are the regression suite for the bug we just fixed: real
    Mistral-7B 4-bit outputs that the previous single-shot JSON parser
    rejected, leaving every transcription without a title or speakers.
    """

    def test_title_speakers_parses_clean_json(self):
        out = parse_llm_title_speakers(
            '{"title": "Présentation des outils RH", '
            '"speakers": {"SPEAKER_00": "Robin", "SPEAKER_01": ""}}'
        )
        self.assertEqual(out["title"], "Présentation des outils RH")
        # Empty values dropped — we never want to label a SPEAKER as "".
        self.assertEqual(out["speakers"], {"SPEAKER_00": "Robin"})
        self.assertEqual(out["technical_terms"], [])

    def test_title_speakers_parses_technical_terms(self):
        out = parse_llm_title_speakers(
            '{"title": "Site CVR", "speakers": {}, '
            '"technical_terms": ["Odoo", "Infomaniak", "Chat GPT"]}'
        )
        self.assertEqual(out["technical_terms"], ["Odoo", "Infomaniak", "Chat GPT"])

    def test_title_speakers_recovers_from_leading_prose(self):
        # mlx_lm sometimes prepends "Voici votre JSON :" before the object.
        text = (
            "Voici votre JSON :\n"
            '{"title": "Migration v19", "speakers": {"SPEAKER_00": "Robin"}}\n'
            "(merci !)"
        )
        out = parse_llm_title_speakers(text)
        self.assertEqual(out["title"], "Migration v19")

    def test_title_speakers_recovers_from_trailing_comma(self):
        # The exact failure mode from the user's app.log.
        text = '{"title": "Stand-up Odoo", "speakers": {"SPEAKER_00": "Robin",}}'
        out = parse_llm_title_speakers(text)
        self.assertEqual(out["title"], "Stand-up Odoo")
        self.assertEqual(out["speakers"], {"SPEAKER_00": "Robin"})

    def test_title_speakers_returns_empty_on_pure_garbage(self):
        out = parse_llm_title_speakers("désolé, je ne sais pas")
        self.assertEqual(out, {})

    def test_corrections_parses_well_formed_markdown(self):
        text = """# Corrections
- [00:12:34] "Adel" -> "Adèle" (raison: prénom dans glossaire)
- [00:14:02] "modifs" -> "modifications" (raison: oral courant)

# Doutes
- [00:18:30] "Captivea" (raison: nom propre incertain)
"""
        out = parse_llm_corrections_markdown(text)
        self.assertEqual(len(out["corrections"]), 2)
        self.assertEqual(out["corrections"][0]["original"], "Adel")
        self.assertEqual(out["corrections"][0]["replacement"], "Adèle")
        self.assertEqual(len(out["uncertain_passages"]), 1)
        self.assertEqual(out["uncertain_passages"][0]["text"], "Captivea")

    def test_corrections_skips_malformed_lines(self):
        # A single bad line shouldn't dynamite the whole pass — that was
        # the central failure of the old single-JSON design.
        text = """# Corrections
- [00:12:34] "Adel" -> "Adèle" (raison: glossaire)
- garbage line nobody can parse
- [00:14:02] "modifs" -> "modifications"
"""
        out = parse_llm_corrections_markdown(text)
        self.assertEqual(len(out["corrections"]), 2)

    def test_corrections_handles_empty_marker(self):
        text = "Aucune correction.\nAucun doute.\n"
        out = parse_llm_corrections_markdown(text)
        self.assertEqual(out, {"corrections": [], "uncertain_passages": []})

    def test_build_title_cmd_carries_glossary(self):
        cmd = build_llm_title_cmd(
            "/v/bin/python",
            "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
            "/tmp/t.txt",
            "Adèle, Odoo",
        )
        self.assertEqual(cmd[0], "/v/bin/python")
        self.assertEqual(cmd[-1], "Adèle, Odoo")
        self.assertIn("speakers", cmd[2])
        self.assertIn("technical_terms", cmd[2])

    def test_build_corrections_cmd_carries_glossary(self):
        cmd = build_llm_corrections_cmd(
            "/v/bin/python",
            "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
            "/tmp/t.txt",
            "Adèle, Odoo",
        )
        self.assertEqual(cmd[0], "/v/bin/python")
        self.assertEqual(cmd[-1], "Adèle, Odoo")
        self.assertIn("Corrections", cmd[2])
        self.assertIn("Doutes", cmd[2])
        self.assertIn("erreur phonétique", cmd[2])
        self.assertIn("Chat GPT", cmd[2])


if __name__ == "__main__":
    unittest.main()
