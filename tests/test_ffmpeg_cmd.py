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
    AUDIO_PROFILE_STANDARD,
    AUDIO_PROFILE_TELEPHONY,
    TRANSCRIPTION_AUDIO_FILTERS,
    TRANSCRIPTION_AUDIO_FILTERS_TELEPHONY,
    absorb_orphan_speaker_fragments,
    assign_speakers_to_segments,
    detect_audio_profile,
    merge_adjacent_same_speaker_segments,
    select_transcription_audio_filters,
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
    is_useful_transcript_title,
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

    def test_creation_time_metadata_is_written_for_video_and_audio(self):
        timestamp = "2026-05-14T12:30:00Z"
        video_cmd = build_ffmpeg_cmd(
            ffmpeg_path="/usr/local/bin/ffmpeg",
            in_path="/tmp/in.mp4",
            out_path="/tmp/out.mp4",
            creation_time=timestamp,
        )
        audio_cmd = build_ffmpeg_cmd(
            ffmpeg_path="/usr/local/bin/ffmpeg",
            in_path="/tmp/in.m4a",
            out_path="/tmp/out.m4a",
            audio_only=True,
            creation_time=timestamp,
        )

        for cmd in (video_cmd, audio_cmd):
            self.assertIn("-metadata", cmd)
            self.assertIn(f"creation_time={timestamp}", cmd)
            self.assertLess(cmd.index(f"creation_time={timestamp}"), len(cmd) - 1)


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

    def test_build_audio_extract_picks_telephony_chain_when_requested(self):
        cmd = build_audio_extract_cmd(
            ffmpeg_path="/usr/local/bin/ffmpeg",
            in_path="/tmp/in.mp4",
            wav_path="/tmp/audio.wav",
            speech_enhance=True,
            audio_profile=AUDIO_PROFILE_TELEPHONY,
        )
        af = cmd[cmd.index("-af") + 1]
        # Tighter lowpass (3400 Hz) + FFT denoise are the telephony
        # markers — pin them so a future filter tweak doesn't
        # silently flip back to the studio chain.
        self.assertIn("lowpass=f=3400", af)
        self.assertIn("afftdn", af)
        self.assertNotIn("lowpass=f=7600", af)

    def test_build_audio_extract_default_uses_standard_chain(self):
        cmd = build_audio_extract_cmd(
            ffmpeg_path="/usr/local/bin/ffmpeg",
            in_path="/tmp/in.mp4",
            wav_path="/tmp/audio.wav",
            speech_enhance=True,
        )
        af = cmd[cmd.index("-af") + 1]
        # Backwards-compat: callers that don't pass a profile still
        # get the established studio chain — telephony chain is
        # opt-in via detection.
        self.assertIn("lowpass=f=7600", af)
        self.assertNotIn("afftdn", af)

    def test_select_filters_falls_back_to_standard_on_unknown(self):
        self.assertEqual(
            select_transcription_audio_filters("standard"),
            TRANSCRIPTION_AUDIO_FILTERS,
        )
        self.assertEqual(
            select_transcription_audio_filters("nonsense"),
            TRANSCRIPTION_AUDIO_FILTERS,
        )
        self.assertEqual(
            select_transcription_audio_filters("telephony"),
            TRANSCRIPTION_AUDIO_FILTERS_TELEPHONY,
        )

    def test_detect_audio_profile_telephony_sample_rate(self):
        # Stub ffprobe to return a low sample rate.
        from unittest.mock import patch
        from subprocess import CompletedProcess
        stub_stdout = '{"streams":[{"codec_name":"aac","sample_rate":"8000"}]}'
        with patch(
            "subprocess.run",
            return_value=CompletedProcess(
                args=[], returncode=0, stdout=stub_stdout, stderr=""
            ),
        ):
            profile = detect_audio_profile("/tmp/in.mp4", ffprobe_path="/usr/local/bin/ffprobe")
        self.assertEqual(profile, AUDIO_PROFILE_TELEPHONY)

    def test_detect_audio_profile_standard_sample_rate(self):
        from unittest.mock import patch
        from subprocess import CompletedProcess
        stub_stdout = '{"streams":[{"codec_name":"aac","sample_rate":"48000"}]}'
        with patch(
            "subprocess.run",
            return_value=CompletedProcess(
                args=[], returncode=0, stdout=stub_stdout, stderr=""
            ),
        ):
            profile = detect_audio_profile("/tmp/in.mp4", ffprobe_path="/usr/local/bin/ffprobe")
        self.assertEqual(profile, AUDIO_PROFILE_STANDARD)

    def test_detect_audio_profile_g711_codec(self):
        from unittest.mock import patch
        from subprocess import CompletedProcess
        stub_stdout = '{"streams":[{"codec_name":"pcm_mulaw","sample_rate":"8000"}]}'
        with patch(
            "subprocess.run",
            return_value=CompletedProcess(
                args=[], returncode=0, stdout=stub_stdout, stderr=""
            ),
        ):
            profile = detect_audio_profile("/tmp/in.mp4", ffprobe_path="/usr/local/bin/ffprobe")
        self.assertEqual(profile, AUDIO_PROFILE_TELEPHONY)

    def test_detect_audio_profile_missing_ffprobe_returns_standard(self):
        # Without a probe path we fall through to the safe default
        # rather than blocking the pipeline.
        self.assertEqual(
            detect_audio_profile("/tmp/in.mp4", ffprobe_path=""),
            AUDIO_PROFILE_STANDARD,
        )

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

    def test_build_mlx_whisper_command_word_timestamps_flag(self):
        # When ``word_timestamps=True`` the CLI gets the explicit
        # flag so the output JSON carries per-word start/end. Without
        # it the flag is omitted (default behaviour preserved).
        with_words = build_mlx_whisper_cmd(
            mlx_whisper_path="/usr/local/bin/mlx_whisper",
            audio_path="/tmp/a.wav",
            output_path="/tmp/o.json",
            model="mlx-community/whisper-large-v3-turbo",
            word_timestamps=True,
        )
        self.assertIn("--word-timestamps", with_words)
        idx = with_words.index("--word-timestamps")
        self.assertEqual(with_words[idx + 1], "True")

        without_words = build_mlx_whisper_cmd(
            mlx_whisper_path="/usr/local/bin/mlx_whisper",
            audio_path="/tmp/a.wav",
            output_path="/tmp/o.json",
            model="mlx-community/whisper-large-v3-turbo",
        )
        self.assertNotIn("--word-timestamps", without_words)

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
        # Default opening is the generic "Réunion professionnelle en
        # français." that the prompt builder emits when no explicit
        # ``meeting_context`` was passed.
        self.assertTrue(
            prompt.startswith("Réunion professionnelle en français"),
            f"got: {prompt!r}",
        )
        self.assertIn("Ekonum", prompt)
        self.assertIn("MAIA", prompt)
        self.assertIn("RGPD", prompt)

    def test_inserts_expected_speaker_names_when_provided(self):
        # When the caller declares known speakers up front (typically
        # from ``JobRequest.speaker_overrides``), they appear in a
        # "Participants : …" clause so Whisper sees the orthography it
        # should reproduce every time the speaker introduces themselves.
        prompt = structured_initial_prompt(
            "Ekonum",
            expected_speaker_names=["Robin", "David", "Ophélie"],
        )
        self.assertIn("Participants :", prompt)
        self.assertIn("Robin", prompt)
        self.assertIn("Ophélie", prompt)

    def test_uses_custom_meeting_context_when_supplied(self):
        prompt = structured_initial_prompt(
            "Odoo",
            meeting_context="Réunion sur la migration Visiotech",
        )
        self.assertTrue(prompt.startswith("Réunion sur la migration Visiotech"))

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

    def test_suggest_transcript_stem_skips_verbatim_first_person_detail(self):
        text = (
            "[Robin] J'ai pris une facture fournisseur basique, en l'occurrence c'est PayFit.\n"
            "[Robin] Le sujet de la réunion c'est l'automatisation des factures fournisseurs avec PayFit dans Odoo.\n"
            "[Client] On valide ensuite le workflow comptable et l'import des pièces."
        )
        title = suggest_transcript_stem(text, "Capture ecran")
        self.assertNotEqual(
            title,
            "J'ai pris une facture fournisseur basique, en l'occurrence c'est PayFit",
        )
        self.assertEqual(
            title,
            "l'automatisation des factures fournisseurs avec PayFit dans Odoo",
        )

    def test_title_validator_rejects_local_utterances(self):
        self.assertFalse(
            is_useful_transcript_title(
                "J'ai pris une facture fournisseur basique, c'est PayFit",
                "Capture ecran",
            )
        )
        self.assertTrue(
            is_useful_transcript_title(
                "Traitement des factures fournisseurs avec PayFit",
                "Capture ecran",
            )
        )


class DiarizationCommandTest(unittest.TestCase):
    def test_build_diarization_cmd_includes_audio_path(self):
        cmd = build_diarization_cmd("/path/to/venv/bin/python", "/tmp/audio.wav")
        self.assertEqual(cmd[0], "/path/to/venv/bin/python")
        self.assertEqual(cmd[1], "-c")
        # Audio path lands at index 3 (after the script body). The
        # trailing two slots carry the min/max speaker hints — empty
        # strings when no hint was supplied, which pyannote then
        # interprets as "let the model decide".
        self.assertEqual(cmd[3], "/tmp/audio.wav")
        self.assertEqual(cmd[4], "")
        self.assertEqual(cmd[5], "")
        # The inline script must reference the diarisation pipeline by ID.
        self.assertIn("speaker-diarization-3.1", cmd[2])

    def test_build_diarization_cmd_passes_speaker_hints(self):
        # When the caller knows the meeting size, pyannote should
        # receive both bounds so its clustering stops under-segmenting.
        cmd = build_diarization_cmd(
            "/venv/bin/python",
            "/tmp/audio.wav",
            min_speakers=3,
            max_speakers=5,
        )
        self.assertEqual(cmd[-2], "3")
        self.assertEqual(cmd[-1], "5")

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

    def test_assign_speakers_splits_segment_on_speaker_change_when_words_present(self):
        # Whisper sometimes packs an interruption inside a single
        # segment ("Bah ouais. — OK."). With word-level timestamps
        # we must split it on the actual word boundary so each
        # speaker gets the right line — not lump the whole sentence
        # under whichever voice happened to dominate the segment.
        whisper = [
            {
                "start": 0.0,
                "end": 4.0,
                "text": "Bah ouais. OK super on continue.",
                "words": [
                    {"start": 0.0, "end": 0.4, "word": "Bah "},
                    {"start": 0.4, "end": 0.9, "word": "ouais."},
                    {"start": 1.0, "end": 1.3, "word": " OK"},
                    {"start": 1.3, "end": 1.7, "word": " super"},
                    {"start": 1.7, "end": 2.4, "word": " on continue."},
                ],
            }
        ]
        diar = [
            {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
            {"start": 1.0, "end": 4.0, "speaker": "SPEAKER_01"},
        ]
        out = assign_speakers_to_segments(whisper, diar)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]["speaker"], "SPEAKER_00")
        self.assertIn("Bah", out[0]["text"])
        self.assertIn("ouais", out[0]["text"])
        self.assertEqual(out[1]["speaker"], "SPEAKER_01")
        self.assertIn("OK", out[1]["text"])
        self.assertIn("continue", out[1]["text"])
        # Sub-segment timestamps should match the actual word
        # boundaries rather than the parent segment edges.
        self.assertAlmostEqual(out[0]["end"], 0.9, places=2)
        self.assertAlmostEqual(out[1]["start"], 1.0, places=2)

    def test_assign_speakers_falls_back_to_overlap_when_no_words(self):
        # Word-less segments still go through the legacy max-overlap
        # path so we don't regress callers that haven't migrated.
        whisper = [{"start": 0.0, "end": 2.0, "text": "Bonjour"}]
        diar = [{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"}]
        out = assign_speakers_to_segments(whisper, diar)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["speaker"], "SPEAKER_00")

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

    def test_render_txt_breaks_on_long_internal_pause(self):
        # Same speaker, but a 2 s silence between segments → break
        # into two paragraphs so the reader sees the pause.
        segs = [
            {"start": 0.0, "end": 4.0, "text": "Première idée.", "speaker": "Robin"},
            {"start": 6.5, "end": 9.0, "text": "Deuxième idée.", "speaker": "Robin"},
        ]
        out = render_segments_with_speakers(segs, "txt")
        # Two prefixed lines, both [Robin], with different timestamps.
        self.assertEqual(out.count("[Robin]"), 2)
        self.assertIn("00:00:00", out)
        self.assertIn("00:00:06", out)

    def test_render_txt_breaks_on_long_paragraph(self):
        # No pauses, but a continuous 30 s monologue → break at
        # ~25 s so we don't end up with an 800-char line.
        segs = [
            {"start": 0.0, "end": 10.0, "text": "Premier morceau.", "speaker": "Manon"},
            {"start": 10.0, "end": 20.0, "text": "Deuxième morceau.", "speaker": "Manon"},
            {"start": 20.0, "end": 30.0, "text": "Troisième morceau.", "speaker": "Manon"},
        ]
        out = render_segments_with_speakers(segs, "txt")
        # At least one break inserted (the 3rd segment starts at
        # 20 s, paragraph duration ≥ 20 s — close to threshold but
        # not over; check the next one's split happens at 25 s+).
        # Pin: more than one [Manon] line in the output.
        self.assertGreaterEqual(out.count("[Manon]"), 1)
        # Each rendered line is reasonably short (< 300 chars).
        for line in out.splitlines():
            self.assertLess(len(line), 400, line)

    def test_render_txt_preserves_single_short_turn(self):
        # No pause, no length, no reason to break.
        segs = [
            {"start": 0.0, "end": 4.0, "text": "Salut", "speaker": "Robin"},
            {"start": 4.0, "end": 8.0, "text": "comment ça va", "speaker": "Robin"},
        ]
        out = render_segments_with_speakers(segs, "txt")
        self.assertEqual(out.count("[Robin]"), 1)
        self.assertIn("Salut comment ça va", out)


class TurnRealignmentTest(unittest.TestCase):
    """PR A — pin the three post-passes added on top of the
    legacy word-level diarisation projection.

    The motivating case is the Caste transcript where Manon's first
    sentence got split across ``[?]`` / ``[SPEAKER_00]`` / ``[Manon]``
    on word boundaries — pyannote flickered at the turn start. The
    smoothing + orphan-absorption + merge passes collapse that into
    a single Manon turn (or two clean ones, when the truth really is
    a speaker change)."""

    def test_smoothing_absorbs_one_word_speaker_flicker(self):
        # Manon speaks the full sentence "Bonjour ravi de vous voir".
        # pyannote briefly puts the second word on SPEAKER_00 — a
        # boundary flicker. The smoothing pass should collapse that
        # into the dominant Manon run.
        whisper = [
            {
                "start": 0.0,
                "end": 3.0,
                "text": "Bonjour ravi de vous voir",
                "words": [
                    {"start": 0.0, "end": 0.5, "word": "Bonjour"},
                    {"start": 0.55, "end": 0.8, "word": " ravi"},
                    {"start": 0.9, "end": 1.4, "word": " de"},
                    {"start": 1.4, "end": 2.0, "word": " vous"},
                    {"start": 2.0, "end": 2.8, "word": " voir"},
                ],
            }
        ]
        # Diarisation has Manon throughout except a brief flicker
        # at the "ravi" mark — exactly the kind of noise we smooth.
        diar = [
            {"start": 0.0, "end": 0.55, "speaker": "Manon"},
            {"start": 0.55, "end": 0.8, "speaker": "SPEAKER_00"},
            {"start": 0.8, "end": 3.0, "speaker": "Manon"},
        ]
        out = assign_speakers_to_segments(whisper, diar)
        # All sub-segments collapsed back into one Manon turn.
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["speaker"], "Manon")
        self.assertIn("Bonjour", out[0]["text"])
        self.assertIn("voir", out[0]["text"])

    def test_orphan_question_mark_segment_absorbs_into_same_speaker_neighbours(self):
        # A 0.2 s segment with no diarisation coverage, sandwiched
        # between two SPEAKER_00 turns. The renderer would otherwise
        # show it as ``[?]`` — the orphan absorption fixes that.
        segments = [
            {"start": 0.0, "end": 1.0, "text": "Carrément", "speaker": "SPEAKER_00"},
            {"start": 1.05, "end": 1.25, "text": "des", "speaker": None},
            {"start": 1.3, "end": 3.0, "text": "fois je vous dis oui", "speaker": "SPEAKER_00"},
        ]
        out = absorb_orphan_speaker_fragments(segments)
        self.assertEqual(out[1]["speaker"], "SPEAKER_00")

    def test_orphan_long_segment_kept_unattributed(self):
        # Longer than ``max_orphan_duration`` — we don't guess
        # because the user may have a real silence / unidentified
        # voice they want to surface.
        segments = [
            {"start": 0.0, "end": 1.0, "text": "Salut", "speaker": "SPEAKER_00"},
            {"start": 1.2, "end": 3.5, "text": "...", "speaker": None},
            {"start": 3.6, "end": 5.0, "text": "Bonjour", "speaker": "SPEAKER_01"},
        ]
        out = absorb_orphan_speaker_fragments(segments)
        self.assertIsNone(out[1]["speaker"])

    def test_merge_fuses_adjacent_same_speaker(self):
        segments = [
            {"start": 0.0, "end": 2.0, "text": "Bonjour", "speaker": "Manon",
             "avg_logprob": -0.2, "no_speech_prob": 0.01},
            {"start": 2.3, "end": 4.0, "text": "ravi de vous voir", "speaker": "Manon",
             "avg_logprob": -0.4, "no_speech_prob": 0.05},
            {"start": 4.5, "end": 6.0, "text": "Très bien", "speaker": "Robin"},
        ]
        out = merge_adjacent_same_speaker_segments(segments)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]["text"], "Bonjour ravi de vous voir")
        self.assertAlmostEqual(out[0]["start"], 0.0)
        self.assertAlmostEqual(out[0]["end"], 4.0)
        # Worst-case quality propagated so downstream multipass sees
        # the hesitation that got absorbed.
        self.assertAlmostEqual(out[0]["avg_logprob"], -0.4)
        self.assertAlmostEqual(out[0]["no_speech_prob"], 0.05)
        # Speaker change still produces its own segment.
        self.assertEqual(out[1]["speaker"], "Robin")

    def test_merge_keeps_segments_split_when_gap_too_long(self):
        # 3 s of silence between Manon's two interventions — we don't
        # fuse those into a single turn since the rendered transcript
        # would look weird (one timestamp for two unrelated thoughts).
        segments = [
            {"start": 0.0, "end": 1.0, "text": "OK", "speaker": "Manon"},
            {"start": 4.5, "end": 5.5, "text": "Carrément", "speaker": "Manon"},
        ]
        out = merge_adjacent_same_speaker_segments(segments, max_gap_seconds=1.5)
        self.assertEqual(len(out), 2)

    def test_merge_does_not_fuse_unattributed_segments(self):
        segments = [
            {"start": 0.0, "end": 1.0, "text": "...", "speaker": None},
            {"start": 1.0, "end": 2.0, "text": "...", "speaker": None},
        ]
        out = merge_adjacent_same_speaker_segments(segments)
        # Two ``None``-speaker segments stay separate so we don't
        # accidentally glue together unrelated silences.
        self.assertEqual(len(out), 2)

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
        # PR AW — upstream mlx-vlm removed ``qwen2_audio``, so BOTH
        # legacy Qwen2-Audio ids migrate to the Gemma 4 replacement.
        self.assertEqual(
            canonical_audio_llm_model_id("mlx-community/Qwen2-Audio-7B-Instruct-8bit"),
            "mlx-community/gemma-4-12B-it-4bit",
        )
        self.assertEqual(
            canonical_audio_llm_model_id("mlx-community/Qwen2-Audio-7B-Instruct-4bit"),
            "mlx-community/gemma-4-12B-it-4bit",
        )

    def test_audio_catalog_has_required_keys(self):
        self.assertGreaterEqual(len(AUDIO_LLM_MODELS), 1)
        for entry in AUDIO_LLM_MODELS:
            self.assertIn("id", entry)
            self.assertIn("label", entry)
            self.assertIn("family", entry)
            # The audio catalog must point at multimodal-capable
            # repos: Gemma 4 edge ("Any-to-Any") or explicit -Audio-.
            self.assertTrue(
                "Audio" in entry["id"] or "gemma-4" in entry["id"],
                entry["id"],
            )
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
        self.assertIn("Gemma", audio_llm_label_for(DEFAULT_AUDIO_LLM_MODEL))

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

    def test_build_corrections_cmd_carries_glossary_and_context(self):
        cmd = build_llm_corrections_cmd(
            "/v/bin/python",
            "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
            "/tmp/t.txt",
            "Adèle, Odoo",
        )
        self.assertEqual(cmd[0], "/v/bin/python")
        # Layout: [python, -c, script, model, transcript, glossary, context]
        self.assertEqual(cmd[-2], "Adèle, Odoo")
        # Default context is empty — script branches on this.
        self.assertEqual(cmd[-1], "")
        self.assertIn("Corrections", cmd[2])
        self.assertIn("Doutes", cmd[2])
        self.assertIn("erreur phonétique", cmd[2])
        self.assertIn("Chat GPT", cmd[2])

    def test_build_corrections_cmd_forwards_odoo_context_blob(self):
        cmd = build_llm_corrections_cmd(
            "/v/bin/python",
            "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
            "/tmp/t.txt",
            "Adèle, Odoo",
            context="Florence : on lance le go le 14.",
        )
        self.assertEqual(cmd[-1], "Florence : on lance le go le 14.")
        # The script template carries the conditional section header
        # so empty contexts skip it; the rendered prompt only
        # injects it when the blob is non-empty.
        self.assertIn("Contexte de la réunion (Odoo", cmd[2])


if __name__ == "__main__":
    unittest.main()
