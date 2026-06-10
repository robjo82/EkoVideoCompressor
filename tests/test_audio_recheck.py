"""
PR F — Multimodal Qwen2-Audio recheck (port du legacy au nouveau moteur).

Covers:
  • Prompt construction (``build_multimodal_recheck_prompt``): the
    glossary block is included when non-empty, omitted when blank.
  • Response parsing (``parse_multimodal_audio_response``): JSON last-
    line extraction, malformed input, error payload.
  • The pipeline's ``_run_audio_recheck``:
      - Returns ``None`` (no StepResult) when the feature is off,
        when there are no uncertain passages, when the venv python
        is missing, when ``mlx_vlm`` isn't installed, or when the
        Whisper WAV doesn't exist.
      - Caps at ``_AUDIO_RECHECK_MAX_PASSAGES`` regardless of how
        many doubts the LLM flagged.
      - Attaches ``suggestion`` and ``clip_path`` to each processed
        passage in place.
      - Keeps going when one passage fails (extract error / mlx
        error / parse error / empty suggestion).
      - Skips passages with empty or malformed timestamps.
  • ``mlx_vlm`` probe is cached: calling
    ``_ensure_mlx_vlm_available`` twice runs the probe once.
  • Format helper ``format_seconds_for_clip`` produces ffmpeg-
    compatible strings.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from ekovideo_engine.models import JobRequest, TranscriptionSettings
from ekovideo_engine.pipeline import TranscriptionPipeline
from transcription_utils import (
    build_multimodal_recheck_prompt,
    format_seconds_for_clip,
    parse_multimodal_audio_response,
)


# ---------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------


class BuildMultimodalRecheckPromptTests(unittest.TestCase):
    def test_glossary_block_included_when_non_empty(self):
        prompt = build_multimodal_recheck_prompt(
            whisper_text="On parle de pouvoir bien.",
            reason="phonetic-doubt",
            glossary="Power BI\nOdoo\nCastel",
        )
        self.assertIn("Vocabulaire métier attendu", prompt)
        self.assertIn("Power BI", prompt)
        self.assertIn("Odoo", prompt)
        self.assertIn("Castel", prompt)

    def test_glossary_block_omitted_when_empty(self):
        prompt = build_multimodal_recheck_prompt(
            whisper_text="pouvoir bien",
            reason="phonetic-doubt",
            glossary="",
        )
        self.assertNotIn("Vocabulaire", prompt)

    def test_includes_whisper_text_and_reason(self):
        prompt = build_multimodal_recheck_prompt(
            whisper_text="cast.fr",
            reason="surface inhabituelle",
            glossary="",
        )
        self.assertIn("« cast.fr »", prompt)
        self.assertIn("surface inhabituelle", prompt)

    def test_whitespace_only_glossary_is_treated_as_empty(self):
        prompt = build_multimodal_recheck_prompt(
            whisper_text="X",
            reason="Y",
            glossary="   \n\t  ",
        )
        self.assertNotIn("Vocabulaire", prompt)


class ParseMultimodalAudioResponseTests(unittest.TestCase):
    def test_parses_clean_json_payload(self):
        payload = parse_multimodal_audio_response(
            '{"suggestion": "On parle de Power BI."}'
        )
        self.assertEqual(payload["suggestion"], "On parle de Power BI.")

    def test_takes_last_non_empty_line(self):
        # mlx_vlm sometimes prints progress chatter before the JSON.
        stdout = (
            "Loading model...\n"
            "Generating...\n"
            "\n"
            '{"suggestion": "Castel et fils."}\n'
        )
        payload = parse_multimodal_audio_response(stdout)
        self.assertEqual(payload["suggestion"], "Castel et fils.")

    def test_returns_empty_dict_on_invalid_json(self):
        self.assertEqual(parse_multimodal_audio_response("not json"), {})

    def test_returns_empty_dict_on_empty_input(self):
        self.assertEqual(parse_multimodal_audio_response(""), {})
        self.assertEqual(parse_multimodal_audio_response("   \n  "), {})

    def test_passes_through_error_payload(self):
        # When ``mlx_vlm`` itself errored, the script emits
        # ``{"error": "..."}``. The parser surfaces it so the caller
        # can log the underlying cause.
        payload = parse_multimodal_audio_response('{"error": "OOM"}')
        self.assertEqual(payload, {"error": "OOM"})

    def test_returns_empty_dict_on_non_object_json(self):
        # Defensive: a model that returns a JSON array instead of
        # an object shouldn't crash the parser.
        self.assertEqual(parse_multimodal_audio_response("[1,2,3]"), {})


class FormatSecondsForClipTests(unittest.TestCase):
    def test_integer_seconds_renders_without_decimals(self):
        self.assertEqual(format_seconds_for_clip(42.0), "42")

    def test_fractional_seconds_keeps_precision(self):
        self.assertEqual(format_seconds_for_clip(42.5), "42.5")

    def test_negative_clamps_to_zero(self):
        self.assertEqual(format_seconds_for_clip(-3.0), "0")

    def test_zero_renders_as_zero(self):
        self.assertEqual(format_seconds_for_clip(0.0), "0")


# ---------------------------------------------------------------------
# Pipeline integration — _run_audio_recheck
# ---------------------------------------------------------------------


def _make_pipeline(
    *,
    audio_recheck_enabled: bool = True,
    venv_python: str = "",
    uncertain_passages: list[dict] | None = None,
    glossary_terms: list[str] | None = None,
) -> TranscriptionPipeline:
    request = JobRequest(
        source_path="/tmp/x.mov",
        output_dir="/tmp/out",
        mode="transcribe",
        transcription_settings=TranscriptionSettings(
            audio_recheck_enabled=audio_recheck_enabled,
            audio_llm_model="mlx-community/Qwen2-Audio-7B-Instruct-4bit",
            venv_python_path=venv_python,
        ),
        glossary_terms=list(glossary_terms or []),
    )
    sink = MagicMock()
    pipeline = TranscriptionPipeline(request=request, sink=sink)
    pipeline._llm_payload = {"uncertain_passages": list(uncertain_passages or [])}
    return pipeline


class RunAudioRecheckGuardTests(unittest.TestCase):
    """The recheck step is opt-in and gracefully degrades."""

    def test_returns_none_when_flag_off(self):
        pipeline = _make_pipeline(
            audio_recheck_enabled=False,
            uncertain_passages=[{"timestamp": "0:30", "text": "X"}],
        )
        result = pipeline._run_audio_recheck(
            Path("/tmp/whisper.wav"), Path("/tmp/work")
        )
        self.assertIsNone(result)

    def test_returns_none_when_no_uncertain_passages(self):
        pipeline = _make_pipeline(audio_recheck_enabled=True)
        result = pipeline._run_audio_recheck(
            Path("/tmp/whisper.wav"), Path("/tmp/work")
        )
        self.assertIsNone(result)

    def test_returns_none_when_venv_python_missing(self):
        pipeline = _make_pipeline(
            venv_python="",  # no managed venv
            uncertain_passages=[{"timestamp": "0:30", "text": "X"}],
        )
        result = pipeline._run_audio_recheck(
            Path("/tmp/whisper.wav"), Path("/tmp/work")
        )
        self.assertIsNone(result)


class EnsureMlxVlmAvailableTests(unittest.TestCase):
    def setUp(self):
        # PR AW — neutralise the upstream gate so these tests keep
        # covering the probe logic itself (the gate has its own tests).
        patcher = patch.object(
            TranscriptionPipeline, "_AUDIO_RECHECK_UPSTREAM_BLOCK", ""
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_probe_is_cached(self):
        pipeline = _make_pipeline(venv_python="/usr/bin/python3")
        with patch("ekovideo_engine.pipeline.subprocess.run") as mock_run, patch(
            "ekovideo_engine.pipeline.Path.exists", return_value=True
        ):
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            first = pipeline._ensure_mlx_vlm_available()
            second = pipeline._ensure_mlx_vlm_available()
        self.assertTrue(first)
        self.assertTrue(second)
        self.assertEqual(mock_run.call_count, 1)

    def test_probe_returns_false_when_subprocess_fails(self):
        pipeline = _make_pipeline(venv_python="/usr/bin/python3")
        with patch("ekovideo_engine.pipeline.subprocess.run") as mock_run, patch(
            "ekovideo_engine.pipeline.Path.exists", return_value=True
        ):
            mock_run.return_value = MagicMock(
                returncode=1, stderr="ModuleNotFoundError: mlx_vlm"
            )
            self.assertFalse(pipeline._ensure_mlx_vlm_available())

    def test_probe_returns_false_when_venv_missing(self):
        pipeline = _make_pipeline(venv_python="")
        self.assertFalse(pipeline._ensure_mlx_vlm_available())

    # -- PR AC: smarter probe (model-submodule check) --------------

    def test_probe_includes_model_submodule_check_for_gemma4(self):
        # PR AW — the default audio model is Gemma 4 E4B; the probe
        # script must reference ``mlx_vlm.models.gemma4``.
        pipeline = _make_pipeline(venv_python="/usr/bin/python3")
        with patch("ekovideo_engine.pipeline.subprocess.run") as mock_run, patch(
            "ekovideo_engine.pipeline.Path.exists", return_value=True
        ):
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            pipeline._ensure_mlx_vlm_available()
        # Inspect the script the probe sent.
        cmd_args, _ = mock_run.call_args
        script = cmd_args[0][2]
        self.assertIn("import mlx_vlm", script)
        self.assertIn("mlx_vlm.models.gemma4_unified", script)

    def test_probe_remaps_legacy_qwen_id_to_gemma4(self):
        # PR AW — upstream mlx-vlm removed ``qwen2_audio``; a persisted
        # Qwen2-Audio setting must remap to Gemma 4 BEFORE the slug
        # lookup, otherwise old installs keep probing a module that can
        # never exist and the recheck stays dead forever.
        pipeline = _make_pipeline(venv_python="/usr/bin/python3")
        pipeline.request.transcription_settings.audio_llm_model = (
            "mlx-community/Qwen2-Audio-7B-Instruct-4bit"
        )
        with patch("ekovideo_engine.pipeline.subprocess.run") as mock_run, patch(
            "ekovideo_engine.pipeline.Path.exists", return_value=True
        ):
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            pipeline._ensure_mlx_vlm_available()
        cmd_args, _ = mock_run.call_args
        script = cmd_args[0][2]
        self.assertIn("mlx_vlm.models.gemma4_unified", script)
        self.assertNotIn("qwen2_audio", script)

    def test_slug_hints_distinguish_unified_12b_from_edge(self):
        # 12B "Unified" checkpoints need gemma4_unified; e2b/e4b use
        # plain gemma4. The needle order in the hints table keeps that
        # distinction (specific before generic).
        self.assertEqual(
            TranscriptionPipeline._mlx_vlm_submodule_for(
                "mlx-community/gemma-4-12B-it-4bit"
            ),
            "gemma4_unified",
        )
        self.assertEqual(
            TranscriptionPipeline._mlx_vlm_submodule_for(
                "mlx-community/gemma-4-e4b-it-4bit"
            ),
            "gemma4",
        )

    def test_probe_distinguishes_missing_submodule_from_missing_package(self):
        # Exit code 2 means mlx_vlm is there but the model
        # submodule isn't — the CVR/Caste failure mode. The probe
        # must return False AND emit a WarningEvent with the
        # actionable "update mlx-vlm" message.
        pipeline = _make_pipeline(venv_python="/usr/bin/python3")
        with patch("ekovideo_engine.pipeline.subprocess.run") as mock_run, patch(
            "ekovideo_engine.pipeline.Path.exists", return_value=True
        ):
            mock_run.return_value = MagicMock(
                returncode=2,
                stderr="No module named 'mlx_vlm.models.qwen2_audio'",
            )
            ok = pipeline._ensure_mlx_vlm_available()
        self.assertFalse(ok)
        # Check the WarningEvent payload — sink is a MagicMock.
        warn_calls = [
            call for call in pipeline.sink.call_args_list
            if call.args and getattr(call.args[0], "event", "") == "warning"
        ]
        self.assertEqual(len(warn_calls), 1)
        warning = warn_calls[0].args[0]
        self.assertIn("gemma4_unified", warning.message.lower())
        self.assertIn("pip install -U mlx-vlm", warning.message)

    def test_probe_emits_missing_package_message_on_rc_1(self):
        # Exit code 1 means mlx_vlm itself missing — different
        # remediation advice.
        pipeline = _make_pipeline(venv_python="/usr/bin/python3")
        with patch("ekovideo_engine.pipeline.subprocess.run") as mock_run, patch(
            "ekovideo_engine.pipeline.Path.exists", return_value=True
        ):
            mock_run.return_value = MagicMock(
                returncode=1,
                stderr="ModuleNotFoundError: No module named 'mlx_vlm'",
            )
            pipeline._ensure_mlx_vlm_available()
        warning = next(
            call.args[0] for call in pipeline.sink.call_args_list
            if call.args and getattr(call.args[0], "event", "") == "warning"
        )
        self.assertIn("n'est pas installé", warning.message)

    def test_probe_skips_submodule_check_for_unknown_model_path(self):
        # A future model like ``foo-bar-7B`` doesn't match any
        # slug hint. The probe degrades to the legacy "just
        # import mlx_vlm" check rather than failing.
        request = JobRequest(
            source_path="/tmp/x.mov",
            output_dir="/tmp/out",
            mode="transcribe",
            transcription_settings=TranscriptionSettings(
                audio_recheck_enabled=True,
                audio_llm_model="some-vendor/Unknown-Model-7B",
                venv_python_path="/usr/bin/python3",
            ),
        )
        pipeline = TranscriptionPipeline(request=request, sink=MagicMock())
        with patch("ekovideo_engine.pipeline.subprocess.run") as mock_run, patch(
            "ekovideo_engine.pipeline.Path.exists", return_value=True
        ):
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            pipeline._ensure_mlx_vlm_available()
        cmd_args, _ = mock_run.call_args
        script = cmd_args[0][2]
        # No ``mlx_vlm.models.<slug>`` reference in the script.
        self.assertIn("import mlx_vlm", script)
        self.assertNotIn("mlx_vlm.models.", script)

    def test_submodule_for_helper_maps_qwen2_audio(self):
        from ekovideo_engine.pipeline import TranscriptionPipeline
        self.assertEqual(
            TranscriptionPipeline._mlx_vlm_submodule_for(
                "mlx-community/Qwen2-Audio-7B-Instruct-4bit"
            ),
            "qwen2_audio",
        )
        self.assertEqual(
            TranscriptionPipeline._mlx_vlm_submodule_for("Qwen2_Audio"),
            "qwen2_audio",
        )
        self.assertEqual(
            TranscriptionPipeline._mlx_vlm_submodule_for(""),
            "",
        )
        self.assertEqual(
            TranscriptionPipeline._mlx_vlm_submodule_for("foo/Bar-7B"),
            "",
        )


class RunAudioRecheckHappyPathTests(unittest.TestCase):
    """Mock the subprocesses and verify orchestrator behaviour."""

    def setUp(self):
        # PR AW — neutralise the upstream gate (see EnsureMlxVlm tests).
        patcher = patch.object(
            TranscriptionPipeline, "_AUDIO_RECHECK_UPSTREAM_BLOCK", ""
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def _patch_subprocess(self, mlx_stdout: str = '{"suggestion": "OK"}'):
        """Return a side_effect callable feeding mlx_vlm probe + per-
        passage extract + mlx_vlm response in the right order."""

        def side_effect(cmd, *args, **kwargs):  # noqa: ARG001
            # mlx_vlm probe: a single "-c" with "import mlx_vlm" body.
            if len(cmd) >= 3 and cmd[1] == "-c" and "import mlx_vlm" in cmd[2]:
                return MagicMock(returncode=0, stderr="")
            # ffmpeg extract: first arg is ffmpeg path.
            if cmd and "ffmpeg" in str(cmd[0]).lower():
                return MagicMock(returncode=0, stdout="", stderr="")
            # multimodal mlx_vlm call: returns the suggestion JSON.
            return MagicMock(returncode=0, stdout=mlx_stdout, stderr="")

        return side_effect

    def test_attaches_suggestion_to_each_processed_passage(self):
        pipeline = _make_pipeline(
            venv_python="/usr/bin/python3",
            uncertain_passages=[
                {"timestamp": "0:30", "text": "pouvoir bien", "reason": "phonetic"},
                {"timestamp": "1:15", "text": "cast.fr", "reason": "uncommon"},
            ],
            glossary_terms=["Power BI", "Castel"],
        )

        with patch(
            "ekovideo_engine.pipeline.subprocess.run",
            side_effect=self._patch_subprocess('{"suggestion": "Power BI."}'),
        ), patch("ekovideo_engine.pipeline.Path.exists", return_value=True):
            result = pipeline._run_audio_recheck(
                Path("/tmp/whisper.wav"), Path("/tmp/work")
            )

        self.assertIsNotNone(result)
        self.assertTrue(result.ok)
        self.assertEqual(result.metrics["rechecked"], 2)
        self.assertEqual(result.metrics["suggestions"], 2)
        passages = pipeline._llm_payload["uncertain_passages"]
        self.assertEqual(passages[0]["suggestion"], "Power BI.")
        self.assertEqual(passages[1]["suggestion"], "Power BI.")
        self.assertIn("clip_path", passages[0])

    def test_caps_at_max_passages(self):
        # 15 doubts ; _AUDIO_RECHECK_MAX_PASSAGES = 10 ; only 10 processed.
        passages = [
            {"timestamp": f"0:{seconds:02d}", "text": f"t{seconds}", "reason": "r"}
            for seconds in range(10, 25)
        ]
        pipeline = _make_pipeline(
            venv_python="/usr/bin/python3",
            uncertain_passages=passages,
        )
        with patch(
            "ekovideo_engine.pipeline.subprocess.run",
            side_effect=self._patch_subprocess(),
        ), patch("ekovideo_engine.pipeline.Path.exists", return_value=True):
            result = pipeline._run_audio_recheck(
                Path("/tmp/whisper.wav"), Path("/tmp/work")
            )
        self.assertEqual(result.metrics["rechecked"], 10)
        # First 10 got suggestions; last 5 untouched.
        suggested = [p for p in passages if "suggestion" in p]
        self.assertEqual(len(suggested), 10)
        unsuggested = [p for p in passages if "suggestion" not in p]
        self.assertEqual(len(unsuggested), 5)

    def test_skips_passages_with_empty_timestamp(self):
        passages = [
            {"timestamp": "", "text": "lost"},
            {"timestamp": "1:00", "text": "found", "reason": "r"},
        ]
        pipeline = _make_pipeline(
            venv_python="/usr/bin/python3",
            uncertain_passages=passages,
        )
        with patch(
            "ekovideo_engine.pipeline.subprocess.run",
            side_effect=self._patch_subprocess(),
        ), patch("ekovideo_engine.pipeline.Path.exists", return_value=True):
            result = pipeline._run_audio_recheck(
                Path("/tmp/whisper.wav"), Path("/tmp/work")
            )
        # The cap counts ALL passages (we slice before filtering),
        # but only the one with a usable timestamp gets a suggestion.
        self.assertEqual(result.metrics["suggestions"], 1)
        self.assertNotIn("suggestion", passages[0])
        self.assertEqual(passages[1]["suggestion"], "OK")

    def test_continues_when_one_mlx_call_fails(self):
        passages = [
            {"timestamp": "0:30", "text": "a", "reason": "r"},
            {"timestamp": "1:15", "text": "b", "reason": "r"},
        ]
        pipeline = _make_pipeline(
            venv_python="/usr/bin/python3",
            uncertain_passages=passages,
        )

        call_state = {"mlx_count": 0}

        def side_effect(cmd, *args, **kwargs):  # noqa: ARG001
            if len(cmd) >= 3 and cmd[1] == "-c" and "import mlx_vlm" in cmd[2]:
                return MagicMock(returncode=0, stderr="")
            if cmd and "ffmpeg" in str(cmd[0]).lower():
                return MagicMock(returncode=0, stdout="", stderr="")
            # First mlx call fails ; second succeeds.
            call_state["mlx_count"] += 1
            if call_state["mlx_count"] == 1:
                return MagicMock(
                    returncode=2, stdout="", stderr="OOM"
                )
            return MagicMock(
                returncode=0, stdout='{"suggestion": "second."}'
            )

        with patch(
            "ekovideo_engine.pipeline.subprocess.run", side_effect=side_effect
        ), patch("ekovideo_engine.pipeline.Path.exists", return_value=True):
            result = pipeline._run_audio_recheck(
                Path("/tmp/whisper.wav"), Path("/tmp/work")
            )

        self.assertEqual(result.metrics["suggestions"], 1)
        self.assertEqual(result.metrics["failures"], 1)
        self.assertNotIn("suggestion", passages[0])
        self.assertEqual(passages[1]["suggestion"], "second.")

    def test_empty_mlx_response_counts_as_failure(self):
        passages = [{"timestamp": "0:30", "text": "x", "reason": "r"}]
        pipeline = _make_pipeline(
            venv_python="/usr/bin/python3",
            uncertain_passages=passages,
        )

        def side_effect(cmd, *args, **kwargs):  # noqa: ARG001
            if len(cmd) >= 3 and cmd[1] == "-c" and "import mlx_vlm" in cmd[2]:
                return MagicMock(returncode=0, stderr="")
            if cmd and "ffmpeg" in str(cmd[0]).lower():
                return MagicMock(returncode=0, stdout="", stderr="")
            # Return an error JSON — no suggestion field.
            return MagicMock(
                returncode=0, stdout='{"error": "no audio"}'
            )

        with patch(
            "ekovideo_engine.pipeline.subprocess.run", side_effect=side_effect
        ), patch("ekovideo_engine.pipeline.Path.exists", return_value=True):
            result = pipeline._run_audio_recheck(
                Path("/tmp/whisper.wav"), Path("/tmp/work")
            )
        self.assertEqual(result.metrics["suggestions"], 0)
        self.assertEqual(result.metrics["failures"], 1)




class UpstreamBlockGateTests(unittest.TestCase):
    """PR AW — the honest gate: while mlx-vlm's Gemma 4 audio
    generation is broken upstream, the probe refuses with ONE
    actionable warning and never spawns a subprocess."""

    def test_gate_blocks_with_warning_and_no_subprocess(self):
        pipeline = _make_pipeline(venv_python="/usr/bin/python3")
        self.assertTrue(TranscriptionPipeline._AUDIO_RECHECK_UPSTREAM_BLOCK)
        with patch("ekovideo_engine.pipeline.subprocess.run") as mock_run:
            ok = pipeline._ensure_mlx_vlm_available()
        self.assertFalse(ok)
        mock_run.assert_not_called()
        warn_calls = [
            call for call in pipeline.sink.call_args_list
            if call.args and getattr(call.args[0], "event", "") == "warning"
        ]
        self.assertEqual(len(warn_calls), 1)
        self.assertEqual(
            warn_calls[0].args[0].code, "audio_recheck_upstream_blocked"
        )

    def test_gate_result_is_cached(self):
        pipeline = _make_pipeline(venv_python="/usr/bin/python3")
        with patch("ekovideo_engine.pipeline.subprocess.run") as mock_run:
            pipeline._ensure_mlx_vlm_available()
            pipeline._ensure_mlx_vlm_available()
        mock_run.assert_not_called()
        warn_calls = [
            call for call in pipeline.sink.call_args_list
            if call.args and getattr(call.args[0], "event", "") == "warning"
        ]
        self.assertEqual(len(warn_calls), 1)  # warned once, not per call


if __name__ == "__main__":
    unittest.main()
