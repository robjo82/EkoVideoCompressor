"""
PR L — Mise en cohérence finale du préset "Maximale".

Covers:
  • ``QUALITY_PRESETS`` matrix is consistent: every preset row lists
    every quality-knob field, no typos.
  • ``max`` enables every quality knob the engine wires today except
    the explicit pending set (``_MAX_PRESET_PENDING``).
  • ``fast`` / ``balanced`` / ``max`` apply via
    ``apply_quality_preset`` and round-trip through
    ``TranscriptionSettings.from_dict``.
  • ``quality_preset_levers`` returns the same matrix for tests
    and docs.

Sentinel intent: when a future PR adds a new ``*_enabled`` /
``hot_*`` / ``condition_*`` boolean to ``TranscriptionSettings``,
this file fails until the developer either flips it on in ``max``
or explicitly opts it out in ``_MAX_PRESET_PENDING`` — preventing
the "Maximale" preset from silently drifting out of sync again.
"""

from __future__ import annotations

import unittest

from ekovideo_engine.models import (
    QUALITY_PRESETS,
    TranscriptionSettings,
    _MAX_PRESET_PENDING,
    apply_quality_preset,
    quality_preset_levers,
)


# Fields on ``TranscriptionSettings`` that the sentinel test treats
# as "quality knobs". A field is a quality knob when its name ends
# in one of these suffixes OR is one of the curated extras below.
# Keeping this here (rather than introspecting bool fields blindly)
# avoids false positives on settings like ``diarization_enabled``,
# which is a user opt-in gated by HF token rather than a preset
# choice — and on cosmetic toggles like ``enhance_audio`` that
# every preset shares.
_QUALITY_FIELD_SUFFIXES = (
    "_enabled",
    "_enrichment",
)
_QUALITY_FIELD_EXTRAS = frozenset(
    {
        "condition_on_previous_text",
    }
)
# Fields that match the suffix above but are NOT preset-driven (user
# explicitly opts in or they're cosmetic). Listed here so the sentinel
# test ignores them.
_NOT_PRESET_DRIVEN = frozenset(
    {
        "diarization_enabled",  # gated by HF token, separate user toggle
        "enhance_audio",  # shared by all presets, always True
    }
)


def _discover_quality_fields() -> set[str]:
    """Find every ``TranscriptionSettings`` field that looks like a
    preset-driven quality knob."""
    found: set[str] = set()
    for field in TranscriptionSettings.__dataclass_fields__.values():
        name = field.name
        if name in _NOT_PRESET_DRIVEN:
            continue
        if name in _QUALITY_FIELD_EXTRAS:
            found.add(name)
            continue
        if any(name.endswith(suffix) for suffix in _QUALITY_FIELD_SUFFIXES):
            found.add(name)
    return found


class QualityPresetMatrixTests(unittest.TestCase):
    def test_every_preset_row_uses_known_fields(self):
        """No typos: every key in QUALITY_PRESETS exists on the dataclass."""
        known = {f.name for f in TranscriptionSettings.__dataclass_fields__.values()}
        for preset, row in QUALITY_PRESETS.items():
            for field_name in row.keys():
                self.assertIn(
                    field_name,
                    known,
                    f"QUALITY_PRESETS[{preset!r}] references unknown field "
                    f"{field_name!r}",
                )

    def test_every_preset_row_lists_every_quality_field(self):
        """Every preset row covers every quality knob — no gaps that
        would silently leak the previous user-selected value."""
        quality_fields = _discover_quality_fields()
        for preset, row in QUALITY_PRESETS.items():
            missing = quality_fields - set(row.keys())
            self.assertFalse(
                missing,
                f"QUALITY_PRESETS[{preset!r}] is missing quality field(s): "
                f"{sorted(missing)}. Add an explicit True/False so the "
                f"preset doesn't leak the previous setting.",
            )

    def test_max_preset_enables_every_wired_quality_knob(self):
        """Sentinel test — the whole reason PR L exists.

        ``max`` must enable every quality-driving boolean on
        ``TranscriptionSettings`` *unless* it's explicitly listed in
        ``_MAX_PRESET_PENDING`` (features whose underlying engine
        support isn't wired yet — e.g. audio_recheck pending PR F).
        """
        max_row = QUALITY_PRESETS["max"]
        offenders: list[str] = []
        for field_name in _discover_quality_fields():
            value = max_row.get(field_name)
            if value is True:
                continue
            if field_name in _MAX_PRESET_PENDING:
                continue
            offenders.append(field_name)

        self.assertFalse(
            offenders,
            "The 'max' preset is supposed to enable every wired "
            "quality knob. Found knob(s) left OFF without being in "
            f"_MAX_PRESET_PENDING: {sorted(offenders)}. "
            "Either flip them to True in QUALITY_PRESETS['max'] or "
            "add them to _MAX_PRESET_PENDING with a comment "
            "explaining the pending engine work.",
        )

    def test_pending_set_is_a_subset_of_known_fields(self):
        """``_MAX_PRESET_PENDING`` can't reference fields that no
        longer exist — would silently make the sentinel test green."""
        known = {f.name for f in TranscriptionSettings.__dataclass_fields__.values()}
        bogus = _MAX_PRESET_PENDING - known
        self.assertFalse(
            bogus,
            f"_MAX_PRESET_PENDING references unknown field(s): {sorted(bogus)}",
        )


class ApplyQualityPresetTests(unittest.TestCase):
    def test_fast_preset_turns_everything_off(self):
        settings = TranscriptionSettings(
            quality_preset="fast",
            vad_enabled=True,
            multipass_enabled=True,
            per_speaker_enabled=True,
            web_enrichment_enabled=True,
            condition_on_previous_text=True,
            hot_prompt_enrichment=True,
        )
        result = apply_quality_preset(settings)
        self.assertFalse(result.vad_enabled)
        self.assertFalse(result.multipass_enabled)
        self.assertFalse(result.per_speaker_enabled)
        self.assertFalse(result.web_enrichment_enabled)
        self.assertFalse(result.condition_on_previous_text)
        self.assertFalse(result.hot_prompt_enrichment)

    def test_balanced_preset_enables_vad_and_multipass_only(self):
        settings = TranscriptionSettings(quality_preset="balanced")
        result = apply_quality_preset(settings)
        self.assertTrue(result.vad_enabled)
        self.assertTrue(result.multipass_enabled)
        # The "heavy" knobs stay off in balanced.
        self.assertFalse(result.per_speaker_enabled)
        self.assertFalse(result.web_enrichment_enabled)
        self.assertFalse(result.condition_on_previous_text)
        self.assertFalse(result.hot_prompt_enrichment)
        self.assertFalse(result.audio_recheck_enabled)

    def test_max_preset_enables_all_wired_knobs(self):
        settings = TranscriptionSettings(quality_preset="max")
        result = apply_quality_preset(settings)
        self.assertTrue(result.vad_enabled)
        self.assertTrue(result.multipass_enabled)
        self.assertTrue(result.per_speaker_enabled)
        self.assertTrue(result.web_enrichment_enabled)
        self.assertTrue(result.condition_on_previous_text)
        self.assertTrue(result.hot_prompt_enrichment)
        # PR F: ``audio_recheck_enabled`` now wired in the engine —
        # the step is opt-in via this flag and degrades to a silent
        # no-op when ``mlx_vlm`` isn't installed.
        self.assertTrue(result.audio_recheck_enabled)

    def test_custom_preset_is_passthrough(self):
        settings = TranscriptionSettings(
            quality_preset="custom",
            vad_enabled=False,
            multipass_enabled=True,
            per_speaker_enabled=True,
            web_enrichment_enabled=False,
            condition_on_previous_text=True,
            hot_prompt_enrichment=False,
        )
        result = apply_quality_preset(settings)
        self.assertFalse(result.vad_enabled)
        self.assertTrue(result.multipass_enabled)
        self.assertTrue(result.per_speaker_enabled)
        self.assertFalse(result.web_enrichment_enabled)
        self.assertTrue(result.condition_on_previous_text)
        self.assertFalse(result.hot_prompt_enrichment)

    def test_unknown_preset_falls_through_to_passthrough(self):
        """Defensive: a typo in a saved JSON shouldn't reset the
        user's individual flags to some hardcoded default."""
        settings = TranscriptionSettings(
            quality_preset="ULTRA-OMG",
            vad_enabled=True,
            multipass_enabled=False,
        )
        result = apply_quality_preset(settings)
        # Passed through untouched.
        self.assertTrue(result.vad_enabled)
        self.assertFalse(result.multipass_enabled)

    def test_from_dict_applies_preset_round_trip(self):
        settings = TranscriptionSettings.from_dict({"quality_preset": "max"})
        self.assertTrue(settings.per_speaker_enabled)
        self.assertTrue(settings.condition_on_previous_text)
        self.assertTrue(settings.hot_prompt_enrichment)


class QualityPresetLeversTests(unittest.TestCase):
    def test_returns_dict_copy_for_known_preset(self):
        levers = quality_preset_levers("max")
        self.assertIsInstance(levers, dict)
        self.assertTrue(levers["per_speaker_enabled"])
        # Caller can mutate the returned dict without affecting the
        # canonical matrix.
        levers["per_speaker_enabled"] = False
        self.assertTrue(QUALITY_PRESETS["max"]["per_speaker_enabled"])

    def test_returns_empty_dict_for_custom(self):
        self.assertEqual(quality_preset_levers("custom"), {})

    def test_normalises_input(self):
        self.assertEqual(
            quality_preset_levers("  MAX  "),
            quality_preset_levers("max"),
        )


if __name__ == "__main__":
    unittest.main()
