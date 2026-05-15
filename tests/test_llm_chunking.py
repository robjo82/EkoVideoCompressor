"""Tests for the LLM transcript chunker.

We have to chunk locally because the embedded mlx_lm correction
script truncates its input at 30 000 chars. On any meeting >
~30 minutes that quietly drops the back half of the recording from
the correction pass — exactly the failure mode we saw on a real
1h40 meeting where every correction landed in the first 4 minutes.

The tests pin: chunks split at line boundaries (never mid-line),
respect the configured max char window, carry overlap so a
correction spanning a window boundary still gets seen, and the
dedup step deals with the resulting duplicates.
"""

from __future__ import annotations

import unittest

from llm_chunking import chunk_transcript_for_llm, dedupe_corrections


def _fake_transcript(lines: int, line_template: str = "[{m:02d}:{s:02d}:00] [Robin] Phrase {i}.") -> str:
    out = []
    for i in range(lines):
        out.append(line_template.format(m=i // 60, s=i % 60, i=i))
    return "\n".join(out)


class ChunkTranscriptTest(unittest.TestCase):
    def test_short_transcript_returns_single_chunk(self):
        text = _fake_transcript(20)
        chunks = chunk_transcript_for_llm(text, max_chars=22_000)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, text)
        self.assertEqual(chunks[0].index, 0)
        self.assertEqual(chunks[0].total, 1)

    def test_long_transcript_splits_into_overlapping_chunks(self):
        text = _fake_transcript(2_500)
        chunks = chunk_transcript_for_llm(text, max_chars=22_000, overlap_chars=1_500)
        # The exact count depends on the synthetic line length, but
        # any window scheme that respects the 22 000-char cap on
        # ~85 000 chars must produce at least 4 chunks.
        self.assertGreaterEqual(len(chunks), 4)
        for chunk in chunks:
            self.assertLessEqual(len(chunk.text), 22_000)
        # All chunks share the same total count.
        self.assertEqual({c.total for c in chunks}, {len(chunks)})
        # Concatenated chunks cover everything in the original text
        # (modulo duplicates from the overlap).
        joined = " ".join(c.text for c in chunks)
        self.assertIn("Phrase 0.", joined)
        self.assertIn("Phrase 2499.", joined)

    def test_chunks_split_on_newline_boundaries(self):
        text = _fake_transcript(1_000)
        chunks = chunk_transcript_for_llm(text, max_chars=10_000, overlap_chars=1_000)
        for chunk in chunks:
            # First and last char shouldn't sit in the middle of a
            # bracket-prefixed line.
            self.assertTrue(
                chunk.text.startswith("["),
                f"chunk {chunk.index} starts with: {chunk.text[:40]!r}",
            )
            self.assertTrue(chunk.text.rstrip().endswith("."))

    def test_overlap_actually_overlaps_consecutive_chunks(self):
        # The whole point of the overlap is that proper-noun
        # references near a chunk boundary still see context.
        text = _fake_transcript(800)
        chunks = chunk_transcript_for_llm(text, max_chars=8_000, overlap_chars=600)
        self.assertGreaterEqual(len(chunks), 2)
        prev_tail = chunks[0].text.splitlines()[-5:]
        next_head = chunks[1].text.splitlines()[:20]
        # At least one of the last 5 lines of chunk 0 reappears in
        # the first 20 lines of chunk 1.
        self.assertTrue(
            any(line in next_head for line in prev_tail),
            "overlap did not preserve context across the boundary",
        )

    def test_empty_input_returns_single_empty_chunk(self):
        chunks = chunk_transcript_for_llm("", max_chars=10_000)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, "")

    def test_invalid_parameters_raise(self):
        with self.assertRaises(ValueError):
            chunk_transcript_for_llm("x", max_chars=0)
        with self.assertRaises(ValueError):
            chunk_transcript_for_llm("x", overlap_chars=-1)
        with self.assertRaises(ValueError):
            chunk_transcript_for_llm("x", max_chars=100, overlap_chars=100)


class DedupeCorrectionsTest(unittest.TestCase):
    def test_drops_exact_duplicates(self):
        items = [
            {"timestamp": "00:01", "original": "foo", "replacement": "bar"},
            {"timestamp": "00:01", "original": "foo", "replacement": "bar"},
        ]
        out = dedupe_corrections(items)
        self.assertEqual(len(out), 1)

    def test_case_and_whitespace_normalisation(self):
        items = [
            {"timestamp": "00:01:00", "original": "  Foo  ", "replacement": "BAR"},
            {"timestamp": "00:01:00", "original": "foo", "replacement": "bar"},
        ]
        # Both entries normalise to the same triple — second one
        # gets dropped.
        out = dedupe_corrections(items)
        self.assertEqual(len(out), 1)

    def test_different_timestamps_keep_both(self):
        items = [
            {"timestamp": "00:01", "original": "foo", "replacement": "bar"},
            {"timestamp": "00:02", "original": "foo", "replacement": "bar"},
        ]
        self.assertEqual(len(dedupe_corrections(items)), 2)


if __name__ == "__main__":
    unittest.main()
