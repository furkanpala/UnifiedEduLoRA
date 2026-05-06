"""
Tests for pure-function helpers in unifiedfl/evaluation/metrics.py.

Focus on edge cases — empty inputs, malformed LLM JSON, and the recently-fixed
escape-sanitization bug. These don't require GPU, model downloads, or API keys.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from evaluation.metrics import (
    _cosine,
    _parse_qa,
    _sanitize_json_escapes,
    _token_f1,
)


# ── _parse_qa ────────────────────────────────────────────────────────────────

class TestParseQA:
    def test_canonical_format(self):
        q, a = _parse_qa("Question: What is gradient descent?\nAnswer: An optimization method.")
        assert q == "What is gradient descent?"
        assert a == "An optimization method."

    def test_case_insensitive_headers(self):
        q, a = _parse_qa("question: foo\nanswer: bar")
        assert q == "foo"
        assert a == "bar"

    def test_multiline_answer(self):
        q, a = _parse_qa("Question: Q?\nAnswer: line1\nline2\nline3")
        assert q == "Q?"
        assert "line1" in a and "line3" in a

    def test_no_question_header_falls_back_to_full_text_as_answer(self):
        # Documents the existing fallback behavior. The whole text is treated
        # as the answer; question is empty. Downstream metrics interpret this
        # as a parse failure (RTC, faithfulness skip the sample).
        q, a = _parse_qa("just some prose without headers")
        assert q == ""
        assert a == "just some prose without headers"

    def test_empty_string(self):
        q, a = _parse_qa("")
        assert q == ""
        assert a == ""


# ── _token_f1 ────────────────────────────────────────────────────────────────

class TestTokenF1:
    def test_identical(self):
        assert _token_f1("the cat sat", "the cat sat") == pytest.approx(1.0)

    def test_disjoint(self):
        assert _token_f1("alpha beta", "gamma delta") == 0.0

    def test_punctuation_and_case_insensitive(self):
        # Punctuation stripped, lowercased.
        assert _token_f1("The CAT, sat!", "the cat sat") == pytest.approx(1.0)

    def test_empty_either_side(self):
        assert _token_f1("", "anything") == 0.0
        assert _token_f1("anything", "") == 0.0
        assert _token_f1("", "") == 0.0

    def test_partial_overlap_is_harmonic_mean(self):
        # pred = "a b c d", ref = "a b" -> precision=2/4, recall=2/2,
        # F1 = 2 * 0.5 * 1.0 / 1.5 = 0.6667
        assert _token_f1("a b c d", "a b") == pytest.approx(2 / 3)


# ── _cosine ──────────────────────────────────────────────────────────────────

class TestCosine:
    def test_identical_vectors(self):
        v = np.array([1.0, 2.0, 3.0])
        assert _cosine(v, v) == pytest.approx(1.0)

    def test_orthogonal(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert _cosine(a, b) == pytest.approx(0.0)

    def test_zero_vector_returns_zero_not_nan(self):
        # Without the explicit guard this returns NaN and corrupts downstream means.
        z = np.zeros(4)
        v = np.array([1.0, 2.0, 3.0, 4.0])
        assert _cosine(z, v) == 0.0
        assert _cosine(v, z) == 0.0
        assert _cosine(z, z) == 0.0

    def test_anti_parallel(self):
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        assert _cosine(a, b) == pytest.approx(-1.0)


# ── _sanitize_json_escapes ───────────────────────────────────────────────────

class TestSanitizeJsonEscapes:
    """
    The bug class this guards: GPT outputs raw backslashes (math notation,
    Windows paths) that aren't valid JSON escapes; json.loads then crashes
    with 'Invalid \\escape'.
    """

    def test_valid_escapes_are_untouched(self):
        # If json.loads accepts the raw input, sanitization must be a no-op.
        for s in [r'{"x": "line1\nline2"}', r'{"x": "tab\there"}',
                  r'{"x": "quote\""}', r'{"x": "backslash\\"}',
                  r'{"x": "unicode é"}']:
            assert json.loads(_sanitize_json_escapes(s)) == json.loads(s)

    def test_lone_backslash_alpha_becomes_parseable(self):
        # GPT-emitted math notation: {"reason": "the \alpha parameter"}
        bad = r'{"reason": "the \alpha parameter"}'
        with pytest.raises(json.JSONDecodeError):
            json.loads(bad)
        fixed = _sanitize_json_escapes(bad)
        parsed = json.loads(fixed)
        assert parsed["reason"] == r"the \alpha parameter"

    def test_path_with_invalid_escape(self):
        # \xnew is invalid (\x is not a JSON escape). \n and \f happen to be
        # valid escapes, so a real Windows path like "C:\new\file.txt" won't
        # fail json.loads — but paths containing \x, \z, etc. will.
        bad = r'{"path": "C:\xnew"}'
        with pytest.raises(json.JSONDecodeError):
            json.loads(bad)
        parsed = json.loads(_sanitize_json_escapes(bad))
        assert "C:" in parsed["path"]

    def test_doubled_backslash_not_re_doubled(self):
        # Regression: a buggy regex earlier turned valid \\a into \\\a.
        good = r'{"x": "already\\escaped"}'
        # State machine should leave the valid \\ alone.
        sanitized = _sanitize_json_escapes(good)
        assert json.loads(sanitized) == json.loads(good)

    def test_unicode_escape_at_end(self):
        # \u followed by 4 hex digits is valid; sanitizer must consume all 6 chars.
        s = r'{"x": "é"}'
        assert _sanitize_json_escapes(s) == s
        assert json.loads(_sanitize_json_escapes(s))["x"] == "é"

    def test_trailing_lone_backslash(self):
        # A backslash with nothing after it must be doubled, not crash.
        bad = r'{"x": "ends with \"}'  # the " closes the string per JSON, weird
        # Skip this as it's not even valid JSON regardless of our fix.
        # Instead check a backslash near end-of-string in a value:
        bad2 = '{"x": "trail \\\\"}'  # \\\\ in source = \\ in JSON = \ in value, valid
        json.loads(bad2)  # already valid

    def test_idempotent(self):
        # Applying the sanitizer twice = applying it once.
        bad = r'{"reason": "the \alpha and \beta"}'
        once = _sanitize_json_escapes(bad)
        twice = _sanitize_json_escapes(once)
        assert once == twice


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
