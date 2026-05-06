"""
Tests for unifiedfl/data/dataset.py.

QADataset doesn't require model downloads (uses any tokenizer); we use a stub
tokenizer so tests stay offline.
"""

from __future__ import annotations

import pytest
import torch

from data.dataset import (
    BLOOM_VERBS,
    PROMPT_TEMPLATES,
    QADataset,
    render_prompt,
)


# ── render_prompt ────────────────────────────────────────────────────────────

class TestRenderPrompt:
    def test_baseline(self):
        p = render_prompt({"context": "MLP context"}, "baseline")
        assert "MLP context" in p
        assert "{question_topic}" not in p  # template not left unfilled

    def test_topic_falls_back_when_field_missing(self):
        # Should not crash on missing question_topic — uses "the main topic".
        p = render_prompt({"context": "ctx"}, "topic")
        assert "the main topic" in p

    def test_topic_uses_provided_topic(self):
        p = render_prompt({"context": "ctx", "question_topic": "Backprop"}, "topic")
        assert "Backprop" in p

    def test_bloom_level_clamped_low(self):
        # Spec says levels 1-6; if upstream stored 0 (e.g. parse failure default),
        # render_prompt clamps to 1.
        p = render_prompt({"context": "ctx", "bloom_level": 0}, "bloom")
        assert "Level-1" in p
        assert BLOOM_VERBS[1] in p

    def test_bloom_level_clamped_high(self):
        p = render_prompt({"context": "ctx", "bloom_level": 99}, "bloom")
        assert "Level-6" in p
        assert BLOOM_VERBS[6] in p

    def test_bloom_level_default_when_missing(self):
        # Default is 2 ("Understand"), per render_prompt.
        p = render_prompt({"context": "ctx"}, "bloom")
        assert "Level-2" in p

    def test_bloom_level_empty_string_falls_back_to_default(self):
        # split.py:88 inserts bloom_level="" when the source qa_pair lacks the
        # field (pre-enrichment data). render_prompt must NOT crash with
        # int("") on this — the bug class is "ValueError mid-training run on
        # the very first batch when --conditioning bloom is used against
        # un-enriched data." Falls back to default level=2 instead.
        p = render_prompt({"context": "ctx", "bloom_level": ""}, "bloom")
        assert "Level-2" in p

    def test_bloom_level_non_int_string_falls_back_to_default(self):
        # If the source data carries a textual bloom level (e.g. "understand"),
        # we silently fall back to the default rather than crash.
        p = render_prompt({"context": "ctx", "bloom_level": "understand"}, "bloom")
        assert "Level-2" in p

    def test_unknown_conditioning_raises(self):
        with pytest.raises(ValueError):
            render_prompt({"context": "ctx"}, "totally_made_up")

    def test_all_three_conditionings_have_templates(self):
        # If we add a fourth conditioning later, this is the gate that reminds us
        # to register a template. The set is curated.
        assert set(PROMPT_TEMPLATES) == {"baseline", "topic", "bloom"}


# ── QADataset (stub tokenizer) ───────────────────────────────────────────────

class _StubTokenizer:
    """Tokenizer that hashes characters into ids in a stable way."""
    pad_token_id = 0

    def __call__(self, text, max_length, padding, truncation, return_tensors):
        # Convert text to a list of token ids (capped to max_length).
        ids = [(ord(c) % 200) + 1 for c in text][:max_length]
        attn = [1] * len(ids)
        # Pad to max_length with pad_token_id=0.
        while len(ids) < max_length:
            ids.append(0)
            attn.append(0)
        return {
            "input_ids":      torch.tensor([ids], dtype=torch.long),
            "attention_mask": torch.tensor([attn], dtype=torch.long),
        }


class TestQADataset:
    @pytest.fixture
    def samples(self):
        return [
            {"context": "ctx1", "question": "Q1?", "answer": "A1.",
             "question_topic": "topic1", "bloom_level": 3},
            {"context": "ctx2", "question": "Q2?", "answer": "A2.",
             "question_topic": "topic2", "bloom_level": 5},
        ]

    def test_len_matches_samples(self, samples):
        ds = QADataset(samples, _StubTokenizer(), max_input_len=64, max_target_len=32)
        assert len(ds) == 2

    def test_returns_three_tensors_with_consistent_shapes(self, samples):
        ds = QADataset(samples, _StubTokenizer(), max_input_len=64, max_target_len=32)
        item = ds[0]
        assert set(item.keys()) == {"input_ids", "attention_mask", "labels"}
        assert item["input_ids"].shape == (64,)
        assert item["attention_mask"].shape == (64,)
        assert item["labels"].shape == (32,)

    def test_pad_tokens_in_labels_are_minus_100(self, samples):
        # Critical correctness invariant for HF seq2seq loss: padding positions
        # in labels must be -100 so cross-entropy ignores them. If this regresses,
        # the model trains to emit pad tokens at end of sequences.
        ds = QADataset(samples, _StubTokenizer(), max_input_len=64, max_target_len=32)
        item = ds[0]
        labels = item["labels"]
        # Every position that is the pad token in the raw tokenization (id 0)
        # should now be -100 in labels.
        # Construct the raw target text and tokenize it the same way to compare.
        from data.dataset import TARGET_TEMPLATE
        target_text = TARGET_TEMPLATE.format(question="Q1?", answer="A1.")
        raw = _StubTokenizer()(target_text, max_length=32, padding="max_length",
                                truncation=True, return_tensors="pt")["input_ids"].squeeze(0)
        for i in range(32):
            if raw[i].item() == 0:
                assert labels[i].item() == -100, f"pad pos {i} not masked"
            else:
                assert labels[i].item() == raw[i].item(), f"non-pad pos {i} corrupted"

    def test_baseline_conditioning_used_by_default(self, samples):
        ds = QADataset(samples, _StubTokenizer(), max_input_len=64, max_target_len=32)
        # Default is "baseline" — render_prompt should produce a string that
        # does NOT mention question_topic or bloom_level.
        # We can't directly inspect the rendered prompt from the tensor, but
        # we can check ds.conditioning attribute.
        assert ds.conditioning == "baseline"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
