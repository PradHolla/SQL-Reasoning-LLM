"""Tests for the Spider SFT trainer's data handling.

The prompt/completion split decides which tokens the model is trained on. Get it
wrong and training still runs, loss still falls, and the model learns the wrong
thing -- there is no crash to tell you.
"""

from __future__ import annotations

import pytest

from sqlrl.tokenizer import CHAT_EOS, build_tokenizer
from sqlrl.training.sft_spider import to_prompt_completion


@pytest.fixture(scope="module")
def tokenizer():
    return build_tokenizer("Qwen/Qwen2.5-0.5B", chat=True)


@pytest.fixture
def row():
    return {
        "messages": [
            {"role": "system", "content": "You are a database expert."},
            {"role": "user", "content": "Schema: CREATE TABLE t (a INT)\nQuestion: how many?"},
            {"role": "assistant", "content": "<think>\nthinking\n</think>\n<answer>\nSELECT count(*) FROM t\n</answer>"},
        ]
    }


def test_prompt_ends_at_the_generation_point(tokenizer, row):
    out = to_prompt_completion(row, tokenizer)
    assert out["prompt"].endswith("<|im_start|>assistant\n")


def test_completion_holds_the_answer(tokenizer, row):
    out = to_prompt_completion(row, tokenizer)
    assert "SELECT count(*) FROM t" in out["completion"]
    # ...and none of the question, which must stay masked.
    assert "CREATE TABLE" not in out["completion"]


def test_completion_does_not_end_with_a_stop_token(tokenizer, row):
    # TRL appends its own eos. Leaving the template's copy in trains the model
    # to emit two stop tokens in a row.
    out = to_prompt_completion(row, tokenizer)
    assert not out["completion"].endswith(CHAT_EOS)
    assert out["completion"].endswith("</answer>")


def test_prompt_and_completion_reconstruct_the_conversation(tokenizer, row):
    out = to_prompt_completion(row, tokenizer)
    full = tokenizer.apply_chat_template(row["messages"], tokenize=False)
    assert full.startswith(out["prompt"] + out["completion"])


def test_prompt_is_the_larger_half(tokenizer, row):
    # Sanity on the reason completion-only loss matters: with full schemas the
    # prompt dominates, so full-sequence loss mostly teaches schema generation.
    out = to_prompt_completion(row, tokenizer)
    assert len(tokenizer(out["prompt"])["input_ids"]) > len(
        tokenizer(out["completion"])["input_ids"]
    )


def test_non_prefix_stable_template_is_rejected(row):
    class Broken:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
            # Renders the prompt differently depending on whether the assistant
            # turn is present -- so the split would mask the wrong tokens.
            return "WITH-TURN" if len(messages) == 3 else "WITHOUT-TURN"

    with pytest.raises(ValueError, match="prefix-stable"):
        to_prompt_completion(row, Broken())
