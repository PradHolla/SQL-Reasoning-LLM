"""Tests for the Spider SFT trainer's data handling.

The prompt/completion split decides which tokens the model is trained on. Get it
wrong and training still runs, loss still falls, and the model learns the wrong
thing -- there is no crash to tell you.
"""

from __future__ import annotations

import inspect

import pytest

from sqlrl.tokenizer import BASE_EOS, CHAT_EOS, build_tokenizer
from sqlrl.training.sft_spider import TRAIN_EOS, to_prompt_completion, train


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


def test_completions_are_trained_to_end_on_a_token_the_base_model_can_emit():
    # Not a style preference. Qwen2.5-0.5B *base* ships the ChatML specials but
    # never trained them: <|im_end|>'s output-embedding row sits in the 1.5th
    # percentile by norm, near random init, while <|endoftext|> is in the 98th.
    # LoRA freezes that embedding (tie_word_embeddings=True, modules_to_save
    # None), so two epochs of loss on <|im_end|> moved nothing and the trained
    # model emitted it never. If anyone "fixes" this back to CHAT_EOS for
    # ChatML tidiness, generation stops terminating and the symptom appears two
    # stages later as a GRPO run where no rollout ends.
    assert TRAIN_EOS == BASE_EOS
    assert TRAIN_EOS != CHAT_EOS


def test_the_trainer_actually_passes_train_eos_to_trl():
    # The constant above is inert unless SFTConfig receives it -- TRL otherwise
    # appends processing_class.eos_token, which is <|im_end|>.
    source = inspect.getsource(train)
    assert "eos_token=TRAIN_EOS" in source


def test_the_trainer_verifies_the_model_stops_before_saving():
    # The defect this guards against produced a checkpoint that looked healthy:
    # loss fell, eval loss fell, and nothing terminated. Only generating catches
    # it, so the trainer must generate before it saves.
    source = inspect.getsource(train)
    assert "assert_model_stops" in source


def test_non_prefix_stable_template_is_rejected(row):
    class Broken:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
            # Renders the prompt differently depending on whether the assistant
            # turn is present -- so the split would mask the wrong tokens.
            return "WITH-TURN" if len(messages) == 3 else "WITHOUT-TURN"

    with pytest.raises(ValueError, match="prefix-stable"):
        to_prompt_completion(row, Broken())
