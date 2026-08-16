"""transformers + peft generation backend. The reference implementation.

Deliberately boring: load the base model in bf16, apply the LoRA adapter with
``PeftModel``, generate greedily. Slow, but it is the thing vLLM gets checked
against.

Four decisions that would each produce a confidently wrong benchmark number:

* **No Unsloth.** It is a training-time optimisation; evaluation should measure
  the model, not the training framework. The v0 adapters were trained against a
  4-bit base, and applying them to the bf16 base is standard QLoRA practice but
  is not numerically identical to training conditions. That is fine as long as
  every model is evaluated the same way.
* **The tokenizer comes from the checkpoint, never from the base model name.**
  Unsloth's ``get_chat_template`` rewrote the vocabulary during v0 training, so
  the SFT and GRPO checkpoints put ``<|im_end|>`` at id 151643 where stock
  Qwen2.5 puts ``<|endoftext|>``. Pair those adapters with a stock tokenizer and
  generation never stops -- every score reads near zero for a reason that has
  nothing to do with the model. ``assert_stops_on`` turns that into an
  immediate error.
* **Left padding.** Decoder-only models generate from the right edge of the
  batch. With the default right padding, every sequence shorter than the
  longest one continues from pad tokens and produces garbage -- and it degrades
  silently, looking like a bad model rather than a bad harness.
* **Greedy decoding.** ``do_sample=False`` and a fixed seed, so a re-run of the
  same checkpoint gives the same number. v0's ``inference.py`` passed
  ``temperature`` without ``do_sample``, which transformers silently ignores.
"""

from __future__ import annotations

from typing import Callable, TypeVar

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

from sqlrl.eval.backends import Backend
from sqlrl.eval.prompts import Prompt
from sqlrl.tokenizer import BASE_EOS, CHAT_EOS, assert_stops_within, build_tokenizer

__all__ = ["MAX_INPUT_TOKENS", "HFBackend", "batched_in_order", "pick_device"]

T = TypeVar("T")

#: Hard ceiling on the rendered prompt. Exceeding it is an error rather than a
#: silent truncation -- see ``_check_fits``.
#:
#: Single-turn Spider prompts top out at 610 tokens, so the old hardcoded 2048
#: never bound anything. The retry loop changes that: a multiturn round-3 prompt
#: is the base prompt plus two (answer, feedback) pairs, so with
#: ``--max-new-tokens 640`` the worst case is 610 + 2*(640 + 200) = 2,290 -- over
#: the old ceiling, and it would have killed a run 15 minutes in. 3072 clears the
#: worst case this project can generate with headroom. It costs KV cache, not the
#: fp32 logits tensor that bounds *training* batch size, so it is cheap here.
MAX_INPUT_TOKENS = 3072


def batched_in_order(
    items: list[str],
    batch_size: int,
    run_batch: Callable[[list[str]], list[T]],
) -> list[T]:
    """Process in length-sorted batches, return results in the *original* order.

    Sorting by length keeps batches from being mostly padding, which is wasted
    compute. Getting the unsort wrong would silently pair every prediction with
    someone else's question, so this is separated out to be tested without
    loading a model.

    Generic in the result type: greedy decoding returns one string per prompt,
    sampling returns a tuple of ``n`` per prompt, and both need the same
    length-sort-and-restore.
    """
    order = sorted(range(len(items)), key=lambda i: len(items[i]))
    results: list[T] = [None] * len(items)  # type: ignore[list-item]

    for start in range(0, len(order), batch_size):
        indices = order[start : start + batch_size]
        outputs = run_batch([items[i] for i in indices])
        if len(outputs) != len(indices):
            raise ValueError(
                f"backend returned {len(outputs)} results for {len(indices)} prompts"
            )
        for index, output in zip(indices, outputs):
            results[index] = output
    return results


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _pick_dtype(device: str) -> torch.dtype:
    # bf16 everywhere it works; float32 on CPU, where bf16 is slower, not faster.
    return torch.float32 if device == "cpu" else torch.bfloat16


class HFBackend(Backend):
    """Generate with transformers, optionally applying a LoRA adapter.

    ``model_path`` is either a full model (base or merged) or a directory of
    LoRA adapters, in which case ``base_model`` must name the base to apply them
    to. Either way the *tokenizer* is built from ``model_path``, because that is
    where the training-time vocabulary lives.

    ``chat`` must match how the checkpoint was trained: True for SFT/GRPO and
    for instruct models, False for CPT and raw base models. Getting it wrong
    does not crash, it just quietly measures the wrong thing.
    """

    def __init__(
        self,
        model_path: str,
        *,
        name: str | None = None,
        base_model: str | None = None,
        chat: bool = True,
        device: str | None = None,
        dtype: torch.dtype | None = None,
        batch_size: int = 8,
        seed: int = 3407,
    ) -> None:
        self.name = name or model_path
        self.model_path = model_path
        self.batch_size = batch_size
        self.device = device or pick_device()
        self.dtype = dtype or _pick_dtype(self.device)
        self.chat = chat

        torch.manual_seed(seed)

        self.tokenizer = build_tokenizer(model_path, chat=chat)
        # Decoder-only generation must pad on the left. See module docstring.
        self.tokenizer.padding_side = "left"
        # And truncate on the left too. The default is the right, which on a
        # ChatML prompt deletes the question and the generation prompt while
        # keeping the schema -- the model is then scored for not answering a
        # question it was never shown. ``_check_fits`` means this should never
        # fire; it is set anyway so the fallback is the less destructive one.
        self.tokenizer.truncation_side = "left"

        # Stop on any end-of-turn token this vocabulary has, not just the one
        # nominated as eos. A base model given a ChatML prompt may end its turn
        # with either, and running past the end only adds text to strip.
        self.stop_ids = sorted(
            {self.tokenizer.eos_token_id}
            | {
                token_id
                for token_id in (
                    self.tokenizer.convert_tokens_to_ids(BASE_EOS),
                    self.tokenizer.convert_tokens_to_ids(CHAT_EOS),
                )
                if token_id is not None and token_id >= 0
            }
        )
        # The invariant is "we stop on something this checkpoint emits", not
        # "eos matches exactly" -- see assert_stops_within. Strict equality would
        # reject running a checkpoint against a prompt format it was not trained
        # on, which is exactly how a weights effect gets separated from a prompt
        # effect.
        assert_stops_within(model_path, self.stop_ids)

        source = base_model or model_path
        model = AutoModelForCausalLM.from_pretrained(source, dtype=self.dtype)
        if base_model is not None:
            model = PeftModel.from_pretrained(model, model_path)

        self.model = model.to(self.device).eval()

    def render(self, prompt: Prompt) -> str:
        """Prompt -> the exact string handed to the tokenizer."""
        if prompt.text is not None:
            return prompt.text
        return self.tokenizer.apply_chat_template(
            prompt.messages, tokenize=False, add_generation_prompt=True
        )

    def _check_fits(self, rendered: list[str]) -> None:
        """Refuse to generate from a prompt that would be silently truncated.

        Checked up front, over every prompt, rather than per batch: a truncation
        that surfaces twenty minutes into a run has already spent the GPU time,
        and one that never surfaces at all produces a number that is wrong for a
        reason nothing in the output would reveal.
        """
        lengths = [len(self.tokenizer(text)["input_ids"]) for text in rendered]
        longest = max(lengths, default=0)
        if longest > MAX_INPUT_TOKENS:
            over = sum(length > MAX_INPUT_TOKENS for length in lengths)
            raise ValueError(
                f"{over} of {len(rendered)} prompts exceed MAX_INPUT_TOKENS "
                f"({longest} > {MAX_INPUT_TOKENS}). Truncating them would drop "
                f"part of the prompt and score the model for it. Shorten the "
                f"schema, lower --max-new-tokens, or raise the ceiling knowing "
                f"what it costs in memory."
            )

    @torch.inference_mode()
    def generate(self, prompts: list[Prompt], max_new_tokens: int = 512) -> list[str]:
        rendered = [self.render(prompt) for prompt in prompts]
        self._check_fits(rendered)
        return batched_in_order(
            rendered,
            self.batch_size,
            lambda texts: self._generate_batch(texts, max_new_tokens),
        )

    @torch.inference_mode()
    def sample(
        self,
        prompts: list[Prompt],
        *,
        n: int,
        temperature: float = 0.8,
        top_p: float = 0.95,
        max_new_tokens: int = 512,
    ) -> list[list[str]]:
        """``n`` sampled completions per prompt, for execution voting.

        ``do_sample=True`` is set explicitly alongside ``temperature``. v0's
        ``inference.py`` passed ``temperature`` on its own, which transformers
        silently ignores -- inference had been greedy the whole time while
        reporting as sampled. That bug is the reason this is a separate method
        from ``generate`` rather than a flag on it: the two decoding regimes
        cannot be confused if they do not share a call site.

        The batch is divided by ``n``, because ``num_return_sequences`` puts
        ``batch_size * n`` sequences in flight at once and the KV cache is what
        runs out first. At 1.5B with 16x8 that is ~19 GB of cache on a 24 GB
        card, which OOMs after the weights; dividing keeps the in-flight count
        at whatever ``--batch-size`` already proved safe.
        """
        rendered = [self.render(prompt) for prompt in prompts]
        self._check_fits(rendered)
        per_batch = max(1, self.batch_size // n)

        flat = batched_in_order(
            # One entry per (prompt, sample) so batched_in_order's length-sorted
            # unshuffle keeps working; the run_batch below expands each prompt.
            rendered,
            per_batch,
            lambda texts: self._sample_batch(texts, n, temperature, top_p, max_new_tokens),
        )
        return [list(group) for group in flat]

    def _sample_batch(
        self,
        texts: list[str],
        n: int,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
    ) -> list[tuple[str, ...]]:
        batch = self.tokenizer(
            texts, return_tensors="pt", padding=True,
            truncation=True, max_length=MAX_INPUT_TOKENS,
        ).to(self.device)

        output = self.model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=n,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.stop_ids,
        )

        prompt_len = batch["input_ids"].shape[1]
        decoded = self.tokenizer.batch_decode(
            output[:, prompt_len:], skip_special_tokens=True
        )
        # generate returns rows grouped by input: prompt 0's n samples, then
        # prompt 1's, and so on. Regrouping wrongly here would attribute one
        # question's samples to another and corrupt every vote silently.
        return [tuple(decoded[i * n : (i + 1) * n]) for i in range(len(texts))]

    def _generate_batch(self, texts: list[str], max_new_tokens: int) -> list[str]:
        batch = self.tokenizer(
            texts, return_tensors="pt", padding=True,
            truncation=True, max_length=MAX_INPUT_TOKENS,
        ).to(self.device)

        output = self.model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.stop_ids,
        )

        # Everything before this index is prompt. Left padding makes the prompt
        # width identical across the batch, so one slice is correct for all rows.
        prompt_len = batch["input_ids"].shape[1]
        return self.tokenizer.batch_decode(
            output[:, prompt_len:], skip_special_tokens=True
        )
