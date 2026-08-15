"""What you need to know about a base model before you train on it.

    uv run python -m sqlrl.base_report Qwen/Qwen2.5-Coder-1.5B
    uv run python -m sqlrl.base_report Qwen/Qwen2.5-0.5B Qwen/Qwen2.5-Coder-1.5B

Phase 3 lost a GRPO pilot run to a fact nobody had checked: Qwen2.5-0.5B **base**
ships the ChatML special tokens in its vocabulary but never trained them, so no
amount of loss on `<|im_end|>` could teach the model to emit it. The embedding
was frozen by LoRA and, under weight tying, that embedding *is* the output head.
The symptom appeared two stages later as rollouts that never terminated.

That was a five-second check that nobody ran. This is the five-second check.

Run it before adopting any new base. It answers:

* **Can this model be taught to end a turn with `<|im_end|>`?** If that token's
  output-embedding row is sitting near random initialisation, the honest answer
  is no, and `TRAIN_EOS` should be `<|endoftext|>` instead. Instruction-tuned
  variants have trained it; base variants generally have not.
* **How big is the logits tensor going to be?** Batch size in this project is
  bounded by `batch x seq_len x vocab_size` upcast to fp32 for the loss, not by
  the weights. That killed a Phase 1.5 run at batch 8, and the arithmetic
  changes with every new base.
* **Are the embeddings tied?** If they are, making the embedding trainable also
  changes the input side, which is a much larger commitment than "let it learn a
  stop token".
"""

from __future__ import annotations

import argparse

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from sqlrl.tokenizer import BASE_EOS, CHAT_EOS

__all__ = ["report"]

SPECIALS = (BASE_EOS, "<|im_start|>", CHAT_EOS)

#: Below this percentile of the vocabulary's row norms, treat a token as
#: effectively untrained. Qwen2.5-0.5B base puts <|im_end|> at the 1.5th
#: percentile against <|endoftext|> at the 98.4th, so the two populations are
#: nowhere near each other and the exact cut hardly matters.
UNTRAINED_PERCENTILE = 10.0


def report(model_name: str, batch: int = 4, seq_len: int = 2048) -> dict:
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"=== {model_name} ===")
    params = getattr(config, "num_parameters", None)
    print(f"  hidden {config.hidden_size}  layers {config.num_hidden_layers}  "
          f"vocab {config.vocab_size}  tied_embeddings {config.tie_word_embeddings}")
    print(f"  tokenizer eos {tokenizer.eos_token!r} "
          f"(id {tokenizer.eos_token_id})")

    # The constraint that actually decides batch size. Logits are materialised
    # for every position and upcast to fp32 for the cross-entropy.
    logits_gb = batch * seq_len * config.vocab_size * 4 / 1024**3
    print(f"\n  logits tensor at batch={batch}, seq={seq_len}: {logits_gb:.1f} GB fp32")
    print(f"    (this, not the weights, is what OOMs a 24 GB card)")

    print("\n  loading weights to inspect the output embedding...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float16, low_cpu_mem_usage=True
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  parameters: {n_params / 1e9:.2f}B")

    weights = model.get_output_embeddings().weight
    norms = weights.float().norm(dim=1)

    print("\n  special tokens in the output embedding:")
    verdict = {}
    for token in SPECIALS:
        tid = tokenizer.convert_tokens_to_ids(token)
        if tid is None or tid >= norms.numel():
            print(f"    {token:15s} NOT IN VOCAB")
            continue
        norm = norms[tid].item()
        pct = (norms < norm).float().mean().item() * 100
        trained = pct >= UNTRAINED_PERCENTILE
        verdict[token] = trained
        flag = "" if trained else "   <- effectively UNTRAINED"
        print(f"    {token:15s} id {tid}  norm {norm:.4f}  percentile {pct:5.2f}%{flag}")

    can_emit_chat_eos = verdict.get(CHAT_EOS, False)
    train_eos = CHAT_EOS if can_emit_chat_eos else BASE_EOS
    print(f"\n  VERDICT: train completions to end on {train_eos!r}")
    if not can_emit_chat_eos:
        print(f"    {CHAT_EOS} is near its random initialisation in this base, and LoRA")
        print(f"    freezes the embedding, so loss on it cannot move the model. Emitting")
        print(f"    it would mean aiming the hidden state at a near-random vector while")
        print(f"    competing with a well-trained {BASE_EOS}.")
        if config.tie_word_embeddings:
            print(f"    Embeddings are tied, so making it trainable also changes the")
            print(f"    input side -- a much bigger commitment than moving a stop token.")
    else:
        print(f"    This base has trained {CHAT_EOS}, so ChatML termination works here")
        print(f"    and the Phase 3 workaround is unnecessary.")

    del model
    return {"train_eos": train_eos, "vocab_size": config.vocab_size,
            "hidden_size": config.hidden_size, "tied": config.tie_word_embeddings}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="+")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=2048)
    args = parser.parse_args()
    for name in args.models:
        report(name, batch=args.batch, seq_len=args.seq_len)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
