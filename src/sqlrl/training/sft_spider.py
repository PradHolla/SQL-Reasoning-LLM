"""SFT on Spider train. Replaces the v0 SFT stage.

    uv run python -m sqlrl.training.sft_spider --inspect     # show what the model sees
    uv run python -m sqlrl.training.sft_spider

v0's SFT produced a checkpoint that scored **4.6%** on Spider test while the
untrained base model scored **17.4%**. Fine-tuning was worse than not
fine-tuning. This run tests whether the training *data* was the cause.

What differs from v0, and why. This is more than "just the data", so the list is
explicit rather than buried:

1. **Spider train, full schemas** (the change under test). v0 trained on
   `sql-create-context`, whose schemas are pruned to the columns each question
   needs -- teaching the model not to read the schema at all.
2. **No Unsloth.** It silently rewrote the vocabulary during v0 training,
   swapping the ids of `<|im_end|>` and `<|endoftext|>`, which made every later
   checkpoint depend on carrying its own broken tokenizer around. Plain
   transformers + peft instead, at 0.5B the speed difference does not justify
   the risk.
3. **Tokenizer from `sqlrl.tokenizer.build_tokenizer`** -- the same function the
   evaluator uses, so training and evaluation cannot disagree about tokens.
4. **Loss on the completion only.** v0 computed loss over the whole sequence.
   That was survivable with pruned schemas; with full ones the prompt is most of
   the sequence, so ~80% of the gradient would be spent teaching the model to
   *generate database schemas* rather than SQL. `--inspect` prints exactly which
   tokens are trained on.
5. **Two real epochs.** v0 ran 0.10 epochs -- 500 steps over 78k rows, so it saw
   about 8k examples once.
6. **lr 1e-4** rather than 2e-5. LoRA adapters start from zero and need a higher
   rate; 2e-5 for a tenth of an epoch is most of why v0 barely moved.

Writes to `models/qwen-0.5b-sft-spider`. The v0 adapters are never touched --
they are the baseline that makes every later number mean something.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer

from sqlrl.tokenizer import (
    BASE_EOS,
    CHAT_EOS,
    assert_model_stops,
    build_tokenizer,
    special_token_ids,
)

__all__ = ["train"]

BASE_MODEL = "Qwen/Qwen2.5-0.5B"
SFT_DATA = Path("data/processed/spider_sft.jsonl")
VAL_DATA = Path("data/processed/spider_val.jsonl")
OUTPUT = Path("models/qwen-0.5b-sft-spider")
CHECKPOINTS = Path("outputs/sft_spider")

#: Identical to v0's adapter_config.json, so the comparison isolates the data
#: rather than confounding it with capacity.
LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

#: Longest example in the dataset is 1,742 tokens. TRL's default is 1,024, which
#: would silently truncate ~5% of the data mid-query.
MAX_LENGTH = 2_048

#: The token completions are trained to end on, and it is deliberately NOT
#: ChatML's <|im_end|>.
#:
#: The first run of this trainer taught the model to end turns with <|im_end|>
#: -- the data was correct, the loss mask covered that token, and two epochs ran
#: over it. The model still emits <|im_end|> essentially never. After </answer>
#: its next-token distribution is near-uniform (top token 0.037%), it picks a
#: junk byte, and only then falls back to <|endoftext|>.
#:
#: The cause is in the base model, not in the data. Qwen2.5-0.5B **base** carries
#: the ChatML specials in its vocabulary but never trained them -- only the
#: Instruct variants use ChatML. Measured on the stock checkpoint's output
#: embedding:
#:
#:     <|endoftext|>   row norm 0.5987   98.40th percentile
#:     <|im_end|>      row norm 0.3010    1.52th percentile
#:     <|im_start|>    row norm 0.3010    1.69th percentile
#:
#: <|im_end|> is still sitting at its random initialisation. LoRA targets only
#: the attention and MLP projections and `modules_to_save` is None, so the
#: embedding is frozen -- and with tie_word_embeddings=True that embedding *is*
#: the output head. To emit <|im_end|> the model would have to steer its hidden
#: state onto a near-random 896-dim vector while competing with a well-trained
#: <|endoftext|>. The gradient had nowhere to go.
#:
#: So train on the token the model can actually produce. The alternative is to
#: put embed_tokens in `modules_to_save` and learn the ChatML specials properly,
#: which is defensible but adds 136M trainable parameters (the embedding is ~27%
#: of this model) and changes the input embeddings too. Not worth it to move a
#: stop token.
#:
#: <|im_end|> stays in the *prompt* as a turn separator. Reading a weakly-trained
#: token is a far easier job than producing one, and 44.6% says it copes.
TRAIN_EOS = BASE_EOS

#: Batch size is bounded by the logits tensor, not the model. Qwen2.5's
#: vocabulary is 151,936, so logits are batch x seq_len x 151,936 upcast to fp32
#: for the loss: at batch 8 and a 1,742-token example that is 8.5 GB on its own,
#: which OOMs a 22 GB A10G. Batch 4 halves it to ~4.2 GB. The model itself is
#: only ~1 GB -- it is never the constraint here.
#:
#: A short smoke run will not catch this: with --limit the long examples are
#: usually not in the sample, and the failure only appears when one is.


def to_prompt_completion(row: dict, tokenizer) -> dict:
    """Split a conversation into the masked prompt and the trained completion."""
    messages = row["messages"]
    full = tokenizer.apply_chat_template(messages, tokenize=False)
    prompt = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True
    )
    if not full.startswith(prompt):
        raise ValueError(
            "chat template is not prefix-stable: rendering the conversation "
            "without the final turn is not a prefix of rendering it with. "
            "Completion-only loss would mask the wrong tokens."
        )

    # The chat template already closes the turn with <|im_end|>, and TRL appends
    # an eos of its own. Left alone this trains the model to emit two stop
    # tokens. Strip the template's copy and let TRL supply the single one --
    # which is TRAIN_EOS, not <|im_end|>. See the note on TRAIN_EOS above.
    completion = full[len(prompt):].rstrip()
    if completion.endswith(CHAT_EOS):
        completion = completion[: -len(CHAT_EOS)].rstrip()
    return {"prompt": prompt, "completion": completion}


def load_split(path: Path, tokenizer, limit: int | None = None):
    dataset = load_dataset("json", data_files=str(path), split="train")
    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))
    return dataset.map(
        lambda row: to_prompt_completion(row, tokenizer),
        remove_columns=dataset.column_names,
    )


def inspect(tokenizer, dataset) -> None:
    """Print one example exactly as the trainer will see it, with the loss mask.

    Worth doing before every training run: it is the difference between knowing
    the model is trained on the answer and assuming it.
    """
    row = dataset[0]
    print("=== PROMPT (masked, no loss) ===")
    print(row["prompt"])
    print("=== COMPLETION (trained on) ===")
    print(row["completion"])

    prompt_ids = tokenizer(row["prompt"])["input_ids"]
    completion_ids = tokenizer(row["completion"])["input_ids"]
    print(f"\nprompt {len(prompt_ids)} tokens (masked) | "
          f"completion {len(completion_ids)} tokens (trained)")
    print(f"fraction of tokens carrying loss: "
          f"{len(completion_ids) / (len(prompt_ids) + len(completion_ids)):.1%}")
    print(f"completion ends with: {tokenizer.convert_ids_to_tokens(completion_ids[-3:])}")
    print(f"special token ids: {special_token_ids(tokenizer)}")


def train(
    *,
    epochs: float = 2.0,
    learning_rate: float = 1e-4,
    batch_size: int = 4,
    grad_accum: int = 4,
    limit: int | None = None,
    max_steps: int = -1,
    seed: int = 3407,
    report_to: str = "wandb",
    output: Path = OUTPUT,
    data: Path = SFT_DATA,
    val_data: Path = VAL_DATA,
) -> None:
    # Clear any previous checkpoint before training, not after. A crashed run
    # that leaves a stale adapter behind is worse than one that leaves nothing:
    # the next evaluation scores it happily and reports a real-looking number
    # for a model that was never trained. This already happened once, with a
    # 5-step smoke checkpoint.
    if output.exists():
        print(f"removing previous checkpoint at {output}")
        shutil.rmtree(output)

    tokenizer = build_tokenizer(BASE_MODEL, chat=True)
    train_dataset = load_split(data, tokenizer, limit)
    eval_dataset = load_split(val_data, tokenizer, limit)
    print(f"train from {data}\nval   from {val_data}")
    print(f"train {len(train_dataset)} | val {len(eval_dataset)}")

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16)
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=16,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=LORA_TARGETS,
        ),
    )
    model.print_trainable_parameters()

    args = SFTConfig(
        output_dir=str(CHECKPOINTS),
        num_train_epochs=epochs,
        max_steps=max_steps,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        bf16=torch.cuda.is_available(),
        max_length=MAX_LENGTH,
        packing=False,
        # The whole point of the prompt/completion split above.
        completion_only_loss=True,
        # Overrides the tokenizer's <|im_end|>, which this base model cannot
        # learn to emit. See TRAIN_EOS.
        eos_token=TRAIN_EOS,
        logging_steps=10,
        eval_strategy="epoch",
        per_device_eval_batch_size=batch_size,
        save_strategy="epoch",
        save_total_limit=2,
        seed=seed,
        report_to=report_to,
        optim="adamw_torch",
        weight_decay=0.01,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=args,
    )
    trainer.train()

    # Ask the weights, not the config. Two epochs of loss on <|im_end|> produced
    # a model that emits it never, and nothing in this script noticed -- the run
    # looked healthy, the loss fell, and the defect only surfaced two phases
    # later as a GRPO pilot where no rollout terminated. Checked here now, while
    # the model is still in memory and rerunning costs ten minutes.
    prompts = [
        tokenizer.apply_chat_template(
            row["messages"][:-1], tokenize=False, add_generation_prompt=True
        )
        for row in load_dataset("json", data_files=str(val_data), split="train").select(range(4))
    ]
    lengths = assert_model_stops(model, tokenizer, prompts, max_new_tokens=320)
    print(f"stop check: completions terminate in {lengths} tokens, "
          f"stopping on {tokenizer.eos_token!r} (id {tokenizer.eos_token_id})")

    output.parent.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output))
    # The tokenizer travels with the checkpoint -- see sqlrl.tokenizer.
    tokenizer.save_pretrained(str(output))
    print(f"saved to {output}")


def main() -> int:
    parser = argparse.ArgumentParser(description="SFT on Spider train.")
    parser.add_argument("--inspect", action="store_true",
                        help="print one example with its loss mask, then exit")
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--report-to", default="wandb")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--data", type=Path, default=SFT_DATA,
                        help="training jsonl; point at spider_sft_traces.jsonl for Phase 3")
    parser.add_argument("--val-data", type=Path, default=VAL_DATA)
    args = parser.parse_args()

    if args.inspect:
        tokenizer = build_tokenizer(BASE_MODEL, chat=True)
        inspect(tokenizer, load_split(args.data, tokenizer, limit=4))
        return 0

    train(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        limit=args.limit,
        max_steps=args.max_steps,
        seed=args.seed,
        report_to=args.report_to,
        output=args.output,
        data=args.data,
        val_data=args.val_data,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
