"""GRPO on Spider train, starting from the SFT checkpoint.

    uv run python -m sqlrl.training.grpo_spider --inspect     # show what it will train on
    uv run python -m sqlrl.training.grpo_spider --pilot       # 30 steps, watch the diagnostics
    uv run python -m sqlrl.training.grpo_spider

A rewrite of `grpo_trainer.py`, not a patch of it. v0's run produced no learning
and the diagnostics said so at the time: `frac_reward_zero_std` sat between 0.5
and 1.0, meaning between half and all of every batch had identical rewards
within the group, an advantage of exactly zero, and therefore no gradient.

What changed, and why each one mattered:

1. **Execution-grounded reward** (`rewards.py`) instead of exact string match.
   `age>56` scored 0 against `age > 56` in v0 -- identical queries, identical
   results. The new reward runs both and compares result sets.
2. **The reward is tiered, not binary.** Partial credit for parsing and for
   running is what gives a group variance on prompts where no sample is
   correct. Those prompts were the majority, and v0 learned nothing from them.
3. **`scale_rewards="none"`.** The single most important line in this file, and
   the least obvious -- see the block comment on the config below.
4. **No format reward.** It sat at 1.000 for all 300 v0 steps. A reward every
   sample earns is a constant: it contributed exactly zero to the advantage
   while inflating the headline number by +1.0.
5. **`num_generations` 4 -> 8.** More samples per group is the cheapest
   available way to reduce zero-variance groups.
6. **A disjoint split.** v0 ran GRPO on `select(range(5000))` of the same rows
   SFT trained on, so nothing was held out. `spider_grpo.jsonl` shares no
   database with the SFT or validation splits.
7. **Starting from a policy worth reinforcing.** v0 started GRPO from an SFT
   checkpoint scoring 4.6%; correct samples were so rare that groups were
   almost always uniformly wrong. `sft-spider-2ep` scores 44.6%, so roughly
   four samples in eight are correct and groups genuinely split. RL amplifies
   what a policy already does sometimes -- it cannot create the behaviour.
8. **No Unsloth.** It rewrote the vocabulary during v0 training. See
   `sqlrl.tokenizer`.

The bar: Spider test execution accuracy above **44.6%**, and
`frac_reward_zero_std` meaningfully below v0's 0.5-1.0. The second without the
first means the reward is being satisfied but not the task.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

from sqlrl.tokenizer import build_tokenizer
from sqlrl.training.rewards import SQLReward, drop_empty_gold

__all__ = ["train"]

BASE_MODEL = "Qwen/Qwen2.5-0.5B"
SFT_CHECKPOINT = Path("models/qwen-0.5b-sft-spider-2ep")
GRPO_DATA = Path("data/processed/spider_grpo.jsonl")
OUTPUT = Path("models/qwen-0.5b-grpo-spider")
CHECKPOINTS = Path("outputs/grpo_spider")

#: Measured over the split with the checkpoint's own tokenizer: prompts run to
#: 401 tokens and gold completions to 234. Both caps sit clear of those, so
#: nothing is silently truncated -- TRL truncates prompts from the *left*, which
#: on our format would eat the schema and leave the question, producing rollouts
#: that cannot succeed and a reward that quietly punishes the model for it.
MAX_PROMPT_LENGTH = 512
MAX_COMPLETION_LENGTH = 320

#: 8 completions per device step x 8 accumulation steps = 8 prompts and 64
#: rollouts per optimiser update, which is the step the reward probe timed at
#: 9-11 ms. In GRPO `per_device_train_batch_size` counts *completions*, not
#: prompts, so this is one prompt per forward pass.
#:
#: TRL requires the generation batch to hold whole groups:
#: (batch x devices x grad_accum) % num_generations == 0. Here 8 x 1 x 8 = 64,
#: which divides by 8. Getting this wrong raises at construction, so it is a
#: loud failure -- unlike almost everything else in this file.
NUM_GENERATIONS = 8
BATCH_SIZE = 8
GRAD_ACCUM = 8


def load_rows(path: Path, limit: int | None = None) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if limit:
        rows = rows[:limit]

    # 22 of the 1,035 rows have gold queries returning nothing. Against those,
    # any empty result compares equal, so they cannot teach correctness -- and
    # producing nothing is the easiest thing in SQL. The reward refuses to pay
    # for them anyway; removing them stops us spending rollouts on questions
    # with no learnable answer.
    kept = drop_empty_gold(rows)
    print(f"{path}: {len(rows)} rows, {len(kept)} after dropping empty gold "
          f"({len(rows) - len(kept)} removed)")
    return kept


def build_dataset(rows: list[dict]) -> Dataset:
    """The prompt, plus the two columns the reward needs.

    TRL hands every dataset column except ``prompt``/``completion`` through to
    the reward function, already expanded to one entry per generation. So
    ``db_path`` and ``gold_sql`` ride along and arrive aligned with the
    completions -- there is no separate lookup to get out of step.
    """
    return Dataset.from_list(
        [
            {
                # System + user only. The assistant turn in the file is the gold
                # answer, which is what the model must now produce for itself.
                "prompt": row["messages"][:-1],
                "db_path": row["db_path"],
                "gold_sql": row["gold_sql"],
            }
            for row in rows
        ]
    )


def assert_prompts_fit(dataset: Dataset, tokenizer, max_prompt_length: int) -> None:
    """Fail loudly if any prompt would be truncated. There is no warning otherwise.

    TRL truncates over-length prompts and says nothing -- no log line, no
    metric, unlike completion clipping which at least gets
    ``completions/clipped_ratio``. Worse, the *direction* is not TRL's to
    choose: it calls the tokenizer with ``truncation=True`` and no
    ``truncation_side``, so the tokenizer's own setting decides. Stock Qwen2.5
    defaults to ``"right"``, which on our format keeps the schema and deletes
    the question -- the model would be asked to write SQL for a question it
    cannot see, and the reward would punish it for failing.

    ``train`` sets ``truncation_side="left"`` to match what TRL documents, but
    left-truncation only trades which half is destroyed. This check is the
    actual protection: measured over the split, prompts reach 401 tokens
    against a 512 cap, so it should never fire -- and if the data changes it
    stops the run instead of quietly training on mutilated inputs.
    """
    lengths = [
        len(tokenizer(
            tokenizer.apply_chat_template(row["prompt"], tokenize=False,
                                          add_generation_prompt=True)
        )["input_ids"])
        for row in dataset
    ]
    longest = max(lengths)
    over = sum(length > max_prompt_length for length in lengths)
    if over:
        raise ValueError(
            f"{over} of {len(lengths)} prompts exceed max_prompt_length="
            f"{max_prompt_length} (longest {longest}). TRL would truncate them "
            f"silently. Raise max_prompt_length above {longest}."
        )
    print(f"prompt lengths: longest {longest} of {max_prompt_length} allowed "
          f"({longest / max_prompt_length:.0%} of budget), 0 truncated")


class RewardOutcomes(TrainerCallback):
    """Log the reward's outcome distribution alongside TRL's own metrics.

    The headline reward can rise for good reasons and bad ones. This is what
    separates them: ``match`` climbing while Spider test does not move means the
    reward is being gamed, and ``too_many_rows``/``wrong_rows`` climbing means
    the policy is collapsing onto the executes tier -- a flat 0.5 plateau that
    every degenerate query reaches, which would recreate v0's zero-variance
    stall through a different door.
    """

    def __init__(self, reward: SQLReward) -> None:
        self.reward = reward

    def on_log(self, args, state, control, logs=None, **kwargs) -> None:
        report = self.reward.report()
        if not report:
            return
        try:
            import wandb

            if wandb.run is not None:
                wandb.log(report, step=state.global_step)
                return
        except ImportError:
            pass
        # Logged either way: a run without W&B should still show the shape of
        # the reward, since that is the only thing that reveals hacking.
        print(f"  step {state.global_step} outcomes: "
              + "  ".join(f"{k.split('/')[-1]} {v:.0%}" for k, v in report.items() if v))


def train(
    *,
    steps: int = -1,
    epochs: float = 1.0,
    learning_rate: float = 1e-5,
    beta: float = 0.0,
    num_generations: int = NUM_GENERATIONS,
    batch_size: int = BATCH_SIZE,
    grad_accum: int = GRAD_ACCUM,
    limit: int | None = None,
    seed: int = 3407,
    report_to: str = "wandb",
    adapter: Path = SFT_CHECKPOINT,
    output: Path = OUTPUT,
) -> None:
    if not adapter.is_dir():
        raise FileNotFoundError(
            f"No SFT adapter at {adapter}. Phase 1.5's checkpoint is the starting "
            f"policy for this run; pull it from S3 before training."
        )

    # Before training, not after. A crashed run that leaves a stale adapter
    # behind is worse than one that leaves nothing: the evaluator scores it
    # happily and reports a real-looking number for a model that was never
    # trained. This has already happened once, with a 5-step smoke checkpoint.
    if output.exists():
        print(f"removing previous checkpoint at {output}")
        shutil.rmtree(output)

    rows = load_rows(GRPO_DATA, limit)
    dataset = build_dataset(rows)

    # Warmed before training: this executes every gold query once, so a missing
    # database or a drifted dataset fails here rather than at step 400, and the
    # first optimiser step does not pay for a thousand cold cache misses.
    reward = SQLReward().warm(rows)
    print(f"reward warm: {len(reward._gold)} distinct gold results cached")

    # The checkpoint's own tokenizer, not the base model's -- see sqlrl.tokenizer
    # for what happens when those disagree.
    tokenizer = build_tokenizer(str(adapter), chat=True)
    # TRL sets this itself *only* when it builds the tokenizer, which it does
    # not do when one is passed in. Left to the Qwen default of "right", an
    # over-length prompt keeps the schema and loses the question.
    tokenizer.truncation_side = "left"
    assert_prompts_fit(dataset, tokenizer, MAX_PROMPT_LENGTH)

    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16)
    # is_trainable=True keeps the SFT adapter as the thing being optimised.
    # Without it peft loads the adapter in inference mode and GRPO would train
    # nothing while reporting a perfectly normal-looking loss.
    model = PeftModel.from_pretrained(base, str(adapter), is_trainable=True)
    model.print_trainable_parameters()

    args = GRPOConfig(
        output_dir=str(CHECKPOINTS),
        num_train_epochs=epochs,
        max_steps=steps,
        learning_rate=learning_rate,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=5,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        num_generations=num_generations,
        max_prompt_length=MAX_PROMPT_LENGTH,
        max_completion_length=MAX_COMPLETION_LENGTH,
        # ------------------------------------------------------------------
        # The line this whole file turns on.
        #
        # TRL defaults to scale_rewards="group", which divides each group's
        # advantages by that group's own standard deviation. That rescales
        # every group to unit variance, and the consequence is not obvious:
        # the *ordering* of the reward tiers survives, but the *spacing between
        # them does not*. Measured on four eight-sample groups:
        #
        #   group of 8 rollouts             "group"   "none"
        #   one correct, rest merely ran      2.474    1.312
        #   one correct, rest no SQL          2.475    1.750
        #   one merely ran, rest db_error     2.473    0.262
        #   one parsed, rest no SQL           2.471    0.175
        #
        # Under the default, a group whose best sample was actually *correct*
        # pushes exactly as hard as one whose best sample merely *parsed* --
        # any two-outcome group normalises to the same advantage, whichever two
        # rungs it straddles. That discards the entire reason for having tiers.
        #
        # "none" keeps the raw spacing, a 7.5x spread between finding the answer
        # and producing something that parses. It is also Dr. GRPO's
        # recommendation, for the related reason that dividing by a group's own
        # variance over-weights questions that are trivially easy or hopeless.
        scale_rewards="none",
        # ------------------------------------------------------------------
        # beta=0 drops the KL penalty and the reference model with it. Also
        # TRL 0.24's own default, but set explicitly because the reasoning
        # matters here: with a PEFT model TRL does not keep a frozen copy of
        # the policy: it calls `model.disable_adapter()` to get reference
        # log-probs (grpo_trainer.py:425-433 and 1443-1467, read to confirm).
        # That makes the reference the *raw base model*, not our SFT
        # checkpoint. So a KL penalty here would not be "stay near where you
        # started" -- it would be "go back towards Qwen2.5-0.5B", pulling
        # against the 44.6% Phase 1.5 just bought.
        #
        # It also explains v0's unexplained KL ~= 1.2 from step 10 that never
        # moved, which the blueprint guessed was a 4-bit artefact. It was not:
        # the SFT policy had already moved that far from base before GRPO
        # began, so the number was measuring the SFT stage, not the RL.
        beta=beta,
        bf16=torch.cuda.is_available(),
        logging_steps=1,
        save_strategy="steps",
        save_steps=25,
        save_total_limit=3,
        seed=seed,
        report_to=report_to,
        optim="adamw_torch",
        log_completions=True,
        num_completions_to_print=2,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward],
        args=args,
        train_dataset=dataset,
        callbacks=[RewardOutcomes(reward)],
    )
    trainer.train()

    output.parent.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output))
    tokenizer.save_pretrained(str(output))
    print(f"saved to {output}")


def inspect(limit: int = 3) -> None:
    """Print what the trainer will actually see, before spending a GPU hour.

    Every silent failure in this project so far has been visible in the inputs
    if anyone had looked at them: pruned schemas, a double eos, a prompt/
    completion split that masked the wrong tokens.
    """
    rows = load_rows(GRPO_DATA, limit=limit)
    dataset = build_dataset(rows)
    reward = SQLReward().warm(rows)

    for row in dataset:
        print("=" * 72)
        for message in row["prompt"]:
            print(f"[{message['role']}] {message['content'][:400]}")
        print(f"[gold_sql] {row['gold_sql']}")
        print(f"[db_path]  {row['db_path']}")
        # What the reward pays for the right answer and for a plausible miss.
        for label, sql in (
            ("gold", row["gold_sql"]),
            ("wrong column", "SELECT no_such_column_zz FROM sqlite_master"),
            ("no sql", "I cannot answer that."),
        ):
            scored, outcome = reward.score(
                f"<answer>{sql}</answer>", row["db_path"], row["gold_sql"]
            )
            print(f"    reward({label:13s}) = {scored:.1f}  [{outcome}]")


def main() -> int:
    parser = argparse.ArgumentParser(description="GRPO on Spider train.")
    parser.add_argument("--inspect", action="store_true",
                        help="print what the trainer will see, then exit")
    parser.add_argument("--pilot", action="store_true",
                        help="30 steps, to check the diagnostics before a full run")
    parser.add_argument("--steps", type=int, default=-1)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--num-generations", type=int, default=NUM_GENERATIONS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--grad-accum", type=int, default=GRAD_ACCUM)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--report-to", default="wandb")
    parser.add_argument("--adapter", type=Path, default=SFT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    if args.inspect:
        inspect()
        return 0

    train(
        steps=30 if args.pilot else args.steps,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        beta=args.beta,
        num_generations=args.num_generations,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        limit=args.limit,
        seed=args.seed,
        report_to=args.report_to,
        adapter=args.adapter,
        output=args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
