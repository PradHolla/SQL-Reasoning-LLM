# SQL Reasoning LLM

> **A note on AI use:** this is a learning project, and I build it with AI models — Claude Sonnet and Claude Opus — as collaborators.

Post-training a 0.5B model (`Qwen2.5-0.5B`) to turn English questions into SQL that actually runs and returns the right rows.

The interesting part of this repo isn't the pipeline. It's that I built the pipeline first, believed it worked, and then built an evaluation that told me it didn't. Most of what follows is me finding out what was wrong and fixing it, in order. I've kept the wrong version around at every step, because a number is only meaningful next to the number it replaced.

## Where it stands

Spider **test** split, 2,147 questions the model has never seen, greedy decoding, pass@1. Execution accuracy = run the predicted SQL and the gold SQL against the real database and compare result sets.

| | EX | executes | parses |
|---|---|---|---|
| the original pipeline (CPT→SFT→GRPO) | 6.4% | 35.2% | 78.8% |
| untrained base model, best prompt | 17.4% | 51.0% | 74.8% |
| SFT rebuilt on Spider | 44.6% | 65.9% | 99.6% |
| **+ GRPO with an execution reward** | **49.7%** | **76.5%** | 99.3% |
| Qwen2.5-Coder-7B-Instruct (14× bigger) | 71.2% | 89.3% | 95.8% |

So: **6.4% → 49.7%**. Same base model, same size, same benchmark.

The 7B row is there on purpose. A model 14× the size still beats this comfortably, and pretending otherwise would be silly.

## How it went wrong, and what fixed it

### The original numbers were not numbers

v0 reported success from W&B training curves — reward climbing to 0.8–1.0, format adherence at 0.95+. All of it measured on the training distribution, with a reward function that was exact string matching. `age>56` scored zero against `age > 56`. Identical queries. Identical results. Zero.

There was no held-out evaluation at all. Not a bad one — none.

### So I built the measuring stick first

`src/sqlrl/eval/executor.py` runs both queries against the actual SQLite file and compares result sets. Everything else in the project depends on it, including the RL reward, which is the whole point: **in RL your evaluation and your reward are the same skill.**

It tracks four numbers, not one, because the gaps between them are the diagnosis. A model that parses 99% and executes 65% has a schema problem. A model that parses 22% has a formatting problem. One accuracy number can't tell you which.

Then the eval immediately started ruining my week.

### The benchmark was half-leaked

**562 of Spider dev's 1,034 questions appear verbatim in the training set**, gold SQL and all. `b-mc2/sql-create-context` is partly built from Spider. Any dev score was partly a memorisation score.

Moved to Spider test (0.7% overlap). Worth noting this affects every project trained on that dataset and benchmarked on Spider dev, which is a lot of them.

### The trained model was worse than the untrained one

6.4% for the full pipeline, 17.4% for the base model I started from. Four months of training had been actively destroying value.

### One of the three training stages was harmful

Continued pre-training was measured as neutral-to-harmful — but CPT was also the only stage evaluated with a different prompt format, so weights and prompt were confounded. Ran the 2×2 to separate them. CPT costs you 17.4% → 3.1% holding the prompt fixed. It wasn't neutral. It was destructive, and it's now deleted from the pipeline.

That 2×2 also showed prompt format was a hidden variable in *every* comparison I'd made until then.

### The training data taught the wrong task

`sql-create-context` gives you schemas **pruned to exactly the columns the answer needs**. Train on that and the model learns it doesn't have to read the schema — which is the actual job. Spider hands over the whole database and expects you to figure out which two of forty columns matter.

Rebuilt on Spider train with real databases, plus: loss on the completion only (with full schemas the prompt is ~75% of the tokens, so most of the gradient was teaching the model to *generate database schemas*), two real epochs instead of 0.10, and a learning rate that suits a from-scratch LoRA.

6.4% → 44.6%.

A thing I did not expect: **validation loss picked the wrong checkpoint.** Epoch 2 looked neutral-to-slightly-worse on eval loss and token accuracy, and was +3.9 points of actual execution accuracy. Loss is a proxy. Run the real metric.

### The RL reward was a constant, not a reward

The old GRPO had two rewards: format (+1.0) and exact string match (+2.0). The format reward fired for *every* sample. The string match fired for almost none. So a group of 4 samples scored `[1.0, 1.0, 1.0, 1.0]`.

GRPO computes each sample's advantage as `(reward − group mean) / group std`. Identical rewards → advantage exactly zero → **no gradient**. The metric for this is `frac_reward_zero_std`, it sat between 0.5 and 1.0, and it hit 1.00 at step 160. A large fraction of those 300 training steps did nothing at all.

The fix is a ladder with partial credit:

| | reward |
|---|---|
| no SQL, or it won't parse | 0.0 |
| parses | 0.2 |
| the database ran it | 0.5 |
| **result set matches gold** | **2.0** |

Now a group where nothing is correct still scores `[0.2, 0.5, 0.5, 0.2]` — real variance, real gradient. `frac_reward_zero_std` dropped to 0.34.

44.6% → **49.7%**, and I checked it properly: McNemar on the paired predictions, χ² = 29.7, p < 0.001. GRPO fixed 251 questions and broke 142. The +5.1 is a net, not a clean sweep.

The mechanism matters more than the score. Going in, 88% of remaining failures were the model inventing columns and tables that don't exist. The "+0.5 for executing" tier existed specifically to pay for fixing that:

```
unknown_column   647 → 458   (−29%)
unknown_table     66 →   7   (−89%)
```

Execution rate went 65.9% → 76.5% while parse rate stayed flat. That's a model that already knew how to write SQL learning to write it against *the schema in front of it*.

## Three bugs that were invisible and expensive

Keeping these because they're the actual lesson, and all three failed silently — no crash, no warning, just a plausible wrong number.

**Unsloth rewrote the vocabulary.** `get_chat_template` swapped the ids of `<|im_end|>` and `<|endoftext|>` during training. Pair those checkpoints with a stock tokenizer and generation never stops, every completion runs to the token limit, and your eval score reads near zero for reasons that have nothing to do with the model. Unsloth is gone now.

**You can't teach a base model a token it was never taught.** My SFT trained the model to end its turn with `<|im_end|>`. The data was right, the loss mask covered that token, two epochs ran over it — and the trained model emits `<|im_end|>` essentially never. GRPO then spent a whole pilot run generating rollouts that never terminated.

The reason is in the base model. Qwen2.5-0.5B **base** carries the ChatML special tokens in its vocabulary but never trained them — only the Instruct variants use ChatML. Their embeddings are still sitting near random initialisation:

```
<|endoftext|>   row norm 0.5987   98.40th percentile
<|im_end|>      row norm 0.3010    1.52th percentile
```

LoRA targets attention and MLP, not the embedding — and Qwen ties the embedding to the output head. So to emit `<|im_end|>` the model would have had to steer its hidden state onto a near-random 896-dimensional vector while competing against a well-trained `<|endoftext|>`. The gradient had nowhere to go. It learned *where* to stop and picked the token it could actually reach.

The wider lesson is the checking, though. I already had an assertion for stop-token mismatches, and **it passes on the broken checkpoint** — it compares the tokenizer's eos to the saved tokenizer's eos, both say `<|im_end|>`, and the model agrees with neither. Two configs agreeing tells you nothing about what the weights do. The only real check is to generate and look, which is what `assert_model_stops` does now, in both trainers, before anything gets saved.

**TRL's default quietly cancelled the reward design.** `scale_rewards="group"` divides each group's advantages by that group's own standard deviation. The *ordering* of the reward tiers survives that; the *spacing* does not. A group where the best sample was genuinely correct ends up pushing exactly as hard as one where the best sample merely parsed — any two-outcome group normalises to the same advantage. I'd have shipped a carefully-designed ladder that behaved like a coin flip. `scale_rewards="none"` fixes it.

I found that last one by writing a script that attacks my own reward with policies that never read the question — `SELECT 1`, `SELECT *`, `WHERE 1=0`, degenerate cross joins — and checking what they can extract. Best one got 27% of what real answers get. It's in `src/sqlrl/training/reward_probe.py` and I'd recommend the habit to anyone doing RL: probe the reward *before* you spend the GPU hours, not after the curve looks weird.

## Running it

```bash
uv sync
```

Needs Linux + an NVIDIA GPU for training. Evaluation runs on a Mac (MPS) or CPU, slowly.

**Data.** Downloads Spider, checks it against the benchmark for contamination, executes every gold query before keeping it, and splits by *database* so no database appears in two splits:

```bash
uv run python -m sqlrl.data_prep.build_spider_datasets
```

**Train.** SFT first, then GRPO from that checkpoint:

```bash
uv run python -m sqlrl.training.sft_spider --inspect   # look at what it'll train on first
uv run python -m sqlrl.training.sft_spider

uv run python -m sqlrl.training.grpo_spider --inspect
uv run python -m sqlrl.training.grpo_spider --pilot    # 30 steps, check the diagnostics
uv run python -m sqlrl.training.grpo_spider
```

Both have `--inspect`, and I'd use it every time. Every silent failure in this project was visible in the inputs if anyone had bothered to look at them.

**Evaluate:**

```bash
uv run python -m sqlrl.eval.run_eval --model all --split test
uv run python -m sqlrl.eval.run_eval --model all --split test --score-only   # rescore saved predictions, no GPU
```

**Poke at the reward:**

```bash
uv run python -m sqlrl.training.reward_probe
```

**Tests:** 193 of them, `uv run pytest`. Heavily weighted toward the places where being wrong produces a believable number instead of a crash.

## Layout

```text
src/sqlrl/
  eval/
    executor.py     run SQL, compare result sets   <- everything depends on this
    metrics.py      the four numbers and a failure taxonomy
    spider.py       download, contamination check, gold verification
    prompts.py      prompt formats, and pulling SQL back out of model output
    run_eval.py     the harness
  training/
    sft_spider.py   SFT on Spider train
    rewards.py      the execution-grounded GRPO reward
    reward_probe.py attacks that reward with lazy policies
    grpo_spider.py  the GRPO run
  tokenizer.py      one place that decides how text becomes tokens
infra/              EC2 scripts: launch, start, stop, run jobs, idle-shutdown
```

The v0 trainers (`cpt_trainer.py`, `sft_trainer.py`, `grpo_trainer.py`) are still here, unmodified. They're the baseline every number above is measured against, so they stay.

## What's next

Real reasoning traces. The `<think>` block in the training data is currently **one hardcoded sentence, repeated on every single example** — the model learned to recite a preamble, not to think. Rejection sampling is the plan: generate a lot of traces, keep only the ones whose SQL executes correctly, retrain on those.

Then scale, then retrieval and an agentic loop that can look at the schema and its own errors.

Longer term, closing the gap to that 7B row without becoming a 7B model.

## Costs

The whole thing runs on one `g5.xlarge` (A10G 24GB) at about $1/hr on-demand, plus an idle-shutdown cron job that has saved me more money than every other optimisation combined. GRPO training plus a full evaluation is about two hours.
