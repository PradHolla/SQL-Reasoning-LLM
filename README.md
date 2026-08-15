# SQL Reasoning LLM

> **A note on AI use:** this is a learning project, and I built it with AI models like Claude Opus, Sonnet, with also a little bit of Google Gemini too, as collaborators.

Teaching a very small language model (0.5B parameters) to answer questions about a database by writing SQL that actually runs and returns the right rows.

## What the model actually does

You hand it a database schema and a question in plain English:

```
Schema:   CREATE TABLE head (head_ID INT, name TEXT, born_state TEXT, age REAL);
          CREATE TABLE department (Department_ID INT, Name TEXT, Budget_in_Billions REAL);
          CREATE TABLE management (department_ID INT, head_ID INT, temporary_acting TEXT)

Question: How many heads of the departments are older than 56?
```

and it writes the query:

```sql
SELECT count(*) FROM head WHERE age > 56
```

Sounds easy. It isn't, because the schema is usually much bigger than that, and the hard part isn't SQL syntax — it's **schema linking**: working out that "heads of departments" means the `head` table and not the `management` table, and that "older than" refers to the `age` column and not one of the thirty-nine others. Get the syntax right and the column wrong and you have a query that runs beautifully and returns the wrong answer.

Most of this project turns out to be about that one problem.

## Terms, so the rest of this makes sense

**Spider** — the benchmark. ~10,000 human-written questions across 200 real SQLite databases, split so that **the databases in the test set never appear in training**. That last part matters: you can't pass it by memorising one company's schema, you have to read a schema you've never seen. It's the standard academic benchmark for this task.

**Execution accuracy (EX)** — the score that counts. Run the model's SQL *and* the correct SQL against the real database, compare the rows that come back. Same rows, same answer, point scored. Notably this doesn't care whether the query *looks* like the reference one — two very different queries that return the same rows are both right, which is exactly how a human would judge it.

The other three numbers in the tables below are there for diagnosis, not scoring:

- **executes** — the database accepted the query and ran it
- **parses** — it's valid SQL syntax at all
- **struct** — the query is *shaped* like the reference one (same tables, joins, aggregates)

The gaps between them are where the diagnosis lives. A model that parses 99% but only executes 65% is inventing columns. A model that parses 22% is producing garbage that isn't SQL. One accuracy number can't tell you which, which is why this project tracks four.

**The training stages**, in the order they run:

- **SFT** (supervised fine-tuning) — show the model thousands of question → correct-SQL pairs and have it imitate them. This is the bulk of the learning.
- **GRPO** (a reinforcement learning algorithm) — let the model write 8 different answers to the same question, score each one, and push it toward whatever scored well. Useful because it learns from its *own* attempts rather than from copied answers.
- **LoRA** — instead of updating all 500 million weights, freeze them and train ~9 million extra ones bolted onto the side. Vastly cheaper, and it's why this whole thing fits on one rented GPU.

**The base model** is `Qwen2.5-0.5B` — a small, general-purpose model that has never been taught to follow instructions or write SQL specifically. Everything below is what happens after.

## Where it stands

Spider **test** split: 2,147 questions over databases the model has never seen. Greedy decoding, one attempt per question.

| | EX | executes | parses | struct |
|---|---|---|---|---|
| the original pipeline (CPT→SFT→GRPO) | 6.4% | 35.2% | 78.8% | 2.7% |
| untrained base model, best prompt | 17.4% | 51.0% | 74.8% | 10.2% |
| SFT rebuilt on Spider | 44.6% | 65.9% | 99.6% | 34.0% |
| **+ GRPO with an execution reward** | **49.7%** | **76.5%** | 99.3% | 38.8% |
| + reasoning traces instead (see Phase 3) | 47.2% | 72.8% | 99.8% | 31.5% |
| Qwen2.5-Coder-7B-Instruct (14× bigger) | 71.2% | 89.3% | 95.8% | 35.9% |

**6.4% → 49.7%.** Same base model, same size, same benchmark.

The 7B row is there on purpose. A model 14× the size still beats this comfortably, and pretending otherwise would be silly.

## The shape of this repo

The interesting part isn't the pipeline. It's that I built the pipeline first, believed it worked, and then built an evaluation that told me it didn't.

Everything below is me finding out what was wrong, in order, and fixing it. I've kept the wrong version at every step, because a number only means something next to the number it replaced. The rest of this README is that story.

---

## How it went wrong, and what fixed it

### The original numbers were not numbers

v0 reported success from training curves — reward climbing to 0.8–1.0, format adherence at 0.95+. All of it measured on the training data, with a reward function that was exact string matching. `age>56` scored zero against `age > 56`. Identical queries. Identical results. Zero.

There was no held-out evaluation at all. Not a bad one — none.

### So I built the measuring stick first

`src/sqlrl/eval/executor.py` runs both queries against the actual SQLite file and compares result sets. Everything else depends on it, including the RL reward, which is the whole point: **in reinforcement learning, your evaluation and your reward are the same skill.** Build one, get the other.

Then the eval immediately started ruining my week.

### The benchmark was half-leaked

**562 of Spider dev's 1,034 questions appear verbatim in the training set**, gold SQL and all — `b-mc2/sql-create-context` is partly built from Spider. Any score on dev was partly a memorisation score.

Moved to Spider test (0.7% overlap). Worth noting this affects every project trained on that dataset and benchmarked on Spider dev, which is a lot of them.

### The trained model was worse than the untrained one

6.4% for the full pipeline, 17.4% for the base model I started from. Four months of training had been actively destroying value.

### One of the three training stages was harmful

Continued pre-training measured as neutral-to-harmful — but CPT was also the only stage evaluated with a different prompt format, so weights and prompt were tangled together. Ran a 2×2 to separate them. Holding the prompt fixed, CPT costs you 17.4% → 3.1%. Not neutral. Destructive, and now deleted.

That 2×2 also revealed prompt format had been a hidden variable in *every* comparison I'd made until then.

### The training data taught the wrong task

`sql-create-context` gives you schemas **pruned to exactly the columns the answer needs**. Train on that and the model learns it doesn't have to read the schema — which is the actual job. Spider hands over the whole database and expects you to work out which two of forty columns matter.

Rebuilt on Spider train with real databases, plus: loss computed on the answer only (with full schemas the prompt is ~75% of the tokens, so most of the gradient was teaching the model to *generate database schemas*), two real epochs instead of 0.10, and a learning rate that suits a from-scratch LoRA.

**6.4% → 44.6%.**

Something I didn't expect: **validation loss picked the wrong checkpoint.** Epoch 2 looked neutral-to-worse on loss and token accuracy, and was +3.9 points of actual execution accuracy. Loss is a proxy. Run the real metric.

### The RL reward was a constant, not a reward

v0's GRPO had two rewards: format (+1.0) and exact string match (+2.0). The format reward fired for *every* sample. The string match fired for almost none. So a group of 4 answers scored `[1.0, 1.0, 1.0, 1.0]`.

GRPO scores each answer relative to its group: `(reward − group average) / group spread`. Identical rewards → every answer is exactly average → **zero gradient, nothing learned.** The metric for this is `frac_reward_zero_std`; it sat between 0.5 and 1.0 and hit 1.00 at step 160. A large fraction of those 300 training steps did nothing at all.

The fix is a ladder with partial credit:

| | reward |
|---|---|
| no SQL, or it won't parse | 0.0 |
| parses | 0.2 |
| the database ran it | 0.5 |
| **result set matches gold** | **2.0** |

Now a group where nothing is fully correct still scores `[0.2, 0.5, 0.5, 0.2]` — real disagreement, real gradient. `frac_reward_zero_std` dropped to 0.34.

**44.6% → 49.7%**, and I checked it properly: McNemar's test on the paired predictions, χ² = 29.7, p < 0.001. GRPO fixed 251 questions and broke 142 — the +5.1 is a net, not a clean sweep.

The mechanism matters more than the score. Going in, 88% of remaining failures were the model inventing columns and tables that don't exist. The "+0.5 for executing" tier existed to pay for fixing exactly that:

```
unknown_column   647 → 458   (−29%)
unknown_table     66 →   7   (−89%)
```

Execution rate 65.9% → 76.5% while parse rate stayed flat. That's a model that already knew how to write SQL learning to write it against *the schema in front of it*.

---

## Phase 3: teaching it to think, and what that cost

The training data told the model to "think" before answering, but the thinking was **one hardcoded sentence repeated on all 5,378 examples**:

> *"I need to analyze the schema to find the correct tables and columns, then construct a valid SQL query."*

As scratch paper that's blank — it carries no information about the specific question. The model learned to type it and then solve the problem in one shot anyway.

**The fix is rejection sampling** (also called STaR): take a bigger model that *can* reason, have it solve your training questions 8 times each while showing its work, then **execute every attempt and keep only the traces whose SQL is actually correct.** The filter is the whole technique — a big model is wrong confidently and fluently, and training on that is worse than not training.

The verifier was already there. `executor.py` has now had three jobs: benchmark metric, RL reward, and data filter.

Result: 4,823 verified traces, 90.8% coverage, median 114 words of reasoning that names real tables and columns and explains which joins aren't needed.

**And the first attempt made the model worse: 44.6% → 41.8%.**

The cause was measurable. With loss computed only on the answer:

```
                completion tokens   of which SQL   share of gradient on the SQL
canned sentence            62.6           33.6            53.7%
real traces               195.1           31.1            15.9%
```

The SQL barely changed size; 114 words of prose grew around it. Same epochs, same learning rate — **the model was spending 3.4× less of its training on the part that gets scored.** Doubling the epochs fixed it: **41.8% → 47.1%**, and +2.5 over the canned baseline.

**But the gains don't stack.** GRPO on the trace-trained model was a perfect wash — +0.1 points, 179 questions fixed and 177 broken:

```
                  sft-spider  grpo-spider  sft-traces-4ep  grpo-traces
unknown_column           647          458             566          541
```

Both interventions fix schema hallucination, and **RL is much better at it** — it cut `unknown_column` by 189 where traces cut it by 81. Traces did a partial job of the thing RL does well and left RL nothing to work with. The best model is still SFT + GRPO at 49.7%.

RL wasn't idle, though: execution rate +3.7, structure +4.3, accuracy flat. It made the SQL *run* without making it *right* — climbing the 0.5 rung of the reward ladder instead of the 2.0 one. Which is precisely the failure the reward probe predicted in writing before any training happened.

---

## Three bugs that were invisible and expensive

All three failed silently. No crash, no warning, just a plausible wrong number.

**Unsloth rewrote the vocabulary.** Its `get_chat_template` swapped the ids of two special tokens during training. Pair those checkpoints with a normal tokenizer and generation never stops, every answer runs to the token limit, and your eval score reads near zero for reasons that have nothing to do with the model. Unsloth is gone now.

**You can't teach a base model a token it was never taught.** My SFT trained the model to end its turn with `<|im_end|>`. The data was right, the loss covered that token, two epochs ran over it — and the trained model emits it essentially never. GRPO then spent an entire pilot run generating answers that never terminated.

The reason is in the base model. Qwen2.5-0.5B **base** carries the chat special tokens in its vocabulary but never trained them — only the instruction-tuned variants use them. Their embeddings are still near random initialisation:

```
<|endoftext|>   row norm 0.5987   98.40th percentile
<|im_end|>      row norm 0.3010    1.52th percentile
```

LoRA trains attention and MLP, not the embedding — and Qwen ties the embedding to the output layer. So to emit `<|im_end|>` the model would have had to aim its hidden state at a near-random 896-dimensional vector while competing with a well-trained `<|endoftext|>`. The gradient had nowhere to go. It learned *where* to stop and picked the token it could actually reach.

The wider lesson is the checking. I already had an assertion for stop-token mismatches, and **it passes on the broken checkpoint** — it compares the tokenizer's setting to the saved tokenizer's setting, both agree, and the model agrees with neither. Two configs agreeing tells you nothing about what the weights do. The only real check is to generate and look, which is what `assert_model_stops` does now, in both trainers, before anything gets saved.

**TRL's default quietly cancelled the reward design.** `scale_rewards="group"` divides each group's scores by that group's own spread. The *ordering* of the reward ladder survives; the *spacing* does not. A group where the best answer was genuinely correct ends up pushing exactly as hard as one where the best answer merely parsed — any two-outcome group normalises to the same thing. I'd have shipped a carefully designed ladder that behaved like a coin flip. `scale_rewards="none"` fixes it.

I found that one by writing a script that attacks my own reward with policies that never read the question — `SELECT 1`, `SELECT *`, `WHERE 1=0`, degenerate cross joins — and measuring what they can extract. Best one got 27% of what real answers get. It's `src/sqlrl/training/reward_probe.py`, and I'd recommend the habit to anyone doing RL: probe the reward *before* you spend the GPU hours, not after the curve looks strange.

---

## Running it

```bash
uv sync
```

Training needs Linux and an NVIDIA GPU. Evaluation runs on a Mac (MPS) or CPU, slowly.

**Data.** Downloads Spider, checks it against the benchmark for contamination, executes every reference query before keeping it, and splits by *database* so no database appears in two splits:

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

Both have `--inspect`, and I'd use it every time. Every silent failure in this project was visible in the inputs, if anyone had bothered to look at them.

**Evaluate:**

```bash
uv run python -m sqlrl.eval.run_eval --model all --split test
uv run python -m sqlrl.eval.run_eval --model all --split test --score-only   # rescore, no GPU
```

**Poke at the reward:**

```bash
uv run python -m sqlrl.training.reward_probe
```

**Reasoning traces** (Phase 3) need vLLM, which conflicts with this project's pinned versions, so it gets its own throwaway environment — see `requirements-teacher.txt`.

**Tests:** 217 of them, `uv run pytest`. Heavily weighted toward the places where being wrong produces a believable number instead of a crash.

## Layout

```text
src/sqlrl/
  eval/
    executor.py       run SQL, compare result sets   <- everything depends on this
    metrics.py        the four numbers and a failure taxonomy
    spider.py         download, contamination check, gold verification
    prompts.py        prompt formats, and pulling SQL back out of model output
    run_eval.py       the harness
  training/
    sft_spider.py     supervised fine-tuning on Spider
    rewards.py        the execution-grounded GRPO reward
    reward_probe.py   attacks that reward with lazy policies
    grpo_spider.py    the RL run
  data_prep/
    sample_traces.py  teacher sampling + execution filtering (Phase 3)
    build_trace_sft.py
  tokenizer.py        one place that decides how text becomes tokens
infra/                EC2 scripts: launch, start, stop, run jobs, idle-shutdown
```

The v0 trainers (`cpt_trainer.py`, `sft_trainer.py`, `grpo_trainer.py`) are still here, unmodified. They're the baseline every number above is measured against, so they stay.

## What's next

A bigger base model. Everything so far has been squeezing a 0.5B model, and that 7B row suggests the remaining gap is mostly capacity rather than technique. The same pipeline on 1.5B or 7B is the obvious next lever — and reasoning traces, which barely paid off here, are much likelier to help a model with the capacity to use them.

After that: schema retrieval for databases too large to fit in the prompt, and an agentic loop that can run its own query, look at the error, and try again.

## Costs

The whole thing runs on one `g5.xlarge` (A10G 24GB) at about $1/hr on-demand, plus an idle-shutdown cron job that has saved me more money than every other optimisation combined. A GRPO run plus a full evaluation is about two hours.
