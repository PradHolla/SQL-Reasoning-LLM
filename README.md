# SQL Reasoning LLM

> **A note on AI use:** this is a learning project, and I built it with AI models like Claude Opus, Sonnet, with also a little bit of Google Gemini too, as collaborators.

Post-training small language models (0.5B and 1.5B) to answer questions about a database by writing SQL that actually runs and returns the right rows.

**6.4% → 68.1%** execution accuracy on the Spider benchmark, ending 3.1 points behind a model five times larger.

## What the model does

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

Sounds easy. It isn't, because real schemas are much bigger and the hard part isn't SQL syntax — it's **schema linking**: working out that "heads of departments" means the `head` table and not `management`, and that "older than" is the `age` column and not one of the other thirty-nine. Get the syntax right and the column wrong and you have a query that runs beautifully and returns the wrong answer.

Almost everything below turns out to be about that one problem.

## The five things worth knowing

1. **The original pipeline scored worse than the untrained model it started from** — 6.4% against 17.4% — and nobody could have known, because there was no held-out evaluation at all.
2. **The benchmark was half-leaked.** 562 of Spider dev's 1,034 questions appear verbatim in the training set, gold SQL included.
3. **One of the three training stages was actively destructive**, costing 17.4% → 3.1% once a 2×2 separated it from a prompt-format confound.
4. **Technique is worth +20.9 points independent of model size**, measured by running the old recipe and the new one on the same larger base. Both technique and scale matter, and they saturate against each other.
5. **Reinforcement learning stopped helping as the base model got better** — +5.1 points at 0.5B, +0.2 and not significant at 1.5B — while every diagnostic metric still improved. The reward paid for something that had stopped being the bottleneck.

Every number here is on held-out data, and differences are checked with a paired significance test rather than eyeballed.

---

## Where it stands

Spider **test** split: 2,147 questions over databases the model has never seen. Greedy decoding, one attempt per question. **EX** (execution accuracy) means the predicted SQL and the reference SQL return the same rows from the real database.

| | EX | executes | parses | struct |
|---|---|---|---|---|
| the original pipeline (CPT→SFT→GRPO) | 6.4% | 35.2% | 78.8% | 2.7% |
| untrained base model, best prompt | 17.4% | 51.0% | 74.8% | 10.2% |
| **0.5B** — SFT rebuilt on Spider | 44.6% | 65.9% | 99.6% | 34.0% |
| **0.5B** — + GRPO with an execution reward | 49.7% | 76.5% | 99.3% | 38.8% |
| **1.5B** — same recipe, code-pretrained base | 67.9% | 88.2% | 99.8% | 46.3% |
| **1.5B** — + GRPO | **68.1%** | **90.6%** | **99.9%** | **47.3%** |
| Qwen2.5-Coder-7B-Instruct (5× bigger) | 71.2% | 89.3% | 95.8% | 35.9% |

The 7B row is there on purpose. It still wins, and pretending otherwise would be silly — though the 1.5B model parses better (99.9% vs 95.8%) and matches query structure better (47.3% vs 35.9%).

The other three columns exist for diagnosis, not scoring. The **gaps between them** are where the diagnosis lives: a model that parses 99% but executes 65% is inventing columns; one that parses 22% is producing text that isn't SQL. A single accuracy number can't tell you which, which is why this tracks four.

## Did you just need a bigger model?

The obvious challenge to any result like this, and worth answering with an experiment rather than an opinion. So: run the *old* recipe and the *new* recipe on the same larger base.

```
                    0.5B      1.5B      scale effect
old recipe          4.6%     46.9%        +42.4
new recipe         44.6%     67.9%        +23.2
technique effect   +40.1     +20.9

technique at 1.5B:  +20.9 pts | 559 fixed, 110 broken | χ² = 300, p < 0.001
scale, new recipe:  +23.2 pts | 577 fixed,  78 broken | χ² = 379, p < 0.001
```

**Both matter, and they saturate.** Technique is worth +20.9 points even on a bigger, code-pretrained base, so the pipeline isn't merely compensating for a weak model. But that's half what it was worth at 0.5B, and the two effects don't add: 4.6% → 67.9% is +63.3 total, against +82 if you could bank them independently.

One caveat that travels with this table: the bottom-left cell is the old recipe *with* a tokenizer bug that stopped generation terminating, while the 1.5B version used the working stop token. So the +42.4 is partly "a bug was also fixed". The two clean numbers are the ones underneath, each holding everything else fixed.

The ablation was built deliberately generous to the old recipe — the destructive training stage omitted, the vocabulary-corrupting library omitted, the working stop token used. It's the strongest fair version of the thing it loses to.

---

## How it went wrong, and what fixed it

The interesting part of this repo isn't the pipeline. It's that I built the pipeline first, believed it worked, then built an evaluation that told me it didn't. I've kept the wrong version at every step, because a number only means something next to the number it replaced.

### The original numbers weren't numbers

The first version reported success from training curves — reward climbing to 0.8–1.0, format adherence 0.95+. All of it measured on the training data, with a reward that was exact string matching. `age>56` scored zero against `age > 56`. Identical queries, identical results, zero.

There was no held-out evaluation. Not a bad one — none.

So the first real work was `src/sqlrl/eval/executor.py`: run both queries against the actual SQLite file, compare result sets. Everything depends on it, including the RL reward, which is the point — **in reinforcement learning your evaluation and your reward are the same skill.** Build one, get the other. It later became the filter for synthetic data too: one component, three jobs.

Then it started ruining my week.

### The benchmark was half-leaked

**562 of Spider dev's 1,034 questions appear verbatim in the training set**, gold SQL and all — `b-mc2/sql-create-context` is partly built from Spider. Any dev score was partly a memorisation score. Moved to Spider test (0.7% overlap).

This affects every project trained on that dataset and evaluated on Spider dev, which is a lot of them.

### A whole training stage was making things worse

Continued pre-training measured as neutral-to-harmful — but it was also the only stage evaluated with a different prompt format, so weights and prompt were tangled. A 2×2 separated them: holding the prompt fixed, that stage costs 17.4% → 3.1%. Deleted.

The same 2×2 revealed prompt format had been a hidden variable in *every* comparison I'd made until then.

### The training data taught the wrong task

`sql-create-context` gives you schemas **pruned to exactly the columns the answer needs**. Train on that and the model learns it doesn't have to read the schema — which is the actual job.

Rebuilt on Spider with real databases, plus loss computed on the answer only (with full schemas the prompt is ~75% of the tokens, so most of the gradient was teaching the model to *generate database schemas*), two real epochs instead of 0.10, and a learning rate suited to a from-scratch adapter. **6.4% → 44.6%.**

Unexpected: **validation loss picked the wrong checkpoint.** Epoch 2 looked slightly worse on loss and token accuracy and was +3.9 points of real accuracy. Loss is a proxy. Run the real metric.

### The RL reward was a constant, not a reward

The old setup had two rewards: format (+1.0) and exact string match (+2.0). Format fired for *every* sample; string match for almost none. So a group of 4 answers scored `[1.0, 1.0, 1.0, 1.0]`.

GRPO scores each answer relative to its group: `(reward − group average) / group spread`. Identical rewards mean every answer is exactly average — **zero gradient, nothing learned.** The metric for this, `frac_reward_zero_std`, sat between 0.5 and 1.0 and hit 1.00 at step 160. A large share of those 300 steps did nothing at all.

The fix is a ladder with partial credit:

| | reward |
|---|---|
| no SQL, or it won't parse | 0.0 |
| parses | 0.2 |
| the database ran it | 0.5 |
| **result matches the reference** | **2.0** |

Now a group where nothing is fully correct still scores `[0.2, 0.5, 0.5, 0.2]` — real disagreement, real gradient. **44.6% → 49.7%**, McNemar χ² = 29.7, p < 0.001. It fixed 251 questions and broke 142; the +5.1 is a net, not a clean sweep.

The mechanism matters more than the score. Going in, 88% of remaining failures were invented columns and tables. The "+0.5 for executing" tier existed to pay for exactly that:

```
unknown_column   647 → 458   (−29%)
unknown_table     66 →   7   (−89%)
```

### Teaching it to reason didn't work, and the reason was measurable

The training data told the model to think before answering, but the thinking was **one hardcoded sentence repeated on all 5,378 examples**. As scratch paper it's blank — no information about the specific question.

The fix is **rejection sampling** (STaR): take a bigger model that can reason, have it solve your training questions 8 times each while showing its work, then **execute every attempt and keep only the traces whose SQL is correct**. The filter is the whole technique — a big model is wrong confidently and fluently, and training on that is worse than not training. Result: 4,823 verified traces, 90.8% coverage, median 114 words of reasoning naming real tables and columns.

**And it made the model worse: 44.6% → 41.8%.**

The cause was measurable:

```
                completion tokens   of which SQL   share of gradient on the SQL
canned sentence            62.6           33.6            53.7%
real traces               195.1           31.1            15.9%
```

The SQL barely changed size; 114 words of prose grew around it. Same epochs, same learning rate, so the model was spending **3.4× less of its training on the part that gets scored**. Doubling the epochs fixed it: **41.8% → 47.1%**.

But the gains didn't stack — RL on top of the trace model was a perfect wash, +0.1 points, 179 fixed and 177 broken. Both interventions fix schema hallucination and RL is better at it, so the traces did a partial job of RL's work and left it nothing to do.

### RL stopped helping once the base model was good

Same reward, same config, on the 1.5B base:

```
RL on 0.5B base:  +5.1 pts | 251 fixed, 142 broken | p < 0.001
RL on 1.5B base:  +0.2 pts | 122 fixed, 117 broken | not significant
```

The mechanism still worked — hallucinated columns down 23%, execution rate 88.2% → 90.6%, syntax errors 5 → 2. **Everything improved except the score.**

The reward pays for executability, and executability had stopped being the bottleneck. At 0.5B, invented columns dominated the failures, so fixing them moved the score. At 1.5B the model already runs 88% of its queries and what remains are queries that execute perfectly and answer the wrong question — which this reward cannot see.

Two diagnostics from that run are worth carrying anywhere else:

**`frac_reward_zero_std` averaged 0.562** — 56% of prompt groups produced no gradient at all. That's the original failure number arriving for the opposite reason: those rollouts were uniformly *wrong*, these are uniformly *right*. **RL gets harder as the policy gets better**, because it runs on within-group disagreement.

**Training reward rose 24% while held-out accuracy moved 0.2 points**, with entropy collapsing monotonically. The same shape as the dashboards that started this whole project, caught this time only because the held-out number exists.

---

## Three bugs that were invisible and expensive

All three failed silently. No crash, no warning, just a plausible wrong number. These are the most portable thing in the repo.

**A library rewrote the vocabulary.** Unsloth's `get_chat_template` swapped the ids of two special tokens during training. Pair those checkpoints with a normal tokenizer and generation never stops, every answer runs to the token limit, and your eval reads near zero for reasons unrelated to the model.

**You can't teach a base model a token it was never taught.** SFT trained the model to end its turn with `<|im_end|>`. The data was right, the loss covered that token, two epochs ran over it — and the trained model emits it essentially never.

The reason is in the base model. `Qwen2.5-0.5B` **base** carries the chat special tokens but never trained them; only instruction-tuned variants use them. Their embeddings sit near random initialisation:

```
<|endoftext|>   row norm 0.5987   98.40th percentile
<|im_end|>      row norm 0.3010    1.52th percentile
```

Adapters train attention and MLP, not the embedding — and Qwen ties the embedding to the output layer. So emitting `<|im_end|>` would mean aiming the hidden state at a near-random 896-dimensional vector while competing with a well-trained `<|endoftext|>`. The gradient had nowhere to go. It learned *where* to stop and picked the token it could reach.

The wider lesson is about checking. I already had an assertion for stop-token mismatches, and **it passes on the broken checkpoint** — it compares the tokenizer's setting to the saved tokenizer's setting, they agree, and the model agrees with neither. Two configs agreeing tells you nothing about what the weights do. The only real check is to generate and look, which is what `assert_model_stops` does now, in both trainers, before anything is saved. `sqlrl.base_report` runs the same check on any new base model before you train on it.

**A library default quietly cancelled the reward design.** TRL's `scale_rewards="group"` divides each group's scores by that group's own spread. The *ordering* of the reward ladder survives; the *spacing* does not. A group where the best answer was genuinely correct pushes exactly as hard as one where the best answer merely parsed. I'd have shipped a carefully designed ladder that behaved like a coin flip.

I found it by writing a script that attacks my own reward with policies that never read the question — `SELECT 1`, `SELECT *`, `WHERE 1=0`, degenerate cross joins — and measuring what they can extract. The best got 27% of what real answers get. It's `reward_probe.py`, and the habit generalises: probe the reward *before* spending the GPU hours, not after the curve looks strange.

---

## Terms

**Spider** — the benchmark. ~10,000 human-written questions across 200 real SQLite databases, split so **the test databases never appear in training**. You can't pass it by memorising one schema; you have to read one you've never seen.

**SFT** (supervised fine-tuning) — show the model thousands of question → correct-SQL pairs and have it imitate them. The bulk of the learning.

**GRPO** — a reinforcement learning algorithm. The model writes 8 answers to the same question, each is scored, and it's pushed toward whatever scored well. It learns from its *own* attempts rather than copied answers.

**LoRA** — instead of updating all the model's weights, freeze them and train a small number of extra ones bolted on the side. Much cheaper, and why this fits on one rented GPU.

**McNemar's test** — how two models are compared here. Both answer the same questions, so most of the data is shared and uninformative; only the questions where they *disagree* say anything. If the models were equally good, disagreements should split evenly. 251 vs 142 is not even, and the test says how unlikely that is by chance.

## Running it

```bash
uv sync
```

Training needs Linux and an NVIDIA GPU. Evaluation runs on a Mac (MPS) or CPU, slowly.

**Check a base model before training on it** — vocabulary facts that cost real GPU hours to discover the hard way:

```bash
uv run python -m sqlrl.base_report Qwen/Qwen2.5-Coder-1.5B
```

**Build the data.** Downloads Spider, checks it against the benchmark for contamination, executes every reference query before keeping it, and splits by *database* so none appears in two splits:

```bash
uv run python -m sqlrl.data_prep.build_spider_datasets
```

**Train.** SFT first, then GRPO from that checkpoint:

```bash
uv run python -m sqlrl.training.sft_spider --inspect    # look at what it'll train on
uv run python -m sqlrl.training.sft_spider --base-model Qwen/Qwen2.5-Coder-1.5B

uv run python -m sqlrl.training.grpo_spider --pilot     # 30 steps, check the diagnostics
uv run python -m sqlrl.training.grpo_spider
```

Both take `--inspect`, and I'd use it every time: every silent failure in this project was visible in the inputs, if anyone had bothered to look. Both also take `--resume`, which pulls the last checkpoint from S3 onto a fresh machine — tested by killing a run and deleting its disk, because spot instances are terminated rather than stopped.

**Evaluate:**

```bash
uv run python -m sqlrl.eval.run_eval --model all --split test
uv run python -m sqlrl.eval.run_eval --model all --split test --score-only   # rescore, no GPU
```

**Attack your own reward:**

```bash
uv run python -m sqlrl.training.reward_probe
```

**Tests:** 234, via `uv run pytest`. Heavily weighted toward the places where being wrong produces a believable number instead of a crash.

## Layout

```text
src/sqlrl/
  eval/
    executor.py       run SQL, compare result sets   <- everything depends on this
    metrics.py        the four numbers and a failure taxonomy
    spider.py         download, contamination check, reference verification
    prompts.py        prompt formats, and pulling SQL back out of model output
    run_eval.py       the harness
  training/
    sft_spider.py     supervised fine-tuning
    rewards.py        the execution-grounded GRPO reward
    reward_probe.py   attacks that reward with lazy policies
    grpo_spider.py    the RL run
    checkpoints.py    S3 checkpoint sync and spot resume
  data_prep/
    sample_traces.py  teacher sampling + execution filtering
    build_trace_sft.py
  base_report.py      what to check about a base model before training on it
  tokenizer.py        one place that decides how text becomes tokens
infra/                EC2: launch, start, stop, detached job runner, idle shutdown
```

The original trainers are still here, unmodified. They're the baseline every number above is measured against, so they stay.

## What's next

Two things, both aimed at the failure class that's left. The remaining errors are queries that run perfectly and answer the wrong question — which the current reward is blind to.

**Schema retrieval**, so databases too large to fit in a prompt become tractable at all.

**An agentic loop** that runs its own query, reads the error or the empty result, and tries again. A model that can *see* its query returned nothing has information no single-shot reward can give it.

## Costs

The whole thing runs on one `g5.xlarge` (A10G 24GB) at about $1/hr on-demand, plus an idle-shutdown cron job that has saved me more money than every other optimisation combined. A training run plus a full evaluation is about two hours.
