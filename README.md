# SQL Reasoning LLM

> **A note on AI use:** this is a learning project, and I built it with AI models like Claude Opus, Sonnet, with also a little bit of Google Gemini too, as collaborators.

Post-training small language models (0.5B and 1.5B) to answer questions about a database by writing SQL that actually runs and returns the right rows.

**6.4% → 68.1%** execution accuracy on the Spider benchmark single-shot, **71.5%** with execution voting — level with a model five times larger, at eight times the inference cost.

And **45.2%** when it has to find its own tables in a 300-table database instead of being handed the right one. That last number is the honest one, and getting to it is most of what Phase 5 was about.

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

## The seven things worth knowing

1. **The original pipeline scored worse than the untrained model it started from** — 6.4% against 17.4% — and nobody could have known, because there was no held-out evaluation at all.
2. **The benchmark was half-leaked.** 562 of Spider dev's 1,034 questions appear verbatim in the training set, gold SQL included.
3. **One of the three training stages was actively destructive**, costing 17.4% → 3.1% once a 2×2 separated it from a prompt-format confound.
4. **Technique is worth +20.9 points independent of model size**, measured by running the old recipe and the new one on the same larger base. Both technique and scale matter, and they saturate against each other.
5. **Reinforcement learning stopped helping as the base model got better** — +5.1 points at 0.5B, +0.2 and not significant at 1.5B — while every diagnostic metric still improved. The reward paid for something that had stopped being the bottleneck.
6. **Handing the model the correct database was doing more work than four phases of training.** Make it retrieve its own tables and 63.5% becomes 45.2% — and most of that loss is *distraction*, not absence: the irrelevant tables cost twice what the missing ones do.
7. **How much the model agrees with itself predicts whether it's right**, for free. Unanimous across 8 samples → correct 84.5% of the time, and that's 70% of questions. A split vote is a coin flip.

Every number here is on held-out data, and differences are checked with a paired significance test rather than eyeballed.

---

## Where it stands

Spider **test** split: 2,147 questions over databases the model has never seen. Greedy decoding, one attempt per question, except the last row. **EX** (execution accuracy) means the predicted SQL and the reference SQL return the same rows from the real database.

| | EX | executes | parses | struct |
|---|---|---|---|---|
| the original pipeline (CPT→SFT→GRPO) | 6.4% | 35.2% | 78.8% | 2.7% |
| untrained base model, best prompt | 17.4% | 51.0% | 74.8% | 10.2% |
| **0.5B** — SFT rebuilt on Spider | 44.6% | 65.9% | 99.6% | 34.0% |
| **0.5B** — + GRPO with an execution reward | 49.7% | 76.5% | 99.3% | 38.8% |
| **1.5B** — same recipe, code-pretrained base | 67.9% | 88.2% | 99.8% | 46.3% |
| **1.5B** — + GRPO | **68.1%** | **90.6%** | **99.9%** | **47.3%** |
| Qwen2.5-Coder-7B-Instruct (5× bigger) | 71.2% | 89.3% | 95.8% | 35.9% |
| **1.5B** — + execution voting, 8 samples | **71.5%** | 95.1% | 100.0% | 48.5% |

The 7B row is there on purpose. Single-shot it still wins, and pretending otherwise would be silly — though the 1.5B model parses better (99.9% vs 95.8%) and matches query structure better (47.3% vs 35.9%).

The voting row draws level with it, but **that's eight times the inference for one model's worth of parameters**, so it's a compute-for-size trade rather than a free win, and it belongs with that caveat attached. [What it costs to run](#what-it-costs-to-run) has the latency.

One number for calibration on how precisely to read any of this: the greedy pass
re-run inside the voting job scored 68.0%, not 68.1% — same weights, same seed,
same questions, different batch composition. **Roughly a tenth of a point is the
noise floor here**, which is why nothing in this README leans on a difference
that small.

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

## Getting more out of the model you already have

Everything above changes the *weights*. At 68.1% the failures split like this:

```
1,462  correct                                          68%
  483  runs perfectly, answers the wrong question       22%
  202  the database rejects it outright                  9%
```

Those two failure types need completely different fixes, and the split is what
Phase 5 is built around.

### Letting it retry barely works, for an interesting reason

When the database rejects a query it says why — *no such column: customer_name*.
So show the model its own error and let it try again, up to three times.

**+1.1 points.** I predicted +4 and was wrong.

The reason is the useful part. Of 202 rejected queries, 63 became something the
database would accept — but only **23 of those were actually correct**. The other
40 moved from obviously broken to quietly wrong.

An error message tells you *that* you are wrong, never *what* is right. And on
the 0.5B model retry did essentially nothing (+0.2, 12 of 500 repaired), because
a model told "no such column" still has no idea which column does exist.

### Voting works, and works where retry can't

Sample 8 candidates, run all of them, group them **by the rows they return
rather than the SQL text**, and answer with the largest group. Two queries
written completely differently that return identical results are probably both
right; a hallucinated one usually returns something nobody else got.

```
                 retry      voting
1.5B model       +1.1        +3.5
0.5B model       +0.2        +3.0
```

Three times better, and 38 of the fixes came out of the *runs-fine-wrong-answer*
pile — the 483 queries retry is structurally blind to, because nothing goes
wrong for it to read.

The asymmetry across model size is the lesson: **retry needs a model good enough
to act on feedback; voting only needs one that's right sometimes**, and then
fishes that answer out. The second is a much weaker requirement, which is why
voting survives at a scale where retry collapses.

Voting also beat retry on retry's own ground — 45 rejected queries repaired
against 23. Seven more samples is a better repair mechanism than one error
message, which is worth knowing before building anything cleverer than sampling.

Grouping goes through the same `compare` the benchmark uses, not a hash of the
rows. It runs a bounded column-permutation search, so `SELECT age, name` and
`SELECT name, age` count as one answer — a hash would split exactly the groups
voting exists to merge. That executor is now doing three jobs: benchmark metric,
RL reward, and vote.

**The trap:** empty result sets all compare equal to each other. `WHERE 1=0`, a
hallucinated filter matching nothing, and a genuinely empty answer form one
cluster and vote as a bloc. Three people shrugging is not a consensus. Empty
clusters are demoted below any non-empty one: worth **+1.3 points**, and it costs
6 questions where the true answer really was empty. Both halves measured.

### Agreement predicts correctness, for free

The best thing to come out of Phase 5 costs no extra compute — it falls out of
the votes you already have:

| agreement across 8 samples | share of questions | how often correct |
|---|---|---|
| all 8 | 69.5% | **84.5%** |
| 6–7 of 8 | 12.3% | 56.1% |
| 1–5 of 8 | 13.2% | 44.0% |
| nothing ran | 4.9% | 0.0% |

**When the model agrees with itself it's right 85% of the time. When it doesn't,
it's a coin flip.** That's the difference between a system that silently returns
wrong numbers and one that can say "I'm not confident about this one" — which
matters more in a product than the 3.5 points do.

It replicates at both model sizes, shifted down at 0.5B, which is what makes it a
property of the method rather than a quirk of one checkpoint.

### The remaining gap is a selection problem, and heuristics can't touch it

Voting picks a wrong answer while holding a correct one **3.6% of the time**. That
looked like the cheapest remaining win — no training, no GPU, the candidates are
already on disk.

I looked at the failures before building anything: **87% are cases where the wrong
answer won 6-votes-to-2.** The model is confidently, consistently wrong, and no
frequency-based rule can override a 6–2 majority. I built two smarter selectors
anyway and measured them at +0.1 and −0.1.

Dead end, and worth the hour it took to prove rather than the week it would have
taken to build. Closing that gap needs a learned verifier, not a tiebreak rule.

---

## The number that would matter in production

Every score above hands the model the complete, correct schema for the exact
database. Real databases have hundreds of tables and don't fit in a context
window. So: can it find its own?

Building a fair test was most of the work, and the first design was wrong.
Spider's databases are tiny — a median of **4 tables** — so retrieval over one
measures nothing. The obvious fix, pooling all 206 databases into one haystack,
fails for a reason worth recording: **125 table names appear in more than one
database**, covering 41% of all tables. `customers` is in 22 of them. A question
about customers is then genuinely ambiguous, no retriever can resolve it, and the
experiment would have measured an impossible task and blamed the retriever.

The pool is instead a **collision-free** subset: 81 databases, 300 tables, 1,457
questions, no duplicate names. It renders to 8,262 tokens against a 3,072-token
limit, so retrieval is mandatory rather than decorative.

```
                     finds all needed tables      EX
keyword matching (BM25)        64.1%            37.1%
embedding matching             86.3%            45.2%
the correct database (control)    —             63.5%
exactly the right tables          —             68.4%
```

Two findings, neither of which I predicted.

**A minimal perfect schema beats the correct database by +4.9 points.** Less
schema is better at question time — the opposite direction to Phase 1.5, where
*training* on pruned schemas was actively harmful. Training on them means never
learning to pick tables out of a crowd; not having to pick at inference is pure
upside. Consistent with `unknown_column` having been the dominant failure since
day one.

**Most of the loss is distraction, not absence.** Split by whether retrieval
actually found everything:

```
all needed tables present   1,257 questions (86.3%)   EX 51.9%
at least one missing          200 questions (13.7%)   EX  3.0%
```

The missing half behaves as expected — 3% is effectively zero, those questions
are unanswerable. That costs about 9 points. But **even with every needed table
in the prompt, accuracy is 51.9% against oracle's 68.4%** — 16.5 points burned
purely by the nine irrelevant tables sitting next to the right ones. Execution
rate falls from 92.1% to 66.8%: the model reaches for a plausible table from a
different database entirely.

I predicted 58–61% by assuming EX = coverage × baseline, i.e. that finding the
tables was the whole problem. The gap between that and 45.2% is exactly the
distraction term that model has no room for. **Retrieval is a noise problem as
much as a search problem, and the noise is more expensive than the misses.**

---

## What it costs to run

The project could say it was 68.1% accurate and nothing whatsoever about whether
a query takes 200ms or 30 seconds. One A10G, one request at a time:

```
mode        accuracy   p50 ms   p95 ms   p99 ms     q/s
greedy         69.0%     2794     5143     8590    0.32
vote8          70.0%     5865    10711    17755    0.15
retry3         70.0%     2866    11686    28140    0.25
```

**The tail is the finding.** `retry3`'s median is indistinguishable from plain
greedy, because most requests never enter the loop — but its p99 is 28 seconds.
Retry is nearly free for the typical request and catastrophic for the unlucky
one, and the mean (3,932ms) hides that completely. Voting costs about 2× across
the board but is far more predictable at the tail. **The mean says retry is
cheaper; the tail says voting is safer**, and users experience the tail.

Accuracy over 100 questions can't resolve a 3.5-point effect and isn't offered as
evidence for one — it's a check that the service agrees with the harness, and
69.0% against 68.1% says it does. The service reuses the same voting and retry
code the numbers were measured with, rather than reimplementing it, precisely so
that check means something.

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

And it stayed invisible for three more phases. Three checkpoints emit their
answer and then keep generating to the token cap — **not one of 2,147 generations
stopped cleanly** — while every checkpoint trained after the fix stops 100% of the
time. Nobody noticed because `extract_sql` cuts at `</answer>` and cleaned up
after it every single time. Accuracy was never affected; the cost was compute,
roughly 6–8× on every evaluation of those checkpoints. A 0.5B model took 2,139
seconds where a *larger* one took 793, and that should have been a red flag.

There is now a `stop` column in the results table, so it can't hide again. Note
what it anchors on: the **first** `</answer>`, not the last. A runaway completion
often emits more `</answer>` tags in the text it rambles into, and anchoring on
the last one credits the model with stopping whenever the ramble happens to end
on a tag — 63 of 2,147 predictions. A metric written to expose this bug would
have hidden a sixteenth of it.

The wider lesson is about checking. I already had an assertion for stop-token mismatches, and **it passes on the broken checkpoint** — it compares the tokenizer's setting to the saved tokenizer's setting, they agree, and the model agrees with neither. Two configs agreeing tells you nothing about what the weights do. The only real check is to generate and look, which is what `assert_model_stops` does now, in both trainers, before anything is saved. `sqlrl.base_report` runs the same check on any new base model before you train on it.

**A library default quietly cancelled the reward design.** TRL's `scale_rewards="group"` divides each group's scores by that group's own spread. The *ordering* of the reward ladder survives; the *spacing* does not. A group where the best answer was genuinely correct pushes exactly as hard as one where the best answer merely parsed. I'd have shipped a carefully designed ladder that behaved like a coin flip.

I found it by writing a script that attacks my own reward with policies that never read the question — `SELECT 1`, `SELECT *`, `WHERE 1=0`, degenerate cross joins — and measuring what they can extract. The best got 27% of what real answers get. It's `reward_probe.py`, and the habit generalises: probe the reward *before* spending the GPU hours, not after the curve looks strange.

---

## Terms

**Spider** — the benchmark. ~10,000 human-written questions across 200 real SQLite databases, split so **the test databases never appear in training**. You can't pass it by memorising one schema; you have to read one you've never seen.

**SFT** (supervised fine-tuning) — show the model thousands of question → correct-SQL pairs and have it imitate them. The bulk of the learning.

**GRPO** — a reinforcement learning algorithm. The model writes 8 answers to the same question, each is scored, and it's pushed toward whatever scored well. It learns from its *own* attempts rather than copied answers.

**LoRA** — instead of updating all the model's weights, freeze them and train a small number of extra ones bolted on the side. Much cheaper, and why this fits on one rented GPU.

**Execution voting** — sample several answers, run them all, and go with whichever *result* the most answers agree on. Grouping by result rather than by query text is the whole trick: two correct queries rarely look alike.

**pass@k / the oracle** — is *any* of the k samples correct? The ceiling a perfect chooser could reach. The gap between it and what voting actually scores tells you whether to invest in the model or in the choosing.

**coverage@k** — of the tables a question needs, did retrieval find *all* of them? Reported instead of recall because finding 1 of 2 needed tables leaves the question just as unanswerable, and recall would generously call that 50%.

**p50 / p95 / p99** — the median request, the slowest 1 in 20, the slowest 1 in 100. Averages hide tails, and tails are what users complain about.

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

**Retrieval, and the end-to-end cost of it.** The first prints how often each
retriever finds every table a question needs; the second runs the model on what
retrieval gave it. `--retrieve gold` is the control, scored over the same
question subset — comparing against the full-split number would be comparing two
different benchmarks:

```bash
uv run python -m sqlrl.eval.retrieval --split test --k 1,3,5,10,20
uv run python -m sqlrl.eval.run_eval --model grpo-coder15 --retrieve dense --top-k 10
uv run python -m sqlrl.eval.run_eval --model grpo-coder15 --retrieve gold
```

**Inference-time techniques:**

```bash
uv run python -m sqlrl.eval.run_eval --model grpo-coder15 --samples 8        # voting
uv run python -m sqlrl.eval.run_eval --model grpo-coder15 --max-attempts 3   # retry
```

**Serve it, and find out what it costs:**

```bash
uv run uvicorn sqlrl.serving.api:app --port 8080
uv run python -m sqlrl.serving.bench --n 100 --modes greedy,vote8,retry3
```

**Tests:** 338, via `uv run pytest`. Heavily weighted toward the places where being wrong produces a believable number instead of a crash.

## Layout

```text
src/sqlrl/
  eval/
    executor.py       run SQL, compare result sets   <- everything depends on this
    metrics.py        the four numbers and a failure taxonomy
    spider.py         download, contamination check, reference verification
    prompts.py        prompt formats, and pulling SQL back out of model output
    voting.py         sample k, group by result rows, take the majority
    retry.py          feed the database's error back and try again
    retrieval.py      find the right tables in a 300-table pool
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
  serving/
    service.py        the model, plus voting/retry, plus a calibrated confidence
    api.py            HTTP: ask a question, get SQL, rows and a confidence
    bench.py          latency reported next to accuracy
  base_report.py      what to check about a base model before training on it
  tokenizer.py        one place that decides how text becomes tokens
infra/                EC2: launch, start, stop, detached job runner, idle shutdown
```

The original trainers are gone as of Phase 5, along with the `unsloth` dependency
whose only remaining importers they were. The old recipe is still reproducible —
`sft_spider.py --full-sequence-loss` is what produced the ablation it loses to —
so deleting them cost no measurement.

Removing that dependency broke three test modules, which was the useful part:
**`peft` was only ever installed because unsloth happened to depend on it**, while
this package imports it directly on the critical path of every number here. A
fresh clone had been broken the whole time and nothing said so. Same defect Phase
0 fixed for `fastapi` and `uvicorn`, wearing a different hat.

## What's next

Both of the things this section used to list — retrieval and an agentic loop —
are built and measured above. What they turned up reshapes the list.

**Cut the distraction, not the misses.** Retrieval's dominant cost is irrelevant
tables, not absent ones, so the obvious move — retrieve more — makes it worse. A
smaller k, or a second pass that prunes the retrieved set before the model sees
it. The oracle row says 68.4% is genuinely available if the schema arrives clean.

**A learned verifier**, for the 3.6% of questions where the model writes a correct
query and the vote picks a different one. Heuristics are measured dead on arrival
there; it needs a model that scores candidates against the question.

**vLLM**, for the 2.8-second median. Batch-of-one serving is the biggest weakness
in the numbers above and continuous batching is the standard answer. Removing
`unsloth` cleared one of the three things that made it unresolvable; whether the
other two still bite needs testing on a CUDA machine.

**Multi-turn RL** was the original plan and I'd skip it. The ceiling isn't how
often the model repairs a rejected query, it's that only 37% of repairs are
right — and at 1.5B, 56% of RL training groups already produce no gradient
because the model agrees with itself. Trajectory-level training makes that worse,
not better.

## Costs

The whole thing runs on one `g5.xlarge` (A10G 24GB) at about $1/hr on-demand, plus an idle-shutdown cron job that has saved me more money than every other optimisation combined. A training run plus a full evaluation is about two hours.
