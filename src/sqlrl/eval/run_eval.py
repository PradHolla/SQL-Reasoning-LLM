"""Run a model over a benchmark and print the model x metric table.

    uv run eval --model sft
    uv run eval --model all --split test
    uv run eval --model all --split test --score-only    # no GPU needed

**Generation and scoring are separate on purpose.** Predictions are written to
``results/<split>/<model>.json`` and scoring reads them back. Every metric in
this project is young enough to still have bugs in it, and re-scoring a fixed
metric should cost seconds of CPU, not another pass over 2,147 questions on a
GPU. ``--score-only`` re-derives every number from saved predictions.

Each run also records the git commit, device and decoding settings alongside
the predictions, because a number whose provenance you cannot reconstruct is
not a measurement.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from sqlrl.eval.executor import read_schema
from sqlrl.eval.metrics import Report, aggregate, format_report, score_example
from sqlrl.eval.prompts import chat_prompt, cpt_prompt, extract_sql, render_schema
from sqlrl.eval.retry import (
    STYLES,
    Attempt,
    Trace,
    at_budget,
    attempt_counts,
    run_retry,
)
from sqlrl.eval.spider import SPLITS, Example, load_split

__all__ = ["MODELS", "ModelSpec", "main"]

DEFAULT_RESULTS = Path("results")
BASE_MODEL = "Qwen/Qwen2.5-0.5B"
#: Phase 4. Code-pretrained and 3x the parameters; ModelSpec.base is per-spec,
#: so 0.5B and 1.5B checkpoints coexist in one table and one eval run.
CODER_1_5B = "Qwen/Qwen2.5-Coder-1.5B"


@dataclass(frozen=True)
class ModelSpec:
    """How to load one baseline.

    ``chat`` is the *tokenizer* mode -- which template and stop token the
    checkpoint was trained with. ``prompt`` is the *prompt shape* sent to it.
    They default together, because in v0 they were decided together: SFT and
    GRPO trained on ChatML, CPT on raw text. Separating them is only for
    diagnostics, where running a checkpoint against the other format is what
    tells a weights effect apart from a prompt-format effect.
    """

    name: str
    path: str
    base: str | None = None
    chat: bool = True
    #: "chat" or "cpt". Defaults to whatever `chat` implies. Set it explicitly
    #: only to run a checkpoint against a prompt format it was not trained on,
    #: which is how a weights effect gets separated from a prompt-format effect.
    prompt: str | None = None

    @property
    def prompt_style(self) -> str:
        return self.prompt or ("chat" if self.chat else "cpt")


#: The five baselines. Greedy, pass@1, identical treatment across all of them --
#: the comparison is only fair if nothing but the weights changes.
BASELINES: dict[str, ModelSpec] = {
    "base": ModelSpec("base", BASE_MODEL, chat=True),
    "cpt": ModelSpec("cpt", "models/qwen-0.5b-cpt-lora", base=BASE_MODEL, chat=False),
    "sft": ModelSpec("sft", "models/qwen-0.5b-sft-lora", base=BASE_MODEL, chat=True),
    "grpo": ModelSpec(
        "grpo", "models/qwen-0.5b-reasoning-final", base=BASE_MODEL, chat=True
    ),
    "coder-7b": ModelSpec("coder-7b", "Qwen/Qwen2.5-Coder-7B-Instruct", chat=True),
    # Phase 1.5: SFT redone on Spider train with full schemas. The bar it has to
    # clear is not v0's SFT (4.6%) but the untrained base model's best prompt,
    # 17.4% -- fine-tuning that lands below that is worse than not training.
    "sft-spider": ModelSpec(
        "sft-spider", "models/qwen-0.5b-sft-spider-2ep", base=BASE_MODEL, chat=True
    ),
    # eval_loss was 0.296 after one epoch and 0.299 after two, so the second
    # epoch may have bought nothing. Cheap to settle rather than assume.
    "sft-spider-1ep": ModelSpec(
        "sft-spider-1ep", "models/qwen-0.5b-sft-spider-1ep", base=BASE_MODEL, chat=True
    ),
    # Phase 2: GRPO from sft-spider-2ep with the execution-grounded reward, one
    # epoch over the 1,013-example GRPO split. The bar is sft-spider's 44.6% and
    # nothing else -- this is the first comparison in the project that isolates
    # what RL contributed, since v0's GRPO differed from its SFT in the split it
    # trained on as well as the algorithm.
    "grpo-spider": ModelSpec(
        "grpo-spider", "models/qwen-0.5b-grpo-spider", base=BASE_MODEL, chat=True
    ),
    # Phase 3: SFT on 4,823 execution-verified teacher traces instead of one
    # hardcoded sentence. The bar is sft-spider's 44.6% *before any RL* -- if
    # real reasoning helps a 0.5B model at all, it has to show up here.
    #
    # Evaluate this one with --max-new-tokens 640, not the 384 default: its
    # completions are trained to run to 485 tokens, and a budget that cuts the
    # reasoning off before it reaches <answer> would score the model wrong for a
    # reason that has nothing to do with SQL. The comparison stays fair because
    # sft-spider's completions are ~43 tokens, so its number was never
    # budget-limited either way.
    "sft-traces": ModelSpec(
        "sft-traces", "models/qwen-0.5b-sft-traces", base=BASE_MODEL, chat=True
    ),
    # 2 epochs of traces scored 41.8%, below the canned-sentence run's 44.6%,
    # and the measured suspect is gradient dilution: completion-only loss spends
    # 15.9% of its budget on the SQL here against 53.7% before. More epochs buy
    # back the absolute amount of SQL training. If this clears 44.6% the traces
    # work and simply cost more; if it stalls near 42%, dilution is not the
    # story and capacity is.
    "sft-traces-4ep": ModelSpec(
        "sft-traces-4ep", "models/qwen-0.5b-sft-traces-4ep", base=BASE_MODEL, chat=True
    ),
    # Phase 3 + Phase 2 stacked: GRPO from the trace-trained checkpoint. Phase 2
    # bought +5.1 points on a 44.6% base; the question is whether that gain
    # survives on a 47.1% base or gets absorbed. Also needs --max-new-tokens 640.
    "grpo-traces": ModelSpec(
        "grpo-traces", "models/qwen-0.5b-grpo-traces", base=BASE_MODEL, chat=True
    ),
    # Phase 4: the identical SFT recipe on a 3x larger, code-pretrained base.
    # Same data, epochs, learning rate, LoRA rank and targets as sft-spider;
    # only the base model differs, so the gap over 44.6% is what capacity and
    # code pretraining buy.
    "sft-coder15": ModelSpec(
        "sft-coder15", "models/qwen-coder-1.5b-sft-spider", base=CODER_1_5B, chat=True
    ),
    # Phase 4 task 4: the Phase 2 reward, unchanged, on the 1.5B base. Run at
    # the identical config on purpose -- retuning num_generations for this
    # policy would answer a different question than "what does the method
    # contribute independent of scale".
    "grpo-coder15": ModelSpec(
        "grpo-coder15", "models/qwen-coder-1.5b-grpo-spider", base=CODER_1_5B, chat=True
    ),
    # Phase 4 task 5, the ablation the whole phase exists for: v0's SFT recipe
    # on the *same* 1.5B base. Pruned-schema data, 500 steps, lr 2e-5, loss over
    # the full sequence. Against sft-coder15's 67.9%, the gap is what the method
    # contributes once parameter count is held fixed.
    #
    # Deliberately generous to v0: CPT omitted (measured destructive), Unsloth
    # omitted (corrupted the vocabulary), and the working stop token used. This
    # is the strongest fair version of v0, not v0 at its worst.
    "v0-coder15": ModelSpec(
        "v0-coder15", "models/qwen-coder-1.5b-v0style", base=CODER_1_5B, chat=True
    ),
}

#: Diagnostics, opt-in by name. These fill in the other two cells of the CPT
#: 2x2: Phase 1 measured CPT as neutral-to-harmful, but CPT was the only stage
#: evaluated with the raw completion prompt, so weights and prompt format were
#: confounded. With all four cells, comparing down a column isolates the weights
#: and across a row isolates the prompt.
DIAGNOSTICS: dict[str, ModelSpec] = {
    "base-cptprompt": ModelSpec(
        "base-cptprompt", BASE_MODEL, chat=False, prompt="cpt"
    ),
    "cpt-chatprompt": ModelSpec(
        "cpt-chatprompt", "models/qwen-0.5b-cpt-lora", base=BASE_MODEL,
        chat=True, prompt="chat",
    ),
    # base scores 17.4% with the completion prompt vs 3.6% with ChatML, which is
    # higher than the entire v0 pipeline. Before concluding the pipeline
    # destroyed value, check whether SFT and GRPO also gain from that prompt --
    # if they do, the format is the story; if they do not, the pipeline really
    # did end up below a correctly-prompted base model.
    "sft-cptprompt": ModelSpec(
        "sft-cptprompt", "models/qwen-0.5b-sft-lora", base=BASE_MODEL,
        chat=True, prompt="cpt",
    ),
    "grpo-cptprompt": ModelSpec(
        "grpo-cptprompt", "models/qwen-0.5b-reasoning-final", base=BASE_MODEL,
        chat=True, prompt="cpt",
    ),
}

MODELS: dict[str, ModelSpec] = {**BASELINES, **DIAGNOSTICS}


@dataclass
class Prediction:
    index: int
    db_id: str
    question: str
    gold_sql: str
    raw: str
    pred_sql: str


@dataclass
class RunRecord:
    """Predictions plus everything needed to reproduce them."""

    model: str
    split: str
    n: int
    device: str
    dtype: str
    max_new_tokens: int
    decoding: str
    seed: int
    git_commit: str
    generated_at: str
    generation_seconds: float
    predictions: list[Prediction] = field(default_factory=list)


@dataclass
class RetryRecord:
    """One retry run: a ``Trace`` per example plus the same provenance as
    ``RunRecord``, so a retry run is reproducible the same way a plain one is.
    """

    model: str
    split: str
    n: int
    device: str
    dtype: str
    max_new_tokens: int
    decoding: str
    seed: int
    git_commit: str
    generated_at: str
    generation_seconds: float
    max_attempts: int
    retry_style: str
    traces: list[Trace] = field(default_factory=list)

    def at_attempt(self, budget: int) -> RunRecord:
        """Collapse every trace to the attempt it would have stopped on with
        only ``budget`` tries, as a plain ``RunRecord``.

        This is the whole point of the design: ``score()``, ``aggregate()``
        and ``format_report()`` are reused unchanged for every budget, so
        there is no separate retry-metric code path that could be wrong.
        """
        predictions = []
        for trace in self.traces:
            attempt = at_budget(trace.attempts, budget)
            predictions.append(
                Prediction(
                    index=trace.index,
                    db_id=trace.db_id,
                    question=trace.question,
                    gold_sql=trace.gold_sql,
                    raw=attempt.raw,
                    pred_sql=attempt.sql,
                )
            )
        return RunRecord(
            model=self.model,
            split=self.split,
            n=self.n,
            device=self.device,
            dtype=self.dtype,
            max_new_tokens=self.max_new_tokens,
            decoding=self.decoding,
            seed=self.seed,
            git_commit=self.git_commit,
            generated_at=self.generated_at,
            generation_seconds=self.generation_seconds,
            predictions=predictions,
        )


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:  # noqa: BLE001 -- provenance is best-effort, not fatal
        return "unknown"


# --------------------------------------------------------------------------
# generation
# --------------------------------------------------------------------------


def build_prompts(examples: list[Example], style: str) -> list:
    """One prompt per example, in the requested shape."""
    schemas: dict[str, str] = {}
    prompts = []
    for example in examples:
        if example.db_id not in schemas:
            schemas[example.db_id] = render_schema(read_schema(example.db_path))
        text = schemas[example.db_id]
        prompts.append(
            chat_prompt(text, example.question) if style == "chat"
            else cpt_prompt(text, example.question)
        )
    return prompts


def generate(
    spec: ModelSpec,
    examples: list[Example],
    *,
    batch_size: int,
    max_new_tokens: int,
    device: str | None,
    seed: int,
) -> RunRecord:
    # Imported here so --score-only never pays for torch.
    from sqlrl.eval.backends.hf import HFBackend

    backend = HFBackend(
        spec.path,
        name=spec.name,
        base_model=spec.base,
        chat=spec.chat,
        device=device,
        batch_size=batch_size,
        seed=seed,
    )
    print(f"  loaded on {backend.device} ({backend.dtype}), "
          f"stops on {backend.stop_ids}, prompt={spec.prompt_style}")

    prompts = build_prompts(examples, spec.prompt_style)
    started = time.perf_counter()
    outputs = backend.generate(prompts, max_new_tokens=max_new_tokens)
    elapsed = time.perf_counter() - started
    print(f"  generated {len(outputs)} in {elapsed / 60:.1f} min "
          f"({elapsed / max(len(outputs), 1):.2f}s each)")

    return RunRecord(
        model=spec.name,
        split="",  # filled by the caller, which knows the split
        n=len(examples),
        device=backend.device,
        dtype=str(backend.dtype),
        max_new_tokens=max_new_tokens,
        decoding="greedy",
        seed=seed,
        git_commit=_git_commit(),
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        generation_seconds=round(elapsed, 1),
        predictions=[
            Prediction(
                index=index,
                db_id=example.db_id,
                question=example.question,
                gold_sql=example.gold_sql,
                raw=raw,
                pred_sql=extract_sql(raw),
            )
            for index, (example, raw) in enumerate(zip(examples, outputs))
        ],
    )


def generate_retry(
    spec: ModelSpec,
    examples: list[Example],
    *,
    batch_size: int,
    max_new_tokens: int,
    device: str | None,
    seed: int,
    max_attempts: int,
    style: str,
    timeout: float,
) -> RetryRecord:
    """Like ``generate``, but running each example through the retry loop.

    Retry needs turns to append feedback onto, which the CPT completion
    format does not have -- see ``retry_prompt``.
    """
    if spec.prompt_style != "chat":
        raise ValueError(
            f"retry requires the chat prompt style, but {spec.name} uses "
            f"{spec.prompt_style!r}"
        )

    # Imported here so --score-only never pays for torch.
    from sqlrl.eval.backends.hf import HFBackend

    backend = HFBackend(
        spec.path,
        name=spec.name,
        base_model=spec.base,
        chat=spec.chat,
        device=device,
        batch_size=batch_size,
        seed=seed,
    )
    print(f"  loaded on {backend.device} ({backend.dtype}), "
          f"stops on {backend.stop_ids}, prompt={spec.prompt_style}")

    prompts = build_prompts(examples, spec.prompt_style)
    db_paths = [example.db_path for example in examples]

    def on_round(attempt_number: int, n_generated: int) -> None:
        print(f"  attempt {attempt_number}: generating {n_generated}")

    started = time.perf_counter()
    histories = run_retry(
        prompts,
        db_paths,
        lambda batch: backend.generate(batch, max_new_tokens=max_new_tokens),
        max_attempts=max_attempts,
        style=style,
        timeout=timeout,
        on_round=on_round,
    )
    elapsed = time.perf_counter() - started
    print(f"  generated {len(histories)} traces in {elapsed / 60:.1f} min")

    traces = [
        Trace(
            index=index,
            db_id=example.db_id,
            question=example.question,
            gold_sql=example.gold_sql,
            attempts=attempts,
        )
        for index, (example, attempts) in enumerate(zip(examples, histories))
    ]

    return RetryRecord(
        model=spec.name,
        split="",  # filled by the caller, which knows the split
        n=len(examples),
        device=backend.device,
        dtype=str(backend.dtype),
        max_new_tokens=max_new_tokens,
        decoding="greedy",
        seed=seed,
        git_commit=_git_commit(),
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        generation_seconds=round(elapsed, 1),
        max_attempts=max_attempts,
        retry_style=style,
        traces=traces,
    )


def save(record: RunRecord, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(record), indent=1))
    print(f"  wrote {path}")


def load(path: Path) -> RunRecord:
    raw = json.loads(path.read_text())
    predictions = [Prediction(**p) for p in raw.pop("predictions")]
    return RunRecord(**raw, predictions=predictions)


def save_retry(record: RetryRecord, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(record), indent=1))
    print(f"  wrote {path}")


def load_retry(path: Path) -> RetryRecord:
    raw = json.loads(path.read_text())
    traces = []
    for trace in raw.pop("traces"):
        attempts = [Attempt(**a) for a in trace.pop("attempts")]
        traces.append(Trace(**trace, attempts=attempts))
    return RetryRecord(**raw, traces=traces)


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------


def score(record: RunRecord, examples: list[Example], timeout: float) -> tuple[Report, Report]:
    """Score a run, returning (all examples, uncontaminated examples only).

    Both are reported because on Spider dev they differ enormously -- 54% of
    that split is in the v0 training data. The gap is the memorisation.
    """
    scores, clean_scores = [], []
    for prediction in record.predictions:
        example = examples[prediction.index]
        assert example.question == prediction.question, (
            f"prediction {prediction.index} does not line up with the benchmark; "
            f"the saved run is stale, regenerate it"
        )
        result = score_example(
            prediction.pred_sql, prediction.gold_sql, example.db_path, timeout=timeout
        )
        scores.append(result)
        if not example.contaminated:
            clean_scores.append(result)
    return aggregate(scores), aggregate(clean_scores)


def format_retry(record: RetryRecord, budgets: list[tuple[int, Report]]) -> str:
    """How the retry budget was spent, next to what it bought.

    The per-budget accuracy rows say whether retry helped. This block says
    whether the loop was even reached, which is what distinguishes "retry
    does not work" from "almost nothing was ever retried". Both are null
    results and they have completely different follow-ups.
    """
    used, still_rejected = attempt_counts([trace.attempts for trace in record.traces])
    total = len(record.traces)

    def row(label: str, count: int) -> str:
        return f"    {label:<30}{count:>6}  {count / total:>7.1%}"

    lines = [f"  retry: {record.retry_style}, up to {record.max_attempts} attempts"]
    for attempts_used, count in used.items():
        plural = "attempt" if attempts_used == 1 else "attempts"
        lines.append(row(f"accepted after {attempts_used} {plural}", count))
    lines.append(row("still rejected at the end", still_rejected))

    if budgets:
        track = "  ->  ".join(
            f"@{budget} {report.execution_accuracy:.1%}" for budget, report in budgets
        )
        delta = budgets[-1][1].execution_accuracy - budgets[0][1].execution_accuracy
        lines.append(f"    EX  {track}   ({delta * 100:+.1f} pts)")
    return "\n".join(lines)


def comparison_table(rows: list[tuple[str, Report, Report]]) -> str:
    """The model x metric table. The deliverable of Phase 1."""
    header = (
        f"{'model':<16} {'EX':>7} {'EX/clean':>9} {'exec':>7} {'parse':>7} "
        f"{'struct':>7} {'n':>6}"
    )
    lines = [header, "-" * len(header)]
    for name, full, clean in rows:
        lines.append(
            f"{name:<16} {full.execution_accuracy:>7.1%} {clean.execution_accuracy:>9.1%} "
            f"{full.execution_rate:>7.1%} {full.parse_rate:>7.1%} "
            f"{full.structural_match:>7.1%} {full.n:>6}"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate models on Spider.")
    parser.add_argument("--model", default="all",
                        help="model name, comma-separated names, 'all' for the "
                             "five baselines, or 'cpt2x2' for the CPT experiment")
    parser.add_argument("--split", choices=SPLITS, default="test")
    parser.add_argument("--limit", type=int, default=None,
                        help="evaluate only the first N examples (smoke tests)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--score-only", action="store_true",
                        help="re-score saved predictions without loading a model")
    parser.add_argument("--max-attempts", type=int, default=1,
                        help="retry budget; 1 leaves generation and scoring on "
                             "the pre-retry path untouched")
    parser.add_argument("--retry-style", choices=STYLES, default="multiturn",
                        help="feedback shape for rounds after the first -- see "
                             "sqlrl.eval.retry")
    args = parser.parse_args()

    if args.model == "all":
        names = list(BASELINES)
    elif args.model == "cpt2x2":
        # The four cells of the CPT disentangling experiment.
        names = ["base", "cpt", "base-cptprompt", "cpt-chatprompt"]
    elif args.model == "promptsweep":
        # Every 0.5B checkpoint against both prompt formats.
        names = ["base", "base-cptprompt", "cpt", "cpt-chatprompt",
                 "sft", "sft-cptprompt", "grpo", "grpo-cptprompt"]
    else:
        names = args.model.split(",")
    unknown = [n for n in names if n not in MODELS]
    if unknown:
        parser.error(f"unknown model(s): {unknown}. Choose from {list(MODELS)}")

    examples = load_split(args.split)
    if args.limit:
        examples = examples[: args.limit]
    contaminated = sum(example.contaminated for example in examples)
    print(f"{args.split}: {len(examples)} examples, {contaminated} contaminated "
          f"({contaminated / len(examples):.1%})\n")

    out_dir = args.results / args.split
    rows: list[tuple[str, Report, Report]] = []

    for name in names:
        spec = MODELS[name]
        print(f"=== {name} ===")

        if args.max_attempts > 1:
            # Separate result files: a retry run is a different measurement
            # from a plain one, over a different (larger) generation budget,
            # and must never be confused with or overwrite it.
            suffix = "-retry" if args.retry_style == "multiturn" else "-retry-restate"
            retry_path = out_dir / f"{name}{suffix}.json"

            if args.score_only:
                if not retry_path.is_file():
                    print(f"  no saved predictions at {retry_path}, skipping\n")
                    continue
                retry_record = load_retry(retry_path)
            else:
                retry_record = generate_retry(
                    spec, examples,
                    batch_size=args.batch_size,
                    max_new_tokens=args.max_new_tokens,
                    device=args.device,
                    seed=args.seed,
                    max_attempts=args.max_attempts,
                    style=args.retry_style,
                    timeout=args.timeout,
                )
                retry_record.split = args.split
                save_retry(retry_record, retry_path)

            # Score every budget -- the whole question this mode exists to
            # answer is how accuracy moves as the budget grows -- but only
            # print the full report for the largest one, so the output stays
            # readable when max_attempts is more than a couple.
            budget_reports: list[tuple[int, Report]] = []
            for budget in range(1, args.max_attempts + 1):
                full, clean = score(retry_record.at_attempt(budget), examples, args.timeout)
                row_name = f"{name}@{budget}"
                if budget == args.max_attempts:
                    print()
                    print(format_report(full, title=f"{row_name} — {args.split} (all {full.n})"))
                    print()
                    print(format_report(
                        clean, title=f"{row_name} — {args.split} (uncontaminated only)"
                    ))
                    print()
                budget_reports.append((budget, full))
                rows.append((row_name, full, clean))
            print(format_retry(retry_record, budget_reports))
            print()
            continue

        path = out_dir / f"{name}.json"

        if args.score_only:
            if not path.is_file():
                print(f"  no saved predictions at {path}, skipping\n")
                continue
            record = load(path)
        else:
            record = generate(
                spec, examples,
                batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens,
                device=args.device,
                seed=args.seed,
            )
            record.split = args.split
            save(record, path)

        full, clean = score(record, examples, args.timeout)
        print()
        print(format_report(full, title=f"{name} — {args.split} (all {full.n})"))
        print()
        print(format_report(clean, title=f"{name} — {args.split} (uncontaminated only)"))
        print()
        rows.append((name, full, clean))

    if rows:
        budget = "greedy, pass@1" if args.max_attempts == 1 else (
            f"greedy, up to {args.max_attempts} attempts ({args.retry_style})"
        )
        print("=" * 60)
        print(f"Spider {args.split} — {budget}")
        print("=" * 60)
        print(comparison_table(rows))
        # summary.txt is the project's headline table, written by the full
        # `--model all` sweep. A one-model run used to overwrite it with a
        # single row -- which has already cost this project the table twice --
        # so anything narrower than the full sweep gets its own file.
        summary = out_dir / (
            "summary.txt" if args.model == "all" and args.max_attempts == 1
            else f"summary-{_slug(args)}.txt"
        )
        summary.write_text(comparison_table(rows) + "\n")
        print(f"\nwrote {summary}")
    return 0


def _slug(args: argparse.Namespace) -> str:
    """Filename stem describing what was run, so runs do not overwrite each other."""
    stem = args.model.replace(",", "+")
    if args.max_attempts > 1:
        stem += f"-retry{args.max_attempts}-{args.retry_style}"
    return stem


if __name__ == "__main__":
    raise SystemExit(main())
