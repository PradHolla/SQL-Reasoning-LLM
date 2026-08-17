"""Latency, reported next to accuracy.

    uv run python -m sqlrl.serving.bench --model grpo-coder15 --n 100 --modes greedy,vote8,retry3

Every number this project has produced so far is an accuracy: 68.1% at
grpo-coder15, +3.5 points from voting, +1.1 from retry. None of it says
whether a query takes 200ms or 30 seconds -- and voting and retry both spend
GPU time to buy their accuracy, so the honest way to report them is latency
sitting right next to the number it paid for. This script is that report.

Questions are drawn from Spider test via ``load_split``, then subsampled with
a seeded ``random.Random`` so ``--n 100 --seed 3407`` picks the same 100
questions every run -- comparable across modes and across re-runs of this
script.
"""

from __future__ import annotations

import argparse
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

from sqlrl.eval.metrics import score_example
from sqlrl.eval.spider import Example, load_split
from sqlrl.serving.service import SqlService

__all__ = ["Mode", "ModeStats", "main", "parse_mode"]


@dataclass(frozen=True)
class Mode:
    """One inference-time technique, as the ``samples``/``max_attempts`` it
    drives through ``SqlService.answer``.
    """

    label: str
    samples: int
    max_attempts: int


def parse_mode(text: str) -> Mode:
    """``"greedy"`` | ``"vote<k>"`` | ``"retry<n>"`` -> the ``Mode`` it means."""
    if text == "greedy":
        return Mode("greedy", samples=1, max_attempts=1)
    if text.startswith("vote"):
        k = int(text[len("vote"):])
        return Mode(text, samples=k, max_attempts=1)
    if text.startswith("retry"):
        n = int(text[len("retry"):])
        return Mode(text, samples=1, max_attempts=n)
    raise ValueError(f"unrecognised mode {text!r}; expected greedy, vote<k> or retry<n>")


@dataclass(frozen=True)
class _Sample:
    total_ms: float
    generate_ms: float
    execute_ms: float
    correct: bool
    sql_tokens: int


@dataclass(frozen=True)
class ModeStats:
    n: int
    accuracy: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    throughput_qps: float
    #: Tokens in the *final SQL string* the model produced, not the full
    #: generation (which includes the <think> trace) -- Answer does not
    #: retain the raw completion, so this is the cheapest honest proxy for
    #: output length available from the service, not a measurement of total
    #: tokens generated. Labelled accordingly in the printed table.
    mean_sql_tokens: float


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(int(len(ordered) * fraction), len(ordered) - 1)
    return ordered[index]


def run_mode(
    service: SqlService, mode: Mode, examples: list[Example], *, timeout: float, temperature: float
) -> list[_Sample]:
    samples = []
    for example in examples:
        answer = service.answer(
            example.question, example.db_id,
            samples=mode.samples, max_attempts=mode.max_attempts, temperature=temperature,
        )
        score = score_example(answer.sql, example.gold_sql, example.db_path, timeout=timeout)
        sql_tokens = len(service.backend.tokenizer(answer.sql)["input_ids"])
        samples.append(_Sample(
            total_ms=answer.timings_ms["total"],
            generate_ms=answer.timings_ms["generate"],
            execute_ms=answer.timings_ms["execute"],
            correct=score.execution_match,
            sql_tokens=sql_tokens,
        ))
    return samples


def summarize(samples: list[_Sample], elapsed_seconds: float) -> ModeStats:
    n = len(samples)
    totals = [s.total_ms for s in samples]
    return ModeStats(
        n=n,
        accuracy=sum(s.correct for s in samples) / n if n else 0.0,
        p50_ms=_percentile(totals, 0.50),
        p95_ms=_percentile(totals, 0.95),
        p99_ms=_percentile(totals, 0.99),
        mean_ms=statistics.fmean(totals) if totals else 0.0,
        throughput_qps=n / elapsed_seconds if elapsed_seconds > 0 else 0.0,
        mean_sql_tokens=statistics.fmean(s.sql_tokens for s in samples) if n else 0.0,
    )


def format_table(rows: list[tuple[str, ModeStats]]) -> str:
    header = (
        f"{'mode':<10} {'n':>5} {'accuracy':>9} {'p50 ms':>8} {'p95 ms':>8} "
        f"{'p99 ms':>8} {'mean ms':>8} {'q/s':>7} {'sql tok':>8}"
    )
    lines = [header, "-" * len(header)]
    for label, stats in rows:
        lines.append(
            f"{label:<10} {stats.n:>5} {stats.accuracy:>9.1%} {stats.p50_ms:>8.0f} "
            f"{stats.p95_ms:>8.0f} {stats.p99_ms:>8.0f} {stats.mean_ms:>8.0f} "
            f"{stats.throughput_qps:>7.2f} {stats.mean_sql_tokens:>8.1f}"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Latency next to accuracy, per inference mode.")
    parser.add_argument("--model", default="grpo-coder15")
    parser.add_argument("--databases", type=Path, default=Path("data/spider/spider_data/test_database"))
    parser.add_argument("--split", choices=("dev", "test"), default="test")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--modes", default="greedy,vote8,retry3")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    modes = [parse_mode(name) for name in args.modes.split(",")]

    examples = load_split(args.split)
    sample = random.Random(args.seed).sample(examples, min(args.n, len(examples)))
    print(f"{args.model} on {len(sample)} {args.split} questions (seed={args.seed})\n")

    service = SqlService(
        args.model, databases=args.databases, device=args.device,
        batch_size=args.batch_size, max_new_tokens=args.max_new_tokens, timeout=args.timeout,
    )

    rows: list[tuple[str, ModeStats]] = []
    for mode in modes:
        started = time.perf_counter()
        mode_samples = run_mode(service, mode, sample, timeout=args.timeout, temperature=args.temperature)
        elapsed = time.perf_counter() - started
        rows.append((mode.label, summarize(mode_samples, elapsed)))
        print(f"  {mode.label}: {len(mode_samples)} questions in {elapsed:.1f}s")

    print()
    print(format_table(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
