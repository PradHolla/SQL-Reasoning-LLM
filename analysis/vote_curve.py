"""vote@k against pass@k -- the data behind the k-curve chart.

Two lines, and the gap between them is the whole point:

* **vote@k** is what the system actually answers with. ``VoteRecord.at_k``
  collapses each ballot to its winner and the existing ``score()`` path grades
  it, so this is not a second implementation of the metric.
* **pass@k** is the ceiling: did *any* of the k candidates return the gold
  rows. The gap is what a perfect selector would buy, and it is the honest way
  to say whether voting is near its limit or leaving points on the table.

Hydration happens once, up front. Vote runs deliberately do not persist result
rows (see ``run_eval.hydrate_votes``), so every k is computed from one
re-execution pass rather than one per budget -- 8 candidates x 2,147 questions
is ~40s of SQLite, and doing it four times because the loop was written the
obvious way would be four times that for no additional information.

    uv run python -m analysis.vote_curve \
        --votes results/test/grpo-coder15-vote8.json \
        --out results/analysis/vote_curve.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from sqlrl.eval.executor import compare, requires_order, run
from sqlrl.eval.run_eval import (
    hydrate_votes,
    load_votes,
    oracle_rate,
    score,
    vote_budgets,
)
from sqlrl.eval.spider import load_split
from sqlrl.eval.voting import select
from sqlrl.serving.service import winner_agreement

__all__ = ["build", "calibration", "main"]


def calibration(record, examples, k: int, timeout: float) -> list[dict[str, Any]]:
    """Accuracy per level of self-agreement -- the data behind the claim that
    voting hands you a confidence signal for free.

    Agreement comes from ``service.winner_agreement`` and selection from
    ``voting.select``, i.e. the two functions the running service uses, so
    these buckets describe the deployed behaviour rather than a
    reconstruction of it.

    Buckets are reported raw, one per agreement level. Collapsing them into
    the coarse high/medium/low bands is ``service.CALIBRATION``'s job, and it
    needs the per-level counts to justify where its boundaries fall -- so this
    deliberately stops short of that and reports the population of each level
    alongside its accuracy, because a level holding 27 ballots and one holding
    1,492 should not be read the same way.
    """
    buckets: dict[int, list[int]] = {}
    for ballot in record.ballots:
        example = examples[ballot.index]
        gold = run(ballot.gold_sql, example.db_path, timeout=timeout)
        if not gold.ok:
            continue
        subset = ballot.candidates[:k]
        index = select(subset)
        winner = subset[index]
        agreement = winner_agreement(subset, index)
        correct = winner.status == "ok" and compare(
            winner.rows, gold.rows, requires_order(ballot.gold_sql)
        )
        counts = buckets.setdefault(agreement, [0, 0])
        counts[0] += correct
        counts[1] += 1

    total = sum(n for _, n in buckets.values())
    return [
        {
            "agreement": agreement,
            "of": k,
            "correct": correct,
            "n": n,
            "accuracy": correct / n if n else 0.0,
            "coverage": n / total if total else 0.0,
        }
        for agreement, (correct, n) in sorted(buckets.items(), reverse=True)
    ]


def build(votes_path: Path, split: str, timeout: float) -> dict[str, Any]:
    """Every budget in one pass. Returns the chart payload, provenance included."""
    record = load_votes(votes_path)
    if record.split != split:
        raise ValueError(
            f"{votes_path} is a {record.split!r} run but --split says {split!r}; "
            f"scoring it against the wrong benchmark would line up by index and "
            f"be wrong by content"
        )

    examples = load_split(split)
    print(f"  hydrating {len(record.ballots)} ballots x {record.samples} candidates...")
    started = time.perf_counter()
    hydrate_votes(record, examples, timeout)

    points = []
    for k in vote_budgets(record.samples):
        report, clean = score(record.at_k(k), examples, timeout)
        oracle = oracle_rate(record.ballots, examples, k, timeout)
        points.append({
            "k": k,
            "vote_ex": report.execution_accuracy,
            "vote_ex_clean": clean.execution_accuracy,
            "oracle_ex": oracle,
            # The selection gap: how much a perfect chooser would add on top of
            # what majority voting already gets. This is the number that says
            # whether to invest in a better selector or in a better model.
            "selection_gap": oracle - report.execution_accuracy,
            "structural_match": report.structural_match,
            "execution_rate": report.execution_rate,
            "scored": report.scored,
        })
        print(
            f"  k={k:<3} vote {report.execution_accuracy:.1%}  "
            f"oracle {oracle:.1%}  gap {oracle - report.execution_accuracy:.1%}"
        )

    print(f"  calibrating at k={record.samples}...")
    buckets = calibration(record, examples, record.samples, timeout)
    for bucket in buckets:
        print(f"    {bucket['agreement']}/{bucket['of']}  "
              f"{bucket['accuracy']:>6.1%}  n={bucket['n']:<5} "
              f"({bucket['coverage']:.1%} of questions)")

    return {
        "kind": "vote_curve",
        "points": points,
        "calibration": buckets,
        "provenance": {
            "source": str(votes_path),
            "model": record.model,
            "split": record.split,
            "n": record.n,
            "samples": record.samples,
            "temperature": record.temperature,
            "top_p": record.top_p,
            "seed": record.seed,
            "git_commit": record.git_commit,
            "generated_at": record.generated_at,
            "generation_seconds": record.generation_seconds,
            "scored_seconds": round(time.perf_counter() - started, 1),
            "timeout": timeout,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--votes", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "test"), default="test")
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--out", type=Path, default=Path("results/analysis/vote_curve.json"))
    args = parser.parse_args()

    payload = build(args.votes, args.split, args.timeout)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1))
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
