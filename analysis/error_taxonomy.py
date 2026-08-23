"""What the remaining failures actually are.

``metrics.classify_error`` already buckets the queries that *do not run* --
schema hallucination, syntax, timeouts. That is reused here untouched. The gap
it cannot describe is the larger one: at grpo-coder15 the execution rate is
90.6% and EX is 68.1%, so roughly 22 points of the split produce a perfectly
valid query that returns the wrong answer, and "wrong" there is a single
undifferentiated bucket.

So executed-but-wrong is split by *how* the result differs from gold:

``empty_predicted``   ran, returned nothing, gold returns rows. Usually an
                      over-restrictive WHERE.
``extra_when_empty``  the mirror: gold is empty and the prediction is not.
``wrong_arity``       different number of columns -- selected the wrong thing
                      rather than filtered it wrongly.
``order_only``        identical rows, wrong order, on a query whose gold has
                      an ORDER BY. The answer is right; the sort is not.
``set_vs_multiset``   identical as *sets*, different as multisets: duplicate
                      rows differ. This bucket is worth its own name because
                      Spider's official evaluator dedupes and this project
                      does not, so it is exactly the population where our
                      numbers read lower than a leaderboard for a reason that
                      is about the metric, not the model.
``wrong_row_count``   right shape, wrong number of rows.
``wrong_values``      right shape, right count, different contents.

    uv run python -m analysis.error_taxonomy \
        --predictions results/test/grpo-coder15.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from sqlrl.eval.executor import compare, requires_order, run
from sqlrl.eval.metrics import score_example
from sqlrl.eval.run_eval import load
from sqlrl.eval.spider import load_split

__all__ = ["EXECUTED_WRONG", "build", "classify_wrong", "main"]

#: Ordered most specific first -- ``classify_wrong`` returns the first match,
#: so a result that is both mis-ordered and mis-counted is reported as the
#: narrower diagnosis.
EXECUTED_WRONG = (
    "extra_when_empty",
    "empty_predicted",
    "wrong_arity",
    "order_only",
    "set_vs_multiset",
    "wrong_row_count",
    "wrong_values",
)


def classify_wrong(pred_rows: list, gold_rows: list, order_matters: bool) -> str:
    """Why a query that ran cleanly still did not match."""
    if not gold_rows:
        return "extra_when_empty"
    if not pred_rows:
        return "empty_predicted"
    if len(pred_rows[0]) != len(gold_rows[0]):
        return "wrong_arity"
    # Relax one rule at a time; whichever relaxation makes it pass is the
    # single thing that was wrong with it.
    if order_matters and compare(pred_rows, gold_rows, False):
        return "order_only"
    if compare(pred_rows, gold_rows, order_matters, dedupe=True):
        return "set_vs_multiset"
    if len(pred_rows) != len(gold_rows):
        return "wrong_row_count"
    return "wrong_values"


def build(predictions_path: Path, split: str, timeout: float) -> dict[str, Any]:
    record = load(predictions_path)
    examples = load_split(split)

    counts: Counter[str] = Counter()
    scored = 0

    for prediction in record.predictions:
        example = examples[prediction.index]
        assert example.question == prediction.question, (
            f"prediction {prediction.index} does not line up with the benchmark; "
            f"the saved run is stale, regenerate it"
        )
        result = score_example(
            prediction.pred_sql, prediction.gold_sql, example.db_path,
            timeout=timeout, raw=prediction.raw,
        )
        if not result.gold_ok:
            continue
        scored += 1

        if result.execution_match:
            counts["correct"] += 1
            continue
        if result.pred_status != "ok":
            # Reuse, do not reimplement: ``error_kind`` is already
            # ``classify_error(status, error)``, so this bucketing and the one
            # the report prints can never disagree about the same run.
            counts[result.error_kind] += 1
            continue

        # Executed cleanly and still wrong -- the bucket this script exists for.
        pred = run(prediction.pred_sql, example.db_path, timeout=timeout)
        gold = run(prediction.gold_sql, example.db_path, timeout=timeout)
        counts[classify_wrong(
            pred.rows, gold.rows, requires_order(prediction.gold_sql)
        )] += 1

    correct = counts.get("correct", 0)
    executed_wrong = {k: counts.get(k, 0) for k in EXECUTED_WRONG if counts.get(k)}
    did_not_run = {
        k: v for k, v in counts.items()
        if k != "correct" and k not in EXECUTED_WRONG
    }

    return {
        "kind": "error_taxonomy",
        "scored": scored,
        "correct": correct,
        "accuracy": correct / scored if scored else 0.0,
        "executed_wrong": dict(sorted(executed_wrong.items(), key=lambda kv: -kv[1])),
        "did_not_run": dict(sorted(did_not_run.items(), key=lambda kv: -kv[1])),
        "executed_wrong_total": sum(executed_wrong.values()),
        "did_not_run_total": sum(did_not_run.values()),
        "provenance": {
            "source": str(predictions_path),
            "model": record.model,
            "split": record.split,
            "n": record.n,
            "seed": record.seed,
            "git_commit": record.git_commit,
            "generated_at": record.generated_at,
            "timeout": timeout,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "test"), default="test")
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    payload = build(args.predictions, args.split, args.timeout)
    out = args.out or Path(
        f"results/analysis/taxonomy-{payload['provenance']['model']}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1))

    scored = payload["scored"]
    print(f"  {payload['correct']}/{scored} correct ({payload['accuracy']:.1%})")
    print(f"  ran but wrong  ({payload['executed_wrong_total']}):")
    for kind, count in payload["executed_wrong"].items():
        print(f"    {kind:<18} {count:>5}  {count / scored:>6.1%}")
    print(f"  did not run    ({payload['did_not_run_total']}):")
    for kind, count in payload["did_not_run"].items():
        print(f"    {kind:<18} {count:>5}  {count / scored:>6.1%}")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
