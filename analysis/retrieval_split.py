"""Why retrieval costs what it costs: absence or distraction.

End-to-end accuracy over the 300-table pool falls a long way below the
control, and the aggregate number cannot say which of two things did it:

* **absence** -- the retriever missed a table the gold query needs, so the
  answer was never reachable from the prompt.
* **distraction** -- every gold table *was* retrieved, and the model still got
  it wrong, with the retrieved-but-irrelevant tables as the only difference
  from the ``oracle`` condition.

Those imply opposite fixes. Absence says retrieve more; distraction says
retrieve less. So this splits the run four ways -- covered/uncovered x
right/wrong -- and prices each side against the ``oracle`` control.

**The retrieved sets are not in the results file.** ``RunRecord.predictions``
keeps only the prompt's output, never its input, so recovering what each
question was actually shown means re-running the retriever with the same mode
and k. That is the one thing here that must match the original run exactly;
``score()`` asserts question alignment, which catches a mismatched example
list, and ``--expect-ex`` below catches a mismatched retriever.

    uv run python -m analysis.retrieval_split \
        --predictions results/test/grpo-coder15-retr-dense10.json \
        --mode dense --top-k 10 --expect-ex 0.452
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sqlrl.eval.metrics import score_example
from sqlrl.eval.retrieval import (
    BM25,
    Dense,
    build_pool,
    coverage_at_k,
    gold_tables,
    pool_questions,
    recall_at_k,
)
from sqlrl.eval.run_eval import load
from sqlrl.eval.spider import load_split

__all__ = ["build", "main"]

#: How far the recomputed EX may drift from the number the run reported before
#: this refuses to write a file. Scoring is deterministic, so any drift at all
#: means the inputs are not what they are believed to be -- a different split,
#: a stale predictions file, a changed comparison rule. 0.001 is "the same
#: number, printed to fewer digits", not a tolerance for real disagreement.
EX_TOLERANCE = 0.001


def build(
    predictions_path: Path,
    mode: str,
    top_k: int,
    split: str,
    timeout: float,
    device: str | None,
    expect_ex: float | None,
) -> dict[str, Any]:
    record = load(predictions_path)

    # Exactly how run_eval.main builds it for --retrieve. Any divergence here
    # misaligns every index in the file, so it is one line, copied, not a
    # reimplementation.
    pool = build_pool(split)
    examples = pool_questions(load_split(split), pool)

    if len(record.predictions) != len(examples):
        raise ValueError(
            f"{predictions_path} has {len(record.predictions)} predictions but the "
            f"pool subset of {split} has {len(examples)} questions; the run and the "
            f"benchmark do not match"
        )

    if mode == "bm25":
        retriever: Any = BM25(pool)
        hits: list[list[int]] | None = [retriever.search(ex.question, top_k) for ex in examples]
    elif mode == "dense":
        retriever = Dense(pool, device=device)
        hits = retriever.search_many([ex.question for ex in examples], top_k)
    else:
        # gold and oracle retrieve nothing: gold shows the whole database and
        # oracle shows exactly the gold query's tables, so coverage is 1.0 by
        # construction in both. They run through the same path anyway so the
        # four conditions come out of one script with one definition of EX --
        # a control scored by different code than the thing it controls for is
        # not a control.
        hits = None

    # covered x correct, in that order.
    cells = {(True, True): 0, (True, False): 0, (False, True): 0, (False, False): 0}
    recalls: list[float] = []
    matches = 0
    scored = 0

    for prediction in record.predictions:
        example = examples[prediction.index]
        assert example.question == prediction.question, (
            f"prediction {prediction.index} does not line up with the pool subset; "
            f"the saved run is stale, regenerate it"
        )
        result = score_example(
            prediction.pred_sql, prediction.gold_sql, example.db_path,
            timeout=timeout, raw=prediction.raw,
        )
        if not result.gold_ok:
            # Same denominator rule as execution_accuracy: an example whose own
            # gold query will not run is not evidence about the retriever.
            continue
        scored += 1
        matches += result.execution_match

        if hits is None:
            covered = True
            recalls.append(1.0)
        else:
            docs = [pool[i] for i in hits[prediction.index]]
            gold = gold_tables(prediction.gold_sql, example.db_id)
            covered = coverage_at_k(docs, gold) if gold else True
            recalls.append(recall_at_k(docs, gold) if gold else 1.0)
        cells[(covered, result.execution_match)] += 1

    ex = matches / scored if scored else 0.0
    if expect_ex is not None and abs(ex - expect_ex) > EX_TOLERANCE:
        raise ValueError(
            f"recomputed EX {ex:.4f} but --expect-ex says {expect_ex:.4f}. The "
            f"predictions, the split or the comparison rule is not what this "
            f"analysis assumes; refusing to write a file that would look right."
        )

    covered_n = cells[(True, True)] + cells[(True, False)]
    uncovered_n = cells[(False, True)] + cells[(False, False)]

    return {
        "kind": "retrieval_split",
        "mode": mode,
        "top_k": top_k,
        "overall_ex": ex,
        "scored": scored,
        "coverage_at_k": covered_n / scored if scored else 0.0,
        "mean_recall_at_k": sum(recalls) / len(recalls) if recalls else 0.0,
        "cells": {
            "covered_correct": cells[(True, True)],
            "covered_wrong": cells[(True, False)],
            "uncovered_correct": cells[(False, True)],
            "uncovered_wrong": cells[(False, False)],
        },
        # The two conditional accuracies are the whole finding. ``ex_covered``
        # is the ceiling retrieval could reach if recall were perfect, and the
        # distance from it to the oracle control is what distractors cost.
        "ex_covered": cells[(True, True)] / covered_n if covered_n else 0.0,
        "ex_uncovered": cells[(False, True)] / uncovered_n if uncovered_n else 0.0,
        "provenance": {
            "source": str(predictions_path),
            "model": record.model,
            "split": record.split,
            "n": record.n,
            "pool_tables": len(pool),
            "pool_questions": len(examples),
            "seed": record.seed,
            "git_commit": record.git_commit,
            "generated_at": record.generated_at,
            "timeout": timeout,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--mode", choices=("gold", "oracle", "bm25", "dense"), required=True)
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--split", choices=("dev", "test"), default="test")
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--expect-ex", type=float, default=None,
                        help="EX this run already reported, as a guard against "
                             "analysing the wrong file (e.g. 0.452)")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    payload = build(args.predictions, args.mode, args.top_k, args.split,
                    args.timeout, args.device, args.expect_ex)
    suffix = str(args.top_k) if args.mode in ("bm25", "dense") else ""
    out = args.out or Path(f"results/analysis/retrieval-{args.mode}{suffix}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1))

    print(f"  {args.mode}@{args.top_k}: EX {payload['overall_ex']:.1%}  "
          f"coverage {payload['coverage_at_k']:.1%}")
    print(f"    all gold tables present -> {payload['ex_covered']:.1%} "
          f"({payload['cells']['covered_correct']}/"
          f"{payload['cells']['covered_correct'] + payload['cells']['covered_wrong']})")
    print(f"    a gold table missing    -> {payload['ex_uncovered']:.1%} "
          f"({payload['cells']['uncovered_correct']}/"
          f"{payload['cells']['uncovered_correct'] + payload['cells']['uncovered_wrong']})")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
