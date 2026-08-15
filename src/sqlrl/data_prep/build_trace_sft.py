"""Turn verified teacher traces into the new SFT dataset.

    uv run python -m sqlrl.data_prep.build_trace_sft

Consumes what `sample_traces` produced and emits the same shape
`build_spider_datasets` emits, so `sft_spider.py` needs no changes at all: the
prompt is still `chat_prompt(schema, question)`, byte for byte, built by the
same function the evaluator uses.

**Exactly one thing changes: what is inside `<think>`.** Before, it was this
sentence, on all 5,378 examples::

    "I need to analyze the schema to find the correct tables and columns,
     then construct a valid SQL query."

After, it is a derivation the teacher produced for *this* question, which names
the tables and columns it needs and says why the others are not needed --
schema linking, reasoned out loud, which is the exact skill Phase 1.5 measured
as missing (88% of execution failures were hallucinated columns and tables).

If the next SFT checkpoint moves, the traces moved it. Nothing else differs.

**What this dataset is biased toward, stated plainly.** The teacher solved
4,881 of 5,378 questions at least once in 8 attempts (90.8%). The 497 it never
solved are dropped, and they are not a random 9% -- they are the hardest
questions, spread across 86 of the 110 databases. So the student now trains on a
distribution slightly easier than the benchmark it is judged on. That is
inherent to rejection sampling, not a bug in this script, and the honest place
to record it is here rather than in a footnote after the result.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from sqlrl.eval.prompts import chat_prompt, extract_sql

__all__ = ["build", "to_example"]

DEFAULT_TRACES = Path("data/processed/spider_traces.jsonl")
DEFAULT_SOURCE = Path("data/processed/spider_sft.jsonl")
DEFAULT_OUT = Path("data/processed/spider_sft_traces.jsonl")

_THINK = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)


def think_of(trace: str) -> str:
    match = _THINK.search(trace)
    return match.group(1).strip() if match else ""


def to_example(schema_text: str, question: str, think: str, sql: str) -> dict:
    """One training row, in the shape `sft_spider.to_prompt_completion` expects.

    The SQL is re-emitted through `extract_sql` by the caller rather than copied
    from the teacher verbatim. Teacher output carries trailing semicolons, line
    breaks and occasional prose, and `extract_sql` is what the evaluator will
    run on the student's output -- so training on its normalised form means the
    student is learning to produce exactly what will later be scored.
    """
    prompt = chat_prompt(schema_text, question)
    return {
        "messages": prompt.messages
        + [
            {
                "role": "assistant",
                "content": f"<think>\n{think}\n</think>\n<answer>\n{sql}\n</answer>",
            }
        ]
    }


def _schema_text(row: dict) -> str:
    user = row["messages"][1]["content"]
    return user.split("Schema: ", 1)[1].rsplit("\nQuestion:", 1)[0]


def build(
    traces: Path = DEFAULT_TRACES,
    source: Path = DEFAULT_SOURCE,
    out: Path = DEFAULT_OUT,
) -> None:
    by_index = {}
    for line in source.read_text().splitlines():
        if line.strip():
            by_index[len(by_index)] = json.loads(line)

    dropped: Counter[str] = Counter()
    records: list[dict] = []
    kept_meta: list[dict] = []

    for line in traces.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not row["trace"]:
            dropped["teacher never got it right"] += 1
            continue

        think = think_of(row["trace"])
        if not think:
            # Correct SQL with no reasoning at all. Keeping it would teach the
            # model that an empty <think> is acceptable, which is the habit this
            # entire phase exists to break.
            dropped["no <think> block"] += 1
            continue

        sql = extract_sql(row["trace"])
        if not sql:
            dropped["no SQL could be extracted"] += 1
            continue

        original = by_index[row["index"]]
        records.append(to_example(_schema_text(original), row["question"], think, sql))
        kept_meta.append({"db_id": row["db_id"], "n_correct": row["n_correct"],
                          "think_words": len(think.split())})

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    print(f"wrote {out}  ({len(records)} examples)")
    _report(records, kept_meta, dropped, by_index)


def _report(records, meta, dropped, by_index) -> None:
    total = len(by_index)
    print("\n=== kept ===")
    print(f"  {len(records)} of {total} ({len(records) / total:.1%})")
    if dropped:
        print("\n=== dropped ===")
        for reason, count in dropped.most_common():
            print(f"  {reason:32s} {count}")

    words = sorted(m["think_words"] for m in meta)
    n = len(words)
    print("\n=== reasoning length (words in <think>) ===")
    print(f"  median {words[n // 2]}   p95 {words[int(n * 0.95)]}   max {words[-1]}")
    print("  the dataset this replaces had 19 words, identical on every example")

    # How much of the training signal comes from questions the teacher found
    # trivial. A set dominated by 8/8 questions teaches less than its size
    # suggests, and this is the number that says so.
    easy = sum(1 for m in meta if m["n_correct"] == 8)
    hard = sum(1 for m in meta if m["n_correct"] <= 2)
    print("\n=== difficulty mix of what was kept ===")
    print(f"  teacher solved 8/8 (easy)   {easy:5d}  ({easy / len(meta):.1%})")
    print(f"  teacher solved <=2/8 (hard) {hard:5d}  ({hard / len(meta):.1%})")

    print(f"\n  databases represented: {len({m['db_id'] for m in meta})}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, default=DEFAULT_TRACES)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    build(args.traces, args.source, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
