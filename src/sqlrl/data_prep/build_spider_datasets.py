"""Build SFT / GRPO / validation datasets from Spider train.

    uv run python -m sqlrl.data_prep.build_spider_datasets

Replaces `b-mc2/sql-create-context` as the training source. Two reasons, both
measured rather than assumed:

1. **`sql-create-context` has no databases.** It carries `CREATE TABLE` text and
   nothing else, so an execution-grounded reward — the entire point of Phase 2 —
   cannot be computed on it. Spider train ships 140 real SQLite files.
2. **Its schemas are pruned to the columns each question needs.** Training on
   "here are exactly the two columns required" teaches the model *not to read
   the schema*, which is precisely the skill Spider tests. The v0 pipeline
   scored 6.4% on Spider test while the untrained base model scored 17.4%:
   fine-tuning on that distribution was worse than not fine-tuning at all.

Three design rules, each guarding against a way this could silently poison
training:

* **Schemas are rendered by the same code the evaluator uses.** `render_schema`
  and `read_schema` are imported, never reimplemented. If train and eval
  disagree about what a schema looks like, we recreate the exact distribution
  mismatch this dataset exists to fix.
* **Splits are by database, not by example.** Spider itself separates train/dev/
  test by database. Splitting randomly would let validation ask new questions
  about databases the model trained on, which measures something easier than
  the benchmark does and would read as progress.
* **Every gold query is executed before it is kept.** A gold query that does not
  run is a training target that teaches noise.

The `<think>` block is deliberately left as v0's single hardcoded sentence.
Replacing it with real reasoning traces is Phase 3's whole job, and changing it
here would make Phase 3's contribution unattributable. One change at a time:
this dataset changes the *data*, nothing else.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from sqlrl.eval.executor import read_schema, run
from sqlrl.eval.prompts import chat_prompt, render_schema
from sqlrl.eval.spider import DEFAULT_ROOT, _norm_question, ensure_data, load_split

__all__ = ["build"]

#: Verbatim from format_sft_data.py. Kept identical on purpose -- see docstring.
THINK = (
    "I need to analyze the schema to find the correct tables and columns, "
    "then construct a valid SQL query."
)

DEFAULT_OUT = Path("data/processed")

#: Token-length reporting only. Qwen2.5 and Qwen2.5-Coder share a 151,936
#: vocabulary, so this does not have to track the base model being trained.
TOKENIZER_FOR_REPORT = "Qwen/Qwen2.5-0.5B"

#: Databases go to validation first, then GRPO, and SFT takes the remainder.
#: Filling the small splits first is what makes them land near their targets.
VAL_TARGET = 500
GRPO_TARGET = 1_000

#: Anything longer than this is truncated by the trainer, which silently teaches
#: the model a cut-off query. Reported, never quietly accepted.
MAX_TOKENS = 2_048


def _load_train(root: Path) -> list[dict]:
    return json.loads((Path(root) / "spider_data" / "train_spider.json").read_text())


def _assign_splits(
    rows: list[dict], seed: int
) -> dict[str, str]:
    """Map db_id -> split name, keeping whole databases together."""
    by_db: dict[str, int] = Counter(row["db_id"] for row in rows)
    dbs = sorted(by_db)  # sort first so the shuffle is reproducible
    random.Random(seed).shuffle(dbs)

    assignment: dict[str, str] = {}
    counts = {"val": 0, "grpo": 0, "sft": 0}
    for db in dbs:
        if counts["val"] < VAL_TARGET:
            split = "val"
        elif counts["grpo"] < GRPO_TARGET:
            split = "grpo"
        else:
            split = "sft"
        assignment[db] = split
        counts[split] += by_db[db]
    return assignment


def build(
    root: Path = DEFAULT_ROOT,
    out_dir: Path = DEFAULT_OUT,
    seed: int = 3407,
    timeout: float = 30.0,
) -> None:
    root = ensure_data(root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    databases = Path(root) / "spider_data" / "database"

    rows = _load_train(root)
    print(f"Spider train: {len(rows)} examples, "
          f"{len({r['db_id'] for r in rows})} databases")

    # Anything the benchmark also asks must not be trained on.
    benchmark = {
        _norm_question(ex.question)
        for split in ("test", "dev")
        for ex in load_split(split, root)
    }

    assignment = _assign_splits(rows, seed)
    buckets: dict[str, list[dict]] = defaultdict(list)
    dropped: Counter[str] = Counter()
    empty_gold: Counter[str] = Counter()
    schemas: dict[str, str] = {}

    for row in rows:
        db_id = row["db_id"]
        db_path = databases / db_id / f"{db_id}.sqlite"
        if not db_path.is_file():
            dropped["missing database"] += 1
            continue
        if _norm_question(row["question"]) in benchmark:
            dropped["question appears in test/dev"] += 1
            continue

        # A gold query that will not run is a target that teaches noise.
        result = run(row["query"], db_path, timeout=timeout)
        if not result.ok:
            dropped[f"gold {result.status}"] += 1
            continue

        if db_id not in schemas:
            schemas[db_id] = render_schema(read_schema(db_path))

        split = assignment[db_id]
        if not result.rows:
            empty_gold[split] += 1

        prompt = chat_prompt(schemas[db_id], row["question"])
        buckets[split].append(
            {
                "db_id": db_id,
                "db_path": str(db_path),
                "question": row["question"],
                "gold_sql": row["query"],
                # The exact system+user turns the evaluator builds, plus the
                # target turn. Reused, not reimplemented, so train and eval
                # cannot drift apart.
                "messages": prompt.messages
                + [
                    {
                        "role": "assistant",
                        "content": f"<think>\n{THINK}\n</think>\n"
                                   f"<answer>\n{row['query']}\n</answer>",
                    }
                ],
            }
        )

    for split, records in buckets.items():
        path = out_dir / f"spider_{split}.jsonl"
        with path.open("w") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
        print(f"  wrote {path}  ({len(records)} examples)")

    _report(buckets, dropped, empty_gold, assignment)


def _report(
    buckets: dict[str, list[dict]],
    dropped: Counter[str],
    empty_gold: Counter[str],
    assignment: dict[str, str],
) -> None:
    from sqlrl.tokenizer import build_tokenizer

    print("\n=== splits ===")
    for split in ("sft", "grpo", "val"):
        records = buckets.get(split, [])
        dbs = {r["db_id"] for r in records}
        print(f"  {split:5s} {len(records):5d} examples  {len(dbs):4d} databases"
              f"  empty gold: {empty_gold[split]}")

    # Databases must not appear in two splits, or validation is measuring
    # questions about databases the model trained on.
    per_split = defaultdict(set)
    for db, split in assignment.items():
        per_split[split].add(db)
    overlap = (per_split["sft"] & per_split["val"]) | (per_split["sft"] & per_split["grpo"])
    print(f"  database overlap between splits: {len(overlap)}"
          f"{'  <- BUG' if overlap else '  (disjoint)'}")

    if dropped:
        print("\n=== dropped ===")
        for reason, count in dropped.most_common():
            print(f"  {reason:32s} {count}")

    print("\n=== token lengths (Qwen2.5 tokenizer, full prompt + answer) ===")
    # Only used for the length report below. Every Qwen2.5 variant we use
    # shares this vocabulary, so the numbers hold across bases.
    tokenizer = build_tokenizer(TOKENIZER_FOR_REPORT, chat=True)
    for split in ("sft", "grpo", "val"):
        records = buckets.get(split, [])
        if not records:
            continue
        # Render to text first, then tokenize. apply_chat_template(tokenize=True)
        # returns a dict in transformers 5.x, so len() on it counts *keys* and
        # reports every example as 2 tokens long.
        lengths = sorted(
            len(tokenizer(tokenizer.apply_chat_template(r["messages"], tokenize=False))["input_ids"])
            for r in records
        )
        over = sum(length > MAX_TOKENS for length in lengths)
        print(f"  {split:5s} median {lengths[len(lengths) // 2]:5d}"
              f"  p95 {lengths[int(len(lengths) * 0.95)]:5d}"
              f"  max {lengths[-1]:5d}"
              f"  over {MAX_TOKENS}: {over}"
              f"{'  <- these would be silently truncated' if over else ''}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()
    build(args.root, args.out, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
