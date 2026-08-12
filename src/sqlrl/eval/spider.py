"""Fetch the Spider benchmark, verify it, and check it for contamination.

    uv run python -m sqlrl.eval.spider --split test
    uv run python -m sqlrl.eval.spider --split dev

**Read this before trusting a Spider number from this project.**

The blueprint assumed Spider *dev* was legitimately held out, on the grounds
that ``b-mc2/sql-create-context`` derives from WikiSQL and Spider *train*. That
assumption is false, and this module is what caught it: **562 of the 1,034
Spider dev questions appear verbatim in the v0 training data**, gold SQL and
all -- including the very first one, "How many singers do we have?". Reporting
v0's score on full Spider dev would have been reporting how well it memorised.

So the primary benchmark here is **Spider test**: 2,147 questions over 40
databases, disjoint from both dev and train, and 99.3% absent from the v0
training set. Dev is kept as a secondary, reported in two slices -- the full
1,034 for rough comparability with published numbers, and the clean 472. The
gap between those two slices *is* the memorisation, measured rather than
assumed.

Provenance, since the two splits come from different places:

* Dev questions and gold SQL come from ``xlangai/spider`` on the Hub, the
  authoritative release.
* The SQLite databases are not in that dataset (parquet only) and the official
  zip lives on Google Drive, which cannot be fetched reproducibly. They come
  from a Hub mirror of that zip -- as does the test split, which the Hub
  dataset does not carry at all.
* ``verify_mirror`` is what makes the mirror usable: it checks the mirror's
  ``dev.json`` against the official dataset row for row. They match exactly, so
  the mirror is a faithful copy of the official zip and its test split inherits
  that credibility. ``verify`` then executes every gold query against the
  mirrored databases, which no mismatched schema could survive.
"""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download

from sqlrl.eval.executor import run

__all__ = ["Example", "SPLITS", "ensure_data", "load_split"]

OFFICIAL_DATASET = "xlangai/spider"
TRAINING_DATASET = "b-mc2/sql-create-context"
MIRROR_REPO = "HAL-9001/spider-databases"
MIRROR_FILE = "spider_data.zip"

DEFAULT_ROOT = Path("data/spider")
SPLITS = ("dev", "test")

#: Directory and question file per split, inside the extracted zip.
_LAYOUT = {
    "dev": ("database", "dev.json"),
    "test": ("test_database", "test.json"),
}

_WANTED = (
    "spider_data/database/",
    "spider_data/test_database/",
    "spider_data/dev.json",
    "spider_data/test.json",
    "spider_data/tables.json",
    "spider_data/test_tables.json",
    "spider_data/train_spider.json",
    "spider_data/train_others.json",
)

_EXPECTED = {"dev": (1034, 20), "test": (2147, 40)}


@dataclass(frozen=True)
class Example:
    db_id: str
    db_path: Path
    question: str
    gold_sql: str
    #: True when this exact question appears in the v0 training data.
    contaminated: bool = False


# --------------------------------------------------------------------------
# fetching
# --------------------------------------------------------------------------


def ensure_data(root: Path = DEFAULT_ROOT) -> Path:
    """Download and extract the Spider archive if it is not already present."""
    root = Path(root)
    if (root / "spider_data" / "test.json").is_file():
        return root

    root.mkdir(parents=True, exist_ok=True)
    archive = hf_hub_download(MIRROR_REPO, MIRROR_FILE, repo_type="dataset")
    with zipfile.ZipFile(archive) as zf:
        members = [
            name
            for name in zf.namelist()
            # __MACOSX holds resource forks, not data.
            if name.startswith(_WANTED) and not name.startswith("__MACOSX")
        ]
        zf.extractall(root, members=members)
    return root


def _pairs(split: str, root: Path) -> list[dict]:
    """Question/gold pairs, from the most authoritative source for that split."""
    if split == "dev":
        # The official release. Verified equal to the mirror by verify_mirror.
        return [
            {"db_id": row["db_id"], "question": row["question"], "query": row["query"]}
            for row in load_dataset(OFFICIAL_DATASET, split="validation")
        ]
    _, questions = _LAYOUT[split]
    return json.loads((Path(root) / "spider_data" / questions).read_text())


def load_split(
    split: str = "test",
    root: Path = DEFAULT_ROOT,
    *,
    clean_only: bool = False,
) -> list[Example]:
    """Load a Spider split, with contaminated examples marked (or dropped)."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")

    root = ensure_data(root)
    databases = Path(root) / "spider_data" / _LAYOUT[split][0]
    contaminated = set(contaminated_indices(split, root))

    examples = [
        Example(
            db_id=row["db_id"],
            db_path=databases / row["db_id"] / f"{row['db_id']}.sqlite",
            question=row["question"],
            gold_sql=row["query"],
            contaminated=index in contaminated,
        )
        for index, row in enumerate(_pairs(split, root))
    ]
    return [ex for ex in examples if not ex.contaminated] if clean_only else examples


# --------------------------------------------------------------------------
# contamination
# --------------------------------------------------------------------------


def _norm_question(text: str) -> str:
    text = re.sub(r"[^a-z0-9 ]+", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def _norm_sql(sql: str) -> str:
    # Strip again after the semicolon: "... singer ;" leaves a trailing space.
    sql = re.sub(r"\s+", " ", sql.lower()).strip().rstrip(";").strip()
    return re.sub(r"\s*([(),])\s*", r"\1", sql)


def contaminated_indices(split: str, root: Path = DEFAULT_ROOT) -> list[int]:
    """Positions in ``split`` whose question appears in the v0 training data.

    Cached to disk: the answer is fixed by two published datasets, and loading
    78,577 training rows to re-derive it on every evaluation would be waste.
    """
    cache = Path(root) / f"contamination_{split}.json"
    if cache.is_file():
        return json.loads(cache.read_text())

    training = {
        _norm_question(question)
        for question in load_dataset(TRAINING_DATASET, split="train")["question"]
    }
    indices = [
        index
        for index, row in enumerate(_pairs(split, Path(root)))
        if _norm_question(row["question"]) in training
    ]
    cache.write_text(json.dumps(indices))
    return indices


# --------------------------------------------------------------------------
# verification
# --------------------------------------------------------------------------


def verify_mirror(root: Path = DEFAULT_ROOT) -> list[str]:
    """Is the mirror a faithful copy of the official Spider release?

    The test split has no authoritative source we can fetch, so its
    trustworthiness rests entirely on this: if the mirror reproduces the
    official dev split exactly, it is the official zip.
    """
    mirror = json.loads((Path(root) / "spider_data" / "dev.json").read_text())
    official = _pairs("dev", Path(root))

    key = lambda rows: [(r["db_id"], r["question"], r["query"]) for r in rows]  # noqa: E731
    if key(mirror) != key(official):
        return [
            f"mirror dev.json does not match {OFFICIAL_DATASET} "
            f"({len(mirror)} vs {len(official)} rows) -- do not trust the test split"
        ]
    return []


def verify(examples: list[Example], split: str, timeout: float = 60.0) -> list[str]:
    """Check the benchmark is intact. Returns problems; empty means fine."""
    problems: list[str] = []
    n_expected, db_expected = _EXPECTED[split]

    if len(examples) != n_expected:
        problems.append(f"expected {n_expected} {split} examples, got {len(examples)}")

    db_ids = {ex.db_id for ex in examples}
    if len(db_ids) != db_expected:
        problems.append(f"expected {db_expected} {split} databases, got {len(db_ids)}")

    missing = sorted({str(ex.db_path) for ex in examples if not ex.db_path.is_file()})
    if missing:
        problems.append(f"{len(missing)} database files missing, e.g. {missing[0]}")
        return problems

    failed, empty = [], 0
    for ex in examples:
        result = run(ex.gold_sql, ex.db_path, timeout=timeout)
        if not result.ok:
            failed.append(f"{ex.db_id}: {result.status}: {result.error}")
        elif not result.rows:
            empty += 1

    if failed:
        problems.append(
            f"{len(failed)} of {len(examples)} gold queries did not run "
            f"(first: {failed[0]})"
        )

    print(f"  gold queries executed:   {len(examples) - len(failed)}/{len(examples)}")
    print(
        f"  gold answers empty:      {empty} ({empty / len(examples):.1%})"
        "  <- ceiling on free points for returning nothing"
    )
    return problems


# --------------------------------------------------------------------------


def _report_contamination(examples: list[Example], split: str, root: Path) -> None:
    dirty = [ex for ex in examples if ex.contaminated]
    clean = [ex for ex in examples if not ex.contaminated]

    training = load_dataset(TRAINING_DATASET, split="train")
    train_sql = {_norm_sql(sql) for sql in training["answer"]}
    sql_overlap = {_norm_sql(ex.gold_sql) for ex in examples} & train_sql

    spider_train = json.loads(
        (Path(root) / "spider_data" / "train_spider.json").read_text()
    )
    train_dbs = {row["db_id"] for row in spider_train}

    print(f"\n  checked against {TRAINING_DATASET} ({len(training):,} examples)")
    print(f"  questions also in training data:  {len(dirty)} / {len(examples)}"
          f"  ({len(dirty) / len(examples):.1%})")
    print(f"  gold SQL also in training data:   {len(sql_overlap)} unique queries")
    print(f"  databases shared with Spider train: "
          f"{len({ex.db_id for ex in examples} & train_dbs)}")

    if dirty:
        print(f"\n  !! {len(dirty)} contaminated, e.g.:")
        for ex in dirty[:3]:
            print(f"       [{ex.db_id}] {ex.question}")
        print(f"\n  clean subset: {len(clean)} examples over "
              f"{len({ex.db_id for ex in clean})} databases")
        counts = Counter(ex.db_id for ex in clean).most_common(5)
        print("  largest clean databases: "
              + ", ".join(f"{db} ({n})" for db, n in counts))
    else:
        print("\n  Clean.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch and verify Spider.")
    parser.add_argument("--split", choices=SPLITS, default="test")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()

    title = f"Spider {args.split}"
    print(title)
    print("=" * 40)

    examples = load_split(args.split, args.root)
    print(f"  loaded {len(examples)} examples over "
          f"{len({ex.db_id for ex in examples})} databases")

    problems = verify_mirror(args.root)
    if problems:
        # Nothing below this line is meaningful if the source is not what it claims.
        print("\nPROBLEMS:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print(f"  mirror matches {OFFICIAL_DATASET} exactly on dev")

    problems += verify(examples, args.split)
    _report_contamination(examples, args.split, args.root)

    if problems:
        print("\nPROBLEMS -- do not trust numbers from this benchmark:")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print("\nBenchmark verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
