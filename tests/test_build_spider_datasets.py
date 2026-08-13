"""Tests for the Spider dataset builder.

The split logic is the part worth pinning. If a database appears in both the
training and validation splits, validation measures "new question about a
database you trained on" — easier than the benchmark, and it would read as
progress that does not exist.
"""

from __future__ import annotations

from collections import Counter, defaultdict

from sqlrl.data_prep.build_spider_datasets import GRPO_TARGET, VAL_TARGET, _assign_splits


def make_rows(per_db: dict[str, int]) -> list[dict]:
    return [{"db_id": db} for db, n in per_db.items() for _ in range(n)]


def fake_corpus(n_dbs: int = 140, per_db: int = 50) -> list[dict]:
    return make_rows({f"db{i:03d}": per_db for i in range(n_dbs)})


def split_counts(rows: list[dict], assignment: dict[str, str]) -> Counter:
    return Counter(assignment[row["db_id"]] for row in rows)


def test_every_database_gets_exactly_one_split():
    rows = fake_corpus()
    assignment = _assign_splits(rows, seed=3407)
    by_split = defaultdict(set)
    for db, split in assignment.items():
        by_split[split].add(db)

    assert by_split["sft"] & by_split["val"] == set()
    assert by_split["sft"] & by_split["grpo"] == set()
    assert by_split["val"] & by_split["grpo"] == set()


def test_all_databases_are_assigned():
    rows = fake_corpus()
    assignment = _assign_splits(rows, seed=3407)
    assert set(assignment) == {row["db_id"] for row in rows}


def test_split_is_deterministic_for_a_seed():
    rows = fake_corpus()
    assert _assign_splits(rows, seed=3407) == _assign_splits(rows, seed=3407)


def test_different_seeds_give_different_splits():
    rows = fake_corpus()
    assert _assign_splits(rows, seed=1) != _assign_splits(rows, seed=2)


def test_small_splits_reach_their_targets():
    rows = fake_corpus()
    counts = split_counts(rows, _assign_splits(rows, seed=3407))
    # Filled first, so they land at or just past target rather than short.
    assert counts["val"] >= VAL_TARGET
    assert counts["grpo"] >= GRPO_TARGET
    # And SFT still gets the bulk of the data.
    assert counts["sft"] > counts["val"] + counts["grpo"]


def test_uneven_database_sizes_do_not_break_assignment():
    # Real Spider databases range from a handful of questions to hundreds.
    rows = make_rows({f"db{i:03d}": (i % 40) + 1 for i in range(140)})
    assignment = _assign_splits(rows, seed=3407)
    counts = split_counts(rows, assignment)
    assert set(assignment) == {row["db_id"] for row in rows}
    assert sum(counts.values()) == len(rows)


def test_a_database_is_never_split_across_two_buckets():
    rows = fake_corpus(n_dbs=20, per_db=100)
    assignment = _assign_splits(rows, seed=11)
    # One split per db_id is the invariant; assignment being a dict enforces it,
    # so assert the consequence: every row of a db lands in the same bucket.
    seen: dict[str, str] = {}
    for row in rows:
        split = assignment[row["db_id"]]
        assert seen.setdefault(row["db_id"], split) == split
