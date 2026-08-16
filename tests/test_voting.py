"""Tests for sqlrl.eval.voting.

No GPU, no model: ``Candidate``s are built directly with hand-written rows,
the same pattern tests/test_retry.py uses for ``Attempt``s. None of the cases
below need an actual database -- ``compare()`` only ever looks at the rows
already attached to each ``Candidate`` -- so there is no SQLite fixture here.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from sqlrl.eval.executor import run
from sqlrl.eval.run_eval import Prediction, RunRecord, VoteRecord
from sqlrl.eval.spider import Example
from sqlrl.eval.voting import (
    Ballot,
    Candidate,
    cluster,
    cluster_stats,
    oracle_at,
    select,
    vote_at,
)


@pytest.fixture(scope="module")
def db(tmp_path_factory) -> str:
    """A real database, needed only by the round-trip tests -- hydration
    re-executes SQL, so those cannot work on hand-written rows.
    """
    path = tmp_path_factory.mktemp("dbs") / "voting.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE people (id INTEGER, name TEXT, age INTEGER);
        INSERT INTO people VALUES (1, 'ada', 36), (2, 'grace', 45);
        """
    )
    conn.commit()
    conn.close()
    return str(path)


def _cand(sql: str) -> Candidate:
    """An unhydrated candidate, as load_votes produces one."""
    return Candidate(raw=f"<answer>{sql}</answer>", sql=sql, status="unknown", rows=[])


def _executed(sql: str, db_path: str) -> Candidate:
    result = run(sql, db_path, timeout=30.0)
    rows = [list(row) for row in result.rows] if result.status == "ok" else []
    return Candidate(raw=f"<answer>{sql}</answer>", sql=sql,
                     status=result.status, rows=rows)


def _ok(rows: list) -> Candidate:
    """An "ok" candidate with the given rows. ``raw``/``sql`` never affect
    clustering except through ``requires_order``, and none of these rows come
    from a query with an ORDER BY, so a fixed unordered stand-in is enough.
    """
    return Candidate(raw="<answer>SELECT 1</answer>", sql="SELECT 1", status="ok", rows=rows)


def _err() -> Candidate:
    return Candidate(raw="<answer>bogus</answer>", sql="bogus", status="error", rows=[])


# --------------------------------------------------------------------------
# cluster
# --------------------------------------------------------------------------


def test_cluster_groups_identical_rows_and_separates_different():
    same_a = _ok([["ada", 36]])
    same_b = _ok([["ada", 36]])
    different = _ok([["grace", 45]])

    clusters = cluster([same_a, same_b, different])

    assert clusters == [[0, 1], [2]]


def test_cluster_groups_rows_that_differ_only_in_column_order():
    # compare() runs a column-permutation search, so these two rows are the
    # same answer with the columns presented in a different order. A hash of
    # the row tuples would treat them as different result sets -- exactly the
    # false split this module exists to avoid.
    same_columns_swapped_a = _ok([["ada", 36]])
    same_columns_swapped_b = _ok([[36, "ada"]])

    clusters = cluster([same_columns_swapped_a, same_columns_swapped_b])

    assert clusters == [[0, 1]]


def test_cluster_excludes_non_ok_candidates_entirely():
    candidates = [_ok([["ada", 36]]), _err(), _ok([["ada", 36]])]

    clusters = cluster(candidates)

    assert clusters == [[0, 2]]
    assert all(1 not in indices for indices in clusters)


def test_cluster_orders_largest_first_ties_broken_by_smallest_index():
    a, b, c = _ok([["a"]]), _ok([["b"]]), _ok([["c"]])
    # index: 0=a 1=b 2=a 3=c 4=b 5=c 6=c
    # clusters as founded: A={0,2} (size 2), B={1,4} (size 2), C={3,5,6} (size 3)
    candidates = [a, b, a, c, b, c, c]

    clusters = cluster(candidates)

    # C is largest and sorts first; A and B tie at size 2, so A (founded at
    # the smaller index, 0) sorts before B (founded at 1).
    assert clusters == [[3, 5, 6], [0, 2], [1, 4]]


# --------------------------------------------------------------------------
# select
# --------------------------------------------------------------------------


def test_select_picks_the_majority_clusters_representative():
    majority_rep = _ok([["ada", 36]])
    minority = _ok([["grace", 45]])
    majority_other = _ok([["ada", 36]])

    winner = select([majority_rep, minority, majority_other])

    assert winner == 0  # majority_rep founded the size-2 cluster


def test_select_empty_result_trap():
    # 3 candidates return [] for different (wrong) reasons, 2 return an
    # identical non-empty answer. Empty results all compare equal to each
    # other regardless of why they are empty, so without the guard the empty
    # bloc outvotes the real answer 3-2.
    empties = [_ok([]) for _ in range(3)]
    nonempty = [_ok([["ada", 36]]), _ok([["ada", 36]])]
    candidates = [*empties, *nonempty]

    # demote_empty=True: the non-empty answer wins despite being outnumbered.
    assert select(candidates, demote_empty=True) == 3
    # demote_empty=False: the larger, empty cluster wins the vote outright.
    assert select(candidates, demote_empty=False) == 0


def test_select_falls_back_to_the_empty_cluster_when_everything_is_empty():
    # Nothing else to pick -- every ok candidate returned zero rows.
    candidates = [_ok([]), _ok([]), _ok([])]

    assert select(candidates, demote_empty=True) == 0
    assert select(candidates, demote_empty=False) == 0


def test_select_returns_zero_when_nothing_executed_ok():
    candidates = [_err(), _err(), _err()]

    assert select(candidates) == 0


# --------------------------------------------------------------------------
# vote_at
# --------------------------------------------------------------------------


def _ballot(candidates: list[Candidate], gold_sql: str = "SELECT 1") -> Ballot:
    return Ballot(index=0, db_id="db", question="q", gold_sql=gold_sql, candidates=candidates)


def test_vote_at_one_returns_exactly_the_greedy_candidate():
    greedy = _ok([["ada", 36]])
    ballot = _ballot([greedy, _ok([["grace", 45]]), _err()])

    assert vote_at(ballot, 1) is greedy
    assert vote_at(ballot, 1, demote_empty=False) is greedy


def test_vote_at_rejects_out_of_range_k():
    ballot = _ballot([_ok([["a"]]), _ok([["a"]])])

    with pytest.raises(ValueError):
        vote_at(ballot, 0)
    with pytest.raises(ValueError):
        vote_at(ballot, 3)


def test_increasing_k_can_change_the_winner():
    # index: 0=a 1=b 2=b 3=a 4=a. At k=3, {b1,b2} outnumbers {a0} -- b wins.
    # At k=5, {a0,a3,a4} outnumbers {b1,b2} -- a wins. Proves the prefix
    # logic is live, not just always returning the first or largest cluster.
    a_row, b_row = [["ada", 36]], [["grace", 45]]
    candidates = [_ok(a_row), _ok(b_row), _ok(b_row), _ok(a_row), _ok(a_row)]
    ballot = _ballot(candidates)

    winner_at_3 = vote_at(ballot, 3)
    winner_at_5 = vote_at(ballot, 5)

    assert winner_at_3.rows == b_row
    assert winner_at_5.rows == a_row


# --------------------------------------------------------------------------
# oracle_at
# --------------------------------------------------------------------------


def test_oracle_at_true_when_any_of_first_k_matches_gold_and_monotonic():
    gold_rows = [["ada", 36]]
    candidates = [
        _ok([["grace", 45]]),  # wrong
        _err(),  # cannot vote or match
        _ok([["ada", 36]]),  # matches gold, at index 2
    ]
    ballot = _ballot(candidates)

    assert oracle_at(ballot, 1, gold_rows, False) is False
    assert oracle_at(ballot, 2, gold_rows, False) is False
    assert oracle_at(ballot, 3, gold_rows, False) is True

    results = [oracle_at(ballot, k, gold_rows, False) for k in range(1, 4)]
    assert results == sorted(results)  # monotonic: never flips back to False


# --------------------------------------------------------------------------
# cluster_stats
# --------------------------------------------------------------------------


def test_cluster_stats_reports_distinct_clusters_and_largest_size():
    candidates = [
        _ok([["ada", 36]]),
        _ok([["grace", 45]]),
        _ok([["ada", 36]]),
        _err(),
    ]
    ballot = _ballot(candidates)

    n_clusters, largest = cluster_stats(ballot, 4)

    assert n_clusters == 2  # {ada-cluster}, {grace-cluster}; the error is excluded
    assert largest == 2  # the ada cluster has 2 members


# --------------------------------------------------------------------------
# VoteRecord.at_k
# --------------------------------------------------------------------------


def test_vote_record_at_k_one_matches_a_plain_greedy_record():
    greedy_raw = "<answer>SELECT name FROM people</answer>"
    ballot = Ballot(
        index=0, db_id="db", question="how many?", gold_sql="SELECT name FROM people",
        candidates=[
            Candidate(raw=greedy_raw, sql="SELECT name FROM people", status="ok",
                      rows=[["ada"], ["grace"]]),
            _ok([["grace"]]),
            _ok([["grace"]]),
        ],
    )
    vote_record = VoteRecord(
        model="m", split="test", n=1, device="cpu", dtype="float32",
        max_new_tokens=64, decoding="greedy+2@T0.8", seed=0, git_commit="deadbeef",
        generated_at="2026-01-01T00:00:00+00:00", generation_seconds=0.0,
        samples=3, temperature=0.8, top_p=0.95, ballots=[ballot],
    )

    plain_record = RunRecord(
        model="m", split="test", n=1, device="cpu", dtype="float32",
        max_new_tokens=64, decoding="greedy", seed=0, git_commit="deadbeef",
        generated_at="2026-01-01T00:00:00+00:00", generation_seconds=0.0,
        predictions=[
            Prediction(
                index=0, db_id="db", question="how many?",
                gold_sql="SELECT name FROM people", raw=greedy_raw,
                pred_sql="SELECT name FROM people",
            ),
        ],
    )

    assert vote_record.at_k(1).predictions == plain_record.predictions


# --------------------------------------------------------------------------
# save / load / hydrate round trip
# --------------------------------------------------------------------------


def test_rows_are_not_persisted_and_hydrate_restores_them(tmp_path, db):
    """Rows are recomputed from the database, never read back from the file.

    A single Spider query returns 26,112 rows and the greedy pass alone is
    268,891 cells, so persisting them would put ~26 MB per run into a repo
    whose results/ is already 74 MB. Re-executing is ~40s of CPU. The risk
    that buys is a hydration that misaligns candidates with their examples,
    which would corrupt every vote without saying so -- hence this test.
    """
    from sqlrl.eval.run_eval import hydrate_votes, load_votes, save_votes

    gold = "SELECT name FROM people"
    ballots = [
        Ballot(index=0, db_id="db", question="q0", gold_sql=gold, candidates=[
            _cand("SELECT name FROM people"), _cand("SELECT nope FROM people"),
        ]),
        Ballot(index=1, db_id="db", question="q1", gold_sql=gold, candidates=[
            _cand("SELECT age FROM people"), _cand("SELECT name FROM people"),
        ]),
    ]
    for ballot in ballots:
        for i, candidate in enumerate(ballot.candidates):
            ballot.candidates[i] = _executed(candidate.sql, db)

    record = VoteRecord(
        model="m", split="test", n=2, device="cpu", dtype="float32",
        max_new_tokens=64, decoding="greedy+1@T0.8", seed=0, git_commit="deadbeef",
        generated_at="2026-01-01T00:00:00+00:00", generation_seconds=0.0,
        samples=2, temperature=0.8, top_p=0.95, ballots=ballots,
    )
    before = [[(c.status, c.rows) for c in b.candidates] for b in record.ballots]

    path = tmp_path / "vote2.json"
    save_votes(record, path)

    payload = json.loads(path.read_text())
    for ballot in payload["ballots"]:
        for candidate in ballot["candidates"]:
            assert "rows" not in candidate and "status" not in candidate

    loaded = load_votes(path)
    assert all(
        c.status == "unknown" and c.rows == []
        for b in loaded.ballots for c in b.candidates
    )

    examples = [
        Example(db_id="db", db_path=Path(db), question=f"q{i}", gold_sql=gold)
        for i in range(2)
    ]
    hydrate_votes(loaded, examples, timeout=30.0)

    after = [[(c.status, c.rows) for c in b.candidates] for b in loaded.ballots]
    assert after == before
    assert loaded.at_k(2).predictions == record.at_k(2).predictions


def test_hydrate_refuses_a_stale_saved_run(tmp_path, db):
    """A ballot that no longer lines up with the benchmark must raise, not
    quietly vote against someone else's question."""
    from sqlrl.eval.run_eval import hydrate_votes

    record = VoteRecord(
        model="m", split="test", n=1, device="cpu", dtype="float32",
        max_new_tokens=64, decoding="greedy", seed=0, git_commit="deadbeef",
        generated_at="2026-01-01T00:00:00+00:00", generation_seconds=0.0,
        samples=1, temperature=0.8, top_p=0.95,
        ballots=[Ballot(index=0, db_id="db", question="the old question",
                        gold_sql="SELECT name FROM people",
                        candidates=[_cand("SELECT name FROM people")])],
    )
    examples = [Example(db_id="db", db_path=Path(db), question="a different question",
                        gold_sql="SELECT name FROM people")]
    with pytest.raises(AssertionError):
        hydrate_votes(record, examples, timeout=30.0)
