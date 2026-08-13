"""Tests for sqlrl.training.rewards.

A reward that quietly gets a case wrong does not crash training -- it produces
a nice-looking curve while teaching the model something other than SQL. Every
test here pins one way that could happen: the tier ladder itself, the
``WHERE 1=0`` / empty-gold hack, row-cap truncation masquerading as a match,
gold caching, and the calling convention TRL actually uses.
"""

from __future__ import annotations

import sqlite3

import pytest

from sqlrl.eval.executor import run
from sqlrl.training.rewards import OUTCOMES, SQLReward, Tiers, drop_empty_gold


@pytest.fixture(scope="module")
def db(tmp_path_factory) -> str:
    path = tmp_path_factory.mktemp("dbs") / "rewards.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE people (id INTEGER, name TEXT, age INTEGER);
        INSERT INTO people VALUES
            (1, 'ada',   36),
            (2, 'grace', 45),
            (3, 'alan',  41),
            (4, 'ada',   29);
        """
    )
    conn.commit()
    conn.close()
    return str(path)


#: age > 40 keeps grace and alan -- two rows, no ORDER BY, so row order does
#: not matter and both scoreable and comparison edge cases have something to
#: bite on.
GOLD_NONEMPTY = "SELECT name FROM people WHERE age > 40"
#: Nothing in the fixture is older than 200, so this gold query returns nothing.
GOLD_EMPTY = "SELECT name FROM people WHERE age > 200"


# --------------------------------------------------------------------------
# the ladder, end to end
# --------------------------------------------------------------------------


def test_ladder_gold_query_matches(db):
    reward = SQLReward()
    pred = f"<answer>{GOLD_NONEMPTY}</answer>"
    assert reward.score(pred, db, GOLD_NONEMPTY) == (2.0, "match")


def test_ladder_runs_cleanly_but_wrong_rows(db):
    reward = SQLReward()
    pred = "<answer>SELECT name FROM people WHERE age < 40</answer>"
    assert reward.score(pred, db, GOLD_NONEMPTY) == (0.5, "wrong_rows")


def test_ladder_parses_but_database_rejects_it(db):
    # Valid SQL syntax, hallucinated column -- sqlite, not sqlglot, catches it.
    reward = SQLReward()
    pred = "<answer>SELECT salary FROM people</answer>"
    assert reward.score(pred, db, GOLD_NONEMPTY) == (0.2, "db_error")


def test_ladder_no_sql_in_completion(db):
    reward = SQLReward()
    pred = "I think the answer is probably yes."
    assert reward.score(pred, db, GOLD_NONEMPTY) == (0.0, "no_sql")


def test_ladder_sql_shaped_text_that_will_not_parse(db):
    # Past MAX_JOINS, so parse_sql refuses it outright rather than hanging.
    joins = " ".join(f"JOIN t{i}" for i in range(1, 15))
    pred = f"<answer>SELECT a FROM t0 {joins}</answer>"
    reward = SQLReward()
    assert reward.score(pred, db, GOLD_NONEMPTY) == (0.0, "unparseable")


# --------------------------------------------------------------------------
# Tiers
# --------------------------------------------------------------------------


def test_tiers_rejects_a_non_monotone_ladder():
    with pytest.raises(ValueError):
        Tiers(nothing=0.0, parses=0.3, executes=0.2, matches=2.0)


def test_tiers_accepts_a_custom_monotone_ladder_and_reward_uses_it(db):
    tiers = Tiers(nothing=-1.0, parses=0.0, executes=1.0, matches=5.0)
    reward = SQLReward(tiers)
    assert reward.score("no query here", db, GOLD_NONEMPTY) == (-1.0, "no_sql")
    pred = f"<answer>{GOLD_NONEMPTY}</answer>"
    assert reward.score(pred, db, GOLD_NONEMPTY) == (5.0, "match")


# --------------------------------------------------------------------------
# empty gold -- the important one
# --------------------------------------------------------------------------


def test_empty_gold_defaults_to_the_executes_tier_not_a_match(db):
    reward = SQLReward()
    pred = "<answer>SELECT name FROM people WHERE age > 300</answer>"
    assert reward.score(pred, db, GOLD_EMPTY) == (0.5, "empty_gold")


def test_empty_gold_pays_the_match_tier_when_opted_in(db):
    reward = SQLReward(pay_for_empty_gold=True)
    pred = "<answer>SELECT name FROM people WHERE age > 300</answer>"
    assert reward.score(pred, db, GOLD_EMPTY) == (2.0, "match")


# --------------------------------------------------------------------------
# the WHERE 1=0 hack
# --------------------------------------------------------------------------


def test_empty_result_against_nonempty_gold_never_reaches_match(db):
    reward = SQLReward()
    pred = "<answer>SELECT name FROM people WHERE 1=0</answer>"
    assert reward.score(pred, db, GOLD_NONEMPTY) == (0.5, "wrong_rows")


# --------------------------------------------------------------------------
# gold caching
# --------------------------------------------------------------------------


def test_gold_is_cached_by_db_and_query(db):
    reward = SQLReward()
    first = reward.gold(db, GOLD_NONEMPTY)
    second = reward.gold(db, GOLD_NONEMPTY)
    assert first is second
    assert len(reward._gold) == 1


def test_different_gold_queries_get_different_cache_entries(db):
    reward = SQLReward()
    reward.gold(db, GOLD_NONEMPTY)
    reward.gold(db, "SELECT name FROM people WHERE age < 40")
    assert len(reward._gold) == 2


def test_the_same_query_against_two_databases_is_cached_separately(db, tmp_path):
    # The cache key has to include the database. Spider reuses question and
    # query text across databases, so keying on SQL alone would answer one
    # database's question with another's rows -- and every rollout for that
    # prompt would be scored against the wrong gold.
    other = tmp_path / "other.sqlite"
    conn = sqlite3.connect(other)
    conn.executescript(
        "CREATE TABLE people (id INTEGER, name TEXT, age INTEGER);"
        "INSERT INTO people VALUES (1, 'zed', 99);"
    )
    conn.commit()
    conn.close()

    reward = SQLReward()
    first = reward.gold(db, GOLD_NONEMPTY)
    second = reward.gold(str(other), GOLD_NONEMPTY)
    assert len(reward._gold) == 2
    assert first.rows != second.rows


def test_warm_populates_the_cache_and_returns_the_instance(db):
    reward = SQLReward()
    rows = [
        {"db_path": db, "gold_sql": GOLD_NONEMPTY},
        {"db_path": db, "gold_sql": "SELECT name FROM people WHERE age < 40"},
    ]
    result = reward.warm(rows)
    assert result is reward
    assert len(reward._gold) == 2


# --------------------------------------------------------------------------
# warm() fails loudly on bad inputs
# --------------------------------------------------------------------------


def test_warm_raises_file_not_found_for_a_missing_database(tmp_path):
    reward = SQLReward()
    rows = [{"db_path": str(tmp_path / "nope.sqlite"), "gold_sql": "SELECT 1"}]
    with pytest.raises(FileNotFoundError):
        reward.warm(rows)


def test_warm_raises_value_error_for_a_gold_query_that_does_not_execute(db):
    rows = [{"db_path": db, "gold_sql": "SELECT nope FROM nowhere"}]
    with pytest.raises(ValueError):
        SQLReward().warm(rows)


# --------------------------------------------------------------------------
# score() never raises on model output
# --------------------------------------------------------------------------

PATHOLOGICAL = [
    pytest.param("z" * 10_000, id="10kb-of-text"),
    pytest.param(
        "<answer>SELECT a FROM t0 "
        + " ".join(f"JOIN t{i}" for i in range(1, 31))
        + "</answer>",
        id="30-on-less-joins",
    ),
    pytest.param("<answer>SELECT 'héllo wörld ☃ 日本語'</answer>", id="unicode"),
    pytest.param("\x00\x01<answer>SELECT 1\x07</answer>\x1b", id="control-characters"),
    pytest.param("<answer>", id="lone-answer-tag"),
    pytest.param("-- just a comment, nothing else", id="sql-comment-only"),
    pytest.param("DROP TABLE people", id="drop-table"),
    pytest.param(
        "<answer>" + "SELECT * FROM (" * 50 + "SELECT 1" + ")" * 50 + "</answer>",
        id="nested-subquery-bomb",
    ),
]


@pytest.mark.parametrize("completion", PATHOLOGICAL)
def test_score_never_raises_and_database_stays_intact(db, completion):
    reward = SQLReward()
    before = run("SELECT * FROM people ORDER BY id", db).rows
    score, _outcome = reward.score(completion, db, GOLD_NONEMPTY)
    assert isinstance(score, float)
    assert reward.tiers.nothing <= score <= reward.tiers.matches
    after = run("SELECT * FROM people ORDER BY id", db).rows
    assert after == before


# --------------------------------------------------------------------------
# ...but FileNotFoundError is not model output, so it propagates
# --------------------------------------------------------------------------


def test_score_raises_file_not_found_for_a_missing_database(tmp_path):
    # Our bug, not the model's -- swallowing it would score every rollout zero
    # while training continued, which is worse than crashing.
    reward = SQLReward()
    missing = str(tmp_path / "nope.sqlite")
    with pytest.raises(FileNotFoundError):
        reward.score("<answer>SELECT 1</answer>", missing, "SELECT 1")


# --------------------------------------------------------------------------
# the row cap
# --------------------------------------------------------------------------


def test_too_many_rows_never_reaches_match(db):
    reward = SQLReward()
    pred = "<answer>SELECT name FROM people</answer>"  # 4 rows, gold has 2
    assert reward.score(pred, db, GOLD_NONEMPTY) == (0.5, "too_many_rows")


def test_truncated_prefix_equal_to_gold_is_still_too_many_rows(db):
    # The truncated result is exactly gold's two rows plus one more ('zzz').
    # If truncation were ever compared instead of gated, this would look like
    # a match -- it must not.
    reward = SQLReward()
    pred = (
        "<answer>"
        f"{GOLD_NONEMPTY} UNION ALL SELECT 'zzz' UNION ALL SELECT 'yyy'"
        "</answer>"
    )
    assert reward.score(pred, db, GOLD_NONEMPTY) == (0.5, "too_many_rows")


# --------------------------------------------------------------------------
# a query that never finishes
# --------------------------------------------------------------------------


def test_timeout_scores_the_parse_tier_not_the_executes_tier(db):
    # Deliberately stricter than the evaluator, whose execution-rate metric
    # counts a timeout as "executed". A query that never finished says nothing
    # about whether it was right, and paying it the executes tier would reward
    # rollouts for burning the timeout budget.
    reward = SQLReward(timeout=0.25)
    pred = (
        "<answer>WITH RECURSIVE c(x) AS ("
        "SELECT 1 UNION ALL SELECT x + 1 FROM c"
        ") SELECT count(*) FROM c</answer>"
    )
    assert reward.score(pred, db, GOLD_NONEMPTY) == (0.2, "timeout")


# --------------------------------------------------------------------------
# row ordering follows the gold query
# --------------------------------------------------------------------------


def test_reordered_rows_do_not_match_when_gold_has_order_by(db):
    gold_sql = "SELECT name FROM people ORDER BY age DESC"
    pred = "<answer>SELECT name FROM people ORDER BY age ASC</answer>"
    reward = SQLReward()
    assert reward.score(pred, db, gold_sql) == (0.5, "wrong_rows")


def test_reordered_rows_match_when_gold_has_no_order_by(db):
    gold_sql = "SELECT name FROM people"
    pred = "<answer>SELECT name FROM people ORDER BY age ASC</answer>"
    reward = SQLReward()
    assert reward.score(pred, db, gold_sql) == (2.0, "match")


# --------------------------------------------------------------------------
# multiset semantics, matching the evaluator
# --------------------------------------------------------------------------


def test_distinct_does_not_match_plain_select_with_duplicates(db):
    # 'ada' appears twice, so SELECT name has 4 rows and SELECT DISTINCT name
    # has 3. Set semantics would call these the same answer; multiset must not.
    gold_sql = "SELECT name FROM people"
    pred = "<answer>SELECT DISTINCT name FROM people</answer>"
    reward = SQLReward()
    assert reward.score(pred, db, gold_sql) == (0.5, "wrong_rows")


# --------------------------------------------------------------------------
# the TRL calling convention
# --------------------------------------------------------------------------


def test_call_accepts_message_dict_completions(db):
    reward = SQLReward()
    completions = [
        [{"role": "assistant", "content": f"<answer>{GOLD_NONEMPTY}</answer>"}],
        [{"role": "assistant", "content": "no query at all"}],
    ]
    scores = reward(
        completions=completions,
        db_path=[db, db],
        gold_sql=[GOLD_NONEMPTY, GOLD_NONEMPTY],
    )
    assert scores == [2.0, 0.0]


def test_call_accepts_plain_string_completions(db):
    reward = SQLReward()
    completions = [f"<answer>{GOLD_NONEMPTY}</answer>", "no query at all"]
    scores = reward(
        completions=completions,
        db_path=[db, db],
        gold_sql=[GOLD_NONEMPTY, GOLD_NONEMPTY],
    )
    assert scores == [2.0, 0.0]


def test_call_returns_one_score_per_completion_in_order(db):
    reward = SQLReward()
    completions = [
        f"<answer>{GOLD_NONEMPTY}</answer>",
        "<answer>SELECT salary FROM people</answer>",
        "no query at all",
    ]
    scores = reward(
        completions=completions,
        db_path=[db, db, db],
        gold_sql=[GOLD_NONEMPTY, GOLD_NONEMPTY, GOLD_NONEMPTY],
    )
    assert scores == [2.0, 0.2, 0.0]


def test_call_tolerates_extra_trl_keyword_arguments(db):
    # TRL passes trainer_state and every other dataset column expanded to one
    # entry per generation. A reward that does not accept **kwargs dies here.
    reward = SQLReward()
    scores = reward(
        completions=[f"<answer>{GOLD_NONEMPTY}</answer>"],
        db_path=[db],
        gold_sql=[GOLD_NONEMPTY],
        trainer_state=object(),
        question=["irrelevant dataset column"],
        prompts=[[{"role": "user", "content": "..."}]],
    )
    assert scores == [2.0]


def test_call_raises_on_mismatched_input_lengths(db):
    reward = SQLReward()
    with pytest.raises(ValueError):
        reward(
            completions=[f"<answer>{GOLD_NONEMPTY}</answer>", "another one"],
            db_path=[db],
            gold_sql=[GOLD_NONEMPTY],
        )


def test_reward_has_the_name_trl_reads_for_logging():
    # TRL labels the reward in its logs with reward_funcs[i].__name__ and dies
    # on an AttributeError without it.
    assert SQLReward().__name__ == "sql_reward"


# --------------------------------------------------------------------------
# report()
# --------------------------------------------------------------------------


def test_report_fractions_sum_to_one(db):
    reward = SQLReward()
    reward.score(f"<answer>{GOLD_NONEMPTY}</answer>", db, GOLD_NONEMPTY)
    reward.score("<answer>SELECT salary FROM people</answer>", db, GOLD_NONEMPTY)
    reward.score("no query here", db, GOLD_NONEMPTY)
    report = reward.report()
    total = sum(report[f"outcome/{name}"] for name in OUTCOMES)
    assert total == pytest.approx(1.0)


def test_report_resets_the_counters(db):
    reward = SQLReward()
    reward.score("no query here", db, GOLD_NONEMPTY)
    first = reward.report()
    assert first
    second = reward.report()
    assert second == {}


def test_report_empty_match_frac_flags_a_zero_row_match(db):
    reward = SQLReward(pay_for_empty_gold=True)
    pred = "<answer>SELECT name FROM people WHERE age > 300</answer>"
    reward.score(pred, db, GOLD_EMPTY)
    report = reward.report()
    assert report["outcome/empty_match_frac"] == pytest.approx(1.0)


def test_report_empty_match_frac_zero_without_one(db):
    reward = SQLReward()
    reward.score(f"<answer>{GOLD_NONEMPTY}</answer>", db, GOLD_NONEMPTY)
    report = reward.report()
    assert report["outcome/empty_match_frac"] == 0.0


# --------------------------------------------------------------------------
# drop_empty_gold
# --------------------------------------------------------------------------


def test_drop_empty_gold_removes_only_the_empty_ones(db):
    rows = [
        {"db_path": db, "gold_sql": GOLD_NONEMPTY},
        {"db_path": db, "gold_sql": GOLD_EMPTY},
        {"db_path": db, "gold_sql": "SELECT name FROM people WHERE age > 28"},
    ]
    kept = drop_empty_gold(rows)
    assert kept == [rows[0], rows[2]]


def test_drop_empty_gold_raises_rather_than_dropping_a_broken_gold(db):
    # "the answer is empty" and "we could not get the answer" are different
    # facts. Treating the second as the first would let a wrong --root or a
    # partial download silently shrink the training set, and a GRPO run on
    # half a dataset looks exactly like one on all of it.
    rows = [{"db_path": db, "gold_sql": "SELECT nope FROM nowhere"}]
    with pytest.raises(ValueError):
        drop_empty_gold(rows)
