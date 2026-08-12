"""Tests for sqlrl.eval.executor.

Every case here is a way the evaluator could report a plausible wrong number.
"""

from __future__ import annotations

import math
import sqlite3

import pytest

from sqlrl.eval.executor import compare, parse_sql, requires_order, run


@pytest.fixture(scope="module")
def db(tmp_path_factory) -> str:
    path = tmp_path_factory.mktemp("dbs") / "test.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE people (id INTEGER, name TEXT, age INTEGER, score REAL);
        INSERT INTO people VALUES
            (1, 'ada',   36, 90.5),
            (2, 'grace', 45, 88.0),
            (3, 'alan',  41, 90.5),
            (4, 'ada',   29, NULL);

        CREATE TABLE pets (id INTEGER, owner_id INTEGER, name TEXT);
        INSERT INTO pets VALUES (1, 1, 'rex'), (2, 2, 'mia');

        CREATE TABLE junk (v TEXT);
        INSERT INTO junk VALUES (CAST(x'6162ff63' AS TEXT));
        """
    )
    conn.commit()
    conn.close()
    return str(path)


# --------------------------------------------------------------------------
# run()
# --------------------------------------------------------------------------


def test_run_returns_rows(db):
    result = run("SELECT name FROM people ORDER BY id", db)
    assert result.status == "ok"
    assert result.ok
    assert result.rows == [("ada",), ("grace",), ("alan",), ("ada",)]
    assert result.error is None


def test_syntax_error_is_data_not_an_exception(db):
    result = run("SELCT name FRM people", db)
    assert result.status == "error"
    assert result.rows == []
    assert result.error


def test_hallucinated_table_and_column(db):
    assert run("SELECT * FROM employees", db).status == "error"
    assert run("SELECT salary FROM people", db).status == "error"


@pytest.mark.parametrize("sql", ["", "   ", "-- nothing here"])
def test_non_queries_are_errors_not_empty_results(db, sql):
    # sqlite3 runs these happily and reports zero rows. Left as "ok" they would
    # compare equal to any empty gold result, scoring a point for generating
    # nothing at all.
    result = run(sql, db)
    assert result.status == "error"
    assert result.rows == []


def test_multiple_statements_are_rejected(db):
    # sqlite3 refuses more than one statement, which is what stops a smuggled
    # second query from running at all.
    assert run("SELECT 1; SELECT 2;", db).status == "error"


def test_trailing_semicolon_is_fine(db):
    assert run("SELECT count(*) FROM people;", db).rows == [(4,)]


def test_database_is_read_only(db):
    dropped = run("DROP TABLE people", db)
    assert dropped.status == "error"
    inserted = run("INSERT INTO people VALUES (9, 'x', 1, 1.0)", db)
    assert inserted.status == "error"
    # The benchmark survived. If this ever fails, every later score is garbage.
    assert run("SELECT count(*) FROM people", db).rows == [(4,)]


def test_timeout_stops_a_runaway_query(db):
    runaway = (
        "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) "
        "SELECT count(*) FROM c"
    )
    result = run(runaway, db, timeout=0.25)
    assert result.status == "timeout"
    assert result.rows == []


def test_row_cap_truncates_instead_of_exploding(db):
    result = run("SELECT a.id, b.id FROM people a, people b", db, max_rows=5)
    assert result.status == "too_many_rows"
    assert len(result.rows) == 5


def test_row_cap_not_tripped_when_result_fits(db):
    result = run("SELECT id FROM people", db, max_rows=4)
    assert result.status == "ok"
    assert len(result.rows) == 4


def test_invalid_utf8_text_is_readable(db):
    # Several Spider databases contain bytes that are not valid UTF-8. The
    # default text factory raises on them, which would look like a query error.
    result = run("SELECT v FROM junk", db)
    assert result.status == "ok"
    assert isinstance(result.rows[0][0], str)


def test_missing_database_raises(tmp_path):
    # Our bug, not the model's -- must not be scored as a wrong answer.
    with pytest.raises(FileNotFoundError):
        run("SELECT 1", tmp_path / "nope.sqlite")


# --------------------------------------------------------------------------
# requires_order()
# --------------------------------------------------------------------------


def test_plain_query_is_unordered():
    assert requires_order("SELECT name FROM people") is False


def test_order_by_makes_order_matter():
    assert requires_order("SELECT name FROM people ORDER BY age DESC") is True


def test_order_by_inside_a_string_literal_does_not_count():
    # The reason this parses instead of substring-matching.
    assert requires_order("SELECT 'order by' FROM people") is False


def test_window_function_ordering_does_not_count():
    sql = "SELECT name, row_number() OVER (ORDER BY age) FROM people"
    assert requires_order(sql) is False


def test_subquery_order_counts_strict():
    sql = "SELECT * FROM (SELECT name FROM people ORDER BY age)"
    assert requires_order(sql) is True


def test_unparseable_falls_back_to_substring():
    assert requires_order("!!! order by !!!") is True
    assert requires_order("!!! nonsense !!!") is False


# --------------------------------------------------------------------------
# compare()
# --------------------------------------------------------------------------


def test_identical_results_match():
    assert compare([(1, "a")], [(1, "a")]) is True


def test_row_order_ignored_unless_it_matters():
    pred, gold = [(1,), (2,)], [(2,), (1,)]
    assert compare(pred, gold, order_matters=False) is True
    assert compare(pred, gold, order_matters=True) is False


def test_distinct_is_not_the_same_answer():
    # Set semantics would call these equal and inflate the score.
    pred, gold = [("ada",)], [("ada",), ("ada",)]
    assert compare(pred, gold) is False
    assert compare(pred, gold, dedupe=True) is True


def test_empty_matches_empty():
    # Documented false positive: `WHERE 1=0` scores on empty-gold questions.
    # metrics.py reports that slice separately rather than fixing it here.
    assert compare([], []) is True


def test_empty_does_not_match_rows():
    assert compare([], [(1,)]) is False
    assert compare([(1,)], []) is False


def test_column_count_mismatch():
    assert compare([(1, 2)], [(1,)]) is False


def test_column_order_ignored_by_default():
    pred, gold = [(36, "ada"), (45, "grace")], [("ada", 36), ("grace", 45)]
    assert compare(pred, gold) is True
    assert compare(pred, gold, column_order_matters=True) is False


def test_column_permutation_does_not_break_row_correspondence():
    # Both columns hold {1, 2} either way, so the cheap per-column prefilter
    # accepts every permutation -- the full row check has to reject these.
    assert compare([(1, 1), (2, 2)], [(1, 2), (2, 1)]) is False


def test_int_and_float_are_the_same_number():
    assert compare([(3,)], [(3.0,)]) is True


def test_float_noise_is_tolerated():
    # SUM(x)/COUNT(x) vs AVG(x) should not be scored as a wrong answer.
    assert compare([(0.1 + 0.2,)], [(0.3,)]) is True


def test_genuinely_different_floats_do_not_match():
    assert compare([(1.0000001,)], [(1.0,)]) is False


def test_nulls_compare_equal_to_nulls():
    assert compare([(None,)], [(None,)]) is True
    assert compare([(None,)], [(0,)]) is False


def test_nan_matches_nan():
    nan = float("nan")
    assert compare([(nan,)], [(nan,)]) is True


def test_strings_are_compared_exactly():
    # Both queries read the same database, so a case difference is a real one.
    assert compare([("Ada",)], [("ada",)]) is False


# --------------------------------------------------------------------------
# the two halves together
# --------------------------------------------------------------------------


def test_end_to_end_equivalent_queries_score_correct(db):
    gold_sql = "SELECT name FROM people WHERE age > 40"
    pred_sql = "SELECT p.name FROM people AS p WHERE p.age >= 41"
    gold = run(gold_sql, db)
    pred = run(pred_sql, db)
    assert gold.ok and pred.ok
    assert compare(pred.rows, gold.rows, requires_order(gold_sql)) is True


def test_end_to_end_ordered_query_catches_wrong_direction(db):
    gold_sql = "SELECT name FROM people ORDER BY age DESC"
    pred_sql = "SELECT name FROM people ORDER BY age ASC"
    gold = run(gold_sql, db)
    pred = run(pred_sql, db)
    assert compare(pred.rows, gold.rows, requires_order(gold_sql)) is False
    # Same rows -- only the ordering requirement separates them.
    assert compare(pred.rows, gold.rows, order_matters=False) is True


def test_end_to_end_avg_agrees_with_manual_average(db):
    gold = run("SELECT avg(age) FROM people", db)
    pred = run("SELECT sum(age) * 1.0 / count(age) FROM people", db)
    assert compare(pred.rows, gold.rows) is True
    assert not math.isnan(gold.rows[0][0])


# --------------------------------------------------------------------------
# parse_sql -- the guard against a parser that never returns
# --------------------------------------------------------------------------


def test_parse_sql_accepts_normal_queries():
    assert parse_sql("SELECT name FROM people WHERE age > 40") is not None


def test_parse_sql_rejects_garbage():
    assert parse_sql("SELECT FROM WHERE (((") is None
    assert parse_sql("") is None


def test_parse_sql_accepts_realistic_join_counts():
    # The most JOINs in any of the 3,181 Spider gold queries is 6.
    sql = "SELECT a FROM t0 " + " ".join(
        f"JOIN t{i} ON t{i}.id = t0.id" for i in range(1, 7)
    )
    assert parse_sql(sql) is not None


def test_parse_sql_refuses_degenerate_join_chains():
    # sqlglot's parser is exponential in ON-less JOINs: 20 of them take 11s and
    # a few more never return. A CPT prediction shaped exactly like this hung an
    # evaluation run until the box idle-shut-down under it.
    sql = "SELECT a FROM t0 " + " ".join(f"JOIN t{i}" for i in range(1, 40))
    assert parse_sql(sql) is None


def test_parse_sql_refuses_overlong_input():
    # Longest legitimate gold query is 608 characters.
    assert parse_sql("SELECT " + "a," * 2000 + "b FROM t") is None


def test_degenerate_query_does_not_hang_the_metrics(db):
    # The end-to-end property that actually matters: pathological model output
    # must return a verdict quickly, not stall the run.
    import time

    sql = "SELECT a FROM t0 " + " ".join(f"JOIN t{i}" for i in range(1, 60))
    start = time.perf_counter()
    assert requires_order(sql) is False
    assert run(sql, db).status == "error"
    assert time.perf_counter() - start < 2.0
