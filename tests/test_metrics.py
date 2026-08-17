"""Tests for sqlrl.eval.metrics."""

from __future__ import annotations

import sqlite3

import pytest

from sqlrl.eval.executor import read_schema
from sqlrl.eval.metrics import (
    ExampleScore,
    aggregate,
    classify_error,
    format_report,
    parses,
    score_example,
    structural_match,
)


@pytest.fixture(scope="module")
def db(tmp_path_factory) -> str:
    path = tmp_path_factory.mktemp("dbs") / "metrics.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE people (id INTEGER, name TEXT, age INTEGER, score REAL);
        INSERT INTO people VALUES
            (1, 'ada',   36, 90.5),
            (2, 'grace', 45, 88.0),
            (3, 'alan',  41, 90.5);

        CREATE TABLE pets (id INTEGER, owner_id INTEGER, name TEXT);
        INSERT INTO pets VALUES (1, 1, 'rex'), (2, 2, 'mia');
        """
    )
    conn.commit()
    conn.close()
    return str(path)


@pytest.fixture(scope="module")
def schema(db) -> dict:
    return read_schema(db)


# --------------------------------------------------------------------------
# read_schema
# --------------------------------------------------------------------------


def test_read_schema(schema):
    assert set(schema) == {"people", "pets"}
    assert list(schema["people"]) == ["id", "name", "age", "score"]
    assert schema["people"]["name"] == "TEXT"


def test_read_schema_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_schema(tmp_path / "nope.sqlite")


# --------------------------------------------------------------------------
# parses / classify_error
# --------------------------------------------------------------------------


def test_parses():
    assert parses("SELECT name FROM people") is True
    assert parses("SELECT FROM WHERE (((") is False


def test_parses_accepts_sql_that_will_not_run():
    # Parsing is about syntax only -- a hallucinated table is still valid SQL.
    # That gap is exactly what separates parse rate from execution rate.
    assert parses("SELECT * FROM employees") is True


@pytest.mark.parametrize(
    "error, expected",
    [
        ("OperationalError: no such table: employees", "unknown_table"),
        ("OperationalError: no such column: salary", "unknown_column"),
        ("OperationalError: ambiguous column name: id", "ambiguous_column"),
        ("OperationalError: no such function: strftime2", "unknown_function"),
        ('OperationalError: near ")": syntax error', "syntax"),
        ("query produced no result set", "not_a_query"),
        ("ProgrammingError: something else entirely", "other"),
    ],
)
def test_classify_error(error, expected):
    assert classify_error("error", error) == expected


def test_classify_non_errors():
    assert classify_error("ok", None) == "ok"
    assert classify_error("timeout", "exceeded 5.0s") == "timeout"
    assert classify_error("too_many_rows", "more than 10000 rows") == "too_many_rows"


# --------------------------------------------------------------------------
# structural_match
# --------------------------------------------------------------------------


def test_identical_queries_match(schema):
    sql = "SELECT name FROM people WHERE age > 40"
    assert structural_match(sql, sql, schema) is True


def test_aliases_and_case_are_cosmetic(schema):
    assert structural_match(
        "SELECT p.NAME FROM People AS p WHERE p.age > 40",
        "select name from people where age > 40",
        schema,
    ) is True


def test_literal_values_are_ignored(schema):
    assert structural_match(
        "SELECT name FROM people WHERE age > 40",
        "SELECT name FROM people WHERE age > 99",
        schema,
    ) is True


def test_limit_values_are_not_ignored(schema):
    # "top 3" and "top 5" are different questions.
    assert structural_match(
        "SELECT name FROM people ORDER BY age DESC LIMIT 3",
        "SELECT name FROM people ORDER BY age DESC LIMIT 5",
        schema,
    ) is False


def test_output_alias_is_cosmetic(schema):
    assert structural_match(
        "SELECT count(*) AS total FROM people",
        "SELECT count(*) FROM people",
        schema,
    ) is True


def test_column_and_conjunct_order_are_cosmetic(schema):
    assert structural_match(
        "SELECT age, name FROM people WHERE name = 'ada' AND age > 40",
        "SELECT name, age FROM people WHERE age > 41 AND name = 'grace'",
        schema,
    ) is True


def test_order_by_direction_is_not_cosmetic(schema):
    assert structural_match(
        "SELECT name FROM people ORDER BY age ASC",
        "SELECT name FROM people ORDER BY age DESC",
        schema,
    ) is False


def test_join_columns_are_not_collapsed(schema):
    # The alias inlining must not lose which table a column came from.
    assert structural_match(
        "SELECT a.id FROM people a JOIN pets b ON a.id = b.owner_id",
        "SELECT b.id FROM people a JOIN pets b ON a.id = b.owner_id",
        schema,
    ) is False


def test_different_aggregate_does_not_match(schema):
    assert structural_match(
        "SELECT avg(age) FROM people",
        "SELECT max(age) FROM people",
        schema,
    ) is False


def test_unparseable_prediction_does_not_match(schema):
    assert structural_match("SELECT FROM (((", "SELECT name FROM people", schema) is False


def test_unresolvable_prediction_does_not_match(schema):
    # Hallucinated column: fails to resolve against the schema. It must not be
    # compared against a gold query that resolved cleanly.
    assert structural_match(
        "SELECT salary FROM people", "SELECT name FROM people", schema
    ) is False


# --------------------------------------------------------------------------
# score_example
# --------------------------------------------------------------------------


def test_score_correct_prediction(db):
    score = score_example(
        "SELECT p.name FROM people AS p WHERE p.age > 40",
        "SELECT name FROM people WHERE age > 40",
        db,
    )
    assert score.execution_match is True
    assert score.structural_match is True
    assert score.parsed and score.executed and score.gold_ok
    assert score.gold_empty is False
    assert score.error_kind == "ok"


def test_score_right_rows_wrong_shape(db):
    # Same answer via a different query: EX credits it, structural match does not.
    score = score_example(
        "SELECT name FROM people WHERE age >= 41 AND age <= 41",
        "SELECT name FROM people WHERE age = 41",
        db,
    )
    assert score.execution_match is True
    assert score.structural_match is False


def test_score_hallucinated_column(db):
    score = score_example(
        "SELECT salary FROM people", "SELECT name FROM people", db
    )
    assert score.parsed is True  # valid SQL...
    assert score.executed is False  # ...that the database rejects
    assert score.execution_match is False
    assert score.error_kind == "unknown_column"


def test_score_empty_prediction(db):
    score = score_example("", "SELECT name FROM people", db)
    assert score.executed is False
    assert score.execution_match is False
    assert score.error_kind == "not_a_query"


def test_empty_prediction_does_not_score_on_empty_gold(db):
    # The trap: gold returns nothing, so an empty result set would match. The
    # executor calls a non-query an error, so there is nothing to compare.
    score = score_example("", "SELECT name FROM people WHERE age > 200", db)
    assert score.gold_empty is True
    assert score.execution_match is False


def test_empty_gold_is_flagged(db):
    score = score_example(
        "SELECT name FROM people WHERE age > 300",
        "SELECT name FROM people WHERE age > 200",
        db,
    )
    # Both empty, so this counts as correct -- and gets flagged so the report
    # can show how much of the score comes from questions like this.
    assert score.execution_match is True
    assert score.gold_empty is True


def test_broken_gold_is_not_scoreable(db):
    score = score_example("SELECT name FROM people", "SELECT nope FROM nowhere", db)
    assert score.gold_ok is False
    assert score.execution_match is False


# --------------------------------------------------------------------------
# score_example: terminated
# --------------------------------------------------------------------------


def test_score_example_without_raw_leaves_terminated_unknown(db):
    score = score_example("SELECT name FROM people", "SELECT name FROM people", db)
    assert score.terminated is None


def test_score_example_with_clean_raw_sets_terminated_true(db):
    score = score_example(
        "SELECT name FROM people", "SELECT name FROM people", db,
        raw="<answer>SELECT name FROM people</answer>",
    )
    assert score.terminated is True


def test_score_example_with_trailing_junk_raw_sets_terminated_false(db):
    score = score_example(
        "SELECT name FROM people", "SELECT name FROM people", db,
        raw="<answer>SELECT name FROM people</answer> extra junk",
    )
    assert score.terminated is False


# --------------------------------------------------------------------------
# aggregate / format_report
# --------------------------------------------------------------------------


def make(**overrides) -> ExampleScore:
    base = dict(
        execution_match=True,
        structural_match=True,
        parsed=True,
        executed=True,
        gold_ok=True,
        gold_empty=False,
        pred_status="ok",
        error_kind="ok",
    )
    return ExampleScore(**{**base, **overrides})


def test_aggregate_rates():
    scores = [make(), make(), make(execution_match=False), make(execution_match=False)]
    report = aggregate(scores)
    assert report.n == 4
    assert report.scored == 4
    assert report.execution_accuracy == 0.5
    assert report.parse_rate == 1.0


def test_unscoreable_examples_leave_the_denominator():
    scores = [make(), make(gold_ok=False, execution_match=False)]
    report = aggregate(scores)
    assert report.n == 2
    assert report.scored == 1
    assert report.gold_failures == 1
    # 1 of 1 scoreable, not 1 of 2 -- a gold we cannot compute is not a miss.
    assert report.execution_accuracy == 1.0


def test_empty_gold_slice_is_separated():
    scores = [
        make(gold_empty=True),
        make(gold_empty=True),
        make(execution_match=False),
        make(execution_match=False),
    ]
    report = aggregate(scores)
    assert report.empty_gold == 2
    assert report.execution_accuracy == 0.5
    # Remove the empty-gold freebies and the real score is zero.
    assert report.execution_accuracy_nonempty == 0.0


def test_error_kinds_counted():
    scores = [
        make(error_kind="unknown_column", execution_match=False),
        make(error_kind="unknown_column", execution_match=False),
        make(error_kind="syntax", execution_match=False),
        make(),
    ]
    report = aggregate(scores)
    assert report.error_kinds == {"unknown_column": 2, "syntax": 1}


def test_aggregate_of_nothing_does_not_divide_by_zero():
    report = aggregate([])
    assert report.n == 0
    assert report.execution_accuracy == 0.0


def test_aggregate_stop_rate():
    scores = [
        make(terminated=True), make(terminated=True), make(terminated=False),
        make(terminated=None),
    ]
    report = aggregate(scores)
    # 2 of 3 known examples stopped cleanly; the unknown one is excluded from
    # both the numerator and the denominator.
    assert report.stop_known == 3
    assert report.stop_rate == pytest.approx(2 / 3)


def test_aggregate_stop_rate_unknown_when_nothing_is_known():
    report = aggregate([make(), make()])
    assert report.stop_known == 0
    assert report.stop_rate == 0.0


def test_format_report_flags_the_empty_gold_prop():
    scores = [make(gold_empty=True)] * 5 + [make(execution_match=False)] * 5
    text = format_report(aggregate(scores), title="test model")
    assert "test model" in text
    assert "execution accuracy" in text
    assert "propped up by empty results" in text


def test_format_report_flags_schema_hallucination():
    scores = [make(executed=False, execution_match=False, error_kind="unknown_column")] * 5
    scores += [make()] * 5
    text = format_report(aggregate(scores))
    assert "inventing tables or columns" in text
    assert "unknown_column" in text


def test_format_report_shouts_about_broken_gold():
    text = format_report(aggregate([make(), make(gold_ok=False)]))
    assert "gold queries did not run" in text


def test_format_report_omits_stop_line_when_unknown():
    text = format_report(aggregate([make(), make()]))
    assert "stopped cleanly" not in text


def test_format_report_includes_stop_line_when_known():
    text = format_report(aggregate([make(terminated=True), make(terminated=False)]))
    assert "stopped cleanly" in text


def test_diagnose_warns_when_stop_rate_is_low():
    scores = [make(terminated=False)] * 9 + [make(terminated=True)]
    text = format_report(aggregate(scores))
    assert "not emitting its stop token" in text


def test_diagnose_does_not_warn_when_stop_rate_is_high():
    scores = [make(terminated=True)] * 9 + [make(terminated=False)]
    text = format_report(aggregate(scores))
    assert "not emitting its stop token" not in text
