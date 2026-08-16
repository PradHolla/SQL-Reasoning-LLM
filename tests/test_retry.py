"""Tests for sqlrl.eval.retry.

No GPU, no model: ``generate`` is a fake injected by each test, following the
same pattern ``run_retry`` was designed for. Execution still goes through the
real ``run()`` against a temporary SQLite database, matching how
tests/test_executor.py and tests/test_metrics.py do it.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from sqlrl.eval.executor import ExecResult, read_schema
from sqlrl.eval.metrics import aggregate, score_example
from sqlrl.eval.prompts import chat_prompt, cpt_prompt, render_schema
from sqlrl.eval.retry import (
    Attempt,
    Trace,
    at_budget,
    attempt_counts,
    feedback,
    retry_prompt,
    run_retry,
)
from sqlrl.eval.run_eval import Prediction, RetryRecord
from sqlrl.eval.spider import Example


@pytest.fixture(scope="module")
def db(tmp_path_factory) -> str:
    path = tmp_path_factory.mktemp("dbs") / "retry.sqlite"
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


@pytest.fixture(scope="module")
def schema_text(db) -> str:
    return render_schema(read_schema(db))


def _raw(sql: str) -> str:
    """A model output shaped like SYSTEM_PROMPT asked for."""
    return f"<think>reasoning</think><answer>{sql}</answer>"


# --------------------------------------------------------------------------
# feedback
# --------------------------------------------------------------------------


def test_feedback_strips_exception_class_prefix():
    text = feedback("SELECT bogus FROM t", "OperationalError: no such column: T1.name")
    assert "OperationalError" not in text
    assert "no such column: T1.name" in text


def test_feedback_leaves_unprefixed_error_alone():
    # No "SomethingError: " prefix to strip -- must pass the message through.
    text = feedback("SELECT 1", "no such column: T1.name")
    assert "no such column: T1.name" in text


def test_feedback_caps_the_quoted_query():
    # A hallucinated join chain quoted back in full is what pushes a round-3
    # multiturn prompt over the backend's input ceiling, and it teaches the
    # model nothing anyway.
    from sqlrl.eval.retry import MAX_QUOTED_SQL

    long_sql = "SELECT " + ", ".join(f"col{i}" for i in range(500))
    assert len(long_sql) > MAX_QUOTED_SQL
    text = feedback(long_sql, "OperationalError: no such column: col0")
    assert long_sql not in text
    assert long_sql[:MAX_QUOTED_SQL] in text
    assert "..." in text


def test_feedback_reports_no_sql_found():
    text = feedback("", "irrelevant")
    assert "no sql query was found" in text.lower()
    assert "<answer>" in text and "</answer>" in text


# --------------------------------------------------------------------------
# retry_prompt
# --------------------------------------------------------------------------


def test_retry_prompt_multiturn_grows_by_one_pair_per_attempt():
    base = chat_prompt("CREATE TABLE t (a INT)", "how many rows?")
    assert len(base.messages) == 2

    attempt1 = Attempt(
        raw=_raw("SELECT bogus FROM t"),
        sql="SELECT bogus FROM t",
        status="error",
        error="OperationalError: no such column: bogus",
    )
    after_one = retry_prompt(base, [attempt1], "multiturn")
    assert len(after_one.messages) == 4
    assert [m["role"] for m in after_one.messages] == ["system", "user", "assistant", "user"]
    assert after_one.messages[2] == {"role": "assistant", "content": attempt1.raw}
    assert "no such column: bogus" in after_one.messages[3]["content"]

    attempt2 = Attempt(
        raw=_raw("SELECT a FROM tt"),
        sql="SELECT a FROM tt",
        status="error",
        error="OperationalError: no such table: tt",
    )
    after_two = retry_prompt(base, [attempt1, attempt2], "multiturn")
    assert len(after_two.messages) == 6
    assert [m["role"] for m in after_two.messages] == [
        "system", "user", "assistant", "user", "assistant", "user",
    ]
    assert after_two.messages[4] == {"role": "assistant", "content": attempt2.raw}
    # The base messages must not have been mutated by the first call.
    assert len(base.messages) == 2


def test_retry_prompt_restate_never_grows_past_two_messages():
    base = chat_prompt("CREATE TABLE t (a INT)", "how many rows?")

    attempt1 = Attempt(
        raw=_raw("SELECT bogus FROM t"),
        sql="SELECT bogus FROM t",
        status="error",
        error="OperationalError: no such column: bogus",
    )
    after_one = retry_prompt(base, [attempt1], "restate")
    assert len(after_one.messages) == 2
    assert after_one.messages[0]["role"] == "system"
    assert "CREATE TABLE t (a INT)" in after_one.messages[1]["content"]
    assert "how many rows?" in after_one.messages[1]["content"]
    assert "no such column: bogus" in after_one.messages[1]["content"]

    attempt2 = Attempt(
        raw=_raw("SELECT a FROM tt"),
        sql="SELECT a FROM tt",
        status="error",
        error="OperationalError: no such table: tt",
    )
    after_two = retry_prompt(base, [attempt1, attempt2], "restate")
    assert len(after_two.messages) == 2
    # Only the most recent failure appears -- the first one's error is gone.
    assert "no such table: tt" in after_two.messages[1]["content"]
    assert "no such column: bogus" not in after_two.messages[1]["content"]
    assert "CREATE TABLE t (a INT)" in after_two.messages[1]["content"]
    assert "how many rows?" in after_two.messages[1]["content"]


def test_retry_prompt_rejects_unknown_style():
    base = chat_prompt("s", "q")
    attempt = Attempt(raw="x", sql="SELECT 1", status="error", error="e")
    with pytest.raises(ValueError):
        retry_prompt(base, [attempt], "bogus-style")


def test_retry_prompt_rejects_text_prompt():
    base = cpt_prompt("s", "q")
    attempt = Attempt(raw="x", sql="SELECT 1", status="error", error="e")
    with pytest.raises(ValueError):
        retry_prompt(base, [attempt], "multiturn")


# --------------------------------------------------------------------------
# run_retry -- the batched loop
# --------------------------------------------------------------------------


def test_loop_only_regenerates_failures(db, schema_text):
    prompts = [
        chat_prompt(schema_text, "how many people?"),
        chat_prompt(schema_text, "list names"),
    ]
    db_paths = [Path(db), Path(db)]
    batches: list[list] = []

    def fake_generate(batch):
        batches.append(batch)
        if len(batches) == 1:
            # Example 0 fails (hallucinated column), example 1 succeeds.
            return [_raw("SELECT bogus FROM people"), _raw("SELECT name FROM people")]
        # Only example 0 should come back for round 2.
        return [_raw("SELECT name FROM people")]

    histories = run_retry(prompts, db_paths, fake_generate, max_attempts=3)

    assert len(batches) == 2
    assert len(batches[0]) == 2
    assert len(batches[1]) == 1  # strictly smaller, and only the failure
    # The retried prompt carries the round-1 history for example 0 only.
    assert len(batches[1][0].messages) == 4
    assert "how many people?" in batches[1][0].messages[1]["content"]

    assert [a.status for a in histories[0]] == ["error", "ok"]
    assert [a.status for a in histories[1]] == ["ok"]


def test_early_stop_when_everything_succeeds_round_one(db, schema_text):
    prompts = [
        chat_prompt(schema_text, "how many people?"),
        chat_prompt(schema_text, "list names"),
    ]
    db_paths = [Path(db), Path(db)]
    calls = []

    def fake_generate(batch):
        calls.append(batch)
        return [_raw("SELECT count(*) FROM people") for _ in batch]

    histories = run_retry(prompts, db_paths, fake_generate, max_attempts=3)

    assert len(calls) == 1
    assert all(len(h) == 1 for h in histories)
    assert all(h[0].status == "ok" for h in histories)


@pytest.mark.parametrize("status", ["timeout", "too_many_rows"])
def test_no_retry_on_non_rejection_statuses(monkeypatch, db, schema_text, status):
    # SQLite *accepted* these queries -- metrics.py already counts them as
    # executed. A stub stands in for run() so the test does not depend on
    # actually triggering a timeout or an oversized result.
    import sqlrl.eval.retry as retry_module

    def fake_run(sql, db_path, timeout=30.0):
        return ExecResult(status, [], "stub")

    monkeypatch.setattr(retry_module, "run", fake_run)

    prompts = [chat_prompt(schema_text, "how many people?")]
    db_paths = [Path(db)]
    calls = []

    def fake_generate(batch):
        calls.append(batch)
        return [_raw("SELECT count(*) FROM people") for _ in batch]

    histories = run_retry(prompts, db_paths, fake_generate, max_attempts=3)

    assert len(calls) == 1  # never retried
    assert len(histories[0]) == 1
    assert histories[0][0].status == status


def test_on_round_callback_reports_shrinking_batches(db, schema_text):
    prompts = [
        chat_prompt(schema_text, "q1"),
        chat_prompt(schema_text, "q2"),
    ]
    db_paths = [Path(db), Path(db)]
    rounds: list[tuple[int, int]] = []

    def fake_generate(batch):
        if not rounds:
            return [_raw("SELECT bogus FROM people"), _raw("SELECT name FROM people")]
        return [_raw("SELECT name FROM people")]

    run_retry(
        prompts, db_paths, fake_generate, max_attempts=3,
        on_round=lambda attempt_number, n: rounds.append((attempt_number, n)),
    )

    assert rounds == [(1, 2), (2, 1)]


def test_generate_output_count_mismatch_raises(db, schema_text):
    prompts = [chat_prompt(schema_text, "q1"), chat_prompt(schema_text, "q2")]
    db_paths = [Path(db), Path(db)]

    with pytest.raises(AssertionError):
        run_retry(prompts, db_paths, lambda batch: [_raw("SELECT 1")], max_attempts=1)


# --------------------------------------------------------------------------
# at_budget
# --------------------------------------------------------------------------


def test_at_budget_one_attempt_trace_clamps_at_every_budget():
    only = Attempt(raw="r", sql="SELECT 1", status="ok", error=None)
    assert at_budget([only], 1) is only
    assert at_budget([only], 2) is only
    assert at_budget([only], 3) is only


def test_at_budget_three_attempt_trace_picks_the_matching_attempt():
    a1 = Attempt(raw="r1", sql="s1", status="error", error="e1")
    a2 = Attempt(raw="r2", sql="s2", status="error", error="e2")
    a3 = Attempt(raw="r3", sql="s3", status="ok", error=None)
    assert at_budget([a1, a2, a3], 1) is a1
    assert at_budget([a1, a2, a3], 2) is a2
    assert at_budget([a1, a2, a3], 3) is a3


def test_at_budget_rejects_budget_below_one():
    only = Attempt(raw="r", sql="s", status="ok", error=None)
    with pytest.raises(ValueError):
        at_budget([only], 0)


def test_at_budget_rejects_empty_attempts():
    with pytest.raises(ValueError):
        at_budget([], 1)


# --------------------------------------------------------------------------
# attempt_counts
# --------------------------------------------------------------------------


def _attempts(*statuses: str) -> list[Attempt]:
    return [
        Attempt(raw="r", sql="SELECT 1", status=s, error=None if s == "ok" else "e")
        for s in statuses
    ]


def test_attempt_counts_buckets_by_attempts_used():
    used, still_rejected = attempt_counts([
        _attempts("ok"),
        _attempts("ok"),
        _attempts("error", "ok"),
        _attempts("error", "error", "ok"),
    ])
    assert used == {1: 2, 2: 1, 3: 1}
    assert still_rejected == 0


def test_attempt_counts_separates_exhausted_budgets():
    # A trace whose last attempt is still "error" never produced SQL the
    # database would accept -- it must not be counted as "accepted after 3".
    used, still_rejected = attempt_counts([
        _attempts("ok"),
        _attempts("error", "error", "error"),
    ])
    assert used == {1: 1}
    assert still_rejected == 1


def test_attempt_counts_treats_non_rejections_as_accepted():
    # timeout and too_many_rows are not rejections; the loop stops on them, so
    # they count as settled, consistent with metrics.py's execution rate.
    used, still_rejected = attempt_counts([_attempts("timeout"), _attempts("too_many_rows")])
    assert used == {1: 2}
    assert still_rejected == 0


# --------------------------------------------------------------------------
# RetryRecord.at_attempt
# --------------------------------------------------------------------------


def _retry_record(traces: list[Trace], max_attempts: int = 3) -> RetryRecord:
    return RetryRecord(
        model="m", split="test", n=len(traces), device="cpu", dtype="float32",
        max_new_tokens=64, decoding="greedy", seed=0, git_commit="deadbeef",
        generated_at="2026-01-01T00:00:00+00:00", generation_seconds=0.0,
        max_attempts=max_attempts, retry_style="multiturn", traces=traces,
    )


def test_execution_accuracy_is_monotonic_in_budget(db):
    # A status=="error" attempt can never execution-match gold -- there is
    # nothing to compare rows against. So collapsing a trace to a later
    # attempt can only hold or raise accuracy as the budget grows, never
    # lower it, as long as every failure is eventually followed by a fix.
    # That structural fact is what this test checks, not any particular
    # numbers.
    gold_sql = "SELECT name FROM people"

    def trace(index: int, sql_and_status: list[tuple[str, str]]) -> Trace:
        attempts = [
            Attempt(
                raw=_raw(sql), sql=sql, status=status,
                error=None if status == "ok" else "OperationalError: no such column",
            )
            for sql, status in sql_and_status
        ]
        return Trace(index=index, db_id="db", question="q", gold_sql=gold_sql, attempts=attempts)

    traces = [
        trace(0, [("SELECT name FROM people", "ok")]),
        trace(1, [
            ("SELECT bogus FROM people", "error"),
            ("SELECT name FROM people", "ok"),
        ]),
        trace(2, [
            ("SELECT bogus FROM people", "error"),
            ("SELECT bogus2 FROM people", "error"),
            ("SELECT name FROM people", "ok"),
        ]),
    ]
    record = _retry_record(traces)
    examples = [
        Example(db_id="db", db_path=Path(db), question="q", gold_sql=gold_sql)
        for _ in traces
    ]

    accuracies = []
    for budget in (1, 2, 3):
        run_record = record.at_attempt(budget)
        scores = [
            score_example(p.pred_sql, p.gold_sql, examples[p.index].db_path)
            for p in run_record.predictions
        ]
        accuracies.append(aggregate(scores).execution_accuracy)

    assert accuracies == sorted(accuracies)
    assert accuracies[0] < accuracies[-1]  # not a vacuous, all-equal case


def test_at_attempt_one_matches_a_plain_single_shot_prediction(db):
    attempt = Attempt(
        raw=_raw("SELECT name FROM people"), sql="SELECT name FROM people",
        status="ok", error=None,
    )
    trace = Trace(index=0, db_id="db", question="q", gold_sql="SELECT name FROM people",
                  attempts=[attempt])
    record = _retry_record([trace])

    run_record = record.at_attempt(1)

    assert run_record.predictions == [
        Prediction(
            index=0, db_id="db", question="q", gold_sql="SELECT name FROM people",
            raw=attempt.raw, pred_sql=attempt.sql,
        )
    ]
    assert run_record.model == record.model
    assert run_record.n == record.n
