"""Tests for the CPU-reachable functions in sqlrl.data_prep.sample_traces.

Rejection sampling is only as good as its filter. If ``pick_trace`` ever let
reasoning quality outrank correctness, or dropped a correct trace because its
``<think>`` block was thin, the run would still finish and print a coverage
report -- it would just be quietly training the student on confident nonsense
instead of derivation. Every test here pins one way that could happen.

``sample()`` itself needs vLLM and a GPU and is out of scope: nothing here
imports vllm, calls ``sample()``, or loads a model.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from sqlrl.data_prep.sample_traces import (
    _done_indices,
    _identifiers,
    _schema_text,
    build_teacher_prompt,
    names_schema_identifiers,
    pick_trace,
    think_of,
)


@pytest.fixture(scope="module")
def db(tmp_path_factory) -> str:
    path = tmp_path_factory.mktemp("dbs") / "traces.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE people (id INTEGER, name TEXT, age INTEGER);
        INSERT INTO people VALUES
            (1, 'ada',   36),
            (2, 'grace', 45),
            (3, 'alan',  41);
        """
    )
    conn.commit()
    conn.close()
    return str(path)


@pytest.fixture(scope="module")
def identifiers(db) -> set[str]:
    return _identifiers(db)


#: age > 40 keeps grace and alan.
GOLD_SQL = "SELECT name FROM people WHERE age > 40"
#: age < 40 keeps only ada -- a different row set from GOLD_SQL.
WRONG_SQL = "SELECT name FROM people WHERE age < 40"

#: Names a real table and a real column, so it should count as grounded.
GROUNDED = "the people table's age column tells us who is older than 40"
#: Generic filler that names nothing from the schema.
VAGUE = "let's think about this step by step and work out the best answer here"


def trace(think: str, sql: str) -> str:
    return f"<think>{think}</think><answer>{sql}</answer>"


# --------------------------------------------------------------------------
# build_teacher_prompt
# --------------------------------------------------------------------------


def test_build_teacher_prompt_embeds_schema_and_question_in_the_student_format():
    # The kept trace has to drop into the SFT set unchanged, so the tail of
    # the prompt has to match chat_prompt's user turn byte for byte.
    schema_text = "CREATE TABLE people (id INTEGER, name TEXT, age INTEGER)"
    question = "Who is older than 40?"
    prompt = build_teacher_prompt(schema_text, question)
    assert prompt.endswith(f"Schema: {schema_text}\nQuestion: {question}")


def test_build_teacher_prompt_demands_think_and_answer_tags_and_named_identifiers():
    prompt = build_teacher_prompt("CREATE TABLE t (a INTEGER)", "q?")
    assert "<think>" in prompt and "</think>" in prompt
    assert "<answer>" in prompt and "</answer>" in prompt
    assert "actual tables and columns" in prompt


# --------------------------------------------------------------------------
# think_of
# --------------------------------------------------------------------------


def test_think_of_extracts_the_block_contents():
    text = "<think>\n  need the people table\n</think><answer>SELECT 1</answer>"
    assert think_of(text) == "need the people table"


def test_think_of_is_case_insensitive_to_the_tag():
    text = "<THINK>reasoning here</THINK><answer>SELECT 1</answer>"
    assert think_of(text) == "reasoning here"


def test_think_of_returns_empty_string_without_raising_when_absent():
    # A correct-but-think-less candidate must not blow up the filter.
    assert think_of("<answer>SELECT 1</answer>") == ""


# --------------------------------------------------------------------------
# names_schema_identifiers
# --------------------------------------------------------------------------


def test_names_schema_identifiers_true_for_a_real_table_name(identifiers):
    assert names_schema_identifiers("I will query the people table", identifiers)


def test_names_schema_identifiers_is_case_insensitive(identifiers):
    assert names_schema_identifiers("Look at the PEOPLE table and AGE column", identifiers)


def test_names_schema_identifiers_matches_whole_words_not_substrings(identifiers):
    # "age" is a real column, but "agency" merely contains it as a substring.
    # A naive `in` check on the raw text would wrongly count this as grounded.
    assert not names_schema_identifiers("check with the agency first", identifiers)


def test_names_schema_identifiers_false_when_nothing_matches(identifiers):
    assert not names_schema_identifiers(VAGUE, identifiers)


# --------------------------------------------------------------------------
# pick_trace -- the filter itself
# --------------------------------------------------------------------------


def test_pick_trace_never_returns_a_wrong_query_no_matter_how_grounded(db, identifiers):
    # Reasoning quality must never override correctness -- that ordering is
    # the entire point of rejection sampling.
    candidates = [trace(GROUNDED, WRONG_SQL)]
    result, n_correct = pick_trace(candidates, GOLD_SQL, db, identifiers)
    assert result is None
    assert n_correct == 0


def test_pick_trace_counts_every_correct_candidate_even_ones_not_kept(db, identifiers):
    candidates = [trace(GROUNDED, GOLD_SQL), trace(VAGUE, GOLD_SQL)]
    _, n_correct = pick_trace(candidates, GOLD_SQL, db, identifiers)
    assert n_correct == 2


def test_pick_trace_does_not_count_a_wrong_candidate_alongside_correct_ones(db, identifiers):
    candidates = [trace(GROUNDED, GOLD_SQL), trace(GROUNDED, WRONG_SQL)]
    _, n_correct = pick_trace(candidates, GOLD_SQL, db, identifiers)
    assert n_correct == 1


def test_pick_trace_prefers_grounded_reasoning_over_vague(db, identifiers):
    candidates = [trace(VAGUE, GOLD_SQL), trace(GROUNDED, GOLD_SQL)]
    result, _ = pick_trace(candidates, GOLD_SQL, db, identifiers)
    assert result == trace(GROUNDED, GOLD_SQL)


def test_pick_trace_returns_a_vague_trace_rather_than_none_when_none_are_grounded(
    db, identifiers
):
    # A correct trace with weak reasoning beats no trace at all.
    candidates = [trace(VAGUE, GOLD_SQL)]
    result, n_correct = pick_trace(candidates, GOLD_SQL, db, identifiers)
    assert result is not None
    assert n_correct == 1


def test_pick_trace_keeps_the_shortest_among_equally_grounded_candidates(db, identifiers):
    # Both name "people" and "age", so both count as grounded; the tie-break
    # is length, because a 0.5B student has a small token budget to spend.
    short = trace("age column, people table", GOLD_SQL)
    long = trace(
        "the people table's age column, considered at great length with many "
        "extra words about the age column and the people table all over again",
        GOLD_SQL,
    )
    result, _ = pick_trace([long, short], GOLD_SQL, db, identifiers)
    assert result == short


def test_pick_trace_accepts_a_correct_candidate_with_no_think_block(db, identifiers):
    # Correctness alone makes a candidate eligible -- a missing <think> block
    # is not a reason to reject an otherwise-correct query.
    candidate = f"<answer>{GOLD_SQL}</answer>"
    assert think_of(candidate) == ""
    result, n_correct = pick_trace([candidate], GOLD_SQL, db, identifiers)
    assert result == candidate
    assert n_correct == 1


# --------------------------------------------------------------------------
# _schema_text
# --------------------------------------------------------------------------


def _row_with_user_turn(content: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": "You are a database expert."},
            {"role": "user", "content": content},
        ]
    }


def test_schema_text_recovers_the_schema_from_the_stored_user_turn():
    schema_text = "CREATE TABLE people (id INTEGER, name TEXT, age INTEGER)"
    question = "Who is older than 40?"
    row = _row_with_user_turn(f"Schema: {schema_text}\nQuestion: {question}")
    assert _schema_text(row) == schema_text


def test_schema_text_split_is_anchored_even_when_the_question_contains_question_colon():
    # "Question:" reappears inside the question text itself, but without a
    # leading newline. rsplit("\nQuestion:", 1) must still land on the one
    # real boundary rather than the embedded substring.
    schema_text = "CREATE TABLE people (id INTEGER, name TEXT, age INTEGER)"
    question = "What is the average Question: field length?"
    row = _row_with_user_turn(f"Schema: {schema_text}\nQuestion: {question}")
    assert _schema_text(row) == schema_text


# --------------------------------------------------------------------------
# _identifiers
# --------------------------------------------------------------------------


def test_identifiers_returns_lowercased_table_and_column_names(tmp_path):
    path = tmp_path / "mixed_case.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript("CREATE TABLE Employees (ID INTEGER, FullName TEXT);")
    conn.commit()
    conn.close()
    assert _identifiers(str(path)) == {"employees", "id", "fullname"}


# --------------------------------------------------------------------------
# _done_indices
# --------------------------------------------------------------------------


def test_done_indices_empty_for_a_missing_file(tmp_path):
    assert _done_indices(tmp_path / "nope.jsonl") == set()


def test_done_indices_includes_rows_whose_trace_is_null(tmp_path):
    # A null trace means the teacher failed every attempt on that question,
    # but it is still written to disk so a restart does not resample it.
    path = tmp_path / "traces.jsonl"
    rows = [
        {"index": 0, "trace": "<answer>SELECT 1</answer>", "n_correct": 1},
        {"index": 1, "trace": None, "n_correct": 0},
        {"index": 2, "trace": "<answer>SELECT 2</answer>", "n_correct": 3},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    assert _done_indices(path) == {0, 1, 2}
