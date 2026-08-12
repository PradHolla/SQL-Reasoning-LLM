"""Tests for sqlrl.eval.prompts.

extract_sql is the one to watch: every bug in it lowers the score, and a low
score looks like a bad model rather than a bad harness.
"""

from __future__ import annotations

import pytest

from sqlrl.eval.prompts import (
    SYSTEM_PROMPT,
    Prompt,
    chat_prompt,
    cpt_prompt,
    extract_sql,
    render_schema,
)

SCHEMA = {
    "singer": {"id": "INTEGER", "name": "TEXT"},
    "concert": {"id": "INTEGER", "year": "INTEGER"},
}


# --------------------------------------------------------------------------
# prompt construction
# --------------------------------------------------------------------------


def test_render_schema_matches_training_format():
    # sql-create-context joins statements with "; " and has no trailing one.
    assert render_schema(SCHEMA) == (
        "CREATE TABLE singer (id INTEGER, name TEXT); "
        "CREATE TABLE concert (id INTEGER, year INTEGER)"
    )


def test_render_empty_schema():
    assert render_schema({}) == ""


def test_chat_prompt_is_verbatim_v0():
    prompt = chat_prompt("CREATE TABLE singer (id INTEGER)", "How many singers?")
    assert prompt.text is None
    assert prompt.messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    # The exact user string SFT and GRPO trained on.
    assert prompt.messages[1]["content"] == (
        "Schema: CREATE TABLE singer (id INTEGER)\nQuestion: How many singers?"
    )


def test_cpt_prompt_matches_the_cpt_template():
    prompt = cpt_prompt("CREATE TABLE singer (id INTEGER)", "How many singers?")
    assert prompt.messages is None
    assert prompt.text == (
        "\n-- Database Schema --\n"
        "CREATE TABLE singer (id INTEGER)\n\n"
        "-- Executed SQL Query --\n"
        "-- Intent: How many singers?\n"
    )


def test_prompt_requires_exactly_one_form():
    with pytest.raises(ValueError):
        Prompt()
    with pytest.raises(ValueError):
        Prompt(messages=[{"role": "user", "content": "x"}], text="x")


# --------------------------------------------------------------------------
# extract_sql
# --------------------------------------------------------------------------


def test_extract_from_answer_tags():
    text = "<think>\nreasoning\n</think>\n<answer>\nSELECT count(*) FROM singer\n</answer>"
    assert extract_sql(text) == "SELECT count(*) FROM singer"


def test_extract_from_unterminated_answer():
    # Hit the token limit mid-answer. There is still a query worth scoring.
    text = "<think>x</think>\n<answer>\nSELECT count(*) FROM singer"
    assert extract_sql(text) == "SELECT count(*) FROM singer"


def test_extract_ignores_reasoning_that_mentions_sql():
    # The <think> block usually contains SQL-ish prose. It must not win.
    text = "<think>Maybe SELECT * FROM wrong</think><answer>SELECT id FROM singer</answer>"
    assert extract_sql(text) == "SELECT id FROM singer"


def test_extract_after_think_when_no_answer_tags():
    text = "<think>reasoning</think>\nSELECT name FROM singer"
    assert extract_sql(text) == "SELECT name FROM singer"


def test_extract_from_markdown_fence():
    text = "Here you go:\n```sql\nSELECT name FROM singer\n```\nHope that helps."
    assert extract_sql(text) == "SELECT name FROM singer"


def test_extract_from_unterminated_fence():
    assert extract_sql("```sql\nSELECT name FROM singer") == "SELECT name FROM singer"


def test_extract_from_mismatched_tags():
    # Observed verbatim from the v0 SFT checkpoint: opens <answer>, closes with
    # </think>. Carrying that tag into the query makes a syntax error out of an
    # answer that might have been right.
    text = (
        "<think>\nI need to analyze the tables.\n<answer>\n"
        "SELECT COUNT(*) FROM singer\n</think>"
    )
    assert extract_sql(text) == "SELECT COUNT(*) FROM singer"


def test_extract_keeps_sql_comparison_operators():
    # The tag-stripping must not eat "<", "<=" or the "<>" not-equal operator.
    assert extract_sql("SELECT a FROM t WHERE x < 3 AND y <= 4 AND z <> 5") == (
        "SELECT a FROM t WHERE x < 3 AND y <= 4 AND z <> 5"
    )


def test_extract_bare_sql():
    assert extract_sql("SELECT name FROM singer") == "SELECT name FROM singer"


def test_extract_strips_preamble_prose():
    text = "Sure! Here is the query you asked for: SELECT name FROM singer"
    assert extract_sql(text) == "SELECT name FROM singer"


def test_extract_cuts_trailing_commentary():
    text = "SELECT name FROM singer; This returns every singer name."
    assert extract_sql(text) == "SELECT name FROM singer"


def test_extract_collapses_whitespace():
    assert extract_sql("SELECT   name\n  FROM\tsinger") == "SELECT name FROM singer"


def test_extract_handles_with_clause():
    text = "<answer>WITH x AS (SELECT 1) SELECT * FROM x</answer>"
    assert extract_sql(text) == "WITH x AS (SELECT 1) SELECT * FROM x"


@pytest.mark.parametrize("text", ["", "   ", "I don't know.", "<answer></answer>"])
def test_extract_returns_empty_when_there_is_no_query(text):
    # Empty means the executor reports not_a_query, which is the honest label.
    assert extract_sql(text) == ""


def test_extract_does_not_invent_a_query_from_refusal():
    assert extract_sql("I cannot answer that question.") == ""
