"""Tests for the CPU-reachable data handling in sqlrl.training.grpo_spider.

``load_rows``, ``build_dataset``, and ``assert_prompts_fit`` are exercised
here. The model, ``GRPOTrainer``, and the training loop itself need a GPU and
are out of scope for this file -- none of it is instantiated or run.

The invariant worth stating up front is ``build_dataset``'s: it must put only
the system and user turns of ``messages`` in the ``prompt`` column. The
assistant turn holds the gold SQL, and if ``messages[:-1]`` ever slipped to
``messages``, the answer would ride along inside the prompt. Every rollout
would then see it, trivially reach the match tier, and the run would look
like a spectacular success while teaching the model nothing.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from sqlrl.tokenizer import build_tokenizer
from sqlrl.training.grpo_spider import assert_prompts_fit, build_dataset, load_rows

SYSTEM = (
    "You are a database expert. You must think step-by-step inside "
    "<think></think> tags, and output ONLY the final SQL query inside "
    "<answer></answer> tags."
)
SCHEMA = "CREATE TABLE people (id INTEGER, name TEXT, age INTEGER)"


@pytest.fixture(scope="module")
def db(tmp_path_factory) -> str:
    path = tmp_path_factory.mktemp("dbs") / "grpo.sqlite"
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
def tokenizer():
    try:
        return build_tokenizer("Qwen/Qwen2.5-0.5B", chat=True)
    except Exception as exc:  # noqa: BLE001 -- any download failure, offline or not
        pytest.skip(f"Qwen2.5-0.5B tokenizer unavailable (no network?): {exc}")


#: age > 40 keeps grace and alan -- two rows in the fixture.
GOLD_NONEMPTY = "SELECT name FROM people WHERE age > 40"
#: Nothing in the fixture is older than 200, so this gold query returns nothing.
GOLD_EMPTY = "SELECT name FROM people WHERE age > 200"


def make_row(db_path: str, question: str, gold_sql: str) -> dict:
    """One row in the shape of data/processed/spider_grpo.jsonl."""
    return {
        "db_id": "people",
        "db_path": db_path,
        "question": question,
        "gold_sql": gold_sql,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Schema: {SCHEMA}\nQuestion: {question}"},
            {
                "role": "assistant",
                "content": f"<think>\nthinking\n</think>\n<answer>\n{gold_sql}\n</answer>",
            },
        ],
    }


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return path


# --------------------------------------------------------------------------
# build_dataset -- the prompt/answer split
# --------------------------------------------------------------------------


def test_build_dataset_prompt_excludes_the_assistant_turn_holding_the_gold_answer(db):
    gold_sql = "SELECT name FROM people WHERE age > 40"
    rows = [make_row(db, "who is older than 40?", gold_sql)]
    dataset = build_dataset(rows)
    prompt = dataset[0]["prompt"]

    assert [message["role"] for message in prompt] == ["system", "user"]
    rendered = " ".join(message["content"] for message in prompt)
    assert gold_sql not in rendered


def test_build_dataset_has_exactly_prompt_db_path_and_gold_sql_columns(db):
    rows = [make_row(db, "how many people?", "SELECT count(*) FROM people")]
    dataset = build_dataset(rows)
    # TRL forwards every non-prompt column to the reward function unchanged, so
    # a stray column here becomes an unexpected keyword argument at train time.
    assert set(dataset.column_names) == {"prompt", "db_path", "gold_sql"}


def test_build_dataset_carries_db_path_and_gold_sql_through_unchanged(db):
    gold_sql = "SELECT name FROM people"
    rows = [make_row(db, "who is in the table?", gold_sql)]
    dataset = build_dataset(rows)
    assert dataset[0]["db_path"] == db
    assert dataset[0]["gold_sql"] == gold_sql


def test_build_dataset_preserves_row_order(db):
    rows = [
        make_row(db, "first", "SELECT name FROM people WHERE age = 36"),
        make_row(db, "second", "SELECT name FROM people WHERE age = 45"),
        make_row(db, "third", "SELECT name FROM people WHERE age = 41"),
    ]
    dataset = build_dataset(rows)
    # db_path/gold_sql have to stay aligned with the prompt they came from --
    # there is no id column to re-sort by if this ever drifted.
    assert [row["gold_sql"] for row in dataset] == [row["gold_sql"] for row in rows]
    assert [row["prompt"][1]["content"] for row in dataset] == [
        row["messages"][1]["content"] for row in rows
    ]


# --------------------------------------------------------------------------
# load_rows -- empty-gold filtering
# --------------------------------------------------------------------------


def test_load_rows_drops_empty_gold_rows_and_keeps_the_rest(tmp_path, db):
    rows = [
        make_row(db, "real one", GOLD_NONEMPTY),
        make_row(db, "empty one", GOLD_EMPTY),
        make_row(db, "real two", "SELECT name FROM people WHERE age < 40"),
    ]
    path = write_jsonl(tmp_path / "rows.jsonl", rows)
    kept = load_rows(path)
    assert kept == [rows[0], rows[2]]


def test_load_rows_limit_is_applied_before_the_empty_gold_filter(tmp_path, db):
    # First two rows have empty gold, last two do not. If the filter ran
    # before the limit, slicing to 2 afterwards would surface the two
    # non-empty rows. Reading the code shows it the other way round: rows are
    # sliced to `limit` first, so both survivors here are the empty-gold ones
    # and get dropped, leaving nothing kept.
    rows = [
        make_row(db, "empty one", GOLD_EMPTY),
        make_row(db, "empty two", GOLD_EMPTY),
        make_row(db, "real one", GOLD_NONEMPTY),
        make_row(db, "real two", "SELECT name FROM people WHERE age < 40"),
    ]
    path = write_jsonl(tmp_path / "rows.jsonl", rows)
    assert load_rows(path, limit=2) == []


def test_load_rows_raises_rather_than_dropping_a_gold_query_that_fails(tmp_path, db):
    # Distinct from the empty-gold case: "the answer is empty" (dropped) and
    # "we could not get the answer" (raised) are different facts. Collapsing
    # the second into the first would silently shrink the training set.
    rows = [make_row(db, "broken", "SELECT no_such_column FROM people")]
    path = write_jsonl(tmp_path / "rows.jsonl", rows)
    with pytest.raises(ValueError):
        load_rows(path)


# --------------------------------------------------------------------------
# assert_prompts_fit
# --------------------------------------------------------------------------


def test_assert_prompts_fit_raises_naming_the_count_and_the_longest_length(tokenizer, db):
    huge_schema = ", ".join(f"column_{i} INTEGER" for i in range(2000))
    row = make_row(db, "huge one", "SELECT name FROM people")
    row["messages"][1]["content"] = f"Schema: CREATE TABLE huge ({huge_schema})\nQuestion: huge one"
    dataset = build_dataset([row])

    with pytest.raises(ValueError, match=r"1 of 1 prompts exceed") as excinfo:
        assert_prompts_fit(dataset, tokenizer, max_prompt_length=100)
    # Names both numbers a fix would need: how many, and how far over.
    assert "max_prompt_length=100" in str(excinfo.value)
    assert "longest" in str(excinfo.value)


def test_assert_prompts_fit_passes_silently_when_everything_fits(tokenizer, db, capsys):
    rows = [make_row(db, "who is older than 40?", GOLD_NONEMPTY)]
    dataset = build_dataset(rows)
    assert_prompts_fit(dataset, tokenizer, max_prompt_length=512)
    assert "0 truncated" in capsys.readouterr().out


def test_assert_prompts_fit_does_not_raise_on_an_empty_dataset(tokenizer):
    # Pinning the documented behaviour. As written, `longest = max(lengths)`
    # is called unconditionally on an empty `lengths` list, which raises
    # `ValueError: max() iterable argument is empty` instead of passing
    # silently -- a genuine bug, left failing here rather than patched around.
    dataset = build_dataset([])
    assert_prompts_fit(dataset, tokenizer, max_prompt_length=512)


# --------------------------------------------------------------------------
# the truncation-side landmine
# --------------------------------------------------------------------------


def test_tokenizer_truncation_side_defaults_to_right(tokenizer):
    # `train()` sets `tokenizer.truncation_side = "left"` explicitly, because
    # TRL only configures this itself when *it* builds the tokenizer -- not
    # when one is passed in, as grpo_spider.py does. Left at the stock
    # default of "right", an over-length prompt keeps the schema and drops
    # the question, so the model is asked for SQL against a question it
    # cannot see. This test exists to fail loudly if a future transformers
    # release changes that default: it would mean the override in `train()`
    # has silently become either unnecessary or, worse, wrong.
    assert tokenizer.truncation_side == "right"
