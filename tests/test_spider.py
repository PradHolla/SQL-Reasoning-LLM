"""Tests for sqlrl.eval.spider.

The normalisation tests matter more than they look: the contamination result --
562 of 1,034 dev questions leaked -- is only as trustworthy as these two
functions. Too aggressive and it invents overlap; too timid and it misses real
leakage and declares a dirty benchmark clean.

Tests that need the ~200 MB download are skipped when it is not present.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sqlrl.eval.spider import (
    DEFAULT_ROOT,
    SPLITS,
    _norm_question,
    _norm_sql,
    contaminated_indices,
    load_split,
)

needs_data = pytest.mark.skipif(
    not (DEFAULT_ROOT / "spider_data" / "test.json").is_file(),
    reason="Spider not downloaded -- run `python -m sqlrl.eval.spider`",
)


# --------------------------------------------------------------------------
# normalisation
# --------------------------------------------------------------------------


def test_question_normalisation_ignores_case_and_punctuation():
    assert _norm_question("How many singers do we have?") == _norm_question(
        "how many singers do we have"
    )


def test_question_normalisation_collapses_whitespace():
    assert _norm_question("  How   many\nsingers ?  ") == "how many singers"


def test_question_normalisation_keeps_different_questions_apart():
    # The failure that would matter: over-normalising into false contamination.
    assert _norm_question("How many singers are there?") != _norm_question(
        "How many songs are there?"
    )
    assert _norm_question("singers over 30") != _norm_question("singers over 40")


def test_sql_normalisation_ignores_whitespace_and_case():
    assert _norm_sql("SELECT  count(*)   FROM singer ;") == _norm_sql(
        "select count(*) from singer"
    )


def test_sql_normalisation_ignores_spacing_around_punctuation():
    assert _norm_sql("SELECT avg(age) , min(age) FROM singer") == _norm_sql(
        "SELECT avg(age), min(age) FROM singer"
    )


def test_sql_normalisation_keeps_different_queries_apart():
    assert _norm_sql("SELECT avg(age) FROM singer") != _norm_sql(
        "SELECT max(age) FROM singer"
    )


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------


def test_unknown_split_is_rejected():
    with pytest.raises(ValueError):
        load_split("train")


@needs_data
@pytest.mark.parametrize("split, n, dbs", [("dev", 1034, 20), ("test", 2147, 40)])
def test_split_sizes(split, n, dbs):
    examples = load_split(split)
    assert len(examples) == n
    assert len({ex.db_id for ex in examples}) == dbs
    assert all(ex.db_path.is_file() for ex in examples)


@needs_data
def test_splits_do_not_share_databases():
    # Why test is a usable held-out set at all.
    dev = {ex.db_id for ex in load_split("dev")}
    test = {ex.db_id for ex in load_split("test")}
    assert dev & test == set()


@needs_data
def test_dev_is_heavily_contaminated():
    # Not a happy test. It pins the finding so it cannot quietly regress into
    # an assumption again.
    dev = load_split("dev")
    dirty = [ex for ex in dev if ex.contaminated]
    assert len(dirty) > 500
    assert any(ex.question == "How many singers do we have?" for ex in dirty)


@needs_data
def test_test_split_is_nearly_clean():
    test = load_split("test")
    dirty = [ex for ex in test if ex.contaminated]
    assert len(dirty) / len(test) < 0.02


@needs_data
def test_clean_only_drops_exactly_the_contaminated():
    full = load_split("dev")
    clean = load_split("dev", clean_only=True)
    assert len(clean) == sum(not ex.contaminated for ex in full)
    assert not any(ex.contaminated for ex in clean)


@needs_data
@pytest.mark.parametrize("split", SPLITS)
def test_contamination_cache_is_stable(split):
    first = contaminated_indices(split)
    assert first == contaminated_indices(split)
    assert (Path(DEFAULT_ROOT) / f"contamination_{split}.json").is_file()
