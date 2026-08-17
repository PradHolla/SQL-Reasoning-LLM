"""Tests for sqlrl.eval.retrieval.

No GPU, no model: ``TableDoc`` lists are built by hand, the same pattern
tests/test_voting.py uses for ``Candidate``s. The one exception is
``build_pool``, which reads the ~206 real Spider test database files -- that
is the one thing worth checking against real data, since it is the property
the whole retrieval design rests on (see the module docstring), and it is
cheap enough to do once.
"""

from __future__ import annotations

import pytest

from sqlrl.eval.executor import parse_sql
from sqlrl.eval.retrieval import (
    BM25,
    TableDoc,
    build_pool,
    coverage_at_k,
    gold_tables,
    pool_questions,
    recall_at_k,
    render_pool_schema,
    tokenize,
)
from sqlrl.eval.spider import DEFAULT_ROOT, load_split

needs_data = pytest.mark.skipif(
    not (DEFAULT_ROOT / "spider_data" / "test.json").is_file(),
    reason="Spider not downloaded -- run `python -m sqlrl.eval.spider`",
)


# --------------------------------------------------------------------------
# tokenize
# --------------------------------------------------------------------------


def test_tokenize_splits_snake_case_and_keeps_joined_form():
    assert tokenize("city_code") == ["city_code", "city", "code"]


def test_tokenize_splits_camel_case_and_keeps_joined_form():
    assert tokenize("singerID") == ["singerid", "singer", "id"]


def test_tokenize_lowercases_and_splits_on_punctuation():
    assert tokenize("How many Singers?") == ["how", "many", "singers"]


def test_tokenize_does_not_duplicate_a_plain_word_within_one_occurrence():
    # "student" has no separator and no case change, so its joined form and
    # its only "part" are identical -- must collapse to one token, not two,
    # or its own term frequency would be inflated for nothing.
    assert tokenize("student") == ["student"]
    # Two real occurrences must still produce two tokens.
    assert tokenize("student student") == ["student", "student"]


# --------------------------------------------------------------------------
# TableDoc
# --------------------------------------------------------------------------


def _doc(db_id: str, table: str, columns: tuple[str, ...]) -> TableDoc:
    return TableDoc(db_id=db_id, table=table, columns=columns)


def test_table_doc_text_is_table_colon_columns():
    doc = _doc("school", "student", ("id", "name", "age", "dept_id"))
    assert doc.text == "student: id, name, age, dept_id"


def test_table_doc_key_lowercases_the_table_name_only():
    doc = _doc("school", "Student", ("id",))
    assert doc.key == ("school", "student")
    # db_id itself is not lowercased -- Spider db_ids are already lowercase
    # and consistent, unlike table names inside gold SQL.
    assert doc.key[0] == "school"


# --------------------------------------------------------------------------
# BM25
# --------------------------------------------------------------------------


def test_bm25_ranks_an_obviously_matching_table_first():
    docs = [
        _doc("db", "singer", ("id", "name", "age", "country")),
        _doc("db", "weather", ("city", "temperature", "date")),
    ]
    bm25 = BM25(docs)

    results = bm25.search("What are the names of all singers?", k=2)

    assert results[0] == 0  # singer, not weather


def test_bm25_is_deterministic_and_ties_break_by_index_ascending():
    # Identical text -> identical score for any query, so only the tie-break
    # rule decides the order.
    docs = [
        _doc("db1", "people", ("id", "name")),
        _doc("db2", "people", ("id", "name")),
    ]
    bm25 = BM25(docs)

    first = bm25.search("how many people", k=2)
    second = bm25.search("how many people", k=2)

    assert first == second == [0, 1]


def test_bm25_returns_at_most_k():
    docs = [_doc("db", f"t{i}", ("a", "b")) for i in range(5)]
    bm25 = BM25(docs)

    results = bm25.search("a b", k=2)

    assert len(results) == 2


def test_bm25_handles_k_larger_than_the_corpus():
    docs = [_doc("db", "t1", ("a",)), _doc("db", "t2", ("b",))]
    bm25 = BM25(docs)

    results = bm25.search("a", k=50)

    assert len(results) == 2
    assert set(results) == {0, 1}


# --------------------------------------------------------------------------
# gold_tables
# --------------------------------------------------------------------------


def test_gold_tables_simple_query():
    assert gold_tables("SELECT name FROM singer", "concert_singer") == {
        ("concert_singer", "singer")
    }


def test_gold_tables_join_returns_both_tables():
    sql = (
        "SELECT s.name FROM singer AS s "
        "JOIN concert AS c ON s.id = c.singer_id"
    )
    assert gold_tables(sql, "concert_singer") == {
        ("concert_singer", "singer"),
        ("concert_singer", "concert"),
    }


def test_gold_tables_unparseable_input_returns_empty_set_without_raising():
    assert gold_tables("", "concert_singer") == set()
    assert gold_tables("this is not sql at all !!! ((((", "concert_singer") == set()


# --------------------------------------------------------------------------
# recall_at_k
# --------------------------------------------------------------------------


def test_recall_at_k_full_when_every_gold_table_is_retrieved():
    retrieved = [_doc("db", "singer", ("id",)), _doc("db", "concert", ("id",))]
    gold = {("db", "singer"), ("db", "concert")}
    assert recall_at_k(retrieved, gold) == 1.0


def test_recall_at_k_partial_when_only_some_gold_tables_are_retrieved():
    retrieved = [_doc("db", "singer", ("id",)), _doc("db", "weather", ("id",))]
    gold = {("db", "singer"), ("db", "concert")}
    assert recall_at_k(retrieved, gold) == 0.5


def test_recall_at_k_zero_when_no_gold_table_is_retrieved():
    retrieved = [_doc("db", "weather", ("id",))]
    gold = {("db", "singer"), ("db", "concert")}
    assert recall_at_k(retrieved, gold) == 0.0


def test_recall_at_k_empty_gold_is_perfect_recall():
    # Nothing was required, so nothing was missed -- an unparseable gold
    # query must not drag the mean down for a question the metric has no
    # opinion about.
    assert recall_at_k([], set()) == 1.0
    assert recall_at_k([_doc("db", "singer", ("id",))], set()) == 1.0


# --------------------------------------------------------------------------
# coverage_at_k
# --------------------------------------------------------------------------


def test_coverage_at_k_true_only_when_every_gold_table_present():
    retrieved = [_doc("db", "singer", ("id",)), _doc("db", "concert", ("id",))]
    gold = {("db", "singer"), ("db", "concert")}
    assert coverage_at_k(retrieved, gold) is True


def test_coverage_at_k_false_when_one_of_two_gold_tables_is_missing():
    # A two-table join missing one table is unanswerable -- the case this
    # metric exists to catch, even though recall_at_k on the same input would
    # read a non-zero 0.5.
    retrieved = [_doc("db", "singer", ("id",))]
    gold = {("db", "singer"), ("db", "concert")}
    assert coverage_at_k(retrieved, gold) is False
    assert recall_at_k(retrieved, gold) == 0.5


def test_coverage_at_k_empty_gold_is_vacuously_covered():
    assert coverage_at_k([], set()) is True


# --------------------------------------------------------------------------
# render_pool_schema
# --------------------------------------------------------------------------


def test_render_pool_schema_parses_and_contains_every_table():
    docs = [
        _doc("school", "student", ("id", "name")),
        _doc("company", "employee", ("id", "salary")),
    ]

    schema_sql = render_pool_schema(docs)

    assert parse_sql(schema_sql) is not None
    assert "CREATE TABLE student" in schema_sql
    assert "CREATE TABLE employee" in schema_sql


def test_render_pool_schema_defaults_every_column_to_text():
    docs = [_doc("school", "student", ("id", "name"))]
    assert render_pool_schema(docs) == "CREATE TABLE student (id TEXT, name TEXT)"


def test_render_pool_schema_uses_real_types_when_given():
    docs = [_doc("school", "student", ("id", "name"))]
    types = {("school", "student"): {"id": "INTEGER", "name": "VARCHAR(50)"}}

    schema_sql = render_pool_schema(docs, types)

    assert schema_sql == "CREATE TABLE student (id INTEGER, name VARCHAR(50))"


# --------------------------------------------------------------------------
# build_pool / pool_questions -- real Spider data
# --------------------------------------------------------------------------


@needs_data
def test_build_pool_has_no_duplicate_table_names():
    # The property the whole design rests on: no two kept databases share a
    # table name, or a question about "customers" would be unresolvable by
    # any retriever, however good -- see the module docstring.
    pool = build_pool("test")

    assert len(pool) > 0
    assert len({doc.table.lower() for doc in pool}) == len(pool)


@needs_data
def test_pool_questions_keeps_only_examples_in_the_pool():
    pool = build_pool("test")
    examples = load_split("test")

    pooled = pool_questions(examples, pool)

    pooled_dbs = {doc.db_id for doc in pool}
    assert len(pooled) > 0
    assert len(pooled) < len(examples)  # a strict subset, not everything
    assert pooled == [example for example in examples if example.db_id in pooled_dbs]
