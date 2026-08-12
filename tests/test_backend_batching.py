"""Tests for the batching in sqlrl.eval.backends.hf.

No model is loaded here. The point is the length-sort/unsort: if it is wrong,
every prediction is scored against a different question's gold query and the
whole benchmark is quietly meaningless. That deserves a test that runs in
milliseconds and always runs.
"""

from __future__ import annotations

import pytest

from sqlrl.eval.backends.hf import batched_in_order


def echo(texts: list[str]) -> list[str]:
    return [f"<{t}>" for t in texts]


def test_results_come_back_in_the_caller_order():
    # Deliberately reverse-sorted by length, so a missing unsort is obvious.
    items = ["aaaaaa", "aaaaa", "aaaa", "aaa", "aa", "a"]
    assert batched_in_order(items, 2, echo) == [f"<{item}>" for item in items]


def test_every_item_is_processed_exactly_once():
    items = [f"{'x' * (i % 7)}#{i}" for i in range(50)]
    seen: list[str] = []

    def record(texts: list[str]) -> list[str]:
        seen.extend(texts)
        return texts

    out = batched_in_order(items, 8, record)
    assert out == items
    assert sorted(seen) == sorted(items)


def test_batches_respect_the_size_limit():
    sizes: list[int] = []

    def measure(texts: list[str]) -> list[str]:
        sizes.append(len(texts))
        return texts

    batched_in_order([f"item{i}" for i in range(10)], 4, measure)
    assert sizes == [4, 4, 2]


def test_batches_are_grouped_by_length():
    # The whole reason for sorting: batches should be internally uniform, so
    # padding is minimal.
    batches: list[list[str]] = []

    def capture(texts: list[str]) -> list[str]:
        batches.append(list(texts))
        return texts

    items = ["a", "aaaaaaaa", "aa", "aaaaaaa", "aaa", "aaaaaa"]
    batched_in_order(items, 2, capture)

    # Batches are drawn in non-decreasing length order, so each one is as
    # internally uniform as the data allows.
    lengths = [len(text) for batch in batches for text in batch]
    assert lengths == sorted(lengths)


def test_duplicate_items_do_not_collide():
    # Position, not value, must decide where a result lands.
    items = ["same", "same", "same"]
    counter = iter(range(3))
    out = batched_in_order(items, 1, lambda t: [f"{t[0]}-{next(counter)}"])
    assert sorted(out) == ["same-0", "same-1", "same-2"]


def test_empty_input():
    assert batched_in_order([], 4, echo) == []


def test_backend_returning_wrong_count_is_an_error():
    with pytest.raises(ValueError, match="results for"):
        batched_in_order(["a", "b"], 2, lambda texts: ["only one"])
