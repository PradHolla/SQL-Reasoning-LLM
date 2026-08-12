"""Run SQL against a SQLite database and decide whether two result sets match.

This is the measuring stick for the whole project. Phase 1 uses it to compute
execution accuracy; Phase 2 calls the same two functions to compute the GRPO
reward. So it has no ML dependencies, no imports from the rest of ``sqlrl``, and
it never raises on bad model output -- a query the model invented is *data*, not
an error condition.

    run(sql, db_path, timeout)                -> ExecResult(status, rows, error)
    compare(pred_rows, gold_rows, ordered)    -> bool
    requires_order(gold_sql)                  -> bool

Decisions worth knowing about, because each one is a way this file could quietly
report a wrong number:

* **The database is opened read-only.** The model can emit ``DROP TABLE``. If
  that ever landed, every subsequent evaluation against that database would be
  wrong and nothing would tell us.
* **The timeout is enforced by a progress handler, not** ``sqlite3.connect(
  timeout=...)``, which only bounds *lock* acquisition and would let a runaway
  join spin forever.
* **Rows are compared as a multiset, not a set.** Set comparison makes
  ``SELECT name`` and ``SELECT DISTINCT name`` indistinguishable, which inflates
  the score. Spider's official script uses set semantics, so our numbers read
  slightly lower than published ones; ``dedupe=True`` reproduces their behaviour
  when we need an apples-to-apples comparison.
* **Column order is ignored by default** (bounded search over column
  permutations), matching the Spider test-suite convention: answering with
  ``age, name`` instead of ``name, age`` is a presentation difference.
* **Row order only matters when the gold query says so** -- see
  ``requires_order``.
* **Numeric cells are compared to 9 significant digits**, so ``SUM(x)/COUNT(x)``
  and ``AVG(x)`` agree despite float noise. Strings are compared exactly; both
  queries read the same database, so any string difference is a real one.

Known limitation, deliberately not fixed here: an empty result set matching an
empty gold result counts as correct, so ``WHERE 1=0`` scores on any question
whose gold answer is empty. That cannot be fixed inside a comparison function --
it needs either multiple database instances (Spider's test-suite approach) or
reporting the empty-gold slice separately. ``metrics.py`` owns that; it has both
row sets and can see ``len(gold_rows) == 0``.
"""

from __future__ import annotations

import math
import re
import sqlite3
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterator, Literal, NamedTuple

import sqlglot
from sqlglot import expressions as exp

__all__ = [
    "ExecResult",
    "Status",
    "compare",
    "parse_sql",
    "read_schema",
    "requires_order",
    "run",
]

Status = Literal["ok", "error", "timeout", "too_many_rows"]

#: How many SQLite VM instructions between deadline checks. Small enough that a
#: runaway query notices quickly, large enough that the check costs nothing.
_PROGRESS_OPS = 1_000

#: Significant digits kept when comparing floats.
_FLOAT_SIGDIGITS = 9

#: NaN never equals itself, which would make a row containing one unmatchable.
_NAN = "\x00nan"

#: Ceiling on the column-permutation search, so a pathological wide result with
#: many identical columns cannot stall the eval loop.
_PERMUTATION_BUDGET = 20_000

#: Guards against sqlglot's parser, whose cost is exponential in the number of
#: ON-less JOINs: 12 joins parse in 0.04s, 16 in 0.68s, 18 in 2.8s, 20 in 11.2s,
#: and a few more never finish. A degenerate model output really does hang the
#: evaluation -- this is not hypothetical, it stopped a run dead.
#:
#: The limits are set from the benchmark itself. Across all 3,181 Spider gold
#: queries the longest is 608 characters and the most JOINs is 6, so these
#: leave large headroom over anything legitimate while capping parse time in
#: the milliseconds. Input beyond them is not "too complex to parse", it is
#: model garbage, and reporting it as unparseable is the honest answer.
MAX_SQL_CHARS = 2_000
MAX_JOINS = 10

_JOIN = re.compile(r"\bJOIN\b", re.IGNORECASE)

Row = tuple[Any, ...]


class ExecResult(NamedTuple):
    """Outcome of running one query.

    ``status`` is the thing to branch on:

    ``ok``              ran to completion; ``rows`` is the full result
    ``error``           SQLite rejected it -- syntax, missing table, missing column
    ``timeout``         exceeded the time budget
    ``too_many_rows``   more rows than ``max_rows``; ``rows`` is truncated

    The ``max_rows`` default is high because real gold queries can be large --
    one Spider gold query returns 20,662 rows. The cap exists to stop a
    hallucinated cross join from exhausting memory, not to bound honest results.
    """

    status: Status
    rows: list[Row]
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def run(
    sql: str,
    db_path: str | Path,
    timeout: float = 5.0,
    max_rows: int = 100_000,
) -> ExecResult:
    """Execute ``sql`` against the SQLite file at ``db_path``.

    Never raises for anything the model could have caused -- bad SQL comes back
    as ``status="error"``. A missing database file *does* raise, because that is
    our bug, not the model's, and silently scoring it as a miss would corrupt
    every number downstream.
    """
    path = Path(db_path)
    if not path.is_file():
        raise FileNotFoundError(f"No SQLite database at {path!r}")

    # mode=ro is the safety belt: model-authored SQL cannot alter the benchmark.
    conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)

    # Several Spider databases contain byte sequences that are not valid UTF-8.
    # The default text factory raises on them mid-fetch, which would look like a
    # query error. Replacing is fine: gold and prediction read the same bytes.
    conn.text_factory = lambda b: b.decode("utf-8", "replace")

    deadline = time.monotonic() + timeout
    timed_out = False

    def watchdog() -> int:
        # Returning non-zero aborts the statement. sqlite3.connect(timeout=...)
        # would not help here -- it bounds lock waits, not query runtime.
        nonlocal timed_out
        if time.monotonic() > deadline:
            timed_out = True
            return 1
        return 0

    conn.set_progress_handler(watchdog, _PROGRESS_OPS)

    try:
        cursor = conn.execute(sql)
        if cursor.description is None:
            # No result set at all: empty string, comments only, or a statement
            # that is not a query. sqlite3 reports these as a successful run of
            # zero rows, which would then compare equal to any empty gold
            # result -- a free point for producing nothing.
            return ExecResult("error", [], "query produced no result set")
        # One extra row tells us the cap was hit without materialising the rest.
        rows = cursor.fetchmany(max_rows + 1)
        if len(rows) > max_rows:
            return ExecResult(
                "too_many_rows", rows[:max_rows], f"more than {max_rows} rows"
            )
        return ExecResult("ok", rows)
    except Exception as exc:  # noqa: BLE001 -- model output is data, not a fault
        if timed_out:
            return ExecResult("timeout", [], f"exceeded {timeout}s")
        return ExecResult("error", [], f"{type(exc).__name__}: {exc}")
    finally:
        conn.close()


def read_schema(db_path: str | Path) -> dict[str, dict[str, str]]:
    """``{table: {column: type}}`` read straight from the database file.

    Used to resolve table aliases when comparing query structure, and later to
    build prompts. Reading the live database rather than a benchmark's metadata
    file means the schema we show the model is the schema its SQL will run
    against -- there is no second source of truth to drift.
    """
    path = Path(db_path)
    if not path.is_file():
        raise FileNotFoundError(f"No SQLite database at {path!r}")

    conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    conn.text_factory = lambda b: b.decode("utf-8", "replace")
    try:
        tables = [
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            )
        ]
        return {
            table: {
                row[1]: row[2]
                # Identifier, not a value, so it cannot be parameterised. The
                # names come from sqlite_master, not from model output.
                for row in conn.execute(f'PRAGMA table_info("{table}")')
            }
            for table in tables
        }
    finally:
        conn.close()


def parse_sql(sql: str) -> exp.Expression | None:
    """Parse SQL, refusing input that would make the parser blow up.

    Returns ``None`` for anything unparseable *or* pathological. Every sqlglot
    call in this project goes through here -- the executor has a timeout, but a
    parser that never returns cannot be interrupted, and this same function is
    what will run inside the Phase 2 reward loop on every rollout.
    """
    if not sql or len(sql) > MAX_SQL_CHARS:
        return None
    if len(_JOIN.findall(sql)) > MAX_JOINS:
        return None
    try:
        return sqlglot.parse_one(sql, read="sqlite")
    except Exception:  # noqa: BLE001 -- unparseable predictions are expected
        return None


def requires_order(sql: str) -> bool:
    """Does this query's result have a meaningful row order?

    True when the query contains an ``ORDER BY`` that is not part of a window
    function. Errs strict: an ``ORDER BY`` buried in a subquery counts, even
    though the outer result may be unordered. Being too strict can only cost us
    credit for a right answer; being too lax would hand out credit for a wrong
    one.
    """
    tree = parse_sql(sql)
    if tree is None:
        # Fall back to the crude check rather than declaring the query unordered.
        return "order by" in sql.lower()
    return any(
        node.find_ancestor(exp.Window) is None for node in tree.find_all(exp.Order)
    )


def compare(
    pred_rows: list[Row],
    gold_rows: list[Row],
    order_matters: bool = False,
    *,
    dedupe: bool = False,
    column_order_matters: bool = False,
) -> bool:
    """Do two result sets represent the same answer?

    ``order_matters`` should come from ``requires_order(gold_sql)``.
    ``dedupe=True`` switches to Spider's official set semantics.
    ``column_order_matters=True`` disables the column-permutation search.
    """
    pred = [tuple(_cell_key(cell) for cell in row) for row in pred_rows]
    gold = [tuple(_cell_key(cell) for cell in row) for row in gold_rows]

    if dedupe:
        pred = _unique(pred)
        gold = _unique(gold)

    if len(pred) != len(gold):
        return False
    if not gold:  # both empty -- see the empty-result caveat in the module docstring
        return True

    n_cols = len(gold[0])
    if len(pred[0]) != n_cols:
        return False

    if column_order_matters or n_cols == 1:
        return _rows_equal(pred, gold, order_matters)

    for perm in _column_permutations(pred, gold, n_cols):
        permuted = [tuple(row[i] for i in perm) for row in pred]
        if _rows_equal(permuted, gold, order_matters):
            return True
    return False


def _rows_equal(pred: list[Row], gold: list[Row], order_matters: bool) -> bool:
    if order_matters:
        # Note: ties under the gold ORDER BY have an arbitrary order, so two
        # equally correct queries can disagree here. Accepted for now.
        return pred == gold
    return Counter(pred) == Counter(gold)


def _unique(rows: list[Row]) -> list[Row]:
    """Deduplicate, preserving first-seen order so ordered comparison survives."""
    seen: set[Row] = set()
    out: list[Row] = []
    for row in rows:
        if row not in seen:
            seen.add(row)
            out.append(row)
    return out


def _column_permutations(
    pred: list[Row], gold: list[Row], n_cols: int
) -> Iterator[tuple[int, ...]]:
    """Yield candidate column mappings, cheapest-to-reject first.

    ``perm[j]`` is the prediction column that should supply gold column ``j``.
    Only columns whose value multisets already match are considered, which
    collapses the search to a handful of candidates on real queries.
    """
    pred_cols = [Counter(row[i] for row in pred) for i in range(n_cols)]
    gold_cols = [Counter(row[j] for row in gold) for j in range(n_cols)]

    candidates = [
        [i for i in range(n_cols) if pred_cols[i] == gold_cols[j]]
        for j in range(n_cols)
    ]
    if any(not choices for choices in candidates):
        return

    budget = _PERMUTATION_BUDGET
    used = [False] * n_cols
    current: list[int] = []

    def walk(j: int) -> Iterator[tuple[int, ...]]:
        nonlocal budget
        if j == n_cols:
            yield tuple(current)
            return
        for i in candidates[j]:
            if used[i]:
                continue
            budget -= 1
            if budget < 0:
                return
            used[i] = True
            current.append(i)
            yield from walk(j + 1)
            current.pop()
            used[i] = False

    yield from walk(0)


def _cell_key(value: Any) -> Any:
    """Canonical form of one cell, so equal values hash and compare equally.

    Integers and floats collapse onto each other for free in Python
    (``3 == 3.0`` and their hashes match), so ``COUNT(*)`` and ``1.0*COUNT(*)``
    agree. Rounding to significant digits absorbs float noise between
    mathematically identical aggregates.
    """
    if isinstance(value, float):
        if math.isnan(value):
            return _NAN
        if math.isinf(value):
            return value
        return float(f"{value:.{_FLOAT_SIGDIGITS}g}")
    return value
