"""Score predictions against gold SQL and aggregate the result into a report.

Four numbers, because one number cannot tell you *why* it is low:

``parse_rate``          fraction sqlglot can parse -- isolates malformed syntax
``execution_rate``      fraction SQLite accepts -- isolates invented tables/columns
``execution_accuracy``  the headline: same rows as the gold query
``structural_match``    same query shape, ignoring literal values and aliases

The gaps between them are the diagnosis. Parse 95% with execution 60% means the
model is inventing column names. Execution 90% with accuracy 30% means it writes
valid SQL that answers a different question. ``format_report`` spells those out.

Two honesty notes that matter when comparing against published numbers:

* ``structural_match`` is **not** Spider's official exact-set-match. Spider's
  metric runs their own SQL parser over a fixed grammar; ours canonicalises with
  sqlglot -- resolve aliases against the real schema, drop output aliases, blank
  literal values, sort the parts of the query that are sets. It answers the same
  question ("did it write the same query, ignoring cosmetics?") but the numbers
  are not interchangeable with published EM scores.
* Execution accuracy counts an **empty result matching an empty gold** as
  correct, so ``WHERE 1 = 0`` scores on any question whose true answer is empty.
  That is unfixable inside a comparison function, so it is measured instead:
  ``empty_gold`` counts those examples and ``execution_accuracy_nonempty``
  reports the score with them removed. If the two accuracies diverge, the
  headline is propped up by questions nobody actually answered.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import sqlglot
from sqlglot import expressions as exp
from sqlglot.optimizer.qualify import qualify
from sqlglot.optimizer.scope import traverse_scope

from sqlrl.eval.executor import compare, read_schema, requires_order, run

__all__ = [
    "ExampleScore",
    "Report",
    "aggregate",
    "classify_error",
    "format_report",
    "parses",
    "score_example",
    "structural_match",
]

_DIALECT = "sqlite"
_LITERAL = "<lit>"


# --------------------------------------------------------------------------
# individual signals
# --------------------------------------------------------------------------


def parses(sql: str) -> bool:
    """Can this be parsed as SQL at all? Isolates pure syntax failure."""
    try:
        return sqlglot.parse_one(sql, read=_DIALECT) is not None
    except Exception:  # noqa: BLE001 -- unparseable predictions are the point
        return False


def classify_error(status: str, error: str | None) -> str:
    """Bucket a failure so the report says *how* the model failed.

    ``unknown_column`` and ``unknown_table`` are schema hallucination;
    ``syntax`` is malformed output; ``not_a_query`` is an empty or non-SELECT
    prediction.
    """
    if status == "ok":
        return "ok"
    if status in ("timeout", "too_many_rows"):
        return status

    message = (error or "").lower()
    if "no result set" in message:
        return "not_a_query"
    if "no such table" in message:
        return "unknown_table"
    if "no such column" in message:
        return "unknown_column"
    if "ambiguous column" in message:
        return "ambiguous_column"
    if "no such function" in message:
        return "unknown_function"
    if any(
        token in message
        for token in ("syntax error", "unrecognized token", "incomplete input")
    ):
        return "syntax"
    return "other"


def structural_match(
    pred_sql: str, gold_sql: str, schema: dict[str, dict[str, str]]
) -> bool:
    """Same query shape, ignoring literal values, aliases and cosmetic order.

    See the module docstring: this is a sqlglot-based approximation, not
    Spider's official exact-set-match.
    """
    pred = _canonicalize(pred_sql, schema)
    gold = _canonicalize(gold_sql, schema)
    if pred is None or gold is None:
        return False
    # Only compare like with like. A prediction that failed to resolve against
    # the schema must not accidentally match a gold query that resolved fine.
    return pred == gold


def _canonicalize(sql: str, schema: dict[str, dict[str, str]]) -> tuple[str, bool] | None:
    """Canonical text plus a flag for whether it resolved against the schema."""
    try:
        tree = sqlglot.parse_one(sql, read=_DIALECT)
    except Exception:  # noqa: BLE001
        return None
    if tree is None:
        return None

    resolved = True
    try:
        # On a copy: qualify mutates as it goes, and a half-qualified tree would
        # make the canonical form depend on where the failure happened.
        tree = _inline_table_aliases(qualify(tree.copy(), schema=schema, dialect=_DIALECT))
    except Exception:  # noqa: BLE001 -- hallucinated columns fail to resolve
        resolved = False

    try:
        tree = _drop_output_aliases(tree)
        tree = _blank_literals(tree)
        tree = _sort_set_like_parts(tree)
        return tree.sql(dialect=_DIALECT, normalize=True).lower(), resolved
    except Exception:  # noqa: BLE001
        return None


def _inline_table_aliases(tree: exp.Expression) -> exp.Expression:
    """Rewrite ``people AS p ... p.name`` to ``people ... people.name``.

    ``qualify`` attaches every column to a source but keeps the alias, so
    without this the metric would mostly be measuring whether the model happens
    to use short aliases. Done per scope, so a column in a subquery resolves
    against that subquery's sources.
    """
    for scope in traverse_scope(tree):
        aliases = {}
        for name, source in scope.sources.items():
            if isinstance(source, exp.Table) and source.alias:
                aliases[name] = source.name
                source.set("alias", None)
        if not aliases:
            continue
        for column in scope.expression.find_all(exp.Column):
            if column.table in aliases:
                column.set("table", exp.to_identifier(aliases[column.table]))
    return tree


def _drop_output_aliases(tree: exp.Expression) -> exp.Expression:
    """``SELECT count(*) AS total`` and ``SELECT count(*)`` are the same answer."""
    for select in tree.find_all(exp.Select):
        select.set(
            "expressions",
            [e.this if isinstance(e, exp.Alias) else e for e in select.expressions],
        )
    return tree


def _blank_literals(tree: exp.Expression) -> exp.Expression:
    """Replace literal values with a placeholder -- the "ignoring values" part.

    ``LIMIT`` is exempt: "top 3" and "top 5" are different questions, not
    different spellings of one.
    """
    for literal in list(tree.find_all(exp.Literal)):
        if literal.find_ancestor(exp.Limit, exp.Offset) is not None:
            continue
        literal.replace(exp.Literal.string(_LITERAL))
    return tree


def _sort_set_like_parts(tree: exp.Expression) -> exp.Expression:
    """Sort the parts of a query where order carries no meaning.

    Selected columns, ``GROUP BY`` keys and ``AND`` conjuncts are sets.
    ``ORDER BY`` is deliberately left alone -- there, order *is* the meaning.
    """
    for select in tree.find_all(exp.Select):
        select.set("expressions", _sorted(select.expressions))

        group = select.args.get("group")
        if group is not None:
            group.set("expressions", _sorted(group.expressions))

        for clause in ("where", "having"):
            node = select.args.get(clause)
            if node is not None and isinstance(node.this, exp.And):
                node.set("this", exp.and_(*_sorted(list(node.this.flatten()))))
    return tree


def _sorted(nodes: list[exp.Expression]) -> list[exp.Expression]:
    return sorted(nodes, key=lambda node: node.sql(dialect=_DIALECT))


# --------------------------------------------------------------------------
# scoring one example
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ExampleScore:
    """Everything worth knowing about one prediction.

    ``gold_ok`` false means we could not compute the gold answer, so this
    example is not scoreable -- it is excluded from accuracy rather than
    counted as a miss, and reported separately so it cannot hide.
    """

    execution_match: bool
    structural_match: bool
    parsed: bool
    executed: bool
    gold_ok: bool
    gold_empty: bool
    pred_status: str
    error_kind: str


@lru_cache(maxsize=64)
def _schema_for(db_path: str) -> tuple[tuple[str, tuple[tuple[str, str], ...]], ...]:
    """Cached, hashable schema. Spider dev reuses 20 databases across 1,034 questions."""
    schema = read_schema(db_path)
    return tuple(
        (table, tuple(columns.items())) for table, columns in sorted(schema.items())
    )


def _schema_dict(db_path: str | Path) -> dict[str, dict[str, str]]:
    return {table: dict(columns) for table, columns in _schema_for(str(db_path))}


def score_example(
    pred_sql: str,
    gold_sql: str,
    db_path: str | Path,
    *,
    timeout: float = 5.0,
    dedupe: bool = False,
) -> ExampleScore:
    """Run both queries and produce every signal for this one example."""
    gold = run(gold_sql, db_path, timeout=timeout)
    pred = run(pred_sql, db_path, timeout=timeout)

    gold_ok = gold.ok
    execution_match = gold_ok and pred.ok and compare(
        pred.rows, gold.rows, requires_order(gold_sql), dedupe=dedupe
    )

    schema = _schema_dict(db_path)
    return ExampleScore(
        execution_match=execution_match,
        structural_match=structural_match(pred_sql, gold_sql, schema),
        parsed=parses(pred_sql),
        # A timeout or an oversized result is not a *rejection* -- the database
        # accepted the query. Only "error" means SQLite refused it.
        executed=pred.status != "error",
        gold_ok=gold_ok,
        gold_empty=gold_ok and not gold.rows,
        pred_status=pred.status,
        error_kind=classify_error(pred.status, pred.error),
    )


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Report:
    n: int
    scored: int
    gold_failures: int
    empty_gold: int
    execution_accuracy: float
    execution_accuracy_nonempty: float
    structural_match: float
    parse_rate: float
    execution_rate: float
    error_kinds: dict[str, int] = field(default_factory=dict)


def aggregate(scores: list[ExampleScore]) -> Report:
    n = len(scores)
    scoreable = [s for s in scores if s.gold_ok]
    nonempty = [s for s in scoreable if not s.gold_empty]

    return Report(
        n=n,
        scored=len(scoreable),
        gold_failures=n - len(scoreable),
        empty_gold=sum(s.gold_empty for s in scoreable),
        execution_accuracy=_rate(sum(s.execution_match for s in scoreable), len(scoreable)),
        execution_accuracy_nonempty=_rate(
            sum(s.execution_match for s in nonempty), len(nonempty)
        ),
        structural_match=_rate(sum(s.structural_match for s in scores), n),
        parse_rate=_rate(sum(s.parsed for s in scores), n),
        execution_rate=_rate(sum(s.executed for s in scores), n),
        error_kinds=dict(
            Counter(s.error_kind for s in scores if s.error_kind != "ok").most_common()
        ),
    )


def _rate(hits: int, total: int) -> float:
    return hits / total if total else 0.0


def format_report(report: Report, title: str = "") -> str:
    """A human-readable block: the metrics, the failure breakdown, the diagnosis."""
    lines: list[str] = []
    if title:
        lines += [title, "=" * len(title)]

    lines.append(f"{report.n} examples, {report.scored} scoreable")
    if report.gold_failures:
        lines.append(
            f"  !! {report.gold_failures} gold queries did not run -- excluded from "
            f"accuracy. Investigate before trusting these numbers."
        )

    lines += [
        "",
        f"  parse rate           {report.parse_rate:6.1%}",
        f"  execution rate       {report.execution_rate:6.1%}",
        f"  structural match     {report.structural_match:6.1%}   (not Spider EM)",
        f"  execution accuracy   {report.execution_accuracy:6.1%}   <- headline",
        f"    excl. empty gold   {report.execution_accuracy_nonempty:6.1%}   "
        f"({report.empty_gold} of {report.scored} gold answers are empty)",
    ]

    if report.error_kinds:
        lines += ["", "  failures:"]
        lines += [f"    {kind:<18} {count}" for kind, count in report.error_kinds.items()]

    notes = _diagnose(report)
    if notes:
        lines += ["", "  reading:"] + [f"    - {note}" for note in notes]
    return "\n".join(lines)


def _diagnose(report: Report) -> list[str]:
    notes = []
    if report.parse_rate - report.execution_rate > 0.15:
        notes.append(
            "parses but will not execute -- the model is inventing tables or columns; "
            "the schema is not reaching it clearly enough"
        )
    if report.execution_rate - report.execution_accuracy > 0.3:
        notes.append(
            "executes but returns the wrong rows -- valid SQL answering a different "
            "question; a reasoning problem, not a syntax one"
        )
    if report.execution_accuracy - report.execution_accuracy_nonempty > 0.05:
        notes.append(
            "the headline is propped up by empty results matching empty gold answers; "
            "trust the excl.-empty number"
        )
    if report.execution_accuracy - report.structural_match > 0.2:
        notes.append(
            "right rows from differently shaped queries -- the model is finding its own "
            "path rather than reproducing gold, which is fine"
        )
    return notes
