"""What can a policy extract from the GRPO reward without ever learning SQL?

    uv run python -m sqlrl.training.reward_probe
    uv run python -m sqlrl.training.reward_probe --limit 100

A reward curve that rises while the benchmark stays flat is the failure mode
this phase exists to avoid, and by the time a training run shows it we have
already spent the GPU hours. So before training, run a set of fixed adversarial
"policies" -- functions from a dataset row to a completion, none of which look
at the question -- against every example in the split and see what the ladder
in ``rewards.py`` pays them.

The number that matters is not any policy's mean reward on its own, it is the
mean *relative to* ``gold``. ``gold`` is the ceiling: a policy that reaches a
large fraction of it while answering nothing is a policy GRPO can find, because
GRPO is a search and these degenerate templates are far easier to stumble into
than correct SQL. Anything close to the ceiling is a hole in the reward.

Five things this deliberately measures rather than assumes:

* **``gold`` itself.** It must score at the top tier nearly everywhere. If it
  does not, the reward and the Phase 1 benchmark disagree about what "correct"
  means, and every conclusion below -- and every reward curve after it -- is
  measuring the wrong thing. That is reported loudly.
* **What a fixed degenerate template pays.** The eleven policies below.
* **What that template pays a *searcher*.** ``template_oracle`` takes the best
  filling of a template per question, because GRPO samples eight completions
  per prompt and follows whichever scored highest. A single fixed filling
  understates what the trainer can reach.
* **The zero-row hack.** ``WHERE 1=0`` is the classic exploit and ``rewards.py``
  defends against it by refusing to pay the match tier when gold is empty.
  ``empty_gold_counterfactual`` re-scores those same policies with the defence
  switched off, so the size of what the defence is holding back is a measured
  number rather than an argument.
* **Cost.** GRPO scores ``batch x num_generations`` completions per optimiser
  step, synchronously, on the training process. If that is slow the reward is a
  throughput tax on every step of the run, so the probe times a realistic step.

One thing it computes rather than measures: ``advantage_table`` shows what
TRL's group normaliser does to the ladder's rung spacing. That is arithmetic,
not an experiment, but it belongs here because it decides how much of what the
tables above show actually reaches the gradient.

Not a framework. One flag, no config, prints a table. If it grows a plugin
system it has stopped being a diagnostic.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from sqlglot import expressions as exp

from sqlrl.eval.executor import parse_sql, read_schema
from sqlrl.training.rewards import OUTCOMES, SQLReward

__all__ = ["POLICIES", "PolicyResult", "main", "run_policy"]

DATASET = Path("data/processed/spider_grpo.jsonl")

#: A realistic optimiser step: TRL scores every generation of every prompt in
#: the batch before it can compute advantages, so this is the whole per-step
#: reward cost, not a per-sample one.
STEP_PROMPTS = 8
STEP_GENERATIONS = 8

Row = Mapping[str, Any]
Policy = Callable[[Row], str]


# --------------------------------------------------------------------------
# schemas, read once per database
# --------------------------------------------------------------------------

#: 1,035 questions over 22 databases. Re-opening a SQLite file per question
#: would make the probe's own setup cost dominate the reward cost it is trying
#: to measure.
_SCHEMAS: dict[str, dict[str, dict[str, str]]] = {}


def schema(db_path: str) -> dict[str, dict[str, str]]:
    cached = _SCHEMAS.get(db_path)
    if cached is None:
        cached = _SCHEMAS[db_path] = read_schema(db_path)
    return cached


def tables(db_path: str) -> list[str]:
    """Table names in ``sqlite_master`` order.

    "First table" below means first in this order. It is creation order, which
    is stable for a given file and has nothing to do with the question -- that
    is the point. These policies are meant to be cheap guesses, not good ones.
    """
    names = list(schema(db_path))
    if not names:
        raise ValueError(f"no tables in {db_path}; cannot build a probe query for it")
    return names


def quote(name: str) -> str:
    """Double-quote an identifier. Some Spider tables and columns have spaces."""
    return '"' + name.replace('"', '""') + '"'


# --------------------------------------------------------------------------
# the policies
# --------------------------------------------------------------------------


def answer(sql: str) -> str:
    """Wrap SQL the way a policy trained by SFT emits it.

    The reward has no format term -- ``extract_sql`` is lenient enough that
    bare SQL scores identically -- so the tags are here only so the completions
    look like what the trainer will actually be handed.
    """
    return f"<think>\nThe schema gives me what I need.\n</think>\n<answer>\n{sql}\n</answer>"


def gold(row: Row) -> str:
    """The ceiling, and the sanity check. Must reach the match tier nearly always."""
    return answer(row["gold_sql"])


def empty_string(row: Row) -> str:
    """The floor: produce nothing at all."""
    return ""


def prose(row: Row) -> str:
    """A plausible English refusal containing no SQL.

    Word choice is not casual. ``extract_sql`` slices from the first ``SELECT``
    or ``WITH`` token anywhere in the completion, so an otherwise SQL-free
    sentence containing the word "with" -- "I cannot help with that" -- comes
    back as the fragment ``with that.``. That still scores zero (it will not
    parse), so it is a curiosity rather than an exploit, but a probe that
    tripped over it would be measuring the extractor instead of the reward.
    """
    return (
        "<think>\nThe question is ambiguous and I am not confident about the "
        "column meanings.\n</think>\n<answer>\nI am unable to produce a query "
        "for this question from the information given.\n</answer>"
    )


def select_1(row: Row) -> str:
    """Valid SQL that touches no table. Parses and runs everywhere."""
    return answer("SELECT 1")


def select_star(row: Row) -> str:
    """Dump a table. Almost always more rows than gold -- what does that pay?"""
    return answer(f"SELECT * FROM {quote(tables(row['db_path'])[0])}")


def count_star(row: Row) -> str:
    """One row, one integer. Spider is full of "how many X" questions, so this
    is the degenerate template most likely to match a gold answer by luck."""
    return answer(f"SELECT COUNT(*) FROM {quote(tables(row['db_path'])[0])}")


def empty_result(row: Row) -> str:
    """The classic zero-row hack, in its cheapest form."""
    return answer("SELECT 1 WHERE 1=0")


def gold_limit_zero(row: Row) -> str:
    """The zero-row hack again, but shaped like the gold answer.

    Same column count as gold and zero rows, so it is the strongest possible
    version of the exploit: whatever ``compare`` would accept from an empty
    result, this gets. Worth separating from ``empty_result`` because if the
    two score differently, the reward is sensitive to the *shape* of the empty
    result and not just to its emptiness.
    """
    return answer(f"SELECT * FROM ({row['gold_sql']}) LIMIT 0")


#: Not in any Spider schema, and unlikely to collide with a real column.
_BOGUS_COLUMN = "no_such_column_zz"


def gold_wrong_column(row: Row) -> str:
    """Gold with one column identifier corrupted: parses, database rejects it.

    This isolates the parses/executes rung. It is what a model that has learned
    query *shape* but not the schema produces, which is exactly the v0 failure
    mode -- so the gap between this and ``gold`` is the reward's headroom for
    the thing we actually want it to teach.
    """
    return answer(_corrupt_column(row["gold_sql"]))


def _corrupt_column(sql: str) -> str:
    tree = parse_sql(sql)
    if tree is None:  # no gold query in this split hits it; keep the probe honest anyway
        return f"SELECT {_BOGUS_COLUMN} FROM ({sql})"

    column = next(tree.find_all(exp.Column), None)
    if column is not None:
        # Keep any table qualifier: "T1.no_such_column_zz" is still a clean
        # "no such column" rejection, and it leaves the rest of the query intact.
        column.set("this", exp.to_identifier(_BOGUS_COLUMN))
        return tree.sql(dialect="sqlite")

    # 27 of the 1,035 gold queries are pure "SELECT count(*) FROM t" and have no
    # column node to corrupt. Adding a bogus one to the projection is the same
    # failure -- valid syntax, unknown identifier.
    if isinstance(tree, exp.Select):
        return tree.select(_BOGUS_COLUMN).sql(dialect="sqlite")
    return f"SELECT {_BOGUS_COLUMN} FROM ({sql})"


def cross_join(row: Row) -> str:
    """A degenerate two-table cross join: the price of merely running.

    Every database in the split has at least two tables. A comma join is used
    rather than ``JOIN`` because ``parse_sql`` caps JOIN keywords at 10 and the
    question here is what the *executor* pays for, not what the parser rejects.
    """
    names = tables(row["db_path"])[:2]
    return answer(f"SELECT * FROM {', '.join(quote(name) for name in names)}")


def first_column(row: Row) -> str:
    """One column of one table, chosen without reading the question."""
    db_path = row["db_path"]
    table = tables(db_path)[0]
    columns = list(schema(db_path)[table])
    if not columns:
        return answer(f"SELECT * FROM {quote(table)}")
    return answer(f"SELECT {quote(columns[0])} FROM {quote(table)}")


#: Ordered as they are defined, which is roughly worst-to-best expected. The
#: printed table re-sorts by measured mean reward.
POLICIES: dict[str, Policy] = {
    "gold": gold,
    "empty_string": empty_string,
    "prose": prose,
    "select_1": select_1,
    "select_star": select_star,
    "count_star": count_star,
    "empty_result": empty_result,
    "gold_limit_zero": gold_limit_zero,
    "gold_wrong_column": gold_wrong_column,
    "cross_join": cross_join,
    "first_column": first_column,
}

#: Policies whose whole strategy is to return zero rows. Re-scored at the end
#: with ``pay_for_empty_gold=True`` to size the defence.
_EMPTY_POLICIES = ("empty_result", "gold_limit_zero")

#: A deliberately pessimistic spread for the step-cost measurement: two
#: full-table scans and a cross join alongside the cheap cases, because the
#: step cost that matters is the slow one, not the average one.
_STEP_MIX = (
    "gold",
    "gold_wrong_column",
    "select_star",
    "cross_join",
    "count_star",
    "first_column",
    "prose",
    "select_1",
)


# --------------------------------------------------------------------------
# running one policy
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PolicyResult:
    name: str
    n: int
    mean: float
    seconds: float
    #: ``outcome/<label>`` fractions straight out of ``SQLReward.report()``, so
    #: the probe reads the same numbers the trainer will log to W&B.
    report: dict[str, float]

    @property
    def match(self) -> float:
        return self.report.get("outcome/match", 0.0)

    @property
    def per_second(self) -> float:
        return self.n / self.seconds if self.seconds else float("inf")


def run_policy(
    name: str, policy: Policy, rows: list[Row], gold_cache: dict
) -> PolicyResult:
    """Score every row under one policy.

    One ``SQLReward`` per policy: the outcome counters are instance state, so a
    shared instance would report one blended distribution and the shape of each
    policy's failure -- the whole point -- would be lost.
    """
    reward = SQLReward()
    # ...but they all share one gold cache. Gold results are a property of the
    # dataset, not of the policy, and re-executing 1,035 gold queries eleven
    # times would put the probe's setup cost inside the throughput numbers.
    reward._gold = gold_cache

    # Completions are built before the clock starts. Generating them is the
    # simulated model's job; only the reward's own work is being timed.
    completions = [policy(row) for row in rows]

    started = time.perf_counter()
    total = sum(
        reward.score(completion, row["db_path"], row["gold_sql"])[0]
        for completion, row in zip(completions, rows, strict=True)
    )
    elapsed = time.perf_counter() - started

    return PolicyResult(
        name=name,
        n=len(rows),
        mean=total / len(rows),
        seconds=elapsed,
        report=reward.report(),
    )


# --------------------------------------------------------------------------
# the table
# --------------------------------------------------------------------------

#: Short headers for OUTCOMES, in the same order. Kept as a separate mapping so
#: a new outcome label in rewards.py shows up as a KeyError here rather than
#: silently vanishing from the table.
_SHORT = {
    "no_sql": "no_sql",
    "unparseable": "unparse",
    "timeout": "timeout",
    "db_error": "db_err",
    "too_many_rows": "toomany",
    "wrong_rows": "wrong",
    "empty_gold": "emptyG",
    "match": "match",
}


def summary_table(results: list[PolicyResult], ceiling: float) -> str:
    """Mean reward, share of the ceiling, match rate, throughput."""
    header = f"{'policy':<18} {'mean':>7} {'% ceiling':>10} {'match':>8} {'rew/s':>8}"
    lines = [header, "-" * len(header)]
    for result in results:
        share = result.mean / ceiling if ceiling else 0.0
        mark = "   <- ceiling (real answers)" if result.name == "gold" else ""
        lines.append(
            f"{result.name:<18} {result.mean:>7.3f} {share:>9.1%} "
            f"{result.match:>8.1%} {result.per_second:>8.0f}{mark}"
        )
    return "\n".join(lines)


def outcome_table(results: list[PolicyResult]) -> str:
    """The full outcome distribution, in percent of examples."""
    header = f"{'policy':<18}" + "".join(f"{_SHORT[name]:>9}" for name in OUTCOMES)
    lines = [header, "-" * len(header)]
    for result in results:
        cells = "".join(
            f"{result.report.get(f'outcome/{name}', 0.0) * 100:>9.1f}"
            for name in OUTCOMES
        )
        lines.append(f"{result.name:<18}{cells}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# what the table cannot show: search, the empty-gold defence, and cost
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class OracleResult:
    name: str
    mean: float
    match: float
    #: Candidate queries tried per row. The width of the search this bound assumes.
    candidates: float


def template_oracle(
    name: str,
    candidates_for: Callable[[Row], list[str]],
    rows: list[Row],
    gold_cache: dict,
) -> OracleResult:
    """The best reward a fixed *template* can earn, given the luckiest filling-in.

    The fixed policies above are pessimistic about what GRPO can find, because
    GRPO is a search: it draws ``num_generations`` samples per prompt and any
    one of them that scores higher pulls the policy toward it. A model that
    learns nothing except "emit ``COUNT(*)`` over some table" is therefore
    better modelled by the *maximum* over that template's fillings than by one
    arbitrary filling.

    This is a loose upper bound and is labelled as one -- it searches every
    table or column in the database, which is far more than eight samples
    explore. If even the bound is far below the ceiling, the template is not a
    way into the reward. If the bound is close, it is.
    """
    reward = SQLReward()
    reward._gold = gold_cache

    total = 0.0
    matches = 0
    tried = 0
    for row in rows:
        best = 0.0
        for sql in candidates_for(row):
            tried += 1
            scored, _ = reward.score(answer(sql), row["db_path"], row["gold_sql"])
            best = max(best, scored)
        total += best
        matches += best >= reward.tiers.matches

    return OracleResult(
        name=name,
        mean=total / len(rows),
        match=matches / len(rows),
        candidates=tried / len(rows),
    )


def count_over_any_table(row: Row) -> list[str]:
    return [f"SELECT COUNT(*) FROM {quote(table)}" for table in tables(row["db_path"])]


def any_single_column(row: Row) -> list[str]:
    db_path = row["db_path"]
    return [
        f"SELECT {quote(column)} FROM {quote(table)}"
        for table in tables(db_path)
        for column in schema(db_path)[table]
    ]


def oracle_table(oracles: list[OracleResult], ceiling: float) -> str:
    header = (
        f"{'template':<22} {'best mean':>10} {'% ceiling':>10} {'match':>8} "
        f"{'tried/row':>10}"
    )
    lines = [header, "-" * len(header)]
    for oracle in oracles:
        lines.append(
            f"{oracle.name:<22} {oracle.mean:>10.3f} "
            f"{oracle.mean / ceiling if ceiling else 0.0:>9.1%} "
            f"{oracle.match:>8.1%} {oracle.candidates:>10.1f}"
        )
    return "\n".join(lines)


def empty_gold_counterfactual(rows: list[Row], gold_cache: dict) -> str:
    """How much is ``pay_for_empty_gold=False`` actually holding back?

    ``rewards.py`` caps empty-gold rows at the executes tier so a zero-row
    answer cannot collect 2.0. That is a design decision defended in prose; this
    turns it into a number by re-scoring the zero-row policies with the cap
    removed. If the counterfactual mean is close to the real ceiling, the cap is
    load-bearing and nothing about it should be relaxed.
    """
    lines = [
        f"{'policy':<18} {'mean (as shipped)':>18} {'mean (if paid)':>16} {'match (if paid)':>17}",
    ]
    lines.append("-" * len(lines[0]))
    for name in _EMPTY_POLICIES:
        shipped = run_policy(name, POLICIES[name], rows, gold_cache)
        paid = SQLReward(pay_for_empty_gold=True)
        paid._gold = gold_cache
        completions = [POLICIES[name](row) for row in rows]
        total = sum(
            paid.score(completion, row["db_path"], row["gold_sql"])[0]
            for completion, row in zip(completions, rows, strict=True)
        )
        report = paid.report()
        lines.append(
            f"{name:<18} {shipped.mean:>18.3f} {total / len(rows):>16.3f} "
            f"{report.get('outcome/match', 0.0):>16.1%}"
        )
    return "\n".join(lines)


#: Groups of ``num_generations=8`` rewards, named by what the best sample in
#: each one did. Chosen from the outcomes the probe actually produced above, not
#: invented: every executable degenerate policy lands on 0.5, so "one ran, rest
#: errored" is the group a lazy policy manufactures for itself.
_GROUPS: dict[str, list[float]] = {
    "one correct, rest merely ran": [2.0] + [0.5] * 7,
    "one correct, rest no SQL": [2.0] + [0.0] * 7,
    "one merely ran, rest db_error": [0.5] + [0.2] * 7,
    "one parsed, rest no SQL": [0.2] + [0.0] * 7,
    "all merely ran (degenerate)": [0.5] * 8,
}

#: TRL adds this before dividing, so a zero-variance group yields zeros rather
#: than NaNs. Reproduced here because it is the only thing separating the
#: bottom row of the table from a division by zero.
_TRL_EPSILON = 1e-4


def advantage_table() -> str:
    """What the group normaliser does to the ladder's spacing.

    ``rewards.py`` argues that "only the ordering and the spacing matter"
    because the advantage divides by the group's standard deviation. The
    ordering survives that. The *spacing between rungs does not*, and this table
    is the demonstration: TRL 0.24 defaults to ``scale_rewards="group"``, which
    rescales every group to unit variance, so a group whose only distinction is
    "one sample happened to run" gets the same push as one where a sample was
    actually correct. Two-outcome groups are indistinguishable after scaling no
    matter which two rungs they sit on.

    That is not a bug in ``rewards.py`` -- the ladder is right -- it is a
    trainer setting that discards what the ladder encodes.
    """
    header = f"{'group of 8 rollouts':<32} {'best':>6} {'scaled adv':>12} {'unscaled':>10}"
    lines = [header, "-" * len(header)]
    for name, rewards in _GROUPS.items():
        n = len(rewards)
        mean = sum(rewards) / n
        variance = sum((r - mean) ** 2 for r in rewards) / (n - 1)  # torch's ddof=1
        std = variance**0.5
        best = max(rewards)
        lines.append(
            f"{name:<32} {best:>6.1f} {(best - mean) / (std + _TRL_EPSILON):>12.3f} "
            f"{best - mean:>10.3f}"
        )
    return "\n".join(lines)


def step_cost(rows: list[Row], gold_cache: dict, repeats: int = 3) -> list[float]:
    """Seconds to score one optimiser step's worth of rollouts.

    Called through ``__call__`` -- the actual TRL entry point, with the column
    lists TRL expands per generation -- rather than through ``score``, so this
    measures the code path training will use. Repeated because the first pass
    warms SQLite's page cache for those databases and a single sample would
    report that instead of the steady state.
    """
    # Spread across the split so several databases are involved, the way a
    # shuffled batch is. A batch drawn from one database would be optimistic.
    stride = max(len(rows) // STEP_PROMPTS, 1)
    batch = [rows[i * stride] for i in range(STEP_PROMPTS)]

    completions: list[str] = []
    db_paths: list[str] = []
    gold_sqls: list[str] = []
    for row in batch:
        for i in range(STEP_GENERATIONS):
            completions.append(POLICIES[_STEP_MIX[i % len(_STEP_MIX)]](row))
            db_paths.append(row["db_path"])
            gold_sqls.append(row["gold_sql"])

    timings = []
    for _ in range(repeats):
        reward = SQLReward()
        reward._gold = gold_cache
        started = time.perf_counter()
        reward(completions=completions, db_path=db_paths, gold_sql=gold_sqls)
        timings.append(time.perf_counter() - started)
    return sorted(timings)


# --------------------------------------------------------------------------


def heading(text: str) -> None:
    print("=" * max(len(text), 72))
    print(text)
    print("=" * max(len(text), 72))


def load(path: Path, limit: int | None) -> list[Row]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return rows[:limit] if limit else rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Probe the GRPO reward with policies that never read the question."
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="probe only the first N examples (smoke test)")
    args = parser.parse_args()

    rows = load(DATASET, args.limit)
    print(f"{DATASET}: {len(rows)} examples, {len({r['db_path'] for r in rows})} databases\n")

    # Also the drift check. If a gold query no longer runs, this raises here
    # rather than producing a table of meaningless numbers.
    warming = time.perf_counter()
    shared = SQLReward().warm(rows)
    for row in rows:  # populate the schema cache too, so it is out of the timings
        tables(row["db_path"])
    print(f"warmed {len(shared._gold)} distinct gold queries and "
          f"{len(_SCHEMAS)} schemas in {time.perf_counter() - warming:.1f}s\n")

    results = [
        run_policy(name, policy, rows, shared._gold)
        for name, policy in POLICIES.items()
    ]
    by_name = {result.name: result for result in results}
    ceiling = by_name["gold"].mean
    results.sort(key=lambda result: result.mean, reverse=True)

    heading("reward extracted by policies that never look at the question")
    print(summary_table(results, ceiling))
    print()
    outcomes = outcome_table(results)
    print("outcome distribution (% of examples)")
    print("-" * len(outcomes.splitlines()[0]))
    print(outcomes)
    print()

    # The one result that invalidates everything else if it fails.
    gold_result = by_name["gold"]
    if gold_result.match < 0.95:
        print("!! " + "=" * 69)
        print(f"!! gold reached the match tier on only {gold_result.match:.1%} of "
              f"examples.")
        print("!! The reward disagrees with the Phase 1 benchmark about what correct")
        print("!! means. Every number above is measuring the wrong thing, and so would")
        print("!! any reward curve from training against it. Fix this before training.")
        print("!! " + "=" * 69)
        print()
    else:
        print(f"gold sanity: match on {gold_result.match:.1%}, "
              f"{gold_result.report.get('outcome/empty_gold', 0.0):.1%} unscoreable "
              f"(empty gold answer). The reward agrees with the benchmark.\n")

    heading("upper bound if the policy learns only to pick a table or column")
    oracles = [
        template_oracle("COUNT(*) over any table", count_over_any_table, rows, shared._gold),
        template_oracle("any single column", any_single_column, rows, shared._gold),
    ]
    print(oracle_table(oracles, ceiling))
    print("  (max over every filling of the template -- a far wider search than")
    print("   8 generations do, so treat these as loose upper bounds)")
    print()

    heading("counterfactual: the empty-result defence, switched off")
    print(empty_gold_counterfactual(rows, shared._gold))
    print()

    heading("what GRPO's group normaliser does to the ladder's spacing")
    print(advantage_table())
    print("  Every two-outcome group scales to the same advantage, whichever two")
    print("  rungs it straddles: 0.2-vs-0.0 pushes as hard as 2.0-vs-0.0. The")
    print("  ladder's spacing only survives with GRPOConfig(scale_rewards=\"none\").")
    print()

    heading(f"cost of one GRPO step ({STEP_PROMPTS} prompts x {STEP_GENERATIONS} "
            f"generations = {STEP_PROMPTS * STEP_GENERATIONS} rewards)")
    timings = step_cost(rows, shared._gold)
    median = timings[len(timings) // 2]
    print(f"  {median:.3f}s median, {timings[-1]:.3f}s worst of {len(timings)} runs "
          f"({STEP_PROMPTS * STEP_GENERATIONS / median:.0f} rewards/s)")
    # Generating 64 completions of a few hundred tokens from a 0.5B model takes
    # tens of seconds on an A10G. A reward that costs a second or two against
    # that is noise; anything approaching the generation time is a real tax.
    if median > 5.0:
        print(f"  BOTTLENECK: {median:.1f}s per step is the same order as generating")
        print("  the rollouts. Parallelise the reward or tighten the timeout.")
    else:
        print(f"  Not a bottleneck: generating {STEP_PROMPTS * STEP_GENERATIONS} rollouts "
              f"from a 0.5B model takes tens of")
        print(f"  seconds on an A10G; the reward adds {median * 1000:.0f}ms on top of that.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
