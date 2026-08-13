"""The GRPO reward: did the model's SQL produce the right answer?

    reward = SQLReward()
    reward.warm(rows)                       # validate + cache, before training
    GRPOTrainer(..., reward_funcs=[reward])

This replaces v0's ``format_reward_func`` + ``correctness_reward_func``. Those
two ran for 300 steps and taught the model nothing, and the reason is worth
stating precisely because everything here follows from it.

GRPO samples ``num_generations`` completions per prompt and gives each one an
advantage relative to *its own group*::

    advantage_i = (r_i - mean(r)) / std(r)

If every sample in a group scores the same, ``mean(r) == r_i``, so every
advantage is exactly zero and the group contributes **no gradient at all**.
v0's format reward fired for every sample (the SFT model always emitted the
tags) and its exact-string-match reward fired for almost none, so a typical
group scored ``[1.0, 1.0, 1.0, 1.0]``. ``frac_reward_zero_std`` sat between 0.5
and 1.0 and touched 1.00 at step 160: between half and all of every batch was a
no-op. **Variance within the group is the fuel; a reward every sample earns is
not a reward, it is a constant.**

So the reward here is a *ladder*, and partial credit is the entire point:

===========================================  =======
outcome                                      reward
===========================================  =======
no SQL found, or it will not parse             0.0
parses as SQL                                  0.2
the database accepted it and it ran            0.5
result set matches gold                        2.0
===========================================  =======

A group where nothing is fully correct can now score ``[0.2, 0.5, 0.5, 0.2]``
-- real variance, real gradient, pointing at "the ones that ran beat the ones
that did not". Those are exactly the prompts v0 learned nothing from.

**This ladder only reaches the gradient under** ``GRPOConfig(scale_rewards=
"none")``. TRL 0.24 defaults to ``"group"``, which divides each group's
advantages by that group's own standard deviation, rescaling every group to
unit variance. The *ordering* of the rungs survives that; the *spacing between
them does not*. Measured on the four groups below, with eight generations::

    group of 8 rollouts               scale_rewards="group"   ="none"
    one correct, rest merely ran                      2.474     1.312
    one correct, rest no SQL                          2.475     1.750
    one merely ran, rest db_error                     2.473     0.262
    one parsed, rest no SQL                           2.471     0.175

Under the default, a group whose best sample was *actually correct* pushes
exactly as hard as one whose best sample merely *parsed* -- any two-outcome
group normalises to the same advantage, whichever two rungs it straddles. That
throws away the entire reason for having tiers. Setting ``scale_rewards="none"``
(Dr. GRPO's recommendation, for the related reason that dividing by a group's
own variance over-weights easy and hopeless questions) restores a 7.5x spread
between "found the answer" and "produced something that parses".

``sqlrl.training.reward_probe`` prints that table, and the measurements below,
from live data.

There is no format reward. ``extract_sql`` is the same lenient extractor the
evaluator uses, so a completion that abandons the ``<think>/<answer>`` tags but
contains a working query still scores -- and one that is malformed enough to
hide its SQL scores zero without any separate rule. We reward what we measure.

Correctness is decided by ``sqlrl.eval.executor``, the same code that computes
the Phase 1 benchmark number. Not a reimplementation of it: if the reward and
the benchmark could disagree about what "correct" means, a rising reward curve
would prove nothing about the metric we actually care about. Confirmed rather
than assumed: replaying the gold queries through this reward reaches the match
tier on 97.9% of the split, and the 2.1% shortfall is exactly the empty-gold
rows described below.

Probed before it was trained against, by eleven policies that never read the
question (``reward_probe``). The best of them reaches 26.8% of gold's mean and
matches on 1.8% of examples; searching every table for the luckiest ``COUNT(*)``
reaches 4.0%. Both zero-row exploits score 0.0% match. Nothing here is a way
into the reward, but note the shape of the failure: **every degenerate policy
that runs at all lands on exactly 0.5**, so the executes rung is a wide plateau.
A policy that collapsed onto it would manufacture zero-variance groups and stall
the same way v0 did. ``outcome/too_many_rows`` and ``outcome/wrong_rows`` in
``report()`` are what would show that happening.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from sqlrl.eval.executor import Row, compare, parse_sql, requires_order, run
from sqlrl.eval.prompts import extract_sql

__all__ = ["Gold", "SQLReward", "Tiers", "drop_empty_gold"]


@dataclass(frozen=True)
class Tiers:
    """Reward at each rung. A rung is *assigned*, not accumulated.

    Must stay monotone -- a strictly better outcome must never score lower --
    or GRPO will happily optimise toward the dip.
    """

    nothing: float = 0.0
    parses: float = 0.2
    executes: float = 0.5
    matches: float = 2.0

    def __post_init__(self) -> None:
        rungs = [self.nothing, self.parses, self.executes, self.matches]
        if any(b < a for a, b in zip(rungs, rungs[1:])):
            raise ValueError(f"tiers must be non-decreasing, got {rungs}")


@dataclass(frozen=True)
class Gold:
    """One gold query's answer, computed once and reused for every rollout."""

    rows: list[Row]
    order_matters: bool

    @property
    def empty(self) -> bool:
        return not self.rows


#: Outcome labels, ordered worst to best. Reported every step: the *shape* of
#: this distribution is how reward hacking announces itself, long before the
#: benchmark notices.
OUTCOMES = (
    "no_sql",        # nothing query-shaped in the completion
    "unparseable",   # sqlglot rejected it, or it was pathological enough to skip
    "timeout",       # never finished
    "db_error",      # SQLite refused it: bad syntax, unknown table, unknown column
    "too_many_rows", # ran, but returned more rows than gold has -- cannot match
    "wrong_rows",    # ran cleanly, wrong answer
    "empty_gold",    # gold returns nothing, so correctness is unknowable here
    "match",         # correct
)


class SQLReward:
    """Execution-grounded reward, shaped for TRL's ``reward_funcs``.

    TRL calls this as ``reward(prompts=..., completions=..., **columns)`` where
    ``columns`` are the dataset's other fields already expanded to one entry per
    generation. We need ``db_path`` and ``gold_sql``; everything else is ignored.
    """

    #: TRL logs rewards under ``reward_funcs[i].__name__``. Instances do not get
    #: one for free, and without it trainer construction dies on an AttributeError.
    __name__ = "sql_reward"

    def __init__(
        self,
        tiers: Tiers = Tiers(),
        *,
        timeout: float = 3.0,
        gold_timeout: float = 30.0,
        dedupe: bool = False,
        pay_for_empty_gold: bool = False,
    ) -> None:
        #: Every gold query in the GRPO split runs in 0.1 ms at the median and
        #: 19.9 ms at the worst, so 3 s is ~150x headroom over anything
        #: legitimate. It is deliberately tighter than the evaluator's 5 s: this
        #: runs ``batch x num_generations`` times per step, and eight
        #: simultaneously pathological rollouts would otherwise stall training
        #: for 40 s of pure nothing.
        self.timeout = timeout
        self.gold_timeout = gold_timeout
        #: Must match the evaluator, or the reward optimises one definition of
        #: correct while the benchmark reports another. Both default to multiset
        #: semantics; ``dedupe=True`` is Spider's official set semantics.
        self.dedupe = dedupe
        self.pay_for_empty_gold = pay_for_empty_gold
        self.tiers = tiers
        self._gold: dict[tuple[str, str], Gold] = {}
        self.outcomes: Counter[str] = Counter()
        #: Rollouts that matched while returning zero rows. The canary for the
        #: single most likely hack in this task -- see ``report``.
        self.empty_matches = 0

    # ------------------------------------------------------------------
    # the TRL entry point
    # ------------------------------------------------------------------

    def __call__(
        self,
        completions: Sequence[Any],
        db_path: Sequence[str],
        gold_sql: Sequence[str],
        **kwargs: Any,
    ) -> list[float]:
        return [
            self.score(text, path, sql)[0]
            for text, path, sql in zip(
                (_text(c) for c in completions), db_path, gold_sql, strict=True
            )
        ]

    def score(self, completion: str, db_path: str, gold_sql: str) -> tuple[float, str]:
        """Reward and outcome label for one rollout. Never raises on model output.

        A reward function that throws kills the run mid-training, so anything the
        model could have caused is caught and scored zero. A missing database is
        *not* in that category -- that is our bug, and swallowing it would score
        every rollout zero while training continued happily, which is the exact
        silent failure this whole phase is trying to avoid.
        """
        try:
            reward, outcome = self._score(completion, db_path, gold_sql)
        except FileNotFoundError:
            raise
        except Exception:  # noqa: BLE001 -- one bad rollout must not end a run
            reward, outcome = self.tiers.nothing, "no_sql"
        self.outcomes[outcome] += 1
        return reward, outcome

    def _score(self, completion: str, db_path: str, gold_sql: str) -> tuple[float, str]:
        sql = extract_sql(completion)
        if not sql:
            return self.tiers.nothing, "no_sql"
        if parse_sql(sql) is None:
            return self.tiers.nothing, "unparseable"

        gold = self.gold(db_path, gold_sql)

        # Capping at one row past gold is both the memory bound and a large
        # speed win: a hallucinated cross join stops after a handful of rows
        # instead of materialising millions. It costs nothing in accuracy --
        # under multiset comparison a result longer than gold cannot match, so
        # the truncated rows were never going to be examined.
        result = run(sql, db_path, timeout=self.timeout, max_rows=len(gold.rows) + 1)

        if result.status == "timeout":
            # Deliberately *not* the executes tier, though the evaluator's
            # execution-rate metric counts timeouts as executed. A query that
            # never finished tells us nothing about whether it was right, and
            # paying for it would put a thumb on the scale toward rollouts that
            # burn three seconds each.
            return self.tiers.parses, "timeout"
        if result.status == "error":
            return self.tiers.parses, "db_error"
        if result.status == "too_many_rows":
            # It ran. Given the cap above, this just means "more rows than gold".
            return self.tiers.executes, "too_many_rows"

        if gold.empty and not self.pay_for_empty_gold:
            # Gold returns nothing, so *any* empty result compares equal -- and
            # producing nothing is the easiest thing in SQL. Paying the match
            # tier here would hand out 2.0 for `WHERE 1=0` and GRPO would find
            # that within a few hundred steps. Capped at the executes tier
            # instead, so the group still differentiates on parse/execute.
            # Belt and braces: `drop_empty_gold` removes these rows entirely.
            return self.tiers.executes, "empty_gold"

        if compare(result.rows, gold.rows, gold.order_matters, dedupe=self.dedupe):
            if not result.rows:
                self.empty_matches += 1
            return self.tiers.matches, "match"
        return self.tiers.executes, "wrong_rows"

    # ------------------------------------------------------------------
    # gold results, computed once
    # ------------------------------------------------------------------

    def gold(self, db_path: str, gold_sql: str) -> Gold:
        """The cached answer to ``gold_sql``, keyed by (database, query).

        The reward runs ``batch x num_generations`` times per step and the gold
        query is identical across a whole group, so without this we would
        re-execute it eight times per prompt per step, forever. Caching is
        unconditional because the whole GRPO split's gold results total 80,080
        cells -- a few megabytes, once.
        """
        key = (str(db_path), gold_sql)
        cached = self._gold.get(key)
        if cached is None:
            result = _run_gold(gold_sql, db_path, self.gold_timeout)
            cached = Gold(result.rows, requires_order(gold_sql))
            self._gold[key] = cached
        return cached

    def warm(self, rows: Iterable[Mapping[str, Any]]) -> "SQLReward":
        """Execute and cache every gold query up front.

        Two jobs. It moves a missing database or a broken gold query from step
        400 of training to before step 1, where it is a five-second fix rather
        than a wasted GPU hour; and it keeps the first optimiser step from
        paying for a thousand cold cache misses.
        """
        for row in rows:
            self.gold(row["db_path"], row["gold_sql"])
        return self

    # ------------------------------------------------------------------
    # what to watch while it trains
    # ------------------------------------------------------------------

    def report(self) -> dict[str, float]:
        """Outcome distribution since the last call, for W&B. Resets the counters.

        Log this every step. The headline reward can rise for good reasons and
        bad ones, and these are what tell them apart:

        * ``match`` rising while the benchmark does not move means the reward is
          being gamed rather than satisfied.
        * ``empty_match_frac`` above roughly zero is the specific hack to fear:
          the model has found that returning nothing is cheap.
        * ``too_many_rows`` or ``timeout`` climbing means degenerate joins are
          being rewarded for merely running.
        * everything collapsing onto one outcome is the v0 failure returning --
          uniform reward, zero variance, no gradient.
        """
        total = sum(self.outcomes.values())
        if not total:
            return {}
        report = {f"outcome/{name}": self.outcomes[name] / total for name in OUTCOMES}
        report["outcome/empty_match_frac"] = self.empty_matches / total
        self.outcomes.clear()
        self.empty_matches = 0
        return report


def _text(completion: Any) -> str:
    """The assistant's text, from either dataset shape.

    TRL hands back a list of message dicts for conversational datasets and a
    plain string otherwise. Ours is conversational, but the reward is also
    called directly from tests and probes, so it accepts both.
    """
    if isinstance(completion, str):
        return completion
    if isinstance(completion, Sequence):
        return "".join(
            str(turn.get("content", ""))
            for turn in completion
            if isinstance(turn, Mapping)
        )
    return ""


def _run_gold(gold_sql: str, db_path: str, timeout: float):
    """Execute a gold query, treating any failure as our bug rather than data.

    Every gold query in every split was executed at dataset-build time, so one
    failing now means the jsonl and the database files have drifted apart --
    a wrong ``--root``, a partial download, the wrong working directory. Left
    quiet, that silently shrinks the training set instead of stopping to say
    so, and a GRPO run on half a dataset looks exactly like one on all of it.
    """
    result = run(gold_sql, db_path, timeout=timeout)
    if not result.ok:
        raise ValueError(
            f"gold query failed ({result.status}: {result.error}) "
            f"on {db_path}: {gold_sql!r}"
        )
    return result


def drop_empty_gold(
    rows: Sequence[Mapping[str, Any]], timeout: float = 30.0
) -> list[Mapping[str, Any]]:
    """Rows whose gold query actually returns something.

    22 of the GRPO split's 1,035 examples have gold queries that return zero
    rows. They cannot teach correctness -- every empty result matches them, for
    right and wrong reasons alike -- so they are worth more removed than
    defended against. ``SQLReward`` refuses to pay for them anyway; this just
    stops us spending rollouts on questions with no learnable answer.

    Drops *only* the empty ones. A gold query that fails to execute raises,
    because "the answer is empty" and "we could not get the answer" are
    different facts and only one of them is about the data.
    """
    return [row for row in rows if _run_gold(row["gold_sql"], row["db_path"], timeout).rows]
