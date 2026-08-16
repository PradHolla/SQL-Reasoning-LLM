"""Agentic retry loop for evaluation: write SQL, run it, retry on rejection.

Every checkpoint so far has been scored pass@1 -- one shot, no feedback. This
module measures the thing nobody has: whether handing the model its own
database error and letting it try again actually helps, at attempt budgets of
1, 2 and 3.

**The stopping rule is the whole point, so it gets this much space.** We retry
only when the executor's ``status`` is ``"error"`` -- a genuine *rejection*,
meaning SQLite refused the query and produced a message worth feeding back.
We do **not** retry on ``"timeout"`` or ``"too_many_rows"``: SQLite *accepted*
those queries, and ``metrics.py`` already counts them as executed
(``executed = pred.status != "error"``, see ``score_example``). Retrying them
would make this module's notion of "failed" disagree with the execution-rate
metric everything else in this project is scored against.

**The loop never looks at gold SQL.** At real inference time gold is not
available, so a stopping rule that consulted it would be measuring a loop that
cannot exist in production -- the whole exercise would be meaningless. Gold is
used exactly once in this project's retry story: by the scorer, after the loop
has already finished and committed to its answers.

**Two feedback shapes, because the checkpoints were never trained for either
one properly.** Every checkpoint here was SFT'd on strictly single-turn
conversations -- one system turn, one user turn, one assistant answer. A retry
loop that keeps appending turns (``"multiturn"``, the shape a real agent
would use) hands those checkpoints a conversation shape they have never seen,
so a loss there could just be "out of distribution", not "retry doesn't
help". ``"restate"`` stays inside the trained shape: it rewrites the single
user turn to include the failed query and its error, so the prompt is still
exactly two messages no matter how many rounds have run. Measuring both is
what separates a real retry effect from a prompt-format artifact.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from sqlrl.eval.executor import run
from sqlrl.eval.prompts import Prompt, extract_sql

__all__ = [
    "STYLES",
    "Attempt",
    "Trace",
    "at_budget",
    "attempt_counts",
    "feedback",
    "retry_prompt",
    "run_retry",
]

#: The two feedback shapes. See the module docstring for why both exist.
STYLES = ("multiturn", "restate")

#: Errors arrive as ``f"{type(exc).__name__}: {exc}"`` (see executor.run). The
#: model should see the message, not our exception plumbing.
_EXC_PREFIX = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*Error: ")

_NO_SQL_FEEDBACK = (
    "No SQL query was found in your answer. You must put the query inside "
    "<answer></answer> tags."
)

#: Longest failed query quoted back to the model. The longest gold query in all
#: of Spider is 608 characters, so anything past this is model garbage -- a
#: hallucinated 2,000-character join chain teaches nothing when echoed back, and
#: quoting it in full is what pushes a round-3 multiturn prompt over the
#: backend's input ceiling. Bounding it here keeps the worst case ~450 tokens
#: below that ceiling instead of ~180 above it.
MAX_QUOTED_SQL = 600


@dataclass(frozen=True)
class Attempt:
    raw: str  # full model output for this attempt
    sql: str  # extract_sql(raw)
    status: str  # ExecResult.status: "ok" | "error" | "timeout" | "too_many_rows"
    error: str | None  # ExecResult.error


@dataclass
class Trace:
    index: int  # index into the benchmark example list
    db_id: str
    question: str
    gold_sql: str
    attempts: list[Attempt]  # length 1..max_attempts, never empty


def feedback(sql: str, error: str | None) -> str:
    """The message shown to the model after a failed attempt.

    Two cases: nothing query-shaped was found (``sql == ""``, so there is no
    query to quote), or the database rejected a real query, in which case the
    exception-class prefix is stripped so the model sees ``"no such column:
    T1.name"`` rather than ``"OperationalError: no such column: T1.name"``.
    """
    if not sql:
        return _NO_SQL_FEEDBACK

    message = _EXC_PREFIX.sub("", error or "")
    quoted = sql if len(sql) <= MAX_QUOTED_SQL else sql[:MAX_QUOTED_SQL] + " ..."
    return (
        f"The query `{quoted}` failed with a database error: {message}\n"
        "Think step-by-step inside <think></think> tags about why it failed, "
        "then give the corrected query inside <answer></answer> tags."
    )


def retry_prompt(base: Prompt, attempts: list[Attempt], style: str) -> Prompt:
    """Build the next round's prompt out of ``base`` and the attempts so far.

    ``"multiturn"`` appends an assistant/user pair per failed attempt onto
    ``base.messages``, so the conversation grows every round. ``"restate"``
    always returns exactly two messages -- system plus a user turn built from
    the *original* ``Schema: ...\\nQuestion: ...`` text with only the most
    recent failure's feedback appended, so the prompt never leaves the shape
    the checkpoints were trained on.
    """
    if style not in STYLES:
        raise ValueError(f"style must be one of {STYLES}, got {style!r}")
    if base.messages is None:
        raise ValueError(
            "retry is chat-only; base prompt has no messages (it is a CPT "
            "completion prompt, which has no turns to append to)"
        )

    if style == "multiturn":
        messages = list(base.messages)
        for attempt in attempts:
            messages.append({"role": "assistant", "content": attempt.raw})
            messages.append(
                {"role": "user", "content": feedback(attempt.sql, attempt.error)}
            )
        return Prompt(messages=messages)

    # "restate": recover the original "Schema: ...\nQuestion: ..." text and
    # append only the latest failure -- the prompt must not grow past round 2.
    last = attempts[-1]
    original = base.messages[-1]["content"]
    user_text = f"{original}\n\n{feedback(last.sql, last.error)}"
    return Prompt(messages=[base.messages[0], {"role": "user", "content": user_text}])


def run_retry(
    prompts: list[Prompt],
    db_paths: list[Path],
    generate: Callable[[list[Prompt]], list[str]],
    *,
    max_attempts: int = 3,
    style: str = "multiturn",
    timeout: float = 30.0,
    on_round: Callable[[int, int], None] | None = None,
) -> list[list[Attempt]]:
    """Run the retry loop in batched rounds, not per example.

    Round 1 generates all ``len(prompts)`` prompts in one batched ``generate``
    call. Only the examples still in ``"error"`` after that go into round 2,
    and so on -- round 2 is typically a tenth the size of round 1. Looping
    per example instead would replace a handful of large batched calls with
    thousands of tiny ones, roughly 50x slower for no gain in the result.

    Returns one ``list[Attempt]`` per input prompt, in the original input
    order, regardless of how many rounds each example needed.
    """
    n = len(prompts)
    active = list(range(n))
    histories: list[list[Attempt]] = [[] for _ in range(n)]

    for round_index in range(max_attempts):
        if not active:
            break

        batch = [
            prompts[i] if round_index == 0 else retry_prompt(prompts[i], histories[i], style)
            for i in active
        ]
        raws = generate(batch)
        assert len(raws) == len(batch), (
            f"generate returned {len(raws)} outputs for {len(batch)} prompts"
        )

        for i, raw in zip(active, raws):
            sql = extract_sql(raw)
            result = run(sql, db_paths[i], timeout=timeout)
            histories[i].append(Attempt(raw, sql, result.status, result.error))

        if on_round:
            on_round(round_index + 1, len(active))

        # Retry only on a database rejection -- see the module docstring for
        # why "timeout" and "too_many_rows" must not loop back here.
        active = [i for i in active if histories[i][-1].status == "error"]

    return histories


def at_budget(attempts: list[Attempt], budget: int) -> Attempt:
    """The attempt the loop would have finished on with only ``budget`` tries.

    A trace that succeeded (or simply ran out of rounds) before ``budget`` was
    reached stays on its last attempt -- the loop stopped there for real, so a
    *larger* budget could not have produced a different answer. ``min(budget,
    len(attempts))`` is what pins that down: a 1-attempt trace clamps to index
    0 at every budget, never index ``budget - 1``. Getting this backwards
    would silently score every early success as if it kept trying, which is
    the main way this file could quietly produce a wrong number.
    """
    if budget < 1:
        raise ValueError(f"budget must be >= 1, got {budget}")
    if not attempts:
        raise ValueError("attempts must not be empty")
    return attempts[min(budget, len(attempts)) - 1]


def attempt_counts(histories: list[list[Attempt]]) -> tuple[dict[int, int], int]:
    """``({attempts_used: count}, still_rejected)`` -- how the budget was spent.

    The accuracy-per-budget table says whether retry helped. This says *how
    much of it was even used*, which is the number that tells the two null
    results apart: a loop nothing reaches (almost everything settles on
    attempt 1) is a different finding from a loop that runs and fixes
    nothing.

    ``still_rejected`` counts traces whose last attempt was ``"error"`` --
    they exhausted the budget without ever producing SQL the database would
    accept. Note this is not the same as "got it wrong": a query can be
    accepted and still answer a different question, which no error-driven
    loop can see.
    """
    used: dict[int, int] = {}
    still_rejected = 0
    for attempts in histories:
        if not attempts:
            continue
        if attempts[-1].status == "error":
            still_rejected += 1
            continue
        used[len(attempts)] = used.get(len(attempts), 0) + 1
    return dict(sorted(used.items())), still_rejected
