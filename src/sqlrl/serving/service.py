"""Serve a checkpoint the way it was actually measured.

Phase 5 found two inference-time techniques worth exposing outside the eval
harness:

* execution voting (``eval.voting``) -- sample k candidates, execute all of
  them, answer with the largest group by result rows. +3.5 EX points at k=8.
* retry (``eval.retry``) -- on a database rejection, feed the error back and
  try again. +1.1 EX points at 3 attempts.

**Both are reused here, not reimplemented.** Clustering, selection and the
retry loop are exactly the code that produced those two numbers; a second
implementation in this module could drift from them silently, and the whole
point of a service is that it provably behaves like the thing that was
measured. This module's job is generation plumbing and timing around calls
into ``eval.voting`` and ``eval.retry`` -- never a second copy of what they
do.

**Composing retry with voting is not attempted.** The clean version --
retrying each of the k sampled candidates independently until it executes,
then voting over the results -- needs a different shape of call into
``run_retry`` than its batched, one-round-per-list-of-examples design
supports (it does not distinguish "the greedy candidate" from "a sampled
candidate" within a batch, which voting's ordering depends on). Rather than
build that and risk it silently diverging from either measured technique,
``answer`` takes the vote path whenever ``samples > 1``, full stop -- retry
is only applied when ``samples == 1``. Pass ``max_attempts=1`` when you want
pure voting (the default) and ``samples=1`` when you want pure retry. See the
``answer`` docstring.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

from sqlrl.eval import voting
from sqlrl.eval.executor import read_schema, run
from sqlrl.eval.prompts import Prompt, chat_prompt, extract_sql, render_schema
from sqlrl.eval.retry import run_retry
from sqlrl.eval.run_eval import MODELS

__all__ = ["Answer", "CALIBRATION", "Confidence", "SqlService", "confidence",
           "winner_agreement"]


@dataclass(frozen=True)
class Confidence:
    agreement: int  # size of the winning group
    samples: int  # how many candidates were generated
    level: str  # "high" | "medium" | "low" | "none" | "unmeasured"
    expected_accuracy: float | None


@dataclass(frozen=True)
class Answer:
    sql: str
    rows: list[list]
    status: str  # ExecResult.status
    error: str | None
    confidence: Confidence
    attempts: int  # how many generation rounds retry used
    timings_ms: dict[str, float]  # keys: "generate", "execute", "total"


#: Fraction of samples agreeing (``agreement / samples``, so this survives a
#: different k) -> ``(level, expected_accuracy)``. Sorted highest fraction
#: first; ``confidence`` walks it top to bottom.
#:
#: These numbers come from ONE measurement: grpo-coder15 on Spider test,
#: k=8, n=2147. They do not automatically transfer to another checkpoint,
#: another k, or another database -- a service reporting a confidence it has
#: not calibrated for its own deployment is worse than reporting none, which
#: is exactly why ``samples == 1`` reports "unmeasured" instead of guessing.
#:
#: The buckets are deliberately coarse. The raw per-agreement curve was:
#:
#:     8/8 -> 84.5%   7/8 -> 62.8%   6/8 -> 47.9%   5/8 -> 44.5%
#:     4/8 -> 44.8%   3/8 -> 29.5%   2/8 -> 55.6%   1/8 -> 50.0%
#:
#: Below 7/8 that is not a curve, it is noise on samples of 27 to 110: 2/8
#: scores above 4/8, and 3/8 below both. Only unanimous and near-unanimous
#: are separable, so everything from 1/8 to 5/8 is one bucket.
#:
#: **Each bucket's number is the pooled accuracy of its own members, not its
#: best member.** Bundling 7/8 (62.8%) with 6/8 (47.9%) and then reporting
#: 62.8% would overstate the weaker half by 15 points -- which is the failure
#: this whole table exists to avoid. Recomputed directly from the ballots:
#:
#:     bucket     n     share   accuracy
#:     high     1493    69.5%      0.845
#:     medium    264    12.3%      0.561
#:     low       284    13.2%      0.440
#:     none      106     4.9%      0.000
CALIBRATION: tuple[tuple[float, str, float], ...] = (
    (1.0, "high", 0.845),
    (0.75, "medium", 0.561),
    (0.0, "low", 0.440),
)


def confidence(agreement: int, samples: int) -> Confidence:
    """The calibrated confidence for a vote of ``agreement`` out of ``samples``.

    ``samples == 1`` is a single greedy sample: there were no other
    candidates to agree or disagree with, so there is nothing measured here
    to report. Level is "unmeasured" and ``expected_accuracy`` is ``None`` --
    inventing a number for it would be exactly the failure mode the module
    docstring above warns about, just one step earlier.
    """
    if samples == 1:
        # agreement=1 is a trivial, self-referential value ("the one sample
        # agrees with itself") -- it carries no signal. level="unmeasured" is
        # the actual message; do not read anything into this number.
        return Confidence(agreement=1, samples=1, level="unmeasured", expected_accuracy=None)

    fraction = agreement / samples
    if fraction <= 0.0:
        return Confidence(agreement=agreement, samples=samples, level="none", expected_accuracy=0.0)
    for threshold, level, expected in CALIBRATION:
        if fraction >= threshold:
            return Confidence(agreement=agreement, samples=samples, level=level, expected_accuracy=expected)
    raise AssertionError("unreachable: CALIBRATION's lowest threshold is 0.0")


class _Outcome(NamedTuple):
    """What ``_single_answer``/``_vote_answer`` produce, before ``answer`` adds
    the total-latency figure that only it can see (it wraps both of them).
    """

    sql: str
    rows: list[list]
    status: str
    error: str | None
    confidence: Confidence
    attempts: int
    generate_ms: float
    execute_ms: float


def _execute(raw: str, db_path: Path, timeout: float) -> tuple[voting.Candidate, str | None]:
    """One candidate, executed. Mirrors ``run_eval._execute_candidate``, kept
    as a local copy because that one is private to ``eval.run_eval`` -- and
    unlike it, this also returns the raw error message, which
    ``voting.Candidate`` does not carry but ``Answer`` needs to report for
    whichever candidate wins the vote.
    """
    sql = extract_sql(raw)
    result = run(sql, db_path, timeout=timeout)
    rows = [list(row) for row in result.rows] if result.status == "ok" else []
    return voting.Candidate(raw=raw, sql=sql, status=result.status, rows=rows), result.error


def winner_agreement(candidates: list[voting.Candidate], winner_index: int) -> int:
    """Size of the cluster ``voting.select`` chose, i.e. how many of the
    candidates agreed with the winning answer.

    Public because the calibration numbers in ``CALIBRATION`` are recomputed
    from saved ballots by ``analysis/vote_curve.py``, and a second definition
    of "how many agreed" there could drift from this one without either side
    noticing -- which is the exact failure the table's own docstring warns
    about, one level down.

    Re-derives clustering with ``voting.cluster`` rather than trusting a
    count computed some other way, so "agreement" can never drift from what
    ``select`` actually picked -- ``cluster`` is deterministic and cheap (see
    its docstring), so recomputing it here costs nothing worth avoiding.
    """
    for indices in voting.cluster(candidates):
        if indices[0] == winner_index:
            return len(indices)
    # select() returns 0 when nothing executed "ok" at all -- there is no
    # cluster containing index 0 in that case, because cluster() drops every
    # non-ok candidate before grouping. Zero candidates agree with an answer
    # that never ran.
    return 0


class SqlService:
    """A checkpoint plus the two inference-time techniques Phase 5 measured,
    behind one call: ``answer``.

    ``model`` is a key into ``run_eval.MODELS`` and the backend is built
    exactly as ``run_eval.generate`` builds it (same ``HFBackend``
    constructor, same arguments), so the weights this serves are the weights
    that were evaluated -- there is no second, slightly different loading
    path that could silently serve something else.
    """

    def __init__(
        self,
        model: str = "grpo-coder15",
        *,
        databases: Path,
        device: str | None = None,
        batch_size: int = 8,
        max_new_tokens: int = 384,
        timeout: float = 5.0,
    ) -> None:
        if model not in MODELS:
            raise ValueError(f"unknown model {model!r}; choose from {sorted(MODELS)}")
        spec = MODELS[model]
        self.model_name = model
        self.max_new_tokens = max_new_tokens
        self.timeout = timeout
        self._schema_cache: dict[str, str] = {}

        self._databases_dir = Path(databases)
        self._db_ids = tuple(
            sorted(
                child.name
                for child in self._databases_dir.iterdir()
                if child.is_dir() and (child / f"{child.name}.sqlite").is_file()
            )
        ) if self._databases_dir.is_dir() else ()

        # Imported here, mirroring run_eval.generate: nothing above this line
        # needs torch, so constructing a SqlService is the only place in this
        # module that pays for it -- importing sqlrl.serving.service (and the
        # api module built on it) stays cheap.
        from sqlrl.eval.backends.hf import HFBackend

        self.backend = HFBackend(
            spec.path,
            name=spec.name,
            base_model=spec.base,
            chat=spec.chat,
            device=device,
            batch_size=batch_size,
        )

    @property
    def databases(self) -> list[str]:
        return list(self._db_ids)

    def schema_for(self, db_id: str) -> str:
        if db_id not in self._schema_cache:
            path = self._databases_dir / db_id / f"{db_id}.sqlite"
            self._schema_cache[db_id] = render_schema(read_schema(path))
        return self._schema_cache[db_id]

    def answer(
        self,
        question: str,
        db_id: str,
        *,
        samples: int = 1,
        max_attempts: int = 1,
        temperature: float = 0.8,
    ) -> Answer:
        """Answer one question, never raising on anything the model produced.

        Bad SQL, a database rejection, a hallucinated table -- all of that is
        data, not an error condition: a question whose every candidate fails
        comes back as an ``Answer`` with the failing SQL, ``status="error"``
        and confidence level ``"none"``. The only things that raise here are
        genuine service bugs: an unknown ``db_id`` (``schema_for`` reads
        straight through to ``read_schema``, which raises ``FileNotFoundError``
        for a missing database file) or an out-of-range ``samples`` /
        ``max_attempts``.

        ``samples > 1`` always takes the voting path and ``max_attempts`` is
        ignored; ``samples == 1`` always takes the retry path (even
        ``max_attempts == 1``, which makes retry.run_retry behave exactly
        like a single plain generate-and-execute call, so there is one code
        path for "no retry" instead of two). See the module docstring for why
        the two are not combined.
        """
        if samples < 1:
            raise ValueError(f"samples must be >= 1, got {samples}")
        if max_attempts < 1:
            raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")

        prompt = chat_prompt(self.schema_for(db_id), question)
        db_path = self._databases_dir / db_id / f"{db_id}.sqlite"

        total_started = time.perf_counter()
        outcome = (
            self._vote_answer(prompt, db_path, samples=samples, temperature=temperature)
            if samples > 1
            else self._single_answer(prompt, db_path, max_attempts=max_attempts)
        )
        total_ms = (time.perf_counter() - total_started) * 1000

        return Answer(
            sql=outcome.sql,
            rows=outcome.rows,
            status=outcome.status,
            error=outcome.error,
            confidence=outcome.confidence,
            attempts=outcome.attempts,
            timings_ms={
                "generate": outcome.generate_ms,
                "execute": outcome.execute_ms,
                "total": total_ms,
            },
        )

    def _single_answer(self, prompt: Prompt, db_path: Path, *, max_attempts: int) -> _Outcome:
        """The retry path: ``eval.retry.run_retry`` on a single-example batch."""
        generate_ms_accum = 0.0

        def timed_generate(batch: list[Prompt]) -> list[str]:
            nonlocal generate_ms_accum
            started = time.perf_counter()
            outputs = self.backend.generate(batch, max_new_tokens=self.max_new_tokens)
            generate_ms_accum += (time.perf_counter() - started) * 1000
            return outputs

        # run_retry interleaves generation and execution across rounds and does
        # not expose them separately -- reimplementing the loop just to split
        # the timer would be exactly the kind of second implementation the
        # module docstring says not to build. Wrapping the generate callable
        # measures pure generation time directly; execute time is the
        # remainder of the wall clock, which is a fair split because
        # generation and SQLite execution are the only two things run_retry
        # does inside its loop.
        wall_started = time.perf_counter()
        histories = run_retry(
            [prompt], [db_path], timed_generate,
            max_attempts=max_attempts, timeout=self.timeout,
        )
        wall_ms = (time.perf_counter() - wall_started) * 1000
        execute_ms = max(wall_ms - generate_ms_accum, 0.0)

        attempts = histories[0]
        final = attempts[-1]

        # retry.Attempt does not carry result rows (see its docstring) -- only
        # status and error. Re-running the final query is the only way to get
        # rows for the response; folded into execute_ms since it is more
        # SQLite execution, not generation.
        exec_started = time.perf_counter()
        result = run(final.sql, db_path, timeout=self.timeout)
        execute_ms += (time.perf_counter() - exec_started) * 1000
        rows = [list(row) for row in result.rows] if result.status == "ok" else []

        return _Outcome(
            sql=final.sql,
            rows=rows,
            status=final.status,
            error=final.error,
            confidence=confidence(agreement=1, samples=1),
            attempts=len(attempts),
            generate_ms=generate_ms_accum,
            execute_ms=execute_ms,
        )

    def _vote_answer(
        self, prompt: Prompt, db_path: Path, *, samples: int, temperature: float
    ) -> _Outcome:
        """The voting path: greedy + ``samples - 1`` sampled candidates,
        clustered and selected by ``eval.voting`` -- same ordering
        ``run_eval.generate_votes`` uses, so ``vote_at``-equivalent behaviour
        holds here too.
        """
        generate_started = time.perf_counter()
        greedy = self.backend.generate([prompt], max_new_tokens=self.max_new_tokens)[0]
        sampled = self.backend.sample(
            [prompt], n=samples - 1, temperature=temperature,
            max_new_tokens=self.max_new_tokens,
        )[0]
        generate_ms = (time.perf_counter() - generate_started) * 1000

        execute_started = time.perf_counter()
        executed = [_execute(raw, db_path, self.timeout) for raw in (greedy, *sampled)]
        execute_ms = (time.perf_counter() - execute_started) * 1000

        candidates = [candidate for candidate, _ in executed]
        winner_index = voting.select(candidates, demote_empty=True)
        winner, winner_error = executed[winner_index]
        agreement = winner_agreement(candidates, winner_index)

        return _Outcome(
            sql=winner.sql,
            rows=winner.rows,
            status=winner.status,
            error=winner_error,
            confidence=confidence(agreement, samples),
            attempts=1,  # one generation round; voting never retries
            generate_ms=generate_ms,
            execute_ms=execute_ms,
        )
