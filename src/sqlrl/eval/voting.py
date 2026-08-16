"""Execution voting for evaluation: generate k candidates, let them vote.

Self-consistency, grounded in execution rather than in the text of the SQL.
Instead of one greedy query, sample k candidates, run every one of them, group
the ones that return identical rows, and answer with the largest group. Two
differently-written queries that land on the same result set are probably both
right; a hallucinated one usually returns something nobody else returns.

**Why this exists.** The retry loop (``retry.py``) only has something to work
with when the database *rejects* a query -- an ``"error"`` gives it a message
to feed back. It has nothing for the 483 of 2,147 Spider test queries that
execute perfectly and answer the wrong question: no error, no feedback, no
loop. Those wrong-but-valid queries usually disagree with their siblings, which
is a signal retry cannot see and voting can.

**Cluster with ``compare``, not a hash.** ``executor.compare`` runs a bounded
column-permutation search, so two result sets that differ only in column order
count as equal -- ``SELECT age, name`` and ``SELECT name, age`` are the same
answer. A hash of the row set would split those into different buckets, which
is exactly the kind of false disagreement voting exists to avoid. O(k^2)
pairwise comparisons at k=8 is 28 calls to ``compare`` per question, at roughly
0.04ms each: free.

**The empty-result trap.** Empty result sets all compare equal to each other,
regardless of *why* they are empty. ``WHERE 1 = 0``, a hallucinated filter that
happens to match nothing, and a genuinely empty answer all land in the same
cluster and vote as a bloc -- unlike every other cluster, where agreement is
evidence of correctness, an empty cluster's size mostly measures how many ways
there are to return nothing. This is the same hazard documented in
``executor.py``'s module docstring for plain scoring, made worse here because
the empties actively collude to outvote a real, minority answer.
``select(demote_empty=True)`` (the default) is the guard: an empty cluster only
wins if there is no non-empty cluster at all. Only 54 of 2,147 Spider test gold
answers are empty (2.52%), so the guard can cost at most that slice, in
exchange for removing the collusion risk on the other 97.5%.

**Gold is used in exactly one place here: ``oracle_at``.** Everything up to and
including ``vote_at`` -- clustering, selection, the k-curve -- never looks at
gold, because at real inference time there is no gold to look at. ``oracle_at``
computes pass@k, the ceiling a perfect selector could reach; it exists purely
to be compared against vote@k after the fact, the same way ``metrics.py``
scores a retry loop's output only once the loop has already committed to an
answer.

    cluster(candidates)                      -> list[list[int]]
    select(candidates, demote_empty=True)     -> int
    vote_at(ballot, k, demote_empty=True)     -> Candidate
    oracle_at(ballot, k, gold_rows, ordered)  -> bool
    cluster_stats(ballot, k)                  -> (n_clusters, largest)
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlrl.eval.executor import compare, requires_order

__all__ = [
    "Ballot",
    "Candidate",
    "cluster",
    "cluster_stats",
    "oracle_at",
    "select",
    "vote_at",
]


@dataclass(frozen=True)
class Candidate:
    raw: str  # full model output
    sql: str  # extract_sql(raw)
    status: str  # ExecResult.status: "ok" | "error" | "timeout" | "too_many_rows"
    rows: list  # ExecResult.rows (empty when not ok)


@dataclass
class Ballot:
    index: int  # index into the benchmark example list
    db_id: str
    question: str
    gold_sql: str
    candidates: list[Candidate]  # candidates[0] is ALWAYS the greedy sample


def cluster(candidates: list[Candidate]) -> list[list[int]]:
    """Group candidates by the rows they return, using ``compare`` -- never a
    hash. See the module docstring for why a hash would split groups the
    scorer considers identical.

    Only ``status == "ok"`` candidates are eligible: a query the database
    refused has no result to agree with, so it is dropped before clustering
    rather than forming a cluster of its own.

    Greedy clustering, one pass: for each eligible candidate, in order,
    compare it against every existing cluster's representative (the candidate
    that founded that cluster) and join the first match, or start a new
    cluster if none matches.

    Returns lists of indices into ``candidates``, largest cluster first, ties
    broken by the smallest index each cluster contains -- deterministic, so
    the same k candidates always cluster the same way.
    """
    clusters: list[list[int]] = []
    for i, candidate in enumerate(candidates):
        if candidate.status != "ok":
            continue

        joined = False
        for indices in clusters:
            representative = candidates[indices[0]]
            # order_matters should come from the gold query's requires_order,
            # but there is no gold at inference time -- see the module
            # docstring. Using the cluster representative's own SQL is an
            # inference-time approximation of what score_example does with
            # gold. It can only affect *grouping* here; the final score always
            # goes through metrics.score_example, which uses gold's own
            # requires_order regardless of how voting clustered anything.
            if compare(
                candidate.rows, representative.rows, requires_order(representative.sql)
            ):
                indices.append(i)
                joined = True
                break
        if not joined:
            clusters.append([i])

    clusters.sort(key=lambda indices: (-len(indices), min(indices)))
    return clusters


def select(candidates: list[Candidate], *, demote_empty: bool = True) -> int:
    """The index of the winning candidate: the largest cluster's representative.

    ``demote_empty=True`` (the default) is the guard against the empty-result
    trap described in the module docstring: a cluster whose representative
    returned zero rows is only chosen if every cluster is empty -- there is
    nothing else to pick. ``demote_empty=False`` skips the guard, so both can
    be measured rather than the choice being asserted.

    Falls back to index 0 (the greedy candidate) when no candidate executed at
    all, so the answer degrades to greedy rather than to nothing.
    """
    clusters = cluster(candidates)
    if not clusters:
        return 0

    if demote_empty:
        for indices in clusters:
            representative = candidates[indices[0]]
            if representative.rows:
                return indices[0]
        # Every ok cluster is empty. Nothing else to pick -- fall through to
        # the largest (still empty) cluster below.

    return clusters[0][0]


def vote_at(ballot: Ballot, k: int, *, demote_empty: bool = True) -> Candidate:
    """Vote using only the first ``k`` candidates -- one generation run yields
    the whole k-curve (1, 2, 4, 8, ...) with no extra GPU time. Same trick as
    ``retry.at_budget``.

    ``ballot.candidates[0]`` is always the greedy sample, so ``vote_at(ballot,
    1)`` must return exactly it -- a wiring-bug detector, since it means
    vote@1 reproduces the existing greedy score exactly. Tie-breaking to the
    smallest index also means that when every candidate in the budget
    disagrees with every other, the winner falls back to greedy.
    """
    if k < 1 or k > len(ballot.candidates):
        raise ValueError(f"k must be between 1 and {len(ballot.candidates)}, got {k}")
    subset = ballot.candidates[:k]
    return subset[select(subset, demote_empty=demote_empty)]


def oracle_at(ballot: Ballot, k: int, gold_rows: list, order_matters: bool) -> bool:
    """pass@k: is ANY of the first k candidates a match for gold?

    The ceiling voting is trying to reach. The gap between vote@k and this is
    how much better a perfect selector could do -- the most informative number
    in the whole experiment. Unlike everything else in this module, this DOES
    take gold: it is a scoring-time diagnostic, never part of the voting loop
    itself, which cannot see gold at inference time.
    """
    return any(
        candidate.status == "ok" and compare(candidate.rows, gold_rows, order_matters)
        for candidate in ballot.candidates[:k]
    )


def cluster_stats(ballot: Ballot, k: int) -> tuple[int, int]:
    """``(number of distinct clusters, size of the largest cluster)`` among the
    first k candidates -- how much the samples actually agreed with each
    other, independent of whether they agreed with gold.
    """
    clusters = cluster(ballot.candidates[:k])
    if not clusters:
        return 0, 0
    return len(clusters), len(clusters[0])
