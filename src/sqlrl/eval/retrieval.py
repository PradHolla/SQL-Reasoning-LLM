"""Schema retrieval: the stress test Spider, handed whole, cannot pose.

    uv run python -m sqlrl.eval.retrieval --split test --k 1,3,5,10,20
    uv run python -m sqlrl.eval.retrieval --split test --k 1,3,5,10,20 --retriever bm25

**Why this module exists.** Every checkpoint here has been evaluated with the
one correct database handed over in full: Spider test's 206 databases have a
median of **4 tables and 19 columns**, and a single database renders to
roughly **102 prompt tokens**. Retrieval over that measures nothing -- there is
nothing to retrieve *from*. Real deployments do not look like this: production
schemas run to hundreds of tables that do not fit in a context window, and the
model has to find the handful it needs before it can write a query at all.
This module builds the pool that makes that gap measurable.

**Pooling all 206 databases naively does not work.** Table names collide
across them: 125 distinct table names appear in more than one database,
covering 436 of 1,053 tables (41%). ``customers`` alone appears in 22
databases, ``addresses`` in 15, ``student`` in 12. A question asking about
"customers" over that pool is genuinely ambiguous -- there is no way for any
retriever, however good, to know which of the 22 the question means, because
the question itself does not disambiguate them. Scoring retrieval over an
unresolvable pool would measure that ambiguity, not retrieval quality.

**The fix is a collision-free subset.** ``build_pool`` takes databases
greedily, in descending order of how many benchmark questions they carry,
skipping any database whose table names clash with a database already taken.
That yields **81 databases, 300 tables, 1,427 columns, and 1,457 of 2,147
questions (68%)** -- smaller than the full test split, but internally
unambiguous: no two kept tables share a name.

**And retrieval is not optional over that pool.** Rendered as CREATE TABLE
statements the same way a single database is, the 300-table pool comes to
**8,262 tokens against the 3,072-token input cap this project's checkpoints
were evaluated under -- 2.7x over.** There is room for roughly 92 tables in
the prompt budget, a third of the pool. Retrieval has to cut that down, every
time, or the prompt does not fit.

    TableDoc(db_id, table, columns)            -> .key, .text
    build_pool(split)                          -> list[TableDoc]
    pool_questions(examples, pool)             -> list[Example]
    gold_tables(gold_sql, db_id)               -> set[(db_id, table)]
    tokenize(text)                             -> list[str]
    BM25(docs).search(question, k)             -> list[int]
    Dense(docs).search(question, k)            -> list[int]
    Dense(docs).search_many(questions, k)      -> list[list[int]]
    recall_at_k(retrieved, gold)               -> float
    coverage_at_k(retrieved, gold)             -> bool
    render_pool_schema(docs, types)            -> str
"""

from __future__ import annotations

import argparse
import math
import re
from collections import Counter
from dataclasses import dataclass

from sqlglot import expressions as exp

from sqlrl.eval.executor import parse_sql, read_schema
from sqlrl.eval.spider import SPLITS, Example, load_split

__all__ = [
    "BM25",
    "Dense",
    "TableDoc",
    "build_pool",
    "coverage_at_k",
    "gold_tables",
    "pool_questions",
    "recall_at_k",
    "render_pool_schema",
    "tokenize",
]

#: A retrieved document per table, one per (db_id, table) pair.
_Key = tuple[str, str]


@dataclass(frozen=True)
class TableDoc:
    """One retrievable unit: a table, with the columns a retriever can match
    a question against. Never carries type information -- types are cosmetic
    for retrieval (see ``render_pool_schema``, which is where they matter).
    """

    db_id: str
    table: str
    columns: tuple[str, ...]  # column names only, in schema (PRAGMA) order

    @property
    def key(self) -> _Key:
        """Identity for collision-checking and gold-table lookups. Lowercased
        because SQL identifiers are case-insensitive in SQLite and gold SQL is
        not consistently cased with the schema that defines it.
        """
        return (self.db_id, self.table.lower())

    @property
    def text(self) -> str:
        """What a retriever indexes: ``"student: id, name, age, dept_id"``.
        The table name carries as much signal as the columns for schema
        linking, so it goes in the text too, not just the doc's identity.
        """
        return f"{self.table}: {', '.join(self.columns)}"


# --------------------------------------------------------------------------
# pool construction
# --------------------------------------------------------------------------


def build_pool(split: str = "test") -> list[TableDoc]:
    """The collision-free subset of ``split`` -- see the module docstring for
    why pooling every database naively is unusable, and what number this
    greedy construction is supposed to land on (81 databases, 300 tables,
    1,427 columns, over Spider test).

    **Candidate databases come from the split's database directory, not from
    the benchmark questions.** Spider's test release ships 206 database
    directories under ``test_database/``, but ``test.json`` only draws
    questions from 40 of them -- the rest carry no question at all. Those 166
    still count as real candidates: a retrieval stress test is about how much
    of a large schema pool a retriever can navigate, and a pool restricted to
    only the 40 question-bearing databases would be a much smaller, easier
    problem (and would not reach the ~8k-token, 2.7x-over-budget pool the
    module docstring describes). The directory is derived from an example's
    own ``db_path`` (``db_path.parent.parent``) rather than hardcoded, so this
    same code resolves ``dev``'s ``database/`` directory correctly too.

    Databases are visited in order of how many benchmark questions they
    carry, descending (the 166 question-less ones all sort after every
    question-bearing one, at count 0) -- keeping the databases that cover the
    most questions is what makes ``pool_questions`` retain as much of the
    benchmark as possible. Ties break on ``db_id`` ascending so the whole
    construction is deterministic: the pool *is* the benchmark from here on,
    and a nondeterministic pool would make every retrieval number
    irreproducible.

    A database is kept only if none of its (lowercased) table names collide
    with a table name already kept by an earlier (higher-priority) database.
    Rejected databases are simply skipped -- there is no partial keeping of a
    database's tables, because that would silently change what "the
    database's schema" means for every question against it.
    """
    examples = load_split(split)
    if not examples:
        return []

    databases_dir = examples[0].db_path.parent.parent
    db_ids = sorted(path.name for path in databases_dir.iterdir() if path.is_dir())

    question_counts: Counter[str] = Counter(example.db_id for example in examples)
    schemas = {
        db_id: read_schema(databases_dir / db_id / f"{db_id}.sqlite") for db_id in db_ids
    }

    order = sorted(db_ids, key=lambda db_id: (-question_counts[db_id], db_id))

    seen_tables: set[str] = set()
    kept_dbs: list[str] = []
    for db_id in order:
        table_names = {table.lower() for table in schemas[db_id]}
        if table_names & seen_tables:
            continue
        kept_dbs.append(db_id)
        seen_tables |= table_names

    docs = [
        TableDoc(db_id=db_id, table=table, columns=tuple(columns))
        for db_id in kept_dbs
        for table, columns in schemas[db_id].items()
    ]
    docs.sort(key=lambda doc: (doc.db_id, doc.table))
    return docs


def pool_questions(examples: list[Example], pool: list[TableDoc]) -> list[Example]:
    """``examples`` restricted to the databases ``pool`` kept -- the 1,457-
    question subset every retrieval number in this module is measured over.

    Scores on this subset are NOT comparable to a score on the full
    2,147-question Spider test table: it is a different, smaller, easier-by-
    construction set of questions (their databases were kept precisely
    because they are large or well-covered). Any comparison against a non-
    retrieval baseline must run that baseline (``--retriever none``, once
    wired into the eval CLI) over this same pooled subset, not the full split
    -- otherwise the "gap" being reported is partly just a different
    denominator.
    """
    pooled_dbs = {doc.db_id for doc in pool}
    return [example for example in examples if example.db_id in pooled_dbs]


def gold_tables(gold_sql: str, db_id: str) -> set[_Key]:
    """Every table the gold query touches -- the ground truth ``recall_at_k``
    and ``coverage_at_k`` are scored against.

    Walks every ``exp.Table`` node in the parsed tree, not only the ones in a
    top-level ``FROM``/``JOIN``: a table referenced solely inside a subquery
    or CTE still has to be in the model's context for the question to be
    answerable, so it counts here too.

    Unparseable gold SQL (``parse_sql`` returns ``None``) returns the empty
    set rather than raising. That is data about the benchmark -- already
    reported through parse rate elsewhere -- not a reason to crash a sweep
    over 1,457 questions.
    """
    tree = parse_sql(gold_sql)
    if tree is None:
        return set()
    return {(db_id, table.name.lower()) for table in tree.find_all(exp.Table)}


# --------------------------------------------------------------------------
# tokenisation
# --------------------------------------------------------------------------

_WORD = re.compile(r"[A-Za-z0-9_]+")
#: Splits camelCase at two boundaries: a lowercase/digit followed by an
#: uppercase letter ("singerID" -> "singer" | "ID"), and an uppercase letter
#: followed by an uppercase-then-lowercase run ("XMLParser" -> "XML" |
#: "Parser"). Underscores are handled separately in ``tokenize``, since they
#: split into their own boundary rule (``str.split("_")``) before this runs.
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def _camel_parts(chunk: str) -> list[str]:
    return [part for part in _CAMEL_BOUNDARY.split(chunk) if part]


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens, expanded so a plain-English question can match a
    schema identifier that spells the same concept in ``snake_case`` or
    ``camelCase``. This is most of what schema linking is: a question says
    "city" and the column is named ``city_code``; a tokenizer that only
    lowercases and splits on punctuation never lets those match, because
    ``city_code`` is a single token to it.

    Every raw word (split on anything that is not a letter, digit, or
    underscore) expands to itself lowercased **and** its snake_case/camelCase
    parts, so ``city_code`` -> ``["city_code", "city", "code"]`` and
    ``singerID`` -> ``["singerid", "singer", "id"]``. The joined form is kept
    alongside the parts, not replaced by them, so a query that uses the
    identifier verbatim still gets credit for the exact match.

    Expansion is deduplicated within one word's own occurrence, not across
    the whole text: a plain word like "student" (no separator, no case
    change) would otherwise contribute itself twice per occurrence for
    nothing, but two real occurrences of "student" in the text must still
    produce two tokens -- BM25's term-frequency term depends on that count
    being real.
    """
    tokens: list[str] = []
    for word in _WORD.findall(text):
        pieces = [word.lower()]
        for chunk in word.split("_"):
            pieces.extend(part.lower() for part in _camel_parts(chunk))

        seen: set[str] = set()
        for piece in pieces:
            if piece and piece not in seen:
                seen.add(piece)
                tokens.append(piece)
    return tokens


# --------------------------------------------------------------------------
# BM25
# --------------------------------------------------------------------------


class BM25:
    """Okapi BM25 over ``doc.text``, implemented directly -- 300 documents is
    far too small to justify a new dependency for this.

    IDF and document lengths are computed once, in ``__init__``, since they
    depend only on the corpus, not the query. The IDF formula is
    ``log(1 + (N - df + 0.5) / (df + 0.5))``, the "BM25+"-style variant that
    stays non-negative even for a term appearing in most documents -- unlike
    the original Robertson-Sparck Jones form, which goes negative there and
    would let a common term actively *penalise* a document for matching it.

    **The query is the question text and nothing else.** ``search`` takes
    ``question: str``, not an ``Example`` or a ``db_id`` -- that is
    deliberate and load-bearing: if the retriever ever saw which database the
    question came from, it could look up that database's tables directly and
    every recall/coverage number in this module would be measuring an
    answer key, not retrieval.
    """

    def __init__(self, docs: list[TableDoc], k1: float = 1.5, b: float = 0.75) -> None:
        self.docs = docs
        self.k1 = k1
        self.b = b

        doc_tokens = [tokenize(doc.text) for doc in docs]
        self._term_freqs = [Counter(tokens) for tokens in doc_tokens]
        self._doc_lens = [len(tokens) for tokens in doc_tokens]
        self._avg_len = sum(self._doc_lens) / len(docs) if docs else 0.0

        doc_freqs: Counter[str] = Counter()
        for freqs in self._term_freqs:
            doc_freqs.update(freqs.keys())
        n = len(docs)
        self._idf = {
            term: math.log(1 + (n - df + 0.5) / (df + 0.5))
            for term, df in doc_freqs.items()
        }

    def search(self, question: str, k: int) -> list[int]:
        """Indices into ``docs``, best match first, at most ``k`` of them.
        Ties break by document index ascending, so results are deterministic.
        """
        query_terms = tokenize(question)
        scores = [self._score(i, query_terms) for i in range(len(self.docs))]
        order = sorted(range(len(self.docs)), key=lambda i: (-scores[i], i))
        return order[:k]

    def _score(self, doc_index: int, query_terms: list[str]) -> float:
        freqs = self._term_freqs[doc_index]
        doc_len = self._doc_lens[doc_index]
        norm = self.k1 * (1 - self.b + self.b * doc_len / self._avg_len) if self._avg_len else 0.0

        total = 0.0
        for term in query_terms:
            tf = freqs.get(term, 0)
            idf = self._idf.get(term)
            if tf == 0 or idf is None:
                continue
            total += idf * (tf * (self.k1 + 1)) / (tf + norm)
        return total


# --------------------------------------------------------------------------
# dense retrieval
# --------------------------------------------------------------------------


class Dense:
    """Sentence embeddings via ``transformers`` directly -- mean pooling over
    the attention mask, then L2 normalisation, then a dot product, which is
    what ``sentence-transformers`` does under the hood for this model family.
    Not adding that library for one encoder call is worth the ~30 lines it
    costs to spell out here.

    Documents are encoded once, in ``__init__`` (300 tables is cheap even on
    CPU). Queries are encoded per call in ``search``, or batched in
    ``search_many`` for a full sweep -- encoding 1,457 questions one at a
    time would be needlessly slow, and this CLI's ``--retriever dense`` sweep
    is exactly that workload.

    Like ``BM25.search``, the query is the question text alone; see that
    class's docstring for why that is load-bearing, not a style choice.

    ``torch``/``transformers`` are imported inside the methods that need
    them, not at module scope -- the ``# Imported here so --score-only never
    pays for torch`` pattern in ``run_eval.py`` -- so ``--retriever bm25`` and
    every BM25 test can run without torch installed at all.
    """

    def __init__(
        self,
        docs: list[TableDoc],
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str | None = None,
        batch_size: int = 64,
    ) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.docs = docs
        self.batch_size = batch_size
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self._doc_embeddings = self._encode([doc.text for doc in docs])

    def search(self, question: str, k: int) -> list[int]:
        """One question at a time. See ``search_many`` to batch several."""
        return self.search_many([question], k)[0]

    def search_many(self, questions: list[str], k: int) -> list[list[int]]:
        """Batched ``search``: one encoder pass over all of ``questions``
        rather than one pass per question. Ties break by document index
        ascending, matching ``BM25.search``.
        """
        query_embeddings = self._encode(questions)
        scores = (query_embeddings @ self._doc_embeddings.T).tolist()
        n_docs = len(self.docs)
        return [
            sorted(range(n_docs), key=lambda i, row=row: (-row[i], i))[:k]
            for row in scores
        ]

    def _encode(self, texts: list[str]):
        import torch

        chunks = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            encoded = self._tokenizer(
                batch, padding=True, truncation=True, return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                output = self._model(**encoded)
            pooled = self._mean_pool(output.last_hidden_state, encoded["attention_mask"])
            chunks.append(torch.nn.functional.normalize(pooled, p=2, dim=1))
        return torch.cat(chunks, dim=0)

    @staticmethod
    def _mean_pool(last_hidden_state, attention_mask):
        """Token embeddings averaged over the (non-pad) attention mask -- the
        pooling all-MiniLM-L6-v2 was trained to be read with; its [CLS]
        embedding, unlike BERT's, is not a usable sentence representation on
        its own.
        """
        mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
        summed = (last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------


def recall_at_k(retrieved: list[TableDoc], gold: set[_Key]) -> float:
    """Fraction of ``gold`` tables present among ``retrieved``.

    Returns 1.0 when ``gold`` is empty -- an unparseable gold query (see
    ``gold_tables``) required nothing, so nothing was missed. Folding that
    into 0.0 would punish the retriever for a benchmark artifact it has no
    way to satisfy.
    """
    if not gold:
        return 1.0
    retrieved_keys = {doc.key for doc in retrieved}
    return sum(1 for key in gold if key in retrieved_keys) / len(gold)


def coverage_at_k(retrieved: list[TableDoc], gold: set[_Key]) -> bool:
    """True only when EVERY gold table is present among ``retrieved``.

    This is the decisive metric, more than recall: missing one table of a
    two-table join makes the question unanswerable regardless of how good the
    model is at writing SQL, and regardless of how high recall reads on
    average across the benchmark. A retriever that gets 90% recall by
    consistently dropping the second table of every join question is not 90%
    as useful as one that gets every join right -- it is useless on every
    join question, which coverage is what exposes.
    """
    retrieved_keys = {doc.key for doc in retrieved}
    return gold.issubset(retrieved_keys)


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------


def render_pool_schema(
    docs: list[TableDoc],
    types: dict[_Key, dict[str, str]] | None = None,
) -> str:
    """The retrieved tables as ``CREATE TABLE ...; CREATE TABLE ...`` text --
    the same shape ``prompts.render_schema`` produces for a single database,
    so what a retriever hands the model is byte-compatible with what the
    checkpoints were trained on.

    ``types`` maps ``TableDoc.key`` (``(db_id, table.lower())``) to a
    ``{column: type}`` dict, typically built by calling ``executor.read_schema``
    once per database in the pool. When it is ``None`` (or missing an entry),
    every column renders as ``TEXT`` -- retrieval and column-name matching
    both work without real types, but a real evaluation prompt should still
    pass them in, since the model was trained on the database's actual SQLite
    types, not a placeholder.
    """
    types = types or {}
    statements = []
    for doc in docs:
        column_types = types.get(doc.key, {})
        columns = ", ".join(
            f"{column} {column_types.get(column, 'TEXT')}" for column in doc.columns
        )
        statements.append(f"CREATE TABLE {doc.table} ({columns})")
    return "; ".join(statements)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

_RETRIEVERS = ("bm25", "dense")


def _mean_scores(
    top_k: list[list[int]], pool: list[TableDoc], golds: list[set[_Key]], k: int
) -> tuple[float, float]:
    recalls = []
    coverages = []
    for indices, gold in zip(top_k, golds):
        retrieved = [pool[i] for i in indices[:k]]
        recalls.append(recall_at_k(retrieved, gold))
        coverages.append(coverage_at_k(retrieved, gold))
    return sum(recalls) / len(recalls), sum(coverages) / len(coverages)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure schema retrieval over the collision-free Spider pool."
    )
    parser.add_argument("--split", choices=SPLITS, default="test")
    parser.add_argument("--k", default="1,3,5,10,20",
                        help="comma-separated list of k values to report")
    parser.add_argument("--retriever", default="bm25,dense",
                        help="comma-separated subset of: " + ", ".join(_RETRIEVERS))
    args = parser.parse_args()

    ks = sorted({int(k) for k in args.k.split(",")})
    retrievers = args.retriever.split(",")
    unknown = [r for r in retrievers if r not in _RETRIEVERS]
    if unknown:
        parser.error(f"unknown retriever(s): {unknown}. Choose from {_RETRIEVERS}")

    pool = build_pool(args.split)
    examples = load_split(args.split)
    questions = pool_questions(examples, pool)

    n_dbs = len({doc.db_id for doc in pool})
    n_cols = sum(len(doc.columns) for doc in pool)
    print(f"pool ({args.split}): {n_dbs} databases, {len(pool)} tables, "
          f"{n_cols} columns")
    print(f"pooled questions: {len(questions)} of {len(examples)} "
          f"({len(questions) / len(examples):.1%}) -- NOT comparable to a "
          "full-split score; see pool_questions' docstring\n")

    golds = [gold_tables(example.gold_sql, example.db_id) for example in questions]
    max_k = max(ks)

    for name in retrievers:
        print(f"=== {name} ===")
        if name == "bm25":
            index = BM25(pool)
            top_k = [index.search(example.question, max_k) for example in questions]
        else:
            index = Dense(pool)
            top_k = index.search_many([example.question for example in questions], max_k)

        print(f"{'k':>4}  {'recall@k':>10}  {'coverage@k':>12}")
        for k in ks:
            mean_recall, mean_coverage = _mean_scores(top_k, pool, golds, k)
            print(f"{k:>4}  {mean_recall:>10.1%}  {mean_coverage:>12.1%}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
