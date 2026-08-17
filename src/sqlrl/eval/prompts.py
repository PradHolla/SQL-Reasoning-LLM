"""Build model inputs and pull SQL back out of model outputs.

Both halves are places where a bug quietly *lowers* the score, so they live in
one file with tests rather than being inlined into the eval loop.

**The prompt must match training exactly.** v0 trained in two different shapes
and they are not interchangeable:

* SFT and GRPO used ChatML with a fixed system message and ``Schema: ...``
  ``Question: ...`` in the user turn -- reproduced verbatim in ``chat_prompt``.
* CPT used raw text with ``-- Database Schema --`` / ``-- Executed SQL Query --``
  headers and no chat template at all. Handing the CPT checkpoint a ChatML
  prompt measures the wrong thing, so ``cpt_prompt`` reproduces that instead.

One difference we cannot paper over: ``sql-create-context`` schemas are
**pruned to the columns the question needs**, and typed loosely (VARCHAR for
everything). Spider hands over the full schema with real SQLite types, so the
model has to do its own schema linking at evaluation time -- work it never had
to do during training. That is a genuine distribution shift, not a bug, and it
is a large part of why the numbers will look bad. Worth remembering when
reading them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = [
    "Prompt",
    "SYSTEM_PROMPT",
    "chat_prompt",
    "cpt_prompt",
    "extract_sql",
    "render_schema",
    "stopped_cleanly",
]

#: Verbatim from format_sft_data.py / grpo_trainer.py. Do not "improve" it --
#: the point is to match what the checkpoints were trained against.
SYSTEM_PROMPT = (
    "You are a database expert. You must think step-by-step inside "
    "<think></think> tags, and output ONLY the final SQL query inside "
    "<answer></answer> tags."
)


@dataclass(frozen=True)
class Prompt:
    """Either a chat conversation or a raw completion string, never both.

    Backends render this themselves, because applying a chat template needs the
    checkpoint's own tokenizer.
    """

    messages: list[dict[str, str]] | None = None
    text: str | None = None

    def __post_init__(self) -> None:
        if (self.messages is None) == (self.text is None):
            raise ValueError("Prompt takes exactly one of messages or text")


def render_schema(schema: dict[str, dict[str, str]]) -> str:
    """``{table: {column: type}}`` as the CREATE TABLE text the models saw.

    Matches ``sql-create-context``: statements joined with ``"; "``, no trailing
    semicolon.
    """
    return "; ".join(
        f"CREATE TABLE {table} ({', '.join(f'{col} {typ}' for col, typ in columns.items())})"
        for table, columns in schema.items()
    )


def chat_prompt(schema_text: str, question: str) -> Prompt:
    """The SFT/GRPO shape, byte-for-byte."""
    return Prompt(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Schema: {schema_text}\nQuestion: {question}"},
        ]
    )


def cpt_prompt(schema_text: str, question: str) -> Prompt:
    """The CPT shape: raw text, cut where the model should start writing SQL."""
    return Prompt(
        text=(
            "\n-- Database Schema --\n"
            f"{schema_text}\n\n"
            "-- Executed SQL Query --\n"
            f"-- Intent: {question}\n"
        )
    )


# --------------------------------------------------------------------------
# getting SQL back out
# --------------------------------------------------------------------------

_ANSWER = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_ANSWER_OPEN = re.compile(r"<answer>(.*)", re.DOTALL | re.IGNORECASE)
_THINK_CLOSE = re.compile(r"</think>(.*)", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"```(?:sql)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_FENCE_OPEN = re.compile(r"```(?:sql)?\s*(.*)", re.DOTALL | re.IGNORECASE)
_START = re.compile(r"\b(SELECT|WITH)\b", re.IGNORECASE)
#: A complete XML-ish tag. Requires a letter after "<" and a closing ">", so it
#: cannot match SQL's "<", "<=" or the "<>" not-equal operator.
_TAG = re.compile(r"</?[a-zA-Z][\w-]*\s*>")


def extract_sql(text: str) -> str:
    """Pull the SQL out of whatever the model produced.

    Handles the tagged format we trained for, markdown fences, and bare SQL with
    prose around it. Unterminated forms are handled deliberately: a model that
    hits the token limit mid-answer has still produced a query worth scoring,
    and treating that as empty would understate the model rather than the
    prompt.

    Returns ``""`` when there is nothing query-shaped, which the executor then
    reports as ``not_a_query``.
    """
    if not text:
        return ""

    body = _first_group(text, _ANSWER, _ANSWER_OPEN)
    if body is None:
        # No answer tags. Anything after </think> is the model's real output.
        body = _first_group(text, _THINK_CLOSE) or text

    fenced = _first_group(body, _FENCE, _FENCE_OPEN)
    if fenced is not None:
        body = fenced

    return _tidy(body)


_ANSWER_CLOSE = re.compile(r"</answer>", re.IGNORECASE)


def stopped_cleanly(text: str) -> bool | None:
    """Did generation stop right after the answer, or keep running past it?

    This is a **text proxy**, not the real signal. What we actually want to
    know is whether generation ended on a stop token or was cut off at
    ``max_new_tokens`` -- but the backend does not currently record that, so
    there is no ground truth to read. What we have instead is the completion
    text, and a model that stopped cleanly leaves nothing after its closing
    ``</answer>`` while one that did not keeps emitting tokens -- typically
    unrelated pretraining memories -- until the token budget runs out. The
    proxy's one big advantage over fixing the backend is that it can be
    computed retroactively over every result file already on disk, without
    regenerating anything.

    Anchored on the **first** closing ``</answer>``, matching what
    ``extract_sql`` does when it builds the prediction. That choice is the
    whole metric: a runaway completion often emits further ``</answer>`` tags
    among the text it rambles into, and anchoring on the last one credits the
    model with stopping cleanly whenever the ramble happens to end with a
    closing tag. On ``sft-spider`` that alone is the difference between 2.9%
    and 0.0% -- a metric built to expose this bug would have hidden 63 cases of
    it.

    Returns ``True`` when the first ``</answer>`` has nothing but whitespace
    after it, ``False`` when it has trailing non-whitespace content, and
    ``None`` when there is no closing ``</answer>`` at all. ``None`` is
    deliberately not folded into ``False``: it covers the CPT completion
    format (no tags to look for, see ``cpt_prompt``) and answers truncated
    mid-block, both of which already show up in parse rate. Counting them here
    too would make those baselines read as 0% stopped for a reason that has
    nothing to do with whether the model knows how to stop.
    """
    match = _ANSWER_CLOSE.search(text)
    if match is None:
        return None
    return text[match.end() :].strip() == ""


def _first_group(text: str, *patterns: re.Pattern[str]) -> str | None:
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            return match.group(1)
    return None


def _tidy(sql: str) -> str:
    sql = sql.strip().strip("`").strip()

    # Drop any preamble before the query actually starts ("Here is the SQL: ...").
    start = _START.search(sql)
    if start:
        sql = sql[start.start() :]
    elif not sql.lower().startswith(("select", "with", "insert", "update", "delete")):
        # Nothing query-shaped at all.
        return ""

    # Cut at the first stray tag. v0's SFT models routinely emit mismatched
    # tags -- "<think>...<answer>SELECT ...</think>" -- and the unterminated
    # branch above would otherwise carry "</think>" into the query and turn a
    # possibly-correct answer into a syntax error.
    tag = _TAG.search(sql)
    if tag:
        sql = sql[: tag.start()]

    # Cut trailing commentary after the statement ends. A single SELECT has no
    # internal semicolon, so the first one is the end of the query.
    semicolon = sql.find(";")
    if semicolon != -1:
        sql = sql[:semicolon]

    return " ".join(sql.split())
