"""Sample reasoning traces from a teacher, keep only the ones whose SQL is correct.

    uv run python -m sqlrl.data_prep.sample_traces --limit 32   # smoke test
    uv run python -m sqlrl.data_prep.sample_traces

Rejection sampling, a.k.a. STaR / RFT, and roughly how R1's cold-start data was
built. Take a stronger model, sample k attempts per question, **execute each
one and keep only the attempts that actually produce the right rows**, then
train the small model on what survives.

The filter is the whole technique, and it is why Phase 1 had to come first:
`eval/executor.py` has been the benchmark metric and the RL reward, and this is
its third job. Correctness-filtered synthetic data beats more data, and it beats
unfiltered teacher output by a distance -- a plausible-looking trace ending in
wrong SQL is worse than no trace at all, because it teaches confident nonsense.

**What this is fixing.** The `<think>` block in the current SFT set is one
hardcoded sentence repeated on all 5,378 examples::

    "I need to analyze the schema to find the correct tables and columns,
     then construct a valid SQL query."

The model learned to recite a preamble, not to think. v0's completion length sat
flat at 55-70 tokens for all 300 GRPO steps for the same reason: there was no
reasoning there to grow.

**One variable.** Only the `<think>` content changes. The prompt the student
sees is still `chat_prompt(schema, question)`, byte for byte, built by the same
function the evaluator uses. If the new SFT checkpoint moves, it moved because
of the traces.

**No rationalisation.** STaR has a fallback where you show the teacher the gold
answer and ask it to justify it, which lifts coverage on hard questions. Not
done here: it teaches post-hoc justification rather than derivation, and it
would make coverage and trace quality two variables instead of one. If coverage
comes out badly skewed toward easy questions, that is a *finding* to report, and
the fallback is a follow-up experiment rather than a silent default.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from sqlrl.eval.executor import read_schema
from sqlrl.eval.metrics import score_example
from sqlrl.eval.prompts import extract_sql

__all__ = ["build_teacher_prompt", "pick_trace", "sample"]

TEACHER = "Qwen/Qwen2.5-Coder-7B-Instruct"
SFT_DATA = Path("data/processed/spider_sft.jsonl")
DEFAULT_OUT = Path("data/processed/spider_traces.jsonl")

#: Attempts per question. More is strictly better for coverage and strictly
#: worse for wall clock; 8 is where a 5,378-question pass stays inside an hour
#: on one A10G with vLLM.
SAMPLES_PER_QUESTION = 8

#: Sampling has to be hot enough that the k attempts actually differ -- k
#: identical greedy traces teaches nothing that one would not. Standard
#: rejection-sampling settings.
TEMPERATURE = 0.8
TOP_P = 0.95
MAX_TOKENS = 512

#: Questions per vLLM call. Only affects how much work an interruption costs:
#: results are appended after every chunk, so a spot reclaim loses at most this
#: many questions rather than the run.
CHUNK = 256

#: The teacher is told to name real tables and columns. This checks it did,
#: because "let me think about this step by step" is not reasoning and we would
#: be replacing one canned sentence with another.
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def build_teacher_prompt(schema_text: str, question: str) -> str:
    """What we ask the teacher for.

    Deliberately demands the *student's* output format, so the kept traces can
    be dropped into the SFT set unchanged, and demands that the reasoning name
    real schema identifiers -- which is the measurable half of Phase 3's goal.
    """
    return (
        "You are a database expert. Work out the SQL that answers the question.\n\n"
        "Think step by step inside <think></think> tags. In your thinking, name "
        "the actual tables and columns from the schema you need and say why you "
        "need them, including any joins and the reason for them. Then give the "
        "final SQLite query inside <answer></answer> tags, and nothing else "
        "after it.\n\n"
        f"Schema: {schema_text}\n"
        f"Question: {question}"
    )


def think_of(text: str) -> str:
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


def names_schema_identifiers(think: str, identifiers: set[str]) -> bool:
    """Does this reasoning actually mention something from the schema?

    Cheap, and it is the difference between a trace that reasons about *this*
    database and one that could have been written without reading the schema.
    """
    return any(word.lower() in identifiers for word in _IDENT.findall(think))


def pick_trace(
    candidates: list[str], gold_sql: str, db_path: str, identifiers: set[str]
) -> tuple[str | None, int]:
    """The best correct trace among k attempts, and how many were correct.

    Correct means the same thing it means everywhere else in this project:
    ``score_example(...).execution_match``, the benchmark's own definition. Not
    a looser one, or the training data would be graded more generously than the
    model ever will be.

    Among correct candidates, prefer ones whose reasoning names real schema
    identifiers, then the shortest. Shortest is a deliberate tie-break: long
    teacher traces tend to wander, and the student has 0.5B parameters and a
    2,048-token budget to spend.
    """
    correct = [
        text
        for text in candidates
        if (sql := extract_sql(text))
        and score_example(sql, gold_sql, db_path).execution_match
    ]
    if not correct:
        return None, 0

    grounded = [t for t in correct if names_schema_identifiers(think_of(t), identifiers)]
    pool = grounded or correct
    return min(pool, key=len), len(correct)


def _identifiers(db_path: str) -> set[str]:
    schema = read_schema(db_path)
    names = {table.lower() for table in schema}
    for columns in schema.values():
        names.update(column.lower() for column in columns)
    return names


def _done_indices(path: Path) -> set[int]:
    """Question indices already written, so a restart resumes instead of repeating.

    This is what makes the run spot-friendly: sampling has no optimiser state,
    so "resume" is just "skip what is already on disk".
    """
    if not path.is_file():
        return set()
    done = set()
    for line in path.read_text().splitlines():
        if line.strip():
            done.add(json.loads(line)["index"])
    return done


def sample(
    teacher: str = TEACHER,
    source: Path = SFT_DATA,
    out: Path = DEFAULT_OUT,
    k: int = SAMPLES_PER_QUESTION,
    limit: int | None = None,
    chunk: int = CHUNK,
    gpu_memory_utilization: float = 0.90,
) -> None:
    # Imported here, not at module scope: vLLM is an optional CUDA-only extra
    # and this module is imported by tests that run on a laptop.
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    if limit:
        rows = rows[:limit]
    for i, row in enumerate(rows):
        row["index"] = i

    out.parent.mkdir(parents=True, exist_ok=True)
    done = _done_indices(out)
    todo = [row for row in rows if row["index"] not in done]
    print(f"{len(rows)} questions, {len(done)} already sampled, {len(todo)} to go")
    if not todo:
        print("nothing to do")
        return

    tokenizer = AutoTokenizer.from_pretrained(teacher)
    llm = LLM(model=teacher, gpu_memory_utilization=gpu_memory_utilization,
              max_model_len=4096)
    params = SamplingParams(n=k, temperature=TEMPERATURE, top_p=TOP_P,
                            max_tokens=MAX_TOKENS)

    kept = 0
    correct_counts: Counter[int] = Counter()
    for start in range(0, len(todo), chunk):
        batch = todo[start : start + chunk]
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user",
                  "content": build_teacher_prompt(
                      _schema_text(row), row["question"])}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for row in batch
        ]
        outputs = llm.generate(prompts, params)

        with out.open("a") as handle:
            for row, output in zip(batch, outputs, strict=True):
                candidates = [o.text for o in output.outputs]
                trace, n_correct = pick_trace(
                    candidates, row["gold_sql"], row["db_path"],
                    _identifiers(row["db_path"]),
                )
                correct_counts[n_correct] += 1
                kept += trace is not None
                handle.write(json.dumps({
                    "index": row["index"],
                    "db_id": row["db_id"],
                    "db_path": row["db_path"],
                    "question": row["question"],
                    "gold_sql": row["gold_sql"],
                    # None when all k attempts were wrong. Written anyway, so a
                    # restart does not resample a question we already know the
                    # teacher cannot do, and so the failures stay countable.
                    "trace": trace,
                    "n_correct": n_correct,
                    "n_sampled": k,
                }) + "\n")

        seen = start + len(batch)
        print(f"  {seen}/{len(todo)} sampled, {kept} kept ({kept / seen:.1%})",
              flush=True)

    _report(correct_counts, kept, len(todo), k)


def _schema_text(row: dict) -> str:
    """The schema exactly as the student sees it, taken from the stored prompt.

    Read back out of the user turn rather than re-rendered, so the teacher is
    reasoning about the same text the student will be given. Re-rendering would
    be a second source of truth and an invitation to drift.
    """
    user = row["messages"][1]["content"]
    return user.split("Schema: ", 1)[1].rsplit("\nQuestion:", 1)[0]


def _report(counts: Counter[int], kept: int, total: int, k: int) -> None:
    print("\n=== coverage ===")
    print(f"  questions sampled : {total}")
    print(f"  usable traces     : {kept} ({kept / total:.1%})")
    print(f"  no correct attempt: {counts[0]} ({counts[0] / total:.1%})")
    print("\n=== how many of k attempts were correct ===")
    print("  (all-k-correct means the question was easy for the teacher;")
    print("   exactly-1 means it barely managed it -- the interesting ones)")
    for n in range(k + 1):
        if counts[n]:
            bar = "#" * round(40 * counts[n] / total)
            print(f"  {n}/{k}  {counts[n]:5d}  {bar}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", default=TEACHER)
    parser.add_argument("--source", type=Path, default=SFT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("-k", "--samples", type=int, default=SAMPLES_PER_QUESTION)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--chunk", type=int, default=CHUNK)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    args = parser.parse_args()

    sample(
        teacher=args.teacher, source=args.source, out=args.out, k=args.samples,
        limit=args.limit, chunk=args.chunk,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
