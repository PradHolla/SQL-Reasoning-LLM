"""Drive the running service and record what it said, for a demo.

Talks HTTP to ``sqlrl.serving.api`` rather than constructing ``SqlService``
in-process, because the thing worth showing is the service: a question goes to
an endpoint, SQL comes back, the SQL has already been run, and the answer
arrives with a calibrated confidence attached. Importing the class directly
would demo the library and quietly skip the part that is actually deployed.

Two outputs, from one pass:

* a terminal rendering, which is what a human watches;
* ``--transcript``, the same run as JSON, which ``analysis.demo_gif`` turns
  into frames. The GIF is therefore built from a real recorded session --
  real SQL, real rows, real timings -- rather than from a script of what the
  session would have said.

    uv run python -m analysis.demo --db-id concert_singer --samples 8 \
        --transcript results/analysis/demo.json
"""

from __future__ import annotations

import argparse
import json
import textwrap
import time
from pathlib import Path
from typing import Any

import httpx

__all__ = ["DEFAULT_QUESTIONS", "ask", "main", "render"]

#: Chosen to show range rather than to flatter: a trivial count, a join, an
#: aggregate with a filter, and one the model is likely to get wrong. A demo
#: that only shows the easy case is an advertisement, not a demo.
DEFAULT_QUESTIONS = [
    "How many singers do we have?",
    "Show the name and the release year of the song by the youngest singer.",
    "What are the names of the concerts that happened in 2014, and how many stadiums hosted each?",
]

RESET = "\033[0m"
DIM = "\033[2m"
BOLD = "\033[1m"
BLUE = "\033[38;5;33m"
GREEN = "\033[38;5;35m"
AMBER = "\033[38;5;178m"
RED = "\033[38;5;167m"

LEVEL_COLOUR = {"high": GREEN, "medium": AMBER, "low": RED, "none": RED, "unmeasured": DIM}
MAX_ROWS = 6
MAX_CELL = 28
#: Wrap generated SQL at this width. One unwrapped join can be 150 characters,
#: and since the GIF sizes its canvas to the longest line, a single long query
#: was producing a frame nearly four times wider than tall -- unreadable in a
#: feed. Wrapping is display-only; the transcript keeps the SQL intact.
SQL_WIDTH = 88


def ask(client: httpx.Client, question: str, db_id: str, samples: int,
        max_attempts: int) -> dict[str, Any]:
    response = client.post("/query", json={
        "question": question, "db_id": db_id,
        "samples": samples, "max_attempts": max_attempts,
    }, timeout=180.0)
    response.raise_for_status()
    return response.json()


def _table(rows: list[list[Any]]) -> list[str]:
    """Result rows as a box-drawn table, truncated. Returns display lines."""
    if not rows:
        return [f"{DIM}(no rows){RESET}"]
    shown = rows[:MAX_ROWS]
    cells = [[_cell(v) for v in row] for row in shown]
    widths = [max(len(row[i]) for row in cells) for i in range(len(cells[0]))]

    def rule(left: str, mid: str, right: str) -> str:
        return DIM + left + mid.join("─" * (w + 2) for w in widths) + right + RESET

    lines = [rule("┌", "┬", "┐")]
    for row in cells:
        body = f"{DIM}│{RESET} " + f" {DIM}│{RESET} ".join(
            value.ljust(width) for value, width in zip(row, widths)
        ) + f" {DIM}│{RESET}"
        lines.append(body)
    lines.append(rule("└", "┴", "┘"))
    if len(rows) > MAX_ROWS:
        lines.append(f"{DIM}… {len(rows) - MAX_ROWS} more rows{RESET}")
    return lines


def _cell(value: Any) -> str:
    text = "NULL" if value is None else str(value)
    return text if len(text) <= MAX_CELL else text[: MAX_CELL - 1] + "…"


def render(question: str, answer: dict[str, Any], db_id: str) -> list[str]:
    """The lines a human sees. Also exactly what the GIF renders."""
    confidence = answer["confidence"]
    level = confidence["level"]
    colour = LEVEL_COLOUR.get(level, DIM)
    timings = answer["timings_ms"]

    lines = [
        f"{BLUE}❯{RESET} {BOLD}{question}{RESET}",
        f"{DIM}  database: {db_id}{RESET}",
        "",
        f"{DIM}  SQL{RESET}",
    ]
    for raw_line in answer["sql"].splitlines():
        wrapped = textwrap.wrap(
            raw_line, width=SQL_WIDTH, break_long_words=False,
            break_on_hyphens=False, subsequent_indent="  ",
        ) or [""]
        lines += [f"    {BLUE}{piece}{RESET}" for piece in wrapped]
    lines.append("")

    if answer["status"] == "ok":
        rows = answer["rows"]
        lines.append(f"{DIM}  RESULT{RESET}  {len(rows)} row{'' if len(rows) == 1 else 's'}")
        lines += [f"    {line}" for line in _table(rows)]
    else:
        lines.append(f"{DIM}  RESULT{RESET}  {RED}{answer['status']}{RESET}")
        if answer.get("error"):
            lines.append(f"    {RED}{answer['error']}{RESET}")
    lines.append("")

    if confidence["expected_accuracy"] is None:
        badge = f"{colour}{level}{RESET}"
    else:
        badge = (f"{colour}{BOLD}{level}{RESET}  "
                 f"{confidence['agreement']}/{confidence['samples']} samples agree  "
                 f"{DIM}~{confidence['expected_accuracy']:.0%} accurate at this level{RESET}")
    lines.append(f"{DIM}  CONFIDENCE{RESET}  {badge}")
    lines.append(
        f"{DIM}  {timings['total'] / 1000:.1f}s total "
        f"({timings['generate'] / 1000:.1f}s generate, "
        f"{timings['execute'] / 1000:.1f}s execute){RESET}"
    )
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--db-id", default="concert_singer")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--max-attempts", type=int, default=1)
    parser.add_argument("--questions", nargs="*", default=None)
    parser.add_argument("--transcript", type=Path, default=None)
    args = parser.parse_args()

    questions = args.questions or DEFAULT_QUESTIONS
    entries: list[dict[str, Any]] = []

    with httpx.Client(base_url=args.url) as client:
        health = client.get("/health", timeout=30.0).json()
        print(f"{DIM}{health['model']} on {health['device']}, "
              f"{health['databases']} databases loaded{RESET}\n")

        for question in questions:
            started = time.perf_counter()
            answer = ask(client, question, args.db_id, args.samples, args.max_attempts)
            lines = render(question, answer, args.db_id)
            print("\n".join(lines))
            print()
            entries.append({
                "question": question,
                "db_id": args.db_id,
                "answer": answer,
                "wall_seconds": round(time.perf_counter() - started, 2),
            })

    if args.transcript:
        args.transcript.parent.mkdir(parents=True, exist_ok=True)
        args.transcript.write_text(json.dumps({
            "kind": "demo_transcript",
            "model": health["model"],
            "device": health["device"],
            "samples": args.samples,
            "max_attempts": args.max_attempts,
            "entries": entries,
        }, indent=1))
        print(f"{DIM}  wrote {args.transcript}{RESET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
