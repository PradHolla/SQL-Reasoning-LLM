"""The LinkedIn carousel, as a multi-page PDF.

LinkedIn renders a document post as static pages, so this is a PDF and not a
deck of images -- text stays vector, which is what keeps 11pt captions legible
after the platform's own downscaling.

Typography and palette are the artifact's, deliberately. The carousel, the
technical write-up and the blog that follows should read as one body of work
rather than three unrelated things, and the artifact's ground (#F1F4F0) is
already not white, so it separates from the feed without a dark deck that
would fight the light-grounded charts.

Portrait 4:5 (1080x1350 at 150 dpi) because that is the aspect that occupies
the most vertical space in a phone feed.

    uv run python -m analysis.carousel --out results/analysis/carousel.pdf
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.image as mpimg  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402

__all__ = ["SLIDES", "build", "main"]

# --- identity -----------------------------------------------------------
GROUND = "#F1F4F0"
PANEL = "#FAFBF9"
INK = "#17201B"
INK_SOFT = "#4E5C54"
INK_FAINT = "#7A8880"
RULE = "#CFD8D0"
ACCENT = "#1C5A4E"
DEFECT = "#97382C"

_F = "/System/Library/Fonts/Supplemental/"
SERIF = FontProperties(fname=f"{_F}Georgia.ttf")
SERIF_B = FontProperties(fname=f"{_F}Georgia Bold.ttf")
MONO = FontProperties(fname="/System/Library/Fonts/Menlo.ttc")

W, H = 7.2, 9.0      # inches; 1080x1350 at 150 dpi
DPI = 150
M = 0.62             # margin, inches -> figure fraction below
CHARTS = Path("results/analysis/charts")


def _x(inches: float) -> float:
    return inches / W


def _y(inches: float) -> float:
    return 1 - inches / H


def _text(fig, x, y, s, *, size, font=SERIF, color=INK, wrap=None,
          leading=1.45, va="top", ha="left") -> float:
    """Draw (optionally wrapped) text; return the y in inches after it."""
    lines = textwrap.wrap(s, wrap) if wrap else s.split("\n")
    step = size * leading / 72
    for i, line in enumerate(lines):
        fig.text(_x(x), _y(y + i * step), line, fontproperties=font,
                 fontsize=size, color=color, va=va, ha=ha)
    return y + len(lines) * step


def _eyebrow(fig, y, s) -> float:
    fig.text(_x(M), _y(y), s.upper(), fontproperties=MONO, fontsize=8.5,
             color=INK_FAINT, va="top")
    return y + 0.30


def _rule(fig, y, width=W - 2 * M) -> float:
    fig.add_artist(plt.Line2D([_x(M), _x(M + width)], [_y(y), _y(y)],
                              color=RULE, linewidth=0.8,
                              transform=fig.transFigure))
    return y + 0.24


#: Content must end above this, or it collides with the page number and then
#: runs off the trim. matplotlib will happily draw text at a negative y and
#: report no error, so this is checked rather than eyeballed.
FLOOR = H - M - 0.28


def _guard(y: float, slide: str) -> float:
    if y > FLOOR:
        raise SystemExit(
            f"slide {slide!r} overflows: content ends at {y:.2f}in, "
            f"floor is {FLOOR:.2f}in. Cut {(y - FLOOR):.2f}in of copy or "
            f"shrink the chart."
        )
    return y


def _page(fig, n, total) -> None:
    fig.text(_x(W - M), _y(H - M + 0.10), f"{n} / {total}",
             fontproperties=MONO, fontsize=8.5, color=INK_FAINT, ha="right")


def _blank():
    fig = plt.figure(figsize=(W, H), dpi=DPI)
    fig.patch.set_facecolor(GROUND)
    return fig


def _chart(fig, name, top, height):
    """Place a chart PNG on a light card, preserving its aspect ratio."""
    path = CHARTS / name
    if not path.is_file():
        raise SystemExit(f"missing chart {path}; run analysis.charts first")
    img = mpimg.imread(path)
    ih, iw = img.shape[0], img.shape[1]
    avail_w = W - 2 * M
    draw_w = avail_w
    draw_h = draw_w * ih / iw
    if draw_h > height:
        draw_h = height
        draw_w = draw_h * iw / ih
    pad = 0.10
    card = fig.add_axes([_x(M), _y(top + draw_h + 2 * pad),
                         (draw_w + 2 * pad) / W, (draw_h + 2 * pad) / H])
    card.set_facecolor(PANEL)
    for sp in card.spines.values():
        sp.set_color(RULE); sp.set_linewidth(0.8)
    card.set_xticks([]); card.set_yticks([])
    ax = fig.add_axes([_x(M + pad), _y(top + draw_h + pad), draw_w / W, draw_h / H])
    ax.imshow(img); ax.axis("off")
    return top + draw_h + 2 * pad + 0.30


# --- slides -------------------------------------------------------------
# Ordered so the first four assume nothing and the rest escalate. Slide 10
# is text-only on purpose: it is the one that separates this from every
# "I fine-tuned a model" post, and a chart would dilute it.

def s01_cover(fig):
    fig.text(_x(M), _y(1.55), "TEXT-TO-SQL  ·  SMALL LANGUAGE MODELS",
             fontproperties=MONO, fontsize=9, color=INK_FAINT, va="top")
    y = _text(fig, M, 2.05, "A 1.5B model that writes SQL as well as a 7B",
              size=40, font=SERIF, wrap=22, leading=1.16)
    y = _rule(fig, y + 0.30)
    y = _text(fig, M, y,
              "What I learned building, measuring and serving it. "
              "And three numbers I got wrong on the way.",
              size=15.5, font=SERIF, color=INK_SOFT, wrap=44)
    y = 6.30
    _rule(fig, y - 0.34)
    for i, (big, small) in enumerate([("71.5%", "execution accuracy"),
                                      ("2,147", "held-out questions"),
                                      ("1.5B", "parameters")]):
        col = M + i * (W - 2 * M) / 3
        fig.text(_x(col), _y(y), big, fontproperties=MONO, fontsize=23,
                 color=INK, va="top")
        fig.text(_x(col), _y(y + 0.42), small.upper(), fontproperties=MONO,
                 fontsize=8, color=INK_FAINT, va="top")
    fig.text(_x(M), _y(H - M - 0.30), "Full technical write-up in progress",
             fontproperties=MONO, fontsize=9.5, color=ACCENT, va="top")


def s02_problem(fig):
    y = _eyebrow(fig, M, "the problem")
    y = _text(fig, M, y, "Ask a database a question in English",
              size=27, font=SERIF, wrap=26, leading=1.2)
    y = _rule(fig, y + 0.24)
    y = _text(fig, M, y,
              "You have a database. Someone asks a question in plain English. "
              "Something has to turn that into SQL that actually runs and "
              "returns the right rows.",
              size=14.5, font=SERIF, color=INK_SOFT, wrap=52) + 0.22
    y = _text(fig, M, y,
              "That is easy on a database you were trained on. This project "
              "does it on databases the model has never seen, at 1.5 billion "
              "parameters, which is small enough to serve without a GPU "
              "budget behind it.",
              size=14.5, font=SERIF, color=INK_SOFT, wrap=52) + 0.55
    for i, (big, small) in enumerate([("2,147", "held-out questions"),
                                      ("206", "unseen databases"),
                                      ("1.5B", "parameters")]):
        col = M + i * (W - 2 * M) / 3
        fig.text(_x(col), _y(y), big, fontproperties=MONO, fontsize=25,
                 color=INK, va="top")
        fig.text(_x(col), _y(y + 0.44), small.upper(), fontproperties=MONO,
                 fontsize=8, color=INK_FAINT, va="top")


def s03_decision(fig):
    y = _eyebrow(fig, M, "the one decision everything rests on")
    y = _text(fig, M, y, "Measure by running the SQL, not by reading it",
              size=27, font=SERIF, wrap=26, leading=1.2)
    y = _rule(fig, y + 0.24)
    y = _text(fig, M, y,
              "The obvious way to grade generated SQL is to compare its text "
              "against a reference query. That marks two identical answers "
              "wrong for choosing a different join order.",
              size=14.5, font=SERIF, color=INK_SOFT, wrap=52) + 0.22
    y = _text(fig, M, y,
              "Instead: run both queries against the real database, and "
              "compare the rows that come back. Two queries that look nothing "
              "alike and return the same answer are both correct.",
              size=14.5, font=SERIF, color=INK, wrap=52) + 0.34
    y = _text(fig, M, y, "That single component then does four jobs:",
              size=14.5, font=SERIF, color=INK_SOFT, wrap=52) + 0.20
    for line in ["the benchmark metric",
                 "the reward signal for reinforcement learning",
                 "the filter that verifies synthetic training data",
                 "the vote at inference time"]:
        fig.text(_x(M + 0.16), _y(y), "·", fontproperties=MONO, fontsize=13,
                 color=ACCENT, va="top")
        y = _text(fig, M + 0.42, y, line, size=14, font=SERIF, color=INK,
                  wrap=48) + 0.06


def s04_system(fig):
    y = _eyebrow(fig, M, "what i built")
    y = _text(fig, M, y, "End to end, not a notebook",
              size=27, font=SERIF, wrap=26, leading=1.2)
    y = _rule(fig, y + 0.24)
    stages = [
        ("Qwen2.5-Coder-1.5B", "the base model"),
        ("Supervised fine-tuning", "on Spider train, full schemas"),
        ("GRPO", "reinforcement learning, reward is whether the query returns the right rows"),
        ("Execution voting", "sample k answers, run them all, take the majority"),
        ("FastAPI service", "returns SQL, rows, a calibrated confidence and a timing breakdown"),
    ]
    for name, detail in stages:
        fig.text(_x(M), _y(y), "▸", fontproperties=MONO, fontsize=11,
                 color=ACCENT, va="top")
        y = _text(fig, M + 0.30, y - 0.02, name, size=15, font=SERIF_B) + 0.02
        y = _text(fig, M + 0.30, y, detail, size=12.5, font=SERIF,
                  color=INK_SOFT, wrap=54) + 0.20
    y += 0.20
    for i, (big, small) in enumerate([("338", "tests"),
                                      ("11", "gpu-hours measuring"),
                                      ("~$14", "total compute")]):
        col = M + i * (W - 2 * M) / 3
        fig.text(_x(col), _y(y), big, fontproperties=MONO, fontsize=22,
                 color=INK, va="top")
        fig.text(_x(col), _y(y + 0.40), small.upper(), fontproperties=MONO,
                 fontsize=8, color=INK_FAINT, va="top")


def _chart_slide(eyebrow, title, chart, body, height=3.05):
    def render(fig):
        y = _eyebrow(fig, M, eyebrow)
        y = _text(fig, M, y, title, size=25, font=SERIF, wrap=30, leading=1.2)
        y = _chart(fig, chart, y + 0.16, height)
        for para in body:
            y = _text(fig, M, y, para, size=12.5, font=SERIF,
                      color=INK_SOFT, wrap=62) + 0.20
        _guard(y, title)
    return render


def s10_wrong(fig):
    y = _eyebrow(fig, M, "the part that matters")
    y = _text(fig, M, y, "Three numbers I got wrong",
              size=28, font=SERIF, wrap=26, leading=1.2)
    y = _rule(fig, y + 0.24)
    items = [
        ("I reported a p99 latency of 28 seconds.",
         "It was measured over 100 requests, and a p99 over 100 samples is "
         "the single slowest request in the sample. Over the full 2,147 it "
         "is 16 seconds, and the ranking I had built on it does not hold."),
        ("I wrote that retrieving fewer tables would help.",
         "It was a guess. I ran it at 5, 10 and 20 tables across two "
         "retrievers. It does not help. The claim is withdrawn."),
        ("I carried a 9-point cost for missing tables.",
         "Measured properly it is 6.7. The other half of that decomposition "
         "reproduced exactly, which is how I knew which number had drifted."),
    ]
    for i, (claim, fix) in enumerate(items, start=1):
        fig.text(_x(M), _y(y), f"0{i}", fontproperties=MONO, fontsize=13,
                 color=DEFECT, va="top")
        y = _text(fig, M + 0.46, y - 0.02, claim, size=14.5, font=SERIF_B,
                  wrap=44) + 0.04
        y = _text(fig, M + 0.46, y, fix, size=12.5, font=SERIF,
                  color=INK_SOFT, wrap=52) + 0.30
    y = _rule(fig, y + 0.06)
    _guard(_text(fig, M, y,
                 "All three sit in the write-up beside the corrected "
                 "versions. The wrong ones are the more useful half.",
                 size=13.5, font=SERIF, color=INK, wrap=56), "three wrong")


def s11_next(fig):
    y = _eyebrow(fig, M, "next")
    y = _text(fig, M, y, "What I would build next",
              size=28, font=SERIF, wrap=26, leading=1.2)
    y = _rule(fig, y + 0.24)
    items = [
        ("A learned verifier.",
         "Majority voting leaves 4.8 points on the table at k=16, and two "
         "hand-built tiebreak rules recovered +0.1 and −0.1. This needs a "
         "model, not a heuristic."),
        ("Training on cluttered schemas.",
         "Distraction costs 2.5x what missing tables do, so the fix belongs "
         "in training rather than in the retriever."),
        ("vLLM in the serving path.",
         "A 2.7 second median is not fast. Nothing about the current stack "
         "is optimised for inference."),
    ]
    for name, detail in items:
        fig.text(_x(M), _y(y), "▸", fontproperties=MONO, fontsize=11,
                 color=ACCENT, va="top")
        y = _text(fig, M + 0.30, y - 0.02, name, size=15, font=SERIF_B) + 0.02
        y = _text(fig, M + 0.30, y, detail, size=12.5, font=SERIF,
                  color=INK_SOFT, wrap=54) + 0.24
    y = _rule(fig, y + 0.10)
    y = _text(fig, M, y,
              "I am writing this up properly, with the methods and the "
              "numbers that did not make it here.",
              size=14, font=SERIF, color=INK, wrap=52) + 0.10
    _guard(_text(fig, M, y, "Follow if you want it when it lands.",
                 size=14, font=SERIF_B, color=ACCENT, wrap=52), "next")


SLIDES = [
    s01_cover,
    s02_problem,
    s03_decision,
    s04_system,
    _chart_slide(
        "the result", "71.5%, level with a model five times larger",
        "vote-curve.png",
        ["Sample eight answers, run all eight, reply with whatever the "
         "majority returned. 68.1% to 71.5%. A 7B scores 71.2%. Measured "
         "cost: 2.1x median latency, not 8x.",
         "But it saturates: +1.2, +1.4, +1.1, then +0.4 per doubling. The "
         "orange ceiling keeps climbing, so the gap between what the model "
         "produces and what it picks widens to 4.8 points. The bottleneck "
         "is no longer generation. It is selection."]),
    _chart_slide(
        "a signal you get for free", "The model already knows when it is wrong",
        "calibration.png",
        ["When all sixteen samples agree it is right 86% of the time, over "
         "65% of questions. When they disagree it is close to a coin flip.",
         "No extra training, no extra infrastructure. You are already "
         "generating the samples in order to vote. The ragged middle is "
         "honest: those buckets hold 23 to 84 questions each, so they are "
         "noise, not signal."]),
    _chart_slide(
        "the finding i almost missed", "A flat number hiding two opposing forces",
        "retrieval-tradeoff.png",
        ["Hand the model a 300-table pool instead of the right database and "
         "it must retrieve first. Accuracy read 44.3%, 45.2%, 44.8% across a "
         "fourfold change in k. Flat enough to conclude k does not matter.",
         "Split by whether retrieval found every needed table: coverage "
         "climbs while accuracy on those same questions falls, in both "
         "retrievers. They cancel. Irrelevant context costs 2.5x what "
         "missing context does."]),
    _chart_slide(
        "what it costs", "Accuracy reported next to its price",
        "latency-percentiles.png",
        ["Almost nobody publishes latency beside accuracy, which is how a "
         "technique that looks free in a benchmark becomes expensive in "
         "production.",
         "Retry's median is indistinguishable from greedy: most requests "
         "never enter the loop. All of its cost sits in the tail, and its "
         "mean hides that from both ends at once."]),
    _chart_slide(
        "what is still broken", "The largest failure is not reasoning",
        "taxonomy-grpo-coder15.png",
        ["Every failure on the held-out split, by kind. Orange never ran. "
         "Blue ran cleanly and returned the wrong rows.",
         "The biggest single one is the model inventing a column name that "
         "does not exist: 8.1%, larger than any wrong-answer category, after "
         "RL already cut it by 29%. The next fix belongs in how the schema is "
         "shown, not in how the model reasons."]),
    s10_wrong,
    s11_next,
]


def build(out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    total = len(SLIDES)
    with PdfPages(out) as pdf:
        for i, render in enumerate(SLIDES, start=1):
            fig = _blank()
            render(fig)
            if i > 1:
                _page(fig, i, total)
            pdf.savefig(fig, facecolor=GROUND)
            plt.close(fig)
    print(f"  wrote {out}  ({total} slides, {out.stat().st_size / 1_000_000:.2f} MB)")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("results/analysis/carousel.pdf"))
    parser.add_argument("--png-dir", type=Path, default=None,
                        help="also write one PNG per slide, for previewing")
    args = parser.parse_args()

    build(args.out)
    if args.png_dir:
        args.png_dir.mkdir(parents=True, exist_ok=True)
        for i, render in enumerate(SLIDES, start=1):
            fig = _blank(); render(fig)
            if i > 1:
                _page(fig, i, len(SLIDES))
            fig.savefig(args.png_dir / f"slide-{i:02d}.png", dpi=DPI,
                        facecolor=GROUND)
            plt.close(fig)
        print(f"  wrote {len(SLIDES)} slide PNGs to {args.png_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
