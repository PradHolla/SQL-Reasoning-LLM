"""Every chart, from the JSON the analysis scripts emit.

Rendering is separated from computing on purpose: the expensive half (executing
tens of thousands of queries against SQLite, re-running a retriever) happens
once and lands in ``results/analysis/*.json``; this reads those files and can be
re-run as many times as it takes to get a label to stop colliding, for free.

**Charts are rendered light, always**, even though the document that embeds them
follows the reader's theme. A figure is an artifact you export -- into a post, a
slide, a blog -- and it should look the same everywhere it lands rather than
inheriting whatever the page around it happens to be. The document puts them on
a light card so the two agree.

Colours are the validated default categorical palette (blue/orange, adjacent
pair CVD dE 24.7) and, for the percentile ramp, blue steps 250/450/650 as an
ordinal scale -- p50 < p95 < p99 is ordered magnitude, not four identities, so
one hue getting darker says what four hues would not.

Missing inputs are skipped, not fatal: this runs after each experiment lands,
not once at the end.

    uv run python -m analysis.charts --out-dir results/analysis/charts
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # no display on the box; must precede the pyplot import
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

__all__ = ["main", "render_all"]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e4e3df"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
RAMP = ("#86b6ef", "#2a78d6", "#104281")  # ordinal: p50 -> p95 -> p99
DPI = 200


def _setup() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": INK,
        "axes.labelcolor": INK_SOFT,
        "xtick.color": INK_SOFT,
        "ytick.color": INK_SOFT,
        "axes.edgecolor": GRID,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 14,
        "figure.autolayout": False,
    })


def _frame(ax, *, ygrid: bool = True, xgrid: bool = False) -> None:
    """Recessive axes: no box, grid behind the marks, ticks without spurs."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(length=0)
    if ygrid:
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    if xgrid:
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _pct(decimals: int = 0):
    return FuncFormatter(lambda v, _: f"{v:.{decimals}%}")


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"  wrote {path}")
    return path


def _load(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        print(f"  skip {path.name} (not built yet)")
        return None
    return json.loads(path.read_text())


# --------------------------------------------------------------------------
# charts
# --------------------------------------------------------------------------


def chart_vote_curve(data: dict, out: Path, reference: float | None,
                     reference_label: str) -> Path:
    """vote@k against its pass@k ceiling. Two series, so a legend is required."""
    points = data["points"]
    ks = [p["k"] for p in points]
    vote = [p["vote_ex"] for p in points]
    oracle = [p["oracle_ex"] for p in points]

    fig, ax = plt.subplots(figsize=(8, 5))
    _frame(ax)

    if reference is not None:
        ax.axhline(reference, color=INK_SOFT, linewidth=1.2, linestyle=(0, (4, 3)))
        # Sits on the surface, not on whatever crosses it: at low k the
        # pass@k line runs straight through this label's box.
        ax.annotate(f"{reference_label}  {reference:.1%}",
                    xy=(ks[0], reference), xytext=(0, 8), textcoords="offset points",
                    ha="left", fontsize=10, color=INK_SOFT, zorder=4,
                    bbox=dict(facecolor=SURFACE, edgecolor="none", pad=2.5))

    ax.plot(ks, oracle, color=ORANGE, linewidth=2, marker="o", markersize=8,
            markeredgecolor=SURFACE, markeredgewidth=2, label="pass@k (any sample correct)")
    ax.plot(ks, vote, color=BLUE, linewidth=2, marker="o", markersize=8,
            markeredgecolor=SURFACE, markeredgewidth=2, label="vote@k (what it answers)")

    # Direct-label the endpoints only -- a number on every marker is noise.
    for series, colour, offset in ((oracle, ORANGE, 10), (vote, BLUE, -18)):
        ax.annotate(f"{series[-1]:.1%}", xy=(ks[-1], series[-1]),
                    xytext=(8, offset), textcoords="offset points",
                    color=colour, fontsize=11, fontweight="bold")

    ax.set_xscale("log", base=2)
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.yaxis.set_major_formatter(_pct())
    ax.set_xlabel("samples generated per question (k)")
    ax.set_ylabel("execution accuracy")
    ax.set_title("Voting closes the gap to a model five times its size")
    ax.legend(frameon=False, loc="center right", fontsize=10)
    ax.margins(x=0.12)

    prov = data["provenance"]
    fig.text(0.0, -0.04,
             f"{prov['model']}  ·  Spider {prov['split']}, n={prov['n']}  ·  "
             f"temperature {prov['temperature']}  ·  {prov['git_commit']}",
             fontsize=9, color=INK_SOFT)
    return _save(fig, out)


def chart_calibration(data: dict, out: Path) -> Path:
    """Accuracy by self-agreement. One series -- the title names it, no legend."""
    buckets = [b for b in data.get("calibration", []) if b["n"] >= 1]
    if not buckets:
        return out
    buckets = sorted(buckets, key=lambda b: b["agreement"])
    labels = [f"{b['agreement']}/{b['of']}\nn={b['n']}" for b in buckets]
    acc = [b["accuracy"] for b in buckets]
    cover = [b["coverage"] for b in buckets]

    fig, ax = plt.subplots(figsize=(8, 5))
    _frame(ax)
    bars = ax.bar(labels, acc, color=BLUE, width=0.62)
    # 4px-equivalent rounded data-end, anchored to the baseline.
    for bar in bars:
        bar.set_joinstyle("round")

    for bar, a in zip(bars, acc):
        ax.annotate(f"{a:.1%}", xy=(bar.get_x() + bar.get_width() / 2, a),
                    xytext=(0, 5), textcoords="offset points",
                    ha="center", fontsize=10, fontweight="bold", color=INK)

    # Selective direct label: the bucket holding most of the traffic is the
    # only one whose coverage changes what you would do about it.
    widest = max(range(len(cover)), key=lambda i: cover[i])
    ax.annotate(f"{cover[widest]:.0%} of all questions land here",
                xy=(bars[widest].get_x() + bars[widest].get_width() / 2, acc[widest]),
                xytext=(0, 34), textcoords="offset points", ha="center",
                fontsize=10, color=INK_SOFT,
                arrowprops=dict(arrowstyle="-", color=GRID, linewidth=1.2))

    ax.yaxis.set_major_formatter(_pct())
    ax.set_ylim(0, max(acc) * 1.34)
    ax.set_xlabel("samples agreeing with the answer")
    ax.set_ylabel("execution accuracy")
    ax.set_title("Only near-unanimous votes carry a usable signal")

    prov = data["provenance"]
    fig.text(0.0, -0.08,
             f"{prov['model']}  ·  Spider {prov['split']}, n={prov['n']}  ·  "
             f"k={prov['samples']}  ·  {prov['git_commit']}",
             fontsize=9, color=INK_SOFT)
    return _save(fig, out)


def chart_retrieval(datasets: list[dict], out: Path) -> Path:
    """Four conditions, and for the retrieved ones the covered/uncovered split.

    Two series (all gold tables present vs one missing) so a legend is present;
    the control conditions have no split and show a single bar.
    """
    order = {"gold": 0, "oracle": 1, "bm25": 2, "dense": 3}
    datasets = sorted(datasets, key=lambda d: (order.get(d["mode"], 9), d["top_k"]))
    labels, overall, covered, uncovered = [], [], [], []
    for d in datasets:
        suffix = f"@{d['top_k']}" if d["mode"] in ("bm25", "dense") else ""
        labels.append(f"{d['mode']}{suffix}")
        overall.append(d["overall_ex"])
        covered.append(d["ex_covered"] if d["mode"] in ("bm25", "dense") else None)
        uncovered.append(d["ex_uncovered"] if d["mode"] in ("bm25", "dense") else None)

    fig, ax = plt.subplots(figsize=(9, 5))
    _frame(ax)
    x = range(len(labels))
    ax.bar([i - 0.17 for i in x], overall, width=0.3, color=BLUE, label="overall")
    ax.bar([i + 0.17 for i in x],
           [c if c is not None else 0 for c in covered],
           width=0.3, color=ORANGE, label="when every gold table was retrieved")

    for i, value in enumerate(overall):
        ax.annotate(f"{value:.1%}", xy=(i - 0.16, value), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=9.5,
                    fontweight="bold", color=INK)
    for i, value in enumerate(covered):
        if value is not None:
            ax.annotate(f"{value:.1%}", xy=(i + 0.16, value), xytext=(0, 4),
                        textcoords="offset points", ha="center", fontsize=9.5,
                        fontweight="bold", color=INK)

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.yaxis.set_major_formatter(_pct())
    ax.set_ylabel("execution accuracy")
    ax.set_title("Retrieval's cost is mostly not the tables it missed")
    ax.legend(frameon=False, fontsize=10, loc="upper right")
    ax.set_ylim(0, max(overall) * 1.25)

    prov = datasets[0]["provenance"]
    fig.text(0.0, -0.04,
             f"{prov['model']}  ·  {prov['pool_tables']}-table pool  ·  "
             f"n={prov['pool_questions']}  ·  {prov['git_commit']}",
             fontsize=9, color=INK_SOFT)
    return _save(fig, out)


def chart_retrieval_tradeoff(datasets: list[dict], out: Path) -> Path:
    """The two forces that overall accuracy hides.

    Raising k retrieves the gold tables for more questions (coverage rises) and
    makes the model worse on the questions it already had them for (accuracy on
    the covered set falls). Plotting only the aggregate shows a flat line for
    dense and a rising one for bm25, and neither says why. Small multiples --
    one panel per retriever, two series each -- because four lines on shared
    axes would need four categorical slots the palette will not give a
    line chart.
    """
    by_mode: dict[str, list[dict]] = {}
    for d in datasets:
        if d["mode"] in ("bm25", "dense"):
            by_mode.setdefault(d["mode"], []).append(d)
    for runs in by_mode.values():
        runs.sort(key=lambda d: d["top_k"])
    if not by_mode:
        return out

    order = [m for m in ("dense", "bm25") if m in by_mode]
    fig, axes = plt.subplots(1, len(order), figsize=(4.9 * len(order), 5),
                             sharey=True)
    axes = axes if len(order) > 1 else [axes]

    for ax, mode in zip(axes, order):
        _frame(ax)
        runs = by_mode[mode]
        ks = [d["top_k"] for d in runs]
        coverage = [d["coverage_at_k"] for d in runs]
        covered = [d["ex_covered"] for d in runs]

        ax.plot(ks, coverage, color=BLUE, linewidth=2, marker="o", markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=2,
                label="questions with every gold table retrieved")
        ax.plot(ks, covered, color=ORANGE, linewidth=2, marker="o", markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=2,
                label="accuracy on those questions")

        for series, colour in ((coverage, BLUE), (covered, ORANGE)):
            for k, value in zip(ks, series):
                ax.annotate(f"{value:.0%}", xy=(k, value), xytext=(0, 9),
                            textcoords="offset points", ha="center",
                            fontsize=9.5, color=colour, fontweight="bold")

        ax.set_xscale("log", base=2)
        ax.set_xticks(ks)
        ax.set_xticklabels([str(k) for k in ks])
        ax.set_xlabel(f"tables retrieved ({mode})")
        ax.margins(x=0.18, y=0.20)
        ax.yaxis.set_major_formatter(_pct())

    axes[0].set_ylabel("share of questions / accuracy")
    axes[0].legend(frameon=False, fontsize=9.5, loc="lower left")
    fig.suptitle("Retrieving more finds more tables and answers fewer questions",
                 x=0.0, ha="left", fontsize=13, fontweight="bold", y=1.02)

    prov = datasets[0]["provenance"]
    fig.text(0.0, -0.04,
             f"{prov['model']}  ·  {prov['pool_tables']}-table pool  ·  "
             f"n={prov['pool_questions']}  ·  {prov['git_commit']}",
             fontsize=9, color=INK_SOFT)
    return _save(fig, out)


def chart_latency_percentiles(data: dict, out: Path) -> Path:
    """p50/p95/p99 per mode. Ordered magnitude, so one hue darkening."""
    modes = list(data["modes"].items())
    labels = [label for label, _ in modes]
    series = [
        ("p50", [s["p50_ms"] for _, s in modes], RAMP[0]),
        ("p95", [s["p95_ms"] for _, s in modes], RAMP[1]),
        ("p99", [s["p99_ms"] for _, s in modes], RAMP[2]),
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    _frame(ax)
    width = 0.26
    x = range(len(labels))
    for offset, (name, values, colour) in zip((-width, 0, width), series):
        ax.bar([i + offset for i in x], values, width=width - 0.04,
               color=colour, label=name)

    # Label only the p99s: the tail is the point of this chart.
    for i, value in enumerate(series[2][1]):
        ax.annotate(f"{value / 1000:.1f}s", xy=(i + width, value), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=10,
                    fontweight="bold", color=INK)

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:g}s"))
    ax.set_ylabel("latency")
    ax.set_ylim(0, max(series[2][1]) * 1.18)
    ax.set_title("Retry looks free at the median and is not")
    ax.legend(frameon=False, fontsize=10, title="", loc="upper left")

    fig.text(0.0, -0.04,
             f"{data['model']}  ·  Spider {data['split']}, n={data['n']}  ·  "
             f"{data['device']}  ·  one request at a time",
             fontsize=9, color=INK_SOFT)
    return _save(fig, out)


def chart_latency_accuracy(data: dict, out: Path) -> Path:
    """What each accuracy point costs. One colour, every point direct-labelled --
    colouring six modes would need six categorical slots in a scatter, which the
    palette's all-pairs cap does not allow, and identity is carried by the label
    anyway.
    """
    modes = list(data["modes"].items())
    xs = [s["p50_ms"] / 1000 for _, s in modes]
    ys = [s["accuracy"] for _, s in modes]

    fig, ax = plt.subplots(figsize=(8, 5))
    _frame(ax, xgrid=True)
    ax.scatter(xs, ys, s=110, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=3)
    for (label, _), x, y in zip(modes, xs, ys):
        ax.annotate(f"{label}\n{y:.1%} · {x:.1f}s", xy=(x, y), xytext=(10, -4),
                    textcoords="offset points", fontsize=10, color=INK)

    ax.yaxis.set_major_formatter(_pct(1))
    ax.set_xlabel("median latency per question (p50)")
    ax.set_ylabel("execution accuracy")
    ax.set_title("What each point of accuracy costs in wall clock")
    ax.margins(x=0.22, y=0.18)

    fig.text(0.0, -0.04,
             f"{data['model']}  ·  Spider {data['split']}, n={data['n']}  ·  "
             f"{data['device']}",
             fontsize=9, color=INK_SOFT)
    return _save(fig, out)


def chart_taxonomy(data: dict, out: Path) -> Path:
    """The failures, largest first. One series, horizontal for long labels."""
    buckets = {**data["executed_wrong"], **data["did_not_run"]}
    items = sorted(buckets.items(), key=lambda kv: kv[1])
    if not items:
        return out
    labels = [k.replace("_", " ") for k, _ in items]
    values = [v for _, v in items]
    scored = data["scored"]
    ran = set(data["executed_wrong"])

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    _frame(ax, ygrid=False, xgrid=True)
    colours = [BLUE if k in ran else ORANGE for k, _ in items]
    ax.barh(labels, values, color=colours, height=0.66)
    for value, label in zip(values, labels):
        ax.annotate(f"{value}  ({value / scored:.1%})", xy=(value, label),
                    xytext=(6, 0), textcoords="offset points", va="center",
                    fontsize=9.5, color=INK_SOFT)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=BLUE),
        plt.Rectangle((0, 0), 1, 1, color=ORANGE),
    ]
    ax.legend(handles, ["ran, wrong answer", "did not run"],
              frameon=False, fontsize=10, loc="lower right")
    ax.set_xlabel("questions")
    ax.set_title(f"What the {1 - data['accuracy']:.0%} it gets wrong looks like")
    ax.margins(x=0.18)

    prov = data["provenance"]
    fig.text(0.0, -0.04,
             f"{prov['model']}  ·  Spider {prov['split']}, n={prov['n']}  ·  "
             f"{prov['git_commit']}",
             fontsize=9, color=INK_SOFT)
    return _save(fig, out)


# --------------------------------------------------------------------------


def render_all(source: Path, out_dir: Path, reference: float | None,
               reference_label: str) -> list[Path]:
    _setup()
    written: list[Path] = []

    votes = _load(source / "vote_curve.json")
    if votes:
        written.append(chart_vote_curve(votes, out_dir / "vote-curve.png",
                                        reference, reference_label))
        if votes.get("calibration"):
            written.append(chart_calibration(votes, out_dir / "calibration.png"))

    retrieval = [d for d in (_load(p) for p in sorted(source.glob("retrieval-*.json"))) if d]
    if retrieval:
        written.append(chart_retrieval(retrieval, out_dir / "retrieval.png"))
        written.append(chart_retrieval_tradeoff(retrieval, out_dir / "retrieval-tradeoff.png"))

    latency = _load(source / "latency.json")
    if latency:
        written.append(chart_latency_percentiles(latency, out_dir / "latency-percentiles.png"))
        written.append(chart_latency_accuracy(latency, out_dir / "latency-accuracy.png"))

    for path in sorted(source.glob("taxonomy-*.json")):
        data = _load(path)
        if data:
            written.append(chart_taxonomy(data, out_dir / f"{path.stem}.png"))

    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, default=Path("results/analysis"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/analysis/charts"))
    parser.add_argument("--reference", type=float, default=0.712,
                        help="horizontal reference line on the k-curve (coder-7b)")
    parser.add_argument("--reference-label", default="Qwen2.5-Coder-7B-Instruct")
    args = parser.parse_args()

    written = render_all(args.source, args.out_dir, args.reference, args.reference_label)
    print(f"  {len(written)} charts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
