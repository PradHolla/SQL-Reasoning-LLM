"""An animated GIF of a real recorded session.

Reads the transcript ``analysis.demo`` wrote and replays it frame by frame:
the question types in, the service thinks for as long as it actually thought,
then the SQL, the rows and the confidence badge appear in the order a person
watching the terminal would have seen them.

**Nothing here is scripted.** The lines rendered are the ones ``demo.render``
produced from the service's real responses -- this module parses that output's
ANSI codes rather than re-deriving what it should have said, so the GIF and
the terminal cannot disagree. The only inventions are the frame timings, and
the "thinking" pause is scaled from the measured latency rather than chosen.

Rendered on a dark terminal ground, unlike the light figures from
``analysis.charts``: this is a recording of a terminal, and a terminal that is
white in a post reads as a screenshot of the wrong thing.

    uv run python -m analysis.demo_gif --transcript results/analysis/demo.json \
        --out results/analysis/charts/demo.gif
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
from PIL import Image, ImageDraw, ImageFont

from analysis.demo import BLUE, BOLD, DIM, RESET, render

__all__ = ["THEME", "build_frames", "main", "parse_ansi"]

ANSI = re.compile(r"\033\[([0-9;]*)m")

#: Dark terminal ground. Keys are the ANSI codes ``demo.render`` emits.
THEME = {
    "bg": "#16161a",
    "fg": "#e6e6e6",
    "dim": "#8a8a8a",
    "38;5;33": "#6aa9f0",   # blue -- prompt marker and SQL
    "38;5;35": "#4ec9a0",   # green -- high confidence
    "38;5;178": "#e0b24c",  # amber -- medium
    "38;5;167": "#e5787a",  # red -- low, and errors
}

FONT_SIZE = 17
LINE_HEIGHT = 25
PAD = 26
TYPE_CHUNK = 3
MS_TYPE = 45
MS_THINK = 130
MS_LINE = 85
MS_HOLD = 1500
MAX_THINK_FRAMES = 14
SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


def parse_ansi(line: str) -> list[tuple[str, str, bool]]:
    """``"\\033[1mhi\\033[0m"`` -> ``[("hi", <fg hex>, <bold>)]``."""
    spans: list[tuple[str, str, bool]] = []
    colour, bold, position = THEME["fg"], False, 0
    for match in ANSI.finditer(line):
        text = line[position:match.start()]
        if text:
            spans.append((text, colour, bold))
        raw = match.group(1) or "0"
        if raw in ("0", ""):
            colour, bold = THEME["fg"], False
        elif raw == "1":
            bold = True
        elif raw == "2":
            colour = THEME["dim"]
        elif raw in THEME:
            colour = THEME[raw]
        position = match.end()
    tail = line[position:]
    if tail:
        spans.append((tail, colour, bold))
    return spans


def _fonts() -> tuple[ImageFont.FreeTypeFont, ImageFont.FreeTypeFont]:
    """DejaVu Sans Mono, from matplotlib's bundled copy -- present wherever
    this package is installed, so the GIF looks the same on the box and on a
    laptop without depending on a system font being there.
    """
    root = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    return (
        ImageFont.truetype(str(root / "DejaVuSansMono.ttf"), FONT_SIZE),
        ImageFont.truetype(str(root / "DejaVuSansMono-Bold.ttf"), FONT_SIZE),
    )


def _draw(lines: list[str], size: tuple[int, int], fonts) -> Image.Image:
    regular, bold = fonts
    image = Image.new("RGB", size, THEME["bg"])
    canvas = ImageDraw.Draw(image)
    advance = regular.getlength("M")
    for row, line in enumerate(lines):
        x = PAD
        for text, colour, is_bold in parse_ansi(line):
            canvas.text((x, PAD + row * LINE_HEIGHT), text,
                        font=bold if is_bold else regular, fill=colour)
            x += advance * len(text)
    return image


def build_frames(transcript: dict, fonts) -> tuple[list[Image.Image], list[int]]:
    """One screen per question: type, think, reveal, hold."""
    screens = [
        (entry, render(entry["question"], entry["answer"], entry["db_id"]))
        for entry in transcript["entries"]
    ]
    plain = [ANSI.sub("", line) for _, lines in screens for line in lines]
    width = int(max(len(line) for line in plain) * fonts[0].getlength("M")) + PAD * 2
    height = max(len(lines) for _, lines in screens) * LINE_HEIGHT + PAD * 2
    size = (width, height)

    frames: list[Image.Image] = []
    durations: list[int] = []

    def push(lines: list[str], ms: int) -> None:
        frames.append(_draw(lines, size, fonts))
        durations.append(ms)

    for entry, lines in screens:
        question_line = lines[0]
        question = entry["question"]

        # 1. the question types in. Composed the same way ``demo.render``
        # composes its first line, rather than sliced back out of it -- the
        # rendered line carries ANSI codes that do not survive being cut at an
        # arbitrary character.
        for cut in range(0, len(question) + TYPE_CHUNK, TYPE_CHUNK):
            push([f"{BLUE}❯{RESET} {BOLD}{question[:cut]}{RESET}"], MS_TYPE)

        # 2. thinking, for as long as it really took
        think_frames = min(MAX_THINK_FRAMES,
                           max(2, int(entry["answer"]["timings_ms"]["total"] / 400)))
        for index in range(think_frames):
            spin = SPINNER[index % len(SPINNER)]
            push([question_line,
                  f"{DIM}  {spin} running "
                  f"{entry['answer']['confidence']['samples']} samples…{RESET}"],
                 MS_THINK)

        # 3. the answer, a line at a time
        for cut in range(1, len(lines) + 1):
            push(lines[:cut], MS_LINE)
        push(lines, MS_HOLD)

    return frames, durations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--transcript", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results/analysis/charts/demo.gif"))
    parser.add_argument("--colors", type=int, default=64,
                        help="palette size; the whole frame is flat colour on a "
                             "flat ground, so a small palette costs nothing visible")
    args = parser.parse_args()

    transcript = json.loads(args.transcript.read_text())
    frames, durations = build_frames(transcript, _fonts())
    quantized = [f.quantize(colors=args.colors, method=Image.MEDIANCUT) for f in frames]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    quantized[0].save(args.out, save_all=True, append_images=quantized[1:],
                      duration=durations, loop=0, optimize=True, disposal=2)
    size_mb = args.out.stat().st_size / 1_000_000
    print(f"  wrote {args.out}  ({len(frames)} frames, "
          f"{frames[0].width}x{frames[0].height}, {size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
