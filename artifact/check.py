"""Structural checks on the artifact source, before it costs a publish.

The document is edited by string substitution, which is fast and has exactly
one failure mode: a replacement that lands in the wrong place, or half-lands,
produces HTML that still renders -- browsers close tags for you -- while the
section numbering silently skips a number or an anchor points at nothing.
None of that raises an error anywhere. So it gets checked here.

    uv run python artifact/check.py
"""

from __future__ import annotations

import html
import re
import sys
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path

SOURCE = Path("artifact/execution-signal.src.html")
CHARTS = Path("results/analysis/charts")
VOID = {"img", "br", "hr", "meta", "link", "input", "source", "col", "area", "base", "wbr"}

SECTION = re.compile(r'<h2 class="sec" id="([^"]+)"><span class="k">(\d+)</span>(.*?)</h2>', re.S)
TOC = re.compile(r'<li><span class="k">(\d+)</span><a href="#([^"]+)">([^<]+)</a>')


class _Balance(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[tuple[str, tuple[int, int]]] = []
        self.errors: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag not in VOID:
            self.stack.append((tag, self.getpos()))

    def handle_endtag(self, tag):
        if tag in VOID:
            return
        if not self.stack:
            self.errors.append(f"stray </{tag}> at line {self.getpos()[0]}")
            return
        opened, pos = self.stack.pop()
        if opened != tag:
            self.errors.append(
                f"</{tag}> at line {self.getpos()[0]} closes <{opened}> from line {pos[0]}"
            )


def main() -> int:
    src = SOURCE.read_text()
    problems: list[str] = []

    balance = _Balance()
    balance.feed(src)
    problems += balance.errors
    problems += [f"unclosed <{t}> at line {p[0]}" for t, p in balance.stack]

    ids = re.findall(r'\sid="([^"]+)"', src)
    problems += [f"duplicate id {i!r}" for i, n in Counter(ids).items() if n > 1]

    anchors = {h[1:] for h in re.findall(r'href="(#[^"]+)"', src)}
    problems += [f"anchor #{a} points at nothing" for a in sorted(anchors - set(ids))]

    sections = [(n, i, " ".join(html.unescape(re.sub(r"<[^>]+>", " ", t)).split()))
                for i, n, t in SECTION.findall(src)]
    numbers = [int(n) for n, _, _ in sections]
    if numbers != list(range(1, len(numbers) + 1)):
        problems.append(f"section numbering is not sequential: {numbers}")

    toc = TOC.findall(src)
    if [int(n) for n, _, _ in toc] != numbers:
        problems.append("table of contents does not match the section numbering")
    for (num, anchor, title), (tnum, thref, ttitle) in zip(sections, toc):
        if anchor != thref:
            problems.append(f"section {num} has id {anchor!r} but the contents link to {thref!r}")
        if title.lower() not in ttitle.lower() and ttitle.lower() not in title.lower():
            problems.append(f"section {num} titled {title!r} but listed as {ttitle!r}")

    figures = re.findall(r'data-src="([^"]+)"', src)
    problems += [f"figure {f} referenced but not built (run analysis.charts)"
                 for f in figures if not (CHARTS / f).is_file()]

    if problems:
        print(f"{len(problems)} problem(s):")
        for p in problems:
            print(f"  {p}")
        return 1

    print(f"  OK  {len(sections)} sections, {len(figures)} figures, "
          f"{len(ids)} anchors, tags balanced")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
