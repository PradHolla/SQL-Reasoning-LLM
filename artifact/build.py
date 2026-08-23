"""Assemble the artifact: source HTML plus figures, inlined.

A published artifact must be self-contained -- a strict CSP blocks every
external host -- so each figure has to travel inside the file as a ``data:``
URI. Base64 of a 150 KB PNG is 200 KB of unreadable text, and seven of those
pasted into the document would make the source impossible to edit or diff.

So the source keeps an empty figure with the file it wants:

    <figure class="fig" data-src="vote-curve.png">
      <figcaption>What the reader should take from it.</figcaption>
    </figure>

and this injects the <img> at build time. The source stays reviewable and
lives in git; the multi-megabyte build is derived and gitignored.

    uv run python artifact/build.py
"""

from __future__ import annotations

import base64
import mimetypes
import re
import sys
from pathlib import Path

SOURCE = Path("artifact/execution-signal.src.html")
OUT = Path("artifact/build/execution-signal.html")
CHARTS = Path("results/analysis/charts")

FIGURE = re.compile(r'<figure class="fig" data-src="([^"]+)">')


def inline(match: re.Match[str]) -> str:
    name = match.group(1)
    path = CHARTS / name
    if not path.is_file():
        raise SystemExit(
            f"figure {name!r} is referenced by the document but not built.\n"
            f"  expected: {path}\n"
            f"  run:      uv run python -m analysis.charts"
        )
    mime = mimetypes.guess_type(name)[0] or "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    # alt is deliberately empty: every figure here is followed by a figcaption
    # that says the same thing, and a screen reader announcing it twice is
    # worse than not announcing it at all.
    return (f'<figure class="fig" data-src="{name}">'
            f'<img alt="" src="data:{mime};base64,{encoded}">')


def main() -> int:
    if not SOURCE.is_file():
        raise SystemExit(f"no source at {SOURCE}")
    html = SOURCE.read_text()

    wanted = FIGURE.findall(html)
    built = FIGURE.sub(inline, html)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(built)

    size = OUT.stat().st_size
    print(f"  {len(wanted)} figures inlined: {', '.join(wanted)}")
    print(f"  wrote {OUT}  ({size / 1_000_000:.2f} MB of a 16 MB budget)")
    if size > 15_000_000:
        print("  WARNING: within 1 MB of the artifact size limit", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
