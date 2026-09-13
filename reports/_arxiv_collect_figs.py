"""Copy into the arXiv package only the figures the .tex actually includes.

The first cut of this used `grep '\includegraphics'`, which also matched a
commented-out figure (fig_swm_loop.pdf, acl_latex.tex L274-277) and shipped a
file arXiv then flagged as "Not used". Comments are stripped here before the
scan, so a figure pulled from the text stops being a dependency.

Run from docs/paper/:  /usr/bin/python3 ../../reports/_arxiv_collect_figs.py
"""
import re
import shutil
from pathlib import Path

TEX = Path("acl_latex.tex")
PKG = Path("arxiv_20260907")


def used_figures(tex):
    out = set()
    for line in tex.read_text().splitlines():
        line = re.sub(r"(?<!\\)%.*$", "", line)          # drop the comment tail
        out |= set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", line))
    return sorted(out)


def main():
    figs = used_figures(TEX)
    PKG.mkdir(exist_ok=True)
    for name in figs:
        src = Path(name if Path(name).suffix else name + ".pdf")
        shutil.copy(src, PKG / src.name)
    stale = [p for p in PKG.glob("*.pdf")
             if p.name not in {Path(f).name for f in figs}]
    for p in stale:
        p.unlink()
    print(f"{len(figs)} figures copied" +
          (f"; removed {len(stale)} no longer used: "
           + ", ".join(p.name for p in stale) if stale else ""))


if __name__ == "__main__":
    main()
