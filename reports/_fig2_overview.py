r"""Paper Figure 2: AgentIdeaBench pipeline diagram.

Schematic only -- no data dependency, nothing here is a measured number. Every
label restates a protocol fact already stated in the paper text or fixed in
config.json (subfield counts, sampling tiers, the Static and Active conditions,
the SEARCH/FETCH/FINAL contract, tool budget, output format, critic count,
the coherence and boilerplate caps, dimensions, weights), so the figure carries
no claim that is not sourced from Sections 3-4 and Appendix G.

STYLE. Dense agent-pipeline convention: dashed rounded group containers with
their own titles, pastel module boxes with monospace names and a small badge
naming what drives the module (LLM vs Semantic Scholar API), a red decision
diamond on the Active loop, and a Symbols legend. The point is that Active is
a *loop with an exit condition*, which a flat left-to-right flow cannot show.

ARROWS (2026-09-06). One head geometry everywhere, and every multi-leg
connection is an orthogonal route() path with rounded corners: the Subfield
split and the two-track join are a stub, a bus and coloured drops, and the
retry loop runs over the agent onto its top edge. The previous version drew
these as arc3 curves whose tails started underneath the box they left, which
is what made the diagram look hand-assembled.

Paper-wide colour semantics are preserved: Static = blue #2b6cb0, Active =
orange #dd6b20, neutral grey (pair CVD-validated). Pastel fills carry stage
identity, not track identity, so they never compete with that contrast:
mint = retrieval, lavender = an LLM call, yellow = a guard that caps a score,
blue-grey = shared artifact or aggregation.

LAYOUT IS IN POINTS, NOT GUESSED FRACTIONS. The vertical stack is a single
top-down cursor in typographic points (see the STACK block), and every box
height comes from need_h(), so a box can never be handed more lines than it can
hold. main() ends by re-measuring every string against the box that holds it
and raising if anything overflows, so a layout regression fails loudly instead
of shipping. Group titles and free labels are measured too, which is how the
v1 title and legend overflows were found.

Canvas is 6.3in wide = ACL \textwidth and the axes fills the figure, so point
sizes here are the point sizes in the compiled paper (include at
width=\textwidth, no rescaling).

Writes fig2_benchmark_overview.{pdf,png} to reports/figures/summary/.
Previous versions: archive/reports/_fig2_overview_v1_20260904.py (5.20in, the
one with crowded Active column and a half-empty Static column) and
archive/reports/_fig2_overview_pre_sciresearcher_20260904.py (flat six-box).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import (FancyBboxPatch, FancyArrowPatch, PathPatch,
                                Polygon)
from matplotlib.path import Path as MplPath

ROOT = Path(__file__).parent.parent
OUT = ROOT / "reports" / "figures" / "summary"
OUT.mkdir(parents=True, exist_ok=True)

FIG_W, FIG_H = 6.3, 4.78
PT_X, PT_Y = 1.0 / (FIG_W * 72), 1.0 / (FIG_H * 72)

# Paper-wide track semantics (fixed 2026-08-03, CVD-validated; do not change).
BLUE, ORANGE = "#2b6cb0", "#dd6b20"
BLUE_BG, ORANGE_BG = "#eaf1f9", "#fdefe3"

MINT, MINT_E = "#dff0e6", "#5f9c80"        # retrieval
LILAC, LILAC_E = "#e4dff2", "#8377b5"      # an LLM call
YELL, YELL_E = "#fdf2d4", "#bb9a30"        # a guard that caps a score
SLATE, SLATE_E = "#e6ecf3", "#7f96ad"      # shared artifact / aggregation

INK, MUTED, ARR = "#1f2937", "#55636f", "#3f4a56"
RED = "#c0392b"
GROUP_E, GROUP_BG = "#9aa7b4", "#fcfdfe"

# Module names 7.6pt bold, detail lines 6.8pt, group titles 8.2pt. These sit
# below the 8pt floor the earlier flat version held: at this density 8pt
# everywhere needs a figure taller than the page can spare, and the detail
# lines are keyword fragments rather than prose.
FS_MOD, FS, FS_SMALL, FS_GROUP = 7.6, 6.8, 6.2, 8.2

plt.rcParams.update({"font.size": FS, "pdf.fonttype": 42, "ps.fonttype": 42})
MONO = "DejaVu Sans Mono"

# Vertical rhythm inside a module box, in points from the box top.
PAD_TOP, LINE_H, PAD_BOT, PAD_SIDE = 9.0, 9.0, 6.0, 5.0

# ------------------------------------------------------------------ STACK --
# One top-down cursor in points. Change a gap here, not inside build().
Y0 = 3.0            # top margin
G_TITLE = 14.0      # gutter a dashed group reserves for its own title
G1_PAD_B = 5.0      # acquisition group, below its boxes
GAP_ACQ_SF = 11.0   # acquisition group -> Subfield
GAP_SF_TG = 14.0    # Subfield -> track groups (the two curved fan-out arrows)
TG_GAP = 11.0       # context box -> synthesis box, inside a track
TG_PAD_B = 14.0     # track group bottom (holds the malformed-rollout note)
GAP_TG_HY = 12.0    # track groups -> Hypothesis
GAP_HY_SG = 15.0    # Hypothesis -> scoring group
SG_PAD_B = 5.0

_CHECKS = []          # (text_artist, allowed_width_in_axes_fraction, where)
_SHORT = []           # boxes whose height cannot hold the lines they were given
# Every primitive is also recorded in points from the canvas top-left, which is
# what _fig2_overview_pptx.py replays as native PowerPoint shapes. Recording
# here rather than re-declaring the layout in the exporter is what keeps the
# .pptx and the .pdf from drifting apart.
_SHAPES = []


def _rec(**kw):
    _SHAPES.append(kw)


def PX(x_frac):
    """Axes-fraction x -> points from the canvas left edge."""
    return x_frac * FIG_W * 72.0


def need_h(n_lines):
    """Points a box needs for its name plus n detail lines."""
    return PAD_TOP + 9.0 + max(n_lines - 1, 0) * LINE_H + PAD_BOT


def Y(pt):
    """Axes-fraction y of a point measured downward from the figure top."""
    return 1.0 - pt * PT_Y


def HF(pt):
    return pt * PT_Y


def _track(t, allowed, where):
    _CHECKS.append((t, allowed, where))
    return t


def group(ax, x, top_pt, w, h_pt, title=None, color=GROUP_E, ls=(0, (4, 2.6))):
    y, h = Y(top_pt + h_pt), HF(h_pt)
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.012",
        linewidth=0.8, edgecolor=color, facecolor=GROUP_BG,
        linestyle=ls, zorder=1))
    _rec(kind="rect", x=PX(x), y=top_pt, w=PX(w), h=h_pt, fill=GROUP_BG,
         edge=color, lw=0.8, dash=True, r=0.012 * FIG_W * 72.0)
    if title:
        _rec(kind="text", x=PX(x + w / 2), y=top_pt + 4.0, s=title,
             size=FS_GROUP, color=color, bold=True, mono=False, ha="center",
             va="top")
        t = ax.text(x + w / 2, Y(top_pt + 4.0), title, fontsize=FS_GROUP,
                    fontweight="bold", color=color, ha="center", va="top",
                    zorder=6)
        _track(t, w - 2 * PAD_SIDE * PT_X, f"group:{title}")


def box(ax, x, top_pt, w, h_pt, name, lines=(), fill=LILAC, edge=LILAC_E,
        badge=None, mono=True, name_color=INK, align="left"):
    """One module: name on top, keyword detail lines beneath it."""
    if lines and h_pt < need_h(len(lines)) - 1e-9:
        _SHORT.append((name, h_pt, need_h(len(lines))))
    y, h = Y(top_pt + h_pt), HF(h_pt)
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.008",
        linewidth=0.9, edgecolor=edge, facecolor=fill, zorder=3))
    _rec(kind="rect", x=PX(x), y=top_pt, w=PX(w), h=h_pt, fill=fill, edge=edge,
         lw=0.9, dash=False, r=0.008 * FIG_W * 72.0)
    # The badge occupies the top-right corner, so the name gets the rest.
    name_room = w - 2 * PAD_SIDE * PT_X - (17.0 * PT_X if badge else 0.0)
    t = ax.text(x + w / 2, Y(top_pt + PAD_TOP), name, fontsize=FS_MOD,
                fontweight="bold", family=MONO if mono else None,
                color=name_color, ha="center", va="baseline", zorder=5)
    _track(t, name_room, f"name:{name}")
    _rec(kind="text", x=PX(x + w / 2), y=top_pt + PAD_TOP, s=name, size=FS_MOD,
         color=name_color, bold=True, mono=mono, ha="center", va="baseline")
    if badge:
        _rec(kind="text", x=PX(x + w - PAD_SIDE * PT_X), y=top_pt + 3.0,
             s=badge, size=FS_SMALL, color=edge, bold=False, mono=True,
             ha="right", va="top")
        ax.text(x + w - PAD_SIDE * PT_X, Y(top_pt + 3.0), badge,
                fontsize=FS_SMALL, family=MONO, ha="right", va="top",
                color=edge, zorder=5)
    tx = x + PAD_SIDE * PT_X if align == "left" else x + w / 2
    ha = "left" if align == "left" else "center"
    for i, ln in enumerate(lines):
        t = ax.text(tx, Y(top_pt + PAD_TOP + 9.0 + i * LINE_H), ln,
                    fontsize=FS, color=MUTED, ha=ha, va="baseline", zorder=5)
        _track(t, w - 2 * PAD_SIDE * PT_X, f"line:{name}/{ln[:18]}")
        _rec(kind="text", x=PX(tx), y=top_pt + PAD_TOP + 9.0 + i * LINE_H,
             s=ln, size=FS, color=MUTED, bold=False, mono=False, ha=ha,
             va="baseline")


def diamond(ax, cx, cy_pt, w, h_pt, lab):
    cy, h = Y(cy_pt), HF(h_pt)
    ax.add_patch(Polygon([(cx, cy + h / 2), (cx + w / 2, cy),
                          (cx, cy - h / 2), (cx - w / 2, cy)],
                         closed=True, facecolor="white", edgecolor=RED,
                         linewidth=1.0, zorder=4))
    _rec(kind="diamond", x=PX(cx - w / 2), y=cy_pt - h_pt / 2, w=PX(w),
         h=h_pt, fill="#ffffff", edge=RED, lw=1.0)
    _rec(kind="text", x=PX(cx), y=cy_pt, s=lab, size=FS, color=RED, bold=True,
         mono=False, ha="center", va="center")
    ax.text(cx, cy, lab, fontsize=FS, fontweight="bold", color=RED,
            ha="center", va="center", zorder=5)


# One arrowhead shape for the whole figure. v2 mixed head lengths from 5.0 to
# 5.5 on shafts between 4pt and 90pt long, so the short connectors rendered as
# bare triangles while the long ones looked thin; a single slimmer head reads
# as the same arrow at every length.
HEAD_W, HEAD_L = 2.1, 4.4


def arrow(ax, p0, p1, color=ARR, lw=1.2, ls="-", rad=0.0, head=HEAD_L,
          zorder=4):
    """A straight (or slightly arced) arrow between two (x_fraction, y_pt)."""
    ax.add_patch(FancyArrowPatch(
        (p0[0], Y(p0[1])), (p1[0], Y(p1[1])),
        arrowstyle=f"-|>,head_width={HEAD_W},head_length={head}",
        mutation_scale=1.0, linewidth=lw, color=color, linestyle=ls,
        connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0, zorder=zorder))
    _rec(kind="line", pts=[(PX(p0[0]), p0[1]), (PX(p1[0]), p1[1])],
         color=color, lw=lw, dash=ls != "-", arrow=True)


def route(ax, pts, color=ARR, lw=1.2, ls="-", r_pt=4.5, head=True,
          zorder=4):
    """Orthogonal polyline through (x_fraction, y_points_from_top) waypoints.

    Every leg is axis-aligned and the corners are rounded with a fixed
    typographic radius, so a fan-out or a loop-back reads as one path instead
    of as the two disconnected arcs v2 drew. The radius is converted per axis
    because the canvas is 6.3x4.78in and the axes span 0..1 on both.
    """
    A = [(x, Y(y)) for x, y in pts]
    rx, ry = r_pt * PT_X, r_pt * PT_Y
    verts, codes = [A[0]], [MplPath.MOVETO]
    for i in range(1, len(A) - 1):
        (xp, yp), (xc, yc), (xn, yn) = A[i - 1], A[i], A[i + 1]

        def step(fx, fy, tx, ty):
            """Point r away from the corner along the leg towards (tx, ty)."""
            dx, dy = tx - fx, ty - fy
            if abs(dx) >= abs(dy):
                s = min(rx, abs(dx)) * (1 if dx > 0 else -1)
                return (fx + s, fy)
            s = min(ry, abs(dy)) * (1 if dy > 0 else -1)
            return (fx, fy + s)

        verts += [step(xc, yc, xp, yp), (xc, yc), step(xc, yc, xn, yn)]
        codes += [MplPath.LINETO, MplPath.CURVE3, MplPath.CURVE3]
    verts.append(A[-1]); codes.append(MplPath.LINETO)
    ax.add_patch(PathPatch(MplPath(verts, codes), fill=False, lw=lw,
                           edgecolor=color, linestyle=ls, capstyle="round",
                           joinstyle="round", zorder=zorder))
    _rec(kind="line", pts=[(PX(x), y) for x, y in pts], color=color, lw=lw,
         dash=ls != "-", arrow=head)
    if head:
        # a head-only stub laid over the final leg, so the head sits exactly on
        # the endpoint and its shaft is hidden by the path it duplicates
        (xa, ya), (xb, yb) = A[-2], A[-1]
        d = np.hypot((xb - xa) / PT_X, (yb - ya) / PT_Y) or 1.0
        f = min(1.0, HEAD_L / d)
        ax.add_patch(FancyArrowPatch(
            (xb - (xb - xa) * f, yb - (yb - ya) * f), (xb, yb),
            arrowstyle=f"-|>,head_width={HEAD_W},head_length={HEAD_L}",
            mutation_scale=1.0, linewidth=lw, color=color, shrinkA=0,
            shrinkB=0, zorder=zorder))


def label(ax, x, y_pt, s, color=MUTED, size=FS_SMALL, allowed=None, where="",
          **kw):
    t = ax.text(x, Y(y_pt), s, fontsize=size, color=color, zorder=6, **kw)
    _rec(kind="text", x=PX(x), y=y_pt, s=s, size=size, color=color,
         bold=kw.get("fontweight") == "bold", mono=False,
         ha=kw.get("ha", "left"), va=kw.get("va", "baseline"),
         italic=kw.get("style") == "italic")
    if allowed is not None:
        _track(t, allowed, where or f"label:{s[:22]}")
    return t


def build(ax):
    H1, H2, H3, H4 = need_h(1), need_h(2), need_h(3), need_h(4)

    # ------------------------------------------- row 1: legend + provenance --
    g1_h = G_TITLE + H2 + G1_PAD_B
    g1_bot = Y0 + g1_h
    bx_top = Y0 + G_TITLE

    # Legend is wider than v1: "guard that caps a score" ran past the old edge.
    group(ax, 0.010, Y0, 0.218, g1_h, ls=(0, (3, 2)))
    label(ax, 0.119, Y0 + 4.0, "Symbols", color=INK, size=FS_GROUP,
          ha="center", va="top", fontweight="bold")
    for i, (tag, txt, col) in enumerate((
            ("LLM", "driven by an LLM call", LILAC_E),
            ("API", "calls Semantic Scholar", MINT_E),
            ("cap", "guard that caps a score", YELL_E))):
        yy = Y0 + 17.5 + i * 7.4
        ax.text(0.022, Y(yy), tag, fontsize=FS_SMALL, family=MONO, color=col,
                fontweight="bold", ha="left", va="center", zorder=6)
        _rec(kind="text", x=PX(0.022), y=yy, s=tag, size=FS_SMALL, color=col,
             bold=True, mono=True, ha="left", va="center")
        label(ax, 0.055, yy, txt, ha="left", va="center",
              allowed=0.228 - 0.006 - 0.055, where=f"legend:{tag}")
    for i, (col, ls, txt) in enumerate((
            (ARR, "-", "forward process"),
            (RED, (0, (2, 1.6)), "loop back / retry"))):
        yy = Y0 + 39.7 + i * 7.4
        arrow(ax, (0.020, yy), (0.050, yy), color=col, lw=1.1, ls=ls)
        label(ax, 0.055, yy, txt, ha="left", va="center",
              allowed=0.228 - 0.006 - 0.055, where=f"legend:{txt[:8]}")

    group(ax, 0.236, Y0, 0.754, g1_h, "Subfield Acquisition")
    for i, (bx, nm, ln, bg) in enumerate((
            (0.244, "SS search", ("100 topics,", "5 disciplines"), True),
            (0.430, "Sample", ("rank 1-5 / 6-10,", "40 per discipline"), False),
            (0.616, "Filter", ("abstract 80-800 w,", "2025-04 \u2013 2026-01"), False),
            (0.802, "Rank refs", ("top-15 pool,", "fixed 5 kept"), True))):
        # 0.168 rather than 0.176: at a 0.186 pitch the wider box left a 4.5pt
        # gap, and a 4.4pt arrowhead in a 4.5pt gap is a bare triangle. The
        # widest detail line here measures 64.5pt against 66.2pt of room.
        box(ax, bx, bx_top, 0.168, H2, nm, ln, MINT, MINT_E,
            badge="API" if bg else None)
        if i < 3:
            arrow(ax, (bx + 0.169, bx_top + H2 / 2), (bx + 0.1855, bx_top + H2 / 2))

    # ------------------------------------------- row 2: the shared task unit --
    sf_top = g1_bot + GAP_ACQ_SF
    arrow(ax, (0.500, g1_bot + 1.5), (0.500, sf_top - 1.5), lw=1.3)
    box(ax, 0.335, sf_top, 0.330, H1, "Subfield",
        ("name  +  5 curated reference papers",), SLATE, SLATE_E,
        mono=False, align="center")

    # ------------------------------------------------------ row 3: two tracks --
    # Asymmetric columns: Active holds a tool contract, a loop and an exit
    # condition, Static holds neither, so equal columns left Static half empty.
    tg_top = sf_top + H1 + GAP_SF_TG
    ctx_top = tg_top + G_TITLE
    mod_top = ctx_top + H3 + TG_GAP
    tg_h = G_TITLE + H3 + TG_GAP + H2 + TG_PAD_B
    tg_bot = tg_top + tg_h

    SL, SR = 0.010, 0.438          # Static group
    AL, AR = 0.452, 0.990          # Active group
    scx, acx = (SL + SR) / 2, (AL + AR) / 2

    # Fan-out as a stub, a bus and two coloured drops. v2 drew two arcs whose
    # tails started underneath the Subfield box, so they read as free-floating
    # curves rather than as one split.
    sf_bot = sf_top + H1
    bus1 = sf_bot + GAP_SF_TG * 0.30
    route(ax, [(0.500, sf_bot + 1.5), (0.500, bus1)], head=False, lw=1.3)
    route(ax, [(scx, bus1), (acx, bus1)], head=False, lw=1.3)
    route(ax, [(scx, bus1), (scx, tg_top - 1.5)], color=BLUE, lw=1.3)
    route(ax, [(acx, bus1), (acx, tg_top - 1.5)], color=ORANGE, lw=1.3)

    group(ax, SL, tg_top, SR - SL, tg_h, color=BLUE)
    label(ax, scx, tg_top + 4.0, "Static  |  curated retrieval",
          color=BLUE, size=FS_GROUP, ha="center", va="top", fontweight="bold",
          allowed=(SR - SL) - 2 * PAD_SIDE * PT_X, where="title:Static")
    box(ax, 0.026, ctx_top, 0.396, H3, "Available context",
        ("subfield name  +  5 reference papers",
         "title + abstract shown",
         "no tool access"), BLUE_BG, BLUE, mono=False, name_color=BLUE,
        align="center")
    arrow(ax, (scx, ctx_top + H3 + 1.5), (scx, mod_top - 1.5), lw=1.3)
    box(ax, 0.062, mod_top, 0.324, H2, "StaticSynthesis",
        ("read the given refs, commit to",
         "one hypothesis; single pass, no revision"),
        LILAC, LILAC_E, badge="LLM", align="center")

    group(ax, AL, tg_top, AR - AL, tg_h, color=ORANGE)
    label(ax, acx, tg_top + 4.0, "Active  |  agent-controlled retrieval",
          color=ORANGE, size=FS_GROUP, ha="center", va="top",
          fontweight="bold", allowed=(AR - AL) - 2 * PAD_SIDE * PT_X,
          where="title:Active")
    box(ax, 0.468, ctx_top, 0.506, H3, "Available tools", (), ORANGE_BG,
        ORANGE, mono=False, name_color=ORANGE)
    for i, (tool, desc) in enumerate((
            ("SEARCH: <query>", "keyword query, 4-10 words"),
            ("FETCH: <paperId>", "pull that paper's references"))):
        yy = ctx_top + PAD_TOP + 9.5 + i * LINE_H
        ax.text(0.482, Y(yy), tool, fontsize=FS_SMALL, family=MONO,
                color=ORANGE, ha="left", va="center", zorder=6)
        _rec(kind="text", x=PX(0.482), y=yy, s=tool, size=FS_SMALL,
             color=ORANGE, bold=False, mono=True, ha="left", va="center")
        label(ax, 0.628, yy, desc, ha="left", va="center",
              allowed=0.974 - PAD_SIDE * PT_X - 0.628, where=f"tool:{tool[:6]}")
    label(ax, 0.721, ctx_top + PAD_TOP + 9.5 + 2 * LINE_H,
          "subfield name only;  budget = 10 calls",
          ha="center", va="center", style="italic",
          allowed=0.506 - 2 * PAD_SIDE * PT_X, where="tools:budget")

    agent_l, agent_r = 0.464, 0.808
    arrow(ax, ((agent_l + agent_r) / 2, ctx_top + H3 + 1.5),
          ((agent_l + agent_r) / 2, mod_top - 1.5), lw=1.3)
    box(ax, agent_l, mod_top, agent_r - agent_l, H2, "ActiveAgent",
        ("SEARCH broadly, FETCH the best lead's",
         "references, widen or deepen, then FINAL"),
        LILAC, LILAC_E, badge="LLM+API", align="center")

    # The exit condition: the whole content of Track C, invisible in a flat
    # flow. The "no" arc stays inside the row instead of climbing into the
    # gap above the box, which is what made v1 look crowded on this side.
    dcx, dcy, dh = 0.888, mod_top + H2 / 2, 30.0
    diamond(ax, dcx, dcy, 0.080, dh, "FINAL?")
    arrow(ax, (agent_r + 0.002, dcy), (dcx - 0.042, dcy))
    # "no" leaves the diamond's top vertex, runs back over the agent and drops
    # onto its top edge: a loop-back that looks like a loop-back. v2 arced it
    # into the box's right flank, where it crossed the forward arrow and struck
    # through its own label.
    loop_y = mod_top - 5.6
    route(ax, [(dcx, dcy - dh / 2 - 1.0), (dcx, loop_y),
               (agent_r - 0.052, loop_y), (agent_r - 0.052, mod_top - 1.0)],
          color=RED, lw=1.1, ls=(0, (2.4, 1.8)), r_pt=2.6)
    label(ax, (agent_r + dcx) / 2, loop_y, "no", color=RED, ha="center",
          va="center", bbox=dict(facecolor="white", edgecolor="none",
                                 pad=0.6))
    label(ax, 0.902, dcy + dh / 2 + 6.0, "yes", color=RED, ha="left",
          va="center")
    label(ax, AL + 0.012, tg_bot - 6.5,
          "3 malformed commands in a row: rollout discarded",
          ha="left", va="center", style="italic",
          allowed=(AR - AL) - 0.024, where="note:malformed")

    # ------------------------------------ row 4: one identical output contract --
    hy_top = tg_bot + GAP_TG_HY
    # Mirror of the fan-out: each track drops to a shared bus, and one neutral
    # arrow enters the Hypothesis box. v2 split this into a headed vertical
    # followed by a separate arc, which read as two unrelated arrows.
    bus2 = tg_bot + GAP_TG_HY * 0.52
    route(ax, [(scx, mod_top + H2 + 1.5), (scx, bus2)], color=BLUE, lw=1.3,
          head=False)
    route(ax, [(dcx, dcy + dh / 2 + 1.0), (dcx, bus2)], color=ORANGE, lw=1.3,
          head=False)
    route(ax, [(scx, bus2), (dcx, bus2)], head=False, lw=1.3)
    route(ax, [(0.500, bus2), (0.500, hy_top - 1.5)], lw=1.3)
    box(ax, 0.254, hy_top, 0.492, H1, "Hypothesis",
        ("one paragraph, 80-150 words, identical format in both tracks",),
        SLATE, SLATE_E, mono=False, align="center")

    # -------------------------------------------------- row 5: scoring chain --
    sg_top = hy_top + H1 + GAP_HY_SG
    sb_top = sg_top + G_TITLE
    sg_h = G_TITLE + H4 + SG_PAD_B
    arrow(ax, (0.500, hy_top + H1 + 1.5), (0.500, sg_top - 2.5), lw=1.3)
    group(ax, 0.010, sg_top, 0.980, sg_h, "Literature-Verified Scoring")
    # widths leave an 0.018 gap (8.2pt) between neighbours for the same reason
    # as the acquisition row: 4.5pt could not hold a readable arrow
    chain = [
        (0.018, 0.174, "QueryExtract",
         ("3 keyword queries", "from the hypothesis"), MINT, MINT_E, None),
        (0.210, 0.159, "PriorArt",
         ("Semantic Scholar,", "<= 2026-05-31,", "up to 8 papers"),
         MINT, MINT_E, "API"),
        (0.387, 0.377, "CriticEnsemble", (), LILAC, LILAC_E, "LLM"),
        (0.782, 0.200, "Aggregation",
         ("trim highest of 3,", "O2 I1.5 F1 C.5 S.5 / 5.5",
          "mean over 3 ideas"), SLATE, SLATE_E, None),
    ]
    for bx, bw, nm, ln, fc, ec, bg in chain:
        box(ax, bx, sb_top, bw, H4, nm, ln, fc, ec, badge=bg)
    for bx, bw, *_ in chain[:-1]:
        arrow(ax, (bx + bw + 0.001, sb_top + H4 / 2),
              (bx + bw + 0.0175, sb_top + H4 / 2))

    # The critic's own text on the left, the score caps as a nested chip on the
    # right, mirroring the reference figure's boxes-inside-boxes convention.
    # Guards is pushed 5pt below the ensemble's name row and inset 0.020 from
    # its right edge so the two badges no longer sit shoulder to shoulder.
    for i, ln in enumerate(("3 of 5 critics, T = 0,", "O/F/C/I/S scored 1-10,",
                            "originality argued", "against the evidence")):
        t = ax.text(0.397, Y(sb_top + PAD_TOP + 9.0 + i * LINE_H), ln,
                    fontsize=FS, color=MUTED, ha="left", va="baseline",
                    zorder=5)
        _track(t, 0.171, f"line:CriticEnsemble/{ln[:18]}")
        _rec(kind="text", x=PX(0.397),
             y=sb_top + PAD_TOP + 9.0 + i * LINE_H, s=ln, size=FS, color=MUTED,
             bold=False, mono=False, ha="left", va="baseline")
    box(ax, 0.574, sb_top + 14.0, 0.180, H2, "Guards",
        ("coherence: O 5, S 6", "boilerplate: O 6, I 6"), YELL, YELL_E,
        badge="cap")

    return sg_top + sg_h


def verify(fig, bottom_pt):
    """Re-measure every tracked string against the box that holds it."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.axes[0].transData.inverted()
    bad = []
    for t, allowed, where in _CHECKS:
        bb = t.get_window_extent(renderer=r)
        x0, _ = inv.transform((bb.x0, bb.y0))
        x1, _ = inv.transform((bb.x1, bb.y1))
        w = abs(x1 - x0)
        if w > allowed + 1e-9:
            bad.append((where, w, allowed))
    for where, w, allowed in bad:
        print(f"  OVERFLOW {where}: {w:.4f} > {allowed:.4f} "
              f"({(w - allowed) * FIG_W * 72:.1f}pt over)")
    for name, h, need in _SHORT:
        print(f"  TOO SHORT {name}: h={h:.1f}pt needs {need:.1f}pt")
    slack = FIG_H * 72 - bottom_pt
    print(f"  bottom margin {slack:.1f}pt")
    return bad + _SHORT


def main():
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    bottom = build(ax)
    bad = verify(fig, bottom)
    if bad:
        raise SystemExit(f"{len(bad)} label(s) overflow their box; fix layout")

    for ext in ("pdf", "png", "svg"):
        # svg.fonttype "none" keeps every string as a <text> node, so the SVG
        # opens in Illustrator/Figma/Inkscape as editable type rather than as
        # outlines. It is the re-layout master; the pdf is what LaTeX includes.
        with plt.rc_context({"svg.fonttype": "none"}):
            fig.savefig(OUT / f"fig2_benchmark_overview.{ext}",
                        dpi=300 if ext == "png" else None)
    plt.close(fig)
    print(f"wrote fig2_benchmark_overview.pdf/.png/.svg  "
          f"({len(_CHECKS)} labels checked, 0 overflow, {FIG_W}x{FIG_H}in, "
          f"{len(_SHAPES)} shapes recorded)")


if __name__ == "__main__":
    main()
