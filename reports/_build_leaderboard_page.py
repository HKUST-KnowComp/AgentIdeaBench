#!/usr/bin/env /usr/bin/python3
"""
Render the agentic leaderboard in the two forms GitHub can serve.

  site/index.html   the standalone page, deployed by .github/workflows/pages.yml
  LEADERBOARD.md    the same ranking as a markdown table, which renders inside
                    the repository itself and needs no Pages deployment

Both are generated from reports/leaderboard_full.json alone, so a clone with no
database and no API key can rebuild them. Markup and styling live in
reports/leaderboard_template.html; this script only substitutes the data blob
and the headline figures quoted in the prose, so no number is typed by hand.

Run reports/_leaderboard_full.py first to refresh the JSON.
"""
import json
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "reports" / "leaderboard_full.json"
TPL = ROOT / "reports" / "leaderboard_template.html"
OUT_HTML = ROOT / "site" / "index.html"
OUT_FRAGMENT = ROOT / "reports" / "leaderboard_fragment.html"
OUT_MD = ROOT / "LEADERBOARD.md"

# The template is authored as page content, without a document skeleton, because
# that is what the Artifact publisher expects: it supplies doctype, charset and
# viewport itself. A plain web server supplies none of them, so a file served
# straight from GitHub Pages or a personal domain would render in quirks mode
# with no mobile viewport. build_html therefore wraps the same content in a real
# document, and build_fragment keeps the unwrapped form for publishing.
SKELETON = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description" content="{desc}">
<link rel="canonical" href="https://moyunxiang.com/agentideabench/">
{head}
</head>
<body>
{body}
</body>
</html>
"""
DESCRIPTION = ("AgentIdeaBench Active-track leaderboard: hypothesis quality for "
               "every model evaluated with agent-controlled literature search.")

DIMS = ["originality", "feasibility", "clarity", "impact", "specificity"]


def excluded_note(data):
    """One sentence naming what the roster leaves out, and why."""
    dup = [e["model"].split("/")[-1] for e in data["excluded"]
           if e["reason"] == "duplicate serving route"]
    solo = [e["model"].split("/")[-1] for e in data["excluded"]
            if e["reason"] == "Active-only, no Static run"]
    parts = []
    if dup:
        parts.append(f"{', '.join(dup)} is left out as a second serving route to "
                     f"a model already listed, not a second model")
    if solo:
        parts.append(f"{', '.join(solo)} ran the Active track only, so "
                     f"{'they have' if len(solo) > 1 else 'it has'} no Static "
                     f"column to rank against")
    return ("Three models are excluded: " if len(dup) + len(solo) == 3
            else "Excluded: ") + "; ".join(parts) + "."


def build_html(data, cells):
    rows = data["rows"]
    paired = [r for r in rows if r["gain"] is not None]
    keep = ("model", "name", "vendor", "access", "active", "active_sem",
            "static", "gain", "dims", "static_dims",
            "n_subfields_active", "n_subfields_static", "turns")
    slim = {"rows": [{k: r[k] for k in keep} for r in rows]}
    best_open = next(r for r in rows if r["access"] == "open")
    subs = {
        "__DATA__": json.dumps(slim, separators=(",", ":")),
        "MODELS_N": str(data["n_models"]),
        "OPEN_N": str(data["n_open"]),
        "CELLS_N": f"{cells:,}",
        "OPEN_BEST": f"{best_open['active']:.2f}",
        "OPEN_BEST_NAME": best_open["name"],
        "CEIL_SCORE": f"{rows[0]['active']:.2f}",
        "CEIL_NAME": rows[0]["name"],
        "GAIN_POS": str(sum(r["gain"] > 0 for r in paired)),
        "GAIN_TOTAL": str(len(paired)),
        "EXCLUDED_NOTE": excluded_note(data),
        "BUILD_DATE": date.today().isoformat(),
    }
    html = TPL.read_text()
    # Longest key first: OPEN_BEST is a prefix of OPEN_BEST_NAME, and replacing
    # the short one first would leave "6.33_NAME" on the page.
    for k in sorted(subs, key=len, reverse=True):
        html = html.replace(k, subs[k])

    OUT_FRAGMENT.write_text(html)

    # The template's title, font links and stylesheet all sit ahead of the first
    # markup, so the end of the <style> block is the head/body boundary.
    cut = html.index("</style>") + len("</style>")
    head, body = html[:cut], html[cut:]
    OUT_HTML.parent.mkdir(exist_ok=True)
    OUT_HTML.write_text(SKELETON.format(
        desc=DESCRIPTION, head=head.strip(), body=body.strip()))
    return OUT_HTML.stat().st_size


def build_markdown(data, cells):
    rows = data["rows"]
    paired = [r for r in rows if r["gain"] is not None]
    pos = sum(r["gain"] > 0 for r in paired)
    best_open = next(r for r in rows if r["access"] == "open")

    L = []
    L.append("# AgentIdeaBench — Active track leaderboard\n")
    L.append(
        f"Every model receives a scientific subfield and nothing else, searches "
        f"Semantic Scholar itself under a budget of 10 tool calls, and proposes a "
        f"testable hypothesis. {data['n_models']} models, 40 subfields in five "
        f"disciplines, 3 hypotheses each, 3 critic models per hypothesis, "
        f"{cells:,} critic scores in total.\n")
    L.append(
        f"The best open-weight model is **{best_open['name']}** at "
        f"{best_open['active']:.2f}. The held-out ceiling is "
        f"{rows[0]['active']:.2f} ({rows[0]['name']}). Of the {len(paired)} "
        f"models run in both tracks, {pos} score higher with the search tool "
        f"than with a supplied reading list, and {len(paired) - pos} score "
        f"lower.\n")
    L.append(
        "Closed-source models are marked † . They are reported here in full and "
        "held out of every headline statistic, exactly as in the paper, whose "
        "claims are scoped to the 28 open-weight models.\n")
    L.append(excluded_note(data) + "\n")
    L.append("| # | Model | Vendor | Static | Active | Δ | Turns |")
    L.append("|--:|---|---|--:|--:|--:|--:|")
    for r in rows:
        dag = " †" if r["access"] == "closed" else ""
        stat = f"{r['static']:.2f}" if r["static"] is not None else "—"
        gain = f"{r['gain']:+.2f}" if r["gain"] is not None else "—"
        turns = f"{r['turns']:.1f}" if r["turns"] is not None else "—"
        L.append(f"| {r['rank']} | {r['name']}{dag} | {r['vendor']} | {stat} | "
                 f"**{r['active']:.2f}** | {gain} | {turns} |")

    L.append("\n#### Column Definitions\n")
    L.append("| Column | Definition | Computation | Unit / Range | Source |")
    L.append("|---|---|---|---|---|")
    defs = [
        ("#", "Rank by Active total, descending.",
         "Position after sorting all scored models.",
         f"1–{data['n_models']}", "computed"),
        ("Model", "The evaluated hypothesis generator.",
         "Display name for the routed model id.", "—",
         "`reports/_leaderboard_full.py` `DISPLAY`"),
        ("Vendor", "Publishing organization. † marks closed weights.",
         "Prefix of the model id; † is the gateway/Gemini rule shared with "
         "`experiments/e36_leaderboard_subscores.py`.", "—", "model id"),
        ("Static", "Weighted hypothesis quality with a supplied reading list "
         "(Track B).",
         "Drop the most generous of 3 critics per hypothesis, weight the five "
         "dimensions O2/I1.5/F1/C0.5/S0.5 normalized by 5.5, mean over 3 "
         "hypotheses, then over subfields.",
         "1–10", "`lit8d_scores_3seed`, track B"),
        ("Active", "Weighted hypothesis quality with agentic search (Track C).",
         "Identical aggregation to Static, on the agentic rollouts.",
         "1–10", "`lit8d_scores_3seed`, track C"),
        ("Δ", "Gain from agentic search over a supplied reading list.",
         "Paired mean of Active − Static over subfields scored in both tracks. "
         "`—` where the model has no Static run.",
         "score points", "computed, paired"),
        ("Turns", "Mean tool calls issued per rollout, out of a budget of 10. "
         "Describes behavior, not quality.",
         "Counted from the logged agent transcript.",
         "0–10 calls", "`reports/e43_active_turns.json`"),
    ]
    for c, d, comp, unit, src in defs:
        L.append(f"| {c} | {d} | {comp} | {unit} | {src} |")

    L.append(
        "\nPer-dimension scores, standard errors and an interactive version of "
        "this table are at <https://moyunxiang.com/agentideabench/>. Row-level scores for every model are in "
        "`release_data/`.\n")
    L.append(
        "Critic scores are model judgments, not measurements of scientific "
        "merit. Standard errors are not shown here; they are in "
        "`reports/leaderboard_full.json` as `active_sem` and `gain_sem`.\n")
    L.append(f"Generated by `reports/_build_leaderboard_page.py` on "
             f"{date.today().isoformat()}.")
    OUT_MD.write_text("\n".join(L) + "\n")
    return len(rows)


def main():
    data = json.loads(SRC.read_text())
    cells = data["n_critic_scores"]
    n_bytes = build_html(data, cells)
    n_rows = build_markdown(data, cells)
    print(f"wrote {OUT_HTML.relative_to(ROOT)}  {n_bytes:,} bytes  (standalone document)")
    print(f"wrote {OUT_FRAGMENT.relative_to(ROOT)}  (unwrapped, for the artifact publisher)")
    print(f"wrote {OUT_MD.relative_to(ROOT)}  {n_rows} rows")


if __name__ == "__main__":
    main()
