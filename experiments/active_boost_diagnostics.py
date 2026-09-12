"""
4 diagnostics on existing Active vs Static data (read-only):

  D1 - Overlap between agent-searched paperIds and survey-curated static refs.
  D2 - Per-dimension boost contribution (which of O/F/C/I/S drives boost?).
  D3 - Tool-call count vs boost (per-model & Spearman across models).
  D4 - Topic alignment (active-searched titles vs target paper title token overlap).

Output: plain text table to stdout + JSON dump to experiments/active_boost_diagnostics.json
"""
import json
import re
import sqlite3
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
DB_R = ROOT / "data" / "results.db"
DB_P = ROOT / "data" / "papers.db"
OUT_JSON = ROOT / "experiments" / "active_boost_diagnostics.json"

WEIGHTS = {"originality": 2.0, "feasibility": 1.0, "clarity": 0.5,
           "impact": 1.5, "specificity": 0.5}
DIMS = list(WEIGHTS)
WEIGHT_SUM = sum(WEIGHTS.values())

STOPWORDS = {
    "with", "from", "using", "based", "through", "this", "that", "their", "have",
    "been", "more", "than", "such", "also", "when", "will", "what", "some", "very",
    "into", "about", "between", "these", "they", "were", "only", "over", "study",
    "paper", "approach", "method", "methods", "result", "results", "data",
    "effect", "effects", "model", "models", "analysis", "research", "novel",
    "new", "high", "low", "via", "under", "show", "shows", "shown", "use", "used",
}

_PID_RE = re.compile(r'"paperId"\s*:\s*"([0-9a-f]{40})"')
_TITLE_RE = re.compile(r'"title"\s*:\s*"((?:[^"\\]|\\.)*)"')
# Escaped variants for raw 4000-char-truncated traces where outer JSON fails to parse
_PID_RE_ESC = re.compile(r'\\"paperId\\"\s*:\s*\\"([0-9a-f]{40})\\"')
_TITLE_RE_ESC = re.compile(r'\\"title\\"\s*:\s*\\"((?:[^"\\]|\\\\.)*?)\\"')
_WORD_RE = re.compile(r"[a-zA-Z]{4,}")
_TOOL_RE = re.compile(r'"tool"\s*:\s*"')


def weighted(scores):
    try:
        return sum(WEIGHTS[d] * float(scores[d]) for d in DIMS) / WEIGHT_SUM
    except (KeyError, TypeError, ValueError):
        return None


def parse_trace(raw):
    """Return (visited_pids: set, visited_titles: list[str], n_tool_calls: int)."""
    if not raw:
        return set(), [], 0
    pids, titles = set(), []
    n_calls = 0
    try:
        d = json.loads(raw)
        trace = d.get("trace") if isinstance(d, dict) else (d if isinstance(d, list) else [])
        if isinstance(d, dict) and "n_tool_calls" in d:
            n_calls = int(d.get("n_tool_calls") or 0)
        for step in trace or []:
            if not isinstance(step, dict):
                continue
            if step.get("tool"):
                if n_calls == 0:
                    n_calls += 1 if not isinstance(d, dict) or "n_tool_calls" not in d else 0
            rp = step.get("result_preview", "")
            if isinstance(rp, str):
                pids.update(_PID_RE.findall(rp))
                titles.extend(_TITLE_RE.findall(rp))
        if isinstance(d, list) and n_calls == 0:
            n_calls = sum(1 for s in d if isinstance(s, dict) and s.get("tool"))
    except json.JSONDecodeError:
        # Try both escaped and unescaped patterns (truncated outer JSON)
        pids.update(_PID_RE.findall(raw))
        pids.update(_PID_RE_ESC.findall(raw))
        titles.extend(_TITLE_RE.findall(raw))
        titles.extend(_TITLE_RE_ESC.findall(raw))
        n_calls = len(_TOOL_RE.findall(raw))
    return pids, titles, n_calls


def get_static_refs(ranked_refs_json, top_n=5):
    if not ranked_refs_json:
        return set()
    try:
        refs = json.loads(ranked_refs_json)
        return {r.get("paperId") for r in refs[:top_n] if r.get("paperId")}
    except Exception:
        return set()


def tokens(s):
    if not s:
        return set()
    return {w for w in (m.lower() for m in _WORD_RE.findall(s)) if w not in STOPWORDS}


def load_active_data():
    cp = sqlite3.connect(str(DB_P))
    cp.row_factory = sqlite3.Row
    papers = {r["paper_id"]: dict(r) for r in cp.execute(
        "SELECT paper_id, title, domain, ranked_refs_json FROM papers")}
    cp.close()
    cr = sqlite3.connect(str(DB_R))
    out = {}
    for pid, m, raw in cr.execute(
        "SELECT paper_id, idea_model, raw_response FROM results "
        "WHERE track='C' AND critic_model=''"):
        if pid not in papers:
            continue
        v_pids, v_titles, n_calls = parse_trace(raw)
        out[(m, pid)] = {
            "visited_pids": v_pids,
            "visited_titles": v_titles,
            "n_calls": n_calls,
            "static_refs": get_static_refs(papers[pid].get("ranked_refs_json")),
            "target_title": papers[pid].get("title", ""),
            "target_domain": papers[pid].get("domain", ""),
        }
    cr.close()
    return out


def load_scores_per_dim():
    """Per (model, track, dim): list of scores. Mean across all critics + ideas + papers."""
    cr = sqlite3.connect(str(DB_R))
    per_dim = defaultdict(list)
    for m, track, sj in cr.execute(
        "SELECT idea_model, track, scores_json FROM results "
        "WHERE track IN ('B','C') AND scores_json IS NOT NULL"):
        try:
            s = json.loads(sj)
        except Exception:
            continue
        for d in DIMS:
            if d in s:
                per_dim[(m, track, d)].append(float(s[d]))
    cr.close()
    return per_dim


def load_aggregate_scores():
    """Per (model, track): mean weighted absolute score from model_scores table."""
    cr = sqlite3.connect(str(DB_R))
    per = defaultdict(list)
    for m, track, sc in cr.execute(
        "SELECT idea_model, track, mean_absolute_score FROM model_scores "
        "WHERE track IN ('B','C') AND mean_absolute_score IS NOT NULL"):
        per[(m, track)].append(sc)
    cr.close()
    return {k: statistics.mean(v) for k, v in per.items()}


def ranks(xs):
    n = len(xs)
    si = sorted(range(n), key=lambda i: xs[i])
    r = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and xs[si[j + 1]] == xs[si[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            r[si[k]] = avg
        i = j + 1
    return r


def spearman(xs, ys):
    if len(xs) < 3:
        return None
    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = sum((r - mx) ** 2 for r in rx)
    dy = sum((r - my) ** 2 for r in ry)
    return num / (dx * dy) ** 0.5 if dx * dy > 0 else None


def diag1_overlap(active):
    per_model = defaultdict(list)
    for (m, pid), d in active.items():
        s, v = d["static_refs"], d["visited_pids"]
        if not s or not v:
            continue
        per_model[m].append({
            "inter": len(s & v),
            "static_n": len(s),
            "visited_n": len(v),
            "jaccard": len(s & v) / len(s | v),
        })
    rows = []
    for m, lst in per_model.items():
        if not lst:
            continue
        mi = statistics.mean(x["inter"] for x in lst)
        mv = statistics.mean(x["visited_n"] for x in lst)
        ms = statistics.mean(x["static_n"] for x in lst)
        mj = statistics.mean(x["jaccard"] for x in lst)
        cov = mi / ms if ms else 0
        rows.append((m, len(lst), mv, ms, mi, mj, cov))
    rows.sort(key=lambda r: -r[6])
    return rows


def diag2_per_dim(per_dim):
    models = sorted({m for (m, t, _) in per_dim if t == "B"}
                    & {m for (m, t, _) in per_dim if t == "C"})
    rows = []
    for m in models:
        row = {"model": m, "per_dim": {}, "per_dim_weighted_contrib": {}}
        total_w = 0.0
        for d in DIMS:
            b = per_dim.get((m, "B", d), [])
            c = per_dim.get((m, "C", d), [])
            if not b or not c:
                row["per_dim"][d] = None
                continue
            delta = statistics.mean(c) - statistics.mean(b)
            row["per_dim"][d] = delta
            row["per_dim_weighted_contrib"][d] = WEIGHTS[d] * delta / WEIGHT_SUM
            total_w += WEIGHTS[d] * delta / WEIGHT_SUM
        row["total_weighted_boost"] = total_w
        rows.append(row)
    rows.sort(key=lambda r: -r["total_weighted_boost"])
    return rows


def diag3_calls_vs_boost(active, agg):
    per = defaultdict(list)
    for (m, _), d in active.items():
        per[m].append(d["n_calls"])
    rows = []
    for m, calls in per.items():
        if (m, "B") not in agg or (m, "C") not in agg:
            continue
        rows.append((m, statistics.mean(calls), agg[(m, "B")],
                     agg[(m, "C")], agg[(m, "C")] - agg[(m, "B")], len(calls)))
    rows.sort(key=lambda r: r[1])
    if len(rows) >= 3:
        rho = spearman([r[1] for r in rows], [r[4] for r in rows])
    else:
        rho = None
    return rows, rho


def diag4_topic_alignment(active):
    per_model = defaultdict(list)
    for (m, _), d in active.items():
        tgt = tokens(d["target_title"])
        if not tgt:
            continue
        vis = set()
        for t in d["visited_titles"]:
            vis |= tokens(t)
        ov = len(tgt & vis) / len(tgt) if tgt else 0
        per_model[m].append({"overlap": ov, "tgt_n": len(tgt), "vis_n": len(vis)})
    rows = []
    for m, lst in per_model.items():
        if not lst:
            continue
        rows.append((m, len(lst),
                     statistics.mean(x["overlap"] for x in lst),
                     statistics.mean(x["tgt_n"] for x in lst),
                     statistics.mean(x["vis_n"] for x in lst)))
    rows.sort(key=lambda r: -r[2])
    return rows


def main():
    print("=" * 86)
    print("ACTIVE-MODE 加成 4-DIAGNOSTIC REPORT")
    print("=" * 86)
    active = load_active_data()
    per_dim = load_scores_per_dim()
    agg = load_aggregate_scores()
    print(f"Loaded {len(active)} (model,paper) active runs; "
          f"{len({m for (m,_) in active})} distinct models with active data")
    print()

    out = {}

    # D1
    print("─" * 86)
    print("D1 — STATIC SURVEY REFS vs ACTIVE-VISITED PAPERS")
    print("─" * 86)
    print(f"{'Model':<42} {'Np':>3} {'Visit':>6} {'Static':>6}  {'∩':>4} {'Jacc':>5}  {'Cov':>5}")
    d1 = diag1_overlap(active)
    for m, n, mv, ms, mi, mj, cov in d1:
        print(f"  {m:<40} {n:>3} {mv:>6.1f} {ms:>6.1f}  {mi:>4.2f} {mj:>5.3f}  {cov*100:>4.1f}%")
    print()
    print("  Cov = mean(|active∩static|) / |static| = % of survey refs the agent rediscovered.")
    out["d1_overlap"] = [{"model": m, "n_papers": n, "mean_visited": mv,
                          "mean_static": ms, "mean_intersect": mi,
                          "mean_jaccard": mj, "coverage": cov}
                         for m, n, mv, ms, mi, mj, cov in d1]
    print()

    # D2
    print("─" * 86)
    print("D2 — PER-DIMENSION BOOST (mean C_dim − mean B_dim, raw 1-10)")
    print("─" * 86)
    print(f"{'Model':<42} {'O':>6} {'F':>6} {'C':>6} {'I':>6} {'S':>6}  {'Σ(w)':>6}")
    print(f"{'(weights →)':<42} {'×2':>6} {'×1':>6} {'×.5':>6} {'×1.5':>6} {'×.5':>6}")
    d2 = diag2_per_dim(per_dim)
    for row in d2:
        cells = []
        for d in DIMS:
            v = row["per_dim"].get(d)
            cells.append(f"{v:+6.2f}" if v is not None else "   n/a")
        print(f"  {row['model']:<40} " + " ".join(cells)
              + f"  {row['total_weighted_boost']:+6.2f}")
    print()
    print("  Σ(w) = weighted-sum boost predicted from per-dim deltas.")
    out["d2_per_dim"] = d2
    print()

    # D3
    print("─" * 86)
    print("D3 — TOOL-CALL COUNT vs BOOST")
    print("─" * 86)
    print(f"{'Model':<42} {'#calls':>7} {'B':>6} {'C':>6} {'Boost':>7} {'Np':>3}")
    d3, rho = diag3_calls_vs_boost(active, agg)
    for m, mc, b, c, bo, n in d3:
        print(f"  {m:<40} {mc:>7.2f} {b:>6.2f} {c:>6.2f} {bo:>+7.2f} {n:>3}")
    print()
    if rho is not None:
        print(f"  Spearman( mean #calls , boost ) = {rho:+.3f}  (n_models = {len(d3)})")
        if abs(rho) < 0.2:
            print("  → essentially no monotonic relationship.")
    out["d3_calls_vs_boost"] = {"rows": d3, "spearman": rho}
    print()

    # D4
    print("─" * 86)
    print("D4 — TOPIC ALIGNMENT (target-title token coverage by visited titles)")
    print("─" * 86)
    print(f"{'Model':<42} {'Np':>3} {'TopicCov':>9}  {'tgt#':>5} {'vis#':>5}")
    d4 = diag4_topic_alignment(active)
    for m, n, ov, tn, vn in d4:
        print(f"  {m:<40} {n:>3} {ov*100:>8.1f}%  {tn:>5.1f} {vn:>5.1f}")
    print()
    print("  TopicCov = mean( |target_title_tokens ∩ visited_title_tokens| / |target_title_tokens| )")
    print("  tokens = lowercase words ≥4 chars, stopwords removed.")
    print("  ⚠ Active mode prompt never reveals target title — high TopicCov means the agent")
    print("     independently searched into the correct subarea from domain alone.")
    out["d4_topic_alignment"] = [{"model": m, "n_papers": n, "topic_coverage": ov,
                                   "mean_tgt_tokens": tn, "mean_visited_tokens": vn}
                                  for m, n, ov, tn, vn in d4]
    print()

    OUT_JSON.write_text(json.dumps(out, indent=2, default=str))
    print(f"Dumped: {OUT_JSON.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
