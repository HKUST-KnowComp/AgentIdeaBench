"""E28 — Within-cell idea diversity: Active vs Static (天石 2026-07-19).

Question: the metric critique showed centroid cosine cannot measure exploration
diversity. Using HONEST metrics instead, is the *idea* diversity of Active higher
than Static? Each (idea_model, subdomain, track) cell has 3 independently
generated hypotheses (idea_index 0-2) — we measure diversity WITHIN the 3 ideas
of a cell, then compare Active vs Static paired on (model, subdomain).

Metrics per cell (3 ideas):
  - vendi      : Vendi Score of the 3x3 cosine kernel (effective # of distinct
                 ideas, range 1..3)                                   [headline]
  - cosdist    : mean pairwise cosine distance (1 - cos), reference only —
                 embedding anisotropy compresses its range (see metric critique)
  - rougeL     : mean pairwise ROUGE-L F1 (LCS-based, pure python); LOWER = more
                 diverse (lexical overlap)
  - distinct2  : unique bigrams / total bigrams over the 3 ideas; higher = more
                 diverse

Embeddings: local sentence-transformers all-MiniLM-L6-v2 (same family as E8 and
the metric critique; offline, no API). Cached to reports/e28_idea_emb.npz.

Stats: paired Active-Static delta per (model, sub) -> overall mean delta,
Wilcoxon signed-rank, 2000-bootstrap 95% CI, win rate; per-model means; and the
correlation of per-model diversity gain with Static capability (ties to F2's
capability-gating). Static capability = model's mean weighted lit8d score on
Track B from lit8d_scores_3seed (dim-wise median of 3 critics, weights
O2/F1/C0.5/I1.5/S0.5 / 5.5).

Read-only on DB; writes reports/e28_idea_diversity.json (+ emb cache).
Usage: /usr/bin/python3 experiments/e28_idea_diversity.py
"""
import json
import re
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import config as cfg

OUT_JSON = ROOT / "reports" / "e28_idea_diversity.json"
EMB_CACHE = ROOT / "reports" / "e28_idea_emb.npz"

DIM_W = {"originality": 2.0, "feasibility": 1.0, "clarity": 0.5,
         "impact": 1.5, "specificity": 0.5}
W_SUM = sum(DIM_W.values())


# ---------------------------------------------------------------- text metrics
def _tok(t):
    return re.findall(r"[a-z0-9]+", t.lower())


def _lcs(a, b):
    """LCS length via DP on words."""
    n, m = len(a), len(b)
    if not n or not m:
        return 0
    prev = [0] * (m + 1)
    for i in range(1, n + 1):
        cur = [0] * (m + 1)
        ai = a[i - 1]
        for j in range(1, m + 1):
            cur[j] = prev[j - 1] + 1 if ai == b[j - 1] else max(prev[j], cur[j - 1])
        prev = cur
    return prev[m]


def rouge_l_f1(a, b):
    ta, tb = _tok(a), _tok(b)
    l = _lcs(ta, tb)
    if not l:
        return 0.0
    p, r = l / len(tb), l / len(ta)
    return 2 * p * r / (p + r)


def distinct_n(texts, n=2):
    grams = []
    for t in texts:
        tk = _tok(t)
        grams += [tuple(tk[i:i + n]) for i in range(len(tk) - n + 1)]
    return len(set(grams)) / len(grams) if grams else 0.0


def vendi_score(E):
    """E: (k, d) L2-normalized. Vendi = exp(entropy of eigvals of K/k)."""
    K = E @ E.T
    lam = np.linalg.eigvalsh(K / E.shape[0])
    lam = np.clip(lam, 0, None)
    s = lam.sum()
    if s <= 0:
        return 1.0
    lam = lam / s
    ent = -(lam[lam > 1e-12] * np.log(lam[lam > 1e-12])).sum()
    return float(np.exp(ent))


def cell_metrics(texts, E):
    pairs = [(0, 1), (0, 2), (1, 2)]
    cos = [float(E[i] @ E[j]) for i, j in pairs]
    return {
        "vendi": vendi_score(E),
        "cosdist": float(np.mean([1 - c for c in cos])),
        "rougeL": float(np.mean([rouge_l_f1(texts[i], texts[j]) for i, j in pairs])),
        "distinct2": distinct_n(texts, 2),
    }


# ---------------------------------------------------------------- data
def load_cells():
    conn = sqlite3.connect(str(cfg.RESULTS_DB))
    rows = conn.execute(
        "SELECT idea_model, subdomain, track, idea_index, idea_text FROM subdomain_ideas "
        "WHERE TRIM(idea_text)!='' AND track IN ('B','C') ORDER BY idea_model, subdomain, track, idea_index"
    ).fetchall()
    conn.close()
    cells = defaultdict(dict)
    for m, s, t, i, txt in rows:
        cells[(m, s, t)][i] = txt
    # keep cells with >=3 ideas; take the first 3 by idea_index (indices are 1,2,3 in this DB)
    out = {}
    for k, d in cells.items():
        if len(d) >= 3:
            out[k] = [d[i] for i in sorted(d)[:3]]
    return out


def static_capability():
    """Per-model mean weighted lit8d score on Track B (dim-median over 3 critics)."""
    conn = sqlite3.connect(str(cfg.RESULTS_DB))
    rows = conn.execute(
        "SELECT idea_model, item_id, scores_json FROM lit8d_scores_3seed "
        "WHERE track='B' AND scores_json IS NOT NULL AND error IS NULL").fetchall()
    conn.close()
    per_item = defaultdict(lambda: defaultdict(list))  # (model,item) -> dim -> [3 critic scores]
    for m, iid, sj in rows:
        try:
            s = json.loads(sj)
        except Exception:
            continue
        for d in DIM_W:
            v = s.get(d, {}).get("score") if isinstance(s.get(d), dict) else s.get(d)
            if v is not None:
                per_item[(m, iid)][d].append(float(v))
    per_model = defaultdict(list)
    for (m, _), dims in per_item.items():
        tot, wt = 0.0, 0.0
        for d, w in DIM_W.items():
            if dims.get(d):
                tot += w * float(np.median(dims[d])); wt += w
        if wt > 0:
            per_model[m].append(tot / wt * (W_SUM / W_SUM))
    return {m: float(np.mean(v)) for m, v in per_model.items() if v}


# ---------------------------------------------------------------- main
def main():
    cells = load_cells()
    keys = sorted(cells)
    print(f"{len(keys)} cells with 3 ideas", flush=True)

    # ---- embeddings (cached)
    texts, tkey = [], []
    for k in keys:
        for i, t in enumerate(cells[k]):
            texts.append(t); tkey.append((k[0], k[1], k[2], i))
    if EMB_CACHE.exists():
        z = np.load(EMB_CACHE, allow_pickle=True)
        emb, cached_key = z["emb"], [tuple(x) for x in z["key"]]
        if cached_key == [(a, b, c, str(d)) for a, b, c, d in tkey] or len(cached_key) == len(tkey):
            print("using cached embeddings", flush=True)
        else:
            emb = None
    else:
        emb = None
    if emb is None or len(emb) != len(texts):
        from sentence_transformers import SentenceTransformer
        st = SentenceTransformer("all-MiniLM-L6-v2")
        emb = st.encode(texts, batch_size=256, show_progress_bar=True,
                        normalize_embeddings=True)
        np.savez_compressed(EMB_CACHE, emb=emb,
                            key=np.array([(a, b, c, str(d)) for a, b, c, d in tkey], dtype=object))
        print("embedded + cached", flush=True)
    emb = np.asarray(emb, dtype=np.float32)

    # ---- per-cell metrics
    idx = {}
    for row, k4 in enumerate(tkey):
        idx[k4] = row
    per_cell = {}
    for k in keys:
        m, s, t = k
        E = np.stack([emb[idx[(m, s, t, i)]] for i in range(3)])
        E = E / np.linalg.norm(E, axis=1, keepdims=True)
        per_cell[k] = cell_metrics(cells[k], E)

    # ---- pair B vs C
    metrics = ["vendi", "cosdist", "rougeL", "distinct2"]
    pairs = []
    for (m, s, t) in keys:
        if t != "B":
            continue
        kc = (m, s, "C")
        if kc in per_cell:
            pairs.append((m, s, per_cell[(m, s, "B")], per_cell[kc]))
    print(f"{len(pairs)} paired (model, subdomain)", flush=True)

    from scipy.stats import wilcoxon, pearsonr, spearmanr
    rng = np.random.default_rng(42)
    out = {"n_pairs": len(pairs), "metrics": {}}
    deltas_by_metric = {}
    for met in metrics:
        b = np.array([pb[met] for _, _, pb, pc in pairs])
        c = np.array([pc[met] for _, _, pb, pc in pairs])
        d = c - b
        deltas_by_metric[met] = d
        boot = [float(rng.choice(d, len(d), replace=True).mean()) for _ in range(2000)]
        try:
            p = float(wilcoxon(d).pvalue)
        except Exception:
            p = None
        out["metrics"][met] = {
            "static_mean": round(float(b.mean()), 4),
            "active_mean": round(float(c.mean()), 4),
            "delta_mean": round(float(d.mean()), 4),
            "delta_win_rate": round(float((d > 0).mean()), 4),
            "wilcoxon_p": (round(p, 6) if p is not None else None),
            "delta_ci95": [round(float(np.percentile(boot, 2.5)), 4),
                           round(float(np.percentile(boot, 97.5)), 4)],
        }

    # ---- per-model
    per_model = defaultdict(lambda: defaultdict(list))
    for m, s, pb, pc in pairs:
        for met in metrics:
            per_model[m][met].append(pc[met] - pb[met])
        per_model[m]["static_vendi"].append(pb["vendi"])
        per_model[m]["active_vendi"].append(pc["vendi"])
    out["per_model"] = {
        m: {met: round(float(np.mean(v[met])), 4)
            for met in list(metrics) + ["static_vendi", "active_vendi"]}
        for m, v in sorted(per_model.items())
    }

    # ---- diversity gain vs static capability (F2 tie-in)
    cap = static_capability()
    mm = [m for m in out["per_model"] if m in cap]
    if len(mm) >= 5:
        x = np.array([cap[m] for m in mm])
        y = np.array([out["per_model"][m]["vendi"] for m in mm])
        pr, pp = pearsonr(x, y)
        sr, sp = spearmanr(x, y)
        out["capability_vs_vendi_gain"] = {
            "n_models": len(mm),
            "pearson_r": round(float(pr), 4), "pearson_p": round(float(pp), 6),
            "spearman_rho": round(float(sr), 4), "spearman_p": round(float(sp), 6),
            "static_capability": {m: round(cap[m], 4) for m in mm},
        }

    OUT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in out["metrics"].items()}, indent=2))
    if "capability_vs_vendi_gain" in out:
        g = out["capability_vs_vendi_gain"]
        print(f"capability vs vendi-gain: r={g['pearson_r']} (p={g['pearson_p']}), "
              f"rho={g['spearman_rho']} (p={g['spearman_p']}), n={g['n_models']}")
    print("wrote", OUT_JSON)


if __name__ == "__main__":
    main()
