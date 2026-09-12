"""E8 — Embedding analysis: is Static generation just restating the seed?

Hypothesis (plan E8): Static mode (Track B, given the paper's refs) may largely
RESTATE the seed paper, while Active mode (Track C, only a domain name) explores
more freely. If so, Static ideas should sit CLOSER (higher cosine) to the seed
(the paper's ground-truth hypothesis) than Active ideas.

Method (fully offline — sentence-transformers, NO API, no rate limit):
  - seed text   = papers.gt_hypothesis (the real paper's hypothesis)
  - Static idea = results.idea_text where track='B'
  - Active idea = results.idea_text where track='C'
    (prompt_version='v1_paper_refs', critic_model='' i.e. the generation rows)
  - Embed everything with all-MiniLM-L6-v2.
  - cos(Static_idea, seed) vs cos(Active_idea, seed).
  - Paired by (idea_model, paper) for the 16 models that have BOTH B and C, so
    model + paper are controlled. Sign test on the paired means.
  - Plot: (a) cosine-to-seed distributions Static vs Active; (b) 2D t-SNE of
    seed / static / active points.

Writes (NEVER touches results / papers DB — read only):
  reports/e8_embedding_static_vs_seed.json
  reports/figures/e8_cosine_static_vs_active.png
  reports/figures/e8_tsne_seed_static_active.png

Usage:
  python experiments/e8_embedding_static_vs_seed.py
  python experiments/e8_embedding_static_vs_seed.py --model all-MiniLM-L6-v2
"""
import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import config as cfg

PROMPT_VERSION = "v1_paper_refs"


def _cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def _sign_test(diffs):
    """Two-sided sign test p-value for diffs != 0 (Static_cos - Active_cos)."""
    from scipy import stats
    pos = sum(1 for d in diffs if d > 0)
    neg = sum(1 for d in diffs if d < 0)
    n = pos + neg
    if n == 0:
        return {"n": 0, "pos": 0, "neg": 0, "p": None}
    k = min(pos, neg)
    p = 2 * stats.binom.cdf(k, n, 0.5)
    return {"n": n, "pos": pos, "neg": neg, "p": float(min(p, 1.0))}


def load_data():
    cp = sqlite3.connect(str(cfg.PAPERS_DB)); cp.row_factory = sqlite3.Row
    seeds = {r["paper_id"]: (r["gt_hypothesis"] or "").strip()
             for r in cp.execute(
                 "SELECT paper_id, gt_hypothesis FROM papers "
                 "WHERE status='filtered' AND gt_hypothesis IS NOT NULL AND gt_hypothesis!=''")}
    cp.close()

    cr = sqlite3.connect(str(cfg.RESULTS_DB)); cr.row_factory = sqlite3.Row
    rows = cr.execute(
        "SELECT paper_id, idea_model, track, idea_index, idea_text FROM results "
        "WHERE prompt_version=? AND critic_model='' AND track IN ('B','C') "
        "  AND idea_text IS NOT NULL AND idea_text!=''", (PROMPT_VERSION,)).fetchall()
    cr.close()
    ideas = [dict(r) for r in rows if r["paper_id"] in seeds]
    return seeds, ideas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all-MiniLM-L6-v2")
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer
    print(f"loading embedder: {args.model}")
    enc = SentenceTransformer(args.model)

    seeds, ideas = load_data()
    print(f"seeds: {len(seeds)} | idea rows (B+C): {len(ideas)}")

    # Embed seeds
    seed_ids = sorted(seeds.keys())
    seed_vecs = enc.encode([seeds[p] for p in seed_ids], show_progress_bar=True,
                           normalize_embeddings=False, batch_size=64)
    seed_emb = {p: v for p, v in zip(seed_ids, seed_vecs)}

    # Embed ideas
    texts = [it["idea_text"] for it in ideas]
    idea_vecs = enc.encode(texts, show_progress_bar=True,
                           normalize_embeddings=False, batch_size=64)

    # cos(idea, its paper's seed)
    for it, v in zip(ideas, idea_vecs):
        it["_vec"] = v
        it["cos_seed"] = _cos(v, seed_emb[it["paper_id"]])

    # ---- Overall distributions ----
    cos_B = [it["cos_seed"] for it in ideas if it["track"] == "B"]
    cos_C = [it["cos_seed"] for it in ideas if it["track"] == "C"]

    def summ(xs):
        a = np.array(xs)
        return {"n": len(xs), "mean": float(a.mean()), "median": float(np.median(a)),
                "std": float(a.std(ddof=1)) if len(xs) > 1 else 0.0,
                "min": float(a.min()), "max": float(a.max())} if xs else {"n": 0}

    # ---- Paired by (model, paper): mean cos over ideas, for models with both B & C ----
    by_mp = defaultdict(lambda: {"B": [], "C": []})
    for it in ideas:
        by_mp[(it["idea_model"], it["paper_id"])][it["track"]].append(it["cos_seed"])
    paired = []
    for (m, p), d in by_mp.items():
        if d["B"] and d["C"]:
            paired.append({"model": m, "paper_id": p,
                           "cos_B": float(np.mean(d["B"])),
                           "cos_C": float(np.mean(d["C"]))})
    diffs = [r["cos_B"] - r["cos_C"] for r in paired]   # >0 means Static closer to seed
    sign = _sign_test(diffs)

    # Per-model paired means
    per_model = defaultdict(lambda: {"B": [], "C": []})
    for r in paired:
        per_model[r["model"]]["B"].append(r["cos_B"])
        per_model[r["model"]]["C"].append(r["cos_C"])
    per_model_out = {}
    for m, d in sorted(per_model.items()):
        per_model_out[m] = {
            "n_papers": len(d["B"]),
            "mean_cos_static": round(float(np.mean(d["B"])), 4),
            "mean_cos_active": round(float(np.mean(d["C"])), 4),
            "static_minus_active": round(float(np.mean(d["B"]) - np.mean(d["C"])), 4),
        }

    result = {
        "embedder": args.model,
        "prompt_version": PROMPT_VERSION,
        "overall": {
            "static_vs_seed": summ(cos_B),
            "active_vs_seed": summ(cos_C),
            "static_minus_active_mean": (round(np.mean(cos_B) - np.mean(cos_C), 4)
                                         if cos_B and cos_C else None),
        },
        "paired_by_model_paper": {
            "n_pairs": len(paired),
            "mean_diff_static_minus_active": round(float(np.mean(diffs)), 4) if diffs else None,
            "sign_test": sign,
        },
        "per_model": per_model_out,
    }

    out_dir = ROOT / "reports"; (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    (out_dir / "e8_embedding_static_vs_seed.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result["overall"], indent=2))
    print(json.dumps(result["paired_by_model_paper"], indent=2))

    # ---- Plots ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    # LEFT: density-normalised (n differs: ~40 models have Static, only 16 ran Active,
    # so raw counts are 3438 vs 318 — density makes the SHAPES comparable).
    ax[0].hist(cos_B, bins=30, density=True, alpha=0.55,
               label=f"Static→seed  (mean={np.mean(cos_B):.3f}, n={len(cos_B)})", color="#1f77b4")
    ax[0].hist(cos_C, bins=30, density=True, alpha=0.55,
               label=f"Active→seed  (mean={np.mean(cos_C):.3f}, n={len(cos_C)})", color="#d62728")
    ax[0].axvline(np.mean(cos_B), color="#1f77b4", ls="--", lw=2)
    ax[0].axvline(np.mean(cos_C), color="#d62728", ls="--", lw=2)
    ax[0].set_xlabel("cosine similarity of generated idea to the REAL hypothesis (seed)")
    ax[0].set_ylabel("density (area = 1, so n-difference removed)")
    ax[0].set_title("(A) Distribution of idea→real-hypothesis similarity\n"
                    "Active (red) sits slightly RIGHT of Static → closer to the real hypothesis")
    ax[0].legend(fontsize=9)
    # RIGHT: paired scatter — controls the n imbalance by pairing same model×paper
    if paired:
        xs = [r["cos_C"] for r in paired]; ys = [r["cos_B"] for r in paired]
        below = sum(1 for x, y in zip(xs, ys) if x > y)   # Active closer
        ax[1].scatter(xs, ys, s=16, alpha=0.5, c="#6a51a3")
        lim = [min(xs + ys) - 0.02, max(xs + ys) + 0.02]
        ax[1].plot(lim, lim, "k--", lw=1.2)
        ax[1].set_xlim(lim); ax[1].set_ylim(lim)
        ax[1].set_xlabel("Active idea → seed cosine")
        ax[1].set_ylabel("Static idea → seed cosine")
        ax[1].annotate("points BELOW the line\n= Active closer to real hypothesis",
                       (0.97, 0.06), xycoords="axes fraction", ha="right", fontsize=9, color="#6a51a3")
        ax[1].set_title(f"(B) Paired by same model×paper (n={len(paired)})\n"
                        f"{below}/{len(paired)} below diagonal → Active closer  (sign test p={sign['p']:.1e})")
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "e8_cosine_static_vs_active.png", dpi=140)
    print("saved figures/e8_cosine_static_vs_active.png")

    # t-SNE of seed/static/active
    try:
        from sklearn.manifold import TSNE
        sample_B = [it for it in ideas if it["track"] == "B"]
        sample_C = [it for it in ideas if it["track"] == "C"]
        # cap for speed
        rng = np.random.default_rng(42)
        def cap(lst, n):
            if len(lst) <= n: return lst
            idx = rng.choice(len(lst), n, replace=False)
            return [lst[i] for i in idx]
        sample_B = cap(sample_B, 400); sample_C = cap(sample_C, 400)
        X = np.vstack([np.array(seed_vecs)] +
                      [it["_vec"] for it in sample_B] +
                      [it["_vec"] for it in sample_C])
        labels = (["seed"] * len(seed_vecs) +
                  ["static"] * len(sample_B) + ["active"] * len(sample_C))
        emb2d = TSNE(n_components=2, random_state=42, init="pca",
                     perplexity=min(30, max(5, len(X) // 4))).fit_transform(X)
        fig2, ax2 = plt.subplots(figsize=(7, 6))
        colors = {"seed": "#2ca02c", "static": "#1f77b4", "active": "#d62728"}
        for lab in ["static", "active", "seed"]:
            m = [i for i, l in enumerate(labels) if l == lab]
            ax2.scatter(emb2d[m, 0], emb2d[m, 1], s=(28 if lab == "seed" else 10),
                        alpha=(0.9 if lab == "seed" else 0.4),
                        c=colors[lab], label=f"{lab} (n={len(m)})",
                        marker=("*" if lab == "seed" else "o"))
        ax2.legend(); ax2.set_title("t-SNE: seed vs Static vs Active ideas")
        ax2.set_xticks([]); ax2.set_yticks([])
        fig2.tight_layout()
        fig2.savefig(out_dir / "figures" / "e8_tsne_seed_static_active.png", dpi=140)
        print("saved figures/e8_tsne_seed_static_active.png")
    except Exception as e:
        print(f"t-SNE skipped: {e}")

    print("\nDONE E8")


if __name__ == "__main__":
    main()
