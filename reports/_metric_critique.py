"""Metric-critique experiment: demonstrate WHY mean-centroid cosine similarity is a
poor metric for measuring how differently two literature searches explored a topic.

Empirical puzzle: STATIC vs ACTIVE reference sets have Jaccard overlap ~0.03 (nearly
disjoint papers) yet centroid-cosine similarity 0.87. This script shows the 0.87 is
essentially the FLOOR of the metric on this data — i.e. it carries almost no
information beyond "same subfield" — while honest set-membership measures (Jaccard,
mutual-kNN overlap, energy distance on the point clouds) do discriminate.

Read-only. Reuses cached embeddings in v3_arrays.npz (no re-encoding, no DB writes).
/usr/bin/python3.

Definitions (all embeddings L2-normalized rows of rmat):
  centroid-cosine(A,B) = cos( mean(A), mean(B) )   [the metric under critique]
  jaccard(A,B)         = |A∩B| / |A∪B|              [honest paper overlap]
  mutual-kNN(A,B)      = mean fraction of each paper's k nearest neighbors (within A∪B)
                         that fall in the OTHER set  [set-interleaving; 0=separated,~0.5=mixed]
  energy-dist(A,B)     = 2*E|a-b| - E|a-a'| - E|b-b'|  on cosine distance [distribution gap]
"""
import json, re, sqlite3, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import config as cfg

RNG = np.random.default_rng(20260713)
_PID = re.compile(r'"paperId":\s*"([0-9a-f]{40})"')

Z = np.load(ROOT / "reports" / "e11_ref_overlap" / "v3_arrays.npz", allow_pickle=True)
RMAT = Z["rmat"].astype(np.float32)
# L2-normalize once so dot == cosine
_n = np.linalg.norm(RMAT, axis=1, keepdims=True)
RMAT = RMAT / np.clip(_n, 1e-9, None)
IDX = {rid: i for i, rid in enumerate(Z["ref_ids"])}


def rows(ids):
    return [IDX[r] for r in ids if r in IDX]


def centroid_cos(ra, rb):
    ca = RMAT[ra].mean(0); cb = RMAT[rb].mean(0)
    na = np.linalg.norm(ca); nb = np.linalg.norm(cb)
    if na == 0 or nb == 0:
        return None
    return float((ca @ cb) / (na * nb))


def energy_dist(ra, rb):
    A = RMAT[ra]; B = RMAT[rb]
    # cosine distance = 1 - dot (rows are unit-norm)
    dab = 1 - A @ B.T
    daa = 1 - A @ A.T
    dbb = 1 - B @ B.T
    return float(2 * dab.mean() - daa.mean() - dbb.mean())


def vendi(rr):
    """Vendi Score = effective number of distinct items = exp(entropy of eigenvalues
    of the normalized (cosine) similarity matrix K/n). VS in [1, n]; higher = more
    diverse. Friedman & Dieng 2023."""
    R = RMAT[rr]
    n = len(rr)
    if n < 2:
        return float(n)
    K = (R @ R.T) / n
    w = np.linalg.eigvalsh(K)
    w = w[w > 1e-12]
    w = w / w.sum()
    return float(np.exp(-(w * np.log(w)).sum()))


def mutual_knn(ra, rb, k=5):
    A = np.array(ra); B = np.array(rb)
    U = np.concatenate([A, B])
    lab = np.concatenate([np.zeros(len(A)), np.ones(len(B))])
    M = RMAT[U] @ RMAT[U].T
    np.fill_diagonal(M, -np.inf)
    kk = min(k, len(U) - 1)
    nn = np.argpartition(-M, kk - 1, axis=1)[:, :kk]
    frac = (lab[nn] != lab[:, None]).mean(axis=1)  # fraction of nbrs in the OTHER set
    return float(frac.mean())


def load_sets():
    conn = sqlite3.connect(str(cfg.RESULTS_DB)); conn.row_factory = sqlite3.Row
    sref = {}
    for r in conn.execute("SELECT subdomain, refs_json FROM subdomain_refs"):
        sref[r["subdomain"]] = [x["paperId"] for x in json.loads(r["refs_json"])
                                if isinstance(x, dict) and x.get("paperId")]
    pairs = []           # (subdomain, static_ids, active_ids) observed
    pool = defaultdict(set)  # subdomain -> union of all pids seen (topic universe)
    for pid in sref:
        pool[pid].update(sref[pid])
    for r in conn.execute("SELECT idea_model,subdomain,telemetry FROM subdomain_ideas "
                          "WHERE track='C' AND TRIM(idea_text)!=''"):
        aids = set()
        if r["telemetry"]:
            try:
                tel = json.loads(r["telemetry"])
                for t in tel.get("trace", []):
                    rp = (t.get("result_preview", "") or "") if isinstance(t, dict) else ""
                    try:
                        items = json.loads(rp)
                        if isinstance(items, list):
                            for it in items:
                                if isinstance(it, dict) and it.get("paperId"):
                                    aids.add(it["paperId"])
                            continue
                    except Exception:
                        pass
                    aids.update(_PID.findall(rp))
            except Exception:
                pass
        sub = r["subdomain"]
        pool[sub].update(aids)
        sids = sref.get(sub, [])
        if sids and aids:
            pairs.append((sub, sids, list(aids)))
    conn.close()
    return sref, pairs, pool


def jaccard(a, b):
    a, b = set(a), set(b)
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def main():
    sref, pairs, pool = load_sets()
    # keep only pooled subdomains with enough embedded papers for disjoint splits
    poolrows = {s: rows(list(ids)) for s, ids in pool.items()}
    big = {s: r for s, r in poolrows.items() if len(r) >= 30}
    print(f"pairs (observed static/active): {len(pairs)}  |  subdomains with >=30 embedded refs: {len(big)}")

    out = {"n_observed_pairs": len(pairs)}

    # --- (A) observed static-vs-active: centroid cosine + jaccard on a sample ---
    samp = pairs if len(pairs) <= 1500 else [pairs[i] for i in RNG.choice(len(pairs), 1500, replace=False)]
    obs_cos, obs_jac = [], []
    for sub, s, a in samp:
        rs, ra = rows(s), rows(a)
        if len(rs) >= 3 and len(ra) >= 3:
            cc = centroid_cos(rs, ra)
            if cc is not None:
                obs_cos.append(cc); obs_jac.append(jaccard(s, a))
    out["observed"] = {"centroid_cos_mean": round(float(np.mean(obs_cos)), 4),
                       "jaccard_mean": round(float(np.mean(obs_jac)), 4),
                       "n": len(obs_cos)}

    # --- (B) same-topic FLOOR: random disjoint split of the same subdomain pool ---
    floor = []
    for s, r in big.items():
        for _ in range(8):
            perm = RNG.permutation(r)
            h = len(perm) // 2
            m = min(h, 15)
            A, B = perm[:m], perm[h:h + m]
            cc = centroid_cos(A, B)
            if cc is not None:
                floor.append(cc)
    out["same_topic_floor"] = {"centroid_cos_mean": round(float(np.mean(floor)), 4),
                               "centroid_cos_std": round(float(np.std(floor)), 4),
                               "p5": round(float(np.percentile(floor, 5)), 4),
                               "p95": round(float(np.percentile(floor, 95)), 4),
                               "n": len(floor)}

    # --- (C) cross-topic reference: sets from DIFFERENT subdomains ---
    subs = list(big.keys())
    cross = []
    for _ in range(1500):
        i, j = RNG.choice(len(subs), 2, replace=False)
        A = RNG.permutation(big[subs[i]])[:15]
        B = RNG.permutation(big[subs[j]])[:15]
        cc = centroid_cos(A, B)
        if cc is not None:
            cross.append(cc)
    out["cross_topic"] = {"centroid_cos_mean": round(float(np.mean(cross)), 4),
                          "p5": round(float(np.percentile(cross, 5)), 4),
                          "p95": round(float(np.percentile(cross, 95)), 4),
                          "n": len(cross)}

    # --- (D) sensitivity curve: centroid cos & jaccard vs swap fraction ---
    fracs = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    curve = {f: {"cos": [], "jac": []} for f in fracs}
    for s, r in big.items():
        if len(r) < 40:
            continue
        base = RNG.permutation(r)[:15]
        rest = [x for x in r if x not in set(base.tolist())]
        for f in fracs:
            k = int(round(f * len(base)))
            keep = base[k:]
            repl = RNG.permutation(rest)[:k]
            new = np.concatenate([keep, repl]).astype(int)
            cc = centroid_cos(base, new)
            if cc is not None:
                curve[f]["cos"].append(cc)
                curve[f]["jac"].append(jaccard(base.tolist(), new.tolist()))
    out["sensitivity"] = {str(f): {"cos_mean": round(float(np.mean(v["cos"])), 4),
                                   "jac_mean": round(float(np.mean(v["jac"])), 4),
                                   "n": len(v["cos"])} for f, v in curve.items()}

    # --- (E) alt metrics: do they separate observed vs same-topic-random split? ---
    # observed pairs
    ed_obs, kn_obs, ed_flo, kn_flo = [], [], [], []
    ssub = samp if len(samp) <= 400 else [samp[i] for i in RNG.choice(len(samp), 400, replace=False)]
    for sub, s, a in ssub:
        rs, ra = rows(s), rows(a)
        if len(rs) >= 5 and len(ra) >= 5:
            ed_obs.append(energy_dist(rs, ra)); kn_obs.append(mutual_knn(rs, ra))
    for s, r in list(big.items()):
        perm = RNG.permutation(r); h = len(perm) // 2; m = min(h, 15)
        A, B = perm[:m], perm[h:h + m]
        if len(A) >= 5 and len(B) >= 5:
            ed_flo.append(energy_dist(A, B)); kn_flo.append(mutual_knn(A, B))
    out["alt_metrics"] = {
        "energy_dist": {"observed_static_vs_active": round(float(np.mean(ed_obs)), 4),
                        "same_topic_random_split": round(float(np.mean(ed_flo)), 4)},
        "mutual_knn_other_frac": {"observed_static_vs_active": round(float(np.mean(kn_obs)), 4),
                                  "same_topic_random_split": round(float(np.mean(kn_flo)), 4),
                                  "note": "0.5 = fully interleaved point clouds"},
    }

    # --- (F) Vendi diversity: does the union hold MORE distinct content than either set? ---
    vs_s, vs_a, vs_u, add = [], [], [], []
    for sub, s, a in ssub:
        rs, ra = rows(s), rows(a)
        if len(rs) >= 3 and len(ra) >= 3:
            vs, va = vendi(rs), vendi(ra)
            vu = vendi(list(dict.fromkeys(rs + ra)))  # union, dedup rows
            vs_s.append(vs); vs_a.append(va); vs_u.append(vu)
            add.append(vu / max(vs, va))  # union effective-count vs the larger single set
    out["vendi"] = {"static": round(float(np.mean(vs_s)), 3),
                    "active": round(float(np.mean(vs_a)), 3),
                    "union": round(float(np.mean(vs_u)), 3),
                    "union_over_max_single": round(float(np.mean(add)), 3),
                    "note": "union effective-distinct-count > either alone => the two searches add genuinely new content despite 0.87 centroid cosine",
                    "n": len(vs_s)}

    (ROOT / "reports" / "e11_ref_overlap" / "metric_critique.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
