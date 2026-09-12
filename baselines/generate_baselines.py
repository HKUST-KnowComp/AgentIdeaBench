"""
Baseline Generators for SciSynthBench

Three baselines that produce idea_text entries in the same format as real
idea models, to be scored by the same Phase 3 pipeline.

  1. random    — shuffled domain keywords, no coherent hypothesis (local, no API)
  2. template  — fixed template filled with title/domain (local, no API)
  3. copy      — picks a reference paper's abstract, rewrites it into structured
                 proposal format via GPT-4o (tests whether judges can distinguish
                 "existing work repackaged as proposal" from genuinely novel ideas)

Usage:
    python baselines/generate_baselines.py --papers-db data/papers.db --results-db data/results.db
    python baselines/generate_baselines.py --papers-db data/papers.db --results-db data/results.db --smoke
"""

import argparse
import hashlib
import json
import logging
import random
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Baseline model identifiers (used as idea_model in results.db)
# ---------------------------------------------------------------------------

BASELINE_RANDOM   = "baseline/random"
BASELINE_TEMPLATE = "baseline/template"
BASELINE_COPY     = "baseline/copy"
BASELINE_GT       = "baseline/gt"

ALL_BASELINES = [BASELINE_RANDOM, BASELINE_TEMPLATE, BASELINE_COPY, BASELINE_GT]

# ---------------------------------------------------------------------------
# Domain keyword pools for random baseline
# ---------------------------------------------------------------------------

_DOMAIN_KEYWORDS = {
    "CS": [
        "neural network", "transformer", "attention mechanism", "gradient descent",
        "reinforcement learning", "knowledge graph", "embedding space", "tokenizer",
        "convolutional", "generative model", "latent space", "fine-tuning",
        "benchmark", "inference", "pre-training", "self-supervised",
    ],
    "Biology": [
        "gene expression", "protein folding", "CRISPR", "transcription factor",
        "cell signaling", "metabolic pathway", "RNA sequencing", "organoid",
        "epigenetic", "microbiome", "antibody", "mutation", "genome-wide",
        "single-cell", "chromatin", "phosphorylation",
    ],
    "Physics": [
        "quantum state", "entanglement", "superconductor", "photonic",
        "topological", "lattice", "coherence", "spin-orbit", "qubit",
        "condensed matter", "dark matter", "gravitational wave",
        "Bose-Einstein", "nanophotonic", "frequency comb", "perovskite",
    ],
    "Chemistry": [
        "catalysis", "molecular dynamics", "density functional", "electrolyte",
        "nanoparticle", "retrosynthesis", "metal-organic framework", "ligand binding",
        "free energy", "reaction mechanism", "selectivity", "quantum dot",
        "photocatalysis", "polymer", "biosensor", "enzyme",
    ],
    "Medicine": [
        "biomarker", "immunotherapy", "clinical trial", "patient stratification",
        "tumor microenvironment", "neurodegeneration", "metabolic syndrome",
        "drug delivery", "liquid biopsy", "T cell", "gene therapy",
        "fibrosis", "nociceptor", "wearable sensor", "ctDNA", "sepsis",
    ],
}

_FALLBACK_KEYWORDS = [
    "hypothesis", "mechanism", "system", "analysis", "framework",
    "interaction", "prediction", "model", "pathway", "function",
]


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

def _random_hypothesis(title: str, domain: str, seed: int) -> str:
    """Generate a random hypothesis from shuffled domain keywords."""
    rng = random.Random(seed)
    keywords = list(_DOMAIN_KEYWORDS.get(domain, _FALLBACK_KEYWORDS))
    rng.shuffle(keywords)
    picked = keywords[:rng.randint(6, 10)]

    connectors = [
        "We hypothesize that",
        "through the interaction of",
        "combined with",
        "leading to novel insights in",
        "which may reveal",
        "by leveraging",
        "resulting in improved understanding of",
        "and further investigating",
    ]
    rng.shuffle(connectors)

    parts = []
    for i, kw in enumerate(picked):
        if i < len(connectors):
            parts.append(f"{connectors[i]} {kw}")
        else:
            parts.append(kw)

    return " ".join(parts) + "."


def _template_hypothesis(title: str, domain: str) -> str:
    """Generate a template-based hypothesis in the active generation format."""
    import config as cfg
    is_structured = cfg.GENERATION_FORMAT == "structured"

    if not is_structured:
        return (
            f"Hypothesis: We hypothesize that a novel approach combining recent "
            f"advances in {domain} with targeted analysis will reveal previously "
            f"uncharacterized interactions. We propose to systematically "
            f"investigate the key factors using state-of-the-art computational "
            f"and experimental methodologies, validate predictions through "
            f"controlled experiments, and quantify the resulting effects with "
            f"appropriate statistical frameworks. We expect measurable effects "
            f"consistent with general trends observed in the broader literature. "
            f"This work has the potential to advance fundamental understanding "
            f"in {domain} and inform future research directions in related "
            f"disciplines."
        )

    return (
        f"## Hypothesis\n"
        f"We hypothesize that a novel approach combining recent advances in {domain} "
        f"with targeted analysis of the mechanisms underlying {title} will reveal "
        f"previously uncharacterized interactions.\n\n"
        f"## Experimental Design\n"
        f"We propose to systematically investigate the key factors and their "
        f"relationships using state-of-the-art computational and experimental "
        f"methodologies. Our approach leverages multi-scale analysis to identify "
        f"critical variables, validate predictions through controlled experiments, "
        f"and quantify the resulting effects with appropriate statistical frameworks.\n\n"
        f"## Expected Outcomes\n"
        f"We expect to observe significant effects that support our hypothesis. "
        f"Null results would prompt re-evaluation of the underlying assumptions.\n\n"
        f"## Contingency Plan\n"
        f"If the primary approach fails, alternative methodologies will be explored "
        f"and the hypothesis will be refined accordingly.\n\n"
        f"## Broader Implications\n"
        f"This work has the potential to advance fundamental understanding in "
        f"{domain} and inform future research directions in related disciplines.\n\n"
        f"## Assumption Audit\n"
        f"We assume standard conditions apply and that common methodological "
        f"assumptions hold for this investigation.\n\n"
        f"## Potential Confounds & Controls\n"
        f"Standard confounding variables will be controlled through established "
        f"experimental practices.\n\n"
        f"## Quantitative Predictions\n"
        f"We expect measurable effects consistent with general trends observed "
        f"in the broader literature."
    )


def _pick_ref_abstract(paper: dict, seed: int) -> str:
    """Pick one reference paper's abstract from ranked_refs_json.

    Returns the raw abstract text, or empty string if no refs available.
    """
    refs_json = paper.get("ranked_refs_json")  # same 5 refs Track B and judge see
    if not refs_json:
        return ""
    try:
        refs = json.loads(refs_json)
        with_abs = [r for r in refs if r.get("abstract")]
        if with_abs:
            rng = random.Random(seed)
            return rng.choice(with_abs)["abstract"].strip()
    except (json.JSONDecodeError, TypeError):
        pass
    return ""


def _rewrite_ref_as_proposal(abstract: str, domain: str) -> str:
    """Rewrite a reference abstract into the same format as real model output.

    Matches config.GENERATION_FORMAT so copy baseline is comparable:
      - paragraph mode: single 80-150 word hypothesis paragraph
      - structured mode: 8-section research proposal
    Tests whether judges can distinguish 'existing work repackaged' from novel ideas.
    """
    from utils.LLM import BaseLLM
    import config as cfg

    is_structured = cfg.GENERATION_FORMAT == "structured"

    if not abstract.strip():
        if is_structured:
            return (
                f"## Hypothesis\nWe propose to investigate aspects of {domain}.\n\n"
                f"## Experimental Design\nStandard methodology will be applied.\n\n"
                f"## Expected Outcomes\nResults are expected to be informative.\n\n"
                f"## Contingency Plan\nAlternative approaches will be considered.\n\n"
                f"## Broader Implications\nFindings may advance the field.\n\n"
                f"## Assumption Audit\nStandard assumptions apply.\n\n"
                f"## Potential Confounds & Controls\nTypical confounds will be addressed.\n\n"
                f"## Quantitative Predictions\nResults are expected to show measurable effects."
            )
        return f"Hypothesis: We propose to investigate aspects of {domain}."

    if is_structured:
        system = (
            "You are a scientific writing assistant. "
            "Rewrite the given abstract as a structured research proposal. "
            "Write in first-person future tense. "
            "Do NOT include specific numerical results or p-values. "
            "Output exactly eight sections with the headers shown."
        )
        prompt = (
            "Rewrite this abstract as a structured research proposal:\n\n"
            "## Hypothesis (80-120 words)\n"
            "## Experimental Design (120-200 words)\n"
            "## Expected Outcomes (60-100 words)\n"
            "## Contingency Plan (60-100 words)\n"
            "## Broader Implications (40-80 words)\n"
            "## Assumption Audit (80-120 words)\n"
            "## Potential Confounds & Controls (60-100 words)\n"
            "## Quantitative Predictions (60-100 words)\n\n"
            f"Abstract:\n{abstract}"
        )
    else:
        system = (
            "You are a scientific writing assistant. "
            "Rewrite the given abstract as a single research hypothesis paragraph. "
            "Write in first-person future tense ('We propose...', 'We hypothesize...'). "
            "Do NOT include specific numerical results or p-values. "
            "Output exactly one paragraph, 80-150 words, starting with 'Hypothesis: '."
        )
        prompt = (
            f"Rewrite this abstract as a single 80-150 word hypothesis paragraph, "
            f"beginning with 'Hypothesis: '.\n\n"
            f"Abstract:\n{abstract}"
        )

    try:
        llm = BaseLLM(model_name="openai/gpt-4o")
        response = llm.completion(prompt, system_prompt=system)
        if isinstance(response, tuple):
            return response[0].strip()
        return response.strip()
    except Exception as e:
        logger.warning(f"  copy baseline rewrite failed: {e}")
        if is_structured:
            return (
                f"## Hypothesis\n{abstract[:200]}\n\n"
                f"## Experimental Design\nMethodology as described above.\n\n"
                f"## Expected Outcomes\nResults consistent with the hypothesis.\n\n"
                f"## Contingency Plan\nAlternative methods will be explored.\n\n"
                f"## Broader Implications\nFindings will advance understanding in {domain}.\n\n"
                f"## Assumption Audit\nStandard assumptions apply.\n\n"
                f"## Potential Confounds & Controls\nTypical confounds will be addressed.\n\n"
                f"## Quantitative Predictions\nResults are expected to show measurable effects."
            )
        return f"Hypothesis: {abstract[:600]}"


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def generate_baselines(db_papers: str, db_results: str,
                       n_ideas: int = 3,
                       smoke: bool = False,
                       idea_model: str = None) -> dict:
    """Generate baseline ideas for all filtered papers.

    If idea_model is given, only generate that specific baseline.
    Otherwise generate all baselines (random, template, copy, gt).
    """
    conn_p = sqlite3.connect(db_papers, timeout=30)
    conn_p.row_factory = sqlite3.Row
    conn_r = sqlite3.connect(db_results, timeout=30)
    conn_r.execute("PRAGMA journal_mode=WAL")
    conn_r.execute("PRAGMA busy_timeout=30000")

    papers = [dict(r) for r in conn_p.execute(
        "SELECT * FROM papers WHERE status='filtered' "
        "AND gt_hypothesis IS NOT NULL AND gt_hypothesis != '' "
        "ORDER BY domain, paper_id"
    ).fetchall()]
    conn_p.close()

    if smoke:
        # 1 paper per domain for balanced coverage
        from collections import defaultdict
        by_domain = defaultdict(list)
        for p in papers:
            by_domain[p["domain"]].append(p)
        papers = [p for ps in by_domain.values() for p in ps[:1]]
        papers.sort(key=lambda p: (p["domain"], p["paper_id"]))

    stats = {"papers": len(papers), "inserted": 0}
    ts = datetime.now(timezone.utc).isoformat()
    cur = conn_r.cursor()

    # Cache for copy baseline: one GPT-4o rewrite per paper (not per idea)
    copy_cache: dict = {}

    for i, paper in enumerate(papers, 1):
        pid    = paper["paper_id"]
        title  = paper["title"]
        domain = paper["domain"]

        logger.info(f"  [{i}/{len(papers)}] {domain} | {title[:50]}")

        active_baselines = [idea_model] if idea_model else ALL_BASELINES
        for baseline_model in active_baselines:
            # Deterministic baselines only need 1 idea (duplicating wastes eval budget)
            # copy/gt rewrite exactly once; random/template are also fixed per paper
            effective_n = 1 if baseline_model in (BASELINE_COPY, BASELINE_GT, BASELINE_TEMPLATE) else n_ideas
            for track in ("B",):
                for idea_idx in range(1, effective_n + 1):
                    # Check if already exists
                    exists = cur.execute(
                        "SELECT 1 FROM results WHERE paper_id=? AND idea_model=? "
                        "AND track=? AND idea_index=? AND critic_model=''",
                        (pid, baseline_model, track, idea_idx)
                    ).fetchone()
                    if exists:
                        continue

                    # Deterministic seed per (paper, baseline, track, index)
                    seed_str = f"{pid}:{baseline_model}:{track}:{idea_idx}"
                    seed = int(hashlib.md5(seed_str.encode()).hexdigest(), 16) % (2**32)

                    if baseline_model == BASELINE_RANDOM:
                        idea_text = _random_hypothesis(title, domain, seed)
                    elif baseline_model == BASELINE_TEMPLATE:
                        idea_text = _template_hypothesis(title, domain)
                    elif baseline_model == BASELINE_COPY:
                        # One API call per paper, cached across tracks/indices
                        if pid not in copy_cache:
                            ref_abs = _pick_ref_abstract(paper, seed)
                            copy_cache[pid] = _rewrite_ref_as_proposal(ref_abs, domain)
                        idea_text = copy_cache[pid]
                    elif baseline_model == BASELINE_GT:
                        # Positive control: use the ground-truth hypothesis directly
                        gt = paper.get("gt_hypothesis") or ""
                        if not gt.strip():
                            continue
                        idea_text = gt
                    else:
                        continue

                    cur.execute("""
                        INSERT OR IGNORE INTO results
                          (paper_id, idea_model, track, idea_index, idea_text,
                           critic_model, created_at)
                        VALUES (?, ?, ?, ?, ?, '', ?)
                    """, (pid, baseline_model, track, idea_idx, idea_text, ts))
                    if cur.rowcount > 0:
                        stats["inserted"] += 1

    conn_r.commit()
    conn_r.close()

    logger.info(f"generate_baselines: {stats['inserted']} baseline rows inserted "
                f"for {stats['papers']} papers")
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    import config as cfg

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(description="Generate baseline ideas")
    parser.add_argument("--papers-db",  default=str(cfg.PAPERS_DB))
    parser.add_argument("--results-db", default=str(cfg.RESULTS_DB))
    parser.add_argument("--smoke",      action="store_true")
    args = parser.parse_args()

    stats = generate_baselines(args.papers_db, args.results_db, smoke=args.smoke)
    print(f"\nDone: {stats['inserted']} baseline rows inserted")
