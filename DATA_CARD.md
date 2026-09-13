# Data card — AgentIdeaBench release

> Covers everything under `release_data/`. Regenerate with
> `python scripts/export_release_data.py`; re-check with `--verify-only`.
> Licensed CC BY 4.0 ([`LICENSE-DATA`](LICENSE-DATA)). The code that produced it is
> MIT ([`LICENSE`](LICENSE)).

51 files, 435,770 rows, 39.0 MB of gzipped CSV. Every file loads with
`pandas.read_csv(path)` — the `.gz` is handled transparently.

`release_data/MANIFEST.json` is the machine-readable index: for each file it records
the row count, byte size, SHA-256, source table, the full column list, and which
columns were dropped on the way out. Where this document and the manifest disagree,
the manifest is right — it is generated, this is written.

---

## 1. What the release does not carry

Three exclusions. Each is a legal obligation, not a size decision, and the export
script enforces all three by construction rather than by inspection.

### 1.1 No abstracts, no reference-list text, no full text

Paper metadata, abstracts and reference lists came from the Semantic Scholar Academic
Graph API under the Semantic Scholar API License Agreement. Academic Graph *metadata*
is ODC-BY 1.0 and may be redistributed with attribution; the *abstracts* are often
publisher content that the API agreement does not license us to redistribute.

So the release carries identifiers where the internal databases carry text. Every
reference list, retrieval trace and prior-art evidence set is reduced to Semantic
Scholar `paperId` values plus counts, years and our own query strings. To get the
text back, refetch it from the API under your own agreement:

```python
import requests
r = requests.get(f"https://api.semanticscholar.org/graph/v1/paper/{ss_paper_id}",
                 params={"fields": "title,abstract,year,externalIds"})
```

These columns are removed from every table on export:

| Dropped column | Where it lived | Why |
|---|---|---|
| `abstract`, `real_abstract`, `idea_abstract` | `papers`, `domain_anchor_pool` | publisher-sourced abstract text |
| `gt_hypothesis` | `papers` | a model rewrite of the source abstract; republishing it republishes the abstract's substance under another surface form |
| `references_json`, `ranked_refs_json`, `paper_refs_backup` | `papers` | reference lists with abstracts and full-text citation contexts nested inside |
| `refs_json` | `subdomain_refs`, `domain_topic_refs`, `e38_replay_refs` | same, for the curated Static reference sets |
| `evidence_json` | `e13_evidence` | retrieved prior art, abstracts included |
| `telemetry`, `telemetry_json`, `trace_json` | idea tables | agent traces whose `result_preview` field is the raw API response |
| `recitation_text` | `prior_probe`, `prior_probe_refs` | a model's attempt to reproduce a held-out abstract; the ROUGE and Jaccard columns are the finding and they stay |
| `raw_response`, `raw_forward`, `raw_swap`, `reasoning_json` | critic tables | raw critic transcripts, which quote the retrieved evidence at length |
| `raw_output` | `active_comparison_ideas` | unparsed Active transcript; the parsed `hypothesis` column stays |

The information worth keeping from the four nested payloads is not lost — it is
lifted into `derived/` as identifiers. See §4.

### 1.2 No Gemini-generated text

The five held-out Gemini models were accessed through the Gemini API, whose additional
terms assign ownership of generated output to the developer while forbidding the use of
that output to build or improve a competing model. Releasing that text under CC BY 4.0
would purport to grant rights we do not hold, so the text is withheld and the fields
carry `[withheld: Gemini API terms, see LICENSE-DATA]` instead.

Their **scores, per-dimension breakdowns and derived statistics are released in full**,
so every published number involving them is reproducible from this release.

| Model | Rows with text withheld |
|---|---:|
| `google/gemini-2.5-flash` | 240 |
| `google/gemini-2.5-flash-lite` | 240 |
| `google/gemini-3-flash-preview` | 240 |
| `google/gemini-3.1-pro-preview` | 240 |
| `google/gemini-3.5-flash` | 240 |

The rule is applied to every `google/gemini*` identifier in the databases, which is a
superset of the five and so strictly more conservative. Gemma models are open-weight
under the Gemma Terms of Use and are not affected.

### 1.3 Reference titles inside model citation footers

The generator is asked to append a `Cited: [14] ...` footer. Internally that column
stores the bracket index *and* the reference title; the release keeps the indices and
drops the titles, since the titles are reference-list text. `cited_refs` is therefore
a JSON array of integers, e.g. `[14, 15, 8, 20]`, indexing the reference list that run
was shown.

### 1.4 One thing the release does carry, deliberately

Weak models sometimes paste a sentence of a reference abstract into their own
hypothesis or citation footer. That text is model output, and rewriting model output
would corrupt the artifact this benchmark exists to release, so it is left intact.

The verifier reports it rather than failing on it. In the 400-shingle sample the
verifier draws, **4 shingles were found inside model-generated text, across 2 files**
(`core/results.csv.gz`, `core/uniform_critic_scores.csv.gz`). This is a detection from
a sample, not a rate: the fraction of generations containing verbatim runs from source
abstracts is **not computed**.

---

## 2. How the boundary is checked

`python scripts/export_release_data.py --verify-only` re-reads every product and
asserts five things. It exits non-zero on any failure.

1. **No dropped column survived.** Every file's header is intersected with the drop list.
2. **No long unwhitelisted text.** Any text value over 300 characters in a column not on
   the whitelist fails. The whitelist is model generations plus our own derived fields,
   so a payload that slipped through under a new column name is caught by length.
3. **No abstract shingles in assembled fields.** 400 sixty-character substrings are
   drawn from real abstracts across all source databases and searched for in the
   products. A hit in a field we assembled is a failure; a hit inside model-generated
   text is counted and reported (§1.4).
4. **No Gemini-generated text survived.** Every row whose generating model matches
   `google/gemini*` is checked to carry the placeholder in every text column.
5. **Checksums match** what `MANIFEST.json` recorded.

---

## 3. Shared column vocabulary

These columns recur across many files and mean the same thing everywhere. Per-file
specifics are in §4 and §5; the exhaustive per-file column list is in `MANIFEST.json`.

| Column | Definition | Computation | Range | Source |
|---|---|---|---|---|
| `idea_model`, `gen_model` | the model that generated the hypothesis | OpenRouter model identifier as dispatched | string, e.g. `deepseek/deepseek-r1-0528` | recorded at generation |
| `critic_model` | the model that scored it | OpenRouter model identifier | string | recorded at scoring |
| `track` | evaluation mode | `B` = Static (given references), `C` = Active (given a search tool) | `B` \| `C` | run configuration |
| `domain` | top-level field | one of five, fixed by the sampling design | Biology, CS, Chemistry, Medicine, Physics | run configuration |
| `subdomain` | the subfield prompt the model was given | the query string that defines one benchmark cell | string, 40 distinct values in the main analysis | run configuration |
| `idea_index` | which of the repeated generations this is | 1-based, up to 3 per cell in the main analysis | integer ≥ 1 | run configuration |
| `item_id` | joins a score row back to its hypothesis | `{model}\|{track}\|{subdomain}` truncated, exactly as written at scoring time | string | derived at scoring |
| `idea_text`, `hypothesis` | the generated hypothesis | verbatim model output, one paragraph | string, or the Gemini placeholder | model output |
| `score_originality`, `score_feasibility`, `score_clarity`, `score_impact`, `score_specificity` | one critic's rating on one dimension | flattened out of `scores_json`; both the flat and the `{score, reasoning}` nested form reduce to the number | integer 1–10, or empty when the call errored | LLM judgment — **not a direct observation** |
| `error` | why a call produced no usable result | exception or parse-failure string, empty on success | string | recorded at call time |
| `created_at` | when the row was written | ISO-8601 UTC | timestamp | recorded at write |
| `ss_paper_id` | Semantic Scholar `paperId` | 40-char hex, resolves at `api.semanticscholar.org/graph/v1/paper/{id}` | string | Semantic Scholar API |

**The five score columns are LLM judgments, not measurements.** They are what a critic
model output under the rubric in §Scoring of the README, after being shown prior-art
evidence frozen in `e13_evidence`. Aggregate them as the paper does — trimmed mean
dropping the highest of three critics, weighted `O:2, I:1.5, F:1, C:0.5, S:0.5`,
normalised by 5.5 — or differently, but they carry judge-relative uncertainty either
way. Per-cell confidence intervals are **not computed** in this release; the paper's
appendix reports bootstrap intervals at the model level.

---

## 4. `derived/` — what was rescued from the dropped payloads

Four side tables, each built by exploding a payload that could not be released whole.

### `paper_references.csv.gz`

One row per reference of one sampled paper, from `papers.references_json` (the full
reference list) and `papers.ranked_refs_json` (our embedding-ranked candidate pool).

| Column | Definition | Computation | Range | Source |
|---|---|---|---|---|
| `paper_id` | the citing paper | Semantic Scholar `paperId` | 40-char hex | Semantic Scholar |
| `list_kind` | which list this row came from | `reference` = the paper's own bibliography; `ranked` = our top-20 embedding-ranked pool | `reference` \| `ranked` | derived |
| `rank` | position in that list | 0-based index as stored | integer ≥ 0 | derived |
| `ss_paper_id` | the cited paper | Semantic Scholar `paperId` | 40-char hex | Semantic Scholar |
| `year`, `citation_count` | the cited paper's year and citations | as returned by the API at fetch time | integer | Semantic Scholar (ODC-BY) |
| `in_paper_citations` | how many times the citing paper cites it | count from the API's citation contexts | integer ≥ 0 | Semantic Scholar |
| `is_influential` | Semantic Scholar's influential-citation flag | as returned | `True` \| `False` | Semantic Scholar |

Titles, abstracts and citation contexts are dropped; `ss_paper_id` is how you refetch them.

### `static_refs.csv.gz`

The curated Static-track reference sets — what a Track B model was actually shown.
Merges three source tables, so a row's key columns depend on `source_table`.

| Column | Definition | Computation | Range | Source |
|---|---|---|---|---|
| `source_table` | which reference set this is | `subdomain_refs` (main analysis), `domain_topic_refs` (early domain-level), `e38_replay_refs` (replay control) | string | derived |
| `key` | that set's identifier | the subdomain name, or the topic query | string | run configuration |
| `rank` | position in the set as presented | 0-based; Static presentation order is deterministically shuffled from this | integer ≥ 0 | derived |
| `ss_paper_id`, `year`, `citation_count` | the reference | as in `paper_references` | — | Semantic Scholar |

### `active_traces.csv.gz`

One row per tool call an Active run made — the retrieval behaviour that Track C
measures and Track B has none of.

| Column | Definition | Computation | Range | Source |
|---|---|---|---|---|
| `source_table` | which experiment | `subdomain_ideas` (main) or `budget_sweep_ideas*` (budget sweep) | string | derived |
| `budget` | the tool-call budget that run was given | empty for the main analysis, which uses the fixed budget of 10 | integer or empty | run configuration |
| `iter` | which loop iteration issued the call | 0-based | integer ≥ 0 | agent loop |
| `tool` | which tool | `search_papers` or `fetch_paper` | string | agent loop |
| `query` | the search string the model composed | verbatim model output; the Gemini placeholder where withheld | string | model output |
| `limit` | results requested | as the model passed it | integer | model output |
| `n_results` | results returned | length of the returned id list | integer ≥ 0 | derived |
| `returned_ss_ids` | which papers came back | JSON array of Semantic Scholar `paperId`, parsed out of the raw response; **the response text itself is discarded** | JSON array | derived from Semantic Scholar |

`query` is model-generated text and is withheld for Gemini models along with the rest.

### `evidence_hits.csv.gz`

One row per prior-art paper retrieved for one critic call. This is the evidence the
critic judged originality against, frozen so scoring is reproducible.

| Column | Definition | Computation | Range | Source |
|---|---|---|---|---|
| `item_id`, `grp` | which hypothesis and experiment group this evidence backs | as written at retrieval | string | derived |
| `cutoff_date` | the prior-art filter applied | a single global idea-conception date, `2026-05-31`, for every model | date | run configuration |
| `rank` | position in the retrieved set | 0-based, up to 8 per call | integer 0–7 | derived |
| `query` | the keyword query that retrieved it | one of three extracted from the hypothesis | string | derived |
| `ss_paper_id`, `year`, `publication_date`, `citation_count` | the retrieved paper | as returned by the API | — | Semantic Scholar |

Titles and abstracts are dropped. The paper describes `e13_evidence` as frozen with
abstracts included — that is true of the internal table; the released form is
identifiers only, and the abstracts refetch from `ss_paper_id`.

---

## 5. File inventory

### `core/` — tables behind the main results

| File | Rows | Size | Source table | Columns dropped |
|---|---:|---:|---|---|
| `lit8d_scores_3seed.csv.gz` | 47,013 | 0.42 MB | `results.db:lit8d_scores_3seed` | — |
| `subdomain_ideas.csv.gz` | 26,098 | 7.64 MB | `results.db:subdomain_ideas` | `telemetry` |
| `e13_evidence.csv.gz` | 19,540 | 1.25 MB | `results.db:e13_evidence` | `evidence_json` |
| `uniform_critic_scores.csv.gz` | 18,646 | 1.98 MB | `results.db:uniform_critic_scores` | `raw_response`, `reasoning_json`, `telemetry` |
| `results.csv.gz` | 17,377 | 3.06 MB | `results.db:results` | `raw_response`, `reasoning_json`, `telemetry` |
| `lit8d_scores.csv.gz` | 3,471 | 0.03 MB | `results.db:lit8d_scores` | — |
| `model_scores.csv.gz` | 1,075 | 0.29 MB | `results.db:model_scores` | — |
| `papers.csv.gz` | 249 | 0.03 MB | `papers.db:papers` | `abstract`, `gt_hypothesis`, `paper_refs_backup`, `ranked_refs_json`, `references_json` |
| `pairwise_results.csv.gz` | 225 | 0.00 MB | `results.db:pairwise_results` | `raw_forward`, `raw_swap`, `telemetry` |
| `subdomain_refs.csv.gz` | 100 | 0.00 MB | `results.db:subdomain_refs` | `refs_json` |
| `e13_anchor_meta.csv.gz` | 28 | 0.00 MB | `results.db:e13_anchor_meta` | — |
| `survey_refs.csv.gz` | 5 | 0.00 MB | `papers.db:survey_refs` | `refs_json` |

### `appendix/` — tables behind the appendix experiments

| File | Rows | Size | Source table | Columns dropped |
|---|---:|---:|---|---|
| `e12_gap_scores.csv.gz` | 4,933 | 0.06 MB | `results.db:e12_gap_scores` | — |
| `budget_sweep_scores_v3.csv.gz` | 3,159 | 0.05 MB | `results.db:budget_sweep_scores_v3` | `raw_response`, `reasoning_json`, `telemetry` |
| `budget_sweep_scores_v2.csv.gz` | 3,150 | 0.05 MB | `results.db:budget_sweep_scores_v2` | `raw_response`, `reasoning_json`, `telemetry` |
| `swm_scores.csv.gz` | 2,853 | 0.03 MB | `results.db:swm_scores` | — |
| `e32_recall_scores.csv.gz` | 2,520 | 0.02 MB | `results.db:e32_recall_scores` | — |
| `e38_replay_scores.csv.gz` | 2,493 | 0.02 MB | `results.db:e38_replay_scores` | — |
| `prior_probe_refs.csv.gz` | 2,200 | 0.05 MB | `results.db:prior_probe_refs` | `recitation_text` |
| `dynamic_critic_sweep.csv.gz` | 1,690 | 0.03 MB | `results.db:dynamic_critic_sweep` | `raw_response`, `reasoning_json`, `telemetry` |
| `e10_fixv2_scores.csv.gz` | 1,590 | 0.01 MB | `results.db:e10_fixv2_scores` | — |
| `e10_fixv3_scores.csv.gz` | 1,590 | 0.01 MB | `results.db:e10_fixv3_scores` | — |
| `e10_deepseek_scores.csv.gz` | 1,515 | 0.01 MB | `results.db:e10_deepseek_scores` | — |
| `budget_sweep_scores.csv.gz` | 1,290 | 0.02 MB | `results.db:budget_sweep_scores` | `raw_response`, `reasoning_json`, `telemetry` |
| `budget_sweep_ideas_v2.csv.gz` | 1,080 | 0.35 MB | `results.db:budget_sweep_ideas_v2` | `trace_json` |
| `e31_bo3_scores.csv.gz` | 1,080 | 0.01 MB | `results.db:e31_bo3_scores` | — |
| `budget_sweep_ideas_v3.csv.gz` | 1,079 | 0.34 MB | `results.db:budget_sweep_ideas_v3` | `trace_json` |
| `swm_ideas.csv.gz` | 981 | 0.34 MB | `results.db:swm_ideas` | — |
| `e32_recall_ideas.csv.gz` | 840 | 0.19 MB | `results.db:e32_recall_ideas` | — |
| `e38_replay_ideas.csv.gz` | 831 | 0.21 MB | `results.db:e38_replay_ideas` | — |
| `e29_nlp_scores.csv.gz` | 741 | 0.01 MB | `results.db:e29_nlp_scores` | — |
| `prior_probe.csv.gz` | 550 | 0.01 MB | `results.db:prior_probe` | `recitation_text` |
| `budget_sweep_ideas.csv.gz` | 489 | 0.17 MB | `results.db:budget_sweep_ideas` | `trace_json` |
| `dynamic_critic_sweep_ideas.csv.gz` | 469 | 0.12 MB | `results.db:dynamic_critic_sweep_ideas` | `telemetry` |
| `e10_model_idea_scores.csv.gz` | 460 | 0.00 MB | `results.db:e10_model_idea_scores` | `reasoning_json` |
| `e31_bo3_ideas.csv.gz` | 360 | 0.12 MB | `results.db:e31_bo3_ideas` | — |
| `wm_scores.csv.gz` | 345 | 0.00 MB | `results.db:wm_scores` | — |
| `e29_pairwise_llm.csv.gz` | 300 | 0.00 MB | `results.db:e29_pairwise_llm` | `raw_forward`, `raw_swap`, `telemetry` |
| `e38_replay_refs.csv.gz` | 280 | 0.00 MB | `results.db:e38_replay_refs` | `refs_json` |
| `cap_ablation_scores.csv.gz` | 279 | 0.00 MB | `results.db:cap_ablation_scores` | `raw_response`, `reasoning_json` |
| `e10_idea_anchor_scores.csv.gz` | 270 | 0.00 MB | `results.db:e10_idea_anchor_scores` | `raw_response`, `reasoning_json` |
| `active_comparison_scores.csv.gz` | 240 | 0.00 MB | `results.db:active_comparison_scores` | `raw_response`, `reasoning_json`, `telemetry_json` |
| `e7_bestpaper_scores.csv.gz` | 200 | 0.00 MB | `results.db:e7_bestpaper_scores` | `raw_response`, `reasoning_json` |
| `domain_anchor_pool.csv.gz` | 185 | 0.02 MB | `results.db:domain_anchor_pool` | `idea_abstract`, `real_abstract` |
| `wm_ideas.csv.gz` | 120 | 0.08 MB | `results.db:wm_ideas` | — |
| `domain_topic_refs.csv.gz` | 92 | 0.00 MB | `results.db:domain_topic_refs` | `refs_json` |
| `active_comparison_ideas.csv.gz` | 30 | 0.01 MB | `results.db:active_comparison_ideas` | `raw_output`, `telemetry_json`, `trace_json` |

### `derived/` — reference lists, traces and evidence, as identifiers

| File | Rows | Size | Source table | Columns dropped |
|---|---:|---:|---|---|
| `evidence_hits.csv.gz` | 144,024 | 6.15 MB | `e13_evidence.evidence_json` | — |
| `active_traces.csv.gz` | 101,455 | 15.36 MB | `subdomain_ideas.telemetry + budget_sweep_ideas*.trace_json` | — |
| `paper_references.csv.gz` | 11,794 | 0.30 MB | `papers.references_json + papers.ranked_refs_json` | — |
| `static_refs.csv.gz` | 4,386 | 0.14 MB | `subdomain_refs / domain_topic_refs / e38_replay_refs .refs_json` | — |

#### Column definitions for the inventory tables

- **File** — the file's name inside its group directory.
- **Rows** — data rows, excluding the header. Computed by the export script and recorded in `MANIFEST.json`.
- **Size** — bytes on disk after gzip, from `MANIFEST.json`.
- **Source table** — the `database:table` the rows came from, or for `derived/`, the columns they were exploded out of.
- **Columns dropped** — which of that table's columns were removed on export, per §1. An em dash means the table had none of them.

`core/` and `appendix/` files keep their source table's column names, minus the dropped
ones, plus the flattened `score_*` and `aux_*` columns. The shared vocabulary is in §3
and the exhaustive per-file list is in `MANIFEST.json`.

---

## 6. Known limits

- **Scores are LLM judgments.** Nothing here is a human rating. The paper reports an
  agreement study against human annotators in its appendix; this release contains that
  study's rows (`appendix/e29_nlp_scores.csv.gz`, `appendix/e29_pairwise_llm.csv.gz`)
  but no raw human annotations.
- **Per-cell uncertainty is not computed** in these files. Aggregate and bootstrap them
  yourself; the paper's intervals are at the model level.
- **The `appendix/` group includes superseded runs.** `budget_sweep_*`, `*_v2` and
  `*_v3` are successive passes of the same sweep, kept because published numbers cite
  specific versions. Check the paper's appendix for which version backs which number.
- **Pre-overwrite backups and archived error rows are not released**
  (`results_archive_*`, `swm_scores_bak_*`, `active_comparison_ideas_v1_archive`).
  They are duplicates or failed calls, retained locally under the project's raw-data
  policy but not part of a release.
- **Intended use is research on evaluating and improving language-model ideation.**
  This is not a validated instrument for deciding what science to fund or publish, and
  the hypotheses in it are untested model output rather than vetted claims.
