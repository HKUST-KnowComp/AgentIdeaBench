# AgentIdeaBench — Active track leaderboard

Every model receives a scientific subfield and nothing else, searches Semantic Scholar itself under a budget of 10 tool calls, and proposes a testable hypothesis. 63 models, 40 subfields in five disciplines, 3 hypotheses each, 3 critic models per hypothesis, 44,754 critic scores in total.

The best open-weight model is **GLM-5.1** at 6.33. The held-out ceiling is 7.14 (Claude Opus 5). Of the 63 models run in both tracks, 40 score higher with the search tool than with a supplied reading list, and 23 score lower.

Closed-source models are marked † . They are reported here in full and held out of every headline statistic, exactly as in the paper, whose claims are scoped to the 28 open-weight models.

Three models are excluded: gpt-5.3-chat is left out as a second serving route to a model already listed, not a second model; qwen3-235b-a22b-thinking-2507, qwen3-vl-8b-thinking ran the Active track only, so they have no Static column to rank against.

| # | Model | Vendor | Static | Active | Δ | Turns |
|--:|---|---|--:|--:|--:|--:|
| 1 | Claude Opus 5 † | Anthropic | 6.89 | **7.14** | +0.25 | 9.7 |
| 2 | GPT-5.6 Sol † | OpenAI | 6.90 | **6.88** | -0.02 | 9.8 |
| 3 | GPT-5.6 Terra † | OpenAI | 6.86 | **6.80** | -0.06 | 9.7 |
| 4 | Claude Opus 4.8 † | Anthropic | 5.78 | **6.79** | +1.00 | 7.0 |
| 5 | Claude Opus 4.7 † | Anthropic | 5.83 | **6.70** | +0.90 | 9.9 |
| 6 | GPT-5.5 † | OpenAI | 6.70 | **6.68** | -0.01 | 10.0 |
| 7 | Claude Sonnet 5 † | Anthropic | 6.05 | **6.66** | +0.61 | 7.1 |
| 8 | GPT-5.2 † | OpenAI | 6.16 | **6.66** | +0.50 | 8.0 |
| 9 | GPT-5.4 † | OpenAI | 6.18 | **6.62** | +0.44 | 8.7 |
| 10 | GPT-5.6 Luna † | OpenAI | 6.70 | **6.43** | -0.27 | 9.7 |
| 11 | GPT-5 † | OpenAI | 6.55 | **6.40** | -0.15 | 9.8 |
| 12 | GPT-5 mini † | OpenAI | 5.88 | **6.38** | +0.49 | 9.1 |
| 13 | GPT-5.4 mini † | OpenAI | 6.06 | **6.36** | +0.30 | 5.1 |
| 14 | GLM-5.1 | Zhipu | 5.13 | **6.33** | +1.21 | 8.8 |
| 15 | Gemini 3 Flash † | Google | 5.59 | **6.29** | +0.70 | 8.5 |
| 16 | Gemini 3.5 Flash † | Google | 5.87 | **6.22** | +0.35 | 9.9 |
| 17 | Gemini 3.1 Pro † | Google | 5.63 | **6.20** | +0.57 | 7.5 |
| 18 | Kimi K2.6 | Moonshot | 5.10 | **6.17** | +1.08 | 9.8 |
| 19 | o3 † | OpenAI | 5.56 | **6.06** | +0.49 | 5.5 |
| 20 | GPT-5.1 † | OpenAI | 5.66 | **6.04** | +0.38 | 4.0 |
| 21 | Claude Opus 4.6 † | Anthropic | 5.80 | **6.03** | +0.25 | 10.0 |
| 22 | Qwen3.5 397B-A17B | Alibaba | 5.06 | **5.95** | +0.89 | 8.9 |
| 23 | Kimi K2.5 | Moonshot | 4.91 | **5.92** | +1.02 | 9.9 |
| 24 | DeepSeek-V4 Pro | DeepSeek | 5.32 | **5.92** | +0.60 | 8.5 |
| 25 | MiMo-V2.5 Pro | Xiaomi | 5.11 | **5.91** | +0.81 | 9.0 |
| 26 | Claude Sonnet 4.6 † | Anthropic | 5.36 | **5.84** | +0.51 | 10.0 |
| 27 | GPT-5.4 nano † | OpenAI | 5.72 | **5.82** | +0.10 | 4.6 |
| 28 | GLM-4.6 | Zhipu | 4.83 | **5.78** | +0.95 | 6.9 |
| 29 | Claude Opus 4.5 † | Anthropic | 5.52 | **5.62** | +0.10 | 10.0 |
| 30 | o4-mini † | OpenAI | 5.39 | **5.61** | +0.22 | 7.6 |
| 31 | Qwen3.5 27B | Alibaba | 4.71 | **5.53** | +0.82 | 8.0 |
| 32 | DeepSeek-V4 Flash | DeepSeek | 5.09 | **5.52** | +0.42 | 9.0 |
| 33 | Claude Sonnet 4.5 † | Anthropic | 5.02 | **5.50** | +0.48 | 10.0 |
| 34 | Gemma 4 31B | Google | 5.14 | **5.49** | +0.35 | 5.0 |
| 35 | MiMo-V2.5 | Xiaomi | 4.90 | **5.46** | +0.56 | 7.8 |
| 36 | Claude Haiku 4.5 † | Anthropic | 5.20 | **5.37** | +0.17 | 9.9 |
| 37 | Mistral Medium 3.1 | Mistral | 4.74 | **5.37** | +0.62 | 6.7 |
| 38 | GPT-5 nano † | OpenAI | 5.75 | **5.36** | -0.39 | 6.7 |
| 39 | MiniMax-M2.7 | MiniMax | 4.70 | **5.33** | +0.62 | 9.7 |
| 40 | DeepSeek-R1 | DeepSeek | 4.86 | **5.30** | +0.44 | 6.0 |
| 41 | Mistral Small 2603 | Mistral | 4.61 | **5.16** | +0.55 | 9.4 |
| 42 | Gemini 2.5 Flash † | Google | 5.10 | **5.05** | -0.05 | 7.7 |
| 43 | Qwen3 30B-A3B | Alibaba | 4.93 | **4.94** | +0.01 | 8.3 |
| 44 | Qwen3.5 9B | Alibaba | 4.23 | **4.87** | +0.64 | 6.9 |
| 45 | Gemma 3 27B | Google | 4.68 | **4.82** | +0.14 | 6.2 |
| 46 | GLM-4.5 Air | Zhipu | 4.31 | **4.82** | +0.51 | 8.9 |
| 47 | o3-mini † | OpenAI | 4.84 | **4.82** | -0.02 | 3.6 |
| 48 | Qwen3 Coder | Alibaba | 4.87 | **4.72** | -0.15 | 8.6 |
| 49 | GPT-4.1 † | OpenAI | 4.92 | **4.70** | -0.22 | 8.7 |
| 50 | o1 † | OpenAI | 4.55 | **4.66** | +0.11 | 8.7 |
| 51 | Qwen3 32B | Alibaba | 4.53 | **4.44** | -0.10 | 3.1 |
| 52 | GPT-4o † | OpenAI | 4.71 | **4.37** | -0.34 | 9.9 |
| 53 | GPT-4.1 nano † | OpenAI | 4.66 | **4.32** | -0.35 | 2.0 |
| 54 | Gemini 2.5 Flash-Lite † | Google | 4.63 | **4.27** | -0.37 | 6.9 |
| 55 | GPT-4.1 mini † | OpenAI | 4.92 | **4.26** | -0.66 | 7.3 |
| 56 | Qwen3 8B | Alibaba | 4.24 | **4.21** | -0.03 | 4.3 |
| 57 | Qwen2.5 72B | Alibaba | 4.20 | **3.97** | -0.23 | 8.8 |
| 58 | Mistral Small 24B | Mistral | 4.19 | **3.93** | -0.26 | 8.9 |
| 59 | Llama 4 Maverick | Meta | 4.34 | **3.84** | -0.50 | 4.5 |
| 60 | GPT-4o mini † | OpenAI | 4.09 | **3.76** | -0.33 | 9.2 |
| 61 | Qwen2.5 7B | Alibaba | 3.97 | **3.67** | -0.30 | 8.8 |
| 62 | Llama 3.1 8B | Meta | 3.80 | **3.54** | -0.26 | 6.7 |
| 63 | Gemma 2 27B | Google | 4.02 | **3.20** | -0.82 | 4.1 |

#### Column Definitions

| Column | Definition | Computation | Unit / Range | Source |
|---|---|---|---|---|
| # | Rank by Active total, descending. | Position after sorting all scored models. | 1–63 | computed |
| Model | The evaluated hypothesis generator. | Display name for the routed model id. | — | `reports/_leaderboard_full.py` `DISPLAY` |
| Vendor | Publishing organization. † marks closed weights. | Prefix of the model id; † is the gateway/Gemini rule shared with `experiments/e36_leaderboard_subscores.py`. | — | model id |
| Static | Weighted hypothesis quality with a supplied reading list (Track B). | Drop the most generous of 3 critics per hypothesis, weight the five dimensions O2/I1.5/F1/C0.5/S0.5 normalized by 5.5, mean over 3 hypotheses, then over subfields. | 1–10 | `lit8d_scores_3seed`, track B |
| Active | Weighted hypothesis quality with agentic search (Track C). | Identical aggregation to Static, on the agentic rollouts. | 1–10 | `lit8d_scores_3seed`, track C |
| Δ | Gain from agentic search over a supplied reading list. | Paired mean of Active − Static over subfields scored in both tracks. `—` where the model has no Static run. | score points | computed, paired |
| Turns | Mean tool calls issued per rollout, out of a budget of 10. Describes behavior, not quality. | Counted from the logged agent transcript. | 0–10 calls | `reports/e43_active_turns.json` |

Per-dimension scores, standard errors and an interactive version of this table are on the leaderboard page. Row-level scores for every model are in `release_data/`.

Critic scores are model judgments, not measurements of scientific merit. Standard errors are not shown here; they are in `reports/leaderboard_full.json` as `active_sem` and `gain_sem`.

Generated by `reports/_build_leaderboard_page.py` on 2026-09-12.
