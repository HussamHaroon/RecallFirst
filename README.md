# RecallFirst — bug-localization controls and a post-cutoff memorization test for coding agents

Companion release for *"Recall First, Then Let the Model Choose: Gemma 4 as a Bug-Localization Selector, Tested on
Post-Cutoff Issues"* (Kaggle × Google DeepMind, Gemma 4 Developer Agent Paper Track, 2026) — PDF in [`paper/`](paper/RecallFirst_paper.pdf).
Author: Hussam Khaskheli (GitHub: HussamHaroon). Code: Apache-2.0. Reproduce every number on CPU: [`notebooks/reproduce.ipynb`](notebooks/reproduce.ipynb).

**What it is for.** Before an LLM file localizer's SWE-bench number can be trusted, three questions need answers:
is it better than strong baselines *on the same instances*, did it lose because the ranker is weak or because the
candidate pool missed the file, and how much comes from reading the issue rather than remembering a public
benchmark? This release answers them for Gemma-4-12B-it and lets you ask them of your own localizer:

- **Same-instance orderings** for 57 SWE-bench Lite instances (8 repositories): BM25, dense, four likelihood
  rerankers, Gemma listwise selection on two pools.
- **Controls**: the selector on BM25's pool only, with the issue removed, with another issue from the same
  repository, under two extra shuffles, and path-only BM25/MiniLM rankings of the same pools (Lite and post-cutoff); a 10-way issue-only probe with hard distractors and four non-LLM readers
  scored on the identical options.
- **A post-cutoff set**: 120 SWE-bench-Live issues created April–August 2025 (84 repositories, after Gemma 4's
  January 2025 cutoff) with BM25/dense top-100 and the frozen selector's answers.
- **Exact paired statistics** (McNemar with b/c counts, Holm within question families, bootstrap CIs, repo-clustered
  bootstrap) that run on a laptop CPU in seconds.

## Headline numbers (all recomputable from `data/`)

| | top-1 | top-5 |
|---|---|---|
| Likelihood scoring (NAMI) of BM25's top-25 | 5% | 26% |
| **Gemma selecting from the same 25 files** | **47%** | **63%** |
| Gemma, BM25-25 ∪ dense-25 pool | 61% | 84% |
| … same pool, issue removed / issue from another task (model's own picks) | 7% / 2% | 21% / 14% |
| … same pool ranked by path strings only (BM25 / MiniLM) | 25% / 28% | 54% / 42% |
| Post-cutoff (120 issues): dense / **Gemma** | 38% / **61%** | 72% / **85%** |
| Post-cutoff, issue removed / path-only BM25 | 11% / 32% | 31% / 60% |

## Quick start (CPU)

```bash
pip install -r requirements.txt
python analysis/controls_postcutoff_stats.py     # every McNemar test, Holm families, CIs, post-cutoff analysis
python analysis/make_figures.py                  # the paper's three figures -> figures/
python -m nami.stats data/orderings_wholerepo.jsonl                      # all Lite methods vs BM25
python -m nami.stats data/orderings_wholerepo.jsonl --add my_localizer.jsonl --baseline dense
python analysis/inpool_analysis.py data/orderings_wholerepo.jsonl INPOOL_ANALYSIS.md   # pool miss vs ranker loss
```

To evaluate **your own localizer**, write one JSON line per instance, `{"instance": "...", "order": ["path/a.py", ...]}`,
and pass it with `--add`; it is compared pairwise with every released method on the identical instances. For the
memorization question, run it on the 120 post-cutoff instances (`data/postcutoff_live120.jsonl` lists IDs, repos,
dates and gold files) and compare its gap to dense with the gap on Lite.

## Layout

| Path | Contents |
|---|---|
| `data/` | Per-instance results (JSON lines) — every field in `DATA_CARD.md` |
| `analysis/` | `controls_postcutoff_stats.py` (paper statistics), `make_figures.py`, `inpool_analysis.py`, `merge_listwise.py` |
| `nami/` | Likelihood scorer (LM head applied in 128-position slices, flat memory), tree-sitter chunker, SWE-bench fetch, issue features, `stats.py` (`compare()` + CLI) |
| `harness/` | GPU scripts that produced the data (originally Colab cells): `run_wholerepo_rerank.py`, `run_listwise.py`, `run_controls57.py`, `run_postcutoff_live.py`, probes, `run_patch_demo.py`, `run_pool61.py` |
| `figures/` | The paper's figures, generated from `data/` |
| `docs/ENGINEERING_LOG.md` | What broke, how it was found, run measurements |

## Running the GPU scripts

Each `harness/*.py` loads `google/gemma-4-12b-it` in bf16 (about 25 GB of GPU memory; an A100-40GB was used) and
writes JSON lines to `$RECALLFIRST_WORKDIR` (default `/content`). They download SWE-bench Lite / SWE-bench-Live from
the Hugging Face Hub and repository tarballs from GitHub; nothing from the repositories is executed except by the
gated patch demo. Accept the Gemma terms on the Hub and set `HF_TOKEN` if needed; **no token is included here.**
Selection is deterministic (greedy decoding, fixed per-instance shuffle seeds `"<instance>|listwise"`); the
post-cutoff sample is drawn with seed 20261107, at most 3 issues per repository, created ≥ 2025-04-01.

## Limitations

One model (Gemma-4-12B-it), one dense embedder (MiniLM; the planned CodeRankEmbed failed to load), Python only;
57 Lite instances; post-cutoff repositories are smaller than Lite's (median 127 vs 718 Python files), so compare
methods *within* a set, not absolute rates across sets. Likelihood orderings cover BM25's top-25 only. The easy probe
did not store its options; the patch demo covers 12 django instances. See the paper.

## Licence and data provenance

Code: Apache-2.0 (`LICENSE`). Data are derived from SWE-bench Lite (MIT) and SWE-bench-Live and contain file
paths, ranks, issue features and short model outputs; repository source code is not redistributed except short
model-generated diff excerpts in `patch_demo.jsonl` (the repositories' own licences apply to code they quote).
Model outputs were produced with Gemma-4-12B-it; see the model's terms of use.
