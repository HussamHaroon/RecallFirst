# Data card

All files are JSON lines, one object per instance (or per instance × condition). Rows containing `error` are runs
that failed (all out-of-memory in likelihood scoring) and are excluded from every analysis (`nami.stats.load` drops
them). Lite files come from the SWE-bench Lite test split (one gold file each); `postcutoff_live120.jsonl` comes from
SWE-bench-Live (one or more gold files). "Names the file" (`gold_stem_in_issue` / `stem_in_issue`) = a gold file's
base name without `.py` occurs in the issue text, case-insensitively, ignoring the stems `__init__`, `base`,
`utils`, `core` and stems of ≤2 characters.

## `orderings_wholerepo.jsonl` — main study (61 rows, 57 usable; 8 repositories)
- `instance`, `repo`, `gold`; `protocol` ("rerank-v2"); `null_repo` (repository of the length-matched null issue).
- `orders` (file lists, best first):
  - `bm25`, `dense`: ranking of **all non-test, non-docs Python files** at the base commit, **truncated to 100**
    (shorter for repositories with fewer files).
  - `nami_bpt`, `nami_bits` (likelihood contrast, per-token / total bits), `qlm` (query likelihood), `h_real`
    (no-null ablation): **complete orderings of the 25-file rerank pool = BM25 top-25** (`n_cand_files`).
  - `gemma_paths`, `gemma_names`: Gemma-4-12B-it's top-10 from the shuffled BM25-25 ∪ dense-25 pool (paths only /
    paths + up to 15 symbol names per file), followed by the rest of that pool in BM25 order (added by
    `analysis/merge_listwise.py` from `listwise_gemma.jsonl`).
- `ranks`: rank of the gold file. For `bm25`/`dense` this is the **full** rank, which can exceed 100 even though
  `orders` stops at 100; for the likelihood methods it is the rank inside the 25-file pool, **−1 = gold not in pool**.
- `ceilings`: whether the gold file is in BM25/dense top-k (k = 5…100). `gold_in_cand`: gold in the rerank pool.
- `issue_feats` (regex counts on the issue: words, identifiers, tracebacks, code fences, .py paths …), `style`
  (identifier / mixed / prose, thresholds fixed before results), `gold_stem_in_issue` (gold file name appears in the
  issue text).
- `n_files_repo`, `n_chunks_scored`, timings in seconds (`score_s`: two NAMI passes; `qlm_s`; `total_s`).

## `listwise_gemma.jsonl` — raw listwise answers (114 rows = 57 × {`paths`, `paths_sigs`})
`pool_size` (28–49), `gold_in_pool`, `gemma_top5`, `gemma_top10`, `ranks` (`gemma_listwise`: position in the top-10
or null; `gemma_fill`: rank in the filled ordering; `bm25`/`dense` copied), `parse_n` (valid indices parsed: 10 for
all rows), `raw_answer`, `prompt_tokens`, `secs` (paths_sigs includes repository download and chunking).

## `controls_probe57.jsonl` — controls on the same 57 instances (342 rows = 57 × 6 conditions)
Produced by `harness/run_controls57.py` with the selector prompt of the paper (paths only), greedy decoding.
- `condition`: `bm25pool` (Gemma selects from BM25's top-25 only — the pool the likelihood methods reranked;
  shuffle seed `"<instance>|bm25pool"`), `union_empty` (union pool, issue replaced by "(no bug report text is
  available)"), `union_mismatch` (union pool, issue of the next instance from the same repository; `issue_from`),
  `union_seed1` / `union_seed2` (union pool, extra shuffles `"<instance>|listwise|union_seedN"`), `probe57`.
- Listwise rows: `pool_size`, `gold_in_pool`, `top10` (model's picks), `rank` (gold rank after appending unpicked
  pool files in BM25-then-dense order; null if not in pool), `parse_n`, `raw`, `secs`.
- `probe57` rows (issue + 10 paths, no code): `cands` (gold + the 9 highest-ranked non-gold files of BM25 and dense,
  interleaved, shuffled with seed 2000+index), `gold_letter`, `picked`, `correct`; readers on the identical options:
  `bm25_correct` / `dense_correct` (best-ranked option under each retriever's ordering — disadvantaged by
  construction, the distractors are their own top non-gold files), `path_tiebreak_correct` (issue-token overlap with
  the path; ties broken by BM25 order), `path_abstain_correct` (ties scored wrong), `path_tied`, `stem_in_issue`.

## `postcutoff_live120.jsonl` — post-cutoff test (120 rows, SWE-bench-Live)
Issues created 2025-04-01 … 2025-08-30, at most 3 per repository (84 repositories), drawn with seed 20261107 from
all splits; pipeline frozen from the Lite study (`harness/run_postcutoff_live.py`), nothing executed.
- `instance`, `repo`, `created_at`, `gold` (list: non-test `.py` files changed by the fix), `n_gold` (47 rows > 1),
  `n_files_repo`, `n_chunks`, `stem_in_issue`, `issue_chars`.
- `pool_size`, `gold_in_pool` (any gold file in BM25-25 ∪ dense-25: 118/120), `ranks` (`bm25`, `dense`,
  `gemma_fill`: rank of the first gold file), `gemma_top10`, `parse_n` (10 on 119/120), `raw`, `bm25_top100`,
  `dense_top100`, `secs` (download + indexing + selection).

## `orderings_pool61.jsonl` — controlled study (25 rows: astropy 6, django 19)
Gold + 60 seeded random distractor files (`n_files` = 61). **Complete** orderings and gold ranks for `bm25`,
`nami_bpt`, `nami_bits`, `qlm`, `h_real`; issue features as above. Fixed scorer.

## Determinism pair
`determinism_original_oct19.jsonl` (first run, 25 rows: gold ranks per method) and
`determinism_rescore_nov01.jsonl` (same pools re-scored 13 days later on a different A100 VM, 24 usable rows, full
orderings). Both use the scorer **before** the padding fix; the gold-file ranks of `nami_bpt`, `nami_bits`, `h_real`
agree on 24/24 instances. The 25th (django-11583) ran out of GPU memory in both runs at the same point; in the
first run it was re-scored separately, which is why `determinism_original_oct19.jsonl` has 25 rows.

## `probe_hard.jsonl` — issue-only probe, hard distractors (25 rows, same instances as the controlled study)
`cands` (the exact 10 options: gold + the 9 highest-ranked non-gold files under BM25 and dense, interleaved,
shuffled with a fixed seed), `gold_letter`, `picked`/`picked_file`/`correct` (Gemma, issue + paths only),
`bm25_pick`/`dense_pick` (readers over the files' code; **the distractors are their own top non-gold files**),
`path_pick` (issue-token match on the paths; `=` means a tie and is scored as wrong), `*_correct`, `raw_answer`, `secs`.

## `probe_easy.jsonl` — issue-only probe, random distractors (25 rows)
`gold`, `gold_letter`, `picked`, `correct`, `n_cands` (10). **The 10 options were not saved**, so this probe is
reported for context only. `open_answer`/`open_hit`: open-ended "name the file" answers, run for 3 instances only
(empty elsewhere).

## `patch_demo.jsonl` — one-turn patch agent (36 rows = 12 django instances × 3 conditions; 10 pass the gate)
`condition` (`bm25`, `dense`, `gemma` top-5 as context), `top5`, `gold_in_top5`, `labels` (FAIL_TO_PASS tests as
Django test labels), `env_ok` / `baseline_pass` / `gold_pass` (sanity gate: tests must fail on the base commit and
pass with the official fix), `apply_ok`, `apply_method`, `resolved` (applied and the target tests pass; null when not
evaluable), `gen_s`, `diff_lines`, `diff_head`/`note` (first lines of the model diff / applier error), `gold_tail` (end of the official-fix test log).

## Known flags
- `pylint-dev__pylint-6506`: the gold file is ranked last in the 25-file pool by `nami_bpt`, `nami_bits` and `qlm`
  (no-null ablation: 16; BM25: 1). Flagged automatically by `analysis/inpool_analysis.py`; not investigated. It may
  be a chunking or truncation artefact (chunks are capped at 6,000 characters for scoring). It affects one instance
  and does not change any reported conclusion (BM25 already ranks it first).
- `probe_easy.jsonl` stores picks but not the 10 options; it cannot be re-scored.
