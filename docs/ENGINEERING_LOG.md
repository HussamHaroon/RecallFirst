# Engineering log (redacted)
Timestamped record of the experiments, failures and fixes behind this release, as written during the work.
Redactions: private notebook IDs, connection tokens and local machine paths. Section titles were added; content is unchanged otherwise.

---

## Worklog (Sep 27 – Nov 2)

# NAMI Worklog

## 2026-09-27 (session 1)

**Infrastructure established:**
- colab-mcp bridge running (TCP 127.0.0.1:7898), notebook connected:
  "NAMI Experiments.ipynb" (Drive id [notebook-id])
- Chrome DevTools CLI attached to user's Chrome (--autoConnect)
- AI Studio research chat: "AI Research Assistant Collabor" — Gemini with
  Google Search grounding ON; user attached Research.md and instructed it to
  serve as objective research assistant. Treat as lead generator; verify all.
- Colab runtime switched CPU -> T4 GPU (15.6 GB VRAM) via Runtime menu.
  Note: Colab Python is 3.12+; `tree-sitter-languages` is dead there —
  use `tree-sitter-language-pack` instead.

**Notebook state (cell ids):**
- `LdEJAMiPLt5X` — setup cell (deps + GPU check). PASSED on T4: torch 2.11.0+cu128.
- `lPjcEJ5GRXWP` — SWE-bench Lite load + inspect (fired, result pending at write time).

**Literature intel (Gemini grounded search, notes/gemini_answer1.json) — to verify:**
- Conditional-entropy-drop/PMI ranking for bug localization OR code retrieval: "none found".
- KILLER-ADJACENT: "Pointwise Mutual Information as a Performance Gauge for
  Retrieval-Augmented Generation" (NAACL 2025, arXiv:2411.07773) — PMI(D;Q)=logP(D|Q)−logP(D)
  with an autoregressive LM for RAG context selection. Mechanism identical to NAMI;
  domain (text QA vs code), task (RAG gauge vs bug localization), and controls differ.
  Action: read this paper in full; cite; differentiate. Also validates feasibility.
- MASPrism (arXiv:2605.07509, 2026): prefill NLL for faulty-step attribution in agent
  trajectories incl. SWE-bench traces — logs, not source code. Cite as adjacent.
- SweRank (2025), CodeScout (2025): SWE-bench localization via embeddings/BM25 — baselines/related.
- Campos et al. 2024 confirmed UNconditioned likelihood ranking (differentiates us).

**Pending:** background subagent literature report (cross-check vs Gemini's findings).

## 2026-09-27 (session 1, continued)

**Novelty verification COMPLETE** — both independent passes agree: exact
formulation unpublished. See NOVELTY_LOG.md (verdict, 5 closest works,
positioning claim) + lit_report_subagent.md (full) + gemini_answer1.json.
Key additions: LLM-BL (ICPC 2025, prompt-based listwise — nearest task prior),
LecPrompt (2410.08241, unconditioned logprob localization), InfoGain-RAG
(2509.12765), PMI-DC null-contrast ancestor (2104.08315). Correction: DePA
(2502.20246) is dataset cleansing, not bug localization.

**Track/model requirements (user-supplied, noted):** main track strictly
requires gemma-4-31b-it-qat-w4a16-ct (LoRA/scaffold submissions only);
paper track permits open ablations (gemma-2-9b etc.). Implication: use Gemma 4
as primary; QAT-signal-preservation becomes a novel ablation.

**Model access matrix (hf_hub_download probe — definitive, file-level):**
- gemma-2-2b / 2-9b / 2-9b-it: DENIED (license not accepted on HF; earlier
  model_info check was misleading — metadata visible without file access)
- gemma-4-12b-it: OK | gemma-4-31b-it: OK | gemma-4-31b-it-qat-w4a16-ct: OK
- No small gemma-4 variants exist under probed names.
**Model plan:** gemma-4-12b-it primary (int8 on Colab T4 for smoke; bf16
sharded on Kaggle 2xT4 for paper numbers); 31b-qat for quantization ablation;
gemma-2-9b cross-gen ablation IF user accepts gemma-2 licenses (asked).

**Smoke test v1 (gemma-2-2b):** FAILED at model load (403 gated). Everything
upstream worked: dataset loads, repo snapshot (astropy-12907, 8168 chunks),
null control ratio 1.0.

**Smoke test v2 (gemma-4-12b-it int8, T4):** running. Subset per Research.md
7.2: gold files + 60 random distractor files. Watcher polls cell output.

## 2026-09-27 (session 1, smoke saga)

Error chain on the quantization path (all model-loading plumbing; NAMI pipeline
itself verified green in v1):
- v2: `load_in_8bit=True` kwarg rejected — Gemma4 loads as
  `Gemma4UnifiedForConditionalGeneration` (multimodal); modern API requires
  `quantization_config=BitsAndBytesConfig(...)`.
- v3: bitsandbytes not installed on py3.13 runtime → added to setup.
- v4: bnb 0.50.2 installed + imports, but `is_bitsandbytes_available()` = False.
- v5: pivoted to torchao; torchao 0.18.0 installed + imports, but
  `is_torchao_available()` = False — SAME failure pattern.
- **ROOT CAUSE (diagnosed via inspect.getsource):** transformers 5.16.1 wraps
  availability checks in `@lru_cache` and evaluates at import time.
  transformers was first imported BEFORE the libs were pip-installed → checks
  froze False for the whole process. Textbook Colab rule: pip-install-then-
  import needs a kernel restart.
- v6 = kernel restart ("Restart session and run all" via UI) → fresh import
  order (setup installs first) → checks evaluate True. Cache-clear lines added
  to smoke preflight as defense. Status: RUNNING (setup→data→smoke→diag).

- v8: OOM at first forward: model fit (11.4GB int8 on 14.56GB usable) but the
  [B=6, L~1900, V=262k] logits tensor tried to allocate 11.86GiB at once.
  Gemma's 262k vocab makes full-logits scoring infeasible on consumer GPUs.
- v9 = decoder-path scoring (the real fix, worth a methods paragraph in the
  paper): run backbone only -> hidden states [B,L,d] (60MB) -> CE through
  lm_head in 128-position slices (~800MB transient, fp32). Identical math,
  logits tensor never materialized. Handles logit softcapping if configured.
- v8/v9 CORRECTED DIAGNOSIS: both OOMed inside from_pretrained's
  `caching_allocator_warmup` (transformers 5.16) — it pre-allocates a
  model-sized fp16 buffer (the 11.86GiB) DURING LOAD, not at forward time.
  int8 model (11.4GB) + warmup (11.9GB) > 14.56GB usable T4 → guaranteed OOM.
  (Initial v8 theory "logits tensor at forward" was wrong; decoder-path fix is
  still correct and valuable for scoring memory.)
- v10: Int4WeightOnlyConfig(group_size=32) → ~6.5GB model + ~3GB warmup fits.
  Also PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True. Watcher bug fixed:
  completion is now gated on the bridge response journal (stale cell outputs
  from previous failed runs previously triggered false FINISHED).
- v11: warmup bypass WORKED (model loaded) but GPU genuinely full: 14.55GB
  used (11.4 int8 + ~3GB bf16 embeddings/lm_head/vision-tower), 9.8MB free —
  no activation headroom. CONCLUSION: Gemma-4-12b does not fit a single T4
  under any quantization trick. Colab T4 abandoned for model runs.

## 2026-09-27 (session 1, Kaggle pivot)

- Pivoted to Kaggle 2xT4 (30GB VRAM, 29GB RAM): kernel `iamantihero/nami-smoke`
  v1 pushed via CLI (`--accelerator nvidiaTeslaT4x2`). Self-contained script:
  pip installs + inline nami modules + bf16 load (device_map auto, warmup
  monkeypatched) + decoder-path scoring + result JSON to /kaggle/working.
- HF token embedded in PRIVATE kernel; build via experiments/build_kernel.py.
  REMINDER: strip token before any publication of the kernel.
- Colab notebook retains verified data/chunking/null-control pipeline and
  stays the interactive exploration surface (small-model work).
- Status: kernel RUNNING, status poller active (45s cadence).

## 2026-09-27 (session 1, Kaggle blocker)

- v1 AND v2 kernels run with NO internet (pip DNS failures) and
  "Accelerator: None" — enable_internet/enable_gpu silently stripped even as
  correct booleans. Diagnosis: account not phone-verified (user confirmed
  verification pending). Kaggle gates internet + T4x2 on verification.
- BLOCKED on user: kaggle.com/settings -> Phone verification. After verify,
  re-push is one command (kernel.py v2 already built in experiments/).
- The running kernel will die at `import datasets` after pip retries —
  harmless CPU run.

## 2026-09-27 (session 1, Colab post-mortem via Gemini, kernel stopped)

- Kaggle kernel v2: user cancelled/ended it during identity verification;
  events panel shows Cancelled, 0 active events. Nothing running on Kaggle.
- Gemini (grounded) Colab diagnosis (full: notes/gemini_colab_diag.json):
  (a) T4 CAN run gemma-4-12b-it but ONLY with true packed 4-bit (~7-8.3GB):
      bnb NF4 + double quant + fp16 compute. T4 is Turing: no bf16 tensor
      cores -> use fp16 for compute/activations.
  (b) CORRECTION of my v11 attribution: the ~3GB unquantized residue is NOT
      a vision tower — Gemma4-12b has none (just ~35M patch projection). It is
      the 262,144 x 3,840 embedding + lm_head (~2GiB) which quantizers keep
      in 16-bit by default.
  (c) Community text-only/fp8 variants exist but do not solve the embedding
      footprint; NF4 double-quant is the practical T4 path.
- PLAN: Kaggle bf16 (once user verifies identity) = paper numbers.
  Colab NF4-fp16 = free iteration fallback if we want Colab back.

## 2026-09-27 (late night — Colab NF4, per user request)

- Reconnected notebook (user clicked Connect + attached runtime). Fresh VM:
  bnb installed BEFORE transformers import -> availability True. 
- v12 (NF4 double-quant + fp16 compute + warmup monkeypatch): LOAD SUCCEEDED —
  7.7GB on GPU (Gemini's prediction exact), decoder path resolved. OOM at
  first scoring forward: 14.38GB in use. Root cause: transformers default
  use_cache=True builds a ~6.7GB KV cache — useless for teacher-forced
  scoring. v13 = use_cache=False in both scoring paths.
- Memory budget v13: 7.7 model + ~1.4 transient attn + ~0.8 CE slices
  = ~10GB of 14.56 -> comfortable margin.
- Watcher at 10s cadence; results also persist to /content/nami_smoke_result.json
  and notebook cell output. User asleep — verdict waits for morning.

## 2026-09-27 (overnight loop, user granted autonomy + end-of-night shutdown)

- v14/v15: same "modules dispatched to CPU" error — v12's model zombie-held.
  v16: full cleanup prologue → GPU free 15.4GB ✓, load 7.7GB ✓, but SAME
  deterministic OOM at first forward (14.38GB in use, +6.7GB unexplained).
- v17: instrumentation + B=2 → revealed "GPU free before load: 1.1GB" —
  failed-run models held by exception tracebacks/sys.last_* — prologue
  cleanup insufficient in same kernel.
- DETERMINISTIC LOOP STEP ADOPTED: SIGKILL ipykernel cell (Colab auto-
  restarts, disk cache survives) → setup → smoke. Yields 15.5GB free GPU.
- v18: clean kernel, reached scoring, died on my own bug (_entropy_nats
  self-reference — it's a class attr not module global). v19: fixed, but
  ran in v18's dirty kernel (7.5GB held).
- v20 (clean kernel + fixed dbg flag): **SCORING WORKS.** [dbg] memory flat:
  7.68 → 7.70 → 7.69GB across decoder+CE at B=2, L=1079. Progress to
  [real] batch 40/470 at ~6s/batch. THEN OOM: single 3.11GiB alloc — a
  ~3.5k-token chunk batch; eager attention is O(L^2).
- v21 (running): trim chunks to 6000 chars + length-sorted batches
  (homogeneous memory profile). ETA ~100 min for both passes.

## 2026-09-27 ~04:00 — SMOKE VERDICT (v21 COMPLETED)

**Result: NAMI gold rank 49/61, BM25 1/61** (astropy-12907, separable.py).
939 chunks / 3730s / 3972ms per chunk / boundary_shifted=3 / flat 7.69GB.

Diagnostics (same kernel, post-hoc — free re-ranking from stored scores):
- h_real alone: gold 16/61. h_null alone: gold 15/61 (!). So the gold file
  is intrinsically low-entropy code; conditioning signal partially reflects
  naturalness. The null cancels it correctly — by design.
- Contrast ranking variants: max-bpt 49, max-bits 41, sum-bits 32,
  length-floored 41. All lose. Distractors (fits/header.py, utils/data.py)
  absorb more issue-conditioned info than gold on this instance.
- Interpretations + tomorrow plan: see MORNING_STATUS.md (n=1 statistics,
  null-domain contamination [matplotlib null vs astropy code], NF4 noise,
  instruct-likelihood pathology).

Artifacts: experiments/smoke_v21_chunks.csv (996 rows), smoke_v21_meta.json.
Teardown executed per user instruction: runtime.unassign(), tabs closed,
Chrome closed, laptop shutdown with 60s grace.
- Post-smoke plan: optional 3-instance mini-run, then closing sequence
  (document → disconnect runtime → close tabs → close Chrome → shutdown PC
  per user's explicit instruction).

Process improvements adopted (per user): preflight asserts before expensive
steps; per-10-batch progress prints from scorer; 10s change-detecting watcher;
verify notebook is actually Executing rather than trusting the fire call.

Model plan update: torchao int8_weight_only on T4 for smoke; bf16 sharded on
Kaggle 2xT4 for paper numbers (avoids quantization noise entirely).

## 2026-11-01 — Strengthening round (reviewer: "loses on Generalizability + Empirical Validation")
External review verdict: budget artifact (union@5=10 files vs BM25@10: 20 vs 19), no real
fusion, no dense baseline, n=25/2 repos, contamination unquantified, 11620 = post-hoc variant,
determinism thin (1 instance), no agent/resolve rate. Fix plan = 4 phases, deadline Nov 12.

Artifacts (all persistent this time):
- src/nami/stats.py — RRF/interleave fusion (pre-registered primary RRF(bm25,bpt) k=60),
  bootstrap CIs (10k, seed 20261101), paired Wilcoxon (reciprocal ranks), exact McNemar.
  Smoke-tested on synthetic data.
- src/build_cells.py — persistent cell builder (embeds nami pkg as JSON payload + injects
  DONE_SEED with regex-verified single sub). Usage: python src/build_cells.py <N> --seed <json>
- src/colab_cell_10_orderings.py — Phase 1: re-score original 25 (SAME pools/rng → doubles as
  25-inst cross-VM determinism check), CORRECT bm25 file agg, dumps full orderings top-61.
  Output: ORD_ROW lines + /content/orderings25.jsonl.
- src/colab_cell_11_rerank.py — Phase 2: ~120 inst stratified 10 repos (round-robin so partial
  runs stay balanced; zlib.crc32 seed per repo), whole-repo BM25 → top-25 files → NAMI rerank,
  dense baseline MiniLM whole-repo, NF4 piggyback (1 astropy + 3 django) at end.
  Output: ORD_ROW + /content/rerank_v1.jsonl (+NF4_ROW).
- src/colab_cell_12_probe.py — contamination probe: issue-only 10-way MC file ID on original 25
  (chance 10%), open-ended for 3 rescues. Output: PROBE_ROW + /content/probe_v1.jsonl.
- src/colab_cell_13_patchgen.py — Phase 3 agent: django-only ≤30 inst × 2 conditions
  (bm25 top-5 vs RRF-fused top-5, identical context rule: 2 largest chunks/file, 1500c),
  one-turn gemma-4-12b-it diff → git apply → tests/runtests.py F2P. Clean-tree baseline per
  instance (env_fail guard). Output: PATCH_ROW + /content/patchgen_v1.jsonl. Risk: old-django-on-
  py3.11 compat; fallback = labeled proxy metric.

Execution order: reconnect Colab → cell 00 setup (fresh VM) → cell 10 (~2.5h A100) → cell 11
(~5-7h A100, checkpointed, can span sessions) → cell 12 (~20min) → local fusion/stats → cell 13
(~2-3h A100) → Phase 4 rewrite ≤3000 words. Quota: may need one $9.99 top-up (A100 ≈ 11 u/h).
Status: cells built+compiled; bridge down (needs user to open notebook + click Connect).

## 2026-11-01 (evening) — v2 research changes (scorer bug fix, QLM + code-embedder baselines)
- **SCORER BUG FIXED** (nami/scorer.py `_entropy_nats`): the chunk-token mask ran
  to the end of the padded batch, so right-padding targets' NLL was averaged into
  every shorter row. Fix multiplies by attn[:,1:]. Pre-fix behaviour kept behind
  `scorer.count_padding = True` (legacy numbers reproducible). All Oct-19 numbers
  and the running old cell 10 are pre-fix; v2 cells use the fixed scorer.
- **QLM baseline** `NAMIScorer.score_qlm()` = mean log2 P(issue | chunk), Zhuang et
  al. 2310.13243, same model/decoder path/batching. Since log P(I) is constant per
  instance, QLM ranks by PMI(I;C) from the opposite direction -> the control that
  decides whether issue-conditioning of the CODE (NAMI) earns its keep.
- **cell 11 -> v2** (out: /content/rerank_v2.jsonl): + CodeRankEmbed (nomic-ai, 137M,
  trust_remote_code, query prefix "Represent this query for searching relevant code: "),
  candidate pool = BM25 top-20 ∪ CodeRankEmbed top-20 (raises recall ceiling; v1 was
  BM25 top-25 only), QLM pass on same pool, recall ceilings @5..100, issue feats,
  top-100 orders. COST: 3 passes over up to 40 files -> est. ~1.5-2x v1 per instance;
  trim REPO_QUOTA or CAND_* if A100 budget is short.
- **cell 10 -> v2** (out: /content/orderings25_v2.jsonl): + QLM, + issue feats,
  + LEGACY_CHECK (12907, 11564, 11815 re-scored with count_padding=True vs Oct-19 ranks).
- **nami/issue_feats.py** (new; embedded by build_cells): regex issue-style features +
  pre-registered style label (identifier / mixed / prose; thresholds fixed before results).
- **nami/stats.py**: rank_of(miss_rank) for partial reranked orders; fusions + RRF(bm25,qlm),
  RRF(bm25,bpt,qlm), RRF(bm25,coderank), RRF(bm25,coderank,bpt); fusion_table (equal-budget
  McNemar vs BM25@k), ceiling_table, router_analysis (Spearman, one-sided MWU). Smoke OK.
- Verified: both v2 cells build via build_cells (to a scratch dir, live *_built.py untouched)
  and py_compile; pyflakes: no undefined names.
- **Number corrections**: union@5 = 18/25 with pre-registered per-token NAMI (20/25 only when
  taking the better of bpt/bits per instance, post hoc); union@10 21 (22). BM25 file top-1 9,
  mean 7.8. Fixed in WRITEUP/VERDICT/REPORT_DAY3. Citations [1] (Zhuang, 2310.13243) and
  [5] (Karampatsis, 1911.04621) titles corrected.
- **Security**: HF token removed from experiments/kernel.py + build_kernel.py (now requires
  env/Secrets). Token still present in old .colab-bridge/*.json scratch logs -> REVOKE IT.
- Old (pre-fix) cell 10 is running on Colab now; first 2 rows reproduce Oct-19 ranks exactly
  (12907: 42/45; 14182: 21/40/36, bm25 6) -> extends cross-VM determinism toward 25 instances.
- Backups: experiments/{scorer,stats,cell11}_backup_20261101.py.

## 2026-11-02 ~01:50 — persistence hardening
- notes/RESULTS_V2.md written: full v2 25-instance tables (equal-budget, fusion+McNemar,
  rescue decomposition, honest interpretation) — the numbers are no longer chat-only.
- backups/: rolling 10-min snapshots of experiments/*.jsonl + RESULTS_V2 + night_state +
  orchestrator tail (keeps newest 12). Independent of bridge/orchestrator processes.
- Recovery chain proven tonight: VM-disk /content/*.jsonl -> dump cell -> local artifact
  (used for orderings25_v2.jsonl after the notebook rollback).

## 2026-11-02 02:55 — PAUSED (user request) after ~12/61 scaled instances
Stored: orderings25.jsonl (25 v1), orderings25_v2.jsonl (25), rerank_v2.jsonl (12 ok
+ 4 err incl. 2 fresh OOMs to retry), RESULTS_V2.md, backups/final_pause_0255/.
Resume: bridge token unchanged (port 7898); seed_c11.json (12 ids); build_cells 11
--seed; update cell uKUS8xy2yH2u; run. Cell-12 probe + cell-13 pending (cell-13
still needs rerank_v2 path bump from code session). Orchestrator/backup loops
stopped; bridge process left alive for tomorrow's reconnect.

## 2026-11-07 13:15 — GPU DISCONNECTED, EXPERIMENT PHASE CLOSED
All 6 experiments complete, all artifacts frozen (20-file snapshot in backups/final_freeze_1300/).
Watchers stopped; bridge left running for potential reconnect. Colab runtime disconnected by user.
Remaining schedule: Nov 8-9 writing, Nov 10 number proofread, Nov 11 submit, Nov 12 buffer.

---

## Handoff and decisions log (Nov 1 – Nov 7)

# HANDOFF_V2 — code changes for the v2 runs (2026-11-01)

Owner split (agreed): **Colab execution + WRITEUP/PDF = execution session.**
**`src/` + analysis = code session.** Don't edit the other side's files; leave a
note here instead.

## (a) `count_padding` — nobody needs to set it by hand

| Run | Scorer behaviour | What to do |
|---|---|---|
| v1 cell 10 (running now, built 20:18) | pre-fix, flag doesn't exist yet | nothing; it's the determinism artifact |
| cell 10 v2 main loop | **fixed** (default `count_padding=False`) | nothing |
| cell 10 v2 `LEGACY_CHECK` block | sets `True` itself, re-scores 12907 / 11564 / 11815, resets to `False` | read the `LEGACY_ROW` lines: `identical: true` expected |
| cell 11 v2 (incl. NF4 piggyback) | fixed | nothing |

Rule: **paper numbers = fixed scorer.** Pre-fix numbers are used only for the
determinism claim (v1 cell 10 vs Oct-19, and the LEGACY_CHECK rows).

## (b) Cost estimates (A100-40GB bf16), to be checked against the first rows

Basis: v1 = 2 passes over 61-file pools, about 4 min per instance (your
measurement). Each pass costs roughly the same, so a 3rd pass adds ~50%.

- **cell 10 v2:** ~6 min/instance × 25 ≈ **2.5 h**, plus LEGACY_CHECK
  3 × ~4 min ≈ 12 min. **Total ≈ 2.7 h.**
- **cell 11 v2: very uncertain, likely MORE than the "1.5–2×" I said earlier.**
  Candidate files are not random. BM25 and the embedder favour big files
  (django `db/models/query.py`, `sql/query.py` …), so 30–40 candidate files can
  hold as many chunks as or more than the 61 random files of v1. Other per-instance
  costs: whole-repo chunking, two dense encoders over every chunk in the repo
  (10–30k chunks; CodeRankEmbed is the slow one), and 3 Gemma passes.
  **Estimate: ~8–14 min/instance → 120 instances ≈ 16–28 h.** That is too much.
  **Measure before committing:** every `ORD_ROW` has `score_s` (NAMI's 2 passes),
  `qlm_s` (QLM pass) and `total_s`. After 3 instances, extrapolate.
  Knobs, in order of preference:
  1. `CAND_BM25 = CAND_DENSE = 15` (≈ 25–30 candidate files) → scoring time −25–35%.
  2. Cut `REPO_QUOTA` totals from 120 to ~80 (round-robin order keeps a partial
     run balanced across repos, so stopping early is also fine).
  3. `CAND_*=10` is the floor. Below that the rerank ceiling gets too low to be
     interesting. Check `ceilings`/`gold_in_cand` in the rows.

## (c) QLM baseline definition

- Method: `NAMIScorer.score_qlm(issue_text, chunks)` in `src/nami/scorer.py`.
- Scored sequence: `"### Code Snippet:\n{chunk}\n\n### Issue Description:\n" + issue`.
  Loss is taken over the **issue tokens only** (the chunk is the conditioning prefix).
- Score = mean log2 P(issue token | chunk, previous issue tokens). **Higher = better.**
  Direction is the reverse of NAMI (NAMI scores the code given the issue). Because
  log P(issue) is constant within an instance, QLM ranks chunks by the same PMI
  from the other side. This is Zhuang et al., EMNLP-F 2023, arXiv:2310.13243.
- File score = max over the file's chunks (same as NAMI and BM25). Order key in rows: `"qlm"`.
- Cost: **1 extra forward pass per chunk** (NAMI = 2: real + null). Same decoder
  path, same adaptive batching, same pad fix.
- Asymmetry to state in the paper: the issue is capped at **4000 chars** for QLM
  (`max_issue_chars`). NAMI uses the full issue as its prefix. The chunk cap is
  6000 chars for both.

## (d) CodeRankEmbed specifics

- `nomic-ai/CodeRankEmbed`, 137M params, loaded with `SentenceTransformer(...,
  trust_remote_code=True)`.
- Query = `"Represent this query for searching relevant code: " + issue[:4000]`.
  Documents = chunk text[:2000] chars, no prefix. `max_seq_length` capped at 512 by us.
- Cosine on normalised embeddings; file score = max chunk. Order key: `"coderank"`.
  MiniLM is kept as `"dense"` (secondary row only).
- **Pre-flight check:** loading is wrapped in try/except, so a failure is NOT fatal.
  It prints `dense model FAILED (non-fatal): coderank …`. The candidate pool then
  silently falls back to BM25 top-25 (v1 protocol) and the coderank columns are
  missing. **Look for `dense model ok: coderank` in the first output lines before
  leaving it overnight.** The remote code may need an extra package (for example
  `einops`). If you see an ImportError, `%pip install` the named package in the setup cell.

## (e) `stats.py` API changes (all backward compatible)

- `rank_of(order, gold, miss_rank=None)` — your multi-gold logic is unchanged. The new
  optional `miss_rank` gives every method the same worst rank when gold is missing
  from a partial (reranked) order. `evaluate` passes `max(len(order)) + 1` per instance.
- `evaluate(...)`: method list is now `bm25, nami_bpt, nami_bits, qlm, h_real, dense,
  coderank, fused, union_oracle`. Missing orders are skipped. A fusion whose orders
  are absent returns an empty `fused` instead of raising. `union_oracle` is still
  bm25 + nami_bpt only (the pre-registered variant).
- New `FUSIONS` keys: `RRF(bm25,qlm)`, `RRF(bm25,bpt,qlm)`, `RRF(bm25,coderank)`,
  `RRF(bm25,coderank,bpt)`. **Primary is still `RRF(bm25,bpt)`.** The key control
  for the novelty claim: `RRF(bm25,bpt)` vs `RRF(bm25,qlm)` at equal k.
- New `fusion_table(rows, ks=(1,5,10))`: every fusion vs BM25@k, with exact McNemar
  (+wins/−losses, p). Fusions whose orders are missing are skipped.
- New `ceiling_table(rows)`: `bm25@k`, `dense@k`, `coderank@k` recall plus
  `gold_in_candidate_pool` (rerank-v2 rows).
- New `router_analysis(rows, a="nami_bpt", b="bm25")`: per-style mean ΔRR,
  Spearman(identifier density, ΔRR), one-sided Mann–Whitney (prose > identifier),
  and a gold-stem-in-issue split. Style labels and thresholds are fixed in
  `nami/issue_feats.py` (pre-registered, do not tune after seeing results).

## Row schema additions (v2)

- **cell 10 v2** (`full-pool-v2`, `/content/orderings25_v2.jsonl`): orders gain `qlm`;
  rows gain `issue_feats`, `style`, `gold_stem_in_issue`.
  `LEGACY_ROW` lines are printed only (not written to the jsonl).
- **cell 11 v2** (`rerank-v2`, `/content/rerank_v2.jsonl`): orders = `bm25, dense,
  coderank` (whole repo, top-100 dumped) and `nami_bpt, nami_bits, h_real, qlm`
  (candidate pool only). Rows gain `n_cand_files`, `gold_in_cand`, `qlm_s`, `ceilings`,
  `issue_feats`, `style`, `gold_stem_in_issue`. Rows with `skip_reason` are terminal,
  not errors.

## Build

`python src/build_cells.py 10` and `... 11` — the embed now carries 5 package files
(adds `issue_feats.py`). Both v2 builds compile and pass pyflakes (verified in a
scratch build dir; your live `*_built.py` files were not touched).

## Execution notes (execution session, 2026-11-01 evening)

- v1 cell 10 running (determinism 5/5 exact so far); v2 cells 10+11 built and
  compile-verified locally (35,328 / 39,925 chars, issue_feats embedded).
- **FLAG for code session:** `colab_cell_13_patchgen.py` still reads
  `RERANK = "/content/rerank_v1.jsonl"` but cell 11 v2 writes `rerank_v2.jsonl`.
  Please bump (the fused-top5 logic is fine with partial nami orders).
- I will follow the measure-first rule for cell 11 v2: read `total_s`/`qlm_s`/
  `score_s` from the first 3 ORD_ROWs, then apply knobs (CAND 15+15, then quota
  ~80) per your preference order, and log the decision here.
- Decisions log: (empty — will fill as runs land)
- 02:00 execution session: knobs applied to cell 11 per handoff order (CAND 15+15;
  quota 120->81). src/ constants only, no logic touched. Cell-13 rerank_v2 path bump
  still PENDING (code session). LEGACY_ROW check output lost with the v2 cell
  (browser reload predating autosave); all 25 v2 rows recovered from VM disk
  (orderings25_v2.jsonl, ok=25 err=0, incl. django-11583). WS bridge patched
  max_size=64MB (was dying at 1MB on get_cells of big outputs).
- 02:55 execution session: run PAUSED by user request after ~11/61 instances (VM-side);
  orchestrator stopped; local artifact = experiments/rerank_v2.jsonl (8 ok rows,
  4 error rows incl. 2 fresh OOMs: matplotlib-23563, pylint-7080 — retry on resume).
  Resume procedure: bridge reconnect -> python src/build_cells.py 11 --seed
  .colab-bridge2/seed_c11.json -> update cell uKUS8xy2yH2u -> run. Missing ~3 rows
  re-score automatically (not in seed). Cell-12 probe NOT yet run.
- Nov 7 03:4x execution session: capture dedupe bug found+fixed — stale ERROR rows for
  an instance blocked its later OK row from being captured (dedupe by instance only).
  Purged all error rows from local rerank_v2.jsonl; retried instances re-capture fine.
  Root cause note for code session: capture_rows() should keep latest row per instance,
  not first.
- Nov 7 04:0x execution session: bumped RERANK path v1->v2 in colab_cell_13.py line 16
  (user-authorized; config constant only, no logic). Reason: cell 13 dead-on-arrival
  without it; pending with code session since Nov 1. Row-format compatibility checked
  (rerank-v2 rows carry orders.bm25 + orders.nami_bpt which is all cell 13 consumes).
  Comment header on line 2 still says v1 — cosmetic, left alone.
- Nov 7 10:1x (review session, user-delegated decision): **cell 13 condition changed: fused -> dense.**
  `src/colab_cell_13_patchgen.py` now compares BM25 top-5 vs MiniLM dense top-5 (`orders.dense`,
  present in all rerank-v2 rows); output -> `/content/patchgen_v2.jsonl`. Reason: RRF(bm25,bpt) is
  significantly worse than BM25 at top-5 (20 vs 28 of 46, p=0.008), dense is the strongest signal.
  Rebuilt `.colab-bridge2/cell13_built.py` (py_compile OK). Old versions kept:
  `experiments/cell13_backup_20261107_bm25_vs_fused.py`, `experiments/cell13_built_backup_20261107_bm25_vs_fused.py`.
  **ACTION for execution session: the copy in the Colab notebook is still the OLD one — push the
  rebuilt cell13_built.py into the notebook before firing cell 13.** Not pushed from here to avoid
  clashing with the running watcher on port 7898.
- Nov 7 10:1x (review session) two findings:
  (1) **Candidate pool is BM25 top-25, not "BM25 top-15".** CodeRankEmbed failed to load (no
  `coderank` order in any row), so candidate_files() fell back to BM25 top-25 for all 49 rows
  (verified: rerank pool set == bm25[:25] in 49/49; gold_in_cand == bm25@25 ceiling in 49/49).
  FACTS.md wording corrected. The planned BM25+dense union pool never ran.
  (2) **watcher_c11.py has a 3h cap; it started 07:57 so it exits ~10:57**, likely before cell 11
  finishes (ETA 11:00-11:30). Restart the watcher after 10:57 (or just dump
  /content/rerank_v2.jsonl from the VM at the end) so the last rows reach experiments/.
- Nov 7 ~11:15 execution session: FIXED chat_generate bug in cells 12+13 (execution-emergency,
  code session informed here): apply_chat_template returns BatchEncoding in this transformers
  version; generate() needs ["input_ids"]. Both cells rebuilt; probe rerunning (25 error rows
  retry automatically). Cell 13 also carried the same bug - caught BEFORE it ever ran.
- Nov 7 11:2x (review session): (a) **Probe caveat — please don't headline "memorization" yet.**
  Distractors are 9 RANDOM repo files, and there is no non-LLM control on the same options. Per
  orderings25_v2 `gold_stem_in_issue`, the gold file's stem appears in the issue on 15 of the 22
  correct picks; random distractors are usually off-topic. Reading the issue suffices; the
  open-ended 0/3 points AWAY from memorization. Conclusive test = hard distractors (BM25 top-9
  non-gold) + a BM25/dense reader on the same 10 options, or post-cutoff instances.
  WRITEUP_v3 §4.5 written with this framing.
  (b) **Cell 13 OOM**: django-11583 bm25+dense both OOM with only 685 MiB free of 39.5 GiB. Some other
  object is likely still holding GPU memory (e.g. cell 11's `scorer` keeps a reference to its model
  after cell 13 does `del model`). Error rows retry on rerun. Before the rerun:
  `del scorer; gc.collect(); torch.cuda.empty_cache()` (or restart the runtime, then rerun setup + cell 13).
  (c) WRITEUP_v3 numbers refreshed to the final 57 rows (my stats.py / inpool numbers match RESULTS_SCALED).
- Nov 7 11:3x (review session): **NEW cell 12b = hard-distractor contamination probe** (user-approved).
  Source `src/colab_cell_12b_probe_hard.py`, built `.colab-bridge2/cell12b_built.py` (34,307 chars,
  py_compile + pyflakes clean; helper logic unit-tested offline). build_cells.py only takes integer
  cell numbers, so it was built with the identical embed template (no change to build_cells.py).
  Same 25 instances + same prompt as cell 12; distractors = top non-gold files by BM25 and MiniLM
  (interleaved); records Gemma's pick plus BM25 / dense / path-token readers on the SAME 10 options.
  Frees `model/tok/scorer/dense_models` first (prints GPU free). Out: /content/probe_hard_v1.jsonl,
  PROBE_HARD_ROW lines, SUMMARY lines, "CELL12B COMPLETE". Needs rank_bm25 + sentence-transformers
  (already installed for cell 11). Est. 25 x (chunk + BM25 + MiniLM + 1 short generate) ≈ 15-30 min.
  **ACTION for execution session: add as a new cell and fire AFTER cell 13 finishes** (not concurrently).
- Nov 7 11:4x (review session): **NEW cell 14 = recall-first pool + Gemma listwise** (per execution
  session's contract). Source `src/colab_cell_14_listwise.py`, built via `python src/build_cells.py 14`
  -> `.colab-bridge2/cell14_built.py` (34,697 chars; pyflakes + py_compile clean).
  Contract: (1) chat_generate uses `apply_chat_template(...)["input_ids"]` (the cell-12/13 fix) from birth;
  (2) reads /content/rerank_v2.jsonl, skips error/skip_reason rows (latest ok per instance), pool =
  bm25[:25] ∪ dense[:25]; out /content/listwise_v1.jsonl + LIST_ROW lines, DONE_SEED = "instance|condition"
  keys; (3) row has gemma_top5 + gemma_top10 + ranks {gemma_listwise, gemma_fill, bm25, dense} +
  gold_in_gemma_top5. Pool is SHUFFLED per instance (fixed seed) so Gemma can't read the retrievers' order.
  Conditions run "paths" over all 57 first, then "paths_sigs" (paths + ≤15 chunk names/file; needs repo
  fetch). To skip sigs: set CONDITIONS = ("paths",). Frees model/tok/scorer/dense first.
  Offline test on the real rerank_v2.jsonl with a stubbed model: 57 rows, union pool 28-49 files
  (mean 41.8), gold in pool 51/57 (89%, matches RESULTS_SCALED); parsing/dedupe/rank asserts pass.
  Fire order agreed: cell 13 -> restart -> cell 13 retry -> cell 14 -> cell 12b (hard probe, ~20 min).
- Nov 7 ~11:50 (review session): **cell 13 v2 results are INVALID — harness rewritten as v3.**
  v2 log: 15/15 attempts "corrupt patch"; every "clean-tree F2P: fail" meant tests/ missing. Bugs:
  (1) parse_diff dropped context lines + trailing newline; (2) nami.data snapshot skips tests/ ->
  runtests.py absent; (3) snapshot not a git repo -> `git checkout` no-op (patches would leak bm25->dense);
  (4) F2P labels "test_x (mod.Class)" passed raw; (5) SWE-bench test_patch never applied.
  v3 (`src/colab_cell_13_patchgen.py`, built `.colab-bridge2/cell13_built.py`, 38,496 chars, pyflakes+compile OK):
  full tarball incl. tests as git repo; test_patch applied + committed as baseline; per-instance sanity
  (baseline must FAIL F2P, baseline+GOLD patch must PASS) -> env_ok, else no generation spent;
  git apply --recount -> fallback `patch --fuzz=3`; reset --hard + clean; labels converted; pip adds
  legacy-cgi (Django<4.2 needs `cgi`, gone in py3.13) + setuptools; MAX_NEW 1024; out /content/patchgen_v3.jsonl.
  TESTED on the user's machine against a real Django 4.2 checkout: download+git ok; label conversion ok;
  runtests passing test -> True (ran 5), failing test -> False; LLM-style diff parse+apply ok (also with
  wrong hunk counts via --recount); reset restores tree; OLD parser reproduces "corrupt patch" exactly.
  Test also caught+fixed a fetch bug (codeload dir name == target dir -> was deleted after extract).
  Not testable here: SWE-bench rows (HF blocked) and py3.13 — the gold sanity check covers that on Colab.
  **ACTION: interrupt v2 cell 13, push v3 cell13_built.py, run. Watch the first "env:" line: gold_pass must
  be True. If gold_pass=False on most instances, paste the gold_tail — do NOT spend GPU on generation.**
- Nov 7 12:1x (review session) — **cell 14 corrections before anything goes in the paper:**
  (1) **n=27 is NOT by design.** /content/rerank_v2.jsonl on the current VM only holds rows scored in this
  VM session (instances 32-61 minus OOMs); rows 1-31 exist only in local experiments/rerank_v2.jsonl (57 ok).
  Fix: upload the local 57-row file to /content/rerank_v2.jsonl and rerun cell 14 (checkpoint skips the 27).
  (2) **Compare on the SAME instances.** On these 27: BM25 top-5 9/27 (33%), dense 15/27 (56%), Gemma paths
  24/27 (89%); top-1 BM25 6, dense 9, Gemma 18 (paths) / 20 (sigs). Do NOT compare Gemma-on-27 with the
  57-instance BM25 51% / dense 65%. Gemma still wins clearly; report paired McNemar on identical instances.
  (3) **Hold the "partly memorization" sentence until cell 12b finishes** — probe v1 cannot separate
  memorisation from reading (random options; gold stem in issue on 15/22). Note paths-only ≈ paths+sigs:
  Gemma barely uses code, which is consistent with BOTH name-reading and memorisation. 12b decides.
  Also: cell "Zs-qlQ-A0bx_" above cell 14 is my neutralized duplicate (one comment line) — safe to delete.
- Nov 7 ~13:00 execution session: ALL EXPERIMENTS FROZEN. Final artifacts:
  experiments/{rerank_v2,orderings25,orderings25_v2,probe_v1,probe_hard_v1,patchgen_v3,
  listwise_complete,nf4_scaled_rows}.jsonl; notes/{RESULTS_SCALED,RESULTS_LISTWISE,
  RESULTS_V2,INPOOL_ANALYSIS,FACTS}.md. Review-session wording corrections applied to
  FACTS.md (no 'contamination confirmed'; upper-bound framing). Cell 14 rerun used the
  pushed full 57-row file; SUMMARY lines verified against local merge (114 rows).
- Nov 7 ~13:00 (review session): **two independent judge reviews done; all their key claims re-verified on the raw files.**
  Paper ~3.1/5, Resource ~2.4/5 (≈3.0 if fully packaged). WRITEUP_v3 rewritten as v3.1 (backup:
  backups/WRITEUP_v3_pre_review_20261107.md). Key changes: (1) §4.4 overclaim removed — hard probe split by
  gold_stem_in_issue: Gemma 15/16 when the issue names the file, 3/9 when it does not; path reader counted 11 ties
  as misses; hard distractors are adversarial to BM25/dense by construction. (2) New fair baselines: RRF(BM25,dense)
  top-1/5/10 20/33/42, per-instance oracle best-of 31/43/48 — Gemma top-5 48 beats even the oracle. (3) Name-free
  issues (n=23): Gemma top-5 18 vs dense 14; top-1 11/14 vs 8. (4) Missing citations added: SWE-Bench Illusion
  (2506.12286), SweRank (2505.07849), CoRNStack (2412.01007), RankGPT (2304.09542), arXiv:2512.10218.
  (5) Timing corrected to median 323 s (IQR 171-455); union rescoring ≈10 A100-h; determinism scoped to gold ranks,
  pre-fix scorer; "pre-registered" removed. (6) **Sixth harness bug**: 6/13 apply failures = model diffs without
  a/ b/ prefixes stripped by -p1 (disclosed in §4.5; fix = try -p0 when headers lack a/). stats.py hard-codes
  method names + default path orderings_v1.jsonl (release blocker for the Resource entry).
- Nov 7 ~13:00 (review session): **Colab reconnected (user-approved); patch demo v4 RUNNING in cell `mGODmvUL_TIk`**
  -> /content/patchgen_v4.jsonl. Please do not start other GPU cells until it prints "CELL13 COMPLETE".
  v4 = v3 + strip-level fix (try -p0 when diff headers lack a/ b/), GNU patch fuzz 3->1 (fuzz 3 applied a garbage
  hunk in a local test), + third condition: Gemma (paths+names) top-5 as context. Same 12 django instances as v3.
  Self-contained (candidates + nami.chunker embedded). Source: src/colab_cell_13b_patchgen_v4.py (pyflakes clean).
  Gold-in-top5 over the 12: bm25 3, dense 7, gemma 10. Locally tested applier: a/b, no-prefix, shifted line numbers
  apply; garbage context rejected. Needs pulling to experiments/patchgen_v4.jsonl when done.

---

## Run measurements referenced in the writeups

- Whole-repository run (Nov 7, A100-SXM4-40GB, bf16), cell output: `model loaded in 85s | 24.6GB` (GPU memory after
  loading Gemma-4-12B-it; scoring stayed flat at this level). Free GPU before load: 42.0 GB.
- NF4 smoke test on a free-tier T4 (Sep 27): flat 7.7 GB across decoder and cross-entropy (see the worklog, v20/v21).
