# Released as part of the localization audit kit. Originally a Google Colab cell; runs as a script from the
# release root (python harness/<name>.py) with a GPU. Working files go to $RECALLFIRST_WORKDIR (default /content).
import os as _os
WORKDIR = _os.environ.get("RECALLFIRST_WORKDIR", "/content")
_os.makedirs(WORKDIR, exist_ok=True)
# === Phase 1: re-score original 25, dump FULL per-file orderings ===
# Same pools as the bf16 run (identical rng seeding). v2 (2026-11-01): scorer
# pad-mask FIX is on, so primary numbers will differ from the Oct-19 run; a
# legacy (count_padding=True) re-score of 3 instances at the end reproduces the
# old rows exactly -> cross-VM determinism check. + QLM baseline + issue feats.
# (Original note:) Correct BM25 file aggregation
# (max chunk score per file) is computed here -- the harness's inline variant
# was the chunk-position bug we already corrected offline.
import os, sys, time, json, random, gc
import torch

from nami.scorer import NAMIScorer
from nami.chunker import chunk_repo
from nami.data import gold_files, fetch_repo_snapshot
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from rank_bm25 import BM25Okapi
from nami.issue_feats import issue_features, gold_stem_in_issue, style_label

MODEL_ID = "google/gemma-4-12b-it"
N_INSTANCES = 25
OUT_PATH = WORKDIR + "/orderings25_v2.jsonl"
LEGACY_CHECK = {  # instance -> Oct-19 bf16 (nami_bpt, nami_bits) ranks, pre-fix scorer
    "astropy__astropy-12907": (42, 45),
    "django__django-11564": (4, 14),
    "django__django-11815": (3, 9),
}
MAX_CHUNK_CHARS = 6000
TOP_DUMP = 61
SCI_OWNERS = {"astropy", "matplotlib", "scipy", "numpy", "pandas", "sympy",
              "networkx", "scikit-learn", "statsmodels"}
DONE_SEED = []

# ---- preflight ----
import shutil, psutil
assert torch.cuda.is_available(), "no GPU"
print(f"preflight: GPU={torch.cuda.get_device_name(0)} | RAM={psutil.virtual_memory().available/1e9:.1f}GB", flush=True)
import transformers.utils.import_utils as _iu
for _fn in ("is_torchao_available", "is_bitsandbytes_available"):
    _f = getattr(_iu, _fn, None)
    if hasattr(_f, "cache_clear"):
        _f.cache_clear()
import transformers.modeling_utils as _mu
_mu.caching_allocator_warmup = lambda *a, **k: None
print("preflight: OK", flush=True)

ds = load_dataset("princeton-nlp/SWE-bench_Lite")
split = "test" if "test" in ds else list(ds.keys())[0]
rows = list(ds[split])

def pick_null_nonsci(rows, issue_text, exclude_repo):
    target = len(issue_text)
    lo, hi = 0.7 * target, 1.3 * target
    best, best_gap = None, float("inf")
    for row in rows:
        owner = row["repo"].split("/")[0]
        if row["repo"] == exclude_repo:
            continue
        n = len(row["problem_statement"])
        gap = abs(n - target) + (20000 if owner in SCI_OWNERS else 0)
        if lo <= n <= hi and gap < best_gap:
            best, best_gap = row, gap
    if best is None:
        for row in rows:
            owner = row["repo"].split("/")[0]
            if row["repo"] == exclude_repo:
                continue
            gap = abs(len(row["problem_statement"]) - target) + (20000 if owner in SCI_OWNERS else 0)
            if gap < best_gap:
                best, best_gap = row, gap
    return best

# ---- checkpoint ----
done_ids = set(DONE_SEED)
if os.path.exists(OUT_PATH):
    for line in open(OUT_PATH):
        try:
            r = json.loads(line)
            if "error" not in r:
                done_ids.add(r["instance"])
        except Exception:
            pass
print(f"checkpoint: {len(done_ids)} done", flush=True)

# ---- model ----
for _v in ("model", "scorer", "tok"):
    if _v in globals():
        del globals()[_v]
gc.collect(); torch.cuda.empty_cache()
sys.last_traceback = None; sys.last_value = None; sys.last_type = None
print("GPU free before load:", round(torch.cuda.mem_get_info()[0]/1e9, 1), "GB", flush=True)
t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = None
for attempt in range(3):
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto",
            attn_implementation="eager")
        break
    except Exception as e:
        print(f"load attempt {attempt+1} failed: {type(e).__name__} {str(e)[:120]}", flush=True)
        model = None; gc.collect(); torch.cuda.empty_cache(); time.sleep(5)
assert model is not None, "model load failed"
scorer = NAMIScorer(model, tok)
print(f"model loaded in {time.time()-t0:.0f}s | {round(torch.cuda.memory_allocated()/1e9,1)}GB", flush=True)

def file_orders(chunks, results, issue_text):
    """Full per-file orderings from per-chunk scores + CORRECT BM25 file agg."""
    def agg(key_fn, better=max):
        fs = {}
        for c, r in zip(chunks, results):
            v = key_fn(r)
            f = c.repo_path
            fs[f] = better(fs.get(f, -1e9 if better is max else 1e9), v)
        return [f for f, _ in sorted(fs.items(), key=lambda kv: -kv[1] if better is max else kv[1])]
    orders = {
        "nami_bpt": agg(lambda r: r.contrast_bpt),
        "nami_bits": agg(lambda r: r.contrast_bits),
        "h_real": agg(lambda r: -r.h_real),
    }
    bm = BM25Okapi([c.text[:MAX_CHUNK_CHARS].lower().split() for c in chunks])
    cs = bm.get_scores(issue_text.lower().split())
    fsc = {}
    for c, s in zip(chunks, cs):
        fsc[c.repo_path] = max(fsc.get(c.repo_path, -1e9), float(s))
    orders["bm25"] = [f for f, _ in sorted(fsc.items(), key=lambda kv: -kv[1])]
    return orders

n_ok = n_err = 0
for idx in range(N_INSTANCES):
    inst = rows[idx]
    iid = inst["instance_id"]
    if iid in done_ids:
        print(f"[{idx+1}/25] {iid} SKIP (done)", flush=True)
        continue
    tI = time.time()
    try:
        gold = gold_files(inst["patch"])
        gold_set = set(gold)
        repo_dir = fetch_repo_snapshot(inst["repo"], inst["base_commit"], WORKDIR + "/repos")
        all_chunks = chunk_repo(repo_dir, "python")
        rng = random.Random(idx)          # SAME seeding as the original run
        other = sorted({c.repo_path for c in all_chunks} - gold_set)
        distract = set(rng.sample(other, k=min(60, len(other))))
        chunks = [c for c in all_chunks if c.repo_path in gold_set or c.repo_path in distract]
        if not any(c.repo_path in gold_set for c in chunks):
            raise RuntimeError("gold file not chunked")
        null_inst = pick_null_nonsci(rows, inst["problem_statement"], inst["repo"])
        pairs = sorted(((c, c.text[:MAX_CHUNK_CHARS]) for c in chunks), key=lambda p: len(p[1]))
        chunks = [c for c, _ in pairs]
        texts = [t for _, t in pairs]
        t0 = time.time()
        results = scorer.score_chunks(inst["problem_statement"],
                                      null_inst["problem_statement"], texts,
                                      batch_size=6, with_uncond=False)
        orders = file_orders(chunks, results, inst["problem_statement"])
        qlm = scorer.score_qlm(inst["problem_statement"], texts)
        fq = {}
        for c, v in zip(chunks, qlm):
            fq[c.repo_path] = max(fq.get(c.repo_path, -1e9), v)
        orders["qlm"] = [f for f, _ in sorted(fq.items(), key=lambda kv: -kv[1])]
        feats = issue_features(inst["problem_statement"])
        def rk(o):
            return next((i + 1 for i, f in enumerate(o) if f in gold_set), -1)
        row = {"instance": iid, "repo": inst["repo"], "protocol": "full-pool-v2",
               "gold": gold[0] if len(gold) == 1 else gold,
               "null_repo": null_inst["repo"], "n_chunks": len(chunks),
               "n_files": len({c.repo_path for c in chunks}),
               "score_s": round(time.time() - t0),
               "ranks": {k: rk(v) for k, v in orders.items()},
               "issue_feats": feats, "style": style_label(feats),
               "gold_stem_in_issue": gold_stem_in_issue(inst["problem_statement"], gold),
               "orders": {k: v[:TOP_DUMP] for k, v in orders.items()}}
        with open(OUT_PATH, "a") as f:
            f.write(json.dumps(row) + "\n")
        print("ORD_ROW " + json.dumps(row), flush=True)
        print(f"[{idx+1}/25] {iid} done {(time.time()-tI)/60:.0f}m | ranks={row['ranks']}", flush=True)
        n_ok += 1
    except Exception as e:
        row = {"instance": iid, "repo": inst.get("repo"),
               "error": f"{type(e).__name__}: {str(e)[:200]}"}
        with open(OUT_PATH, "a") as f:
            f.write(json.dumps(row) + "\n")
        print("ORD_ROW " + json.dumps(row), flush=True)
        print(f"[{idx+1}/25] {iid} ERROR: {type(e).__name__} {str(e)[:150]}", flush=True)
        n_err += 1

print(f"CELL10 MAIN COMPLETE. ok={n_ok} err={n_err} (+{len(done_ids)} resumed)", flush=True)

# ---- legacy determinism check: pre-fix scorer must reproduce Oct-19 ranks ----
scorer.count_padding = True
for iid, (old_bpt, old_bits) in LEGACY_CHECK.items():
    try:
        idx = next(i for i in range(N_INSTANCES) if rows[i]["instance_id"] == iid)
        inst = rows[idx]
        gold_set = set(gold_files(inst["patch"]))
        repo_dir = fetch_repo_snapshot(inst["repo"], inst["base_commit"], WORKDIR + "/repos")
        all_chunks = chunk_repo(repo_dir, "python")
        rng = random.Random(idx)
        other = sorted({c.repo_path for c in all_chunks} - gold_set)
        distract = set(rng.sample(other, k=min(60, len(other))))
        chunks = [c for c in all_chunks if c.repo_path in gold_set or c.repo_path in distract]
        null_inst = pick_null_nonsci(rows, inst["problem_statement"], inst["repo"])
        pairs = sorted(((c, c.text[:MAX_CHUNK_CHARS]) for c in chunks), key=lambda p: len(p[1]))
        chunks = [c for c, _ in pairs]; texts = [t for _, t in pairs]
        res = scorer.score_chunks(inst["problem_statement"], null_inst["problem_statement"],
                                  texts, batch_size=6, with_uncond=False)
        o = file_orders(chunks, res, inst["problem_statement"])
        rk = lambda od: next((i + 1 for i, f in enumerate(od) if f in gold_set), -1)
        new = (rk(o["nami_bpt"]), rk(o["nami_bits"]))
        print("LEGACY_ROW " + json.dumps({"instance": iid, "old": [old_bpt, old_bits],
              "new": list(new), "identical": new == (old_bpt, old_bits)}), flush=True)
    except Exception as e:
        print(f"LEGACY_ROW error {iid}: {type(e).__name__} {str(e)[:150]}", flush=True)
scorer.count_padding = False
print("CELL10 COMPLETE", flush=True)
