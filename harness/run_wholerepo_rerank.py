# Released as part of the localization audit kit. Originally a Google Colab cell; runs as a script from the
# release root (python harness/<name>.py) with a GPU. Working files go to $RECALLFIRST_WORKDIR (default /content).
import os as _os
WORKDIR = _os.environ.get("RECALLFIRST_WORKDIR", "/content")
_os.makedirs(WORKDIR, exist_ok=True)
# === Phase 2: scaled rerank protocol v2 ===
# Whole-repo BM25 + whole-repo dense (MiniLM, CodeRankEmbed) -> candidate pool =
# BM25 top-20 UNION CodeRankEmbed top-20 -> rerank pool with NAMI (contrast) AND
# QLM (Zhuang-style log P(issue|chunk), same model, same pool). ~120 instances
# stratified across 10 repos, round-robin so partial runs stay balanced.
# Rows carry recall ceilings, issue-style features, full top-100 orderings.
# v2 changes (2026-11-01): + QLM baseline, + CodeRankEmbed, candidate union,
# ceilings, issue feats, scorer pad-mask fix (see nami/scorer.py).
import os, sys, time, json, random, gc, zlib
import torch

from nami.scorer import NAMIScorer
from nami.chunker import chunk_repo
from nami.data import gold_files, fetch_repo_snapshot
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from rank_bm25 import BM25Okapi
from nami.issue_feats import issue_features, gold_stem_in_issue, style_label

MODEL_ID = "google/gemma-4-12b-it"
DENSE_MODELS = {  # name -> (hf id, query prefix); doc side gets no prefix
    "dense": ("sentence-transformers/all-MiniLM-L6-v2", ""),
    "coderank": ("nomic-ai/CodeRankEmbed",
                 "Represent this query for searching relevant code: "),
}
OUT_PATH = WORKDIR + "/rerank_v2.jsonl"
MAX_CHUNK_CHARS = 6000
CAND_BM25 = 15      # scope cut 2026-11-02 (capped run; HANDOFF knob order: 15+15 first)
CAND_DENSE = 15     # ... UNION CodeRankEmbed top-15 (falls back to BM25 top-25)
CAND_DENSE_KEY = "coderank"
CEIL_KS = (5, 10, 20, 25, 50, 100)
TOP_DUMP = 100
SCI_OWNERS = {"astropy", "matplotlib", "scipy", "numpy", "pandas", "sympy",
              "networkx", "scikit-learn", "statsmodels"}
REPO_QUOTA = {"django/django": 20, "astropy/astropy": 8, "sympy/sympy": 8,
              "xarray-dev/xarray": 7, "pylint-dev/pylint": 7, "pallets/flask": 7,
              "psf/requests": 7, "scikit-learn/scikit-learn": 7,
              "matplotlib/matplotlib": 5, "zzzeek/sqlalchemy": 5}
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
from transformers.utils import is_bitsandbytes_available as _bnb_ok
import transformers.modeling_utils as _mu
_mu.caching_allocator_warmup = lambda *a, **k: None
print(f"preflight: OK | bnb={_bnb_ok()}", flush=True)

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

# ---- stratified, round-robin instance selection (deterministic) ----
per_repo = {}
for repo, q in REPO_QUOTA.items():
    cand = sorted([r for r in rows if r["repo"] == repo], key=lambda r: r["instance_id"])
    rng = random.Random(zlib.crc32(repo.encode()))
    per_repo[repo] = rng.sample(cand, min(q, len(cand)))
sel, queues = [], {r: list(v) for r, v in per_repo.items()}
while any(queues.values()):
    for repo in REPO_QUOTA:
        if queues[repo]:
            sel.append(queues[repo].pop(0))
plan = {r: len(v) for r, v in per_repo.items()}
print("INSTANCE_PLAN " + json.dumps(plan), flush=True)
print(f"selected {len(sel)} instances", flush=True)

# ---- checkpoint ----
done_ids = set(DONE_SEED)
if os.path.exists(OUT_PATH):
    for line in open(OUT_PATH):
        try:
            r = json.loads(line)
            if "error" not in r:  # terminal rows: success or skip_reason
                done_ids.add(r["instance"])
        except Exception:
            pass
print(f"checkpoint: {len(done_ids)} done", flush=True)

# ---- models ----
for _v in ("model", "scorer", "tok", "dense_model", "dense_tok", "dense_models"):
    if _v in globals():
        del globals()[_v]
gc.collect(); torch.cuda.empty_cache()
sys.last_traceback = None; sys.last_value = None; sys.last_type = None
print("GPU free before load:", round(torch.cuda.mem_get_info()[0]/1e9, 1), "GB", flush=True)

from sentence_transformers import SentenceTransformer
dense_models = {}
for _name, (_hf, _qp) in DENSE_MODELS.items():
    try:
        _m = SentenceTransformer(_hf, device="cuda", trust_remote_code=True)
        _m.max_seq_length = min(512, _m.max_seq_length or 512)
        dense_models[_name] = (_m, _qp)
        print(f"dense model ok: {_name} ({_hf})", flush=True)
    except Exception as e:
        print(f"dense model FAILED (non-fatal): {_name} {type(e).__name__} {str(e)[:150]}", flush=True)

# encode-time probe (2026-11-02): a model can load fine but crash on encode
# (nomic custom code vs transformers 5.x). Drop those here so the run degrades
# gracefully instead of erroring every instance.
for _name in list(dense_models):
    try:
        dense_models[_name][0].encode("def f():\n    pass", normalize_embeddings=True)
    except Exception as e:
        print(f"dense model DROPPED at encode probe: {_name} {type(e).__name__} {str(e)[:120]}", flush=True)
        del dense_models[_name]

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

def bm25_file_order(chunks, issue_text):
    bm = BM25Okapi([c.text[:MAX_CHUNK_CHARS].lower().split() for c in chunks])
    cs = bm.get_scores(issue_text.lower().split())
    fsc = {}
    for c, s in zip(chunks, cs):
        fsc[c.repo_path] = max(fsc.get(c.repo_path, -1e9), float(s))
    return [f for f, _ in sorted(fsc.items(), key=lambda kv: -kv[1])]

def dense_file_order(name, chunks, issue_text):
    dm, qprefix = dense_models[name]
    qv = dm.encode(qprefix + issue_text[:4000], normalize_embeddings=True)
    ce = dm.encode([c.text[:2000] for c in chunks], batch_size=128,
                   normalize_embeddings=True, show_progress_bar=False)
    sims = ce @ qv
    fsc = {}
    for c, s in zip(chunks, sims):
        fsc[c.repo_path] = max(fsc.get(c.repo_path, -1e9), float(s))
    return [f for f, _ in sorted(fsc.items(), key=lambda kv: -kv[1])]

def nami_orders(chunks, results):
    def agg(key_fn):
        fs = {}
        for c, r in zip(chunks, results):
            fs[c.repo_path] = max(fs.get(c.repo_path, -1e9), key_fn(r))
        return [f for f, _ in sorted(fs.items(), key=lambda kv: -kv[1])]
    return {"nami_bpt": agg(lambda r: r.contrast_bpt),
            "nami_bits": agg(lambda r: r.contrast_bits),
            "h_real": agg(lambda r: -r.h_real)}

def max_agg(chunks, scores):
    fs = {}
    for c, v in zip(chunks, scores):
        fs[c.repo_path] = max(fs.get(c.repo_path, -1e9), v)
    return [f for f, _ in sorted(fs.items(), key=lambda kv: -kv[1])]

def candidate_files(bm25_order, dense_orders):
    """Ordered union: BM25 top-CAND_BM25 then unseen CodeRankEmbed top-CAND_DENSE."""
    cand = list(bm25_order[:CAND_BM25])
    if CAND_DENSE_KEY in dense_orders:
        for f in dense_orders[CAND_DENSE_KEY][:CAND_DENSE]:
            if f not in cand:
                cand.append(f)
    else:  # dense model unavailable: plain BM25 top-25 (v1 protocol)
        cand = list(bm25_order[:25])
    return cand

n_ok = n_skip = n_err = 0
for prog, inst in enumerate(sel, 1):
    iid = inst["instance_id"]
    if iid in done_ids:
        print(f"[{prog}/{len(sel)}] {iid} SKIP (done)", flush=True)
        continue
    tI = time.time()
    try:
        gold = gold_files(inst["patch"])
        gold_set = set(gold)
        repo_dir = fetch_repo_snapshot(inst["repo"], inst["base_commit"], WORKDIR + "/repos")
        all_chunks = chunk_repo(repo_dir, "python")
        n_repo_files = len({c.repo_path for c in all_chunks})
        if not any(c.repo_path in gold_set for c in all_chunks):
            row = {"instance": iid, "repo": inst["repo"], "protocol": "rerank-v2",
                   "skip_reason": "gold_not_in_corpus", "gold": gold[0] if len(gold) == 1 else gold}
            with open(OUT_PATH, "a") as f:
                f.write(json.dumps(row) + "\n")
            print("ORD_ROW " + json.dumps(row), flush=True)
            n_skip += 1
            continue
        issue = inst["problem_statement"]
        bm25_order = bm25_file_order(all_chunks, issue)
        dense_orders = {n: dense_file_order(n, all_chunks, issue) for n in dense_models}
        cand = candidate_files(bm25_order, dense_orders)
        cand_set = set(cand)
        rk_chunks = [c for c in all_chunks if c.repo_path in cand_set]
        null_inst = pick_null_nonsci(rows, inst["problem_statement"], inst["repo"])
        pairs = sorted(((c, c.text[:MAX_CHUNK_CHARS]) for c in rk_chunks), key=lambda p: len(p[1]))
        rk_chunks = [c for c, _ in pairs]
        texts = [t for _, t in pairs]
        t0 = time.time()
        results = scorer.score_chunks(inst["problem_statement"],
                                      null_inst["problem_statement"], texts,
                                      batch_size=6, with_uncond=False)
        score_s = round(time.time() - t0)
        t1 = time.time()
        qlm = scorer.score_qlm(issue, texts)
        qlm_s = round(time.time() - t1)
        orders = nami_orders(rk_chunks, results)
        orders["qlm"] = max_agg(rk_chunks, qlm)
        orders["bm25"] = bm25_order
        orders.update(dense_orders)
        def rk(o, cap=None):
            return next((i + 1 for i, f in enumerate(o) if f in gold_set), -1)
        def first_hit(o):
            return next((i + 1 for i, f in enumerate(o) if f in gold_set), 10**6)
        ceilings = {f"{m}@{k}": int(first_hit(o) <= k)
                    for m, o in [("bm25", bm25_order)] + list(dense_orders.items())
                    for k in CEIL_KS}
        feats = issue_features(issue)
        row = {"instance": iid, "repo": inst["repo"], "protocol": "rerank-v2",
               "gold": gold[0] if len(gold) == 1 else gold,
               "null_repo": null_inst["repo"], "n_files_repo": n_repo_files,
               "n_cand_files": len(cand), "gold_in_cand": bool(gold_set & cand_set),
               "n_chunks_scored": len(rk_chunks), "score_s": score_s, "qlm_s": qlm_s,
               "total_s": round(time.time() - tI),
               "ranks": {k: rk(v) for k, v in orders.items()},
               "ceilings": ceilings,
               "issue_feats": feats, "style": style_label(feats),
               "gold_stem_in_issue": gold_stem_in_issue(issue, gold),
               "orders": {k: v[:TOP_DUMP] for k, v in orders.items()}}
        with open(OUT_PATH, "a") as f:
            f.write(json.dumps(row) + "\n")
        print("ORD_ROW " + json.dumps(row), flush=True)
        print(f"[{prog}/{len(sel)}] {iid} done {(time.time()-tI)/60:.1f}m | {row['ranks']} | repo_files={n_repo_files}", flush=True)
        n_ok += 1
    except Exception as e:
        row = {"instance": iid, "repo": inst.get("repo"),
               "error": f"{type(e).__name__}: {str(e)[:200]}"}
        with open(OUT_PATH, "a") as f:
            f.write(json.dumps(row) + "\n")
        print("ORD_ROW " + json.dumps(row), flush=True)
        print(f"[{prog}/{len(sel)}] {iid} ERROR: {type(e).__name__} {str(e)[:150]}", flush=True)
        n_err += 1

print(f"PHASE2 MAIN COMPLETE. ok={n_ok} skip={n_skip} err={n_err} (+{len(done_ids)} resumed)", flush=True)

# ---- NF4 piggyback: 4 instances, quantization sensitivity ----
try:
    nf4_pick = [r for r in sel if r["repo"] == "astropy/astropy"][:1] + \
               [r for r in sel if r["repo"] == "django/django"][:3]
    for _v in ("model", "scorer"):
        if _v in globals():
            del globals()[_v]
    gc.collect(); torch.cuda.empty_cache()
    time.sleep(3)
    print("loading NF4 for piggyback...", flush=True)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16)
    nf4_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb, device_map="auto",
        attn_implementation="eager")
    nf4_scorer = NAMIScorer(nf4_model, tok)
    for prog, inst in enumerate(nf4_pick, 1):
        iid = inst["instance_id"]
        tI = time.time()
        gold = gold_files(inst["patch"]); gold_set = set(gold)
        repo_dir = fetch_repo_snapshot(inst["repo"], inst["base_commit"], WORKDIR + "/repos")
        all_chunks = chunk_repo(repo_dir, "python")
        bm25_order = bm25_file_order(all_chunks, inst["problem_statement"])
        dense_orders = {n: dense_file_order(n, all_chunks, inst["problem_statement"]) for n in dense_models}
        cand_set = set(candidate_files(bm25_order, dense_orders))
        rk_chunks = [c for c in all_chunks if c.repo_path in cand_set]
        null_inst = pick_null_nonsci(rows, inst["problem_statement"], inst["repo"])
        pairs = sorted(((c, c.text[:MAX_CHUNK_CHARS]) for c in rk_chunks), key=lambda p: len(p[1]))
        rk_chunks = [c for c, _ in pairs]; texts = [t for _, t in pairs]
        results = nf4_scorer.score_chunks(inst["problem_statement"],
                                          null_inst["problem_statement"], texts,
                                          batch_size=6, with_uncond=False)
        orders = nami_orders(rk_chunks, results)
        row = {"instance": iid, "repo": inst["repo"], "protocol": "rerank-v2-nf4",
               "ranks": {"nami_bpt": next((i+1 for i, f in enumerate(orders['nami_bpt']) if f in gold_set), -1),
                         "nami_bits": next((i+1 for i, f in enumerate(orders['nami_bits']) if f in gold_set), -1)}}
        print("NF4_ROW " + json.dumps(row), flush=True)
        print(f"NF4 [{prog}/{len(nf4_pick)}] {iid} {(time.time()-tI)/60:.1f}m | {row['ranks']}", flush=True)
except Exception as e:
    print(f"NF4 piggyback failed (non-fatal): {type(e).__name__} {str(e)[:200]}", flush=True)

print("CELL11 COMPLETE", flush=True)
