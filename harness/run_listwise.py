# Released as part of the localization audit kit. Originally a Google Colab cell; runs as a script from the
# release root (python harness/<name>.py) with a GPU. Working files go to $RECALLFIRST_WORKDIR (default /content).
import os as _os
WORKDIR = _os.environ.get("RECALLFIRST_WORKDIR", "/content")
_os.makedirs(WORKDIR, exist_ok=True)
# === Cell 14: recall-first pool + Gemma listwise localization (and hard-distractor contamination) ===
# For every ok row of the whole-repo run (/content/rerank_v2.jsonl): pool = BM25 top-25 ∪ dense top-25
# (from the row's stored whole-repo orders, deduped). The pool is SHUFFLED with a fixed per-instance seed
# so Gemma never sees the retrievers' ordering. Gemma ranks its top-10 files from the issue text.
# Conditions (paths first over ALL instances, then sigs):
#   "paths"      - numbered file paths only (no code): cheap; also the realistic-distractor contamination test
#   "paths_sigs" - paths + up to 15 chunk names per file (classes / functions / methods) from the
#                  repo at base commit; needs the repo snapshot. Set CONDITIONS to skip it.
# Ranks reported per row (1-based, None = not found):
#   gemma_listwise : gold position in Gemma's top-10 (None if absent)
#   gemma_fill     : rank in Gemma top-10 followed by the rest of the pool in BM25 order (full ordering,
#                    comparable to bm25/dense; gold outside the pool -> None)
#   bm25, dense    : whole-repo ranks copied from the rerank-v2 row (same numbers as the paper tables)
# Output: LIST_ROW lines + /content/listwise_v1.jsonl, checkpointed per instance|condition (DONE_SEED
# holds "instance|condition" keys). Error rows retry automatically.
import os, sys, time, json, random, gc, re
import torch

from nami.chunker import chunk_repo
from nami.data import fetch_repo_snapshot
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "google/gemma-4-12b-it"
RERANK = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "data", "orderings_wholerepo.jsonl")
OUT_PATH = WORKDIR + "/listwise_v1.jsonl"
DONE_SEED = []
CONDITIONS = ("paths", "paths_sigs")
POOL_BM25 = 25
POOL_DENSE = 25
TOP_K = 10
MAX_NAMES = 15
ISSUE_CHARS = 4000

assert os.path.exists(RERANK), f"missing {RERANK}"
assert torch.cuda.is_available(), "no GPU"

# --- free GPU memory held by earlier cells (cell 11 `scorer` keeps a model reference) ---
for _v in ("model", "tok", "scorer", "dense_models", "dense_model", "dense_tok", "dense"):
    if _v in globals():
        del globals()[_v]
gc.collect(); torch.cuda.empty_cache()
sys.last_traceback = None; sys.last_value = None; sys.last_type = None
_free, _tot = torch.cuda.mem_get_info()
print(f"GPU free before load: {_free/1e9:.1f} / {_tot/1e9:.1f} GB", flush=True)

# --- data: latest ok row per instance ---
latest = {}
for line in open(RERANK):
    try:
        r = json.loads(line)
    except Exception:
        continue
    if r.get("error") or r.get("skip_reason"):
        continue
    latest[r["instance"]] = r
inst_rows = list(latest.values())
print(f"listwise candidates: {len(inst_rows)} instances | conditions={CONDITIONS}", flush=True)

_ds = load_dataset("princeton-nlp/SWE-bench_Lite")
_split = "test" if "test" in _ds else list(_ds.keys())[0]
swe = {r["instance_id"]: r for r in _ds[_split]}

done = set(DONE_SEED)
if os.path.exists(OUT_PATH):
    for line in open(OUT_PATH):
        try:
            r = json.loads(line)
            if "error" not in r:
                done.add(f'{r["instance"]}|{r["condition"]}')
        except Exception:
            pass
print(f"checkpoint: {len(done)} done", flush=True)

tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto",
    attn_implementation="eager")
model.eval()
print("model loaded", flush=True)

def chat_generate(user_msg, max_new=80):
    msgs = [{"role": "user", "content": user_msg}]
    # BatchEncoding in this transformers version -> index ["input_ids"] (bug that broke cells 12/13)
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")["input_ids"].to(model.device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip(), int(ids.shape[1])

def golds(r):
    g = r["gold"]
    return list(g) if isinstance(g, list) else [g]

def union_pool(r):
    pool = []
    for f in r["orders"]["bm25"][:POOL_BM25] + r["orders"]["dense"][:POOL_DENSE]:
        if f not in pool:
            pool.append(f)
    return pool

def rank_in(order, gs):
    for i, f in enumerate(order, 1):
        if f in gs:
            return i
    return None

def file_names(repo_dir, files):
    """Up to MAX_NAMES chunk names per file: classes/top-level functions first, then methods."""
    want = set(files)
    by = {}
    for c in chunk_repo(repo_dir, "python"):
        if c.repo_path in want and c.name != "<file>":
            by.setdefault(c.repo_path, []).append(c.name)
    out = {}
    for f in files:
        names = list(dict.fromkeys(by.get(f, [])))
        names.sort(key=lambda n: ("." in n, ))          # stable: top-level first
        out[f] = names[:MAX_NAMES]
    return out

def parse_numbers(ans, n):
    seen = []
    for m in re.finditer(r"\d+", ans):
        k = int(m.group())
        if 1 <= k <= n and k not in seen:
            seen.append(k)
        if len(seen) >= TOP_K:
            break
    return seen

n_ok = n_err = 0
for cond in CONDITIONS:
    for prog, r in enumerate(inst_rows, 1):
        iid = r["instance"]
        key = f"{iid}|{cond}"
        if key in done:
            continue
        t0 = time.time()
        try:
            gs = set(golds(r))
            pool = union_pool(r)
            shown = list(pool)
            random.Random(f"{iid}|listwise").shuffle(shown)
            inst = swe[iid]
            issue = inst["problem_statement"][:ISSUE_CHARS]
            if cond == "paths":
                listing = "\n".join(f"{i}. {f}" for i, f in enumerate(shown, 1))
            else:
                repo_dir = fetch_repo_snapshot(inst["repo"], inst["base_commit"], WORKDIR + "/repos")
                names = file_names(repo_dir, shown)
                listing = "\n".join(
                    f"{i}. {f}" + (f"  [{', '.join(names[f])}]" if names.get(f) else "")
                    for i, f in enumerate(shown, 1))
            q = (f"Below is a bug report from the {inst['repo']} project, followed by a numbered list of "
                 f"{len(shown)} candidate source files from the repository"
                 + (" (with the classes/functions each file defines)" if cond != "paths" else "")
                 + f". Rank the {TOP_K} files most likely to be modified to fix this bug, most likely first. "
                 f"Answer with only the {TOP_K} numbers, comma-separated.\n\n"
                 f"--- BUG REPORT ---\n{issue}\n\n--- CANDIDATE FILES ---\n{listing}\n\n"
                 f"Answer ({TOP_K} numbers, most likely first):")
            ans, n_prompt = chat_generate(q)
            nums = parse_numbers(ans, len(shown))
            top = [shown[k - 1] for k in nums]
            bm_rest = [f for f in r["orders"]["bm25"] if f in set(pool) and f not in top]
            fill = top + bm_rest + [f for f in pool if f not in top and f not in bm_rest]
            row = {"instance": iid, "repo": r["repo"], "condition": cond,
                   "gold": r["gold"], "pool_size": len(pool), "gold_in_pool": bool(gs & set(pool)),
                   "gemma_top5": top[:5], "gemma_top10": top,
                   "ranks": {"gemma_listwise": rank_in(top, gs), "gemma_fill": rank_in(fill, gs),
                             "bm25": r["ranks"].get("bm25"), "dense": r["ranks"].get("dense")},
                   "gold_in_gemma_top5": rank_in(top[:5], gs) is not None,
                   "parse_n": len(nums), "raw_answer": ans[:160], "prompt_tokens": n_prompt,
                   "secs": round(time.time() - t0, 1)}
            with open(OUT_PATH, "a") as f:
                f.write(json.dumps(row) + "\n")
            print("LIST_ROW " + json.dumps(row), flush=True)
            print(f"[{cond} {prog}/{len(inst_rows)}] {iid} gemma={row['ranks']['gemma_listwise']} "
                  f"bm25={row['ranks']['bm25']} dense={row['ranks']['dense']} parsed={len(nums)} {row['secs']}s",
                  flush=True)
            n_ok += 1
        except Exception as e:
            row = {"instance": iid, "condition": cond, "error": f"{type(e).__name__}: {str(e)[:200]}"}
            with open(OUT_PATH, "a") as f:
                f.write(json.dumps(row) + "\n")
            print("LIST_ROW " + json.dumps(row), flush=True)
            n_err += 1
            if "out of memory" in str(e).lower():
                gc.collect(); torch.cuda.empty_cache()

# --- summary (latest ok row per instance|condition) ---
lat = {}
for line in open(OUT_PATH):
    try:
        x = json.loads(line)
    except Exception:
        continue
    if "error" not in x:
        lat[(x["instance"], x["condition"])] = x
for cond in CONDITIONS:
    R = [x for (i, c), x in lat.items() if c == cond]
    n = len(R)
    if not n:
        continue
    def topk(m, k):
        return sum(1 for x in R if x["ranks"][m] is not None and x["ranks"][m] <= k)
    for k in (1, 5, 10):
        print(f"SUMMARY {cond} top-{k}: gemma_fill {topk('gemma_fill', k)}/{n} | "
              f"bm25 {topk('bm25', k)}/{n} | dense {topk('dense', k)}/{n}", flush=True)
    print(f"SUMMARY {cond} gold in union pool: {sum(x['gold_in_pool'] for x in R)}/{n} | "
          f"mean parsed numbers {sum(x['parse_n'] for x in R)/n:.1f}", flush=True)
print(f"CELL14 COMPLETE. ok={n_ok} err={n_err} (+{len(done)} resumed)", flush=True)
