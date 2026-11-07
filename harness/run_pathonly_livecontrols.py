# === Cell 17 (2026-11-07 night): review-3 optional checks. Frozen prompt, greedy, same pools as the paper. ===
#  A  path-only baselines (no model): rank each instance's BM25-25 ∪ dense-25 pool by BM25 / MiniLM over the PATH
#     STRINGS only (the information the selector sees) -> Lite 57 and post-cutoff 120.
#  B  post-cutoff controls with Gemma (union pool, same shuffle seed "<id>|listwise"):
#     live_empty    : issue replaced by "(no bug report text is available)"
#     live_mismatch : issue of the next instance from the same repository (other repository if the repo has 1 issue)
# Inputs: data/orderings_wholerepo.jsonl + data/postcutoff_live120.jsonl from the RecallFirst repo (cloned if public,
# else upload both files to /content). Output: /content/c17.jsonl (checkpointed), C17_ROW lines, SUMMARY lines.
import os, re, json, random, time, gc, subprocess
OUT = "/content/c17.jsonl"; TOP_K = 10; ISSUE_CHARS = 4000; MODEL_ID = "google/gemma-4-12b-it"

def find(name):
    for p in (f"/content/{name}", f"/content/RecallFirst/data/{name}"):
        if os.path.exists(p): return p
    return None
if not (find("orderings_wholerepo.jsonl") and find("postcutoff_live120.jsonl")):
    subprocess.run("cd /content && git clone -q https://github.com/HussamHaroon/RecallFirst", shell=True)
P_LITE, P_LIVE = find("orderings_wholerepo.jsonl"), find("postcutoff_live120.jsonl")
assert P_LITE and P_LIVE, "Upload orderings_wholerepo.jsonl and postcutoff_live120.jsonl to /content (Files panel)"
LITE = [x for x in map(json.loads, open(P_LITE)) if "ranks" in x]
LIVE = [x for x in map(json.loads, open(P_LIVE)) if "ranks" in x]
assert len(LITE) == 57 and len(LIVE) == 120, (len(LITE), len(LIVE))
items = [{"i": x["instance"], "r": x["repo"], "g": [x["gold"]], "b": x["orders"]["bm25"][:25], "d": x["orders"]["dense"][:25],
          "st": x["gold_stem_in_issue"], "set": "lite"} for x in LITE] + \
        [{"i": x["instance"], "r": x["repo"], "g": x["gold"], "b": x["bm25_top100"][:25], "d": x["dense_top100"][:25],
          "st": x["stem_in_issue"], "set": "live", "pool_size": x["pool_size"]} for x in LIVE]
for it in items:
    it["pool"] = list(dict.fromkeys(it["b"] + it["d"]))
    if "pool_size" in it: assert len(it["pool"]) == it["pool_size"], it["i"]

from datasets import load_dataset
ISS = {r["instance_id"]: r["problem_statement"] for r in load_dataset("princeton-nlp/SWE-bench_Lite")["test"]}
for split, ds in load_dataset("SWE-bench-Live/SWE-bench-Live").items():
    for r in ds: ISS.setdefault(r["instance_id"], r["problem_statement"])
assert all(it["i"] in ISS for it in items), [it["i"] for it in items if it["i"] not in ISS][:5]

done = set()
if os.path.exists(OUT):
    for l in open(OUT):
        try:
            x = json.loads(l)
            if "error" not in x: done.add((x["instance"], x["condition"]))
        except Exception: pass
def write(row):
    with open(OUT, "a") as f: f.write(json.dumps(row) + "\n")
    print("C17_ROW " + json.dumps(row), flush=True)
def first_rank(order, gs): return next((k for k, f in enumerate(order, 1) if f in gs), None)

# ---------- A: path-only baselines ----------
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
_T = re.compile(r"[^a-z0-9]+")
def toks(t): return [w for w in _T.split(t.lower()) if len(w) >= 2 and w != "py"]
mini = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")
for it in items:
    if (it["i"], "path_bm25") in done: continue
    issue = ISS[it["i"]]; pool = it["pool"]; bpos = {f: k for k, f in enumerate(it["b"] + it["d"])}
    sc = BM25Okapi([toks(f) for f in pool]).get_scores(toks(issue))
    o_bm = [f for _, f in sorted(zip(sc, pool), key=lambda t: (-t[0], bpos[t[1]]))]
    pe = mini.encode([f.replace("/", " ").replace("_", " ").removesuffix(".py") for f in pool], normalize_embeddings=True)
    qe = mini.encode(issue[:ISSUE_CHARS], normalize_embeddings=True)
    o_de = [f for _, f in sorted(zip((pe @ qe).tolist(), pool), key=lambda t: (-t[0], bpos[t[1]]))]
    for cond, order in (("path_bm25", o_bm), ("path_dense", o_de)):
        write({"instance": it["i"], "repo": it["r"], "set": it["set"], "condition": cond, "gold": it["g"],
               "stem_in_issue": it["st"], "rank": first_rank(order, set(it["g"])), "top10": order[:10]})
del mini; gc.collect()

# ---------- B: post-cutoff controls with Gemma ----------
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
for _v in ("model", "tok"):
    if _v in globals(): del globals()[_v]
gc.collect(); torch.cuda.empty_cache()
tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="eager")
model.eval(); print("model loaded", flush=True)
def gen(msg, max_new=80):
    ids = tok.apply_chat_template([{"role": "user", "content": msg}], add_generation_prompt=True, return_tensors="pt")["input_ids"].to(model.device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
def prompt(repo, issue, shown):   # identical to the paper's selector prompt (paths only)
    listing = "\n".join(f"{i}. {f}" for i, f in enumerate(shown, 1))
    return (f"Below is a bug report from the {repo} project, followed by a numbered list of "
            f"{len(shown)} candidate source files from the repository"
            f". Rank the {TOP_K} files most likely to be modified to fix this bug, most likely first. "
            f"Answer with only the {TOP_K} numbers, comma-separated.\n\n"
            f"--- BUG REPORT ---\n{issue}\n\n--- CANDIDATE FILES ---\n{listing}\n\n"
            f"Answer ({TOP_K} numbers, most likely first):")
live = sorted([it for it in items if it["set"] == "live"], key=lambda t: t["i"])
by_repo = {}
for it in live: by_repo.setdefault(it["r"], []).append(it["i"])
def mismatch(it):
    ids = by_repo[it["r"]]
    if len(ids) > 1: return ids[(ids.index(it["i"]) + 1) % len(ids)], True
    j = [t["i"] for t in live].index(it["i"]); return live[(j + 1) % len(live)]["i"], False
for cond in ("live_empty", "live_mismatch"):
    for it in live:
        if (it["i"], cond) in done: continue
        try:
            shown = list(it["pool"]); random.Random(f'{it["i"]}|listwise').shuffle(shown)
            extra = {}
            if cond == "live_empty": issue = "(no bug report text is available)"
            else:
                other, same = mismatch(it); issue = ISS[other][:ISSUE_CHARS]; extra = {"issue_from": other, "same_repo": same}
            t0 = time.time(); ans = gen(prompt(it["r"], issue, shown))
            nums = []
            for m in re.finditer(r"\d+", ans):
                k = int(m.group())
                if 1 <= k <= len(shown) and k not in nums: nums.append(k)
                if len(nums) >= TOP_K: break
            top = [shown[k - 1] for k in nums]
            fill = top + [f for f in it["b"] + it["d"] if f not in top]
            fill = list(dict.fromkeys(fill))
            write({"instance": it["i"], "repo": it["r"], "set": "live", "condition": cond, "gold": it["g"],
                   "stem_in_issue": it["st"], "rank": first_rank(fill, set(it["g"])), "top10": top, "parse_n": len(nums),
                   "raw": ans[:120], "secs": round(time.time() - t0, 1), **extra})
        except Exception as e:
            write({"instance": it["i"], "condition": cond, "error": f"{type(e).__name__}: {str(e)[:200]}"})

# ---------- summary ----------
R = {}
for l in open(OUT):
    x = json.loads(l)
    if "error" not in x: R[(x["instance"], x["condition"])] = x
h = lambda r, k: r is not None and r <= k
for st in ("lite", "live"):
    for cond in ("path_bm25", "path_dense", "live_empty", "live_mismatch"):
        S = [x for (i, c), x in R.items() if c == cond and x["set"] == st]
        if not S: continue
        nf = [x for x in S if not x["stem_in_issue"]]
        print(f"SUMMARY {st} {cond} n={len(S)} top1 {sum(h(x['rank'],1) for x in S)} top5 {sum(h(x['rank'],5) for x in S)} "
              f"top10 {sum(h(x['rank'],10) for x in S)} | name-free n={len(nf)} top1 {sum(h(x['rank'],1) for x in nf)} "
              f"top5 {sum(h(x['rank'],5) for x in nf)}", flush=True)
print("CELL17 COMPLETE", flush=True)
