# === Cell 15 (2026-11-07 evening): controls requested by three independent reviews. No repo access needed. ===
# On the SAME 57 instances as the paper, with the SAME prompt as cell 14 (paths only):
#   A  bm25pool      : Gemma selects from BM25 top-25 only (the pool NAMI reranked) -> clean same-pool comparison
#   P  probe57       : hard issue-only 10-way probe on the 57 (gold + 9 top non-gold of BM25/dense, interleaved);
#                      readers on the same options: BM25 / dense (rank order), path-token match (ties -> BM25 order,
#                      plus the old abstain-on-tie variant)
#   C1 union_empty   : union pool (BM25-25 ∪ dense-25), issue replaced by a placeholder -> repo/path prior only
#   C2 union_mismatch: union pool, issue of ANOTHER instance from the same repo -> prior + generic issue signal
#   S  union_seed1/2 : union pool, two extra shuffles of the candidate list -> position-bias / shuffle variance
# Output: C15_ROW lines + /content/c15_controls.jsonl (checkpointed per instance|condition), SUMMARY at the end.
import os, json, random, re, time, gc, string
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "google/gemma-4-12b-it"
OUT_PATH = os.path.join(os.environ.get("RECALLFIRST_WORKDIR", "/content"), "c15_controls.jsonl")
TOP_K = 10
ISSUE_CHARS = 4000
# Inputs rebuilt from the released orderings: BM25/dense top-25 per instance.
D = [{"i": x["instance"], "r": x["repo"], "g": x["gold"], "b": x["orders"]["bm25"][:25],
      "d": x["orders"]["dense"][:25], "st": x["gold_stem_in_issue"]}
     for x in map(json.loads, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "orderings_wholerepo.jsonl"))) if "ranks" in x]
assert len(D) == 57, len(D)
LETTERS = string.ascii_uppercase

swe = {r["instance_id"]: r for r in load_dataset("princeton-nlp/SWE-bench_Lite")["test"]}
done = set()
if os.path.exists(OUT_PATH):
    for l in open(OUT_PATH):
        try:
            x = json.loads(l)
            if "error" not in x: done.add(f'{x["instance"]}|{x["condition"]}')
        except Exception: pass
print(f"instances {len(D)} | checkpoint {len(done)}", flush=True)

for _v in ("model", "tok", "scorer", "dense_models", "dense"):
    if _v in globals(): del globals()[_v]
gc.collect(); torch.cuda.empty_cache()
tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto",
                                             attn_implementation="eager")
model.eval(); print("model loaded", flush=True)

def gen(msg, max_new=80):
    ids = tok.apply_chat_template([{"role": "user", "content": msg}], add_generation_prompt=True,
                                  return_tensors="pt")["input_ids"].to(model.device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()

def listwise_prompt(repo, issue, shown):   # identical wording to cell 14 (paths condition)
    listing = "\n".join(f"{i}. {f}" for i, f in enumerate(shown, 1))
    return (f"Below is a bug report from the {repo} project, followed by a numbered list of "
            f"{len(shown)} candidate source files from the repository"
            f". Rank the {TOP_K} files most likely to be modified to fix this bug, most likely first. "
            f"Answer with only the {TOP_K} numbers, comma-separated.\n\n"
            f"--- BUG REPORT ---\n{issue}\n\n--- CANDIDATE FILES ---\n{listing}\n\n"
            f"Answer ({TOP_K} numbers, most likely first):")

def probe_prompt(repo, issue, cands):      # identical wording to cell 12b
    listing = "\n".join(f"{LETTERS[i]}. {f}" for i, f in enumerate(cands))
    return (f"Below is a bug report from the {repo} project, followed by a list of "
            f"source files from the repository at the relevant commit. Which single file is "
            f"most likely modified to fix this bug? Answer with just the letter.\n\n"
            f"--- BUG REPORT ---\n{issue}\n"
            f"--- FILES ---\n{listing}\n\nAnswer (single letter):")

def parse_nums(ans, n):
    seen = []
    for m in re.finditer(r"\d+", ans):
        k = int(m.group())
        if 1 <= k <= n and k not in seen: seen.append(k)
        if len(seen) >= TOP_K: break
    return seen

def union(d):
    pool = []
    for f in d["b"] + d["d"]:
        if f not in pool: pool.append(f)
    return pool

def rank_in(order, g):
    return next((i for i, f in enumerate(order, 1) if f == g), None)

_SPLIT = re.compile(r"[^a-z0-9]+")
def ptoks(p):
    return set(t for t in _SPLIT.split(p.lower().removesuffix(".py")) if len(t) >= 3) - \
           {"init", "django", "astropy", "sympy", "sklearn", "lib", "src", "utils"}

by_repo = {}
for d in sorted(D, key=lambda x: x["i"]):
    by_repo.setdefault(d["r"], []).append(d["i"])
def mismatch_issue(d):
    ids = by_repo[d["r"]]; j = ids.index(d["i"])
    other = ids[(j + 1) % len(ids)]
    return other, swe[other]["problem_statement"][:ISSUE_CHARS]

def write(row):
    with open(OUT_PATH, "a") as f: f.write(json.dumps(row) + "\n")
    print("C15_ROW " + json.dumps(row), flush=True)

def run_listwise(d, cond, pool, issue, seed_key, extra=None):
    shown = list(pool); random.Random(seed_key).shuffle(shown)
    t0 = time.time()
    ans = gen(listwise_prompt(d["r"], issue, shown))
    nums = parse_nums(ans, len(shown)); top = [shown[k - 1] for k in nums]
    rest = [f for f in d["b"] + d["d"] if f in set(pool) and f not in top]
    fill = top + [f for f in dict.fromkeys(rest)]
    row = {"instance": d["i"], "repo": d["r"], "condition": cond, "gold": d["g"], "pool_size": len(pool),
           "gold_in_pool": d["g"] in pool, "top10": top, "rank": rank_in(fill, d["g"]),
           "parse_n": len(nums), "raw": ans[:120], "secs": round(time.time() - t0, 1)}
    if extra: row.update(extra)
    write(row)

PLAN = ["bm25pool", "probe57", "union_empty", "union_mismatch", "union_seed1", "union_seed2"]
for cond in PLAN:
    for idx, d in enumerate(sorted(D, key=lambda x: x["i"])):
        if f'{d["i"]}|{cond}' in done: continue
        try:
            issue = swe[d["i"]]["problem_statement"][:ISSUE_CHARS]
            if cond == "bm25pool":
                run_listwise(d, cond, d["b"][:25], issue, f'{d["i"]}|bm25pool')
            elif cond == "union_empty":
                run_listwise(d, cond, union(d), "(no bug report text is available)", f'{d["i"]}|listwise')
            elif cond == "union_mismatch":
                other, oiss = mismatch_issue(d)
                run_listwise(d, cond, union(d), oiss, f'{d["i"]}|listwise', {"issue_from": other})
            elif cond.startswith("union_seed"):
                run_listwise(d, cond, union(d), issue, f'{d["i"]}|listwise|{cond}')
            elif cond == "probe57":
                g = d["g"]; bn = [f for f in d["b"] if f != g]; dn = [f for f in d["d"] if f != g]
                distract = []
                for a, b in zip(bn, dn):
                    for f in (a, b):
                        if f not in distract and len(distract) < 9: distract.append(f)
                    if len(distract) >= 9: break
                cands = [g] + distract; random.Random(2000 + idx).shuffle(cands)
                gl = LETTERS[cands.index(g)]
                t0 = time.time(); ans = gen(probe_prompt(d["r"], issue, cands), max_new=8)
                picked = next((ch for ch in ans if ch in LETTERS[:10]), "?")
                def pick_rank(order):   # best-ranked option under a full ordering; unseen -> last
                    pos = {f: k for k, f in enumerate(order)}
                    return LETTERS[min(range(10), key=lambda k: pos.get(cands[k], 10**6))]
                itoks = set(t for t in _SPLIT.split(swe[d["i"]]["problem_statement"].lower()) if len(t) >= 3)
                sc = [sum(1 for t in ptoks(f) if t in itoks) for f in cands]
                best = max(sc); tied = [k for k in range(10) if sc[k] == best]
                bpos = {f: k for k, f in enumerate(d["b"])}
                path_tb = LETTERS[min(tied, key=lambda k: bpos.get(cands[k], 10**6))]
                path_abst = LETTERS[tied[0]] if len(tied) == 1 else "="
                row = {"instance": d["i"], "repo": d["r"], "condition": cond, "gold": g, "cands": cands,
                       "gold_letter": gl, "picked": picked, "correct": picked == gl,
                       "bm25_correct": pick_rank(d["b"]) == gl, "dense_correct": pick_rank(d["d"]) == gl,
                       "path_tiebreak_correct": path_tb == gl, "path_abstain_correct": path_abst == gl,
                       "path_tied": len(tied) > 1, "stem_in_issue": d["st"], "raw": ans[:20],
                       "secs": round(time.time() - t0, 1)}
                write(row)
        except Exception as e:
            write({"instance": d["i"], "condition": cond, "error": f"{type(e).__name__}: {str(e)[:200]}"})

# ---------- summary (identical instances; exact McNemar printed later offline) ----------
lat = {}
for l in open(OUT_PATH):
    x = json.loads(l)
    if "error" not in x: lat[(x["instance"], x["condition"])] = x
for cond in PLAN:
    R = [x for (i, c), x in lat.items() if c == cond]
    if not R: continue
    if cond == "probe57":
        ks = ["correct", "bm25_correct", "dense_correct", "path_tiebreak_correct", "path_abstain_correct"]
        print(f"SUMMARY probe57 n={len(R)} " + " | ".join(f"{k} {sum(x[k] for x in R)}" for k in ks), flush=True)
        for st in (True, False):
            S = [x for x in R if x["stem_in_issue"] == st]
            print(f"SUMMARY probe57 stem_in_issue={st} n={len(S)} gemma {sum(x['correct'] for x in S)} "
                  f"path_tb {sum(x['path_tiebreak_correct'] for x in S)} dense {sum(x['dense_correct'] for x in S)}", flush=True)
    else:
        h = lambda k: sum(1 for x in R if x["rank"] is not None and x["rank"] <= k)
        print(f"SUMMARY {cond} n={len(R)} top1 {h(1)} top5 {h(5)} top10 {h(10)} | gold_in_pool "
              f"{sum(x['gold_in_pool'] for x in R)}", flush=True)
print("CELL15 COMPLETE", flush=True)
