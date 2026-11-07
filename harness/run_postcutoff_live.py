# === Cell 16 (2026-11-07 evening): post-cutoff check on SWE-bench-Live (issues created after Gemma 4's cutoff) ===
# Gemma 4 knowledge cutoff: January 2025 (Google EU training-data transparency summary). We use issues created on or
# after 2025-04-01 (3-month margin). Pipeline FROZEN from the paper: whole-repo BM25 + MiniLM dense over non-test,
# non-docs Python files at the base commit, pool = BM25-25 ∪ dense-25 shuffled (seed "<id>|listwise"), Gemma
# listwise prompt identical to cell 14 (paths only). No re-tuning. Nothing is executed from the repositories.
# Gold = non-test .py files changed by the fix (multi-file allowed; Any@k = any gold file within top-k).
# Output: C16_ROW lines + /content/c16_live.jsonl (checkpointed), SUMMARY at the end. Time budget guard.
import os, sys, io, re, json, time, random, tarfile, urllib.request, gc
import torch
from datasets import load_dataset
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # release root, for nami/
from nami.chunker import chunk_repo

MODEL_ID = "google/gemma-4-12b-it"
OUT_PATH = os.path.join(os.environ.get("RECALLFIRST_WORKDIR", "/content"), "c16_live.jsonl")
CUTOFF = "2025-04-01"
N_TARGET = 120  # the paper used 120 (first run 60, then extended; selection is a prefix, so identical)
PER_REPO = 3
TIME_BUDGET_S = 100 * 60
MAX_CHUNKS = 30000
TOP_K = 10
ISSUE_CHARS = 4000
SEED = 20261107
T_START = time.time()

# ---------- data ----------
dsd = load_dataset("SWE-bench-Live/SWE-bench-Live")
print("splits:", {k: len(v) for k, v in dsd.items()}, flush=True)
rows, seen_ids = [], set()
for split, ds in dsd.items():
    cols = set(ds.column_names)
    print(f"  {split} columns: {sorted(cols)[:14]}", flush=True)
    for r in ds:
        if r["instance_id"] in seen_ids: continue
        seen_ids.add(r["instance_id"]); rows.append(r)
DATE_KEY = next(k for k in ("created_at", "createdAt", "created") if k in rows[0])
DIFF_RE = re.compile(r"^diff --git a/(.+?) b/(.+?)$", re.M)
def gold_of(patch):
    out = []
    for m in DIFF_RE.finditer(patch or ""):
        p = m.group(2)
        parts = p.split("/")
        if p.endswith(".py") and not any(x in ("tests", "test", "docs", "doc", "testing") for x in parts[:-1]) \
           and not parts[-1].startswith("test_") and not parts[-1].endswith("_test.py"):
            out.append(p)
    return sorted(set(out))
cand = [r for r in rows if str(r[DATE_KEY])[:10] >= CUTOFF and gold_of(r["patch"])]
print(f"rows {len(rows)} | created >= {CUTOFF} with non-test .py gold: {len(cand)} | "
      f"repos {len({r['repo'] for r in cand})}", flush=True)
rng = random.Random(SEED); rng.shuffle(cand)
picked, per = [], {}
for r in cand:
    if per.get(r["repo"], 0) < PER_REPO:
        picked.append(r); per[r["repo"]] = per.get(r["repo"], 0) + 1
    if len(picked) >= N_TARGET: break
print(f"selected {len(picked)} instances from {len(per)} repos", flush=True)

done = set()
if os.path.exists(OUT_PATH):
    for l in open(OUT_PATH):
        try:
            x = json.loads(l); done.add(x["instance"])
        except Exception: pass

# ---------- models ----------
for _v in ("model", "tok", "scorer", "dense_models", "dense"):
    if _v in globals(): del globals()[_v]
gc.collect(); torch.cuda.empty_cache()
dense = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")
tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto",
                                             attn_implementation="eager")
model.eval(); print("models loaded", flush=True)

def gen(msg, max_new=80):
    ids = tok.apply_chat_template([{"role": "user", "content": msg}], add_generation_prompt=True,
                                  return_tensors="pt")["input_ids"].to(model.device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()

def fetch(repo, commit, iid):
    owner, name = repo.split("/")
    dest = f"/content/live_repos/{iid}"
    if os.path.isdir(dest) and os.listdir(dest):
        return os.path.join(dest, os.listdir(dest)[0])
    os.makedirs(dest, exist_ok=True)
    data = urllib.request.urlopen(f"https://codeload.github.com/{owner}/{name}/tar.gz/{commit}", timeout=180).read()
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
        mem = [m for m in tf.getmembers() if "/tests/" not in m.name and "/docs/" not in m.name
               and "/.github/" not in m.name]
        tf.extractall(dest, members=mem)
    return os.path.join(dest, os.listdir(dest)[0])

def file_order(chunks, scores):
    fs = {}
    for c, s in zip(chunks, scores):
        fs[c.repo_path] = max(fs.get(c.repo_path, -1e9), float(s))
    return [f for f, _ in sorted(fs.items(), key=lambda kv: -kv[1])]

def prompt(repo, issue, shown):   # identical to cell 14 (paths)
    listing = "\n".join(f"{i}. {f}" for i, f in enumerate(shown, 1))
    return (f"Below is a bug report from the {repo} project, followed by a numbered list of "
            f"{len(shown)} candidate source files from the repository"
            f". Rank the {TOP_K} files most likely to be modified to fix this bug, most likely first. "
            f"Answer with only the {TOP_K} numbers, comma-separated.\n\n"
            f"--- BUG REPORT ---\n{issue}\n\n--- CANDIDATE FILES ---\n{listing}\n\n"
            f"Answer ({TOP_K} numbers, most likely first):")

def first_rank(order, gs):
    return next((i for i, f in enumerate(order, 1) if f in gs), None)

def stem_in_issue(text, gs):
    low = text.lower()
    for p in gs:
        stem = p.rsplit("/", 1)[-1].rsplit(".", 1)[0].lower()
        if stem not in ("__init__", "base", "utils", "core") and len(stem) > 2 and stem in low:
            return True
    return False

def write(row):
    with open(OUT_PATH, "a") as f: f.write(json.dumps(row) + "\n")
    print("C16_ROW " + json.dumps(row), flush=True)

for n, r in enumerate(picked, 1):
    iid = r["instance_id"]
    if iid in done: continue
    if time.time() - T_START > TIME_BUDGET_S:
        print("TIME BUDGET reached - stopping", flush=True); break
    t0 = time.time()
    try:
        gs = gold_of(r["patch"]); issue_full = r["problem_statement"]; issue = issue_full[:ISSUE_CHARS]
        root = fetch(r["repo"], r["base_commit"], iid)
        chunks = chunk_repo(root, "python")
        files = {c.repo_path for c in chunks}
        base = {"instance": iid, "repo": r["repo"], "created_at": str(r[DATE_KEY])[:10], "gold": gs,
                "n_gold": len(gs), "n_files_repo": len(files), "n_chunks": len(chunks),
                "stem_in_issue": stem_in_issue(issue_full, gs), "issue_chars": len(issue_full)}
        if not (set(gs) & files):
            write({**base, "skip_reason": "gold_not_in_corpus"}); continue
        if len(chunks) > MAX_CHUNKS:
            write({**base, "skip_reason": f"too_many_chunks>{MAX_CHUNKS}"}); continue
        bm = BM25Okapi([c.text[:6000].lower().split() for c in chunks])
        bm25 = file_order(chunks, bm.get_scores(issue_full.lower().split()))
        qv = dense.encode(issue_full[:4000], normalize_embeddings=True)
        ce = dense.encode([c.text[:2000] for c in chunks], batch_size=256, normalize_embeddings=True,
                          show_progress_bar=False)
        dn = file_order(chunks, ce @ qv)
        pool = list(dict.fromkeys(bm25[:25] + dn[:25]))
        shown = list(pool); random.Random(f"{iid}|listwise").shuffle(shown)
        ans = gen(prompt(r["repo"], issue, shown))
        nums = []
        for m in re.finditer(r"\d+", ans):
            k = int(m.group())
            if 1 <= k <= len(shown) and k not in nums: nums.append(k)
            if len(nums) >= TOP_K: break
        top = [shown[k - 1] for k in nums]
        fill = top + [f for f in bm25 if f in set(pool) and f not in top]
        write({**base, "pool_size": len(pool), "gold_in_pool": bool(set(gs) & set(pool)),
               "ranks": {"bm25": first_rank(bm25, gs), "dense": first_rank(dn, gs),
                         "gemma_fill": first_rank(fill, gs)},
               "gemma_top10": top, "parse_n": len(nums), "raw": ans[:120],
               "bm25_top100": bm25[:100], "dense_top100": dn[:100], "secs": round(time.time() - t0, 1)})
        print(f"[{n}/{len(picked)}] {iid} gemma={first_rank(fill, gs)} bm25={first_rank(bm25, gs)} "
              f"dense={first_rank(dn, gs)} {time.time()-t0:.0f}s", flush=True)
    except Exception as e:
        write({"instance": iid, "repo": r["repo"], "error": f"{type(e).__name__}: {str(e)[:200]}"})
    finally:
        os.system(f"rm -rf /content/live_repos/{iid}")

# ---------- summary ----------
X = [json.loads(l) for l in open(OUT_PATH)]
X = [x for x in X if "ranks" in x]
def hk(m, k, S): return sum(1 for x in S if x["ranks"][m] is not None and x["ranks"][m] <= k)
for name, S in (("all", X), ("name-free", [x for x in X if not x["stem_in_issue"]]),
                ("single-file", [x for x in X if x["n_gold"] == 1])):
    if not S: continue
    print(f"SUMMARY {name} n={len(S)} pool_ceiling {sum(x['gold_in_pool'] for x in S)} | " +
          " | ".join(f"{m} t1 {hk(m,1,S)} t5 {hk(m,5,S)} t10 {hk(m,10,S)}" for m in ("bm25", "dense", "gemma_fill")),
          flush=True)
print(f"CELL16 COMPLETE in {(time.time()-T_START)/60:.0f} min", flush=True)
