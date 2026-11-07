# Released as part of the localization audit kit. Originally a Google Colab cell; runs as a script from the
# release root (python harness/<name>.py) with a GPU. Working files go to $RECALLFIRST_WORKDIR (default /content).
import os as _os
WORKDIR = _os.environ.get("RECALLFIRST_WORKDIR", "/content")
_os.makedirs(WORKDIR, exist_ok=True)
# === Contamination probe v2 (HARD distractors + non-LLM readers on the SAME options) ===
# Why: probe v1 (cell 12) used 9 RANDOM repo files as distractors -> 22/25 correct, but the
# gold file's stem appears in the issue on 15/22 correct picks and random options are mostly
# off-topic, so reading the issue suffices; v1 cannot separate memorisation from reading.
# v2: same 25 instances, same prompt wording as v1, but the 9 distractors are the strongest
# NON-gold files by BM25 and MiniLM dense over the whole repo (interleaved, deduped).
# Readers on the identical 10 options (no LLM):
#   bm25_pick   - BM25 over chunk content, max-chunk per file (same as cell 11)
#   dense_pick  - MiniLM cosine over chunk content, max-chunk per file (same as cell 11)
#   path_pick   - lexical match of issue tokens against PATH tokens only (sees what Gemma sees)
# Reading the result: Gemma >> path_pick on hard options = the model knows more than the
# visible text supports (memorisation OR genuine domain understanding — still not proof);
# Gemma ~ path_pick / bm25_pick = v1's 88% was the easy-distractor effect.
# Output: /content/probe_hard_v1.jsonl + PROBE_HARD_ROW lines. Checkpointed; errors retry.
import os, sys, time, json, random, gc, re, string
import torch

from nami.chunker import chunk_repo
from nami.data import gold_files, fetch_repo_snapshot
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

MODEL_ID = "google/gemma-4-12b-it"
DENSE_ID = "sentence-transformers/all-MiniLM-L6-v2"
OUT_PATH = WORKDIR + "/probe_hard_v1.jsonl"
DONE_SEED = []
N_INSTANCES = 25          # identical instance set to probe v1 (first 25 of the Lite test split)
N_DISTRACT = 9
MAX_CHUNK_CHARS = 6000    # as cell 11
LETTERS = string.ascii_uppercase

assert torch.cuda.is_available(), "no GPU"
print(f"preflight: GPU={torch.cuda.get_device_name(0)}", flush=True)

# --- free GPU memory held by earlier cells (cell 11 `scorer` keeps a model reference) ---
for _v in ("model", "tok", "scorer", "dense_models", "dense_model", "dense_tok"):
    if _v in globals():
        del globals()[_v]
gc.collect(); torch.cuda.empty_cache()
sys.last_traceback = None; sys.last_value = None; sys.last_type = None
free, total = torch.cuda.mem_get_info()
print(f"GPU free before load: {free/1e9:.1f} / {total/1e9:.1f} GB", flush=True)

ds = load_dataset("princeton-nlp/SWE-bench_Lite")
split = "test" if "test" in ds else list(ds.keys())[0]
rows = list(ds[split])

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

dense = SentenceTransformer(DENSE_ID, device="cuda")
tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto",
    attn_implementation="eager")
model.eval()
print("model loaded", flush=True)

def chat_generate(user_msg, max_new=8):
    msgs = [{"role": "user", "content": user_msg}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")["input_ids"].to(model.device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()

def file_scores_bm25(chunks, issue):
    bm = BM25Okapi([c.text[:MAX_CHUNK_CHARS].lower().split() for c in chunks])
    cs = bm.get_scores(issue.lower().split())
    fs = {}
    for c, s in zip(chunks, cs):
        fs[c.repo_path] = max(fs.get(c.repo_path, -1e9), float(s))
    return fs

def file_scores_dense(chunks, issue):
    qv = dense.encode(issue[:4000], normalize_embeddings=True)
    ce = dense.encode([c.text[:2000] for c in chunks], batch_size=128,
                      normalize_embeddings=True, show_progress_bar=False)
    sims = ce @ qv
    fs = {}
    for c, s in zip(chunks, sims):
        fs[c.repo_path] = max(fs.get(c.repo_path, -1e9), float(s))
    return fs

_SPLIT = re.compile(r"[^a-z0-9]+")
def path_tokens(p):
    toks = set(t for t in _SPLIT.split(p.lower().removesuffix(".py")) if len(t) >= 3)
    return toks - {"init", "django", "astropy", "sympy", "sklearn", "lib", "src", "utils"}

def path_score(p, issue_tokens):
    return sum(1 for t in path_tokens(p) if t in issue_tokens)

def pick_by(scores, cands):
    """Letter of the best-scoring option; ties -> no pick ('=') so ties never count as correct."""
    vals = [scores.get(f, -1e9) for f in cands]
    best = max(vals)
    if vals.count(best) > 1:
        return "="
    return LETTERS[vals.index(best)]

n_ok = n_err = 0
for idx in range(N_INSTANCES):
    inst = rows[idx]
    iid = inst["instance_id"]
    if iid in done_ids:
        print(f"[{idx+1}] {iid} SKIP (done)", flush=True)
        continue
    try:
        t0 = time.time()
        gold = gold_files(inst["patch"])
        gold_set = set(gold)
        issue = inst["problem_statement"]
        repo_dir = fetch_repo_snapshot(inst["repo"], inst["base_commit"], WORKDIR + "/repos")
        chunks = chunk_repo(repo_dir, "python")
        bm = file_scores_bm25(chunks, issue)
        dn = file_scores_dense(chunks, issue)
        bm_neg = [f for f, _ in sorted(bm.items(), key=lambda kv: -kv[1]) if f not in gold_set]
        dn_neg = [f for f, _ in sorted(dn.items(), key=lambda kv: -kv[1]) if f not in gold_set]
        distract = []
        for a, b in zip(bm_neg, dn_neg):          # interleave: BM25 #1, dense #1, BM25 #2, ...
            for f in (a, b):
                if f not in distract and len(distract) < N_DISTRACT:
                    distract.append(f)
            if len(distract) >= N_DISTRACT:
                break
        cands = gold[:1] + distract
        rng = random.Random(2000 + idx)
        rng.shuffle(cands)
        gold_letter = LETTERS[cands.index(gold[0])]
        listing = "\n".join(f"{LETTERS[i]}. {f}" for i, f in enumerate(cands))
        q = (f"Below is a bug report from the {inst['repo']} project, followed by a list of "
             f"source files from the repository at the relevant commit. Which single file is "
             f"most likely modified to fix this bug? Answer with just the letter.\n\n"
             f"--- BUG REPORT ---\n{issue[:4000]}\n"
             f"--- FILES ---\n{listing}\n\nAnswer (single letter):")
        ans = chat_generate(q)
        picked = next((ch for ch in ans if ch in LETTERS[:len(cands)]), "?")
        issue_tokens = set(t for t in _SPLIT.split(issue.lower()) if len(t) >= 3)
        bm_pick = pick_by(bm, cands)
        dn_pick = pick_by(dn, cands)
        pa_pick = pick_by({f: path_score(f, issue_tokens) for f in cands}, cands)
        row = {"instance": iid, "repo": inst["repo"], "gold": gold[0], "n_gold": len(gold),
               "cands": cands, "gold_letter": gold_letter,
               "picked": picked, "correct": picked == gold_letter,
               "bm25_pick": bm_pick, "bm25_correct": bm_pick == gold_letter,
               "dense_pick": dn_pick, "dense_correct": dn_pick == gold_letter,
               "path_pick": pa_pick, "path_correct": pa_pick == gold_letter,
               "picked_file": cands[LETTERS.index(picked)] if picked in LETTERS[:len(cands)] else None,
               "raw_answer": ans[:40], "secs": round(time.time() - t0, 1)}
        with open(OUT_PATH, "a") as f:
            f.write(json.dumps(row) + "\n")
        print("PROBE_HARD_ROW " + json.dumps(row), flush=True)
        print(f"[{idx+1}] {iid} gemma={'OK' if row['correct'] else 'x'} bm25={'OK' if row['bm25_correct'] else 'x'} "
              f"dense={'OK' if row['dense_correct'] else 'x'} path={'OK' if row['path_correct'] else 'x'} "
              f"{row['secs']}s", flush=True)
        n_ok += 1
    except Exception as e:
        row = {"instance": iid, "error": f"{type(e).__name__}: {str(e)[:200]}"}
        with open(OUT_PATH, "a") as f:
            f.write(json.dumps(row) + "\n")
        print("PROBE_HARD_ROW " + json.dumps(row), flush=True)
        n_err += 1

# --- summary over everything in the output file (latest ok row per instance) ---
lat = {}
for line in open(OUT_PATH):
    try:
        r = json.loads(line)
    except Exception:
        continue
    if "error" not in r:
        lat[r["instance"]] = r
R = list(lat.values()); n = len(R)
if n:
    for k in ("correct", "bm25_correct", "dense_correct", "path_correct"):
        print(f"SUMMARY {k}: {sum(r[k] for r in R)}/{n}", flush=True)
    both = sum(r["correct"] and not r["path_correct"] for r in R)
    print(f"SUMMARY gemma right where path-match wrong: {both}/{n}", flush=True)
    from collections import Counter
    print("SUMMARY gemma letter distribution:", dict(Counter(r["picked"] for r in R)), flush=True)
print(f"CELL12B COMPLETE. ok={n_ok} err={n_err} (+{len(done_ids)} resumed)", flush=True)
