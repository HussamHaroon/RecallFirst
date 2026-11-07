# Released as part of the localization audit kit. Originally a Google Colab cell; runs as a script from the
# release root (python harness/<name>.py) with a GPU. Working files go to $RECALLFIRST_WORKDIR (default /content).
import os as _os
WORKDIR = _os.environ.get("RECALLFIRST_WORKDIR", "/content")
_os.makedirs(WORKDIR, exist_ok=True)
# === Contamination probe: issue-only file identification ===
# For the original 25 instances: can gemma-4-12b-it name the gold file from
# the issue text ALONE (no repo code)? 10-way multiple choice (gold + 9
# seeded random non-gold files from the same repo). Chance = 10%.
# Also open-ended "which file would you modify?" for the 3 rescue instances.
# High accuracy on rescues would mean NAMI's wins may ride on memorized
# issue->file associations; low accuracy supports the contrast-signal story.
import os, sys, time, json, random, gc
import torch

from nami.chunker import chunk_repo
from nami.data import gold_files, fetch_repo_snapshot
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "google/gemma-4-12b-it"
OUT_PATH = WORKDIR + "/probe_v1.jsonl"
DONE_SEED = []
RESCUES = {"django__django-11564", "django__django-11815", "django__django-11620"}
N_INSTANCES = 25

assert torch.cuda.is_available(), "no GPU"
print(f"preflight: GPU={torch.cuda.get_device_name(0)}", flush=True)

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

for _v in ("model", "tok"):
    if _v in globals():
        del globals()[_v]
gc.collect(); torch.cuda.empty_cache()
sys.last_traceback = None; sys.last_value = None; sys.last_type = None
tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto",
    attn_implementation="eager")
model.eval()
print("model loaded", flush=True)

def chat_generate(user_msg, max_new=220):
    msgs = [{"role": "user", "content": user_msg}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")["input_ids"].to(model.device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()

import string
LETTERS = string.ascii_uppercase
n_ok = n_err = 0
for idx in range(N_INSTANCES):
    inst = rows[idx]
    iid = inst["instance_id"]
    if iid in done_ids:
        print(f"[{idx+1}] {iid} SKIP (done)", flush=True)
        continue
    try:
        gold = gold_files(inst["patch"])
        gold_set = set(gold)
        repo_dir = fetch_repo_snapshot(inst["repo"], inst["base_commit"], WORKDIR + "/repos")
        all_files = sorted({c.repo_path for c in chunk_repo(repo_dir, "python")})
        rng = random.Random(1000 + idx)
        others = [f for f in all_files if f not in gold_set]
        distract = rng.sample(others, k=min(9, len(others)))
        cands = gold[:1] + distract
        rng.shuffle(cands)
        gold_letter = LETTERS[cands.index(gold[0])]
        listing = "\n".join(f"{LETTERS[i]}. {f}" for i, f in enumerate(cands))
        q = (f"Below is a bug report from the {inst['repo']} project, followed by a list of "
             f"source files from the repository at the relevant commit. Which single file is "
             f"most likely modified to fix this bug? Answer with just the letter.\n\n"
             f"--- BUG REPORT ---\n{inst['problem_statement'][:4000]}\n"
             f"--- FILES ---\n{listing}\n\nAnswer (single letter):")
        ans = chat_generate(q, max_new=8)
        picked = next((ch for ch in ans if ch in LETTERS[:len(cands)]), "?")
        open_ans, open_hit = "", None
        if iid in RESCUES:
            q2 = (f"Bug report from {inst['repo']}:\n\n{inst['problem_statement'][:4000]}\n\n"
                  f"Which source file would you modify to fix this? Reply with just the file path.")
            open_ans = chat_generate(q2, max_new=80)
            open_hit = any(g.rstrip('/') in open_ans for g in gold) or gold[0] in open_ans
        row = {"instance": iid, "repo": inst["repo"], "gold": gold[0],
               "gold_letter": gold_letter, "picked": picked,
               "correct": picked == gold_letter, "n_cands": len(cands),
               "open_answer": open_ans[:200] if open_ans else "",
               "open_hit": open_hit}
        with open(OUT_PATH, "a") as f:
            f.write(json.dumps(row) + "\n")
        print("PROBE_ROW " + json.dumps(row), flush=True)
        n_ok += 1
    except Exception as e:
        row = {"instance": iid, "error": f"{type(e).__name__}: {str(e)[:200]}"}
        with open(OUT_PATH, "a") as f:
            f.write(json.dumps(row) + "\n")
        print("PROBE_ROW " + json.dumps(row), flush=True)
        n_err += 1

print(f"CELL12 COMPLETE. ok={n_ok} err={n_err} (+{len(done_ids)} resumed)", flush=True)
