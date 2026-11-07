# Released as part of the localization audit kit. Originally a Google Colab cell; runs as a script from the
# release root (python harness/<name>.py) with a GPU. Working files go to $RECALLFIRST_WORKDIR (default /content).
import os as _os
WORKDIR = _os.environ.get("RECALLFIRST_WORKDIR", "/content")
_os.makedirs(WORKDIR, exist_ok=True)
# === Phase 3 (v3, harness fixed 2026-11-07): one-turn patch agent, BM25-top5 vs dense(MiniLM)-top5 ===
# v2 (bm25 vs dense) produced 0 valid results because of 5 harness bugs, all fixed here:
#  1 parse_diff dropped every context line + the trailing newline  -> keep the diff block verbatim
#  2 repo snapshot excluded tests/ (nami.data skips tests/docs)    -> separate FULL snapshot incl. tests
#  3 snapshot was not a git repo, `git checkout` was a no-op         -> git init + commit, reset --hard/clean
#  4 F2P labels "test_x (mod.Class)" passed raw to runtests.py      -> converted to "mod.Class.test_x"
#  5 SWE-bench test_patch never applied (F2P tests didn't exist)    -> applied and committed as baseline
# Sanity per instance (no GPU): baseline must FAIL F2P, baseline + GOLD patch must PASS F2P. If the gold
# patch does not pass, the environment can't evaluate this instance -> env_fail, no generation spent.
# Old versions: experiments/cell13_backup_20261107_bm25_vs_fused.py (v1),
#               experiments/cell13_v2_backup_20261107_before_harnessfix.py (v2).
import os, sys, time, json, gc, subprocess, re, io, tarfile, urllib.request
import torch

from nami.chunker import chunk_repo
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "google/gemma-4-12b-it"
RERANK = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "data", "orderings_wholerepo.jsonl")
OUT_PATH = WORKDIR + "/patchgen_v4.jsonl"
FULL_DIR = WORKDIR + "/repos_full"
MAX_INSTANCES = 12
CHUNK_CHARS = 1500
MAX_NEW = 1024
TEST_TIMEOUT = 300
DONE_SEED = []

assert os.path.exists(RERANK), f"missing {RERANK} - run cell 11 first (or restore file)"
# legacy-cgi: Django < 4.2 imports `cgi`, removed in Python 3.13; setuptools supplies distutils
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "sqlparse", "pytz", "asgiref",
                "legacy-cgi", "setuptools"], check=False)

_ds = load_dataset("princeton-nlp/SWE-bench_Lite")
_split = "test" if "test" in _ds else list(_ds.keys())[0]
rows_dt = {r["instance_id"]: r for r in _ds[_split]}

latest = {}
for line in open(RERANK):
    try:
        r = json.loads(line)
    except Exception:
        continue
    if r.get("error") or r.get("skip_reason"):
        continue
    if r.get("repo") == "django/django":
        latest[r["instance"]] = r
cands = list(latest.values())[:MAX_INSTANCES]
print(f"patch-agent candidates: {len(cands)} django instances", flush=True)

done_keys = set(DONE_SEED)
if os.path.exists(OUT_PATH):
    for line in open(OUT_PATH):
        try:
            r = json.loads(line)
            if "error" not in r:
                done_keys.add(f'{r["instance"]}|{r["condition"]}')
        except Exception:
            pass
print(f"checkpoint: {len(done_keys)} done", flush=True)

# ---------------- environment helpers (CPU only) ----------------
GIT_ENV = dict(os.environ, GIT_AUTHOR_NAME="h", GIT_AUTHOR_EMAIL="h@x", GIT_COMMITTER_NAME="h",
               GIT_COMMITTER_EMAIL="h@x")

def sh(args, cwd, inp=None, timeout=120):
    return subprocess.run(args, cwd=cwd, input=inp, capture_output=True, text=True,
                          timeout=timeout, env=GIT_ENV)

def fetch_full(repo, commit):
    """Full tarball (tests included) as a git repo with one commit. Cached."""
    owner, name = repo.split("/")
    root = os.path.join(FULL_DIR, f"{name}-{commit}")
    if os.path.isdir(os.path.join(root, ".git")):
        return root
    os.makedirs(FULL_DIR, exist_ok=True)
    if os.path.isdir(root):                      # stale partial checkout (no .git): remove BEFORE extracting
        subprocess.run(["rm", "-rf", root])
    data = urllib.request.urlopen(f"https://codeload.github.com/{owner}/{name}/tar.gz/{commit}",
                                  timeout=180).read()
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
        top = tf.getmembers()[0].name.split("/")[0]
        tf.extractall(FULL_DIR)
    src_dir = os.path.join(FULL_DIR, top)
    if os.path.abspath(src_dir) != os.path.abspath(root):   # codeload names it {name}-{sha}: often == root
        os.rename(src_dir, root)
    sh(["git", "init", "-q"], root); sh(["git", "add", "-A"], root)
    sh(["git", "commit", "-qm", "base"], root, timeout=300)
    return root

def reset(repo_dir):
    sh(["git", "reset", "--hard", "-q"], repo_dir); sh(["git", "clean", "-fdq"], repo_dir)

def _strip_level(diff):
    """1 if the ---/+++ headers carry git-style a/ b/ prefixes, else 0 (models often omit them)."""
    heads = [l[4:].split("\t")[0].strip() for l in diff.splitlines() if l.startswith(("--- ", "+++ "))]
    heads = [h for h in heads if h != "/dev/null"]
    return 1 if heads and all(h.startswith(("a/", "b/")) for h in heads) else 0

def apply_patch(repo_dir, diff):
    """git apply (recount) -> fallback GNU patch with fuzz, at the strip level the headers imply
    (v4 fix: v3 always used -p1, which broke diffs without a/ b/ prefixes). Returns (ok, method, err)."""
    lvl = _strip_level(diff)
    errs = []
    for p in (lvl, 1 - lvl):                       # implied level first, then the other one
        a = sh(["git", "apply", "--recount", "--whitespace=nowarn", f"-p{p}", "-"], repo_dir, inp=diff)
        if a.returncode == 0:
            return True, f"git-p{p}", ""
        b = sh(["patch", f"-p{p}", "--fuzz=1", "--batch", "--forward", "--no-backup-if-mismatch"], repo_dir, inp=diff)
        if b.returncode == 0:
            return True, f"patch-fuzz-p{p}", ""
        reset(repo_dir)                                  # undo partial hunks + .rej files before the next try
        errs.append(a.stderr[-120:] + " | " + b.stdout[-120:])
    return False, "none", " || ".join(errs)

_LBL = re.compile(r"^\s*(.*?)\s+\(([\w.]+)\)\s*$")
def to_labels(tests):
    out = []
    for t in tests:
        m = _LBL.match(t)
        if m and m.group(1).isidentifier():
            lab = f"{m.group(2)}.{m.group(1)}"
        elif m:
            lab = m.group(2)                 # docstring-style name -> run the whole class
        else:
            lab = t.removeprefix("tests.")
        if lab not in out:
            out.append(lab)
    return out

def run_tests(repo_dir, labels):
    env = dict(os.environ, PYTHONUNBUFFERED="1",
               PYTHONPATH=os.pathsep.join(p for p in (repo_dir, os.environ.get("PYTHONPATH", "")) if p))
    try:
        p = subprocess.run([sys.executable, "tests/runtests.py", "--settings=test_sqlite",
                            "--parallel", "1", "-v", "1"] + labels,
                           cwd=repo_dir, env=env, capture_output=True, text=True, timeout=TEST_TIMEOUT)
        out = p.stdout + p.stderr
        m = re.search(r"Ran (\d+) test", out)
        return p.returncode == 0, (int(m.group(1)) if m else 0), out[-300:]
    except subprocess.TimeoutExpired:
        return False, 0, "TIMEOUT"

def parse_diff(ans):
    """Keep the diff verbatim (context lines included); ensure trailing newline."""
    seg = None
    for fence in ("```diff", "```patch", "```"):
        if fence in ans:
            seg = ans.split(fence, 1)[1]
            seg = seg.split("```", 1)[0]
            break
    if seg is None:
        m = re.search(r"^(diff --git |--- )", ans, re.M)
        seg = ans[m.start():] if m else ""
    seg = seg.lstrip("\n").replace("\r\n", "\n")
    if not re.search(r"^@@", seg, re.M):
        return ""
    return seg if seg.endswith("\n") else seg + "\n"

def build_context(src_dir, files):
    chunks = [c for c in chunk_repo(src_dir, "python") if c.repo_path in set(files)]
    by_file = {}
    for c in chunks:
        by_file.setdefault(c.repo_path, []).append(c.text[:CHUNK_CHARS])
    out = []
    for f in files:
        best = sorted(by_file.get(f, []), key=len, reverse=True)[:2]
        for i, t in enumerate(best):
            out.append(f"--- {f} (excerpt {i+1}) ---\n{t}")
    return "\n\n".join(out)

# ---------------- model ----------------
for _v in ("model", "tok", "scorer", "dense_models", "dense_model", "dense_tok", "dense"):
    if _v in globals():
        del globals()[_v]
gc.collect(); torch.cuda.empty_cache()
sys.last_traceback = None; sys.last_value = None; sys.last_type = None
_free, _tot = torch.cuda.mem_get_info()
print(f"GPU free before load: {_free/1e9:.1f} / {_tot/1e9:.1f} GB", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="eager")
model.eval()
print("model loaded", flush=True)

def chat_generate(user_msg, max_new=MAX_NEW):
    ids = tok.apply_chat_template([{"role": "user", "content": user_msg}],
                                  add_generation_prompt=True, return_tensors="pt")["input_ids"].to(model.device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()

def write(row):
    with open(OUT_PATH, "a") as f:
        f.write(json.dumps(row) + "\n")
    print("PATCH_ROW " + json.dumps(row), flush=True)

# ---------------- main loop ----------------
n_done = 0
for prog, rr in enumerate(cands, 1):
    iid = rr["instance"]
    inst = rows_dt[iid]
    conds = [("bm25", rr["orders"]["bm25"][:5]), ("dense", rr["orders"]["dense"][:5])]
    if rr["orders"].get("gemma_names"):
        conds.append(("gemma", rr["orders"]["gemma_names"][:5]))
    if all(f"{iid}|{c}" in done_keys for c, _ in conds):
        continue
    try:
        repo_dir = fetch_full(inst["repo"], inst["base_commit"])
        reset(repo_dir)
        ok_tp, _, err_tp = apply_patch(repo_dir, inst["test_patch"])
        if not ok_tp:
            raise RuntimeError(f"test_patch did not apply: {err_tp}")
        sh(["git", "add", "-A"], repo_dir); sh(["git", "commit", "-qm", "tests"], repo_dir)
        labels = to_labels(json.loads(inst["FAIL_TO_PASS"]))
        base_ok, base_n, base_tail = run_tests(repo_dir, labels)
        ok_g, _, err_g = apply_patch(repo_dir, inst["patch"])
        gold_ok, gold_n, gold_tail = run_tests(repo_dir, labels) if ok_g else (False, 0, err_g)
        reset(repo_dir)
        env_ok = (not base_ok) and gold_ok and gold_n > 0
        print(f"[{prog}] {iid} env: baseline_pass={base_ok} (ran {base_n}) gold_pass={gold_ok} "
              f"(ran {gold_n}) -> env_ok={env_ok}", flush=True)
    except Exception as e:
        for c, _ in conds:
            write({"instance": iid, "condition": c, "error": f"env {type(e).__name__}: {str(e)[:200]}"})
        continue
    for condition, files in conds:
        key = f"{iid}|{condition}"
        if key in done_keys:
            continue
        gold = rr["gold"] if isinstance(rr["gold"], list) else [rr["gold"]]
        row = {"instance": iid, "condition": condition, "top5": files,
               "gold_in_top5": any(g in files for g in gold), "labels": labels,
               "env_ok": env_ok, "baseline_pass": base_ok, "gold_pass": gold_ok,
               "gold_tail": "" if gold_ok else gold_tail[-200:]}
        if not env_ok:
            row.update({"apply_ok": False, "resolved": None, "note": "env_fail: gold patch does not pass "
                        "or baseline already passes - instance not evaluable"})
            write(row); n_done += 1
            continue
        tI = time.time()
        try:
            ctx = build_context(repo_dir, files)
            q = ("You are an expert Python developer fixing a bug in the Django repository. "
                 "Below is the bug report and code excerpts from the 5 most relevant source files. "
                 "Write a minimal patch that fixes the issue. Output ONLY a unified diff that applies "
                 "with `git apply -p1` (paths like a/django/... and b/django/...), including 3 lines of "
                 "unchanged context around each change with exact original indentation. No explanations.\n\n"
                 f"--- BUG REPORT ---\n{inst['problem_statement'][:3500]}\n\n"
                 f"--- CODE CONTEXT ---\n{ctx}\n\n--- PATCH (unified diff) ---")
            t0 = time.time()
            ans = chat_generate(q)
            row["gen_s"] = round(time.time() - t0)
            diff = parse_diff(ans)
            row["diff_lines"] = len(diff.splitlines())
            if not diff:
                row.update({"apply_ok": False, "resolved": False, "note": "no diff parsed",
                            "raw_head": ans[:200]})
            else:
                ok, method, err = apply_patch(repo_dir, diff)
                row.update({"apply_ok": ok, "apply_method": method})
                if not ok:
                    row.update({"resolved": False, "note": err[-200:], "diff_head": diff[:300]})
                else:
                    passed, n_ran, tail = run_tests(repo_dir, labels)
                    row.update({"resolved": bool(passed and n_ran > 0), "tests_ran": n_ran,
                                "test_tail": tail[-200:]})
            reset(repo_dir)
            write(row); n_done += 1
            print(f"[{prog}] {iid} {condition} {(time.time()-tI)/60:.1f}m apply={row.get('apply_ok')} "
                  f"resolved={row.get('resolved')}", flush=True)
        except Exception as e:
            reset(repo_dir)
            write({"instance": iid, "condition": condition, "error": f"{type(e).__name__}: {str(e)[:200]}"})
            if "out of memory" in str(e).lower():
                gc.collect(); torch.cuda.empty_cache()

# ---------------- summary ----------------
lat = {}
for line in open(OUT_PATH):
    try:
        x = json.loads(line)
    except Exception:
        continue
    if "error" not in x:
        lat[(x["instance"], x["condition"])] = x
for c in ("bm25", "dense", "gemma"):
    R = [x for (i, cc), x in lat.items() if cc == c]
    ev = [x for x in R if x.get("env_ok")]
    print(f"SUMMARY {c}: evaluable {len(ev)}/{len(R)} | gold-in-top5 {sum(x['gold_in_top5'] for x in ev)} | "
          f"applied {sum(bool(x.get('apply_ok')) for x in ev)} | resolved {sum(bool(x.get('resolved')) for x in ev)}",
          flush=True)
print(f"CELL13 COMPLETE. rows this run: {n_done}", flush=True)
