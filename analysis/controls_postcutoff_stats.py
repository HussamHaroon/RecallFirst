"""Exact paired stats for cells 15 (controls) and 16 (post-cutoff). Usage: python analysis/c15_c16_stats.py"""
import json, math, random
from collections import defaultdict
B = "data/orderings_wholerepo.jsonl"; C15 = "data/controls_probe57.jsonl"; C16 = "data/postcutoff_live120.jsonl"

def mcnemar(b, c):  # exact two-sided binomial
    n = b + c
    if n == 0: return 1.0
    k = min(b, c); p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)
def pair(name, A, Bv):  # A, Bv: dict id->bool
    ids = sorted(set(A) & set(Bv)); b = sum(A[i] and not Bv[i] for i in ids); c = sum(Bv[i] and not A[i] for i in ids)
    p = mcnemar(b, c); TESTS.append((name, b, c, p))
    print(f"  {name:58s} n={len(ids)} {sum(A[i] for i in ids)} vs {sum(Bv[i] for i in ids)}  +{b}/-{c}  p={p:.4f}")
TESTS = []
hit = lambda r, k: r is not None and r <= k

base = {}
for l in open(B):
    x = json.loads(l)
    if "ranks" not in x: continue
    rk = lambda o: next((i for i, f in enumerate(x["orders"].get(o) or [], 1) if f == x["gold"]), None)
    base[x["instance"]] = {m: rk(m) for m in ("nami_bpt", "bm25", "dense", "gemma_paths", "gemma_names")}
c15 = defaultdict(dict)
for l in open(C15):
    x = json.loads(l); c15[x["condition"]][x["instance"]] = x
ids = sorted(c15["bm25pool"]); assert len(ids) == 57 and all(i in base for i in ids), [i for i in ids if i not in base]
print("== Lite controls, same 57 instances (paper 4.1, 4.3, 4.4) ==")
for k in (1, 5):
    G = {i: hit(c15["bm25pool"][i]["rank"], k) for i in ids}; N = {i: hit(base[i]["nami_bpt"], k) for i in ids}
    Bm = {i: hit(base[i]["bm25"], k) for i in ids}
    pair(f"top{k} Gemma(BM25-25 pool) vs NAMI(BM25-25)", G, N)
    pair(f"top{k} Gemma(BM25-25 pool) vs BM25", G, Bm)
inpool = [i for i in ids if c15["bm25pool"][i]["gold_in_pool"]]
print(f"  in-pool n={len(inpool)}: Gemma t5 {sum(hit(c15['bm25pool'][i]['rank'],5) for i in inpool)}  NAMI t5 {sum(hit(base[i]['nami_bpt'],5) for i in inpool)}  BM25 t5 {sum(hit(base[i]['bm25'],5) for i in inpool)}")
# random-order reference on BM25-25: P(gold in top5 | in pool) = 5/25
print(f"  random order of BM25-25: expected top5 = {len(inpool)*5/25:.1f}/57 ({len(inpool)*5/25/57:.0%}); top1 = {len(inpool)/25:.1f}")
P = c15["probe57"]
for name, key in (("path_tiebreak", "path_tiebreak_correct"), ("dense", "dense_correct"), ("bm25", "bm25_correct")):
    pair(f"probe57 Gemma vs {name}", {i: P[i]["correct"] for i in ids}, {i: P[i][key] for i in ids})
nf = [i for i in ids if not P[i]["stem_in_issue"]]
for name, key in (("path_tiebreak", "path_tiebreak_correct"), ("dense", "dense_correct")):
    pair(f"probe57 name-free(n={len(nf)}) Gemma vs {name}", {i: P[i]["correct"] for i in nf}, {i: P[i][key] for i in nf})
for k in (1, 5):
    Real = {i: hit(base[i]["gemma_paths"], k) for i in ids}
    for cond in ("union_empty", "union_mismatch", "union_seed1", "union_seed2"):
        pair(f"top{k} Gemma real issue vs {cond}", Real, {i: hit(c15[cond][i]["rank"], k) for i in ids})
# shuffle stability
S = ["gemma_paths", "union_seed1", "union_seed2"]
def r_of(i, s): return base[i]["gemma_paths"] if s == "gemma_paths" else c15[s][i]["rank"]
for k in (1, 5):
    v = [sum(hit(r_of(i, s), k) for i in ids) for s in S]; m = sum(v) / 3
    sd = (sum((a - m) ** 2 for a in v) / 2) ** .5
    agree = sum(len({hit(r_of(i, s), k) for s in S}) == 1 for i in ids)
    print(f"  shuffles top{k}: {v} mean {m:.1f} sd {sd:.2f}; instances identical across 3 orders {agree}/57")

print("== Post-cutoff, SWE-bench-Live (paper 4.4) ==")
X = [json.loads(l) for l in open(C16)]; X = [x for x in X if "ranks" in x]
print(f"  n={len(X)} repos={len({x['repo'] for x in X})} dates {min(x['created_at'] for x in X)}..{max(x['created_at'] for x in X)} multi-file {sum(x['n_gold']>1 for x in X)} name-free {sum(not x['stem_in_issue'] for x in X)}")
D = {x["instance"]: x for x in X}
for sub, S2 in (("all", list(D)), ("name-free", [i for i in D if not D[i]["stem_in_issue"]]), ("single-file", [i for i in D if D[i]["n_gold"] == 1])):
    for k in (1, 5):
        g = {i: hit(D[i]["ranks"]["gemma_fill"], k) for i in S2}
        pair(f"live {sub} top{k} Gemma vs dense", g, {i: hit(D[i]["ranks"]["dense"], k) for i in S2})
        pair(f"live {sub} top{k} Gemma vs BM25", g, {i: hit(D[i]["ranks"]["bm25"], k) for i in S2})
# repo-clustered bootstrap of Gemma-dense top5 delta, Lite vs Live
def boot(rows, f, B=10000, seed=0):
    byr = defaultdict(list)
    for r in rows: byr[r[0]].append(r[1])
    reps = list(byr); rng = random.Random(seed); out = []
    for _ in range(B):
        s = [v for r in (rng.choice(reps) for _ in reps) for v in byr[r]]
        out.append(sum(s) / len(s))
    out.sort(); return out[int(.025 * B)], out[int(.975 * B)]
lite = [(base[i]["gemma_paths"] is not None and 0, 0) for i in ids]
lite = [(c15["bm25pool"][i]["repo"], hit(base[i]["gemma_paths"], 5) - hit(base[i]["dense"], 5)) for i in ids]
live = [(D[i]["repo"], hit(D[i]["ranks"]["gemma_fill"], 5) - hit(D[i]["ranks"]["dense"], 5)) for i in D]
for nm, rows in (("Lite", lite), ("Live", live)):
    d = sum(v for _, v in rows) / len(rows); lo, hi = boot(rows, None)
    print(f"  top5 delta Gemma-dense {nm}: {d:+.3f} [{lo:+.3f},{hi:+.3f}] repo-clustered 95% CI")
# ---- earlier-round McNemar tests reported in the paper (b/c recomputed in the v3 review pass; union-pool ones recomputed here)
for k in (1, 5):
    for g in ("gemma_paths", "gemma_names"):
        pair(f"union top{k} {g} vs dense", {i: hit(base[i][g], k) for i in ids}, {i: hit(base[i]["dense"], k) for i in ids})
pair("union top5 gemma_paths vs bm25", {i: hit(base[i]["gemma_paths"], 5) for i in ids}, {i: hit(base[i]["bm25"], 5) for i in ids})
pair("union top1 gemma_names vs gemma_paths", {i: hit(base[i]["gemma_names"], 1) for i in ids}, {i: hit(base[i]["gemma_paths"], 1) for i in ids})
for name, b, c in (("v3 4.1 RRF(BM25,NAMI) vs BM25 top5", 0, 8), ("v3 4.2 in-pool dense vs BM25 top5", 5, 6),
                   ("v3 4.2 in-pool NAMI vs BM25 top5", 3, 17), ("v3 4.2 in-pool QLM vs BM25 top5", 4, 19),
                   ("v3 4.2 in-pool no-null vs BM25 top5", 3, 17)):
    p = mcnemar(b, c); TESTS.append((name, b, c, p)); print(f"  {name:58s} +{b}/-{c}  p={p:.4f}")
def fam(n):
    if n.startswith(("top1 Gemma(BM25", "top5 Gemma(BM25", "v3 4.1")): return "4.1 same pool"
    if n.startswith("v3 4.2"): return "4.2 in-pool rerankers"
    if n.startswith("union"): return "4.3 union pool"
    if "real issue" in n: return "4.4 controls"
    if n.startswith("probe57"): return "4.4 probe"
    if n.startswith("live"): return "4.4 post-cutoff"
    return "other"
F = defaultdict(list)
for t in TESTS: F[fam(t[0])].append(t)
print(f"== Holm within families ({len(TESTS)} tests, {len(F)} families) ==")
for f in sorted(F):
    T = sorted(F[f], key=lambda t: t[3]); m = len(T); prev = 0
    print(f" [{f}] m={m}")
    for j, (n, b, c, p) in enumerate(T):
        adj = max(prev, min(1, (m - j) * p)); prev = adj
        print(f"  {n:58s} +{b}/-{c} p={p:.4f} holm={adj:.4f} {'SIG' if adj < .05 else 'ns'}")
