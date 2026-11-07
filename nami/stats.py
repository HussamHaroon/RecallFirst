"""Paired evaluation statistics for file-localization orderings.

Quick start (CPU, seconds):
  python -m nami.stats data/orderings_wholerepo.jsonl                 # all methods vs BM25
  python -m nami.stats data/orderings_wholerepo.jsonl --baseline dense
  python -m nami.stats data/orderings_wholerepo.jsonl --add my_localizer.jsonl
  (my_localizer.jsonl: one JSON object per line: {"instance": "...", "order": ["path/a.py", ...]})
Use compare() for any set of methods; evaluate()/fusion_table() reproduce the paper's NAMI tables.

Original data contract (paper runs), one row per instance:
  instance, repo, protocol, gold,
  orders: {"bm25": [file,...], "nami_bpt": [...], "nami_bits": [...],
           "dense": [...], "fused_*": [...] (computed here, never dumped)},
  optional: meta {...}

Pre-registered fusion: PRIMARY = RRF(bm25, nami_bpt), k=60.
  Sensitivity (reported, not selected): RRF(bm25, nami_bits),
  RRF(bm25, bpt, bits), interleave(bm25, bpt).
Statistics: 95% bootstrap CIs (10k, seeded), paired Wilcoxon signed-rank on
  reciprocal ranks vs BM25, exact McNemar for paired binary outcomes.
"""
import json, math, os
from bisect import bisect_left

import numpy as np
from scipy import stats as sps

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ORDERINGS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "orderings_wholerepo.jsonl")

RRF_K = 60
BOOT_N = 10_000
SEED = 20261101


# ---------------------------------------------------------------- fusion
def rrf_scores(order):
    """Reciprocal-rank-fusion scores for one full ordering (list of file ids)."""
    return {f: 1.0 / (RRF_K + r) for r, f in enumerate(order, start=1)}


def rrf_fuse(*orders):
    """Fuse full orderings into one ordering (ties broken by BM25 position)."""
    acc = {}
    for order in orders:
        for f, s in rrf_scores(order).items():
            acc[f] = acc.get(f, 0.0) + s
    bm25_pos = {f: i for i, f in enumerate(orders[0])}
    return sorted(acc, key=lambda f: (-acc[f], bm25_pos.get(f, 10**9)))


def interleave(*orders):
    """Round-robin interleave; skip files already emitted."""
    out, seen = [], set()
    for round_i in range(max(len(o) for o in orders)):
        for order in orders:
            if round_i < len(order) and order[round_i] not in seen:
                out.append(order[round_i]); seen.add(order[round_i])
    return out


def rank_of(order, gold, miss_rank=None):
    """1-based rank of gold in order. Accepts str or list-of-str (multi-file
    gold: best rank over members); missing -> miss_rank (default len+1).
    For reranked partial orders pass a common miss_rank so methods compare fairly."""
    golds = [gold] if isinstance(gold, str) else list(gold)
    for i, f in enumerate(order):
        if f in golds:
            return i + 1
    return miss_rank if miss_rank is not None else len(order) + 1  # gold not in pool: worst


# ---------------------------------------------------------------- stats
def bootstrap_ci(values, stat=np.mean, n=BOOT_N, seed=SEED, alpha=0.05):
    """Percentile bootstrap CI for stat over instance-level values."""
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n, len(v)))
    samples = stat(v[idx], axis=1)
    lo, hi = np.quantile(samples, [alpha / 2, 1 - alpha / 2])
    return float(stat(v)), float(lo), float(hi)


def wilcoxon_rr(ranks_a, ranks_b):
    """Paired Wilcoxon signed-rank on reciprocal ranks (a vs b; >0 means a better)."""
    rr_a = 1.0 / np.asarray(ranks_a, dtype=float)
    rr_b = 1.0 / np.asarray(ranks_b, dtype=float)
    diff = rr_a - rr_b
    if np.allclose(diff, 0):
        return float("nan"), 1.0
    res = sps.wilcoxon(rr_a, rr_b, alternative="two-sided", zero_method="wilcox")
    return float(np.median(diff)), float(res.pvalue)


def mcnemar_exact(b_a, b_b):
    """Exact McNemar on paired binary outcomes (arrays of 0/1, same instances).

    Discordant pairs: n01 = b wins, n10 = a wins; p = binom two-sided on the
    smaller count. Returns (n01, n10, p).
    """
    a = np.asarray(b_a, dtype=int); b = np.asarray(b_b, dtype=int)
    n01 = int(np.sum((a == 1) & (b == 0)))  # a resolved, b not
    n10 = int(np.sum((a == 0) & (b == 1)))
    nd = n01 + n10
    if nd == 0:
        return n01, n10, 1.0
    k = min(n01, n10)
    p = min(1.0, 2 * sps.binom.cdf(k, nd, 0.5))
    return n01, n10, float(p)


# ---------------------------------------------------------------- tables
FUSIONS = {  # name -> (callable, orders used); PRIMARY first
    "RRF(bm25,bpt)":    lambda o: rrf_fuse(o["bm25"], o["nami_bpt"]),
    "RRF(bm25,bits)":   lambda o: rrf_fuse(o["bm25"], o["nami_bits"]),
    "RRF(bm25,bpt,bits)": lambda o: rrf_fuse(o["bm25"], o["nami_bpt"], o["nami_bits"]),
    "interleave(bm25,bpt)": lambda o: interleave(o["bm25"], o["nami_bpt"]),
    # v2 controls: does fusing NAMI beat fusing query likelihood / a code embedder?
    "RRF(bm25,qlm)":    lambda o: rrf_fuse(o["bm25"], o["qlm"]),
    "RRF(bm25,bpt,qlm)": lambda o: rrf_fuse(o["bm25"], o["nami_bpt"], o["qlm"]),
    "RRF(bm25,coderank)": lambda o: rrf_fuse(o["bm25"], o["coderank"]),
    "RRF(bm25,coderank,bpt)": lambda o: rrf_fuse(o["bm25"], o["coderank"], o["nami_bpt"]),
}


def load(path=ORDERINGS, keep_errors=False):
    """Load per-instance rows; rows with an 'error' or 'skip_reason' are dropped unless keep_errors."""
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    if not keep_errors:
        rows = [r for r in rows if not r.get("error") and not r.get("skip_reason") and "orders" in r]
    return rows


def add_method(rows, path, name=None):
    """Attach a new localizer's orderings to rows (in place). File: JSON lines with
    {"instance": id, "order": [file, ...]} (optionally "method"). Instances missing from the
    file get an empty order (= miss). Returns the method name used."""
    name = name or os.path.splitext(os.path.basename(path))[0]
    new = {}
    for l in open(path, encoding="utf-8"):
        if l.strip():
            x = json.loads(l); new[x["instance"]] = x["order"]
    for r in rows:
        r["orders"][name] = new.get(r["instance"], [])
    return name


def compare(rows, methods=None, baseline="bm25", ks=(1, 5, 10)):
    """Every method vs a baseline on the SAME instances, at equal budget k.

    methods: list of keys in row["orders"]; default = every key present in all rows.
    Per method and k: hits, bootstrap 95% CI, exact McNemar (+wins/-losses vs baseline);
    plus MRR and paired Wilcoxon on reciprocal rank vs baseline. A common miss rank per
    instance (longest order + 1) is used when the gold file is absent from an ordering.
    """
    if methods is None:
        methods = sorted(set.intersection(*[set(r["orders"]) for r in rows]))
    if baseline not in methods:
        methods = [baseline] + list(methods)
    ranks = {m: [] for m in methods}
    for r in rows:
        miss = max(len(v) for v in r["orders"].values()) + 1
        for m in methods:
            ranks[m].append(rank_of(r["orders"].get(m, []), r["gold"], miss))
    out = []
    for m in methods:
        rk = np.asarray(ranks[m], dtype=float)
        line = {"method": m, "n": len(rk), "MRR": round(float(np.mean(1.0 / rk)), 3)}
        for k in ks:
            hit = (rk <= k).astype(int)
            pt, lo, hi = bootstrap_ci(hit)
            line[f"top{k}"] = f"{int(hit.sum())}/{len(hit)} = {pt*100:.0f}% [{lo*100:.0f},{hi*100:.0f}]"
            if m != baseline:
                w, l, p = mcnemar_exact(hit, (np.asarray(ranks[baseline]) <= k).astype(int))
                line[f"top{k}_vs_{baseline}"] = f"+{w}/-{l} p={p:.3g}"
        if m != baseline:
            _, p = wilcoxon_rr(rk, ranks[baseline])
            line[f"wilcoxon_rr_vs_{baseline}"] = f"{p:.2g}"
        out.append(line)
    return out


def _print_table(lines):
    keys = []
    for l in lines:
        for k in l:
            if k not in keys: keys.append(k)
    print("\t".join(keys))
    for l in lines:
        print("\t".join(str(l.get(k, "")) for k in keys))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Paired comparison of localization orderings.")
    ap.add_argument("data", nargs="?", default=ORDERINGS)
    ap.add_argument("--add", action="append", default=[], help="JSONL of new orderings (repeatable)")
    ap.add_argument("--baseline", default="bm25")
    ap.add_argument("--methods", nargs="*", default=None)
    a = ap.parse_args()
    rows = load(a.data)
    for f in a.add:
        add_method(rows, f)
    _print_table(compare(rows, a.methods, a.baseline))


def evaluate(rows, ks=(1, 5, 10), fusion="RRF(bm25,bpt)"):
    """Equal-budget comparison: every method gets the same k files.

    Returns per-method rank lists and a summary table with bootstrap CIs and
    paired Wilcoxon p-values (vs BM25).
    """
    base = ["bm25", "nami_bpt", "nami_bits", "qlm", "h_real", "dense", "coderank"]
    methods = base + ["fused", "union_oracle"]
    ranks = {m: [] for m in methods}
    for row in rows:
        o = row["orders"]; gold = row["gold"]
        miss = max(len(v) for v in o.values()) + 1  # common worst rank per instance
        try:
            fused = FUSIONS[fusion](o)
        except KeyError:  # fusion needs an order these rows lack
            fused = []
        for m, order in [(m, o.get(m, [])) for m in base] + [("fused", fused)]:
            if order:
                ranks[m].append(rank_of(order, gold, miss))
        # oracle union at k files == min rank across the two fused components
        ranks["union_oracle"].append(min(rank_of(o["bm25"], gold, miss),
                                         rank_of(o["nami_bpt"], gold, miss)) if o.get("nami_bpt") else math.inf)

    summary = []
    n = len(rows)
    for m in methods:
        r = [x for x in ranks[m] if x != math.inf] or [math.inf]
        if all(x == math.inf for x in r):
            continue
        line = {"method": m, "n": len(r), "mean_rank": float(np.mean(r))}
        for k in ks:
            hits = (np.asarray(r) <= k).astype(int)
            point, lo, hi = bootstrap_ci(hits)
            line[f"top{k}"] = f"{point*100:.0f}% [{lo*100:.0f}, {hi*100:.0f}]"
        if m != "bm25" and len(r) == len(ranks["bm25"]):
            _, p = wilcoxon_rr(r, ranks["bm25"])
            line["wilcoxon_p_vs_bm25"] = round(p, 4)
        summary.append(line)
    return ranks, summary


# ---------------------------------------------------------------- v2 analyses
def fusion_table(rows, ks=(1, 5, 10)):
    """Every fusion at equal budget k vs BM25@k (exact McNemar on top-k hits)."""
    out = []
    bm = [rank_of(r["orders"]["bm25"], r["gold"]) for r in rows]
    for name, fn in FUSIONS.items():
        try:
            fr = [rank_of(fn(r["orders"]), r["gold"]) for r in rows]
        except KeyError:
            continue  # orders for this fusion not present in these rows
        line = {"fusion": name, "n": len(fr), "mean_rank": round(float(np.mean(fr)), 2)}
        for k in ks:
            a = (np.asarray(fr) <= k).astype(int); b = (np.asarray(bm) <= k).astype(int)
            n01, n10, p = mcnemar_exact(a, b)
            line[f"top{k}"] = f"{a.sum()}/{len(a)} vs bm25 {b.sum()} (+{n01}/-{n10}, p={p:.3f})"
        out.append(line)
    return out


def ceiling_table(rows):
    """Candidate-generation recall ceilings (rerank-protocol rows only)."""
    keys = sorted({k for r in rows for k in r.get("ceilings", {})})
    tab = {k: f"{sum(r['ceilings'][k] for r in rows if k in r.get('ceilings', {}))}/"
              f"{sum(1 for r in rows if k in r.get('ceilings', {}))}" for k in keys}
    if any("gold_in_cand" in r for r in rows):
        tab["gold_in_candidate_pool"] = f"{sum(bool(r.get('gold_in_cand')) for r in rows)}/{len(rows)}"
    return tab


def router_analysis(rows, a="nami_bpt", b="bm25"):
    """Pre-registered test: does NAMI's advantage over BM25 track issue style?

    dRR = 1/rank_a - 1/rank_b per instance. Reports mean dRR per style label,
    Spearman(ident_per_100w, dRR), one-sided Mann-Whitney prose > identifier,
    and the gold-stem-in-issue diagnostic split.
    """
    d, x, lab, stem = [], [], [], []
    for r in rows:
        if "issue_feats" not in r or a not in r["orders"]:
            continue
        miss = max(len(v) for v in r["orders"].values()) + 1
        ra = rank_of(r["orders"][a], r["gold"], miss)
        rb = rank_of(r["orders"][b], r["gold"], miss)
        d.append(1 / ra - 1 / rb); x.append(r["issue_feats"]["ident_per_100w"])
        lab.append(r.get("style", "?")); stem.append(bool(r.get("gold_stem_in_issue")))
    if len(d) < 5:
        return {"n": len(d), "note": "too few rows with issue_feats"}
    d = np.asarray(d); lab = np.asarray(lab); stem = np.asarray(stem)
    res = {"n": len(d)}
    for L in ("identifier", "mixed", "prose"):
        m = lab == L
        if m.any():
            res[f"mean_dRR[{L}]"] = (f"{d[m].mean():+.3f} (n={m.sum()}, {a} wins "
                                     f"{int((d[m] > 0).sum())}, {b} wins {int((d[m] < 0).sum())})")
    rho, p = sps.spearmanr(x, d)
    res["spearman(ident_density, dRR)"] = f"rho={rho:+.3f} p={p:.4f}"
    if (lab == "identifier").sum() >= 3 and (lab == "prose").sum() >= 3:
        u = sps.mannwhitneyu(d[lab == "prose"], d[lab == "identifier"], alternative="greater")
        res["MWU prose>identifier (one-sided)"] = f"p={u.pvalue:.4f}"
    for flag in (True, False):
        m = stem == flag
        if m.any():
            res[f"mean_dRR[gold_stem_in_issue={flag}]"] = f"{d[m].mean():+.3f} (n={m.sum()})"
    return res


def _smoke():  # self-test on synthetic data (was the old __main__ block)
    # smoke test on synthetic data
    rng = np.random.default_rng(0)
    rows = []
    for i in range(30):
        pool = [f"f{j}" for j in range(50)]
        rng.shuffle(pool)
        gold = pool[rng.integers(0, 10)]
        bm25 = pool[:]
        nami = pool[:]
        # make nami partially complementary
        nami.remove(gold); nami.insert(int(rng.integers(0, 8)), gold)
        rows.append({"instance": f"i{i}", "repo": "r/r", "protocol": "synthetic",
                     "gold": gold,
                     "orders": {"bm25": bm25, "nami_bpt": nami, "nami_bits": nami[:]}})
    ranks, table = evaluate(rows)
    assert all(1 <= r <= 51 for m in ranks for r in ranks[m] if r != math.inf)
    fused_beats = np.mean(ranks["fused"]) < np.mean(ranks["bm25"])
    print("smoke OK | synthetic mean ranks: bm25 %.1f fused %.1f | fused<=bm25: %s"
          % (np.mean(ranks["bm25"]), np.mean(ranks["fused"]), fused_beats))
    n01, n10, p = mcnemar_exact([1]*6 + [0]*4, [1]*3 + [0]*7)
    print(f"mcnemar smoke: n01={n01} n10={n10} p={p:.3f}")
    for line in table:
        print(line)
    # v2 paths: qlm/coderank orders, ceilings, issue feats, partial rerank orders
    for i, r in enumerate(rows):
        o = r["orders"]
        o["qlm"] = o["nami_bpt"][:25]           # partial (reranked pool) order
        o["coderank"] = list(reversed(o["bm25"]))
        r["ceilings"] = {"bm25@10": int(rank_of(o["bm25"], r["gold"]) <= 10)}
        r["gold_in_cand"] = r["gold"] in o["qlm"]
        r["issue_feats"] = {"ident_per_100w": float(i % 10)}
        r["style"] = ["identifier", "mixed", "prose"][i % 3]
        r["gold_stem_in_issue"] = i % 4 == 0
    ranks2, _ = evaluate(rows)
    assert len(ranks2["qlm"]) == len(rows) and max(ranks2["qlm"]) <= 51
    for line in fusion_table(rows):
        print(line)
    print(ceiling_table(rows))
    print(router_analysis(rows))
    print("v2 smoke OK")
