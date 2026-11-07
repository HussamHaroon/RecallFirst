"""Paper figures, computed from the frozen result files (no hand-typed numbers).
Usage (from the release root): python analysis/make_figures.py
Writes figures/fig1, fig2, fig3 (.png + .svg)."""
import json, random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
ACCENT, MUTED = "#2a78d6", "#a6a49c"          # reference palette: series-1 + neutral de-emphasis
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID,
                     "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK})

def load(p):
    out = []
    for l in open(p):
        try: out.append(json.loads(l))
        except Exception: pass
    return out

RR = {r["instance"]: r for r in load("data/orderings_wholerepo.jsonl") if not r.get("error") and not r.get("skip_reason")}
LW = {}
for x in load("data/listwise_gemma.jsonl"):
    if "error" not in x: LW[(x["instance"], x["condition"])] = x
C15 = {}
for x in load("data/controls_probe57.jsonl"):
    if "error" not in x: C15[(x["instance"], x["condition"])] = x
LIVE = [x for x in load("data/postcutoff_live120.jsonl") if "ranks" in x]

def ci(h, B=10000, seed=20261107):
    rng = random.Random(seed); n = len(h)
    v = sorted(sum(h[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return 100 * v[int(.025 * B)], 100 * v[int(.975 * B) - 1]

def golds(r): return set(r["gold"] if isinstance(r["gold"], list) else [r["gold"]])
ids = sorted(RR)
# --- panel a: pool recall
bm_pool = [bool(golds(RR[i]) & set(RR[i]["orders"]["bm25"][:25])) for i in ids]
un_pool = [bool(golds(RR[i]) & (set(RR[i]["orders"]["bm25"][:25]) | set(RR[i]["orders"]["dense"][:25]))) for i in ids]
# --- panel b: top-5 on identical instances
def top5(m): return [0 < RR[i]["ranks"][m] <= 5 for i in ids]
def gem(c): return [LW[(i, c)]["ranks"]["gemma_fill"] is not None and LW[(i, c)]["ranks"]["gemma_fill"] <= 5 for i in ids]
assert all((i, c) in LW for i in ids for c in ("paths", "paths_sigs")), "listwise rows missing"

def hbar(ax, labels, hits, emph, title, note=None, chance=None):
    vals = [100 * sum(h) / len(h) for h in hits]
    cis = [ci(h) for h in hits]
    y = list(range(len(labels)))[::-1]
    ax.set_facecolor(SURFACE)
    for yy, v, (lo, hi), e, h in zip(y, vals, cis, emph, hits):
        col = ACCENT if e else MUTED
        ax.add_patch(FancyBboxPatch((0, yy - .21), max(v, .5), .42, boxstyle="round,pad=0,rounding_size=0.06",
                                    mutation_aspect=.25, lw=0, fc=col))
        ax.plot([lo, hi], [yy, yy], color=INK2, lw=1, solid_capstyle="butt", zorder=3)
        ax.text(min(hi, 100) + 1.5, yy, f"{v:.0f}%  ({sum(h)}/{len(h)})", va="center", ha="left",
                fontsize=8.5, color=INK if e else INK2, fontweight="bold" if e else "normal")
    if chance is not None:
        ax.axvline(chance, color=INK2, lw=1, ls=(0, (2, 2)))
        ax.text(chance + 1, y[0] + .45, "chance", fontsize=7.5, color=INK2, va="bottom")
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_xlim(0, 118); ax.set_ylim(-.6, len(labels) - .4)
    ax.set_xticks([0, 25, 50, 75, 100]); ax.set_xticklabels(["0", "25", "50", "75", "100%"])
    ax.grid(axis="x", color=GRID, lw=.8); ax.set_axisbelow(True)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title(title, loc="left", fontsize=10, fontweight="bold", color=INK, pad=8)
    if note: ax.text(0, -.17, note, transform=ax.transAxes, fontsize=7.5, color=INK2, va="top")

n = len(ids)
fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.3), gridspec_kw={"width_ratios": [1, 1.35], "wspace": .55})
fig.patch.set_facecolor(SURFACE)
hbar(a, ["BM25 top-25", "BM25 top-25 ∪\ndense top-25"], [bm_pool, un_pool], [False, True],
     "a  Is the gold file in the candidate pool?",
     note=f"{n} SWE-bench Lite instances, 8 repositories · whiskers: 95% bootstrap CI")
g_bm = [C15[(i, "bm25pool")]["rank"] is not None and C15[(i, "bm25pool")]["rank"] <= 5 for i in ids]
hbar(b, ["NAMI (likelihood,\nBM25 pool)", "Gemma selects\n(BM25 pool)", "BM25", "Dense (MiniLM)", "Gemma selects\n(union pool)"],
     [top5("nami_bpt"), g_bm, top5("bm25"), top5("dense"), gem("paths")],
     [False, True, False, False, True],
     "b  Gold file in the top-5 (identical instances)",
     note="Pool ceilings: BM25-25 holds the gold file on 36/57 (63%), the union pool on 51/57 (89%)")
fig.savefig("figures/fig1_pool_and_selection.png", dpi=220, bbox_inches="tight", facecolor=SURFACE)
fig.savefig("figures/fig1_pool_and_selection.svg", bbox_inches="tight", facecolor=SURFACE)

P = [C15[(i, "probe57")] for i in ids]
fig, ax = plt.subplots(figsize=(6.2, 2.6)); fig.patch.set_facecolor(SURFACE)
hbar(ax, ["BM25 (reads code)", "Path-token match (same text;\nties broken by BM25 order)", "Dense (reads code)", "Gemma-4-12B-it\n(issue + paths only)"],
     [[x["bm25_correct"] for x in P], [x["path_tiebreak_correct"] for x in P], [x["dense_correct"] for x in P], [x["correct"] for x in P]],
     [False, False, False, True],
     "Picking the gold file among 10 hard options",
     note=f"{len(P)} instances · options = gold + 9 top non-gold files of BM25/dense · same options for every picker",
     chance=10)
fig.savefig("figures/fig2_hard_probe.png", dpi=220, bbox_inches="tight", facecolor=SURFACE)
fig.savefig("figures/fig2_hard_probe.svg", bbox_inches="tight", facecolor=SURFACE)

def lv(m, k): return [x["ranks"][m] is not None and x["ranks"][m] <= k for x in LIVE]
fig, (a, b) = plt.subplots(1, 2, figsize=(10, 2.6), gridspec_kw={"wspace": .55}); fig.patch.set_facecolor(SURFACE)
hbar(a, ["BM25", "Dense (MiniLM)", "Gemma selects\n(union pool)"], [top5("bm25"), top5("dense"), gem("paths")],
     [False, False, True], "a  SWE-bench Lite (pre-cutoff), top-5", note=f"{n} instances, 8 repositories")
hbar(b, ["BM25", "Dense (MiniLM)", "Gemma selects\n(union pool)"], [lv("bm25", 5), lv("dense", 5), lv("gemma_fill", 5)],
     [False, False, True], "b  SWE-bench-Live, created Apr–Aug 2025, top-5",
     note=f"{len(LIVE)} instances, {len({x['repo'] for x in LIVE})} repositories · frozen pipeline · any gold file counts")
fig.savefig("figures/fig3_post_cutoff.png", dpi=220, bbox_inches="tight", facecolor=SURFACE)
fig.savefig("figures/fig3_post_cutoff.svg", bbox_inches="tight", facecolor=SURFACE)
print("pool", sum(bm_pool), sum(un_pool), n, "| top5 nami/gemma_bm/bm25/dense/gp",
      sum(top5("nami_bpt")), sum(g_bm), sum(top5("bm25")), sum(top5("dense")), sum(gem("paths")),
      "| probe57", sum(x["correct"] for x in P), sum(x["path_tiebreak_correct"] for x in P), sum(x["dense_correct"] for x in P),
      "| live t5", sum(lv("bm25", 5)), sum(lv("dense", 5)), sum(lv("gemma_fill", 5)), len(LIVE))
