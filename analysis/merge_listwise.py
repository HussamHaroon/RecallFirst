"""Add Gemma listwise selections to the whole-repository records as ordinary orderings.

orders["gemma_paths"] / orders["gemma_names"] = Gemma's top-10 (paths only / paths + symbol names),
followed by the rest of its candidate pool (BM25 top-25 ∪ dense top-25) in BM25 order — the same
"fill" ordering used in the paper, so top-1/5/10 equal Gemma's own choices.
Idempotent: overwrites only those two keys. Usage (from the release root):
    python analysis/merge_listwise.py data/orderings_wholerepo.jsonl data/listwise_gemma.jsonl
"""
import json, sys

def main(rows_path, lw_path):
    lw = {}
    for l in open(lw_path, encoding="utf-8"):
        x = json.loads(l)
        if "error" not in x:
            lw[(x["instance"], x["condition"])] = x
    out, n = [], 0
    for l in open(rows_path, encoding="utf-8"):
        r = json.loads(l)
        if "orders" in r and not r.get("error"):
            pool = []
            for f in r["orders"]["bm25"][:25] + r["orders"]["dense"][:25]:
                if f not in pool:
                    pool.append(f)
            for cond, key in (("paths", "gemma_paths"), ("paths_sigs", "gemma_names")):
                x = lw.get((r["instance"], cond))
                if x is None:
                    continue
                top = x["gemma_top10"]
                rest = [f for f in r["orders"]["bm25"] if f in pool and f not in top]
                r["orders"][key] = top + rest + [f for f in pool if f not in top and f not in rest]
                n += 1
        out.append(r)
    with open(rows_path, "w", encoding="utf-8") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")
    print(f"merged {n} listwise orderings into {rows_path}")

if __name__ == "__main__":
    main(*sys.argv[1:3])
