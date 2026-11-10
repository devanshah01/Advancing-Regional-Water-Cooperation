"""Combine the four raters' sheets: Fleiss' kappa, majority vote, ties, and win rates (baseline vs updated).

  python3 fleiss_ratings.py --raters rater1.csv rater2.csv rater3.csv rater4.csv --key rating_key.csv
Each rater file has item_id, correct (0/1), grounded (0/1). The key (kept by the coordinator, NOT given to
raters) maps item_id -> id (query) and system (baseline | updated). Majority vote needs >= 3 of 4;
2-2 ties are written to ties.csv for the team to settle by discussion. Writes ratings_final.csv, ties.csv, win_rates.csv.
"""
import argparse
import csv
import numpy as np
from stats_utils import fleiss_kappa, mcnemar_exact, wilson, fmt_p


def read(path):
    return {r["item_id"]: r for r in csv.DictReader(open(path, newline="", encoding="utf-8"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raters", nargs="+", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--out", default=".")
    ap.add_argument("--resolved", default=None, help="csv item_id,criterion,value with ties settled by discussion")
    a = ap.parse_args()
    raters = [read(p) for p in a.raters]
    key = read(a.key)
    ids = [i for i in key if all(i in r for r in raters)]
    resolved = {}
    if a.resolved:
        for r in csv.DictReader(open(a.resolved, newline="")):
            resolved[(r["item_id"], r["criterion"])] = int(r["value"])
    final, ties = {}, []
    print(f"{len(ids)} items rated by {len(raters)} raters")
    for crit in ("correct", "grounded"):
        M = np.array([[int(r[i][crit]) for r in raters] for i in ids])
        print(f"Fleiss' kappa ({crit}): {fleiss_kappa(M):.3f}   (raters agree unanimously on {100*np.mean(M.min(1)==M.max(1)):.0f}% of items)")
        for i, row in zip(ids, M):
            s = int(row.sum())
            if (i, crit) in resolved:
                v = resolved[(i, crit)]
            elif s * 2 == len(row):
                ties.append({"item_id": i, "criterion": crit, "votes": "".join(map(str, row))}); v = ""
            else:
                v = int(s * 2 > len(row))
            final.setdefault(i, {})[crit] = v
    with open(f"{a.out}/ratings_final.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["item_id", "id", "system", "correct", "grounded"])
        for i in ids: w.writerow([i, key[i]["id"], key[i]["system"], final[i]["correct"], final[i]["grounded"]])
    with open(f"{a.out}/ties.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["item_id", "criterion", "votes"]); w.writeheader(); w.writerows(ties)
    print(f"{len(ties)} ties written to ties.csv (settle by discussion, then rerun with --resolved)")

    # win rates: per query, updated vs baseline (only complete pairs)
    by = {}
    for i in ids:
        by.setdefault(key[i]["id"], {})[key[i]["system"]] = final[i]
    out = []
    for crit in ("correct", "grounded"):
        pairs = [(d["baseline"][crit], d["updated"][crit]) for d in by.values()
                 if "baseline" in d and "updated" in d and d["baseline"][crit] != "" and d["updated"][crit] != ""]
        x, y = [p[0] for p in pairs], [p[1] for p in pairs]
        wins = sum(1 for p in pairs if p[1] > p[0]); losses = sum(1 for p in pairs if p[1] < p[0]); same = len(pairs) - wins - losses
        mc = mcnemar_exact(x, y) if pairs else {"p": float("nan")}
        out.append({"criterion": crit, "n_pairs": len(pairs), "baseline_rate": f"{sum(x)}/{len(x)}", "updated_rate": f"{sum(y)}/{len(y)}",
                    "updated_wins": wins, "baseline_wins": losses, "ties": same, "p_exact_mcnemar": fmt_p(mc["p"]) if pairs else ""})
    with open(f"{a.out}/win_rates.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
    for o in out: print(o)


if __name__ == "__main__":
    main()
