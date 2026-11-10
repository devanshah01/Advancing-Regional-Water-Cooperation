"""H1: table-preserving chunking (and k = 4) vs the 500-character, top-3 baseline.

Inputs (written by evaluate.py on the 42-query test set, temperature 0):
  --fixed  results_42/qa_fixed.csv   conditions k=3|basic, k=4|basic, baseline(no retrieval)
  --table  results_42/qa_table.csv   conditions k=3|basic, k=4|basic  (table-preserving chunking)
Primary test: source_hit on the answerable queries, exact McNemar, Bonferroni alpha = 0.05/3.
Also reports fact recall in retrieved text, answers containing every expected fact, false refusals and correct refusals.
Writes h1_table1.csv (Table 1) and h1_tests.csv.
"""
import argparse
import csv
import numpy as np
from stats_utils import wilson, mcnemar_exact, paired_diff_ci, boot_mean_ci, bonferroni_alpha, fmt_p


def load(path, cond):
    rows = [r for r in csv.DictReader(open(path, newline="", encoding="utf-8")) if r["condition"] == cond and r["repeat"] == "1"]
    return {r["id"]: r for r in rows}


def metrics(d):
    ans = [r for r in d.values() if r["in_scope"] == "1"]
    out = [r for r in d.values() if r["in_scope"] == "0"]
    f = lambda x: float(x) if x not in ("", None) else np.nan
    return {
        "n_ans": len(ans), "n_out": len(out),
        "hit": [int(float(r["source_hit"])) if r["source_hit"] != "" else 0 for r in ans],
        "recall": [f(r["context_recall"]) for r in ans],
        "complete": [int(f(r["context_recall"]) == 1.0) for r in ans],
        "allfacts": [int(f(r["keyword_coverage"]) == 1.0) for r in ans],
        "cover": [f(r["keyword_coverage"]) for r in ans],
        "false_ref": [int(r["refused"]) for r in ans],
        "correct_ref": [int(r["refused"]) for r in out],
        "ids_ans": [r["id"] for r in ans], "ids_out": [r["id"] for r in out],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixed", required=True)
    ap.add_argument("--table", required=True)
    ap.add_argument("--out", default="h1")
    a = ap.parse_args()
    systems = {
        "Baseline (500-char chunks, top 3)": metrics(load(a.fixed, "k=3|basic")),
        "Fixed chunks, top 4": metrics(load(a.fixed, "k=4|basic")),
        "Table-preserving chunks, top 3": metrics(load(a.table, "k=3|basic")),
        "Updated (table-preserving chunks, top 4)": metrics(load(a.table, "k=4|basic")),
    }
    rows = []
    for name, m in systems.items():
        nA, nO = m["n_ans"], m["n_out"]
        h, c, fr, cr = sum(m["hit"]), sum(m["complete"]), sum(m["false_ref"]), sum(m["correct_ref"])
        rec = boot_mean_ci(np.nan_to_num(m["recall"]))
        cov = boot_mean_ci(np.nan_to_num(m["cover"]))
        rows.append({
            "system": name, "n_answerable": nA, "n_unanswerable": nO,
            "correct_retrieval": f"{h}/{nA} ({100*h/nA:.0f}%)", "correct_retrieval_ci": "[%.0f, %.0f]" % tuple(100 * x for x in wilson(h, nA)),
            "fact_recall_mean": f"{100*rec[0]:.0f}%", "fact_recall_ci": f"[{100*rec[1]:.0f}, {100*rec[2]:.0f}]",
            "all_facts_in_answer": f"{sum(m['allfacts'])}/{nA}", "answer_fact_coverage_mean": f"{100*cov[0]:.0f}%",
            "false_refusals": f"{fr}/{nA} ({100*fr/nA:.0f}%)", "false_refusals_ci": "[%.0f, %.0f]" % tuple(100 * x for x in wilson(fr, nA)),
            "correct_refusals": f"{cr}/{nO} ({100*cr/nO:.0f}%)", "correct_refusals_ci": "[%.0f, %.0f]" % tuple(100 * x for x in wilson(cr, nO)),
        })
    with open(a.out + "_table1.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    base = systems["Baseline (500-char chunks, top 3)"]
    alpha = bonferroni_alpha(0.05, 3)
    tests = []
    for name in ["Table-preserving chunks, top 3", "Updated (table-preserving chunks, top 4)", "Fixed chunks, top 4"]:
        s = systems[name]
        for metric, label in [("hit", "correct retrieval (PRIMARY)"), ("complete", "all expected facts retrieved"),
                              ("allfacts", "all expected facts in answer"), ("false_ref", "false refusals")]:
            x, y = base[metric], s[metric]
            mc = mcnemar_exact(x, y)
            d, lo, hi = paired_diff_ci(x, y)
            tests.append({"comparison": f"{name} vs baseline", "metric": label, "n": len(x), "baseline": sum(x), "system": sum(y),
                          "only_baseline": mc["only_1"], "only_system": mc["only_2"],
                          "diff_pct_points": f"{100*d:+.1f}", "diff_ci": f"[{100*lo:+.1f}, {100*hi:+.1f}]",
                          "p_exact_mcnemar": fmt_p(mc["p"]), "significant_at_bonferroni_alpha": int(mc["p"] < alpha)})
    with open(a.out + "_tests.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(tests[0].keys())); w.writeheader(); w.writerows(tests)
    print(f"Bonferroni alpha = {alpha:.4f}\n")
    for r in rows:
        print(r)
    print()
    for t in tests:
        print(t)


if __name__ == "__main__":
    main()
