"""H2 (computed figures inserted verbatim) and H3 (constrained, quotation-backed stance prompt).

Input: results_42/policy.csv from evaluate_policy.py (7 targets x 3 repeats x conditions pilot_full, verbatim, constrained).
Pairing: brief (target, repeat) is paired across conditions.
  H2: 4 required figures x 21 briefs = 84 paired figure scores, pilot_full vs verbatim (and vs constrained).
  H3: 5 governments x 21 briefs = 105 paired stance scores (1 = stance equals the documented position in
      gold_stances.csv), pilot_full vs constrained; secondary: verbatim vs constrained.
Tests: exact McNemar (primary, as in the plan's fallback) and a mixed-effects logistic model with a random
intercept for brief (BinomialBayesMixedGLM, variational Bayes); when that model cannot be fit (for example,
one condition is 100% correct, which makes the effect unbounded) the result says so and McNemar stands.
Bonferroni alpha = 0.05/3 across H1-H3.
Writes policy_summary.csv (Table 2) and policy_tests.csv.
"""
import argparse
import csv
import numpy as np
from stats_utils import wilson, mcnemar_exact, paired_diff_ci, bonferroni_alpha, fmt_p

FIGS = ["fig_drip_ha", "fig_drip_cost", "fig_rice_ha", "fig_lining_pct"]
COUNTRIES = ["Kyrgyzstan", "Tajikistan", "Kazakhstan", "Uzbekistan", "Turkmenistan"]


def load(path):
    return list(csv.DictReader(open(path, newline="", encoding="utf-8")))


def paired_units(rows, cond, kind):
    """{(brief_id, item): 0/1} for figures or stances; unscorable stances are skipped."""
    out = {}
    for r in rows:
        if r["condition"] != cond:
            continue
        if kind == "fig":
            for k in FIGS:
                out[(r["brief_id"], k)] = int(r[k])
        else:
            for c in COUNTRIES:
                v = r[f"match_{c}"]
                if v != "":
                    out[(r["brief_id"], c)] = int(v)
    return out


def mixed_model(keys, x, y):
    """Logit with random intercept per brief; returns (odds_ratio, lo, hi) or a reason string."""
    try:
        import pandas as pd
        from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
        df = pd.DataFrame({"brief": [k[0] for k in keys] * 2, "y": list(x) + list(y),
                           "cond": [0] * len(x) + [1] * len(y)})
        if df.groupby("cond")["y"].nunique().min() < 2:
            return "not fit: one condition has no variation (perfect or zero accuracy)"
        m = BinomialBayesMixedGLM.from_formula("y ~ cond", {"brief": "0 + C(brief)"}, df).fit_vb()
        i = list(m.model.exog_names).index("cond")
        b, s = float(m.fe_mean[i]), float(m.fe_sd[i])
        return float(np.exp(b)), float(np.exp(b - 1.96 * s)), float(np.exp(b + 1.96 * s))
    except Exception as e:  # noqa: BLE001
        return f"not fit: {type(e).__name__}"


def compare(rows, c1, c2, kind, label):
    u1, u2 = paired_units(rows, c1, kind), paired_units(rows, c2, kind)
    keys = sorted(set(u1) & set(u2))
    x, y = [u1[k] for k in keys], [u2[k] for k in keys]
    mc = mcnemar_exact(x, y)
    d, lo, hi = paired_diff_ci(x, y)
    mm = mixed_model(keys, x, y)
    mm_txt = (f"OR {mm[0]:.2f} [{mm[1]:.2f}, {mm[2]:.2f}]" if isinstance(mm, tuple) else mm)
    return {"hypothesis": label, "comparison": f"{c2} vs {c1}", "unit": "figures" if kind == "fig" else "stances",
            "n_pairs": len(keys), f"{c1}_correct": sum(x), f"{c2}_correct": sum(y),
            "only_first": mc["only_1"], "only_second": mc["only_2"],
            "diff_pct_points": f"{100*d:+.1f}", "diff_ci": f"[{100*lo:+.1f}, {100*hi:+.1f}]",
            "p_exact_mcnemar": fmt_p(mc["p"]), "mixed_model": mm_txt}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--out", default="policy")
    a = ap.parse_args()
    rows = load(a.policy)
    conds = list(dict.fromkeys(r["condition"] for r in rows))

    summary = []
    for c in conds:
        sub = [r for r in rows if r["condition"] == c]
        fig_ok = sum(int(r[k]) for r in sub for k in FIGS); fig_n = 4 * len(sub)
        st = [int(r[f"match_{n}"]) for r in sub for n in COUNTRIES if r[f"match_{n}"] != ""]
        unscored = 5 * len(sub) - len(st)
        stances = [r[f"stance_{n}"] for r in sub for n in COUNTRIES if r[f"stance_{n}"]]
        n_sup = sum(1 for s in stances if s == "Supports"); n_opp = sum(1 for s in stances if s == "Opposes")
        # by position: upstream vs downstream stance agreement
        up = [int(r[f"match_{n}"]) for r in sub for n in ["Kyrgyzstan", "Tajikistan"] if r[f"match_{n}"] != ""]
        dn = [int(r[f"match_{n}"]) for r in sub for n in ["Kazakhstan", "Uzbekistan", "Turkmenistan"] if r[f"match_{n}"] != ""]
        summary.append({
            "condition": c, "n_briefs": len(sub),
            "figures_correct": f"{fig_ok}/{fig_n} ({100*fig_ok/fig_n:.0f}%)",
            "figures_ci": "[%.0f, %.0f]" % tuple(100 * x for x in wilson(fig_ok, fig_n)),
            "stances_matching": f"{sum(st)}/{len(st)} ({100*sum(st)/max(len(st),1):.0f}%)",
            "stances_ci": "[%.0f, %.0f]" % tuple(100 * x for x in wilson(sum(st), len(st))),
            "stances_not_scorable": unscored,
            "upstream_matching": f"{sum(up)}/{len(up)}", "downstream_matching": f"{sum(dn)}/{len(dn)}",
            "supports_count": f"{n_sup}/{len(stances)}", "opposes_count": f"{n_opp}/{len(stances)}",
            "median_feasibility_1_5": "", "median_balance_1_5": "",
        })
        fe = [float(r["feasibility_1_5"]) for r in sub if r.get("feasibility_1_5")]
        ba = [float(r["balance_1_5"]) for r in sub if r.get("balance_1_5")]
        if fe: summary[-1]["median_feasibility_1_5"] = float(np.median(fe))
        if ba: summary[-1]["median_balance_1_5"] = float(np.median(ba))
    with open(a.out + "_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys())); w.writeheader(); w.writerows(summary)

    tests = []
    if "pilot_full" in conds and "verbatim" in conds:
        tests.append(compare(rows, "pilot_full", "verbatim", "fig", "H2 (primary)"))
    if "pilot_full" in conds and "constrained" in conds:
        tests.append(compare(rows, "pilot_full", "constrained", "fig", "H2 (updated system)"))
        tests.append(compare(rows, "pilot_full", "constrained", "stance", "H3 (primary)"))
    if "verbatim" in conds and "constrained" in conds:
        tests.append(compare(rows, "verbatim", "constrained", "stance", "H3 (prompt effect only)"))
    if tests:
        with open(a.out + "_tests.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=sorted({k for t in tests for k in t}, key=lambda k: list(tests[0]).index(k) if k in tests[0] else 99))
            w.writeheader(); w.writerows(tests)
    print(f"Bonferroni alpha = {bonferroni_alpha(0.05, 3):.4f}\n")
    for s in summary: print(s)
    print()
    for t in tests: print(t)


if __name__ == "__main__":
    main()
