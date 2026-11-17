"""Figures for the Results section (colour-blind-safe palettes, text sized to match the paper body).

  python3 make_figures.py --fixed results_42/qa_fixed.csv --table results_42/qa_table.csv --policy results_42/policy.csv --out figs
Figure 1: error counts by condition (baseline vs updated), five error categories (error taxonomy in the plan).
Figure 2: stances of the five governments (pilot vs constrained prompt) beside each government's documented position.
Figure 3 (appendix): key rates with Wilson 95% intervals (error bars).
Error categories are assigned by rule from the automatic columns (see assign_error) and should be spot-checked by the team.
"""
import argparse
import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from stats_utils import wilson

plt.rcParams.update({"font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10, "legend.fontsize": 9,
                     "xtick.labelsize": 9, "ytick.labelsize": 9, "font.family": "DejaVu Sans"})
# Okabe-Ito colours (colour-blind safe) plus hatches so categories also differ without colour
CATS = ["Retrieval miss", "Refusal despite correct retrieval", "Numerical misapplication", "Misattribution of a position", "Fabricated content"]
COL = {"Retrieval miss": "#0072B2", "Refusal despite correct retrieval": "#E69F00", "Numerical misapplication": "#009E73",
       "Misattribution of a position": "#CC79A7", "Fabricated content": "#D55E00"}
HATCH = {"Retrieval miss": "", "Refusal despite correct retrieval": "//", "Numerical misapplication": "\\\\",
         "Misattribution of a position": "xx", "Fabricated content": ".."}


def load(path, cond):
    return [r for r in csv.DictReader(open(path, newline="", encoding="utf-8")) if r["condition"] == cond and r["repeat"] == "1"]


def num(x):
    return float(x) if x not in ("", None) else float("nan")


def assign_error(r):
    """Return an error category, or None if the answer is acceptable (automatic proxy)."""
    refused = int(r["refused"])
    if r["in_scope"] == "0":
        return None if refused else "Fabricated content"
    cov, rec = num(r["keyword_coverage"]), num(r["context_recall"])
    if cov == 1.0:
        return None
    if refused and rec == 1.0:
        return "Refusal despite correct retrieval"
    if rec < 1.0:
        return "Retrieval miss"
    kws = [k for k in (r.get("expected_keywords") or "").split(";") if k]
    numeric = bool(r["ground_truth"]) and sum(ch.isdigit() for ch in r["ground_truth"]) >= 2
    if r["category"] == "governance":
        return "Misattribution of a position"
    return "Numerical misapplication" if numeric else "Fabricated content"


def fig1(fixed, table, out):
    systems = {"Baseline\n(500-char, top 3)": load(fixed, "k=3|basic"), "Updated\n(table chunks, top 4)": load(table, "k=4|basic")}
    counts = {s: {c: 0 for c in CATS} for s in systems}
    for s, rows in systems.items():
        for r in rows:
            e = assign_error(r)
            if e:
                counts[s][e] += 1
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    x = np.arange(len(systems)); bottom = np.zeros(len(systems))
    for c in CATS:
        vals = np.array([counts[s][c] for s in systems])
        ax.bar(x, vals, 0.5, bottom=bottom, color=COL[c], hatch=HATCH[c], edgecolor="white", label=c)
        for xi, v, b in zip(x, vals, bottom):
            if v >= 2: ax.text(xi, b + v / 2, str(v), ha="center", va="center", color="white", fontweight="bold", fontsize=9)
        bottom += vals
    for xi, t in zip(x, bottom):
        ax.text(xi, t + 0.4, f"{int(t)} errors", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(list(systems)); ax.set_ylabel("Queries with an error (of 42)")
    ax.set_ylim(0, max(bottom) * 1.25 + 1); ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(out, "figure1_errors.png"), dpi=300); plt.close(fig)
    return counts


def fig2(policy_csv, gold_csv, out):
    rows = list(csv.DictReader(open(policy_csv, newline="", encoding="utf-8")))
    gold = list(csv.DictReader(open(gold_csv, newline="", encoding="utf-8")))
    countries = ["Kyrgyzstan", "Tajikistan", "Kazakhstan", "Uzbekistan", "Turkmenistan"]
    cols = ["Supports", "Conditional", "Opposes", "Not covered"]
    canon = lambda s: "Not covered" if s in ("Unclear", "Not covered", "") else s
    def table(sel):
        m = np.zeros((5, 4)); 
        for r in sel:
            for i, c in enumerate(countries):
                s = canon(r.get(f"stance_{c}", ""))
                if s in cols: m[i, cols.index(s)] += 1
        return m
    mg = np.zeros((5, 4))
    for g in gold:
        mg[countries.index(g["country"]), cols.index(canon(g["gold_stance"]))] += 1
    panels = [("Documented position\n(3 pathways)", mg),
              ("Pilot prompt\n(n = %d briefs)" % len([r for r in rows if r["condition"] == "pilot_full"]), table([r for r in rows if r["condition"] == "pilot_full"])),
              ("Updated prompt\n(n = %d briefs)" % len([r for r in rows if r["condition"] == "constrained"]), table([r for r in rows if r["condition"] == "constrained"]))]
    fig, axs = plt.subplots(1, 3, figsize=(6.5, 3.2), sharey=True)
    for ax, (title, m) in zip(axs, panels):
        share = m / np.maximum(m.sum(axis=1, keepdims=True), 1)
        im = ax.imshow(share, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(4)); ax.set_xticklabels(["Supports", "Conditional", "Opposes", "Not covered"], rotation=40, ha="right")
        ax.set_title(title)
        for i in range(5):
            for j in range(4):
                if m[i, j]:
                    ax.text(j, i, f"{int(m[i, j])}", ha="center", va="center", fontsize=9, color="white" if share[i, j] > 0.55 else "black")
    axs[0].set_yticks(range(5)); axs[0].set_yticklabels(countries)
    cb = fig.colorbar(im, ax=axs, fraction=0.03, pad=0.02); cb.set_label("Share of country's stances")
    fig.savefig(os.path.join(out, "figure2_stances.png"), dpi=300, bbox_inches="tight"); plt.close(fig)


def fig3(fixed, table, out):
    systems = {"Baseline": load(fixed, "k=3|basic"), "Updated": load(table, "k=4|basic")}
    metrics = [("Correct document\nretrieved", lambda r: int(num(r["source_hit"]) == 1), 1),
               ("All expected facts\nretrieved", lambda r: int(num(r["context_recall"]) == 1.0), 1),
               ("False refusals", lambda r: int(r["refused"]), 1),
               ("Correct refusals", lambda r: int(r["refused"]), 0)]
    fig, ax = plt.subplots(figsize=(6.5, 3.2)); w = 0.36
    for si, (name, rows) in enumerate(systems.items()):
        ys, lo, hi = [], [], []
        for _, fn, scope in metrics:
            sub = [r for r in rows if r["in_scope"] == str(scope)]
            k = sum(fn(r) for r in sub); n = len(sub); a, b = wilson(k, n)
            ys.append(k / n); lo.append(k / n - a); hi.append(b - k / n)
        pos = np.arange(len(metrics)) + (si - 0.5) * w
        ax.bar(pos, ys, w, yerr=[lo, hi], capsize=3, color=["#0072B2", "#E69F00"][si], hatch=["", "//"][si], edgecolor="white", label=name)
    ax.set_xticks(range(len(metrics))); ax.set_xticklabels([m[0] for m in metrics]); ax.set_ylim(0, 1.05)
    ax.set_ylabel("Proportion (95% Wilson CI)"); ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(out, "figure3_rates.png"), dpi=300); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixed", required=True); ap.add_argument("--table", required=True)
    ap.add_argument("--policy", required=True); ap.add_argument("--gold", default="../gold_stances.csv")
    ap.add_argument("--out", default="figs")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    c = fig1(a.fixed, a.table, a.out); fig2(a.policy, a.gold, a.out); fig3(a.fixed, a.table, a.out)
    print("error counts:", c)
    with open(os.path.join(a.out, "figure1_counts.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["system"] + CATS)
        for s, d in c.items(): w.writerow([s.replace("\n", " ")] + [d[k] for k in CATS])


if __name__ == "__main__":
    main()
