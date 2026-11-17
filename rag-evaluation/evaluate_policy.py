"""Policy-brief evaluation for hypotheses H2 and H3 of the evaluation plan.

Five governments: upstream Kyrgyzstan and Tajikistan, downstream Kazakhstan, Uzbekistan and Turkmenistan.
Conditions per target (BCM/yr) and repeat (see policy.py):
  pilot_full   the pilot system re-run (control)
  verbatim     H2: computed table inserted verbatim by code
  constrained  H2 + H3: verbatim table + constrained, quotation-backed stance prompt
  no_calc / baseline   appendix ablations (add with --conditions)

Default: 7 targets x 3 conditions x 3 repeats at temperature 0.7 = 63 briefs. Repeat r of every condition uses
seed (--seed + r), so brief (target, repeat) is PAIRED across conditions; 7 targets x 3 repeats = 21 paired
briefs, i.e. 21 briefs x 5 governments = 105 stances per condition and 21 x 4 = 84 required figures.

Real run:   python3 evaluate_policy.py --docs ../data --db ./db_policy --out results_42/policy
Dry run:    python3 evaluate_policy.py --docs ../data --dry-run      (fake models; plumbing only)

Automatic columns: recommended_pathway, stance_<country>, match_<country> (1/0 against gold_stances.csv, ''
if not scorable), fig_* (1/0 per required figure), figures_correct (0-4), numbers_not_in_support.
Optional human columns: feasibility_1_5 and balance_1_5 (OSCE experts, if available) and notes.
"""
import argparse
import csv
import statistics

import policy as pol
import rag_core as rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default=rc.DOCS_FOLDER)
    ap.add_argument("--db", default="./db_policy")
    ap.add_argument("--targets", default="2,3,4,5,6,8,10", help="BCM/yr targets, comma separated")
    ap.add_argument("--conditions", default="pilot_full,verbatim,constrained")
    ap.add_argument("--k-each", type=int, default=3, help="chunks retrieved per country")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gold", default=None, help="path to gold_stances.csv (default: next to policy.py)")
    ap.add_argument("--out", default="policy_results")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.dry_run:
        rc.DRY_RUN = True
        print("*** DRY RUN: fake model outputs. Do NOT report these numbers. ***")
    embed_fn, generate_fn = rc.default_embed_fn(), rc.default_generate_fn()
    out_prefix = args.out + ("_DRYRUN" if args.dry_run else "")

    col, status, info = rc.build_index(args.docs, args.db, embed_fn=embed_fn, rebuild=args.dry_run)
    if col is None:
        raise SystemExit(f"No documents found in {args.docs}")
    print(f"Index ({status}): {info['documents']} documents, {info['chunks']} chunks")
    gold = pol.load_gold(args.gold)
    if not gold:
        print("WARNING: gold_stances.csv not found; stance matching will be blank.")

    targets = [float(x) for x in args.targets.split(",") if x.strip()]
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    rows = []
    for t in targets:
        for rep in range(args.repeats):
            for cond in conditions:
                out = pol.policy_brief(t, col, condition=cond, k_each=args.k_each, embed_fn=embed_fn,
                                       generate_fn=generate_fn, temperature=args.temperature,
                                       seed=args.seed + rep)
                support = out["calc_text"] + " " + " ".join(c["text"] for cs in out["chunks"].values() for c in cs)
                pw, stances, match = pol.stance_matches(out["text"], gold)
                figs = pol.figure_scores(out["text"], t)
                scored = [v for v in match.values() if v != ""]
                row = {
                    "brief_id": f"T{t:g}-R{rep + 1}", "target_bcm": t, "repeat": rep + 1, "condition": cond,
                    "recommended_pathway": pw or "",
                }
                for n in pol.STAKEHOLDERS:
                    row[f"stance_{n}"] = stances.get(n) or ""
                for n in pol.STAKEHOLDERS:
                    row[f"match_{n}"] = match[n]
                row.update({
                    "stances_scored": len(scored), "stances_matching": sum(scored),
                    "fig_drip_ha": figs["drip_ha"], "fig_drip_cost": figs["drip_cost"],
                    "fig_rice_ha": figs["rice_ha"], "fig_lining_pct": figs["lining_pct"],
                    "figures_correct": sum(figs.values()),
                    "numbers_not_in_support": ("" if cond == "baseline" else len(pol.numbers_not_in_support(
                        out["model_text"], support + f" {t:g}"))),
                    "seconds": out["seconds"],
                    "brief": out["text"].replace("\r", " ").strip(),
                    "verified_calculations": out["calc_text"].replace("\n", " | "),
                    "feasibility_1_5": "", "balance_1_5": "", "notes": "",
                })
                rows.append(row)
        print(f"  done: target {t:g} BCM")

    with open(out_prefix + ".csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\n=== SUMMARY (automatic columns only) ===")
    print("condition | n_briefs | figures_correct (of 4) | stances_matching/scored | stakeholders_with_a_stance")
    for cond in conditions:
        sub = [r for r in rows if r["condition"] == cond]
        sm = sum(r["stances_matching"] for r in sub)
        ss = sum(r["stances_scored"] for r in sub)
        cov = statistics.mean(sum(1 for n in pol.STAKEHOLDERS if r[f"stance_{n}"]) for r in sub)
        print(f"{cond} | {len(sub)} | {statistics.mean(r['figures_correct'] for r in sub):.2f} | "
              f"{sm}/{ss} | {cov:.2f}")
    print(f"\nWrote {out_prefix}.csv")


if __name__ == "__main__":
    main()
