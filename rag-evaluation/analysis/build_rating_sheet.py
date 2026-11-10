"""Build the blinded rating sheets (run by the coordinator).

  python3 build_rating_sheet.py --fixed results_42/qa_fixed.csv --table results_42/qa_table.csv \
        --calib results_39docs/results.csv --out rating
Main set: the baseline answer (fixed chunks, k=3) and the updated answer (table-preserving chunks, k=4) for every
query, shuffled, with the system label hidden -> rating_sheet.csv (give to all four raters; each saves their own copy)
and rating_key.csv (kept private). Calibration set: 5 other answers, with the coordinator's own 'reference' ratings
in calibration_key.csv, to align on the rubric before independent rating.
"""
import argparse
import csv
import random


def rows(path, cond):
    return [r for r in csv.DictReader(open(path, newline="", encoding="utf-8")) if r["condition"] == cond and r["repeat"] == "1"]


def sheet_row(item_id, r):
    ctx = r["retrieved_chunks"].replace(" ||| ", "\n---\n")
    return {"item_id": item_id, "question": r["query"],
            "answerable_from_documents": "yes" if r["in_scope"] == "1" else "no",
            "reference_answer": r["ground_truth"], "retrieved_passages": ctx[:2500],
            "model_answer": r["answer"], "correct": "", "grounded": ""}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixed", required=True)
    ap.add_argument("--table", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--out", default="rating")
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    items = [("baseline", r) for r in rows(a.fixed, "k=3|basic")] + [("updated", r) for r in rows(a.table, "k=4|basic")]
    random.Random(a.seed).shuffle(items)
    sheet, key = [], []
    for n, (system, r) in enumerate(items, 1):
        iid = f"R{n:03d}"
        sheet.append(sheet_row(iid, r)); key.append({"item_id": iid, "id": r["id"], "system": system})
    with open(a.out + "_sheet.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(sheet[0].keys())); w.writeheader(); w.writerows(sheet)
    with open(a.out + "_key.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["item_id", "id", "system"]); w.writeheader(); w.writerows(key)

    pool = rows(a.calib, "k=3|structured")
    random.Random(a.seed + 1).shuffle(pool)
    calib = pool[:5]
    with open(a.out + "_calibration_sheet.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(sheet[0].keys())); w.writeheader()
        for n, r in enumerate(calib, 1): w.writerow(sheet_row(f"C{n:02d}", r))
    with open(a.out + "_calibration_key.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["item_id", "id", "note"])
        for n, r in enumerate(calib, 1): w.writerow([f"C{n:02d}", r["id"], "coordinator fills reference ratings, discusses with team"])
    print(f"{len(sheet)} main items, {len(calib)} calibration items")


if __name__ == "__main__":
    main()
