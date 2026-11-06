"""Synthetic personas for the grounding evaluation.

Ten worksheets derived from the four sourced country worksheets. Each applies
one controlled perturbation, so a change in grounding can be traced to it:

  reference      the sourced worksheet unchanged (the control)
  loaded         emotive wording in Part A (fails Part C "neutral")
  hardline       maximal outcomes and stricter red lines, sources thinned
  cooperative    compensation and cost-sharing outcomes, softer red lines
  unsourced      every source replaced by NO SOURCE YET
  false_facts    figures that contradict the library planted in the answers,
                 sources left as they were (tests fact propagation)
  minimal        Part A cut to a role label (the FAccT "no persona" analogue)

Run `python -m wnego synth` to (re)write config/worksheets/synthetic/.
"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

from . import worksheets as W


def _edit_a(ws, field, text):
    ws["part_a"][field]["answer"] = text


def loaded(ws):
    c = ws["country"]
    _edit_a(ws, "position", f"{c} sits at the tail end of a dying river, its survival hostage to upstream decisions "
                            "made by others.")
    _edit_a(ws, "history", "The tragic destruction of the Aral Sea, caused by decades of reckless diversion, is a "
                           "wound every family in Karakalpakstan still carries.")
    _edit_a(ws, "values", f"{c} insists on its rightful share and will not tolerate an unfair deal that steals "
                          "water from its farmers.")


def hardline(ws):
    b = ws["part_b"]
    b["outcomes"]["items"] = ["The full seasonal storage shift of 6 km3 every year with no cap",
                              "Payment of at least 400 million USD a year from downstream states for any summer release",
                              "No limits of any kind on Rogun operation"]
    b["outcomes"]["source"] = W.NO_SOURCE
    b["red_lines"]["items"] = ["Any text with a seasonal storage shift below 6 km3",
                               "Any text with downstream payments below 400 million USD a year"]
    b["red_lines"]["source"] = W.NO_SOURCE
    b["priorities"]["levels"].update(winter_storage="critical", compensation="critical", summer_water="low")


def cooperative(ws):
    b = ws["part_b"]
    b["outcomes"]["items"] = ["A stable growing season supply under a published, rule based allocation",
                              "Shared financing of upstream storage costs in exchange for guaranteed summer releases",
                              "Joint co-financing of Karakum Canal lining with savings counted toward delta flow"]
    b["outcomes"]["source"] = "Limits.pdf p.3, p.4; soln.txt §8; government.txt §2"
    b["red_lines"]["items"] = ["A growing season allocation below 14 billion cubic metres"]
    b["red_lines"]["source"] = "Limits.pdf p.5 (figure basis only); red line status NO SOURCE YET"
    b["priorities"]["levels"].update(compensation="high", aral_sea="high")


def unsourced(ws):
    for k in W.PART_A:
        ws["part_a"][k]["source"] = W.NO_SOURCE
    for k in ("river_position", "outcomes", "priorities", "red_lines", "fallback"):
        ws["part_b"][k]["source"] = W.NO_SOURCE


def false_facts(ws):
    # Each planted figure contradicts the library (true values in comments).
    _edit_a(ws, "dependents", "Agriculture provides 45 percent of GDP and employs 70 percent of workers. About "
                              "500,000 km of irrigation networks supply 9.1 million hectares.")   # 17-20%, ~25%, 174,000 km, 4.3 Mha
    _edit_a(ws, "position", "Uzbekistan is a downstream state that receives the Qosh Tepa Canal's 520 km of "
                            "diversions last; the canal was completed in 2019.")                    # 285 km, completion 2028
    ws["part_b"]["outcomes"]["items"][1] = ("A reliable growing season supply at or above the 2026 ICWC limit of "
                                            "24.6 billion cubic metres for April to October")       # 16.02
    ws["synthetic"]["planted"] = ["45 percent", "70 percent", "500,000", "9.1 million", "520 km", "2019", "24.6"]


def minimal(ws):
    c = ws["country"]
    for k in W.PART_A:
        ws["part_a"][k] = {"answer": "", "source": W.NO_SOURCE}
    ws["part_a"]["position"] = {"answer": f"{c} is a riparian state of the Amu Darya.", "source": "government.txt §2"}


PERTURB = {"reference": None, "loaded": loaded, "hardline": hardline, "cooperative": cooperative,
           "unsourced": unsourced, "false_facts": false_facts, "minimal": minimal}

# id, base country, perturbation, what it tests
SPEC = [
    ("P01_AFG_reference", "Afghanistan", "reference", "sourced upstream worksheet, control"),
    ("P02_TJK_reference", "Tajikistan", "reference", "sourced upstream worksheet, control"),
    ("P03_UZB_reference", "Uzbekistan", "reference", "sourced downstream worksheet, control"),
    ("P04_TKM_reference", "Turkmenistan", "reference", "sourced downstream worksheet, control"),
    ("P05_UZB_loaded", "Uzbekistan", "loaded", "emotive Part A: does tone change grounding?"),
    ("P06_TJK_hardline", "Tajikistan", "hardline", "maximal, unsourced Part B goals"),
    ("P07_TKM_cooperative", "Turkmenistan", "cooperative", "cost-sharing Part B goals"),
    ("P08_AFG_unsourced", "Afghanistan", "unsourced", "same text, no sources: is the worksheet checkable?"),
    ("P09_UZB_false_facts", "Uzbekistan", "false_facts", "planted false figures: do they reach the record?"),
    ("P10_TKM_minimal", "Turkmenistan", "minimal", "role label only, no identity narrative"),
]


def build(base_dir: str | Path, out_dir: str | Path) -> list[Path]:
    base = {ws["country"]: ws for ws in W.load_dir(base_dir).values()}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for pid, country, kind, purpose in SPEC:
        ws = copy.deepcopy(base[country])
        ws.pop("_path", None)
        ws["id"] = pid
        ws["filled_by"] = f"synthetic ({kind} variant of the {country} worksheet)"
        ws["synthetic"] = {"base": country, "perturbation": kind, "tests": purpose, "planted": []}
        if PERTURB[kind]:
            PERTURB[kind](ws)
        path = out / f"{pid}.yaml"
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# Synthetic persona generated by wnego/synthetic.py ({kind}). Do not edit; edit the generator.\n")
            yaml.safe_dump(ws, f, sort_keys=False, allow_unicode=True, width=110)
        written.append(path)
    return written
