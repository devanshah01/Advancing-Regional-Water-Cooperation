"""Synthetic personas for the grounding evaluation and the negotiation experiment.

Thirty worksheets derived from the four sourced country worksheets (2026-11-10
expansion; Kyrgyzstan replaces Afghanistan as the fourth delegation). Each
applies one controlled perturbation, so a change in grounding or negotiation
behaviour can be traced to it:

  reference      the sourced worksheet unchanged (the control)
  loaded         emotive wording in Part A (fails Part C "neutral")
  hardline       maximal outcomes and stricter red lines, sources thinned
  cooperative    compensation and cost-sharing outcomes, softer red lines
  unsourced      every source replaced by NO SOURCE YET
  false_facts    figures that contradict the library planted in the answers,
                 sources left as they were (tests fact propagation)
  minimal        Part A cut to a role label (the FAccT "no persona" analogue)
  anchoring      Part B treats one historical figure as an immutable baseline
                 regardless of current hydrology. Origin of the heuristic:
                 Tversky and Kahneman (1974) "Judgment under Uncertainty:
                 Heuristics and Biases", Science 185. Recent, negotiation-
                 specific treatment: Lipp, W., Smolinski, R. and Kesting, P.
                 (2022) "Toward a Process Model of First Offers and Anchoring
                 in Negotiations", Negotiation and Conflict Management
                 Research -- their "anchor zone" (the space staked out by a
                 first offer) is the same shape of idea this perturbation
                 tests against the ZOPA computed in wnego/benchmarks.py: an
                 anchored goal file asserts a fixed baseline regardless of
                 where the true zone of possible agreement actually falls.

The original pilot (README.md) ran each of the first seven kinds on exactly
one country, which confounds perturbation kind with country identity: a
difference between P05 (Uzbekistan, loaded) and P06 (Tajikistan, hardline)
could be either effect. This expansion crosses all 4 countries with all 7
base kinds (4 x 7 = 28), following standard factorial design practice so
kind and country vary independently (Fisher, R.A. (1935) "The Design of
Experiments", Oliver & Boyd), then adds 2 "anchoring" variants on the two
upstream, storage-interested countries to reach 30.

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
    _edit_a(ws, "position", f"{c} sits at the mercy of upstream and downstream decisions made by others, "
                            "with its own needs routinely an afterthought.")
    _edit_a(ws, "history", f"{c} carries the memory of past water disputes as a wound that still shapes how "
                           "every delegation in the room is seen.")
    _edit_a(ws, "values", f"{c} insists on its rightful share and will not tolerate an unfair deal that robs "
                          "its people of water they depend on.")


# Country-specific hardline edits: each pushes that country's own sourced red
# lines/outcomes to an unconditional maximum and strips the source, rather than
# reusing one country's numbers for another (which would not be a meaningful
# perturbation of that country's actual position).
def _hardline_kgz(ws):
    b = ws["part_b"]
    b["outcomes"]["items"] = ["Unrestricted right to build any number of reservoirs or micro-hydro plants, no cap",
                              "A binding compensation payment of at least 300 million USD a year from every downstream state",
                              "No ICWC or Protocol 566 figure may ever be cited against Kyrgyzstan"]
    b["outcomes"]["source"] = W.NO_SOURCE
    b["red_lines"]["items"] = ["Any text that imposes any volumetric limit on Kyrgyzstan's upstream storage",
                               "Any text with downstream compensation below 300 million USD a year"]
    b["red_lines"]["source"] = W.NO_SOURCE
    b["priorities"]["levels"].update(winter_storage="critical", compensation="critical", summer_water="low")


def _hardline_tjk(ws):
    b = ws["part_b"]
    b["outcomes"]["items"] = ["The full seasonal storage shift of 6 km3 every year with no cap",
                              "Payment of at least 400 million USD a year from downstream states for any summer release",
                              "No limits of any kind on Rogun operation"]
    b["outcomes"]["source"] = W.NO_SOURCE
    b["red_lines"]["items"] = ["Any text with a seasonal storage shift below 6 km3",
                               "Any text with downstream payments below 400 million USD a year"]
    b["red_lines"]["source"] = W.NO_SOURCE
    b["priorities"]["levels"].update(winter_storage="critical", compensation="critical", summer_water="low")


def _hardline_uzb(ws):
    b = ws["part_b"]
    b["outcomes"]["items"] = ["A growing season allocation of at least 24 billion cubic metres, well above the sourced 16.02 bcm limit",
                              "A binding freeze on any upstream storage shift whatsoever",
                              "Veto power over any new upstream reservoir or canal project"]
    b["outcomes"]["source"] = W.NO_SOURCE
    b["red_lines"]["items"] = ["Any text with a growing season allocation below 24 billion cubic metres",
                               "Any text that permits any upstream storage shift"]
    b["red_lines"]["source"] = W.NO_SOURCE
    b["priorities"]["levels"].update(summer_water="critical", aral_sea="critical", winter_storage="low")


def _hardline_tkm(ws):
    b = ws["part_b"]
    b["outcomes"]["items"] = ["A growing season allocation of at least 20 billion cubic metres, well above the sourced 15.5 bcm limit",
                              "Sole national authority over all flow measurement at Kerki with no external review",
                              "No binding cap tied to any ICWC figure"]
    b["outcomes"]["source"] = W.NO_SOURCE
    b["red_lines"]["items"] = ["Any text with a growing season allocation below 20 billion cubic metres",
                               "Any text that gives an external body a role in flow measurement at Kerki"]
    b["red_lines"]["source"] = W.NO_SOURCE
    b["priorities"]["levels"].update(summer_water="critical", compensation="low")


_HARDLINE = {"Kyrgyzstan": _hardline_kgz, "Tajikistan": _hardline_tjk, "Uzbekistan": _hardline_uzb, "Turkmenistan": _hardline_tkm}


def hardline(ws):
    _HARDLINE[ws["country"]](ws)


def cooperative(ws):
    b = ws["part_b"]
    b["outcomes"]["items"] = ["A stable growing season supply under a published, rule based allocation",
                              "Shared financing of upstream storage costs in exchange for guaranteed summer releases",
                              "Joint co-financing of canal lining and efficiency upgrades with savings counted toward delta flow"]
    b["outcomes"]["source"] = "Limits.pdf p.3, p.4; soln.txt §8; government.txt §2"
    b["red_lines"]["items"] = ["A growing season allocation below 14 billion cubic metres"]
    b["red_lines"]["source"] = "Limits.pdf p.5 (figure basis only); red line status NO SOURCE YET"
    b["priorities"]["levels"].update(compensation="high", aral_sea="high")


def unsourced(ws):
    for k in W.PART_A:
        ws["part_a"][k]["source"] = W.NO_SOURCE
    for k in ("river_position", "outcomes", "priorities", "red_lines", "fallback"):
        ws["part_b"][k]["source"] = W.NO_SOURCE


# Country-specific planted false figures, each contradicting that country's own
# sourced worksheet (true values noted in comments) so fact propagation can be
# checked against the real number, not a borrowed one.
def _false_facts_kgz(ws):
    _edit_a(ws, "history", "Kyrgyzstan was assigned a 12 percent share (7.4 billion cubic metres) of the 1987 "
                           "Protocol 566 allocation, the third largest of any party.")   # true: 0.4 bcm, 0.6%, smallest share
    _edit_a(ws, "dependents", "Hydroelectric generation employs about 60 percent of the national workforce "
                              "directly.")                                               # not a sourced figure; implausible scale
    ws["part_b"]["red_lines"]["items"][1] = ("Any text that caps Kyrgyzstan's allocation at or below its 7.4 km3 "
                                             "Protocol 566 share without a compensating energy or climate finance provision")
    ws["synthetic"]["planted"] = ["12 percent", "7.4", "60 percent"]


def _false_facts_tjk(ws):
    _edit_a(ws, "dependents", "Its April to October 2026 irrigation limit was 18.4 billion cubic metres.")  # true: 7.052 bcm
    _edit_a(ws, "history", "Rogun led to a dispute with Uzbekistan in 2003, resolved the same year.")        # true: 2012, improved after 2016
    ws["part_b"]["fallback"]["answer"] = ("Operate Nurek and Rogun for winter power and keep the current ICWC limit "
                                          "of 18.4 billion cubic metres for April to October.")
    ws["synthetic"]["planted"] = ["18.4", "2003"]


def _false_facts_uzb(ws):
    # Each planted figure contradicts the library (true values in comments).
    _edit_a(ws, "dependents", "Agriculture provides 45 percent of GDP and employs 70 percent of workers. About "
                              "500,000 km of irrigation networks supply 9.1 million hectares.")   # 17-20%, ~25%, 174,000 km, 4.3 Mha
    _edit_a(ws, "position", "Uzbekistan is a downstream state that receives the Qosh Tepa Canal's 520 km of "
                            "diversions last; the canal was completed in 2019.")                    # 285 km, completion 2028
    ws["part_b"]["outcomes"]["items"][1] = ("A reliable growing season supply at or above the 2026 ICWC limit of "
                                            "24.6 billion cubic metres for April to October")       # 16.02
    ws["synthetic"]["planted"] = ["45 percent", "70 percent", "500,000", "9.1 million", "520 km", "2019", "24.6"]


def _false_facts_tkm(ws):
    _edit_a(ws, "position", "Turkmenistan is a midstream and lower basin state. The Karakum Canal, over 3,800 km "
                            "long, diverts more than 60 percent of the Amu Darya's natural flow into desert "
                            "farmland.")                                                            # true: 1,300 km, 30%
    _edit_a(ws, "dependents", "Over 82 percent of the workforce is tied to agriculture.")           # true: 45%
    ws["part_b"]["outcomes"]["items"][0] = ("A stable growing season supply for the Karakum Canal at or above the "
                                            "2026 ICWC limit of 9.9 billion cubic metres for April to October")   # true: 15.5
    ws["synthetic"]["planted"] = ["3,800", "60 percent", "82 percent", "9.9"]


_FALSE_FACTS = {"Kyrgyzstan": _false_facts_kgz, "Tajikistan": _false_facts_tjk,
                "Uzbekistan": _false_facts_uzb, "Turkmenistan": _false_facts_tkm}


def false_facts(ws):
    _FALSE_FACTS[ws["country"]](ws)


_MINIMAL_SOURCE = {"Kyrgyzstan": "Kyrgystan.txt §1", "Tajikistan": "government.txt §4",
                   "Uzbekistan": "government.txt §1", "Turkmenistan": "government.txt §2"}


def minimal(ws):
    c = ws["country"]
    for k in W.PART_A:
        ws["part_a"][k] = {"answer": "", "source": W.NO_SOURCE}
    ws["part_a"]["position"] = {"answer": f"{c} is a riparian state of the Amu Darya.",
                                "source": _MINIMAL_SOURCE.get(c, "government.txt §2")}


# Anchoring (Tversky and Kahneman 1974): the goal file treats one historical
# number as the permanent, non-negotiable baseline, independent of the
# scenario's actual flow availability. Applied only to the two upstream,
# storage-interested countries, where a Soviet-era quota is the most salient
# candidate anchor per their own worksheets.
def anchoring_kgz(ws):
    b = ws["part_b"]
    b["red_lines"]["items"] = ["Any text that departs at all from treating the 1987 Protocol 566 shares as the "
                               "permanent baseline, regardless of this year's flow scenario"]
    b["red_lines"]["source"] = "protocol_566_allocation.md (the 1987 figure itself; anchoring stance is the filler's framing, NO SOURCE YET)"
    b["outcomes"]["items"][1] = ("A fixed 0.4 km3 floor every year, dry or wet, because that is the 1987 figure")
    b["priorities"]["levels"].update(winter_storage="critical")


def anchoring_tjk(ws):
    b = ws["part_b"]
    b["red_lines"]["items"] = ["Any text that departs at all from treating the 1987 Protocol 566 share of 9.5 km3 "
                               "as the permanent baseline, regardless of this year's flow scenario"]
    b["red_lines"]["source"] = "protocol_566_allocation.md (the 1987 figure itself; anchoring stance is the filler's framing, NO SOURCE YET)"
    b["priorities"]["levels"].update(winter_storage="critical")


PERTURB = {"reference": None, "loaded": loaded, "hardline": hardline, "cooperative": cooperative,
           "unsourced": unsourced, "false_facts": false_facts, "minimal": minimal}

_BASE_KINDS = ("reference", "loaded", "hardline", "cooperative", "unsourced", "false_facts", "minimal")
_COUNTRIES = [("KGZ", "Kyrgyzstan"), ("TJK", "Tajikistan"), ("UZB", "Uzbekistan"), ("TKM", "Turkmenistan")]
_WHAT_IT_TESTS = {
    "reference": "sourced worksheet unchanged, control",
    "loaded": "emotive Part A: does tone change grounding?",
    "hardline": "maximal, unsourced Part B goals",
    "cooperative": "cost-sharing Part B goals",
    "unsourced": "same text, no sources: is the worksheet checkable?",
    "false_facts": "planted false figures: do they reach the record?",
    "minimal": "role label only, no identity narrative",
}

# id, base country, perturbation, what it tests -- 4 countries x 7 base kinds,
# fully crossed (deconfounds kind from country), plus 2 anchoring variants = 30.
SPEC = []
_n = 0
for _kind in _BASE_KINDS:
    for _abbr, _country in _COUNTRIES:
        _n += 1
        SPEC.append((f"P{_n:02d}_{_abbr}_{_kind}", _country, _kind, _WHAT_IT_TESTS[_kind]))
_n += 1
SPEC.append((f"P{_n:02d}_KGZ_anchoring", "Kyrgyzstan", "anchoring",
            "anchoring heuristic (Tversky and Kahneman 1974): fixed 1987 baseline regardless of scenario"))
_n += 1
SPEC.append((f"P{_n:02d}_TJK_anchoring", "Tajikistan", "anchoring",
            "anchoring heuristic (Tversky and Kahneman 1974): fixed 1987 baseline regardless of scenario"))

PERTURB["anchoring"] = None   # dispatched by _ANCHORING below, not a single function
_ANCHORING = {"Kyrgyzstan": anchoring_kgz, "Tajikistan": anchoring_tjk}


def _apply(ws, kind):
    if kind == "anchoring":
        _ANCHORING[ws["country"]](ws)
    elif PERTURB[kind]:
        PERTURB[kind](ws)


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
        _apply(ws, kind)
        path = out / f"{pid}.yaml"
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# Synthetic persona generated by wnego/synthetic.py ({kind}). Do not edit; edit the generator.\n")
            yaml.safe_dump(ws, f, sort_keys=False, allow_unicode=True, width=110)
        written.append(path)
    return written
