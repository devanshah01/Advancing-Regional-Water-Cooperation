"""Policy-brief mode: connects an 'optimal' water-saving pathway to stakeholder interests.

Research question: can data-driven policy bridge the gap between an optimal water
management solution and the interests of Central Asian stakeholders?

How this module serves that question:
  1. "Optimal" is made explicit and computable: the pathway table (calculator.py) gives, for a
     target of X BCM/yr, the hectares/percent needed and the cost of each single measure.
  2. Stakeholder interests are retrieved PER COUNTRY (one retrieval each), so a single top-k
     lookup cannot crowd out four of the five governments.
  3. The five governments are the ones named in the evaluation plan:
     upstream Kyrgyzstan and Tajikistan, downstream Kazakhstan, Uzbekistan and Turkmenistan.

Conditions (one brief per target x condition x repeat):
  pilot_full   the PILOT system re-run: the model is given the verified calculations and told to copy them;
               stance evidence comes from government.txt + info.txt.            (control for H2 and H3)
  verbatim     H2: code inserts the computed table VERBATIM at the top of the brief; the model is told
               NOT to write hectares, costs or percentages itself. Otherwise identical to pilot_full.
  constrained  H2 + H3: verbatim table, and a constrained stance prompt: each country's stance must be
               justified by a quotation from that country's own file, else "Not covered".
  no_calc      per-country retrieval, model does the arithmetic itself (appendix ablation)
  baseline     no retrieval, no calculator (plain Llama 3.2) (appendix ablation)
"""
from __future__ import annotations

import csv
import os
import re
import time

import calculator as calc
import rag_core as rc

STAKEHOLDERS = ["Kyrgyzstan", "Tajikistan", "Kazakhstan", "Uzbekistan", "Turkmenistan"]
UPSTREAM = ["Kyrgyzstan", "Tajikistan"]
DOWNSTREAM = ["Kazakhstan", "Uzbekistan", "Turkmenistan"]
COUNTRY_FILES = {"Kyrgyzstan": "Kyrgystan.txt", "Tajikistan": "Tajikistan.txt", "Kazakhstan": "Kazakhstan.txt",
                 "Uzbekistan": "Uzbekistan.txt", "Turkmenistan": "Turkmenistan.txt"}
STANCE_WORDS = ["Supports", "Conditional", "Opposes", "Unclear", "Not covered"]
PILOT_SOURCES = ["government.txt", "info.txt"]     # the pilot's only stance evidence
SCALING_SOURCES = ["scaling.txt"]
CONDITIONS = ("pilot_full", "verbatim", "constrained", "no_calc", "baseline")

PATHWAY_WORDS = {
    "drip": r"drip",
    "rice": r"rice|wheat|crop[- ]switch",
    "lining": r"lining|lined|geomembrane|concrete",
}
PATHWAY_LABEL = {"drip": "cotton drip irrigation", "rice": "rice-to-wheat switching", "lining": "canal lining"}


# ---------------------------------------------------------------------- prompts
def _context_block(chunks_by_name):
    parts = []
    for name, chunks in chunks_by_name.items():
        text = "\n".join(c["text"] for c in chunks) if chunks else "(nothing retrieved)"
        parts.append(f"[{name}]\n{text}")
    return "\n\n".join(parts)


def _format(constrained=False):
    names = STAKEHOLDERS
    if constrained:
        line = '<Supports / Conditional / Opposes / Not covered> - "<short quotation from that country\'s passages>" - <what would make it acceptable>'
    else:
        line = "<Supports / Conditional / Opposes / Unclear / Not covered> - <one-sentence reason from the context> - <what would make it acceptable>"
    stakes = "\n".join(f"- {n}: {line}" if i == 0 else f"- {n}: (same format)" for i, n in enumerate(names))
    return f"""Write the brief in exactly this format:

Recommended pathway: <2-4 sentences naming which measure to prioritize and why>

Stakeholder positions:
{stakes}

Main risks and open questions: <2-3 sentences>
"""


def build_policy_prompt(target_bcm, condition, calc_text, context_text, extra=""):
    head = ("You are drafting an exploratory policy brief on reducing water losses in the Amu Darya Basin. "
            "This is decision support, not an official directive.\n\n")
    if extra:
        head += f"ADDITIONAL USER INSTRUCTION: {extra}\n\n"
    if condition == "baseline":
        return head + f"TARGET: save {target_bcm:g} BCM of water per year in the Amu Darya Basin.\n\nUse your general knowledge.\n\n" + _format()
    rules = ("Rules:\n- Use ONLY the material below. Do not invent facts, figures or quotes.\n"
             "- If the material does not say what a stakeholder's position is, write \"Not covered\" for that stakeholder.\n")
    if condition == "pilot_full":
        return (head + rules + "- Do not do your own arithmetic: copy numbers from VERIFIED CALCULATIONS.\n\n"
                f"TARGET: save {target_bcm:g} BCM of water per year.\n\n"
                f"VERIFIED CALCULATIONS (computed by code from the scaling rules):\n{calc_text}\n\n"
                f"CONTEXT FROM THE DOCUMENTS:\n{context_text}\n\n" + _format())
    if condition == "verbatim":
        return (head + rules + "- A table of computed figures will be attached to your brief by code. Do NOT write any hectare, "
                "cost or percentage figures yourself; refer to \"the table\" instead.\n\n"
                f"TARGET: save {target_bcm:g} BCM of water per year.\n\n"
                f"COMPUTED FIGURES (for your reference; they will be attached automatically):\n{calc_text}\n\n"
                f"CONTEXT FROM THE DOCUMENTS:\n{context_text}\n\n" + _format())
    if condition == "constrained":
        strict = ("- Each stakeholder's passages are labelled with that country's name. For each country use ONLY its own passages.\n"
                  "- A stance word must be supported by a short quotation, in quotation marks, from that country's passages that shows its "
                  "position on the pathway you recommend. If its passages say nothing about that pathway, write \"Not covered\" and give "
                  "no quotation.\n- Never give a country a stance because other countries hold it.\n")
        return (head + rules + strict + "- A table of computed figures will be attached to your brief by code. Do NOT write any hectare, "
                "cost or percentage figures yourself; refer to \"the table\" instead.\n\n"
                f"TARGET: save {target_bcm:g} BCM of water per year.\n\n"
                f"COMPUTED FIGURES (for your reference; they will be attached automatically):\n{calc_text}\n\n"
                f"CONTEXT FROM THE DOCUMENTS:\n{context_text}\n\n" + _format(constrained=True))
    # no_calc: the model must compute from the scaling rules itself
    return (head + rules + "- Use the scaling rules in the context to calculate hectares and costs yourself, showing the numbers.\n\n"
            f"TARGET: save {target_bcm:g} BCM of water per year.\n\n"
            f"CONTEXT FROM THE DOCUMENTS:\n{context_text}\n\n" + _format())


def policy_brief(target_bcm, collection, condition="pilot_full", k_each=3, embed_fn=None, generate_fn=None,
                 temperature=None, seed=None, network_km=None, extra=""):
    """Return dict(text, model_text, rows, chunks, calc_text, prompt, seconds)."""
    assert condition in CONDITIONS, condition
    embed_fn = embed_fn or rc.default_embed_fn()
    generate_fn = generate_fn or rc.default_generate_fn()
    t0 = time.time()
    rows = calc.pathway_table(target_bcm, network_km)
    calc_text = calc.format_table(rows)

    chunks = {}
    if condition != "baseline" and collection is not None:
        for name in STAKEHOLDERS:
            q = (f"{name} official stance position on drip irrigation, rice-to-wheat crop switching and canal lining; "
                 f"vulnerabilities, red lines, policy priorities, water")
            sources = [COUNTRY_FILES[name]] if condition == "constrained" else PILOT_SOURCES
            chunks[name] = rc.retrieve(collection, embed_fn(q), k_each, sources=sources)
        if condition == "no_calc":
            q = "scaling rules hectares to convert to save BCM cost per hectare drip lining rice wheat"
            chunks["Scaling rules"] = rc.retrieve(collection, embed_fn(q), 4, sources=SCALING_SOURCES)
    prompt = build_policy_prompt(target_bcm, condition, calc_text, _context_block(chunks), extra)
    model_text = generate_fn(prompt, temperature=temperature, seed=seed)
    if condition in ("verbatim", "constrained"):
        text = ("Computed figures (inserted by code from the scaling rules; not written by the model):\n"
                f"{calc_text}\n\n{model_text}")
    else:
        text = model_text
    return {"text": text, "model_text": model_text, "rows": [r.as_dict() for r in rows], "chunks": chunks,
            "calc_text": calc_text, "prompt": prompt, "seconds": round(time.time() - t0, 2)}


# ----------------------------------------------------------------- stance parsing
def _norm(text):
    return re.sub(r"(?<=\d),(?=\d{3}\b)", "", text.lower())


def parse_stances(text):
    """{country: 'Supports'|'Conditional'|'Opposes'|'Unclear'|'Not covered'|None}"""
    out = {}
    block = text.split("Stakeholder positions", 1)[-1]
    for name in STAKEHOLDERS:
        found = None
        for m in re.finditer(rf"{name}\W{{0,6}}", block):
            window = block[m.end(): m.end() + 40].lower()
            hits = [(window.find(w.lower()), w) for w in STANCE_WORDS if w.lower() in window]
            if hits:
                found = min(hits)[1]
                break
        out[name] = found
    return out


def stakeholders_covered(text):
    return sum(1 for v in parse_stances(text).values() if v)


def recommended_pathway(text):
    """First pathway (drip | rice | lining) named in the 'Recommended pathway' paragraph, else None."""
    m = re.search(r"Recommended pathway:(.*?)(?:\n\s*\n|Stakeholder positions)", text, flags=re.S | re.I)
    seg = (m.group(1) if m else "").lower()
    best = None
    for key, pat in PATHWAY_WORDS.items():
        mm = re.search(pat, seg)
        if mm and (best is None or mm.start() < best[0]):
            best = (mm.start(), key)
    return best[1] if best else None


def load_gold(path=None):
    """{(country, pathway): gold_stance} from gold_stances.csv (documented positions, verified by the team)."""
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "gold_stances.csv")
    gold = {}
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                gold[(r["country"].strip(), r["pathway"].strip())] = r["gold_stance"].strip()
    return gold


def _canon(stance):
    if stance is None:
        return None
    return "Not covered" if stance in ("Unclear", "Not covered") else stance


def stance_matches(text, gold):
    """Per country: 1 = stance equals the documented position, 0 = not, '' = not scorable
    (no recommended pathway found, no stance parsed, or no gold entry). Unclear == Not covered."""
    pw = recommended_pathway(text)
    stances = parse_stances(text)
    res = {}
    for name in STAKEHOLDERS:
        g = gold.get((name, pw)) if pw else None
        s = _canon(stances.get(name))
        res[name] = "" if (g is None or s is None) else int(s == _canon(g))
    return pw, stances, res


# ------------------------------------------------------------- figure scoring (H2)
_FIG_CONTEXT = {"drip_ha": r"drip", "drip_cost": r"drip|cost|\$|usd", "rice_ha": r"rice|wheat", "lining_pct": r"lining|lined|network|canal"}


def _forms(key, target_bcm):
    if key == "drip_ha":
        x = calc.drip_hectares(target_bcm)
    elif key == "drip_cost":
        x = calc.drip_cost_usd(target_bcm)
    elif key == "rice_ha":
        x = calc.rice_to_wheat_hectares(target_bcm)
    else:
        p = calc.lining_percent(target_bcm)
        return {f"{p:.1f}%", f"{p:.0f}%", f"{p:.2f}%", f"{p:.1f} percent", f"{p:.0f} percent", f"{p:.2f} percent",
                f"{int(p)}%", f"{int(p)} percent"}
    forms = {f"{x:.0f}", f"{int(x)}", f"{round(x, -2):.0f}", f"{round(x, -3):.0f}"}
    if x >= 1e9:
        forms |= {f"{x / 1e9:.2f} billion", f"{x / 1e9:.1f} billion"}
    if x >= 1e6:
        forms |= {f"{x / 1e6:.0f} million", f"{x / 1e6:.2f} million", f"{x / 1e6:.1f} million"}
    if 1e5 <= x < 1e6:
        forms |= {f"{x / 1e6:.2f} million", f"{x / 1e3:.0f} thousand"}
    return forms


def figure_scores(text, target_bcm):
    """Automatic H2 scoring. A required figure is CORRECT if it appears in a common number format within
    80 characters of a word for its measure, and (lining) is not misused as 'water saved'
    (the pilot wrote 'can save up to 16% of water', where 16% is the share of the network to line).
    This is a proxy; the rating sheet lets the team confirm."""
    t = re.sub(r"(?<=\d)\s+%", "%", _norm(text))   # "16.7 %" -> "16.7%"
    out = {}
    for key in ("drip_ha", "drip_cost", "rice_ha", "lining_pct"):
        ok = 0
        for f in _forms(key, target_bcm):
            for m in re.finditer(re.escape(f), t):
                win = t[max(0, m.start() - 80): m.end() + 80]
                if re.search(_FIG_CONTEXT[key], win):
                    ok = 1
                    if key == "lining_pct" and re.search(rf"save[sd]?[^.]{{0,25}}{re.escape(f)}[^.]{{0,12}}(of )?(the )?water", win):
                        ok = 0
                if ok:
                    break
            if ok:
                break
        out[key] = ok
    return out


def calc_numbers_correct(text, target_bcm):
    """Share of the four required figures (drip hectares, drip cost, rice hectares, lining %) that are correct."""
    s = figure_scores(text, target_bcm)
    return round(sum(s.values()) / len(s), 3)


def numbers_not_in_support(text, support_text):
    """Numbers (>10) in the generated text that are absent from the retrieved/verified material."""
    def nums(s):
        out = set()
        for m in re.findall(r"\d[\d,]*\.?\d*", _norm(s)):
            m = m.rstrip(".,")
            try:
                v = float(m.replace(",", ""))
            except ValueError:
                continue
            if v > 10:
                out.add(m.replace(",", ""))
        return out
    return sorted(nums(text) - nums(support_text))
