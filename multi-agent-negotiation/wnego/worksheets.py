"""Country worksheets: the team's template for persona and goals.

  Part A (who the country is)       -> persona text   (identity, no demands)
  Part B (what its government wants) -> goal file     (instructions, no feelings)
  Part C (check before submitting)   -> check()       (run automatically)

One YAML file per country in config/worksheets/. Every answer carries a source
naming a document in the RAG library (file name plus p.N or §N), or the
literal text NO SOURCE YET. The negotiation never sees the sources; the
grounding evaluation checks them against the library.
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml

NO_SOURCE = "NO SOURCE YET"
PART_A = ("position", "dependents", "history", "values")
PART_A_LABELS = {"position": "Position on the river", "dependents": "Who depends on the river",
                 "history": "History that shapes its view of water", "values": "What it values in regional water management"}
PRIORITY_KEYS = ("summer_water", "winter_water", "winter_storage", "aral_sea", "compensation")
LEVELS = ("low", "medium", "high", "critical")

# Part C: describe, don't take sides. Word stems, matched at word starts.
LOADED = ("tragic", "rightful", "unfair", "survival", "dying", "catastroph", "existential", "aggressive",
          "desperate", "devastat", "theft", "steal", "stole", "victim", "betray", "outrageous", "unjust",
          "robbed", "starv", "doomed", "selfish", "greed", "hostage", "plunder", "suffer")
DEMANDS = ("must", "demand", "insist", "require", "should", "need to", "will not accept", "never accept")
FEELINGS = ("feel", "fear", "hope", "proud", "pride", "anger", "angry", "frustrat", "resent", "memory of",
            "carry the", "hurt", "grief", "betray", "humiliat")
VAGUE = ("unfair", "fair deal", "reasonable", "acceptable", "adequate", "appropriate", "sufficient")
SOURCE_FILE = re.compile(r"([A-Za-z0-9_\- ]+?\.(?:txt|pdf|md))", re.I)


def _has(text: str, stems) -> list[str]:
    low = f" {text.lower()}"
    return [s for s in stems if re.search(r"[^a-z]" + re.escape(s), low)]


def load(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        ws = yaml.safe_load(f)
    ws["_path"] = str(path)
    ws.setdefault("id", Path(path).stem)
    return ws


def load_dir(folder: str | Path) -> dict[str, dict]:
    """{worksheet id: worksheet}. Base worksheets use the country as id."""
    return {(ws := load(p))["id"]: ws for p in sorted(Path(folder).glob("*.yaml"))}


def answers(ws: dict) -> list[dict]:
    """Every sourced answer as {field, part, text, source}, one row per list item."""
    rows = []
    for k in PART_A:
        e = ws["part_a"].get(k) or {}
        rows.append({"field": k, "part": "A", "text": (e.get("answer") or "").strip(), "source": e.get("source") or ""})
    b = ws["part_b"]
    for k in ("river_position", "fallback"):
        e = b.get(k) or {}
        rows.append({"field": k, "part": "B", "text": (e.get("answer") or "").strip(), "source": e.get("source") or ""})
    for k in ("outcomes", "red_lines"):
        e = b.get(k) or {}
        for i, item in enumerate(e.get("items") or []):
            rows.append({"field": f"{k}[{i}]", "part": "B", "text": str(item).strip(), "source": e.get("source") or ""})
    return rows


def source_files(source: str) -> list[str]:
    return [m.strip() for m in SOURCE_FILE.findall(source or "")]


def word_count(ws: dict) -> int:
    return sum(len(r["text"].split()) for r in answers(ws))


def check(ws: dict, peers: list[dict] | None = None, tolerance: float = 0.3) -> list[dict]:
    """Part C of the worksheet as code. Returns issues; an empty list means it passes.
    peers: the other worksheets, for the 'roughly the same length' check."""
    issues = []

    def add(check_name, field, msg):
        issues.append({"check": check_name, "field": field, "message": msg})

    for r in answers(ws):
        if not r["text"]:
            add("answered", r["field"], "no answer")
        if not r["source"].strip():
            add("sourced", r["field"], f"no source (write {NO_SOURCE} if none exists)")
        for w in _has(r["text"], LOADED):
            add("neutral", r["field"], f"loaded word '{w}'")
        if r["part"] == "A":
            for w in _has(r["text"], DEMANDS):
                add("part_a_no_demands", r["field"], f"demand language '{w}' belongs in Part B")
        else:
            for w in _has(r["text"], FEELINGS):
                add("part_b_no_feelings", r["field"], f"feeling language '{w}' belongs in Part A or nowhere")
    b = ws["part_b"]
    pr = (b.get("priorities") or {})
    if not (pr.get("source") or "").strip():
        add("sourced", "priorities", "no source")
    for k in PRIORITY_KEYS:
        if (pr.get("levels") or {}).get(k) not in LEVELS:
            add("priorities", k, f"level must be one of {', '.join(LEVELS)}")
    for i, item in enumerate((b.get("red_lines") or {}).get("items") or []):
        if _has(item, VAGUE) and not re.search(r"\d", item):
            add("checkable_red_lines", f"red_lines[{i}]", "too vague to check; name a volume, share or specific clause")
    if not (b.get("red_lines") or {}).get("items"):
        add("answered", "red_lines", "no red lines")
    if not 2 <= len((b.get("outcomes") or {}).get("items") or []) <= 3:
        add("answered", "outcomes", "list 2 or 3 outcomes")
    if peers:
        counts = [word_count(p) for p in peers]
        mean = sum(counts) / len(counts)
        n = word_count(ws)
        if mean and abs(n - mean) / mean > tolerance:
            add("similar_length", "all", f"{n} words vs {mean:.0f} average; keep within {tolerance:.0%}")
    return issues


def persona_text(ws: dict) -> str:
    """Part A as the delegate's identity."""
    a = ws["part_a"]
    lines = [f"You lead {ws['country']}'s delegation in a negotiation over Amu Darya water. Background on your country:"]
    for k in PART_A:
        text = ((a.get(k) or {}).get("answer") or "").strip()
        if text:
            lines.append(f"- {PART_A_LABELS[k]}: {text}")
    return "\n".join(lines)


def goal_file(ws: dict) -> dict:
    """Part B in the GoalBook format (bloc, objectives, priorities, red_lines, fallback)."""
    b = ws["part_b"]
    levels = dict((b.get("priorities") or {}).get("levels") or {})
    other = levels.pop("other", None)
    if isinstance(other, dict) and other.get("name"):
        levels[other["name"]] = other.get("level", "medium")
    return {"bloc": ((b.get("river_position") or {}).get("answer") or "").strip(),
            "objectives": list((b.get("outcomes") or {}).get("items") or []),
            "priorities": levels,
            "red_lines": list((b.get("red_lines") or {}).get("items") or []),
            "fallback": ((b.get("fallback") or {}).get("answer") or "").strip()}


def report(folder: str | Path) -> dict[str, list[dict]]:
    sheets = list(load_dir(folder).values())
    return {ws["id"]: check(ws, [p for p in sheets if p is not ws]) for ws in sheets}
