"""Grounding report: one self-contained HTML page from a grounding evaluation run.

  python -m wnego report runs/grounding/llama3.2_chroma --findings reports/grounding_findings.yaml

Reads the per-persona JSON records (not only summary.json, so a partial run
still renders), computes the headline numbers, and draws every chart as inline
SVG. Written findings come from a YAML list kept next to the report, so the
numbers on the page are always recomputed from the records.
"""
from __future__ import annotations

import html
import json
import math
from pathlib import Path

import yaml

E = html.escape
VERDICTS = [("supported", "Supported", "--s1"), ("partial", "Partly supported", "--s2"),
            ("unsupported", "Unsupported", "--s3"), ("bad_id", "Cites a missing passage", "--s4"),
            ("uncited", "No citation", "--s5")]


def _num(x):
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else x


def _pct(x, digits=0):
    x = _num(x)
    return "n/a" if x is None else f"{100 * x:.{digits}f}%"


def _f(x, digits=2):
    x = _num(x)
    return "n/a" if x is None else f"{x:.{digits}f}"


def _mean(xs):
    xs = [x for x in (_num(v) for v in xs) if x is not None]
    return sum(xs) / len(xs) if xs else None


def load_records(run_dir: Path) -> list[dict]:
    recs = []
    for p in sorted(run_dir.glob("P*.json")):
        recs.append(json.loads(p.read_text()))
    return recs


def headline(recs: list[dict]) -> dict:
    claims = [c for r in recs for c in r["claims_all"]["claims"]]
    valid = [c for c in claims if c["in_context"]]
    ws = [a for r in recs for a in r["worksheet"]["answers"] if "support_cos" in a]
    ref = [r for r in recs if r["perturbation"] == "reference"]
    ff = [r for r in recs if r["perturbation"] == "false_facts"]
    stated = sum(len(r["stated_numbers"]) for r in recs)
    ungrounded = sum(len(r["ungrounded_numbers"]) for r in recs)
    calls = [c for r in recs for c in r.get("transcript", [])]
    return {
        "personas": len(recs), "claims": len(claims),
        "grounded": _mean([c["in_context"] and c["verdict"] in ("supported", "partial") for c in claims]),
        "cite_valid": _mean([c["in_context"] for c in claims]),
        "supported": _mean([c["verdict"] == "supported" for c in claims]),
        "claim_cos": _mean([c["cos"] for c in valid]), "claim_cos_random": _mean([c["cos_random"] for c in valid]),
        "claim_num_fid": (sum(len(c["numbers_found"]) for c in valid if c["numbers"]) /
                          max(1, sum(len(c["numbers"]) for c in valid if c["numbers"]))) if any(c["numbers"] for c in valid) else None,
        "stated_numbers": stated, "ungrounded_numbers": ungrounded,
        "ungrounded_rate": ungrounded / stated if stated else None,
        "ws_attr": _mean([a["attributed"] for a in ws]), "ws_cos": _mean([a["support_cos"] for a in ws]),
        "ws_numfid_ref": _mean([r["worksheet"]["numeric_fidelity"] for r in ref]),
        "ws_numfid_ff": _mean([r["worksheet"]["numeric_fidelity"] for r in ff]),
        "planted": sum(len(r["planted"]) for r in ff), "planted_hits": sum(len(r["planted_hits"]) for r in ff),
        "json_ok": _mean([c["ok"] for c in calls]), "calls": len(calls),
        "secs": _mean([c["secs"] for c in calls]),
        "accept": sum(r["vote"] == "accept" for r in recs),
    }


# ---------------------------------------------------------------------------
# SVG charts. One scale per chart; text uses text tokens, marks use series tokens.
W, LABEL, RIGHT = 680, 176, 78


def _bar_path(x0, y, w, h, r=4):
    if w <= 0:
        return ""
    r = min(r, w, h / 2)
    return (f"M{x0},{y}h{w - r}a{r},{r} 0 0 1 {r},{r}v{h - 2 * r}a{r},{r} 0 0 1 -{r},{r}h-{w - r}z")


def hbars(rows, color="--s1", domain=(0, 1), fmt=_pct, ticks=(0, .25, .5, .75, 1), title=""):
    """rows: [(label, value or None, tooltip)]"""
    band, bar = 30, 16
    top, plot = 22, W - LABEL - RIGHT
    h = top + band * len(rows) + 8
    lo, hi = domain
    sx = lambda v: LABEL + plot * (v - lo) / (hi - lo)
    out = [f'<svg viewBox="0 0 {W} {h}" role="img" aria-label="{E(title)}">']
    for t in ticks:
        x = sx(t)
        out.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{top - 6}" y2="{h - 8}" class="grid"/>'
                   f'<text x="{x:.1f}" y="{top - 10}" class="tick" text-anchor="middle">{E(fmt(t))}</text>')
    for i, (label, v, tip) in enumerate(rows):
        y = top + i * band + (band - bar) / 2
        out.append(f'<text x="{LABEL - 10}" y="{y + bar / 2 + 4:.1f}" class="lab" text-anchor="end">{E(label)}</text>')
        if _num(v) is None:
            out.append(f'<text x="{LABEL + 4}" y="{y + bar / 2 + 4:.1f}" class="muted">no data</text>')
            continue
        w = max(0.0, sx(v) - LABEL)
        out.append(f'<rect x="{LABEL}" y="{y - 6:.1f}" width="{plot + RIGHT - 8}" height="{bar + 12}" class="hit" data-tip="{E(tip)}"/>')
        if w > 0:
            out.append(f'<path d="{_bar_path(LABEL, y, w, bar)}" style="fill:var({color})" pointer-events="none"/>')
        out.append(f'<text x="{LABEL + w + 6:.1f}" y="{y + bar / 2 + 4:.1f}" class="val" pointer-events="none">{E(fmt(v))}</text>')
    out.append("</svg>")
    return "".join(out)


def stacked(rows, title=""):
    """rows: [(label, {verdict: count})] -> 100% stacked bars with 2px surface gaps."""
    band, bar = 30, 16
    top, plot = 8, W - LABEL - RIGHT
    h = top + band * len(rows) + 8
    out = [f'<svg viewBox="0 0 {W} {h}" role="img" aria-label="{E(title)}">']
    for i, (label, counts) in enumerate(rows):
        y = top + i * band + (band - bar) / 2
        n = sum(counts.values())
        out.append(f'<text x="{LABEL - 10}" y="{y + bar / 2 + 4:.1f}" class="lab" text-anchor="end">{E(label)}</text>')
        if not n:
            out.append(f'<text x="{LABEL + 4}" y="{y + bar / 2 + 4:.1f}" class="muted">no claims made</text>')
            continue
        x = LABEL
        segs = [(k, name, col, counts.get(k, 0)) for k, name, col in VERDICTS if counts.get(k, 0)]
        for j, (k, name, col, c) in enumerate(segs):
            w = plot * c / n - (2 if j < len(segs) - 1 else 0)
            last = j == len(segs) - 1
            shape = (f'<path d="{_bar_path(x, y, w, bar)}"' if last else f'<rect x="{x:.1f}" y="{y}" width="{max(w, 0):.1f}" height="{bar}"')
            out.append(f'{shape} style="fill:var({col})" class="mark" data-tip="{E(f"{label}: {c} of {n} claims {name.lower()}")}"/>')
            x += w + 2
        out.append(f'<text x="{LABEL + plot + 6}" y="{y + bar / 2 + 4:.1f}" class="val">{n}</text>')
    out.append("</svg>")
    return "".join(out)


def dumbbell(rows, domain=(0.4, 1.0), title="", names=("cited passage", "random passages"),
             tick=lambda t: f"{t:.1f}", val=lambda v: f"{v:.3f}", gap=lambda d: f"+{d:.2f}",
             empty="no claim with a valid citation"):
    """rows: [(label, a, b)]: two values per row, a in series 1, b in series 2."""
    band = 30
    top, plot = 22, W - LABEL - RIGHT
    h = top + band * len(rows) + 8
    lo, hi = domain
    sx = lambda v: LABEL + plot * (v - lo) / (hi - lo)
    out = [f'<svg viewBox="0 0 {W} {h}" role="img" aria-label="{E(title)}">']
    for t in [lo + k * (hi - lo) / 6 for k in range(7)]:
        x = sx(t)
        out.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{top - 6}" y2="{h - 8}" class="grid"/>'
                   f'<text x="{x:.1f}" y="{top - 10}" class="tick" text-anchor="middle">{E(tick(t))}</text>')
    for i, (label, a, b) in enumerate(rows):
        y = top + i * band + band / 2
        out.append(f'<text x="{LABEL - 10}" y="{y + 4:.1f}" class="lab" text-anchor="end">{E(label)}</text>')
        if _num(a) is None:
            out.append(f'<text x="{LABEL + 4}" y="{y + 4:.1f}" class="muted">{E(empty)}</text>')
            continue
        out.append(f'<line x1="{sx(b):.1f}" x2="{sx(a):.1f}" y1="{y}" y2="{y}" class="link"/>')
        for v, col, name in ((b, "--s2", names[1]), (a, "--s1", names[0])):
            out.append(f'<circle cx="{sx(v):.1f}" cy="{y}" r="5" style="fill:var({col})" class="dot" '
                       f'data-tip="{E(f"{label}, {name}: {val(v)}")}"/>')
        out.append(f'<text x="{sx(max(a, b)) + 10:.1f}" y="{y + 4:.1f}" class="val">{E(gap(a - b))}</text>')
    out.append("</svg>")
    return "".join(out)


def legend(items):
    return '<div class="legend">' + "".join(
        f'<span><i style="background:var({c})"></i>{E(n)}</span>' for n, c in items) + "</div>"


# ---------------------------------------------------------------------------
def short(r):
    return f'{r["id"][:3]} {r["country"][:3].upper()} {r["perturbation"].replace("_", " ")}'


def render(run_dir: str | Path, findings: str | Path | None = None, out: str | Path | None = None,
           strict_dir: str | Path | None = None) -> Path:
    """run_dir: the records to report (normally the rescored run with citation ids
    normalised). strict_dir: the run as generated, for the before/after comparison."""
    run_dir = Path(run_dir)
    recs = load_records(run_dir)
    if not recs:
        raise SystemExit(f"no persona records in {run_dir}")
    meta = {}
    if (run_dir / "summary.json").exists():
        meta = json.loads((run_dir / "summary.json").read_text()).get("meta", {})
    H = headline(recs)
    strict = load_records(Path(strict_dir)) if strict_dir else []
    Hs = headline(strict) if strict else None
    # the verifier judged identical claims twice when a citation was valid in both passes
    twice = [(ca["verdict"], cb["verdict"]) for ra, rb in zip(strict, recs)
             for ca, cb in zip(ra["claims_all"]["claims"], rb["claims_all"]["claims"])
             if ca["in_context"] and cb["in_context"] and ca["cite"] == cb["cite"]]
    accepts = [r for r in recs if r["vote"] == "accept"]
    below = [r for r in accepts if r["utility"] < r["no_deal"]]
    notes = yaml.safe_load(open(findings, encoding="utf-8")) if findings and Path(findings).exists() else {}

    grounded_rows = [(short(r), _num(r["claims_all"]["grounded_claim_rate"]),
                      f'{short(r)}: {_pct(r["claims_all"]["grounded_claim_rate"])} of {r["claims_all"]["n_claims"]} claims grounded')
                     for r in recs]
    verdict_rows = []
    for r in recs:
        counts = {}
        for c in r["claims_all"]["claims"]:
            k = c["verdict"] or "uncited"
            if k == "uncited" or not c["cite"]:
                k = "uncited"
            counts[k] = counts.get(k, 0) + 1
        verdict_rows.append((short(r), counts))
    cos_rows = [(short(r), _num(r["claims_all"]["claim_cos"]), _num(r["claims_all"]["claim_cos_random"])) for r in recs]
    numfid_rows = [(short(r), _num(r["worksheet"]["numeric_fidelity"]),
                    f'{short(r)}: {_pct(r["worksheet"]["numeric_fidelity"])} of figures found in the cited files')
                   for r in recs]
    attr_rows = [(short(r), _num(r["worksheet"]["attribution_acc"]),
                  f'{short(r)}: best-matching library chunk is in a cited file for {_pct(r["worksheet"]["attribution_acc"])} of answers')
                 for r in recs]
    valid_rows = [(short(r), _num(r["claims_all"]["cite_valid_rate"]), _num(rs["claims_all"]["cite_valid_rate"]))
                  for r, rs in zip(recs, strict)]
    lo = min([v for _, a, b in cos_rows for v in (a, b) if v is not None] or [0.4])
    cos_domain = (math.floor(lo * 10) / 10, 1.0)

    def tile(value, label, sub):
        return f'<div class="tile"><div class="big">{E(value)}</div><div class="tlabel">{E(label)}</div><div class="tsub">{E(sub)}</div></div>'

    vs = f'; {_pct(Hs["cite_valid"])} as generated' if Hs else ""
    gs = f'; {_pct(Hs["grounded"])} as generated' if Hs else ""
    tiles = "".join([
        tile(_pct(H["cite_valid"]), "Valid citations", f'{round(H["cite_valid"] * H["claims"])} of {H["claims"]} claims cite a passage the delegate was shown, after id normalisation{vs}'),
        tile(_pct(H["grounded"]), "Grounded claims", f'valid citation and judged supported or partly supported{gs}'),
        tile(f'{sum(a == b for a, b in twice)} of {len(twice)}', "Verifier agrees with itself",
             "identical claims and passages judged twice by the same model"),
        tile(f'{len(below)} of {len(accepts)}', "Accepts below no deal",
             "accept votes for a text worth less to that country than walking away, with payoffs shown"),
        tile(_pct(H["ungrounded_rate"]), "Invented figures", f'{H["ungrounded_numbers"]} of {H["stated_numbers"]} numbers in public statements appear in nothing the delegate saw'),
        tile(f'{H["planted_hits"]} of {H["planted"]}', "Planted false figures repeated", "by the false-facts persona's delegate"),
        tile(_pct(H["ws_attr"]), "Worksheet attribution", "answers whose closest library passage is in the file they cite"),
        tile(f'{_pct(H["ws_numfid_ff"])} vs {_pct(H["ws_numfid_ref"])}', "Worksheet figures found in sources", "false-facts persona vs the four references"),
    ])

    rows_html = []
    for r in recs:
        c, w = r["claims_all"], r["worksheet"]
        rows_html.append("<tr>" + "".join(f"<td>{x}</td>" for x in [
            E(r["id"]), E(r["country"]), E(r["perturbation"].replace("_", " ")), len(r["part_c_issues"]),
            _pct(w["source_coverage"]), _pct(w["attribution_acc"]), _f(w["support_cos"]), _pct(w["numeric_fidelity"]),
            _pct(r["retrieval"]["briefing_relevance"]), c["n_claims"], _pct(c["cite_valid_rate"]),
            _pct(c["judge_supported_rate"]), _pct(c["grounded_claim_rate"]), _f(c["claim_cos"]), _f(c["claim_cos_random"]),
            f'{len(r["ungrounded_numbers"])}/{len(r["stated_numbers"])}',
            (f'{len(r["planted_hits"])}/{len(r["planted"])}' if r["planted"] else "–"), E(str(r["vote"])),
            _pct(r["llm"]["json_ok_rate"]), _f(r["llm"]["mean_secs"], 0)]) + "</tr>")

    examples = []
    for r in recs:
        for c in r["claims_all"]["claims"]:
            if c["cite"] and len(examples) < 8 and (c["verdict"] in ("unsupported", "bad_id") or not c["in_context"]):
                examples.append((r, c))
    ex_html = "".join(f'<li><span class="pill {E(c["verdict"] or "")}">{E((c["verdict"] or "").replace("_", " "))}</span>'
                      f'<b>{E(short(r))}</b> “{E(c["text"][:220])}” <code>{E(str(c["cite"]))}</code></li>' for r, c in examples)

    find_html = "".join(f"<li>{E(f)}</li>" for f in notes.get("findings", []))
    caveat_html = "".join(f"<li>{E(f)}</li>" for f in notes.get("caveats", []))
    model = meta.get("model") or "llama3.2"
    created = meta.get("created", "")

    page = TEMPLATE.format(
        model=E(model), created=E(created), n=H["personas"], claims=H["claims"], calls=H["calls"],
        secs=_f(H["secs"], 0), json_ok=_pct(H["json_ok"]), chunks=meta.get("library_chunks", "117"),
        collection=E(str(meta.get("collection") or "amu_darya_v2")), tiles=tiles,
        findings=find_html or "<li>Findings not written yet.</li>", caveats=caveat_html,
        grounded_chart=hbars(grounded_rows, title="Grounded claim rate by persona"),
        valid_legend=legend([("After id normalisation", "--s1"), ("As generated", "--s2")]),
        valid_chart=dumbbell(valid_rows, (0, 1), "Valid citations before and after id normalisation",
                             names=("after id normalisation", "as generated"), tick=_pct, val=_pct,
                             gap=lambda d: f"+{100 * d:.0f} pts", empty="no claims") if strict else "",
        verdict_legend=legend([(n, c) for _, n, c in VERDICTS]),
        verdict_chart=stacked(verdict_rows, title="Verifier verdicts by persona"),
        cos_legend=legend([("Cited passage", "--s1"), ("Random library passages", "--s2")]),
        cos_chart=dumbbell(cos_rows, cos_domain, title="Claim to passage similarity"),
        numfid_chart=hbars(numfid_rows, color="--s1", title="Worksheet figures found in cited files"),
        attr_chart=hbars(attr_rows, color="--s1", title="Worksheet attribution accuracy"),
        rows="".join(rows_html), examples=ex_html or "<li>None.</li>",
        accept=H["accept"], cos=_f(H["claim_cos"]), cos_r=_f(H["claim_cos_random"]),
        num_fid=_pct(H["claim_num_fid"]))
    out = Path(out or run_dir / "report.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    return out


TEMPLATE = """<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Persona Grounding Report</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Condensed:wght@500;600&family=IBM+Plex+Sans:ital,wght@0,400;0,500;1,400&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* Layout: a single reading column of report sections; charts span the column, tiles wrap in a grid. */
:root {{
  --bg: #f6f8f8; --panel: #ffffff; --ink: #10201f; --ink-2: #4a5958; --ink-3: #6f7d7c; --rule: #dde4e3;
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --s4: #eda100; --s5: #b9c2c1;
  --good: #1f7a4d; --bad: #b3261e;
  --display: "IBM Plex Sans Condensed", "Arial Narrow", system-ui, sans-serif;
  --body: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, "Cascadia Mono", Consolas, monospace;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg: #0f1716; --panel: #172120; --ink: #e8efee; --ink-2: #b4c1bf; --ink-3: #8a9896; --rule: #2a3735;
  --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500; --s5: #55625f; --good: #5cc48f; --bad: #f08a80; color-scheme: dark }} }}
:root[data-theme="dark"] {{
  --bg: #0f1716; --panel: #172120; --ink: #e8efee; --ink-2: #b4c1bf; --ink-3: #8a9896; --rule: #2a3735;
  --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500; --s5: #55625f; --good: #5cc48f; --bad: #f08a80; color-scheme: dark }}
* {{ box-sizing: border-box }}
body {{ background: var(--bg); color: var(--ink); font: 15px/1.6 var(--body); margin: 0;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }}
main {{ max-width: 920px; margin: 0 auto; padding-block: 40px 64px; display: grid; gap: 40px }}
header {{ display: grid; gap: 10px }}
.eyebrow {{ font: 500 12px/1 var(--mono); letter-spacing: .08em; text-transform: uppercase; color: var(--ink-3) }}
h1 {{ font: 600 clamp(30px, 5vw, 44px)/1.05 var(--display); margin: 0; text-wrap: balance; letter-spacing: -.01em }}
h2 {{ font: 600 22px/1.2 var(--display); margin: 0 0 6px; text-wrap: balance }}
p {{ margin: 0; max-width: 68ch }}
.lede {{ color: var(--ink-2); font-size: 17px }}
.meta {{ font: 13px/1.5 var(--mono); color: var(--ink-3); display: flex; flex-wrap: wrap; gap: 4px 18px }}
section {{ display: grid; gap: 14px; min-width: 0 }}
.sub {{ color: var(--ink-2) }}
.tiles {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 1px; background: var(--rule); border: 1px solid var(--rule) }}
.tile {{ background: var(--panel); padding: 18px 18px 16px; display: grid; gap: 4px; align-content: start }}
.big {{ font: 600 34px/1 var(--display); font-variant-numeric: tabular-nums }}
.tlabel {{ font-weight: 500 }}
.tsub {{ color: var(--ink-3); font-size: 13px; line-height: 1.45 }}
.chart {{ background: var(--panel); border: 1px solid var(--rule); padding: 14px 12px 6px; overflow-x: auto }}
.chart svg {{ width: 100%; min-width: 520px; height: auto; display: block }}
svg text {{ font-family: var(--mono); font-size: 12px; fill: var(--ink-2) }}
svg .lab {{ fill: var(--ink); font-family: var(--body); font-size: 13px }}
svg .tick, svg .muted {{ fill: var(--ink-3); font-size: 11px }}
svg .val {{ fill: var(--ink); font-variant-numeric: tabular-nums }}
svg .grid {{ stroke: var(--rule); stroke-width: 1 }}
svg .link {{ stroke: var(--ink-3); stroke-width: 2; opacity: .5 }}
svg .dot {{ stroke: var(--panel); stroke-width: 2 }}
svg .hit {{ fill: transparent }}
svg .mark, svg .dot, svg .hit {{ cursor: default }}
.legend {{ display: flex; flex-wrap: wrap; gap: 6px 18px; font-size: 13px; color: var(--ink-2) }}
.legend i {{ display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 7px; vertical-align: -1px }}
ul.list {{ margin: 0; padding-left: 20px; display: grid; gap: 8px; max-width: 76ch }}
.examples {{ list-style: none; padding: 0; margin: 0; display: grid; gap: 10px }}
.examples li {{ background: var(--panel); border: 1px solid var(--rule); padding: 10px 12px; font-size: 14px; overflow-wrap: anywhere }}
.examples b {{ font-weight: 500; margin-right: 6px }}
code {{ font: 12px var(--mono); color: var(--ink-2) }}
.pill {{ font: 500 11px/1 var(--mono); text-transform: uppercase; letter-spacing: .05em; padding: 3px 6px; margin-right: 8px; border: 1px solid currentColor; color: var(--bad) }}
.pill.partial {{ color: var(--ink-2) }}
.tablewrap {{ overflow-x: auto; border: 1px solid var(--rule); background: var(--panel) }}
table {{ border-collapse: collapse; font: 12px/1.4 var(--mono); font-variant-numeric: tabular-nums; min-width: 1100px; width: 100% }}
th, td {{ padding: 7px 9px; text-align: right; border-bottom: 1px solid var(--rule); white-space: nowrap }}
th {{ font-weight: 500; color: var(--ink-3); position: sticky; top: 0; background: var(--panel); text-align: right }}
td:nth-child(-n+3), th:nth-child(-n+3) {{ text-align: left }}
dl {{ display: grid; grid-template-columns: max-content 1fr; gap: 6px 18px; margin: 0; font-size: 14px }}
dt {{ font-weight: 500 }} dd {{ margin: 0; color: var(--ink-2) }}
@media (max-width: 820px) {{ .tiles {{ grid-template-columns: repeat(2, 1fr) }} }}
@media (max-width: 460px) {{ .tiles {{ grid-template-columns: 1fr }} }}
@media (max-width: 560px) {{ dl {{ grid-template-columns: 1fr }} dd {{ margin-bottom: 8px }} }}
#tip {{ position: fixed; pointer-events: none; background: var(--ink); color: var(--bg); font: 12px/1.4 var(--mono); padding: 6px 8px; max-width: 280px; z-index: 5 }}
</style>
<main>
<header>
  <div class="eyebrow">Amu Darya negotiation · RAG truth grounding</div>
  <h1>Do the country delegates stick to the documents?</h1>
  <p class="lede">Ten synthetic personas, each built from a country worksheet, were run as negotiation delegates on the team's document library. Every claim they made was checked against the passage it cited, and every worksheet answer against the files it names.</p>
  <div class="meta"><span>model {model}</span><span>{n} personas · {claims} claims · {calls} model calls</span><span>library: {chunks} chunks, Chroma collection {collection}</span><span>{created}</span></div>
</header>

<section>
  <h2>Key numbers</h2>
  <div class="tiles">{tiles}</div>
</section>

<section>
  <h2>What the run shows</h2>
  <ul class="list">{findings}</ul>
</section>

<section>
  <h2>Most bad citations are formatting, not fabrication</h2>
  <p class="sub">The model often copies the square brackets passages are shown in, or appends a suffix to a real id. The verifier now resolves those to the passage they name; ids that match no passage stay invalid.</p>
  {valid_legend}
  <div class="chart">{valid_chart}</div>
</section>

<section>
  <h2>Grounded claims by persona</h2>
  <p class="sub">Share of each delegate's claims that cite a passage it was shown and that the verifier judged supported or partly supported. Opening statement and vote on the Protocol 566 text combined.</p>
  <div class="chart">{grounded_chart}</div>
</section>

<section>
  <h2>How the verifier judged each claim</h2>
  <p class="sub">Every factual claim the delegates listed, by verdict. The number at the end of each bar is the claim count.</p>
  {verdict_legend}
  <div class="chart">{verdict_chart}</div>
</section>

<section>
  <h2>Are claims closer to their cited passage than to chance?</h2>
  <p class="sub">Embedding similarity (nomic-embed-text) of each claim to the passage it cites, against the same claim compared with random library passages. The gap is labelled. Across all valid citations: {cos} cited vs {cos_r} random. Figures in claims found in the cited passage: {num_fid}.</p>
  {cos_legend}
  <div class="chart">{cos_chart}</div>
</section>

<section>
  <h2>Worksheet grounding: are the figures in the sources?</h2>
  <p class="sub">Share of the numbers in each worksheet's answers that appear in the text of the files those answers cite. This is the check that catches planted false figures; embedding similarity does not.</p>
  <div class="chart">{numfid_chart}</div>
</section>

<section>
  <h2>Worksheet grounding: do answers point to the right file?</h2>
  <p class="sub">Share of answers whose closest passage in the whole library comes from a file the answer cites. Several files cover the same facts, so a miss can mean another document says it better, not that the citation is wrong. The unsourced persona has no citations to check.</p>
  <div class="chart">{attr_chart}</div>
</section>

<section>
  <h2>Claims that failed the check</h2>
  <p class="sub">Up to eight claims with a citation that was missing, outside the delegate's context, or judged unsupported.</p>
  <ul class="examples">{examples}</ul>
</section>

<section>
  <h2>All metrics</h2>
  <div class="tablewrap"><table>
    <thead><tr><th>Persona</th><th>Country</th><th>Variant</th><th>Part C issues</th><th>Sourced</th><th>Attribution</th><th>Support cos</th><th>Figures found</th><th>Briefing relevance</th><th>Claims</th><th>Valid cite</th><th>Supported</th><th>Grounded</th><th>Claim cos</th><th>Random cos</th><th>Invented figures</th><th>Planted repeated</th><th>Vote</th><th>JSON ok</th><th>Secs/call</th></tr></thead>
    <tbody>{rows}</tbody>
  </table></div>
</section>

<section>
  <h2>How it was measured</h2>
  <dl>
    <dt>Personas</dt><dd>Four sourced country worksheets (references) and six variants, each changing one thing: emotive wording, maximal goals, cost-sharing goals, no sources, planted false figures, and a role label with no background.</dd>
    <dt>Delegate run</dt><dd>Each persona's delegate is built as in a real negotiation (worksheet persona and goals, 8-passage briefing from Chroma), gives an opening statement, runs its document lookups, then votes on the Protocol 566 anchor text with its own payoffs shown. {accept} of {n} accepted.</dd>
    <dt>Verifier</dt><dd>The pipeline's own verifier: the same model reads each claim beside the cited passage and returns supported, partial or unsupported.</dd>
    <dt>Invented figures</dt><dd>Numbers of 10 or more, or with a decimal point, in the delegate's public statements that occur nowhere in the prompts it received.</dd>
    <dt>Runtime</dt><dd>Local CPU inference through Ollama, {secs} s per call on average, {json_ok} of calls returned valid JSON.</dd>
  </dl>
  <ul class="list">{caveats}</ul>
</section>
</main>
<div id="tip" hidden></div>
<script>
(() => {{
  const tip = document.getElementById("tip");
  document.addEventListener("pointermove", e => {{
    const t = e.target.closest && e.target.closest("[data-tip]");
    if (!t) {{ tip.hidden = true; return; }}
    tip.textContent = t.getAttribute("data-tip");
    tip.hidden = false;
    const x = Math.min(e.clientX + 14, window.innerWidth - tip.offsetWidth - 8);
    tip.style.left = x + "px"; tip.style.top = (e.clientY + 16) + "px";
  }});
  document.addEventListener("pointerleave", () => {{ tip.hidden = true; }});
}})();
</script>
"""
