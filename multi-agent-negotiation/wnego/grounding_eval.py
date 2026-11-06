"""RAG truth grounding evaluation of persona worksheets.

For each persona (a worksheet, usually from config/worksheets/synthetic/):

1. Worksheet grounding (no LLM). Each answer is embedded and compared with the
   RAG library:
     source_coverage      answers with a source other than NO SOURCE YET
     source_resolvable    cited file names that exist in the library
     attribution_acc      the closest chunk in the whole library comes from a cited file
     support_cos          cosine of the answer to its best chunk in the cited files
     numeric_fidelity     numbers in the answer that appear in the cited files' text

2. Delegate run. The country's delegate is built exactly as in a negotiation
   (worksheet persona and goals, Chroma briefing) and makes an opening statement,
   runs its lookups, then votes on the Protocol 566 anchor text.

3. Claim grounding of the delegate's output:
     cite_valid           cited id is a passage the delegate was actually shown
     judge_*              the pipeline's Verifier (LLM judge): supported, partial, unsupported
     grounded_claim_rate  valid citation and judged supported or partial
     claim_numeric_fid    numbers in a claim that appear in its cited passage
     claim_cos            cosine of claim to its cited passage, against a baseline of
                          the same claims compared with random library passages
     ungrounded_numbers   numbers in public statements found nowhere in the prompts
                          the delegate saw (hallucinated figures)
     planted_propagation  planted false figures (false_facts persona) that reach the
                          delegate's statements or claims

Numbers are compared as normalised tokens (commas removed); only values >= 10 or
with a decimal point count, so small counts like "2 outcomes" do not dominate.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np

from . import worksheets as W
from .agents import Condition, Delegate, Toolbox, Verifier
from .environment import Basin
from .goals import GoalBook

_NUM = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?")
KEY_TERMS = {
    "Afghanistan": ("afghan", "qosh tepa", "panj", "kunduz", "kokcha"),
    "Tajikistan": ("tajik", "rogun", "nurek", "vakhsh", "bahri tojik", "dushanbe"),
    "Uzbekistan": ("uzbek", "khorezm", "karakalpak", "nukus", "tashkent"),
    "Turkmenistan": ("turkmen", "karakum", "kerki", "avaza"),
}


def numbers(text: str) -> set[str]:
    out = set()
    for m in _NUM.findall(text or ""):
        v = m.replace(",", "").rstrip(".")
        try:
            f = float(v)
        except ValueError:
            continue
        if f >= 10 or "." in v:
            out.add(v.rstrip("0").rstrip(".") if "." in v else v)
    return out


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n == 0, 1, n)


class Recorder:
    """Wraps an LLM: logs every prompt, parsed output, latency and failure."""

    def __init__(self, llm, phase: str):
        self.llm, self.phase, self.log = llm, phase, []

    @property
    def calls(self):
        return self.llm.calls

    def complete_json(self, system, user, meta=None, retries=2):
        t, before = time.perf_counter(), self.llm.calls
        try:
            out, ok, err = self.llm.complete_json(system, user, meta, retries), True, None
        except Exception as e:      # keep the evaluation going; a failed call is itself a metric
            out, ok, err = {}, False, repr(e)
        self.log.append({"phase": self.phase, "system": system, "user": user, "output": out, "ok": ok, "error": err,
                         "attempts": self.llm.calls - before, "secs": round(time.perf_counter() - t, 2)})
        return out


class Library:
    """The RAG library as an embedding matrix, read from the Chroma collection
    the agents use (or embedded from the documents folder for BM25 runs)."""

    def __init__(self, retriever, embed, docs_dir: Path):
        col = getattr(retriever, "col", None)
        if col is not None:
            r = col.get(include=["documents", "metadatas", "embeddings"])
            self.ids, self.texts = r["ids"], r["documents"]
            self.sources = [m.get("source", "") for m in r["metadatas"]]
            self.E = _unit(np.array(r["embeddings"]))
        else:
            from .retrieval import iter_chunks
            rows = list(iter_chunks(docs_dir))
            self.ids, self.texts = [c for c, *_ in rows], [t for _, t, *_ in rows]
            self.sources = [m["source"] for _, _, m, _ in rows]
            self.E = _unit(np.array([embed(t) for t in self.texts]))
        self.embed = embed
        self._cache: dict[str, np.ndarray] = {}

    def vec(self, text: str) -> np.ndarray:
        if text not in self._cache:
            self._cache[text] = _unit(np.array(self.embed(text)))
        return self._cache[text]

    def files(self) -> set[str]:
        return set(self.sources)

    def text_of(self, files) -> str:
        return "\n".join(t for t, s in zip(self.texts, self.sources) if s in files)


# ---------------------------------------------------------------------------
def worksheet_grounding(ws: dict, lib: Library) -> dict:
    rows, known = [], lib.files()
    for a in W.answers(ws):
        if not a["text"]:
            continue
        files = W.source_files(a["source"])
        row = {"field": a["field"], "text": a["text"], "source": a["source"], "sourced": bool(files),
               "files": files, "resolvable": [f in known for f in files]}
        if files and any(row["resolvable"]):
            v = lib.vec(a["text"])
            sims = lib.E @ v
            mask = np.array([s in files for s in lib.sources])
            best_all = int(np.argmax(sims))
            row["support_cos"] = float(sims[mask].max())
            row["best_chunk"] = lib.ids[int(np.flatnonzero(mask)[np.argmax(sims[mask])])]
            row["best_any"] = lib.ids[best_all]
            row["attributed"] = lib.sources[best_all] in files
            nums = numbers(a["text"])
            src_nums = numbers(lib.text_of(files))
            row["numbers"] = sorted(nums)
            row["numbers_found"] = sorted(nums & src_nums)
        rows.append(row)
    scored = [r for r in rows if "support_cos" in r]
    with_nums = [r for r in scored if r["numbers"]]
    all_files = [ok for r in rows for ok in r["resolvable"]]
    return {
        "answers": rows,
        "source_coverage": float(np.mean([r["sourced"] for r in rows])) if rows else np.nan,
        "source_resolvable": float(np.mean(all_files)) if all_files else np.nan,
        "attribution_acc": float(np.mean([r["attributed"] for r in scored])) if scored else np.nan,
        "support_cos": float(np.mean([r["support_cos"] for r in scored])) if scored else np.nan,
        "numeric_fidelity": (float(sum(len(r["numbers_found"]) for r in with_nums) / sum(len(r["numbers"]) for r in with_nums))
                             if with_nums else np.nan),
    }


def _claims_metrics(checked: list[dict], shown: dict, lib: Library, rng) -> dict:
    n = len(checked)
    rows = []
    for c in checked:
        cite = c.get("cite")
        p = shown.get(cite) if cite else None
        row = {"text": c.get("text", ""), "cite": cite, "verdict": c.get("verdict"), "in_context": p is not None}
        if p is not None:
            cv = lib.vec(row["text"] or " ")
            row["cos"] = float(cv @ lib.vec(p.text))
            row["cos_random"] = float(np.mean(lib.E[rng.choice(len(lib.ids), size=min(5, len(lib.ids)), replace=False)] @ cv))
            nums = numbers(row["text"])
            row["numbers"] = sorted(nums)
            row["numbers_found"] = sorted(nums & numbers(p.text))
        rows.append(row)
    valid = [r for r in rows if r["in_context"]]
    with_nums = [r for r in valid if r["numbers"]]

    def rate(pred, pool=rows):
        return float(np.mean([pred(r) for r in pool])) if pool else np.nan

    return {
        "claims": rows, "n_claims": n,
        "cite_valid_rate": rate(lambda r: r["in_context"]),
        "uncited_rate": rate(lambda r: not r["cite"]),
        "bad_id_rate": rate(lambda r: r["verdict"] == "bad_id"),
        "judge_supported_rate": rate(lambda r: r["verdict"] == "supported"),
        "judge_partial_rate": rate(lambda r: r["verdict"] == "partial"),
        "judge_unsupported_rate": rate(lambda r: r["verdict"] == "unsupported"),
        "grounded_claim_rate": rate(lambda r: r["in_context"] and r["verdict"] in ("supported", "partial")),
        "claim_numeric_fid": (float(sum(len(r["numbers_found"]) for r in with_nums) / sum(len(r["numbers"]) for r in with_nums))
                              if with_nums else np.nan),
        "claim_cos": float(np.mean([r["cos"] for r in valid])) if valid else np.nan,
        "claim_cos_random": float(np.mean([r["cos_random"] for r in valid])) if valid else np.nan,
    }


def eval_persona(ws: dict, llm, retriever, embed, lib: Library, basin: Basin, seed: int = 0) -> dict:
    country = ws["country"]
    rng = np.random.default_rng(seed)
    cond = Condition(persona="eval", payoff="own", grounding="rag", scenario=basin.scenario, max_rounds=6)
    goals = GoalBook({country: W.goal_file(ws)})
    agent_llm, judge_llm = Recorder(llm, "delegate"), Recorder(llm, "verifier")
    tools = Toolbox(basin, retriever, cond)
    d = Delegate(country, agent_llm, basin, cond, {"eval": {country: W.persona_text(ws)}}, retriever, goals, tools, rng)
    verifier = Verifier(judge_llm, retriever)
    briefing = list(d.briefing)

    opening = d.opening([])
    shown_open = {p.id: p for p in d.known_passages}
    opening["claims_checked"] = verifier.check(opening.get("claims") or [])
    d.run_tools(opening, 0)
    public = [f"[Round 0] {country}: {opening.get('statement', '')}"]
    text = basin.protocol_566_proposal()
    resp = d.respond(text, 1, public)
    shown_resp = {p.id: p for p in d.known_passages}
    resp["claims_checked"] = verifier.check(resp.get("claims") or [])

    terms = KEY_TERMS.get(country, (country.lower(),))
    claims = _claims_metrics(opening["claims_checked"], shown_open, lib, rng)
    claims_r = _claims_metrics(resp["claims_checked"], shown_resp, lib, rng)
    all_claims = _claims_metrics(opening["claims_checked"] + resp["claims_checked"], {**shown_open, **shown_resp}, lib, rng)

    seen = "\n".join(e["system"] + "\n" + e["user"] for e in agent_llm.log)
    said = " ".join(str(o.get(k) or "") for o in (opening, resp) for k in ("statement",))
    stated_nums = numbers(said)
    ungrounded = sorted(stated_nums - numbers(seen))
    out_text = said + " " + " ".join(c.get("text", "") for c in opening["claims_checked"] + resp["claims_checked"])
    planted = (ws.get("synthetic") or {}).get("planted") or []
    planted_hits = [p for p in planted if p.lower() in out_text.lower()]
    u, nd = basin.utilities(text)[country], basin.no_deal_utilities()[country]
    calls = agent_llm.log + judge_llm.log
    return {
        "id": ws["id"], "country": country, "perturbation": (ws.get("synthetic") or {}).get("perturbation", "base"),
        "tests": (ws.get("synthetic") or {}).get("tests", ""),
        "part_c_issues": W.check(ws),
        "worksheet": worksheet_grounding(ws, lib),
        "retrieval": {
            "briefing": [{"id": p.id, "source": p.source, "score": p.score} for p in briefing],
            "briefing_relevance": float(np.mean([any(t in p.text.lower() for t in terms) for p in briefing])) if briefing else np.nan,
            "briefing_sources": len({p.source for p in briefing}),
            "briefing_score": float(np.mean([p.score for p in briefing])) if briefing else np.nan,
            "lookups": [t for t in tools.log if t["tool"] == "lookup"],
        },
        "claims_all": all_claims, "claims_opening": claims, "claims_response": claims_r,
        "statements": {"opening": opening.get("statement", ""), "response": resp.get("statement", "")},
        "vote": resp.get("vote"), "utility": u, "no_deal": nd,
        "vote_payoff_consistent": int((resp.get("vote") == "accept") == (u >= nd)),
        "stated_numbers": sorted(stated_nums), "ungrounded_numbers": ungrounded,
        "ungrounded_number_rate": len(ungrounded) / len(stated_nums) if stated_nums else 0.0,
        "planted": planted, "planted_hits": planted_hits,
        "planted_propagation": len(planted_hits) / len(planted) if planted else np.nan,
        "llm": {"calls": len(calls), "json_ok_rate": float(np.mean([c["ok"] for c in calls])) if calls else np.nan,
                "retries": sum(max(0, c["attempts"] - 1) for c in calls),
                "mean_secs": float(np.mean([c["secs"] for c in calls])) if calls else np.nan},
        "transcript": [{k: v for k, v in c.items() if k != "system"} for c in calls],
    }


def rescore(run_dir, out_dir, llm, retriever, embed, docs_dir, log=print) -> None:
    """Re-verify a finished run's claims with citation ids normalised (brackets
    stripped, appended suffixes resolved to the passage they start with). The
    delegates' outputs are not regenerated: same claims, same passages shown,
    only the id matching and the verifier call are redone."""
    run_dir, out = Path(run_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    lib = Library(retriever, embed, Path(docs_dir))
    for f in sorted(run_dir.glob("P*.json")):
        r = json.loads(f.read_text())
        shown_ids = {b["id"] for b in r["retrieval"]["briefing"]} | {
            pid for t in r["retrieval"]["lookups"] for pid in t.get("results", [])}
        shown = {pid: p for pid in shown_ids if (p := retriever.get(pid)) is not None}
        raw = [{"text": c["text"], "cite": c.get("cite_raw") or c["cite"]} for c in r["claims_all"]["claims"]]
        judge = Recorder(llm, "verifier")
        checked = Verifier(judge, retriever, normalize_ids=True).check(raw)
        r["claims_all"] = _claims_metrics(checked, shown, lib, np.random.default_rng(0))
        r["rescored"] = {"from": str(run_dir), "normalize_ids": True, "verifier_calls": len(judge.log)}
        (out / f.name).write_text(json.dumps(r, indent=1, default=float))
        log(f"{r['id']}: valid {r['claims_all']['cite_valid_rate']} grounded {r['claims_all']['grounded_claim_rate']}")
    summary = run_dir / "summary.json"
    if summary.exists():
        s = json.loads(summary.read_text())
        s["meta"]["rescored"] = "citation ids normalised"
        (out / "summary.json").write_text(json.dumps(s, indent=1, default=float))


FLAT = [("worksheet", "source_coverage"), ("worksheet", "source_resolvable"), ("worksheet", "attribution_acc"),
        ("worksheet", "support_cos"), ("worksheet", "numeric_fidelity"),
        ("retrieval", "briefing_relevance"), ("retrieval", "briefing_score"), ("retrieval", "briefing_sources"),
        ("claims_all", "n_claims"), ("claims_all", "cite_valid_rate"), ("claims_all", "uncited_rate"),
        ("claims_all", "bad_id_rate"), ("claims_all", "judge_supported_rate"), ("claims_all", "judge_partial_rate"),
        ("claims_all", "judge_unsupported_rate"), ("claims_all", "grounded_claim_rate"),
        ("claims_all", "claim_numeric_fid"), ("claims_all", "claim_cos"), ("claims_all", "claim_cos_random"),
        ("llm", "calls"), ("llm", "json_ok_rate"), ("llm", "retries"), ("llm", "mean_secs")]


def flatten(r: dict) -> dict:
    row = {"id": r["id"], "country": r["country"], "perturbation": r["perturbation"],
           "part_c_issues": len(r["part_c_issues"]), "vote": r["vote"],
           "vote_payoff_consistent": r["vote_payoff_consistent"],
           "ungrounded_number_rate": r["ungrounded_number_rate"], "n_stated_numbers": len(r["stated_numbers"]),
           "planted_propagation": r["planted_propagation"]}
    for sec, k in FLAT:
        row[k if sec != "worksheet" else f"ws_{k}"] = r[sec][k]
    return row


def run(persona_dir, llm, retriever, embed, docs_dir, out_dir, scenario="normal", basin_path=None, seed=0, log=print):
    import pandas as pd
    from .experiment import ROOT
    basin = Basin.load(basin_path or ROOT / "config/basin.yaml", scenario)
    lib = Library(retriever, embed, Path(docs_dir))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sheets = W.load_dir(persona_dir)
    peers = list(W.load_dir(Path(persona_dir).parent).values()) if Path(persona_dir).parent.joinpath("Afghanistan.yaml").exists() else None
    results = []
    for i, (pid, ws) in enumerate(sheets.items(), 1):
        t = time.time()
        try:
            r = eval_persona(ws, llm, retriever, embed, lib, basin, seed)
        except Exception as e:      # one bad persona should not cost the other runs
            log(f"[{i}/{len(sheets)}] {pid}: FAILED {e!r}")
            (out / f"{pid}.error.txt").write_text(repr(e))
            continue
        r["part_c_issues"] = W.check(ws, peers)
        results.append(r)
        (out / f"{pid}.json").write_text(json.dumps(r, indent=1, default=float))
        log(f"[{i}/{len(sheets)}] {pid}: claims={r['claims_all']['n_claims']} "
            f"grounded={r['claims_all']['grounded_claim_rate']} ws_attr={r['worksheet']['attribution_acc']} "
            f"({time.time() - t:.0f}s)")
    df = pd.DataFrame([flatten(r) for r in results])
    df.to_csv(out / "summary.csv", index=False)
    claims = [c for r in results for c in r["claims_all"]["claims"]]
    valid = [c for c in claims if c["in_context"]]
    ws_scored = [a for r in results for a in r["worksheet"]["answers"] if "support_cos" in a]
    headline = {
        "personas": len(results), "claims": len(claims),
        "grounded_claim_rate": float(np.mean([c["in_context"] and c["verdict"] in ("supported", "partial") for c in claims])) if claims else None,
        "cite_valid_rate": float(np.mean([c["in_context"] for c in claims])) if claims else None,
        "judge_supported_rate": float(np.mean([c["verdict"] == "supported" for c in claims])) if claims else None,
        "claim_cos": float(np.mean([c["cos"] for c in valid])) if valid else None,
        "claim_cos_random": float(np.mean([c["cos_random"] for c in valid])) if valid else None,
        "ws_attribution_acc": float(np.mean([a["attributed"] for a in ws_scored])) if ws_scored else None,
        "ws_support_cos": float(np.mean([a["support_cos"] for a in ws_scored])) if ws_scored else None,
        "ungrounded_number_rate": float(df["ungrounded_number_rate"].mean()),
        "json_ok_rate": float(df["json_ok_rate"].mean()),
        "mean_secs_per_call": float(df["mean_secs"].mean()),
        "llm_calls": int(df["calls"].sum()),
    }
    meta = {"scenario": scenario, "seed": seed, "model": getattr(llm, "model", "") or getattr(llm, "name", ""),
            "backend": getattr(llm, "name", ""), "retriever": type(retriever).__name__,
            "collection": getattr(getattr(retriever, "col", None), "name", None), "library_chunks": len(lib.ids),
            "created": time.strftime("%Y-%m-%d %H:%M")}
    (out / "summary.json").write_text(json.dumps({"meta": meta, "headline": headline,
                                                  "personas": df.to_dict(orient="records")}, indent=1, default=float))
    return df, headline
