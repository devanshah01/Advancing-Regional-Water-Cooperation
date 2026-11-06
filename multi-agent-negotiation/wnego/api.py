"""HTTP API for the web front end (web/). Start with: python -m wnego serve

  GET  /api/health                  retriever, collection size, Ollama reachability
  GET  /api/worksheets              base and synthetic worksheets with Part C results
  GET  /api/worksheets/{id}         one worksheet
  PUT  /api/worksheets/{id}         save a base worksheet (synthetic ones are generated)
  POST /api/worksheets/check        Part C checks on an unsaved worksheet
  GET  /api/search?q=&k=&doc_type=  passages from the RAG library the agents use
  POST /api/runs                    start a negotiation in the background
  GET  /api/runs, /api/runs/{id}    status and results
  GET  /api/grounding               latest grounding evaluation (runs/grounding/*/summary.json)
  GET  /api/grounding/{id}          one persona's full grounding record
"""
from __future__ import annotations

import copy
import json
import math
import threading
import time
import uuid
from pathlib import Path

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import worksheets as W
from .agents import Condition
from .experiment import ROOT, load_configs, run_one
from .retrieval import make_retriever

BASE_DIR = ROOT / "config" / "worksheets"
SYNTH_DIR = BASE_DIR / "synthetic"
GROUND_DIR = ROOT / "runs" / "grounding"
WEB_DIST = ROOT / "web" / "dist"

app = FastAPI(title="Multi-agent negotiation API")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["*"], allow_headers=["*"])

_retriever = None
_runs: dict[str, dict] = {}
_lock = threading.Lock()


def retriever():
    global _retriever
    if _retriever is None:
        _retriever = make_retriever(load_configs()["rag"], ROOT)
    return _retriever


def _safe(x):
    """NaN and infinity (no deal, no claims) are not valid JSON; send null."""
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, dict):
        return {k: _safe(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_safe(v) for v in x]
    return x


def _clean(ws: dict) -> dict:
    return {k: v for k, v in ws.items() if not k.startswith("_")}


def _all_sheets() -> dict[str, dict]:
    sheets = {i: {**ws, "_kind": "base"} for i, ws in W.load_dir(BASE_DIR).items()}
    if SYNTH_DIR.exists():
        sheets.update({i: {**ws, "_kind": "synthetic"} for i, ws in W.load_dir(SYNTH_DIR).items()})
    return sheets


def _base_peers(exclude: str | None = None) -> list[dict]:
    return [ws for i, ws in W.load_dir(BASE_DIR).items() if i != exclude]


@app.get("/api/health")
def health():
    cfg = load_configs()["rag"]
    out = {"retriever": cfg.get("backend"), "collection": cfg.get("chroma", {}).get("collection")}
    try:
        r = retriever()
        out["chunks"] = r.col.count() if hasattr(r, "col") else len(getattr(r, "passages", []))
        out["retriever_ok"] = True
    except Exception as e:
        out.update(retriever_ok=False, error=repr(e))
    try:
        import ollama
        out["ollama_models"] = [m.model for m in ollama.list().models]
    except Exception:
        out["ollama_models"] = None
    return out


@app.get("/api/worksheets")
def list_worksheets():
    out = []
    for wid, ws in _all_sheets().items():
        issues = W.check(ws, _base_peers(wid))
        out.append({"id": wid, "country": ws["country"], "kind": ws["_kind"], "filled_by": ws.get("filled_by"),
                    "synthetic": ws.get("synthetic"), "words": W.word_count(ws), "issues": issues})
    return out


@app.get("/api/worksheets/{wid}")
def get_worksheet(wid: str):
    sheets = _all_sheets()
    if wid not in sheets:
        raise HTTPException(404, f"no worksheet {wid}")
    ws = sheets[wid]
    return {"worksheet": _clean(ws), "kind": ws["_kind"], "issues": W.check(ws, _base_peers(wid)),
            "persona_text": W.persona_text(ws), "goal_file": W.goal_file(ws)}


class WorksheetBody(BaseModel):
    worksheet: dict


@app.post("/api/worksheets/check")
def check_worksheet(body: WorksheetBody):
    ws = body.worksheet
    try:
        return {"issues": W.check(ws, _base_peers(ws.get("id"))), "persona_text": W.persona_text(ws),
                "goal_file": W.goal_file(ws)}
    except (KeyError, TypeError, AttributeError) as e:
        raise HTTPException(422, f"malformed worksheet: {e!r}")


@app.put("/api/worksheets/{wid}")
def save_worksheet(wid: str, body: WorksheetBody):
    path = BASE_DIR / f"{wid}.yaml"
    if not path.exists():
        raise HTTPException(403, "only the base country worksheets can be edited; synthetic ones come from wnego/synthetic.py")
    ws = _clean(copy.deepcopy(body.worksheet))
    ws.pop("id", None)
    issues = W.check({**ws, "id": wid}, _base_peers(wid))
    head = "".join(line for line in path.read_text(encoding="utf-8").splitlines(keepends=True) if line.startswith("#"))
    with open(path, "w", encoding="utf-8") as f:
        f.write(head + "\n")
        yaml.safe_dump(ws, f, sort_keys=False, allow_unicode=True, width=110)
    return {"saved": str(path.relative_to(ROOT)), "issues": issues}


@app.get("/api/search")
def search(q: str, k: int = 6, doc_type: str | None = None):
    where = {"doc_type": doc_type} if doc_type else None
    return [{"id": p.id, "title": p.title, "source": p.source, "page": p.meta.get("page"),
             "doc_type": p.meta.get("doc_type"), "score": p.score, "text": p.text}
            for p in retriever().search(q, k, where)]


class RunBody(BaseModel):
    backend: str = "mock"
    model: str = ""
    persona: str = "worksheet"
    payoff: str = "hidden"
    scenario: str = "normal"
    events: str = "none"
    legal: str = "on"
    rounds: int = 6
    seed: int = 0


def _do_run(rid: str, body: RunBody):
    try:
        cond = Condition(persona=body.persona, payoff=body.payoff, grounding="rag", legal=body.legal,
                         events=body.events, scenario=body.scenario, max_rounds=body.rounds)
        spec = {"name": body.model or body.backend, "type": body.backend, "model": body.model}
        res = run_one(spec, cond, body.seed, retriever=retriever())
        keep = ("agreement", "rounds", "final_text", "final_utilities", "no_deal_utilities", "votes", "claims",
                "public_record", "caucus", "tool_calls", "llm_calls", "metrics", "speaking_order")
        result = _safe(json.loads(json.dumps({k: res[k] for k in keep}, default=float)))
        (ROOT / "runs" / "web").mkdir(parents=True, exist_ok=True)
        (ROOT / "runs" / "web" / f"{rid}.json").write_text(json.dumps(result, indent=1))
        with _lock:
            _runs[rid].update(status="done", result=result, finished=time.time())
    except Exception as e:
        with _lock:
            _runs[rid].update(status="failed", error=repr(e), finished=time.time())


@app.post("/api/runs")
def start_run(body: RunBody):
    rid = uuid.uuid4().hex[:8]
    with _lock:
        _runs[rid] = {"id": rid, "status": "running", "params": body.model_dump(), "started": time.time()}
    threading.Thread(target=_do_run, args=(rid, body), daemon=True).start()
    return _runs[rid]


@app.get("/api/runs")
def list_runs():
    with _lock:
        return [{k: v for k, v in r.items() if k != "result"} for r in _runs.values()]


@app.get("/api/runs/{rid}")
def get_run(rid: str):
    if rid not in _runs:
        raise HTTPException(404, f"no run {rid}")
    return _runs[rid]


def _latest_grounding() -> Path | None:
    runs = sorted((p for p in GROUND_DIR.glob("*/summary.json")), key=lambda p: p.stat().st_mtime)
    return runs[-1].parent if runs else None


@app.get("/api/grounding")
def grounding():
    d = _latest_grounding()
    if not d:
        raise HTTPException(404, "no grounding evaluation yet; run python -m wnego ground")
    return _safe({"run": d.name, **json.loads((d / "summary.json").read_text())})


@app.get("/api/grounding/{pid}")
def grounding_persona(pid: str):
    d = _latest_grounding()
    f = d / f"{pid}.json" if d else None
    if not f or not f.exists():
        raise HTTPException(404, f"no grounding record for {pid}")
    r = json.loads(f.read_text())
    r.pop("transcript", None)
    return _safe(r)


if WEB_DIST.exists():       # serve the built front end from the same port
    app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="web")
