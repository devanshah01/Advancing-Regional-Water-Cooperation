"""Factorial experiment runner and statistics.

Mirrors the design in Manoranjan and Gaikwad: every model x persona x payoff
visibility (x grounding) cell, repeated over seeds, with omnibus chi square
tests, Cramer's V and Holm corrected pairwise Fisher exact tests on agreement.
"""
from __future__ import annotations

import itertools
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import chi2_contingency, fisher_exact

from . import worksheets
from .agents import Condition
from .benchmarks import nash_bargaining_solution, zopa_bounds
from .environment import Basin
from .evaluation import evaluate
from .llm import make_llm
from .protocol import Negotiation, save_transcript
from .retrieval import BaseRetriever, make_retriever

ROOT = Path(__file__).resolve().parent.parent
_NBS_CACHE: dict = {}
_ZOPA_CACHE: dict = {}


def _git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "no-git"


def load_configs(root: Path = ROOT) -> dict:
    """Worksheets in config/worksheets/ (one per country) supply the `worksheet`
    persona (Part A) and the goal files (Part B). goals.yaml is used only for a
    country without a worksheet."""
    c = root / "config"
    personas = yaml.safe_load(open(c / "personas.yaml"))
    goals = yaml.safe_load(open(c / "goals.yaml"))
    sheets = {ws["country"]: ws for ws in worksheets.load_dir(c / "worksheets").values()}
    if sheets:
        personas["worksheet"] = {k: worksheets.persona_text(ws) for k, ws in sheets.items()}
        goals.update({k: worksheets.goal_file(ws) for k, ws in sheets.items()})
    return {"personas": personas, "goals": goals, "worksheets": sheets,
            "events": yaml.safe_load(open(c / "events.yaml")) or {},
            "rag": yaml.safe_load(open(c / "rag.yaml"))}


def run_one(model_spec: dict, cond: Condition, seed: int, temperature: float = 0.7,
            basin_path=ROOT / "config/basin.yaml", retriever: BaseRetriever | None = None,
            nbs_cache: dict | None = None, zopa_cache: dict | None = None, configs: dict | None = None,
            approver=None, verifier_temperature: float | None = None):
    nbs_cache = _NBS_CACHE if nbs_cache is None else nbs_cache
    zopa_cache = _ZOPA_CACHE if zopa_cache is None else zopa_cache
    cfg = configs or load_configs()
    basin = Basin.load(basin_path, cond.scenario)
    cond.briefing_size = cfg["rag"].get("briefing_size", cond.briefing_size)
    if retriever is None and cond.grounding == "rag":
        retriever = make_retriever(cfg["rag"], ROOT)
    llm = make_llm(model_spec, temperature)
    # Verifier runs colder by default (0.0): the pilot found it judged the same 7 claims
    # twice and agreed with itself only 3 times at the shared temperature=0.7.
    verifier_llm = (make_llm(model_spec, verifier_temperature) if verifier_temperature is not None
                     and verifier_temperature != temperature else None)
    result = Negotiation(llm, basin, cond, cfg["personas"], retriever, cfg["goals"], cfg["events"],
                         seed=seed, approver=approver, verifier_llm=verifier_llm).run()
    key = (str(basin_path), cond.scenario)
    if key not in nbs_cache:
        nbs_cache[key] = nash_bargaining_solution(basin)
    if key not in zopa_cache:
        zopa_cache[key] = zopa_bounds(basin)
    result["model"] = model_spec.get("name", model_spec.get("type"))
    result["metrics"] = evaluate(result, basin, nbs_cache[key], zopa_cache[key])
    return result


def run_experiment(config_path: str | Path, out_dir: str | Path | None = None) -> pd.DataFrame:
    cfg = yaml.safe_load(open(config_path))
    out = Path(out_dir or ROOT / "runs" / cfg["name"])
    (out / "transcripts").mkdir(parents=True, exist_ok=True)
    configs = load_configs()
    retriever = make_retriever(configs["rag"], ROOT)
    nbs_cache: dict = {}
    zopa_cache: dict = {}
    verifier_temperature = cfg.get("verifier_temperature")
    f = cfg["factors"]
    rows = []
    grid = list(itertools.product(cfg["models"], f["persona"], f["payoff"], f["grounding"], f.get("legal", ["on"]),
                                  f.get("events", ["none"]), cfg["scenarios"], range(cfg["seeds"])))
    for i, (model, persona, payoff, grounding, legal, events, scenario, seed) in enumerate(grid, 1):
        legal = {True: "on", False: "off"}.get(legal, legal)   # YAML reads bare on/off as booleans
        cond = Condition(persona=persona, payoff=payoff, grounding=grounding, legal=legal, events=events,
                         scenario=scenario, max_rounds=cfg.get("max_rounds", 6))
        try:
            res = run_one(model, cond, seed, cfg.get("temperature", 0.7), retriever=retriever if grounding == "rag" else None,
                          nbs_cache=nbs_cache, zopa_cache=zopa_cache, configs=configs,
                          verifier_temperature=verifier_temperature)
        except Exception as e:  # keep the sweep going, record the failure
            rows.append({"model": model.get("name"), **cond.__dict__, "seed": seed, "error": repr(e)})
            print(f"[{i}/{len(grid)}] FAILED {cond.key()} seed={seed}: {e}")
            continue
        tid = f"{res['model']}__{persona}__{payoff}__{grounding}__legal-{legal}__{events}__{scenario}__s{seed}"
        save_transcript(res, out / "transcripts" / f"{tid}.json")
        rows.append({"model": res["model"], **cond.__dict__, "seed": seed, "transcript": tid, **res["metrics"]})
        print(f"[{i}/{len(grid)}] {tid}: agreement={res['agreement']} rounds={res['rounds']}")
    df = pd.DataFrame(rows)
    df.to_csv(out / "results.csv", index=False)
    summarize(df, out)
    (out / "run_info.json").write_text(json.dumps({"config": cfg, "git": _git_hash()}, indent=1))
    return df


def _holm(pvals: list[float]) -> list[float]:
    order = np.argsort(pvals)
    adj, running = [0.0] * len(pvals), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = running
    return adj


def summarize(df: pd.DataFrame, out: Path):
    df = df[df.get("error").isna()] if "error" in df else df
    cells = ["model", "scenario", "persona", "payoff", "grounding", "legal", "events"]
    metrics = ["agreement", "rounds", "below_batna_accept_rate", "payoff_consistent_rate", "gain_ratio_vs_nbs",
               "nbs_distance_km3", "anchor_566_distance_km3", "min_gain_share", "delta_flow_met",
               "claims_supported_rate", "claims_unsupported_rate", "legal_concerns", "lookups", "what_ifs",
               "caucus_messages", "goal_requests", "zopa_exists", "mean_batna_score"]
    metrics = [m for m in metrics if m in df]
    summary = df.groupby(cells)[metrics].mean().round(3)
    summary["n"] = df.groupby(cells).size()
    summary.to_csv(out / "summary.csv")

    lines = []
    for (model, scenario), g in df.groupby(["model", "scenario"]):
        g = g.assign(cell=g["persona"] + "|" + g["payoff"] + "|" + g["grounding"] + "|legal-" + g["legal"] + "|" + g["events"])
        tab = pd.crosstab(g["cell"], g["agreement"])
        lines.append(f"== {model} / {scenario}: agreement by condition ==")
        lines.append(g.groupby("cell")["agreement"].agg(["mean", "count"]).round(3).to_string())
        if tab.shape[1] == 2 and tab.shape[0] > 1:
            chi2, p, dof, _ = chi2_contingency(tab)
            v = np.sqrt(chi2 / (tab.values.sum() * (min(tab.shape) - 1)))
            lines.append(f"omnibus chi2({dof}) = {chi2:.2f}, p = {p:.3g}, Cramer's V = {v:.2f}")
            pairs = list(itertools.combinations(tab.index, 2))
            ps = [fisher_exact(tab.loc[[a, b]].values)[1] for a, b in pairs]
            for (a, b), praw, padj in zip(pairs, ps, _holm(ps)):
                if padj < 0.05:
                    lines.append(f"  {a} vs {b}: Fisher p = {praw:.3g}, Holm p = {padj:.3g}")
        else:
            lines.append("no variation in agreement, tests not defined")
        lines.append("")
    (out / "stats.txt").write_text("\n".join(lines))
