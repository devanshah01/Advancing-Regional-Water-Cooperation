"""Command line entry point.

  python -m wnego run --backend ollama --model qwen2.5:32b --rag chroma --events qosh_tepa_shock
  python -m wnego run --backend mock --scenario dry                 # offline plumbing test
  python -m wnego run --backend ollama --model qwen2.5:32b --hitl final   # human approves the final text
  python -m wnego search "obligation not to cause significant harm" --rag chroma
  python -m wnego index --docs ../data/raw --db ../chroma_db        # improved index (amu_darya_v2)
  python -m wnego experiment config/experiment.yaml
  python -m wnego benchmarks --scenario normal
  python -m wnego worksheets                                        # Part C checks on config/worksheets
  python -m wnego synth                                             # write the 10 synthetic personas
  python -m wnego ground --backend ollama --model llama3.2         # RAG truth grounding evaluation
  python -m wnego rescore runs/grounding/llama3.2_chroma           # re-verify with citation ids normalised
  python -m wnego report runs/grounding/llama3.2_chroma_normalized --strict runs/grounding/llama3.2_chroma
  python -m wnego serve                                             # API for the web front end
"""
import argparse
import json

from .agents import Condition
from .benchmarks import nash_bargaining_solution, utilitarian_optimum
from .environment import Basin
from .experiment import ROOT, load_configs, run_experiment, run_one
from .protocol import save_transcript
from .retrieval import build_index, make_retriever


def _configs_with_rag(a):
    cfg = load_configs()
    if getattr(a, "rag", None):
        cfg["rag"]["backend"] = a.rag
    if getattr(a, "collection", None):
        cfg["rag"]["chroma"]["collection"] = a.collection
    return cfg


def main():
    ap = argparse.ArgumentParser(prog="wnego")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run one negotiation and print the public record")
    r.add_argument("--backend", default="mock", choices=["mock", "ollama", "anthropic", "openai_compat"])
    r.add_argument("--model", default="")
    r.add_argument("--base-url", default=None, help="openai_compat server URL")
    r.add_argument("--host", default=None, help="Ollama host, e.g. http://localhost:11434")
    r.add_argument("--persona", default="worksheet", help="none, rich, rich_v2, rich_v3, worksheet")
    r.add_argument("--payoff", default="hidden", choices=["hidden", "own", "full"])
    r.add_argument("--grounding", default="rag", choices=["rag", "none"])
    r.add_argument("--legal", default="on", choices=["on", "off"])
    r.add_argument("--events", default="none")
    r.add_argument("--hitl", default="off", choices=["off", "final"])
    r.add_argument("--scenario", default="normal")
    r.add_argument("--rounds", type=int, default=6)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--rag", default=None, choices=["bm25", "chroma"], help="override config/rag.yaml backend")
    r.add_argument("--collection", default=None, help="Chroma collection (local_docs or amu_darya_v2)")
    r.add_argument("--out", default=str(ROOT / "runs" / "single_run.json"))

    s = sub.add_parser("search", help="test the retriever the agents will use")
    s.add_argument("query")
    s.add_argument("--k", type=int, default=4)
    s.add_argument("--rag", default=None, choices=["bm25", "chroma"])
    s.add_argument("--collection", default=None)
    s.add_argument("--doc-type", default=None, help="filter, e.g. legal")

    ix = sub.add_parser("index", help="build the improved Chroma collection from a documents folder")
    ix.add_argument("--docs", required=True, help="the RAG app's documents folder (data/raw)")
    ix.add_argument("--db", required=True, help="the RAG app's chroma_db folder")
    ix.add_argument("--collection", default="amu_darya_v2")
    ix.add_argument("--embed-model", default="nomic-embed-text")
    ix.add_argument("--max-chars", type=int, default=1000)

    e = sub.add_parser("experiment", help="run a factorial experiment from a YAML config")
    e.add_argument("config")
    e.add_argument("--out", default=None)

    bm = sub.add_parser("benchmarks", help="print no deal, Protocol 566, Nash bargaining and utilitarian reference points")
    bm.add_argument("--scenario", default="normal")

    sub.add_parser("worksheets", help="run the Part C checks on config/worksheets")
    sub.add_parser("synth", help="write the synthetic personas to config/worksheets/synthetic")

    g = sub.add_parser("ground", help="RAG truth grounding evaluation of persona worksheets")
    g.add_argument("--backend", default="ollama", choices=["mock", "ollama", "anthropic", "openai_compat"])
    g.add_argument("--model", default="llama3.2")
    g.add_argument("--base-url", default=None)
    g.add_argument("--host", default=None)
    g.add_argument("--personas", default=str(ROOT / "config/worksheets/synthetic"))
    g.add_argument("--docs", default=str(ROOT.parent / "data"), help="documents folder (used only with --rag bm25)")
    g.add_argument("--scenario", default="normal")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--temperature", type=float, default=0.7)
    g.add_argument("--rag", default=None, choices=["bm25", "chroma"])
    g.add_argument("--collection", default=None)
    g.add_argument("--out", default=None)

    rs = sub.add_parser("rescore", help="re-verify a grounding run's claims with citation ids normalised")
    rs.add_argument("run_dir")
    rs.add_argument("--backend", default="ollama")
    rs.add_argument("--model", default="llama3.2")
    rs.add_argument("--out", default=None, help="default: <run_dir>_normalized")

    rp = sub.add_parser("report", help="HTML report of a grounding evaluation run")
    rp.add_argument("run_dir")
    rp.add_argument("--findings", default=str(ROOT / "reports/grounding_findings.yaml"))
    rp.add_argument("--out", default=str(ROOT / "reports/grounding_report.html"))
    rp.add_argument("--strict", default=None, help="the run as generated, when run_dir is a rescored run")

    sv = sub.add_parser("serve", help="HTTP API for the web front end")
    sv.add_argument("--port", type=int, default=8000)

    a = ap.parse_args()
    if a.cmd == "run":
        spec = {"name": a.model or a.backend, "type": a.backend, "model": a.model}
        if a.base_url:
            spec["base_url"] = a.base_url
        if a.host:
            spec["host"] = a.host
        cond = Condition(persona=a.persona, payoff=a.payoff, grounding=a.grounding, legal=a.legal, events=a.events,
                         hitl=a.hitl, scenario=a.scenario, max_rounds=a.rounds)
        res = run_one(spec, cond, a.seed, configs=_configs_with_rag(a))
        print("\n".join(res["public_record"]))
        print("\nAGREEMENT" if res["agreement"] else "\nNO DEAL", "after", res["rounds"], "rounds")
        print(json.dumps(res["metrics"], indent=1, default=float))
        save_transcript(res, a.out)
        print("transcript saved to", a.out)
    elif a.cmd == "search":
        cfg = _configs_with_rag(a)
        ret = make_retriever(cfg["rag"], ROOT)
        where = {"doc_type": a.doc_type} if a.doc_type else None
        for p in ret.search(a.query, a.k, where):
            print(f"\n[{p.id}] score {p.score}  ({p.source}, doc_type={p.meta.get('doc_type', 'n/a')})\n{p.text[:500]}")
    elif a.cmd == "index":
        from .retrieval import ollama_embedder
        stats = build_index(a.docs, a.db, a.collection, ollama_embedder(a.embed_model), a.max_chars)
        print(stats)
    elif a.cmd == "worksheets":
        from . import worksheets
        for wid, issues in worksheets.report(ROOT / "config/worksheets").items():
            print(f"{wid}: {'PASS' if not issues else 'FAIL'}")
            for i in issues:
                print(f"   [{i['check']}] {i['field']}: {i['message']}")
    elif a.cmd == "synth":
        from .synthetic import build
        for p in build(ROOT / "config/worksheets", ROOT / "config/worksheets/synthetic"):
            print("wrote", p.relative_to(ROOT))
    elif a.cmd == "ground":
        import time
        from . import grounding_eval
        from .llm import make_llm
        from .retrieval import ollama_embedder
        cfg = _configs_with_rag(a)
        spec = {"name": a.model or a.backend, "type": a.backend, "model": a.model}
        if a.base_url:
            spec["base_url"] = a.base_url
        if a.host:
            spec["host"] = a.host
        ret = make_retriever(cfg["rag"], ROOT)
        embed = getattr(ret, "embed", None) or ollama_embedder(cfg["rag"]["chroma"].get("embed_model", "nomic-embed-text"))
        out = a.out or ROOT / "runs" / "grounding" / time.strftime("%Y%m%d_%H%M")
        _, head = grounding_eval.run(a.personas, make_llm(spec, a.temperature), ret, embed, a.docs, out, a.scenario, seed=a.seed)
        print(json.dumps(head, indent=1, default=float))
        print("results in", out)
    elif a.cmd == "rescore":
        from .grounding_eval import rescore
        from .llm import make_llm
        cfg = load_configs()
        ret = make_retriever(cfg["rag"], ROOT)
        rescore(a.run_dir, a.out or a.run_dir.rstrip("/\\") + "_normalized",
                make_llm({"type": a.backend, "model": a.model}), ret, ret.embed, ROOT.parent / "data")
    elif a.cmd == "report":
        from .report import render
        print("wrote", render(a.run_dir, a.findings, a.out, a.strict))
    elif a.cmd == "serve":
        import uvicorn
        uvicorn.run("wnego.api:app", host="127.0.0.1", port=a.port)
    elif a.cmd == "experiment":
        run_experiment(a.config, a.out)
    elif a.cmd == "benchmarks":
        b = Basin.load(ROOT / "config/basin.yaml", a.scenario)
        print("NO DEAL\n" + b.describe(b.no_deal_proposal()), "\n", b.no_deal_utilities())
        p = b.protocol_566_proposal()
        print("\nPROTOCOL 566 ANCHOR\n" + b.describe(p), "\n", b.utilities(p))
        for name, fn in [("NASH BARGAINING SOLUTION", nash_bargaining_solution), ("UTILITARIAN OPTIMUM", utilitarian_optimum)]:
            sol = fn(b)
            print(f"\n{name}\n" + b.describe(sol["proposal"]), "\n", sol["utilities"])


if __name__ == "__main__":
    main()
