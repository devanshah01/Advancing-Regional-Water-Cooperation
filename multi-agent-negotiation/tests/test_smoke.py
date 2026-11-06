"""Offline tests. Run with: python -m pytest -q"""
import json

import yaml

from wnego.agents import Condition
from wnego.benchmarks import nash_bargaining_solution
from wnego.environment import Basin
from wnego.experiment import ROOT, load_configs, run_one
from wnego.llm import LLM, extract_json
from wnego.protocol import Negotiation
from wnego.retrieval import BM25Retriever as Retriever

BASIN = ROOT / "config/basin.yaml"


def test_extract_json_handles_fences_and_prose():
    assert extract_json('Sure.\n```json\n{"a": {"b": "}"}}\n```') == {"a": {"b": "}"}}
    assert extract_json('prefix {"vote": "accept"} suffix') == {"vote": "accept"}


def test_no_deal_is_feasible_and_nbs_beats_it():
    for sc in ("normal", "dry", "wet"):
        b = Basin.load(BASIN, sc)
        assert not b.validate(b.no_deal_proposal())
        nbs = nash_bargaining_solution(b)
        d = b.no_deal_utilities()
        assert all(nbs["utilities"][c] > d[c] for c in b.countries)
        assert not b.validate(nbs["proposal"])


def test_retriever_returns_cited_passages():
    r = Retriever(ROOT / "corpus")
    hits = r.search("Qosh Tepa canal diversion", 3)
    assert hits and hits[0].id.startswith("qosh_tepa_canal#")
    assert r.get(hits[0].id).text == hits[0].text


def test_mock_episode_all_conditions():
    for persona in ("none", "rich", "rich_v2"):
        for payoff in ("hidden", "own", "full"):
            for grounding, legal, events in (("rag", "on", "qosh_tepa_shock"), ("none", "off", "none")):
                res = run_one({"name": "mock", "type": "mock"},
                              Condition(persona=persona, payoff=payoff, grounding=grounding, legal=legal,
                                        events=events, max_rounds=4), seed=1)
                assert "agreement" in res["metrics"]
                json.dumps(res, default=str)


class ScriptedLLM(LLM):
    """Returns canned text replies to exercise the real prompt and parsing path."""
    name = "scripted"

    def _complete(self, system, user):
        if system.startswith("You are an independent legal advisor"):
            return '{"assessments": [{"principle": "no significant harm", "finding": "concern", "explanation": "x", "cite": null}], "summary": "ok"}'
        if "Table the single negotiating text" in user:
            return json.dumps({"proposal": {"allocations": {
                "Afghanistan": {"growing": 5, "winter": 2}, "Tajikistan": {"growing": 5, "winter": 3},
                "Uzbekistan": {"growing": 14, "winter": 8}, "Turkmenistan": {"growing": 11, "winter": 6}},
                "tajik_storage_shift": 3}, "explanation": "draft", "caucus_replies": {"Uzbekistan": "noted"}, "claims": []})
        if "opening round" in user:
            return 'Here you go:\n{"statement": "hello", "ask": {}, "claims": [{"text": "Qosh Tepa is 285 km", "cite": "qosh_tepa_canal#c0"}], "caucus": "flexible on winter", "lookups": ["Rogun winter"]}'
        if "Verdicts" in system:
            return '{"results": [{"i": 0, "verdict": "supported"}]}'
        return '{"vote": "accept", "statement": "ok", "what_if": {"tajik_storage_shift": 2}, "claims": [{"text": "made up", "cite": "nonexistent#9"}]}'


def test_text_llm_path_and_prompts():
    b = Basin.load(BASIN, "normal")
    cfg = load_configs()
    cond = Condition(persona="rich", payoff="own", grounding="rag", max_rounds=3)
    neg = Negotiation(ScriptedLLM(), b, cond, cfg["personas"], Retriever(ROOT / "corpus"), cfg["goals"], cfg["events"],
                      seed=0, parallel=False)
    res = neg.run()
    assert res["agreement"] and res["rounds"] == 1
    verdicts = {c["verdict"] for c in res["claims"]}
    assert "bad_id" in verdicts and "supported" in verdicts
    assert any("UNVERIFIED" in line for line in res["public_record"])
    assert "DOCUMENT PASSAGES" not in neg.delegates["Uzbekistan"].system   # passages and goals go in the user turn
    from wnego.evaluation import evaluate
    m = evaluate(res, b, None)
    assert m["legal_concerns"] == 1
    assert m["what_ifs"] == 4 and m["lookups"] == 4
    assert any(c["to"] == "Uzbekistan" for c in res["caucus"])


def test_human_gate_can_block_a_deal():
    calls = []
    def reject_then_approve(desc, record):
        calls.append(desc)
        return (len(calls) > 1, "needs delta flow guarantee")
    b = Basin.load(BASIN, "normal")
    cfg = load_configs()
    cond = Condition(max_rounds=3, hitl="final")
    res = Negotiation(ScriptedLLM(), b, cond, cfg["personas"], Retriever(ROOT / "corpus"), cfg["goals"], cfg["events"],
                      seed=0, parallel=False, approver=reject_then_approve).run()
    assert res["agreement"] and res["rounds"] == 2
    assert res["human_approvals"][0]["approved"] is False
