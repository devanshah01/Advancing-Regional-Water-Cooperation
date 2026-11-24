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
                "Kyrgyzstan": {"growing": 5, "winter": 2}, "Tajikistan": {"growing": 5, "winter": 3},
                "Uzbekistan": {"growing": 14, "winter": 8}, "Turkmenistan": {"growing": 11, "winter": 6}},
                "tajik_storage_shift": 3}, "explanation": "draft", "caucus_replies": {"Uzbekistan": "noted"}, "claims": []})
        if "opening round" in user:
            return 'Here you go:\n{"statement": "hello", "ask": {}, "claims": [{"text": "Qosh Tepa is 285 km", "cite": "qosh_tepa_canal#c0"}], "caucus": "flexible on winter", "lookups": ["Rogun winter"]}'
        if "Verdicts" in system:
            passage = user.split("]: ", 1)[1].split("\n")[0]          # quote the first words of the passage
            return json.dumps({"results": [{"i": 0, "evidence": " ".join(passage.split()[:8]), "verdict": "supported"}]})
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


class RecordingLLM(ScriptedLLM):
    """ScriptedLLM that keeps every prompt, and can be told how the legal advisor rules."""

    def __init__(self, legal_finding="concern", verifier_evidence=None, **kw):
        super().__init__(**kw)
        self.prompts, self.legal_finding, self.verifier_evidence = [], legal_finding, verifier_evidence

    def _complete(self, system, user):
        self.prompts.append((system, user))
        if system.startswith("You are an independent legal advisor"):
            return json.dumps({"assessments": [{"principle": "existing arrangements", "finding": self.legal_finding,
                                                "explanation": "x", "cite": None}], "summary": "ok"})
        if "Verdicts" in system and self.verifier_evidence is not None:
            return json.dumps({"results": [{"i": 0, "evidence": self.verifier_evidence, "verdict": "supported"}]})
        return super()._complete(system, user)


def _negotiate(llm, **cond_kw):
    b = Basin.load(BASIN, "normal")
    cfg = load_configs()
    cond = Condition(**{"persona": "rich", "payoff": "own", "grounding": "rag", "max_rounds": 2, **cond_kw})
    return Negotiation(llm, b, cond, cfg["personas"], Retriever(ROOT / "corpus"), cfg["goals"], cfg["events"],
                       seed=0, parallel=False).run()


def test_every_agent_gets_the_six_step_procedure():
    llm = RecordingLLM()
    _negotiate(llm)
    users = [u for s, u in llm.prompts if "Verdicts" not in s]
    assert users and all("HOW TO DECIDE" in u and '"because"' in u for u in users)
    roles = {"opening round" in u for u in users} | {"Table the single negotiating text" in u for u in users}
    assert roles == {True, False}
    assert any(s.startswith("You are an independent legal advisor") for s, _ in llm.prompts)


def test_free_reasoning_restores_the_old_prompts():
    llm = RecordingLLM()
    _negotiate(llm, reasoning="free")
    assert not any("HOW TO DECIDE" in u for _, u in llm.prompts)
    assert any("your private reasoning, never shown to anyone" in u for _, u in llm.prompts)


def test_payoff_lines_state_the_margin_against_walk_away():
    from wnego.agents import payoff_lines
    b = Basin.load(BASIN, "normal")
    below = payoff_lines(b, b.protocol_566_proposal(), "Tajikistan", "own")   # 545 against 1,248
    assert "BELOW walk-away" in below and "worse than no deal" in below
    nbs = nash_bargaining_solution(b)["proposal"]
    above = payoff_lines(b, nbs, "", "full")
    assert above.count("ABOVE walk-away") == 4


def test_decision_rule_and_binding_note_reach_the_delegates():
    b = Basin.load(BASIN, "normal")
    cfg = load_configs()
    neg = Negotiation(ScriptedLLM(), b, Condition(decision_rule="batna", legal="binding"), cfg["personas"], None,
                      cfg["goals"], cfg["events"], seed=0, parallel=False)
    system = neg.delegates["Uzbekistan"].system
    assert "DECISION RULE" in system and "LEGAL REVIEW IS BINDING" in system
    assert "LEGAL REVIEW IS BINDING" in neg.mediator.system
    plain = Negotiation(ScriptedLLM(), b, Condition(), cfg["personas"], None, cfg["goals"], cfg["events"], seed=0)
    assert "DECISION RULE" not in plain.delegates["Uzbekistan"].system


def test_legal_setting_accepts_old_on_off_values():
    assert Condition(legal="on").legal == "advisory" and Condition(legal=True).legal == "advisory"
    assert Condition(legal=False).legal == "off"
    import pytest
    with pytest.raises(ValueError):
        Condition(legal="sometimes")


def test_binding_legal_review_blocks_an_inconsistent_text():
    res = _negotiate(RecordingLLM(legal_finding="inconsistent"), legal="binding")
    assert not res["agreement"] and len(res["legal_blocks"]) == 2      # unanimous both rounds, blocked both times
    assert any("binding review blocks" in line for line in res["public_record"])
    advisory = _negotiate(RecordingLLM(legal_finding="inconsistent"), legal="advisory")
    assert advisory["agreement"] and advisory["rounds"] == 1           # same votes, advice only
    ok = _negotiate(RecordingLLM(legal_finding="concern"), legal="binding")
    assert ok["agreement"]                                             # a concern does not block


def test_verifier_rejects_a_quote_that_is_not_in_the_passage():
    res = _negotiate(RecordingLLM(verifier_evidence="the canal will carry 99 billion cubic metres"), max_rounds=1)
    cited = [c for c in res["claims"] if c.get("cite") == "qosh_tepa_canal#c0"]
    assert cited and all(c["verdict"] == "unsupported" for c in cited)


def test_structured_reasoning_metrics():
    from wnego.evaluation import evaluate
    b = Basin.load(BASIN, "normal")
    full = {k: "x" for k in ("facts", "rules", "value", "options", "because")}
    res = {"no_deal_utilities": b.no_deal_utilities(), "final_utilities": b.no_deal_utilities(), "votes": [],
           "agreement": False, "rounds": 1, "final_text": b.no_deal_proposal(), "claims": [],
           "events": [{"phase": "response", "output": {"vote": "accept", "reasoning": {**full, "decision": "Accept as is."}}},
                      {"phase": "response", "output": {"vote": "accept", "reasoning": {**full, "decision": "Reject: below walk-away"}}},
                      {"phase": "response", "output": {"vote": "reject", "reasoning": {**full, "decision": ""}}}]}
    m = evaluate(res, b, None)
    assert abs(m["reasoning_complete_rate"] - 2 / 3) < 1e-9
    assert m["reasoning_vote_mismatch_rate"] == 0.5


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
