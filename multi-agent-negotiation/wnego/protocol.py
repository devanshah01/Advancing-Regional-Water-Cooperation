"""Orchestrator for one episode (hub and spoke).

Round 0   delegates give opening statements and asks; caucus notes go to the mediator
Round r   scenario controller fires due events (public notice + goal file updates)
          mediator tables the single text (hydrology check, up to 2 repairs)
          mediator answers each caucus privately
          legal advisor reviews the text (if on); review is public
          delegates vote, amend, caucus, and request tools, all simultaneously
          verifier checks cited claims before they enter the public record
          tool requests run (lookups, what_if); results reach the agent next turn
          unanimous accept -> human approval gate (if on) -> deal
After max_rounds without an approved unanimous text, no deal applies.
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import numpy as np

from .agents import Condition, Delegate, LegalAdvisor, Mediator, Toolbox, Verifier, fit_to_constraints, legal_text
from .environment import Basin
from .goals import GoalBook
from .llm import LLM
from .retrieval import BaseRetriever


def _fmt_claims(checked: list[dict]) -> str:
    if not checked:
        return ""
    bits = []
    for c in checked:
        tag = {"supported": "verified", "partial": "partly verified"}.get(c.get("verdict"), "UNVERIFIED")
        cite = f" [{c['cite']}]" if c.get("cite") else ""
        bits.append(f"{c.get('text', '')}{cite} ({tag})")
    return " Claims: " + "; ".join(bits)


def console_approver(text_desc: str, record: list[str]) -> tuple[bool, str]:
    print("\n=== HUMAN APPROVAL GATE ===\nAll four delegations accepted:\n" + text_desc)
    ans = input("Approve this text? [y/N] ").strip().lower()
    note = "" if ans.startswith("y") else input("Note for the delegations (why not): ").strip()
    return ans.startswith("y"), note


class Negotiation:
    def __init__(self, llm: LLM, basin: Basin, cond: Condition, personas: dict, retriever: BaseRetriever | None,
                 goals: dict, events: dict | None = None, seed: int = 0, parallel: bool = True,
                 approver: Callable | None = None, verifier_llm: LLM | None = None):
        self.llm, self.basin, self.cond, self.seed, self.parallel = llm, basin, cond, seed, parallel
        rng = np.random.default_rng(seed)
        self.order = [str(c) for c in rng.permutation(basin.countries)]       # logged for order effects
        child = rng.spawn(len(self.order))
        self.goals = GoalBook(goals, (events or {}).get(cond.events, []))
        self.tools = Toolbox(basin, retriever, cond)
        self.delegates = {c: Delegate(c, llm, basin, cond, personas, retriever, self.goals, self.tools, child[i])
                          for i, c in enumerate(self.order)}
        self.mediator = Mediator(llm, basin, cond, retriever)
        self.legal = LegalAdvisor(llm, basin, cond, retriever) if cond.legal == "on" else None
        # Verifier can run on a separate (typically colder) LLM instance: the pilot found it
        # judged the same 7 claims twice and agreed with itself only 3 times, and the README's
        # own next step was to "run the verifier at temperature 0 or on a stronger model."
        self.verifier = Verifier(verifier_llm or llm, retriever)
        self.approver = approver or (console_approver if cond.hitl == "final" else None)
        self.public: list[str] = []
        self.events: list[dict] = []
        self.caucus_log: list[dict] = []

    def _log(self, **e):
        e["t"] = round(time.time(), 2)
        self.events.append(e)

    def _map(self, fn, items):
        if self.parallel:
            with ThreadPoolExecutor(max_workers=len(items)) as ex:
                return dict(zip(items, ex.map(fn, items)))
        return {i: fn(i) for i in items}

    def _after_moves(self, round_no: int, outs: dict, with_vote: bool) -> dict:
        """Verify claims, publish statements, route caucus notes, run tool requests."""
        caucus_in = {}
        for c in self.order:
            o = outs[c]
            o["claims_checked"] = self.verifier.check(o.get("claims") or [])
            vote = f" Vote: {o['vote']}." if with_vote else ""
            self.public.append(f"[Round {round_no}] {c}:{vote} {o.get('statement', '')}{_fmt_claims(o['claims_checked'])}")
            msg = (o.get("caucus") or "").strip()
            if msg:
                caucus_in[c] = msg
                self.delegates[c].caucus_thread.append(f"[Round {round_no}] you to mediator: {msg}")
                self.caucus_log.append({"round": round_no, "from": c, "to": "Mediator", "text": msg})
            self.delegates[c].run_tools(o, round_no)
        return caucus_in

    def run(self) -> dict:
        b, cond = self.basin, self.cond
        d = b.no_deal_utilities()

        snap = list(self.public)
        openings = self._map(lambda c: self.delegates[c].opening(snap), self.order)
        caucus_in = self._after_moves(0, openings, with_vote=False)
        asks = {c: o.get("ask", {}) for c, o in openings.items()}
        for c, o in openings.items():
            self._log(round=0, phase="opening", agent=c, output=o)

        text, responses, review, agreement, votes, goal_trace, hitl = None, None, None, False, [], [], []
        for r in range(1, cond.max_rounds + 1):
            for ev in self.goals.due(r):
                changes = self.goals.apply(ev, r)
                self.public.append(f"[Round {r}] NEWS: {ev.get('name')}. {ev.get('public', '').strip()}")
                self._log(round=r, phase="event", agent="ScenarioController", event=ev.get("name"), changes=changes)
            goal_trace.append({"round": r, **{c: v["version"] for c, v in self.goals.snapshot().items()}})

            errors = None
            for attempt in range(3):
                m = self.mediator.draft(r, self.public, caucus_in, asks if r == 1 else None, text, responses, review, errors)
                cand = b.normalize(m.get("proposal") or {})
                errors = b.validate(cand)
                self._log(round=r, phase="mediator_draft", agent="Mediator", attempt=attempt, output=m, errors=errors)
                if not errors:
                    break
            if errors:
                cand = fit_to_constraints(b, cand)
                self._log(round=r, phase="auto_repair", agent="Environment", errors=errors)
            text = cand
            med_claims = self.verifier.check(m.get("claims") or [])
            self.public.append(f"[Round {r}] Mediator tables a text: {m.get('explanation', '')}{_fmt_claims(med_claims)}")
            for c, msg in (m.get("caucus_replies") or {}).items():
                if c in self.delegates and msg:
                    self.delegates[c].caucus_thread.append(f"[Round {r}] mediator to you: {msg}")
                    self.caucus_log.append({"round": r, "from": "Mediator", "to": c, "text": msg})

            review = None
            if self.legal:
                review = self.legal.review(text, r)
                self.public.append(f"[Round {r}] Legal advisor: {legal_text(review)}")
                self._log(round=r, phase="legal_review", agent="LegalAdvisor", output=review)

            u = b.utilities(text)
            self._log(round=r, phase="text", agent="Environment", text=text, utilities=u)
            snap = list(self.public)
            responses = self._map(lambda c: self.delegates[c].respond(text, r, snap), self.order)
            caucus_in = self._after_moves(r, responses, with_vote=True)
            for c, o in responses.items():
                votes.append({"round": r, "country": c, "vote": o["vote"], "utility": u[c], "no_deal": d[c],
                              "goal_version": self.goals.version[c]})
                self._log(round=r, phase="response", agent=c, output=o)

            if all(o["vote"] == "accept" for o in responses.values()):
                if self.approver:
                    ok, note = self.approver(b.describe(text), self.public)
                    hitl.append({"round": r, "approved": ok, "note": note})
                    if not ok:
                        self.public.append(f"[Round {r}] Human reviewer did not approve the text. {note}")
                        continue
                agreement = True
                break

        final = text if agreement else b.no_deal_proposal()
        final_u = b.utilities(final) if agreement else d
        claims = [c for e in self.events if e["phase"] in ("opening", "response")
                  for c in e["output"].get("claims_checked", [])]
        reviews = [e["output"] for e in self.events if e["phase"] == "legal_review"]
        return {
            "condition": cond.__dict__, "seed": self.seed, "speaking_order": self.order,
            "agreement": agreement, "rounds": r, "final_text": final, "final_utilities": final_u,
            "no_deal_utilities": d, "last_text": text, "votes": votes, "claims": claims,
            "legal_reviews": reviews, "caucus": self.caucus_log, "tool_calls": self.tools.log,
            "goal_history": self.goals.history, "goal_versions_by_round": goal_trace,
            "goal_revision_requests": self.goals.requests, "human_approvals": hitl,
            "public_record": self.public, "events": self.events, "llm_calls": self.llm.calls,
        }


def save_transcript(result: dict, path):
    with open(path, "w") as f:
        json.dump(result, f, indent=1, default=str)
