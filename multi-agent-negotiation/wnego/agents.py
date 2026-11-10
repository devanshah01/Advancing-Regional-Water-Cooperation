"""Agents and tools (hub and spoke).

Agents
  Delegate       one per country (tagged upstream or downstream in its goal file)
  Mediator       the hub: tables the single text, reads every caucus, replies privately
  LegalAdvisor   a spoke with no stake: reviews each draft against water law
  Verifier       checks every cited claim against the cited passage
Tools
  lookup         document retrieval (BM25 or the team's Chroma pipeline)
  simulate       hydrology and feasibility check of a hypothetical text
Agents request tools through fields in their JSON reply ("lookups", "what_if");
the orchestrator runs them and returns results on the agent's next turn. This
works with any model, including small local ones without reliable native
function calling, and every tool call is logged.

Mapping to Guo et al. (IJCAI 2024): profiling = persona (pre defined) + goal
file + data derived briefing; structure = centralized hub; paradigm =
competitive delegates, cooperative mediator and advisor; feedback = environment
(simulator, feasibility) and agents (votes, legal review); memory = per agent
caucus thread and position history.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

from . import prompts as P
from .environment import SEASONS, Basin
from .goals import GoalBook
from .llm import LLM
from .retrieval import BaseRetriever, Passage, tokenize


@dataclass
class Condition:
    persona: str = "rich"        # key in personas.yaml
    payoff: str = "hidden"       # hidden | own | full
    grounding: str = "rag"       # rag | none
    scenario: str = "normal"
    legal: str = "on"            # on | off  (legal advisor present)
    events: str = "none"         # key in events.yaml
    hitl: str = "off"            # off | final  (human approves a unanimous text)
    max_rounds: int = 6
    briefing_size: int = 8
    mediator_sees_payoffs: bool = False

    def key(self) -> str:
        return (f"persona={self.persona}|payoff={self.payoff}|grounding={self.grounding}|legal={self.legal}"
                f"|events={self.events}|scenario={self.scenario}")


def persona_text(personas: dict, variant: str, country: str) -> str:
    table = personas[variant]
    return table.get(country, table.get("_all", "")).format(country=country).strip()


def payoff_lines(basin: Basin, proposal: dict, country: str, mode: str) -> str:
    if mode == "hidden":
        return ""
    u, d = basin.utilities(proposal), basin.no_deal_utilities()
    who = [country] if mode == "own" else basin.countries
    return "Value (USD million equivalent):\n" + "\n".join(f"  {c}: this text {u[c]:.0f}, no deal {d[c]:.0f}" for c in who)


# ---------------------------------------------------------------------------
class Toolbox:
    def __init__(self, basin: Basin, retriever: BaseRetriever | None, cond: Condition):
        self.basin, self.retriever, self.cond = basin, retriever, cond
        self.log: list[dict] = []

    def lookup(self, agent: str, queries: list[str], round_no: int, where: dict | None = None) -> list[Passage]:
        if not self.retriever or self.cond.grounding != "rag":
            return []
        queries = [q for q in (queries or []) if isinstance(q, str) and q.strip()][:2]
        hits = self.retriever.multi_search(queries, k_each=2, limit=4, where=where) if queries else []
        if queries:
            self.log.append({"round": round_no, "agent": agent, "tool": "lookup", "queries": queries,
                             "results": [p.id for p in hits]})
        return hits

    def simulate(self, agent: str, proposal: dict, round_no: int) -> str:
        """Hydrology check of a hypothetical text. Currently a seasonal water
        ledger; replace Basin with a node by node river model to report flow at
        each gauge without changing any agent code."""
        b = self.basin
        p = b.normalize(proposal)
        errors = b.validate(p)
        lines = [f"What if result (round {round_no}):", "Feasible." if not errors else "NOT feasible: " + "; ".join(errors),
                 b.describe(p)]
        pay = payoff_lines(b, p, agent, self.cond.payoff)
        if pay:
            lines.append(pay)
        self.log.append({"round": round_no, "agent": agent, "tool": "simulate", "proposal": p, "feasible": not errors})
        return "\n".join(lines)


# ---------------------------------------------------------------------------
class Delegate:
    def __init__(self, country: str, llm: LLM, basin: Basin, cond: Condition, personas: dict,
                 retriever: BaseRetriever | None, goals: GoalBook, tools: Toolbox, rng):
        self.country, self.llm, self.basin, self.cond, self.goals, self.tools, self.rng = \
            country, llm, basin, cond, goals, tools, rng
        self.briefing: list[Passage] = []
        if cond.grounding == "rag" and retriever is not None:
            g = goals.goals.get(country, {})
            self.briefing = retriever.multi_search(
                [f"{country} Amu Darya water use {g.get('bloc', '')}"]
                + [f"{country} {o}" for o in g.get("objectives", [])[:2]]
                + ["Protocol 566 allocation shares", "Aral Sea delta inflow", "seasonal storage winter release irrigation"],
                k_each=2, limit=cond.briefing_size)
        payoff_rule = {"hidden": P.PAYOFF_HIDDEN, "own": P.PAYOFF_OWN, "full": P.PAYOFF_FULL}[cond.payoff]
        whatif_payoff = {"hidden": "", "own": ", and its value to your country",
                         "full": ", and its value to every delegation"}[cond.payoff]
        self.system = "\n\n".join([
            persona_text(personas, cond.persona, country),
            P.DELEGATE_RULES.format(max_rounds=cond.max_rounds, scenario=cond.scenario,
                                    max_shift=basin.cfg["max_storage_shift"], whatif_payoff=whatif_payoff,
                                    countries=", ".join(basin.countries)),
            P.GROUNDING_RAG if cond.grounding == "rag" else P.GROUNDING_NONE,
            payoff_rule,
        ])
        self.caucus_thread: list[str] = []      # private channel with the mediator
        self.found: list[Passage] = []          # passages from lookups
        self.tool_results: list[str] = []       # simulator reports waiting to be read
        self.memory: list[str] = []

    @property
    def known_passages(self) -> list[Passage]:
        return self.briefing + self.found

    def _context(self, round_no: int, public_record: list[str]) -> str:
        parts = [self.goals.render(self.country, round_no)]
        if self.known_passages:
            parts.append("DOCUMENT PASSAGES\n" + "\n".join(p.cite_block() for p in self.known_passages[-12:]))
        if self.tool_results:
            parts.append("TOOL RESULTS\n" + "\n\n".join(self.tool_results))
            self.tool_results = []
        if public_record:
            parts.append("PUBLIC RECORD (most recent last)\n" + "\n".join(public_record[-12:]))
        if self.caucus_thread:
            parts.append("CONFIDENTIAL CAUCUS WITH THE MEDIATOR\n" + "\n".join(self.caucus_thread[-6:]))
        if self.memory:
            parts.append("YOUR PREVIOUS POSITIONS\n" + "\n".join(self.memory[-4:]))
        return "\n\n".join(parts)

    def opening(self, public_record) -> dict:
        user = self._context(0, public_record) + "\n\n" + P.OPENING_SCHEMA
        out = self.llm.complete_json(self.system, user, meta={"mock": self._mock_opening})
        self.memory.append(f"Opening ask: {out.get('ask', {})}")
        return out

    def respond(self, text: dict, round_no: int, public_record) -> dict:
        user = "\n\n".join(x for x in [
            self._context(round_no, public_record),
            f"ROUND {round_no} of {self.cond.max_rounds}. CURRENT SINGLE NEGOTIATING TEXT\n{self.basin.describe(text)}",
            payoff_lines(self.basin, text, self.country, self.cond.payoff),
            P.DELEGATE_SCHEMA] if x)
        out = self.llm.complete_json(self.system, user, meta={"mock": lambda: self._mock_respond(text, round_no)})
        out["vote"] = "accept" if str(out.get("vote", "")).lower().startswith("acc") else "reject"
        self.memory.append(f"Round {round_no}: voted {out['vote']}; amendments {out.get('amendments', {})}")
        return out

    def run_tools(self, out: dict, round_no: int):
        hits = self.tools.lookup(self.country, out.get("lookups") or [], round_no)
        known = {p.id for p in self.known_passages}
        self.found += [p for p in hits if p.id not in known]
        if isinstance(out.get("what_if"), dict) and out["what_if"]:
            self.tool_results.append(self.tools.simulate(self.country, out["what_if"], round_no))
        self.goals.log_request(self.country, round_no, out.get("goal_revision_request") or "")

    # mock policies: plumbing tests only, they ignore persona and framing ----
    def _mock_claims(self):
        if self.briefing:
            p = self.briefing[0]
            return [{"text": " ".join(p.text.split()[:18]), "cite": p.id}]
        return []

    def _mock_opening(self):
        par = self.basin.cfg["countries"][self.country]
        ref = self.basin.cfg["protocol_566_km3"][self.country]
        upstream = str(par.get("position", "")).startswith("upstream")
        bump = 1.3 if upstream else 1.05
        # generalized Qosh-Tepa-style aggressive ask: any country with a unilateral-take
        # overhang beyond its historical Protocol 566 share opens asking for that overhang too
        take = par.get("unilateral_take") or {}
        extra = max(0.0, sum(take.values()) - ref)
        ask = {"allocations": {self.country: {"growing": round(0.7 * ref * bump + extra, 2),
                                              "winter": round(0.3 * ref * bump, 2)}}}
        if self.country == self.basin.cfg["storage_country"]:
            ask["tajik_storage_shift"] = self.basin.cfg["max_storage_shift"]
        return {"reasoning": "mock", "statement": f"{self.country} opening position (mock).", "ask": ask,
                "caucus": f"{self.country} could show flexibility on winter water (mock).",
                "lookups": [f"{self.country} irrigation needs"], "claims": self._mock_claims()}

    def _mock_respond(self, text, round_no):
        u = self.basin.utilities(text)[self.country]
        d = self.basin.no_deal_utilities()[self.country]
        margin = 80 * max(0.0, 1 - round_no / self.cond.max_rounds) + self.rng.normal(0, 10)
        if u >= d + margin:
            return {"reasoning": f"mock: u={u:.0f} d={d:.0f}", "vote": "accept", "statement": "We can accept.",
                    "caucus": "", "amendments": {}, "lookups": [], "what_if": None, "claims": self._mock_claims()}
        cur = self.basin.normalize(text)
        amend = {"allocations": {self.country: {"growing": round(cur["allocations"][self.country]["growing"] + 1.0, 2),
                                                "winter": cur["allocations"][self.country]["winter"]}}}
        if self.country == self.basin.cfg["storage_country"]:
            amend["tajik_storage_shift"] = min(self.basin.cfg["max_storage_shift"], cur["tajik_storage_shift"] + 1.0)
        return {"reasoning": f"mock: u={u:.0f} d={d:.0f}", "vote": "reject", "statement": "We need more growing season water.",
                "caucus": "We could accept with one more km3 in the growing season (mock).", "amendments": amend,
                "lookups": [], "what_if": {**cur, "allocations": {**cur["allocations"], **amend["allocations"]}},
                "claims": self._mock_claims()}


# ---------------------------------------------------------------------------
class Mediator:
    def __init__(self, llm: LLM, basin: Basin, cond: Condition, retriever: BaseRetriever | None):
        self.llm, self.basin, self.cond = llm, basin, cond
        self.briefing = retriever.multi_search(
            ["Amu Darya allocation agreement ICWC", "Aral delta inflow", "seasonal storage winter release",
             "Qosh Tepa diversion", "energy for water exchange"], limit=cond.briefing_size) \
            if (retriever and cond.grounding == "rag") else []
        f, env = basin.base_flow, basin.cfg["environmental_flow"]
        evidence = ("EVIDENCE: cite passage ids for factual claims.\nDOCUMENT PASSAGES\n"
                    + "\n".join(p.cite_block() for p in self.briefing)) if self.briefing else ""
        self.system = P.MEDIATOR_SYSTEM.format(scenario=cond.scenario, g=f["growing"], w=f["winter"],
                                               max_shift=basin.cfg["max_storage_shift"], eg=env["growing"],
                                               ew=env["winter"], evidence=evidence)

    def draft(self, round_no, public_record, caucus_in: dict, asks=None, last_text=None, responses=None,
              legal=None, errors=None) -> dict:
        parts = ["PUBLIC RECORD\n" + "\n".join(public_record[-14:])]
        if caucus_in:
            parts.append("CONFIDENTIAL CAUCUS MESSAGES (do not reveal)\n" + "\n".join(f"{c}: {m}" for c, m in caucus_in.items() if m))
        if last_text:
            parts.append("PREVIOUS TEXT\n" + self.basin.describe(last_text))
            if self.cond.mediator_sees_payoffs:
                parts.append(payoff_lines(self.basin, last_text, "", "full"))
        if legal:
            parts.append("LEGAL ADVISOR'S REVIEW OF THE PREVIOUS TEXT\n" + legal_text(legal))
        if responses:
            parts.append("VOTES AND AMENDMENTS\n" + "\n".join(
                f"{c}: {r['vote']}; amendments {r.get('amendments', {})}" for c, r in responses.items()))
        if asks:
            parts.append("OPENING ASKS\n" + "\n".join(f"{c}: {a}" for c, a in asks.items()))
        if errors:
            parts.append("YOUR LAST DRAFT FAILED THE HYDROLOGY CHECK. FIX THESE PROBLEMS:\n" + "\n".join(errors))
        parts.append(f"Table the single negotiating text for round {round_no}.\n" + P.MEDIATOR_SCHEMA)
        return self.llm.complete_json(self.system, "\n\n".join(parts),
                                      meta={"mock": lambda: self._mock_draft(asks, last_text, responses)})

    def _mock_draft(self, asks, last_text, responses):
        b = self.basin
        if last_text is None:
            base = b.protocol_566_proposal()
            for c, a in (asks or {}).items():
                for s in SEASONS:
                    v = ((a or {}).get("allocations") or {}).get(c, {}).get(s)
                    if v is not None:
                        base["allocations"][c][s] = float(v)
            base["tajik_storage_shift"] = b.cfg["max_storage_shift"] / 2
        else:
            base = copy.deepcopy(b.normalize(last_text))
            rejecters = [c for c, r in (responses or {}).items() if r["vote"] == "reject"]
            accepters = [c for c in b.countries if c not in rejecters]
            for c in rejecters:
                r = responses[c]
                am = b.normalize(r.get("amendments") or {})
                want = ((r.get("amendments") or {}).get("allocations") or {}).get(c, {})
                for s in SEASONS:
                    if s in want:
                        step = 0.5 * (am["allocations"][c][s] - base["allocations"][c][s])
                        base["allocations"][c][s] += step
                        for a in accepters:
                            base["allocations"][a][s] = max(0.0, base["allocations"][a][s] - step / max(len(accepters), 1))
                if "tajik_storage_shift" in (r.get("amendments") or {}):
                    base["tajik_storage_shift"] += 0.5 * (am["tajik_storage_shift"] - base["tajik_storage_shift"])
                for a in accepters:
                    base["transfers"].append({"from": a, "to": c, "amount": 25, "for": "mock compensation"})
        return {"reasoning": "mock", "proposal": fit_to_constraints(b, base), "explanation": "Mock mediator text.",
                "caucus_replies": {c: "Noted, I will reflect this where I can (mock)." for c in b.countries}, "claims": []}


def fit_to_constraints(b: Basin, prop: dict) -> dict:
    """Scale allocations down proportionally so each season fits availability minus delta flow."""
    p = b.normalize(prop)
    p["tajik_storage_shift"] = min(max(p["tajik_storage_shift"], 0.0), b.cfg["max_storage_shift"])
    avail, env = b.available(p["tajik_storage_shift"]), b.cfg["environmental_flow"]
    for s in SEASONS:
        cap = avail[s] - env[s]
        tot = sum(p["allocations"][c][s] for c in b.countries)
        if tot > cap > 0:
            for c in b.countries:
                p["allocations"][c][s] = round(p["allocations"][c][s] * cap / tot, 3)
    return p


# ---------------------------------------------------------------------------
def legal_text(review: dict) -> str:
    rows = [f"- {a.get('principle')}: {a.get('finding')}. {a.get('explanation', '')}"
            + (f" [{a['cite']}]" if a.get("cite") else "") for a in review.get("assessments", [])]
    return review.get("summary", "") + ("\n" + "\n".join(rows) if rows else "")


class LegalAdvisor:
    def __init__(self, llm: LLM, basin: Basin, cond: Condition, retriever: BaseRetriever | None):
        self.llm, self.basin = llm, basin
        self.briefing = retriever.multi_search(
            ["equitable and reasonable utilization factors", "obligation not to cause significant harm",
             "notification planned measures", "protection of ecosystems", "Protocol 566 Almaty Agreement ICWC parties"],
            k_each=2, limit=10, where={"doc_type": "legal"}) if (retriever and cond.grounding == "rag") else []

    def review(self, text: dict, round_no: int) -> dict:
        user = "\n\n".join(x for x in [
            ("LEGAL PASSAGES\n" + "\n".join(p.cite_block() for p in self.briefing)) if self.briefing
            else "No legal documents are available; say where your findings rest on general principles only.",
            f"DRAFT TEXT, ROUND {round_no}\n{self.basin.describe(text)}",
            P.LEGAL_SCHEMA] if x)
        return self.llm.complete_json(P.LEGAL_SYSTEM, user, meta={"mock": lambda: self._mock(text)})

    def _mock(self, text):
        d = self.basin.delta_flow(self.basin.normalize(text))
        env = self.basin.cfg["environmental_flow"]
        ok = all(d[s] >= env[s] - 1e-6 for s in SEASONS)
        cite = self.briefing[0].id if self.briefing else None
        return {"assessments": [{"principle": "protection of ecosystems", "finding": "consistent" if ok else "concern",
                                 "explanation": "Delta minimum flow check (mock).", "cite": cite}],
                "summary": "Mock legal review."}


# ---------------------------------------------------------------------------
class Verifier:
    """Checks every cited claim against the cited passage before it enters the
    shared record, so one agent's hallucination does not cascade to the others."""

    def __init__(self, llm: LLM, retriever: BaseRetriever | None, normalize_ids: bool = True):
        self.llm, self.retriever, self.normalize_ids = llm, retriever, normalize_ids

    def resolve(self, cite: str) -> tuple[str, object]:
        """Map a cited id to a passage. Small models copy the brackets passages are
        shown in ("[info#c2]") or append a suffix ("Limits#p1#c0#c0"); with
        normalize_ids those resolve to the passage they start with. Ids that match
        no passage stay unresolved."""
        p = self.retriever.get(cite)
        if p is not None or not self.normalize_ids:
            return cite, p
        cid = cite.strip().strip("[]").strip()
        while cid:
            p = self.retriever.get(cid)
            if p is not None:
                return cid, p
            if cid.count("#") < 2:
                break
            cid = cid.rsplit("#", 1)[0]
        return cite, None

    def check(self, claims: list[dict]) -> list[dict]:
        out, judge, passages = [], [], {}
        for c in claims or []:
            if not isinstance(c, dict):
                continue
            cite = c.get("cite")
            if isinstance(cite, list):          # small models sometimes cite several ids; judge the first
                cite = next((x for x in cite if isinstance(x, str) and x.strip()), None)
            elif cite is not None and not isinstance(cite, str):
                cite = str(cite)
            p = None
            if cite and self.retriever:
                resolved, p = self.resolve(cite)
                if resolved != cite:
                    c = {**c, "cite_raw": cite}
                cite = resolved
            c = {**c, "cite": cite}
            if not cite or self.retriever is None:
                out.append({**c, "verdict": "uncited"})
            elif p is None:
                out.append({**c, "verdict": "bad_id"})
            else:
                out.append(dict(c))
                judge.append(len(out) - 1)
                passages[len(out) - 1] = p
        if not judge:
            return out
        listing = "\n\n".join(f"Claim {k}: {out[i].get('text', '')}\nPassage [{out[i]['cite']}]: {passages[i].text}"
                              for k, i in enumerate(judge))
        res = self.llm.complete_json(P.VERIFIER_SYSTEM, listing + "\n\n" + P.VERIFIER_SCHEMA,
                                     meta={"mock": lambda: self._mock(out, judge, passages)})
        verdicts = {r.get("i"): r.get("verdict", "unsupported") for r in res.get("results", []) if isinstance(r, dict)}
        for k, i in enumerate(judge):
            out[i]["verdict"] = verdicts.get(k, "unsupported")
        return out

    @staticmethod
    def _mock(out, judge, passages):
        res = []
        for k, i in enumerate(judge):
            claim, passage = set(tokenize(out[i].get("text", ""))), set(tokenize(passages[i].text))
            overlap = len(claim & passage) / max(len(claim), 1)
            res.append({"i": k, "verdict": "supported" if overlap > 0.7 else "partial" if overlap > 0.4 else "unsupported"})
        return {"results": res}
