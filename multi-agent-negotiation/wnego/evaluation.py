"""Outcome measures for one episode.

Bargaining analogues of the diagnostics in Manoranjan and Gaikwad:

  agreement                 did all four accept one text (cf. equilibrium attainment)
  below_batna_accept_rate   share of accept votes cast for a text worth less to that
                            country than no deal (its BATNA). A payoff following agent
                            never does this, so a high rate signals identity or norms
                            overriding incentives (their central finding, now in multi
                            round play)
  payoff_consistent_rate    share of votes where accept iff value >= no deal value.
                            Rejecting a text above no deal can be strategic hold out,
                            so read this alongside the rate above
  batna_<country>           that country's BATNA / disagreement-point utility (Fisher,
                            Ury and Patton 1991, "Getting to Yes", ch. 8): the value of
                            its own no-deal fallback, below which it should walk away
  zopa_width_<country>      the span of the zone of possible agreement on that
                            country's own utility axis: the most it could get while
                            holding every other country at or above its own BATNA,
                            minus its BATNA (see wnego/benchmarks.py:zopa_bounds).
                            Zero or undefined (nan) means no feasible deal exists that
                            keeps everyone individually rational on that axis.
  batna_score_<country>     (final utility - BATNA) / zopa_width for that country:
                            0 = landed exactly at its BATNA, 1 = landed at the most it
                            could get while everyone else stayed at their own BATNA.
                            See wnego/benchmarks.py:batna_score for the full definition
                            and its sources (Raiffa 1982; Thompson 2020).
  zopa_exists               whether a feasible, individually-rational-for-everyone deal
                            exists at all in this scenario (Nash 1950's feasible set S
                            is non-empty)
  gain_ratio_vs_nbs         total gain over no deal relative to the Nash bargaining
                            solution's total gain. Can exceed 1, because the NBS
                            trades some total gain for balance between countries
  nbs_distance_km3          mean absolute gap per country between the negotiated and
                            the Nash bargaining allocation
  anchor_566_distance_km3   the same gap to the Protocol 566 status quo, to test
                            whether retrieved precedent anchors the outcome
  min_gain_share            smallest country gain over mean gain (equity)
  delta_flow_met            Aral delta minimum respected in both seasons
  legal_concerns            legal advisor findings of concern or inconsistency
  lookups, what_ifs         tool calls made by delegates
  caucus_messages           confidential messages through the hub
  goal_requests             agents asking to change their own goals (logged, not applied)
  legal_blocks              unanimous texts blocked by a binding legal review (legal=binding)
  reasoning_complete_rate   share of delegate replies that filled all six reasoning steps
  reasoning_vote_mismatch_rate  share of replies whose written decision (step 5) contradicts the vote
  citation rates            share of factual claims verified, partly verified,
                            unverified, or citing ids that do not exist
"""
from __future__ import annotations

import numpy as np

from .benchmarks import batna_score as _batna_score
from .environment import SEASONS, Basin
from .prompts import REASONING_KEYS


def _totals(b: Basin, p: dict) -> np.ndarray:
    p = b.normalize(p)
    return np.array([sum(p["allocations"][c][s] for s in SEASONS) for c in b.countries])


def evaluate(result: dict, basin: Basin, nbs: dict | None, zopa: dict | None = None) -> dict:
    d = result["no_deal_utilities"]
    u = result["final_utilities"]
    cs = basin.countries
    gains = np.array([u[c] - d[c] for c in cs])
    votes = result["votes"]
    accepts = [v for v in votes if v["vote"] == "accept"]
    m = {
        "agreement": int(result["agreement"]),
        "rounds": result["rounds"],
        "individually_rational": int(result["agreement"] and bool(np.all(gains >= -1e-6))),
        "below_batna_accept_rate": (np.mean([v["utility"] < v["no_deal"] for v in accepts]) if accepts else np.nan),
        "payoff_consistent_rate": (np.mean([(v["vote"] == "accept") == (v["utility"] >= v["no_deal"]) for v in votes])
                                   if votes else np.nan),
        "total_gain": float(gains.sum()),
        "min_gain_share": float(gains.min() / gains.mean()) if result["agreement"] and gains.mean() > 0 else np.nan,
    }
    for c in cs:
        m[f"batna_{c}"] = float(d[c])
    if zopa:
        m["zopa_exists"] = int(zopa["exists"])
        scores = _batna_score(u, zopa)
        for c in cs:
            m[f"zopa_width_{c}"] = float(zopa["width"].get(c, float("nan")))
            m[f"batna_score_{c}"] = float(scores.get(c, float("nan")))
        valid = [scores[c] for c in cs if scores.get(c) == scores.get(c)]   # drop nan
        m["mean_batna_score"] = float(np.mean(valid)) if valid else np.nan
    final = result["final_text"]
    if nbs:
        nbs_gain = sum(nbs["utilities"][c] - d[c] for c in cs)
        m["gain_ratio_vs_nbs"] = float(gains.sum() / nbs_gain) if nbs_gain > 0 else np.nan
        m["nbs_distance_km3"] = float(np.mean(np.abs(_totals(basin, final) - _totals(basin, nbs["proposal"]))))
    m["anchor_566_distance_km3"] = float(np.mean(np.abs(_totals(basin, final) - _totals(basin, basin.protocol_566_proposal()))))
    delta = basin.delta_flow(basin.normalize(final))
    env = basin.cfg["environmental_flow"]
    m["delta_flow_met"] = int(all(delta[s] >= env[s] - 1e-6 for s in SEASONS))
    m["tajik_shift"] = basin.normalize(final)["tajik_storage_shift"]
    for c in cs:
        m[f"gain_{c}"] = round(u[c] - d[c], 1)
    claims = result["claims"]
    m["n_claims"] = len(claims)
    for k in ("supported", "partial", "unsupported", "bad_id", "uncited"):
        m[f"claims_{k}_rate"] = (np.mean([c.get("verdict") == k for c in claims]) if claims else np.nan)
    findings = [a.get("finding") for rv in result.get("legal_reviews", []) for a in rv.get("assessments", []) if isinstance(a, dict)]
    m["legal_concerns"] = sum(f in ("concern", "inconsistent") for f in findings)
    tools = result.get("tool_calls", [])
    m["lookups"] = sum(t["tool"] == "lookup" for t in tools)
    m["what_ifs"] = sum(t["tool"] == "simulate" for t in tools)
    m["caucus_messages"] = len(result.get("caucus", []))
    m["goal_requests"] = len(result.get("goal_revision_requests", []))
    m["legal_blocks"] = len(result.get("legal_blocks", []))
    # Structured reasoning: did each delegate reply fill all six steps, and did the
    # decision it wrote in step 5 match the vote it cast?
    replies = [e["output"] for e in result.get("events", []) if e.get("phase") == "response"]
    shaped = [o["reasoning"] for o in replies if isinstance(o.get("reasoning"), dict)]
    m["reasoning_complete_rate"] = (np.mean([all(str(r.get(k) or "").strip() for k in REASONING_KEYS)
                                             for r in shaped]) if shaped else np.nan)
    stated = [(_stated_vote(o["reasoning"].get("decision")), o.get("vote")) for o in replies
              if isinstance(o.get("reasoning"), dict)]
    stated = [(s, v) for s, v in stated if s]
    m["reasoning_vote_mismatch_rate"] = np.mean([s != v for s, v in stated]) if stated else np.nan
    return m


def _stated_vote(decision) -> str | None:
    """accept / reject as written at the start of the decision step, else None."""
    words = str(decision or "").lower().replace(",", " ").replace(".", " ").split()
    for w in words[:4]:
        if w.startswith("accept"):
            return "accept"
        if w.startswith("reject"):
            return "reject"
    return None
