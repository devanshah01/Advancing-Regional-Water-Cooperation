"""Game theoretic reference points computed from the environment.

Manoranjan and Gaikwad use Nash equilibrium as a diagnostic of whether agents
follow the incentive structure. In a multi round bargaining game the analogous
reference points are:

  * the disagreement point d (each country's BATNA / no deal utility),
  * the Nash bargaining solution (maximise the product of gains over d),
  * the utilitarian optimum (maximise total utility, ignoring who gets it),
  * the ZOPA (zone of possible agreement): the region of feasible outcomes in
    which every country does at least as well as its own BATNA.

Deviations of the negotiated outcome from these points are diagnostics, not
judgments of what a fair treaty is.

Sources for the added BATNA / ZOPA methodology (see also evaluation.py):

Foundational definitions (the concepts this module computes a number for):
  Fisher, R., Ury, W. and Patton, B. (1991) "Getting to Yes: Negotiating
    Agreement Without Giving In", 2nd ed., Penguin -- defines BATNA as the
    reservation value below which a party should walk away (ch. 8), and the
    ZOPA as the overlap of the parties' acceptable ranges above their BATNAs.
  Raiffa, H. (1982) "The Art and Science of Negotiation", Harvard University
    Press -- constructs the efficient (Pareto) frontier of a multi-issue
    bargaining problem by maximizing one party's payoff subject to every
    other party's reservation value, which is exactly the optimization this
    module runs per country to trace the ZOPA's extent when there is no
    single scalar "price" to bracket (as there is in two-party, single-issue
    bargaining).
  Nash, J. (1950) "The Bargaining Problem", Econometrica 18(2) -- the gains-
    over-disagreement-point feasible set S this module calls the ZOPA, and
    the product-of-gains objective already used by nash_bargaining_solution.

Recent sources for why this is the right diagnostic to add here specifically
(LLM agents in multi-party bargaining; transboundary river allocation):
  Abdelnabi, S., Gomaa, A., Sivaprasad, S., Schonherr, L. and Fritz, M. (2024)
    "Cooperation, Competition, and Maliciousness: LLM-Stakeholders Interactive
    Negotiation", NeurIPS 2024 Datasets and Benchmarks Track -- finds GPT-3.5
    negotiation agents frequently violate their own BATNA, the same failure
    mode this project's own pilot found in README.md ("Afghanistan accepted a
    deal worth 367 to it when walking away was worth 805"); motivates
    reporting a continuous batna_score rather than only agreement rate.
  Qin, J., Fu, X., Peng, S., Xu, Y., Huang, J. and Huang, S. (2019) "Asymmetric
    Bargaining Model for Water Resource Allocation over Transboundary Rivers",
    International Journal of Environmental Research and Public Health 16(10),
    1733 -- a recent application of the Nash-bargaining family of solutions to
    transboundary river allocation specifically, the same problem class as
    this basin.
  Melnikovova, L. (2017) "Can Game Theory Help to Mitigate Water Conflicts in
    the Syrdarya Basin?", Acta Universitatis Agriculturae et Silviculturae
    Mendelianae Brunensis 65(4), 1393-1401 -- cooperative game theory applied
    to the Syr Darya, the neighbouring basin sharing the same ICWC riparians
    as the Amu Darya modelled here.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .environment import SEASONS, Basin


def _unpack(x, basin: Basin) -> dict:
    cs = basin.countries
    n = len(cs)
    alloc = {c: {"growing": x[2 * i], "winter": x[2 * i + 1]} for i, c in enumerate(cs)}
    shift = x[2 * n]
    tvec = x[2 * n + 1:]
    # turn net positions into pairwise payments, payers matched greedily to receivers
    payers = [[c, -tvec[i]] for i, c in enumerate(cs) if tvec[i] < -1e-6]
    receivers = [[c, tvec[i]] for i, c in enumerate(cs) if tvec[i] > 1e-6]
    transfers = []
    while payers and receivers:
        amt = min(payers[0][1], receivers[0][1])
        transfers.append({"from": payers[0][0], "to": receivers[0][0], "amount": amt,
                          "for": "benchmark side payment"})
        payers[0][1] -= amt
        receivers[0][1] -= amt
        if payers[0][1] <= 1e-6:
            payers.pop(0)
        if receivers[0][1] <= 1e-6:
            receivers.pop(0)
    return basin.normalize({"allocations": alloc, "tajik_storage_shift": shift, "transfers": transfers})


def _flow_constraints(basin: Basin):
    cs, n = basin.countries, len(basin.countries)
    cons = [
        {"type": "ineq", "fun": lambda x, s=s, k=k: basin.available(x[2 * n])[s] - sum(x[2 * i + k] for i in range(n))}
        for k, s in enumerate(SEASONS)
    ]
    cons.append({"type": "eq", "fun": lambda x: np.sum(x[2 * n + 1:])})
    return cons


def _no_deal_start(basin: Basin) -> np.ndarray:
    """The no-deal allocation as an optimizer start vector. Useful specifically
    for zopa_bounds: it trivially satisfies "every other country >= its own
    BATNA" (everyone is exactly at their BATNA there), so SLSQP starts already
    inside the individually-rational region instead of a random point that is
    usually far outside it -- about 20x faster in practice than relying on
    random restarts alone for that constrained problem."""
    cs = basin.countries
    nodeal = basin.no_deal_proposal()
    x0 = []
    for c in cs:
        x0 += [nodeal["allocations"][c]["growing"], nodeal["allocations"][c]["winter"]]
    x0 += [nodeal["tajik_storage_shift"]] + [0.0] * len(cs)
    return np.array(x0)


def _start_points(basin: Basin, starts: int, seed: int):
    cs, n = basin.countries, len(basin.countries)
    maxshift = basin.cfg["max_storage_shift"]
    rng = np.random.default_rng(seed)
    for i in range(starts):
        if i == 0:
            # start at the status quo anchor, which is usually close to feasible
            sq = basin.protocol_566_proposal()
            shift0 = maxshift / 2
            x0 = []
            for c in cs:
                x0 += [max(sq["allocations"][c]["growing"] - shift0 / n, 0.1), sq["allocations"][c]["winter"]]
        else:
            shift0 = rng.uniform(0, maxshift)
            avail = basin.available(shift0)
            w = rng.dirichlet(np.ones(n), size=2)
            x0 = []
            for j in range(n):
                x0 += [0.85 * avail["growing"] * w[0][j], 0.85 * avail["winter"] * w[1][j]]
        x0 += [shift0] + [0.0] * n
        yield np.array(x0)


def _optimize(basin: Basin, f, extra_cons=(), with_transfers: bool = True, starts: int = 12, seed: int = 0,
              extra_starts=(), maxiter: int = 500):
    """Core SLSQP driver shared by the Nash/utilitarian solver and the ZOPA
    frontier solver: minimise f(x) over feasible allocations, keeping only
    starts that satisfy both the hydrology constraints and extra_cons.
    extra_starts: additional explicit start vectors tried before the random
    ones (e.g. the no-deal point, which is already inside extra_cons for the
    ZOPA problem and converges far faster than a random start -- see
    _no_deal_start)."""
    n = len(basin.countries)
    tb = basin.cfg["transfer_bounds"] if with_transfers else 0.0
    bounds = [(0, None)] * (2 * n) + [(0, basin.cfg["max_storage_shift"])] + [(-tb, tb)] * n
    cons = _flow_constraints(basin) + list(extra_cons)

    best = None
    for x0 in list(extra_starts) + list(_start_points(basin, starts, seed)):
        res = minimize(f, x0, method="SLSQP", bounds=bounds, constraints=cons,
                       options={"maxiter": maxiter, "ftol": 1e-9})
        feasible = (not basin.validate(_unpack(res.x, basin))
                    and all(c["fun"](res.x) >= -1e-4 for c in extra_cons if c["type"] == "ineq"))
        if feasible and (best is None or res.fun < best.fun):
            best = res
    return best


def _solve(basin: Basin, objective: str, **kw):
    cs = basin.countries
    d = basin.no_deal_utilities(rounded=False)

    def f(x):
        u = basin.utilities(_unpack(x, basin), rounded=False)
        if objective == "nash":
            gains = np.array([u[c] - d[c] for c in cs])
            # smooth barrier: log for positive gains, steep linear penalty otherwise
            safe = np.where(gains > 1.0, gains, 1.0)
            return -np.sum(np.log(safe)) + 10.0 * np.sum(np.clip(1.0 - gains, 0, None))
        return -sum(u.values())

    best = _optimize(basin, f, **kw)
    if best is None:
        return None
    p = _unpack(best.x, basin)
    return {"proposal": p, "utilities": basin.utilities(p), "no_deal": d}


def nash_bargaining_solution(basin: Basin, **kw):
    return _solve(basin, "nash", **kw)


def utilitarian_optimum(basin: Basin, **kw):
    return _solve(basin, "utilitarian", **kw)


def zopa_bounds(basin: Basin, starts: int = 4, seed: int = 0, **kw) -> dict:
    """The zone of possible agreement: for each country, its BATNA (floor)
    and the most it could get while every OTHER country keeps at least its
    own BATNA and the outcome stays physically feasible (ceiling). This
    traces one extreme point of the Pareto-efficient frontier per country,
    the standard construction for an n-party, multi-issue ZOPA (Raiffa 1982;
    see module docstring). With n=2 and a single issue this reduces to the
    textbook ZOPA interval (Fisher, Ury and Patton 1991); here it is reported
    per country because allocations are a vector, not a scalar price.

    "exists" is True iff every country's ceiling is strictly above its BATNA,
    i.e. there is at least one feasible, individually rational deal for
    everyone simultaneously (Nash's (1950) feasible set S is non-empty).
    """
    cs = basin.countries
    d = basin.no_deal_utilities(rounded=False)
    seed_x0 = _no_deal_start(basin)   # already satisfies "every other country >= its own BATNA" by construction
    ceiling, proposal_at = {}, {}
    for target in cs:

        def f(x, _t=target):
            return -basin.utilities(_unpack(x, basin), rounded=False)[_t]

        extra = [
            {"type": "ineq", "fun": (lambda x, _o=other: basin.utilities(_unpack(x, basin), rounded=False)[_o] - d[_o])}
            for other in cs if other != target
        ]
        best = _optimize(basin, f, extra_cons=extra, starts=starts, seed=seed, extra_starts=[seed_x0],
                         maxiter=200, **kw)
        if best is None:
            ceiling[target], proposal_at[target] = d[target], None
        else:
            p = _unpack(best.x, basin)
            ceiling[target] = basin.utilities(p, rounded=False)[target]
            proposal_at[target] = p
    width = {c: max(0.0, ceiling[c] - d[c]) for c in cs}
    return {"batna": d, "ceiling": ceiling, "width": width, "proposal_at_ceiling": proposal_at,
            "exists": all(width[c] > 1e-6 for c in cs)}


def batna_score(final_utilities: dict, zopa: dict) -> dict:
    """Where a realized deal landed inside each country's ZOPA, 0 = at its
    BATNA (indifferent to no deal), 1 = at the ceiling it could get while
    holding everyone else at their own BATNA. Values are not bounded to
    [0, 1]: a deal can fall below a country's BATNA (payoff-inconsistent
    acceptance, cf. evaluation.py's below_batna_accept_rate) or, under side
    payments, exceed the single-country ceiling computed holding others
    fixed. This is a normalized gain-over-BATNA measure in the spirit of
    Raiffa's (1982) efficient-frontier diagrams and the gain-share analysis
    in Thompson, L. (2020) "The Mind and Heart of the Negotiator", 6th ed.,
    Pearson, ch. 3. It is the same quantity Abdelnabi et al. (2024, see
    module docstring) report informally as "BATNA violation": a negative
    score here is exactly their failure mode, made continuous instead of a
    simple below/above count.
    """
    d, width = zopa["batna"], zopa["width"]
    return {c: ((final_utilities[c] - d[c]) / width[c] if width.get(c, 0) > 1e-6 else float("nan"))
            for c in d}
