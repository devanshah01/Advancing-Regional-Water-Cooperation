"""Game theoretic reference points computed from the environment.

Manoranjan and Gaikwad use Nash equilibrium as a diagnostic of whether agents
follow the incentive structure. In a multi round bargaining game the analogous
reference points are:

  * the disagreement point d (each country's no deal utility),
  * the Nash bargaining solution (maximise the product of gains over d),
  * the utilitarian optimum (maximise total utility, ignoring who gets it).

Deviations of the negotiated outcome from these points are diagnostics, not
judgments of what a fair treaty is.
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


def _solve(basin: Basin, objective: str, with_transfers: bool = True, starts: int = 12, seed: int = 0):
    cs = basin.countries
    n = len(cs)
    d = basin.no_deal_utilities(rounded=False)
    maxshift = basin.cfg["max_storage_shift"]
    tb = basin.cfg["transfer_bounds"] if with_transfers else 0.0
    bounds = [(0, None)] * (2 * n) + [(0, maxshift)] + [(-tb, tb)] * n

    def utils(x):
        return basin.utilities(_unpack(x, basin), rounded=False)

    def f(x):
        u = utils(x)
        if objective == "nash":
            gains = np.array([u[c] - d[c] for c in cs])
            # smooth barrier: log for positive gains, steep linear penalty otherwise
            safe = np.where(gains > 1.0, gains, 1.0)
            return -np.sum(np.log(safe)) + 10.0 * np.sum(np.clip(1.0 - gains, 0, None))
        return -sum(u.values())

    cons = [
        {"type": "ineq", "fun": lambda x, s=s, k=k: basin.available(x[2 * n])[s] - sum(x[2 * i + k] for i in range(n))}
        for k, s in enumerate(SEASONS)
    ]
    cons.append({"type": "eq", "fun": lambda x: np.sum(x[2 * n + 1:])})

    rng = np.random.default_rng(seed)
    best = None
    for _ in range(starts):
        if _ == 0:
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
            for i in range(n):
                x0 += [0.85 * avail["growing"] * w[0][i], 0.85 * avail["winter"] * w[1][i]]
        x0 += [shift0] + [0.0] * n
        res = minimize(f, np.array(x0), method="SLSQP", bounds=bounds, constraints=cons,
                       options={"maxiter": 500, "ftol": 1e-9})
        feasible = not basin.validate(_unpack(res.x, basin))
        if feasible and (best is None or res.fun < best.fun):
            best = res
    if best is None:
        return None
    p = _unpack(best.x, basin)
    return {"proposal": p, "utilities": basin.utilities(p), "no_deal": d}


def nash_bargaining_solution(basin: Basin, **kw):
    return _solve(basin, "nash", **kw)


def utilitarian_optimum(basin: Basin, **kw):
    return _solve(basin, "utilitarian", **kw)
