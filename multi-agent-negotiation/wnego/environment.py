"""Sandbox environment: the basin water ledger.

The environment is the only component allowed to say whether a proposal is
feasible and what it is worth. Agents never compute payoffs themselves; they
receive them (or not) depending on the payoff visibility condition.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path

import yaml

SEASONS = ("growing", "winter")


@dataclass
class Basin:
    cfg: dict
    scenario: str = "normal"

    @classmethod
    def load(cls, path: str | Path, scenario: str = "normal") -> "Basin":
        with open(path) as f:
            cfg = yaml.safe_load(f)
        if scenario not in cfg["flow_scenarios"]:
            raise ValueError(f"unknown scenario {scenario}")
        return cls(cfg=cfg, scenario=scenario)

    # basic facts ---------------------------------------------------------
    @property
    def countries(self) -> list[str]:
        return list(self.cfg["countries"].keys())

    @property
    def base_flow(self) -> dict:
        return self.cfg["flow_scenarios"][self.scenario]

    def available(self, shift: float) -> dict:
        return {
            "growing": self.base_flow["growing"] - shift,
            "winter": self.base_flow["winter"] + shift,
        }

    # proposals -----------------------------------------------------------
    def normalize(self, proposal: dict) -> dict:
        """Fill missing fields with zeros so every proposal has the same shape."""
        p = copy.deepcopy(proposal or {})
        alloc = p.get("allocations") or {}
        p["allocations"] = {
            c: {s: float((alloc.get(c) or {}).get(s, 0.0) or 0.0) for s in SEASONS}
            for c in self.countries
        }
        p["tajik_storage_shift"] = float(p.get("tajik_storage_shift", 0.0) or 0.0)
        p["transfers"] = [
            t for t in (p.get("transfers") or [])
            if t.get("from") in self.countries and t.get("to") in self.countries
        ]
        p["provisions"] = list(p.get("provisions") or [])
        return p

    def validate(self, proposal: dict) -> list[str]:
        p = self.normalize(proposal)
        errors = []
        shift = p["tajik_storage_shift"]
        if not 0 <= shift <= self.cfg["max_storage_shift"]:
            errors.append(f"tajik_storage_shift must be between 0 and {self.cfg['max_storage_shift']}")
        avail = self.available(shift)
        for s in SEASONS:
            total = sum(p["allocations"][c][s] for c in self.countries)
            if total > avail[s] + 1e-6:
                errors.append(f"{s} allocations total {total:.2f} km3 but only {avail[s]:.2f} km3 is available")
            for c in self.countries:
                if p["allocations"][c][s] < 0:
                    errors.append(f"{c} {s} allocation is negative")
        net = self.net_transfers(p)
        bound = self.cfg["transfer_bounds"]
        for c, v in net.items():
            if abs(v) > bound:
                errors.append(f"net transfer for {c} is {v:.0f}, beyond the bound of {bound}")
        return errors

    def net_transfers(self, p: dict) -> dict:
        net = {c: 0.0 for c in self.countries}
        for t in p["transfers"]:
            amt = float(t.get("amount", 0) or 0)
            net[t["from"]] -= amt
            net[t["to"]] += amt
        return net

    def delta_flow(self, p: dict) -> dict:
        avail = self.available(p["tajik_storage_shift"])
        return {s: avail[s] - sum(p["allocations"][c][s] for c in self.countries) for s in SEASONS}

    # payoffs -------------------------------------------------------------
    def utilities(self, proposal: dict, include_conflict_cost: bool = False, rounded: bool = True) -> dict:
        p = self.normalize(proposal)
        delta = self.delta_flow(p)
        env = self.cfg["environmental_flow"]
        shortfall = sum(max(0.0, env[s] - delta[s]) for s in SEASONS)
        net = self.net_transfers(p)
        out = {}
        for c, par in self.cfg["countries"].items():
            a = p["allocations"][c]
            u = par["irrigation_growing"] * math.log1p(max(a["growing"], 0))
            u += par["irrigation_winter"] * math.log1p(max(a["winter"], 0))
            if c == self.cfg["storage_country"]:
                u += par["hydropower_per_km3_shift"] * p["tajik_storage_shift"]
            u -= par["delta_penalty_per_km3"] * shortfall
            u += net[c]
            if include_conflict_cost:
                u -= self.cfg["no_deal"]["conflict_cost"]
            out[c] = round(u, 2) if rounded else u
        return out

    def no_deal_proposal(self) -> dict:
        shift = self.cfg["no_deal"]["storage_shift"]
        avail = self.available(shift)
        alloc = {}
        for c, par in self.cfg["countries"].items():
            if "unilateral_take" in par:
                alloc[c] = dict(par["unilateral_take"])
        rest = [c for c in self.countries if c not in alloc]
        ref = self.cfg["protocol_566_km3"]
        wsum = sum(ref[c] for c in rest)
        fixed = list(alloc)
        for s in SEASONS:
            left = avail[s] - sum(alloc[c][s] for c in fixed)
            for c in rest:
                alloc.setdefault(c, {})[s] = max(0.0, left) * ref[c] / wsum
        return self.normalize({"allocations": alloc, "tajik_storage_shift": shift})

    def no_deal_utilities(self, rounded: bool = True) -> dict:
        return self.utilities(self.no_deal_proposal(), include_conflict_cost=True, rounded=rounded)

    def protocol_566_proposal(self) -> dict:
        """Status quo anchor: Protocol 566 ratios scaled to this scenario, no storage shift."""
        ref = self.cfg["protocol_566_km3"]
        tot = sum(ref.values())
        avail = self.available(0.0)
        env = self.cfg["environmental_flow"]
        alloc = {c: {s: (avail[s] - env[s]) * ref[c] / tot for s in SEASONS} for c in self.countries}
        return self.normalize({"allocations": alloc, "tajik_storage_shift": 0.0})

    def describe(self, proposal: dict) -> str:
        p = self.normalize(proposal)
        lines = []
        for c in self.countries:
            a = p["allocations"][c]
            lines.append(f"  {c}: growing {a['growing']:.1f}, winter {a['winter']:.1f}, total {a['growing'] + a['winter']:.1f} km3")
        lines.append(f"  Tajik storage shift to winter: {p['tajik_storage_shift']:.1f} km3")
        d = self.delta_flow(p)
        lines.append(f"  Flow left for the Aral delta: growing {d['growing']:.1f}, winter {d['winter']:.1f} km3")
        for t in p["transfers"]:
            lines.append(f"  Transfer: {t['from']} pays {t['to']} {float(t.get('amount', 0)):.0f} ({t.get('for', 'unspecified')})")
        for pr in p["provisions"]:
            lines.append(f"  Provision: {pr}")
        return "\n".join(lines)
