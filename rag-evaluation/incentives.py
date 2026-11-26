"""
incentives.py: penalties/rewards + Nash equilibrium tools for the Amu Darya
multi-agent negotiation.

Three parts:
  1. Negotiation-stage incentives: delay cost + credibility scoring
  2. Compliance game after the deal: payoff table + pure Nash finder
  3. Nash bargaining benchmark for the proposals themselves

All payoff numbers are HYPOTHETICAL placeholders. Replace them with values
derived from your scaling rules / cost files, or label them as hypothetical
in the paper.

Requires: numpy, scipy (only for the bargaining solution)
    pip install numpy scipy
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from itertools import product

# ----------------------------------------------------------------------------
# Players and actions
# ----------------------------------------------------------------------------
UPSTREAM = ["Kyrgyzstan", "Tajikistan"]
DOWNSTREAM = ["Kazakhstan", "Uzbekistan", "Turkmenistan"]
PLAYERS = UPSTREAM + DOWNSTREAM

ACTIONS = {p: ["RELEASE_AGREED", "HOLD_BACK"] for p in UPSTREAM}
ACTIONS.update({p: ["WITHDRAW_QUOTA", "OVERDRAW"] for p in DOWNSTREAM})

COOPERATIVE = {p: ACTIONS[p][0] for p in PLAYERS}
DEFECT = {p: ACTIONS[p][1] for p in PLAYERS}
A_COOP = tuple(COOPERATIVE[p] for p in PLAYERS)
A_DEFECT = tuple(DEFECT[p] for p in PLAYERS)


# ----------------------------------------------------------------------------
# Configuration: one object per experimental condition
# ----------------------------------------------------------------------------
@dataclass
class IncentiveConfig:
    # --- negotiation stage ---
    delay_rate: float = 0.0          # payoff shrinks by this fraction each round (0 = off)
    nodeal_decay: float = 0.0        # no-deal default worsens by this fraction per round
    credibility_on: bool = False
    cred_start: float = 5.0
    cred_reward: float = 1.0         # verified claim
    cred_penalty: float = 2.0        # unsupported / fabricated claim
    cred_min: float = 0.0
    cred_max: float = 10.0

    # --- compliance stage ---
    base_utility: float = 10.0       # payoff from the agreement if everyone complies
    defect_gain: float = 4.0         # private gain from defecting
    externality: float = 1.5         # harm to each OTHER country per defector
    penalty: float = 0.0             # enforced cost on defectors (0 = no enforcement)
    detection_prob: float = 1.0      # chance a defection is caught (expected-value penalty)
    cred_stake: float = 0.3          # reputation lost per credibility point if defecting

    # --- information condition (paper's "payoff visibility" factor) ---
    tell_agents_about_incentives: bool = True


# ----------------------------------------------------------------------------
# 1a. Delay cost
# ----------------------------------------------------------------------------
def delay_factor(round_idx: int, rate: float) -> float:
    """Multiplier on all payoffs after `round_idx` rounds (round 0 -> 1.0)."""
    return (1.0 - rate) ** round_idx


def discounted(utils: dict, round_idx: int, cfg: IncentiveConfig) -> dict:
    f = delay_factor(round_idx, cfg.delay_rate)
    return {p: u * f for p, u in utils.items()}


def no_deal_payoffs(base_no_deal: dict, round_idx: int, cfg: IncentiveConfig) -> dict:
    """The default allocation gets worse the longer negotiation drags on.
    Assumes base_no_deal values are on the same scale as proposal utilities."""
    f = (1.0 - cfg.nodeal_decay) ** round_idx
    return {p: u * f for p, u in base_no_deal.items()}


def beats_no_deal(proposal_utils: dict, base_no_deal: dict, round_idx: int,
                  cfg: IncentiveConfig) -> dict:
    """Per-country: does this proposal (after delay discount) beat the
    (decayed) no-deal payoff? Use this to log your 'better than no deal'
    metric and to see who SHOULD have accepted."""
    prop = discounted(proposal_utils, round_idx, cfg)
    nd = no_deal_payoffs(base_no_deal, round_idx, cfg)
    return {p: prop[p] > nd[p] for p in prop}


# ----------------------------------------------------------------------------
# 1b. Credibility scoring (plug into your verifier)
# ----------------------------------------------------------------------------
class CredibilityLedger:
    def __init__(self, cfg: IncentiveConfig, players=PLAYERS):
        self.cfg = cfg
        self.scores = {p: cfg.cred_start for p in players}
        self.log = []  # (round, player, verified, new_score)

    def update(self, player: str, verified: bool, round_idx: int = -1) -> float:
        if not self.cfg.credibility_on:
            return self.scores[player]
        delta = self.cfg.cred_reward if verified else -self.cfg.cred_penalty
        s = self.scores[player] + delta
        self.scores[player] = max(self.cfg.cred_min, min(self.cfg.cred_max, s))
        self.log.append((round_idx, player, verified, self.scores[player]))
        return self.scores[player]

    def update_from_verifier(self, player: str, verifier_results: list[bool],
                             round_idx: int = -1):
        """verifier_results: one bool per cited claim this delegate made."""
        for ok in verifier_results:
            self.update(player, ok, round_idx)

    def weight(self, player: str) -> float:
        """Normalised weight for the mediator (sums to 1 across players)."""
        total = sum(self.scores.values()) or 1.0
        return self.scores[player] / total

    def weighted_support(self, votes: dict) -> float:
        """votes: {player: True/False}. Returns credibility-weighted yes share."""
        return sum(self.weight(p) for p, v in votes.items() if v)

    def briefing_text(self) -> str:
        """Paste into each agent's prompt if tell_agents_about_incentives."""
        lines = [f"- {p}: {s:.1f}" for p, s in self.scores.items()]
        return ("Credibility scores (verified claims raise your score, "
                "unsupported claims lower it; the mediator weights proposals "
                "by score):\n" + "\n".join(lines))


# ----------------------------------------------------------------------------
# 2. Compliance game: payoffs + pure Nash equilibria
# ----------------------------------------------------------------------------
def is_defect(player: str, action: str) -> bool:
    return action == DEFECT[player]


def compliance_payoffs(profile: tuple, cfg: IncentiveConfig,
                       ledger: CredibilityLedger | None = None,
                       round_idx: int = 0,
                       base_by_player: dict | None = None) -> dict:
    """Payoff to each player for a joint action profile (ordered as PLAYERS).

    payoff_i = base_i
             + defect_gain                    if i defects
             - externality * (#other defectors)
             - detection_prob * penalty       if i defects   (enforced in code)
             - cred_stake * credibility_i     if i defects   (reputation at stake)
    then scaled by the delay factor.
    """
    defects = {p: is_defect(p, a) for p, a in zip(PLAYERS, profile)}
    n_def = sum(defects.values())
    f = delay_factor(round_idx, cfg.delay_rate)
    out = {}
    for p in PLAYERS:
        u = (base_by_player or {}).get(p, cfg.base_utility)
        u -= cfg.externality * (n_def - (1 if defects[p] else 0))
        if defects[p]:
            u += cfg.defect_gain
            u -= cfg.detection_prob * cfg.penalty
            if cfg.credibility_on and ledger is not None:
                u -= cfg.cred_stake * ledger.scores[p]
        out[p] = u * f
    return out


def build_payoff_table(cfg: IncentiveConfig, ledger=None, round_idx=0,
                       base_by_player=None) -> dict:
    """{profile: {player: utility}} for all 2^5 = 32 profiles."""
    return {
        prof: compliance_payoffs(prof, cfg, ledger, round_idx, base_by_player)
        for prof in product(*[ACTIONS[p] for p in PLAYERS])
    }


def pure_nash(table: dict) -> list[tuple]:
    """All pure-strategy Nash equilibria. Uses the paper's weak condition:
    no player strictly gains by a unilateral deviation."""
    eqs = []
    for prof, util in table.items():
        stable = True
        for i, p in enumerate(PLAYERS):
            for alt in ACTIONS[p]:
                if alt == prof[i]:
                    continue
                dev = prof[:i] + (alt,) + prof[i + 1:]
                if table[dev][p] > util[p] + 1e-12:
                    stable = False
                    break
            if not stable:
                break
        if stable:
            eqs.append(prof)
    return eqs


def classify(profile: tuple, table: dict) -> dict:
    """Outcome measures matching Manoranjan & Gaikwad: Nash attainment + type."""
    nash_set = pure_nash(table)
    is_nash = profile in nash_set
    if profile == A_COOP:
        kind = "cooperative"
    elif profile == A_DEFECT:
        kind = "full_defection"
    else:
        kind = "mixed"
    return {
        "nash": is_nash,
        "type": kind if is_nash else f"non_nash_{kind}",
        "n_defectors": sum(is_defect(p, a) for p, a in zip(PLAYERS, profile)),
        "equilibria_in_game": nash_set,
        "coop_is_equilibrium": A_COOP in nash_set,
        "defect_is_equilibrium": A_DEFECT in nash_set,
    }


def minimum_deterrent_penalty(cfg: IncentiveConfig, ledger=None) -> float:
    """Penalty at which cooperation becomes a (weak) equilibrium, assuming all
    others cooperate. Useful for choosing penalty levels to sweep."""
    worst = 0.0
    for p in PLAYERS:
        stake = cfg.cred_stake * ledger.scores[p] if (cfg.credibility_on and ledger) else 0.0
        need = (cfg.defect_gain - stake) / max(cfg.detection_prob, 1e-9)
        worst = max(worst, need)
    return worst


# ----------------------------------------------------------------------------
# Getting actions from LLM agents (the "payoff visibility" factor)
# ----------------------------------------------------------------------------
def build_action_prompt(player: str, cfg: IncentiveConfig, table: dict,
                        ledger: CredibilityLedger | None = None,
                        show_payoffs: bool = True, scenario: str = "") -> str:
    acts = ACTIONS[player]
    txt = [f"You are the delegate of {player}. {scenario}",
           "An allocation has been agreed. Now each country independently "
           "chooses an action. Nobody sees the others' choices."]
    if cfg.tell_agents_about_incentives:
        if cfg.penalty > 0:
            txt.append(f"Enforcement: a defector pays a penalty of {cfg.penalty} "
                       f"(detected with probability {cfg.detection_prob}).")
        if cfg.credibility_on and ledger is not None:
            txt.append(ledger.briefing_text())
    if show_payoffs:
        txt.append("Full payoff table (profile order: "
                   + ", ".join(PLAYERS) + "):")
        for prof, u in table.items():
            vals = ", ".join(f"{u[p]:.1f}" for p in PLAYERS)
            txt.append(f"  {prof} -> ({vals})")
    txt.append(f"Available actions: {acts[0]} or {acts[1]}.")
    txt.append('Respond in JSON exactly as: {"action": "<KEY>", "rationale": "<short>"}')
    return "\n".join(txt)


def parse_action(player: str, raw: str) -> str | None:
    """Pull the action out of the model's JSON; returns None if invalid."""
    m = re.search(r"\{.*\}", raw, flags=re.S)
    if not m:
        return None
    try:
        a = json.loads(m.group(0)).get("action")
    except json.JSONDecodeError:
        return None
    return a if a in ACTIONS[player] else None


def run_compliance_game(llm_call, cfg: IncentiveConfig, ledger=None,
                        show_payoffs=True, round_idx=0, scenario="",
                        base_by_player=None) -> dict:
    """llm_call(prompt:str) -> str. Wrap your ollama.generate here.

    Example:
        llm_call = lambda p: ollama.generate(model="llama3.2", prompt=p)["response"]
    """
    table = build_payoff_table(cfg, ledger, round_idx, base_by_player)
    chosen = []
    for p in PLAYERS:
        prompt = build_action_prompt(p, cfg, table, ledger, show_payoffs, scenario)
        a = parse_action(p, llm_call(prompt))
        if a is None:  # one retry, then record invalid
            a = parse_action(p, llm_call(prompt))
        chosen.append(a)
    if any(a is None for a in chosen):
        return {"valid": False, "profile": tuple(chosen)}
    profile = tuple(chosen)
    res = classify(profile, table)
    res.update({"valid": True, "profile": profile,
                "payoffs": table[profile]})
    return res


# ----------------------------------------------------------------------------
# 3. Nash bargaining benchmark for the negotiation itself
# ----------------------------------------------------------------------------
def nash_bargaining_allocation(total_water: float, values: dict,
                               disagreement: dict, min_share: dict | None = None):
    """Allocation x_i (sum = total_water) maximising prod_i (u_i - d_i), with
    diminishing-returns utility u_i = values[i] * sqrt(x_i). Swap `utility`
    below for your own function if you have one (e.g. from the scaling rules).
    Returns (allocation dict, gains dict) or (None, None) if infeasible.
    """
    import numpy as np
    from scipy.optimize import minimize

    ps = list(values)
    n = len(ps)
    d = np.array([disagreement[p] for p in ps])
    v = np.array([values[p] for p in ps])

    def utility(x):
        return v * np.sqrt(np.maximum(x, 1e-12))

    def neg_log_nash(x):
        gains = utility(x) - d
        if np.any(gains <= 1e-9):
            return 1e9 + 1e6 * np.sum(np.maximum(0, 1e-9 - gains))
        return -np.sum(np.log(gains))

    lo = np.array([(min_share or {}).get(p, 0.0) for p in ps])
    x0 = np.full(n, total_water / n)
    res = minimize(
        neg_log_nash, x0, method="SLSQP",
        bounds=[(l, total_water) for l in lo],
        constraints=[{"type": "eq", "fun": lambda x: np.sum(x) - total_water}],
    )
    if not res.success or res.fun >= 1e8:
        return None, None
    x = res.x
    gains = utility(x) - d
    return dict(zip(ps, x)), dict(zip(ps, gains))


def nash_product(gains: dict) -> float:
    prod = 1.0
    for g in gains.values():
        if g <= 0:
            return 0.0
        prod *= g
    return prod


def distance_to_nbs(agreed: dict, nbs: dict) -> float:
    """L1 distance between the agents' allocation and the NBS allocation,
    as a share of total water (0 = identical, 1 = max difference)."""
    total = sum(nbs.values()) or 1.0
    return sum(abs(agreed[p] - nbs[p]) for p in nbs) / (2 * total)


# ----------------------------------------------------------------------------
# Sanity check: python incentives.py
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    print("Compliance game equilibria as penalty rises")
    print(f"{'penalty':>8} {'cred':>5} {'#NE':>4} {'coop NE':>8} {'defect NE':>10}")
    for cred in (False, True):
        for pen in (0, 2, 4, 6):
            cfg = IncentiveConfig(penalty=pen, credibility_on=cred)
            led = CredibilityLedger(cfg)
            tbl = build_payoff_table(cfg, led)
            eq = pure_nash(tbl)
            print(f"{pen:>8} {str(cred):>5} {len(eq):>4} "
                  f"{str(A_COOP in eq):>8} {str(A_DEFECT in eq):>10}")

    print("\nDelay cost example (10%/round): payoff of 10 after 0,3,6 rounds:",
          [round(10 * delay_factor(r, 0.10), 2) for r in (0, 3, 6)])

    cfg = IncentiveConfig(credibility_on=True)
    led = CredibilityLedger(cfg)
    led.update_from_verifier("Uzbekistan", [True, False, False], round_idx=2)
    print("Credibility after 1 good, 2 bad claims (Uzbekistan):",
          led.scores["Uzbekistan"])

    alloc, gains = nash_bargaining_allocation(
        total_water=40.0,  # BCM, hypothetical
        values={"Kazakhstan": 3, "Uzbekistan": 6, "Turkmenistan": 5,
                "Tajikistan": 4, "Kyrgyzstan": 3},
        disagreement={"Kazakhstan": 3, "Uzbekistan": 8, "Turkmenistan": 8,
                      "Tajikistan": 4, "Kyrgyzstan": 3},
    )
    print("\nNash bargaining allocation (BCM):",
          {k: round(v, 1) for k, v in alloc.items()} if alloc else "infeasible")