"""Goal files and the scenario controller that changes them.

Goals are data. The prompt template renders whatever the current goal file says;
events change goal files at set rounds; every change is versioned and logged.
"""
from __future__ import annotations

import copy
import hashlib
import json

from . import prompts as P


class GoalBook:
    def __init__(self, goals: dict, events: list[dict] | None = None):
        self.goals = copy.deepcopy(goals)
        self.events = sorted(events or [], key=lambda e: e["round"])
        self.version = {c: 1 for c in self.goals}
        self.changed_at = {c: 0 for c in self.goals}
        self.history: list[dict] = []
        self.requests: list[dict] = []

    def fingerprint(self, country: str) -> str:
        return hashlib.sha1(json.dumps(self.goals[country], sort_keys=True).encode()).hexdigest()[:8]

    def render(self, country: str, round_no: int) -> str:
        g = self.goals[country]
        changed = ", updated this round" if self.changed_at[country] == round_no and round_no > 0 else ""
        return P.GOALS_BLOCK.format(
            version=self.version[country], changed=changed, bloc=g.get("bloc", ""),
            objectives="\n".join(f"- {o}" for o in g.get("objectives", [])),
            priorities=", ".join(f"{k} {v}" for k, v in (g.get("priorities") or {}).items()),
            red_lines="\n".join(f"- {r}" for r in g.get("red_lines", [])) or "- none",
            fallback=g.get("fallback", ""))

    def due(self, round_no: int) -> list[dict]:
        return [e for e in self.events if e["round"] == round_no]

    def apply(self, event: dict, round_no: int) -> dict:
        changes = {}
        for country, ops in (event.get("goal_updates") or {}).items():
            if country not in self.goals:
                continue
            g = self.goals[country]
            for field, value in (ops.get("set") or {}).items():
                g[field] = value
            for field, value in (ops.get("add") or {}).items():
                g.setdefault(field, [])
                g[field] += value if isinstance(value, list) else [value]
            for field, value in (ops.get("remove") or {}).items():
                g[field] = [x for x in g.get(field, []) if str(value) not in str(x)]
            if ops.get("priorities"):
                g.setdefault("priorities", {}).update(ops["priorities"])
            self.version[country] += 1
            self.changed_at[country] = round_no
            changes[country] = {"version": self.version[country], "fingerprint": self.fingerprint(country)}
        self.history.append({"round": round_no, "event": event.get("name"), "changes": changes})
        return changes

    def log_request(self, country: str, round_no: int, text: str):
        if text:
            self.requests.append({"round": round_no, "country": country, "request": text, "applied": False})

    def snapshot(self) -> dict:
        return {c: {"version": self.version[c], "fingerprint": self.fingerprint(c)} for c in self.goals}
