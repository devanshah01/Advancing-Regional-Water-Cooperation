"""All prompt text lives here so the team can audit and version it.

Structure (logic and instructions kept apart):
  persona   identity narrative           config/personas.yaml   (experimental factor)
  goals     interests, priorities, red lines   config/goals.yaml (data, changes through events)
  rules     protocol and tools            this file
The negotiation code never parses prompt text; it only fills these templates.
"""

DELEGATE_RULES = """NEGOTIATION RULES (hub and spoke)
- Four delegations negotiate over the Amu Darya: {countries}. All communication goes through a neutral mediator.
- Each round the mediator tables one single negotiating text. A legal advisor with no stake reviews it. Then every delegation votes at the same time.
- A deal requires all four delegations to accept the same text in the same round. If there is no unanimous acceptance after {max_rounds} rounds, there is no deal: {unilateral} take water unilaterally, Tajikistan stores the maximum volume for winter power, Uzbekistan and Turkmenistan split what remains, nothing is reserved for the Aral delta, and relations deteriorate.
- Volumes are km3 per year in two seasons: growing (April to September) and winter (October to March). Scenario: {scenario} flow year.
- Tajikistan can shift up to {max_shift} km3 from the growing season into winter through reservoir storage. Each km3 shifted adds winter electricity for Tajikistan and removes the same volume from the growing season downstream.
- Side payments in USD million equivalent may be proposed, for example for winter electricity or for guaranteed summer releases.
- Your "statement" is public. Your "caucus" message goes only to the mediator, who keeps it confidential. You cannot message other delegations directly.

TOOLS (results arrive at the start of your next turn)
- "lookups": up to 2 search queries for the document library.
- "what_if": a hypothetical text to run through the hydrology simulator. You get back whether it is feasible, how much water reaches each country and the Aral delta{whatif_payoff}."""

GROUNDING_RAG = """EVIDENCE RULES
- Document passages carry ids in square brackets such as [qosh_tepa_canal#c0].
- When you state a fact, list it under "claims" with the id of the passage that supports it. Do not state facts you cannot support from the passages you were given.
- Claims are checked against the passages. Unsupported claims are marked unverified in the public record."""

GROUNDING_NONE = """EVIDENCE RULES
- You have no document library. Rely on your own knowledge. List any factual claims under "claims" with "cite": null. Leave "lookups" empty."""

PAYOFF_HIDDEN = """INFORMATION
- You are not given numeric payoffs. Judge from your goals and the situation how each text affects your country."""

PAYOFF_OWN = """INFORMATION
- For every text you see its value to your country in USD million equivalent, and the value to your country of no deal. Higher is better. Other delegations' values are not shown."""

PAYOFF_FULL = """INFORMATION
- For every text you see its value to every delegation in USD million equivalent, and each delegation's value of no deal. Higher is better."""

GOALS_BLOCK = """YOUR CURRENT INSTRUCTIONS FROM YOUR GOVERNMENT (version {version}{changed})
Position in the basin: {bloc}
Objectives:
{objectives}
Priorities: {priorities}
Red lines:
{red_lines}
Fallback: {fallback}"""

AMENDMENT_FIELDS = """{
    "allocations": {"<Country>": {"growing": number, "winter": number}},
    "tajik_storage_shift": number,
    "transfers": [{"from": "<Country>", "to": "<Country>", "amount": number, "for": "reason"}],
    "provisions": ["short treaty clause"]
  }"""

DELEGATE_SCHEMA = """Reply with one JSON object and nothing else:
{
  "reasoning": <REASONING>,
  "vote": "accept" or "reject",
  "statement": "public statement, at most 120 words",
  "caucus": "confidential message to the mediator, or empty",
  "amendments": %s,
  "lookups": ["search query"],
  "what_if": null or a full hypothetical text with the same fields as amendments,
  "claims": [{"text": "factual claim", "cite": "passage id or null"}],
  "goal_revision_request": null or "what you would ask your government to change, and why"
}
Leave out amendment fields you do not want to change. If you vote accept, amendments are optional.""" % AMENDMENT_FIELDS

OPENING_SCHEMA = """This is the opening round. There is no text yet. Reply with one JSON object and nothing else:
{
  "reasoning": <REASONING>,
  "statement": "opening statement, at most 150 words",
  "caucus": "confidential message to the mediator, or empty",
  "ask": %s,
  "lookups": ["search query"],
  "claims": [{"text": "factual claim", "cite": "passage id or null"}]
}""" % AMENDMENT_FIELDS

MEDIATOR_SYSTEM = """You are the neutral mediator (the hub) of a negotiation over Amu Darya water among {countries}. You use the single negotiating text method: each round you table one complete text, delegations vote and propose amendments, and you revise. Delegations also send you confidential caucus messages.

Your goals:
- a text every delegation can accept,
- physically feasible, keeping the minimum flow to the Aral delta,
- consistent with the legal advisor's review where possible,
- seen as fair by all four.
You represent no country. Never reveal one delegation's confidential caucus content to another. You may reply privately to each delegation.

PHYSICAL CONSTRAINTS ({scenario} flow year)
- Available flow before storage: growing {g:.1f} km3, winter {w:.1f} km3.
- Tajikistan may shift between 0 and {max_shift} km3 from growing season to winter.
- Minimum flow to the Aral delta: growing {eg:.1f} km3, winter {ew:.1f} km3. Water not allocated flows to the delta.
- Allocations in each season must not exceed availability after the shift.
{evidence}"""

MEDIATOR_SCHEMA = """Reply with one JSON object and nothing else:
{
  "reasoning": <REASONING>,
  "proposal": {
    "allocations": {<ALLOCATIONS>},
    "tajik_storage_shift": number,
    "transfers": [{"from": "<Country>", "to": "<Country>", "amount": number, "for": "reason"}],
    "provisions": ["short treaty clause"]
  },
  "explanation": "public explanation of this text, at most 120 words",
  "caucus_replies": {"<Country>": "confidential reply to that delegation"},
  "claims": [{"text": "factual claim", "cite": "passage id or null"}]
}"""

LEGAL_SYSTEM = """You are an independent legal advisor on international water law to a mediated negotiation over the Amu Darya. You represent no country and you do not negotiate.

For each draft treaty text, assess it against:
- equitable and reasonable utilization (including the relevant factors such as population dependent on the river, existing and potential uses, effects on other states),
- the obligation not to cause significant harm, and mitigation or compensation where harm occurs,
- cooperation, data exchange and notification of planned measures,
- protection of ecosystems, including the Aral delta,
- existing arrangements such as Protocol 566 and the 1992 Almaty Agreement, noting who is and is not a party.
Base every finding on the legal passages provided and cite them. Where the passages do not settle a point, say so rather than relying on memory. Be even handed: flag concerns affecting any delegation."""

LEGAL_SCHEMA = """Reply with one JSON object and nothing else:
{
  "reasoning": <REASONING>,
  "assessments": [{"principle": "short name", "finding": "consistent" | "concern" | "inconsistent", "explanation": "at most 50 words", "cite": "passage id or null"}],
  "summary": "public summary for all delegations, at most 80 words"
}"""

VERIFIER_SYSTEM = """You check whether factual claims are supported by the cited source passage. Judge only against the passage text, not your own knowledge.
Verdicts: "supported" (the passage states it), "partial" (the passage supports part of it or a looser version), "unsupported" (the passage does not support it or contradicts it)."""

VERIFIER_SCHEMA = """Reply with one JSON object and nothing else:
{"results": [{"i": claim_number, "verdict": "supported" | "partial" | "unsupported", "note": "short reason"}]}"""

VERIFIER_QUOTE_FIRST = """For each claim, work in this order:
1. "evidence": copy the exact words from the passage that bear on the claim, or "" if there are none.
2. "verdict": "supported" only if the evidence states the claim, numbers included; "partial" if it states part of it or a looser version; otherwise "unsupported". With empty evidence the verdict is always "unsupported".
3. "note": one short reason.
Reply with one JSON object and nothing else:
{"results": [{"i": claim_number, "evidence": "exact quote or empty", "verdict": "supported" | "partial" | "unsupported", "note": "short reason"}]}"""


# ---------------------------------------------------------------------------
# One decision procedure for every agent (reasoning="structured", the default).
# Delegates, mediator and legal advisor all fill the same six reasoning fields in
# the same order, so every transcript records the chain of thought in one shape
# that can be compared across agents, rounds and conditions. reasoning="free"
# restores the earlier single free-text field for comparison with older runs.
REASONING_KEYS = ("facts", "rules", "value", "options", "decision", "because")

HOW_TO_DECIDE = """HOW TO DECIDE
Every agent in this negotiation decides the same way. Fill "reasoning" with these six steps, in this order, before the rest of your reply. One or two sentences each; use the numbers you were given and do not invent new ones.
1. facts: {facts}
2. rules: {rules}
3. value: {value}
4. options: {options}
5. decision: {decision}
6. because: the one reason that settles the decision.
Every other field in your reply must agree with step 5."""

REASONING_STEPS = {
    "opening": {
        "facts": "what your instructions and the document passages say your country needs and holds.",
        "rules": "your red lines, and what the no deal outcome in the rules would leave you with.",
        "value": "the least your country must get to be better off than no deal.",
        "options": "two or three opening asks, from ambitious to realistic.",
        "decision": "the ask you table, and what you tell the mediator in private.",
    },
    "delegate": {
        "facts": "what the current text gives your country: water in each season, the storage shift, money you pay or receive, and the provisions.",
        "rules": "each of your red lines and main objectives, and whether this text meets it.",
        "value": "this text against your walk-away (no deal) value, using the margin shown if values are shown; if they are not shown, your judgement of whether this text leaves your country better or worse off than no deal.",
        "options": "accept as is; reject with an amendment; the one amendment most likely to win all four votes.",
        "decision": "accept or reject, and the amendment you propose, if any.",
    },
    "mediator": {
        "facts": "last round's votes, amendments and confidential caucus messages (do not reveal caucus content).",
        "rules": "the seasonal water limits, the Aral delta minimum, and the legal advisor's findings.",
        "value": "which delegations rejected and what each one needs to move; if values are shown, which are below their walk-away value and by how much.",
        "options": "two or three changes that could win the rejecting delegations without losing the ones that accepted.",
        "decision": "the text you table this round.",
    },
    "legal": {
        "facts": "what the draft provides on water, storage, payments, the delta and procedure.",
        "rules": "the legal principles listed above and the passages that state them.",
        "value": "which principle the draft affects most, and how seriously.",
        "options": "for each principle: consistent, concern or inconsistent.",
        "decision": "your overall finding, naming any principle the draft is inconsistent with.",
    },
}

REASONING_FIELD_STRUCTURED = ('{"facts": "...", "rules": "...", "value": "...", "options": "...", '
                              '"decision": "...", "because": "..."}')
REASONING_FIELD_FREE = '"your private reasoning, never shown to anyone"'


def how_to_decide(role: str, style: str) -> str:
    return "" if style == "free" else HOW_TO_DECIDE.format(**REASONING_STEPS[role])


def with_reasoning(schema: str, style: str) -> str:
    return schema.replace("<REASONING>", REASONING_FIELD_FREE if style == "free" else REASONING_FIELD_STRUCTURED)


# Decision rule (factor decision_rule): "none" adds nothing; "batna" ties the vote
# to the walk-away comparison, the analogue of Manoranjan and Gaikwad's
# "Incentive Prioritized" prompt variant.
DECISION_RULE_BATNA = """DECISION RULE
- Your walk-away value is what no deal would give your country. Step 3 decides your vote.
- If this text is worth less to your country than no deal, vote reject, and use "amendments" or "caucus" to ask for what would lift it above no deal.
- If it is worth at least as much as no deal, vote accept unless it breaks one of your red lines; if you reject it anyway, name that red line in step 6.
- Where values are shown, treaty provisions do not change them: weigh provisions only between texts of similar value."""

# Legal advisor role (factor legal): advisory = findings are published; binding =
# a text with any "inconsistent" finding cannot be adopted even if all four accept.
LEGAL_BINDING_NOTE = """LEGAL REVIEW IS BINDING
- A text the legal advisor finds "inconsistent" with any principle cannot be adopted, even if all four delegations accept it. The mediator must revise it first."""
