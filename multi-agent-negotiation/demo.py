"""Streamlit demo of the Amu Darya multi-agent negotiation.

Run from the repository root (that is where the theme in .streamlit/config.toml lives,
and where Streamlit Community Cloud starts the app):

    streamlit run multi-agent-negotiation/demo.py

Backends: Claude when ANTHROPIC_API_KEY is set (environment or Streamlit secrets),
Ollama when it is running locally, and mock always. Live Claude runs are billed, so
DEMO_PASSWORD (optional) locks them and DEMO_MAX_RUNS (default 5) caps them per
visitor. Recorded runs in demo_runs/ can be replayed by anyone.

Layout and look follow the Figma "chat" file: a guided flow of six screens along
a top navigation bar, warm off-white pages, white cards and deep green actions
(theme in .streamlit/config.toml, card and callout styles in CSS below).

The page follows the codebase. When any file in wnego/ or config/ changes, it
reloads the package and reruns, so the controls, agents, prompts, benchmarks and
metrics on screen are always the current code's. The settings are built from the
Condition dataclass and the config files, so a new factor or level appears here
without editing this file.
"""
from __future__ import annotations

import dataclasses
import hmac
import html
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Amu Darya Talks", page_icon="🌊", layout="wide")


def _load_secrets():
    """On Streamlit Cloud, keys live in the app's secrets settings; expose them as environment variables."""
    for name in ("ANTHROPIC_API_KEY", "DEMO_PASSWORD", "DEMO_MAX_RUNS"):
        if os.environ.get(name):
            continue
        try:
            value = st.secrets.get(name)
        except Exception:          # no secrets file (local runs)
            value = None
        if value:
            os.environ[name] = str(value)


_load_secrets()
HERE = Path(__file__).resolve().parent
DEMO_RUNS = HERE / "demo_runs"                      # recorded runs shipped with the app
CLAUDE_MODELS = ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"]
MAX_LIVE_RUNS = int(os.environ.get("DEMO_MAX_RUNS", "5"))   # live Claude runs per visitor session
WATCHED = (HERE / "wnego", HERE / "config")
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


# ---------------------------------------------------------------------------
# Follow the codebase: reload wnego whenever its code or config changes
def code_fingerprint() -> tuple[int, str, float]:
    files = [p for d in WATCHED for p in d.rglob("*")
             if p.is_file() and p.suffix in {".py", ".yaml", ".yml"} and "__pycache__" not in p.parts]
    latest = max(files, key=lambda p: p.stat().st_mtime)
    fp = hash(tuple(sorted((str(p), p.stat().st_mtime_ns) for p in files)))
    return fp, str(latest.relative_to(HERE)).replace("\\", "/"), latest.stat().st_mtime


_state = sys.modules.setdefault("_wnego_demo_state", type(sys)("_wnego_demo_state"))   # survives reruns
FP, LATEST_FILE, LATEST_TIME = code_fingerprint()
if getattr(_state, "fp", None) != FP:
    for name in [m for m in sys.modules if m == "wnego" or m.startswith("wnego.")]:
        del sys.modules[name]          # all of them, so modules that import a changed one reload too
    st.cache_data.clear()
    st.cache_resource.clear()
    _state.fp = FP

from wnego import prompts as P                                              # noqa: E402
from wnego.agents import Condition                                          # noqa: E402
from wnego.benchmarks import nash_bargaining_solution, utilitarian_optimum  # noqa: E402
from wnego.environment import Basin                                         # noqa: E402
from wnego.experiment import ROOT, load_configs, run_one                    # noqa: E402
from wnego.llm import MockLLM                                               # noqa: E402
from wnego.protocol import Negotiation                                      # noqa: E402
from wnego.retrieval import make_retriever                                  # noqa: E402

BASIN_PATH = ROOT / "config/basin.yaml"


# ---------------------------------------------------------------------------
# Look: cards, callouts and headings (colours are in .streamlit/config.toml)
st.html("""
<style>
  .block-container { padding-top: 4.5rem; max-width: 1280px; }
  [class*="st-key-card"] { background: #FFFFFF; border: 1px solid #E2DCCF; border-radius: 0.75rem;
                           padding: 1.25rem 1.4rem; box-shadow: 0 1px 2px rgba(31, 42, 38, 0.04); }
  [class*="st-key-panel"] { background: #FBF9F4; border: 1px solid #E2DCCF; border-radius: 0.75rem;
                            padding: 1.1rem 1.2rem; }
  .eyebrow { font-size: 0.75rem; letter-spacing: 0.08em; text-transform: uppercase; color: #5F6B66;
             margin-bottom: 0.2rem; }
  .lede { color: #4A5551; font-size: 1.02rem; margin: -0.4rem 0 1.4rem 0; max-width: 54rem; }
  .callout { border-radius: 0.6rem; padding: 0.75rem 0.95rem; margin: 0.4rem 0 0.8rem 0; font-size: 0.93rem;
             line-height: 1.45; border: 1px solid; }
  .callout.note  { background: #E7F0EA; border-color: #B9D3C3; color: #1F3A30; }
  .callout.warn  { background: #FBF1D9; border-color: #EBCB82; color: #4A3A12; }
  .callout.alert { background: #F8E4DF; border-color: #E3AFA2; color: #5A231A; }
  .callout b { font-weight: 600; }
  .chip { display: inline-block; padding: 0.1rem 0.55rem; border-radius: 999px; font-size: 0.8rem;
          border: 1px solid #E2DCCF; background: #F6F3EC; margin: 0 0.3rem 0.3rem 0; color: #1F2A26; }
  .foot { color: #7A847F; font-size: 0.78rem; margin-top: 2.5rem; border-top: 1px solid #E2DCCF; padding-top: 0.6rem; }
</style>
""")


def callout(text: str, kind: str = "note", title: str | None = None):
    """A tinted box. Text is escaped; **bold** markers become bold."""
    body = html.escape(text).replace("\n", "<br>")
    body = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", body)
    head = f"<b>{html.escape(title)}</b><br>" if title else ""
    st.html(f'<div class="callout {kind}">{head}{body}</div>')


def page_header(step: str, title: str, lede: str):
    st.html(f'<div class="eyebrow">{html.escape(step)}</div>')
    st.title(title)
    st.html(f'<div class="lede">{html.escape(lede)}</div>')


def chips(items):
    st.html("".join(f'<span class="chip">{html.escape(str(i))}</span>' for i in items))


def git_version() -> str:
    try:
        run = lambda *a: subprocess.run(["git", "-C", str(HERE), *a], capture_output=True, text=True, timeout=5).stdout.strip()
        head = run("rev-parse", "--short", "HEAD")
        return f"{head} + uncommitted changes" if run("status", "--porcelain", "--", ".") else head
    except Exception:
        return "unknown"


def footer():
    st.html(f'<div class="foot">Code version {html.escape(git_version())} · last change '
            f'<code>{html.escape(LATEST_FILE)}</code> at {time.strftime("%H:%M:%S", time.localtime(LATEST_TIME))} · '
            'the page reloads when wnego/ or config/ changes</div>')


# ---------------------------------------------------------------------------
# Labels and help (anything not listed falls back to its field name)
LABELS = {
    "persona": ("Persona", "How much country background each delegate gets (keys in config/personas.yaml)."),
    "payoff": ("Payoff visibility", "hidden: no numbers; own: its own value and walk-away; full: everyone's."),
    "grounding": ("Document grounding", "rag: delegates get document passages and can search; none: no documents."),
    "scenario": ("Flow year", "Water available this year (flow_scenarios in config/basin.yaml)."),
    "legal": ("Legal advisor", "advisory: findings published; binding: an 'inconsistent' finding blocks adoption."),
    "events": ("News events", "Shocks injected mid-negotiation (config/events.yaml)."),
    "max_rounds": ("Rounds", "Rounds before the talks end with no deal."),
    "briefing_size": ("Passages per briefing", "Document passages each agent starts with."),
    "mediator_sees_payoffs": ("Mediator sees payoffs", "Show the mediator every country's value of the last text."),
    "decision_rule": ("Decision rule", "batna: delegates are told to reject any text worth less than no deal."),
    "reasoning": ("Reasoning style", "structured: every agent fills the same six steps; free: one free-text field."),
}
HIDDEN_FIELDS = {"hitl"}   # the human approval gate waits on a console prompt, which a web page cannot answer

AVATAR = {"delegate": "🧑‍💼", "Mediator": "🤝", "LegalAdvisor": "⚖️", "Environment": "🌊"}
VERDICT = {"supported": "✅ supported", "partial": "🟡 partly supported", "unsupported": "❌ unsupported",
           "bad_id": "⛔ cites a passage that does not exist", "uncited": "⚪ no citation"}
FINDING = {"consistent": "✅ consistent", "concern": "⚠️ concern", "inconsistent": "❌ inconsistent"}
PHASE_LABEL = {"briefing": "briefing agents from the database", "lookup": "searching the database",
               "what_if": "testing a what-if", "opening": "opening statements", "event": "news", "mediator_draft": "mediator drafting",
               "auto_repair": "river model repairing the draft", "legal_review": "legal review",
               "text": "text tabled", "response": "delegations voting", "legal_block": "legal block"}


# ---------------------------------------------------------------------------
# Data helpers (cached until the code changes)
@st.cache_resource(show_spinner=False)
def retriever_for(backend: str):
    cfg = load_configs()
    cfg["rag"]["backend"] = backend
    return make_retriever(cfg["rag"], ROOT)


@st.cache_data(show_spinner="Solving the bargaining benchmarks…")
def benchmark_table(scenario: str) -> pd.DataFrame:
    b = Basin.load(BASIN_PATH, scenario)
    d = b.no_deal_utilities()
    deals = {"No deal (walk-away)": d, "Protocol 566 shares": b.utilities(b.protocol_566_proposal()),
             "Nash bargaining solution": nash_bargaining_solution(b)["utilities"],
             "Utilitarian optimum": utilitarian_optimum(b)["utilities"]}
    rows = []
    for name, u in deals.items():
        row = {"Deal": name, **{c: round(u[c], 1) for c in b.countries}, "Total": round(sum(u.values()), 1)}
        row["Everyone at or above walk-away?"] = "✅ yes" if all(u[c] >= d[c] - 1e-6 for c in b.countries) else "❌ no"
        rows.append(row)
    return pd.DataFrame(rows).set_index("Deal")


@st.cache_data(show_spinner=False)
def corpus_docs() -> list[dict]:
    docs = []
    for f in sorted((ROOT / "corpus").glob("*.md")):
        text = f.read_text(encoding="utf-8", errors="ignore")
        meta, body = {}, text
        if text.startswith("---"):
            head, _, body = text[3:].partition("---")
            for line in head.splitlines():
                k, sep, v = line.partition(":")
                if sep:
                    meta[k.strip()] = v.strip()
        para = next((p.strip() for p in body.split("\n\n") if p.strip() and not p.strip().startswith("#")), "")
        docs.append({"title": meta.get("title", f.stem), "type": meta.get("doc_type", ""),
                     "summary": (para[:260] + "…") if len(para) > 260 else para})
    return docs


def database_status() -> tuple[bool, str]:
    """Is the team's Chroma database built, and how big is it?"""
    c = load_configs()["rag"]["chroma"]
    try:
        r = retriever_for("chroma")
        n = r.col.count()
        docs = {(m or {}).get("source") for m in r.col.get(include=["metadatas"])["metadatas"]}
        return True, (f"Connected to the team database: collection {c.get('collection')} · {n} passages "
                      f"from {len(docs)} documents.")
    except Exception as ex:
        return False, (f"The team database is not ready ({ex}). Build it with "
                       "`python -m wnego index --docs ../data --db ../chroma_db`.")


def placeholder_count() -> tuple[int, int]:
    text = BASIN_PATH.read_text(encoding="utf-8", errors="ignore")
    return text.count("ILLUSTRATIVE"), text.count("SOURCED")


def ollama_models() -> list[str] | None:
    try:
        import ollama
        listing = ollama.Client().list()
        models = listing.get("models", []) if isinstance(listing, dict) else getattr(listing, "models", [])
        names = [getattr(m, "model", None) or (m.get("model") or m.get("name")) for m in models]
        return sorted(n for n in names if n and "embed" not in n)
    except Exception:
        return None


def available_backends() -> dict[str, str]:
    """Live backends this machine can use, best first; mock always works."""
    out = {}
    if os.environ.get("ANTHROPIC_API_KEY"):
        out["claude"] = "Claude (live, Anthropic API)"
    if ollama_models():
        out["ollama"] = "Ollama (live, local models)"
    out["mock"] = "Mock (instant, rule-based)"
    return out


def live_gate(opts: dict) -> tuple[bool, str | None]:
    """Paid live runs (Claude) need the demo password, if one is set, and stay under the per-visitor limit."""
    if opts["spec"].get("type") != "anthropic":
        return True, None
    password = os.environ.get("DEMO_PASSWORD")
    if password and not st.session_state.get("unlocked"):
        typed = st.text_input("Demo password", type="password",
                              help="Live Claude runs are billed, so they are password protected.")
        if typed and hmac.compare_digest(typed, password):
            st.session_state["unlocked"] = True
        else:
            return False, ("Enter the demo password to run a live Claude negotiation. Recorded and mock runs "
                           "are open to everyone." if not typed else "That password is not right.")
    used = st.session_state.get("live_runs", 0)
    if used >= MAX_LIVE_RUNS:
        return False, f"This session has used its {MAX_LIVE_RUNS} live runs. Recorded and mock runs still work."
    return True, f"Live run {used + 1} of {MAX_LIVE_RUNS} for this session."


def recorded_runs() -> list[Path]:
    """Runs shipped with the app first, then local runs/ (not deployed)."""
    shipped = sorted(DEMO_RUNS.glob("*.json")) if DEMO_RUNS.exists() else []
    local = sorted((ROOT / "runs").rglob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True) \
        if (ROOT / "runs").exists() else []
    return shipped + [p for p in local if p.name != "run_info.json"]


def open_run(path: Path) -> bool:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "events" not in data or "condition" not in data:
        return False
    st.session_state["result"] = data
    st.session_state["source"] = f"recorded run · {path.name}"
    return True


def current_cond() -> tuple[Condition | None, Exception | None]:
    try:
        return Condition(**st.session_state.get("settings", {})), None
    except Exception as ex:
        return None, ex


# ---------------------------------------------------------------------------
# Rendering helpers
def as_text(v) -> str:
    if isinstance(v, dict):
        return "; ".join(f"{k}: {as_text(x)}" for k, x in v.items())
    if isinstance(v, (list, tuple)):
        return " | ".join(as_text(x) for x in v)
    return str(v)


def show_reasoning(reasoning, label="Reasoning (private)"):
    if not reasoning:
        return
    with st.expander(label):
        if isinstance(reasoning, dict):
            keys = [k for k in P.REASONING_KEYS if k in reasoning] + [k for k in reasoning if k not in P.REASONING_KEYS]
            for i, k in enumerate(keys, 1):
                st.markdown(f"**{i}. {k}**: {as_text(reasoning[k])}")
        else:
            st.markdown(as_text(reasoning))


def show_claims(claims):
    for c in claims or []:
        cite = f" `[{c['cite']}]`" if c.get("cite") else ""
        st.markdown(f"- {c.get('text', '')}{cite}: {VERDICT.get(c.get('verdict'), c.get('verdict', ''))}")
        if c.get("evidence"):
            st.caption(f"Evidence quoted by the verifier: “{c['evidence']}”")


def proposal_view(basin: Basin, prop: dict):
    p = basin.normalize(prop)
    df = pd.DataFrame({c: {"Growing season (km³)": a["growing"], "Winter (km³)": a["winter"],
                           "Total (km³)": a["growing"] + a["winter"]} for c, a in p["allocations"].items()}).T
    st.dataframe(df.round(2), width="stretch")
    delta, env = basin.delta_flow(p), basin.cfg["environmental_flow"]
    ok = all(delta[s] >= env[s] - 1e-6 for s in ("growing", "winter"))
    st.caption(f"Tajik storage shift to winter: {p['tajik_storage_shift']:.1f} km³ · Left for the Aral delta: "
               f"growing {delta['growing']:.1f}, winter {delta['winter']:.1f} km³ "
               f"(minimum {env['growing']:.1f} / {env['winter']:.1f}) {'✅' if ok else '❌ below the minimum'}")
    for t in p["transfers"]:
        st.markdown(f"- 💵 {t['from']} pays {t['to']} {float(t.get('amount', 0)):g} ({t.get('for', 'unspecified')})")
    for pr in p["provisions"]:
        st.markdown(f"- 📄 {pr}")


def value_table(utilities: dict, no_deal: dict) -> pd.DataFrame:
    rows = []
    for c, u in utilities.items():
        gap = u - no_deal[c]
        rows.append({"Country": c, "This text": round(u, 1), "Walk-away (no deal)": round(no_deal[c], 1),
                     "Margin": round(gap, 1),
                     "Verdict": "✅ better than no deal" if gap >= 0 else "❌ worse than no deal"})
    return pd.DataFrame(rows).set_index("Country")


AGENT_NAME = {"Mediator": "the mediator", "LegalAdvisor": "the legal advisor"}


def show_passages(passages):
    for p in passages:
        page = f", p.{p['page']}" if p.get("page") not in (None, "", 0) else ""
        st.markdown(f"**{p.get('title') or p['id']}**{page} · `{p['id']}`")
        st.caption(p.get("text", ""))


def render_event(e: dict, basin: Basin):
    ph, agent, out = e.get("phase"), e.get("agent", ""), e.get("output") or {}
    if ph == "briefing":
        ps = e.get("passages") or []
        who = AGENT_NAME.get(agent, agent)
        with st.expander(f"📚 Input: what {who} was given from the document database ({len(ps)} passages)"):
            show_passages(ps) if ps else st.caption("No documents (grounding is off).")
    elif ph == "lookup":
        ps = e.get("passages") or []
        qs = "; ".join(f"“{q}”" for q in e.get("queries", []))
        with st.expander(f"🔎 {agent} searched the database for {qs}: {len(ps)} passages, read next round"):
            show_passages(ps)
    elif ph == "what_if":
        st.caption(f"🧪 {agent} asked the river model to test a hypothetical text: "
                   + ("feasible ✅" if e.get("feasible") else "not feasible ❌") + " (result read next round)")
    elif ph == "event":
        callout(f"Round {e['round']}: {e.get('event')}", "warn", "News")
    elif ph in ("opening", "response"):
        with st.chat_message(agent, avatar=AVATAR["delegate"]):
            vote = out.get("vote")
            badge = "" if ph == "opening" else (" · ✅ **accept**" if vote == "accept" else " · ❌ **reject**")
            st.markdown(f"**{agent}** · {'opening statement' if ph == 'opening' else 'round ' + str(e['round'])}{badge}")
            st.write(out.get("statement", ""))
            if out.get("caucus"):
                st.caption(f"🔒 Private note to the mediator: {out['caucus']}")
            ask = out.get("ask") if ph == "opening" else out.get("amendments")
            if ask:
                with st.expander("Opening ask" if ph == "opening" else "Proposed changes"):
                    st.json(ask)
            show_claims(out.get("claims_checked"))
            show_reasoning(out.get("reasoning"))
    elif ph == "mediator_draft":
        with st.chat_message("Mediator", avatar=AVATAR["Mediator"]):
            st.markdown(f"**Mediator** · round {e['round']} draft" + (f" (attempt {e['attempt'] + 1})" if e.get("attempt") else ""))
            st.write(out.get("explanation", ""))
            if out.get("proposal"):
                proposal_view(basin, out["proposal"])
            if e.get("errors"):
                callout("The river model rejected this draft: " + "; ".join(e["errors"]), "alert")
            if out.get("caucus_replies"):
                with st.expander("🔒 Private replies to each delegation"):
                    for c, m in out["caucus_replies"].items():
                        st.markdown(f"**{c}:** {m}")
            show_claims(out.get("claims"))
            show_reasoning(out.get("reasoning"))
    elif ph == "auto_repair":
        callout("The mediator could not fix the draft, so the river model scaled allocations down to fit: "
                + "; ".join(e.get("errors") or []), "warn")
    elif ph == "legal_review":
        with st.chat_message("Legal advisor", avatar=AVATAR["LegalAdvisor"]):
            st.markdown(f"**Legal advisor** · round {e['round']}")
            st.write(out.get("summary", ""))
            rows = [{"Principle": a.get("principle"), "Finding": FINDING.get(a.get("finding"), a.get("finding")),
                     "Why": a.get("explanation", ""), "Source": a.get("cite") or ""}
                    for a in out.get("assessments", []) if isinstance(a, dict)]
            if rows:
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            show_reasoning(out.get("reasoning"))
    elif ph == "text":
        with st.chat_message("River model", avatar=AVATAR["Environment"]):
            st.markdown(f"**River model** · value of the round {e['round']} text to each country")
            st.caption("Computed by code, not by an agent. Delegates see these numbers only when payoffs are shown.")
            st.dataframe(value_table(e["utilities"], basin.no_deal_utilities()), width="stretch")
    elif ph == "legal_block":
        callout(f"Binding legal review blocks this text: inconsistent with {', '.join(e.get('principles', []))}.", "alert")


def render_feed(res: dict, basin: Basin):
    """Replay a finished run, one expander per round, the last one open."""
    by_round: dict[int, list] = {}
    for e in res["events"]:
        by_round.setdefault(e.get("round", 0), []).append(e)
    last = max(by_round)
    for r, events in sorted(by_round.items()):
        with st.expander("Briefing and opening statements" if r == 0 else f"Round {r}", expanded=r == last):
            for e in events:
                render_event(e, basin)


def last_text_event(res: dict) -> dict | None:
    return next((e for e in reversed(res["events"]) if e.get("phase") == "text"), None)


# ---------------------------------------------------------------------------
# Settings that survive moving between pages (widget state is dropped off-page)
def setting(name: str, widget, label: str, *, default=None, options=None, **kw):
    store = st.session_state.setdefault("settings_ui", {})
    key = f"w_{name}"
    if key not in st.session_state and name in store and (options is None or store[name] in options):
        st.session_state[key] = store[name]
    if options is not None:
        if key in st.session_state and st.session_state[key] not in options:
            del st.session_state[key]                   # the code no longer offers this value
        if key not in st.session_state:
            kw["index"] = options.index(default) if default in options else 0
        val = widget(label, options, key=key, **kw)
    else:
        if key not in st.session_state and default is not None:
            kw["value"] = default
        val = widget(label, key=key, **kw)
    store[name] = val
    return val


# ---------------------------------------------------------------------------
# Screens
def screen_past():
    page_header("Step 1 of 6 · Context", "Understand the past before the process",
                "The talks start from a 1987 allocation that no longer matches the basin. "
                "These are the documents every agent can cite.")
    b = Basin.load(BASIN_PATH)
    left, right = st.columns([2, 1], gap="large")
    with left:
        with st.container(key="card_past"):
            st.subheader("Protocol 566 shares")
            ref = b.cfg["protocol_566_km3"]
            total = sum(ref.values())
            st.dataframe(pd.DataFrame([{"Country": c, "Share (km³ a year)": v, "Share of total": f"{v / total:.1%}"}
                                       for c, v in ref.items()]).set_index("Country"), width="stretch")
            st.caption(f"Total {total:.1f} km³ a year, allocated under Soviet-era Protocol 566.")
            st.subheader("Document library")
            for d in corpus_docs():
                st.markdown(f"**{d['title']}**  \n<span class='chip'>{html.escape(d['type'])}</span>",
                            unsafe_allow_html=True)
                if d["summary"]:
                    st.caption(d["summary"])
    with right:
        with st.container(key="panel_past"):
            shipped = [p for p in recorded_runs() if p.parent == DEMO_RUNS]
            if shipped:
                st.markdown("**New here?**")
                if st.button("▶ Watch a recorded negotiation", type="primary", width="stretch", key="watch_ctx"):
                    if open_run(shipped[0]):
                        st.switch_page(PAGES["deal"])
                st.divider()
            st.markdown("**Who is at the table**")
            for c, par in b.cfg["countries"].items():
                st.markdown(f"**{c}**  \n{par.get('position', '')}")
            db_ok, db_note = database_status()
            callout(db_note + (" Agents can cite these in the negotiation." if db_ok else ""),
                    "note" if db_ok else "warn", "Team database")
            callout(f"The library on the left is the {len(corpus_docs())} short summaries used when the team "
                    "database is not selected. Claims the documents do not support are marked unverified "
                    "in the record.", "warn", "Evidence base")
    footer()


def screen_water():
    page_header("Step 2 of 6 · The river model", "Follow the water before its assumptions",
                "Code, not an agent, decides whether a deal is feasible and what it is worth to each country. "
                "Check its numbers before trusting any outcome.")
    b = Basin.load(BASIN_PATH)
    cond, _ = current_cond()
    scenario = cond.scenario if cond else "normal"
    left, right = st.columns([2, 1], gap="large")
    with left:
        with st.container(key="card_water"):
            st.subheader("Water available each year (km³)")
            st.dataframe(pd.DataFrame(b.cfg["flow_scenarios"]).T.rename(
                columns={"growing": "Growing season", "winter": "Winter"}), width="stretch")
            env = b.cfg["environmental_flow"]
            st.markdown(f"- Minimum flow to the Aral delta: **{env['growing']} km³** in the growing season, "
                        f"**{env['winter']} km³** in winter.\n"
                        f"- {b.cfg['storage_country']} can shift up to **{b.cfg['max_storage_shift']} km³** "
                        "from summer to winter for hydropower.")
            st.subheader(f"Reference deals ({scenario} year)")
            st.caption("Utility units. A deal everyone can accept leaves each country at or above its walk-away.")
            st.dataframe(benchmark_table(scenario), width="stretch")
    with right:
        with st.container(key="panel_water"):
            illustrative, sourced = placeholder_count()
            callout(f"{illustrative} values in config/basin.yaml are marked ILLUSTRATIVE and {sourced} are SOURCED. "
                    "Outcomes model the incentives; they are not measured values.", "warn", "Placeholder numbers")
            st.markdown("**How a country values a deal**")
            st.caption("Irrigation water (diminishing returns) + winter hydropower for the storage country "
                       "− a penalty if the Aral delta gets less than its minimum + side payments received.")
            callout("The Nash bargaining solution is the benchmark: it keeps every country above its walk-away.",
                    "note")
    footer()


def screen_setup():
    page_header("Step 3 of 6 · Setup", "Give the delegates a shared starting position",
                "Choose the model and the experiment settings, read what each delegation is told, then start the talks.")
    cfg = load_configs()
    flow_years = list(Basin.load(BASIN_PATH).cfg["flow_scenarios"])
    choices = {"persona": list(cfg["personas"]), "events": list(dict.fromkeys(["none", *cfg["events"]])),
               "scenario": flow_years, **{k: list(v) for k, v in Condition.CHOICES.items()}}
    left, right = st.columns([2, 1], gap="large")
    with left:
        with st.container(key="card_model"):
            st.subheader("Model")
            backends = available_backends()
            c1, c2 = st.columns(2)
            with c1:
                backend = setting("backend", st.selectbox, "Backend", options=list(backends),
                                  default=next(iter(backends)), format_func=lambda k: backends[k],
                                  help="Live backends run real language models; mock runs instant rule-based agents.")
            run = {"spec": {"name": "mock", "type": "mock"}, "verifier_spec": None, "temperature": 0.7,
                   "verifier_temperature": 0.0, "ready": True}
            if backend == "claude":
                with c2:
                    model = setting("claude_model", st.selectbox, "Agent model", options=CLAUDE_MODELS,
                                    default=CLAUDE_MODELS[0])
                c3, c4 = st.columns(2)
                with c3:
                    effort = setting("effort", st.selectbox, "Reasoning effort",
                                     options=["model default", "low", "medium", "high"], default="model default",
                                     help="Higher effort thinks longer, and costs more.")
                with c4:
                    vmodel = setting("claude_verifier", st.selectbox, "Verifier model",
                                     options=["same as agents", *CLAUDE_MODELS], default="same as agents")
                extra = {} if effort == "model default" else {"effort": effort}
                run["spec"] = {"name": model, "type": "anthropic", "model": model, **extra}
                if vmodel != "same as agents":
                    run["verifier_spec"] = {"name": vmodel, "type": "anthropic", "model": vmodel, **extra}
                st.caption("A run makes roughly 30 to 60 Claude calls, takes a few minutes, and is billed to the "
                           "API key. More rounds cost more.")
            elif backend == "ollama":
                models = ollama_models()
                if not models:
                    callout("Ollama is not reachable. Start it with `ollama serve`.", "alert")
                    run["ready"] = False
                else:
                    with c2:
                        default = next((m for m in models if m.startswith("qwen2.5:7b")), models[0])
                        model = setting("model", st.selectbox, "Agent model", options=models, default=default)
                    run["spec"] = {"name": model, "type": "ollama", "model": model}
                    c3, c4 = st.columns(2)
                    with c3:
                        run["temperature"] = setting("temperature", st.slider, "Agent temperature",
                                                     default=0.7, min_value=0.0, max_value=1.0, step=0.05)
                        vmodel = setting("verifier_model", st.selectbox, "Verifier model",
                                         options=["same as agents", *models], default="same as agents")
                    with c4:
                        run["verifier_temperature"] = setting("verifier_temperature", st.slider, "Verifier temperature",
                                                              default=0.0, min_value=0.0, max_value=1.0, step=0.05)
                    if vmodel != "same as agents":
                        run["verifier_spec"] = {"name": vmodel, "type": "ollama", "model": vmodel}
                    st.caption("A real run makes dozens of model calls and can take several minutes.")
            c5, c6 = st.columns(2)
            db_ok, db_note = database_status()
            with c5:
                run["rag_backend"] = setting("rag_backend", st.selectbox, "Document database",
                                             options=["chroma", "bm25"], default="chroma" if db_ok else "bm25",
                                             format_func=lambda k: {"chroma": "Team database (data/ folder)",
                                                                    "bm25": "Built-in summaries (corpus/)"}[k],
                                             help="Where the agents' briefings and searches come from.")
            if run["rag_backend"] == "chroma":
                callout(db_note, "note" if db_ok else "alert")
                run["ready"] = run["ready"] and db_ok
            with c6:
                run["seed"] = int(setting("seed", st.number_input, "Seed", default=0, min_value=0, max_value=10_000))

        with st.container(key="card_settings"):
            st.subheader("Experiment settings")
            values, cols = {}, st.columns(2)
            fields = [f for f in dataclasses.fields(Condition) if f.name not in HIDDEN_FIELDS]
            for i, f in enumerate(fields):
                label, help_ = LABELS.get(f.name, (f.name.replace("_", " ").capitalize(), None))
                with cols[i % 2]:
                    if f.name in choices:
                        values[f.name] = setting(f.name, st.selectbox, label, options=choices[f.name],
                                                 default=f.default, help=help_)
                    elif isinstance(f.default, bool):
                        values[f.name] = setting(f.name, st.checkbox, label, default=f.default, help=help_)
                    elif isinstance(f.default, int):
                        values[f.name] = int(setting(f.name, st.number_input, label, default=f.default,
                                                     min_value=1, max_value=50, help=help_))
                    else:
                        values[f.name] = setting(f.name, st.text_input, label, default=str(f.default), help=help_)
            st.session_state["settings"] = values
            st.session_state["run_options"] = run
    with right:
        cond, err = current_cond()
        with st.container(key="panel_setup"):
            st.markdown("**What each delegation is told**")
            goals = load_configs()["goals"]
            for c in Basin.load(BASIN_PATH).countries:
                g = goals.get(c, {})
                with st.expander(c):
                    st.caption(g.get("bloc", ""))
                    for o in g.get("objectives", []):
                        st.markdown(f"- {o}")
                    if g.get("red_lines"):
                        st.markdown("**Red lines**")
                        for r in g["red_lines"]:
                            st.markdown(f"- {r}")
                    if g.get("fallback"):
                        st.caption(f"Walk-away plan: {g['fallback']}")
            if err:
                callout(f"These settings are not valid for the current code: {err}", "alert")
            else:
                chips([f"{k}: {v}" for k, v in values.items() if k in Condition.CHOICES or k in ("persona", "scenario")])
            gate_ok, gate_note = live_gate(st.session_state["run_options"])
            if gate_note:
                callout(gate_note, "note" if gate_ok else "warn")
            if st.button("Start the negotiation", type="primary", width="stretch",
                         disabled=bool(err) or not st.session_state["run_options"]["ready"] or not gate_ok):
                st.session_state["pending_run"] = True
                st.switch_page(PAGES["deal"])
    footer()


def screen_deal():
    page_header("Step 4 of 6 · Negotiation", "A proposed sharing deal",
                "The mediator tables one text each round. The river model scores it, the legal advisor reviews it, "
                "and all four delegations vote at once.")
    left, right = st.columns([2, 1], gap="large")
    if st.session_state.pop("pending_run", False):
        cond, err = current_cond()
        opts = st.session_state.get("run_options")
        if err or not opts:
            callout(f"Set up the talks first: {err or 'no settings yet'}", "alert")
            return
        if opts["spec"].get("type") == "anthropic":
            ok, note = live_gate(opts)
            if not ok:
                callout(note or "Live runs are not available.", "warn")
                return
            st.session_state["live_runs"] = st.session_state.get("live_runs", 0) + 1
        run_cfg = load_configs()
        run_cfg["rag"]["backend"] = opts["rag_backend"]
        basin = Basin.load(BASIN_PATH, cond.scenario)
        with right:
            side = st.container(key="panel_live")
            with side:
                st.markdown("**Live**")
                latest = st.empty()
        with left:
            with st.status("Starting the negotiation…", expanded=True) as status:
                def on_event(e):
                    status.update(label=f"Round {e.get('round', 0)}: {PHASE_LABEL.get(e.get('phase'), e.get('phase'))}…")
                    render_event(e, basin)
                    if e.get("phase") == "text":
                        with latest.container():
                            st.caption(f"Value of the round {e['round']} text")
                            st.dataframe(value_table(e["utilities"], basin.no_deal_utilities())[["Margin", "Verdict"]],
                                         width="stretch")
                st.session_state["running"] = True
                try:
                    res = run_one(opts["spec"], cond, opts["seed"], opts["temperature"], configs=run_cfg,
                                  retriever=retriever_for(opts["rag_backend"]) if cond.grounding == "rag" else None,
                                  verifier_temperature=opts["verifier_temperature"],
                                  verifier_spec=opts["verifier_spec"], on_event=on_event)
                    status.update(label="Deal reached" if res["agreement"] else "No deal", state="complete",
                                  expanded=False)
                    st.session_state["result"] = res
                    st.session_state["source"] = f"live run · {opts['spec']['name']} · {time.strftime('%H:%M:%S')}"
                except Exception as ex:
                    status.update(label="The run failed", state="error", expanded=True)
                    st.exception(ex)
                finally:
                    st.session_state["running"] = False
        with right:
            if "result" in st.session_state:
                if st.button("See the outcome", type="primary", width="stretch"):
                    st.switch_page(PAGES["outcome"])
        footer()
        return

    res = st.session_state.get("result")
    if not res:
        with left:
            with st.container(key="card_empty_deal"):
                st.markdown("No negotiation yet. Watch a recorded one, or set up your own.")
                shipped = [p for p in recorded_runs() if p.parent == DEMO_RUNS]
                if shipped and st.button("▶ Watch a recorded negotiation", type="primary"):
                    if open_run(shipped[0]):
                        st.rerun()
                if st.button("Set up the talks"):
                    st.switch_page(PAGES["setup"])
        footer()
        return
    basin = Basin.load(BASIN_PATH, res["condition"].get("scenario", "normal"))
    with left:
        st.caption(f"Showing: {st.session_state.get('source', 'saved run')}")
        render_feed(res, basin)
    with right:
        with st.container(key="panel_deal"):
            ev = last_text_event(res)
            if ev:
                st.markdown(f"**Latest text (round {ev['round']})**")
                proposal_view(basin, ev["text"])
                st.dataframe(value_table(ev["utilities"], res["no_deal_utilities"])[["Margin", "Verdict"]],
                             width="stretch")
            if st.button("See the outcome", type="primary", width="stretch"):
                st.switch_page(PAGES["outcome"])
            if st.button("Run again", width="stretch"):
                st.session_state["pending_run"] = True
                st.rerun()
    footer()


def screen_outcome():
    res = st.session_state.get("result")
    title = ("Agreement reached." if res and res["agreement"] else "Partial convergence. Conditions remain.")
    page_header("Step 5 of 6 · Outcome", title,
                "Where each country ends up against its walk-away, how the votes moved, and how far the "
                "agents followed their own numbers.")
    if not res:
        callout("Run a negotiation, or open a past run, to see its outcome.", "note")
        footer()
        return
    basin = Basin.load(BASIN_PATH, res["condition"].get("scenario", "normal"))
    m = res.get("metrics", {})
    left, right = st.columns([2, 1], gap="large")
    with left:
        with st.container(key="card_outcome"):
            if res["agreement"]:
                callout(f"All four delegations accepted the round {res['rounds']} text.", "note", "Deal reached")
            else:
                accepted = [v["country"] for v in res.get("votes", []) if v["round"] == res["rounds"] and v["vote"] == "accept"]
                callout(f"No unanimous text after {res['rounds']} rounds. In the last round "
                        f"{len(accepted)} of {len(basin.countries)} accepted"
                        + (f" ({', '.join(accepted)})" if accepted else "")
                        + ". Every country falls back to its walk-away outcome.", "warn", "No deal")
            st.subheader("Final text" if res["agreement"] else "Last text tabled")
            proposal_view(basin, res["final_text"] if res["agreement"] else (res.get("last_text") or res["final_text"]))
            st.subheader("What each country ends up with")
            st.dataframe(value_table(res["final_utilities"], res["no_deal_utilities"]), width="stretch")
            if res.get("votes"):
                st.subheader("Votes by round")
                votes = pd.DataFrame(res["votes"])
                votes["vote"] = votes["vote"].map({"accept": "✅ accept", "reject": "❌ reject"})
                st.dataframe(votes.pivot(index="round", columns="country", values="vote"), width="stretch")
    with right:
        with st.container(key="panel_outcome"):
            fmt = lambda v: "n/a" if v is None or v != v else f"{v:.0%}"
            st.metric("Payoff-consistent votes", fmt(m.get("payoff_consistent_rate")),
                      help="Votes that accept exactly when the text beats walking away.")
            st.metric("Accepts below walk-away", fmt(m.get("below_batna_accept_rate")))
            st.metric("Six reasoning steps filled", fmt(m.get("reasoning_complete_rate")))
            st.metric("Claims supported by their source", fmt(m.get("claims_supported_rate")))
            with st.expander("All metrics"):
                st.dataframe(pd.Series(m, name="value").astype(str), width="stretch")
            with st.expander("Public record"):
                st.text("\n".join(res.get("public_record", [])))
            st.download_button("Download transcript (JSON)", json.dumps(res, indent=1, default=str),
                               file_name=f"negotiation_{time.strftime('%Y%m%d_%H%M%S')}.json",
                               mime="application/json", width="stretch")
            if st.button("Why it did not close" if not res["agreement"] else "Where agents disagreed",
                         type="primary", width="stretch"):
                st.switch_page(PAGES["disagree"])
    footer()


def screen_disagree():
    page_header("Step 6 of 6 · Learn from it", "Make the disagreement useful",
                "Who objected and why, what the legal advisor flagged, where agents voted against their own "
                "numbers, and what to try next.")
    res = st.session_state.get("result")
    if not res:
        callout("Run a negotiation first.", "note")
        footer()
        return
    basin = Basin.load(BASIN_PATH, res["condition"].get("scenario", "normal"))
    d = res["no_deal_utilities"]
    last_r = res["rounds"]
    replies = {e["agent"]: e["output"] for e in res["events"] if e.get("phase") == "response" and e.get("round") == last_r}
    ev = last_text_event(res)
    u = ev["utilities"] if ev else {}
    review = next((e["output"] for e in reversed(res["events"]) if e.get("phase") == "legal_review"), None)
    flagged = [a for a in (review or {}).get("assessments", []) if isinstance(a, dict)
               and a.get("finding") in ("concern", "inconsistent")]
    off_numbers = [v for v in res.get("votes", []) if v["round"] == last_r
                   and (v["vote"] == "accept") != (v["utility"] >= v["no_deal"])]
    below = {c: u[c] - d[c] for c in u if u[c] < d[c]}

    left, right = st.columns([2, 1], gap="large")
    with left:
        with st.container(key="card_disagree"):
            st.subheader(f"Objections in round {last_r}")
            rejecters = {c: o for c, o in replies.items() if o.get("vote") == "reject"}
            if not rejecters:
                st.markdown("Every delegation accepted the last text.")
            for c, o in rejecters.items():
                st.markdown(f"**{c}** rejected")
                r = o.get("reasoning")
                why = r.get("because") or r.get("rules") if isinstance(r, dict) else None
                st.write(why or o.get("statement", ""))
                if o.get("amendments"):
                    with st.expander("What it asked for"):
                        st.json(o["amendments"])
            st.subheader("Votes against the country's own numbers")
            if not off_numbers:
                st.markdown("None: every vote in the last round matched whether the text beat walking away.")
            for v in off_numbers:
                kind = "accepted a text worth less than no deal" if v["vote"] == "accept" else "rejected a text worth more than no deal"
                callout(f"**{v['country']}** {kind} ({v['utility']:.0f} against a walk-away of {v['no_deal']:.0f}).", "warn")
            st.subheader("Legal findings")
            if not flagged:
                st.markdown("No concerns in the last review." if review else "The legal advisor was off.")
            for a in flagged:
                callout(f"**{a.get('principle')}**: {a.get('explanation', '')}",
                        "alert" if a.get("finding") == "inconsistent" else "warn")
    with right:
        with st.container(key="panel_disagree"):
            st.markdown("**What to try next**")
            cond = res["condition"]
            tips = []
            for c, gap in sorted(below.items(), key=lambda kv: kv[1]):
                tips.append(f"{c} is {-gap:.0f} below its walk-away under the last text: it needs at least that much "
                            "more in water value or side payments.")
            if off_numbers and cond.get("decision_rule") == "none":
                tips.append("Agents voted against their own numbers. Try **Decision rule: batna**.")
            if off_numbers and cond.get("payoff") == "hidden":
                tips.append("Delegates could not see their values. Try **Payoff visibility: own**.")
            if any(a.get("finding") == "inconsistent" for a in flagged) and cond.get("legal") == "advisory":
                tips.append("The legal advisor found the text inconsistent but had no power. Try **Legal advisor: binding**.")
            m = res.get("metrics", {})
            if (m.get("claims_unsupported_rate") or 0) > 0.3:
                tips.append("Many claims were unsupported. Try a larger verifier model or more source documents.")
            if not tips:
                tips.append("Nothing stands out. Rerun with another seed to see whether the result holds.")
            for t in tips:
                callout(t, "note")
            if st.button("Change the setup", type="primary", width="stretch"):
                st.switch_page(PAGES["setup"])
    footer()


def screen_prompts():
    page_header("Inspect", "Prompts", "The exact prompts the agents receive with the current settings and code. "
                                      "The text shown is the Protocol 566 split, tabled in round 1.")
    cond, err = current_cond()
    if err:
        callout(f"Fix the settings first: {err}", "alert")
        return
    cfg = load_configs()
    captured: list[tuple[str, str]] = []

    class Capture(MockLLM):
        def complete_json(self, system, user, meta=None, retries=0):
            captured.append((system, user))
            return super().complete_json(system, user, meta)

    basin = Basin.load(BASIN_PATH, cond.scenario)
    ret = retriever_for("bm25") if cond.grounding == "rag" else None
    neg = Negotiation(Capture(), basin, cond, cfg["personas"], ret, cfg["goals"], cfg["events"], seed=0, parallel=False)
    text = basin.protocol_566_proposal()
    with st.container(key="card_prompts"):
        who = st.selectbox("Agent", [*basin.countries, "Mediator", "Legal advisor", "Verifier"])
        if who in basin.countries:
            neg.delegates[who].respond(text, 1, [])
        elif who == "Mediator":
            neg.mediator.draft(1, ["[Round 0] (opening statements appear here)"], {})
        elif who == "Legal advisor":
            if neg.legal:
                neg.legal.review(text, 1)
            else:
                st.info("The legal advisor is off in the current settings.")
        else:
            st.markdown("**Verifier system prompt**")
            st.code(P.VERIFIER_SYSTEM, language=None)
            st.markdown("**Verifier instructions** (sent after the claims and their cited passages)")
            st.code(P.VERIFIER_QUOTE_FIRST if cond.reasoning != "free" else P.VERIFIER_SCHEMA, language=None)
        for system, user in captured[-1:]:
            st.markdown("**System prompt**")
            st.code(system, language=None)
            st.markdown("**User prompt**")
            st.code(user, language=None)
    footer()


def screen_past_runs():
    page_header("Inspect", "Past runs", "Open a saved transcript to replay it on the negotiation and outcome screens.")
    files = recorded_runs()
    with st.container(key="card_past_runs"):
        pick = st.selectbox("Recorded runs", ["—", *[str(f.relative_to(ROOT)) for f in files[:200]]],
                            help="demo_runs/ ships with the app; runs/ holds your local runs.")
        upload = st.file_uploader("…or upload a transcript", type="json")
        if st.button("Open", type="primary"):
            try:
                if upload is not None:
                    data, src = json.load(upload), upload.name
                elif pick != "—":
                    data, src = json.loads((ROOT / pick).read_text(encoding="utf-8")), pick
                else:
                    data, src = None, None
                if data is not None:
                    if "events" not in data or "condition" not in data:
                        callout("That file is not a negotiation transcript.", "alert")
                    else:
                        st.session_state["result"], st.session_state["source"] = data, src
                        st.switch_page(PAGES["deal"])
            except Exception as ex:
                st.exception(ex)
    footer()


# ---------------------------------------------------------------------------
@st.fragment(run_every="3s")
def watch_code():
    # A config edit does not trigger Streamlit's own file watcher, so poll the fingerprint.
    if not st.session_state.get("running") and code_fingerprint()[0] != getattr(_state, "fp", None):
        st.rerun(scope="app")


if "settings" not in st.session_state:              # sensible defaults before the setup screen is visited
    st.session_state["settings"] = {f.name: f.default for f in dataclasses.fields(Condition) if f.name not in HIDDEN_FIELDS}
    st.session_state["run_options"] = {"spec": {"name": "mock", "type": "mock"}, "verifier_spec": None,
                                       "temperature": 0.7, "verifier_temperature": 0.0, "ready": True,
                                       "rag_backend": load_configs()["rag"].get("backend", "bm25"), "seed": 0}

SCREENS = [("past", screen_past, "Context", "context"), ("water", screen_water, "River model", "river"),
           ("setup", screen_setup, "Setup", "setup"), ("deal", screen_deal, "Negotiation", "negotiation"),
           ("outcome", screen_outcome, "Outcome", "outcome"), ("disagree", screen_disagree, "Disagreement", "disagreement"),
           ("prompts", screen_prompts, "Prompts", "prompts"), ("runs", screen_past_runs, "Past runs", "past-runs")]
START = os.environ.get("WNEGO_DEMO_START", "context")      # opening screen; used by the headless tests
PAGES = {key: st.Page(fn, title=title, url_path=url, default=url == START) for key, fn, title, url in SCREENS}
nav = st.navigation(list(PAGES.values()), position="top")
watch_code()
nav.run()
