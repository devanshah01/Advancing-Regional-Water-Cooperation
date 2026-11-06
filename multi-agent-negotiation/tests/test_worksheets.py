"""Offline tests for worksheets, synthetic personas and the grounding evaluation."""
import copy
import json
import shutil
import tempfile
from pathlib import Path

from wnego import grounding_eval as G
from wnego import worksheets as W
from wnego.experiment import ROOT, load_configs
from wnego.llm import MockLLM
from wnego.report import render
from wnego.retrieval import BM25Retriever
from wnego.synthetic import SPEC, build

from test_chroma import fake_embed

BASE = ROOT / "config/worksheets"


def test_base_worksheets_pass_part_c():
    assert W.report(BASE) == {c: [] for c in ("Afghanistan", "Tajikistan", "Turkmenistan", "Uzbekistan")}


def test_part_c_catches_each_rule():
    ws = copy.deepcopy(W.load(BASE / "Uzbekistan.yaml"))
    ws["part_a"]["values"] = {"answer": "Uzbekistan demands its rightful share.", "source": ""}
    ws["part_b"]["fallback"]["answer"] = "We fear the canal."
    ws["part_b"]["red_lines"]["items"] = ["An unfair deal"]
    checks = {i["check"] for i in W.check(ws)}
    assert {"sourced", "neutral", "part_a_no_demands", "part_b_no_feelings", "checkable_red_lines"} <= checks


def test_worksheets_feed_persona_and_goals():
    cfg = load_configs()
    assert set(cfg["personas"]["worksheet"]) == {"Afghanistan", "Tajikistan", "Uzbekistan", "Turkmenistan"}
    g = cfg["goals"]["Tajikistan"]
    assert g["priorities"]["winter_storage"] == "critical" and len(g["objectives"]) == 3
    assert "source" not in cfg["personas"]["worksheet"]["Tajikistan"].lower()


def test_synthetic_personas_and_grounding_run():
    tmp = Path(tempfile.mkdtemp())
    try:
        build(BASE, tmp / "synthetic")
        sheets = W.load_dir(tmp / "synthetic")
        assert len(sheets) == len(SPEC) == 10
        assert sheets["P09_UZB_false_facts"]["synthetic"]["planted"]
        ret = BM25Retriever(ROOT / "corpus")
        df, head = G.run(tmp / "synthetic", MockLLM(), ret, fake_embed, ROOT / "corpus", tmp / "out", log=lambda *_: None)
        assert len(df) == 10 and head["claims"] > 0
        summary = json.loads((tmp / "out/summary.json").read_text())
        assert summary["headline"]["personas"] == 10
        by = df.set_index("id")
        assert by.loc["P08_AFG_unsourced", "ws_source_coverage"] == 0
        assert by.loc["P05_UZB_loaded", "part_c_issues"] > 0
        page = render(tmp / "out", out=tmp / "report.html").read_text(encoding="utf-8")
        assert "<svg" in page and "P09" in page
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_number_extraction():
    assert G.numbers("285 km, 16.02 bcm, 1,300 km and 2 outcomes in 2026") == {"285", "16.02", "1300", "2026"}
