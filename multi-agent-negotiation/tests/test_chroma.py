"""Chroma integration tests. They rebuild a database exactly the way the team's
Streamlit RAG app does (collection `local_docs`, ids "0", "1", ..., metadata
{"source": filename}, 500 character chunks with 50 overlap) and run the agents
on it. A hashing embedder stands in for Ollama's nomic-embed-text, so no Ollama
server is needed; on a real machine the same code calls ollama.embeddings."""
import hashlib
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

chromadb = pytest.importorskip("chromadb")

from wnego.agents import Condition
from wnego.experiment import ROOT, run_one
from wnego.retrieval import ChromaRetriever, build_index, read_document, tokenize


def fake_embed(text: str, dim: int = 256) -> list[float]:
    v = np.zeros(dim)
    for t in tokenize(text):
        v[int(hashlib.md5(t.encode()).hexdigest(), 16) % dim] += 1
    n = np.linalg.norm(v)
    return (v / n if n else v).tolist()


def build_like_streamlit_app(docs: Path, db: Path):
    """Mirror of initialize_vector_db() in the team's app."""
    col = chromadb.PersistentClient(path=str(db)).get_or_create_collection("local_docs")
    doc_id = 0
    for path in sorted(docs.glob("*.md")):
        pages, _ = read_document(path)
        text = pages[0][1]
        for start in range(0, len(text), 450):
            chunk = text[start:start + 500]
            col.add(ids=[str(doc_id)], embeddings=[fake_embed(chunk)], documents=[chunk],
                    metadatas=[{"source": path.name}])
            doc_id += 1


@pytest.fixture()
def db():
    d = Path(tempfile.mkdtemp())
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_reads_the_apps_legacy_collection(db):
    build_like_streamlit_app(ROOT / "corpus", db)
    r = ChromaRetriever(db, "local_docs", embed_fn=fake_embed)
    hits = r.search("Qosh Tepa canal diversion", 3)
    assert hits and any(h.source == "qosh_tepa_canal.md" for h in hits)
    pid = hits[0].id                              # e.g. "qosh_tepa_canal#7"
    assert "#" in pid
    fresh = ChromaRetriever(db, "local_docs", embed_fn=fake_embed)   # a new process can still resolve the id
    assert fresh.get(pid).text == hits[0].text
    # legacy index has no doc_type: filtered search falls back to unfiltered
    assert r.multi_search(["significant harm"], where={"doc_type": "legal"})


def test_improved_index_is_stable_and_incremental(db):
    s1 = build_index(ROOT / "corpus", db, "amu_darya_v2", embed_fn=fake_embed, log=lambda *_: None)
    assert s1["indexed"] == 6 and s1["chunks"] > 0
    s2 = build_index(ROOT / "corpus", db, "amu_darya_v2", embed_fn=fake_embed, log=lambda *_: None)
    assert s2["indexed"] == 0 and s2["skipped"] == 6
    r = ChromaRetriever(db, "amu_darya_v2", embed_fn=fake_embed)
    legal = r.search("obligation not to cause significant harm", 3, where={"doc_type": "legal"})
    assert legal and all(p.meta["doc_type"] == "legal" for p in legal)
    assert legal[0].id.startswith(("un_watercourses", "protocol_566"))


def test_full_negotiation_on_chroma(db):
    build_like_streamlit_app(ROOT / "corpus", db)
    r = ChromaRetriever(db, "local_docs", embed_fn=fake_embed)
    res = run_one({"name": "mock", "type": "mock"}, Condition(max_rounds=4, events="energy_crisis"), seed=2, retriever=r)
    assert res["claims"] and all(c["verdict"] != "bad_id" for c in res["claims"])
    assert res["goal_history"] and res["goal_history"][0]["changes"]["Tajikistan"]["version"] == 2
    assert res["tool_calls"] and res["caucus"]
