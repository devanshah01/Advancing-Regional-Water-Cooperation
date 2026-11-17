"""Shared RAG pipeline for the Amu Darya Basin decision-support prototype.

Used by the Streamlit app (app.py) and the evaluation scripts, so the numbers you
report come from exactly the code the app runs.

Defaults match the original app: llama3.2 (3B) via Ollama, nomic-embed-text,
500-character chunks with a 50-character overlap, top-3 retrieval, and the original
"ONLY the provided context" prompt (style="basic").
"""
import glob
import hashlib
import math
import os
import re
import time

import chromadb
from pypdf import PdfReader

# --- CONFIGURATION ---
MODEL_NAME = "llama3.2"
EMBED_MODEL = "nomic-embed-text"
DB_PATH = os.environ.get("RAG_DB", "./chroma_db")
DOCS_FOLDER = os.environ.get("RAG_DOCS", ".")
COLLECTION_NAME = "local_docs"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
DEFAULT_K = 3
FALLBACK = "I cannot find the answer in the provided documents."
DRY_RUN = os.environ.get("RAG_DRY_RUN") == "1"  # fake models, for testing the plumbing only

# Files in the docs folder that are not knowledge-base content.
EXCLUDE_FILES = {"requirements.txt"}


# ---------------------------------------------------------------- model calls
def ollama_embed(text, model=EMBED_MODEL):
    import ollama  # imported lazily so tests can run without Ollama installed
    return ollama.embeddings(model=model, prompt=text)["embedding"]


def ollama_generate(prompt, model=MODEL_NAME, temperature=None, seed=None):
    import ollama
    options = {}
    if temperature is not None:
        options["temperature"] = temperature
    if seed is not None:
        options["seed"] = seed
    resp = ollama.generate(model=model, prompt=prompt, options=options or None)
    return resp["response"]


def fake_embed(text, dim=64):
    """Bag-of-words hash embedding. NOT meaningful; only exercises the pipeline."""
    v = [0.0] * dim
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        v[int(hashlib.md5(w.encode()).hexdigest(), 16) % dim] += 1.0
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


def fake_generate(prompt, temperature=None, seed=None):
    if "VERIFIED CALCULATIONS" in prompt or "Stakeholder positions" in prompt:
        names = ["Kyrgyzstan", "Tajikistan", "Kazakhstan", "Uzbekistan", "Turkmenistan"]
        body = "\n".join(f"- {n}: Conditional - stub reason - stub condition" for n in names)
        return f"Recommended pathway: Prioritize cotton drip irrigation (stub).\n\nStakeholder positions:\n{body}\n\nMain risks and open questions: stub."
    if "Context:" not in prompt:
        return "Stub baseline answer."
    ctx = prompt.split("Context:")[1].split("Question:")[0].strip()
    return f"Stub answer based on: {ctx[:80]}"


def default_embed_fn():
    return fake_embed if DRY_RUN else ollama_embed


def default_generate_fn():
    return fake_generate if DRY_RUN else ollama_generate


# ------------------------------------------------------------------- documents
def clean_text(text):
    """Strip web-page chrome from PDFs saved from news sites (menus, footers, print headers)."""
    text = re.sub(r"The .{0,3}Daryo.{0,3} online publication.*?2013.2025", " ", text, flags=re.S)
    text = re.sub(r"^\s*\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{2} [AP]M .*$", "", text, flags=re.M)
    text = re.sub(r"^\s*https?://\S+.*$", "", text, flags=re.M)
    text = re.sub(r"^\s*Editions: English\s*$", "", text, flags=re.M)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _doc_paths(folder):
    return sorted(glob.glob(os.path.join(folder, "*.txt")) + glob.glob(os.path.join(folder, "*.pdf")))


def load_documents(folder=DOCS_FOLDER, clean=True):
    """Read every .txt and .pdf in `folder` (non-recursive), in sorted order."""
    documents = []
    for path in _doc_paths(folder):
        name = os.path.basename(path)
        if name in EXCLUDE_FILES:
            continue
        if path.lower().endswith(".txt"):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        else:
            try:
                reader = PdfReader(path)
                pages = [page.extract_text() or "" for page in reader.pages]
                text = "".join(p + "\n" for p in pages if p)
            except Exception as e:  # unreadable PDF: skip but say so
                print(f"Could not read {path}: {e}")
                continue
            if clean:
                text = clean_text(text)
        if text.strip():
            documents.append({"source": name, "text": text})
    return documents


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    return [text[s:s + chunk_size] for s in range(0, len(text), chunk_size - overlap)]


_TABLE_START = re.compile(r"^\s*Table\s+\d+\b", re.I)
_TABLE_STOP = re.compile(r"^\s*(Figure\s+\d+|Table\s+\d+)\b", re.I)
_DATE_ROW = re.compile(r"\d{2}\.\d{2}\.\d{4}")


def _is_prose(line):
    """A long, mostly alphabetic line: ends a table block."""
    words = line.split()
    letters = sum(ch.isalpha() for ch in line)
    return len(words) >= 12 and letters > 0.6 * max(len(line.strip()), 1)


def chunk_text_table_aware(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP, max_table=2000):
    """H1 chunking: a table ("Table N" caption up to the next caption, figure or prose paragraph)
    is kept in ONE chunk so a row and its column headers are never separated. A table longer than
    max_table characters is split by rows, and the caption and date header row are repeated on every
    piece. Text outside tables is chunked exactly like chunk_text()."""
    lines = text.splitlines()
    blocks, cur, kind = [], [], "text"
    seen_rows = False   # a prose line only ends a table after at least one numeric row was seen
    def flush():
        nonlocal cur, kind
        if cur:
            blocks.append((kind, "\n".join(cur)))
        cur = []
    for ln in lines:
        if kind == "text" and _TABLE_START.match(ln):
            flush(); kind = "table"; cur = [ln]; seen_rows = False
        elif kind == "table" and (_TABLE_STOP.match(ln) or (seen_rows and _is_prose(ln))):
            flush(); kind = "text"
            if _TABLE_START.match(ln):
                kind = "table"; seen_rows = False
            cur = [ln]
        else:
            if kind == "table" and len(re.findall(r"\d[\d.,]*", ln)) >= 3:
                seen_rows = True
            cur.append(ln)
    flush()
    chunks = []
    for kind, body in blocks:
        if kind == "text":
            if body.strip():
                chunks.extend(chunk_text(body, chunk_size, overlap))
            continue
        if len(body) <= max_table:
            chunks.append(body)
            continue
        blines = body.splitlines()
        head = [blines[0]]
        for ln in blines[1:6]:
            head.append(ln)
            if _DATE_ROW.search(ln):
                break
        header = "\n".join(head)
        piece = header
        for ln in blines[len(head):]:
            if len(piece) + len(ln) + 1 > max_table and piece != header:
                chunks.append(piece)
                piece = header
            piece += "\n" + ln
        if piece != header:
            chunks.append(piece)
    return chunks


def _fingerprint(folder, chunk_size, overlap, embed_model, clean, header=False, chunking="fixed"):
    """Changes whenever a document, the chunking, the cleaning or the embedding model changes."""
    h = hashlib.sha256()
    h.update(f"{chunk_size}|{overlap}|{embed_model}|{clean}|{DRY_RUN}|{header}|{chunking}".encode())
    for path in _doc_paths(folder):
        name = os.path.basename(path)
        if name in EXCLUDE_FILES:
            continue
        h.update(f"{name}|{os.path.getsize(path)}|{int(os.path.getmtime(path))}".encode())
    return h.hexdigest()


# ----------------------------------------------------------------------- index
def build_index(folder=DOCS_FOLDER, db_path=DB_PATH, embed_fn=None,
                chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP, embed_model=EMBED_MODEL,
                rebuild=False, clean=True, header=False, chunking="fixed"):
    """Return (collection, status, info).

    The index is rebuilt automatically when a document, chunk setting or cleaning
    option changes. status is "loaded_existing", "created_new" or "no_documents".
    chunking="table" keeps each "Table N" block in one chunk (hypothesis H1); "fixed" is the 500/50 baseline.
    header=True prefixes every chunk with "[Source: <file name>]" before embedding, which helps tell
    near-identical documents apart (e.g. the four monitoring reports that share one template).
    """
    embed_fn = embed_fn or default_embed_fn()
    os.makedirs(db_path, exist_ok=True)
    fp_file = os.path.join(db_path, "fingerprint.txt")
    fp = _fingerprint(folder, chunk_size, overlap, embed_model, clean, header, chunking)
    client = chromadb.PersistentClient(path=db_path)

    if not rebuild and os.path.exists(fp_file):
        with open(fp_file) as f:
            if f.read().strip() == fp:
                try:
                    col = client.get_collection(COLLECTION_NAME)
                    if col.count() > 0:
                        return col, "loaded_existing", _info(col)
                except Exception:
                    pass

    documents = load_documents(folder, clean=clean)
    if not documents:
        return None, "no_documents", {"documents": 0, "chunks": 0, "sources": []}

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    col = client.get_or_create_collection(COLLECTION_NAME)

    ids, embs, docs, metas = [], [], [], []
    n = 0
    for doc in documents:
        pieces = (chunk_text_table_aware(doc["text"], chunk_size, overlap) if chunking == "table"
                  else chunk_text(doc["text"], chunk_size, overlap))
        for chunk in pieces:
            if header:
                chunk = f"[Source: {doc['source']}]\n{chunk}"
            ids.append(str(n))
            embs.append(embed_fn(chunk))
            docs.append(chunk)
            metas.append({"source": doc["source"]})
            n += 1
    for i in range(0, len(ids), 500):
        col.add(ids=ids[i:i + 500], embeddings=embs[i:i + 500],
                documents=docs[i:i + 500], metadatas=metas[i:i + 500])
    with open(fp_file, "w") as f:
        f.write(fp)
    return col, "created_new", _info(col)


def _info(col):
    got = col.get(include=["metadatas"])
    counts = {}
    for m in got["metadatas"]:
        counts[m["source"]] = counts.get(m["source"], 0) + 1
    return {"documents": len(counts), "chunks": col.count(), "sources": sorted(counts.items())}


# ------------------------------------------------------------------- retrieval
def retrieve(collection, query_embedding, k=DEFAULT_K, sources=None):
    """Top-k chunks by embedding distance. `sources` optionally restricts to named files;
    if that restriction returns nothing, falls back to the whole index."""
    where = {"source": {"$in": list(sources)}} if sources else None
    res = collection.query(query_embeddings=[query_embedding], n_results=k, where=where)
    if where and not res["documents"][0]:
        res = collection.query(query_embeddings=[query_embedding], n_results=k)
    return [
        {"text": t, "source": m["source"], "distance": d}
        for t, m, d in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]


# --------------------------------------------------------------------- prompts
def build_prompt(query, chunks, style="basic"):
    """style="basic": the original prompt, unchanged.
    style="structured": adds an explicit Answer / Evidence / Calculation format.
    style="none": no retrieval, plain-model baseline (chunks ignored)."""
    context = "\n\n---\n\n".join(c["text"] for c in chunks)
    if style == "none":
        return f"Question: {query}\n"
    if style == "structured":
        return f"""You are a research assistant supporting a study of the Amu Darya Basin water crisis. Answer the question using ONLY the provided context below. If the context does not contain the answer, reply exactly: "{FALLBACK}"

Format your reply as:
Answer: <2-4 sentences answering the question>
Evidence: <the specific facts or figures from the context that you relied on>
Calculation: <if the question needs cost scaling, show each step; otherwise write N/A>

Context:
{context}

Question: {query}
"""
    return f"""You are a helpful assistant. Answer the question using ONLY the provided context below. If you don't know, say "{FALLBACK}"


Context:
{context}


Question: {query}
"""


def answer_query(query, collection=None, k=DEFAULT_K, style="basic",
                 embed_fn=None, generate_fn=None, temperature=None, seed=None,
                 query_embedding=None):
    """Run one question end to end. Pass collection=None (or style="none") for the baseline."""
    embed_fn = embed_fn or default_embed_fn()
    generate_fn = generate_fn or default_generate_fn()
    t0 = time.time()
    chunks = []
    if style != "none" and collection is not None:
        if query_embedding is None:
            query_embedding = embed_fn(query)
        chunks = retrieve(collection, query_embedding, k)
    prompt = build_prompt(query, chunks, style)
    answer = generate_fn(prompt, temperature=temperature, seed=seed)
    sources = sorted({c["source"] for c in chunks})
    return {"answer": answer, "chunks": chunks, "sources": sources,
            "prompt": prompt, "seconds": round(time.time() - t0, 2)}
