"""RAG layer. Two interchangeable retrievers behind one interface.

  ChromaRetriever  plugs into the team's existing pipeline: the same ChromaDB
                   folder, the same Ollama `nomic-embed-text` embeddings. It
                   reads the original `local_docs` collection as is, or the
                   improved `amu_darya_v2` collection built by `wnego index`.
  BM25Retriever    no services needed; used for offline tests and as a
                   transparent baseline.

Every passage has an id the agents cite and the verifier looks up again, so a
citation can always be checked against the exact text.
"""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import yaml

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = set("the a an and or of to in on for by with is are was were be as at from that this it its their".split())


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


@dataclass
class Passage:
    id: str
    title: str
    source: str
    text: str
    score: float = 0.0
    meta: dict = field(default_factory=dict)

    def cite_block(self) -> str:
        page = f", p.{self.meta['page']}" if self.meta.get("page") not in (None, "", 0) else ""
        return f"[{self.id}] ({self.title}{page}) {self.text}"


class BaseRetriever:
    def search(self, query: str, k: int = 4, where: dict | None = None) -> list[Passage]:
        raise NotImplementedError

    def get(self, pid: str) -> Passage | None:
        raise NotImplementedError

    def multi_search(self, queries: list[str], k_each: int = 3, limit: int = 8, where: dict | None = None) -> list[Passage]:
        seen, out = set(), []
        for q in queries:
            hits = self.search(q, k_each, where)
            if where and not hits:          # filter matched nothing (e.g. legacy index without metadata)
                hits = self.search(q, k_each)
            for p in hits:
                if p.id not in seen:
                    seen.add(p.id)
                    out.append(p)
        return out[:limit]


def _match(meta: dict, where: dict | None) -> bool:
    """Tiny subset of Chroma's filter syntax: {"key": value} and {"key": {"$in": [...]}}."""
    for k, v in (where or {}).items():
        have = meta.get(k)
        if isinstance(v, dict) and "$in" in v:
            if have not in v["$in"]:
                return False
        elif have != v:
            return False
    return True


# ---------------------------------------------------------------------------
# Reading and chunking documents (shared by BM25 and the Chroma indexer)
# ---------------------------------------------------------------------------
def read_document(path: Path) -> tuple[list[tuple[int, str]], dict]:
    """Return [(page_number, text)] and metadata from front matter, if any."""
    meta = {"title": path.stem.replace("_", " "), "source": path.name}
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        pages = [(i + 1, p.extract_text() or "") for i, p in enumerate(PdfReader(str(path)).pages)]
        return pages, meta
    text = path.read_text(encoding="utf-8", errors="ignore")
    if text.startswith("---"):
        _, front, text = text.split("---", 2)
        front_meta = {k: str(v) for k, v in (yaml.safe_load(front) or {}).items()}
        if "source" in front_meta:          # keep `source` = file name (the indexer matches on it)
            front_meta["url"] = front_meta.pop("source")
        meta.update(front_meta)
    return [(0, text)], meta


def chunk_text(text: str, max_chars: int = 1000, min_chars: int = 200) -> list[str]:
    """Paragraph aware chunks: whole paragraphs packed up to max_chars; long
    paragraphs split on sentence boundaries. Treaty clauses stay intact."""
    paras = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", text)]
    pieces = []
    for p in paras:
        if not p:
            continue
        if len(p) <= max_chars:
            pieces.append(p)
            continue
        buf = ""
        for s in re.split(r"(?<=[.;:])\s+", p):
            if buf and len(buf) + len(s) + 1 > max_chars:
                pieces.append(buf)
                buf = s
            else:
                buf = f"{buf} {s}".strip()
        if buf:
            pieces.append(buf)
    chunks, buf = [], ""
    for p in pieces:
        if buf and len(buf) + len(p) + 2 > max_chars:
            chunks.append(buf)
            buf = p
        else:
            buf = f"{buf}\n\n{p}".strip()
    if buf:
        if chunks and len(buf) < min_chars:
            chunks[-1] += "\n\n" + buf
        else:
            chunks.append(buf)
    return chunks


def load_sidecar(folder: Path) -> dict:
    """Optional docs_meta.yaml next to the documents:
         rogun_assessment.pdf: {doc_type: technical, countries: Tajikistan,Uzbekistan, year: 2014}
       doc_type values the agents filter on: legal, technical, background."""
    f = folder / "docs_meta.yaml"
    if not f.exists():
        return {}
    return yaml.safe_load(open(f)) or {}


def iter_chunks(folder: str | Path, max_chars: int = 1000):
    folder = Path(folder)
    side = load_sidecar(folder)
    for path in sorted(folder.glob("*")):
        if path.suffix.lower() not in {".md", ".txt", ".pdf"}:
            continue
        pages, meta = read_document(path)
        meta.update({k: str(v) for k, v in (side.get(path.name) or {}).items()})
        meta.setdefault("doc_type", "background")
        n = 0
        for page, text in pages:
            for chunk in chunk_text(text, max_chars):
                cid = f"{path.stem}#p{page}#c{n}" if page else f"{path.stem}#c{n}"
                yield cid, chunk, {**meta, "page": page, "chunk_id": cid}, path
                n += 1


# ---------------------------------------------------------------------------
class BM25Retriever(BaseRetriever):
    def __init__(self, corpus_dir: str | Path, max_chars: int = 1000):
        self.passages = [Passage(cid, m["title"], m["source"], text, meta=m)
                         for cid, text, m, _ in iter_chunks(corpus_dir, max_chars)]
        self.by_id = {p.id: p for p in self.passages}
        self.k1, self.b = 1.5, 0.75
        self.tf = [Counter(tokenize(p.text)) for p in self.passages]
        self.lens = [sum(t.values()) for t in self.tf]
        self.avg = sum(self.lens) / max(len(self.lens), 1)
        df = Counter(w for t in self.tf for w in t)
        n = len(self.passages)
        self.idf = {w: math.log(1 + (n - f + 0.5) / (f + 0.5)) for w, f in df.items()}

    def get(self, pid):
        return self.by_id.get(pid)

    def search(self, query, k=4, where=None):
        q = tokenize(query)
        scored = []
        for i, p in enumerate(self.passages):
            if not _match(p.meta, where):
                continue
            s = 0.0
            for w in q:
                f = self.tf[i].get(w, 0)
                if f:
                    s += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.lens[i] / self.avg))
            if s > 0:
                scored.append((s, i))
        scored.sort(reverse=True)
        return [Passage(**{**self.passages[i].__dict__, "score": round(s, 3)}) for s, i in scored[:k]]


Retriever = BM25Retriever   # backwards compatible name


# ---------------------------------------------------------------------------
def ollama_embedder(model: str = "nomic-embed-text", host: str | None = None) -> Callable[[str], list[float]]:
    """Same call the existing Streamlit app makes: ollama.embeddings(model, prompt)."""
    import ollama
    client = ollama.Client(host=host) if host else ollama
    return lambda text: client.embeddings(model=model, prompt=text)["embedding"]


class ChromaRetriever(BaseRetriever):
    """Reads a ChromaDB collection built by the team's pipeline.

    Works with both layouts:
      legacy `local_docs`  ids are "0", "1", ...; metadata has only `source`.
                           Passage ids become "<file stem>#<chroma id>". Those ids
                           change if the database is rebuilt, so rerun the
                           verifier against the same database you negotiated on.
      `amu_darya_v2`       built by `python -m wnego index`; ids are stable
                           ("rogun_assessment#p12#c3") and metadata carries page,
                           doc_type and countries for filtering.
    """

    def __init__(self, db_path: str | Path, collection: str = "local_docs", embed_model: str = "nomic-embed-text",
                 ollama_host: str | None = None, embed_fn: Callable | None = None):
        import chromadb
        self.client = chromadb.PersistentClient(path=str(Path(db_path).expanduser()))
        self.col = self.client.get_collection(collection)
        if self.col.count() == 0:
            raise RuntimeError(f"Chroma collection '{collection}' at {db_path} is empty")
        self.embed = embed_fn or ollama_embedder(embed_model, ollama_host)
        self._seen: dict[str, Passage] = {}
        self._qcache: dict[str, list[float]] = {}

    def _passage(self, cid, doc, meta, dist=None) -> Passage:
        meta = dict(meta or {})
        src = meta.get("source", "document")
        pid = meta.get("chunk_id") or f"{Path(src).stem}#{cid}"
        p = Passage(pid, meta.get("title", Path(src).stem.replace("_", " ")), src, doc,
                    score=round(1 - dist, 3) if dist is not None else 0.0, meta={**meta, "chroma_id": cid})
        self._seen[pid] = p
        return p

    def search(self, query, k=4, where=None):
        if query not in self._qcache:
            self._qcache[query] = self.embed(query)
        kw = dict(query_embeddings=[self._qcache[query]], n_results=k, include=["documents", "metadatas", "distances"])
        if where:
            kw["where"] = where if len(where) == 1 else {"$and": [{k2: v} for k2, v in where.items()]}
        try:
            r = self.col.query(**kw)
        except Exception:
            if not where:
                raise
            return []        # unknown metadata key in a legacy index; multi_search retries unfiltered
        return [self._passage(c, d, m, s) for c, d, m, s in
                zip(r["ids"][0], r["documents"][0], r["metadatas"][0], r["distances"][0])]

    def get(self, pid):
        if pid in self._seen:
            return self._seen[pid]
        for cid in (pid, pid.rsplit("#", 1)[-1]):      # v2 id is the chroma id; legacy id ends in it
            r = self.col.get(ids=[cid], include=["documents", "metadatas"])
            if r["ids"]:
                return self._passage(r["ids"][0], r["documents"][0], r["metadatas"][0])
        return None


def build_index(docs_folder: str | Path, db_path: str | Path, collection: str = "amu_darya_v2",
                embed_fn: Callable | None = None, max_chars: int = 1000, log=print) -> dict:
    """Build or update an improved collection next to the team's `local_docs`.

    Fixes in the original indexer: stable chunk ids, page numbers, doc_type and
    country metadata, paragraph aware chunks, and incremental updates (a changed
    or new file is reindexed; the original only indexed when the DB was empty).
    The Streamlit app keeps working; point its collection name here to share it.
    """
    import chromadb
    embed = embed_fn or ollama_embedder()
    # cosine space: nomic-embed-text vectors are not unit length, and Chroma's default (l2)
    # both ranks them by norm and makes the 1 - distance score meaningless
    col = chromadb.PersistentClient(path=str(Path(db_path).expanduser())).get_or_create_collection(
        collection, metadata={"hnsw:space": "cosine"})
    by_file: dict[Path, list] = {}
    for cid, text, meta, path in iter_chunks(docs_folder, max_chars):
        by_file.setdefault(path, []).append((cid, text, meta))
    stats = {"indexed": 0, "skipped": 0, "removed": 0, "chunks": 0}
    present = {p.name for p in by_file}
    existing = col.get(include=["metadatas"])
    old_sources = {m.get("source") for m in existing["metadatas"]}
    for gone in old_sources - present:
        col.delete(where={"source": gone})
        stats["removed"] += 1
    for path, chunks in by_file.items():
        digest = hashlib.sha1(path.read_bytes()).hexdigest()
        prev = col.get(where={"source": path.name}, include=["metadatas"], limit=1)
        if prev["ids"] and prev["metadatas"][0].get("file_hash") == digest:
            stats["skipped"] += 1
            continue
        if prev["ids"]:
            col.delete(where={"source": path.name})
        col.add(ids=[c for c, _, _ in chunks], documents=[t for _, t, _ in chunks],
                embeddings=[embed(t) for _, t, _ in chunks],
                metadatas=[{**m, "file_hash": digest} for _, _, m in chunks])
        stats["indexed"] += 1
        stats["chunks"] += len(chunks)
        log(f"indexed {path.name}: {len(chunks)} chunks")
    return stats


def make_retriever(cfg: dict, root: Path) -> BaseRetriever | None:
    """cfg comes from config/rag.yaml."""
    kind = cfg.get("backend", "bm25")
    if kind == "none":
        return None
    if kind == "bm25":
        return BM25Retriever(root / cfg.get("corpus_dir", "corpus"))
    if kind == "chroma":
        c = cfg["chroma"]
        db = Path(c["db_path"]).expanduser()
        return ChromaRetriever(db if db.is_absolute() else root / db, c.get("collection", "local_docs"),
                               c.get("embed_model", "nomic-embed-text"), c.get("ollama_host"))
    raise ValueError(f"unknown retriever backend {kind}")
