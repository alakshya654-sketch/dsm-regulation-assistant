"""
Index: embed chunks, store them in Chroma, and build a BM25 keyword index.

Implements Design §4.4 (embeddings) and §4.5 (vector store). Retrieval logic lives in
retrieve.py; this module only builds and opens the indexes.

    embedder = load_embedder()                       # bge-small-en-v1.5, runs on CPU
    collection = build_vector_index(chunks, INDEX_DIR, embedder)
    keyword = KeywordIndex(chunks)

The vector index is persisted to disk (index/); the keyword index is rebuilt from
chunks.jsonl in under a second, so it is not stored.
"""
from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Protocol

import numpy as np

VERSION = "4b.1"

COLLECTION = "dsm_chunks"
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
# bge models expect this instruction in front of queries (not in front of passages).
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


# ---------------------------------------------------------------------------
# Embedders
# ---------------------------------------------------------------------------
class Embedder(Protocol):
    name: str
    def embed_passages(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """Local embedding model via sentence-transformers. Downloads once (~130 MB), then
    works offline. Vectors are normalised, so cosine similarity = dot product."""

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "cpu"):
        from sentence_transformers import SentenceTransformer
        self.name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.query_prefix = BGE_QUERY_PREFIX if "bge" in model_name.lower() else ""

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, batch_size=32, normalize_embeddings=True,
                                 show_progress_bar=len(texts) > 64)

    def embed_query(self, text: str) -> np.ndarray:
        return self.model.encode([self.query_prefix + text], normalize_embeddings=True)[0]


class HashingEmbedder:
    """Deterministic stand-in for tests: a bag-of-words hashed into a fixed-size vector.
    No download, no model — only used where the real model is unavailable."""

    def __init__(self, dim: int = 512):
        self.name = f"hashing-{dim}"
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in tokenize(text):
            v[hash_token(tok) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vec(t) for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._vec(text)


def hash_token(tok: str) -> int:
    """Stable across runs (Python's built-in hash() is randomised per process)."""
    h = 2166136261
    for ch in tok.encode("utf-8"):
        h = ((h ^ ch) * 16777619) & 0xFFFFFFFF
    return h


def load_embedder(model_name: str = DEFAULT_MODEL) -> Embedder:
    return SentenceTransformerEmbedder(model_name)


# ---------------------------------------------------------------------------
# Keyword index (BM25)
# ---------------------------------------------------------------------------
STOPWORDS = set("""a an and are as at be by for from has have in is it its of on or shall
such that the their this to under was were which will with""".split())


def tokenize(text: str) -> list[str]:
    """Lower-case words and numbers. Keeps regulation-style tokens such as '8', '4a', '50.05'
    and percentages, which matter for exact-term questions."""
    toks = re.findall(r"[a-z]+|\d+(?:\.\d+)?[a-z]?", text.lower())
    return [t for t in toks if t not in STOPWORDS]


class KeywordIndex:
    """BM25 over each chunk's breadcrumb + text (Design §4.6, step 2)."""

    def __init__(self, chunks):
        from rank_bm25 import BM25Okapi
        self.chunks = list(chunks)
        self.ids = [c.chunk_id for c in self.chunks]
        self.bm25 = BM25Okapi([tokenize(c.embed_text) for c in self.chunks])

    def search(self, query: str, k: int = 10, allowed: set[str] | None = None) -> list[tuple[str, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        order = np.argsort(-scores)
        out = []
        for i in order:
            if scores[i] <= 0:
                break
            if allowed is not None and self.ids[i] not in allowed:
                continue
            out.append((self.ids[i], float(scores[i])))
            if len(out) == k:
                break
        return out


# ---------------------------------------------------------------------------
# Vector index (Chroma)
# ---------------------------------------------------------------------------
def chunk_metadata(c) -> dict:
    """Chroma metadata: registry fields (already store-safe) plus chunk fields."""
    return {
        **c.meta,
        "chunk_id": c.chunk_id,
        "regulation_no": c.regulation_no,
        "clause_no": c.clause_no,
        "breadcrumb": c.breadcrumb,
        "page_start": c.page_start,
        "page_end": c.page_end,
        "is_table": c.is_table,
        "tokens": c.tokens,
    }


def _client(index_dir: Path):
    import chromadb
    from chromadb.config import Settings
    return chromadb.PersistentClient(path=str(index_dir), settings=Settings(anonymized_telemetry=False))


def build_vector_index(chunks, index_dir: Path, embedder: Embedder, batch: int = 64, log=print):
    """(Re)build the collection from scratch, so the index always matches chunks.jsonl."""
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    client = _client(index_dir)
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    col = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine", "embedder": embedder.name})
    start = time.time()
    vectors = embedder.embed_passages([c.embed_text for c in chunks])
    log(f"Embedded {len(chunks)} chunks with {embedder.name} in {time.time() - start:.1f} s")
    for i in range(0, len(chunks), batch):
        part = chunks[i:i + batch]
        col.add(ids=[c.chunk_id for c in part],
                documents=[c.text for c in part],
                embeddings=[v.tolist() for v in vectors[i:i + batch]],
                metadatas=[chunk_metadata(c) for c in part])
    return col


def open_vector_index(index_dir: Path):
    return _client(Path(index_dir)).get_collection(COLLECTION)
