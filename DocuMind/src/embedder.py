"""
embedder.py — Sentence-transformer embeddings + FAISS index management.

Responsibilities
----------------
1. Load the embedding model once (cached with @st.cache_resource or module-level).
2. Encode a list of text strings into float32 numpy vectors.
3. Build / save / load a FAISS flat-L2 index for a document.
4. Search an index and return the top-k chunk indices.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from src.config import EMBEDDING_MODEL, INDEX_DIR, TOP_K

# ── Model singleton ──────────────────────────────────────────────────────────
_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """Return a cached SentenceTransformer model (lazy-loaded)."""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


# ── Embedding helpers ─────────────────────────────────────────────────────────

def embed_texts(texts: list[str], batch_size: int = 64,
                show_progress: bool = False) -> np.ndarray:
    """
    Encode a list of texts into a 2-D float32 numpy array.
    Shape: (len(texts), embedding_dim).
    """
    model = get_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,   # cosine similarity via dot-product
    )
    return embeddings.astype(np.float32)


# ── FAISS index helpers ───────────────────────────────────────────────────────

def _index_path(doc_id: int) -> Path:
    """Return the FAISS index file path for a given document id."""
    return INDEX_DIR / f"doc_{doc_id}.faiss"


def build_index(doc_id: int, texts: list[str],
                chunk_db_ids: list[int]) -> Path:
    """
    Embed texts, build a flat-IP FAISS index (inner product = cosine since
    embeddings are normalised), and persist it to disk.

    Parameters
    ----------
    doc_id      : DB document id (used to name the index file).
    texts       : list of chunk text strings in order.
    chunk_db_ids: list of SQLite chunk.id values in the same order as texts.

    Returns
    -------
    Path to the saved index file.
    """
    embeddings = embed_texts(texts, show_progress=True)
    dim = embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)           # inner product (cosine)
    index = faiss.IndexIDMap(index)          # attach custom IDs
    ids = np.array(chunk_db_ids, dtype=np.int64)
    index.add_with_ids(embeddings, ids)

    idx_path = _index_path(doc_id)
    faiss.write_index(index, str(idx_path))
    return idx_path


def load_index(index_path: str | Path) -> faiss.Index:
    """Load a FAISS index from disk."""
    return faiss.read_index(str(index_path))


def search_index(index: faiss.Index, query: str,
                 top_k: int = TOP_K) -> list[tuple[int, float]]:
    """
    Search the FAISS index for the top-k most similar chunks.

    Returns
    -------
    List of (chunk_db_id, similarity_score) tuples, sorted by score descending.
    """
    query_vec = embed_texts([query])           # shape (1, dim)
    scores, ids = index.search(query_vec, top_k)

    results: list[tuple[int, float]] = []
    for chunk_id, score in zip(ids[0], scores[0]):
        if chunk_id != -1:                     # -1 means no result
            results.append((int(chunk_id), float(score)))

    return results


def merge_indexes(index_paths: list[str | Path]) -> faiss.Index | None:
    """
    Merge multiple FAISS indexes (one per document) into a single in-memory
    index for multi-document search.
    """
    if not index_paths:
        return None

    merged: faiss.IndexIDMap | None = None

    for path in index_paths:
        idx = load_index(path)
        if merged is None:
            # Clone the first index structure
            dim = idx.d
            flat = faiss.IndexFlatIP(dim)
            merged = faiss.IndexIDMap(flat)
        # Extract all vectors + IDs and add to merged
        n = idx.ntotal
        if n == 0:
            continue
        vecs = faiss.rev_swig_ptr(idx.get_xb(), n * idx.d).reshape(n, idx.d)
        ids_arr = faiss.vector_to_array(idx.get_ids())
        merged.add_with_ids(vecs.copy(), ids_arr)

    return merged
