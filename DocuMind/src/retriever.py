"""
retriever.py — Orchestrates the RAG retrieval pipeline.

Given a user question and a set of document IDs, it:
1. Loads (or merges) the FAISS indexes for those documents.
2. Searches for top-k relevant chunk IDs.
3. Fetches the chunk text + metadata from the database.
4. Returns a list of passage dicts ready for the LLM prompt.
"""
from __future__ import annotations
from src import database as db
from src.embedder import load_index, search_index, merge_indexes
from src.config import TOP_K


def retrieve_passages(question: str,
                      doc_ids: list[int],
                      top_k: int = TOP_K) -> list[dict]:
    """
    Retrieve the most relevant passages for a question from the given documents.

    Parameters
    ----------
    question : User's natural-language question.
    doc_ids  : List of document IDs to search within.
    top_k    : Number of passages to retrieve.

    Returns
    -------
    List of passage dicts with keys:
        text, page_num, doc_id, doc_filename, chunk_id, score, source_label
    """
    if not doc_ids:
        return []

    # ── 1. Collect index paths ────────────────────────────────────────────────
    index_paths: list[str] = []
    doc_map: dict[int, str] = {}  # doc_id → filename

    for doc_id in doc_ids:
        doc = db.get_document(doc_id)
        if doc and doc.get("index_path"):
            index_paths.append(doc["index_path"])
            doc_map[doc_id] = doc["filename"]

    if not index_paths:
        return []

    # ── 2. Load / merge indexes ───────────────────────────────────────────────
    if len(index_paths) == 1:
        index = load_index(index_paths[0])
    else:
        index = merge_indexes(index_paths)

    if index is None:
        return []

    # ── 3. Vector search ──────────────────────────────────────────────────────
    results = search_index(index, question, top_k=top_k)

    if not results:
        return []

    # ── 4. Fetch chunk metadata from DB ───────────────────────────────────────
    chunk_ids = [chunk_id for chunk_id, _score in results]
    score_map = dict(results)
    chunks = db.get_chunks_by_ids(chunk_ids)

    # Build a map: chunk_id → doc_id (via a quick all-docs lookup)
    all_docs = {d["id"]: d["filename"] for d in db.get_all_documents()}

    passages: list[dict] = []
    for i, chunk in enumerate(chunks, 1):
        chunk_id = chunk["id"]
        doc_id = chunk["doc_id"]
        passages.append({
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "doc_filename": all_docs.get(doc_id, "Unknown"),
            "page_num": chunk["page_num"],
            "text": chunk["text"],
            "score": score_map.get(chunk_id, 0.0),
            "source_label": f"Source {i}",
        })

    # Sort by score descending
    passages.sort(key=lambda p: p["score"], reverse=True)
    return passages
