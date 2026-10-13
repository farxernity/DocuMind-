"""
ingestor.py — End-to-end pipeline: PDF → DB + FAISS index.

Call `ingest_document(filepath, filename)` from the Streamlit upload handler.
"""
from __future__ import annotations
from pathlib import Path
from src.pdf_processor import process_pdf
from src.embedder import build_index
from src import database as db


def ingest_document(filepath: str | Path, filename: str) -> int:
    """
    Full ingestion pipeline for a single PDF.

    Steps
    -----
    1. Parse PDF → extract pages and chunk text.
    2. Insert document record into SQLite.
    3. Insert chunk records into SQLite.
    4. Generate embeddings and build FAISS index.
    5. Update document record with index path and counts.

    Returns
    -------
    doc_id : The SQLite primary key of the newly inserted document.
    """
    filepath = Path(filepath)

    # ── Step 1: Process PDF ───────────────────────────────────────────────────
    num_pages, chunk_dicts = process_pdf(filepath)

    # ── Step 2: Insert document record ───────────────────────────────────────
    doc_id = db.insert_document(
        filename=filename,
        filepath=str(filepath),
        num_pages=num_pages,
        num_chunks=len(chunk_dicts),
    )

    # ── Step 3: Insert chunks ─────────────────────────────────────────────────
    db.insert_chunks(doc_id, chunk_dicts)

    # ── Step 4: Retrieve chunk DB ids in order ────────────────────────────────
    db_chunks = db.get_chunks_for_doc(doc_id)
    texts = [c["text"] for c in db_chunks]
    chunk_db_ids = [c["id"] for c in db_chunks]

    # ── Step 5: Build FAISS index ─────────────────────────────────────────────
    idx_path = build_index(doc_id, texts, chunk_db_ids)

    # ── Step 6: Update document with final stats ──────────────────────────────
    db.update_document(
        doc_id,
        num_chunks=len(db_chunks),
        index_path=str(idx_path),
    )

    return doc_id
