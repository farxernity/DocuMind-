"""
pdf_processor.py — Extract and chunk text from PDF files using PyMuPDF.

Responsibilities
----------------
1. Open a PDF and extract text page-by-page.
2. Split the full text into overlapping chunks.
3. Return chunk dicts ready for embedding and DB insertion.
"""
from pathlib import Path
from dataclasses import dataclass, field
import pymupdf as fitz               # PyMuPDF (new import path)
from src.config import CHUNK_SIZE, CHUNK_OVERLAP


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class TextChunk:
    chunk_index: int
    page_num: int          # 1-based page number where this chunk begins
    text: str
    char_start: int        # character offset in the full concatenated text
    char_end: int


# ── Core functions ────────────────────────────────────────────────────────────

def extract_pages(pdf_path: str | Path) -> list[tuple[int, str]]:
    """
    Open a PDF and return a list of (page_number, page_text) tuples.
    Page numbers are 1-based.
    """
    pdf_path = Path(pdf_path)
    pages: list[tuple[int, str]] = []

    with fitz.open(str(pdf_path)) as doc:
        for page_idx, page in enumerate(doc):
            text = page.get_text("text")   # plain text; use "blocks" for layout
            pages.append((page_idx + 1, text))

    return pages


def build_full_text(pages: list[tuple[int, str]]) -> tuple[str, list[int]]:
    """
    Concatenate all page texts into a single string.
    Returns (full_text, page_start_offsets) where page_start_offsets[i]
    is the character offset where page i+1 begins.
    """
    parts: list[str] = []
    offsets: list[int] = []
    cursor = 0
    for _page_num, text in pages:
        offsets.append(cursor)
        parts.append(text)
        cursor += len(text)
    return "".join(parts), offsets


def _page_for_offset(char_offset: int, page_start_offsets: list[int]) -> int:
    """Return the 1-based page number that contains a given character offset."""
    page = 1
    for i, start in enumerate(page_start_offsets):
        if char_offset >= start:
            page = i + 1
        else:
            break
    return page


def chunk_text(full_text: str,
               page_start_offsets: list[int],
               chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> list[TextChunk]:
    """
    Split full_text into overlapping fixed-size chunks.
    Returns a list of TextChunk objects with page attribution.
    """
    chunks: list[TextChunk] = []
    start = 0
    idx = 0
    text_len = len(full_text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk_text_str = full_text[start:end].strip()

        if chunk_text_str:   # skip empty chunks
            page = _page_for_offset(start, page_start_offsets)
            chunks.append(TextChunk(
                chunk_index=idx,
                page_num=page,
                text=chunk_text_str,
                char_start=start,
                char_end=end,
            ))
            idx += 1

        # Advance with overlap
        start += chunk_size - overlap

    return chunks


def process_pdf(pdf_path: str | Path) -> tuple[int, list[dict]]:
    """
    High-level entry point: process a PDF and return
    (num_pages, list_of_chunk_dicts).

    Each chunk dict has keys matching the `chunks` DB table columns:
    chunk_index, page_num, text, char_start, char_end.
    """
    pdf_path = Path(pdf_path)
    pages = extract_pages(pdf_path)
    full_text, offsets = build_full_text(pages)
    chunks = chunk_text(full_text, offsets)

    chunk_dicts = [
        {
            "chunk_index": c.chunk_index,
            "page_num": c.page_num,
            "text": c.text,
            "char_start": c.char_start,
            "char_end": c.char_end,
        }
        for c in chunks
    ]
    return len(pages), chunk_dicts
