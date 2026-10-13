"""
database.py — SQLite helpers for DocuMind.

Tables
------
documents       — metadata for each uploaded file
chunks          — text chunks extracted from documents
conversations   — chat sessions
messages        — individual Q&A turns within a session
feedback        — thumbs-up / thumbs-down on answers
"""
import sqlite3
import json
from datetime import datetime, timezone
from pathlib import Path
from contextlib import contextmanager
from src.config import DB_PATH


# ── Connection factory ────────────────────────────────────────────────────────

@contextmanager
def get_connection():
    """Yield a SQLite connection with row_factory set to Row."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")   # better concurrent read/write
    conn.execute("PRAGMA foreign_keys=ON;")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── Schema creation ───────────────────────────────────────────────────────────

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS documents (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    filename    TEXT    NOT NULL,
    filepath    TEXT    NOT NULL UNIQUE,
    num_pages   INTEGER,
    num_chunks  INTEGER,
    index_path  TEXT,                        -- path to FAISS index file
    uploaded_at TEXT    NOT NULL,
    meta_json   TEXT    DEFAULT '{}'         -- arbitrary extra metadata
);

CREATE TABLE IF NOT EXISTS chunks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id      INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    page_num    INTEGER,
    text        TEXT    NOT NULL,
    char_start  INTEGER,
    char_end    INTEGER
);

CREATE TABLE IF NOT EXISTS conversations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    title       TEXT,
    doc_ids_json TEXT DEFAULT '[]',           -- JSON list of document IDs in scope
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role            TEXT NOT NULL CHECK(role IN ('user','assistant')),
    content         TEXT NOT NULL,
    sources_json    TEXT DEFAULT '[]',         -- JSON list of cited chunk IDs
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS feedback (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id  INTEGER NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    rating      INTEGER CHECK(rating IN (-1, 1)),  -- -1 = bad, 1 = good
    comment     TEXT,
    created_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc_id     ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_messages_conv_id  ON messages(conversation_id);
"""


def init_db() -> None:
    """Create all tables if they don't exist yet."""
    with get_connection() as conn:
        conn.executescript(_SCHEMA_SQL)


# ── Document helpers ──────────────────────────────────────────────────────────

def insert_document(filename: str, filepath: str, num_pages: int = 0,
                    num_chunks: int = 0, index_path: str = "",
                    meta: dict | None = None) -> int:
    """Insert a new document record and return its id."""
    with get_connection() as conn:
        cur = conn.execute(
            """INSERT INTO documents (filename, filepath, num_pages, num_chunks,
               index_path, uploaded_at, meta_json)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (filename, filepath, num_pages, num_chunks, index_path,
             _now(), json.dumps(meta or {}))
        )
        return cur.lastrowid


def update_document(doc_id: int, **kwargs) -> None:
    """Update arbitrary columns on a document row."""
    allowed = {"num_pages", "num_chunks", "index_path", "meta_json"}
    updates = {k: v for k, v in kwargs.items() if k in allowed}
    if not updates:
        return
    cols = ", ".join(f"{k} = ?" for k in updates)
    vals = list(updates.values()) + [doc_id]
    with get_connection() as conn:
        conn.execute(f"UPDATE documents SET {cols} WHERE id = ?", vals)


def get_all_documents() -> list[dict]:
    """Return all documents as a list of dicts."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM documents ORDER BY uploaded_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]


def get_document(doc_id: int) -> dict | None:
    """Return a single document dict or None."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM documents WHERE id = ?", (doc_id,)
        ).fetchone()
        return dict(row) if row else None


def delete_document(doc_id: int) -> None:
    """Delete a document and cascade to chunks."""
    with get_connection() as conn:
        conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))


# ── Chunk helpers ─────────────────────────────────────────────────────────────

def insert_chunks(doc_id: int, chunks: list[dict]) -> None:
    """
    Bulk-insert chunks for a document.
    Each chunk dict: {chunk_index, page_num, text, char_start, char_end}
    """
    rows = [
        (doc_id, c["chunk_index"], c.get("page_num"), c["text"],
         c.get("char_start"), c.get("char_end"))
        for c in chunks
    ]
    with get_connection() as conn:
        conn.executemany(
            """INSERT INTO chunks (doc_id, chunk_index, page_num, text,
               char_start, char_end) VALUES (?, ?, ?, ?, ?, ?)""",
            rows,
        )


def get_chunks_for_doc(doc_id: int) -> list[dict]:
    """Return all chunks belonging to a document."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM chunks WHERE doc_id = ? ORDER BY chunk_index",
            (doc_id,)
        ).fetchall()
        return [dict(r) for r in rows]


def get_chunks_by_ids(chunk_ids: list[int]) -> list[dict]:
    """Fetch specific chunks by their primary keys."""
    if not chunk_ids:
        return []
    placeholders = ",".join("?" * len(chunk_ids))
    with get_connection() as conn:
        rows = conn.execute(
            f"SELECT * FROM chunks WHERE id IN ({placeholders})", chunk_ids
        ).fetchall()
        return [dict(r) for r in rows]


# ── Conversation helpers ──────────────────────────────────────────────────────

def create_conversation(title: str = "New chat",
                        doc_ids: list[int] | None = None) -> int:
    """Create a new conversation and return its id."""
    with get_connection() as conn:
        cur = conn.execute(
            """INSERT INTO conversations (title, doc_ids_json, created_at)
               VALUES (?, ?, ?)""",
            (title, json.dumps(doc_ids or []), _now())
        )
        return cur.lastrowid


def get_all_conversations() -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM conversations ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]


def get_conversation_messages(conv_id: int) -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at",
            (conv_id,)
        ).fetchall()
        return [dict(r) for r in rows]


def add_message(conv_id: int, role: str, content: str,
                sources: list[int] | None = None) -> int:
    with get_connection() as conn:
        cur = conn.execute(
            """INSERT INTO messages (conversation_id, role, content,
               sources_json, created_at) VALUES (?, ?, ?, ?, ?)""",
            (conv_id, role, content, json.dumps(sources or []), _now())
        )
        return cur.lastrowid


# ── Feedback helpers ──────────────────────────────────────────────────────────

def add_feedback(message_id: int, rating: int, comment: str = "") -> None:
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO feedback (message_id, rating, comment, created_at)
               VALUES (?, ?, ?, ?)""",
            (message_id, rating, comment, _now())
        )


# ── Internal utilities ────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
