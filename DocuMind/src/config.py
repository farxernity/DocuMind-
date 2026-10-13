"""
config.py — Centralised configuration loaded from environment / .env file.
All other modules import from here so settings live in one place.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# ── Load .env if it exists ────────────────────────────────────────────────────
_root = Path(__file__).resolve().parent.parent  # project root
load_dotenv(_root / ".env")


# ── Gemini ─────────────────────────────────────────────────────────────────────
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

# ── Embeddings ────────────────────────────────────────────────────────────────
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 800))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 150))

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K: int = int(os.getenv("TOP_K", 5))

# ── Storage paths ─────────────────────────────────────────────────────────────
DB_PATH: Path = _root / os.getenv("DB_PATH", "data/db/documind.db")
INDEX_DIR: Path = _root / os.getenv("INDEX_DIR", "data/indexes")
UPLOAD_DIR: Path = _root / os.getenv("UPLOAD_DIR", "data/uploads")

# Ensure directories exist at import time
for _d in (DB_PATH.parent, INDEX_DIR, UPLOAD_DIR):
    _d.mkdir(parents=True, exist_ok=True)
