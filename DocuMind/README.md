# 🧠 DocuMind — AI Document Intelligence & Research Assistant

> Upload research papers and PDFs, ask questions in natural language, and receive
> answers grounded in the exact pages and passages that support them.

---

## Architecture Overview

```
PDF Upload
    │
    ▼
[pdf_processor.py]  → extract pages, chunk text
    │
    ▼
[embedder.py]       → sentence-transformers → float32 vectors
    │                  FAISS index (saved per document)
    ▼
[database.py]       → SQLite: documents, chunks, conversations, messages, feedback
    │
    ▼
[retriever.py]      → vector search → top-K chunk IDs → fetch from DB
    │
    ▼
[llm.py]            → RAG prompt → Gemini API → grounded answer
    │
    ▼
[app.py]            → Streamlit UI (Chat | Library | Settings)
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.13 |
| Frontend | Streamlit |
| PDF Parsing | PyMuPDF (fitz) |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Vector Store | FAISS (CPU) |
| LLM | Google Gemini API |
| Database | SQLite |

---

## Quick Start

### 1. Clone & enter the project

```bash
cd DocuMind
```

### 2. Create & activate a virtual environment (recommended)

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure your API key

```bash
# Copy the example env file
copy .env.example .env   # Windows
cp .env.example .env     # macOS / Linux

# Edit .env and paste your Gemini API key
# Get one free at https://aistudio.google.com/app/apikey
```

### 5. Run the app

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## Project Structure

```
DocuMind/
├── app.py                  # Streamlit frontend (3 pages)
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── config.py           # Centralised settings (reads .env)
│   ├── database.py         # SQLite helpers (documents, chunks, conversations)
│   ├── pdf_processor.py    # PyMuPDF extraction + chunking
│   ├── embedder.py         # sentence-transformers + FAISS index management
│   ├── retriever.py        # RAG retrieval orchestrator
│   ├── ingestor.py         # End-to-end pipeline: PDF → DB + FAISS
│   └── llm.py              # Gemini API wrapper + RAG prompt builder
└── data/
    ├── uploads/            # Saved PDF files
    ├── indexes/            # FAISS index files (one per document)
    └── db/
        └── documind.db     # SQLite database
```

---

## Configuration Reference

All settings live in `.env` (copy from `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | *(required)* | Your Gemini API key |
| `GEMINI_MODEL` | `gemini-2.0-flash` | Gemini model to use |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | sentence-transformers model |
| `CHUNK_SIZE` | `800` | Characters per chunk |
| `CHUNK_OVERLAP` | `150` | Overlap between chunks |
| `TOP_K` | `5` | Passages retrieved per query |

---

## Roadmap (Next Steps)

- [ ] Multi-document cross-referencing in a single question
- [ ] Conversation history summarisation for long sessions
- [ ] Keyword search fallback (BM25) alongside vector search
- [ ] Export chat session as PDF report
- [ ] User authentication (for multi-user deployments)
- [ ] MySQL migration for production
