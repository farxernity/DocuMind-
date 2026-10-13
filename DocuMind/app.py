"""
app.py — DocuMind Streamlit frontend.

Run with:
    streamlit run app.py
"""
from __future__ import annotations
import json
import shutil
import streamlit as st
from pathlib import Path

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="DocuMind — AI Document Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Local imports (after st.set_page_config) ──────────────────────────────────
from src.config import UPLOAD_DIR, GEMINI_API_KEY
from src import database as db
from src.ingestor import ingest_document
from src.retriever import retrieve_passages
from src.llm import generate_answer

# ── DB init ───────────────────────────────────────────────────────────────────
db.init_db()

# ════════════════════════════════════════════════════════════════════════════════
# Custom CSS
# ════════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Background ── */
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    min-height: 100vh;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(20px);
    border-right: 1px solid rgba(255,255,255,0.08);
}

[data-testid="stSidebar"] * {
    color: #e0e0ff !important;
}

/* ── Cards ── */
.dm-card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(10px);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.dm-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(120,90,255,0.2);
}

/* ── Hero header ── */
.dm-hero {
    text-align: center;
    padding: 2.5rem 0 1.5rem 0;
}
.dm-hero h1 {
    font-size: 3rem;
    font-weight: 700;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.4rem;
}
.dm-hero p {
    color: rgba(255,255,255,0.55);
    font-size: 1.05rem;
    margin: 0;
}

/* ── Section headers ── */
.dm-section-title {
    color: #a78bfa;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.75rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid rgba(167,139,250,0.25);
}

/* ── Chat bubbles ── */
.dm-user-bubble {
    background: linear-gradient(135deg, #7c3aed, #4f46e5);
    border-radius: 18px 18px 4px 18px;
    padding: 0.9rem 1.2rem;
    color: #fff;
    margin-bottom: 0.5rem;
    max-width: 80%;
    margin-left: auto;
    box-shadow: 0 4px 15px rgba(124,58,237,0.35);
}
.dm-assistant-bubble {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 18px 18px 18px 4px;
    padding: 0.9rem 1.2rem;
    color: #e2e8f0;
    margin-bottom: 0.5rem;
    max-width: 85%;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
}

/* ── Source citation badge ── */
.dm-source-badge {
    display: inline-block;
    background: rgba(96,165,250,0.15);
    border: 1px solid rgba(96,165,250,0.3);
    border-radius: 20px;
    padding: 0.25rem 0.75rem;
    font-size: 0.78rem;
    color: #93c5fd;
    margin: 0.25rem 0.25rem 0 0;
    cursor: default;
}

/* ── Upload drop zone ── */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.03);
    border: 2px dashed rgba(167,139,250,0.4);
    border-radius: 14px;
    transition: border-color 0.2s ease;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(167,139,250,0.8);
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: opacity 0.2s ease, transform 0.15s ease !important;
}
.stButton > button:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
}

/* ── Inputs ── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: #7c3aed !important;
    box-shadow: 0 0 0 2px rgba(124,58,237,0.3) !important;
}

/* ── Selectbox / multiselect ── */
.stMultiSelect [data-baseweb="tag"] {
    background: rgba(124,58,237,0.35) !important;
    border-radius: 8px !important;
}

/* ── Metric ── */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.05);
    border-radius: 12px;
    padding: 0.8rem 1rem;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(167,139,250,0.4); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# Session state initialisation
# ════════════════════════════════════════════════════════════════════════════════
def _init_session():
    defaults = {
        "active_conv_id": None,
        "chat_history": [],      # list of {role, content, sources}
        "selected_doc_ids": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_session()


# ════════════════════════════════════════════════════════════════════════════════
# Sidebar
# ════════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🧠 DocuMind")
    st.markdown("*AI Document Intelligence*")
    st.divider()

    # ── Navigation ────────────────────────────────────────────────────────────
    page = st.radio(
        "Navigate",
        ["💬 Chat", "📚 Document Library", "⚙️ Settings"],
        label_visibility="collapsed",
    )
    st.divider()

    # ── Quick stats ───────────────────────────────────────────────────────────
    all_docs = db.get_all_documents()
    all_convs = db.get_all_conversations()
    col1, col2 = st.columns(2)
    col1.metric("📄 Docs", len(all_docs))
    col2.metric("💬 Chats", len(all_convs))
    st.divider()

    # ── Document selector (for chat) ──────────────────────────────────────────
    if all_docs:
        st.markdown('<p class="dm-section-title">Search Scope</p>',
                    unsafe_allow_html=True)
        doc_options = {d["filename"]: d["id"] for d in all_docs}
        selected_names = st.multiselect(
            "Documents",
            options=list(doc_options.keys()),
            default=list(doc_options.keys()),
            label_visibility="collapsed",
        )
        st.session_state.selected_doc_ids = [
            doc_options[n] for n in selected_names
        ]
    else:
        st.info("Upload documents to start chatting.")


# ════════════════════════════════════════════════════════════════════════════════
# Page: Chat
# ════════════════════════════════════════════════════════════════════════════════
if page == "💬 Chat":
    st.markdown("""
    <div class="dm-hero">
        <h1>🧠 DocuMind</h1>
        <p>Ask anything about your uploaded documents — grounded answers with citations</p>
    </div>
    """, unsafe_allow_html=True)

    # ── API key warning ───────────────────────────────────────────────────────
    if not GEMINI_API_KEY:
        st.warning("⚠️ **Gemini API key not configured.** "
                   "Go to ⚙️ Settings or add GEMINI_API_KEY to your .env file.",
                   icon="🔑")

    # ── Chat history display ──────────────────────────────────────────────────
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="dm-user-bubble">🙋 {msg["content"]}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="dm-assistant-bubble">🤖 {msg["content"]}</div>',
                    unsafe_allow_html=True,
                )
                # Source badges
                if msg.get("sources"):
                    badges = "".join(
                        f'<span class="dm-source-badge">'
                        f'📄 {s["doc_filename"]} · p.{s["page_num"]}'
                        f'</span>'
                        for s in msg["sources"]
                    )
                    st.markdown(badges, unsafe_allow_html=True)
                    with st.expander("📎 View source passages"):
                        for s in msg["sources"]:
                            st.markdown(
                                f"**{s['source_label']}** — "
                                f"*{s['doc_filename']}*, Page {s['page_num']} "
                                f"(score: {s['score']:.3f})"
                            )
                            st.markdown(
                                f"> {s['text'][:400]}{'…' if len(s['text']) > 400 else ''}"
                            )
                            st.divider()

    # ── Input row ─────────────────────────────────────────────────────────────
    st.markdown("---")
    col_inp, col_btn = st.columns([5, 1])
    with col_inp:
        question = st.text_input(
            "Question",
            placeholder="Ask a question about your documents…",
            label_visibility="collapsed",
            key="question_input",
        )
    with col_btn:
        send = st.button("Send 🚀", use_container_width=True)

    col_c1, col_c2 = st.columns([1, 1])
    with col_c1:
        if st.button("🗑️ Clear chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.active_conv_id = None
            st.rerun()
    with col_c2:
        top_k = st.slider("Passages to retrieve", 1, 10, 5)

    # ── Send logic ────────────────────────────────────────────────────────────
    if send and question.strip():
        if not st.session_state.selected_doc_ids:
            st.error("Please select at least one document in the sidebar.")
        elif not GEMINI_API_KEY:
            st.error("Please configure your Gemini API key in ⚙️ Settings.")
        else:
            st.session_state.chat_history.append({
                "role": "user", "content": question, "sources": []
            })

            with st.spinner("🔍 Retrieving relevant passages…"):
                passages = retrieve_passages(
                    question,
                    st.session_state.selected_doc_ids,
                    top_k=top_k,
                )

            with st.spinner("✍️ Generating answer…"):
                answer = generate_answer(question, passages)

            # Persist to DB
            if st.session_state.active_conv_id is None:
                conv_id = db.create_conversation(
                    title=question[:60],
                    doc_ids=st.session_state.selected_doc_ids,
                )
                st.session_state.active_conv_id = conv_id
            else:
                conv_id = st.session_state.active_conv_id

            db.add_message(conv_id, "user", question)
            msg_id = db.add_message(
                conv_id, "assistant", answer,
                sources=[p["chunk_id"] for p in passages]
            )

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer,
                "sources": passages,
                "msg_id": msg_id,
            })
            st.rerun()


# ════════════════════════════════════════════════════════════════════════════════
# Page: Document Library
# ════════════════════════════════════════════════════════════════════════════════
elif page == "📚 Document Library":
    st.markdown("""
    <div class="dm-hero">
        <h1>📚 Document Library</h1>
        <p>Upload, manage, and explore your document collection</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Upload section ────────────────────────────────────────────────────────
    st.markdown('<p class="dm-section-title">Upload New Document</p>',
                unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Drop a PDF here or click to browse",
        type=["pdf"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        save_path = UPLOAD_DIR / uploaded_file.name
        col_up1, col_up2 = st.columns([3, 1])
        with col_up1:
            st.markdown(
                f'<div class="dm-card">📄 <strong>{uploaded_file.name}</strong>'
                f' &nbsp;·&nbsp; {uploaded_file.size / 1024:.1f} KB</div>',
                unsafe_allow_html=True,
            )
        with col_up2:
            if st.button("⚡ Ingest", use_container_width=True):
                # Save file
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                with st.spinner(
                    "📖 Parsing PDF… ✂️ Chunking… 🔢 Embedding… 🗄️ Indexing…"
                ):
                    doc_id = ingest_document(save_path, uploaded_file.name)

                st.success(
                    f"✅ **{uploaded_file.name}** ingested successfully! "
                    f"(doc_id={doc_id})"
                )
                st.rerun()

    st.divider()

    # ── Document list ─────────────────────────────────────────────────────────
    st.markdown('<p class="dm-section-title">Uploaded Documents</p>',
                unsafe_allow_html=True)

    docs = db.get_all_documents()
    if not docs:
        st.info("No documents yet. Upload a PDF above to get started.")
    else:
        for doc in docs:
            with st.container():
                col_d1, col_d2, col_d3, col_d4, col_del = st.columns(
                    [3, 1, 1, 2, 1]
                )
                col_d1.markdown(f"**📄 {doc['filename']}**")
                col_d2.markdown(f"📃 {doc['num_pages']} pages")
                col_d3.markdown(f"✂️ {doc['num_chunks']} chunks")
                col_d4.markdown(
                    f"🕐 {doc['uploaded_at'][:19].replace('T', ' ')}"
                )
                if col_del.button("🗑️", key=f"del_{doc['id']}",
                                   help="Delete this document"):
                    # Remove FAISS index
                    if doc.get("index_path"):
                        try:
                            Path(doc["index_path"]).unlink(missing_ok=True)
                        except Exception:
                            pass
                    # Remove uploaded file
                    try:
                        Path(doc["filepath"]).unlink(missing_ok=True)
                    except Exception:
                        pass
                    db.delete_document(doc["id"])
                    st.rerun()
                st.divider()


# ════════════════════════════════════════════════════════════════════════════════
# Page: Settings
# ════════════════════════════════════════════════════════════════════════════════
elif page == "⚙️ Settings":
    st.markdown("""
    <div class="dm-hero">
        <h1>⚙️ Settings</h1>
        <p>Configure API keys and model parameters</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p class="dm-section-title">Gemini API</p>',
                unsafe_allow_html=True)

    api_key_input = st.text_input(
        "Gemini API Key",
        type="password",
        value=GEMINI_API_KEY,
        placeholder="Paste your Gemini API key here…",
        help="Get your key at https://aistudio.google.com/app/apikey",
    )

    if st.button("💾 Save API key to .env"):
        env_path = Path(__file__).parent / ".env"
        lines: list[str] = []
        found = False
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("GEMINI_API_KEY="):
                    lines.append(f"GEMINI_API_KEY={api_key_input}")
                    found = True
                else:
                    lines.append(line)
        if not found:
            lines.append(f"GEMINI_API_KEY={api_key_input}")
        env_path.write_text("\n".join(lines) + "\n")
        st.success("API key saved! Please restart the app for changes to take effect.")

    st.divider()
    st.markdown('<p class="dm-section-title">Environment Info</p>',
                unsafe_allow_html=True)

    from src.config import (EMBEDDING_MODEL, GEMINI_MODEL, CHUNK_SIZE,
                             CHUNK_OVERLAP, TOP_K, DB_PATH, INDEX_DIR, UPLOAD_DIR)

    info = {
        "Gemini Model": GEMINI_MODEL,
        "Embedding Model": EMBEDDING_MODEL,
        "Chunk Size (chars)": CHUNK_SIZE,
        "Chunk Overlap (chars)": CHUNK_OVERLAP,
        "Top-K Retrieval": TOP_K,
        "Database Path": str(DB_PATH),
        "Index Directory": str(INDEX_DIR),
        "Upload Directory": str(UPLOAD_DIR),
    }
    for k, v in info.items():
        col1, col2 = st.columns([2, 3])
        col1.markdown(f"**{k}**")
        col2.code(str(v), language=None)
