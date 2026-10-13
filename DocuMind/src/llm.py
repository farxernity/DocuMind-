"""
llm.py — Gemini API integration for DocuMind.
Uses the current `google-genai` SDK (google.genai).

Features
--------
- Automatic model fallback: if the primary model returns 503/429, the next
  model in the FALLBACK_MODELS list is tried automatically.
- Reads GEMINI_MODEL fresh from .env on every call (no stale-cache issues).
"""
from __future__ import annotations
import os
import time
import logging
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import ServerError, ClientError
from src.config import GEMINI_API_KEY

log = logging.getLogger(__name__)

# ── Fallback model chain ──────────────────────────────────────────────────────
# If the primary model is 503/429, each subsequent model is tried in order.
FALLBACK_MODELS = [
    "gemini-3.8-flash",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite-preview-06-17",
    "gemini-2.5-pro",
]


def _get_model() -> str:
    """Read primary model name fresh from .env on every call."""
    load_dotenv(override=True)
    return os.getenv("GEMINI_MODEL", "gemini-3.8-flash")


def _model_chain() -> list[str]:
    """Return [primary] + fallbacks (deduped, primary first)."""
    primary = _get_model()
    chain = [primary] + [m for m in FALLBACK_MODELS if m != primary]
    return chain


# ── Client singleton ─────────────────────────────────────────────────────────

def _make_client() -> genai.Client:
    if not GEMINI_API_KEY:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set. "
            "Add it to your .env file or environment variables."
        )
    return genai.Client(api_key=GEMINI_API_KEY)


_client: genai.Client = _make_client()


# ── Prompt builder ────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are DocuMind, an expert research assistant.
You answer questions based ONLY on the document passages provided below.

Rules:
- If the passages contain the answer, provide a clear, accurate response.
- Cite every claim with the source label (e.g. [Source 1, Page 3]).
- If the passages do NOT contain sufficient information, say:
  "The uploaded documents do not contain enough information to answer this question."
- Do NOT use any knowledge outside the provided passages.
- Be concise but complete. Use bullet points where helpful.
"""


def build_rag_prompt(question: str, passages: list[dict]) -> str:
    """
    Build the user-turn text for the RAG prompt.

    Parameters
    ----------
    question : The user's question string.
    passages : list of dicts with keys: text, page_num, doc_filename, source_label
    """
    context_parts: list[str] = []
    for i, p in enumerate(passages, 1):
        label = p.get("source_label", f"Source {i}")
        page = p.get("page_num", "?")
        filename = p.get("doc_filename", "Unknown document")
        context_parts.append(
            f"[{label}] (File: {filename}, Page {page})\n{p['text']}"
        )

    context_block = "\n\n---\n\n".join(context_parts)

    return (
        f"DOCUMENT PASSAGES:\n\n{context_block}\n\n"
        f"QUESTION: {question}\n\n"
        "Answer:"
    )


# ── Config helper ─────────────────────────────────────────────────────────────

def _make_config() -> types.GenerateContentConfig:
    return types.GenerateContentConfig(
        system_instruction=_SYSTEM_PROMPT,
    )


# ── Retry / fallback helper ───────────────────────────────────────────────────

def _is_retryable(exc: Exception) -> bool:
    """Return True for 503 (overload) and 429 (rate limit) errors."""
    if isinstance(exc, ServerError) and "503" in str(exc):
        return True
    if isinstance(exc, ClientError) and "429" in str(exc):
        return True
    return False


def _call_with_fallback(fn, *args, **kwargs):
    """
    Try fn(model, *args, **kwargs) for each model in the fallback chain.
    fn signature: fn(model: str, *args, **kwargs) -> result
    """
    models = _model_chain()
    last_exc: Exception | None = None

    for i, model in enumerate(models):
        try:
            if i > 0:
                log.warning("Falling back to model: %s", model)
                time.sleep(1)          # brief pause before fallback
            return fn(model, *args, **kwargs)
        except Exception as exc:
            if _is_retryable(exc) and i < len(models) - 1:
                log.warning("Model %s unavailable (%s), trying next...", model, exc)
                last_exc = exc
            else:
                raise          # non-retryable or exhausted all models

    raise last_exc  # should not reach here


# ── Answer generation ─────────────────────────────────────────────────────────

def generate_answer(question: str, passages: list[dict],
                    stream: bool = False):
    """
    Generate an answer grounded in the supplied passages.
    Automatically falls back to the next model on 503/429 errors.

    Returns
    -------
    str (stream=False) or generator of str chunks (stream=True).
    """
    user_message = build_rag_prompt(question, passages)
    config = _make_config()

    if stream:
        def _stream(model):
            resp = _client.models.generate_content_stream(
                model=model, contents=user_message, config=config,
            )
            return (chunk.text for chunk in resp if chunk.text)
        return _call_with_fallback(_stream)
    else:
        def _sync(model):
            return _client.models.generate_content(
                model=model, contents=user_message, config=config,
            ).text
        return _call_with_fallback(_sync)


def generate_answer_with_history(question: str, passages: list[dict],
                                  history: list[types.Content],
                                  stream: bool = False):
    """
    Multi-turn variant with fallback support.
    history : list of types.Content objects (role='user' or 'model').
    """
    user_message = build_rag_prompt(question, passages)
    config = _make_config()

    def _call(model):
        chat = _client.chats.create(model=model, config=config, history=history)
        if stream:
            resp = chat.send_message_stream(user_message)
            return (chunk.text for chunk in resp if chunk.text)
        else:
            return chat.send_message(user_message).text

    return _call_with_fallback(_call)
