"""
Conversation Memory with Controlled Summarization (RAG-optimized)
"""
import streamlit as st
import google.generativeai as genai
from config import GOOGLE_API_KEY, GEMINI_MODEL, MAX_HISTORY_TURNS, ENABLE_SUMMARIZER

genai.configure(api_key=GOOGLE_API_KEY)

_summarizer = genai.GenerativeModel(
    model_name=GEMINI_MODEL,
    generation_config={
        "temperature": 0.0,  
        "max_output_tokens": 80
    }
)
# ── Init ──────────────────────────────────────────────────────────────────────
def _init():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "history_summary" not in st.session_state:
        st.session_state.history_summary = ""

# ── Save turn ─────────────────────────────────────────────────────────────────
def save_turn(query: str, response: str):
    _init()

    st.session_state.chat_history.append({
        "query": query,
        "response": response
    })

    # trigger summarization (optional; disabled in low-quota mode)
    if ENABLE_SUMMARIZER and len(st.session_state.chat_history) > MAX_HISTORY_TURNS:
        _summarize()
# ── Summarization (CRITICAL UPGRADE) ──────────────────────────────────────────
def _summarize():
    history = st.session_state.chat_history

    # keep last 2 turns fresh
    old_turns = history[:-2]
    recent_turns = history[-2:]

    text = "\n".join(
        f"User: {t['query']}\nAssistant: {t['response']}"
        for t in old_turns
    )

    prompt = f"""
You are summarizing a medical conversation for a retrieval system.

Rules:
- Preserve key medical facts, symptoms, conditions, and advice
- Keep it concise but information-dense
- Do NOT hallucinate or add new info
- Maintain clinical clarity

Conversation:
{text}

Summary:
"""
    try:
        summary = _summarizer.generate_content(
            prompt,
            request_options={"timeout": 20},
        ).text.strip()

        # length guard
        if len(summary) > 1000:
            summary = summary[:1000]

        st.session_state.history_summary = summary
        st.session_state.chat_history = recent_turns

    except Exception:
        # fallback: keep only recent turns
        st.session_state.chat_history = recent_turns
# ── Build context for LLM ─────────────────────────────────────────────────────
def get_history_text(max_chars: int = 300) -> str:
    _init()

    parts = []

    # include summary first
    if st.session_state.history_summary:
        parts.append(
            f"[Conversation Summary]\n{st.session_state.history_summary}"
        )

    # include recent turns
    for t in st.session_state.chat_history:
        parts.append(
            f"User: {t['query']}\nAssistant: {t['response']}"
        )

    full_text = "\n\n".join(parts)

    # token/length control
    if len(full_text) > max_chars:
        full_text = full_text[-max_chars:]

    return full_text
# ── Clear ─────────────────────────────────────────────────────────────────────
def clear_history():
    st.session_state.chat_history = []
    st.session_state.history_summary = ""