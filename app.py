"""
Medical RAG Chatbot — Streamlit UI
Run: streamlit run app.py
"""
import io
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
from streamlit.components.v1 import declare_component
from PIL import Image
from dotenv import dotenv_values


browser_location = declare_component(
    "browser_location",
    path=str(Path(__file__).resolve().parent / "browser_location_component"),
)


def load_env_file():
    dotenv_path = Path(__file__).resolve().parent / ".env"

    if not dotenv_path.exists():
        return None

    for encoding in ("utf-8-sig", "utf-16", "utf-16-le", "cp1252"):
        try:
            content = dotenv_path.read_text(encoding=encoding)
            values = dotenv_values(stream=io.StringIO(content))
            for key, value in values.items():
                if value is not None:
                    os.environ.setdefault(key, value)
            return encoding
        except UnicodeDecodeError:
            continue

    raise UnicodeDecodeError(".env", b"", 0, 1, "Unable to decode .env file with known encodings")


load_env_file()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MedRAG · Medical AI Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background: #0d1117;
    color: #e2e8f0;
}

section[data-testid="stSidebar"] {
    background: #111827;
    border-right: 1px solid #1e2a3a;
}

.main .block-container { max-width: 860px; padding-top: 2rem; }

.user-bubble {
    background: #1a2744;
    border: 1px solid #2d4a7a;
    border-radius: 12px 12px 2px 12px;
    padding: 14px 18px;
    margin: 8px 0;
    font-size: 0.95rem;
}

.bot-bubble {
    background: #0f1f1a;
    border: 1px solid #1a4a35;
    border-radius: 12px 12px 12px 2px;
    padding: 14px 18px;
    margin: 8px 0;
    font-family: 'DM Mono', monospace;
    font-size: 0.88rem;
}

.pipeline-step {
    display: inline-block;
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 4px;
    padding: 2px 8px;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #94a3b8;
    margin: 2px;
}

.disclaimer {
    background: #1c1208;
    border-left: 3px solid #f59e0b;
    padding: 10px;
    border-radius: 4px;
    font-size: 0.8rem;
    color: #fbbf24;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🩺 MedRAG")
    st.caption("Multimodal Medical RAG")

    api_key = st.text_input("Google API Key", type="password",
                            value=os.getenv("GOOGLE_API_KEY", ""))

    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
        import config
        config.GOOGLE_API_KEY = api_key

    show_eval = st.toggle("Show evaluation", False)
    show_docs = st.toggle("Show retrieved docs", False)
    st.caption("Browser location will be used automatically for urgent hospital lookup.")
    browser_coords = browser_location(key="browser_location")
    if browser_coords:
        st.session_state.browser_coords = browser_coords

    st.divider()

    if st.button("🗑 Clear chat"):
        from memory.chat_memory import clear_history
        clear_history()
        st.session_state.display_history = []
        st.rerun()

    st.divider()

    st.markdown("### Data Pipeline")

    if st.button("🕷 Scrape"):
        from ingestion.scraper import scrape
        with st.spinner("Scraping..."):
            scrape()
        st.success("Done")

    if st.button("🧹 Preprocess"):
        from ingestion.preprocess import preprocess_all
        with st.spinner("Preprocessing..."):
            preprocess_all()
        st.success("Done")

    if st.button("✂ Chunk"):
        from ingestion.chunker import chunk_all
        with st.spinner("Chunking..."):
            chunk_all()
        st.success("Done")

    if st.button("🔢 Embed"):
        from ingestion.embedder import embed_all
        with st.spinner("Embedding..."):
            embed_all()
        st.success("Done")

    st.markdown('<div class="disclaimer">⚠ Not medical advice</div>', unsafe_allow_html=True)

# ── Main UI ───────────────────────────────────────────────────────────────────
st.title("Medical AI Assistant")

if "display_history" not in st.session_state:
    st.session_state.display_history = []

# ── Chat Display ──────────────────────────────────────────────────────────────
for turn in st.session_state.display_history:
    st.markdown(f'<div class="user-bubble">👤 {turn["query"]}</div>', unsafe_allow_html=True)

    if turn.get("image_used"):
        st.caption("📷 Image analyzed")

    st.markdown(f'<div class="bot-bubble">🩺 {turn["response"]}</div>', unsafe_allow_html=True)

    if turn.get("urgent_care"):
        st.error("This sounds urgent. Seek immediate medical attention.")

        hospitals = turn.get("nearby_hospitals") or []
        if hospitals:
            with st.expander("Nearby hospitals"):
                for hospital in hospitals:
                    title = hospital.get("name") or "Hospital"
                    address = hospital.get("address") or "Address unavailable"
                    distance = hospital.get("distance_km")
                    if distance is not None:
                        st.markdown(f"- **{title}** - {address} ({distance:.1f} km away)")
                    else:
                        st.markdown(f"- **{title}** - {address}")
        else:
            st.info("Enter your city or address in the sidebar to list nearby hospitals.")

    if show_eval and turn.get("metrics"):
        m = turn["metrics"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Relevance", f"{m['relevance']*100:.0f}%")
        c2.metric("Grounding", f"{m['grounding']*100:.0f}%")
        c3.metric("No Hallucination", f"{m['no_hallucination']*100:.0f}%")

    if show_docs and turn.get("docs"):
        with st.expander("Retrieved Docs"):
            for i, d in enumerate(turn["docs"]):
                st.markdown(f"**Doc {i+1}:** {d[:300]}...")

    st.markdown("---")

# ── Input ─────────────────────────────────────────────────────────────────────
uploaded_image = st.file_uploader(
    "Upload image (optional)",
    type=["png", "jpg", "jpeg"],
    label_visibility="collapsed",
)

if uploaded_image:
    img = Image.open(uploaded_image)
    st.image(img, width=250)

with st.form("chat_form", clear_on_submit=True):
    query = st.text_area("Ask medical question", height=100)
    submitted = st.form_submit_button("Send")

# ── RAG Execution ─────────────────────────────────────────────────────────────
if submitted and query.strip():

    from config import FAISS_INDEX_PATH, CHUNKS_PATH

    if not os.path.exists(FAISS_INDEX_PATH):
        st.error("Run ingestion pipeline first.")
        st.stop()

    if not os.getenv("GOOGLE_API_KEY"):
        st.error("Enter API key")
        st.stop()

    pil_image = Image.open(uploaded_image) if uploaded_image else None

    with st.spinner("Thinking..."):
        from rag.pipeline import run_pipeline
        from utils.locationiq import find_nearby_hospitals, is_urgent_case

        # ✅ FIXED: single retrieval source of truth
        docs, response = run_pipeline(
            query,
            image=pil_image,
            return_docs=True
        )

        urgent_care = is_urgent_case(query, response)
        nearby_hospitals = []
        if urgent_care:
            nearby_hospitals = find_nearby_hospitals(st.session_state.get("browser_coords"))

        metrics = {}
        if show_eval:
            from evaluation.metrics import evaluate
            metrics = evaluate(query, response, docs)

    st.session_state.display_history.append({
        "query": query,
        "response": response,
        "image_used": pil_image is not None,
        "docs": docs,
        "metrics": metrics,
        "urgent_care": urgent_care,
        "nearby_hospitals": nearby_hospitals,
    })

    st.rerun()