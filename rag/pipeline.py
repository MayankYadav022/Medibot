"""
Core RAG pipeline (optimized + safe)
"""
from multimodal.image_processor import process_image
from rag.retriever      import retrieve
from rag.prompt_builder import build_prompt
from rag.generator      import generate
from memory.chat_memory import get_history_text, save_turn
from utils.logger       import get_logger

log = get_logger(__name__)


def _fallback_answer(query: str, docs: list[str]) -> str:
    if not docs:
        return "I don't have enough information in the retrieved context to answer that."

    keywords = [word.lower() for word in query.split() if len(word) > 3]
    scored_docs = []

    for doc in docs:
        lowered = doc.lower()
        score = sum(1 for word in keywords if word in lowered)
        scored_docs.append((score, doc))

    scored_docs.sort(key=lambda item: item[0], reverse=True)
    best_docs = [doc for score, doc in scored_docs[:2] if score > 0]

    if not best_docs:
        best_docs = docs[:2]

    snippets = [" ".join(doc.split())[:220] for doc in best_docs]
    joined = "\n- ".join(snippets)

    return (
        "Gemini generation is unavailable for this key, so here is a retrieval-based summary:\n"
        f"- {joined}\n\n"
        "This is a context-only fallback, not a full generated answer."
    )


def run_pipeline(query: str, image=None, return_docs: bool = False):
    original_query = query  # 🔥 preserve original

    # ── 1. Process image (if any) ─────────────────────────────────────────────
    img_description = ""
    if image is not None:
        log.info("Processing image …")
        img_description = process_image(image)

    # ── 2. Build retrieval query (controlled) ─────────────────────────────────
    retrieval_query = query
    if img_description:
        retrieval_query = f"{query} {img_description[:300]}"  # limit noise

    # ── 3. Retrieve relevant docs ─────────────────────────────────────────────
    log.info("Retrieving docs …")
    docs = retrieve(retrieval_query)

    # fallback if nothing useful
    if not docs:
        log.warning("No documents retrieved")
        docs = ["No relevant medical context found."]

    # ── 4. Get chat history ───────────────────────────────────────────────────
    history = get_history_text()

    # ── 5. Build prompt (include image separately) ────────────────────────────
    if img_description:
        query_for_prompt = f"{original_query}\n\n[Image Analysis]: {img_description}"
    else:
        query_for_prompt = original_query

    prompt = build_prompt(query_for_prompt, docs, history)

    # ── 6. Generate response ──────────────────────────────────────────────────
    log.info("Generating response …")
    response = generate(prompt)

    if response in {"__QUOTA_EXHAUSTED__", "__MODEL_UNAVAILABLE__", "__OLLAMA_UNAVAILABLE__"}:
        response = _fallback_answer(query_for_prompt, docs)

    # ── 7. Save original query (not modified) ─────────────────────────────────
    save_turn(original_query, response)

    if return_docs:
        return docs, response

    return response