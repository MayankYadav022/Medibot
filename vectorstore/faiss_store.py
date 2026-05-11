"""
FAISS vector store (robust + RAG-optimized)
"""
import numpy as np
import faiss

from config import FAISS_INDEX_PATH, EMBEDDINGS_DIR
from utils.helpers import load_json
from utils.logger import get_logger

log = get_logger(__name__)

_index = None
_chunks = None      

def _trim_chunk(text: str, limit: int = 200) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"
# ── Load index + chunks ───────────────────────────────────────────────────────
def _load(): 
    global _index, _chunks

    if _index is None:
        _index = faiss.read_index(FAISS_INDEX_PATH)

        # 🔥 IMPORTANT: use cleaned aligned chunks
        _chunks = load_json(f"{EMBEDDINGS_DIR}/chunks_clean.json")

        log.info(f"FAISS index loaded ({_index.ntotal} vectors)")
# ── Search ────────────────────────────────────────────────────────────────────
def search(query_vector: np.ndarray, top_k: int = 2) -> list[str]:
    _load()

    if _index.ntotal == 0:
        log.warning("FAISS index is empty")
        return []

    vec = np.array([query_vector]).astype("float32")
    faiss.normalize_L2(vec)

    scores, indices = _index.search(vec, top_k)

    results = []

    for idx, score in zip(indices[0], scores[0]):
        if idx < 0 or idx >= len(_chunks):
            continue

        # optional: filter low similarity
        if score < 0.2:
            continue
        results.append(_trim_chunk(_chunks[idx]))
    return results