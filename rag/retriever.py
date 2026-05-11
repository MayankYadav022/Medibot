import numpy as np
import faiss
import google.generativeai as genai

from config import GOOGLE_API_KEY, EMBEDDING_MODEL, TOP_K
from vectorstore.faiss_store import search

genai.configure(api_key=GOOGLE_API_KEY)
def embed_query(text: str) -> np.ndarray:
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=text,
        task_type="retrieval_query"
    )

    vec = np.array(result["embedding"]).astype("float32")

    # 🔥 CRITICAL: normalize query vector
    faiss.normalize_L2(vec.reshape(1, -1))

    return vec
def retrieve(query: str, top_k: int = TOP_K) -> list[str]:
    vec = embed_query(query)
    docs = search(vec, top_k)
    return docs