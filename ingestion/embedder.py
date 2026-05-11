"""
Embed chunks + build FAISS index (optimized)
Run: python -m ingestion.embedder
"""
import io
import os
import time
import re
import hashlib
from pathlib import Path

import numpy as np
import faiss
from google import genai
from dotenv import dotenv_values


def load_env_file():
    dotenv_path = Path(__file__).resolve().parents[1] / ".env"

    if not dotenv_path.exists():
        return

    for encoding in ("utf-8-sig", "utf-16", "utf-16-le", "cp1252"):
        try:
            content = dotenv_path.read_text(encoding=encoding)
            values = dotenv_values(stream=io.StringIO(content))
            for key, value in values.items():
                if value is not None:
                    os.environ.setdefault(key, value)
            return
        except UnicodeDecodeError:
            continue


load_env_file()

from config import (
    GOOGLE_API_KEY, EMBEDDING_MODEL,
    CHUNKS_PATH, FAISS_INDEX_PATH, EMBEDDINGS_DIR
)
from utils.logger import get_logger
from utils.helpers import ensure_dirs, load_json, save_json

log = get_logger(__name__)

client = genai.Client(api_key=GOOGLE_API_KEY)


class QuotaExhaustedError(RuntimeError):
    """Raised when embedding quota appears hard-exhausted beyond short retry windows."""


# ── Embed one chunk ───────────────────────────────────────────────────────────
def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _parse_retry_delay(exc_text: str) -> float | None:
    # Try to find patterns like 'Please retry in 48.494282845s' or 'retryDelay': '48s'
    m = re.search(r"Please retry in ([0-9]+(?:\.[0-9]+)?)s", exc_text)
    if m:
        return float(m.group(1))
    m2 = re.search(r"retryDelay'[:\"]?\s*[:=]?\s*'?(\d+)s'?", exc_text)
    if m2:
        return float(m2.group(1))
    return None


def _is_quota_error(exc_text: str) -> bool:
    return (
        "RESOURCE_EXHAUSTED" in exc_text
        or "'code': 429" in exc_text
        or '"code": 429' in exc_text
        or "Quota exceeded" in exc_text
    )


def _checkpoint_embeddings(chunks: list[str], final_vectors_by_pos: list, chunks_clean_path: str, vectors_npy_path: str) -> tuple[int, int]:
    """Persist current progress so resume can continue from the latest successful batch."""
    valid_chunks = []
    vectors = []
    for i, vec in enumerate(final_vectors_by_pos):
        if vec is None:
            continue
        valid_chunks.append(chunks[i])
        vectors.append(np.array(vec, dtype="float32"))

    save_json(valid_chunks, chunks_clean_path)
    if vectors:
        np.save(vectors_npy_path, np.vstack(vectors))

    return len(valid_chunks), len(chunks)


def embed_batch(texts: list[str], max_retries: int = 8):
    """Embed a list of texts in one API call (if supported) with retries/backoff.

    Returns list of vectors (or None for failures in same order).
    """
    if not texts:
        return []

    attempt = 0
    last_error = ""
    while attempt < max_retries:
        try:
            res = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=texts,
                config={"task_type": "retrieval_document"},
            )

            # res.embeddings should be a list aligned with texts
            if not res.embeddings:
                return [None] * len(texts)

            return [e.values if e and hasattr(e, "values") else None for e in res.embeddings]

        except Exception as e:
            et = str(e)
            last_error = et
            is_quota = _is_quota_error(et)
            delay = _parse_retry_delay(et)

            if is_quota and delay is not None and delay > 300:
                raise QuotaExhaustedError(f"Quota retry delay too high ({delay:.1f}s).") from e

            if delay is not None:
                sleep_for = min(delay + 1.0, 65.0)
            else:
                sleep_for = min(60, (2 ** attempt))

            log.warning("Retry embedding... %s. Sleeping %.1fs", et, sleep_for)
            time.sleep(sleep_for)
            attempt += 1

    if _is_quota_error(last_error):
        raise QuotaExhaustedError(f"Quota exhausted after {max_retries} retries.")

    raise RuntimeError(f"Failed to embed batch after {max_retries} attempts")


# ── Main embedding pipeline ───────────────────────────────────────────────────
def embed_all(batch_size: int = 64, resume: bool = True):
    """Embed all chunks into a FAISS index with batching, caching and resume support.

    - `batch_size`: number of texts to embed per API request.
    - `resume`: if True, will reuse existing embeddings found in
      EMBEDDINGS_DIR/chunks_clean.json + vectors.npy and only embed missing ones.
    """
    ensure_dirs(EMBEDDINGS_DIR)

    chunks = load_json(CHUNKS_PATH)
    if not chunks:
        log.error(f"No chunks found at {CHUNKS_PATH}. Run chunking first.")
        return

    batch_size = max(1, min(int(batch_size), 100))
    log.info(f"Embedding {len(chunks)} chunks (batch_size={batch_size})...")

    chunks_clean_path = os.path.join(EMBEDDINGS_DIR, "chunks_clean.json")
    vectors_npy_path = os.path.join(EMBEDDINGS_DIR, "vectors.npy")

    # Load existing embeddings if available and consistent
    existing_hash_to_vec = {}
    if resume and os.path.exists(chunks_clean_path) and os.path.exists(vectors_npy_path):
        try:
            existing_chunks = load_json(chunks_clean_path) or []
            existing_matrix = np.load(vectors_npy_path)
            if len(existing_chunks) == existing_matrix.shape[0]:
                for idx, c in enumerate(existing_chunks):
                    existing_hash_to_vec[_sha256(c)] = existing_matrix[idx]
                log.info("Loaded %d existing embeddings; will resume and skip them", len(existing_chunks))
            else:
                log.warning("Existing chunks count doesn't match vectors.npy rows — ignoring resume cache")
        except Exception as e:
            log.warning("Failed to load existing embeddings for resume: %s", e)

    # Prepare lists of which chunks need embedding
    pending_texts = []
    pending_positions = []

    # For final assembling
    final_vectors_by_pos = [None] * len(chunks)

    for i, chunk in enumerate(chunks):
        h = _sha256(chunk)
        if h in existing_hash_to_vec:
            final_vectors_by_pos[i] = existing_hash_to_vec[h]
        else:
            pending_texts.append(chunk)
            pending_positions.append(i)

    log.info("Pending to embed: %d (skipped %d already-embedded)", len(pending_texts), len(chunks) - len(pending_texts))

    # Batch embed pending_texts
    total = len(pending_texts)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = pending_texts[start:end]

        try:
            results = embed_batch(batch)
        except QuotaExhaustedError as e:
            done, total_chunks = _checkpoint_embeddings(chunks, final_vectors_by_pos, chunks_clean_path, vectors_npy_path)
            log.error(
                "Quota limit reached. Checkpoint saved (%d/%d). Update GOOGLE_API_KEY and rerun to resume. Details: %s",
                done,
                total_chunks,
                e,
            )
            return

        for j, vec in enumerate(results):
            pos = pending_positions[start + j]
            if vec is None:
                raise RuntimeError(f"Unexpected empty embedding at chunk position {pos}")
            final_vectors_by_pos[pos] = np.array(vec, dtype="float32")

        done, total_chunks = _checkpoint_embeddings(chunks, final_vectors_by_pos, chunks_clean_path, vectors_npy_path)
        log.info("Embedded %d/%d pending | checkpoint %d/%d", end, total, done, total_chunks)
        # small safety pause
        time.sleep(0.2)

    # Assemble final valid chunks and vectors in original order
    valid_chunks = []
    vectors = []
    for i, vec in enumerate(final_vectors_by_pos):
        if vec is None:
            continue
        valid_chunks.append(chunks[i])
        vectors.append(vec)

    if not vectors:
        log.error("No embeddings were generated. Check the embedding model/API key and quota.")
        return

    if len(valid_chunks) != len(chunks):
        log.error("Embeddings incomplete (%d/%d). Checkpoint saved; rerun to continue.", len(valid_chunks), len(chunks))
        return

    matrix = np.vstack([np.array(v, dtype="float32") for v in vectors])

    # Normalize for cosine similarity
    faiss.normalize_L2(matrix)

    dim = matrix.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(matrix)

    # Save index and aligned chunks/vectors
    faiss.write_index(index, FAISS_INDEX_PATH)
    save_json(valid_chunks, chunks_clean_path)
    np.save(vectors_npy_path, matrix)

    log.info(f"FAISS index saved → {FAISS_INDEX_PATH}")
    log.info(f"Vectors: {matrix.shape[0]} | Dim: {dim}")


if __name__ == "__main__":
    embed_all()