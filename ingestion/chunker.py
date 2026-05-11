"""
Semantic Chunker (paragraph-aware, RAG optimized)
Run: python -m ingestion.chunker
"""
import os
from config import (
    PROCESSED_DATA_DIR,
    CHUNKS_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDINGS_DIR
)
from utils.logger import get_logger
from utils.helpers import ensure_dirs, read_txt, save_json

log = get_logger(__name__)


# ── Split large paragraphs ────────────────────────────────────────────────────
def split_large_paragraph(paragraph: str, size: int, overlap: int):
    chunks = []
    start = 0

    while start < len(paragraph):
        end = min(start + size, len(paragraph))
        chunk = paragraph[start:end].strip()

        if len(chunk) > 100:
            chunks.append(chunk)

        start += size - overlap

    return chunks


# ── MAIN CHUNKING LOGIC (CRITICAL) ────────────────────────────────────────────
def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    chunks = []

    # Step 1: split by paragraphs (KEY FIX)
    paragraphs = text.split("\n\n")

    for para in paragraphs:
        para = para.strip()

        if not para:
            continue

        # Step 2: if small → keep as is
        if len(para) <= size:
            if len(para) > 100:
                chunks.append(para)

        # Step 3: if large → split smartly
        else:
            chunks.extend(split_large_paragraph(para, size, overlap))

    return chunks


# ── Process all files ─────────────────────────────────────────────────────────
def chunk_all():
    ensure_dirs(EMBEDDINGS_DIR)

    files = [f for f in os.listdir(PROCESSED_DATA_DIR) if f.endswith(".txt")]
    log.info(f"Chunking {len(files)} files...")

    all_chunks = []

    for fname in files:
        text = read_txt(os.path.join(PROCESSED_DATA_DIR, fname))

        chunks = chunk_text(text)

        all_chunks.extend(chunks)

    log.info(f"Total chunks: {len(all_chunks)}")

    save_json(all_chunks, CHUNKS_PATH)

    return all_chunks


if __name__ == "__main__":
    chunk_all()