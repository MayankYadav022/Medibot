"""
Preprocess WebMD text for RAG (preserve structure)
Run: python -m ingestion.preprocess
"""
import os, re
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from utils.logger import get_logger
from utils.helpers import ensure_dirs, read_txt, write_txt
log = get_logger(__name__)
def clean(text: str) -> str:
    # ── Remove URLs ───────────────────────────────────────────────
    text = re.sub(r"http\S+", " ", text)
    # ── Remove weird artifacts but KEEP medical symbols ───────────
    text = re.sub(r"[^\w\s\.\,\!\?\:\;\-\(\)%/]", " ", text)
    # ── Normalize spacing but PRESERVE paragraphs ────────────────
    text = re.sub(r"\n\s*\n", "\n\n", text)  # keep paragraph gaps
    text = re.sub(r"[ \t]+", " ", text)
    # ── Trim lines ───────────────────────────────────────────────
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    # ── Rebuild text with paragraph structure ────────────────────
    cleaned = "\n".join(lines)

    return cleaned.strip()

def preprocess_all():
    ensure_dirs(PROCESSED_DATA_DIR)
    files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith(".txt")]
    log.info(f"Preprocessing {len(files)} files...")
    for fname in files:
        raw = read_txt(os.path.join(RAW_DATA_DIR, fname))
        cleaned = clean(raw)
        # quality filter
        if len(cleaned) < 200:
            continue

        write_txt(os.path.join(PROCESSED_DATA_DIR, fname), cleaned)

    log.info("Preprocessing complete.")


if __name__ == "__main__":
    preprocess_all()