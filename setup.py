"""
One-shot script to run the entire ingestion pipeline:
  scrape → preprocess → chunk → embed

Run: python setup.py
"""
import io
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from dotenv import dotenv_values
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

from ingestion.scraper    import scrape
from ingestion.preprocess import preprocess_all
from ingestion.chunker    import chunk_all
from ingestion.embedder   import embed_all
from utils.logger         import get_logger

log = get_logger("setup")

if not os.getenv("GOOGLE_API_KEY"):
    log.error("GOOGLE_API_KEY not set. Export it or create a .env file.")
    sys.exit(1)

log.info("=== Step 1/4: Scraping WebMD ===")
scrape()

log.info("=== Step 2/4: Preprocessing ===")
preprocess_all()

log.info("=== Step 3/4: Chunking ===")
chunk_all()

log.info("=== Step 4/4: Embedding + FAISS ===")
embed_all()
log.info("✅ Ingestion complete. Run:  streamlit run app.py")