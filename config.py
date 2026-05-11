import os

# ── API Keys ──────────────────────────────────────────────────────────────────
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
LOCATIONIQ_API_KEY = os.getenv("LOCATIONIQ_API_KEY", "")

# ── Models ────────────────────────────────────────────────────────────────────
# Local generation uses Ollama by default; embeddings can still use Gemini.
OLLAMA_BASE_URL     = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL        = os.getenv("OLLAMA_MODEL", "llama3.1:8b-instruct")
GEMINI_MODEL        = os.getenv("GEMINI_MODEL", "models/text-bison-001")
GEMINI_VISION_MODEL = os.getenv("GEMINI_VISION_MODEL", "gemini-2.0-mini")
EMBEDDING_MODEL     = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")  # supported Gemini embedding model

# ── Chunking (FINAL) ──────────────────────────────────────────────────────────
CHUNK_SIZE    = 800   # increased for better semantic context
CHUNK_OVERLAP = 100   # higher overlap to preserve continuity

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K = 5

# Enable summarizer (may consume generation quota). Turn off for strict low-quota mode.
ENABLE_SUMMARIZER = False

# ── Paths ─────────────────────────────────────────────────────────────────────
RAW_DATA_DIR       = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
EMBEDDINGS_DIR     = "data/embeddings"
FAISS_INDEX_PATH   = "data/embeddings/faiss.index"
CHUNKS_PATH        = "data/embeddings/chunks.json"

# ── Memory ────────────────────────────────────────────────────────────────────
MAX_HISTORY_TURNS = 6

# ── Scraping ──────────────────────────────────────────────────────────────────
WEBMD_BASE_URL = "https://www.webmd.com"
MAX_PAGES      = 1600  