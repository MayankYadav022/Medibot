import json, os, numpy as np, sys, pathlib

# ensure project root is on sys.path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from config import CHUNKS_PATH, EMBEDDINGS_DIR

chunks = json.load(open(CHUNKS_PATH, 'r', encoding='utf-8'))
chunks_count = len(chunks)

chunks_clean_path = os.path.join(EMBEDDINGS_DIR, 'chunks_clean.json')
vectors_path = os.path.join(EMBEDDINGS_DIR, 'vectors.npy')
existing_chunks = 0
vectors_rows = 0
if os.path.exists(chunks_clean_path) and os.path.exists(vectors_path):
    try:
        existing_chunks = len(json.load(open(chunks_clean_path, 'r', encoding='utf-8')))
        vectors_rows = int(np.load(vectors_path).shape[0])
    except Exception:
        existing_chunks = 0
        vectors_rows = 0

print(chunks_count)
print(existing_chunks)
print(vectors_rows)
