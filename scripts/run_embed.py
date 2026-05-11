import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from ingestion.embedder import embed_all

if __name__ == '__main__':
    # Run API-only embedding (this will consume Google embedding quota).
    embed_all(batch_size=100, resume=True)
