# jobs/migrate_to_pinecone.py
"""
Re-embed local malicious texts and upsert into Pinecone staging index in batches.

Usage:
  # Powershell
  $env:PINECONE_API_KEY="your-key"
  $env:PINECONE_INDEX="antigravity-malicious-staging"
  $env:VECTOR_BACKEND="pinecone"
  python jobs/migrate_to_pinecone.py --source tests/attacks/seed_attacks.json --batch 256
"""
import os
import argparse
import json
import time
import sys

# Ensure parent path is in sys.path to import detector modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentence_transformers import SentenceTransformer
import numpy as np
from detector.vector_client import upsert_vectors, count_vectors
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("migrate")

MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

def load_source(path):
    if not os.path.exists(path):
        logger.error(f"Source file not found: {path}")
        return []
        
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    # Accept either list[str] or list[{"text": "..."}]
    items = []
    for i, rec in enumerate(data):
        if isinstance(rec, dict) and "text" in rec:
            txt = rec["text"]
        else:
            txt = rec
        # Creating a stable ID logic could be better, but index-based is fine for seed
        items.append({"id": f"seed-{i}", "text": txt})
    return items

def embed_texts(model, texts):
    # show_progress_bar might not work well in non-interactive
    embs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return embs.astype(np.float32)

def run_migration(source_path, batch_size=256):
    # Check limit before starting
    curr = count_vectors()
    if curr is not None and curr > 290000:
        logger.error(f"Index size {curr} is near free-tier limit (300k). Aborting migration.")
        return

    logger.info("Loading model %s ...", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)
    
    items = load_source(source_path)
    total = len(items)
    if total == 0:
        logger.warning("No items to migrate.")
        return

    logger.info("Loaded %d items for migration", total)
    
    for i in range(0, total, batch_size):
        batch = items[i:i+batch_size]
        texts = [b["text"] for b in batch]
        ids = [b["id"] for b in batch]
        
        logger.info(f"Embedding batch {i}...")
        embs = embed_texts(model, texts)
        
        upsert_items = []
        for j, vid in enumerate(ids):
            metadata = {"source": "seed", "text": texts[j], "index_pos": i+j}
            upsert_items.append((vid, embs[j].tolist(), metadata))
        
        logger.info(f"Upserting batch {i} to Pinecone...")
        ok = upsert_vectors(upsert_items)
        if not ok:
            logger.error("Upsert failed at batch %d - exiting", i)
            return False
        
        logger.info("Upserted %d/%d", min(i+batch_size, total), total)
        time.sleep(0.5) # rate limit politeness

    final_count = count_vectors()
    logger.info("Migration complete. Index count (approx): %s", final_count)
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="tests/attacks/seed_attacks.json")
    parser.add_argument("--batch", type=int, default=256)
    args = parser.parse_args()
    run_migration(args.source, batch_size=args.batch)
