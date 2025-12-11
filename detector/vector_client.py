# detector/vector_client.py
"""
Vector backend adapter for Antigravity.
Supports two backends:
 - FAISS (local) -- default fallback
 - PINECONE (managed) -- enabled by env VECTOR_BACKEND=pinecone

Usage:
  from detector.vector_client import query_vectors, upsert_vectors, count_vectors

Environment variables:
  VECTOR_BACKEND (faiss | pinecone) default: faiss
  PINECONE_API_KEY, PINECONE_ENV, PINECONE_INDEX  (if using pinecone)
  FAISS_INDEX_PATH, FAISS_META_PATH  (if using faiss)
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

BACKEND = os.getenv("VECTOR_BACKEND", "faiss").lower()

_pine_index = None
_pine_available = False
_faiss_index = None
_faiss_texts = []

# --- Pinecone backend -------------------------------------------------------
if BACKEND == "pinecone":
    try:
        from pinecone import Pinecone
        PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
        PINECONE_INDEX = os.getenv("PINECONE_INDEX", "antigravity-malicious-staging")

        if not PINECONE_API_KEY:
            raise RuntimeError("PINECONE_API_KEY not set — Pinecone disabled.")

        # Updated for Pinecone v3/v4 SDK
        pc = Pinecone(api_key=PINECONE_API_KEY)
        _pine_index = pc.Index(PINECONE_INDEX)
        
        logger.info("Pinecone client initialized, index=%s", PINECONE_INDEX)
        _pine_available = True
    except Exception as e:
        logger.warning("Pinecone init failed or not configured: %s — falling back to FAISS", e)
        BACKEND = "faiss"
        _pine_available = False

# --- FAISS backend ----------------------------------------------------------
# We initialize FAISS if backend is faiss OR if pinecone failed specific init
if BACKEND == "faiss":
    try:
        import faiss
    except Exception as e:
        # If faiss is missing, we are in trouble unless strictly using pinecone (but we fell back here)
        logger.error("faiss not available: %s", e)
        # We don't raise immediately to allow import, but query will fail.
    
    FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "malicious.index")
    FAISS_META_PATH = os.getenv("FAISS_META_PATH", "malicious_texts.json")
    
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(FAISS_META_PATH):
        try:
            import faiss
            _faiss_index = faiss.read_index(FAISS_INDEX_PATH)
            with open(FAISS_META_PATH, "r", encoding="utf-8") as fh:
                _faiss_texts = json.load(fh)
            logger.info("FAISS index loaded (%s vectors)", len(_faiss_texts))
        except Exception as e:
             logger.error("Failed to load FAISS index: %s", e)
    else:
        logger.warning("FAISS index file not found at %s — functionality limited.", FAISS_INDEX_PATH)

# --- Common helper functions ------------------------------------------------
def _to_numpy(emb: Any) -> np.ndarray:
    """Ensure embedding is a 2D numpy array float32"""
    arr = np.array(emb, dtype=np.float32)
    if arr.ndim == 1:
        arr = np.expand_dims(arr, 0)
    return arr

# --- Public API: query_vectors ----------------------------------------------
def query_vectors(emb: Any, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Query the configured vector backend.

    Returns a list of matches with structure:
      {"id": <id_or_index>, "score": <float>, "metadata": {...}}
    """
    emb_arr = _to_numpy(emb)

    if BACKEND == "pinecone" and _pine_available:
        try:
            # Pinecone client expects a list of floats for a single query vector
            # query(vector=[...], ...)
            vec_list = emb_arr[0].tolist() 
            res = _pine_index.query(vector=vec_list, top_k=top_k, include_metadata=True, include_values=False)
            matches = res.get("matches", [])
            out = []
            for m in matches:
                out.append({
                    "id": m["id"],
                    "score": float(m["score"]),
                    "metadata": m.get("metadata", {})
                })
            return out
        except Exception as e:
            logger.exception("Pinecone query failed: %s. Falling back to FAISS if available.", e)
            # Fall through to FAISS logic below if index exists? 
            # If backend was explicitly pinecone, maybe just error or return empty?
            # User feedback says "fails gracefully (stays on FAISS)". 
            # But earlier decision was explicit switch. 
            # I'll check _faiss_index to see if we can fallback.
            if not _faiss_index:
                return []

    # FAISS fallback
    if _faiss_index is None:
        logger.warning("No FAISS index available for query.")
        return []

    D, I = _faiss_index.search(emb_arr, top_k)
    out = []
    # D is [[score1, score2...]], I is [[idx1, idx2...]]
    for scores, idxs in zip(D, I):
        for score, idx in zip(scores, idxs):
            if idx < 0:
                continue
            meta = _faiss_texts[idx] if idx < len(_faiss_texts) else {"text": ""}
            # Standardize output
            # FAISS raw index is just an int, we convert to str id
            out.append({
                "id": str(idx),
                "score": float(score),
                "metadata": {"text": meta.get("text", "") if isinstance(meta, dict) else meta}
            })
    return out

# --- Public API: upsert_vectors ---------------------------------------------
def upsert_vectors(items: List[Tuple[str, List[float], Dict[str, Any]]]) -> bool:
    """
    Upsert vectors into backend.
    items: list of (id_str, vector_list, metadata_dict)
    Returns True on success.
    """
    if BACKEND == "pinecone" and _pine_available:
        try:
            # Pinecone expects vectors=[{"id": "A", "values": [...], "metadata": ...}]
            vectors = []
            for (vid, vvals, vmeta) in items:
                # vvals might be numpy
                vals_clean = vvals
                if isinstance(vals_clean, np.ndarray):
                    vals_clean = vals_clean.tolist()
                vectors.append({"id": vid, "values": vals_clean, "metadata": vmeta})
            
            _pine_index.upsert(vectors=vectors)
            return True
        except Exception as e:
            logger.exception("Pinecone upsert failed: %s", e)
            return False
    else:
        logger.warning("Upsert requested but not supported for FAISS backend.")
        return False

# --- Public API: count_vectors, index_stats ---------------------------------
def count_vectors() -> Optional[int]:
    if BACKEND == "pinecone" and _pine_available:
        try:
            stats = _pine_index.describe_index_stats()
            return int(stats.get("total_vector_count", 0))
        except Exception as e:
            logger.warning("Pinecone describe_index_stats failed: %s", e)
            return None
    elif _faiss_index:
        return _faiss_index.ntotal
    return 0

def index_stats() -> Dict[str, Any]:
    s = {"backend": BACKEND}
    try:
        s["count"] = count_vectors()
    except Exception:
        s["count"] = None
    return s
