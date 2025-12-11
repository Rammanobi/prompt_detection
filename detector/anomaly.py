import os, pickle
from pathlib import Path
from typing import List, Tuple
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
import math

MODEL_DIR = Path("detector/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

ANOMALY_MODEL_PKL = MODEL_DIR / "isolation_forest.pkl"
PCA_PKL = MODEL_DIR / "pca.pkl"
SCALER_PKL = MODEL_DIR / "scaler.pkl"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def clean_text(text: str) -> str:
    # simple cleaner matching main code
    return text.lower().strip()

def meta_features(text: str) -> List[float]:
    text = text or ""
    n_chars = len(text)
    n_words = len(text.split())
    n_punct = sum(1 for c in text if c in ".,;:!?")
    n_digits = sum(1 for c in text if c.isdigit())
    non_ascii = sum(1 for c in text if ord(c) > 127)
    zero_width = sum(1 for c in text if c in ("\u200B","\u200C","\u200D","\uFEFF"))
    
    entropy = 0.0
    if n_chars > 0:
        from collections import Counter
        counts = Counter(text)
        probs = [v/n_chars for v in counts.values()]
        entropy = -sum(p*math.log2(p) for p in probs if p>0)
        
    return [n_chars, n_words, n_punct, n_digits, non_ascii, zero_width, entropy]

def build_features(texts: List[str], embed_model: SentenceTransformer, pca: PCA=None) -> Tuple[np.ndarray, PCA]:
    cleaned = [clean_text(t) for t in texts]
    embs = embed_model.encode(cleaned, convert_to_numpy=True, normalize_embeddings=True)
    if embs.ndim == 1:
        embs = np.expand_dims(embs, 0)
    
    if pca is None:
        # Fit PCA
        dim = min(32, embs.shape[1], len(texts))
        pca = PCA(n_components=dim)
        pca.fit(embs)
    
    emb_reduced = pca.transform(embs)
    metas = np.array([meta_features(t) for t in texts], dtype=float)
    X = np.hstack([emb_reduced, metas])
    return X, pca

def train(normal_texts: List[str], contamination:float=0.01) -> None:
    print(f"Training anomaly model with {len(normal_texts)} samples...")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    X, pca = build_features(normal_texts, embed_model, pca=None)
    
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    
    clf = IsolationForest(n_estimators=200, contamination=contamination, random_state=42, n_jobs=-1)
    clf.fit(Xs)
    
    with open(ANOMALY_MODEL_PKL, "wb") as fh: pickle.dump(clf, fh)
    with open(PCA_PKL, "wb") as fh: pickle.dump(pca, fh)
    with open(SCALER_PKL, "wb") as fh: pickle.dump(scaler, fh)
    print("Training complete. Models saved.")

def load() -> Tuple[SentenceTransformer, PCA, StandardScaler, IsolationForest]:
    if not ANOMALY_MODEL_PKL.exists():
        raise FileNotFoundError("Anomaly model not found.")
    
    with open(ANOMALY_MODEL_PKL, "rb") as fh: clf = pickle.load(fh)
    with open(PCA_PKL, "rb") as fh: pca = pickle.load(fh)
    with open(SCALER_PKL, "rb") as fh: scaler = pickle.load(fh)
    
    # Load embed model (cached)
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    return embed_model, pca, scaler, clf

def score_texts(texts: List[str], embed_model, pca, scaler, clf) -> List[float]:
    X, _ = build_features(texts, embed_model, pca=pca)
    Xs = scaler.transform(X)
    
    # decision_function: positive=normal, negative=outlier
    # Invert to make 1.0 = highly anomalous, 0.0 = normal
    raw = clf.decision_function(Xs)
    # simple heuristic scaling: raw < 0 is outlier. 
    # Sigmoid or linear scaling can refer here.
    scores = 1 / (1 + np.exp(raw)) # Sigmoid-like: raw high -> score low
    return scores.tolist()
