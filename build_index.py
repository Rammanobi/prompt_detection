# build_index.py
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from datasets import load_dataset
from detector.utils import clean_text
from tqdm import tqdm
import os

OUTPUT_DIR = Path("data/malicious")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TEXTS_PATH = Path("malicious_texts.json")
INDEX_PATH = Path("malicious.index")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 128

# Datasets to try
HF_DATASETS = [
    "deepset/prompt-injections",
    "Mindgard/evaded-prompt-injection-and-jailbreak-samples"
]

def load_from_hf():
    collected = []
    for ds_id in HF_DATASETS:
        try:
            ds = load_dataset(ds_id)
        except Exception as e:
            print(f"Could not load {ds_id}: {e}")
            continue
        # iterate each split
        for split in ds.keys():
            dataset = ds[split]
            for i, ex in enumerate(dataset):
                # try common fields
                for key in ("text", "prompt", "attack", "input", "instruction", "example"):
                    if key in ex and ex[key]:
                        # ex[key] might be list or str
                        if isinstance(ex[key], list):
                            for s in ex[key]:
                                collected.append({"source": ds_id, "split": split, "text": clean_text(s)})
                        else:
                            collected.append({"source": ds_id, "split": split, "text": clean_text(ex[key])})
                        break
                else:
                    # fallback: any long string field
                    for k,v in ex.items():
                        if isinstance(v, str) and len(v) > 20:
                            collected.append({"source": ds_id, "split": split, "text": clean_text(v)})
    # deduplicate by text
    unique = {}
    for ex in collected:
        txt = ex["text"]
        if txt and txt not in unique:
            unique[txt] = ex
    return list(unique.values())

def load_seed_file():
    # also load local seed attacks if present
    path = Path("tests/attacks/seed_attacks.json")
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        out = []
        for x in data:
            t = clean_text(x.get("text",""))
            if t:
                out.append({"source":"local_seed","split":None,"text":t})
        return out
    return []

def build_index(all_examples):
    model = SentenceTransformer(MODEL_NAME)
    texts = [ex["text"] for ex in all_examples]
    n = len(texts)
    if n == 0:
        raise ValueError("No texts to index.")
    print(f"Embedding {n} texts ...")
    embs_batches = []
    for i in tqdm(range(0, n, BATCH_SIZE)):
        batch_texts = texts[i:i+BATCH_SIZE]
        embs = model.encode(batch_texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        if embs.ndim == 1:
            embs = embs.reshape(1, -1)
        embs_batches.append(embs.astype("float32"))
    embs = np.vstack(embs_batches)
    d = embs.shape[1]
    print(f"Building FAISS IndexFlatIP (d={d}) ...")
    index = faiss.IndexFlatIP(d)
    index.add(embs)
    faiss.write_index(index, str(INDEX_PATH))
    TEXTS_PATH.write_text(json.dumps(all_examples, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved index and texts.")

def load_generated_attacks():
    path = Path("data/generated_attacks.json")
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            print(f"Loaded {len(data)} generated attacks.")
            return data
        except Exception as e:
            print(f"Error loading generated attacks: {e}")
    return []

def main():
    seed = load_seed_file()
    hf = load_from_hf()
    generated = load_generated_attacks()
    
    # Combined with generated
    combined = seed + hf + generated
    
    # ensure we have at least the seed set; dedupe
    unique = {}
    final = []
    for ex in combined:
        t = ex["text"]
        if t and t not in unique:
            unique[t] = True
            final.append(ex)
    print(f"Total examples to index: {len(final)}")
    build_index(final)

if __name__ == "__main__":
    main()
