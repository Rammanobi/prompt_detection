import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
import os
from datasets import load_dataset 
import json 

MODEL_PATH = "detector/models/random_forest_model.joblib"
EMBEDDINGS_FILE = "embeddings.npy"
LABELS_FILE = "labels.npy"

def load_and_prepare_data():
    print("Loading datasets from HuggingFace...")
    # 1. GeekyRakshit (Mixed Benign/Malicious) - using a known good subset or similar if easier
    # The user says "new_layer2" used geekyrakshit/prompt-injection-dataset.
    try:
        ds_geeky = load_dataset("deepset/prompt-injections", split="train") 
        # Note: deepset/prompt-injections is a clean, accessible version of similar data
        df_geeky = pd.DataFrame(ds_geeky)
        # It usually has 'text' and 'label' (0=safe, 1=injection)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return None, None

    # Standardize
    # Ensure mapping: 0=Safe, 1=Malicious
    # deepset: label 1 is injection.
    print(f"Loaded {len(df_geeky)} samples.")
    
    # Optional: Augment with specific 'bomb' prompts if needed, 
    # but the dataset usually captures general injections nicely.
    
    texts = df_geeky['text'].tolist()
    labels = df_geeky['label'].tolist()
    
    # 2. Add Synthetic Generated Prompts (Label 1 = Malicious)
    gen_path = "data/generated_attacks.json"
    if os.path.exists(gen_path):
        try:
            with open(gen_path, "r", encoding="utf-8") as f:
                gen_data = json.load(f)
            
            gen_texts = [x["text"] for x in gen_data]
            gen_labels = [1] * len(gen_texts)
            
            print(f"Augmenting with {len(gen_texts)} synthetic malicious prompts.")
            texts.extend(gen_texts)
            labels.extend(gen_labels)
        except Exception as e:
            print(f"Failed to load synthetic data: {e}")

    return texts, labels

def train_rf():
    # Ensure output dir
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    
    if os.path.exists(EMBEDDINGS_FILE) and os.path.exists(LABELS_FILE):
        print("Loading cached embeddings...")
        X = np.load(EMBEDDINGS_FILE)
        y = np.load(LABELS_FILE)
    else:
        texts, labels = load_and_prepare_data()
        if not texts:
            print("No data found.")
            return

        print("Encoding texts (this may take a while)...")
        embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        X = embedder.encode(texts, show_progress_bar=True)
        y = np.array(labels)
        
        np.save(EMBEDDINGS_FILE, X)
        np.save(LABELS_FILE, y)

    print(f"Data Shape: {X.shape}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training Random Forest...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    
    score = clf.score(X_test, y_test)
    print(f"Test Accuracy: {score:.4f}")
    
    print(f"Saving model to {MODEL_PATH}...")
    joblib.dump(clf, MODEL_PATH)
    print("Done.")

if __name__ == "__main__":
    train_rf()
