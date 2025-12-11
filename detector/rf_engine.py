import joblib
from sentence_transformers import SentenceTransformer
import os
import numpy as np

class RFAnomalyLayer:
    def __init__(self, model_path='detector/models/random_forest_model.joblib', embedding_model_name='sentence-transformers/all-MiniLM-L6-v2'):
        print("Initializing Random Forest Anomaly Layer...")
        if not os.path.exists(model_path):
            # Fallback or strict error depending on deployment
            # In Docker, we bake the model, so strictly require it.
            # But locally, we just trained it.
            raise FileNotFoundError(f"RF Model not found at {model_path}.")
            
        self.model = joblib.load(model_path)
        # We re-use the sentence transformer. In main.py we might want to pass the existing instance to save RAM.
        # But for cleaner encapsulation, we can load or accept it. 
        # Ideally, accept it in check_prompt if possible, or load strictly.
        # To match the main app's model:
        try:
            self.embedder = SentenceTransformer(embedding_model_name)
        except:
            # If model name differs or offline
             self.embedder = None
        print("RF Layer Ready.")

    def check_prompt(self, prompt, embedder=None):
        """
        Checks a prompt using Random Forest Classifier.
        Args:
            prompt: Text string to check.
            embedder: Optional pre-loaded SentenceTransformer instance.
        """
        if embedder:
            emb_model = embedder
        elif self.embedder:
            emb_model = self.embedder
        else:
            raise ValueError("No embedding model available for RF Layer.")

        # Encode (returns numpy array)
        embedding = emb_model.encode([prompt], convert_to_numpy=True, normalize_embeddings=True)
        
        # Prediction: 0 = Benign, 1 = Anomaly
        # Note: In deepset/prompt-injections, 1 = Injection.
        pred = self.model.predict(embedding)[0]
        # Proba: [prob_0, prob_1]
        probs = self.model.predict_proba(embedding)[0]
        
        # If classes are [0, 1], probs[1] is probability of Anomaly
        anomaly_score = probs[1] 
        
        is_anomaly = (pred == 1)
        status = "ANOMALY FOUND" if is_anomaly else "SAFE"
        
        return {
            "is_anomaly": bool(is_anomaly),
            "status": status,
            "anomaly_score": float(anomaly_score),
            "confidence": float(max(probs)),
            "model": "RandomForest"
        }
