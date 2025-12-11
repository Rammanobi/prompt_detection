from sentence_transformers import SentenceTransformer
import os

model_name = os.getenv("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
print(f"Downloading model: {model_name}...")
model = SentenceTransformer(model_name)
model.save("model_cache")
print("Model saved to model_cache")
