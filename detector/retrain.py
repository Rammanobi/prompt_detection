from google.cloud import firestore
from detector.anomaly import train
import argparse
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=5000)
    args = parser.parse_args()
    
    try:
        db = firestore.Client()
        docs = db.collection("normal_prompts").limit(args.limit).stream()
        texts = [d.to_dict().get("text") for d in docs if d.to_dict().get("text")]
    except Exception as e:
        print(f"Firestore error: {e}")
        texts = []

    if len(texts) < 50:
        print("Not enough data to retrain (need >50).")
        sys.exit(0)
        
    train(texts)

if __name__ == "__main__":
    main()
