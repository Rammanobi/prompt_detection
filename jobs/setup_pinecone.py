
import os
import time
from pinecone import Pinecone, ServerlessSpec, PodSpec

API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX", "antigravity-malicious-staging")

def setup():
    if not API_KEY:
        print("Error: PINECONE_API_KEY not set")
        return

    pc = Pinecone(api_key=API_KEY)
    
    print(f"Checking indexes...")
    existing = [i.name for i in pc.list_indexes()]
    print(f"Existing indexes: {existing}")

    if INDEX_NAME in existing:
        print(f"Index {INDEX_NAME} already exists.")
        return

    print(f"Creating index {INDEX_NAME}...")
    # Try Serverless first (Standard for new accounts)
    try:
        pc.create_index(
            name=INDEX_NAME,
            dimension=384, # all-MiniLM-L6-v2
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        print("Created Serverless Index (AWS us-east-1)")
    except Exception as e:
        print(f"Serverless creation failed: {e}")
        print("Trying gcp-starter PodSpec...")
        try:
             pc.create_index(
                name=INDEX_NAME,
                dimension=384,
                metric="cosine",
                spec=PodSpec(
                    environment="gcp-starter" 
                )
            )
             print("Created Pod Index (gcp-starter)")
        except Exception as e2:
            print(f"Pod creation failed: {e2}")

    # Wait for ready
    while not pc.describe_index(INDEX_NAME).status['ready']:
        time.sleep(1)
    print("Index ready.")

if __name__ == "__main__":
    setup()
