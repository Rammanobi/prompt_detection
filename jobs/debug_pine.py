
import pinecone
print(f"Version: {pinecone.__version__}")
try:
    from pinecone import Pinecone
    pc = Pinecone(api_key="TEST")
    print("Init OK")
except Exception as e:
    print(f"Error: {e}")
