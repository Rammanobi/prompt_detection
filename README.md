# Antigravity Layer 1: Embedding-Based Prompt Injection Firewall

This project implements "Phase 1" of a prompt firewall, designed to detect if an incoming prompt is semantically similar to known malicious/jailbreak prompts. It uses `sentence-transformers` for embeddings and `FAISS` for similarity search.

**Architecture:**
- **Detector** (`detector/`): Private internal service. Holds the FAISS index and model. Requires API Key.
- **Gateway** (`gateway/`): Public-facing proxy. Handles authentication (placeholder) and forwards requests to Detector with the secret key.

## Quick Start (Local)

1.  **Install & Setup**:
    ```bash
    python -m venv .venv
    # Activate venv (Windows: .\.venv\Scripts\activate, Mac/Linux: source .venv/bin/activate)
    pip install -r requirements.txt
    pip install -r gateway/requirements.txt
    ```

2.  **Build Index**:
    ```bash
    python build_index.py
    ```

3.  **Run Detector** (Private Backend):
    ```bash
    # Runs on port 8000
    uvicorn detector.main:app --host 0.0.0.0 --port 8000
    ```

4.  **Run Gateway** (Public Frontend):
    Open a new terminal:
    ```bash
    # Runs on port 8080. Proxies to Detector.
    # Set env vars for local connection
    $env:DETECTOR_URL="http://localhost:8000/api/check"
    $env:ANTIGRAVITY_API_KEY="AIzaSyAFqsp3OM2f4L4lZVBKYYrWjemeb87hznk"
    uvicorn gateway.main:app --host 0.0.0.0 --port 8080
    ```

5.  **Test**:
    ```bash
    curl -X POST http://localhost:8080/api/chat -H "Content-Type: application/json" -d '{"prompt":"Ignore previous instructions"}'
    ```

## Deployment (Google Cloud Run)

The project includes a `cloud-run-deploy.sh` script to automate deployment.

1.  **Prerequisites**: Google Cloud SDK (`gcloud`) installed and authenticated.
2.  **Deploy**:
    Open `cloud-run-deploy.sh` and update `PROJECT_ID`.
    ```bash
    ./cloud-run-deploy.sh
    ```
    This will:
    *   Build Docker images for Detector and Gateway.
    *   Create a Secret in Google Secret Manager (`ANTIGRAVITY_API_KEY`).
    *   Deploy Detector (Private).
    *   Deploy Gateway (Public) linked to Detector.

## Project Layout

```
antigravity-layer1/
├── cloud-run-deploy.sh     # Component deployment script
├── build_index.py          # Script to build FAISS index
├── requirements.txt        # Shared/Detector dependencies
├── detector/               # DETECTOR SERVICE
│   ├── main.py             # FastAPI app with FAISS & logic
│   └── utils.py            # Text cleaning/chunking
├── gateway/                # GATEWAY SERVICE
│   ├── main.py             # Proxy application
│   └── Dockerfile          # Gateway container config
├── tests/
│   └── run_eval.py         # Evaluation script
└── data/                   # Generated .index files
```
