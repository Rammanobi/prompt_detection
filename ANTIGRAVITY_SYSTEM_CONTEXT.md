# Antigravity System State: Layers 1-4 Complete

**To External Agent:** Please ingest the following comprehensive technical state of the **Antigravity** project. We have successfully architected, implemented, and deployed the first 4 layers of defense.

---

## 1. Project Overview
**Goal:** Build a production-grade "AI Firewall" to intercept and block malicious prompts (Jailbreaks, Injections, PII leaks) before they reach an LLM.
**Tech Stack:** Python 3.11, FastAPI, Docker, Google Cloud Run, Firestore, generic LLM SDKs.

## 2. Architecture: Multi-Stage Waterfall Defense
The system processes every request through 4 sequential layers. If any layer triggers a block, the request is rejected immediately (or reviewed).

### **Layer 1: Semantic Defense (Similarity)**
*   **Theory:** "If it looks like a known attack, it is an attack."
*   **Implementation:**
    *   **Engine:** `sentence-transformers` (all-MiniLM-L6-v2) to vectorize triggers.
    *   **Database:** `FAISS` (Facebook AI Similarity Search) running in-memory.
    *   **Data:** A seed list of ~250 confirmed attacks (`tests/attacks/seed_attacks.json`).
    *   **Logic:** Calculates cosine similarity. If `score > 0.85` -> **BLOCK**.

### **Layer 2: Deterministic Rules (Heuristics)**
*   **Theory:** "Catch the obvious stuff instantly."
*   **Implementation:**
    *   **Engine:** Optimized Python RegEx (`detector/rules.py`).
    *   **Data:** `rules/rules.yaml` definition file (~50+ patterns).
    *   **Coverage:**
        *   **Jailbreaks:** "Ignore previous instructions", "DAN Mode".
        *   **Dangerous Ops:** `rm -rf`, `wget`, `curl`.
        *   **Secrets:** PEM Keys, AWS Keys.
    *   **Logic:** If Severity `block` -> **BLOCK** (Score 1.0).

### **Layer 3: Advanced Anomaly Detection (AI Classifier)**
*   **Theory:** "Detect the *style* and *intent* of attacks, not just keywords."
*   **Implementation:**
    *   **Model:** **Random Forest Classifier** (`scikit-learn`), trained on 5,000+ examples (HuggingFace: `deepset/prompt-injections`).
    *   **Feature Engineering:** Re-uses the 384-dimensional embeddings from Layer 1.
    *   **File:** `detector/rf_engine.py` & `models/random_forest_model.joblib`.
    *   **Logic:** Returns an `anomaly_score` (0.0 to 1.0). If `score > 0.75` -> **BLOCK**.

### **Layer 4: Automated AI Triage (The "Judge")**
*   **Theory:** "Don't burn human time on False Positives. Let an AI double-check."
*   **Status:** **ACTIVE** (Deployed in Step 2200).
*   **Implementation:**
    *   **Orchestrator:** `detector/triage.py`.
    *   **Reviewer:** Google Gemini 1.5 Flash (via `google-generativeai`).
    *   **Workflow:**
        1.  If Layers 1/2/3 flag or block a request...
        2.  The **Detector** compiles an "Evidence Packet" (Prompt Hash, Rule ID, Sim Score, RF Confidence).
        3.  The **Reviewer** answers a structured JSON prompt: `{"label": "false_positive", "reason": "It's just a poem."}`.
        4.  **Enforcement:**
            *   `auto_allowed`: System overrides the block and allows the user.
            *   `needs_human` / `confirm_malicious`: System blocks and logs to Firestore.
    *   **Logging:** All events saved to **Google Cloud Firestore** (`quarantine` collection) for audit.

---

## 3. Deployment Topology
*   **Service A: Gateway (`antigravity-layer1-gateway`)**
    *   Public-facing URL.
    *   Handles Auth (API Keys) & Client Communication.
    *   Passes requests to Detector via internal API.
*   **Service B: Detector (`antigravity-layer1-detector`)**
    *   Private (Ingress internal-only).
    *   Hosts the "Brain" (FAISS Index, Rules Engine, RF Model, Triage Logic).
    *   **Scalability:** Stateless, scaling on Cloud Run (currently 2GB Memory).

## 4. Current State
*   **Functionality:** Fully operational. Tested against SQL injections (Blocked), Poems (Allowed/Reviewed), and Greeting (Allowed).
*   **Next Phase:** Ready for **Layer 5** (Production Scaling / Managed Vector DB).
