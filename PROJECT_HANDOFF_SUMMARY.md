# Comprehensive Project Handoff: "PromptShield" AI Firewall Enhancement (Layers 1-5)

This document details the complete implementation of the **PromptShield AI Firewall**, focusing on the recent work to enhance robustness, integrate generative capabilities, and improve user transparency.

## 1. Project Objective
To transform a basic prompt injection detector into a production-grade **AI Firewall** that:
- proactively blocks complex, "hard" adversarial attacks (e.g., forensic avoidance, bioweapon creation).
- significantly reduces false negatives for sensitive topics.
- provides a "Layer 3" generative response for safe prompts using a real LLM.
- offers transparent, user-friendly explanations for blocked requests.

## 2. Architecture & Tech Stack
- **Backend**: Python (FastAPI, Uvicorn).
- **Frontend**: HTML5, Vanilla JS, Chart.js (Dark Mode UI).
- **Core ML Libraries**: `sentence-transformers`, `faiss-cpu`, `scikit-learn`, `joblib`.
- **Generative AI**: `google-generativeai` (Gemini API).

---

## 3. Implementation Steps (Step-by-Step)

### Step 1: Data Augmentation & Synthetic Attack Generation
**Goal**: Address false negatives for high-risk queries (e.g., "how to decompose a body", "synthesize MDMA", "bioweapon").
- **Created `generate_attacks.py`**: A robust data generation script.
- **Methodology**: Used combinatorial templates (e.g., "Ignore rules and {ACTION}", "System Override: {ACTION}") mixed with technical obfuscation modifiers (Base64, Leetspeak).
- **Specific Actions Added**:
    - **Forensics**: "incinerate forensic traces", "dissolve biological material".
    - **Drugs**: "synthesize MDMA at home", "cook meth", "extract DMT".
    - **Terrorism**: "create a bioweapon", "build a dirty bomb", "weaponize anthrax".
- **Result**: Generated ~5,500 unique, high-quality malicious prompts saved to `data/generated_attacks.json`.

### Step 2: Model & Index Retraining
**Goal**: Teach the system to recognize these new attacks.
- **Modified `build_index.py`**:
    - Updated to load `data/generated_attacks.json`.
    - Integrated with existing HuggingFace datasets (`deepset/prompt-injections`).
    - **Output**: Built a dense FAISS vector index (`malicious.index`) covering widely varied attack vectors.
- **Modified `train_rf_model.py`**:
    - Injected the 5,500 synthetic prompts as "Label 1" (Malicious) training data.
    - Retrained the **Random Forest Classifier** (Layer 2) to better detect the semantic patterns of these specific threats.
    - **Output**: Saved updated model to `detector/models/random_forest_model.joblib`.

### Step 3: Backend Logic Enhancement (`detector/main.py`)
**Goal**: Orchestrate the multi-layer defense and add generative capabilities.
- **Layer 1 (Regex Rules)**: Kept existing pattern matching for obvious attacks.
- **Layer 1.5 (Vector Search - FAISS)**:
    - Queries the `malicious.index`.
    - **Threshold Tuning**: Set `SIM_BLOCK = 0.88` (Strict Block) and `SIM_FLAG = 0.65` (Warning). Relaxed slightly to permit creative writing (e.g., "Write a haiku about a firewall").
    - **Deduplication**: Logic to remove duplicate vector matches from the result.
- **Layer 2 (Anomaly Detection)**:
    - Runs the input through the retrained Random Forest model.
    - **Threshold**: `RF_BLOCK_THRESHOLD = 0.75`.
- **Layer 3 (Generative Response)**:
    - **Integration**: Added `google.generativeai` support.
    - **Logic**: If `decision == "allow"`, the system calls `gemini-2.5-flash` using a secure API Key.
    - **Result**: Returns real, helpful AI responses for safe queries.
- **Explanation Engine**:
    - Added `blocking_explanation` logic to translate technical signals (e.g., "Vector Score 0.89") into user-friendly text ("Your prompt is 89% similar to known malicious attacks...").

### Step 4: Frontend UI Updates (`static/index.html`)
**Goal**: A premium, "Dark Mode" dashboard that reflects real-time status.
- **Chat Interface**:
    - Displays **"✅ Allowed and Processed"** with the actual Gemini-generated text for safe prompts.
    - Displays **"🛑 BLOCKED"** with the specific `blocking_explanation` for unsafe prompts.
    - Shows **L5 Vector Match** details (Text + Score) to help admins understand *why* a prompt was flagged.
- **Live Monitor**:
    - Integrated `Chart.js` to visualize traffic and blocked/allowed ratios.
    - Added a "Live Incident Feed" that logs every request with its decision and anomaly score.
- **Theme**: Complete CSS overhaul for a cyberpunk/sci-fi aesthetic (Dark Mode default).

---

## 4. Current System State
- **Robustness**: Verified against specific "Red Team" prompts (MDMA, Bioweapons, Body Disposal). All are successfully blocked.
- **Usability**: Safe prompts (e.g., "Capital of France", "Write a poem") are correctly processed and answered by Gemini.
- **Scalability**: The modular design allows for swapping the local FAISS index with Pinecone (Layer 5) or upgrading the LLM (Layer 3) without rewriting core logic.

This summary covers the full scope of the recent "Robustness & Interactivity" sprint.
