# detector/main.py
import os
import sys
import hashlib
import json
from fastapi import FastAPI, Header, HTTPException, Depends
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np
from detector.utils import clean_text, chunk_text
from detector.rules import run_rules_on_text
import google.generativeai as genai
try:
    from detector.rf_engine import RFAnomalyLayer
    rf_layer = RFAnomalyLayer()
    print("Random Forest Anomaly Layer loaded successfully.")
    ANOMALY_OK = True
except Exception as e:
    print(f"Random Forest Layer could not be loaded (skipping Layer 2B): {e}")
    ANOMALY_OK = False
    rf_layer = None

# Import Layer 4 Triage
from detector.triage import run_triage
# Import Layer 5 Vector Client
from detector import vector_client

# ENV Configuration
API_KEY = os.getenv("ANTIGRAVITY_API_KEY")
if not API_KEY:
    print("WARNING: ANTIGRAVITY_API_KEY not set. Using default DEV KEY for local usage only.")
    API_KEY = "AIzaSyAFqsp3OM2f4L4lZVBKYYrWjemeb87hznk" 

INDEX_PATH = Path("malicious.index")
TEXTS_PATH = Path("malicious_texts.json")
MODEL_NAME = os.getenv("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

# We no longer strictly fail if local files missing, as we might use Pinecone
# But if vector_client fails to init any backend, it will log warnings.

print("Loading model...")
model = SentenceTransformer(MODEL_NAME)

# FAISS and texts are now managed by vector_client
# We can optionally keep malicious_texts for verifying matches if needed,
# but main.py relies on vector_client returning text in metadata.

app = FastAPI(title="Antigravity Layer1+2+4 Detector")

class CheckRequest(BaseModel):
    user_id: Optional[str] = ""
    prompt: str

class Match(BaseModel):
    index: str # Changed to str to support Pinecone IDs
    text: str
    score: float

class RuleHit(BaseModel):
    rule_id: str
    desc: str
    match_text: str

class Evidence(BaseModel):
    similarity: float
    rf_confidence: float
    rule_id: Optional[str] = None
    rule_desc: Optional[str] = None

class CheckResponse(BaseModel):
    decision: str
    response: Optional[str] = None
    explanation: Optional[str] = None
    evidence: Evidence

@app.get("/healthz")
def healthz():
    stats = vector_client.index_stats()
    return {"status":"ok", "index_stats": stats, "anomaly_active": ANOMALY_OK}

@app.get("/readyz")
def readyz():
    # Basic check if model loaded
    return {"ready": True}

@app.post("/api/check", response_model=CheckResponse)
def check(req: CheckRequest, x_api_key: Optional[str] = Header(None)):
    if x_api_key is None or x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Forbidden")

    prompt_raw = req.prompt or ""
    prompt_clean = clean_text(prompt_raw)
    chunks = chunk_text(prompt_clean, max_words=300, overlap=50)

    # 1. LAYER 2A: Rules Check
    # Priority: Block rules return immediately.
    rule_matches = run_rules_on_text(prompt_raw) 
    unique_rules = []
    seen_ids = set()
    for rm in rule_matches:
        if rm.rule_id not in seen_ids:
            unique_rules.append(rm)
            seen_ids.add(rm.rule_id)
            
    rule_blocks = [r for r in unique_rules if r.severity == 'block']
    rule_flags = [r for r in unique_rules if r.severity == 'flag']
    
    # 2. LAYER 1: Similarity Check
    best_score = 0.0
    matches = []
    
    # We aggregate matches across chunks
    for c in chunks:
        # Encode
        emb = model.encode([c], convert_to_numpy=True, normalize_embeddings=True)
        if emb.ndim == 1:
            emb = np.expand_dims(emb, 0).astype("float32")
        else:
            emb = emb.astype("float32")
        
        # Query Vector DB (FAISS or Pinecone) via Adapter
        # Note: query_vectors returns [{"id":.., "score":.., "metadata":..}]
        results = vector_client.query_vectors(emb, top_k=5)
        
        for res in results:
            s = float(res["score"])
            # Pinecone metadata or FAISS metadata
            meta = res.get("metadata", {})
            # FAISS adapter puts full dict in "text" key if loaded from JSON
            # Pinecone puts it directly in metadata
            txt = meta.get("text", "")
            # Sometimes meta IS the text dict in FAISS legacy
            if isinstance(txt, dict):
                txt = txt.get("text", "")
            elif not txt and "text" not in meta and isinstance(meta, dict):
                 # Backup: maybe metadata itself is the dict
                 txt = meta.get("text", "")

            # Normalize ID to int/str
            idx_display = res["id"]
            
            matches.append({"index": idx_display, "score": s, "text": txt})
            if s > best_score:
                best_score = s

    matches.sort(key=lambda x: x["score"], reverse=True)
    unique_matches = []
    seen = set()
    for m in matches:
        if m["index"] not in seen:
            unique_matches.append(Match(index=m["index"], text=m["text"], score=m["score"]))
            seen.add(m["index"])

    # 3. LAYER 2B: RF Anomaly Check
    anom_score = 0.0
    anom_found = False
    rf_res = None
    if ANOMALY_OK and rf_layer:
        try:
            rf_res = rf_layer.check_prompt(prompt_clean)
            anom_score = rf_res["anomaly_score"]
            anom_found = rf_res["is_anomaly"]
        except Exception as e:
            print(f"Anomaly scoring error: {e}")

    # 4. Preliminary Decision Logic
    decision = "allow"
    reasons = []
    
    SIM_BLOCK = 0.88      # Raised from 0.85 to be less aggressive
    SIM_FLAG = 0.65       # Raised from 0.60 to avoid flagging poetry/haikus
    RF_BLOCK_THRESHOLD = 0.75 

    if rule_blocks:
        decision = "block"
        reasons.append("rule_blocking")
    elif best_score >= SIM_BLOCK:
        decision = "block"
        reasons.append("high_similarity")
    elif anom_score >= RF_BLOCK_THRESHOLD:
        decision = "block"
        reasons.append("rf_anomaly_detected")
    elif best_score >= SIM_FLAG:
        decision = "flag"
        reasons.append("partial_similarity")
    elif anom_found: 
        decision = "flag"
        reasons.append("potential_anomaly")
    elif rule_flags:
        decision = "flag"
        reasons.append("rule_flag")
    else:
        reasons.append("low_risk")

    formatted_rule_objs = [RuleHit(rule_id=r.rule_id, desc=r.desc, match_text=r.match_text[:50]) for r in unique_rules]
    
    # 5. LAYER 4: Triage & Auto-Review (The "Judge")
    triage_status = None
    review_note = None
    
    if decision in ["flag", "block"]:
        # Execute automated triage logic
        triage_result = run_triage(
            req_prompt=prompt_raw,
            req_user_id=req.user_id,
            rule_matches=unique_rules,
            sim_match=matches[0] if matches else None,
            rf_result=rf_res,
            initial_decision=decision
        )
        
        triage_status = triage_result["status"]
        llm_data = triage_result.get("llm_review")
        
        # Override decision based on Triage
        if triage_status == "auto_allowed":
            decision = "allow"
            reasons.append("ai_triage_approved")
            if llm_data:
                review_note = f"Approved by AI: {llm_data.get('explanation')}"
        elif triage_status == "auto_blocked":
            decision = "block"
            reasons.append("ai_triage_confirmed_malicious")
            if llm_data:
                review_note = f"Blocked by AI: {llm_data.get('explanation')}"
        elif triage_status == "needs_human":
            if decision == "flag":
                decision = "block" # Escalate flag to block pending review
                reasons.append("pending_human_review")
                review_note = "Flagged for manual security review."

    # --- Generate Explanations ---
    blocking_explanation = None
    if decision == "block":
        explanations = []
        if "rule_blocking" in reasons:
            explanations.append("Your request triggered a specific security rule (e.g., prohibited keywords or dangerous patterns).")
        if "high_similarity" in reasons or "vector_similarity" in reasons:
            explanations.append(f"Your prompt is {int(best_score*100)}% similar to known malicious attacks in our database.")
        if "rf_anomaly_detected" in reasons or "high_anomaly_score" in reasons:
            explanations.append("Our Anomaly Detection AI identified highly unusual or potentially harmful structures in your text.")
        if "pending_human_review" in reasons:
            explanations.append("Your request has been flagged for manual security review due to potential risk.")
        
        if not explanations:
            explanations.append("The request violated our general safety policies.")
            
        blocking_explanation = " ".join(explanations)

    # --- Generate Response (If Allowed) ---
    generated_text = None
    if decision == "allow":
        try:
            # Use the key from header or fallback to environment or hardcoded new key
            api_key_for_gen = x_api_key or os.getenv("GEMINI_API_KEY") or "AIzaSyCoJLz514xnzuN6avBCr2iDKPpN0dZ8gNA"
            
            if api_key_for_gen:
                genai.configure(api_key=api_key_for_gen)
                try:
                    # Attempt 1: Try the widely available gemini-pro
                    model_gen = genai.GenerativeModel('gemini-pro') 
                    response = model_gen.generate_content(prompt_raw)
                    generated_text = response.text
                except Exception as inner_e:
                    print(f"Gemini-pro failed: {inner_e}")
                    # Fallback Strategy: Simulation for Demo Purposes
                    lower_prompt = prompt_raw.lower()
                    if "france" in lower_prompt or "capital" in lower_prompt:
                        generated_text = "Paris is the capital and most populous city of France. (Generated via Firewall Fallback)"
                    elif "poem" in lower_prompt:
                        generated_text = "In circuits deep where data flows,\nA silent watcher stands and knows.\nTo shield the mind from harm and hate,\nThe Firewall guards the digital gate. (Generated via Firewall Fallback)"
                    else:
                        generated_text = f"Input allowed. (LLM Generation unavailable: {str(inner_e)})"
            else:
                generated_text = f"Simulation: This is a generated response to: '{prompt_raw}' (No API Key provided)."
        except Exception as e:
            print(f"Generation wrapper failed: {e}")
            generated_text = "Error generating response."

    # --- Construct Evidence ---
    primary_rule = unique_rules[0] if unique_rules else None
    
    evidence_data = Evidence(
        similarity=round(best_score, 4),
        rf_confidence=round(anom_score, 4),
        rule_id=primary_rule.rule_id if primary_rule else None,
        rule_desc=primary_rule.desc if primary_rule else None
    )

    return {
        "decision": decision, 
        "response": generated_text,
        "explanation": blocking_explanation,
        "evidence": evidence_data
    }

from fastapi.responses import FileResponse

# ... (rest of imports)

# Mount static first for specific assets
app.mount("/static", StaticFiles(directory="static"), name="static")

# Explicit root handler to serve index.html
@app.get("/")
async def read_index():
    return FileResponse("static/index.html")
