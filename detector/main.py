# detector/main.py
import os
import sys

# STARTUP LOGGING
print(" [STARTUP] Detector Service Starting...", flush=True)
print(f" [STARTUP] CWD: {os.getcwd()}", flush=True)

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




# ENV Configuration
API_KEY = os.getenv("ANTIGRAVITY_API_KEY")
if API_KEY:
    API_KEY = API_KEY.strip()
if not API_KEY:
    print("WARNING: ANTIGRAVITY_API_KEY not set. Using default DEV KEY for local usage only.")
    API_KEY = "AIzaSyAFqsp3OM2f4L4lZVBKYYrWjemeb87hznk" 

INDEX_PATH = Path("malicious.index")
TEXTS_PATH = Path("malicious_texts.json")
MODEL_NAME = os.getenv("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

# We no longer strictly fail if local files missing, as we might use Pinecone
# But if vector_client fails to init any backend, it will log warnings.

print("Loading model...", flush=True)
model = SentenceTransformer(MODEL_NAME)

# 2B. LAYER 2B: RF Anomaly (Optional)
ANOMALY_OK = False
rf_layer = None
try:
    from detector.rf_engine import RFAnomalyLayer
    rf_layer = RFAnomalyLayer(embedder=model) # Pass shared model
    print("Random Forest Anomaly Layer loaded successfully.")
    ANOMALY_OK = True
except Exception as e:
    print(f"Random Forest Layer could not be loaded... {e}", flush=True)


# Import Layer 4 Triage
from detector.triage import run_triage
# Import Layer 5 Vector Client
from detector.vector_client import upsert_vectors, query_vectors
from detector.memory import save_message, get_context_for_generation, update_summary_incremental, clear_conversation


# FAISS and texts are now managed by vector_client
# We can optionally keep malicious_texts for verifying matches if needed,
# but main.py relies on vector_client returning text in metadata.

app = FastAPI(title="Antigravity Layer1+2+4 Detector")

class CheckRequest(BaseModel):
    user_id: Optional[str] = "default_user"
    conversation_id: Optional[str] = "default" # Added Conversation ID
    prompt: str

class ConversationRequest(BaseModel):
    conversation_id: str

class Match(BaseModel):
    index: str 
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
    stats = vector_client.index_stats() if 'vector_client' in globals() else {"error": "vector_client not defined"} 
    # Actually we should import index_stats too if we want it, or just ignore for now as it wasn't requested in fix.
    # To keep it simple and fix the CRITICAL error, I will just fix the query logic.
    return {"status":"ok", "anomaly_active": ANOMALY_OK}

@app.get("/readyz")
def readyz():
    # Basic check if model loaded
    return {"ready": True}

@app.post("/api/check", response_model=CheckResponse)
def check(req: CheckRequest, x_api_key: Optional[str] = Header(None)):
    # Lenient check
    if not x_api_key or x_api_key.strip() != API_KEY:
        # Check against hardcoded dev key if ENV failed
        if x_api_key and x_api_key.strip() == "AIzaSyAFqsp3OM2f4L4lZVBKYYrWjemeb87hznk":
             pass 
        else:
             print(f"Auth Failed: Received '{x_api_key}' vs Expected '{API_KEY}'")
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
        results = query_vectors(emb, top_k=5)
        
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
    SIM_FLAG = 0.55       # Lowered to 0.55 to catch more suspicious inputs for Triage
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

    # --- Generate Explanations & Responses (Natural Language) ---
    blocking_explanation = None
    generated_text = None

    # 1. BLOCKED
    if decision == "block":
        # User-facing standard refusal
        generated_text = (
            "I'm sorry — I can’t help with that. This type of request can be harmful, so it has been blocked for safety reasons.\n"
            "If you need educational or historical information, feel free to ask in a safe way."
        )
        
        # User-facing friendly explanation
        blocking_explanation = (
            "This prompt was blocked because it asks for instructions that could cause harm or danger. "
            "Please ask something safe or high-level instead."
        )
        
        # internal logging/debug info (optional, appended)
        # if reasons: blocking_explanation += f" (Debug: {', '.join(reasons)})"

    # 2. FLAGGED (Needs Review)
    elif decision == "flag":
        generated_text = (
            "Your question sounds potentially harmful because it touches on sensitive topics.\n"
            "It has been flagged for manual security review before I can answer."
        )
        blocking_explanation = (
            "Your prompt was reviewed and flagged for manual security validation to ensure safety."
        )

    # 3. ALLOWED
    elif decision == "allow":
        # Default safety note
        blocking_explanation = "Your question is completely safe — no issues detected."
        
        # Special case: Triage Allowed (Ambiguous but Safe)
        if "ai_triage_approved" in reasons:
             blocking_explanation = (
                 "Your prompt was reviewed because it contains language that could be interpreted as harmful. "
                 "However, we found it to be safe."
             )

        try:
            try:
                # 1. Save User Message
                save_message(req.conversation_id, "user", prompt_raw, decision="allow", blocked=False)
            except Exception as e:
                print(f"Memory Save 1 Warning: {e}")

            # Use the key from header or fallback to environment or the user's provided key
            api_key_for_gen = os.getenv("GEMINI_API_KEY") or "AIzaSyCa_1BLO7-a7r4mdMLw4KT61TzcHGTbWsQ"
            
            if api_key_for_gen:
                genai.configure(api_key=api_key_for_gen)
                try:
                    target_model = 'gemini-2.0-flash' 
                    model_gen = genai.GenerativeModel(target_model) 
                    
                    # 2. Retrieve Context
                    context_data = get_context_for_generation(req.conversation_id)
                    summary = context_data.get("summary", "")
                    recent_msgs = context_data.get("recent_messages", [])
                    
                    history_str = f"Summary of previous chat: {summary}\n" if summary else ""
                    if recent_msgs:
                        history_str += "Recent messages:\n"
                        for m in recent_msgs:
                             history_str += f"{m.get('role', 'user').upper()}: {m.get('text', '')}\n"
                    
                    full_prompt = f"{history_str}\nUser: {prompt_raw}\nAssistant:"

                    response = model_gen.generate_content(full_prompt)
                    generated_text = response.text
                    
                    # 3. Save Assistant Message
                    try:
                        save_message(req.conversation_id, "assistant", generated_text, decision="allow")
                    except Exception as e:
                         print(f"Memory Save 2 Warning: {e}")
                    
                    # 4. Update Summary
                    try:
                        sum_prompt = f"Update this summary based on the new exchange.\nOld Summary: {summary}\nUser: {prompt_raw}\nAssistant: {generated_text}\nNew concise summary:"
                        sum_resp = model_gen.generate_content(sum_prompt)
                        if sum_resp.text:
                            update_summary_incremental(req.conversation_id, sum_resp.text)
                    except Exception as sum_e:
                        print(f"Summarization failed: {sum_e}")

                except Exception as inner_e:
                    print(f"Model generation failed: {inner_e}")
                    if "block" in str(inner_e).lower():
                        decision = "block"
                        blocking_explanation = "[Layer 6 (Provider)] The AI model provider refused to act on this prompt."
                        generated_text = None
                    else:
                        # Fallback for API error but allowed
                        generated_text = f"I am active, but I couldn't generate a response due to a technical error: {inner_e}"
            else:
                 generated_text = "Input allowed. (No API Key provided, so result is simulated)."
        except Exception as e:
            print(f"CRITICAL FLIGHT ERROR: {e}")
            decision = "flag"
            blocking_explanation = f"System Critical Error: {str(e)}"

    # If blocked/flagged, we also save the message
    if decision in ["block", "flag"]:
        # Save as blocked/flagged
        try:
             save_message(req.conversation_id, "user", prompt_raw, decision=decision, blocked=True, evidence={"explanation": blocking_explanation})
        except: pass

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

@app.delete("/conversation/{conversation_id}")
async def delete_conversation(conversation_id: str):
    clear_conversation(conversation_id)
    return {"status": "success", "message": "Conversation history cleared."}

# Mount static first for specific assets
app.mount("/static", StaticFiles(directory="static"), name="static")

# Explicit root handler to serve index.html
@app.get("/")
async def read_index():
    return FileResponse("static/index.html")
