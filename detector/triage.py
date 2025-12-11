
import os
import hashlib
import time
from typing import Dict, Any, List
from detector.layer4_llm import call_llm_review

# Firestore setup
try:
    from google.cloud import firestore
    db = firestore.Client()
    USE_FIRESTORE = True
except Exception:
    print("Layer 4 Warning: Firestore not available. Logging disabled.")
    db = None
    USE_FIRESTORE = False

def mask_prompt(prompt: str, visible_chars=200) -> str:
    """
    Masks the prompt to avoid logging excessive PII/Secrets.
    Keeps the first N chars and masks the rest, plus naive PII masking.
    """
    if len(prompt) > visible_chars:
        snippet = prompt[:visible_chars] + "...[TRUNCATED]"
    else:
        snippet = prompt
    
    # Simple redaction example (expand as needed)
    snippet = snippet.replace("AIza", "AIza[MASKED]")
    return snippet

def run_triage(req_prompt: str, req_user_id: str, 
               rule_matches: List[Any], 
               sim_match: Dict[str, Any], 
               rf_result: Dict[str, Any],
               initial_decision: str) -> Dict[str, Any]:
    """
    Orchestrates the Layer 4 Triage Process.
    Returns the FINAL decision packet (e.g., status, review_data).
    """
    
    # 1. Prepare Evidence Payload
    prompt_hash = hashlib.sha256(req_prompt.encode("utf-8")).hexdigest()
    user_hash = hashlib.sha256((req_user_id or "").encode("utf-8")).hexdigest() if req_user_id else None
    
    evidence = {
        "prompt_hash": prompt_hash,
        "prompt_snippet_masked": mask_prompt(req_prompt),
        "rule_matches": [{"id": r.rule_id, "text": r.match_text} for r in rule_matches],
        "sim_match_snippet": sim_match.get("text", "")[:100] if sim_match else "",
        "sim_score": sim_match.get("score", 0.0) if sim_match else 0.0,
        "rf_confidence": rf_result.get("confidence", 0.0) if rf_result else 0.0,
        "rf_status": rf_result.get("status", "UNKNOWN") if rf_result else "UNKNOWN"
    }
    
    # 2. Call LLM Review
    # We only call LLM if flagged/blocked/uncertain to save costs.
    # But user wants "Automated Review for Flagged Prompts".
    llm_result = None
    final_status = "pending_review"
    
    if initial_decision in ["flag", "block"]:
        print(f"Layer 4: Triggering Review for {initial_decision} decision...")
        llm_result = call_llm_review(evidence)
    
    # 3. Interpret Result
    if llm_result:
        label = llm_result.get("label")
        conf = float(llm_result.get("confidence", 0.0))
        
        if label == "confirm_malicious" and conf > 0.8:
            final_status = "auto_blocked"
        elif label == "false_positive" and conf > 0.8:
            final_status = "auto_allowed"
        else:
            final_status = "needs_human"
    else:
        # LLM Failed or not called
        final_status = "needs_human" if initial_decision in ["flag", "block"] else "allowed"

    # 4. Log to Firestore (The "Quarantine" Record)
    if USE_FIRESTORE and initial_decision in ["flag", "block"]:
        try:
            doc_data = {
                "prompt_hash": prompt_hash,
                "user_hash": user_hash,
                "evidence": evidence,
                "initial_decision": initial_decision,
                "created_at": firestore.SERVER_TIMESTAMP,
                "status": final_status,
                "llm_review": llm_result
            }
            # Use hash as ID to dedup same attacks
            doc_ref = db.collection("quarantine").document(prompt_hash)
            doc_ref.set(doc_data, merge=True)
            print(f"Layer 4: Logged to quarantine/{prompt_hash}")
        except Exception as e:
            print(f"Layer 4 logging failed: {e}")

    # 5. Return Final Actionable Result
    return {
        "status": final_status,
        "llm_review": llm_result
    }
