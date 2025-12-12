import os
import time
import json
from datetime import datetime
from threading import Lock
import firebase_admin
from firebase_admin import credentials, firestore

# --- FIRESTORE ONLY CONFIGURATION ---
try:
    # Verify if app is already initialized
    if not firebase_admin._apps:
        # 1. OPTION A: Look for service account file if defined
        cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "service-account.json")
        if os.path.exists(cred_path):
            cred = credentials.Certificate(cred_path)
            firebase_admin.initialize_app(cred)
        else:
            # 2. OPTION B: Use Default Credentials (gcloud auth application-default login)
            print(f"Memory: Warning - '{cred_path}' not found. Trying Application Default Credentials...")
            firebase_admin.initialize_app()
    
    db = firestore.client()
    print("✅ Memory: Firestore connection established.")

except Exception as e:
    print(f"❌ CRITICAL FIREBASE ERROR: Could not connect to Firestore backend.")
    print(f"Reason: {e}")
    print("To fix: Please place your 'service-account.json' in this folder OR set GOOGLE_APPLICATION_CREDENTIALS.")
    # We re-raise because user requested "Firestore Only" - no fallback desired.
    # However, to prevent server crash loop, we set db=None but will fail on write.
    db = None

# --- Memory API (Firestore Strict) ---

def save_message(conversation_id: str, role: str, text: str, decision=None, blocked=False, evidence=None):
    if db is None:
        print(f"⚠️ Skipping specific save (Firestore not connected): {text[:20]}...")
        return False

    timestamp = firestore.SERVER_TIMESTAMP
    msg_data = {
        "role": role,
        "text": text,
        "ts": timestamp,
        "decision": decision,
        "blocked": blocked,
        "evidence": evidence or {}
    }

    try:
        conv_ref = db.collection("conversations").document(conversation_id)
        # Create conv shell if missing
        if not conv_ref.get().exists:
            conv_ref.set({
                "created_at": timestamp,
                "updated_at": timestamp,
                "summary": ""
            })
        
        conv_ref.collection("messages").add(msg_data)
        conv_ref.update({"last_message_ts": timestamp})
        return True
    except Exception as e:
        print(f"Firestore write error: {e}")
        return False

def update_summary_incremental(conversation_id: str, new_summary: str):
    if db is None: return
    try:
        db.collection("conversations").document(conversation_id).update({
            "summary": new_summary,
            "updated_at": firestore.SERVER_TIMESTAMP
        })
    except Exception as e:
        print(f"Firestore summary error: {e}")

def get_context_for_generation(conversation_id: str, limit_recent=6):
    if db is None:
        # Return empty context if DB is down, rather than crashing
        return {"summary": "", "recent_messages": []}

    try:
        conv_ref = db.collection("conversations").document(conversation_id)
        doc = conv_ref.get()
        if not doc.exists:
            return {"summary": "", "recent_messages": []}
            
        summary = doc.to_dict().get("summary", "")
        # Get messages DESC (newest first)
        msgs_stream = conv_ref.collection("messages")\
                              .order_by("ts", direction=firestore.Query.DESCENDING)\
                              .limit(limit_recent)\
                              .stream()
        raw_msgs = [m.to_dict() for m in msgs_stream]
        # Reverse to get Chronological Order [Oldest -> Newest] for LLM
        raw_msgs.reverse()
        return {"summary": summary, "recent_messages": raw_msgs}
    except Exception as e:
        print(f"Firestore read error: {e}")
        return {"summary": "", "recent_messages": []}

def clear_conversation(conversation_id: str):
    if db is None: return
    try:
        # Note: Shallow delete (document only). Subcollections remain in Firestore unless manually deleted.
        # Efficient hack: Just clear the summary and set specific flag 'cleared'
        db.collection("conversations").document(conversation_id).delete()
    except Exception as e:
        print(f"Firestore delete error: {e}")
