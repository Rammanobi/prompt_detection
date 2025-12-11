# gateway/main.py
import os
import requests
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel

DETECTOR_URL = os.getenv("DETECTOR_URL")  # e.g. https://antigravity-detector-xyz.a.run.app/api/check
ANTIG_KEY = os.getenv("ANTIGRAVITY_API_KEY")
if ANTIG_KEY:
    ANTIG_KEY = ANTIG_KEY.strip()

# Allow running without extensive checks locally if desired, but warn
if not DETECTOR_URL:
    print("WARNING: DETECTOR_URL not set.")
if not ANTIG_KEY:
    print("WARNING: ANTIGRAVITY_API_KEY not set.")

app = FastAPI(title="Antigravity Gateway")

class ChatReq(BaseModel):
    user_id: str | None = ""
    prompt: str

@app.post("/api/chat")
def chat(req: ChatReq):
    # Here you would verify the user's JWT/session. For dev, optionally skip.
    # Example: token = request.headers.get("Authorization"); verify JWT...

    if not DETECTOR_URL or not ANTIG_KEY:
         raise HTTPException(status_code=500, detail="Gateway misconfigured")

    # Forward to detector using server-side secret
    try:
        resp = requests.post(DETECTOR_URL, json={"user_id": req.user_id, "prompt": req.prompt},
                             headers={"x-api-key": ANTIG_KEY}, timeout=30) # Increase timeout for LLM Review
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Detector unreachable: {str(e)}")

    if resp.status_code != 200:
        # Pass through the error from detector or generic 502
        # If Detector returns 400 (which it might for blocks if configured strictly), pass it up
        try:
             detail = resp.json()
             raise HTTPException(status_code=resp.status_code, detail=detail)
        except:
             raise HTTPException(status_code=resp.status_code, detail="Detector rejected request")
    
    data = resp.json()
    decision = data.get("decision", "allow")
    
    # Gateway Enforcer
    if decision == "block":
         block_msg = data.get("review_note") or "Request rejected by security policy."
         raise HTTPException(status_code=400, detail=block_msg)
    
    # If Flag, we might allow but warn? 
    # Current Detector Logic: If Triage says 'needs_human' for a flag, it upgrades to 'block'
    # So if we see 'flag' here, it means 'allow with warning' or 'monitor'.
    
    return data
