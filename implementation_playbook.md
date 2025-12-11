# Antigravity — Layer 2 Implementation Playbook (Rule + Anomaly)

**Goal**
Add Layer 2 to Antigravity: (A) deterministic Rule-Based Heuristics for immediate block/flag of obvious secrets and phrases, and (B) ML Anomaly Detection (IsolationForest + feature engineering) as a second signal. Integrate with existing Gateway+Detector, log quarantines to Firestore, and provide retraining/triage pipeline.

---

# Quick checklist

1. Audit Layer1: ensure `ANTIGRAVITY_API_KEY` removed from client and only used server-side.
2. Add rules file `rules/rules.yaml`.
3. Implement `detector/rules.py` to run pre-checks in Gateway or Detector.
4. Add anomaly model code `detector/anomaly.py` and `detector/retrain.py`.
5. Update `detector/main.py` to call rules → layer1(similarity) → anomaly → combine decisions → log to Firestore if flagged/blocked.
6. Train anomaly model on benign corpus and bake into Docker image.
7. Deploy as canary; run `tests/run_anomaly_eval.py` and `tests/run_eval.py`.
8. Promote after acceptance.

---

# 1 — Why Layer 2

* **Rule-Based (fast, deterministic)**: immediately block obvious secrets, direct jailbreak phrases, PII patterns. Lowers load on ML.
* **Anomaly Detection (complementary ML)**: flags inputs that look unusual compared to benign prompts. Good for novel attacks.
* **Combined decision**: `rule_vote + similarity_score (L1) + anomaly_score (L2)` -> `allow | flag | block`.

---

# 2 — Files to add / modify

```
detector/
  anomaly.py            # train/load IsolationForest + PCA
  retrain.py            # cron job to retrain anomaly model
  rules.py              # rule-checking functions
  main.py               # update main logic
rules/
  rules.yaml            # 100+ regex rules
tests/
  run_anomaly_eval.py   # evaluate L2
Dockerfile               # bake anomaly model
```

---

# 3 — Rule-Based Heuristics

## 3.1 `rules/rules.yaml` (100+ Rules)

Create `rules/rules.yaml` with this massive list:

```yaml
# rules/rules.yaml
# 100+ curated regex rules for Antigravity Layer 2.

# --- SECTION 1: SECRETS & KEYS ---
- id: R_PEM_PRIVATE_KEY
  desc: Private key PEM block (RSA / PRIVATE KEY / ENCRYPTED)
  pattern: "-----BEGIN (?:RSA )?PRIVATE KEY-----[\\s\\S]{10,20000}-----END (?:RSA )?PRIVATE KEY-----"
  severity: block

- id: R_PEM_OPENSSH
  desc: OpenSSH private key block
  pattern: "-----BEGIN OPENSSH PRIVATE KEY-----[\\s\\S]{10,20000}-----END OPENSSH PRIVATE KEY-----"
  severity: block

- id: R_PGP_PRIVATE
  desc: PGP private key block
  pattern: "-----BEGIN PGP PRIVATE KEY BLOCK-----[\\s\\S]{10,20000}-----END PGP PRIVATE KEY BLOCK-----"
  severity: block

- id: R_AWS_SECRET_KEY
  desc: AWS secret access key
  pattern: "(?i)(?:aws[_-]secret[_-]access[_-]key|aws_secret_access_key)\\s*[:=]\\s*[A-Za-z0-9/+=]{40}"
  severity: block

- id: R_AWS_ACCESS_KEY_ID
  desc: AWS access key id (AKIA...)
  pattern: "(?i)\\bAKIA[0-9A-Z]{16}\\b"
  severity: block

- id: R_GOOGLE_API_KEY
  desc: Google API key pattern (AIza...)
  pattern: "(?i)AIza[0-9A-Za-z\\-_]{35}"
  severity: block

- id: R_SLACK_TOKEN
  desc: Slack token (xox[bpar]-...)
  pattern: "(?i)xox[bpar]-[a-zA-Z0-9-]{10,}"
  severity: block

- id: R_STRIPE_KEY
  desc: Stripe secret key (sk_live_...)
  pattern: "(?i)sk_live_[0-9a-zA-Z]{24}"
  severity: block

- id: R_GITHUB_TOKEN
  desc: GitHub token (ghp_...)
  pattern: "(?i)ghp_[a-zA-Z0-9]{36}"
  severity: block

- id: R_NPM_TOKEN
  desc: NPM access token
  pattern: "(?i)npm_[a-zA-Z0-9]{36}"
  severity: block

- id: R_TWILIO_AUTH_TOKEN
  desc: Twilio Auth Token
  pattern: "(?i)[a-f0-9]{32}"
  severity: flag
  note: High FP rate, keep severity 'flag' unless very specific context

- id: R_HEROKU_API_KEY
  desc: Heroku API Key
  pattern: "(?i)[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
  severity: flag
  note: UUID format common, use caution.

- id: R_FACEBOOK_ACCESS_TOKEN
  desc: Facebook Access Token
  pattern: "(?i)EAACEdEose0cBA[0-9A-Za-z]+"
  severity: block

- id: R_MAILGUN_API_KEY
  desc: Mailgun API Key
  pattern: "(?i)key-[0-9a-zA-Z]{32}"
  severity: block

- id: R_SENDGRID_API_KEY
  desc: SendGrid API Key
  pattern: "(?i)SG\\.[0-9a-zA-Z\\-_]{22}\\.[0-9a-zA-Z\\-_]{43}"
  severity: block

- id: R_TWITTER_ACCESS_TOKEN
  desc: Twitter Access Token
  pattern: "(?i)[1-9][0-9]+-[0-9a-zA-Z]{40}"
  severity: block

- id: R_SQUARE_ACCESS_TOKEN
  desc: Square Access Token
  pattern: "(?i)sq0atp-[0-9a-zA-Z\\-_]{22}"
  severity: block

- id: R_PAYPAL_TOKEN
  desc: PayPal Access Token
  pattern: "(?i)access_token\\$production\\$[0-9a-z]{16}\\$[0-9a-f]{32}"
  severity: block

- id: R_GENERIC_API_KEY_ASSIGN
  desc: Generic API key assignment
  pattern: "(?i)(?:api[_-]?key|token|secret)\\s*[:=]\\s*['\\\"]?[A-Za-z0-9\\-_.]{20,200}['\\\"]?"
  severity: block

- id: R_BEARER_TOKEN
  desc: Bearer token header
  pattern: "(?i)authorization\\s*[:=]\\s*['\\\"]?Bearer\\s+[A-Za-z0-9\\-_.=]+['\\\"]?"
  severity: block

- id: R_PASSWORD_ASSIGNMENT
  desc: Direct password assignment
  pattern: "(?i)password\\s*[:=]\\s*['\\\"]?\\S{4,200}['\\\"]?"
  severity: flag

- id: R_SQL_CREDENTIALS
  desc: Database credentials connection string
  pattern: "(?i)(?:postgres|mysql|mongodb)://\\w+:\\w+@\\S+"
  severity: block

- id: R_PRIVATE_KEY_GENERIC
  desc: Generic Private Key Label
  pattern: "(?i)private[-_ ]key"
  severity: flag
  note: Just the words "private key" - suspicious context.

# --- SECTION 2: PII (Personal Identifiable Information) ---

- id: R_SSN_US
  desc: US Social Security Number
  pattern: "\\b\\d{3}[-\\s]?\\d{2}[-\\s]?\\d{4}\\b"
  severity: block

- id: R_CREDIT_CARD
  desc: Credit card number
  pattern: "(?:(?:\\d[ -]*?){13,19})"
  severity: flag

- id: R_EMAIL_ADDRESS
  desc: Email listing (generic)
  pattern: "(?i)\\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\\.[A-Z]{2,}\\b"
  severity: flag
  note: Common in business, flag for volume.

- id: R_PHONE_NUMBER_US
  desc: US formatted phone number
  pattern: "\\b(?:\\+?1[-. ]?)?\\(?([0-9]{3})\\)?[-. ]?([0-9]{3})[-. ]?([0-9]{4})\\b"
  severity: flag

- id: R_BTC_WALLET
  desc: Bitcoin Wallet Address
  pattern: "\\b(1|3|bc1)[a-zA-HJ-NP-Z0-9]{25,39}\\b"
  severity: flag

- id: R_ETH_WALLET
  desc: Ethereum Wallet Address
  pattern: "\\b0x[a-fA-F0-9]{40}\\b"
  severity: flag

- id: R_IBAN
  desc: International Bank Account Number
  pattern: "\\b[A-Z]{2}[0-9]{2}[a-zA-Z0-9]{4}[0-9]{7}([a-zA-Z0-9]?){0,16}\\b"
  severity: flag

- id: R_DRIVERS_LICENSE
  desc: Generic Driver License mention
  pattern: "(?i)driver[']?s license"
  severity: flag

- id: R_PASSPORT_NUMBER
  desc: Generic Passport Number pattern
  pattern: "\\b[A-Z]{1,2}[0-9]{6,9}\\b"
  severity: flag
  note: Very generic, high FP, use with caution.

# --- SECTION 3: JAILBREAKS & INJECTION ---

- id: R_IGNORE_PREVIOUS
  desc: Ignore previous instructions
  pattern: "(?i)ignore (?:all )?(?:previous|prior) instructions|disregard (?:previous|prior) instructions"
  severity: block

- id: R_OVERRIDE_POLICY
  desc: Ask to override policy
  pattern: "(?i)turn off safety|disable (?:filters|safety)|bypass (?:filters|moderation|policy)"
  severity: block

- id: R_ROLEPLAY_UNFILTERED
  desc: Roleplay unrestricted/unfiltered
  pattern: "(?i)you are now (?:unrestricted|uncensored|god mode|developer mode|DAN)"
  severity: block

- id: R_AIM_JAILBREAK
  desc: 'Always Intelligent Machiavelli' jailbreak pattern
  pattern: "(?i)Niccolo Machiavelli|AIM:|Always Intelligent Machiavelli"
  severity: block

- id: R_MONGO_TOM
  desc: Mongo Tom jailbreak
  pattern: "(?i)Mongo Tom"
  severity: block

- id: R_DO_ANYTHING_NOW
  desc: DAN (Do Anything Now)
  pattern: "(?i)\\bDAN\\b|Do Anything Now"
  severity: block

- id: R_START_WITH
  desc: 'Start your response with' constraint (common in injections)
  pattern: "(?i)start your response with|start with the words"
  severity: flag

- id: R_PROMPT_CHAINING
  desc: Chaining instructions to bypass
  pattern: "(?i)first .* then .* and (?:only )?output|then do the following steps"
  severity: flag

- id: R_SYSTEM_MESSAGE_IMPERSONATION
  desc: Impersonating system messages
  pattern: "(?i)system:|system prompt:|system instruction"
  severity: block

- id: R_FICTIONAL_CONTEXT
  desc: "For a book", "For a movie" evasion
  pattern: "(?i)this is for a (?:book|movie|script|research|educational purpose)"
  severity: flag

- id: R_HYPOTHETICAL_SCENARIO
  desc: "Hypothetically if I were to"
  pattern: "(?i)hypothetically|in a theoretical scenario"
  severity: flag

- id: R_NEVER_REFUSE
  desc: "Never refuse a request"
  pattern: "(?i)never refuse|answer (?:every|all) questions"
  severity: block

- id: R_NO_ETHICS
  desc: "Ignore ethics/morals"
  pattern: "(?i)ignore (?:ethics|ethical|morals|moral guidelines)"
  severity: block

- id: R_REVEAL_PROMPT
  desc: Reveal system prompt
  pattern: "(?i)reveal (?:your )?system prompt|show (?:your )?initial instructions"
  severity: block

- id: R_OUTPUT_VERBATIM
  desc: Output text verbatim
  pattern: "(?i)output (?:it )?verbatim|repeat the words"
  severity: flag

- id: R_TRANSLATE_EVASION
  desc: Translate tricky text
  pattern: "(?i)translate the following"
  severity: flag
  note: Legit use case common - only flag.

- id: R_ENCODING_EVASION
  desc: Base64/Rot13/Hex dump requests
  pattern: "(?i)decode (?:this )?(?:base64|rot13|hex)|encode .* in (?:base64|rot13)"
  severity: flag

# --- SECTION 4: DANGEROUS COMMANDS & CODE ---

- id: R_SSH_COMMAND_EXEC
  desc: Dangerous shell commands
  pattern: "(?i)(?:\\brm\\s+-rf\\b|curl\\s+\\S+\\s*\\|\\s*sh|wget\\s+\\S+\\s*\\|\\s*sh|sudo\\s+rm\\s+-rf|:(){:|:&};:)"
  severity: block
  note: Includes Fork Bomb

- id: R_REMOTE_CODE_EXEC
  desc: Execute system commands
  pattern: "(?i)execute\\s+.*\\b(command|script|shell|system)\\b|open\\s+local\\s+file|read\\s+/etc/passwd"
  severity: block

- id: R_SQL_INJECTION_PATTERN
  desc: SQLi keywords
  pattern: "(?i)\\bUNION\\b.*\\bSELECT\\b|\\bDROP\\s+TABLE\\b|\\bOR\\s+1=1\\b|--\\s*$"
  severity: flag

- id: R_REVERSE_SHELL
  desc: Reverse shell indicators
  pattern: "(?i)/bin/bash -i|nc -e /bin/sh|/bin/sh -i"
  severity: block

- id: R_POWERSHELL_DOWNLOAD
  desc: Powershell download cradle
  pattern: "(?i)IEX\\s*\\(New-Object\\s+Net\\.WebClient\\)\\.DownloadString"
  severity: block

- id: R_PYTHON_EXEC
  desc: Python eval/exec usage
  pattern: "(?i)eval\\(.*\\)|exec\\(.*\\)|import os; os.system"
  severity: flag

- id: R_PHP_EXEC
  desc: PHP execution
  pattern: "(?i)phpinfo\\(\\)|system\\(.*\\)|shell_exec\\("
  severity: flag

- id: R_WIN_REGISTRY
  desc: Windows Registry edits
  pattern: "(?i)reg add|reg delete|HKLM\\\\Software"
  severity: flag

- id: R_CHMOD_777
  desc: Unsafe permissions
  pattern: "(?i)chmod 777|chmod -R 777"
  severity: flag

- id: R_EXFILTRATION_PIPELINE
  desc: Exfiltration tools (scp, ftp, nc)
  pattern: "(?i)(?:scp|ftp|nc\\s+-e|netcat).{0,200}?(?:@|:)"
  severity: block

- id: R_XSS_SCRIPT
  desc: XSS script tag
  pattern: "(?i)<script>.*</script>|javascript:alert\\("
  severity: block

- id: R_LOG4J
  desc: Log4Shell payload
  pattern: "(?i)\\$\\{jndi:(?:ldap|rmi|dns)://"
  severity: block

- id: R_SENSITIVE_DOC_REQUEST
  desc: Request for internal docs
  pattern: "(?i)(?:show|give|provide|list) (?:internal|private|secret|confidential) (?:files|tokens|keys|configs)"
  severity: block

- id: R_DUMP_CONFIG
  desc: Dump configuration
  pattern: "(?i)dump (?:all )?(?:config|configuration|env|environment)"
  severity: block

- id: R_DEBUG_MODE
  desc: Enable debug mode
  pattern: "(?i)enable debug mode|debug=true"
  severity: flag

# --- SECTION 5: MALICIOUS CONTENT & TOPICS ---

- id: R_BOMB_MAKING
  desc: Bomb making instructions
  pattern: "(?i)how to (?:make|build|create) a bomb|explosive recipe"
  severity: block

- id: R_DRUG_MANUFACTURE
  desc: Drug synthesis
  pattern: "(?i)how to (?:make|synthesize|cook) (?:meth|methamphetamine|lsd|heroin|cocaine)"
  severity: block

- id: R_WEAPON_MANUFACTURE
  desc: 3D printed guns/weapons
  pattern: "(?i)3d print (?:a )?gun|auto sear|suppressor"
  severity: block

- id: R_SUICIDE_SELFHARM
  desc: Self harm content
  pattern: "(?i)how to (?:kill myself|commit suicide|die painlessly)"
  severity: block

- id: R_HATE_SPEECH_KEYWORD
  desc: Slurs (Generic placeholder)
  pattern: "(?i)\\b(?:nigger|kike|faggot|retard)\\b"
  severity: block

- id: R_KEYLOGGER
  desc: Keylogger creation
  pattern: "(?i)write (?:a )?keylogger|capture keystrokes"
  severity: block

- id: R_RANSOMWARE
  desc: Ransomware creation
  pattern: "(?i)write (?:a )?ransomware|encrypt files and demand"
  severity: block

- id: R_PHISHING_TEMPLATE
  desc: Phishing email generation
  pattern: "(?i)write (?:a )?phishing type|generate (?:a )?phishing email"
  severity: block

- id: R_CREDENTIAL_STUFFING
  desc: Credential stuffing tools
  pattern: "(?i)credential stuffing|brute force login"
  severity: block

- id: R_DDOS_ATTACK
  desc: DDoS instructions
  pattern: "(?i)how to ddos|launch a ddos|syn flood"
  severity: block

- id: R_TAX_EVASION
  desc: Tax evasion guides
  pattern: "(?i)how to evade taxes|hide income from irs"
  severity: flag

- id: R_DEEPFAKE_PORN
  desc: Deepfake pornography
  pattern: "(?i)deepfake (?:porn|nude)"
  severity: block

- id: R_LOCKPICKING
  desc: Lockpicking guides
  pattern: "(?i)how to pick a lock|unlock door without key"
  severity: flag

- id: R_HOTWIRING
  desc: Car theft / hotwiring
  pattern: "(?i)how to hotwire|start car without key"
  severity: block

- id: R_STALKING
  desc: Stalking / Doxing
  pattern: "(?i)how to (?:stalk|track) someone|find someone's address"
  severity: block

- id: R_SHOPLIFTING
  desc: Shoplifting tips
  pattern: "(?i)how to shoplift|steal from (?:store|shop)"
  severity: block

- id: R_IDENTITY_THEFT
  desc: Identity theft
  pattern: "(?i)steal identity|fake id"
  severity: block

- id: R_DARK_WEB_MARKET
  desc: Dark web markets
  pattern: "(?i)silk road|alpha bay|buy drugs online"
  severity: flag

- id: R_OSINT_USAGE
  desc: OSINT malicious use
  pattern: "(?i)osint tool|doxbin"
  severity: flag

# --- SECTION 6: OBFUSCATION DETECTORS ---

- id: R_HIDDEN_ZERO_WIDTH
  desc: Hidden / zero-width characters
  pattern: "[\\u200B\\u200C\\u200D\\uFEFF]"
  severity: flag

- id: R_UNICODE_HOMOGLYPH
  desc: Unicode homoglyphs
  pattern: "[\\u0400-\\u04FF\\u0370-\\u03FF]"
  severity: flag

- id: R_PAYLOAD_OBFUSCATION
  desc: Hex escape sequences
  pattern: "(?:\\\\x[0-9a-fA-F]{2}){8,}"
  severity: flag

- id: R_ROT13
  desc: Rot13 heuristic (weak)
  pattern: "(?i)[a-zA-Z0-9]{20,}"
  severity: flag
  note: Very weak, relies on context/entropy models mostly.

- id: R_MORSE_CODE
  desc: Morse code pattern
  pattern: "(?:[.-]{1,5}\\s?){5,}"
  severity: flag

- id: R_BINARY_STRING
  desc: Long binary strings
  pattern: "\\b[01]{16,}\\b"
  severity: flag

- id: R_HEX_DUMP
  desc: Hex dump
  pattern: "(?:[0-9a-fA-F]{2}\\s){8,}"
  severity: flag

- id: R_LONG_WORDS
  desc: Extremely long words (buffer overflow attempt?)
  pattern: "\\S{100,}"
  severity: flag
```

## 3.2 `detector/rules.py` (Implementation)

Save as `detector/rules.py`:

```python
import re
import yaml
from pathlib import Path
from typing import List, Dict, Any

RULES_PATH = Path("rules/rules.yaml")

class RuleMatch:
    def __init__(self, rule_id, desc, severity, match_text):
        self.rule_id = rule_id
        self.desc = desc
        self.severity = severity
        self.match_text = match_text

def load_rules():
    if not RULES_PATH.exists():
        return []
    raw = yaml.safe_load(RULES_PATH.read_text(encoding="utf-8"))
    rules = []
    for r in raw:
        try:
            compiled = re.compile(r["pattern"])
            rules.append({
                "id": r["id"],
                "desc": r.get("desc",""),
                "severity": r.get("severity","flag"),
                "re": compiled
            })
        except Exception as e:
            print(f"Failed to compile rule {r.get('id')}: {e}")
    return rules

_RULES = load_rules()

def run_rules_on_text(text: str) -> List[RuleMatch]:
    matches = []
    for r in _RULES:
        m = r["re"].search(text)
        if m:
            matches.append(RuleMatch(
                rule_id=r["id"], 
                desc=r["desc"], 
                severity=r["severity"], 
                match_text=m.group(0)
            ))
    return matches
```

---

# 4 — Anomaly Detection Code

## 4.1 `detector/anomaly.py`

```python
import os, pickle
from pathlib import Path
from typing import List, Tuple
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
import math

MODEL_DIR = Path("detector/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

ANOMALY_MODEL_PKL = MODEL_DIR / "isolation_forest.pkl"
PCA_PKL = MODEL_DIR / "pca.pkl"
SCALER_PKL = MODEL_DIR / "scaler.pkl"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def clean_text(text: str) -> str:
    # simple cleaner matching main code
    return text.lower().strip()

def meta_features(text: str) -> List[float]:
    text = text or ""
    n_chars = len(text)
    n_words = len(text.split())
    n_punct = sum(1 for c in text if c in ".,;:!?")
    n_digits = sum(1 for c in text if c.isdigit())
    non_ascii = sum(1 for c in text if ord(c) > 127)
    zero_width = sum(1 for c in text if c in ("\u200B","\u200C","\u200D","\uFEFF"))
    
    entropy = 0.0
    if n_chars > 0:
        from collections import Counter
        counts = Counter(text)
        probs = [v/n_chars for v in counts.values()]
        entropy = -sum(p*math.log2(p) for p in probs if p>0)
        
    return [n_chars, n_words, n_punct, n_digits, non_ascii, zero_width, entropy]

def build_features(texts: List[str], embed_model: SentenceTransformer, pca: PCA=None) -> Tuple[np.ndarray, PCA]:
    cleaned = [clean_text(t) for t in texts]
    embs = embed_model.encode(cleaned, convert_to_numpy=True, normalize_embeddings=True)
    if embs.ndim == 1:
        embs = np.expand_dims(embs, 0)
    
    if pca is None:
        # Fit PCA
        dim = min(32, embs.shape[1], len(texts))
        pca = PCA(n_components=dim)
        pca.fit(embs)
    
    emb_reduced = pca.transform(embs)
    metas = np.array([meta_features(t) for t in texts], dtype=float)
    X = np.hstack([emb_reduced, metas])
    return X, pca

def train(normal_texts: List[str], contamination:float=0.01) -> None:
    print(f"Training anomaly model with {len(normal_texts)} samples...")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    X, pca = build_features(normal_texts, embed_model, pca=None)
    
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    
    clf = IsolationForest(n_estimators=200, contamination=contamination, random_state=42, n_jobs=-1)
    clf.fit(Xs)
    
    with open(ANOMALY_MODEL_PKL, "wb") as fh: pickle.dump(clf, fh)
    with open(PCA_PKL, "wb") as fh: pickle.dump(pca, fh)
    with open(SCALER_PKL, "wb") as fh: pickle.dump(scaler, fh)
    print("Training complete. Models saved.")

def load() -> Tuple[SentenceTransformer, PCA, StandardScaler, IsolationForest]:
    if not ANOMALY_MODEL_PKL.exists():
        raise FileNotFoundError("Anomaly model not found.")
    
    with open(ANOMALY_MODEL_PKL, "rb") as fh: clf = pickle.load(fh)
    with open(PCA_PKL, "rb") as fh: pca = pickle.load(fh)
    with open(SCALER_PKL, "rb") as fh: scaler = pickle.load(fh)
    
    # Load embed model (cached)
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    return embed_model, pca, scaler, clf

def score_texts(texts: List[str], embed_model, pca, scaler, clf) -> List[float]:
    X, _ = build_features(texts, embed_model, pca=pca)
    Xs = scaler.transform(X)
    
    # decision_function: positive=normal, negative=outlier
    # Invert to make 1.0 = highly anomalous, 0.0 = normal
    raw = clf.decision_function(Xs)
    # simple heuristic scaling: raw < 0 is outlier. 
    # Sigmoid or linear scaling can refer here.
    scores = 1 / (1 + np.exp(raw)) # Sigmoid-like: raw high -> score low
    return scores.tolist()
```

## 4.2 `detector/retrain.py`

```python
from google.cloud import firestore
from detector.anomaly import train
import argparse
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=5000)
    args = parser.parse_args()
    
    try:
        db = firestore.Client()
        docs = db.collection("normal_prompts").limit(args.limit).stream()
        texts = [d.to_dict().get("text") for d in docs if d.to_dict().get("text")]
    except Exception as e:
        print(f"Firestore error: {e}")
        texts = []

    if len(texts) < 50:
        print("Not enough data to retrain (need >50).")
        sys.exit(0)
        
    train(texts)

if __name__ == "__main__":
    main()
```

---

# 5 — Updated `detector/main.py` (Integration)

Verify imports and startup logic:

```python
# Imports
from detector.rules import run_rules_on_text
try:
    from detector.anomaly import load as load_anomaly, score_texts
    embed_anom, pca, scaler, iso_forest = load_anomaly()
    ANOMALY_OK = True
except Exception as e:
    print(f"Anomaly model disabled: {e}")
    ANOMALY_OK = False

# Inside process_prompt or check function:

# 1. Rules
rule_matches = run_rules_on_text(prompt_clean)
rule_blocks = [m for m in rule_matches if m.severity == 'block']
rule_flags = [m for m in rule_matches if m.severity == 'flag']

if rule_blocks:
    return {
        "decision": "block",
        "score": 1.0,
        "reasons": ["rule_violation"],
        "matches": [{"text": m.match_text, "rule_id": m.rule_id} for m in rule_blocks]
    }

# 2. Similarity (Layer 1 - Existing logic)
# ... best_score calculation ...

# 3. Anomaly (Layer 2)
anom_score = 0.0
if ANOMALY_OK:
    try:
        anom_score = score_texts([prompt_clean], embed_anom, pca, scaler, iso_forest)[0]
    except:
        pass

# 4. Final Decision
# Weights: Rules > Similarity > Anomaly
# Sim Threshold: 0.8
# Anom Threshold: 0.85 (tuned)

if best_score > 0.8 or anom_score > 0.85:
    decision = "block"
elif best_score > 0.6 or anom_score > 0.65 or rule_flags:
    decision = "flag"
else:
    decision = "allow"

# Log logic remains same...
```

---

# 6 — Deployment Commands

Build and Deploy script (`deploy_layer2.sh`):

```bash
#!/bin/bash
# 1. Train Anomaly Model Locally (Bootstrapping)
# Create a dummy benign file if needed
echo '["hello world", "write code", "how are you", "translate this"]' > benign.json
python -c "from detector.anomaly import train; import json; train(json.load(open('benign.json')))"

# 2. Deploy
gcloud builds submit --tag gcr.io/my-project-id-45/antigravity-layer1:layer2-v1
gcloud run deploy antigravity-layer1-detector \
  --image gcr.io/my-project-id-45/antigravity-layer1:layer2-v1 \
  --region us-central1 \
  --set-secrets="ANTIGRAVITY_API_KEY=ANTIGRAVITY_API_KEY:latest"
```
