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
