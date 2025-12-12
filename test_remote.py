import requests
import json
import os

# Gateway URL from deployment
url = "https://antigravity-layer1-gateway-267340528308.us-central1.run.app/api/check"
headers = {
    "Content-Type": "application/json",
    # The gateway usually doesn't require x-api-key from public if allow-unauthenticated is set
    # but the frontend sends one if configured? 
    # Frontend logic: it might send one if logged in, but let's try without first as per 'allow-unauthenticated'
}
data = {
    "prompt": "Hello is this working?",
    "user_id": "test_script",
    "conversation_id": "test_conn"
}

print(f"Testing URL: {url}")
try:
    response = requests.post(url, headers=headers, json=data)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")
