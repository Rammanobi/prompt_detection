import requests
import json

url = "http://127.0.0.1:8081/api/check"
headers = {
    "Content-Type": "application/json",
    "x-api-key": "AIzaSyAFqsp3OM2f4L4lZVBKYYrWjemeb87hznk"
}
data = {
    "prompt": "test"
}

try:
    response = requests.post(url, headers=headers, json=data)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")
