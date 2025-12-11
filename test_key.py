import google.generativeai as genai
import os

key = "AIzaSyCoJLz514xnzuN6avBCr2iDKPpN0dZ8gNA"
genai.configure(api_key=key)

try:
    print("Listing available models...")
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"Found model: {m.name}")
    
    print("\nAttempting generation with 'gemini-pro' as fallback...")
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content("Hello")
    print(f"SUCCESS with gemini-pro: {response.text}")
except Exception as e:
    print(f"FAILED: {e}")
