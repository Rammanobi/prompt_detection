import google.generativeai as genai
import os

key = "AIzaSyCa_1BLO7-a7r4mdMLw4KT61TzcHGTbWsQ"
print(f"Testing Key: {key[:10]}... with gemini-2.0-flash-exp")

try:
    genai.configure(api_key=key)
    # Using the latest reliable model
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
    print("Generating...")
    response = model.generate_content("Reply with 'SUCCESS'")
    print("FINAL RESULT:", response.text)
except Exception as e:
    print("\n--- FAILURE REASON ---")
    print(e)
