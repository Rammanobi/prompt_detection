
# tests/train_bootstrap.py
from detector.anomaly import train
import json

# A small set of benign prompts to bootstrap the model
BENIGN_DATA = [
    "Hello, how are you?",
    "What is the weather like today?",
    "Write a poem about sunflowers.",
    "Translate 'hello' to Spanish.",
    "Summarize this text for me.",
    "Explain quantum physics in simple terms.",
    "Write a python function to add two numbers.",
    "Who won the world cup in 2018?",
    "I need help with my homework.",
    "Draft an email to my boss.",
    "How do I bake a cake?",
    "What is the capital of France?",
    "Show me a recipe for pasta.",
    "Debug this code snippet.",
    "Create a marketing plan for a coffee shop.",
    "What are the best movies of 2023?",
    "Convert this JSON to XML.",
    "Tell me a joke.",
    "Ignore the previous error and try again.", 
    "System status normal.",
    "Log in to the dashboard.", 
    "My name is Alice.",
    "123 Main St, New York, NY",
    "Customer service contact number.",
    "Refund policy for the store.",
    "Terms and conditions apply.",
    "Copyright 2024.",
    "All rights reserved.",
    "Loading data...",
    "Please wait.",
    "Confirm your email address.",
    "Reset password link.",
    "Verify your identity.",
    "Two-factor authentication code.",
    "Payment successful.",
    "Order confirmation #12345.",
    "Track your shipment.",
    "Unsubscribe from this list.",
    "Privacy policy update.",
    "New features available.",
    "Download the app now.",
    "Connect with us on social media.",
    "Frequently asked questions.",
    "Support ticket #9876.",
    "Maintenance scheduled for tonight.",
    "Server upgrade in progress.",
    "Database connection established.",
    "API rate limit exceeded.",
    "404 Not Found.",
    "500 Internal Server Error.",
]

if __name__ == "__main__":
    print(f"Bootstrapping anomaly model with {len(BENIGN_DATA)} examples...")
    train(BENIGN_DATA, contamination=0.01)
    print("Done.")
