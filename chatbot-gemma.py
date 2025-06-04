import os
import json
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load variables from .env file
load_dotenv()
token = os.getenv("HF_TOKEN")

model_id = "google/gemma-2b-it"

tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=token)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
    use_auth_token=token
)

print("✅ Gemma 2B model and tokenizer loaded successfully.")
print("💬 Chat with Gemma 2B! Type 'exit' or 'quit' to stop.")

while True:
    user_input = input("\nYou: ")
    if user_input.lower() in ["exit", "quit"]:
        print("👋 Goodbye!")
        break

    # Prompt formatting
    prompt = f"""
You are a helpful finance assistant. Extract the stock-related intent, ticker symbol, and time period from the user's sentence. 
If any of them are missing or unclear, use null.

Current valid intents (use EXACT spelling):
- "compares_stock"
- "shows_info"
- "calculate_something"
- "predict_something"

Ticker and period MUST be compatible with yfinance (AAPL, TSLA, 1y, 6m, etc...)

Return your answer in this JSON format only:
{{
  "intent": "...",
  "ticker": [...],
  "period": "..."
}}

WARNING: You are a parser, do not reply to the user. Only extract structured data.
Now process this sentence: "{user_input}"
"""

    # Tokenize prompt
    encoded = tokenizer(
        prompt + tokenizer.eos_token,
        return_tensors="pt",
        padding=True,
    )
    input_ids = encoded["input_ids"].to(model.device)
    attention_mask = encoded["attention_mask"].to(model.device)

    # Generate response
    output_ids = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )

    # Decode and extract JSON
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    response = response[len(prompt):].strip()

    try:
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        json_str = response[json_start:json_end]
        extracted = json.loads(json_str)
        print("📊 Extracted Info:", extracted)
    except Exception as e:
        print("⚠️ Could not parse structured response.")
        print("Raw response:", response)
