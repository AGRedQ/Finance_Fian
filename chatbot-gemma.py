import os
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load Hugging Face token from .env
load_dotenv()
token = os.getenv("HF_TOKEN")

# Model setup
model_id = "google/gemma-2b-it"

tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=token)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
    use_auth_token=token
)

print("\n✅ Gemma 2B model loaded successfully.")
print("💬 Start chatting with Gemma! (type 'exit' to quit)\n")

while True:
    user_input = input("You: ").strip()
    if user_input.lower() in ["exit", "quit"]:
        print("👋 Goodbye!")
        break

    # Construct prompt
    prompt = f"You are a helpful finance assistant.\nUser: {user_input}\nAssistant:"

    # Tokenize input
    encoded = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate model response
    with torch.no_grad():
        output = model.generate(
            **encoded,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode and print response
    response = tokenizer.decode(output[0], skip_special_tokens=True)

    # Extract model response after the last "Assistant:"
    if "Assistant:" in response:
        reply = response.split("Assistant:")[-1].strip()
    else:
        reply = response.strip()

    print(f"Gemma: {reply}\n")
