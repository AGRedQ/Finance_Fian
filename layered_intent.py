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

def query_gemma(prompt, max_length=512, temperature=0.7, top_p=0.9):
    encoded = tokenizer(
        prompt + tokenizer.eos_token,
        return_tensors="pt",
        padding=True,
    )
    input_ids = encoded["input_ids"].to(model.device)
    attention_mask = encoded["attention_mask"].to(model.device)

    output_ids = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    response = response[len(prompt):].strip()

    # Extract JSON from response
    try:
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        json_str = response[json_start:json_end]
        extracted = json.loads(json_str)
        return extracted
    except Exception as e:
        print("⚠️ Could not parse JSON response.")
        print("Raw response:", response)
        return None

# Layer 1: Extract intent, tickers, and period together
def layer1_extract_intent_ticker_period(user_input):
    prompt = f"""
You are a helpful finance assistant. Extract the high-level intent, stock ticker(s), and time period from the user's sentence.
Possible intents: show_info, calculate, predict, compare, null.
Tickers must be valid yfinance symbols (e.g. AAPL, TSLA).
Period examples: 1d, 5d, 1mo, 6mo, 1y, 5y, max.
If any information is missing or unclear, use null.

Return your answer in this JSON format only:
{{
  "intent": "...",
  "tickers": [...],
  "period": "..."
}}
WARNING: You are a parser, do not reply to the user, your only job is to extract the information.
Now process this sentence: "{user_input}"
"""
    return query_gemma(prompt)

# Layer 2: Extract indicator only
def layer2_extract_indicator(user_input):
    indicators = [
        "RelativeStrengthIndex",        # RSI
        "SimpleMovingAverage",          # SMA
        "ExponentialMovingAverage",     # EMA
        "MovingAverageConvergenceDivergence",  # MACD
        "BollingerBands",
        "AverageTrueRange",             # ATR
        "OnBalanceVolume",              # OBV
        "StochasticOscillator",
        "StochasticRSI",
        "CommodityChannelIndex",        # CCI
        "ParabolicSAR",
        "MomentumIndicator",
        "ChaikinMoneyFlow",
        "MoneyFlowIndex",
        "WilliamsR",
        "UltimateOscillator",
        "ADX",                         # Average Directional Index
        "Aroon",
        "ForceIndex",
        "KeltnerChannel",
        "PivotPoints",
        "VolumeWeightedAveragePrice",  # VWAP
        "PriceRateOfChange",
        "TRIX",
        "DirectionalMovementIndex",    # DMI
        "LinearRegressionSlope",
        "WilliamsAccumulationDistribution",
        "VolumePriceTrend",
        "ZigZagIndicator",
        "MassIndex",
    ]
    indicator_list_str = ", ".join(indicators)

    prompt = f"""
You are a helpful finance assistant. Extract the technical indicator mentioned in the user's sentence.
Choose ONLY from this list, even if the indicator is shortened (example: RSI is RelativeStrengthIndex, etc...):
{indicator_list_str}
If none match, return null.
Return your answer in this JSON format only:
{{\

  "indicator": "..."
}}
WARNING: You are a parser, do not reply to the user, your only job is to extract the information.
Now process this sentence: "{user_input}"
"""
    return query_gemma(prompt)

def parse_user_input():
    print("✅ Finance Chatbot with 2-Layer Intent Parsing")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break

        # Layer 1
        layer1_res = layer1_extract_intent_ticker_period(user_input)
        if not layer1_res:
            print("⚠️ Layer 1: Could not parse response.")
            continue
        intent = layer1_res.get("intent", "null")
        tickers = layer1_res.get("tickers", [])
        period = layer1_res.get("period", "null")
        print("🔍 Layer 1 Result:", layer1_res)

        # Layer 2
        layer2_res = layer2_extract_indicator(user_input)
        indicator = layer2_res.get("indicator") if layer2_res else "null"
        print("🔍 Layer 2 Result:", layer2_res)

        final_result = {
            "intent": intent,
            "tickers": tickers,
            "period": period,
            "indicator": indicator
        }
        return final_result


if __name__ == "__main__":
      
    queries = parse_user_input()
    print("\n🎯 Final Parsed Output:")
    print(json.dumps(queries, indent=2))

