
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Bian.NLP.extractors import extract_tickers, extract_intent, extract_period


def process_query(query):
    # Extract intent
    intent = extract_intent(query)
    
    # Extract period
    period = extract_period(query)
    
    # Extract tickers
    tickers = extract_tickers(query)
    
    return {
        "intent": intent,
        "period": period,
        "tickers": tickers
    }

def chat():
    print("Welcome to the finance assistant! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        result = process_query(user_input)
        print("Bot:", result)

if __name__ == "__main__":
    chat()