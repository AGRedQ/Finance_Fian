
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Bian.NLP.extractors import extract_tickers, extract_intent, extract_period


raw_query = "What is the stock price of Apple Inc. for the last 5 years?"

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

print(process_query(raw_query))