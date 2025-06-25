import os
import sys
import yfinance as yf
import warnings

# Local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Bian.bian_utils import run_NER, extract_entities, to_yf_period, yfinance_search_company
from Bian.resources import model


def extract_tickers(text): # Backend
    entities = run_NER(text)
    company_names = extract_entities(entities, "ORG")
    tickers = yfinance_search_company(company_names)
    return tickers

def extract_intent(text): # Backend
    prompt = (
        "You are an intention extraction parser. Given the user's input, return only one of the following intent categories: "
        "'display_price', 'compare_stocks', 'calculate_indicator', 'predict_indicator', or 'not_stock'.\n"
        "Descriptions:\n"
        "display_price: The user wants to know the current price or see a visual chart of a specific stock.\n"
        "compare_stocks: The user wants to compare two or more stocks.\n"
        "calculate_indicator: The user wants to calculate a technical indicator (e.g., RSI, SMA, MACD) for a stock.\n"
        "predict_indicator: The user wants a prediction or forecast of a technical indicator for a stock.\n"
        "not_stock: The user's input does not relate to stock trading or analysis.\n"
        "User: " + text + "\n"
        "Intent category:"
    )
    response = model.generate_content(prompt)
    # Extract the intent category from the response (assume it's the first word/line)
    intent = response.text.strip().split('\n')[0].strip().lower()
    # Map to expected categories if needed
    valid_intents = ["display_price", "compare_stocks", "calculate_indicator", "predict_indicator", "not_stock"]
    if intent in valid_intents:
        return intent
    # fallback: 
    return "ask_again_for_intent"

def extract_period(text): # Backend
    entities = run_NER(text)
    date_entities = extract_entities(entities, "DATE")

    if len(date_entities) >= 2: 
        return "1y"

    if len(date_entities) == 1:
        period = to_yf_period(text)
        if period:
            return period
        else:
            return "1y"  

    warnings.filterwarnings("ignore")
    return "1y"

def extract_indicator(text): # Backend
    # Prompt engineering: add system instruction for better responses
    indicators = [
    "MACD", "MACD_signal", "MACD_diff", "ADX", "CCI", "Ichimoku_a", "Ichimoku_b",
    "PSAR", "STC", "RSI", "Stoch", "Stoch_signal", "AwesomeOsc", "KAMA", "ROC", "TSI",
    "UO", "ATR", "Bollinger_hband", "Bollinger_lband", "Bollinger_mavg", "Donchian_hband",
    "Donchian_lband", "Keltner_hband", "Keltner_lband", "Donchian_width", "SMA_5", "EMA_5",
    "WMA_5", "DEMA_5", "TEMA_5", "SMA_10", "EMA_10", "WMA_10", "DEMA_10", "TEMA_10", 
    "SMA_20", "EMA_20", "WMA_20", "DEMA_20", "TEMA_20", "SMA_50", "EMA_50", "WMA_50", 
    "DEMA_50", "TEMA_50", "SMA_100", "EMA_100", "WMA_100", "DEMA_100", "TEMA_100", 
    "SMA_200", "EMA_200", "WMA_200", "DEMA_200", "TEMA_200"
    ]
    indicator_comma = ', '.join(indicators)

    prompt = (
        "You are an indicator extraction parser. Given the user's input, return a list of technical indicators they want to calculate or predict.\n"
        "User: " + text + "\n"
        "Indicators (comma-separated): " + indicator_comma + "\n"
    )
    response = model.generate_content(prompt)
    # Extract the indicators from the response (assume they are comma-separated)        
    indicators_text = response.text.strip()
    if indicators_text:
        # Split by commas and strip whitespace
        indicators_list = [indicator.strip() for indicator in indicators_text.split(',')]
        # Filter out any empty strings
        return [indicator for indicator in indicators_list if indicator]
    return "ask_again_for_indicator"
