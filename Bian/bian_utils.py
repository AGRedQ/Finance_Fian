import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys

# Local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Bian.backend_bian import BackendBian
from Fian.frontend_fian import FrontendFian
bian = BackendBian()
fian = FrontendFian()
from Bian.configs import indicator_plot_config


# import libs
import yfinance as yf
import re
import json



def extract_data_yf(tickers, Period="1y",temp_file=False):  # Backend 
    # Note: Remember to make a way to delete these files after use. Since they are only temporary files.
    data = {}
    for ticker in tickers:
        df = yf.download(ticker, period=Period, interval="1d", auto_adjust=True, progress=False)
        filename = f"temp_{ticker}_{Period}.csv"
        if temp_file:
            df.to_csv(filename)
        data[ticker] = df
    return data

def preprocess_query(text): # SpaCy # Experimental
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return ' '.join(tokens)


def run_NER(text): # SpaCy # Experimental
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

def extract_entities(entities, label): # SpaCy # Experimental
    return [ent_text for ent_text, ent_label in entities if ent_label == label]

def to_yf_period(text): # SpaCy # Experimental
    match = re.search(r'(\d+)\s*(year|years|month|months|week|weeks|day|days)', text, re.IGNORECASE)
    if match:
        number = match.group(1)
        unit = match.group(2).lower()

        if 'year' in unit:
            return f"{number}y"
        elif 'month' in unit:
            return f"{number}mo"
        elif 'week' in unit:
            days = int(number) * 7
            return f"{days}d"
        elif 'day' in unit:
            return f"{number}d"
    return None

def yfinance_search_company(company_names): # Backend
    results = {}
    for name in company_names:
        s = yf.Search(name, max_results=1)
        if s.quotes:
            results[name] = s.quotes[0].get("symbol")
        else:
            results[name] = None
    # Return a list of ticker symbols (filtering out any None values)
    return [ticker for ticker in results.values() if ticker]

def load_resources():
    from Bian.resources import model #,nlp, stop_words, lemmatizer
    model = model
    # nlp = nlp
    # stop_words = stop_words
    # lemmatizer = lemmatizer

def check_valid_ticker(ticker): # Backend
    """Check if a ticker is valid by trying to download data for it"""
    try:
        test_extract = yf.download(ticker, period="1d", auto_adjust=True, progress=False)
        if not test_extract.empty:
            return True  # Ticker is valid
        else:
            return False  # Ticker is invalid
    except Exception as e:
        return False  # Ticker is invalid
    
def handle_input_type(input_text): # For Chatbot
    if input_text[0] == "/":
        return "command"
    return "chat"

def run_command(command): # For Chatbot
    if command.startswith("/help"):
        return (
            "Available commands:<br>"
            "/help - Show this help message<br>"
            "/calculate <indicator> <ticker> - Calculate an indicator (e.g., /calculate RSI AAPL)<br>"
            "/compare <ticker1> <ticker2> - Compare two stocks (e.g., /compare AAPL MSFT)<br>"
            "/predict <ticker> - Predict stock price (e.g., /predict TSLA)<br>"
            "/display <ticker> - Display stock information (e.g., /display GOOGL)<br>"
        )
    elif command.startswith("/compare"):
        try:
            # Parse tickers
            parts = command.strip().split()
            if len(parts) < 3:
                return "Usage: /compare <ticker1> <ticker2>"
            ticker1, ticker2 = parts[1].upper(), parts[2].upper()
            # Fetch data
            data = bian.extract_data_yf([ticker1, ticker2], Period="6mo")
            if ticker1 not in data or ticker2 not in data:
                return f"Could not fetch data for {ticker1} or {ticker2}."
            # Use default chart size or get from session state
            import streamlit as st
            chart_width = st.session_state.get("chart_width", 12)
            chart_height = st.session_state.get("chart_height", 6)
            fian.compare_stocks(data[ticker1], data[ticker2], chart_width, chart_height, title=f"{ticker1} vs {ticker2} Close Price")
            return f"Comparing {ticker1} and {ticker2}..."
        except Exception as e:
            return f"Error comparing stocks: {e}"
    elif command.startswith("/calculate"):
        return "Calculate command placeholder."
    elif command.startswith("/predict"):
        return "Predict command placeholder."
    elif command.startswith("/display"):
        return "Display command placeholder."
    else:
        return "Unknown command. Type /help for available commands."

def run_query(query): # For Chatbot
    """Run a query using the generative model with a system prompt for persona and instructions"""
    from Bian.resources import model
    system_prompt = (
        "You are Trian, a lovely stock market assistant. "
        "You provide basic knowledge about things like technical indicators and news. "
        "If the user wants to use commands, kindly remind them to use /help for a list of available commands."
    )
    prompt = f"{system_prompt}\nUser query: {query}"
    response = model.generate_content(prompt)
    return response.text

def process_query(query):
    in_type = handle_input_type(query)
    if in_type == "command":
        response = run_command(query)
    elif in_type == "chat":
        response = run_query(query)
    return response



if __name__ == "__main__":
    # Example usage
    pass





