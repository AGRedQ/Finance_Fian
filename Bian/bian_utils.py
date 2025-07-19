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

def format_currency(value, currency_code, ticker_symbol=None): # Backend
    """Format currency based on the currency code"""
    if value is None:
        return "N/A"
    
    # Currency mapping
    currency_map = {
        'USD': ('$', 2),
        'VND': ('₫', 0),  # Vietnamese Dong - no decimals
        'EUR': ('€', 2),
        'GBP': ('£', 2),
        'JPY': ('¥', 0),  # Japanese Yen - no decimals
        'KRW': ('₩', 0),  # Korean Won - no decimals
        'CNY': ('¥', 2),  # Chinese Yuan
        'INR': ('₹', 2),  # Indian Rupee
        'CAD': ('C$', 2),
        'AUD': ('A$', 2),
        'HKD': ('HK$', 2),
        'SGD': ('S$', 2),
        'THB': ('฿', 2),
    }
    
    # Get currency symbol and decimal places
    symbol, decimals = currency_map.get(currency_code, ('', 2))
    
    # Format based on currency
    if currency_code == 'VND':
        # Vietnamese Dong - use thousands separators
        return f"{symbol}{value:,.0f}"
    elif currency_code in ['JPY', 'KRW']:
        # No decimals for these currencies
        return f"{symbol}{value:,.0f}"
    else:
        # Standard formatting with decimals
        return f"{symbol}{value:,.{decimals}f}"

def get_currency_info(info, ticker_symbol): # Backend
    """Get currency information from ticker info"""
    # Try to get currency from info
    currency = info.get('currency', 'USD')
    
    # Fallback: detect from ticker symbol
    if not currency or currency == 'USD':
        if '.VN' in ticker_symbol:
            currency = 'VND'
        elif '.TO' in ticker_symbol or '.TRT' in ticker_symbol:
            currency = 'CAD'
        elif '.L' in ticker_symbol:
            currency = 'GBP'
        elif '.T' in ticker_symbol:
            currency = 'JPY'
        elif '.KS' in ticker_symbol:
            currency = 'KRW'
        elif '.HK' in ticker_symbol:
            currency = 'HKD'
        elif '.SI' in ticker_symbol:
            currency = 'SGD'
        elif '.BK' in ticker_symbol:
            currency = 'THB'
        # Add more ticker suffix mappings as needed
    
    return currency

def get_currency_symbol(currency_code): # Backend
    """Get just the currency symbol for chart labels"""
    currency_symbols = {
        'USD': '$', 'VND': '₫', 'EUR': '€', 'GBP': '£', 'JPY': '¥',
        'KRW': '₩', 'CNY': '¥', 'INR': '₹', 'CAD': 'C$', 'AUD': 'A$',
        'HKD': 'HK$', 'SGD': 'S$', 'THB': '฿'
    }
    return currency_symbols.get(currency_code, currency_code)
    
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
            "/compare <ticker1> <ticker2> - Compare two stocks side by side (e.g., /compare AAPL MSFT)<br>"
            "/predict <ticker> - Predict stock price (e.g., /predict TSLA)<br>"
            "/display <ticker> - Display candlestick chart, volume, and company info (e.g., /display GOOGL)<br>"
        )
    elif command.startswith("/compare"):
        try:
            # Parse tickers
            parts = command.strip().split()
            if len(parts) < 3:
                return "Usage: /compare <ticker1> <ticker2>"
            ticker1, ticker2 = parts[1].upper(), parts[2].upper()
            
            # Validate tickers
            if not check_valid_ticker(ticker1):
                return f"Invalid ticker: {ticker1}"
            if not check_valid_ticker(ticker2):
                return f"Invalid ticker: {ticker2}"
            
            # Fetch data using the standalone function
            data = extract_data_yf([ticker1, ticker2], Period="6mo")
            if ticker1 not in data or ticker2 not in data or data[ticker1].empty or data[ticker2].empty:
                return f"Could not fetch data for {ticker1} or {ticker2}."
            
            # Use Fian for visualization
            fian.compare_stocks(data[ticker1], data[ticker2], title=f"{ticker1} vs {ticker2} Comparison")
            return f"📊 Comparison chart displayed for {ticker1} vs {ticker2}"
        except Exception as e:
            return f"Error comparing stocks: {str(e)}"
    elif command.startswith("/calculate"):
        return "Calculate command placeholder."
    elif command.startswith("/predict"):
        return "Predict command placeholder."
    elif command.startswith("/display"):
        try:
            # Parse ticker
            parts = command.strip().split()
            if len(parts) < 2:
                return "Usage: /display <ticker>"
            ticker = parts[1].upper()
            
            # Validate ticker
            if not check_valid_ticker(ticker):
                return f"Invalid ticker: {ticker}"
            
            # Fetch data
            data = extract_data_yf([ticker], Period="1y")
            if ticker not in data or data[ticker].empty:
                return f"Could not fetch data for {ticker}."
            
            # Use Fian for visualization
            fian.display_stock(data[ticker], ticker)
            return f"📈 Stock information and chart displayed for {ticker}"
        except Exception as e:
            return f"Error displaying stock: {str(e)}"
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





