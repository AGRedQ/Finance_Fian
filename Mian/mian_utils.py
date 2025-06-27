import json
import os

def load_tracking_tickers():
    filename = 'Mian/tracking_tickers.json'
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    else:
        return []
def check_valid_ticker(ticker):
    # Funny enough, just try to extract the ticker from yfinance
    # period is like... a day? If it can extract the ticker, it is valid
    import yfinance as yf
    try:
        test_extract = yf.download(ticker, period="1d", auto_adjust=True, progress=False)
        if not test_extract.empty:
            return True # Ticker is valid
        else:
            return False # Ticker is invalid
    except Exception as e:
        return False # Ticker is invalid
    
    
def add_tracking_ticker(ticker):  
    if ticker is None or ticker.strip() == "":
        return None

    if not check_valid_ticker(ticker):
        return "invalid_ticker"
    
    filename = 'Mian/tracking_tickers.json'
    tickers = []
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            tickers = json.load(f)
        if ticker in tickers:
            return "existed"
    else:
        tickers = []
    tickers.append(ticker)
    with open(filename, 'w') as f:
        json.dump(tickers, f)
    return "added"

    
if __name__ == "__main__":
    # Test the functions
    print(add_tracking_ticker("NOTVALID"))  # Should return False for an invalid ticker

