import json
import os
from datetime import datetime

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

def get_current_price(ticker):
    """Get current price for a ticker using yfinance"""
    import yfinance as yf
    try:
        ticker_data = yf.Ticker(ticker)
        hist = ticker_data.history(period="1d")
        if not hist.empty:
            return float(hist['Close'].iloc[-1])
        return None
    except Exception as e:
        return None

def calculate_percent_change(old_price, new_price):
    """Calculate percentage change between old and new price"""
    if old_price and new_price:
        return ((new_price - old_price) / old_price) * 100
    return 0

def calculate_mfi(ticker, period=14):
    """Calculate Money Flow Index for a ticker"""
    import yfinance as yf
    try:
        ticker_data = yf.Ticker(ticker)
        hist = ticker_data.history(period="60d")  # Get enough data for calculation
        
        if len(hist) < period + 1:
            return None
            
        # Calculate typical price
        typical_price = (hist['High'] + hist['Low'] + hist['Close']) / 3
        
        # Calculate money flow
        money_flow = typical_price * hist['Volume']
        
        # Calculate positive and negative money flow
        positive_mf = []
        negative_mf = []
        
        for i in range(1, len(typical_price)):
            if typical_price.iloc[i] > typical_price.iloc[i-1]:
                positive_mf.append(money_flow.iloc[i])
                negative_mf.append(0)
            elif typical_price.iloc[i] < typical_price.iloc[i-1]:
                positive_mf.append(0)
                negative_mf.append(money_flow.iloc[i])
            else:
                positive_mf.append(0)
                negative_mf.append(0)
        
        # Calculate MFI for the last period
        if len(positive_mf) >= period:
            pos_mf_sum = sum(positive_mf[-period:])
            neg_mf_sum = sum(negative_mf[-period:])
            
            if neg_mf_sum == 0:
                return 100
            
            money_ratio = pos_mf_sum / neg_mf_sum
            mfi = 100 - (100 / (1 + money_ratio))
            return round(mfi, 2)
        
        return None
    except Exception as e:
        return None

def calculate_sma_20(ticker):
    """Calculate 20-day Simple Moving Average for a ticker"""
    import yfinance as yf
    try:
        ticker_data = yf.Ticker(ticker)
        hist = ticker_data.history(period="30d")  # Get 30 days to ensure we have enough data
        
        if len(hist) >= 20:
            sma_20 = hist['Close'].rolling(window=20).mean().iloc[-1]
            return round(float(sma_20), 2)
        return None
    except Exception as e:
        return None

def get_ticker_indicators(ticker):
    """Get current price, MFI, and SMA 20 for a ticker"""
    current_price = get_current_price(ticker)
    mfi = calculate_mfi(ticker)
    sma_20 = calculate_sma_20(ticker)
    
    return {
        "price": current_price,
        "mfi": mfi,
        "sma_20": sma_20
    }

# =======================
# User's Setting
# =======================

def load_user_settings():
    filename = 'Mian/user_settings.json'
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    else: return "load_user_settings_error"

def save_user_settings(settings ):
    with open("Mian/user_settings.json", "w") as f:
        json.dump(settings, f, indent=2)
    return settings

# =======================
# Tracking Tickers
# =======================

def load_tracking_tickers():
    filename = 'Mian/tracking_tickers.json'
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    else:
        return {}  # Return empty dict if file doesn't exist

def save_tracking_tickers(tickers_data):
    """Save tracking tickers data to JSON file"""
    with open("Mian/tracking_tickers.json", "w") as f:
        json.dump(tickers_data, f, indent=2)
    return tickers_data

def add_tracking_ticker(ticker, current_price=None):
    """Add a new ticker to tracking with current timestamp, price, and indicators"""
    tickers_data = load_tracking_tickers()
    
    if current_price is None:
        current_price = get_current_price(ticker)
    
    # Get initial indicators
    indicators = get_ticker_indicators(ticker)
    
    tickers_data[ticker] = {
        "added_at": datetime.now().isoformat(),
        "last_checked": datetime.now().isoformat(),
        "last_price": current_price,
        "last_mfi": indicators.get("mfi"),
        "last_sma_20": indicators.get("sma_20"),
        "history": []
    }
    
    save_tracking_tickers(tickers_data)
    return tickers_data

def remove_tracking_ticker(ticker):
    """Remove a ticker from tracking"""
    tickers_data = load_tracking_tickers()
    if ticker in tickers_data:
        del tickers_data[ticker]
        save_tracking_tickers(tickers_data)
    return tickers_data

def update_ticker_price(ticker, new_price=None):
    """Update a ticker's price, MFI, SMA 20 and add to history"""
    tickers_data = load_tracking_tickers()
    if ticker in tickers_data:
        # Get current indicators
        indicators = get_ticker_indicators(ticker)
        current_price = new_price or indicators.get("price")
        
        # Add current data to history
        tickers_data[ticker]["history"].append({
            "timestamp": tickers_data[ticker]["last_checked"],
            "price": tickers_data[ticker]["last_price"],
            "mfi": tickers_data[ticker].get("last_mfi"),
            "sma_20": tickers_data[ticker].get("last_sma_20")
        })
        
        # Update with new data
        tickers_data[ticker]["last_price"] = current_price
        tickers_data[ticker]["last_mfi"] = indicators.get("mfi")
        tickers_data[ticker]["last_sma_20"] = indicators.get("sma_20")
        tickers_data[ticker]["last_checked"] = datetime.now().isoformat()
        
        save_tracking_tickers(tickers_data)
    return tickers_data

def get_ticker_list():
    """Get list of all tracked ticker symbols"""
    tickers_data = load_tracking_tickers()
    return list(tickers_data.keys())

def get_ticker_data(ticker):
    """Get data for a specific ticker"""
    tickers_data = load_tracking_tickers()
    return tickers_data.get(ticker, None)




    
if __name__ == "__main__":
    # Test the functions
    print(len(get_ticker_list()))
