
import os
import sys
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Mian.mian_utils import (
    load_user_settings, save_user_settings,
    load_tracking_tickers, save_tracking_tickers,
    add_tracking_ticker, remove_tracking_ticker,
    update_ticker_price, get_ticker_list, get_ticker_data,
    check_valid_ticker, get_current_price, calculate_percent_change,
    calculate_mfi, calculate_sma_20, get_ticker_indicators
)

class MemoryMian:
    def __init__(self): 
        pass

    # =======================
    # Tracking Tickers
    # =======================
    
    def load_tracking_tickers(self):
        """Load all tracking tickers data"""
        return load_tracking_tickers()
    
    def save_tracking_tickers(self, tickers_data):
        """Save tracking tickers data"""
        return save_tracking_tickers(tickers_data)
    
    def add_tracking_ticker(self, ticker):
        """Add a new ticker to tracking"""
        if check_valid_ticker(ticker):
            return add_tracking_ticker(ticker)
        else:
            return f"Invalid ticker: {ticker}"
    
    def remove_tracking_ticker(self, ticker):
        """Remove a ticker from tracking"""
        return remove_tracking_ticker(ticker)
    
    def get_ticker_list(self):
        """Get list of all tracked ticker symbols"""
        return get_ticker_list()
    
    def get_ticker_data(self, ticker):
        """Get data for a specific ticker"""
        return get_ticker_data(ticker)
    
    def update_ticker_price(self, ticker, new_price):
        """Update a ticker's price"""
        return update_ticker_price(ticker, new_price)
    
    def get_current_price(self, ticker):
        """Get current price for a ticker"""
        return get_current_price(ticker)
    
    def get_ticker_indicators(self, ticker):
        """Get current price, MFI, and SMA 20 for a ticker"""
        return get_ticker_indicators(ticker)
    
    def calculate_mfi(self, ticker):
        """Calculate Money Flow Index for a ticker"""
        return calculate_mfi(ticker)
    
    def calculate_sma_20(self, ticker):
        """Calculate 20-day Simple Moving Average for a ticker"""
        return calculate_sma_20(ticker)
    
    def calculate_percent_change(self, ticker):
        """Calculate percent change since last check"""
        ticker_data = get_ticker_data(ticker)
        if ticker_data and ticker_data.get("last_price"):
            current_price = get_current_price(ticker)
            if current_price:
                return calculate_percent_change(ticker_data["last_price"], current_price)
        return 0
    
    def update_all_ticker_prices(self):
        """Update prices and indicators for all tracked tickers"""
        tickers_data = load_tracking_tickers()
        updated_data = {}
        
        for ticker in tickers_data:
            indicators = get_ticker_indicators(ticker)
            if indicators.get("price"):
                update_ticker_price(ticker)
                updated_data[ticker] = indicators
        
        return updated_data

    def get_total_tickers(self):
        return len(load_tracking_tickers())

    # =======================
    # User's Setting
    # =======================
    def load_user_settings(self):
        return load_user_settings()
    
    def save_user_settings(self, settings):
        settings = save_user_settings(settings)
        return settings

    # =======================
    # Activity Tracking
    # =======================
    def log_activity(self, activity_message):
        """Log a new activity with timestamp"""
        from datetime import datetime
        
        activity_file = os.path.join(os.path.dirname(__file__), "user_activities.json")
        
        # Load existing activities
        try:
            with open(activity_file, 'r') as f:
                activities = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            activities = []
        
        # Add new activity with timestamp
        new_activity = {
            "message": activity_message,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "date": datetime.now().strftime("%Y-%m-%d")
        }
        
        activities.insert(0, new_activity)  # Add to beginning
        
        # Keep only last 50 activities
        activities = activities[:50]
        
        # Save back to file
        try:
            with open(activity_file, 'w') as f:
                json.dump(activities, f, indent=2)
        except Exception as e:
            print(f"Error saving activity: {e}")
        
        return activities
    
    def get_recent_activities(self, limit=10):
        """Get recent activities"""
        from datetime import datetime
        
        activity_file = os.path.join(os.path.dirname(__file__), "user_activities.json")
        
        try:
            with open(activity_file, 'r') as f:
                activities = json.load(f)
            
            # Return limited activities with formatted timestamps
            recent = activities[:limit]
            for activity in recent:
                # Format timestamp for display
                try:
                    dt = datetime.strptime(activity["timestamp"], "%Y-%m-%d %H:%M:%S")
                    activity["display_time"] = dt.strftime("%I:%M %p")
                    activity["display_date"] = dt.strftime("%b %d")
                except:
                    activity["display_time"] = "Unknown"
                    activity["display_date"] = "Unknown"
            
            return recent
        except (FileNotFoundError, json.JSONDecodeError):
            return []
    
    def clear_activities(self):
        """Clear all activities"""
        activity_file = os.path.join(os.path.dirname(__file__), "user_activities.json")
        try:
            with open(activity_file, 'w') as f:
                json.dump([], f)
            return True
        except Exception as e:
            print(f"Error clearing activities: {e}")
            return False





    
if __name__ == "__main__":
    # Test the functions
    mian = MemoryMian()
