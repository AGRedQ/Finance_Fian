
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Mian.mian_utils import add_tracking_ticker, load_tracking_tickers, check_valid_ticker
class MemoryMian:
    def __init__(self): 
        self.tracking_tickers = self.load_tracking_tickers() # Note: Out an Array

    def load_tracking_tickers(self):
        tracking_tickers = load_tracking_tickers()
        return tracking_tickers

    def add_tracking_ticker(self, ticker):
        return add_tracking_ticker(ticker)
        



    
if __name__ == "__main__":
    # Test the functions
    mian = MemoryMian()
    print(mian.add_tracking_ticker("àsdfasdf"))  # Should return "invalid ticker" for an invalid ticker
