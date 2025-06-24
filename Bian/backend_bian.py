# This will be the spine of the Bian backend
# It will be handled as a Object
# If you are testing this file without a loop, model will be load each time you run it.
# If you are using it in a loop, it will be loaded only once.

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class BackendBian:
    def __init__(self):
        pass

    def extract_tickers(self, text):
        from Bian.NLP.extractors import extract_tickers
        return extract_tickers(text)

    def extract_intent(self, text):
        from Bian.NLP.extractors import extract_intent
        return extract_intent(text)

    def extract_period(self, text):
        from Bian.NLP.extractors import extract_period
        return extract_period(text)
    
    def extract_indicator(self, text):
        from Bian.NLP.extractors import extract_indicator
        return extract_indicator(text)
    
    
    




if __name__ == "__main__":
    backend = BackendBian()
    # Example usage
    text = "What is the current price of AAPL?"
    tickers = backend.extract_tickers(text)
    intent = backend.extract_intent(text)
    period = backend.extract_period(text)
    indicator = backend.extract_indicator(text)

    print(f"Tickers: {tickers}")
    print(f"Intent: {intent}")
    print(f"Period: {period}")
    print(f"Indicator: {indicator}")