import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from Bian.backend_bian import BackendBian

# This will be the spine of the Fian frontend
bian = BackendBian()


class FrontendFian:
    def __init__(self, fian):
        self.fian = fian

    def process_query(self, query):
        # Extract intent
        intent = self.fian.extract_intent(query)
        
        # Extract period
        period = self.fian.extract_period(query)
        
        # Extract tickers
        tickers = self.fian.extract_tickers(query)
        
        # Extract indicators if any
        indicators = self.fian.extract_indicator(query)
        
        return {
            "intent": intent,
            "period": period,
            "tickers": tickers,
            "indicators": indicators
        }