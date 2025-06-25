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
        from Bian.extractors import extract_tickers
        return extract_tickers(text)

    def extract_intent(self, text):
        from Bian.extractors import extract_intent
        return extract_intent(text)

    def extract_period(self, text):
        from Bian.extractors import extract_period
        return extract_period(text)
    
    def extract_indicator(self, text):
        from Bian.extractors import extract_indicator
        return extract_indicator(text)

    def fetch_data(self, tickers, period="1y"):
        from Bian.bian_utils import extract_data_yf
        return extract_data_yf(tickers, Period=period)
    
    def calculate_indicators(self, tickers, period="1y", indicators=None):
        from Bian.bian_utils import extract_data_yf
        from Bian.configs import indicator_funcs

        temp_data = extract_data_yf(tickers, Period=period)
        if indicators is None:
            return temp_data

        if isinstance(indicators, str):
            indicators = [indicators]

        for ticker, df in temp_data.items():
            df = df.copy()
            for name in indicators:
                func = indicator_funcs.get(name)
                if func is not None:
                    try:
                        df[name] = func(df)
                    except Exception as e:
                        print(f"Error calculating {name} for {ticker}: {e}")
                else:
                    print(f"Indicator '{name}' not supported.")
            temp_data[ticker] = df

        return temp_data



if __name__ == "__main__":
    bian = BackendBian()
    # Example usage
    text = "What is the current RSI of AAPL?"
    tickers = bian.extract_tickers(text)
    intent = bian.extract_intent(text)
    period = bian.extract_period(text)
    indicator = bian.extract_indicator(text)

    data = bian.calculate_indicators(tickers, period, indicator)
    for ticker, df in data.items():
        print(f"Data for {ticker}:\n", df.tail())