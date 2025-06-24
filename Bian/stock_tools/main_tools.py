import os
import sys

# Local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from Bian.stock_tools.util_functions import extract_data_yf, line_graph, stock_data_side_by_side, line_graphs_compare, visualize_indicator
from Bian.configs import indicator_funcs


def display_stock(tickers, period="1y", visualize=True):  
    temp_data = extract_data_yf(tickers, Period=period)
    for ticker, df in temp_data.items():
        print(f"\nExtracted data for {ticker}:")
        print(df)
        if visualize:
            line_graph(df, field="Close", title=f"{ticker} Closing Price Over Time")


def compare_stocks(tickers, period="1y", visualize=True):
    temp_data = extract_data_yf(tickers, Period=period)
    stock_data_side_by_side(temp_data, period=period)
    if visualize:
        line_graphs_compare(temp_data, field="Close", title=f"Stock Closing Price Comparison for {', '.join(tickers)}")
    return temp_data

def calculate_indicator(tickers, period="1y", indicators=None, visualize=True):
    temp_data = extract_data_yf(tickers, Period=period)
    if indicators is None:
        print("No indicators specified. Returning raw data.")
        return temp_data

    # Ensure indicators is a list
    if isinstance(indicators, str):
        indicators = [indicators]

    for ticker, df in temp_data.items():
        print(f"Calculating indicators for {ticker}...")
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
        print(df.tail())

    if visualize:
        for indicator_name in indicators:
            print(f"Visualization for {indicator_name}")
            visualize_indicator(temp_data, indicator_name)

    return temp_data




if __name__ == "__main__":
    # Example usage
    tickers = ["AAPL", "GOOGL", "MSFT"]
    period = "1y"
    calculate_indicator(tickers, period=period, indicators=["MACD", "RSI"], visualize=True)