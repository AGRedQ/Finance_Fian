import os
import sys

# Local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from Bian.stock_tools.util_functions import extract_data_yf, line_graph, stock_data_side_by_side, line_graphs_compare


def display_stock(tickers, period="1y", visualize=True):  
    temp_data = extract_data_yf(tickers, Period=period)
    for ticker, df in temp_data.items():
        print(f"\nExtracted data for {ticker}:")
        print(df)
        if visualize:
            line_graph(df, field="Close", title=f"{ticker} Closing Price Over Time")


# Main Function
def compare_stocks(tickers, period="1y", visualize=True):
    temp_data = extract_data_yf(tickers, Period=period)
    stock_data_side_by_side(temp_data, period=period)
    if visualize:
        line_graphs_compare(temp_data, field="Close", title=f"Stock Closing Price Comparison for {', '.join(tickers)}")
    return temp_data


if __name__ == "__main__":
    # Example usage
    tickers = ["AAPL", "GOOGL", "MSFT"]
    period = "1y"
    display_stock(tickers, period=period, visualize=True)
