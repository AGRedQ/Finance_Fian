import matplotlib.pyplot as plt
import pandas as pd

def line_graph(df, field: str = "Close", title: str = None):  # Frontend
    plt.figure(figsize=(10, 5))
    plt.plot(df[field])
    plt.xlabel("Date")
    plt.ylabel(field)
    if title:
        plt.title(title)
    else:
        plt.title(f"{field} Over Time")
    plt.show()

def line_graphs_compare(multiple_dfs, field="Close", title=None):
    import matplotlib.pyplot as plt

    combined_df = pd.DataFrame()
    for ticker, df in multiple_dfs.items():
        if field in df.columns:
            combined_df[ticker] = df[field]
        else:
            print(f"Field '{field}' not found in {ticker} DataFrame.")
    print("Combined DataFrame for comparison:")
    print(combined_df)
    combined_df.plot(title=title or f"{field} Comparison")
    plt.xlabel("Date")
    plt.ylabel(field)
    plt.legend(title="Ticker")
    plt.tight_layout()
    plt.show()
    if title:
        print(title)

def stock_data_side_by_side(multiple_dfs, period="1y"):
    if not isinstance(multiple_dfs, dict):
        raise ValueError("Input must be a dictionary of DataFrames.")


    combined_df = pd.DataFrame()

    for ticker, df in multiple_dfs.items():
        df.index = pd.to_datetime(df.index)
        df = df.resample('D').ffill()
        df.columns = [f"{col}_{ticker}" for col in df.columns]
        combined_df = pd.concat([combined_df, df], axis=1)

    print("Stocks Side by Side (first 20 rows):")
    print(combined_df.head(20))