import pandas as pd
import matplotlib.pyplot as plt
from Bian.configs import indicator_plot_config

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


def visualize_indicator(data, indicator_name, title=None): # Frontend
    config = indicator_plot_config.get(indicator_name, {"type": "line", "subplot": False})
    plot_type = config.get("type", "line")
    guides = config.get("guides", [])
    subplot = config.get("subplot", False)
    paired = config.get("paired", None)

    for ticker, df in data.items():
        if indicator_name not in df.columns:
            print(f"{indicator_name} not found in {ticker} data.")
            continue

        chart_df = pd.DataFrame(index=df.index)
        chart_df[indicator_name] = df[indicator_name]

        # Add paired indicator if specified
        if paired and paired in df.columns:
            chart_df[paired] = df[paired]

        plt.figure(figsize=(10, 5))
        if plot_type == "line":
            plt.plot(chart_df.index, chart_df[indicator_name], label=indicator_name)
            if paired and paired in chart_df.columns:
                plt.plot(chart_df.index, chart_df[paired], label=paired)
        elif plot_type == "histogram":
            plt.bar(chart_df.index, chart_df[indicator_name], label=indicator_name)
        elif plot_type == "scatter":
            plt.scatter(chart_df.index, chart_df[indicator_name], label=indicator_name)
        else:
            plt.plot(chart_df.index, chart_df[indicator_name], label=indicator_name)

        # Show guides as reference lines
        for guide in guides:
            plt.axhline(y=guide, color='r', linestyle='--', linewidth=1, label=f'Guide {guide}')

        plt.xlabel("Date")
        plt.ylabel(indicator_name)
        plt.title(title or f"{ticker} - {indicator_name} Visualization")
        plt.legend()
        plt.tight_layout()
        plt.show()