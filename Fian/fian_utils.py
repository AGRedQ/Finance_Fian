import pandas as pd
import matplotlib.pyplot as plt
from Bian.configs import indicator_plot_config






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