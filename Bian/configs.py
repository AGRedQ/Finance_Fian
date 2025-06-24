# list of supported indicators
indicators_list = [
    "MACD", "MACD_signal", "MACD_diff", "ADX", "CCI", "Ichimoku_a", "Ichimoku_b",
    "PSAR", "STC", "RSI", "Stoch", "Stoch_signal", "AwesomeOsc", "KAMA", "ROC", "TSI",
    "UO", "ATR", "Bollinger_hband", "Bollinger_lband", "Bollinger_mavg", "Donchian_hband",
    "Donchian_lband", "Keltner_hband", "Keltner_lband", "Donchian_width", "SMA_5", "EMA_5",
    "WMA_5", "DEMA_5", "TEMA_5", "SMA_10", "EMA_10", "WMA_10", "DEMA_10", "TEMA_10", 
    "SMA_20", "EMA_20", "WMA_20", "DEMA_20", "TEMA_20", "SMA_50", "EMA_50", "WMA_50", 
    "DEMA_50", "TEMA_50", "SMA_100", "EMA_100", "WMA_100", "DEMA_100", "TEMA_100", 
    "SMA_200", "EMA_200", "WMA_200", "DEMA_200", "TEMA_200"
]

# Configuration for plotting indicators
indicator_plot_config = {
    # MACD Family
    "MACD": {"type": "line", "guides": [0], "subplot": True, "paired": "MACD_signal"},
    "MACD_signal": {"type": "line", "guides": [0], "subplot": True, "paired": "MACD"},
    "MACD_diff": {"type": "histogram", "guides": [0], "subplot": True},

    # Trend Strength
    "ADX": {"type": "line", "guides": [20, 40], "subplot": True},
    "CCI": {"type": "line", "guides": [-100, 100], "subplot": True},

    # Ichimoku Cloud
    "Ichimoku_a": {"type": "line", "subplot": False},
    "Ichimoku_b": {"type": "line", "subplot": False},

    # Price Overlay
    "PSAR": {"type": "scatter", "subplot": False},

    # Momentum Oscillators
    "STC": {"type": "line", "guides": [25, 75], "subplot": True},
    "RSI": {"type": "line", "guides": [30, 70], "subplot": True},
    "Stoch": {"type": "line", "guides": [20, 80], "subplot": True, "paired": "Stoch_signal"},
    "Stoch_signal": {"type": "line", "guides": [20, 80], "subplot": True, "paired": "Stoch"},
    "AwesomeOsc": {"type": "histogram", "guides": [0], "subplot": True},
    "KAMA": {"type": "line", "subplot": False},
    "ROC": {"type": "line", "guides": [0], "subplot": True},
    "TSI": {"type": "line", "guides": [0], "subplot": True},
    "UO": {"type": "line", "guides": [30, 70], "subplot": True},

    # Volatility
    "ATR": {"type": "line", "subplot": True},
    "Bollinger_hband": {"type": "line", "subplot": False},
    "Bollinger_lband": {"type": "line", "subplot": False},
    "Bollinger_mavg": {"type": "line", "subplot": False},
    "Donchian_hband": {"type": "line", "subplot": False},
    "Donchian_lband": {"type": "line", "subplot": False},
    "Keltner_hband": {"type": "line", "subplot": False},
    "Keltner_lband": {"type": "line", "subplot": False},
    "Donchian_width": {"type": "line", "subplot": True},

    # Moving Averages (Overlays)
    "SMA_5": {"type": "line", "subplot": False},
    "EMA_5": {"type": "line", "subplot": False},
    "WMA_5": {"type": "line", "subplot": False},
    "DEMA_5": {"type": "line", "subplot": False},
    "TEMA_5": {"type": "line", "subplot": False},
    "SMA_10": {"type": "line", "subplot": False},
    "EMA_10": {"type": "line", "subplot": False},
    "WMA_10": {"type": "line", "subplot": False},
    "DEMA_10": {"type": "line", "subplot": False},
    "TEMA_10": {"type": "line", "subplot": False},
    "SMA_20": {"type": "line", "subplot": False},
    "EMA_20": {"type": "line", "subplot": False},
    "WMA_20": {"type": "line", "subplot": False},
    "DEMA_20": {"type": "line", "subplot": False},
    "TEMA_20": {"type": "line", "subplot": False},
    "SMA_50": {"type": "line", "subplot": False},
    "EMA_50": {"type": "line", "subplot": False},
    "WMA_50": {"type": "line", "subplot": False},
    "DEMA_50": {"type": "line", "subplot": False},
    "TEMA_50": {"type": "line", "subplot": False},
    "SMA_100": {"type": "line", "subplot": False},
    "EMA_100": {"type": "line", "subplot": False},
    "WMA_100": {"type": "line", "subplot": False},
    "DEMA_100": {"type": "line", "subplot": False},
    "TEMA_100": {"type": "line", "subplot": False},
    "SMA_200": {"type": "line", "subplot": False},
    "EMA_200": {"type": "line", "subplot": False},
    "WMA_200": {"type": "line", "subplot": False},
    "DEMA_200": {"type": "line", "subplot": False},
    "TEMA_200": {"type": "line", "subplot": False},
}

