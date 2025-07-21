import ta
from collections import OrderedDict

# list of supported indicators
indicators_list = [
    "MACD", "MACD_signal", "MACD_diff", "ADX", "CCI", "Ichimoku_a", "Ichimoku_b",
    "PSAR", "STC", "RSI", "Stoch", "Stoch_signal", "AwesomeOsc", "KAMA", "ROC", "TSI",
    "UO", "MFI", "ATR", "Bollinger_hband", "Bollinger_lband", "Bollinger_mavg", "Donchian_hband",
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
    "MFI": {"type": "line", "guides": [20, 80], "subplot": True},

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


indicator_funcs = OrderedDict({
    # Trend indicators
    "MACD": lambda df: ta.trend.MACD(close=df["Close"].squeeze()).macd(),
    "MACD_signal": lambda df: ta.trend.MACD(close=df["Close"].squeeze()).macd_signal(),
    "MACD_diff": lambda df: ta.trend.MACD(close=df["Close"].squeeze()).macd_diff(),
    "ADX": lambda df: ta.trend.ADXIndicator(high=df["High"].squeeze(), low=df["Low"].squeeze(), close=df["Close"].squeeze()).adx(),
    "CCI": lambda df: ta.trend.CCIIndicator(high=df["High"].squeeze(), low=df["Low"].squeeze(), close=df["Close"].squeeze()).cci(),
    "Ichimoku_a": lambda df: ta.trend.IchimokuIndicator(high=df["High"].squeeze(), low=df["Low"].squeeze()).ichimoku_a(),
    "Ichimoku_b": lambda df: ta.trend.IchimokuIndicator(high=df["High"].squeeze(), low=df["Low"].squeeze()).ichimoku_b(),
    "PSAR": lambda df: ta.trend.PSARIndicator(high=df["High"].squeeze(), low=df["Low"].squeeze(), close=df["Close"].squeeze()).psar(),
    "STC": lambda df: ta.trend.STCIndicator(close=df["Close"].squeeze()).stc(),

    # Momentum indicators
    "RSI": lambda df: ta.momentum.RSIIndicator(close=df["Close"].squeeze()).rsi(),
    "Stoch": lambda df: ta.momentum.StochasticOscillator(high=df["High"].squeeze(), low=df["Low"].squeeze(), close=df["Close"].squeeze()).stoch(),
    "Stoch_signal": lambda df: ta.momentum.StochasticOscillator(high=df["High"].squeeze(), low=df["Low"].squeeze(), close=df["Close"].squeeze()).stoch_signal(),
    "AwesomeOsc": lambda df: ta.momentum.AwesomeOscillatorIndicator(high=df["High"].squeeze(), low=df["Low"].squeeze()).awesome_oscillator(),
    "KAMA": lambda df: ta.momentum.KAMAIndicator(close=df["Close"].squeeze()).kama(),
    "ROC": lambda df: ta.momentum.ROCIndicator(close=df["Close"].squeeze()).roc(),
    "TSI": lambda df: ta.momentum.TSIIndicator(close=df["Close"].squeeze()).tsi(),
    "UO": lambda df: ta.momentum.UltimateOscillator(high=df["High"].squeeze(), low=df["Low"].squeeze(), close=df["Close"].squeeze()).ultimate_oscillator(),
    "MFI": lambda df: ta.volume.MFIIndicator(high=df["High"].squeeze(), low=df["Low"].squeeze(), close=df["Close"].squeeze(), volume=df["Volume"].squeeze()).money_flow_index(),

    # Volatility indicators
    "ATR": lambda df: ta.volatility.AverageTrueRange(high=df["High"].squeeze(), low=df["Low"].squeeze(), close=df["Close"].squeeze()).average_true_range(),
    "Bollinger_hband": lambda df: ta.volatility.BollingerBands(close=df["Close"].squeeze()).bollinger_hband(),
    "Bollinger_lband": lambda df: ta.volatility.BollingerBands(close=df["Close"].squeeze()).bollinger_lband(),
    "Bollinger_mavg": lambda df: ta.volatility.BollingerBands(close=df["Close"].squeeze()).bollinger_mavg(),
    "Donchian_hband": lambda df: ta.volatility.DonchianChannel(high=df["High"].squeeze(), low=df["Low"].squeeze()).donchian_channel_hband(),
    "Donchian_lband": lambda df: ta.volatility.DonchianChannel(high=df["High"].squeeze(), low=df["Low"].squeeze()).donchian_channel_lband(),
    "Keltner_hband": lambda df: ta.volatility.KeltnerChannel(high=df["High"].squeeze(), low=df["Low"].squeeze(), close=df["Close"].squeeze()).keltner_channel_hband(),
    "Keltner_lband": lambda df: ta.volatility.KeltnerChannel(high=df["High"].squeeze(), low=df["Low"].squeeze(), close=df["Close"].squeeze()).keltner_channel_lband(),
    "Donchian_width": lambda df: ta.volatility.DonchianChannel(high=df["High"].squeeze(), low=df["Low"].squeeze()).donchian_channel_width(),
})

# Add SMA with different window sizes to indicator_funcs
for win in [5, 10, 20, 50, 100, 200]:
    indicator_funcs[f"SMA_{win}"] = lambda df, w=win: ta.trend.SMAIndicator(close=df["Close"].squeeze(), window=w).sma_indicator()
    indicator_funcs[f"EMA_{win}"] = lambda df, w=win: ta.trend.EMAIndicator(close=df["Close"].squeeze(), window=w).ema_indicator()
    indicator_funcs[f"WMA_{win}"] = lambda df, w=win: ta.trend.WMAIndicator(close=df["Close"].squeeze(), window=w).wma()
    indicator_funcs[f"DEMA_{win}"] = lambda df, w=win: ta.trend.DEMAIndicator(close=df["Close"].squeeze(), window=w).dema_indicator()
    indicator_funcs[f"TEMA_{win}"] = lambda df, w=win: ta.trend.TEMAIndicator(close=df["Close"].squeeze(), window=w).tema_indicator()

