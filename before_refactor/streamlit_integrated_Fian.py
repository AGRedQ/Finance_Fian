import re
import spacy
import yfinance as yf
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from joblib import load
import warnings
import google.generativeai as genai
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import os
import sys
# Import locals
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Bian.configs import indicators_list, indicator_plot_config


# ====================================================================================================
# Section: Memory Class
# ====================================================================================================

# Memory_Manager class to handle memory operations
# She will be used to personalize the chatbot and store user preferences, chat history, etc.
# Note: Testing JSON format for memory storage






# ====================================================================================================
# Section: Technical Indicators
# ====================================================================================================

# Gemini Gemma-3-12b-it
api_key = "AIzaSyByTfCk2a6m4gkeJAuCpWGmWi8qfyHBQ3w"
generative_model = "gemma-3-12b-it"
genai.configure(api_key=api_key)
model = genai.GenerativeModel(generative_model)
print(generative_model + " model loaded")
# NLP SpaCy "en_core_web_trf"
spacy_model = "en_core_web_trf"

nlp = spacy.load(spacy_model)
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
print(spacy_model + " model loaded")

# ====================================================================================================
# Section: Natual Language Understanding (NLU)
# ====================================================================================================

## == == == -- -- -- Helper Functions -- -- -- == == == ##

def preprocess_query(text): # Backend
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return ' '.join(tokens)


def run_NER(text): # Backend
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

def extract_entities(entities, label): # Backend
    return [ent_text for ent_text, ent_label in entities if ent_label == label]

def to_yf_period(text): # Backend
    match = re.search(r'(\d+)\s*(year|years|month|months|week|weeks|day|days)', text, re.IGNORECASE)
    if match:
        number = match.group(1)
        unit = match.group(2).lower()

        if 'year' in unit:
            return f"{number}y"
        elif 'month' in unit:
            return f"{number}mo"
        elif 'week' in unit:
            days = int(number) * 7
            return f"{days}d"
        elif 'day' in unit:
            return f"{number}d"
    return None

def yfinance_search_company(company_names): # Backend
    results = {}
    for name in company_names:
        s = yf.Search(name, max_results=1)
        if s.quotes:
            results[name] = s.quotes[0].get("symbol")
        else:
            results[name] = None
    # Return a list of ticker symbols (filtering out any None values)
    return [ticker for ticker in results.values() if ticker]

def extract_tickers(text): # Backend
    entities = run_NER(text)
    company_names = extract_entities(entities, "ORG")
    tickers = yfinance_search_company(company_names)
    return tickers


def extract_intent(text): # Backend
    prompt = (
        "You are an intention extraction parser. Given the user's input, return only one of the following intent categories: "
        "'display_price', 'compare_stocks', 'calculate_indicator', 'predict_indicator', or 'not_stock'.\n"
        "Descriptions:\n"
        "display_price: The user wants to know the current price or see a visual chart of a specific stock.\n"
        "compare_stocks: The user wants to compare two or more stocks.\n"
        "calculate_indicator: The user wants to calculate a technical indicator (e.g., RSI, SMA, MACD) for a stock.\n"
        "predict_indicator: The user wants a prediction or forecast of a technical indicator for a stock.\n"
        "not_stock: The user's input does not relate to stock trading or analysis.\n"
        "User: " + text + "\n"
        "Intent category:"
    )
    response = model.generate_content(prompt)
    # Extract the intent category from the response (assume it's the first word/line)
    intent = response.text.strip().split('\n')[0].strip().lower()
    # Map to expected categories if needed
    valid_intents = ["display_price", "compare_stocks", "calculate_indicator", "predict_indicator", "not_stock"]
    if intent in valid_intents:
        return intent
    # fallback: 
    return "ask_again_for_intent"

def extract_period(text): # Backend
    entities = run_NER(text)
    date_entities = extract_entities(entities, "DATE")

    if len(date_entities) >= 2: 
        print("Multiple Date Ranges are not compatible YET. I will add later. Default: 1y") 
        return "1y"

    if len(date_entities) == 1:
        period = to_yf_period(text)
        if period:
            return period
        else:
            return "1y"  

    warnings.filterwarnings("ignore")
    return "1y"

def extract_indicator(text): # Backend
    global indicators_list
    # Prompt engineering: add system instruction for better responses 
    indicator_comma = ', '.join(indicators_list)

    prompt = (
        "You are an indicator extraction parser. Given the user's input, return a list of technical indicators they want to calculate or predict.\n"
        "User: " + text + "\n"
        "Indicators (comma-separated): " + indicator_comma + "\n"
    )
    response = model.generate_content(prompt)
    # Extract the indicators from the response (assume they are comma-separated)        
    indicators_text = response.text.strip()
    if indicators_text:
        # Split by commas and strip whitespace
        indicators_list = [indicator.strip() for indicator in indicators_text.split(',')]
        # Filter out any empty strings
        return [indicator for indicator in indicators_list if indicator]
    return "ask_again_for_indicator"


# ====================================================================================================
# Section: Fuction for display_price
# ====================================================================================================

import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd

def extract_data_yf(tickers, Period = "1y"): # Backend 
    # Note: Remember to make a way to delelte these files after use. Since they are only temporary files.
    data = {}
    for ticker in tickers:
        df = yf.download(ticker, period=Period, interval="1d",auto_adjust=True, progress=False)
        filename = f"temp_{ticker}_{Period}.csv"
        df.to_csv(filename)
        data[ticker] = df
    return data

def display_price(data, n_rows: int = 10):  # Frontend
    if isinstance(data, dict):
        for ticker, df in data.items():
            st.subheader(f"{ticker}: First {n_rows} rows")
            st.dataframe(df.head(n_rows))
    elif isinstance(data, pd.DataFrame):
        st.subheader(f"First {n_rows} rows")
        st.dataframe(data.head(n_rows))
    else:
        st.warning("Input data must be a dict of DataFrames or a DataFrame.")

def line_graph(df, field: str = "Close", title: str = None):  # Frontend
    st.line_chart(df[field])
    if title:
        st.caption(title)

def display_stock(tickers, period="1y", visualize=True):  # Frontend
    temp_data = extract_data_yf(tickers, Period=period)
    for ticker, df in temp_data.items():
        st.write(f"Extracted data for {ticker}:")
        st.dataframe(df, height=400, use_container_width=True)
        if visualize:
            st.line_chart(df["Close"])
            st.caption(f"{ticker} Closing Price Over Time")


# ====================================================================================================
# Section: Function for compare_stocks
# ====================================================================================================

# compare_stocks function
from IPython.display import display

## == == == -- -- -- Helper Functions -- -- -- == == == ##


def stock_data_side_by_side(multiple_dfs, period="1y"):  # Frontend (Streamlit compatible)
    if not isinstance(multiple_dfs, dict):
        raise ValueError("Input must be a dictionary of DataFrames.")

    combined_df = pd.DataFrame()

    for ticker, df in multiple_dfs.items():
        df.index = pd.to_datetime(df.index)
        df = df.resample('D').ffill()
        df.columns = [f"{col}_{ticker}" for col in df.columns]
        combined_df = pd.concat([combined_df, df], axis=1)

    st.subheader("Stocks Side by Side")
    st.dataframe(combined_df.head(20), use_container_width=True)

def line_graphs_compare(multiple_dfs, field="Close", title=None):  # Frontend (Streamlit integrated)
    st.subheader(title or f"{field} Comparison")
    combined_df = pd.DataFrame()
    for ticker, df in multiple_dfs.items():
        if field in df.columns:
            combined_df[ticker] = df[field]
        else:
            st.warning(f"Field '{field}' not found in {ticker} DataFrame.")
    st.dataframe(combined_df, use_container_width=True)
    st.line_chart(combined_df)
    if title:
        st.caption(title)
## == == == -- -- -- Main Execute Functions -- -- -- == == == ##

def compare_stocks(tickers, period="1y", visualize=True): # Frontend
    temp_data = extract_data_yf(tickers, Period=period)
    stock_data_side_by_side(temp_data, period=period)
    if visualize:
        line_graphs_compare(temp_data, field="Close", title=f"Stock Closing Price Comparison for {', '.join(tickers)}")
    return temp_data


# ====================================================================================================
# Section: Function for calculate_indicator
# ====================================================================================================

# calculate_indicator function

from collections import OrderedDict
import ta
import pandas as pd
import streamlit as st
# Define 30 most used technical indicators using 'ta' library
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



def visualize_indicator(data, indicator_name, title=None):  # Streamlit integrated
    config = indicator_plot_config.get(indicator_name, {"type": "line", "subplot": False})
    plot_type = config.get("type", "line")
    guides = config.get("guides", [])
    subplot = config.get("subplot", False)
    paired = config.get("paired", None)

    for ticker, df in data.items():
        if indicator_name not in df.columns:
            st.warning(f"{indicator_name} not found in {ticker} data.")
            continue

        chart_df = pd.DataFrame(index=df.index)
        chart_df[indicator_name] = df[indicator_name]

        # Add paired indicator if specified
        if paired and paired in df.columns:
            chart_df[paired] = df[paired]

        st.subheader(f"{ticker} - {indicator_name} Visualization")
        if plot_type == "line":
            st.line_chart(chart_df)
        elif plot_type == "histogram":
            st.bar_chart(chart_df[indicator_name])
        elif plot_type == "scatter":
            st.scatter_chart(chart_df)
        else:
            st.line_chart(chart_df)

        # Show guides as reference lines (Streamlit doesn't support native hlines, so just info)
        if guides:
            st.info(f"Reference lines at: {guides}")

        if title:
            st.caption(title)


def calculate_indicator(tickers, period="1y", indicators=None, visualize=True):  # Streamlit integrated
    temp_data = extract_data_yf(tickers, Period=period)
    if indicators is None:
        st.warning("No indicators specified. Returning raw data.")
        return temp_data

    # Ensure indicators is a list
    if isinstance(indicators, str):
        indicators = [indicators]

    for ticker, df in temp_data.items():
        st.write(f"Calculating indicators for **{ticker}**...")
        df = df.copy()
        for name in indicators:
            func = indicator_funcs.get(name)
            if func is not None:
                try:
                    df[name] = func(df)
                except Exception as e:
                    st.error(f"Error calculating {name} for {ticker}: {e}")
            else:
                st.warning(f"Indicator '{name}' not supported.")
        temp_data[ticker] = df
        st.dataframe(df.tail(), use_container_width=True)

    if visualize:
        for indicator_name in indicators:
            st.subheader(f"Visualization for {indicator_name}")
            visualize_indicator(temp_data, indicator_name)

    return temp_data


# ====================================================================================================
# Section: Funtions for predict_price
# ====================================================================================================








# ====================================================================================================
# Section: Assembly of the Intent Processing Pipeline
# ====================================================================================================

def process_intent(intent, raw_query):

    if intent == "not_stock":
        prompt = (
            "This user query does not relate to stock trading or analysis.\n"
            "Chat with the user, but do let them know that you are a small stock market assistant and cannot help with this query.\n"
            "You can fetch live data, calculate indicators, and predict stock prices.\n"
            "User: " + raw_query + "\n"
            "Response: "
        )
        response = model.generate_content(prompt)
        st.info(response.text)

    elif intent == "display_price":
        period = extract_period(raw_query)
        tickers = extract_tickers(raw_query)
        if tickers:
            display_stock(tickers, period=period, visualize=True)
        else:
            st.warning("No valid stock tickers found in the query.")

    elif intent == "compare_stocks":
        period = extract_period(raw_query)
        tickers = extract_tickers(raw_query)
        if tickers and len(tickers) > 1:
            compare_stocks(tickers, period=period, visualize=True)
        else:
            st.warning("Not enough valid stock tickers found for comparison.")

    elif intent == "calculate_indicator":
        period = extract_period(raw_query)
        tickers = extract_tickers(raw_query)
        indicators = extract_indicator(raw_query)
        if tickers and indicators:
            calculate_indicator(tickers, period, indicators, True)
        else:
            st.warning("Could not extract tickers or indicators from your query.")

    elif intent == "predict_price":
        period = extract_period(raw_query)
        tickers = extract_tickers(raw_query)
        indicator = extract_indicator(raw_query)
        if tickers and indicator:
            st.info(f"Prediction functionality is not yet implemented. Would predict {indicator} for {', '.join(tickers)} over the period of {period}.")
        else:
            st.warning("No valid stock tickers or indicators found for prediction.")
    else:  # fallback
        st.warning("Intent not recognized or not implemented. Please try again with a different query.")




# ====================================================================================================
# Section: Main Execution
# ====================================================================================================

st.set_page_config(page_title="Stock Market Assistant", layout="wide")

st.title("Stock Market Assistant Chatbot 🤖")
st.write("Ask me about stock prices, compare stocks, calculate indicators, or get predictions!")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("Type your query here:", key="user_input")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    intent = extract_intent(user_input)
    with st.spinner("Processing..."):
        process_intent(intent, user_input)  # Working: not_stock, display_price
    st.session_state.chat_history.append({"role": "assistant", "content": f"Processed intent: {intent}"})
    # Do not attempt to modify st.session_state.user_input here to avoid Streamlit widget error

st.subheader("Chat History")
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    else:
        st.markdown(f"**Assistant:** {msg['content']}")