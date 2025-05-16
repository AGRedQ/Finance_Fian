import streamlit as st
import requests
import yfinance as yf
import matplotlib.pyplot as plt
import openai
import os

openai.api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

SYSTEM_PROMPT = """
You are Fian, a helpful stock assistant. From a user message, extract:
- company: the company name they mention
- period: a Yahoo Finance period string (e.g., "6mo", "1y", "5d").
If not found, default to "6mo". Respond in this exact JSON format:

{"company": "Company Name", "period": "6mo"}
"""

def extract_company_and_period(user_input):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input}
        ]
    )
    try:
        text = response["choices"][0]["message"]["content"]
        return eval(text)
    except Exception as e:
        st.error(f"Failed to parse AI response: {e}")
        return None

def company_to_ticker(company_name):
    url = f"https://query2.finance.yahoo.com/v1/finance/search?q={company_name}"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        quotes = response.json().get("quotes")
        return quotes[0]["symbol"] if quotes else None
    except Exception as e:
        st.error(f"Error while fetching ticker: {e}")
        return None

def fetch_and_plot_stock(ticker, period):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)

    if df.empty:
        st.error("No data found for this ticker.")
        return

    df["SMA50"] = df["Close"].rolling(window=50).mean()
    df["SMA200"] = df["Close"].rolling(window=200).mean()

    st.write(f"### {ticker} — Close Price Over {period}")
    st.dataframe(df.tail(5))

    fig, ax = plt.subplots()
    df["Close"].plot(ax=ax, label="Close", alpha=0.6)
    df["SMA50"].plot(ax=ax, label="SMA50")
    df["SMA200"].plot(ax=ax, label="SMA200")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.set_title(f"{ticker} Closing Price")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

# --- Streamlit Chat UI ---
st.title("📈 Chat with Fian — Your Stock Assistant")
st.write("Ask me about any stock and I'll plot it for you!")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_input = st.chat_input("Type your message here...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    result = extract_company_and_period(user_input)
    if result:
        company = result.get("company")
        period = result.get("period", "6mo")
        st.chat_message("assistant").write(f"Got it! Checking **{company}** over **{period}**...")

        ticker = company_to_ticker(company)
        if ticker:
            fetch_and_plot_stock(ticker, period)
        else:
            st.chat_message("assistant").write("Couldn't find a ticker for that company.")
    else:
        st.chat_message("assistant").write("Sorry, I couldn’t understand your request.")
