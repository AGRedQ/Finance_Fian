import streamlit as st
import requests
import yfinance as yf
import matplotlib.pyplot as plt

# Function to convert company name to ticker symbol
def company_to_ticker(company_name):
    url = f"https://query2.finance.yahoo.com/v1/finance/search?q={company_name}"
    headers = {
        "User-Agent": "Mozilla/5.0"  # Pretend to be a browser
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise if not 200
        data = response.json()
        quotes = data.get("quotes")
        if quotes and len(quotes) > 0:
            return quotes[0]["symbol"]
        else:
            return None
    except Exception as e:
        st.error(f"Error while fetching ticker: {e}")
        return None

# Function to fetch stock data and plot it
def fetch_and_plot_stock(ticker):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="max")

        if df.empty:
            st.error("No data found for ticker.")
            return

        # Calculate SMAs
        df["SMA50"] = df["Close"].rolling(window=50).mean()
        df["SMA200"] = df["Close"].rolling(window=200).mean()

        st.write(f"Displaying full historical stock data for **{ticker}**")
        st.dataframe(df.tail(5))

        # Plot Closing Price + SMA50 + SMA200
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(df.index, df["Close"], label="Close Price", color="blue", alpha=0.6)
        ax.plot(df.index, df["SMA50"], label="SMA50", color="orange")
        ax.plot(df.index, df["SMA200"], label="SMA200", color="green")

        ax.set_title(f"{ticker} Closing Price with SMA50 & SMA200")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.grid(True)
        ax.legend()

        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error fetching stock data: {e}")

# --- Streamlit UI ---
st.title("Stock Price Viewer")
company = st.text_input("Enter a company name (e.g., Tesla, Apple, FPT):")

if company:
    ticker = company_to_ticker(company)
    
    if ticker:
        fetch_and_plot_stock(ticker)
    else:
        st.error("Company not found. Please try a different name.")
