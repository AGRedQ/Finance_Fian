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
        df = stock.history(period="6mo")

        if df.empty:
            st.error("No data found for ticker.")
            return

        st.write(f"Displaying stock data for {ticker}")
        st.write(df.tail(5))

        # Plot stock data
        fig, ax = plt.subplots()
        df["Close"].plot(ax=ax, title=f"{ticker} Stock Closing Prices (6 months)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Close Price")
        ax.grid(True)
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
