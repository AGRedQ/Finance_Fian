import requests
import yfinance as yf
import matplotlib.pyplot as plt

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
        print(f"Error while fetching ticker: {e}")
        return None


def data_preprocessing(df):

    print("Data Extracted Successfully")
    # Keep only the Close price
    close_prices = df[['Close']].copy()

    # Replace the Date index with day counters: 0, 1, 2, ...
    close_prices['Day'] = range(len(close_prices))

    # Reorder columns for clarity
    clean_df = close_prices[['Day', 'Close']]

    # Optional: Save to a new CSV
    clean_df.to_csv('cleaned_close_data.csv', index=False)

    print(clean_df.head())




def fetch_and_plot_stock(ticker):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="6mo")

        if df.empty:
            print("No data found for ticker.")
            return

        print(f"Displaying stock data for {ticker}")
        print(df.tail(5))

        df["Close"].plot(title=f"{ticker} Stock Closing Prices (6 months)")
        plt.xlabel("Date")
        plt.ylabel("Close Price")
        plt.grid(True)
        plt.show()
        return df
    except Exception as e:
        print(f"Error fetching stock data: {e}")

# --- Main program ---
company = input("Enter a company name (e.g., Tesla, Apple, FPT): ")
ticker = company_to_ticker(company)

if ticker:
    df = fetch_and_plot_stock(ticker)
    data_preprocessing(df)
else:
    print("Company not found. Please try a different name.")
