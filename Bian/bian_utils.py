import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys

# Local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Bian.backend_bian import BackendBian
from Fian.frontend_fian import FrontendFian
bian = BackendBian()
fian = FrontendFian()
from Bian.configs import indicator_plot_config


# import libs
import yfinance as yf
import re
import json



def extract_data_yf(tickers, Period="1y", temp_file=False):  # Backend 
    # Note: Remember to make a way to delete these files after use. Since they are only temporary files.
    data = {}
    for ticker in tickers:
        try:
            # Download data with progress=False to avoid output issues
            df = yf.download(ticker, period=Period, interval="1d", auto_adjust=True, progress=False)
            
            # Handle multi-level columns if they exist
            if isinstance(df.columns, pd.MultiIndex):
                # Flatten multi-level columns - take the first level
                df.columns = df.columns.get_level_values(0)
            
            # Ensure we have the required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                print(f"Warning: Missing columns {missing_columns} for {ticker}")
                continue
            
            # Remove any NaN rows
            df = df.dropna()
            
            if df.empty:
                print(f"Warning: No valid data for {ticker}")
                continue
                
            filename = f"temp_{ticker}_{Period}.csv"
            if temp_file:
                df.to_csv(filename)
            data[ticker] = df
            
        except Exception as e:
            print(f"Error downloading data for {ticker}: {str(e)}")
            continue
            
    return data

def preprocess_query(text): # SpaCy # Experimental
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return ' '.join(tokens)


def run_NER(text): # SpaCy # Experimental
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

def extract_entities(entities, label): # SpaCy # Experimental
    return [ent_text for ent_text, ent_label in entities if ent_label == label]

def to_yf_period(text): # SpaCy # Experimental
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

def load_resources():
    from Bian.resources import model #,nlp, stop_words, lemmatizer
    model = model
    # nlp = nlp
    # stop_words = stop_words
    # lemmatizer = lemmatizer

def check_valid_ticker(ticker): # Backend
    """Check if a ticker is valid by trying to download data for it"""
    try:
        test_extract = yf.download(ticker, period="1d", auto_adjust=True, progress=False)
        
        # Handle multi-level columns if they exist
        if isinstance(test_extract.columns, pd.MultiIndex):
            test_extract.columns = test_extract.columns.get_level_values(0)
        
        # Check if we have valid data and required columns
        if not test_extract.empty and 'Close' in test_extract.columns:
            return True  # Ticker is valid
        else:
            return False  # Ticker is invalid
    except Exception as e:
        print(f"Error validating ticker {ticker}: {str(e)}")
        return False  # Ticker is invalid

def format_currency(value, currency_code, ticker_symbol=None): # Backend
    """Format currency based on the currency code"""
    if value is None or pd.isna(value):
        return "N/A"
    
    # Convert to float if it's a pandas Series or other numeric type
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "N/A"
    
    # Currency mapping
    currency_map = {
        'USD': ('$', 2),
        'VND': ('₫', 0),  # Vietnamese Dong - no decimals
        'EUR': ('€', 2),
        'GBP': ('£', 2),
        'JPY': ('¥', 0),  # Japanese Yen - no decimals
        'KRW': ('₩', 0),  # Korean Won - no decimals
        'CNY': ('¥', 2),  # Chinese Yuan
        'INR': ('₹', 2),  # Indian Rupee
        'CAD': ('C$', 2),
        'AUD': ('A$', 2),
        'HKD': ('HK$', 2),
        'SGD': ('S$', 2),
        'THB': ('฿', 2),
    }
    
    # Get currency symbol and decimal places
    symbol, decimals = currency_map.get(currency_code, ('', 2))
    
    # Format based on currency
    if currency_code == 'VND':
        # Vietnamese Dong - use thousands separators
        return f"{symbol}{value:,.0f}"
    elif currency_code in ['JPY', 'KRW']:
        # No decimals for these currencies
        return f"{symbol}{value:,.0f}"
    else:
        # Standard formatting with decimals
        return f"{symbol}{value:,.{decimals}f}"

def get_currency_info(info, ticker_symbol): # Backend
    """Get currency information from ticker info"""
    # Try to get currency from info
    currency = info.get('currency', 'USD')
    
    # Fallback: detect from ticker symbol
    if not currency or currency == 'USD':
        if '.VN' in ticker_symbol:
            currency = 'VND'
        elif '.TO' in ticker_symbol or '.TRT' in ticker_symbol:
            currency = 'CAD'
        elif '.L' in ticker_symbol:
            currency = 'GBP'
        elif '.T' in ticker_symbol:
            currency = 'JPY'
        elif '.KS' in ticker_symbol:
            currency = 'KRW'
        elif '.HK' in ticker_symbol:
            currency = 'HKD'
        elif '.SI' in ticker_symbol:
            currency = 'SGD'
        elif '.BK' in ticker_symbol:
            currency = 'THB'
        # Add more ticker suffix mappings as needed
    
    return currency

def get_currency_symbol(currency_code): # Backend
    """Get just the currency symbol for chart labels"""
    currency_symbols = {
        'USD': '$', 'VND': '₫', 'EUR': '€', 'GBP': '£', 'JPY': '¥',
        'KRW': '₩', 'CNY': '¥', 'INR': '₹', 'CAD': 'C$', 'AUD': 'A$',
        'HKD': 'HK$', 'SGD': 'S$', 'THB': '฿'
    }
    return currency_symbols.get(currency_code, currency_code)
    
def handle_input_type(input_text): # For Chatbot
    if input_text[0] == "/":
        return "command"
    return "chat"

def run_command(command): # For Chatbot
    # Import MemoryMian for activity logging
    from Mian.memory_mian import MemoryMian
    mian = MemoryMian()
    
    if command.startswith("/help"):
        mian.log_activity("❓ Viewed help commands")
        return (
            "Available commands:\n"
            "/help - Show this help message\n"
            "/calculate <indicator> <ticker> - Calculate technical indicators (e.g., /calculate RSI AAPL, /calculate SMA_20 MSFT)\n"
            "  • Supported indicators: RSI, MACD, SMA_20, EMA_50, ATR, MFI, Bollinger_hband, and many more\n"
            "/compare <ticker1> <ticker2> - Compare two stocks side by side (e.g., /compare AAPL MSFT)\n"
            "/predict <ticker> - Predict stock price (e.g., /predict TSLA)\n"
            "/display <ticker> - Display candlestick chart, volume, and company info (e.g., /display GOOGL)<br>"
        )
    elif command.startswith("/compare"):
        try:
            # Parse tickers
            parts = command.strip().split()
            if len(parts) < 3:
                return "Usage: /compare <ticker1> <ticker2>"
            ticker1, ticker2 = parts[1].upper(), parts[2].upper()
            
            # Validate tickers
            if not check_valid_ticker(ticker1):
                return f"Invalid ticker: {ticker1}"
            if not check_valid_ticker(ticker2):
                return f"Invalid ticker: {ticker2}"
            
            # Fetch data using the standalone function
            data = extract_data_yf([ticker1, ticker2], Period="6mo")
            if ticker1 not in data or ticker2 not in data or data[ticker1].empty or data[ticker2].empty:
                return f"Could not fetch data for {ticker1} or {ticker2}."
            
            # Log the activity
            mian.log_activity(f"📊 Compared {ticker1} vs {ticker2}")
            
            # Use Fian for visualization
            fian.compare_stocks(data[ticker1], data[ticker2], title=f"{ticker1} vs {ticker2} Comparison")
            return f"📊 Comparison chart displayed for {ticker1} vs {ticker2}"
        except Exception as e:
            return f"Error comparing stocks: {str(e)}"
    elif command.startswith("/calculate"):
        try:
            # Parse indicator and ticker
            parts = command.strip().split()
            if len(parts) < 3:
                return "Usage: /calculate <indicator> <ticker>\nExample: /calculate RSI AAPL\nAvailable indicators: RSI, MACD, SMA_20, EMA_50, Bollinger_hband, ATR, MFI, etc."
            
            indicator_name = parts[1].upper()
            ticker = parts[2].upper()
            
            # Import the indicators list and functions
            from Bian.configs import indicators_list, indicator_funcs, indicator_plot_config
            
            # Check if indicator is supported
            if indicator_name not in indicators_list:
                # Categorize indicators based on the configs structure
                trend_indicators = [name for name in indicators_list if any(x in name for x in ["MACD", "ADX", "CCI", "Ichimoku", "PSAR", "STC", "SMA", "EMA", "WMA", "DEMA", "TEMA"])][:8]
                momentum_indicators = [name for name in indicators_list if any(x in name for x in ["RSI", "Stoch", "AwesomeOsc", "KAMA", "ROC", "TSI", "UO", "MFI"])]
                volatility_indicators = [name for name in indicators_list if any(x in name for x in ["ATR", "Bollinger", "Donchian", "Keltner"])]
                
                return (f"Indicator '{indicator_name}' not supported.\n"
                       f"📈 Trend: {', '.join(trend_indicators)}\n"
                       f"⚡ Momentum: {', '.join(momentum_indicators)}\n"
                       f"📊 Volatility: {', '.join(volatility_indicators)}\n"
                       f"💡 Tip: Try /calculate RSI AAPL or /calculate SMA_20 MSFT")
            
            # Validate ticker
            if not check_valid_ticker(ticker):
                return f"Invalid ticker: {ticker}"
            
            # Fetch data for calculation (need enough data for indicators)
            data = extract_data_yf([ticker], Period="1y")
            if ticker not in data or data[ticker].empty:
                return f"Could not fetch data for {ticker}."
            
            ticker_data = data[ticker]
            if len(ticker_data) < 50:  # Need sufficient data for indicators
                return f"Insufficient data for {ticker} to calculate {indicator_name}"
            
            # Calculate the indicator
            if indicator_name not in indicator_funcs:
                return f"Calculation function not available for {indicator_name}"
            
            try:
                # Calculate indicator values
                indicator_values = indicator_funcs[indicator_name](ticker_data)
                
                # Get the latest value
                latest_value = indicator_values.dropna().iloc[-1] if not indicator_values.dropna().empty else None
                
                if latest_value is None:
                    return f"Could not calculate {indicator_name} for {ticker} - insufficient data or calculation error"
                
                # Add indicator to dataframe for visualization
                ticker_data[indicator_name] = indicator_values
                
                # Log the activity
                mian.log_activity(f"🧮 Calculated {indicator_name} for {ticker}")
                
                # Create visualization using Fian
                data_dict = {ticker: ticker_data}
                fian.visualize_indicator(data_dict, indicator_name, title=f"{ticker} - {indicator_name} Analysis")
                
                # Get indicator configuration for smarter interpretation
                config = indicator_plot_config.get(indicator_name, {})
                guides = config.get("guides", [])
                plot_type = config.get("type", "line")
                paired_indicator = config.get("paired", None)
                
                # Add paired indicator info if available
                paired_info = ""
                if paired_indicator and paired_indicator in ticker_data.columns:
                    paired_value = ticker_data[paired_indicator].dropna().iloc[-1]
                    paired_info = f"\n🔗 {paired_indicator}: {paired_value:.2f}"
                
                # Format the latest value with intelligent interpretation based on config
                if guides:  # Use guides from config for interpretation
                    formatted_value = f"{latest_value:.2f}"
                    interpretation = ""
                    
                    # Smart interpretation based on guide levels
                    if len(guides) == 1:  # Single guide (usually zero line)
                        if latest_value > guides[0]:
                            interpretation = f" (Above {guides[0]} - positive signal)"
                        else:
                            interpretation = f" (Below {guides[0]} - negative signal)"
                    elif len(guides) == 2:  # Two guides (usually overbought/oversold)
                        lower_guide, upper_guide = sorted(guides)
                        if latest_value > upper_guide:
                            interpretation = f" (Above {upper_guide} - overbought zone)"
                        elif latest_value < lower_guide:
                            interpretation = f" (Below {lower_guide} - oversold zone)"
                        else:
                            interpretation = f" (Between {lower_guide}-{upper_guide} - neutral zone)"
                    
                    # Add specific interpretations for well-known indicators
                    if indicator_name == "RSI":
                        if latest_value > 70:
                            interpretation = " (Overbought - consider selling)"
                        elif latest_value < 30:
                            interpretation = " (Oversold - consider buying)"
                        else:
                            interpretation = " (Neutral zone)"
                    elif indicator_name == "MFI":
                        if latest_value > 80:
                            interpretation = " (Overbought - high selling pressure)"
                        elif latest_value < 20:
                            interpretation = " (Oversold - high buying opportunity)"
                        else:
                            interpretation = " (Neutral money flow)"
                    elif indicator_name in ["Stoch", "Stoch_signal"]:
                        if latest_value > 80:
                            interpretation = " (Overbought - potential reversal)"
                        elif latest_value < 20:
                            interpretation = " (Oversold - potential bounce)"
                        else:
                            interpretation = " (Neutral momentum)"
                    
                    return f"📊 {indicator_name} for {ticker}: {formatted_value}{interpretation}{paired_info}"
                    
                elif "SMA" in indicator_name or "EMA" in indicator_name or "WMA" in indicator_name or "DEMA" in indicator_name or "TEMA" in indicator_name:
                    # Moving averages - compare with current price and get currency info
                    current_price = float(ticker_data['Close'].iloc[-1])
                    
                    # Get currency information for proper formatting
                    try:
                        import yfinance as yf
                        stock = yf.Ticker(ticker)
                        info = stock.info
                        currency = get_currency_info(info, ticker)
                        
                        formatted_value = format_currency(latest_value, currency)
                        formatted_current = format_currency(current_price, currency)
                    except:
                        formatted_value = f"${latest_value:.2f}"
                        formatted_current = f"${current_price:.2f}"
                    
                    price_vs_ma = "above" if current_price > latest_value else "below"
                    percentage_diff = abs((current_price - latest_value) / latest_value * 100)
                    
                    trend_signal = "📈 Bullish trend" if current_price > latest_value else "📉 Bearish trend"
                    
                    return f"📊 {indicator_name} for {ticker}: {formatted_value}\nCurrent price ({formatted_current}) is {price_vs_ma} the {indicator_name} by {percentage_diff:.2f}%\n{trend_signal}{paired_info}"
                    
                else:
                    # General indicators - format based on typical ranges
                    if latest_value > 1000:
                        formatted_value = f"{latest_value:,.0f}"
                    elif latest_value > 10:
                        formatted_value = f"{latest_value:.2f}"
                    else:
                        formatted_value = f"{latest_value:.4f}"
                    
                    # Add context based on plot type
                    if plot_type == "histogram":
                        context = " (Histogram indicator - check chart for trend)"
                    elif config.get("subplot", False):
                        context = " (Oscillator - check against historical levels)"
                    else:
                        context = " (Price overlay indicator)"
                    
                    return f"📊 {indicator_name} for {ticker}: {formatted_value}{context}{paired_info}"
                
            except Exception as calc_error:
                return f"Error calculating {indicator_name} for {ticker}: {str(calc_error)}"
            
        except Exception as e:
            return f"Error processing calculate command: {str(e)}"
    elif command.startswith("/predict"):
        mian.log_activity("🔮 Used predict command")
        return "Predict command placeholder."
    elif command.startswith("/display"):
        try:
            # Parse ticker
            parts = command.strip().split()
            if len(parts) < 2:
                return "Usage: /display <ticker>"
            ticker = parts[1].upper()
            
            # Validate ticker
            if not check_valid_ticker(ticker):
                return f"Invalid ticker: {ticker}"
            
            # Fetch data with enhanced error handling
            data = extract_data_yf([ticker], Period="1y")
            if ticker not in data or data[ticker].empty:
                return f"Could not fetch data for {ticker}."
            
            # Ensure data is properly formatted
            ticker_data = data[ticker]
            if ticker_data.empty:
                return f"No data available for {ticker}"
            
            # Log the activity
            mian.log_activity(f"📈 Displayed detailed analysis for {ticker}")
            
            # Use Fian for visualization
            fian.display_stock(ticker_data, ticker)
            return f"📈 Stock information and chart displayed for {ticker}"
        except Exception as e:
            import traceback
            import gc
            
            # Clean up memory on error
            try:
                import matplotlib.pyplot as plt
                plt.close('all')
                gc.collect()
            except:
                pass
            
            error_details = traceback.format_exc()
            print(f"Full error traceback: {error_details}")
            
            # Handle specific error types
            if "bad allocation" in str(e).lower():
                return f"Memory error displaying {ticker}. Try again or contact support if this persists."
            elif "no data" in str(e).lower():
                return f"No data available for {ticker}. Please check the ticker symbol."
            else:
                return f"Error displaying stock: {str(e)}"
    else:
        return "Unknown command. Type /help for available commands."

def run_query(query): # For Chatbot
    """Run a query using the generative model with a system prompt for persona and instructions"""
    from Bian.resources import model
    from Mian.memory_mian import MemoryMian
    
    # Log the chat activity
    mian = MemoryMian()
    mian.log_activity("💬 Asked Trian a question")
    
    system_prompt = (
        "You are Trian, a lovely stock market assistant. "
        "You provide basic knowledge about things like technical indicators and news. "
        "If the user wants to use commands, kindly remind them to use /help for a list of available commands."
    )
    prompt = f"{system_prompt}\nUser query: {query}"
    response = model.generate_content(prompt)
    return response.text

def process_query(query):
    in_type = handle_input_type(query)
    if in_type == "command":
        response = run_command(query)
    elif in_type == "chat":
        response = run_query(query)
    return response



if __name__ == "__main__":
    # Example usage
    pass