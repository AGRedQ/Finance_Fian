import pandas as pd
import matplotlib.pyplot as plt
from Bian.configs import indicator_plot_config
import numpy as np



def compare_stocks(data1, data2, chart_width, chart_height, title=None): # Chatbot
    import streamlit as st
    
    if data1.empty or data2.empty:
        st.error("No data to compare.")
        return
    
    common_dates = data1.index.intersection(data2.index)
    if common_dates.empty:
        st.error("No overlapping dates to compare.")
        return

    # Get ticker names from the title or use generic names
    if title and " vs " in title:
        ticker1, ticker2 = title.split(" vs ")[0], title.split(" vs ")[1].split(" ")[0]
    else:
        ticker1, ticker2 = "Stock 1", "Stock 2"

    # Normalize both stocks to percentage change (starting at 100)
    stock1_prices = data1.loc[common_dates, 'Close']
    stock2_prices = data2.loc[common_dates, 'Close']
    
    # Calculate percentage change from first day (normalized to 100)
    stock1_normalized = (stock1_prices / stock1_prices.iloc[0]) * 100
    stock2_normalized = (stock2_prices / stock2_prices.iloc[0]) * 100

    fig, ax = plt.subplots(figsize=(chart_width, chart_height))
    ax.plot(common_dates, stock1_normalized, label=f'{ticker1} (Normalized)', linewidth=2, color='blue')
    ax.plot(common_dates, stock2_normalized, label=f'{ticker2} (Normalized)', linewidth=2, color='red')
    
    # Add horizontal line at 100 (starting point)
    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.7, label='Starting Point (100)')
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Normalized Price (%)')
    ax.set_title(title or 'Stock Comparison (Percentage Change)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add performance summary (ensure we get scalar values)
    final_perf1 = float(stock1_normalized.iloc[-1]) - 100
    final_perf2 = float(stock2_normalized.iloc[-1]) - 100
    
    # Add text box with performance summary
    textstr = f'{ticker1}: {final_perf1:+.1f}%\n{ticker2}: {final_perf2:+.1f}%'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # Display performance summary below the chart
    col1, col2 = st.columns(2)
    with col1:
        color1 = "green" if final_perf1 >= 0 else "red"
        st.markdown(f"**{ticker1} Performance:** <span style='color:{color1}'>{final_perf1:+.2f}%</span>", unsafe_allow_html=True)
    with col2:
        color2 = "green" if final_perf2 >= 0 else "red"
        st.markdown(f"**{ticker2} Performance:** <span style='color:{color2}'>{final_perf2:+.2f}%</span>", unsafe_allow_html=True)
    
    return



def visualize_indicator(data, indicator_name, chart_width = 12, chart_height = 6, title=None): # Chatbot
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

        plt.figure(figsize=(chart_width, chart_height))
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
        
        import streamlit as st
        st.pyplot(plt.gcf())
        plt.close()


def display_stock_info(data, ticker, chart_width=12, chart_height=6):
    """Display stock information with candlestick chart, volume, and general info"""
    import streamlit as st
    import yfinance as yf
    from datetime import datetime
    import sys
    import os
    
    # Import Bian functions
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from Bian.backend_bian import BackendBian
    bian = BackendBian()
    
    if data.empty:
        st.error(f"No data available for {ticker}")
        return
    
    # Get current stock info
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
    except:
        info = {}
    
    # Get currency info
    currency = bian.get_currency_info(info, ticker)
    
    # Display general information
    st.subheader(f"📊 {ticker} Stock Information")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        current_price = data['Close'].iloc[-1] if not data.empty else "N/A"
        st.metric("Current Price", bian.format_currency(current_price, currency))
    
    with col2:
        if len(data) >= 2:
            prev_close = data['Close'].iloc[-2]
            change = current_price - prev_close
            change_pct = (change / prev_close) * 100
            st.metric("Daily Change", bian.format_currency(change, currency), f"{change_pct:.2f}%")
        else:
            st.metric("Daily Change", "N/A")
    
    with col3:
        volume = data['Volume'].iloc[-1] if not data.empty else "N/A"
        st.metric("Volume", f"{volume:,.0f}" if volume != "N/A" else "N/A")
    
    with col4:
        market_cap = info.get('marketCap', 'N/A')
        if market_cap != 'N/A':
            st.metric("Market Cap", f"${market_cap/1e9:.2f}B")
        else:
            st.metric("Market Cap", "N/A")
    
    # Create subplots for candlestick and volume
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(chart_width, chart_height), 
                                   gridspec_kw={'height_ratios': [3, 1]})
    
    # Get currency symbol for chart
    currency_symbol = bian.get_currency_symbol(currency)
    
    # Candlestick chart (simplified as OHLC line chart)
    ax1.plot(data.index, data['Close'], label='Close', color='blue', linewidth=2)
    ax1.fill_between(data.index, data['Low'], data['High'], alpha=0.3, color='lightblue', label='High-Low Range')
    ax1.set_title(f"{ticker} Price Chart")
    ax1.set_ylabel(f"Price ({currency_symbol})")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Volume chart
    colors = ['red' if data['Close'].iloc[i] < data['Open'].iloc[i] else 'green' 
              for i in range(len(data))]
    ax2.bar(data.index, data['Volume'], color=colors, alpha=0.7)
    ax2.set_title("Volume")
    ax2.set_ylabel("Volume")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # Additional company information
    if info:
        st.subheader("📝 Company Information")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Company Name:** {info.get('longName', 'N/A')}")
            st.write(f"**Sector:** {info.get('sector', 'N/A')}")
            st.write(f"**Industry:** {info.get('industry', 'N/A')}")
        
        with col2:
            st.write(f"**P/E Ratio:** {info.get('trailingPE', 'N/A')}")
            high_52w = info.get('fiftyTwoWeekHigh')
            low_52w = info.get('fiftyTwoWeekLow')
            st.write(f"**52W High:** {bian.format_currency(high_52w, currency)}")
            st.write(f"**52W Low:** {bian.format_currency(low_52w, currency)}")
        
        # Business summary
        summary = info.get('longBusinessSummary', '')
        if summary:
            st.subheader("📄 Business Summary")
            st.write(summary[:500] + "..." if len(summary) > 500 else summary)