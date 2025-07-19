import streamlit as st
import json
import os
import sys
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle

# Ensure the Mian module is accessible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from Mian.memory_mian import MemoryMian
mian = MemoryMian()
from Bian.backend_bian import BackendBian
bian = BackendBian()
# =======================

st.set_page_config(
    page_title="Ticker Search - Finance Assistant",
    page_icon="🔍",
    layout="wide"
)

# Add a 'Return to Main Menu' button at the top
if st.button('⬅️ Return to Main Menu', key='return_main_menu'):
    st.switch_page('pages/main_menu.py')

st.title("🔍 Ticker Search")
st.markdown("Enter a ticker symbol to get comprehensive statistics and analysis")

# Create input section
col1, col2 = st.columns([3, 1])

with col1:
    ticker_input = st.text_input(
        "Enter Ticker Symbol", 
        placeholder="e.g., AAPL, MSFT, GOOGL",
        help="Enter a valid stock ticker symbol"
    )

with col2:
    search_button = st.button("🔍 Search", type="primary")

# Only proceed if we have a ticker input
if ticker_input and (search_button or ticker_input):
    ticker = ticker_input.upper().strip()
    
    # Validate ticker
    if not bian.check_valid_ticker(ticker):
        st.error(f"❌ Invalid ticker symbol: {ticker}")
        st.stop()
    
    # Display loading message
    with st.spinner(f"Fetching data for {ticker}..."):
        # Get ticker data
        ticker_obj = yf.Ticker(ticker)
        
        try:
            # Get basic info
            info = ticker_obj.info
            
            # Get historical data
            hist_1d = ticker_obj.history(period="1d")
            hist_30d = ticker_obj.history(period="30d")
            hist_1y = ticker_obj.history(period="1y")
            
            # Get technical indicators
            indicators = mian.get_ticker_indicators(ticker)
            
            # Get currency information
            currency = bian.get_currency_info(info, ticker)
            
            # Create main layout
            st.header(f"📊 {ticker} - {info.get('longName', 'N/A')}")
            
            # Basic Info Cards
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                current_price = indicators.get('price', 0)
                st.metric(
                    label="💰 Current Price",
                    value=bian.format_currency(current_price, currency) if current_price else "N/A",
                    delta=f"{((current_price - hist_30d['Close'].iloc[0]) / hist_30d['Close'].iloc[0] * 100):.2f}%" if current_price and len(hist_30d) > 0 else None
                )
            
            with col2:
                mfi_value = indicators.get('mfi')
                mfi_color = "normal"
                if mfi_value:
                    if mfi_value > 80:
                        mfi_color = "inverse"  # Red background
                    elif mfi_value < 20:
                        mfi_color = "off"  # Green background
                
                st.metric(
                    label="📈 MFI (14)",
                    value=f"{mfi_value:.2f}" if mfi_value else "N/A",
                    help="Money Flow Index - Overbought >80, Oversold <20"
                )
            
            with col3:
                sma_20 = indicators.get('sma_20')
                sma_trend = None
                if sma_20 and current_price:
                    sma_trend = "📈" if current_price > sma_20 else "📉"
                
                st.metric(
                    label="📊 SMA 20",
                    value=bian.format_currency(sma_20, currency) if sma_20 else "N/A",
                    delta=sma_trend,
                    help="20-day Simple Moving Average"
                )
            
            with col4:
                market_cap = info.get('marketCap')
                if market_cap:
                    if market_cap >= 1e12:
                        market_cap_str = f"${market_cap/1e12:.2f}T"
                    elif market_cap >= 1e9:
                        market_cap_str = f"${market_cap/1e9:.2f}B"
                    elif market_cap >= 1e6:
                        market_cap_str = f"${market_cap/1e6:.2f}M"
                    else:
                        market_cap_str = f"${market_cap:,.0f}"
                else:
                    market_cap_str = "N/A"
                
                st.metric(
                    label="🏢 Market Cap",
                    value=market_cap_str
                )
            
            # Technical Analysis Section
            st.subheader("📈 Technical Analysis")
            
            # MFI Analysis
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Money Flow Index (MFI) Analysis:**")
                if mfi_value:
                    if mfi_value > 80:
                        st.error(f"🔴 **Overbought Signal** - MFI: {mfi_value:.2f}")
                        st.markdown("The stock may be overvalued and due for a correction.")
                    elif mfi_value < 20:
                        st.success(f"🟢 **Oversold Signal** - MFI: {mfi_value:.2f}")
                        st.markdown("The stock may be undervalued and due for a rebound.")
                    else:
                        st.info(f"🔵 **Neutral** - MFI: {mfi_value:.2f}")
                        st.markdown("The stock is in a neutral zone.")
                else:
                    st.warning("MFI data not available")
            
            with col2:
                st.markdown("**SMA 20 Analysis:**")
                if sma_20 and current_price:
                    difference = current_price - sma_20
                    percentage_diff = (difference / sma_20) * 100
                    
                    if current_price > sma_20:
                        st.success(f"📈 **Above SMA 20** - {bian.format_currency(difference, currency)} ({percentage_diff:.2f}%)")
                        st.markdown("Price is above the 20-day average, indicating upward momentum.")
                    else:
                        st.error(f"📉 **Below SMA 20** - {bian.format_currency(abs(difference), currency)} ({abs(percentage_diff):.2f}%)")
                        st.markdown("Price is below the 20-day average, indicating downward momentum.")
                else:
                    st.warning("SMA 20 data not available")
            
            # Company Information
            st.subheader("🏢 Company Information")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if info.get('sector'):
                    st.markdown(f"**Sector:** {info.get('sector')}")
                if info.get('industry'):
                    st.markdown(f"**Industry:** {info.get('industry')}")
                if info.get('country'):
                    st.markdown(f"**Country:** {info.get('country')}")
                if info.get('employees'):
                    st.markdown(f"**Employees:** {info.get('employees'):,}")
            
            with col2:
                if info.get('website'):
                    st.markdown(f"**Website:** [{info.get('website')}]({info.get('website')})")
                if info.get('dividendYield'):
                    st.markdown(f"**Dividend Yield:** {info.get('dividendYield')*100:.2f}%")
                if info.get('beta'):
                    st.markdown(f"**Beta:** {info.get('beta'):.2f}")
                if info.get('trailingPE'):
                    st.markdown(f"**P/E Ratio:** {info.get('trailingPE'):.2f}")
            
            # Company Description
            if info.get('longBusinessSummary'):
                st.markdown("**Business Summary:**")
                st.write(info.get('longBusinessSummary'))
            
            # Price Chart
            st.subheader("📊 Price Chart")
            
            # Chart time period selector
            chart_period = st.selectbox(
                "Select Time Period",
                ["1D", "5D", "1M", "3M", "6M", "1Y", "2Y", "5Y"],
                index=3
            )
            
            period_map = {
                "1D": "1d",
                "5D": "5d", 
                "1M": "1mo",
                "3M": "3mo",
                "6M": "6mo",
                "1Y": "1y",
                "2Y": "2y",
                "5Y": "5y"
            }
            
            chart_data = ticker_obj.history(period=period_map[chart_period])
            
            if not chart_data.empty:
                # Create matplotlib figure
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Plot candlestick data as OHLC line chart
                ax.plot(chart_data.index, chart_data['Close'], label=f'{ticker} Close Price', linewidth=2, color='blue')
                
                # Add SMA 20 if available
                if len(chart_data) >= 20:
                    sma_20_series = chart_data['Close'].rolling(window=20).mean()
                    ax.plot(chart_data.index, sma_20_series, label='SMA 20', linewidth=2, color='orange', linestyle='--')
                
                # Format the chart
                ax.set_title(f"{ticker} Price Chart ({chart_period})", fontsize=16, fontweight='bold')
                ax.set_ylabel(f"Price ({bian.get_currency_symbol(currency)})", fontsize=12)
                ax.set_xlabel("Date", fontsize=12)
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Format x-axis dates
                if len(chart_data) > 30:
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                    ax.xaxis.set_major_locator(mdates.MonthLocator())
                else:
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
                    ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(chart_data)//10)))
                
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                # Display in Streamlit
                st.pyplot(fig)
                plt.close(fig)  # Close figure to free memory
            
            # Financial Metrics
            st.subheader("💰 Financial Metrics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Valuation Metrics:**")
                if info.get('trailingPE'):
                    st.markdown(f"• P/E Ratio: {info.get('trailingPE'):.2f}")
                if info.get('forwardPE'):
                    st.markdown(f"• Forward P/E: {info.get('forwardPE'):.2f}")
                if info.get('priceToBook'):
                    st.markdown(f"• P/B Ratio: {info.get('priceToBook'):.2f}")
                if info.get('enterpriseToRevenue'):
                    st.markdown(f"• EV/Revenue: {info.get('enterpriseToRevenue'):.2f}")
            
            with col2:
                st.markdown("**Profitability:**")
                if info.get('profitMargins'):
                    st.markdown(f"• Profit Margin: {info.get('profitMargins')*100:.2f}%")
                if info.get('operatingMargins'):
                    st.markdown(f"• Operating Margin: {info.get('operatingMargins')*100:.2f}%")
                if info.get('returnOnEquity'):
                    st.markdown(f"• ROE: {info.get('returnOnEquity')*100:.2f}%")
                if info.get('returnOnAssets'):
                    st.markdown(f"• ROA: {info.get('returnOnAssets')*100:.2f}%")
            
            with col3:
                st.markdown("**Growth & Volatility:**")
                if info.get('earningsGrowth'):
                    st.markdown(f"• Earnings Growth: {info.get('earningsGrowth')*100:.2f}%")
                if info.get('revenueGrowth'):
                    st.markdown(f"• Revenue Growth: {info.get('revenueGrowth')*100:.2f}%")
                if info.get('beta'):
                    st.markdown(f"• Beta: {info.get('beta'):.2f}")
                if info.get('52WeekChange'):
                    st.markdown(f"• 52W Change: {info.get('52WeekChange')*100:.2f}%")
            
            # Add to tracking section
            st.subheader("⭐ Add to Tracking")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"Track **{ticker}** to monitor its MFI and SMA 20 indicators over time.")
            
            with col2:
                if st.button(f"➕ Add {ticker} to Tracking", key=f"add_{ticker}"):
                    try:
                        mian.add_tracking_ticker(ticker)
                        st.success(f"✅ {ticker} added to tracking!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error adding {ticker}: {str(e)}")
            
        except Exception as e:
            st.error(f"❌ Error fetching data for {ticker}: {str(e)}")
            st.info("Please check if the ticker symbol is correct and try again.")

else:
    # Show help information when no ticker is entered
    st.info("👆 Enter a ticker symbol above to get started!")
    
    st.markdown("""
    ### What you'll get:
    
    **📊 Real-time Data:**
    - Current stock price with 30-day change
    - Technical indicators (MFI, SMA 20)
    - Market capitalization
    
    **📈 Technical Analysis:**
    - Money Flow Index (MFI) signals for overbought/oversold conditions
    - Simple Moving Average (SMA 20) trend analysis
    - Interactive price charts with multiple time periods
    
    **🏢 Company Information:**
    - Business sector and industry
    - Company description and key metrics
    - Financial ratios and performance indicators
    
    **⭐ Additional Features:**
    - Add tickers to your tracking list
    - Historical price visualization
    - Comprehensive financial metrics
    
    ### Popular Tickers to Try:
    `AAPL` `MSFT` `GOOGL` `AMZN` `TSLA` `META` `NVDA` `NFLX`
    """)