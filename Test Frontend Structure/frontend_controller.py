import streamlit as st
import os
import sys
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Mian.memory_mian import MemoryMian
mian =  MemoryMian()
from Bian.backend_bian import BackendBian
bian = BackendBian()

# Extract existing user settings (Note: Defaults are my settings)

st.set_page_config(
    page_title="Finance Assistant",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load resources


with st.spinner("Loading resources..."):
    bian.load_resources()
st.success("Resources loaded successfully!")

# Load tracking tickers
with st.spinner("Loading tracked tickers..."):
    tracking_tickers_data = mian.load_tracking_tickers()
st.success("Tracked tickers loaded successfully!")

# Load user settings
loaded_user_settings = mian.load_user_settings()

if loaded_user_settings == "load_user_settings_error":
    st.error("Error loading user settings. Please check the configuration file.")
else:
    # Initialize session state from loaded_user_settings
    for section, section_settings in loaded_user_settings.items():
        if isinstance(section_settings, dict):
            for key, value in section_settings.items():
                if key not in st.session_state:
                    st.session_state[key] = value
        else:
            # For flat settings (if any)
            if section not in st.session_state:
                st.session_state[section] = section_settings

# Initialize tracking tickers in session state
if "tracking_tickers" not in st.session_state:
    st.session_state.tracking_tickers = list(tracking_tickers_data.keys())

# Main page content - Landing/Home page
st.title("💰 Finance Assistant")
st.markdown("### Welcome to your AI-powered financial analysis tool!")

# Quick navigation
st.markdown("---")
st.subheader("🚀 Quick Navigation")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    **🏠 Main Menu**
    - Dashboard overview
    - Quick stats
    - Recent activity
    """)
    if st.button("Go to Main Menu", key="nav_main"):
        st.switch_page("pages/main_menu.py")

with col2:
    st.markdown("""
    **💬 Chatbot**
    - Ask questions
    - Get analysis
    - Interactive help
    """)
    if st.button("Start Chatting", key="nav_chat"):
        st.switch_page("pages/chatbot.py")

with col3:
    st.markdown("""
    **⚙️ Settings**
    - Configure preferences
    - Manage tickers
    - Model training
    """)
    if st.button("Open Settings", key="nav_settings"):
        st.switch_page("pages/settings.py")
with col4:
    st.markdown("""
    **⚙️ Ticker Search**
    - Comprehensive information
    - Visualizations
    - Basic analysis
    """)
    if st.button("Open Ticker Search", key="nav_ticker"):
        st.switch_page("pages/ticker_search.py")
# Current status overview
st.markdown("---")
st.subheader("📊 Current Status")

col1, col2, col3 = st.columns(3)
with col1:
    st.info(f"📈 **Tracked Tickers**: {len(st.session_state.get('tracking_tickers', []))}")
with col2:
    st.info(f"📊 **Visualizations**: {'ON' if st.session_state.get('visualize', True) else 'OFF'}")
with col3:
    st.info(f"🤖 **Auto Training**: {'ON' if st.session_state.get('auto_train', False) else 'OFF'}")

# Show tracked tickers with price changes
if st.session_state.get('tracking_tickers'):
    st.subheader("📈 Your Tracked Tickers")
    
    # Display up to 4 tickers with price changes
    display_tickers = st.session_state.tracking_tickers[:4]
    if display_tickers:
        cols = st.columns(len(display_tickers))
        
        for i, ticker in enumerate(display_tickers):
            with cols[i]:
                ticker_data = mian.get_ticker_data(ticker)
                if ticker_data:
                    indicators = mian.get_ticker_indicators(ticker)
                    current_price = indicators.get("price")
                    mfi = indicators.get("mfi")
                    sma_20 = indicators.get("sma_20")
                    
                    if current_price and mfi is not None and sma_20 is not None:
                        # Display current indicators without comparisons
                        price_vs_sma = "📈" if current_price > sma_20 else "📉"
                        
                        if mfi > 80:
                            st.error(f"**{ticker}**\n{price_vs_sma} SMA20: ${sma_20:.2f}\n📊 MFI: {mfi:.1f}")
                        elif mfi < 20:
                            st.success(f"**{ticker}**\n{price_vs_sma} SMA20: ${sma_20:.2f}\n📊 MFI: {mfi:.1f}")
                        else:
                            st.info(f"**{ticker}**\n{price_vs_sma} SMA20: ${sma_20:.2f}\n📊 MFI: {mfi:.1f}")
                    else:
                        st.warning(f"**{ticker}**\n❌ Data unavailable")
                else:
                    st.info(f"**{ticker}**\n🔄 Loading...")
    
    if len(st.session_state.tracking_tickers) > 4:
        st.caption(f"... and {len(st.session_state.tracking_tickers) - 4} more tickers")

# Getting started guide
with st.expander("🎯 Getting Started Guide"):
    st.markdown("""
    1. **📈 Add Tickers**: Go to Settings → Tickers to add stocks you want to track
    2. **⚙️ Configure**: Set your visualization and training preferences in Settings
    3. **💬 Ask Questions**: Use the Chatbot to ask about stock prices, indicators, and predictions
    4. **📊 View Dashboard**: Check the Main Menu for overview and quick stats
    """)

# Footer
st.markdown("---")
st.caption("🔗 Use the sidebar navigation to switch between different sections of the app.")