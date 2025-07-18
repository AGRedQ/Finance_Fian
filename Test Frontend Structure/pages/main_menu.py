import streamlit as st
import os
import sys 

# Local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Mian.memory_mian import MemoryMian
mian = MemoryMian()

st.set_page_config(
    page_title="Main Menu - Finance Assistant",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 Finance Assistant Dashboard")
st.write("Welcome to your personal finance assistant!")

# Quick stats/overview
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="📈 Tracked Tickers",
        value=len(st.session_state.get("tracking_tickers", [])),
        delta="Tracked"
    )

with col2:
    st.metric(
        label="🤖 Models Status",
        value=f"3/3 trained",
        delta="Online"
    )

with col3:
    st.metric(
        label="📊 Visualizations",
        value="ON" if st.session_state.get("visualize", True) else "OFF",
        delta="Ready"
    )

with col4:
    st.metric(
        label="🔄 Auto Train",
        value="ON" if st.session_state.get("auto_train", False) else "OFF",
        delta="Scheduled" if st.session_state.get("auto_train", False) else "Manual"
    )

# Quick actions
st.header("Quick Actions")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("💬 Start Chat", use_container_width=True):
        st.switch_page("pages/chatbot.py")

with col2:
    if st.button("⚙️ Settings", use_container_width=True):
        st.switch_page("pages/settings.py")

with col3:
    if st.button("📈 Add Ticker", use_container_width=True):
        with st.form("quick_add_ticker"):
            ticker = st.text_input("Enter ticker symbol:")
            if st.form_submit_button("Add"):
                if ticker:
                    if "tracking_tickers" not in st.session_state:
                        st.session_state.tracking_tickers = []
                    if ticker.upper() not in st.session_state.tracking_tickers:
                        st.session_state.tracking_tickers.append(ticker.upper())
                        st.success(f"Added {ticker.upper()}!")
                    else:
                        st.warning(f"{ticker.upper()} already tracked!")

# Current tracking tickers overview
if st.session_state.get("tracking_tickers"):
    st.header("📊 Your Tracked Tickers")
    ticker_cols = st.columns(min(len(st.session_state.tracking_tickers), 4))
    
    for i, ticker in enumerate(st.session_state.tracking_tickers[:4]):  # Show max 4
        with ticker_cols[i % 4]:
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
                        st.error(f"**{ticker}**\n{price_vs_sma} SMA20: ${sma_20:.2f}\n� MFI: {mfi:.1f}")
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

# Recent activity (mock)
st.header("📋 Recent Activity")
activities = [
    "Added AAPL to tracking list",
    "Trained price prediction model",
    "Generated RSI analysis for GOOGL",
    "Updated visualization settings"
]

for activity in activities[:3]:  # Show last 3
    st.write(f"• {activity}")

# Navigation helper
st.markdown("---")
st.info("💡 **Navigation**: Use the sidebar to switch between different sections of the app")