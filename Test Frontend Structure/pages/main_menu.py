import streamlit as st
import os
import sys 

# Local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Mian.memory_mian import MemoryMian
mian = MemoryMian()

# Function to check Gemini API status
def check_gemini_status():
    """Check if Gemini API is working properly"""
    try:
        from Bian.resources import model
        # Try a simple test query
        response = model.generate_content("test")
        return "✅ Online", "Connected"
    except Exception as e:
        return "❌ Offline", "Error"

# Function to check market status
def check_market_status():
    """Check if US markets are currently open"""
    from datetime import datetime, time
    import pytz
    
    try:
        # Get current time in US Eastern timezone
        eastern = pytz.timezone('US/Eastern')
        now = datetime.now(eastern)
        current_time = now.time()
        current_day = now.weekday()  # 0=Monday, 6=Sunday
        
        # Market hours: 9:30 AM to 4:00 PM ET, Monday to Friday
        market_open = time(9, 30)
        market_close = time(16, 0)
        
        # Check if it's a weekday and within market hours
        if current_day < 5 and market_open <= current_time <= market_close:
            return "🟢 Open", "Trading"
        else:
            return "🔴 Closed", "After Hours"
    except Exception as e:
        return "❓ Unknown", "Error"

# Function to get real ticker data with indicators
def get_ticker_metrics(ticker):
    """Get current price, MFI, and SMA20 for a ticker"""
    try:
        # Import backend functions
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        from Bian.backend_bian import BackendBian
        
        bian = BackendBian()
        
        # Fetch data with indicators
        data = bian.calculate_indicators([ticker], period="3mo", indicators=["MFI", "SMA_20"])
        
        if ticker not in data or data[ticker].empty:
            return {"price": "N/A", "change_pct": "N/A", "mfi": "N/A", "sma20": "N/A"}
        
        ticker_data = data[ticker]
        
        # Get latest values
        latest_price = float(ticker_data['Close'].iloc[-1])
        
        # Calculate daily change
        if len(ticker_data) >= 2:
            prev_price = float(ticker_data['Close'].iloc[-2])
            change_pct = ((latest_price - prev_price) / prev_price) * 100
        else:
            change_pct = 0
        
        # Get MFI and SMA20
        mfi = ticker_data['MFI'].iloc[-1] if 'MFI' in ticker_data.columns else None
        sma20 = ticker_data['SMA_20'].iloc[-1] if 'SMA_20' in ticker_data.columns else None
        
        return {
            "price": f"${latest_price:.2f}",
            "change_pct": f"{change_pct:+.1f}%",
            "mfi": f"{mfi:.1f}" if mfi is not None else "N/A",
            "sma20": f"${sma20:.2f}" if sma20 is not None else "N/A"
        }
        
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return {"price": "N/A", "change_pct": "N/A", "mfi": "N/A", "sma20": "N/A"}

st.set_page_config(
    page_title="Main Menu - Finance Assistant",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 Finance Assistant Dashboard")
st.write("Welcome to your personal finance assistant!")

# Initialize tracking tickers from memory if not in session state
if "tracking_tickers" not in st.session_state:
    try:
        saved_tickers = mian.load_tracking_tickers()
        st.session_state.tracking_tickers = saved_tickers if saved_tickers else []
    except:
        st.session_state.tracking_tickers = []

# Quick stats/overview
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="📈 Tracked Tickers",
        value=len(st.session_state.get("tracking_tickers", [])),
        delta="Tracked"
    )

with col2:
    # Check Gemini API status
    gemini_status, gemini_delta = check_gemini_status()
    st.metric(
        label="🤖 Gemini API Status",
        value=gemini_status,
        delta=gemini_delta
    )

with col3:
    # Check market status
    market_status, market_delta = check_market_status()
    st.metric(
        label="� Market Status",
        value=market_status,
        delta=market_delta
    )

# Quick actions
st.header("Quick Actions")
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("💬 Start Chat", use_container_width=True):
        st.switch_page("pages/chatbot.py")

with col2:
    if st.button("⚙️ Settings", use_container_width=True):
        st.switch_page("pages/settings.py")

with col3:
    if st.button("🔮 Prediction Assistant", use_container_width=True):
        st.switch_page("pages/prediction_assistant.py")

with col4:
    if st.button("📈 Ticker Search", use_container_width=True):
        st.switch_page("pages/ticker_search.py")

# Quick Add Ticker Section
st.header("📈 Quick Add Ticker")
with st.container():
    col1, col2 = st.columns([3, 1])
    with col1:
        new_ticker = st.text_input("Add ticker to your watchlist", placeholder="e.g., AAPL, GOOGL, MSFT", key="quick_add_ticker")
    with col2:
        st.write("")  # Add spacing
        if st.button("➕ Add", use_container_width=True, key="quick_add_btn"):
            if new_ticker:
                # Initialize tracking_tickers from memory if not in session
                if "tracking_tickers" not in st.session_state:
                    try:
                        saved_tickers = mian.load_tracking_tickers()
                        st.session_state.tracking_tickers = saved_tickers if saved_tickers else []
                    except:
                        st.session_state.tracking_tickers = []
                
                ticker_upper = new_ticker.upper()
                if ticker_upper not in st.session_state.tracking_tickers:
                    # Add to session state
                    st.session_state.tracking_tickers.append(ticker_upper)
                    # Save to persistent memory
                    try:
                        mian.save_tracking_tickers(st.session_state.tracking_tickers)
                        st.success(f"✅ Added {ticker_upper} to your watchlist!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error saving ticker: {e}")
                else:
                    st.warning(f"⚠️ {ticker_upper} is already in your watchlist!")
            else:
                st.error("Please enter a ticker symbol!")


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
st.info("💡 **Navigation**: Use the sidebar to switch between different sections of the app!")