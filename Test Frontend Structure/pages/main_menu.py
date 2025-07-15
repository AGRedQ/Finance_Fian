import streamlit as st

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
        delta="Active"
    )

with col2:
    st.metric(
        label="🤖 Models Status",
        value="3/3",
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
            st.info(f"**{ticker}**\n📈 $123.45\n📊 +2.3%")  # Mock data
    
    if len(st.session_state.tracking_tickers) > 4:
        st.write(f"... and {len(st.session_state.tracking_tickers) - 4} more")

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