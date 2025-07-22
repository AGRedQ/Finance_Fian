import streamlit as st
import json
import os
import sys
# Ensure the Mian module is accessible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from Mian.memory_mian import MemoryMian
mian = MemoryMian()


st.set_page_config(
    page_title="Settings - Finance Assistant",
    page_icon="⚙️",
    layout="wide"
)


st.title("⚙️ Settings")

# Create tabs for different setting categories
tab1, tab2, tab3 = st.tabs(["📊 Visualization", "📋 History", "📈 Tickers"])

# Visualization Settings Tab
with tab1:
    st.header("📊 Visualization Settings")
    
    # Visualizations are always enabled - removed the toggle
    st.success("✅ Visualizations are always enabled")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Chart Settings")
        show_volume = st.checkbox("Show Volume", value=True)
        show_indicators = st.checkbox("Show Technical Indicators", value=True)
    
    with col2:
        st.subheader("Display Options")
        show_legends = st.checkbox("Show Chart Legends", value=True)
        decimal_places = st.slider("Price Decimal Places", 0, 6, 2)
    
    st.subheader("Chart Dimensions")
    col1, col2 = st.columns(2)
    with col1:
        chart_width = st.slider("Chart Width", 400, 1200, 800)
    with col2:
        chart_height = st.slider("Chart Height", 300, 800, 400)

# History Tab
with tab2:
    st.header("📋 Activity History")
    
    # Get all recent activities
    recent_activities = mian.get_recent_activities()
    
    if recent_activities:
        st.subheader(f"📊 Recent Activities ({len(recent_activities)})")
        
        # Clear activities button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col3:
            if st.button("🗑️ Clear History", type="secondary"):
                mian.clear_activities()
                st.success("Activity history cleared!")
                st.rerun()
        
        # Display activities in a nice format
        for i, activity in enumerate(recent_activities):
            with st.container():
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    # Format timestamp
                    from datetime import datetime
                    try:
                        timestamp_str = activity.get('timestamp', '')
                        if 'T' in timestamp_str or 'Z' in timestamp_str:
                            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        else:
                            timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                        formatted_time = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                    except:
                        formatted_time = activity.get('timestamp', 'Unknown time')
                    
                    # Handle both 'action' and 'message' keys for backward compatibility
                    activity_text = activity.get('action', activity.get('message', 'Unknown activity'))
                    st.markdown(f"**{activity_text}**")
                    
                    if 'details' in activity and activity['details']:
                        st.markdown(f"*{activity['details']}*")
                
                with col2:
                    st.caption(f"🕒 {formatted_time}")
                
                # Add separator except for last item
                if i < len(recent_activities) - 1:
                    st.markdown("---")
    
    else:
        st.info("No activities recorded yet. Start using the app to see your activity history!")
        
        # Show example of what activities look like
        st.subheader("📝 Activity Types")
        st.markdown("""
        Your activity history will include:
        - 📊 Stock data requests
        - 🔍 Stock comparisons
        - ⚙️ Settings changes
        - 📈 Ticker additions/removals
        - 💬 Chatbot interactions
        """)

# Ticker Management Tab
with tab3:
    st.header("📈 Ticker Management")
    
    # Initialize tracking tickers from memory if not in session state
    if "tracking_tickers" not in st.session_state:
        try:
            saved_tickers = mian.load_tracking_tickers()
            st.session_state.tracking_tickers = saved_tickers if saved_tickers else []
        except:
            st.session_state.tracking_tickers = []
    
    # Add new ticker
    col1, col2 = st.columns([3, 1])
    with col1:
        new_ticker = st.text_input("Add New Ticker", placeholder="e.g., AAPL, GOOGL, MSFT")
    with col2:
        if st.button("➕ Add", use_container_width=True):
            if new_ticker and new_ticker.upper() not in st.session_state.tracking_tickers:
                st.session_state.tracking_tickers.append(new_ticker.upper())
                st.success(f"Added {new_ticker.upper()}!")
                st.rerun()
            elif new_ticker.upper() in st.session_state.tracking_tickers:
                st.warning(f"{new_ticker.upper()} already tracked!")
    
    # Display tracked tickers using the same logic as frontend_controller.py
    if st.session_state.get('tracking_tickers'):
        st.subheader(f"📈 Your Tracked Tickers ({len(st.session_state.tracking_tickers)})")
        
        # Display tickers with real data (show all tickers, not just 4)
        display_tickers = st.session_state.tracking_tickers
        
        # Display in rows of 4
        for row_start in range(0, len(display_tickers), 4):
            row_tickers = display_tickers[row_start:row_start + 4]
            cols = st.columns(len(row_tickers))
            
            for i, ticker in enumerate(row_tickers):
                with cols[i]:
                    ticker_data = mian.get_ticker_data(ticker)
                    if ticker_data:
                        show_indicators = st.session_state.get("show_indicators", True)
                        
                        if show_indicators:
                            indicators = mian.get_ticker_indicators(ticker)
                            current_price = indicators.get("price")
                            mfi = indicators.get("mfi")
                            sma_20 = indicators.get("sma_20")
                            
                            if current_price and mfi is not None and sma_20 is not None:
                                # Display current indicators
                                price_vs_sma = "📈" if current_price > sma_20 else "📉"
                                
                                # Create ticker display with remove button
                                ticker_container = st.container()
                                with ticker_container:
                                    if mfi > 80:
                                        st.error(f"**{ticker}**\n{price_vs_sma} SMA20: ${sma_20:.2f}\n📊 MFI: {mfi:.1f}")
                                    elif mfi < 20:
                                        st.success(f"**{ticker}**\n{price_vs_sma} SMA20: ${sma_20:.2f}\n📊 MFI: {mfi:.1f}")
                                    else:
                                        st.info(f"**{ticker}**\n{price_vs_sma} SMA20: ${sma_20:.2f}\n📊 MFI: {mfi:.1f}")
                                    
                                    # Add remove button
                                    if st.button("❌ Remove", key=f"remove_{ticker}_{row_start}_{i}"):
                                        st.session_state.tracking_tickers.remove(ticker)
                                        st.rerun()
                            else:
                                st.warning(f"**{ticker}**\n❌ Data unavailable")
                                if st.button("❌ Remove", key=f"remove_na_{ticker}_{row_start}_{i}"):
                                    st.session_state.tracking_tickers.remove(ticker)
                                    st.rerun()
                        else:
                            # Show ticker without indicators when show_indicators is disabled
                            st.info(f"**{ticker}**\n📊 Tracking enabled")
                            if st.button("❌ Remove", key=f"remove_{ticker}_{row_start}_{i}"):
                                st.session_state.tracking_tickers.remove(ticker)
                                st.rerun()
                    else:
                        st.info(f"**{ticker}**\n🔄 Loading...")
                        if st.button("❌ Remove", key=f"remove_loading_{ticker}_{row_start}_{i}"):
                            st.session_state.tracking_tickers.remove(ticker)
                            st.rerun()
        
        # Bulk actions
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🗑️ Clear All"):
                st.session_state.tracking_tickers = []
                st.success("All tickers cleared!")
                st.rerun()
        with col2:
            if st.button("📤 Export List"):
                ticker_text = ", ".join(st.session_state.tracking_tickers)
                st.code(ticker_text)
        with col3:
            if st.button("🔄 Refresh Data"):
                st.success("Data will refresh on next load!")
    
    else:
        st.info("No tickers in your tracking list. Add some tickers to get started!")
    
    # Popular tickers section
    st.subheader("🌟 Popular Tickers")
    popular_tickers = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "META", "NVDA", "NFLX"]
    cols = st.columns(4)
    for i, ticker in enumerate(popular_tickers):
        with cols[i % 4]:
            if st.button(f"+ {ticker}", key=f"popular_{ticker}"):
                if ticker not in st.session_state.tracking_tickers:
                    st.session_state.tracking_tickers.append(ticker)
                    st.success(f"{ticker} added!")
                    st.rerun()

# Save settings button
st.markdown("---")
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    if st.button("💾 Save All Settings", use_container_width=True, type="primary"):
        # Save settings to file with nested structure for better management
        user_settings = {
            "visualization_settings": {
            "show_volume": st.session_state.get("show_volume", True),
            "show_indicators": st.session_state.get("show_indicators", True),
            "show_legends": st.session_state.get("show_legends", True),
            "decimal_places": st.session_state.get("decimal_places", 2),
            "chart_width": st.session_state.get("chart_width", 800),
            "chart_height": st.session_state.get("chart_height", 400),
            }
        } 
        mian.save_user_settings(user_settings)
        mian.save_tracking_tickers(st.session_state.tracking_tickers)
        st.balloons()
        st.success("🎉 All settings saved successfully!")