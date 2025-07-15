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
tab1, tab2, tab3, tab4 = st.tabs(["📊 Visualization", "🤖 Auto Train", "📈 Tickers", "🎨 Appearance"])

# Visualization Settings Tab
with tab1:
    st.header("📊 Visualization Settings")
    
    # Main visualization toggle
    st.session_state.visualize = st.checkbox(
        "Enable Visualizations", 
        value=st.session_state.visualize,
        help="Turn on/off chart and graph displays"
    )
    
    if st.session_state.visualize:
        st.success("✅ Visualizations are enabled")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Chart Settings")
            chart_type = st.selectbox("Default Chart Type", ["Line", "Candlestick", "Bar", "Area"])
            chart_theme = st.selectbox("Chart Theme", ["Light", "Dark", "Auto"])
            show_volume = st.checkbox("Show Volume", value=True)
            show_grid = st.checkbox("Show Grid Lines", value=True)
        
        with col2:
            st.subheader("Display Options")
            show_indicators = st.checkbox("Show Technical Indicators", value=True)
            show_legends = st.checkbox("Show Chart Legends", value=True)
            animation_enabled = st.checkbox("Enable Chart Animations", value=False)
            decimal_places = st.slider("Price Decimal Places", 0, 6, 2)
        
        st.subheader("Chart Dimensions")
        col1, col2 = st.columns(2)
        with col1:
            chart_width = st.slider("Chart Width", 400, 1200, 800)
        with col2:
            chart_height = st.slider("Chart Height", 300, 800, 400)
            
    else:
        st.warning("⚠️ Visualizations are disabled")

# Auto Train Settings Tab
with tab2:
    st.header("🤖 Auto Train Model Settings")
    
    # Main auto train toggle
    st.session_state.auto_train = st.checkbox(
        "Enable Auto Training", 
        value=st.session_state.auto_train,
        help="Automatically retrain models when performance drops"
    )
    
    if st.session_state.auto_train:
        st.success("✅ Auto training is enabled")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Training Schedule")
            training_frequency = st.selectbox("Training Frequency", ["Daily", "Weekly", "Monthly"])
            training_time = st.time_input("Preferred Training Time")
            retrain_threshold = st.slider("Retrain Accuracy Threshold", 0.5, 1.0, 0.8)
        
        with col2:
            st.subheader("Resource Limits")
            max_training_time = st.slider("Max Training Time (minutes)", 5, 120, 30)
            training_data_period = st.selectbox("Training Data Period", ["1 year", "2 years", "5 years", "All available"])
            cpu_usage_limit = st.slider("CPU Usage Limit (%)", 10, 100, 70)
        
        # Model status display
        # Developer Note: This will becomes available after models are read 
        st.subheader("📊 Model Status")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Short Term", "92%", "↑2%")
        with col2:
            st.metric("Medium Term", "87%", "↓1%")
        with col3:
            st.metric("Long Term", "89%", "↑3%")
        
        # Manual training buttons
        st.subheader("Manual Training")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button("🎯 Train Intent Classifier"):
                with st.spinner("Training..."):
                    st.success("Intent classifier training started!")
        with col2:
            if st.button("📈 Train Price Predictor"):
                with st.spinner("Training..."):
                    st.success("Price predictor training started!")
        with col3:
            if st.button("📊 Train Indicator Models"):
                with st.spinner("Training..."):
                    st.success("Indicator models training started!")
        with col4:
            if st.button("🔄 Train All Models"):
                with st.spinner("Training all models..."):
                    st.success("All models training started!")
    else:
        st.warning("⚠️ Auto training is disabled")
        st.info("You can still train models manually using the buttons below:")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Train Intent Classifier", key="manual_intent"):
                st.info("Manual training started...")
        with col2:
            if st.button("Train Price Predictor", key="manual_price"):
                st.info("Manual training started...")
        with col3:
            if st.button("Train All Models", key="manual_all"):
                st.info("Manual training started...")

# Ticker Management Tab
with tab3:
    st.header("📈 Ticker Management")
    
    # Add new ticker
    col1, col2 = st.columns([3, 1])
    with col1:
        new_ticker = st.text_input("Add New Ticker", placeholder="e.g., AAPL, GOOGL, MSFT")
    with col2:
        if st.button("➕ Add", use_container_width=True):
            if new_ticker:
                if new_ticker.upper() not in st.session_state.get('tracking_tickers', []):
                    # Add to memory manager
                    result = mian.add_tracking_ticker(new_ticker.upper())
                    if isinstance(result, str) and "Invalid" in result:
                        st.error(result)
                    else:
                        # Add to session state
                        if 'tracking_tickers' not in st.session_state:
                            st.session_state.tracking_tickers = []
                        st.session_state.tracking_tickers.append(new_ticker.upper())
                        st.success(f"Added {new_ticker.upper()}!")
                        st.rerun()
                else:
                    st.warning(f"{new_ticker.upper()} already tracked!")
    
    # Display and manage current tickers
    if st.session_state.get('tracking_tickers'):
        st.subheader(f"Current Tickers ({len(st.session_state.tracking_tickers)})")
        
        # Create columns for ticker display
        cols_per_row = 4
        for i in range(0, len(st.session_state.tracking_tickers), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, ticker in enumerate(st.session_state.tracking_tickers[i:i+cols_per_row]):
                with cols[j]:
                    ticker_data = mian.get_ticker_data(ticker)
                    if ticker_data:
                        indicators = mian.get_ticker_indicators(ticker)
                        current_price = indicators.get("price")
                        mfi = indicators.get("mfi")
                        sma_20 = indicators.get("sma_20")
                        
                        st.markdown(f"**{ticker}**")
                        if current_price:
                            st.caption(f"💰 ${current_price:.2f}")
                            if sma_20:
                                trend = "📈" if current_price > sma_20 else "📉"
                                st.caption(f"{trend} SMA20: ${sma_20:.2f}")
                            if mfi is not None:
                                if mfi > 80:
                                    st.caption(f"� MFI: {mfi:.1f} (Overbought)")
                                elif mfi < 20:
                                    st.caption(f"🟢 MFI: {mfi:.1f} (Oversold)")
                                else:
                                    st.caption(f"🔵 MFI: {mfi:.1f}")
                        else:
                            st.caption("❌ Data unavailable")
                    else:
                        st.markdown(f"**{ticker}**")
                        st.caption("🔄 Loading...")
                    
                    if st.button("❌", key=f"remove_{ticker}_{i}_{j}"):
                        # Remove from memory manager
                        mian.remove_tracking_ticker(ticker)
                        # Remove from session state
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
            if st.button("🔄 Refresh Status"):
                st.info("Ticker status refreshed!")
    
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

# Appearance Settings Tab
with tab4:
    st.header("🎨 Appearance Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Theme")
        theme = st.selectbox("App Theme", ["Auto", "Light", "Dark"])
        color_scheme = st.selectbox("Color Scheme", ["Default", "Blue", "Green", "Purple"])
        font_size = st.selectbox("Font Size", ["Small", "Medium", "Large"])
    
    with col2:
        st.subheader("Layout")
        sidebar_state = st.selectbox("Default Sidebar State", ["Expanded", "Collapsed"])
        page_width = st.selectbox("Page Width", ["Normal", "Wide", "Centered"])
        show_tooltips = st.checkbox("Show Help Tooltips", value=True)

# Save settings button
st.markdown("---")
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    if st.button("💾 Save All Settings", use_container_width=True, type="primary"):
        # Save settings to file with nested structure for better management
        user_settings = {
            "visualization_settings": {
            "visualize": st.session_state.get("visualize", True),
            "chart_type": st.session_state.get("chart_type", "Line"),
            "chart_theme": st.session_state.get("chart_theme", "Light"),
            "show_volume": st.session_state.get("show_volume", True),
            "show_grid": st.session_state.get("show_grid", True),
            "show_indicators": st.session_state.get("show_indicators", True),
            "show_legends": st.session_state.get("show_legends", True),
            "animation_enabled": st.session_state.get("animation_enabled", False),
            "decimal_places": st.session_state.get("decimal_places", 2),
            "chart_width": st.session_state.get("chart_width", 800),
            "chart_height": st.session_state.get("chart_height", 400),
            },
            "auto_train_settings": {
            "auto_train": st.session_state.get("auto_train", False),
            "training_frequency": st.session_state.get("training_frequency", "Daily"),
            "training_time": str(st.session_state.get("training_time", "")),
            "retrain_threshold": st.session_state.get("retrain_threshold", 0.8),
            "max_training_time": st.session_state.get("max_training_time", 30),
            "training_data_period": st.session_state.get("training_data_period", "1 year"),
            "cpu_usage_limit": st.session_state.get("cpu_usage_limit", 70),
            },
            "appearance_settings": {
            "theme": st.session_state.get("theme", "Auto"),
            "color_scheme": st.session_state.get("color_scheme", "Default"),
            "font_size": st.session_state.get("font_size", "Medium"),
            "sidebar_state": st.session_state.get("sidebar_state", "Expanded"),
            "page_width": st.session_state.get("page_width", "Normal"),
            "show_tooltips": st.session_state.get("show_tooltips", True),
            }
        } 
        mian.save_user_settings(user_settings)
        mian.save_tracking_tickers(st.session_state.tracking_tickers)
        st.balloons()
        st.success("🎉 All settings saved successfully!")