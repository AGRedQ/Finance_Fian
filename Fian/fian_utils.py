import pandas as pd
import matplotlib.pyplot as plt
from Bian.configs import indicator_plot_config
import numpy as np

def reset_matplotlib_style():
    """Reset matplotlib to default style to avoid conflicts"""
    plt.style.use('default')
    plt.rcdefaults()

def apply_chart_theme(chart_theme):
    """Apply chart theme and return colors for consistency"""
    if chart_theme == "Dark":
        plt.style.use('dark_background')
        return {
            'bg_color': '#1e1e1e',
            'grid_color': '#404040',
            'text_color': 'white'
        }
    elif chart_theme == "Light":
        plt.style.use('default')
        return {
            'bg_color': 'white',
            'grid_color': '#cccccc',
            'text_color': 'black'
        }
    else:  # Auto - use default
        plt.style.use('default')
        return {
            'bg_color': 'white',
            'grid_color': '#cccccc',
            'text_color': 'black'
        }



def compare_stocks(data1, data2, title=None): # Chatbot
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

    # Get chart customization settings
    chart_theme = st.session_state.get("chart_theme", "Light")
    show_legends = st.session_state.get("show_legends", True)
    
    # Get chart dimensions from settings (convert to inches for matplotlib)
    chart_width = max(4, min(16, st.session_state.get("chart_width", 800) / 100))  # 4-16 inches
    chart_height = max(3, min(12, st.session_state.get("chart_height", 400) / 100))  # 3-12 inches

    # Normalize both stocks to percentage change (starting at 100)
    stock1_prices = data1.loc[common_dates, 'Close']
    stock2_prices = data2.loc[common_dates, 'Close']
    
    # Calculate percentage change from first day (normalized to 100)
    stock1_normalized = (stock1_prices / stock1_prices.iloc[0]) * 100
    stock2_normalized = (stock2_prices / stock2_prices.iloc[0]) * 100

    fig, ax = plt.subplots(figsize=(chart_width, chart_height))
    
    # Apply chart theme
    reset_matplotlib_style()
    theme_colors = apply_chart_theme(chart_theme)
    fig.patch.set_facecolor(theme_colors['bg_color'])
    
    ax.plot(common_dates, stock1_normalized, label=f'{ticker1} (Normalized)', linewidth=2, color='blue')
    ax.plot(common_dates, stock2_normalized, label=f'{ticker2} (Normalized)', linewidth=2, color='red')
    
    # Add horizontal line at 100 (starting point)
    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.7, label='Starting Point (100)')
    
    ax.set_xlabel('Date', color=theme_colors['text_color'])
    ax.set_ylabel('Normalized Price (%)', color=theme_colors['text_color'])
    ax.set_title(title or 'Stock Comparison (Percentage Change)', color=theme_colors['text_color'])
    
    # Apply settings
    if show_legends:
        ax.legend()
    ax.grid(True, alpha=0.3, color=theme_colors['grid_color'])
    
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



def visualize_indicator(data, indicator_name, title=None): # Chatbot
    import streamlit as st
    config = indicator_plot_config.get(indicator_name, {"type": "line", "subplot": False})
    plot_type = config.get("type", "line")
    guides = config.get("guides", [])
    subplot = config.get("subplot", False)
    paired = config.get("paired", None)

    # Get chart customization settings
    chart_theme = st.session_state.get("chart_theme", "Light")
    show_legends = st.session_state.get("show_legends", True)
    
    # Get chart dimensions from settings (convert to inches for matplotlib)
    chart_width = max(4, min(16, st.session_state.get("chart_width", 800) / 100))  # 4-16 inches
    chart_height = max(3, min(12, st.session_state.get("chart_height", 400) / 100))  # 3-12 inches

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
        
        # Apply chart theme
        reset_matplotlib_style()
        theme_colors = apply_chart_theme(chart_theme)
        plt.gca().set_facecolor(theme_colors['bg_color'])
        
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

        plt.xlabel("Date", color=theme_colors['text_color'])
        plt.ylabel(indicator_name, color=theme_colors['text_color'])
        plt.title(title or f"{ticker} - {indicator_name} Visualization", color=theme_colors['text_color'])
        
        # Apply settings
        if show_legends:
            plt.legend()
        plt.grid(True, alpha=0.3, color=theme_colors['grid_color'])
        
        plt.tight_layout()
        
        import streamlit as st
        st.pyplot(plt.gcf())
        plt.close()


def display_stock_info(data, ticker):
    """Display stock information with candlestick chart, volume, and general info"""
    import streamlit as st
    import yfinance as yf
    from datetime import datetime
    import sys
    import os
    import gc  # Garbage collection for memory management
    
    # Import Bian functions
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from Bian.backend_bian import BackendBian
    bian = BackendBian()
    
    if data.empty:
        st.error(f"No data available for {ticker}")
        return
    
    # Ensure we have required columns
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        st.error(f"Missing required columns: {missing_columns}")
        return
    
    # Data validation and cleaning
    try:
        # Remove any infinite or NaN values that could cause memory issues
        data = data.replace([np.inf, -np.inf], np.nan).dropna()
        
        if data.empty:
            st.error(f"No valid data available for {ticker} after cleaning")
            return
            
        # Limit data size to prevent memory issues (max 2 years of data)
        if len(data) > 500:
            data = data.tail(500)
            
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        return
    
    # Get current stock info
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
    except:
        info = {}
    
    # Get currency info
    currency = bian.get_currency_info(info, ticker)
    
    # Get chart customization settings
    chart_theme = st.session_state.get("chart_theme", "Light")
    chart_type = st.session_state.get("chart_type", "Line")
    show_volume = st.session_state.get("show_volume", True)
    show_legends = st.session_state.get("show_legends", True)
    
    # Get chart dimensions from settings (convert to inches for matplotlib)
    chart_width = max(4, min(16, st.session_state.get("chart_width", 800) / 100))  # 4-16 inches  
    chart_height = max(3, min(12, st.session_state.get("chart_height", 400) / 100))  # 3-12 inches

    # Display general information
    st.subheader(f"📊 {ticker} Stock Information")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        current_price = float(data['Close'].iloc[-1]) if not data.empty else None
        st.metric("Current Price", bian.format_currency(current_price, currency))
    
    with col2:
        if len(data) >= 2:
            prev_close = float(data['Close'].iloc[-2])
            change = current_price - prev_close
            change_pct = (change / prev_close) * 100
            st.metric("Daily Change", bian.format_currency(change, currency), f"{change_pct:.2f}%")
        else:
            st.metric("Daily Change", "N/A")
    
    with col3:
        volume = float(data['Volume'].iloc[-1]) if not data.empty else None
        st.metric("Volume", f"{volume:,.0f}" if volume is not None else "N/A")
    
    with col4:
        market_cap = info.get('marketCap', 'N/A')
        if market_cap != 'N/A':
            st.metric("Market Cap", f"${market_cap/1e9:.2f}B")
        else:
            st.metric("Market Cap", "N/A")
    
    # Create a professional combined chart with price and volume
    try:
        # Clear any existing figures to free memory
        plt.close('all')
        gc.collect()
        
        # Reduce chart size if data is large to prevent memory issues
        if len(data) > 200:
            chart_width = min(chart_width, 10)
            chart_height = min(chart_height, 6)
        
        if show_volume:
            fig, ax1 = plt.subplots(figsize=(chart_width, chart_height))
        else:
            fig, ax1 = plt.subplots(figsize=(chart_width, chart_height))
        
        # Apply chart theme
        reset_matplotlib_style()
        theme_colors = apply_chart_theme(chart_theme)
        fig.patch.set_facecolor(theme_colors['bg_color'])
        
        # Get currency symbol for chart
        currency_symbol = bian.get_currency_symbol(currency)
        
        # Price chart on primary y-axis - with memory-safe data conversion
        close_values = data['Close'].astype(float).values
        high_values = data['High'].astype(float).values
        low_values = data['Low'].astype(float).values
        date_index = data.index
        
        # Validate data arrays before plotting
        if len(close_values) == 0 or len(high_values) == 0 or len(low_values) == 0:
            st.error("Invalid data arrays for plotting")
            plt.close(fig)
            return
        
        # Apply different chart types
        if chart_type == "Area":
            ax1.fill_between(date_index, close_values, alpha=0.3, color='#2E86C1', label='Close Price')
            ax1.plot(date_index, close_values, color='#2E86C1', linewidth=2)
        elif chart_type == "Bar":
            ax1.bar(date_index, close_values, color='#2E86C1', alpha=0.7, label='Close Price')
        else:  # Line or Candlestick (simplified as line for now)
            ax1.plot(date_index, close_values, label='Close Price', color='#2E86C1', linewidth=2.5)
        
        # High-Low range fill (only for Line and Area charts)
        if chart_type in ["Line", "Area"]:
            ax1.fill_between(date_index, low_values, high_values, alpha=0.2, color='#85C1E9', label='High-Low Range')
        
        ax1.set_xlabel('Date', fontsize=12, fontweight='bold', color=theme_colors['text_color'])
        ax1.set_ylabel(f'Price ({currency_symbol})', fontsize=12, fontweight='bold', color='#2E86C1')
        ax1.tick_params(axis='y', labelcolor='#2E86C1')
        
        # Apply grid setting
        ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, color=theme_colors['grid_color'])
    
        # Create secondary y-axis for volume (only if show_volume is enabled)
        if show_volume:
            ax2 = ax1.twinx()
            
            volume_values = data['Volume'].astype(float).values
            open_values = data['Open'].astype(float).values
            
            # Validate volume data
            if len(volume_values) == 0 or len(open_values) == 0:
                st.warning("Volume data not available")
            else:
                # Color bars based on price movement (green for up, red for down)
                colors = ['#27AE60' if close_values[i] >= open_values[i] else '#E74C3C' 
                          for i in range(len(data))]
                
                bars = ax2.bar(date_index, volume_values, color=colors, alpha=0.4, width=1.0)
                ax2.set_ylabel('Volume', fontsize=12, fontweight='bold', color='#7D3C98')
                ax2.tick_params(axis='y', labelcolor='#7D3C98')
                
                # Format volume labels (show in millions/thousands)
                max_volume = max(volume_values) if len(volume_values) > 0 else 0
                if max_volume >= 1_000_000:
                    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1_000_000:.1f}M'))
                elif max_volume >= 1_000:
                    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1_000:.1f}K'))
        
        # Set title and layout
        plt.title(f'{ticker} - Price Analysis', fontsize=16, fontweight='bold', pad=20, color=theme_colors['text_color'])
        
        # Create custom legend (only if show_legends is enabled)
        if show_legends:
            lines1, labels1 = ax1.get_legend_handles_labels()
            if show_volume:
                lines2 = [plt.Rectangle((0,0),1,1, color='#27AE60', alpha=0.4), 
                          plt.Rectangle((0,0),1,1, color='#E74C3C', alpha=0.4)]
                labels2 = ['Volume (Up Days)', 'Volume (Down Days)']
                ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
            else:
                ax1.legend(lines1, labels1, loc='upper left', fontsize=10)
        
        plt.tight_layout()
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Error creating chart: {str(e)}")
        print(f"Chart creation error: {str(e)}")
    finally:
        # Always clean up memory
        plt.close('all')
        gc.collect()
    
    # Additional company information (without business summary)
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