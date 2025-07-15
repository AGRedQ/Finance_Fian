import streamlit as st

st.set_page_config(
    page_title="Chatbot - Finance Assistant",
    page_icon="💬",
    layout="wide"
)

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

def get_bot_response(user_input):
    # Enhanced mock response with different types
    user_input_lower = user_input.lower()
    
    if "price" in user_input_lower:
        return f"📈 I can help you get price information! For '{user_input}', I would normally fetch real-time data and show you current prices with charts."
    elif "rsi" in user_input_lower or "macd" in user_input_lower or "indicator" in user_input_lower:
        return f"📊 Technical indicator analysis requested! For '{user_input}', I would calculate the indicators and provide visual analysis."
    elif "predict" in user_input_lower:
        return f"🔮 Price prediction requested! For '{user_input}', I would use LSTM models to forecast future prices."
    elif "compare" in user_input_lower:
        return f"⚖️ Stock comparison requested! For '{user_input}', I would show side-by-side analysis of multiple stocks."
    else:
        return f"🤖 I understand you're asking about '{user_input}'. This would normally be processed through our AI pipeline to provide detailed financial analysis!"

st.title("💬 Finance Assistant Chatbot")
st.write("Ask me anything about stocks, indicators, predictions, or comparisons!")

# Example queries
with st.expander("💡 Example Queries"):
    st.write("Try asking me:")
    st.code("Show me AAPL stock price for the last month")
    st.code("Calculate RSI for GOOGL")
    st.code("Predict TSLA price for next week")
    st.code("Compare AAPL vs MSFT performance")

# Chat interface
with st.container():
    # Chat input
    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([4, 1])
        with col1:
            user_input = st.text_input(
                "Your message:", 
                placeholder="e.g., Show me AAPL stock price trends...",
                label_visibility="collapsed"
            )
        with col2:
            submitted = st.form_submit_button("Send 📤", use_container_width=True)
        
        if submitted and user_input:
            # Add user message
            st.session_state.messages.append({
                "role": "user", 
                "content": user_input,
                "timestamp": st.session_state.get('message_count', 0)
            })
            
            # Get bot response
            bot_response = get_bot_response(user_input)
            st.session_state.messages.append({
                "role": "assistant", 
                "content": bot_response,
                "timestamp": st.session_state.get('message_count', 0) + 1
            })
            
            st.session_state.message_count = st.session_state.get('message_count', 0) + 2
            st.rerun()

# Display chat history
if st.session_state.messages:
    st.markdown("### 💭 Conversation")
    
    # Create a container for chat messages
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"""
                <div style="background-color: #e3f2fd; padding: 10px; border-radius: 10px; margin: 5px 0; text-align: right;">
                    <strong>🧑 You:</strong> {message["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background-color: #f3e5f5; padding: 10px; border-radius: 10px; margin: 5px 0;">
                    <strong>🤖 Assistant:</strong> {message["content"]}
                </div>
                """, unsafe_allow_html=True)
    
    # Chat controls
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.session_state.message_count = 0
            st.rerun()
    
    with col2:
        if st.button("💾 Save Chat"):
            st.success("Chat saved! (Mock functionality)")
    
    with col3:
        st.write(f"Messages: {len(st.session_state.messages)}")

else:
    st.info("👋 Start a conversation by typing a message above!")
    
    # Quick start buttons
    st.markdown("### 🚀 Quick Start")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📈 Ask about stock prices"):
            st.session_state.messages.append({
                "role": "user", 
                "content": "Show me current stock prices for AAPL",
                "timestamp": 0
            })
            bot_response = get_bot_response("Show me current stock prices for AAPL")
            st.session_state.messages.append({
                "role": "assistant", 
                "content": bot_response,
                "timestamp": 1
            })
            st.rerun()
    
    with col2:
        if st.button("📊 Ask about indicators"):
            st.session_state.messages.append({
                "role": "user", 
                "content": "Calculate RSI for GOOGL",
                "timestamp": 0
            })
            bot_response = get_bot_response("Calculate RSI for GOOGL")
            st.session_state.messages.append({
                "role": "assistant", 
                "content": bot_response,
                "timestamp": 1
            })
            st.rerun()