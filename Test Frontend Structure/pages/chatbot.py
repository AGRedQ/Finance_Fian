# Page Init
import streamlit as st
import os
import sys

# Local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Bian.backend_bian import BackendBian
bian = BackendBian()

# Note: Don't worry bois, gemini model is loaded in resources.py and imported via bian.load_resources
# and loaded when the app starts


# --- Streamlit Page Setup ---
st.set_page_config(
    page_title="Chatbot - Finance Assistant",
    page_icon="💬",
    layout="wide"
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "message_count" not in st.session_state:
    st.session_state.message_count = 0

st.title("💬 Finance Assistant Chatbot")
st.write("Chat with me about stocks, indicators, and financial analysis!")
st.write("Start your sentence with a '/' to use commands, '/help' to start")

with st.expander("💡 Example Queries"):
    st.write("Try typing me:")
    st.code("What is the MFI in stock market?")
    st.code("/help")
    st.code("/predict TSLA short") 
    st.code("/compare AAPL MSFT")

# --- Chat Interface ---
with st.container():
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

# Process input outside the form to avoid rerun issues
if submitted and user_input:
    # Add user message
    st.session_state.messages.append({
        "role": "user", 
        "content": user_input,
        "timestamp": st.session_state.get('message_count', 0)
    })
    
    # Process the query and handle both text and potential charts
    try:
        # Import the needed functions directly
        from Bian.bian_utils import handle_input_type, run_command, run_query
        
        input_type = handle_input_type(user_input)
        
        if input_type == "command":
            # For commands, show immediate output
            st.markdown("### 🤖 Processing command...")
            bot_response = run_command(user_input)
            st.success(bot_response)
        else:
            # For regular chat, use the query function
            bot_response = run_query(user_input)
        
    except Exception as e:
        bot_response = f"Sorry, I encountered an error: {str(e)}"
        st.error(bot_response)
    
    # Add bot response
    st.session_state.messages.append({
        "role": "assistant", 
        "content": bot_response,
        "timestamp": st.session_state.get('message_count', 0) + 1
    })
    st.session_state.message_count = st.session_state.get('message_count', 0) + 2

# --- Display Chat History ---
if st.session_state.messages:
    st.markdown("### 💭 Conversation")
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"""
                <div style="background-color: transparent; padding: 10px; border-radius: 10px; margin: 5px 0; text-align: right;">
                    <strong>🧑 You:</strong> {message["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background-color: transparent; padding: 10px; border-radius: 10px; margin: 5px 0;">
                    <strong>🤖 Assistant:</strong> {message["content"]}
                </div>
                """, unsafe_allow_html=True)
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