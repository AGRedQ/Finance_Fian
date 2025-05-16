import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

st.title("Chatbot with DialoGPT (Hugging Face)")

# Load tokenizer and model once
@st.cache(allow_output_mutation=True)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
    return tokenizer, model

tokenizer, model = load_model()

if "chat_history_ids" not in st.session_state:
    st.session_state.chat_history_ids = None
if "step" not in st.session_state:
    st.session_state.step = 0

user_input = st.text_input("You:")

if user_input:
    # Encode the input and add to chat history
    new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

    if st.session_state.chat_history_ids is not None:
        bot_input_ids = torch.cat([st.session_state.chat_history_ids, new_user_input_ids], dim=-1)
    else:
        bot_input_ids = new_user_input_ids

    # Generate response
    st.session_state.chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    # Decode and display output
    bot_output = tokenizer.decode(st.session_state.chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

    # Save conversation in session_state to display
    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.session_state.messages.append(("You", user_input))
    st.session_state.messages.append(("Bot", bot_output))

# Show conversation
if "messages" in st.session_state:
    for sender, msg in st.session_state.messages:
        if sender == "You":
            st.markdown(f"<div style='text-align: right; color: blue;'><b>You:</b> {msg}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='text-align: left; color: green;'><b>Bot:</b> {msg}</div>", unsafe_allow_html=True)
