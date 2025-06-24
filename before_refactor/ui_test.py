import streamlit as st

class StreamlitApp:
    def __init__(self):
        if "page" not in st.session_state:
            st.session_state.page = 1
        if "messages" not in st.session_state:
            st.session_state.messages = []

    def go_to_page(self, page_num):
        st.session_state.page = page_num

    def page_one(self):
        st.title("Welcome to My Streamlit App")
        st.write("This is a fresh Streamlit application. Start building your UI here!")

        with st.container():
            col1, col2 = st.columns([8, 3])
            with col2:
                option = st.selectbox(
                    "Choose an option",
                    ["Option 1", "Option 2", "Option 3"],
                    label_visibility="collapsed"
                )

        st.markdown("---")
        st.subheader("Chatbox")

        with st.form("chat_form", clear_on_submit=True):
            user_input = st.text_input("Type your message:", key="chat_input")
            submitted = st.form_submit_button("Send")
            if submitted and user_input:
                st.session_state.messages.append(("You", user_input))

        for sender, message in st.session_state.messages:
            st.write(f"**{sender}:** {message}")

        if st.button("Go to Page 2"):
            self.go_to_page(2)
            st.rerun()

    def page_two(self):
        st.write("This is page 2")
        if st.button("Back to Page 1"):
            self.go_to_page(1)
            st.rerun()

    def run(self):
        if st.session_state.page == 1:
            self.page_one()
        elif st.session_state.page == 2:
            self.page_two()

if __name__ == "__main__":
    app = StreamlitApp()
    app.run()