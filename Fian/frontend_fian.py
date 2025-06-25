import os 
import sys
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from Bian.backend_bian import BackendBian

# This will be the spine of the Fian frontend
bian = BackendBian()


class FrontendFian:
    def __init__(self):
        pass

    def handle_intent(self, intent, user_input):
        if intent == "display_price":
            tickers = bian.extract_tickers(user_input)
            period = bian.extract_period(user_input)
            return bian.fetch_data(tickers, period)
        elif intent == "compare_stocks":
            tickers = bian.extract_tickers(user_input)
            period = bian.extract_period(user_input)
            return bian.fetch_data(tickers, period)
        elif intent == "calculate_indicator":
            tickers = bian.extract_tickers(user_input)
            period = bian.extract_period(user_input)
            indicator = bian.extract_indicator(user_input)
            return bian.calculate_indicator(tickers, period, indicator)
        elif intent == "predict_indicator":
            return bian.predict_price()
        else:
            return "This query does not relate to stock trading or analysis."



    def streamlit_settings(self):
        st.set_page_config(
            page_title="Finance Assistant",
            page_icon=":money_with_wings:",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        st.sidebar.title("Settings")
        st.sidebar.write("Configure your preferences here.")
        self.visualize = st.sidebar.checkbox("Visualize", value=True)
        st.sidebar.write(f"Visualize is {'on' if self.visualize else 'off'}")

    def init_streamlit(self):
        st.title("Finance Assistant")
        st.write("Type your query below:")

        user_input = st.text_input("Your Query", "")
        if st.button("Submit"):
            if user_input:
                intent = bian.extract_intent(user_input)
                st.write(f"Detected Intent: {intent}")

            else:
                st.warning("Please enter a query.")

    def run(self):
        import streamlit as st
        self.streamlit_settings()
        self.init_streamlit()

if __name__ == "__main__":
    fian = FrontendFian()
    fian.run()