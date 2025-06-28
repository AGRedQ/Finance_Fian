import os 
import sys
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Bian.backend_bian import BackendBian
from Mian.memory_mian import MemoryMian


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
            tickers = bian.extract_tickers(user_input)
            indicator = bian.extract_indicator(user_input)
            # Due to the nature of the prediction, we assume a period of {being tested} days
            return bian.predict_price(tickers, "max", indicator)
        else:
            return "This query does not relate to stock trading or analysis."


    def visualize_setting_init(self):
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


    def autotrain_setting_init(self):
        st.sidebar.title("Auto Train Model")
        st.sidebar.write("Enable or disable automatic model training.")
        self.auto_train = st.sidebar.checkbox("Auto Train", value=False)


    def tracking_ticker_init(self):
        st.sidebar.title("Tracking Ticker")
        st.sidebar.write("Add or remove tickers to track.")
        ticker_input = st.sidebar.text_input("Ticker Symbol", "")
        if ticker_input:
            if not ticker_input:
                st.sidebar.error("Please enter a ticker symbol.")
            result = mian.add_tracking_ticker(ticker_input)
            if result == "added":
                st.sidebar.success(f"{ticker_input} added to tracking list.")
            elif result == "existed":
                st.sidebar.warning(f"{ticker_input} already exists in tracking list.")
            elif result == "invalid_ticker":
                st.sidebar.error("Invalid ticker symbol or error adding ticker.")


    
    def streamlit_settings(self):
        import streamlit as st  
        self.visualize_setting_init()
        self.autotrain_setting_init()
        self.tracking_ticker_init()

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

fian = FrontendFian()
bian = BackendBian()
mian = MemoryMian()

if __name__ == "__main__":



    bian.load_resources()
    fian.run()