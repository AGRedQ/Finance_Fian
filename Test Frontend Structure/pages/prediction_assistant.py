# app.py
# -*- coding: utf-8 -*-

import streamlit as st
from plotly import graph_objs as go
from datetime import date
import pandas as pd
import random # <<< ADD THIS IMPORT >>>
from service import PredictionService
from sklearn.metrics import classification_report

# ... (Keep GLOBAL_TICKER_LIST the same) ...
GLOBAL_TICKER_LIST = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "ORCL", "ADBE", "CRM", "INTC", "AMD", "QCOM", "CSCO",
    "JPM", "BAC", "WFC", "GS", "MS", "V", "MA", "AXP", "BLK", "JNJ", "PFE", "MRK", "UNH", "LLY", "ABBV",
    "WMT", "COST", "HD", "NKE", "MCD", "KO", "PEP", "PG", "BA", "CAT", "XOM", "CVX", "ASML", "LVMUY",
    "NVO", "SAP", "SIE.DE", "TM", "SHEL", "HSBC", "TSM", "BABA", "TCEHY", "SONY", "SSNLF",
    "FPT.VN", "VNM.VN", "HPG.VN", "VCB.VN", "TCB.VN", "MSN.VN",
]

class TrendPredictorApp:
    # ... (Keep __init__, _setup_page_config, and _plot... functions the same) ...
    def __init__(self, service: PredictionService):
        self.service = service

    def _setup_page_config(self):
        st.set_page_config(page_title="Stock Trend Prediction Assistant", page_icon="📈", layout="wide")

    def _plot_signals_on_price_chart(self, actual_price, predictions, test_dates, ticker, term_name, label_map):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=test_dates, y=actual_price, mode='lines', name='Actual Price', line=dict(color='cyan', width=2)))
        up_label_val = next((k for k, v in label_map.items() if v == 'Up'), 2)
        down_label_val = next((k for k, v in label_map.items() if v == 'Down'), 0)
        up_signals = test_dates[predictions == up_label_val]
        down_signals = test_dates[predictions == down_label_val]
        fig.add_trace(go.Scatter(x=up_signals, y=actual_price.loc[up_signals], mode='markers', name='Predicted Up', marker=dict(color='lime', size=10, symbol='triangle-up')))
        fig.add_trace(go.Scatter(x=down_signals, y=actual_price.loc[down_signals], mode='markers', name='Predicted Down', marker=dict(color='red', size=10, symbol='triangle-down')))
        fig.update_layout(title=f"{term_name} prediction signals on actual price of {ticker}", xaxis_title="Date", yaxis_title="Price", template="plotly_dark", legend_title="Legend", xaxis_rangeslider_visible=True)
        st.plotly_chart(fig, use_container_width=True)

    def _plot_future_signals_chart(self, future_predictions, future_price_baseline, ticker, label_map):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=future_price_baseline.index, y=future_price_baseline, mode='lines', name='Forecast baseline price (Prophet)', line=dict(color='cyan', width=2, dash='dash')))
        predictions = future_predictions['labels']
        up_label_val = next((k for k, v in label_map.items() if v == 'Up'), 2)
        down_label_val = next((k for k, v in label_map.items() if v == 'Down'), 0)
        up_signal_dates = predictions.index[predictions == up_label_val]
        down_signal_dates = predictions.index[predictions == down_label_val]
        if not up_signal_dates.empty:
            up_prices = future_price_baseline.loc[future_price_baseline.index.intersection(up_signal_dates)]
            if not up_prices.empty:
                fig.add_trace(go.Scatter(x=up_prices.index, y=up_prices.values, mode='markers', name='Predicted Up', marker=dict(color='lime', size=10, symbol='triangle-up')))
        if not down_signal_dates.empty:
            down_prices = future_price_baseline.loc[future_price_baseline.index.intersection(down_signal_dates)]
            if not down_prices.empty:
                fig.add_trace(go.Scatter(x=down_prices.index, y=down_prices.values, mode='markers', name='Predicted Down', marker=dict(color='red', size=10, symbol='triangle-down')))
        fig.update_layout(title=f"Future trend forecast signals for {ticker}", xaxis_title="Date", yaxis_title="Price", template="plotly_dark", legend_title="Legend")
        st.plotly_chart(fig, use_container_width=True)

    def _render_batch_test_results(self, results_df, ticker_of_interest=None):
        st.subheader(f"Performance Test Results on {len(results_df)} tickers")
        if results_df is None or results_df.empty:
            st.warning("No results to display.")
            return

        def highlight_ticker(s):
            return ['background-color: #004466' if s.ticker == ticker_of_interest else '' for _ in s]

        st.dataframe(results_df.style.apply(highlight_ticker, axis=1).format({
            'up_precision': '{:.1%}', 'up_recall': '{:.1%}', 'up_f1_score': '{:.2f}',
            'down_precision': '{:.1%}', 'down_recall': '{:.1%}', 'down_f1_score': '{:.2f}',
            'accuracy': '{:.1%}'
        }), height=300)
        
        st.subheader("Average Performance Comparison")
        avg_metrics = results_df[results_df['best_model'] != 'ERROR'].drop(columns=['ticker', 'best_model']).mean()
        
        if ticker_of_interest and ticker_of_interest in results_df['ticker'].values:
            ticker_metrics = results_df.loc[results_df['ticker'] == ticker_of_interest].drop(columns=['ticker', 'best_model']).iloc[0]
            comparison_df = pd.DataFrame({'Average Performance': avg_metrics, f'Performance for {ticker_of_interest}': ticker_metrics})
            st.dataframe(comparison_df.style.format('{:.2f}'))
        else:
            comparison_df = pd.DataFrame({'Average Performance': avg_metrics})
            st.dataframe(comparison_df.style.format('{:.2f}'))
        
        st.info(f"The table above compares the model performance on your selected ticker with the average performance on a random sample of {len(results_df)} global tickers.")


    def run(self):
        self._setup_page_config()
        
        # Return to Main Menu button
        if st.button("← Return to Main Menu", type="secondary"):
            st.switch_page("pages/main_menu.py")
        
        st.title("📈 Stock Trend Prediction Assistant")
        
        # --- SECTION 1: GENERAL SETTINGS ---
        st.header("1. Analysis Settings")
        col1, col2, col3 = st.columns(3)
        with col1:
            ticker_option = st.selectbox("Select stock ticker for analysis", GLOBAL_TICKER_LIST, index=4)
            custom_ticker = st.text_input("Or enter another ticker").upper()
            selected_ticker = custom_ticker if custom_ticker else ticker_option
        with col2:
            prediction_type = st.selectbox("Select analysis type", ["Short-term Analysis", "Medium-term Analysis"])
        with col3:
            horizons = {"Next 30 days": 30, "Next 3 months": 63, "Next 6 months": 126}
            selected_horizon_label = st.selectbox("Future prediction timeframe", list(horizons.keys()))
            periods_to_forecast = horizons[selected_horizon_label]
        
        st.markdown("---")

        # --- SECTION 2: COMPARISON (BENCHMARKING) ---
        st.header("2. Performance Comparison Options")
        
        # <<< CHANGE 1: ADD SLIDER >>>
        num_tickers_to_compare = st.slider(
            "Select number of random tickers to compare",
            min_value=10, 
            max_value=len(GLOBAL_TICKER_LIST), 
            value=25, # Default value
            step=5,
            help="Choose the number of stock tickers from the global list to run tests and compare performance. Higher numbers give more reliable results but take longer."
        )

        # --- SECTION 3: BUTTON AND EXECUTION ---
        if st.button("🚀 Start Analysis and Comparison!", type="primary", use_container_width=True):
            if not selected_ticker:
                st.warning("Please select or enter a stock ticker.")
            else:
                # --- Run individual analysis ---
                st.subheader(f"Analysis Results for {selected_ticker}")
                term_name = "Short-term" if "Short-term" in prediction_type else "Medium-term"
                with st.spinner(f"Performing {term_name} analysis for ticker **{selected_ticker}**..."):
                    if term_name == "Short-term":
                        single_results = self.service.run_short_term_prediction(selected_ticker, periods_to_forecast)
                    else:
                        single_results = self.service.run_mid_term_prediction(selected_ticker, periods_to_forecast)

                if single_results is None:
                    st.error(f"Could not complete analysis for {selected_ticker}.")
                else:
                    # Display individual analysis results...
                    info = single_results['data_info']
                    st.info(f"🏆 **Recommended model:** `{single_results['best_model']}`")
                    self._plot_signals_on_price_chart(single_results['test_data']['actual_price'], single_results['predictions_on_test'][single_results['best_model']], single_results['test_data']['dates'], selected_ticker, term_name, info['label_map'])
                    st.subheader("Future trend forecast")
                    if single_results['future_forecasts'] and single_results['best_model'] in single_results['future_forecasts']:
                         self._plot_future_signals_chart(single_results['future_forecasts'][single_results['best_model']], single_results["future_price_baseline"], selected_ticker, info['label_map'])

                st.markdown("---")

                # --- Run general test for comparison ---
                st.header("Comparison with Global Performance")
                analysis_type_param = 'short_term' if term_name == 'Short-term' else 'mid_term'
                
                # <<< CHANGE 2: CREATE RANDOM SAMPLE >>>
                # Get a random sample from the global list
                other_tickers = [t for t in GLOBAL_TICKER_LIST if t != selected_ticker]
                random_sample = random.sample(other_tickers, k=min(num_tickers_to_compare - 1, len(other_tickers)))
                
                # Ensure the ticker being analyzed is always in the list
                ticker_list_to_test = [selected_ticker] + random_sample
                # Shuffle the final list so the ticker being analyzed isn't always first
                random.shuffle(ticker_list_to_test)
                
                with st.spinner(f"Running tests on {len(ticker_list_to_test)} random tickers for benchmarking... (this may take a while)"):
                    batch_results_df = self.service.run_batch_test(ticker_list_to_test, analysis_type_param)
                
                self._render_batch_test_results(batch_results_df, ticker_of_interest=selected_ticker)


if __name__ == "__main__":
    prediction_service = PredictionService()
    app = TrendPredictorApp(service=prediction_service)
    app.run()