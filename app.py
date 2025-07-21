# app.py
# -*- coding: utf-8 -*-

import streamlit as st
from plotly import graph_objs as go
from datetime import date
import pandas as pd
import random # <<< THÊM IMPORT NÀY
from service import PredictionService
from sklearn.metrics import classification_report

# ... (Danh sách GLOBAL_TICKER_LIST giữ nguyên) ...
GLOBAL_TICKER_LIST = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "ORCL", "ADBE", "CRM", "INTC", "AMD", "QCOM", "CSCO",
    "JPM", "BAC", "WFC", "GS", "MS", "V", "MA", "AXP", "BLK", "JNJ", "PFE", "MRK", "UNH", "LLY", "ABBV",
    "WMT", "COST", "HD", "NKE", "MCD", "KO", "PEP", "PG", "BA", "CAT", "XOM", "CVX", "ASML", "LVMUY",
    "NVO", "SAP", "SIE.DE", "TM", "SHEL", "HSBC", "TSM", "BABA", "TCEHY", "SONY", "SSNLF",
    "FPT.VN", "VNM.VN", "HPG.VN", "VCB.VN", "TCB.VN", "MSN.VN",
]

class TrendPredictorApp:
    # ... (Các hàm __init__, _setup_page_config, và các hàm _plot... giữ nguyên) ...
    def __init__(self, service: PredictionService):
        self.service = service

    def _setup_page_config(self):
        st.set_page_config(page_title="Trợ Lý Dự Đoán Xu Hướng Cổ Phiếu", page_icon="📈", layout="wide")

    def _plot_signals_on_price_chart(self, actual_price, predictions, test_dates, ticker, term_name, label_map):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=test_dates, y=actual_price, mode='lines', name='Giá thực tế', line=dict(color='cyan', width=2)))
        up_label_val = next((k for k, v in label_map.items() if v == 'Up'), 2)
        down_label_val = next((k for k, v in label_map.items() if v == 'Down'), 0)
        up_signals = test_dates[predictions == up_label_val]
        down_signals = test_dates[predictions == down_label_val]
        fig.add_trace(go.Scatter(x=up_signals, y=actual_price.loc[up_signals], mode='markers', name='Dự đoán Lên', marker=dict(color='lime', size=10, symbol='triangle-up')))
        fig.add_trace(go.Scatter(x=down_signals, y=actual_price.loc[down_signals], mode='markers', name='Dự đoán Xuống', marker=dict(color='red', size=10, symbol='triangle-down')))
        fig.update_layout(title=f"Tín hiệu dự đoán {term_name} trên giá thực tế của {ticker}", xaxis_title="Ngày", yaxis_title="Giá", template="plotly_dark", legend_title="Chú giải", xaxis_rangeslider_visible=True)
        st.plotly_chart(fig, use_container_width=True)

    def _plot_future_signals_chart(self, future_predictions, future_price_baseline, ticker, label_map):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=future_price_baseline.index, y=future_price_baseline, mode='lines', name='Giá cơ sở dự báo (Prophet)', line=dict(color='cyan', width=2, dash='dash')))
        predictions = future_predictions['labels']
        up_label_val = next((k for k, v in label_map.items() if v == 'Up'), 2)
        down_label_val = next((k for k, v in label_map.items() if v == 'Down'), 0)
        up_signal_dates = predictions.index[predictions == up_label_val]
        down_signal_dates = predictions.index[predictions == down_label_val]
        if not up_signal_dates.empty:
            up_prices = future_price_baseline.loc[future_price_baseline.index.intersection(up_signal_dates)]
            if not up_prices.empty:
                fig.add_trace(go.Scatter(x=up_prices.index, y=up_prices.values, mode='markers', name='Dự đoán Lên', marker=dict(color='lime', size=10, symbol='triangle-up')))
        if not down_signal_dates.empty:
            down_prices = future_price_baseline.loc[future_price_baseline.index.intersection(down_signal_dates)]
            if not down_prices.empty:
                fig.add_trace(go.Scatter(x=down_prices.index, y=down_prices.values, mode='markers', name='Dự đoán Xuống', marker=dict(color='red', size=10, symbol='triangle-down')))
        fig.update_layout(title=f"Tín hiệu dự báo xu hướng tương lai cho {ticker}", xaxis_title="Ngày", yaxis_title="Giá", template="plotly_dark", legend_title="Chú giải")
        st.plotly_chart(fig, use_container_width=True)

    def _render_batch_test_results(self, results_df, ticker_of_interest=None):
        st.subheader(f"Kết quả Kiểm tra Hiệu suất trên {len(results_df)} mã")
        if results_df is None or results_df.empty:
            st.warning("Không có kết quả nào để hiển thị.")
            return

        def highlight_ticker(s):
            return ['background-color: #004466' if s.ticker == ticker_of_interest else '' for _ in s]

        st.dataframe(results_df.style.apply(highlight_ticker, axis=1).format({
            'up_precision': '{:.1%}', 'up_recall': '{:.1%}', 'up_f1_score': '{:.2f}',
            'down_precision': '{:.1%}', 'down_recall': '{:.1%}', 'down_f1_score': '{:.2f}',
            'accuracy': '{:.1%}'
        }), height=300)
        
        st.subheader("So sánh Hiệu suất Trung bình")
        avg_metrics = results_df[results_df['best_model'] != 'ERROR'].drop(columns=['ticker', 'best_model']).mean()
        
        if ticker_of_interest and ticker_of_interest in results_df['ticker'].values:
            ticker_metrics = results_df.loc[results_df['ticker'] == ticker_of_interest].drop(columns=['ticker', 'best_model']).iloc[0]
            comparison_df = pd.DataFrame({'Hiệu suất Trung bình': avg_metrics, f'Hiệu suất cho {ticker_of_interest}': ticker_metrics})
            st.dataframe(comparison_df.style.format('{:.2f}'))
        else:
            comparison_df = pd.DataFrame({'Hiệu suất Trung bình': avg_metrics})
            st.dataframe(comparison_df.style.format('{:.2f}'))
        
        st.info(f"Bảng trên so sánh hiệu suất của mô hình trên mã bạn chọn với hiệu suất trung bình trên một mẫu ngẫu nhiên gồm {len(results_df)} mã toàn cầu.")


    def run(self):
        self._setup_page_config()
        st.title("📈 Trợ Lý Dự Đoán Xu Hướng Cổ Phiếu")
        
        # --- PHẦN 1: CÀI ĐẶT CHUNG ---
        st.header("1. Cài đặt Phân tích")
        col1, col2, col3 = st.columns(3)
        with col1:
            ticker_option = st.selectbox("Chọn mã cổ phiếu để phân tích", GLOBAL_TICKER_LIST, index=4)
            custom_ticker = st.text_input("Hoặc nhập mã khác").upper()
            selected_ticker = custom_ticker if custom_ticker else ticker_option
        with col2:
            prediction_type = st.selectbox("Chọn loại phân tích", ["Phân tích Ngắn hạn", "Phân tích Trung hạn"])
        with col3:
            horizons = {"30 ngày tới": 30, "3 tháng tới": 63, "6 tháng tới": 126}
            selected_horizon_label = st.selectbox("Thời gian dự đoán tương lai", list(horizons.keys()))
            periods_to_forecast = horizons[selected_horizon_label]
        
        st.markdown("---")

        # --- PHẦN 2: SO SÁNH (BENCHMARKING) ---
        st.header("2. Tùy chọn So sánh Hiệu suất")
        
        # <<< THAY ĐỔI 1: THÊM THANH TRƯỢT >>>
        num_tickers_to_compare = st.slider(
            "Chọn số lượng mã ngẫu nhiên để so sánh",
            min_value=10, 
            max_value=len(GLOBAL_TICKER_LIST), 
            value=25, # Giá trị mặc định
            step=5,
            help="Chọn số lượng mã cổ phiếu từ danh sách toàn cầu để chạy kiểm tra và so sánh hiệu suất. Số lượng lớn hơn cho kết quả đáng tin cậy hơn nhưng mất nhiều thời gian hơn."
        )

        # --- PHẦN 3: NÚT BẤM VÀ THỰC THI ---
        if st.button("🚀 Bắt đầu Phân tích và So sánh!", type="primary", use_container_width=True):
            if not selected_ticker:
                st.warning("Vui lòng chọn hoặc nhập một mã cổ phiếu.")
            else:
                # --- Chạy phân tích đơn lẻ ---
                st.subheader(f"Kết quả Phân tích cho {selected_ticker}")
                term_name = "Ngắn hạn" if "Ngắn hạn" in prediction_type else "Trung hạn"
                with st.spinner(f"Đang thực hiện phân tích {term_name} cho mã **{selected_ticker}**..."):
                    if term_name == "Ngắn hạn":
                        single_results = self.service.run_short_term_prediction(selected_ticker, periods_to_forecast)
                    else:
                        single_results = self.service.run_mid_term_prediction(selected_ticker, periods_to_forecast)

                if single_results is None:
                    st.error(f"Không thể hoàn thành phân tích cho {selected_ticker}.")
                else:
                    # Hiển thị kết quả phân tích đơn lẻ...
                    info = single_results['data_info']
                    st.info(f"🏆 **Mô hình đề xuất:** `{single_results['best_model']}`")
                    self._plot_signals_on_price_chart(single_results['test_data']['actual_price'], single_results['predictions_on_test'][single_results['best_model']], single_results['test_data']['dates'], selected_ticker, term_name, info['label_map'])
                    st.subheader("Dự báo xu hướng tương lai")
                    if single_results['future_forecasts'] and single_results['best_model'] in single_results['future_forecasts']:
                         self._plot_future_signals_chart(single_results['future_forecasts'][single_results['best_model']], single_results["future_price_baseline"], selected_ticker, info['label_map'])

                st.markdown("---")

                # --- Chạy kiểm tra tổng quát để so sánh ---
                st.header("So sánh với Hiệu suất Toàn cầu")
                analysis_type_param = 'short_term' if term_name == 'Ngắn hạn' else 'mid_term'
                
                # <<< THAY ĐỔI 2: TẠO MẪU NGẪU NHIÊN >>>
                # Lấy một mẫu ngẫu nhiên từ danh sách toàn cầu
                other_tickers = [t for t in GLOBAL_TICKER_LIST if t != selected_ticker]
                random_sample = random.sample(other_tickers, k=min(num_tickers_to_compare - 1, len(other_tickers)))
                
                # Đảm bảo mã đang phân tích luôn có trong danh sách
                ticker_list_to_test = [selected_ticker] + random_sample
                # Xáo trộn danh sách cuối cùng để mã đang phân tích không luôn ở đầu
                random.shuffle(ticker_list_to_test)
                
                with st.spinner(f"Đang chạy kiểm tra trên {len(ticker_list_to_test)} mã ngẫu nhiên để làm benchmark... (việc này có thể mất một lúc)"):
                    batch_results_df = self.service.run_batch_test(ticker_list_to_test, analysis_type_param)
                
                self._render_batch_test_results(batch_results_df, ticker_of_interest=selected_ticker)


if __name__ == "__main__":
    prediction_service = PredictionService()
    app = TrendPredictorApp(service=prediction_service)
    app.run()