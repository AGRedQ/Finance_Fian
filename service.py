# service.py
# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_datareader.data as web
from datetime import date, timedelta 

# === MODELS (CLASSIFICATION) ===
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# === METRICS AND PREPROCESSING (CLASSIFICATION) ===
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.preprocessing import MinMaxScaler 
from sklearn.utils.class_weight import compute_sample_weight # <<< CẢI TIẾN: Thêm để xử lý mất cân bằng cho XGBoost

import warnings
from prophet import Prophet 

warnings.filterwarnings('ignore') 

@st.cache_data(ttl="1h")
def get_stock_data(_ticker, _start_date, _end_date):
    """
    Tải dữ liệu cổ phiếu lịch sử và làm sạch.
    """
    try:
        data = yf.download(_ticker, start=_start_date, end=_end_date, interval="1d", auto_adjust=True, progress=False)
        if data.empty: return None 
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in data.columns.values]
        
        standard_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        new_columns = {}
        for col in data.columns:
            for std_col in standard_cols:
                if col.startswith(f"{std_col}_") or col == std_col:
                    new_columns[col] = std_col
                    break
        data.rename(columns=new_columns, inplace=True)

        data = data.loc[:,~data.columns.duplicated()]
        data.reset_index(inplace=True)
        if 'index' in data.columns: data.rename(columns={'index': 'Date'}, inplace=True)
        if 'Date' not in data.columns or 'Close' not in data.columns: return None
        data['Date'] = pd.to_datetime(data['Date'])
        data.dropna(subset=['Date', 'Close'], inplace=True)
        if data.empty: return None
        data.set_index('Date', inplace=True)
        data.sort_index(inplace=True)
        return data
    except Exception as e:
        print(f"[get_stock_data] Error for '{_ticker}': {e}")
        return None

class PredictionService:
    def __init__(self, random_state=42):
        self.random_state = random_state
    def _calculate_rsi(self, data, window=14):
        """Calculates Relative Strength Index (RSI) using Exponential Moving Average."""
        delta = data.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=window - 1, min_periods=window).mean()
        avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()

        with np.errstate(divide='ignore', invalid='ignore'):
            rs = avg_gain / avg_loss
        
        rsi = 100 - (100 / (1 + rs))
        return rsi.replace([np.inf, -np.inf], np.nan)

    def _split_and_scale_data(self, X, y, test_size=0.2):
        if len(X) < 20: 
            return [None] * 6
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        if X_train.empty or X_test.empty:
            return [None] * 6
            
        scaler_X = MinMaxScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler_X, X_test
        
    def _get_final_results_classification(self, predictions, y_test):
        results = {}
        for name, pred in predictions.items():
            metrics = {
                "Accuracy": accuracy_score(y_test, pred),
                "F1 Score (Macro)": f1_score(y_test, pred, average='macro', zero_division=0),
                "Precision (Macro)": precision_score(y_test, pred, average='macro', zero_division=0),
                "Recall (Macro)": recall_score(y_test, pred, average='macro', zero_division=0)
            }
            results[name] = metrics
        
        results_df = pd.DataFrame.from_dict(results, orient='index')
        if not results_df.empty:
            results_df = results_df.sort_values(by="F1 Score (Macro)", ascending=False)
            best_model_name = results_df.index[0]
            top_models_list = results_df.index.tolist()
        else:
            best_model_name, top_models_list = "N/A", []
        return results_df, best_model_name, top_models_list
        
    def predict_future_features(self, prophet_feature_models, feature_names, periods_to_forecast, last_historical_date):
        future_dates = pd.bdate_range(start=last_historical_date + timedelta(days=1), periods=periods_to_forecast)
        future_dates_df = pd.DataFrame({'ds': future_dates})
        predicted_features = pd.DataFrame(index=future_dates_df['ds'])
        for feature_name in feature_names:
            if feature_name in prophet_feature_models:
                m_feature = prophet_feature_models[feature_name]
                feature_forecast = m_feature.predict(future_dates_df)
                predicted_features[feature_name] = feature_forecast['yhat'].values
            else:
                predicted_features[feature_name] = 0.0
        return predicted_features

    def _run_hybrid_future_prediction_classification(self, trained_ml_models, scaler_X, feature_names, prophet_feature_models, periods_to_forecast, last_historical_date):
        future_features_df = self.predict_future_features(prophet_feature_models, feature_names, periods_to_forecast, last_historical_date)
        if future_features_df is None or future_features_df.empty: return None
        X_future_scaled = scaler_X.transform(future_features_df[feature_names])
        future_predictions = {}
        for name, model in trained_ml_models.items():
            if model:
                if hasattr(model, 'predict_proba'):
                    labels = model.predict(X_future_scaled)
                    probas = model.predict_proba(X_future_scaled)
                    future_predictions[name] = {
                        'labels': pd.Series(labels, index=future_features_df.index),
                        'probabilities': pd.DataFrame(probas, index=future_features_df.index, columns=model.classes_)
                    }
                else: 
                     labels = model.predict(X_future_scaled)
                     future_predictions[name] = { 'labels': pd.Series(labels, index=future_features_df.index) }
        return future_predictions

    def _run_classification_pipeline(self, ticker, features_list, horizon_days, threshold_up, threshold_down, periods_to_forecast_future, external_data=None):
        df_raw = get_stock_data(ticker, "2015-01-01", date.today().strftime("%Y-%m-%d"))
        if df_raw is None or df_raw.empty: return None
        
        df_features_full = df_raw.copy()
        close_price_full = df_features_full['Close']
        if 'SMA_20' in features_list: df_features_full['SMA_20'] = close_price_full.rolling(window=20).mean()
        if 'RSI_14' in features_list: df_features_full['RSI_14'] = self._calculate_rsi(close_price_full, window=14)
        if 'Volatility' in features_list: df_features_full['Volatility'] = close_price_full.rolling(window=20).std()
        if 'DGS10' in features_list:
            if external_data is not None and 'DGS10' in external_data.columns:
                print(f"[{ticker}] Joining pre-fetched DGS10 data.")
                df_features_full = df_features_full.join(external_data[['DGS10']])
            else:
                print(f"[{ticker}] Fetching DGS10 data individually.")
                try:
                    dgs10 = web.DataReader('DGS10', 'fred', df_features_full.index.min(), df_features_full.index.max())
                    df_features_full = df_features_full.join(dgs10)
                except Exception as e:
                    print(f"[{ticker}] Could not fetch DGS10 individually. Error: {e}")
                    df_features_full['DGS10'] = 0 # Fallback
    
            df_features_full['DGS10'].fillna(method='ffill', inplace=True)
            df_features_full['DGS10'].fillna(method='bfill', inplace=True)
            df_features_full['DGS10'].fillna(0, inplace=True)
        df_ml = df_features_full.copy()
        future_return = (df_ml['Close'].shift(-horizon_days) - df_ml['Close']) / df_ml['Close']
        
        df_ml['target'] = np.select(
            [future_return > threshold_up, future_return < -threshold_down],
            [2, 0], # Up: 2, Down: 0
            default=1 # Sideways: 1
        )
        
        df_ml.dropna(inplace=True) # Bây giờ mới dropna để chuẩn bị cho ML
        
        actual_features = [f for f in features_list if f in df_ml.columns]
        X, y = df_ml[actual_features], df_ml["target"]
        print(f"Class distribution for {ticker} ({horizon_days} days, Up > {threshold_up:.2%}, Down < {-threshold_down:.2%}):\n{y.value_counts(normalize=True)}")
        if X.empty or y.nunique() < 2: return None
        X_train_scaled, X_test_scaled, y_train, y_test, scaler_X, X_test = self._split_and_scale_data(X, y)
        if X_test is None: return None
        models = {
            "Logistic Regression": LogisticRegression(random_state=self.random_state, class_weight='balanced', max_iter=1000),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=self.random_state, n_jobs=-1, class_weight='balanced'),
            "XGBoost": XGBClassifier(random_state=self.random_state, use_label_encoder=False, eval_metric='mlogloss'),
            "SVC": SVC(kernel='rbf', probability=True, random_state=self.random_state, class_weight='balanced'),
            "KNN": KNeighborsClassifier(n_neighbors=10) # Tăng neighbors cho ổn định hơn
        }
    
        predictions, trained_models = {}, {}
        for name, model in models.items():
            if name == "XGBoost":
                sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
                model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
            else:
                model.fit(X_train_scaled, y_train)
            predictions[name] = model.predict(X_test_scaled)
            trained_models[name] = model
        results_df, best_model, top_models_list = self._get_final_results_classification(predictions, y_test)

        # 6. FORECAST FUTURE FEATURES WITH PROPHET
        prophet_feature_models = {}
        for feature in actual_features:
            feature_df = df_features_full[[feature]].dropna().reset_index()
            feature_df.rename(columns={'Date': 'ds', feature: 'y'}, inplace=True)
            if not feature_df.empty:
                try:
                    m = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True).fit(feature_df)
                    prophet_feature_models[feature] = m
                except Exception as e: print(f"Prophet error on {feature}: {e}")

        # 7. HYBRID FUTURE PREDICTION
        last_hist_date = X.index[-1]
        future_forecasts = self._run_hybrid_future_prediction_classification(trained_models, scaler_X, actual_features, prophet_feature_models, periods_to_forecast_future, last_hist_date)
        
        # 8. FORECAST BASELINE PRICE WITH PROPHET (FOR VISUALIZATION)
        future_price_baseline = None
        try:
            prophet_price_model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
            price_df_for_prophet = df_raw[['Close']].reset_index().rename(columns={'Date': 'ds', 'Close': 'y'})
            prophet_price_model.fit(price_df_for_prophet)
            
            future_dates_df = prophet_price_model.make_future_dataframe(periods=periods_to_forecast_future, freq='B') # <<< CẢI TIẾN 4
            price_forecast = prophet_price_model.predict(future_dates_df)
            
            future_price_baseline = price_forecast.set_index('ds').loc[last_hist_date:, 'yhat'].iloc[1:]
        except Exception as e:
            print(f"Could not generate Prophet price baseline: {e}")

        # 9. RETURN ALL RESULTS
        return {
            "results_df": results_df,
            "best_model": best_model,
            "top_models_list": top_models_list,
            "trained_models": trained_models,
            "test_data": {
                "X_test": X_test,
                "y_test": y_test,
                "dates": X_test.index,
                "actual_price": df_raw.loc[X_test.index, 'Close']
            },
            "predictions_on_test": predictions,
            "future_forecasts": future_forecasts,
            "future_price_baseline": future_price_baseline,
            "data_info": {
                "horizon": horizon_days,
                "threshold_up": threshold_up,
                "threshold_down": threshold_down,
                "class_distribution": y.value_counts(normalize=True).to_dict(),
                # Map lại để nhất quán: 0:Down, 1:Sideways, 2:Up
                "label_map": {0: 'Down', 1: 'Sideways', 2: 'Up'}
            }
        }

    def run_short_term_prediction(self, ticker, periods_to_forecast_future):
        features = ["Open", "High", "Low", "Volume", 'SMA_20', 'RSI_14', 'BB_Upper', 'BB_Lower']
        return self._run_classification_pipeline(
            ticker=ticker,
            features_list=features,
            horizon_days=5,
            threshold_up=0.02,
            threshold_down=0.02,
            periods_to_forecast_future=periods_to_forecast_future
        )

    def run_mid_term_prediction(self, ticker, periods_to_forecast_future):
        features = ["Volume", 'SMA_20', 'SMA_50', 'RSI_14', 'Volatility', 'DGS10']
        return self._run_classification_pipeline(
            ticker=ticker,
            features_list=features,
            horizon_days=21,
            threshold_up=0.04,
            threshold_down=0.03,
            periods_to_forecast_future=periods_to_forecast_future
    def run_batch_test(self, ticker_list, analysis_type='short_term'):
        """
        Chạy kiểm tra trên một danh sách các mã cổ phiếu và tổng hợp kết quả.
        Phiên bản này tải trước dữ liệu bên ngoài và có logic xử lý kết quả đúng.
        """
        all_results = []
        
        progress_container = st.container()
        progress_text_area = progress_container.empty()
        progress_bar = progress_container.progress(0)
        total_tickers = len(ticker_list)
        external_data = pd.DataFrame()
        try:
            print("[BATCH TEST] Pre-fetching DGS10 data for all tickers...")
            dgs10_data = web.DataReader('DGS10', 'fred', "2015-01-01", date.today())
            external_data['DGS10'] = dgs10_data['DGS10']
            print("[BATCH TEST] DGS10 data pre-fetched successfully.")
        except Exception as e:
            print(f"[BATCH TEST] WARNING: Could not pre-fetch DGS10 data. Error: {e}")
        for i, ticker in enumerate(ticker_list):
            try:
                progress_text = f"Đang xử lý mã {i+1}/{total_tickers}: {ticker}"
                progress_text_area.text(progress_text)
                progress_bar.progress((i + 1) / total_tickers)
                
                # Xác định các tham số dựa trên loại phân tích
                if analysis_type == 'short_term':
                    features = ["Open", "High", "Low", "Volume", 'SMA_20', 'RSI_14', 'BB_Upper', 'BB_Lower']
                    horizon, thresh_up, thresh_down = 5, 0.02, 0.02
                else: # mid_term
                    features = ["Volume", 'SMA_20', 'SMA_50', 'RSI_14', 'Volatility', 'DGS10']
                    horizon, thresh_up, thresh_down = 21, 0.04, 0.03
                
                # Gọi hàm pipeline với dữ liệu bên ngoài đã được tải trước
                result = self._run_classification_pipeline(
                    ticker=ticker,
                    features_list=features,
                    horizon_days=horizon,
                    threshold_up=thresh_up,
                    threshold_down=thresh_down,
                    periods_to_forecast_future=1,
                    external_data=external_data # <<< TRUYỀN DỮ LIỆU VÀO ĐÂY
                )
                
                # --- BƯỚC 3: KIỂM TRA KẾT QUẢ VÀ XỬ LÝ (PHẦN BỊ THIẾU) ---
                # Đây là khối logic quan trọng nhất đã bị thiếu trong code của bạn
                if result and 'best_model' in result:
                    # Trích xuất các biến cần thiết từ dictionary 'result'
                    best_model_name = result['best_model']
                    test_data = result['test_data']
                    predictions_on_test = result['predictions_on_test'][best_model_name]
                    label_map = result['data_info']['label_map']
                    # Sắp xếp tên nhãn để đảm bảo báo cáo nhất quán
                    target_names_from_map = [label_map[key] for key in sorted(label_map.keys())]

                    # Bây giờ mới tính toán báo cáo
                    report = classification_report(
                        test_data['y_test'], 
                        predictions_on_test, 
                        output_dict=True, 
                        zero_division=0,
                        labels=sorted(label_map.keys()),
                        target_names=target_names_from_map
                    )
                    # Thêm kết quả vào danh sách
                    all_results.append({
                        'ticker': ticker,
                        'best_model': best_model_name,
                        'up_precision': report.get('Up', {}).get('precision', 0),
                        'up_recall': report.get('Up', {}).get('recall', 0),
                        'up_f1_score': report.get('Up', {}).get('f1-score', 0),
                        'down_precision': report.get('Down', {}).get('precision', 0),
                        'down_recall': report.get('Down', {}).get('recall', 0),
                        'down_f1_score': report.get('Down', {}).get('f1-score', 0),
                        'accuracy': report.get('accuracy', 0)
                    })
                else:
                    # Ghi nhận trường hợp không có kết quả trả về
                    print(f"Không có kết quả hợp lệ cho mã {ticker}. Bỏ qua.")
                    all_results.append({'ticker': ticker, 'best_model': 'NO_DATA', 'accuracy': 0})
            except Exception as e:
                print(f"Lỗi nghiêm trọng khi xử lý mã {ticker} trong batch test: {e}")
                # Thêm mã này vào danh sách kết quả với giá trị lỗi để biết nó đã được xử lý
                all_results.append({'ticker': ticker, 'best_model': 'ERROR', 'accuracy': 0})
                continue
        progress_container.empty() # Xóa thanh tiến trình khi hoàn tất
        if not all_results:
            return None
        return pd.DataFrame(all_results)
