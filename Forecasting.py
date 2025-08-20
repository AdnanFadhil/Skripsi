import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import os
from datetime import timedelta
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow import keras
import plotly.express as px
import plotly.graph_objects as go
from another_utils import declare_path, declare_vars, emd_decom_wavelet, get_integration_form, integrate, declare_LSTM_vars, Respective_LSTM_Testing, load_and_preprocess_data, predict_future, evl,plot_direction_accuracy, analyze_comparison_df

def backtest(df_real, df_pred, periods=100, verbose=True):
    """
    Compare predictions to real data only where actual values are available.
    Automatically adjusts periods if real data is insufficient.
    """
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

    max_len = min(len(df_real), len(df_pred), periods)
    if max_len == 0:
        st.warning("⚠️ No overlapping data available for backtesting.")
        return None

    y_true = df_real[-max_len:].values
    y_pred = df_pred[:max_len]

    try:
        r2 = r2_score(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
    except Exception as e:
        st.error(f"❌ Error calculating metrics: {str(e)}")
        return None

    if verbose:
        st.subheader("📉 Backtest Results (Using Available Real Data)")
        st.write(f"**Periods Compared:** {max_len}")
        st.write(f"**R² Score:** {r2:.4f}")
        st.write(f"**RMSE:** {rmse:.4f}")
        st.write(f"**MAE:** {mae:.4f}")

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(y_true, label='Actual', color='blue')
        ax.plot(y_pred, label='Predicted', color='orange')
        ax.set_title("Backtest Comparison: Actual vs Predicted")
        ax.legend()
        st.pyplot(fig)

    return {"R2": r2, "RMSE": rmse, "MAE": mae}



# Define Paths
PATH = 'I:\\Streamlit_test\\'
FIGURE_PATH = PATH + 'figures\\'
LOG_PATH = PATH + 'subset\\'
MODEL_PATH = PATH + 'model\\'
DATASET_FOLDER = PATH + 'dataset'

# Global variables (ensure these are set before calling declare_vars)
MODE = 'ceemdan'  # Decomposition mode, e.g., 'emd', 'ceemdan', 'eemd'
FORM = ''  # Integration form, if applicable
DATE_BACK = 30  # The number of previous days related to today
PERIODS = 100  # The length of the days to forecast
EPOCHS = 200  # Default LSTM epochs (will be overwritten by user input)
PATIENCE = 10  # Patience for early stopping, suggest 1-20

# Declare LSTM model variables
CELLS = 32
DROPOUT = 0.2 
OPTIMIZER_LOSS = 'mse'
BATCH_SIZE = 16
VALIDATION_SPLIT = 0.2
VERBOSE = 1
SHUFFLE = True

LSTM_MODEL = None 

# IMF Weight Configuration (add near other globals)
IMF_WEIGHTS = {
    'co-imf0': 0.30,  # High-frequency (reduced weight due to poor R²)
    'co-imf1': 0.65,  # Mid-frequency (your best performer)
    'co-imf2': 0.5   # Low-frequency
}

def run():
    st.title("📊 Stock Forecasting")
    st.subheader("This page shows the forecasted stock data.")

    # Sidebar controls
    st.sidebar.header("Select the parameter from below")

    # Initialize session state
    if 'start_date' not in st.session_state:
        st.session_state.start_date = '2021-01-01'
    if 'end_date' not in st.session_state:
        st.session_state.end_date = '2025-12-31'
    if 'ticker' not in st.session_state:
        st.session_state.ticker = "BNII.JK"
    if 'forecasting' not in st.session_state:
        st.session_state.forecasting = False

    start_date = st.sidebar.date_input("Start Date", pd.to_datetime(st.session_state.start_date))
    
    end_date = st.sidebar.date_input("End Date", pd.to_datetime(st.session_state.end_date))

    if pd.Timestamp(end_date).date() >= pd.Timestamp.today().date():
        st.warning("⚠️ You selected a forecasting end date that includes today or future dates. "
               "Some evaluations (like backtest) may not have complete real data.")

    ticker = st.sidebar.text_input("Enter company ticker", value=st.session_state.ticker)

    # Let user pick the number of epochs
    global EPOCHS
    EPOCHS = st.sidebar.slider("⏳ LSTM Epochs", 10, 1000, 500, step=10)
    backtest_enabled = st.sidebar.checkbox("🔁 Enable Backtest Evaluation", value=True)

    st.session_state.start_date = start_date
    st.session_state.end_date = end_date
    st.session_state.ticker = ticker

    # Forecast/Stop toggle button
    button_label = "🟢 Forecast" if not st.session_state.forecasting else "🔴 Stop"
    if st.sidebar.button(button_label):
        st.session_state.forecasting = not st.session_state.forecasting

    if not st.session_state.forecasting:
        st.info("Click '🟢 Forecast' to start.")
        return

    try:
        with st.status("🚀 Running forecasting pipeline...", expanded=True) as status:
            # ---- Data Acquisition ----
            st.write("📥 Downloading market data...")
            data = yf.download(ticker, start=start_date, end=end_date)
            
            if data.empty:
                st.error("No data found. Please check your date range or ticker.")
                st.session_state.forecasting = False
                return

            # ---- Data Preprocessing ----
            st.write("🛠 Processing raw data...")
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel(0)
            
            data.reset_index(inplace=True)
            if 'Adj Close' in data.columns:
                data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
                data.drop(columns=['Adj Close'], inplace=True)
            else:
                data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']

            base_ticker = ticker.replace(".", "")
            dataset_path = os.path.join(DATASET_FOLDER, f"{base_ticker}.csv")
            data.to_csv(dataset_path, index=False)

            # ---- Data Visualization ----
            st.write("📊 Visualizing raw data...")
            
            st.dataframe(data.tail(10), height=300)

            
            fig = px.line(data, x='Date', y='Close', title=f"{ticker} Closing Prices")
            st.plotly_chart(fig, use_container_width=True)

            # ---- Decomposition ----
            st.write("🔍 Performing CEEMDAN decomposition...")
            declare_path(ticker)
            declare_vars(mode=MODE, form=FORM, data_back=DATE_BACK, 
                        periods=PERIODS, epochs=EPOCHS, patience=PATIENCE)

            imfs_wvlt = emd_decom_wavelet(trials=10, base_ticker=base_ticker)
            num_imfs = imfs_wvlt.shape[1]
            
            # Show decomposition results without nested expander
            st.write(f"🕵️ Found {num_imfs} IMF components")

            st.dataframe(imfs_wvlt)

            # Show decomposition image
            image_path = os.path.join(FIGURE_PATH, f"CEEMDAN_{base_ticker.replace('.', '').upper()}.svg")
            if os.path.exists(image_path):
                st.image(image_path, 
                        caption=f"CEEMDAN Decomposition - {base_ticker.upper()}",
                        use_column_width=True)
            else:
                st.warning(f"CEEMDAN image not found at {image_path}")

            # ---- Integration ----
            st.write("🧩 Integrating components...")
            inte_form = get_integration_form(num_imfs)
            if not inte_form:
                st.error(f"No default integration form for {num_imfs} IMFs.")
                st.session_state.forecasting = False
                return

            co_imfs = integrate(df=imfs_wvlt, inte_form=inte_form, ticker=ticker)
            integrated_path = os.path.join(LOG_PATH, f"ceemdan_{base_ticker}_integrated.csv")
            co_imfs.to_csv(integrated_path, index=False)
            
            # Show integrated components
            st.write("📅 Integrated Co-IMFs:")
            st.dataframe(co_imfs)

            # ---- LSTM Forecasting ----
            st.write("🧠 Training LSTM models...")
            declare_LSTM_vars()
            
            with st.spinner(f"Training {len([col for col in co_imfs.columns if col.startswith('co-imf')])} LSTM models..."):
                df_pred, model, scalarY = Respective_LSTM_Testing(
                    df=co_imfs, 
                    base_ticker=base_ticker
                )

            # ---- Results Processing ----
            st.write("📈 Generating forecasts...")
            df_res = pd.read_csv(f"{LOG_PATH}respective_ceemdan_{base_ticker}_pred.csv")
            res_pred = df_res.T.sum().values
        # Optional Backtest Evaluation
        if backtest_enabled:
            st.write("📊 Performing backtest using available actual data...")
            actual_series = data.set_index('Date')['Close']
            backtest(actual_series, res_pred, periods=PERIODS)

                
            # Normalize for evaluation
            series = data.set_index('Date')['Close']
            rate = series.max()-series.min()
            series_nor = (series-series.min())/float(rate)
            res_pred_nor = (res_pred-series.min())/float(rate)
                
                # Evaluation metrics
            df_evl = evl(series_nor[-PERIODS:].values, res_pred_nor, scale='Close')
                
            # ---- Future Predictions ----
            st.write("🔮 Predicting future values...")
            csv_path = os.path.join(DATASET_FOLDER, f"{base_ticker}.csv")
            series_full, series_scaled, _ = load_and_preprocess_data(csv_path, scalarY=scalarY)
                
            future_preds = predict_future(
                    model, 
                    series_scaled, 
                    scalarY, 
                    days=30
                )
                
                
            # Create future dates
            last_date = series.index[-1]

            # Ambil data trading ke depan (minimal 60 hari agar aman untuk diambil 30 hari trading)
            future_market_data = yf.download(ticker, start=last_date + timedelta(days=1), period='60d')

            if future_market_data.empty or len(future_market_data) < 30:
                st.warning("❌ Tidak cukup data trading ke depan untuk evaluasi akurasi.")
                return

            future_market_data.reset_index(inplace=True)
            future_dates = future_market_data['Date'].iloc[:30].tolist()

                
            # ---- Results Visualization ----
            st.write("🎨 Preparing visualizations...")
                
                
            # 1. Main forecast plot
            fig1, ax1 = plt.subplots(figsize=(12, 6))
            ax1.plot(series.index, series.values, label="Historical", color='#0070C0')
            ax1.plot(future_dates, future_preds, label="30-Day Forecast", 
                        color='#F27F19', linestyle='--')
            ax1.set_title(f"{ticker} Price Forecast\n(R²: {df_evl.get('R2', 0):.2f} | RMSE: {df_evl.get('RMSE', 0):.2f})")
            ax1.legend()
            st.pyplot(fig1)
                
            # 4. Future predictions table
            st.write("📅 30-Day Forecast Details:")
            future_df = pd.DataFrame({
                    'Date': future_dates,
                    'Predicted Close': future_preds.flatten(),
                    'Change %': np.concatenate([
                        [np.nan],
                        (future_preds.flatten()[1:] / future_preds.flatten()[:-1] - 1) * 100
                    ])
                })
            # Format the table
            st.dataframe(
                    future_df.style.format({
                        "Predicted Close": "{:.2f}",
                        "Change %": "{:.2f}%"
                    }).applymap(
                        lambda x: 'color: green' if isinstance(x, str) and '+' in x else (
                            'color: red' if isinstance(x, str) and '-' in x else ''
                        ),
                        subset=['Change %']
                    )
                )
                
                # In your existing code after getting predictions:
            st.write("## Directional Accuracy Analysis")

                # Get actual and predicted prices
            actual_prices = series.values[-PERIODS:]  # Use your actual variable names
            predicted_prices = res_pred[:PERIODS]     # Use your predicted variable names

                # Plot the directional accuracy
            direction_fig = plot_direction_accuracy(actual_prices, predicted_prices)
            st.plotly_chart(direction_fig, use_container_width=True)

                # Calculate and display accuracy metrics
            correct_directions = sum(
                    (np.diff(actual_prices) * np.diff(predicted_prices)) >= 0
                )
            accuracy = 100 * correct_directions / (len(actual_prices)-1)

            st.metric("Direction Prediction Accuracy", f"{accuracy:.1f}%")
                            
                
            # 2. Enhanced Original vs Ensemble Prediction Comparison
            st.subheader("Original vs Ensemble Prediction Comparison")
                
                # Inverse scaling for all predictions
            original_values = series.values[-PERIODS:]
                
                # Inverse transform the normalized predictions
            if isinstance(scalarY, MinMaxScaler):
                    ensemble_pred = scalarY.inverse_transform(res_pred_nor.reshape(-1, 1)).flatten()
            else:
                    ensemble_pred = res_pred_nor * (series.max()-series.min()) + series.min()
                
                # Create comparison DataFrame
            comparison_df = pd.DataFrame({
                    'Date': series.index[-PERIODS:],
                    'Original': original_values,
                    'Ensemble Prediction': ensemble_pred,
                    'Error': ensemble_pred - original_values
                })
                
                # Create the comparison plot
            fig_comparison = go.Figure()
                
                # Add original values
            fig_comparison.add_trace(go.Scatter(
                    x=comparison_df['Date'],
                    y=comparison_df['Original'],
                    name='Original Values',
                    line=dict(color='#0070C0', width=2),
                    mode='lines'
                ))
                
                # Add ensemble predictions
            fig_comparison.add_trace(go.Scatter(
                    x=comparison_df['Date'],
                    y=comparison_df['Ensemble Prediction'],
                    name='Ensemble Prediction',
                    line=dict(color='#F27F19', width=2, dash='dot'),
                    mode='lines'
                ))
                
                # Add error as bar chart (secondary y-axis)
            fig_comparison.add_trace(go.Bar(
                    x=comparison_df['Date'],
                    y=comparison_df['Error'],
                    name='Prediction Error',
                    marker_color=np.where(comparison_df['Error'] >= 0, 'green', 'red'),
                    opacity=0.3,
                    yaxis='y2'
                ))
                
                # Update layout with dual y-axes
            fig_comparison.update_layout(
                    title=f'{ticker} - Original vs Ensemble Prediction',
                    xaxis_title='Date',
                    yaxis_title='Price',
                    yaxis2=dict(
                        title='Error',
                        overlaying='y',
                        side='right',
                        showgrid=False
                    ),
                    hovermode='x unified',
                    legend=dict(
                        orientation='h',
                        yanchor='bottom',
                        y=1.02,
                        xanchor='right',
                        x=1
                    ),
                    plot_bgcolor='rgba(240,240,240,0.8)'
                )
                
                # Add R² and RMSE annotations
            fig_comparison.add_annotation(
                    x=0.05,
                    y=0.95,
                    xref='paper',
                    yref='paper',
                    text=f"R²: {df_evl.get('R2', 0):.3f}<br>RMSE: {df_evl.get('RMSE', 0):.3f}",
                    showarrow=False,
                    bgcolor='white',
                    bordercolor='black',
                    borderwidth=1
                )
                
            st.plotly_chart(fig_comparison, use_container_width=True)
                
                # 3. Error Distribution Analysis
            st.subheader("Prediction Error Analysis")
                
            col1, col2 = st.columns(2)
                
            with col1:
                    # Error distribution plot
                    fig_error_dist = px.histogram(
                        comparison_df,
                        x='Error',
                        nbins=30,
                        title='Error Distribution',
                        color_discrete_sequence=['#FF7F0E'],
                        marginal='box'
                    )
                    fig_error_dist.update_layout(
                        xaxis_title='Prediction Error',
                        yaxis_title='Count'
                    )
                    st.plotly_chart(fig_error_dist, use_container_width=True)
                
            with col2:
                    # Error metrics
                    error_metrics = {
                        'Mean Error': comparison_df['Error'].mean(),
                        'Std Dev': comparison_df['Error'].std(),
                        'Max Overestimation': comparison_df['Error'].max(),
                        'Max Underestimation': comparison_df['Error'].min(),
                        'Mean Absolute Error': (abs(comparison_df['Error'])).mean()
                    }
                    st.metric(label="Average Error", value=f"{error_metrics['Mean Error']:.2f}")
                    st.metric(label="Error Variability", value=f"{error_metrics['Std Dev']:.2f}")
                    st.metric(label="Worst Overestimation", value=f"{error_metrics['Max Overestimation']:.2f}")
                    st.metric(label="Worst Underestimation", value=f"{error_metrics['Max Underestimation']:.2f}")
                    st.metric(label="MAE (Mean Absolute Error)", value=f"{error_metrics['Mean Absolute Error']:.2f}")
                
                
            # === Compare Forecast with Real Data (if available) ===
            st.subheader("📥 Real Data Comparison During Forecast Period")

            # Tentukan rentang tanggal
            forecast_start_date = future_dates[0]
            forecast_end_date = future_dates[-1]

            # Ambil data aktual dari Yahoo Finance
            st.write(f"📡 Mengambil data aktual dari {forecast_start_date.date()} hingga {forecast_end_date.date()} ...")
            # Ambil data real
            real_data = yf.download(ticker, start=forecast_start_date, end=forecast_end_date + timedelta(days=1))

            # Reset index supaya 'Date' jadi kolom biasa
            real_data.reset_index(inplace=True)

            # Jika kolomnya MultiIndex, flatten dulu
            if isinstance(real_data.columns, pd.MultiIndex):
                real_data.columns = ['_'.join([str(i) for i in col if i]).strip('_') for col in real_data.columns]

            # Debug tampilkan kolom hasil flatten
            st.write("Kolom data real setelah flatten:", real_data.columns.tolist())

            # Cari kolom yang mengandung 'Close' dan 'Date'
            date_col = [col for col in real_data.columns if 'Date' in col][0]
            close_col = [col for col in real_data.columns if 'Close' in col][0]

            # Buat DataFrame actual_df dengan kolom yang sesuai dan rename
            actual_df = real_data[[date_col, close_col]].copy()
            actual_df.columns = ['Date', 'Actual Close']


            if real_data.empty:
                st.warning("⚠️ Data aktual tidak tersedia untuk periode forecast.")
            else:
                real_data.reset_index(inplace=True)
                # Deteksi kolom 'Date' dan kolom yang mengandung 'Close'
                date_col = [col for col in real_data.columns if 'Date' in col][0]
                close_col = [col for col in real_data.columns if 'Close' in col][0]  # misalnya: 'Close_BNII.JK'

                # Buat DataFrame baru dan ganti nama kolom
                actual_df = real_data[[date_col, close_col]].copy()
                actual_df.columns = ['Date', 'Actual Close']

                
                # Buat DataFrame prediksi
                forecast_df = pd.DataFrame({
                    'Date': future_dates,
                    'Predicted Close': future_preds.flatten()[:len(future_dates)]
                })

                print(forecast_df.index)
                print(forecast_df.columns)
                print(actual_df.index)
                print(actual_df.columns)
                print(type(forecast_df.index))
                print(type(actual_df.index))
                print(isinstance(actual_df.index, pd.MultiIndex))

                # Lakukan inner join agar hanya tanggal yang tersedia di kedua data yang digunakan
                comparison_df = pd.merge(forecast_df, actual_df, on='Date', how='inner')

                if comparison_df.empty:
                    st.warning("⚠️ Tidak ada tanggal yang cocok antara data prediksi dan data aktual.")
                else:
                    st.write("✅ Analisis hasil forecast 30 hari ke depan:")
                    analyze_comparison_df(comparison_df)
                
                
                # Add download button
            csv = future_df.to_csv(index=False)
            st.download_button(
                    label="Download Forecast Data",
                    data=csv,
                    file_name=f"{ticker}_forecast.csv",
                    mime='text/csv'
                )
                
                # Save results
            future_path = os.path.join(DATASET_FOLDER, f"{base_ticker}_future_predictions.csv")
            future_df.to_csv(future_path, index=False)
                
            status.update(label="✅ Forecasting complete!", state="complete")
                
    except Exception as e:
        st.session_state.forecasting = False
        st.error(f"""
        ❌ Forecasting failed with error:
        {str(e)}
        """, icon="🚨")
        raise e

    st.session_state.forecasting = False
    

