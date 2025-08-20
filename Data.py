import streamlit as st
import yfinance as yf
import pandas as pd
import os
import plotly.express as px

# Define Paths
PATH = 'I:\\CEEMDAN_LSTM_WAVELET\\'
FIGURE_PATH = PATH + 'figures\\'
LOG_PATH = PATH + 'subset\\'
MODEL_PATH = PATH + 'model\\'
DATASET_FOLDER = PATH + 'dataset'

def run():
    st.title("📊 Stock Forecasting")
    st.subheader("This page shows the forecasted stock data.")

    # Sidebar controls
    st.sidebar.header("Select the parameter from below")

    # Use session_state to store the values
    if 'start_date' not in st.session_state:
        st.session_state.start_date = '2020-01-01'
    if 'end_date' not in st.session_state:
        st.session_state.end_date = '2021-12-31'
    if 'ticker' not in st.session_state:
        st.session_state.ticker = "BNII.JK"

    start_date = st.sidebar.date_input("Start Date", pd.to_datetime(st.session_state.start_date))
    end_date = st.sidebar.date_input("End Date", pd.to_datetime(st.session_state.end_date))
    ticker_list = ["BNII.JK", "AAPL"]
    ticker = st.sidebar.selectbox("Select the company", ticker_list, index=ticker_list.index(st.session_state.ticker))

    # Store these values in session_state
    st.session_state.start_date = start_date
    st.session_state.end_date = end_date
    st.session_state.ticker = ticker

    # Fetch data
    data = yf.download(ticker, start=start_date, end=end_date)

    # Clean MultiIndex if exists
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(0)

    data.reset_index(inplace=True)

    # Check for 'Adj Close' column and adjust
    if 'Adj Close' in data.columns:
        data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        data.drop(columns=['Adj Close'], inplace=True)
    else:
        data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']

    # Show data
    st.write("### Raw Data")
    st.dataframe(data)

    # Close price chart using plotly
    fig = px.line(data, x='Date', y='Close', title=f"Closing Price of {ticker}")
    st.plotly_chart(fig)

    # Column selection for forecasting
    column = st.selectbox("Select the column to be used for forecasting", data.columns[1:])
    subset = data[["Date", column]]
    st.write("### Selected Data for Forecasting")
    st.write(subset)