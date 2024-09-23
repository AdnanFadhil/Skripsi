import streamlit as st
import plotly.graph_objects as go
from backend import fetch_historical_data, generate_stock_predictions, fetch_company_info
from FetchData import indonesian_companies

# Create a dropdown menu for selecting the stock
st.sidebar.markdown("## **Select Stock**")
stock = st.sidebar.selectbox("Choose a stock", [company['ticker'] for company in indonesian_companies])

# Create a dropdown menu for selecting the period
st.sidebar.markdown("## **Select Period**")
period = st.sidebar.selectbox("Choose a period", ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"])

# Create a dropdown menu for selecting the interval
st.sidebar.markdown("## **Select Interval**")
interval = st.sidebar.selectbox("Choose an interval",
                                ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1d", "5d", "1wk", "1mo"])

# Fetch historical data for the selected stock and period
historical_data = fetch_historical_data(stock, period, interval)

# Create a candlestick chart for historical data
fig = go.Figure(data=[
    go.Candlestick(x=historical_data.index, open=historical_data['Open'], high=historical_data['High'],
                   low=historical_data['Low'], close=historical_data['Close'])])
fig.update_layout(xaxis_rangeslider_visible=False)
st.plotly_chart(fig, use_container_width=True)

# Generate stock predictions
predictions = generate_stock_predictions(historical_data)

# Create a line chart for stock predictions
fig = go.Figure(data=[go.Scatter(x=range(len(predictions)), y=predictions, name='LSTM')])
fig.update_layout(xaxis_rangeslider_visible=False)
st.plotly_chart(fig, use_container_width=True)

# Fetch company information
company_info = fetch_company_info(stock)

# Display company information
st.markdown("## **Company Information**")
st.write(f"**Security Code:** {company_info['Security Code']}")
st.write(f"**Issuer Name:** {company_info['Issuer Name']}")
st.write(f"**Security Id:** {company_info['Security Id']}")
st.write(f"**Security Name:** {company_info['Security Name']}")
st.write(f"**Status:** {company_info['Status']}")
st.write(f"**Group:** {company_info['Group']}")
st.write(f"**Face Value:** {company_info['Face Value']}")
st.write(f"**ISIN No:** {company_info['ISIN No']}")
st.write(f"**Industry:** {company_info['Industry']}")
st.write(f"**Instrument:** {company_info['Instrument']}")
st.write(f"**Sector Name:** {company_info['Sector Name']}")
st.write(f"**Industry New Name:** {company_info['Industry New Name']}")
st.write(f"**Igroup Name:** {company_info['Igroup Name']}")
st.write(f"**ISubgroup Name:** {company_info['ISubgroup Name']}")