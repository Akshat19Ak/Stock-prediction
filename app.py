import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import datetime
from datetime import date, timedelta
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm


# ================================
# APP TITLE
# ================================
st.title("Stock Prediction App")
st.subheader("This app predicts the stock trends")


# ================================
# SIDEBAR INPUTS
# ================================
st.sidebar.header("Select Start and End date")

start_date = st.sidebar.date_input('Start Date', date(2016, 1, 1))
end_date = st.sidebar.date_input('End Date', datetime.date.today())

Stocks_symbol = [
    "RELIANCE.NS",
    "SBIN.NS",
    "AXISBANK.NS",
    "ITC.NS",
    "ASIANPAINT.NS",
    "HDFCBANK.NS"
]

selected_stock = st.sidebar.selectbox("Choose your stock", Stocks_symbol)


# ================================
# FETCH STOCK DATA
# ================================
stocks_data = yf.download(selected_stock, start=start_date, end=end_date)

if stocks_data.empty:
    st.error("No data available for selected date range")
    st.stop()

# Reset Date index
stocks_data.reset_index(inplace=True)

# Flatten multi-index columns if present
if isinstance(stocks_data.columns, pd.MultiIndex):
    stocks_data.columns = stocks_data.columns.get_level_values(0)

# Ensure Date is datetime
stocks_data['Date'] = pd.to_datetime(stocks_data['Date'])

st.write("Data from", start_date, "to", end_date)
st.write(stocks_data.tail(10))


# ================================
# VISUALIZATION
# ================================
st.header("Visualizing data")

fig = px.line(
    stocks_data,
    x='Date',
    y=['Open', 'Close', 'High', 'Low'],
    title='Different Variations of Stocks',
    width=800,
    height=400
)

st.plotly_chart(fig)


# ================================
# COLUMN SELECTION
# ================================
column = st.selectbox(
    "Select Column whose data being forecasted",
    ['Open', 'Close', 'High', 'Low']
)

data = stocks_data[['Date', column]]

st.write("Selected data")
st.write(data.tail(10))


# ================================
# DECOMPOSITION
# ================================
decomposition = seasonal_decompose(
    data[column],
    model='additive',
    period=5   # weekly trading pattern
)

trend_fig = px.line(
    x=stocks_data['Date'],
    y=decomposition.trend,
    title='Trend',
    width=800,
    height=400,
    labels={'x': 'Date', 'y': 'Price'}
)

trend_fig.update_traces(line_color='red')
st.plotly_chart(trend_fig)


# ================================
# MODEL PARAMETERS
# ================================
p = 1
d = 1
q = 2


# ================================
# FORECAST INPUT
# ================================
st.markdown(
    "<p style='color:green;font-size:35px;font-weight:bold;'>Forecasting Stocks Data</p>",
    unsafe_allow_html=True
)

forecast_period = st.number_input(
    "Select Number of days for forecasting",
    value=5,
    min_value=1
)


# ================================
# SARIMAX MODEL
# ================================
model = sm.tsa.statespace.SARIMAX(
    data[column],
    order=(p, d, q),
    seasonal_order=(p, d, q, 5),
    enforce_stationarity=False,
    enforce_invertibility=False
)

model_fit = model.fit(disp=False)


# ================================
# PREDICTIONS (BUSINESS DAYS)
# ================================
predictions = model_fit.get_forecast(steps=forecast_period)
predicted_mean = predictions.predicted_mean

future_dates = pd.bdate_range(
    start=data['Date'].iloc[-1],
    periods=forecast_period + 1
)[1:]

predictions_df = pd.DataFrame({
    'Date': future_dates,
    'predicted_mean': predicted_mean.values
})

st.write("## Predictions")
st.write(predictions_df)

st.write("## Actual Data")
st.write(data.tail())


# ================================
# FINAL COMPARISON PLOT
# ================================
fig_final = go.Figure()

fig_final.add_trace(
    go.Scatter(
        x=data['Date'],
        y=data[column],
        mode='lines',
        name='Actual',
        line=dict(color='green')
    )
)

fig_final.add_trace(
    go.Scatter(
        x=predictions_df['Date'],
        y=predictions_df['predicted_mean'],
        mode='lines',
        name='Predicted',
        line=dict(color='red')
    )
)

fig_final.update_layout(
    title='Actual vs Predicted',
    xaxis_title='Date',
    yaxis_title='Price',
    width=800,
    height=400
)

st.plotly_chart(fig_final)


# ================================
# SEPARATE GRAPHS
# ================================
if st.button("Show Separate Graphs"):

    actual_fig = px.line(
        x=data['Date'],
        y=data[column],
        title='Actual Prices',
        width=1000,
        height=400,
        labels={'x': 'Date', 'y': 'Price'}
    )

    actual_fig.update_traces(line_color='green')
    st.plotly_chart(actual_fig)

    pred_fig = px.line(
        x=predictions_df['Date'],
        y=predictions_df['predicted_mean'],
        title='Predicted Prices',
        width=1000,
        height=400,
        labels={'x': 'Date', 'y': 'Price'}
    )

    pred_fig.update_traces(line_color='red')
    st.plotly_chart(pred_fig)