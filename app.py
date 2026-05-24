import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import os
import pickle
import plotly.graph_objects as go
import plotly.express as px
import datetime
from datetime import date
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm

try:
    from keras.models import load_model as keras_load_model
except Exception:
    keras_load_model = None

try:
    from tensorflow.keras.models import load_model as tf_load_model
except Exception:
    tf_load_model = None


ARTIFACT_DIR = "artifacts/lstm"


def _load_model_compat(model_path):
    """Try multiple Keras loaders to handle keras/tf.keras environment differences."""
    loaders = []
    if keras_load_model is not None:
        loaders.append(("keras", keras_load_model))
    if tf_load_model is not None:
        loaders.append(("tensorflow.keras", tf_load_model))

    if not loaders:
        raise RuntimeError("TensorFlow/Keras is not installed.")

    errors = []
    for loader_name, loader in loaders:
        try:
            return loader(model_path, compile=False)
        except Exception as ex:
            errors.append(f"{loader_name}: {ex}")

    raise RuntimeError(" | ".join(errors))


@st.cache_resource
def load_lstm_artifacts():
    """Load notebook-exported files used for LSTM inference in Streamlit."""
    required = {
        "model": os.path.join(ARTIFACT_DIR, "lstm_model.keras"),
        "scaler": os.path.join(ARTIFACT_DIR, "scaler.pkl"),
        "features": os.path.join(ARTIFACT_DIR, "features.pkl"),
        "last_sequence": os.path.join(ARTIFACT_DIR, "last_sequence.pkl")
    }

    missing = [name for name, path in required.items() if not os.path.exists(path)]
    if missing:
        return None, f"Missing LSTM artifact files: {', '.join(missing)}"

    if keras_load_model is None and tf_load_model is None:
        return None, "TensorFlow is not installed. Add tensorflow to requirements and install dependencies."

    try:
        model = _load_model_compat(required["model"])

        with open(required["scaler"], "rb") as f:
            scaler = pickle.load(f)

        with open(required["features"], "rb") as f:
            features = pickle.load(f)

        with open(required["last_sequence"], "rb") as f:
            last_sequence = pickle.load(f)

        metrics_path = os.path.join(ARTIFACT_DIR, "metrics.pkl")
        history_path = os.path.join(ARTIFACT_DIR, "history_payload.pkl")
        future_path = os.path.join(ARTIFACT_DIR, "future_payload.pkl")

        metrics = None
        history_payload = None
        future_payload = None

        if os.path.exists(metrics_path):
            with open(metrics_path, "rb") as f:
                metrics = pickle.load(f)

        if os.path.exists(history_path):
            with open(history_path, "rb") as f:
                history_payload = pickle.load(f)

        if os.path.exists(future_path):
            with open(future_path, "rb") as f:
                future_payload = pickle.load(f)

        payload = {
            "model": model,
            "scaler": scaler,
            "features": features,
            "last_sequence": np.array(last_sequence),
            "metrics": metrics,
            "history_payload": history_payload,
            "future_payload": future_payload
        }
        return payload, None
    except Exception as ex:
        err = str(ex)
        if "InputLayer" in err and "Unrecognized keyword arguments" in err:
            return None, (
                "Unable to load LSTM artifacts due to TensorFlow/Keras version mismatch. "
                "The model was saved with a different Keras config format than the app runtime supports. "
                "Re-generate artifacts using the same TensorFlow/Keras version as this app, "
                "or align app dependencies with the notebook environment. "
                f"Raw error: {err}"
            )
        return None, f"Unable to load LSTM artifacts: {err}"


def forecast_with_lstm(payload, forecast_days):
    """Use saved model + last sequence to generate multi-step forecast."""
    model = payload["model"]
    scaler = payload["scaler"]
    last_sequence = payload["last_sequence"].copy()

    future_scaled = []
    for _ in range(forecast_days):
        pred_input = last_sequence[-60:].reshape(1, last_sequence.shape[0], last_sequence.shape[1])
        next_scaled = model.predict(pred_input, verbose=0)[0, 0]
        future_scaled.append(next_scaled)

        # Keep non-target feature values from last known row, update close with prediction.
        appended_row = np.concatenate(([next_scaled], last_sequence[-1, 1:]))
        last_sequence = np.vstack([last_sequence[1:], appended_row])

    dummy_cols = np.zeros((len(future_scaled), last_sequence.shape[1] - 1))
    restore_input = np.concatenate((np.array(future_scaled).reshape(-1, 1), dummy_cols), axis=1)
    future_values = scaler.inverse_transform(restore_input)[:, 0]
    return future_values


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
@st.cache_data
def fetch_stock_data(symbol, start, end):
    return yf.download(symbol, start=start, end=end)

stocks_data = fetch_stock_data(selected_stock, start_date, end_date)

if stocks_data.empty:
    st.error("No data available for selected date range")
    st.stop()

# Ensure the index is named 'Date' before resetting
if stocks_data.index.name is None or stocks_data.index.name.lower() == 'date':
    stocks_data.index.name = 'Date'

# Reset Date from index to column
stocks_data = stocks_data.reset_index()

# Flatten multi-index columns if present
if isinstance(stocks_data.columns, pd.MultiIndex):
    stocks_data.columns = stocks_data.columns.get_level_values(0)

# Final safety check for Date column
if 'Date' not in stocks_data.columns:
    st.error(f"Unable to extract Date column. Available columns: {list(stocks_data.columns)}")
    st.stop()

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


# ================================
# LSTM PREDICTION SECTION
# ================================
st.header("LSTM Predictions (From Notebook Artifacts)")

lstm_payload, lstm_error = load_lstm_artifacts()
if lstm_error:
    st.warning(
        "LSTM artifacts are not ready yet. Run the last cell in STOCK MARKET PREDICITION PROJECT.ipynb "
        f"to generate files in '{ARTIFACT_DIR}'. Details: {lstm_error}"
    )
else:
    lstm_days = st.number_input(
        "Select Number of business days for LSTM forecasting",
        value=30,
        min_value=1,
        max_value=180,
        step=1,
        key="lstm_days"
    )

    lstm_forecast_values = forecast_with_lstm(lstm_payload, int(lstm_days))
    lstm_future_dates = pd.bdate_range(start=stocks_data['Date'].iloc[-1], periods=int(lstm_days) + 1)[1:]

    lstm_forecast_df = pd.DataFrame({
        'Date': lstm_future_dates,
        'LSTM_Prediction': lstm_forecast_values
    })

    st.write("## LSTM Forecast Table")
    st.write(lstm_forecast_df)

    lstm_fig = go.Figure()

    history_payload = lstm_payload.get("history_payload")
    if history_payload and "close_series" in history_payload:
        history_close = pd.Series(np.array(history_payload["close_series"]).flatten())  # historical close from notebook ticker run
        lstm_fig.add_trace(
            go.Scatter(
                x=history_close.index,
                y=history_close.values,
                mode='lines',
                name='LSTM Training History (Close)',
                line=dict(color='royalblue')
            )
        )

    if history_payload and "valid_predictions" in history_payload and "valid_close" in history_payload:
        valid_close = pd.Series(np.array(history_payload["valid_close"]).flatten())
        valid_pred = pd.Series(np.array(history_payload["valid_predictions"]).flatten())

        lstm_fig.add_trace(
            go.Scatter(
                x=valid_close.index,
                y=valid_close.values,
                mode='lines',
                name='LSTM Validation Close',
                line=dict(color='orange')
            )
        )
        lstm_fig.add_trace(
            go.Scatter(
                x=valid_pred.index,
                y=valid_pred.values,
                mode='lines',
                name='LSTM Validation Prediction',
                line=dict(color='green', dash='dash')
            )
        )

    lstm_fig.add_trace(
        go.Scatter(
            x=lstm_forecast_df['Date'],
            y=lstm_forecast_df['LSTM_Prediction'],
            mode='lines',
            name='LSTM Future Forecast',
            line=dict(color='red', dash='dot')
        )
    )

    lstm_fig.update_layout(
        title='LSTM Based Stock Forecast',
        xaxis_title='Date',
        yaxis_title='Price',
        width=1000,
        height=450
    )
    st.plotly_chart(lstm_fig)

    metrics = lstm_payload.get("metrics")
    if metrics:
        st.write("## LSTM Notebook Metrics")
        metrics_df = pd.DataFrame([metrics])
        st.write(metrics_df)