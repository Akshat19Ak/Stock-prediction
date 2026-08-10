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

# ================================
# COLOUR PALETTE & PLOT HELPERS
# ================================
PLOT_TEMPLATE   = "plotly_dark"
COLOR_OPEN      = "#FFB703"   # amber / gold
COLOR_CLOSE     = "#06D6A0"   # mint green
COLOR_HIGH      = "#FF6B6B"   # coral red
COLOR_LOW       = "#C77DFF"   # violet / purple
COLOR_TREND     = "#FF6B6B"
COLOR_ACTUAL    = "#06D6A0"
COLOR_PREDICTED = "#FFD166"
COLOR_LSTM_HIST       = "#4361EE"
COLOR_LSTM_VAL_CLOSE  = "#F8961E"
COLOR_LSTM_VAL_PRED   = "#90BE6D"
COLOR_LSTM_FUTURE     = "#F94144"


def apply_common_layout(fig, title, xaxis_title="Date", yaxis_title="Price (₹)", legend_visible=True):
    """Apply a consistent, readable dark layout to every Plotly figure."""
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=20, color="white"),
            x=0.5,
            xanchor="center",
        ),
        template=PLOT_TEMPLATE,
        xaxis=dict(
            title=dict(text=xaxis_title, font=dict(size=14, color="#cccccc")),
            tickfont=dict(size=12, color="#cccccc"),
            showgrid=True,
            gridcolor="rgba(255,255,255,0.1)",
            zeroline=False,
        ),
        yaxis=dict(
            title=dict(text=yaxis_title, font=dict(size=14, color="#cccccc")),
            tickfont=dict(size=12, color="#cccccc"),
            showgrid=True,
            gridcolor="rgba(255,255,255,0.1)",
            zeroline=False,
        ),
        legend=dict(
            visible=legend_visible,
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=13, color="white"),
            bgcolor="rgba(30,30,30,0.7)",
            bordercolor="rgba(255,255,255,0.2)",
            borderwidth=1,
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=90, b=60, l=60, r=20),
        hovermode="x unified",
    )
    return fig

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
# PAGE CONFIG & HEADER
# ================================
st.set_page_config(
    page_title="Stock Prediction App",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Stock Market Price Prediction")
st.caption("Select a stock and date range from the sidebar. The app will forecast prices using SARIMAX and LSTM models.")
st.divider()


# ================================
# SIDEBAR INPUTS
# ================================
st.sidebar.header("⚙️ Configuration")
st.sidebar.markdown("Choose the stock and the date range for analysis.")

start_date = st.sidebar.date_input('Start Date', date(2016, 1, 1))
end_date   = st.sidebar.date_input('End Date', datetime.date.today())

Stocks_symbol = [
    "RELIANCE.NS",
    "SBIN.NS",
    "AXISBANK.NS",
    "ITC.NS",
    "ASIANPAINT.NS",
    "HDFCBANK.NS"
]

selected_stock = st.sidebar.selectbox("Choose your stock", Stocks_symbol)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Stocks legend:**\n"
    "- RELIANCE.NS — Reliance Industries\n"
    "- SBIN.NS — State Bank of India\n"
    "- AXISBANK.NS — Axis Bank\n"
    "- ITC.NS — ITC Limited\n"
    "- ASIANPAINT.NS — Asian Paints\n"
    "- HDFCBANK.NS — HDFC Bank"
)


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




# ================================
# VISUALIZATION — OHLC Price Chart
# ================================
st.header("📉 Historical Price Chart")
st.caption("🟡 Open  ·  🟢 Close  ·  🔴 High (dotted)  ·  🟣 Low (dashed)  — Click legend to toggle. Drag to zoom, double-click to reset.")

ohlc_fig = go.Figure()
ohlc_traces = [
    ("Open",  COLOR_OPEN,  "solid"),
    ("Close", COLOR_CLOSE, "solid"),
    ("High",  COLOR_HIGH,  "dot"),
    ("Low",   COLOR_LOW,   "dash"),
]
for col_name, color, dash in ohlc_traces:
    ohlc_fig.add_trace(
        go.Scatter(
            x=stocks_data['Date'],
            y=stocks_data[col_name],
            mode='lines',
            name=col_name,
            line=dict(color=color, dash=dash, width=1.5),
            hovertemplate=f"<b>{col_name}</b>: ₹%{{y:,.2f}}<extra></extra>",
        )
    )

ohlc_fig = apply_common_layout(ohlc_fig, f"{selected_stock} — Open / Close / High / Low Prices")
st.plotly_chart(ohlc_fig, use_container_width=True)
st.divider()


# ================================
# COLUMN SELECTION
# ================================
column = st.sidebar.selectbox(
    "Column to Forecast",
    ['Close', 'Open', 'High', 'Low'],
    help="Close price is most commonly used for stock forecasting."
)

data = stocks_data[['Date', column]]


# ================================
# DECOMPOSITION — TREND
# ================================
st.header("📈 Long-Term Trend")
st.caption("Seasonal decomposition (additive, 5-day period) isolates the underlying price direction, stripping out weekly noise.")

decomp_series = data[column].ffill().bfill()
decomposition = seasonal_decompose(
    decomp_series,
    model='additive',
    period=5,
    extrapolate_trend='freq'
)

trend_fig = go.Figure()
trend_fig.add_trace(
    go.Scatter(
        x=stocks_data['Date'],
        y=decomposition.trend,
        mode='lines',
        name='Trend',
        line=dict(color=COLOR_TREND, width=2),
        hovertemplate="<b>Trend</b>: ₹%{y:,.2f}<extra></extra>",
    )
)
trend_fig = apply_common_layout(
    trend_fig,
    f"{selected_stock} — {column} Long-Term Trend",
)
st.plotly_chart(trend_fig, use_container_width=True)
st.divider()


# ================================
# MODEL PARAMETERS
# ================================
p = 1
d = 1
q = 2


# ================================
# FORECAST INPUT
# ================================
st.header("🔮 SARIMAX Short-Term Forecast")
st.caption("SARIMAX (p=1, d=1, q=2, seasonal period=5) fitted live. Shaded band = 95% confidence interval.")

forecast_period = st.sidebar.number_input(
    "SARIMAX Forecast Days",
    value=5,
    min_value=1,
    max_value=60,
    help="Number of future business days to predict."
)


# ================================
# SARIMAX MODEL
# ================================
with st.spinner("Fitting SARIMAX model…"):
    model_sarimax = sm.tsa.statespace.SARIMAX(
        data[column],
        order=(p, d, q),
        seasonal_order=(p, d, q, 5),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    model_fit = model_sarimax.fit(disp=False)


# ================================
# PREDICTIONS (BUSINESS DAYS)
# ================================
predictions    = model_fit.get_forecast(steps=forecast_period)
predicted_mean = predictions.predicted_mean
conf_int       = predictions.conf_int(alpha=0.05)  # 95% CI

future_dates = pd.bdate_range(
    start=data['Date'].iloc[-1],
    periods=forecast_period + 1
)[1:]

predictions_df = pd.DataFrame({
    'Date':          future_dates,
    'predicted_mean': predicted_mean.values,
    'lower_95':      conf_int.iloc[:, 0].values,
    'upper_95':      conf_int.iloc[:, 1].values,
})




# ================================
# FINAL COMPARISON PLOT
# ================================
st.header(f"📊 {selected_stock} — Actual vs SARIMAX Forecast ({int(forecast_period)}-Day)")

fig_final = go.Figure()

fig_final.add_trace(
    go.Scatter(
        x=data['Date'],
        y=data[column],
        mode='lines',
        name=f'Actual {column}',
        line=dict(color=COLOR_ACTUAL, width=2),
        hovertemplate="<b>Actual</b>: ₹%{y:,.2f}<extra></extra>",
    )
)

fig_final.add_trace(
    go.Scatter(
        x=list(predictions_df['Date']) + list(predictions_df['Date'][::-1]),
        y=list(predictions_df['upper_95']) + list(predictions_df['lower_95'][::-1]),
        fill='toself',
        fillcolor='rgba(255,209,102,0.2)',
        line=dict(color='rgba(255,209,102,0)'),
        hoverinfo='skip',
        name='95% Confidence Interval',
        showlegend=True,
    )
)

fig_final.add_trace(
    go.Scatter(
        x=predictions_df['Date'],
        y=predictions_df['predicted_mean'],
        mode='lines+markers',
        name='SARIMAX Forecast',
        line=dict(color=COLOR_PREDICTED, width=3, dash='dot'),
        marker=dict(size=8, symbol='circle', color=COLOR_PREDICTED),
        hovertemplate="<b>Forecast</b>: ₹%{y:,.2f}<extra></extra>",
    )
)

fig_final = apply_common_layout(
    fig_final,
    f"{selected_stock} — {column}: Actual vs SARIMAX Forecast"
)
st.plotly_chart(fig_final, use_container_width=True)


st.divider()


# ================================
# LSTM PREDICTION SECTION
# ================================
st.header("🤖 LSTM Deep Learning Forecast")
st.caption("Pre-trained LSTM (60-day rolling window). Inference is millisecond-fast — model runs entirely offline from saved artifacts.")

lstm_payload, lstm_error = load_lstm_artifacts()
if lstm_error:
    st.warning(
        "⚠️ LSTM artifacts are not ready yet. Run the last cell in "
        "**STOCK MARKET PREDICITION PROJECT.ipynb** to generate files in "
        f"`{ARTIFACT_DIR}`.\n\nDetails: {lstm_error}"
    )
else:
    lstm_days = st.sidebar.number_input(
        "LSTM Forecast Days",
        value=30,
        min_value=1,
        max_value=180,
        step=1,
        key="lstm_days",
        help="Number of future business days to predict using the LSTM model."
    )

    with st.spinner("Running LSTM inference…"):
        lstm_forecast_values = forecast_with_lstm(lstm_payload, int(lstm_days))

    # =====================================================================
    # DATE RECONSTRUCTION — must happen BEFORE building the table/chart
    # The LSTM artifacts were saved from the Notebook's training run.
    # The model's "present" is the END of the validation split, NOT today.
    # All traces must be anchored to that timeline so the chart is continuous.
    # =====================================================================
    all_dates      = pd.to_datetime(stocks_data['Date'].values)
    history_payload = lstm_payload.get("history_payload")

    # --- Training history dates ---
    if history_payload and "close_series" in history_payload:
        history_vals = np.array(history_payload["close_series"]).flatten()
        n_history    = len(history_vals)
        if n_history <= len(all_dates):
            history_dates = all_dates[:n_history]           # first N dates
        else:
            history_dates = pd.bdate_range(end=all_dates[n_history - 1], periods=n_history)
    else:
        history_vals  = None
        history_dates = all_dates

    # --- Validation dates (immediately follow training) ---
    last_train_date = history_dates[-1]

    if history_payload and "valid_close" in history_payload and "valid_predictions" in history_payload:
        valid_close_vals = np.array(history_payload["valid_close"]).flatten()
        valid_pred_vals  = np.array(history_payload["valid_predictions"]).flatten()
        n_valid          = len(valid_close_vals)
        valid_dates      = pd.bdate_range(start=last_train_date, periods=n_valid + 1)[1:]
        forecast_anchor  = valid_dates[-1]   # future forecast starts HERE
    else:
        valid_close_vals = None
        valid_pred_vals  = None
        valid_dates      = None
        forecast_anchor  = last_train_date

    # --- Future forecast dates (anchored to end of validation, NOT today) ---
    lstm_future_dates = pd.bdate_range(start=forecast_anchor, periods=int(lstm_days) + 1)[1:]

    lstm_forecast_df = pd.DataFrame({
        'Date': lstm_future_dates,
        'LSTM_Prediction': lstm_forecast_values
    })

    # --- Bridge: actual prices from validation-end → today (fills the gap) ---
    # Find the index in all_dates that is closest to / just after forecast_anchor
    bridge_mask  = all_dates > forecast_anchor
    bridge_dates = all_dates[bridge_mask]
    if 'Close' in stocks_data.columns and len(bridge_dates) > 0:
        bridge_vals = stocks_data.loc[bridge_mask, 'Close'].values
    else:
        bridge_dates = None
        bridge_vals  = None

    st.caption("🔵 Training  ·  🟠 Validation Actual  ·  🟢 Validation Prediction  ·  ⚪ Recent Actual  ·  🔴 Future Forecast")

    # =====================================================================
    # CHART
    # =====================================================================
    lstm_fig = go.Figure()

    # 1. Training history
    if history_vals is not None:
        lstm_fig.add_trace(
            go.Scatter(
                x=history_dates,
                y=history_vals,
                mode='lines',
                name='Training History (Close)',
                line=dict(color=COLOR_LSTM_HIST, width=1.5),
                hovertemplate="<b>Train Close</b>: ₹%{y:,.2f}<extra></extra>",
            )
        )

    # 2. Validation actual
    if valid_close_vals is not None:
        lstm_fig.add_trace(
            go.Scatter(
                x=valid_dates,
                y=valid_close_vals,
                mode='lines',
                name='Validation Actual (Close)',
                line=dict(color=COLOR_LSTM_VAL_CLOSE, width=2),
                hovertemplate="<b>Val Actual</b>: ₹%{y:,.2f}<extra></extra>",
            )
        )

    # 3. Validation prediction
    if valid_pred_vals is not None:
        lstm_fig.add_trace(
            go.Scatter(
                x=valid_dates,
                y=valid_pred_vals,
                mode='lines',
                name='Validation Prediction',
                line=dict(color=COLOR_LSTM_VAL_PRED, width=2, dash='dash'),
                hovertemplate="<b>Val Prediction</b>: ₹%{y:,.2f}<extra></extra>",
            )
        )

    # 4. Bridge trace — real prices from validation-end to today
    if bridge_dates is not None and len(bridge_dates) > 0:
        lstm_fig.add_trace(
            go.Scatter(
                x=bridge_dates,
                y=bridge_vals,
                mode='lines',
                name='Recent Actual (Bridge to Today)',
                line=dict(color='#AAAAAA', width=1.5),
                hovertemplate="<b>Recent Close</b>: ₹%{y:,.2f}<extra></extra>",
            )
        )

    # 5. Future forecast — anchored just after bridge ends (= today)
    lstm_fig.add_trace(
        go.Scatter(
            x=lstm_future_dates,
            y=lstm_forecast_values,
            mode='lines+markers',
            name=f'Future Forecast (next {int(lstm_days)} days)',
            line=dict(color=COLOR_LSTM_FUTURE, width=3, dash='dot'),
            marker=dict(size=6, symbol='circle', color=COLOR_LSTM_FUTURE),
            hovertemplate="<b>LSTM Forecast</b>: ₹%{y:,.2f}<extra></extra>",
        )
    )

    lstm_fig = apply_common_layout(
        lstm_fig,
        f"{selected_stock} — LSTM Deep Learning Forecast (Continuous Timeline)"
    )
    lstm_fig.update_layout(
        height=540,
        margin=dict(t=90, b=170, l=60, r=20),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.28,
            xanchor="center",
            x=0.5,
            font=dict(size=11, color="white"),
            bgcolor="rgba(30,30,30,0.8)",
            bordercolor="rgba(255,255,255,0.3)",
            borderwidth=1,
            tracegroupgap=4,
        )
    )
    st.plotly_chart(lstm_fig, use_container_width=True)

    metrics = lstm_payload.get("metrics")
    if metrics:
        st.markdown("#### 📐 Model Accuracy — Validation Set")
        st.caption("RMSE & MAE: prediction error in ₹ (lower = better). R² closer to 1.0 = better fit.")
        st.dataframe(pd.DataFrame([metrics]), use_container_width=True)

st.divider()
st.caption(
    "📌 **Disclaimer:** This app is for educational purposes only. "
    "Stock price predictions are not financial advice. Past performance does not guarantee future results."
)