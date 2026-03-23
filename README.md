# Stock Prediction App

This repository contains a Streamlit stock forecasting application with:
- SARIMAX based forecasting from the live selected stock data
- LSTM based forecasting integrated from notebook-generated artifacts

## Project Files
- app.py: Streamlit UI and forecasting logic
- STOCK MARKET PREDICITION PROJECT.ipynb: LSTM training notebook
- requirements.txt: Minimal dependencies to run the app
- Readme.txt: Original run notes

## How to Run
1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run Streamlit:

```bash
streamlit run app.py
```

## LSTM Model Merge (Notebook -> Streamlit)
This repository now includes the LSTM flow from STOCK MARKET PREDICITION PROJECT.ipynb in the Streamlit UI.

### What was merged
- A new LSTM section was added in the app UI.
- The app now loads notebook-generated artifacts from artifacts/lstm.
- The app shows:
  - LSTM forecast table
  - LSTM forecast graph
  - Notebook evaluation metrics (if available)

### Required artifact generation step
Before LSTM predictions appear in Streamlit, run the last code cell in STOCK MARKET PREDICITION PROJECT.ipynb.
That last cell exports these artifacts:
- artifacts/lstm/lstm_model.keras
- artifacts/lstm/scaler.pkl
- artifacts/lstm/features.pkl
- artifacts/lstm/last_sequence.pkl
- artifacts/lstm/metrics.pkl
- artifacts/lstm/history_payload.pkl
- artifacts/lstm/future_payload.pkl

### Why this approach
- Training an LSTM inside Streamlit on every app run is slow.
- The notebook is used for model training/export.
- Streamlit is used for fast inference and visualization.

### Important note
The notebook currently trains on ticker GOOGL by default. The LSTM section in Streamlit will therefore reflect that trained model unless you retrain/export artifacts for another ticker.
