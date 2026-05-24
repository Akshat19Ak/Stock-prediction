# 📈 Stock Market Price Prediction System
*An End-to-End Hybrid Forecasting Application (SARIMAX + LSTM)*

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit)
![TensorFlow](https://img.shields.io/badge/TensorFlow-CPU_Optimized-FF6F00?style=for-the-badge&logo=tensorflow)
![Statsmodels](https://img.shields.io/badge/Statsmodels-SARIMAX-8CA1AF?style=for-the-badge)

Welcome to the **Stock Market Price Prediction System**, a production-grade, highly interactive web application designed to forecast financial time-series data. 

Rather than relying on a single mathematical approach, this project implements a **Dual-Model Architecture**. It combines the rapid, interpretable forecasting of classical statistical models (SARIMAX) with the deep, non-linear pattern recognition of Long Short-Term Memory (LSTM) neural networks.

---

## 🌟 Why This Project Stands Out (Core Features)

This repository was engineered with production scaling, cloud limitations, and user experience in mind. Here is what makes this project unique in chronological order of the data pipeline:

### 1. Robust Data Ingestion & Aggressive Caching
- **Live Market Data:** Integrates with the `yfinance` API to pull real-time historical data for major market tickers.
- **`@st.cache_data` Implementation:** Web applications naturally trigger full-script re-runs on user interaction. By intelligently caching the data payloads, the app completely eliminates redundant API calls, dropping load times to milliseconds and preventing Yahoo Finance IP rate-limiting.

### 2. Dual-Model Mathematical Engine
- **SARIMAX (Classical Auto-Regressive Modeling):** Automatically decomposes the data into Trend, Seasonality, and Residuals. Fits a SARIMAX model live in the browser session to provide instant, highly interpretable forecasts.
- **LSTM (Deep Learning Recurrent Neural Networks):** Utilizes a sophisticated TensorFlow/Keras LSTM architecture to capture hidden, long-term market dependencies. 

### 3. "Train Offline, Infer Online" Artifact Pipeline
- Training deep neural networks in a web server is an anti-pattern that leads to UI timeouts. 
- **The Solution:** The LSTM is trained offline in a dedicated Jupyter environment. The weights, metrics, and data scalers are serialized and exported to an `artifacts/` directory.
- The Streamlit application utilizes `@st.cache_resource` to load this 3.5MB model exactly once into RAM. Inference is executed in real-time, keeping the UI blazingly fast.

### 4. Cloud-Optimized for Zero-Cost Deployments
- Standard TensorFlow setups can add significant footprint and startup overhead.
- **Current Setup:** This repository uses `tensorflow==2.21.0` (CPU inference on this project setup), plus caching and prebuilt artifacts to keep runtime responsive and deployment-friendly.

### 5. Interactive, Dynamic Visualizations
- Migrated away from static Matplotlib images to fully dynamic **Plotly Express & Graph Objects**.
- Users can zoom, pan, hover for exact data points, and isolate specific model predictions directly on the dashboard.

---

## 🏗️ Technical Architecture
```mermaid
flowchart TB
    subgraph UI[Streamlit Frontend]
        direction TB
        UI_Input[User Inputs: ticker, date range, options]
        UI_Controls[Sidebar & Controls]
        UI_Visuals[Plotly Visualizations]
    end

    subgraph Ingest[Data Ingestion Layer]
        direction LR
        YF[yfinance Download]
        Cache[@st.cache_data]
        Preproc[Preprocessing & Feature Engineering]
    end

    subgraph Engine[Prediction Engine]
        direction LR
        Decision{Execution Split}
        SARIMAX[Statsmodels SARIMAX — live fit]
        Ensemble[Ensembling / Comparator]
        Artifacts[Artifacts Store (artifacts/lstm/)]
        LSTM_Load[load_model() @st.cache_resource]
        LSTM_Infer[LSTM Inference (Keras predict())]
        Scalers[scaler.pkl & sequence buffers]
    end

    subgraph Offline[Offline Training & CI]
        direction TB
        Notebook[Retrain Notebook (.ipynb)]
        Trainer[Training Environment (GPU/Local)]
        Export[Export: .keras, .pkl -> artifacts/]
        CI[Optional CI/CD / Retrain Scheduler]
    end

    subgraph Infra[Deployment & Observability]
        direction TB
        Hosting[Streamlit Cloud / VPS / Docker]
        Logs[App Logging & Metrics]
        Monitoring[Simple Health Checks]
    end

    %% UI -> Ingest
    UI_Input -->|requests| YF
    UI_Controls -->|controls| UI_Input
    YF -->|raw CSV/DF| Cache
    Cache --> Preproc

    %% Preproc -> Engine
    Preproc --> Decision
    Decision -->|fit live| SARIMAX
    Decision -->|use artifacts| LSTM_Load

    %% LSTM path
    LSTM_Load --> Scalers
    Scalers --> LSTM_Infer
    Artifacts --> LSTM_Load

    %% Combine outputs
    SARIMAX --> Ensemble
    LSTM_Infer --> Ensemble
    Ensemble --> UI_Visuals

    %% Offline loop
    Notebook --> Trainer --> Export --> Artifacts
    CI --> Notebook

    %% Infra
    Hosting -->|serves| UI
    Hosting --> Logs
    Logs --> Monitoring
```

**Diagram Notes (high level):**
- UI: `app.py` (Streamlit) collects user inputs and renders Plotly visuals.
- Ingest: `yfinance` -> cached DataFrame (`@st.cache_data`) -> preprocessing (resampling, scaling, sequence windows).
- Engine: a runtime decision either fits `SARIMAX` live (for interpretability) or loads pre-trained LSTM artifacts (from `artifacts/lstm/`) with `@st.cache_resource` for single-load model objects.
- LSTM artifacts include the `.keras` weights, `scaler.pkl`, and the last sequence buffers required to create input windows for inference.
- Ensemble: simple comparator/merger that presents both model outputs and confidence cues to the UI (no heavy ensembling required).
- Offline: training happens in `STOCK MARKET PREDICITION PROJECT.ipynb` -> produces artifacts committed to `artifacts/` or pushed to an artifact store. Optionally automated via CI/scheduler.
- Infra: host on Streamlit Cloud, VPS, or containerize; add simple logging and health checks for production readiness.

**Implementation Mapping (file references):**
- App: [app.py](app.py)
- Notebook / Training: [STOCK MARKET PREDICITION PROJECT.ipynb](STOCK%20MARKET%20PREDICITION%20PROJECT.ipynb)
- Artifacts folder: [artifacts/lstm](artifacts/lstm)

**Operational Considerations:**
- Caching: use `@st.cache_data` for dataframes and `@st.cache_resource` for model objects to avoid re-downloading or re-loading heavy objects.
- Model Drift: schedule periodic retraining (weekly/monthly) and update `artifacts/` via the offline notebook or CI pipeline.
- Monitoring: capture simple inference latency and cache hit-rate metrics in logs to detect regressions.
---

## 🛠️ Step-by-Step Setup Guide

Follow these instructions to clone, install, and run this application on your local machine.

### Prerequisites
- Python 3.9+ installed on your system.
- Git installed.

### 1. Clone the Repository
```bash
git clone <YOUR-REPOSITORY-URL>
cd <YOUR-REPOSITORY-DIRECTORY>
```

### 2. Create an Isolated Virtual Environment
Creating a virtual environment ensures that the project's dependencies do not conflict with your global Python setup.
```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
Install the cloud-optimized dependencies. (Note: `matplotlib` and `seaborn` have been stripped out to ensure a lean, fast deployment).
```bash
pip install -r requirements.txt
```

### 4. Run the Application
Boot up the Streamlit server. The app will automatically open in your default web browser at `http://localhost:8501`.
```bash
streamlit run app.py
```

---

## ☁️ 1-Click Deployment to Streamlit Cloud

Because of aggressive caching and prebuilt LSTM artifacts, this repository is ready for free cloud deployment with the pinned dependencies.

1. Commit and push your code to your GitHub repository. *(Ensure the `artifacts/` folder is included in the commit!)*
2. Log in to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Click **New App** and authorize your GitHub account.
4. Select your repository, set the branch to `main`, and set the Main file path to `app.py`.
5. Click **Deploy**. Your app will be live and shareable via a public URL in less than 2 minutes!

---

## 📓 Retraining the LSTM Model

The LSTM model currently predicts based on the artifacts checked into the repository. If you want to train the model on a different stock ticker or adjust the hyper-parameters (epochs, batch size, neurons):

1. Open `STOCK MARKET PREDICITION PROJECT.ipynb` in your preferred Jupyter environment.
2. Modify the `ticker` variable in the data loading function.
3. Run all cells from top to bottom.
4. The notebook will automatically generate new `.keras` and `.pkl` files and save them directly into the `artifacts/lstm/` directory.
5. Restart your Streamlit app, and the new model will instantly be served!

---

# 🧠 Engineering Deep Dive & Interview Preparation

*This section is an architectural deep dive into the engineering decisions behind the project. It serves as a study guide for discussing this project in technical interviews.*

## 1. Feature Tradeoff Analysis

### Feature 1: Dual-Model Engine (SARIMAX + LSTM)
Instead of forcing the user to rely on a single prediction, the app runs two completely different mathematical approaches in parallel.

*   **SARIMAX (Seasonal Auto-Regressive Integrated Moving Average with eXogenous factors):** A classical statistical model. It is exceptionally fast to fit on the fly and is highly interpretable. It explicitly models linear trends and seasonality.
*   **LSTM (Long Short-Term Memory):** A Recurrent Neural Network (RNN) architecture. It excels at finding non-linear, hidden, and complex patterns over long sequences of historical data.

**✅ Benefits:**
*   **Confidence via Consensus:** If both the linear model (SARIMAX) and non-linear model (LSTM) predict an upward trend, confidence is much higher.
*   **Speed:** SARIMAX fits instantly in the browser session for quick experimentation.

**❌ Drawbacks:**
*   SARIMAX struggles with sudden, non-linear market shocks.
*   LSTM is a "black box"—it is very difficult to explain exactly *why* it made a specific prediction.

### 📊 Dual-Model Summary Table
| Aspect | SARIMAX | LSTM |
| :--- | :--- | :--- |
| **Type** | Classical Statistical | Deep Learning (RNN) |
| **Execution** | Fit live in browser | Pre-trained offline, inferred live |
| **Strengths** | Interpretable, fast, handles linear seasonality | Captures deep, non-linear dependencies |
| **Weaknesses** | Fails on complex, hidden patterns | Black-box, computationally expensive to train |

---

### Feature 2: Offline Artifact Pipeline (The "Train Once, Infer Fast" Pattern)
Training a neural network inside a web application is an anti-pattern. It causes timeouts, freezes the UI, and costs massive amounts of server compute. To solve this, the LSTM is trained in a Jupyter Notebook. 

Once trained, the model weights (`lstm_model.keras`), data scalers (`scaler.pkl`), and last known sequences are exported to an `artifacts/` directory. The Streamlit app simply deserializes these artifacts and runs a rapid `model.predict()`.

**✅ Benefits:**
*   **Blazing Fast UI:** Inference takes milliseconds.
*   **Zero Compute Cost:** No expensive GPUs are required to host the web app.

**❌ Drawbacks:**
*   **Static Weights:** The LSTM model does not learn from new data live. If the market fundamentally changes, the model will degrade (Model Drift) until a developer manually re-runs the notebook and updates the artifacts.

### 📊 Artifact Pipeline Summary Table
| Concept | Explanation | Benefit to Application |
| :--- | :--- | :--- |
| **Artifact Export** | Saving `.keras` and `.pkl` files offline. | Decouples heavy training from the web server. |
| **Inference Only** | App only calls `predict()`, never `fit()`. | UI remains responsive; zero timeout errors. |
| **Model Drift** | Weights become outdated over time. | *Drawback*: Requires manual retraining schedules. |

---

### Feature 3: Aggressive Caching (`@st.cache_data` & `@st.cache_resource`)
Streamlit's execution model dictates that every time a user clicks a button or changes a date, the *entire Python script runs from top to bottom*. Without caching, the app would re-download data from Yahoo Finance and re-load the 3.5MB LSTM model on every click.

*   `@st.cache_data` is used for `yfinance` API calls. It hashes the ticker symbol and dates. If they haven't changed, it returns data instantly from memory.
*   `@st.cache_resource` is used for the LSTM Keras model. It ensures the heavy deep-learning object is initialized exactly as a singleton (once per server lifecycle).

**✅ Benefits:**
*   Prevents Yahoo Finance from IP-banning the server for rate-limiting.
*   Reduces memory usage and drops load times to near zero.

**❌ Drawbacks:**
*   Memory footprint can grow significantly if users query hundreds of different tickers in a single session without cache-clearing mechanisms.

### 📊 Caching Summary Table
| Decorator | Target | Purpose |
| :--- | :--- | :--- |
| `@st.cache_data` | `yfinance.download()` | Caches serializable data (Pandas DataFrames) to stop API rate-limiting. |
| `@st.cache_resource` | `load_model()` | Caches un-serializable objects (Keras Models) as singletons in RAM. |

---

## 🎯 Feature Purpose and Real User Use-Cases

Each visible feature is mapped to a concrete purpose so the app remains useful, not decorative:

1. **Ticker + Date Filters (Sidebar)**
- **Purpose:** Let users scope analysis to a relevant instrument and time range.
- **Use-case:** "I want to compare trend behavior for `HDFCBANK.NS` over the last 2 years."

2. **Seasonal Decomposition + SARIMAX Forecast**
- **Purpose:** Give a transparent, interpretable baseline forecast that updates live.
- **Use-case:** "I need quick directional insight and seasonality behavior without retraining ML models."

3. **LSTM Forecast From Artifacts**
- **Purpose:** Provide fast deep-learning inference without expensive online training.
- **Use-case:** "I want to view non-linear model output immediately in the dashboard with low latency."

4. **Interactive Plotly Visuals**
- **Purpose:** Make model behavior inspectable (zoom, hover, compare traces).
- **Use-case:** "I need to explain model outputs to stakeholders with exact values and trend overlays."

5. **Notebook-to-App Artifact Pipeline**
- **Purpose:** Separate heavy training from serving for reliability and lower costs.
- **Use-case:** "Data scientist retrains offline weekly; app users get stable inference daily."

---

## ⚠️ Current Practical Limitations (Important)

1. **LSTM artifact scope**
- LSTM predictions are based on the saved artifact set in `artifacts/lstm` and may not perfectly match the currently selected sidebar ticker/date unless artifacts were trained for that exact context.

2. **Not investment advice**
- Forecasts are educational/analytical outputs and should not be used as sole basis for trading decisions.

3. **Data source constraints**
- `yfinance` is convenient but can be rate-limited and may not match exchange-grade data quality.

---

### Feature 4: Cloud-Optimized Dependency Management
Standard `tensorflow` installations include massive GPU libraries (CUDA, cuDNN) by default. This easily balloons the container size to >2GB. Free-tier hosting platforms (like Streamlit Community Cloud or Heroku) typically kill applications that exceed 1GB of RAM.

By pinning TensorFlow consistently in `requirements.txt` and serving only prebuilt artifacts, the runtime stays stable and lightweight for this project.

**✅ Benefits:**
*   Guaranteed successful deployment on free-tier cloud architectures.
*   Massively faster CI/CD pipeline and Docker build times.

**❌ Drawbacks:**
*   CPU inference is technically slower than GPU inference, but for predicting a 30-day array on a single ticker, the difference is negligible (milliseconds).

### 📊 Dependency Summary Table
| Dependency | Original | Optimized | Why it was changed |
| :--- | :--- | :--- | :--- |
| **TensorFlow** | Unpinned / mismatched envs | `tensorflow==2.21.0` (pinned) | Reduces train/serve mismatch risk and improves reproducibility. |

---

