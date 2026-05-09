# Stock Prediction App 📈

An interactive, production-ready Streamlit app that forecasts stock prices using two complementary approaches:
- **SARIMAX** for fast statistical forecasting from live ticker data.
- **LSTM** for deep-learning forecasts loaded from pre-trained, notebook-exported artifacts.

This project is fully scalable, optimized for low memory usage, and ready to be deployed to platforms like **Streamlit Community Cloud** or **GitHub Pages**.

## Key Features
- **Live Data Pulling**: Uses `yfinance` with `@st.cache_data` to prevent rate-limiting.
- **Pre-trained LSTM Inference**: Uses `@st.cache_resource` for instant load times without heavy computations in the browser.
- **Optimized for Deployment**: Utilizes `tensorflow-cpu` to comfortably fit within 1GB memory limits on free hosting tiers.
- **Interactive Visualizations**: Powered by Plotly.

---

## 🚀 How to Run Locally

1. **Clone the repository & create a virtual environment**
```bash
git clone <your-repo-url>
cd <your-repo-name>
python -m venv venv
venv\Scripts\activate   # (On Windows)
source venv/bin/activate # (On Mac/Linux)
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the Streamlit App**
```bash
streamlit run app.py
```

---

## ☁️ Deploying to Streamlit Community Cloud

This project is perfectly optimized for free deployment on Streamlit Community Cloud.

1. Commit and push this entire repository to GitHub (ensure the `artifacts/` folder is pushed!).
2. Go to [share.streamlit.io](https://share.streamlit.io/) and connect your GitHub account.
3. Click **New app**.
4. Select the repository, set the branch to `main`, and the main file path to `app.py`.
5. Click **Deploy!**

The app uses `tensorflow-cpu`, ensuring it easily fits under Streamlit's 1GB resource limit.

---

## 🧠 LSTM Artifacts (Notebook -> App)

The LSTM section is designed for **fast inference**. The model was trained in a Jupyter Notebook, and its artifacts are checked into the `artifacts/lstm/` directory.

To train on a new ticker or with new parameters:
1. Open `STOCK MARKET PREDICITION PROJECT.ipynb`.
2. Change the ticker or parameters.
3. Run all cells to export the new `.keras` model and `.pkl` artifacts.
4. Restart the Streamlit app.

## Why Two Models?
- **SARIMAX**: quick, interpretable, handles trend/seasonality.
- **LSTM**: learns nonlinear patterns and long dependencies from historical data.

This dual approach makes the app both **fast** and **highly expressive**.

## Need a Deep Dive?
See [SELF_README.md](SELF_README.md) for detailed tech-stack reasoning, tradeoffs, and design decisions.
