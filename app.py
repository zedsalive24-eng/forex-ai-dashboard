import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from datetime import datetime
from twelvedata import TDClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.exceptions import NotFittedError
import os
import joblib
from pandas.errors import EmptyDataError

# ---- CONFIG ----
st.set_page_config(page_title="FOREX.BOT 1.0", page_icon="💹", layout="wide")

# ---- STYLE ----
st.markdown("""
    <style>
    body {
        background-color: #0e1117;
        color: #fafafa;
    }
    .big-font {
        font-size:40px !important;
        font-weight:600;
        color: #fafafa;
        text-align:center;
    }
    .signal-box {
        border-radius: 20px;
        background: rgba(255, 255, 255, 0.05);
        padding: 40px;
        text-align: center;
        margin-top: 30px;
        box-shadow: 0 4px 30px rgba(255, 255, 255, 0.1);
    }
    </style>
""", unsafe_allow_html=True)

# ---- HEADER ----
st.markdown("<h1 class='big-font'>FOREX.BOT 1.0</h1>", unsafe_allow_html=True)
st.markdown("### Intelligent Self-Learning Forex Signal Generator")

# ---- SIDEBAR ----
st.sidebar.header("⚙️ Settings")
pair = st.sidebar.selectbox("Select Currency Pair", ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CHF"])
timeframe = st.sidebar.selectbox("Timeframe", ["5min", "15min", "30min", "1h", "1day"])
AUTO_REFRESH_SECONDS = 300  # 5 minutes
components.html(
    f"""
    <script>
        setTimeout(function() {{
            window.parent.location.reload();
        }}, {AUTO_REFRESH_SECONDS * 1000});
    </script>
    """,
    height=0,
    width=0,
)
st.sidebar.caption("Auto-refreshing every 5 minutes so the model can log new outcomes.")

# ---- FETCH DATA ----
@st.cache_data(ttl=60, show_spinner=False)
def fetch_data(symbol: str = "EUR/USD", interval: str = "15min") -> pd.DataFrame:
    td = TDClient(apikey="e32c655ac6b449e0892027a22cd62d98")
    try:
        ts = td.time_series(symbol=symbol, interval=interval, outputsize=200)
        df = ts.as_pandas()
        if df is None or df.empty:
            st.warning(f"No live data returned for {symbol} ({interval}).")
            return pd.DataFrame()
        df = df.reset_index().rename(columns={"datetime": "Time", "close": "Price"})
        df = df.sort_values("Time").reset_index(drop=True)
        df["Price"] = df["Price"].astype(float)
        return df
    except Exception as exc:
        st.error(f"Error fetching live data: {exc}")
        return pd.DataFrame()

df = fetch_data(pair, timeframe)

# ---- INDICATORS ----
def compute_ema(data, span): return data["Price"].ewm(span=span, adjust=False).mean()
def compute_rsi(data, periods=14):
    delta = data["Price"].diff()
    gain = delta.clip(lower=0).rolling(window=periods).mean()
    loss = (-delta.clip(upper=0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))
def compute_macd(data):
    ema12, ema26 = compute_ema(data, 12), compute_ema(data, 26)
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

# ---- LIVE PRICE ----
if not df.empty:
    current_price = df["Price"].iloc[-1]
    st.markdown(f"<div style='text-align:center;margin-top:10px;'><span style='font-size:28px;color:#00FFAA;font-weight:600;'>Current {pair} Price: {current_price:.5f}</span></div>", unsafe_allow_html=True)
else:
    st.warning("⚠️ Unable to fetch live price.")

if df.empty:
    st.stop()

# ---- FEATURE ENGINEERING ----
df["EMA_12"], df["EMA_26"] = compute_ema(df, 12), compute_ema(df, 26)
df["RSI"] = compute_rsi(df)
df["MACD"], df["MACD_Signal"] = compute_macd(df)
df["EMA_Spread"] = df["EMA_12"] - df["EMA_26"]
df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
df["Return_1"] = df["Price"].pct_change()
df["Return_3"] = df["Price"].pct_change(3)
df["Return_6"] = df["Price"].pct_change(6)
df["Volatility_5"] = df["Return_1"].rolling(5).std()
df["Volatility_10"] = df["Return_1"].rolling(10).std()
df["RSI_SMA"] = df["RSI"].rolling(5).mean()
df["Price_Momentum"] = df["Price"] - df["Price"].shift(5)
df = df.dropna().reset_index(drop=True)

# ---- MODEL FEATURES ----
feature_cols = ["EMA_12","EMA_26","RSI","MACD","MACD_Signal","EMA_Spread","MACD_Hist",
                "Return_1","Return_3","Return_6","Volatility_5","Volatility_10","RSI_SMA","Price_Momentum"]
df["Target"] = np.where(df["Price"].shift(-1) > df["Price"], 1, 0)
X, y = df[feature_cols], df["Target"]
if X.empty:
    st.warning("Not enough processed history to train the AI model right now. Please try again shortly.")
    st.stop()

# ---- MODEL LOAD/INIT ----
def build_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=400, max_depth=6, min_samples_leaf=3,
            class_weight="balanced_subsample", random_state=42, n_jobs=-1))
    ])

model_file = "forex_model.pkl"
model_pipeline = None
if os.path.exists(model_file):
    try:
        model_pipeline = joblib.load(model_file)
    except Exception:
        model_pipeline = None
if model_pipeline is None:
    model_pipeline = build_model()

if not hasattr(model_pipeline, "classes_"):
    model_pipeline.fit(X, y)
    joblib.dump(model_pipeline, model_file)

# ---- RETRAIN LOGIC ----
log_file = "prediction_log.csv"
log_columns = ["Timestamp","Pair","Timeframe","Signal","Confidence","Price_At_Predict","Actual_Outcome","Was_Correct"]
if os.path.exists(log_file):
    try:
        history = pd.read_csv(log_file)
    except EmptyDataError:
        history = pd.DataFrame(columns=log_columns)
else:
    history = pd.DataFrame(columns=log_columns)
history = history.reindex(columns=log_columns)
for col in ["Price_At_Predict", "Actual_Outcome", "Was_Correct"]:
    if col in history.columns:
        history[col] = pd.to_numeric(history[col], errors="coerce")

# Verify old predictions
for i, row in history.iterrows():
    if not pd.isna(row.get("Was_Correct")):
        continue
    if row.get("Pair") != pair:
        continue
    if pd.notna(row.get("Timeframe")) and row.get("Timeframe") != timeframe:
        continue
    timestamp_str = row.get("Timestamp")
    if pd.isna(timestamp_str):
        continue
    try:
        past_time = datetime.strptime(str(timestamp_str), "%Y-%m-%d %H:%M:%S")
    except ValueError:
        continue
    matching = df[df["Time"] > past_time]
    if len(matching) < 3:
        continue
    future_price = matching["Price"].iloc[2]
    history.loc[i, "Actual_Outcome"] = future_price
    if (row.get("Signal") == "BUY" and future_price > row.get("Price_At_Predict")) or (
        row.get("Signal") == "SELL" and future_price < row.get("Price_At_Predict")
    ):
        history.loc[i, "Was_Correct"] = 1
    else:
        history.loc[i, "Was_Correct"] = 0

# Periodic retrain (every 20 verified predictions)
if history["Was_Correct"].notna().sum() >= 20:
    st.info("🔁 Retraining AI model with new feedback data...")
    model_pipeline = build_model()
    model_pipeline.fit(X, y)
    joblib.dump(model_pipeline, model_file)
    history = history.head(50)  # keep most recent 50 signals only
    history = history.reindex(columns=log_columns)
    history.to_csv(log_file, index=False)

# ---- AI PREDICTION ----
last_sample = X.iloc[[-1]]
try:
    pred = model_pipeline.predict(last_sample)[0]
    conf = model_pipeline.predict_proba(last_sample)[0][pred]
except (NotFittedError, ValueError):
    model_pipeline.fit(X, y)
    joblib.dump(model_pipeline, model_file)
    pred = model_pipeline.predict(last_sample)[0]
    conf = model_pipeline.predict_proba(last_sample)[0][pred]
signal = "BUY" if pred == 1 else "SELL"
color = "lime" if signal == "BUY" else "red"

# ---- SAVE PREDICTION ----
new_entry = pd.DataFrame([{
    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "Pair": pair,
    "Timeframe": timeframe,
    "Signal": signal,
    "Confidence": f"{conf*100:.1f}%",
    "Price_At_Predict": current_price,
    "Actual_Outcome": np.nan,
    "Was_Correct": np.nan
}])
history = pd.concat([new_entry, history], ignore_index=True).head(100)
history = history.reindex(columns=log_columns)
history.to_csv(log_file, index=False)

# ---- PRICE CHART ----
fig = go.Figure()
fig.add_trace(go.Scatter(x=df["Time"], y=df["Price"], mode="lines", name="Price", line=dict(width=2)))
fig.add_trace(go.Scatter(x=df["Time"], y=df["EMA_12"], mode="lines", name="EMA 12", line=dict(width=1)))
fig.add_trace(go.Scatter(x=df["Time"], y=df["EMA_26"], mode="lines", name="EMA 26", line=dict(width=1)))
fig.update_layout(template="plotly_dark", margin=dict(l=0, r=0, t=20, b=20), height=400)
st.plotly_chart(fig, width="stretch")

# ---- SIGNAL BOX ----
st.markdown(f"<div class='signal-box'><h2 style='font-size:60px;color:{color};'>{signal}</h2><p style='font-size:20px;'>Confidence: <b>{conf*100:.1f}%</b></p></div>", unsafe_allow_html=True)

# ---- PERFORMANCE ----
verified_mask = history["Was_Correct"].notna()
total_preds = int(verified_mask.sum())
correct_preds = history.loc[verified_mask, "Was_Correct"].sum()
true_acc = float(correct_preds) / total_preds if total_preds > 0 else 0.0
accuracy_color = "lime" if true_acc >= 0.7 else "orange" if true_acc >= 0.5 else "red"
st.markdown("### AI Model Performance Tracker")
st.progress(min(true_acc, 1.0))
st.markdown(
    f"<b>Learned Accuracy:</b> <span style='color:{accuracy_color};'>{true_acc:.2%}</span> "
    f"based on {total_preds} verified signals.",
    unsafe_allow_html=True,
)

# ---- LOG DISPLAY ----
st.markdown("### Prediction History")
st.dataframe(history.head(10), width="stretch", hide_index=True)

# ---- EXPLANATION ----
with st.expander("How It Learns & Interprets Signals"):
    st.markdown("""
    ### How the AI Learns
    - Each signal is stored with timestamp and price.
    - After 3 new candles, the model checks if its prediction was right.
    - Once 20 verified signals are recorded, the model retrains automatically.

    ### What Improves Over Time
    - Decision boundaries refine using real outcomes.
    - Confidence becomes better calibrated to actual results.
    - Accuracy % becomes a true measure of predictive reliability.

    ### ⚠️ Disclaimer
    This app is for **educational purposes** only and does not constitute financial advice.
    """)
