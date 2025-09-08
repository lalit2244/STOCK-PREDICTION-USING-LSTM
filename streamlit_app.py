
import os, json, time, datetime as dt
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator
from pandas_datareader import data as pdr
import joblib
import tensorflow as tf
import requests
def get_news(ticker: str, limit: int = 10):
    """
    Fetch latest news headlines for a given stock ticker using yfinance.
    """
    try:
        ticker_obj = yf.Ticker(ticker)
        news = ticker_obj.news or []
        news_items = []
        for n in news[:limit]:
            news_items.append({
                "headline": n.get("title", "No title"),
                "link": n.get("link", "#")
            })
        return pd.DataFrame(news_items)
    except Exception as e:
        print(f"⚠️ Error fetching news: {e}")
        return pd.DataFrame(columns=["headline", "link"])

def get_news(ticker: str, limit: int = 10):
    """
    Fetch latest news headlines for a given stock ticker.
    Uses Yahoo Finance API as source.
    """
    try:
        url = f"https://query1.finance.yahoo.com/v1/finance/search?q={ticker}"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()

        # Extract news articles (if available)
        news_items = []
        if "news" in data:
            for item in data["news"][:limit]:
                news_items.append({
                    "headline": item.get("title", "No title"),
                    "link": item.get("link", "#")
                })

        return pd.DataFrame(news_items)

    except Exception as e:
        print(f"⚠️ Error fetching news: {e}")
        return pd.DataFrame(columns=["headline", "link"])

# ---------------------------
# Initialize FinBERT (cached) and VADER fallback
# Paste this after your imports section
# ---------------------------
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

@st.cache_resource
def load_finbert_pipeline():
    """
    Returns a HuggingFace pipeline for finbert-tone (PyTorch backend).
    Returns None on failure so we can fallback to VADER.
    """
    try:
        return pipeline(
            "sentiment-analysis",
            model="yiyanghkust/finbert-tone",
            tokenizer="yiyanghkust/finbert-tone",
            framework="pt"   # force PyTorch to avoid Keras/TF issues
        )
    except Exception as e:
        # If loading FinBERT fails (no torch, network error, etc.) return None
        return None

# Ensure sentiment pipeline object is present in session_state
if "sentiment_pipeline" not in st.session_state:
    st.session_state["sentiment_pipeline"] = load_finbert_pipeline()

# Ensure a VADER fallback exists (lightweight)
if "vader" not in st.session_state:
    try:
        st.session_state["vader"] = SentimentIntensityAnalyzer()
    except Exception:
        st.session_state["vader"] = None


# ------------------ Settings ------------------
st.set_page_config(page_title="StockPredictorPro", page_icon="📈", layout="wide")

APP_DIR = os.path.dirname(__file__)
BASE_DIR = os.path.abspath(os.path.join(APP_DIR, ".."))
DATA_DIR = os.path.join(APP_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

DEFAULT_TICKERS = ["AAPL","GOOGL","TSLA","MSFT","AMZN","META","NVDA","NFLX","JPM","WMT"]
THEMES = {
    "Dark": {"plotly":"plotly_dark", "css":"body {background-color:#0e1117;}"},
    "Light": {"plotly":"plotly", "css":"body {background-color:#ffffff;}"},
    "Colorful": {"plotly":"ggplot2", "css":"body {background: linear-gradient(120deg,#f6d365,#fda085);}"},
    "Professional": {"plotly":"simple_white", "css":"body {background-color:#f5f7fb;}"}
}
analyzer = SentimentIntensityAnalyzer()

def load_state():
    state_path = os.path.join(DATA_DIR, "state.json")
    if os.path.exists(state_path):
        with open(state_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"favorites": [], "history_path": os.path.join(DATA_DIR, "history.csv")}

def save_state(state):
    with open(os.path.join(DATA_DIR, "state.json"), "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

def load_history(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame(columns=["timestamp","ticker","predicted_next_close"])

def save_history(df, path):
    df.to_csv(path, index=False)

STATE = load_state()
HISTORY = load_history(STATE["history_path"])

# ------------------ Sidebar ------------------
st.sidebar.header("🎨 Theme & Settings")
theme_choice = st.sidebar.selectbox("Theme", list(THEMES.keys()), index=0)
st.sidebar.markdown("Once you change theme, charts follow the style.")

ticker = st.sidebar.selectbox("Choose a Stock", DEFAULT_TICKERS, index=0)
custom = st.sidebar.text_input("Or type another ticker", "")
if custom.strip():
    ticker = custom.strip().upper()

if st.sidebar.button("⭐ Add to Favorites"):
    if ticker not in STATE["favorites"]:
        STATE["favorites"].append(ticker)
        save_state(STATE)
        st.sidebar.success(f"Added {ticker} to favorites")
    else:
        st.sidebar.info("Already in favorites!")

if STATE["favorites"]:
    st.sidebar.write("**Your Favorites**")
    st.sidebar.write(", ".join(STATE["favorites"]))

st.sidebar.divider()
auto_refresh = st.sidebar.toggle("Auto-refresh every minute", value=True)
refresh_interval = 60 * 1000


   # Auto refresh support
if "st_autorefresh" not in globals():
    try:
        from streamlit_autorefresh import st_autorefresh
    except ImportError:
        def st_autorefresh(*args, **kwargs):
            st.rerun()


# ------------------ Header ------------------
st.title("📈 StockPredictorPro")
st.caption("Educational demo • Not financial advice")

colA, colB, colC = st.columns([2,2,1])
with colA:
    st.subheader(f"Ticker: {ticker}")
with colB:
    st.selectbox("Quick Pick", DEFAULT_TICKERS, key="quickpick", on_change=lambda: None)
with colC:
    st.write(dt.datetime.now().strftime("Time: %Y-%m-%d %H:%M:%S"))

# Theme apply
plotly_template = THEMES[theme_choice]["plotly"]

# ------------------ Data Fetch ------------------
@st.cache_data(show_spinner=False, ttl=60)
def get_price_data(t, period="1y", interval="1d"):
    try:
        df = yf.download(t, period=period, interval=interval, progress=False, auto_adjust=False)
        if df is None or df.empty:
            return pd.DataFrame()
        df.dropna(inplace=True)
        return df
    except Exception:
        return pd.DataFrame()

# Try minute first for "live" feel, fallback to 5m/1d
df = get_price_data(ticker, period="5d", interval="1m")
if df.empty:
    df = get_price_data(ticker, period="1mo", interval="5m")
if df.empty:
    df = get_price_data(ticker, period="1y", interval="1d")

if df.empty:
    st.error("Couldn't load price data. Try another ticker or check connection.")
    st.stop()

# Indicators
close_series = pd.Series(df["Close"].values.squeeze(), index=df.index)

df["SMA20"] = SMAIndicator(close=close_series, window=20, fillna=True).sma_indicator()
df["SMA50"] = SMAIndicator(close=close_series, window=50, fillna=True).sma_indicator()
df["RSI14"] = RSIIndicator(close=close_series, window=14, fillna=True).rsi()


# ------------------ Price Cards ------------------
last_close = float(df["Close"].iloc[-1])
prev_close = float(df["Close"].iloc[-2]) if len(df) > 1 else last_close
chg = last_close - prev_close
pct = (chg/prev_close*100) if prev_close else 0.0

kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric("Last Close", f"{last_close:,.2f}", f"{chg:+.2f}")
kpi2.metric("Change %", f"{pct:+.2f}%")
kpi3.metric("RSI(14)", f"{df['RSI14'].iloc[-1]:.1f}")

# ------------------ Candlestick Chart ------------------
fig = go.Figure(data=[go.Candlestick(
    x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price"
)])
fig.add_trace(go.Scatter(x=df.index, y=df["SMA20"], name="SMA20"))
fig.add_trace(go.Scatter(x=df.index, y=df["SMA50"], name="SMA50"))
fig.update_layout(template=plotly_template, height=500, margin=dict(l=10,r=10,t=40,b=10))
st.plotly_chart(fig, use_container_width=True)

with st.expander("Show RSI"):
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=df.index, y=df["RSI14"], name="RSI14"))
    fig_rsi.add_hline(y=70, line_dash="dash")
    fig_rsi.add_hline(y=30, line_dash="dash")
    fig_rsi.update_layout(template=plotly_template, height=250, margin=dict(l=10,r=10,t=40,b=10))
    st.plotly_chart(fig_rsi, use_container_width=True)

# ------------------ Quick LSTM Predict ------------------
st.subheader("🔮 Quick Predict (LSTM)")
st.caption("Trains a small LSTM on recent data and predicts the next close. Saves model to /models.")

def make_sequences(array, seq_len=60):
    X, y = [], []
    for i in range(seq_len, len(array)):
        X.append(array[i-seq_len:i])
        y.append(array[i, 0])
    return np.array(X), np.array(y)

if st.button("Train & Predict"):
    feats = df[["Close","SMA20","SMA50","RSI14"]].fillna(method="ffill").values.astype("float32")
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(feats)
    X, y = make_sequences(scaled, 60)

    if len(X) < 100:
        st.warning("Not enough data to train. Try a longer interval.")
    else:
        split = int(len(X)*0.8)
        X_train, y_train = X[:split], y[:split]
        X_test, y_test = X[split:], y[split:]

        tf.keras.backend.clear_session()
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(48, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(24),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer="adam", loss="mse")
        model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.1, verbose=0)

        pred = model.predict(X_test)[-1][0]
        last_row = scaled[-1].copy()
        last_row[0] = pred
        inv_pred = scaler.inverse_transform([last_row])[0][0]
        st.success(f"Predicted next close: {inv_pred:,.2f}")

        # Save model + scaler
        path_model = os.path.join(MODELS_DIR, f"lstm_{ticker}.keras")
        model.save(path_model)
        joblib.dump(scaler, os.path.join(MODELS_DIR, f"scaler_{ticker}.joblib"))

        # Save to History
        if "HISTORY" not in st.session_state:
            st.session_state["HISTORY"] = pd.DataFrame(columns=["timestamp", "ticker", "predicted_next_close"])

        row = {
            "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
            "ticker": ticker,
            "predicted_next_close": inv_pred
        }
        st.session_state["HISTORY"] = pd.concat(
            [st.session_state["HISTORY"], pd.DataFrame([row])],
            ignore_index=True
        )

        save_history(st.session_state["HISTORY"], STATE["history_path"])

# ------------------ Show History ------------------
with st.expander("📚 Analysis History"):
    if "HISTORY" in st.session_state:
        st.dataframe(st.session_state["HISTORY"].tail(100), use_container_width=True)
        st.download_button(
            "Download History CSV",
            st.session_state["HISTORY"].to_csv(index=False).encode(),
            file_name="history.csv"
        )
    else:
        st.info("No predictions yet.")

# ------------------ News + Sentiment ------------------
from newsapi import NewsApiClient

NEWS_API_KEY = "05795e3ad5904ccfa280f6804dcf7308"  # <--- paste your key here
newsapi = NewsApiClient(api_key=NEWS_API_KEY)

st.subheader("📰 Latest News + 📊 Sentiment")

if "sentiment_pipeline" not in st.session_state:
    from transformers import pipeline
    st.session_state.sentiment_pipeline = pipeline(
        "sentiment-analysis", model="ProsusAI/finbert"
    )

try:
    # Fetch news for the ticker
    news = newsapi.get_everything(
        q=ticker, language="en", sort_by="publishedAt", page_size=10
    )
    articles = news.get("articles", [])

    if not articles:
        st.info("No news found from NewsAPI.")
    else:
        for a in articles:
            title = a["title"]
            link = a["url"]
            pub = a.get("source", {}).get("name", "")
            time_str = a.get("publishedAt", "")

            # Run FinBERT sentiment
            result = st.session_state.sentiment_pipeline(title)[0]
            label = result["label"]
            score = result["score"]

            # Color mapping
            if label == "Positive":
                color = "#4CAF50"  # green
            elif label == "Negative":
                color = "#F44336"  # red
            else:
                color = "#9E9E9E"  # gray

            # Display headline card
            st.markdown(
                f"""
                <div style="border:1px solid #ddd; border-radius:10px; padding:12px; margin:8px 0; background:#fafafa;">
                    <b>{title}</b><br>
                    <small>{time_str} — {pub}</small><br><br>
                    <span style="background:{color}; color:white; padding:4px 10px; border-radius:20px;">
                        {label} ({score*100:.1f}%)
                    </span><br><br>
                    <a href="{link}" target="_blank">🔗 Read full article</a>
                </div>
                """,
                unsafe_allow_html=True
            )

except Exception as e:
    st.warning(f"Couldn't load news: {e}")









# ------------------ Crypto ------------------
st.subheader("₿ Crypto Snapshot")
cols = st.columns(2)
for i, ct in enumerate(["BTC-USD","ETH-USD"]):
    cdf = get_price_data(ct, period="7d", interval="1h")
    if cdf.empty:
        cols[i].warning(f"No data for {ct}")
    else:
        last = float(cdf["Close"].iloc[-1])
        figc = px.line(
    cdf,
    y=cdf["Close"].squeeze(),  # ensure 1D
    labels={"y": "Close"},
    template=plotly_template,
    title=f"{ct} — Last: {last:,.2f}"
)

        figc.update_layout(height=260, showlegend=False, margin=dict(l=10,r=10,t=40,b=10))
        cols[i].plotly_chart(figc, use_container_width=True)

# ------------------ Economic Indicators (FRED) ------------------
st.subheader("📊 Economic Indicators")
try:
    gdp = pdr.DataReader("GDP", "fred").dropna().iloc[-1].values[0]
    unrate = pdr.DataReader("UNRATE", "fred").dropna().iloc[-1].values[0]
    cpi = pdr.DataReader("CPIAUCSL", "fred").dropna().iloc[-1].values[0]
    col1, col2, col3 = st.columns(3)
    col1.metric("US GDP (Billions, last)", f"{gdp:,.0f}")
    col2.metric("US Unemployment Rate (last %)", f"{unrate:.1f}%")
    col3.metric("US CPI (last)", f"{cpi:,.1f}")
except Exception as e:
    st.info("Indicators not available right now (needs internet).")

st.caption("© Educational demo. Built with Streamlit + yfinance + Keras.")
