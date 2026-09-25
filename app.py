# app.py — Advanced AI Index Dashboard v2
# Multi-timeframe confirmation | Ensemble ML | 1000+ candles | Persistent settings

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import json, os
from datetime import datetime, time
import pytz
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.model_selection import TimeSeriesSplit
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="AI Pro Dashboard", layout="wide", initial_sidebar_state="expanded")

CONFIG_FILE = "settings.json"
IST = pytz.timezone("Asia/Kolkata")

# ---------------- PERSISTENT SETTINGS ----------------
DEFAULTS = {
    "index": "NIFTY 50", "timeframe": "15m", "candles": 1000,
    "fast_ema": 12, "slow_ema": 26, "buy_score": 78, "sell_score": 22,
    "sl_atr": 1.5, "target_atr": 2.5, "cost_pct": 0.03, "auto_refresh": True,
    "use_mtf": True, "mtf_interval": "1h", "min_folds_acc": 3
}

def load_settings():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE) as f:
            saved = json.load(f)
        return {**DEFAULTS, **saved}
    return DEFAULTS.copy()

def save_settings(cfg):
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f)

if "cfg" not in st.session_state:
    st.session_state.cfg = load_settings()
cfg = st.session_state.cfg

def upd(key):
    def _cb():
        cfg[key] = st.session_state[f"w_{key}"]
        save_settings(cfg)
    return _cb

SYMBOL_MAP = {"NIFTY 50": "^NSEI", "BANK NIFTY": "^NSEBANK", "SENSEX": "^BSESN", "FINNIFTY": "^CNXFIN"}

with st.sidebar:
    st.header("⚙️ Settings")
    st.selectbox("Index", list(SYMBOL_MAP.keys()),
                 index=list(SYMBOL_MAP.keys()).index(cfg["index"]),
                 key="w_index", on_change=upd("index"))
    st.selectbox("Timeframe", ["5m","15m","1h","1d"],
                 index=["5m","15m","1h","1d"].index(cfg["timeframe"]),
                 key="w_timeframe", on_change=upd("timeframe"))
    st.slider("Candles", 300, 2000, cfg["candles"], step=100, key="w_candles", on_change=upd("candles"))

    st.subheader("Multi-Timeframe Filter")
    st.checkbox("Use higher-TF trend confirmation", cfg["use_mtf"], key="w_use_mtf", on_change=upd("use_mtf"))
    st.selectbox("Higher TF", ["1h","1d"],
                 index=["1h","1d"].index(cfg["mtf_interval"]),
                 key="w_mtf_interval", on_change=upd("mtf_interval"))

    st.subheader("Indicators")
    st.number_input("Fast EMA", 3, 50, cfg["fast_ema"], key="w_fast_ema", on_change=upd("fast_ema"))
    st.number_input("Slow EMA", 5, 100, cfg["slow_ema"], key="w_slow_ema", on_change=upd("slow_ema"))

    st.subheader("Signal Thresholds")
    st.slider("Buy score ≥", 50, 95, cfg["buy_score"], key="w_buy_score", on_change=upd("buy_score"))
    st.slider("Sell score ≤", 5, 50, cfg["sell_score"], key="w_sell_score", on_change=upd("sell_score"))

    st.subheader("Risk Management")
    st.number_input("Stop Loss (x ATR)", 0.5, 5.0, cfg["sl_atr"], step=0.1, key="w_sl_atr", on_change=upd("sl_atr"))
    st.number_input("Target (x ATR)", 0.5, 5.0, cfg["target_atr"], step=0.1, key="w_target_atr", on_change=upd("target_atr"))

    st.checkbox("Auto refresh (30s live)", cfg["auto_refresh"], key="w_auto_refresh", on_change=upd("auto_refresh"))

# ---------------- MARKET STATUS ----------------
def market_open():
    now = datetime.now(IST)
    if now.weekday() >= 5:
        return False
    return time(9,15) <= now.time() <= time(15,30)

is_open = market_open()
if cfg["auto_refresh"] and is_open:
    st_autorefresh(interval=30_000, key="refresh")

# ---------------- DATA ----------------
@st.cache_data(ttl=25)
def fetch_data(symbol, interval, period):
    df = yf.download(symbol, interval=interval, period=period, progress=False)
    df = df.reset_index()
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df

period_map = {"5m":"60d","15m":"60d","1h":"2y","1d":"5y"}
symbol = SYMBOL_MAP[cfg["index"]]
raw = fetch_data(symbol, cfg["timeframe"], period_map[cfg["timeframe"]])
df = raw.tail(cfg["candles"]).reset_index(drop=True)

if cfg["use_mtf"]:
    htf_raw = fetch_data(symbol, cfg["mtf_interval"], period_map[cfg["mtf_interval"]])
    htf_raw["EMA_htf"] = htf_raw["Close"].ewm(span=21).mean()
    htf_raw["trend_up"] = htf_raw["Close"] > htf_raw["EMA_htf"]

# ---------------- FEATURE ENGINEERING ----------------
def rsi(series, period=14):
    d = series.diff()
    gain = d.clip(lower=0).rolling(period).mean()
    loss = -d.clip(upper=0).rolling(period).mean()
    return 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

def atr(df, period=14):
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift()).abs(),
        (df["Low"] - df["Close"].shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def macd(series, fast=12, slow=26, signal=9):
    ema_f, ema_s = series.ewm(span=fast).mean(), series.ewm(span=slow).mean()
    macd_line = ema_f - ema_s
    signal_line = macd_line.ewm(span=signal).mean()
    return macd_line, signal_line, macd_line - signal_line

def adx(df, period=14):
    up = df["High"].diff()
    down = -df["Low"].diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = atr(df, period)
    plus_di = 100 * pd.Series(plus_dm).rolling(period).mean() / tr
    minus_di = 100 * pd.Series(minus_dm).rolling(period).mean() / tr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.rolling(period).mean()

def stochastic(df, k=14, d=3):
    low_min = df["Low"].rolling(k).min()
    high_max = df["High"].rolling(k).max()
    pct_k = 100 * (df["Close"] - low_min) / (high_max - low_min)
    return pct_k, pct_k.rolling(d).mean()

def vwap(df):
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    return (tp * df["Volume"]).cumsum() / df["Volume"].cumsum()

df["EMA_fast"] = df["Close"].ewm(span=cfg["fast_ema"]).mean()
df["EMA_slow"] = df["Close"].ewm(span=cfg["slow_ema"]).mean()
df["EMA200"] = df["Close"].ewm(span=200).mean()
df["RSI"] = rsi(df["Close"])
df["ATR"] = atr(df)
df["MACD"], df["MACD_signal"], df["MACD_hist"] = macd(df["Close"])
df["ADX"] = adx(df)
df["Stoch_K"], df["Stoch_D"] = stochastic(df)
df["VWAP"] = vwap(df)
df["BB_mid"] = df["Close"].rolling(20).mean()
df["BB_std"] = df["Close"].rolling(20).std()
df["BB_up"] = df["BB_mid"] + 2*df["BB_std"]
df["BB_dn"] = df["BB_mid"] - 2*df["BB_std"]

df["r1"] = df["Close"].pct_change(1)
df["r3"] = df["Close"].pct_change(3)
df["r5"] = df["Close"].pct_change(5)
df["r10"] = df["Close"].pct_change(10)
df["r20"] = df["Close"].pct_change(20)
df["vol_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()
df["ema_diff"] = (df["EMA_fast"] - df["EMA_slow"]) / df["Close"]
df["dist_vwap"] = (df["Close"] - df["VWAP"]) / df["Close"]
df["dist_ema200"] = (df["Close"] - df["EMA200"]) / df["Close"]
df["bb_pos"] = (df["Close"] - df["BB_dn"]) / (df["BB_up"] - df["BB_dn"] + 1e-9)
df["hl_range"] = (df["High"] - df["Low"]) / df["Close"]

# MTF trend merge
if cfg["use_mtf"]:
    htf_raw["Datetime"] = pd.to_datetime(htf_raw["Datetime"])
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = pd.merge_asof(df.sort_values("Datetime"), htf_raw[["Datetime","trend_up"]].sort_values("Datetime"),
                        on="Datetime", direction="backward")
    df["trend_up"] = df["trend_up"].astype(float)
else:
    df["trend_up"] = 0.5

df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

FEATURES = ["r1","r3","r5","r10","r20","RSI","ema_diff","vol_ratio","MACD_hist",
            "ADX","Stoch_K","dist_vwap","dist_ema200","bb_pos","hl_range","trend_up"]
data = df.dropna(subset=FEATURES + ["target"]).reset_index(drop=True)

# ---------------- WALK-FORWARD VALIDATION (multiple folds) ----------------
n_folds = max(cfg["min_folds_acc"], 3)
tscv = TimeSeriesSplit(n_splits=n_folds)
fold_accs = []

for train_idx, test_idx in tscv.split(data):
    tr, te = data.iloc[train_idx], data.iloc[test_idx]
    m = VotingClassifier([
        ("gb", GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.05)),
        ("rf", RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42))
    ], voting="soft")
    m.fit(tr[FEATURES], tr["target"])
    fold_accs.append((m.predict(te[FEATURES]) == te["target"]).mean())

avg_acc = np.mean(fold_accs)
std_acc = np.std(fold_accs)

# Final model trained on all data for live signal
final_model = VotingClassifier([
    ("gb", GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.05)),
    ("rf", RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42))
], voting="soft")
final_model.fit(data[FEATURES], data["target"])

# feature importance from GB estimator
gb_est = final_model.estimators_[0]
importances = pd.Series(gb_est.feature_importances_, index=FEATURES).sort_values(ascending=False)

df["ai_prob"] = np.nan
valid_idx = df.dropna(subset=FEATURES).index
df.loc[valid_idx, "ai_prob"] = final_model.predict_proba(df.loc[valid_idx, FEATURES])[:,1]
df["score"] = (df["ai_prob"] * 100).round(0)

latest = df.iloc[-1]
if latest["score"] >= cfg["buy_score"]:
    signal = "BUY"
elif latest["score"] <= cfg["sell_score"]:
    signal = "SELL"
else:
    signal = "HOLD"

sl = latest["Close"] - cfg["sl_atr"]*latest["ATR"] if signal=="BUY" else latest["Close"] + cfg["sl_atr"]*latest["ATR"]
target = latest["Close"] + cfg["target_atr"]*latest["ATR"] if signal=="BUY" else latest["Close"] - cfg["target_atr"]*latest["ATR"]

# ---------------- HEADER ----------------
c1,c2,c3,c4,c5 = st.columns(5)
c1.metric(cfg["index"], f"{latest['Close']:.2f}", f"{df['Close'].pct_change().iloc[-1]*100:.2f}%")
c2.metric("AI Signal", signal)
c3.metric("Score (0-100)", f"{latest['score']:.0f}")
c4.metric(f"Walk-fwd Acc ({n_folds} folds)", f"{avg_acc*100:.1f}% ±{std_acc*100:.1f}")
c5.metric("HTF Trend", "UP" if latest["trend_up"]==1 else "DOWN")

st.caption(f"{'🟢 MARKET OPEN' if is_open else '🔴 MARKET CLOSED'} | SL: {sl:.1f} | Target: {target:.1f} | "
           f"Candles: {len(df)} | Model: GB+RF Ensemble")

if avg_acc < 0.55:
    st.warning(f"Walk-forward accuracy {avg_acc*100:.1f}% hai — variance ±{std_acc*100:.1f}% ke saath. "
               f"Isse pata chalta hai model abhi consistently edge nahi de raha. Feature/timeframe adjust kar ke retest karo.")

with st.expander("📊 Feature Importance (AI ne kin cheezon pe dhyan diya)"):
    st.bar_chart(importances.head(10))

# ---------------- PROFESSIONAL CHART ----------------
fig = make_subplots(rows=4, cols=1, shared_xaxes=True, row_heights=[0.5,0.15,0.15,0.2],
                     vertical_spacing=0.02)

fig.add_trace(go.Candlestick(
    x=df["Datetime"], open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
    name="Price", increasing_line_color="#26a69a", decreasing_line_color="#ef5350"
), row=1, col=1)
fig.add_trace(go.Scatter(x=df["Datetime"], y=df["EMA_fast"], name=f"EMA{cfg['fast_ema']}", line=dict(color="orange", width=1)), row=1, col=1)
fig.add_trace(go.Scatter(x=df["Datetime"], y=df["EMA_slow"], name=f"EMA{cfg['slow_ema']}", line=dict(color="cyan", width=1)), row=1, col=1)
fig.add_trace(go.Scatter(x=df["Datetime"], y=df["EMA200"], name="EMA200", line=dict(color="white", width=1, dash="dash")), row=1, col=1)
fig.add_trace(go.Scatter(x=df["Datetime"], y=df["VWAP"], name="VWAP", line=dict(color="yellow", width=1)), row=1, col=1)
fig.add_trace(go.Scatter(x=df["Datetime"], y=df["BB_up"], line=dict(color="gray", width=0.5, dash="dot"), name="BB Up"), row=1, col=1)
fig.add_trace(go.Scatter(x=df["Datetime"], y=df["BB_dn"], line=dict(color="gray", width=0.5, dash="dot"), name="BB Dn"), row=1, col=1)

buys = df[df["score"] >= cfg["buy_score"]]
sells = df[df["score"] <= cfg["sell_score"]]
fig.add_trace(go.Scatter(x=buys["Datetime"], y=buys["Low"]*0.997, mode="markers",
              marker=dict(symbol="triangle-up", color="lime", size=11), name="Buy"), row=1, col=1)
fig.add_trace(go.Scatter(x=sells["Datetime"], y=sells["High"]*1.003, mode="markers",
              marker=dict(symbol="triangle-down", color="red", size=11), name="Sell"), row=1, col=1)

fig.add_trace(go.Bar(x=df["Datetime"], y=df["Volume"], name="Volume", marker_color="#555"), row=2, col=1)

fig.add_trace(go.Scatter(x=df["Datetime"], y=df["RSI"], name="RSI", line=dict(color="violet")), row=3, col=1)
fig.add_hline(y=70, line_dash="dot", line_color="red", row=3, col=1)
fig.add_hline(y=30, line_dash="dot", line_color="green", row=3, col=1)

fig.add_trace(go.Scatter(x=df["Datetime"], y=df["MACD"], name="MACD", line=dict(color="cyan")), row=4, col=1)
fig.add_trace(go.Scatter(x=df["Datetime"], y=df["MACD_signal"], name="Signal", line=dict(color="orange")), row=4, col=1)
fig.add_trace(go.Bar(x=df["Datetime"], y=df["MACD_hist"], name="Histogram", marker_color="gray"), row=4, col=1)

fig.update_layout(
    template="plotly_dark", height=950, xaxis_rangeslider_visible=False,
    dragmode="pan", margin=dict(l=10,r=10,t=30,b=10),
    legend=dict(orientation="h", y=1.03), uirevision="keep"
)
# default zoom to last ~150 candles for readability, full history still scrollable
fig.update_xaxes(range=[df["Datetime"].iloc[-150], df["Datetime"].iloc[-1]], row=1, col=1)

st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True, "displaylogo": False})

st.caption(f"Last update: {datetime.now(IST).strftime('%d %b %H:%M:%S IST')} | "
           f"Folds: {n_folds} | Features: {len(FEATURES)} | Paper testing only.")
