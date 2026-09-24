import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh

st.set_page_config(layout="wide", page_title="Index Signal Bot")
st_autorefresh(interval=15000, key="refresh")

SYMBOLS = {"NIFTY 50": "^NSEI", "BANK NIFTY": "^NSEBANK"}

with st.sidebar:
    name = st.selectbox("Index", list(SYMBOLS))
    tf = st.selectbox("Timeframe", ["1m", "5m", "15m"], index=1)
    fast = st.number_input("Fast EMA", 3, 50, 9)
    slow = st.number_input("Slow EMA", 10, 200, 21)
    rsi_len = st.number_input("RSI length", 5, 30, 14)
    cost = st.number_input("Cost per trade (%)", 0.0, 1.0, 0.03, step=0.01)

@st.cache_data(ttl=10)
def get_data(sym, tf):
    df = yf.download(sym, period="5d", interval=tf,
                     progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna()
    df.index = df.index.tz_convert("Asia/Kolkata")
    return df

def rsi(s, n):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)

# ---------- STRATEGY (yahan apni strategy daalo) ----------
def strategy(df, fast, slow, rsi_len):
    df = df.copy()
    df["ema_f"] = df["Close"].ewm(span=fast, adjust=False).mean()
    df["ema_s"] = df["Close"].ewm(span=slow, adjust=False).mean()
    df["rsi"] = rsi(df["Close"], rsi_len)
    long_ = (df.ema_f > df.ema_s) & (df.rsi > 50)
    short_ = (df.ema_f < df.ema_s) & (df.rsi < 50)
    df["pos"] = 0
    df.loc[long_, "pos"] = 1
    df.loc[short_, "pos"] = -1
    df["signal"] = df["pos"].diff().fillna(0)
    return df

def backtest(df, cost_pct):
    ret = df["Close"].pct_change().fillna(0)
    trades = df["pos"].diff().abs().fillna(0)
    strat = df["pos"].shift(1).fillna(0) * ret - trades * cost_pct / 100
    eq = (1 + strat).cumprod()
    dd = eq / eq.cummax() - 1
    return eq, dd, int((trades > 0).sum())

df = strategy(get_data(SYMBOLS[name], tf), fast, slow, rsi_len)
eq, dd, n_trades = backtest(df, cost)

last = df.iloc[-1]
c1, c2, c3, c4 = st.columns(4)
c1.metric(name, f"{last.Close:,.2f}")
c2.metric("Position", {1: "LONG", -1: "SHORT", 0: "FLAT"}[int(last.pos)])
c3.metric("Backtest return", f"{(eq.iloc[-1]-1)*100:.2f}%")
c4.metric("Max drawdown", f"{dd.min()*100:.2f}%")

fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                    row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.03)
fig.add_trace(go.Candlestick(x=df.index, open=df.Open, high=df.High,
              low=df.Low, close=df.Close, name="Price"), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df.ema_f, name=f"EMA {fast}"), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df.ema_s, name=f"EMA {slow}"), row=1, col=1)
buys = df[df.signal > 0]
sells = df[df.signal < 0]
fig.add_trace(go.Scatter(x=buys.index, y=buys.Low * 0.9995, mode="markers",
              marker=dict(symbol="triangle-up", size=11, color="green"),
              name="Buy"), row=1, col=1)
fig.add_trace(go.Scatter(x=sells.index, y=sells.High * 1.0005, mode="markers",
              marker=dict(symbol="triangle-down", size=11, color="red"),
              name="Sell"), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df.rsi, name="RSI"), row=2, col=1)
fig.add_trace(go.Scatter(x=eq.index, y=eq, name="Equity"), row=3, col=1)
fig.update_layout(height=800, xaxis_rangeslider_visible=False,
                  template="plotly_dark", margin=dict(l=10, r=10, t=30, b=10))
st.plotly_chart(fig, use_container_width=True)
st.caption(f"Trades (backtest window): {n_trades} | Sirf paper testing ke liye.")
