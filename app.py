import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
from zoneinfo import ZoneInfo

st.set_page_config(layout="wide", page_title="AI Signal Pro",
                   initial_sidebar_state="collapsed")

IST = ZoneInfo("Asia/Kolkata")
SYMBOLS = {"NIFTY 50": "^NSEI", "BANK NIFTY": "^NSEBANK"}
TF = {"5m": ("5m", "60d"), "15m": ("15m", "60d"),
      "1h": ("60m", "1y"), "1D": ("1d", "5y")}

now = datetime.now(IST)
hm = (now.hour, now.minute)
market_open = now.weekday() < 5 and (9, 15) <= hm <= (15, 30)

with st.sidebar:
    st.header("Settings")
    name = st.selectbox("Index", list(SYMBOLS))
    tf = st.selectbox("Timeframe", list(TF), index=1)
    bars = st.slider("Candles dikhao", 50, 400, 150, step=10)
    fast = st.number_input("Fast EMA", 3, 50, 9)
    slow = st.number_input("Slow EMA", 10, 200, 21)
    buy_th = st.slider("Buy score >=", 55, 85, 62)
    sell_th = 100 - buy_th
    st.caption(f"Sell score <= {sell_th}")
    sl_mult = st.number_input("Stop Loss (x ATR)", 0.5, 5.0, 1.5, step=0.1)
    tg_mult = st.number_input("Target (x ATR)", 0.5, 8.0, 2.5, step=0.1)
    cost = st.number_input("Cost per trade (%)", 0.0, 1.0, 0.03, step=0.01)
    auto = st.toggle("Auto refresh (30s)", value=True)

if auto and market_open:
    st_autorefresh(interval=30000, key="refresh")

@st.cache_data(ttl=20)
def get_data(sym, interval, period):
    df = yf.download(sym, period=period, interval=interval,
                     progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna()
    if len(df) == 0:
        return df
    idx = df.index
    if idx.tz is not None:
        idx = idx.tz_convert("Asia/Kolkata").tz_localize(None)
    df.index = idx
    return df

def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)

def indicators(df, fast, slow):
    df = df.copy()
    c, h, l, o = df.Close, df.High, df.Low, df.Open
    df["ema_f"] = c.ewm(span=fast, adjust=False).mean()
    df["ema_s"] = c.ewm(span=slow, adjust=False).mean()
    df["rsi"] = rsi(c)
    tr = pd.concat([h - l, (h - c.shift()).abs(),
                    (l - c.shift()).abs()], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1/14, adjust=False).mean()
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    df["hist"] = macd - macd.ewm(span=9, adjust=False).mean()
    ma20, sd20 = c.rolling(20).mean(), c.rolling(20).std()
    df["pctb"] = (c - (ma20 - 2 * sd20)) / (4 * sd20 + 1e-9)
    df["hi20"] = h.rolling(20).max()
    df["lo20"] = l.rolling(20).min()
    df["pos20"] = (c - df.lo20) / (df.hi20 - df.lo20 + 1e-9)
    return df

def make_features(df):
    c = df.Close
    f = pd.DataFrame(index=df.index)
    for n in (1, 3, 5, 10):
        f[f"r{n}"] = c.pct_change(n)
    f["ema_dist"] = c / df.ema_f - 1
    f["ema_gap"] = df.ema_f / df.ema_s - 1
    f["rsi"] = df.rsi
    f["atr_pct"] = df.atr / c
    f["vol_ratio"] = f.atr_pct / f.atr_pct.rolling(50).mean()
    f["hist"] = df["hist"] / c
    f["pctb"] = df.pctb
    f["pos20"] = df.pos20
    f["body"] = (c - df.Open) / (df.High - df.Low + 1e-9)
    if df.index.to_series().diff().median() < pd.Timedelta(days=1):
        f["hour"] = df.index.hour + df.index.minute / 60
    return f.replace([np.inf, -np.inf], np.nan)

@st.cache_data(show_spinner="AI model train ho raha hai...")
def run_ml(_df, key):
    X = make_features(_df)
    y = (_df.Close.shift(-1) > _df.Close).astype(int)
    lab = X.iloc[:-1].dropna()
    yl = y.loc[lab.index]
    n = len(lab)
    out = {"ok": False}
    if n < 250:
        return out
    mk = lambda: RandomForestClassifier(
        n_estimators=150, max_depth=5, min_samples_leaf=25,
        n_jobs=-1, random_state=0)
    proba = pd.Series(np.nan, index=X.index)
    start = int(n * 0.5)
    edges = np.linspace(start, n, 6).astype(int)
    for i in range(5):
        m = mk().fit(lab.iloc[:edges[i]], yl.iloc[:edges[i]])
        te = lab.iloc[edges[i]:edges[i + 1]]
        if len(te):
            proba.loc[te.index] = m.predict_proba(te)[:, 1]
    final = mk().fit(lab, yl)
    if not X.iloc[[-1]].isna().any(axis=1).iloc[0]:
        proba.iloc[-1] = final.predict_proba(X.iloc[[-1]])[:, 1][0]
    oos = proba.iloc[:-1].dropna()
    yt = yl.loc[oos.index]
    pred = (oos > 0.5).astype(int)
    acc = (pred == yt).mean() * 100
    base = max(yt.mean(), 1 - yt.mean()) * 100
    hc = (oos > 0.55) | (oos < 0.45)
    hc_acc = (pred[hc] == yt[hc]).mean() * 100 if hc.sum() else float("nan")
    imp = pd.Series(final.feature_importances_, index=lab.columns)
    imp = imp.sort_values(ascending=False).head(5)
    return {"ok": True, "proba": proba, "oos_start": oos.index[0],
            "acc": acc, "base": base, "hc_acc": hc_acc,
            "hc_n": int(hc.sum()), "n_oos": len(oos), "imp": imp}

def build_signals(df, p, buy_th, sell_th):
    df = df.copy()
    df["p"] = p
    trend = np.tanh((df.ema_f - df.ema_s) / (df.atr + 1e-9))
    mom = (0.6 * ((df.rsi - 50) / 25).clip(-1, 1)
           + 0.4 * np.tanh(df["hist"] / (df.atr + 1e-9) * 3))
    struct = ((df.pos20 - 0.5) * 2).clip(-1, 1)
    ml = ((df.p - 0.5) * 6).clip(-1, 1).fillna(0)
    x = (0.30 * trend + 0.20 * mom + 0.15 * struct + 0.35 * ml).clip(-1, 1)
    df["score"] = 50 + 50 * x
    raw = pd.Series(np.nan, index=df.index)
    raw[(df.score - 50).abs() < 5] = 0
    raw[(df.score >= buy_th) & (df.p > 0.5)] = 1
    raw[(df.score <= sell_th) & (df.p < 0.5)] = -1
    df["pos"] = raw.ffill().fillna(0)
    df["signal"] = df.pos.diff().fillna(0)
    return df

def backtest(d, cost_pct):
    ret = d.Close.pct_change().fillna(0)
    chg = d.pos.diff().abs().fillna(0)
    strat = d.pos.shift(1).fillna(0) * ret - chg * cost_pct / 100
    eq = (1 + strat).cumprod()
    dd = eq / eq.cummax() - 1
    seg = (d.pos != d.pos.shift()).cumsum()
    tr = strat.groupby(seg).sum()
    active = d.pos.groupby(seg).first() != 0
    return eq, dd, tr[active]

interval, period = TF[tf]
raw_df = get_data(SYMBOLS[name], interval, period)
if len(raw_df) < 300:
    st.error("Data kam hai ya nahi mila. Alag timeframe try karo.")
    st.stop()

base = indicators(raw_df, fast, slow)
ml = run_ml(base, f"{name}-{tf}-{fast}-{slow}-{raw_df.index[-1]}")
if not ml["ok"]:
    st.error("AI ke liye data kam hai. 15m ya 1D timeframe try karo.")
    st.stop()

df = build_signals(base, ml["proba"], buy_th, sell_th)
bt = df.loc[ml["oos_start"]:]
eq, dd, tr = backtest(bt, cost)
bh = (bt.Close / bt.Close.iloc[0] - 1) * 100

last, prev = df.iloc[-1], df.iloc[-2]
chg = (last.Close / prev.Close - 1) * 100
px_col = "#26a69a" if chg >= 0 else "#ef5350"
state = {1: "BUY", -1: "SELL", 0: "WAIT"}[int(last.pos)]
s_col = {"BUY": "#00e676", "SELL": "#ff1744", "WAIT": "#9aa0a6"}[state]
status = "🟢 MARKET OPEN" if market_open else "🔴 MARKET CLOSED"
p_up = last.p * 100 if pd.notna(last.p) else float("nan")

if state == "BUY":
    sl, tg = last.Close - sl_mult * last.atr, last.Close + tg_mult * last.atr
elif state == "SELL":
    sl, tg = last.Close + sl_mult * last.atr, last.Close - tg_mult * last.atr
else:
    sl = tg = None

st.markdown(f"""
<div style="display:flex;flex-wrap:wrap;gap:22px;align-items:flex-end;">
  <div><div style="opacity:.6;font-size:13px">{name} · {status}</div>
  <div style="font-size:32px;font-weight:700;color:{px_col}">{last.Close:,.2f}
  <span style="font-size:16px">({chg:+.2f}%)</span></div></div>
  <div><div style="opacity:.6;font-size:13px">AI SIGNAL</div>
  <div style="font-size:30px;font-weight:800;color:{s_col}">{state}</div></div>
  <div><div style="opacity:.6;font-size:13px">Score (0-100)</div>
  <div style="font-size:26px;font-weight:700">{last.score:.0f}</div></div>
  <div><div style="opacity:.6;font-size:13px">AI: agli candle UP</div>
  <div style="font-size:26px;font-weight:700">{p_up:.0f}%</div></div>
  {"" if sl is None else f'''<div><div style="opacity:.6;font-size:13px">Stop Loss / Target</div>
  <div style="font-size:20px;font-weight:700"><span style="color:#ff1744">{sl:,.0f}</span> / <span style="color:#00e676">{tg:,.0f}</span></div></div>'''}
</div>
""", unsafe_allow_html=True)

st.write("")
wins, losses = tr[tr > 0], tr[tr < 0]
win_rate = len(wins) / len(tr) * 100 if len(tr) else 0
pf = wins.sum() / abs(losses.sum()) if len(losses) and losses.sum() != 0 else float("nan")

c1, c2, c3, c4 = st.columns(4)
c1.metric("AI accuracy (unseen data)", f"{ml['acc']:.1f}%",
          f"{ml['acc'] - ml['base']:+.1f}% vs baseline")
c2.metric("Strong-signal accuracy", f"{ml['hc_acc']:.1f}%",
          f"{ml['hc_n']} signals")
c3.metric("Strategy vs Buy&Hold",
          f"{(eq.iloc[-1] - 1) * 100:+.2f}%", f"B&H {bh.iloc[-1]:+.2f}%")
c4.metric("Trades / Win% / PF", f"{len(tr)} / {win_rate:.0f}% / {pf:.2f}")

edge = ml["acc"] - ml["base"]
if edge >= 2 and ml["hc_n"] >= 30:
    st.success("AI ko unseen data pe halka edge mila. Phir bhi paper trading karo.")
else:
    st.warning("AI ko abhi koi pakka edge nahi mila (baseline ke kareeb). "
               "Is signal pe akele bharosa mat karo. Timeframe/settings badal ke test karo.")

view = df.tail(bars)
eqv = eq.reindex(view.index)
bhv = (1 + bh / 100).reindex(view.index)

fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                    row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.02)
fig.add_trace(go.Candlestick(
    x=view.index, open=view.Open, high=view.High, low=view.Low, close=view.Close,
    increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
    increasing_fillcolor="#26a69a", decreasing_fillcolor="#ef5350",
    name="Price", showlegend=False), row=1, col=1)
fig.add_trace(go.Scatter(x=view.index, y=view.ema_f, name=f"EMA {fast}",
              line=dict(color="#f5a623", width=1.3)), row=1, col=1)
fig.add_trace(go.Scatter(x=view.index, y=view.ema_s, name=f"EMA {slow}",
              line=dict(color="#4fc3f7", width=1.3)), row=1, col=1)
fig.add_trace(go.Scatter(x=view.index, y=view.hi20, name="Resistance",
              line=dict(color="rgba(239,83,80,.5)", width=1, dash="dot")), row=1, col=1)
fig.add_trace(go.Scatter(x=view.index, y=view.lo20, name="Support",
              line=dict(color="rgba(38,166,154,.5)", width=1, dash="dot")), row=1, col=1)
buys, sells = view[view.signal > 0], view[view.signal < 0]
fig.add_trace(go.Scatter(x=buys.index, y=buys.Low * 0.9993, mode="markers",
              marker=dict(symbol="triangle-up", size=13, color="#00e676",
                          line=dict(width=1, color="white")), name="Buy"), row=1, col=1)
fig.add_trace(go.Scatter(x=sells.index, y=sells.High * 1.0007, mode="markers",
              marker=dict(symbol="triangle-down", size=13, color="#ff1744",
                          line=dict(width=1, color="white")), name="Sell"), row=1, col=1)
fig.add_hline(y=float(last.Close), line_dash="dot", line_color=px_col,
              line_width=1, row=1, col=1)
if sl is not None:
    fig.add_hline(y=float(sl), line_color="#ff1744", line_width=1, row=1, col=1)
    fig.add_hline(y=float(tg), line_color="#00e676", line_width=1, row=1, col=1)

fig.add_trace(go.Scatter(x=view.index, y=view.score, name="Score",
              line=dict(color="#ab47bc", width=1.4), showlegend=False), row=2, col=1)
for lvl in (sell_th, 50, buy_th):
    fig.add_hline(y=lvl, line_dash="dot", line_color="gray",
                  line_width=0.6, row=2, col=1)
fig.add_trace(go.Scatter(x=eqv.index, y=eqv, name="Strategy",
              line=dict(color="#66bb6a", width=1.5)), row=3, col=1)
fig.add_trace(go.Scatter(x=bhv.index, y=bhv, name="Buy&Hold",
              line=dict(color="#9aa0a6", width=1, dash="dot")), row=3, col=1)

breaks = [dict(bounds=["sat", "mon"])]
if tf != "1D":
    breaks.append(dict(bounds=[15.5, 9.25], pattern="hour"))
fig.update_xaxes(rangebreaks=breaks, rangeslider_visible=False,
                 showspikes=True, spikemode="across", spikethickness=1,
                 spikecolor="#888", showgrid=False)
fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)", side="right")
fig.update_yaxes(range=[0, 100], row=2, col=1)
fig.update_layout(height=760, template="plotly_dark", dragmode="pan",
                  hovermode="x unified", margin=dict(l=5, r=5, t=10, b=10),
                  legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
                  paper_bgcolor="#0e1117", plot_bgcolor="#0e1117")
st.plotly_chart(fig, use_container_width=True, config={
    "scrollZoom": True, "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"]})

with st.expander("AI ne kin cheezon pe dhyan diya (top 5)"):
    st.dataframe(ml["imp"].rename("importance").round(3), use_container_width=True)
with st.expander("Last 10 signals"):
    sg = df[df.signal != 0][["Close", "score", "signal"]].tail(10).copy()
    sg["type"] = sg.signal.apply(lambda v: "BUY" if v > 0 else "SELL")
    st.dataframe(sg[["Close", "score", "type"]].iloc[::-1], use_container_width=True)

st.caption(f"Last update: {now.strftime('%d %b %H:%M:%S')} IST | Backtest sirf unseen "
           f"({ml['n_oos']} candles) data pe | Data Yahoo (delayed) | Sirf paper testing.")
