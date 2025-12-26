import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import requests
from textblob import TextBlob
import statsmodels.api as sm
import zipfile
import io

# ==========================================================
# PAGE CONFIG
# ==========================================================
st.set_page_config(page_title="Market Sentiment Dashboard", layout="wide")

st.title("📊 Market Sentiment Dashboard")
st.caption("A portfolio-style dashboard analyzing price action, volatility, sentiment, simulations, and factor risk.")

# ==========================================================
# SIDEBAR — CONTROLS
# ==========================================================
with st.sidebar:

    st.header("⚙️ Dashboard Controls")
    st.markdown("---")

    # -------- STOCK SELECTION ----------
    st.subheader("📌 Select Ticker")
    ticker = st.text_input("Enter Stock Symbol:", "AAPL").upper()
    st.markdown("---")

    # -------- TIME RANGE ----------
    st.subheader("⏳ Time Range")
    period = st.selectbox(
        "Select Duration:",
        ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
        index=3
    )
    st.markdown("---")

    # -------- MONTE CARLO ----------
    st.subheader("🎲 Monte Carlo Simulation")
    num_sim = st.slider("Simulations", 100, 2000, 200)
    num_days = st.slider("Forecast Horizon (Days)", 50, 500, 252)
    st.markdown("---")

    # -------- STRESS TEST ----------
    st.subheader("⚠️ Stress Test Parameters")
    vol_shock = st.slider("Volatility Shock (%)", 0, 200, 50)
    corr_shock = st.slider("Correlation Tightening (%)", 0, 100, 50)
    st.markdown("---")

    # -------- FACTOR MODEL ----------
    enable_factor = st.checkbox("Enable Fama–French Model", True)
    st.markdown("---")

# ==========================================================
# FETCH PRICE DATA
# ==========================================================
st.markdown("## 📈 Market Overview")

data = yf.download(ticker, period=period)

if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.droplevel(1)

data = data.reset_index()

# ---------------- PRICE CHART + VIX ----------------
col1, col2 = st.columns(2)

with col1:
    st.subheader(f"{ticker} — Price Chart")
    fig = px.line(data, x="Date", y="Close", title=f"{ticker} Closing Price")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("VIX — Market Fear Index")
    vix = yf.download("^VIX", period="1y")
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.droplevel(1)
    vix = vix.reset_index()
    fig_vix = px.line(vix, x="Date", y="Close", title="VIX 1-Year Trend")
    st.plotly_chart(fig_vix, use_container_width=True)

# ==========================================================
# VOLATILITY METRICS
# ==========================================================
st.markdown("## 📊 Return & Volatility Metrics")

data["Returns"] = data["Close"].pct_change()
returns = data["Returns"].dropna()

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Annualized Volatility", f"{returns.std() * np.sqrt(252):.2%}")
with c2:
    st.metric("Avg Daily Return", f"{returns.mean():.4%}")
with c3:
    cum = (1 + returns).cumprod()
    max_dd = (cum / cum.cummax() - 1).min()
    st.metric("Max Drawdown", f"{max_dd:.2%}")

# ==========================================================
# NEWS SENTIMENT
# ==========================================================
st.markdown("## 📰 News Sentiment")

def fetch_news_sentiment(tkr):
    stock = yf.Ticker(tkr)
    try:
        news = stock.news
    except:
        return None, 0
    
    if not news:
        return None, 0

    scores = []
    for n in news[:10]:
        text = n.get("summary", "")
        if text:
            scores.append(TextBlob(text).sentiment.polarity)

    return (sum(scores) / len(scores)) if scores else 0, len(scores)

sent_score, sent_count = fetch_news_sentiment(ticker)

if sent_score is None:
    st.warning("No news available.")
else:
    label = "😊 Positive" if sent_score > 0.1 else "😐 Neutral" if sent_score > -0.1 else "😡 Negative"
    st.metric("Sentiment Score", f"{sent_score:.2f}", label)
    st.caption(f"Articles analyzed: {sent_count}")

# ==========================================================
# MONTE CARLO SIMULATION
# ==========================================================
st.markdown("## 🎲 Monte Carlo Simulation")

last_price = data["Close"].iloc[-1]
sim_df = pd.DataFrame()

for s in range(num_sim):
    prices = [last_price]
    for _ in range(num_days):
        prices.append(prices[-1] * (1 + np.random.normal(returns.mean(), returns.std())))
    sim_df[s] = prices

fig_mc = px.line(sim_df, title="Monte Carlo Forecast")
st.plotly_chart(fig_mc, use_container_width=True)

end_prices = sim_df.iloc[-1]

c1, c2, c3 = st.columns(3)
c1.metric("Worst 5%", f"${np.percentile(end_prices, 5):,.2f}")
c2.metric("Median", f"${np.percentile(end_prices, 50):,.2f}")
c3.metric("Best 95%", f"${np.percentile(end_prices, 95):,.2f}")

# ==========================================================
# STRESS TEST
# ==========================================================
st.markdown("## ⚠️ Stress Test (Crash Scenario)")

shock_sigma = returns.std() * (1 + vol_shock/100) * (1 + corr_shock/100)

stress_df = pd.DataFrame()
for s in range(num_sim):
    prices = [last_price]
    for _ in range(num_days):
        prices.append(prices[-1] * (1 + np.random.normal(returns.mean(), shock_sigma)))
    stress_df[s] = prices

fig_str = px.line(stress_df, title="Stressed Monte Carlo Simulation")
st.plotly_chart(fig_str, use_container_width=True)

# ==========================================================
# FACTOR MODEL — Fama–French 4-Factor Regression
# ==========================================================
if enable_factor:

    st.markdown("## 📘 Regression Summary")

    # ---------------- DOWNLOAD FAMA–FRENCH FACTORS ----------------
    @st.cache_data
    def load_ff_factors():
        ff_url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"
        mom_url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip"

        ff_zip = zipfile.ZipFile(io.BytesIO(requests.get(ff_url).content))
        mom_zip = zipfile.ZipFile(io.BytesIO(requests.get(mom_url).content))

        ff_csv = ff_zip.namelist()[0]
        mom_csv = mom_zip.namelist()[0]

        ff = pd.read_csv(ff_zip.open(ff_csv), skiprows=3)
        mom = pd.read_csv(mom_zip.open(mom_csv), skiprows=13)

        ff.rename(columns={ff.columns[0]: "Date"}, inplace=True)
        mom.rename(columns={mom.columns[0]: "Date"}, inplace=True)

        ff = ff[pd.to_numeric(ff["Date"], errors="coerce").notnull()]
        mom = mom[pd.to_numeric(mom["Date"], errors="coerce").notnull()]

        ff["Date"] = ff["Date"].astype(int)
        mom["Date"] = mom["Date"].astype(int)

        for col in ["Mkt-RF", "SMB", "HML", "RF"]:
            ff[col] /= 100
        mom["Mom"] /= 100

        merged = pd.merge(ff, mom[["Date", "Mom"]], on="Date")
        return merged

    factors = load_ff_factors()

    # ---------------- MERGE WITH RETURNS ----------------
    ret_df = data[["Date", "Returns"]].dropna().copy()
    ret_df["Date"] = ret_df["Date"].dt.strftime("%Y%m%d").astype(int)

    merged = pd.merge(ret_df, factors, on="Date")
    Y = merged["Returns"] - merged["RF"]
    X = merged[["Mkt-RF", "SMB", "HML", "Mom"]]
    X = sm.add_constant(X)

    model = sm.OLS(Y, X).fit()

    # ---------------- DISPLAY REGRESSION TABLE ----------------
summary_html = model.summary().as_html()

st.markdown("""
<style>
.reg-box {
    background: #ffffff;
    padding: 20px;
    border-radius: 8px;
    border: 1px solid #e6e6e6;
    box-shadow: 0px 2px 6px rgba(0,0,0,0.05);
    max-height: 600px;
    overflow-y: scroll;
    width: 100%;
    font-size: 14px;
    white-space: nowrap;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="reg-box">' + summary_html + '</div>', unsafe_allow_html=True)