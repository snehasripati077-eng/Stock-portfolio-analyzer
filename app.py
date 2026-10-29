"""Stock Portfolio Performance Analyzer - Streamlit app.

Run with:  streamlit run app.py
"""
from datetime import date, timedelta

import pandas as pd
import streamlit as st

from utils import analytics as A
from utils import charts as C
from utils import data_loader as D

st.set_page_config(page_title="Stock Portfolio Performance Analyzer", page_icon="📈", layout="wide")

st.markdown(
    """
    <style>
    div[data-testid="stMetric"] {background:#F3F4F8; padding:14px 16px; border-radius:12px;}
    .block-container {padding-top: 2rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=3600, show_spinner="Loading price data...")
def load_prices(tickers: tuple, start, end, source: str) -> pd.DataFrame:
    if source == "Demo data (offline)":
        return D.generate_demo_data(list(tickers), start, end)
    return D.fetch_live_data(list(tickers), start, end)


# ---------------------------- Sidebar ----------------------------
st.sidebar.title("⚙️ Portfolio Setup")

source = st.sidebar.radio("Data source", ["Live (Yahoo Finance)", "Demo data (offline)"],
                          help="Use demo data if you have no internet access.")

names = st.sidebar.multiselect("Select stocks (Nifty 50)", list(D.NIFTY_STOCKS.keys()),
                               default=["Reliance Industries", "TCS", "HDFC Bank", "Infosys", "ITC"])

default_start = date.today() - timedelta(days=3 * 365)
start = st.sidebar.date_input("Start date", default_start, max_value=date.today() - timedelta(days=60))
end = st.sidebar.date_input("End date", date.today(), max_value=date.today())

weight_mode = st.sidebar.radio("Allocation", ["Equal weight", "Custom weights"])
rebalance = st.sidebar.radio("Rebalancing", ["Daily (constant weights)", "Buy & hold"]) == "Daily (constant weights)"
risk_free = st.sidebar.number_input("Risk-free rate (% p.a.)", 0.0, 15.0, 6.5, 0.1) / 100
vol_window = st.sidebar.slider("Rolling volatility window (days)", 10, 120, 30, 5)

if not names:
    st.title("📈 Stock Portfolio Performance Analyzer")
    st.info("Select at least one stock in the sidebar to begin.")
    st.stop()

tickers = [D.NIFTY_STOCKS[n] for n in names]

raw_weights = {}
if weight_mode == "Custom weights":
    st.sidebar.caption("Weights are normalised to 100%.")
    for n, t in zip(names, tickers):
        raw_weights[t] = st.sidebar.number_input(f"{n} (%)", 0.0, 100.0, round(100 / len(names), 1), 1.0)
else:
    raw_weights = {t: 1.0 for t in tickers}
weights = A.normalize_weights(raw_weights)
weights = {t: w for t, w in weights.items() if w > 0}
if not weights:
    st.warning("All weights are zero.")
    st.stop()

# ---------------------------- Data ----------------------------
if start >= end:
    st.error("Start date must be before end date.")
    st.stop()

try:
    prices = load_prices(tuple(sorted(weights.keys())), start, end, source)
except Exception as exc:  # network errors, bad tickers, etc.
    st.error(f"Could not load data: {exc}")
    st.info("Switch to **Demo data (offline)** in the sidebar to try the app without internet.")
    st.stop()

if len(prices) < 40:
    st.error("Not enough data points. Choose a longer date range.")
    st.stop()

stock_prices = prices[list(weights.keys())]
bench_prices = prices[D.BENCHMARK]
rets = A.daily_returns(prices)
bench_ret = rets[D.BENCHMARK]
port_ret = A.portfolio_returns(prices, weights, rebalance).loc[bench_ret.index]

metrics = A.summary_metrics(port_ret, bench_ret, risk_free)
bench_metrics = A.summary_metrics(bench_ret, bench_ret, risk_free)
stock_df = A.stock_summary(prices, weights, D.TICKER_TO_NAME, risk_free)

# ---------------------------- Header ----------------------------
st.title("📈 Stock Portfolio Performance Analyzer")
st.caption(f"{prices.index[0]:%d %b %Y} → {prices.index[-1]:%d %b %Y} · {len(weights)} stocks · "
           f"benchmark: Nifty 50 · {'demo data' if source.startswith('Demo') else 'Yahoo Finance'}")
if source.startswith("Demo"):
    st.warning("Showing simulated demo data, not real market prices.")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Return", f"{metrics['Total Return']:.1%}",
          f"{(metrics['Total Return'] - bench_metrics['Total Return']) * 100:+.1f} pts vs Nifty")
k2.metric("CAGR", f"{metrics['CAGR']:.1%}", f"{(metrics['CAGR'] - bench_metrics['CAGR']) * 100:+.1f} pts vs Nifty")
k3.metric("Volatility (ann.)", f"{metrics['Volatility']:.1%}",
          f"{(metrics['Volatility'] - bench_metrics['Volatility']) * 100:+.1f} pts vs Nifty", delta_color="inverse")
k4.metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.2f}",
          f"{metrics['Sharpe Ratio'] - bench_metrics['Sharpe Ratio']:+.2f} vs Nifty")

k5, k6, k7, k8 = st.columns(4)
k5.metric("Max Drawdown", f"{metrics['Max Drawdown']:.1%}")
k6.metric("Beta vs Nifty", f"{metrics['Beta']:.2f}")
k7.metric("Alpha (ann.)", f"{metrics['Alpha']:.1%}")
k8.metric("Nifty 50 Return", f"{bench_metrics['Total Return']:.1%}")

tab_over, tab_stocks, tab_risk, tab_data = st.tabs(["Overview", "Stocks", "Risk", "Data & Export"])

# ---------------------------- Overview ----------------------------
with tab_over:
    left, right = st.columns([2, 1])
    left.plotly_chart(C.cumulative_chart(A.cumulative_returns(port_ret), A.cumulative_returns(bench_ret)))
    right.plotly_chart(C.allocation_pie(weights, D.TICKER_TO_NAME))

    st.subheader("Key insights")
    for line in A.generate_insights(metrics, bench_metrics, stock_df):
        st.markdown(f"- {line}")

    st.plotly_chart(C.monthly_heatmap(A.monthly_returns_table(port_ret)))

# ---------------------------- Stocks ----------------------------
with tab_stocks:
    normalized = st.toggle("Rebase prices to 100", value=True)
    st.plotly_chart(C.price_chart(stock_prices, D.TICKER_TO_NAME, normalized))

    c1, c2 = st.columns(2)
    c1.plotly_chart(C.stock_return_bar(stock_df))
    c2.plotly_chart(C.risk_return_scatter(stock_df))

    st.subheader("Stock performance table")
    display = stock_df.drop(columns=["Ticker"]).copy()
    st.dataframe(
        display.style.format({
            "Weight": "{:.1%}", "Total Return": "{:.1%}", "CAGR": "{:.1%}", "Volatility": "{:.1%}",
            "Sharpe": "{:.2f}", "Max Drawdown": "{:.1%}", "Contribution": "{:.1%}",
        }),
        hide_index=True,
    )

# ---------------------------- Risk ----------------------------
with tab_risk:
    r1, r2 = st.columns(2)
    r1.plotly_chart(C.rolling_vol_chart(A.rolling_volatility(port_ret, vol_window).dropna(),
                                        A.rolling_volatility(bench_ret, vol_window).dropna(), vol_window))
    r2.plotly_chart(C.drawdown_chart(A.drawdown_series(port_ret), A.drawdown_series(bench_ret)))
    if len(weights) > 1:
        st.plotly_chart(C.correlation_heatmap(rets[list(weights.keys())], D.TICKER_TO_NAME))
    else:
        st.info("Add more stocks to see the correlation matrix.")

# ---------------------------- Data ----------------------------
with tab_data:
    st.subheader("Portfolio vs benchmark summary")
    summary = pd.DataFrame({"Portfolio": metrics, "Nifty 50": bench_metrics})
    st.dataframe(summary.style.format("{:.3f}"))

    export = pd.DataFrame({
        "Portfolio daily return": port_ret,
        "Nifty 50 daily return": bench_ret,
        "Portfolio cumulative": A.cumulative_returns(port_ret),
        "Nifty 50 cumulative": A.cumulative_returns(bench_ret),
    })
    st.download_button("⬇️ Download returns (CSV)", export.to_csv().encode(), "portfolio_returns.csv", "text/csv")
    st.download_button("⬇️ Download prices (CSV)", prices.to_csv().encode(), "prices.csv", "text/csv")

    with st.expander("Raw prices"):
        st.dataframe(prices.round(2))

st.divider()
st.caption("For educational purposes only. Not investment advice. Past performance does not guarantee future results.")
