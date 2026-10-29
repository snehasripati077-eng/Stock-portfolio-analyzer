# 📈 Stock Portfolio Performance Analyzer

Interactive Streamlit dashboard that analyses a stock portfolio against the **Nifty 50** benchmark.

**Stack:** Python · Pandas · NumPy · Streamlit · Plotly · yfinance

## Features
- Pick Nifty 50 stocks, date range, equal or custom weights, daily-rebalanced or buy-and-hold
- Cumulative returns, CAGR, historical (and rolling) volatility, Sharpe ratio, max drawdown, beta, alpha
- Interactive charts: allocation, price movement, portfolio vs Nifty 50, drawdown, correlation, monthly returns
- Auto-generated plain-English insights
- CSV export
- Offline **demo mode** with simulated data (no internet needed)

## Run locally
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
Opens at http://localhost:8501

## Project structure
```
app.py                  Streamlit UI
utils/data_loader.py    Yahoo Finance + demo data
utils/analytics.py      Returns, volatility, risk metrics, insights
utils/charts.py         Plotly charts
.streamlit/config.toml  Theme
```

## Deploy free
Push to GitHub, then deploy on [Streamlit Community Cloud](https://streamlit.io/cloud) with `app.py` as the entry point.

## Methodology
- Returns are simple daily returns from adjusted close prices
- Annualised volatility = daily std × √252
- Sharpe = (annualised mean return − risk-free) / volatility (default risk-free 6.5%)
- Beta/alpha computed against the Nifty 50 (^NSEI)

*Educational use only. Not investment advice.*
