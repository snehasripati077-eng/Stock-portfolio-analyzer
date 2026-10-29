"""Data loading: live Yahoo Finance prices, or reproducible demo data for offline use."""
import zlib

import numpy as np
import pandas as pd

BENCHMARK = "^NSEI"  # Nifty 50 index

# A selection of Nifty 50 constituents (Yahoo Finance NSE tickers)
NIFTY_STOCKS = {
    "Reliance Industries": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "Infosys": "INFY.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "Hindustan Unilever": "HINDUNILVR.NS",
    "ITC": "ITC.NS",
    "State Bank of India": "SBIN.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
    "Larsen & Toubro": "LT.NS",
    "Kotak Mahindra Bank": "KOTAKBANK.NS",
    "Axis Bank": "AXISBANK.NS",
    "Asian Paints": "ASIANPAINT.NS",
    "Maruti Suzuki": "MARUTI.NS",
    "Sun Pharma": "SUNPHARMA.NS",
    "Titan": "TITAN.NS",
    "Tata Motors": "TATAMOTORS.NS",
    "Wipro": "WIPRO.NS",
    "HCL Technologies": "HCLTECH.NS",
    "Bajaj Finance": "BAJFINANCE.NS",
}

TICKER_TO_NAME = {v: k for k, v in NIFTY_STOCKS.items()}
TICKER_TO_NAME[BENCHMARK] = "Nifty 50"


def generate_demo_data(tickers, start, end):
    """Simulate correlated daily prices (market factor + idiosyncratic noise).

    Deterministic per ticker, so the demo looks the same on every run.
    """
    dates = pd.bdate_range(start, end)
    n = len(dates)
    if n < 2:
        raise ValueError("Date range is too short.")

    market = np.random.default_rng(2024).normal(0.0005, 0.009, n)

    prices = {BENCHMARK: 18000 * np.cumprod(1 + market)}
    for t in tickers:
        seed = zlib.crc32(t.encode())
        rng = np.random.default_rng(seed)
        beta = 0.7 + (seed % 70) / 100          # 0.70 - 1.39
        drift = 0.0001 + (seed % 7) * 0.00007   # stock-specific drift
        vol = 0.006 + (seed % 5) * 0.001
        r = beta * market + rng.normal(drift, vol, n)
        prices[t] = (200 + seed % 3000) * np.cumprod(1 + r)

    return pd.DataFrame(prices, index=dates)


def fetch_live_data(tickers, start, end):
    """Download adjusted close prices from Yahoo Finance (needs internet)."""
    import yfinance as yf

    symbols = list(tickers) + [BENCHMARK]
    raw = yf.download(symbols, start=start, end=end, auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        raise ValueError("No data returned. Check your internet connection or date range.")

    prices = raw["Close"]
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(symbols[0])
    prices = prices.dropna(how="all").ffill().dropna()
    missing = [s for s in symbols if s not in prices.columns]
    if missing:
        raise ValueError(f"No data for: {', '.join(missing)}")
    return prices[symbols]
