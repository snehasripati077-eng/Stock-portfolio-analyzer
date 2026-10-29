"""Portfolio analytics: returns, volatility, risk metrics and auto-generated insights."""
import numpy as np
import pandas as pd

TRADING_DAYS = 252


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna()


def normalize_weights(weights: dict) -> dict:
    total = sum(weights.values())
    if total <= 0:
        n = len(weights)
        return {k: 1 / n for k in weights}
    return {k: v / total for k, v in weights.items()}


def portfolio_returns(prices, weights, rebalance_daily=True) -> pd.Series:
    """Daily portfolio returns.

    rebalance_daily=True  -> constant weights (rebalanced every day)
    rebalance_daily=False -> buy & hold (weights drift with prices)
    """
    cols = list(weights.keys())
    w = pd.Series(weights)[cols]
    if rebalance_daily:
        return (daily_returns(prices[cols]) * w).sum(axis=1)
    value = (prices[cols] / prices[cols].iloc[0] * w).sum(axis=1)
    return value.pct_change().dropna()


def cumulative_returns(returns):
    return (1 + returns).cumprod() - 1


def total_return(returns) -> float:
    return float((1 + returns).prod() - 1)


def cagr(returns) -> float:
    years = len(returns) / TRADING_DAYS
    if years <= 0:
        return 0.0
    return float((1 + returns).prod() ** (1 / years) - 1)


def annualized_volatility(returns) -> float:
    return float(returns.std() * np.sqrt(TRADING_DAYS))


def rolling_volatility(returns, window=30) -> pd.Series:
    return returns.rolling(window).std() * np.sqrt(TRADING_DAYS)


def sharpe_ratio(returns, risk_free=0.065) -> float:
    vol = annualized_volatility(returns)
    if vol == 0:
        return 0.0
    return float((returns.mean() * TRADING_DAYS - risk_free) / vol)


def drawdown_series(returns) -> pd.Series:
    wealth = (1 + returns).cumprod()
    return wealth / wealth.cummax() - 1


def max_drawdown(returns) -> float:
    return float(drawdown_series(returns).min())


def beta_alpha(returns, benchmark_returns, risk_free=0.065):
    """Beta and annualised Jensen's alpha versus the benchmark."""
    df = pd.concat([returns, benchmark_returns], axis=1).dropna()
    df.columns = ["p", "b"]
    var_b = df["b"].var()
    if var_b == 0:
        return 0.0, 0.0
    beta = df["p"].cov(df["b"]) / var_b
    rf_d = risk_free / TRADING_DAYS
    alpha = (df["p"].mean() - (rf_d + beta * (df["b"].mean() - rf_d))) * TRADING_DAYS
    return float(beta), float(alpha)


def summary_metrics(returns, benchmark_returns, risk_free=0.065) -> dict:
    beta, alpha = beta_alpha(returns, benchmark_returns, risk_free)
    return {
        "Total Return": total_return(returns),
        "CAGR": cagr(returns),
        "Volatility": annualized_volatility(returns),
        "Sharpe Ratio": sharpe_ratio(returns, risk_free),
        "Max Drawdown": max_drawdown(returns),
        "Beta": beta,
        "Alpha": alpha,
    }


def stock_summary(prices, weights, names, risk_free=0.065) -> pd.DataFrame:
    rets = daily_returns(prices[list(weights.keys())])
    rows = []
    for t, w in weights.items():
        r = rets[t]
        tr = total_return(r)
        rows.append({
            "Stock": names.get(t, t),
            "Ticker": t,
            "Weight": w,
            "Total Return": tr,
            "CAGR": cagr(r),
            "Volatility": annualized_volatility(r),
            "Sharpe": sharpe_ratio(r, risk_free),
            "Max Drawdown": max_drawdown(r),
            "Contribution": w * tr,
        })
    return pd.DataFrame(rows).sort_values("Total Return", ascending=False).reset_index(drop=True)


def monthly_returns_table(returns) -> pd.DataFrame:
    monthly = (1 + returns).groupby([returns.index.year, returns.index.month]).prod() - 1
    table = monthly.unstack()
    table.columns = [pd.Timestamp(2000, int(m), 1).strftime("%b") for m in table.columns]
    table.index.name = "Year"
    return table


def generate_insights(metrics, bench_metrics, stock_df) -> list:
    """Turn the numbers into plain-English takeaways."""
    out = []
    diff = metrics["Total Return"] - bench_metrics["Total Return"]
    verb = "outperformed" if diff >= 0 else "underperformed"
    out.append(
        f"The portfolio {verb} the Nifty 50 by {abs(diff) * 100:.1f} percentage points "
        f"({metrics['Total Return'] * 100:.1f}% vs {bench_metrics['Total Return'] * 100:.1f}%)."
    )

    more = metrics["Volatility"] > bench_metrics["Volatility"]
    out.append(
        f"It was {'more' if more else 'less'} volatile than the benchmark "
        f"({metrics['Volatility'] * 100:.1f}% vs {bench_metrics['Volatility'] * 100:.1f}% annualised)."
    )

    better = metrics["Sharpe Ratio"] > bench_metrics["Sharpe Ratio"]
    out.append(
        f"Risk-adjusted returns {'beat' if better else 'lagged'} the index "
        f"(Sharpe {metrics['Sharpe Ratio']:.2f} vs {bench_metrics['Sharpe Ratio']:.2f})."
    )

    b = metrics["Beta"]
    out.append(
        f"Beta of {b:.2f} means the portfolio tends to move {'more' if b > 1 else 'less'} than the market."
    )

    if len(stock_df) > 0:
        best, worst = stock_df.iloc[0], stock_df.iloc[-1]
        out.append(f"Best performer: {best['Stock']} ({best['Total Return'] * 100:+.1f}%).")
        out.append(f"Weakest performer: {worst['Stock']} ({worst['Total Return'] * 100:+.1f}%).")
        top = stock_df.sort_values("Contribution", ascending=False).iloc[0]
        out.append(
            f"{top['Stock']} contributed the most to portfolio return ({top['Contribution'] * 100:.1f} pts)."
        )

    out.append(f"Worst peak-to-trough fall: {metrics['Max Drawdown'] * 100:.1f}%.")
    return out
