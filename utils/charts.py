"""Plotly chart builders."""
import plotly.express as px
import plotly.graph_objects as go

PORTFOLIO_COLOR = "#4F46E5"
BENCH_COLOR = "#F59E0B"
LAYOUT = dict(template="plotly_white", margin=dict(l=10, r=10, t=50, b=10), hovermode="x unified",
              legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))


def allocation_pie(weights, names):
    labels = [names.get(t, t) for t in weights]
    fig = go.Figure(go.Pie(labels=labels, values=list(weights.values()), hole=0.5, textinfo="label+percent"))
    fig.update_layout(title="Portfolio Allocation", showlegend=False, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def cumulative_chart(port_cum, bench_cum):
    fig = go.Figure()
    fig.add_scatter(x=port_cum.index, y=port_cum * 100, name="Portfolio", line=dict(color=PORTFOLIO_COLOR, width=2.5))
    fig.add_scatter(x=bench_cum.index, y=bench_cum * 100, name="Nifty 50", line=dict(color=BENCH_COLOR, width=2.5))
    fig.update_layout(title="Cumulative Return: Portfolio vs Nifty 50", yaxis_title="Return (%)", **LAYOUT)
    return fig


def price_chart(prices, names, normalized=True):
    df = prices / prices.iloc[0] * 100 if normalized else prices
    fig = go.Figure()
    for col in df.columns:
        fig.add_scatter(x=df.index, y=df[col], name=names.get(col, col), mode="lines")
    fig.update_layout(
        title="Stock Price Movement" + (" (rebased to 100)" if normalized else ""),
        yaxis_title="Rebased price" if normalized else "Price (INR)", **LAYOUT)
    return fig


def drawdown_chart(port_dd, bench_dd):
    fig = go.Figure()
    fig.add_scatter(x=port_dd.index, y=port_dd * 100, name="Portfolio", fill="tozeroy",
                    line=dict(color=PORTFOLIO_COLOR))
    fig.add_scatter(x=bench_dd.index, y=bench_dd * 100, name="Nifty 50", line=dict(color=BENCH_COLOR))
    fig.update_layout(title="Drawdown", yaxis_title="Drawdown (%)", **LAYOUT)
    return fig


def rolling_vol_chart(port_vol, bench_vol, window):
    fig = go.Figure()
    fig.add_scatter(x=port_vol.index, y=port_vol * 100, name="Portfolio", line=dict(color=PORTFOLIO_COLOR))
    fig.add_scatter(x=bench_vol.index, y=bench_vol * 100, name="Nifty 50", line=dict(color=BENCH_COLOR))
    fig.update_layout(title=f"Rolling {window}-Day Volatility (annualised)", yaxis_title="Volatility (%)", **LAYOUT)
    return fig


def stock_return_bar(stock_df):
    df = stock_df.sort_values("Total Return")
    colors = ["#16A34A" if v >= 0 else "#DC2626" for v in df["Total Return"]]
    fig = go.Figure(go.Bar(x=df["Total Return"] * 100, y=df["Stock"], orientation="h", marker_color=colors,
                           text=[f"{v * 100:.1f}%" for v in df["Total Return"]], textposition="outside"))
    fig.update_layout(title="Total Return by Stock", xaxis_title="Return (%)", template="plotly_white",
                      margin=dict(l=10, r=30, t=50, b=10))
    return fig


def risk_return_scatter(stock_df):
    fig = px.scatter(stock_df, x=stock_df["Volatility"] * 100, y=stock_df["CAGR"] * 100, size="Weight",
                     text="Stock", color="Sharpe", color_continuous_scale="RdYlGn")
    fig.update_traces(textposition="top center")
    fig.update_layout(title="Risk vs Return (bubble size = weight)", xaxis_title="Volatility (%)",
                      yaxis_title="CAGR (%)", template="plotly_white", margin=dict(l=10, r=10, t=50, b=10))
    return fig


def correlation_heatmap(returns, names):
    corr = returns.corr()
    labels = [names.get(c, c) for c in corr.columns]
    fig = go.Figure(go.Heatmap(z=corr.values, x=labels, y=labels, zmin=-1, zmax=1, colorscale="RdBu_r",
                               text=corr.round(2).values, texttemplate="%{text}"))
    fig.update_layout(title="Return Correlation", template="plotly_white", margin=dict(l=10, r=10, t=50, b=10))
    return fig


def monthly_heatmap(table):
    fig = go.Figure(go.Heatmap(z=table.values * 100, x=list(table.columns), y=[str(i) for i in table.index],
                               colorscale="RdYlGn", zmid=0, text=(table.values * 100).round(1),
                               texttemplate="%{text}%"))
    fig.update_layout(title="Monthly Portfolio Returns (%)", template="plotly_white",
                      margin=dict(l=10, r=10, t=50, b=10))
    return fig
