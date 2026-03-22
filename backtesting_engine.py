"""
backtesting_engine.py
=====================
Strategy Backtesting Engine — tests trading strategies against historical data.

Implements three strategies from scratch:
  1. Moving Average Crossover  (trend-following)
  2. Mean Reversion            (buy the dip)
  3. Momentum                  (buy recent winners)

Reports: total return, annualised return, Sharpe ratio, max drawdown,
         win rate, number of trades, and a comparison chart vs buy-and-hold.

Author  : Your Name
GitHub  : github.com/yourusername/backtesting-engine

Requirements:
    pip install numpy pandas matplotlib yfinance

Usage:
    python backtesting_engine.py
    python backtesting_engine.py --ticker NVDA --strategy momentum --start 2020-01-01
"""

import argparse
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import yfinance as yf

warnings.filterwarnings("ignore", category=FutureWarning)

# ─────────────────────────────────────────────
# DEFAULT CONFIGURATION
# ─────────────────────────────────────────────

DEFAULT_TICKER    = "AAPL"
DEFAULT_START     = "2019-01-01"
DEFAULT_END       = "2024-01-01"
INITIAL_CAPITAL   = 10_000.0
RISK_FREE_RATE    = 0.045
TRADING_DAYS      = 252
TRANSACTION_COST  = 0.001   # 0.1% per trade — realistic for retail brokers


# ─────────────────────────────────────────────
# DATA LAYER
# ─────────────────────────────────────────────

def fetch_prices(ticker: str, start: str, end: str) -> pd.Series:
    """Download adjusted closing prices for a single ticker."""
    print(f"\nFetching data for {ticker} ({start} → {end})...")
    raw = yf.download(ticker, start=start, end=end,
                      auto_adjust=True, progress=False)["Close"]

    if isinstance(raw, pd.DataFrame):
        raw = raw.squeeze()

    raw.dropna(inplace=True)

    if raw.empty:
        sys.exit(f"Error: no data returned for {ticker}.")

    print(f"  OK — {len(raw)} trading days")
    return raw


# ─────────────────────────────────────────────
# STRATEGIES
# Each strategy returns a pd.Series of signals:
#   1  = hold long (invested)
#   0  = hold cash
# ─────────────────────────────────────────────

def strategy_ma_crossover(
    prices: pd.Series,
    short_window: int = 50,
    long_window: int  = 200
) -> pd.Series:
    """
    Moving Average Crossover — "Golden Cross / Death Cross"

    Logic:
      - Compute SMA(short) and SMA(long)
      - Signal = 1 when SMA(short) > SMA(long)  → uptrend, stay invested
      - Signal = 0 when SMA(short) < SMA(long)  → downtrend, go to cash

    This is a trend-following strategy: it tries to be invested
    during sustained uptrends and out of the market during downtrends.
    It lags real turning points (by definition of moving averages)
    but filters out a lot of short-term noise.
    """
    sma_short = prices.rolling(window=short_window).mean()
    sma_long  = prices.rolling(window=long_window).mean()

    signal = (sma_short > sma_long).astype(int)

    # Shift by 1: we can only act on yesterday's signal (no look-ahead bias)
    # Look-ahead bias is a critical error in backtesting — using future data
    # to make past decisions. The shift(1) prevents this.
    signal = signal.shift(1).fillna(0)

    return signal


def strategy_mean_reversion(
    prices: pd.Series,
    window: int      = 20,
    z_entry: float   = -1.5,
    z_exit: float    = 0.5
) -> pd.Series:
    """
    Mean Reversion — "Buy the Dip"

    Logic:
      - Compute rolling mean and std over `window` days
      - Z-score = (price − mean) / std
      - Signal = 1 (buy) when Z-score < z_entry  (price is unusually low)
      - Signal = 0 (sell) when Z-score > z_exit  (price has recovered)

    This strategy bets that prices revert to their recent average.
    Works well in range-bound markets, poorly in strong trends.
    The z-score is the same concept as in statistics — how many
    standard deviations away from the mean is the current price.
    """
    rolling_mean = prices.rolling(window=window).mean()
    rolling_std  = prices.rolling(window=window).std()
    z_score      = (prices - rolling_mean) / rolling_std

    signal = pd.Series(0, index=prices.index)
    position = 0

    for i in range(len(prices)):
        z = z_score.iloc[i]
        if np.isnan(z):
            signal.iloc[i] = 0
            continue
        if position == 0 and z < z_entry:
            position = 1          # enter: price dipped below threshold
        elif position == 1 and z > z_exit:
            position = 0          # exit: price recovered to mean
        signal.iloc[i] = position

    # Shift to avoid look-ahead bias
    return signal.shift(1).fillna(0)


def strategy_momentum(
    prices: pd.Series,
    lookback: int   = 90,
    holding: int    = 30
) -> pd.Series:
    """
    Momentum — "Buy Recent Winners"

    Logic:
      - Every `holding` days, check whether the stock's return
        over the past `lookback` days is positive
      - If yes: hold for the next `holding` days
      - If no:  stay in cash for the next `holding` days

    Momentum is one of the most robust anomalies in finance —
    stocks that have performed well recently tend to continue
    outperforming over the next 1–12 months (Jegadeesh & Titman, 1993).
    """
    signal = pd.Series(0, index=prices.index)

    for i in range(lookback, len(prices), holding):
        past_return = prices.iloc[i] / prices.iloc[i - lookback] - 1
        position = 1 if past_return > 0 else 0
        end = min(i + holding, len(prices))
        signal.iloc[i:end] = position

    return signal.shift(1).fillna(0)


STRATEGIES = {
    "ma_crossover":  strategy_ma_crossover,
    "mean_reversion": strategy_mean_reversion,
    "momentum":       strategy_momentum,
}


# ─────────────────────────────────────────────
# BACKTEST ENGINE
# ─────────────────────────────────────────────

def run_backtest(
    prices: pd.Series,
    signal: pd.Series,
    initial_capital: float = INITIAL_CAPITAL,
    transaction_cost: float = TRANSACTION_COST
) -> pd.DataFrame:
    """
    Core simulation loop.

    Walks through every trading day, applies the signal, and tracks:
      - portfolio value
      - daily returns
      - position changes (trades)
      - transaction costs

    Returns a DataFrame with one row per day.

    Key formula:
      V_t = V_{t-1} × (1 + r_t × position_t) − cost_t
    """
    # Align signal to prices
    signal = signal.reindex(prices.index).fillna(0)

    daily_returns = prices.pct_change().fillna(0)

    portfolio_value = [initial_capital]
    position_held   = [0]
    trades          = [0]
    costs           = [0]

    current_position = 0

    for i in range(1, len(prices)):
        sig = signal.iloc[i]
        ret = daily_returns.iloc[i]

        # Detect trade
        trade_occurred = int(sig != current_position)
        cost = transaction_cost * trade_occurred

        # Update position
        current_position = sig

        # Update portfolio value
        prev_value = portfolio_value[-1]
        new_value  = prev_value * (1 + ret * current_position) * (1 - cost)

        portfolio_value.append(new_value)
        position_held.append(current_position)
        trades.append(trade_occurred)
        costs.append(prev_value * cost)

    results = pd.DataFrame({
        "price":           prices.values,
        "signal":          signal.values,
        "position":        position_held,
        "portfolio_value": portfolio_value,
        "daily_return":    daily_returns.values,
        "trade":           trades,
        "cost":            costs,
    }, index=prices.index)

    return results


# ─────────────────────────────────────────────
# PERFORMANCE METRICS
# ─────────────────────────────────────────────

def compute_metrics(
    results: pd.DataFrame,
    rf: float = RISK_FREE_RATE
) -> dict:
    """
    Compute the 6 standard performance metrics.

    These are the exact metrics used in professional strategy evaluation.
    """
    values  = results["portfolio_value"]
    returns = values.pct_change().dropna()

    # 1. Total return
    total_return = (values.iloc[-1] / values.iloc[0]) - 1

    # 2. Annualised return
    n_years = len(results) / TRADING_DAYS
    annualised_return = (1 + total_return) ** (1 / n_years) - 1

    # 3. Sharpe ratio (annualised)
    daily_rf      = rf / TRADING_DAYS
    excess_returns = returns - daily_rf
    sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(TRADING_DAYS) \
             if excess_returns.std() > 0 else 0.0

    # 4. Maximum drawdown
    # Running peak: at each day, what was the highest value ever reached?
    running_peak = values.cummax()
    drawdown     = (values - running_peak) / running_peak
    max_drawdown = drawdown.min()   # most negative value = worst drawdown

    # 5. Win rate — % of trading days with positive return
    positive_days = (returns > 0).sum()
    win_rate = positive_days / len(returns) if len(returns) > 0 else 0

    # 6. Number of trades
    n_trades = results["trade"].sum()

    # 7. Total transaction costs paid
    total_costs = results["cost"].sum()

    return {
        "total_return":       total_return,
        "annualised_return":  annualised_return,
        "sharpe_ratio":       sharpe,
        "max_drawdown":       max_drawdown,
        "win_rate":           win_rate,
        "n_trades":           int(n_trades),
        "total_costs":        total_costs,
        "final_value":        values.iloc[-1],
    }


def buy_and_hold_benchmark(
    prices: pd.Series,
    initial_capital: float = INITIAL_CAPITAL
) -> pd.DataFrame:
    """
    The benchmark: what if you just bought on day 1 and held forever?
    Every strategy should be compared against this.
    If your strategy can't beat buy-and-hold, it's not worth using.
    """
    daily_returns   = prices.pct_change().fillna(0)
    portfolio_value = initial_capital * (1 + daily_returns).cumprod()
    signal          = pd.Series(1, index=prices.index)

    return pd.DataFrame({
        "price":           prices.values,
        "signal":          signal.values,
        "position":        1,
        "portfolio_value": portfolio_value.values,
        "daily_return":    daily_returns.values,
        "trade":           0,
        "cost":            0,
    }, index=prices.index)


# ─────────────────────────────────────────────
# REPORTING
# ─────────────────────────────────────────────

def print_metrics(label: str, metrics: dict) -> None:
    print(f"\n{'─'*50}")
    print(f"  {label}")
    print(f"{'─'*50}")
    print(f"  Total return       : {metrics['total_return']*100:>8.2f}%")
    print(f"  Annualised return  : {metrics['annualised_return']*100:>8.2f}%")
    print(f"  Sharpe ratio       : {metrics['sharpe_ratio']:>8.3f}")
    print(f"  Max drawdown       : {metrics['max_drawdown']*100:>8.2f}%")
    print(f"  Win rate           : {metrics['win_rate']*100:>8.1f}%")
    print(f"  Number of trades   : {metrics['n_trades']:>8d}")
    print(f"  Transaction costs  : ${metrics['total_costs']:>7.2f}")
    print(f"  Final value        : ${metrics['final_value']:>8.2f}")


def print_comparison(
    strategy_name: str,
    strategy_metrics: dict,
    benchmark_metrics: dict
) -> None:
    print(f"\n{'─'*60}")
    print(f"  Strategy vs Buy-and-Hold Benchmark")
    print(f"{'─'*60}")
    print(f"  {'Metric':<22} {'Strategy':>12} {'Buy & Hold':>12} {'Edge':>10}")
    print(f"  {'─'*22} {'─'*12} {'─'*12} {'─'*10}")

    metrics_to_compare = [
        ("Total return",      "total_return",      "%",   100),
        ("Ann. return",       "annualised_return",  "%",   100),
        ("Sharpe ratio",      "sharpe_ratio",       "",    1  ),
        ("Max drawdown",      "max_drawdown",       "%",   100),
    ]

    for label, key, unit, scale in metrics_to_compare:
        s = strategy_metrics[key] * scale
        b = benchmark_metrics[key] * scale
        edge = s - b
        edge_str = f"{'+' if edge >= 0 else ''}{edge:.2f}{unit}"
        # For drawdown, less negative is better (flip the edge sign display)
        if key == "max_drawdown":
            edge_str = f"{'+' if edge <= 0 else ''}{abs(edge):.2f}{'%'} {'better' if edge > 0 else 'worse'}"
        print(f"  {label:<22} {s:>11.2f}{unit} {b:>11.2f}{unit} {edge_str:>10}")


# ─────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────

def plot_results(
    prices: pd.Series,
    strategy_results: pd.DataFrame,
    benchmark_results: pd.DataFrame,
    strategy_metrics: dict,
    benchmark_metrics: dict,
    strategy_name: str,
    ticker: str,
    output_path: str = "backtest_results.png"
) -> None:
    """
    Four-panel chart:
      Top left    — equity curve: strategy vs buy-and-hold
      Top right   — drawdown over time
      Bottom left — buy/sell signals overlaid on price
      Bottom right — daily returns distribution
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.patch.set_facecolor("#fafaf8")
    fig.suptitle(
        f"{ticker} — {strategy_name.replace('_', ' ').title()} Strategy Backtest",
        fontsize=13, fontweight="normal", y=0.98
    )

    ax1, ax2, ax3, ax4 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_facecolor("#fafaf8")

    # ── Panel 1: Equity curve ────────────────────────────────
    ax1.plot(strategy_results.index,
             strategy_results["portfolio_value"],
             color="#1D9E75", linewidth=1.8,
             label=f"Strategy  (${strategy_metrics['final_value']:,.0f})")
    ax1.plot(benchmark_results.index,
             benchmark_results["portfolio_value"],
             color="#534AB7", linewidth=1.8, linestyle="--",
             label=f"Buy & hold  (${benchmark_metrics['final_value']:,.0f})")
    ax1.axhline(INITIAL_CAPITAL, color="gray", linewidth=0.7, linestyle=":")
    ax1.set_title("Portfolio value", fontsize=10)
    ax1.set_ylabel("Value ($)", fontsize=9)
    ax1.legend(fontsize=8)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"${x:,.0f}"))

    # ── Panel 2: Drawdown ────────────────────────────────────
    strat_dd = (strategy_results["portfolio_value"] /
                strategy_results["portfolio_value"].cummax() - 1) * 100
    bench_dd = (benchmark_results["portfolio_value"] /
                benchmark_results["portfolio_value"].cummax() - 1) * 100

    ax2.fill_between(strategy_results.index, strat_dd, 0,
                     alpha=0.5, color="#E24B4A", label="Strategy")
    ax2.fill_between(benchmark_results.index, bench_dd, 0,
                     alpha=0.25, color="#534AB7", label="Buy & hold")
    ax2.set_title("Drawdown", fontsize=10)
    ax2.set_ylabel("Drawdown (%)", fontsize=9)
    ax2.legend(fontsize=8)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))

    # ── Panel 3: Price + signals ─────────────────────────────
    ax3.plot(prices.index, prices.values,
             color="gray", linewidth=1.0, alpha=0.8, label="Price")

    # Mark buy signals (position goes from 0 → 1)
    buys  = strategy_results[strategy_results["trade"] == 1].copy()
    buys  = buys[buys["signal"] == 1]
    sells = strategy_results[strategy_results["trade"] == 1].copy()
    sells = sells[sells["signal"] == 0]

    ax3.scatter(buys.index,  prices.loc[buys.index],
                marker="^", color="#1D9E75", s=45, zorder=5, label="Buy")
    ax3.scatter(sells.index, prices.loc[sells.index],
                marker="v", color="#E24B4A", s=45, zorder=5, label="Sell")
    ax3.set_title(f"Buy / sell signals ({strategy_metrics['n_trades']} trades)", fontsize=10)
    ax3.set_ylabel("Price ($)", fontsize=9)
    ax3.legend(fontsize=8)

    # ── Panel 4: Returns distribution ───────────────────────
    strategy_daily = strategy_results["portfolio_value"].pct_change().dropna() * 100
    benchmark_daily = benchmark_results["portfolio_value"].pct_change().dropna() * 100

    ax4.hist(benchmark_daily, bins=60, alpha=0.4,
             color="#534AB7", label="Buy & hold", density=True)
    ax4.hist(strategy_daily,  bins=60, alpha=0.5,
             color="#1D9E75", label="Strategy",   density=True)
    ax4.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    ax4.set_title("Daily returns distribution", fontsize=10)
    ax4.set_xlabel("Daily return (%)", fontsize=9)
    ax4.legend(fontsize=8)
    ax4.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))

    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"\n  Chart saved → {output_path}")
    plt.show()


# ─────────────────────────────────────────────
# RUN ALL STRATEGIES (comparison mode)
# ─────────────────────────────────────────────

def run_all_strategies(
    prices: pd.Series,
    ticker: str,
    rf: float = RISK_FREE_RATE
) -> None:
    """
    Run all three strategies on the same ticker and print a
    side-by-side comparison table — useful for README and LinkedIn.
    """
    benchmark = buy_and_hold_benchmark(prices)
    bm_metrics = compute_metrics(benchmark, rf)

    print(f"\n{'═'*70}")
    print(f"  All Strategies — {ticker}")
    print(f"{'═'*70}")
    print(f"  {'Strategy':<22} {'Ann.Ret':>8} {'Sharpe':>8} {'MaxDD':>8} {'Trades':>8}")
    print(f"  {'─'*22} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")

    for name, func in STRATEGIES.items():
        signal  = func(prices)
        results = run_backtest(prices, signal)
        m       = compute_metrics(results, rf)
        print(f"  {name:<22} {m['annualised_return']*100:>7.2f}% "
              f"{m['sharpe_ratio']:>8.3f} "
              f"{m['max_drawdown']*100:>7.2f}% "
              f"{m['n_trades']:>8d}")

    print(f"  {'buy_and_hold':<22} {bm_metrics['annualised_return']*100:>7.2f}% "
          f"{bm_metrics['sharpe_ratio']:>8.3f} "
          f"{bm_metrics['max_drawdown']*100:>7.2f}% "
          f"{'0':>8}")
    print(f"{'═'*70}")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strategy backtesting engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python backtesting_engine.py
  python backtesting_engine.py --ticker NVDA --strategy momentum
  python backtesting_engine.py --ticker TSLA --strategy mean_reversion --start 2020-01-01
  python backtesting_engine.py --ticker AAPL --all-strategies
        """
    )
    parser.add_argument("--ticker",        default=DEFAULT_TICKER,
                        help="Yahoo Finance ticker symbol (default: AAPL)")
    parser.add_argument("--strategy",
                        choices=list(STRATEGIES.keys()),
                        default="ma_crossover",
                        help="Strategy to run (default: ma_crossover)")
    parser.add_argument("--start",         default=DEFAULT_START)
    parser.add_argument("--end",           default=DEFAULT_END)
    parser.add_argument("--capital",       type=float, default=INITIAL_CAPITAL,
                        help="Starting capital in $ (default: 10000)")
    parser.add_argument("--rf",            type=float, default=RISK_FREE_RATE)
    parser.add_argument("--cost",          type=float, default=TRANSACTION_COST,
                        help="Transaction cost per trade as decimal (default: 0.001)")
    parser.add_argument("--all-strategies", action="store_true",
                        help="Run all strategies and print comparison table")
    parser.add_argument("--no-plot",       action="store_true")
    parser.add_argument("--output",        default="backtest_results.png")
    return parser.parse_args()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    print("=" * 50)
    print("  Backtesting Engine")
    print("=" * 50)

    prices = fetch_prices(args.ticker, args.start, args.end)

    if args.all_strategies:
        run_all_strategies(prices, args.ticker, rf=args.rf)
        return

    # Single strategy run
    strategy_fn = STRATEGIES[args.strategy]
    print(f"\nRunning strategy: {args.strategy}")

    signal    = strategy_fn(prices)
    results   = run_backtest(prices, signal,
                             initial_capital=args.capital,
                             transaction_cost=args.cost)
    metrics   = compute_metrics(results, rf=args.rf)

    benchmark         = buy_and_hold_benchmark(prices, args.capital)
    benchmark_metrics = compute_metrics(benchmark, rf=args.rf)

    print_metrics(f"{args.strategy.replace('_',' ').title()} Strategy", metrics)
    print_metrics("Buy-and-Hold Benchmark", benchmark_metrics)
    print_comparison(args.strategy, metrics, benchmark_metrics)

    if not args.no_plot:
        plot_results(
            prices, results, benchmark,
            metrics, benchmark_metrics,
            args.strategy, args.ticker,
            output_path=args.output
        )

    print("\n" + "=" * 50)
    print("  Complete.")
    print("=" * 50)


if __name__ == "__main__":
    main()
