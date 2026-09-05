"""Tests for backtester core."""
import pandas as pd
import numpy as np
from backtester.engine import run_backtest
from backtester.metrics import sharpe, sortino, max_drawdown
from backtester.walkforward import WalkForward
from strategies.sma_crossover import sma_crossover


def make_synthetic_prices(n: int = 500, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    rets = np.random.normal(0.0005, 0.02, n)
    close = 100 * (1 + rets).cumprod()
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.DataFrame({"Close": close, "Open": close}, index=dates)


def test_run_backtest_basic():
    prices = make_synthetic_prices()
    weights = sma_crossover(prices, fast=10, slow=30)
    res = run_backtest(prices, weights, capital=100_000)
    assert res.initial_capital == 100_000
    assert len(res.equity_curve) == len(prices)
    assert "sharpe" in res.metrics


def test_metrics_have_keys():
    prices = make_synthetic_prices()
    weights = sma_crossover(prices, fast=10, slow=30)
    res = run_backtest(prices, weights)
    for k in ("sharpe", "sortino", "max_drawdown", "calmar", "win_rate"):
        assert k in res.metrics


def test_walkforward_folds():
    prices = make_synthetic_prices(800)
    wf = WalkForward(prices, train_size=300, test_size=100, step=50)
    folds = wf.folds()
    assert len(folds) >= 3
    for f in folds:
        assert f.train_end < f.test_start


def test_sharpe_zero_for_constant_returns():
    returns = pd.Series([0.0] * 100)
    assert sharpe(returns) == 0.0


if __name__ == "__main__":
    test_run_backtest_basic()
    test_metrics_have_keys()
    test_walkforward_folds()
    test_sharpe_zero_for_constant_returns()
    print("All tests passed")
