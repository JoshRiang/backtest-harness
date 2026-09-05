"""Core backtest engine: bar-by-bar simulation, position tracking, PnL accounting."""
from dataclasses import dataclass, field
from typing import Callable, List, Optional
import pandas as pd
import numpy as np


@dataclass
class Trade:
    entry_date: pd.Timestamp
    exit_date: Optional[pd.Timestamp]
    entry_price: float
    exit_price: Optional[float]
    size: float
    pnl: float = 0.0
    side: str = "long"  # "long" or "short"


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trades: List[Trade]
    initial_capital: float
    final_equity: float
    n_bars: int
    metrics: dict = field(default_factory=dict)


class BacktestEngine:
    """Simple long-only backtester. Strategy is a function that takes a price df
    and returns a Series of target weights in [0, 1].
    """
    def __init__(self, initial_capital: float = 100_000.0, commission: float = 0.0005):
        self.initial_capital = initial_capital
        self.commission = commission

    def run(self, prices: pd.DataFrame, weights: pd.Series) -> BacktestResult:
        if "Close" not in prices.columns:
            raise ValueError("prices must have 'Close' column")
        close = prices["Close"]
        equity = pd.Series(index=close.index, dtype=float)
        equity.iloc[0] = self.initial_capital
        prev_w = 0.0
        trades: List[Trade] = []
        open_trade: Optional[Trade] = None
        for i, (dt, price) in enumerate(close.items()):
            target_w = float(weights.iloc[i]) if i < len(weights) else 0.0
            target_w = max(0.0, min(1.0, target_w))
            if i == 0:
                equity.iloc[i] = self.initial_capital
            else:
                # mark-to-market equity
                prev_price = close.iloc[i - 1]
                ret = (price - prev_price) / prev_price if prev_price else 0.0
                equity.iloc[i] = equity.iloc[i - 1] * (1 + prev_w * ret)
                # trade on weight change
                if abs(target_w - prev_w) > 1e-6:
                    # close existing
                    if open_trade is not None:
                        open_trade.exit_date = dt
                        open_trade.exit_price = price
                        open_trade.pnl = (price - open_trade.entry_price) * open_trade.size
                        trades.append(open_trade)
                        open_trade = None
                    # open new
                    if target_w > 0:
                        size = (equity.iloc[i] * target_w) / price
                        cost = equity.iloc[i] * target_w * self.commission
                        equity.iloc[i] -= cost
                        open_trade = Trade(
                            entry_date=dt,
                            exit_date=None,
                            entry_price=price,
                            exit_price=None,
                            size=size,
                            side="long",
                        )
            prev_w = target_w
        # close any remaining open trade at last price
        if open_trade is not None:
            last_dt = close.index[-1]
            last_px = close.iloc[-1]
            open_trade.exit_date = last_dt
            open_trade.exit_price = last_px
            open_trade.pnl = (last_px - open_trade.entry_price) * open_trade.size
            trades.append(open_trade)
        result = BacktestResult(
            equity_curve=equity,
            trades=trades,
            initial_capital=self.initial_capital,
            final_equity=float(equity.iloc[-1]),
            n_bars=len(close),
        )
        result.metrics = self._compute_metrics(result)
        return result

    def _compute_metrics(self, result: BacktestResult) -> dict:
        from .metrics import sharpe, sortino, max_drawdown, calmar
        eq = result.equity_curve
        rets = eq.pct_change().dropna()
        return {
            "sharpe": float(sharpe(rets)),
            "sortino": float(sortino(rets)),
            "max_drawdown": float(max_drawdown(eq)),
            "calmar": float(calmar(eq, rets)),
            "total_return": float(eq.iloc[-1] / eq.iloc[0] - 1),
            "n_trades": len(result.trades),
            "win_rate": float(sum(1 for t in result.trades if t.pnl > 0) / max(1, len(result.trades))),
        }


def run_backtest(prices: pd.DataFrame, weights: pd.Series, capital: float = 100_000.0) -> BacktestResult:
    """One-liner convenience."""
    return BacktestEngine(initial_capital=capital).run(prices, weights)
