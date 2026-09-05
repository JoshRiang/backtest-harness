"""Risk and performance metrics."""
import pandas as pd
import numpy as np


def sharpe(returns: pd.Series, periods_per_year: int = 252, rf: float = 0.0) -> float:
    if returns.std() == 0:
        return 0.0
    excess = returns - rf / periods_per_year
    return float(np.sqrt(periods_per_year) * excess.mean() / returns.std())


def sortino(returns: pd.Series, periods_per_year: int = 252, rf: float = 0.0) -> float:
    downside = returns[returns < 0]
    if len(downside) == 0 or downside.std() == 0:
        return 0.0
    excess = returns - rf / periods_per_year
    return float(np.sqrt(periods_per_year) * excess.mean() / downside.std())


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = (equity - peak) / peak
    return float(dd.min())


def calmar(equity: pd.Series, returns: pd.Series, periods_per_year: int = 252) -> float:
    mdd = abs(max_drawdown(equity))
    if mdd == 0:
        return 0.0
    ann_ret = (equity.iloc[-1] / equity.iloc[0]) ** (periods_per_year / len(equity)) - 1
    return float(ann_ret / mdd)
