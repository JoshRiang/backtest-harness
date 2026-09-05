"""Built-in strategies. Each returns a Series of target weights in [0, 1]."""
import pandas as pd


def sma_crossover(prices: pd.DataFrame, fast: int = 20, slow: int = 50) -> pd.Series:
    """Long when fast SMA > slow SMA, else flat."""
    close = prices["Close"]
    f = close.rolling(fast).mean()
    s = close.rolling(slow).mean()
    weights = (f > s).astype(float)
    weights.index = close.index
    return weights


def momentum(prices: pd.DataFrame, lookback: int = 252) -> pd.Series:
    """Long when 12-month return > 0, else flat."""
    close = prices["Close"]
    ret = close / close.shift(lookback) - 1
    weights = (ret > 0).astype(float)
    weights.index = close.index
    return weights


def mean_reversion(prices: pd.DataFrame, lookback: int = 20, threshold: float = -0.05) -> pd.Series:
    """Long when recent return is below threshold (oversold)."""
    close = prices["Close"]
    ret = close / close.shift(lookback) - 1
    weights = (ret < threshold).astype(float)
    weights.index = close.index
    return weights
