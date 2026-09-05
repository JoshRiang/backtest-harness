"""Walk-forward optimization: split data into rolling train/test windows."""
from dataclasses import dataclass
from typing import Callable, List
import pandas as pd
import numpy as np


@dataclass
class WFold:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


class WalkForward:
    """Generate walk-forward windows and run an optimizer across them.

    Example:
        wf = WalkForward(prices, train_size=504, test_size=126, step=63)
        results = wf.run(lambda tr: sma_optimizer(tr))
    """
    def __init__(self, prices: pd.DataFrame, train_size: int = 504, test_size: int = 126, step: int = 63):
        self.prices = prices
        self.train_size = train_size
        self.test_size = test_size
        self.step = step

    def folds(self) -> List[WFold]:
        n = len(self.prices)
        folds = []
        i = 0
        while i + self.train_size + self.test_size <= n:
            tr = self.prices.iloc[i:i + self.train_size]
            te = self.prices.iloc[i + self.train_size:i + self.train_size + self.test_size]
            folds.append(WFold(
                train_start=tr.index[0], train_end=tr.index[-1],
                test_start=te.index[0], test_end=te.index[-1],
            ))
            i += self.step
        return folds

    def run(self, optimizer: Callable) -> pd.DataFrame:
        """Run `optimizer(train_prices) -> weights` on each fold. Returns DataFrame of OOS metrics."""
        rows = []
        for fold in self.folds():
            tr = self.prices.loc[fold.train_start:fold.train_end]
            te = self.prices.loc[fold.test_start:fold.test_end]
            try:
                weights = optimizer(tr)
            except Exception:
                continue
            # assume weights index aligned to tr; align to te
            if isinstance(weights, pd.Series):
                weights_te = weights.reindex(te.index).fillna(0.0)
            else:
                weights_te = pd.Series(0.0, index=te.index)
            from .engine import run_backtest
            res = run_backtest(te, weights_te)
            row = {"train_start": fold.train_start, "test_start": fold.test_start, **res.metrics}
            rows.append(row)
        return pd.DataFrame(rows)
