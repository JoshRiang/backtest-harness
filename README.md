# Backtest Harness

Walk-forward long-only backtest engine with risk metrics and parameter sweep.

Self-contained bar-by-bar simulator. A strategy is a function that takes a price DataFrame and returns a Series of target weights in `[0, 1]`. The engine steps through each bar, marks positions to market, rebalances on weight changes, and records every trade with entry/exit dates, prices, size, and PnL. Built-in strategies: SMA crossover, momentum (12-month return > 0), mean reversion (recent return below threshold). Metrics: Sharpe, Sortino, max drawdown, Calmar, total return, win rate. Walk-forward wrapper runs rolling train/test windows with an optimizer callback. CLI pulls data from yfinance and runs any registered strategy.

```bash
git clone https://github.com/JoshRiang/backtest-harness.git
cd backtest-harness
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

```bash
# Run SMA crossover on SPY via CLI
python -m backtester --strategy sma_crossover --symbol SPY --start 2015-01-01 --capital 100000 --out spy_metrics.json

# Run walk-forward optimization
python -c "
import pandas as pd
import yfinance as yf
from backtester.walkforward import WalkForward
from backtester.engine import run_backtest
from strategies.sma_crossover import sma_crossover

df = yf.download('SPY', start='2010-01-01', end='2024-01-01')
wf = WalkForward(df, train_size=504, test_size=126, step=63)
results = wf.run(lambda tr: sma_crossover(tr, fast=20, slow=50))
print(results[['train_start','test_start','sharpe','max_drawdown','win_rate']])
"
```

```python
from backtester import BacktestEngine, run_backtest
from strategies.sma_crossover import sma_crossover
import yfinance as yf

df = yf.download('SPY', start='2020-01-01', progress=False)
weights = sma_crossover(df, fast=20, slow=50)
res = run_backtest(df, weights, capital=100_000)

print(f\"Total return: {res.metrics['total_return']:.2%}\")
print(f\"Sharpe: {res.metrics['sharpe']:.2f}\")
print(f\"Max drawdown: {res.metrics['max_drawdown']:.2%}\")
print(f\"Win rate: {res.metrics['win_rate']:.1%}\")
print(f\"Trades: {len(res.trades)}\")
```

Project structure:

```
backtest-harness/
├── backtester/
│   ├── __init__.py   # package exports: BacktestEngine, run_backtest, metrics, WalkForward
│   ├── engine.py     # BacktestEngine (bar-by-bar sim), run_backtest one-liner
│   ├── metrics.py    # sharpe, sortino, max_drawdown, calmar
│   ├── walkforward.py# WalkForward rolling train/test windows
│   └── __main__.py   # CLI: --strategy --symbol --start --capital --out
├── strategies/
│   ├── __init__.py
│   └── sma_crossover.py   # sma_crossover, momentum, mean_reversion weight functions
├── tests/
│   ├── __init__.py
│   └── test_engine.py     # synthetic data tests: run, metrics keys, walk-forward, sharpe edge cases
├── requirements.txt   # yfinance, pandas, numpy, matplotlib
├── LICENSE
├── CHANGELOG.md
└── README.md
```

Testing:

```bash
pytest -v
```

Tests use synthetic price data (deterministic seed) — no network required.

License: MIT — see [LICENSE](LICENSE).

Author: Joshua Riangkamang — [github.com/JoshRiang](https://github.com/JoshRiang)
