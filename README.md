# Project 1: Backtest Harness

Backtest engine with walk-forward validation, risk metrics, parameter sweep, and equity curve visualization. Built for evaluating systematic strategies before live deployment.

## Why this exists
Most retail backtests overfit. This tool forces walk-forward analysis: optimize on in-sample, validate on out-of-sample, report both, and surface the gap.

## Features
- Walk-forward optimizer (in-sample window -> out-of-sample validation)
- Risk metrics: Sharpe, Sortino, Calmar, max drawdown, win rate, profit factor
- Equity curve vs benchmark (SPY by default)
- Parameter sweep via grid
- CSV export of trades and metrics

## Quick start
```bash
pip install -r requirements.txt
python -m backtester --strategy sma_crossover --symbol SPY --start 2010-01-01
```

## Project structure
- `backtester/` - core engine
- `strategies/` - built-in strategy plugins
- `examples/` - notebook examples
- `tests/` - unit tests

## License
MIT
