"""CLI entrypoint: python -m backtester --strategy sma_crossover --symbol SPY"""
import argparse
import json
import sys
import yfinance as yf
import pandas as pd
from backtester import BacktestEngine
from strategies.sma_crossover import sma_crossover, momentum, mean_reversion


STRATEGIES = {
    "sma_crossover": sma_crossover,
    "momentum": momentum,
    "mean_reversion": mean_reversion,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--strategy", required=True, choices=list(STRATEGIES.keys()))
    p.add_argument("--symbol", default="SPY")
    p.add_argument("--start", default="2010-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--capital", type=float, default=100000.0)
    p.add_argument("--out", default=None, help="write JSON metrics to this path")
    args = p.parse_args()

    print(f"Downloading {args.symbol} from {args.start}...")
    df = yf.download(args.symbol, start=args.start, end=args.end, progress=False)
    if df.empty:
        print("No data", file=sys.stderr)
        sys.exit(1)

    strat = STRATEGIES[args.strategy]
    weights = strat(df)
    engine = BacktestEngine(initial_capital=args.capital)
    res = engine.run(df, weights)

    print("\n=== Backtest Results ===")
    for k, v in res.metrics.items():
        print(f"  {k:20s}: {v}")
    print(f"  equity curve length: {len(res.equity_curve)} bars")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"symbol": args.symbol, "strategy": args.strategy, **res.metrics}, f, indent=2)
        print(f"\nMetrics saved to {args.out}")


if __name__ == "__main__":
    main()
