"""
Backtesting Module

Author: Tichaona Peter Chiripa
Project: Portfolio Analytics Engine

This package contains modules for evaluating portfolio performance
using historical out-of-sample data.

Modules
-------
backtester.py
    Core portfolio backtesting engine.

benchmark.py
    Benchmark portfolio construction and comparison.

performance.py
    Backtest performance analytics.

rolling_metrics.py
    Rolling performance and risk statistics.
"""

from .backtester import (
    portfolio_backtest,
    portfolio_value,
    cumulative_returns,
    drawdown,
    backtest_report,
)