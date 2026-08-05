"""
Portfolio Backtesting Engine

Author: Tichaona Peter Chiripa
Project: Portfolio Analytics Engine

This module evaluates an optimized portfolio using
out-of-sample historical returns.
"""

import numpy as np
import pandas as pd


# =============================================================================
# PORTFOLIO DAILY RETURNS
# =============================================================================

def portfolio_backtest(
    weights: np.ndarray,
    test_returns: pd.DataFrame,
) -> pd.Series:
    """
    Compute portfolio daily returns over the test period.

    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights.

    test_returns : pd.DataFrame
        Daily asset returns.

    Returns
    -------
    pd.Series
        Daily portfolio returns.
    """

    portfolio = test_returns.dot(weights)

    portfolio.name = "Portfolio Return"

    return portfolio


# =============================================================================
# PORTFOLIO VALUE
# =============================================================================

def portfolio_value(
    portfolio_returns: pd.Series,
    initial_investment: float = 1.0,
) -> pd.Series:
    """
    Compute cumulative portfolio value.

    Parameters
    ----------
    portfolio_returns : pd.Series
        Daily portfolio returns.

    initial_investment : float
        Starting portfolio value.

    Returns
    -------
    pd.Series
        Portfolio value through time.
    """

    value = initial_investment * (1 + portfolio_returns).cumprod()

    value.name = "Portfolio Value"

    return value


# =============================================================================
# CUMULATIVE RETURNS
# =============================================================================

def cumulative_returns(
    portfolio_returns: pd.Series,
) -> pd.Series:
    """
    Compute cumulative portfolio returns.

    Parameters
    ----------
    portfolio_returns : pd.Series
        Daily portfolio returns.

    Returns
    -------
    pd.Series
        Cumulative return series.
    """

    cumulative = (1 + portfolio_returns).cumprod() - 1

    cumulative.name = "Cumulative Return"

    return cumulative


# =============================================================================
# DRAWDOWN
# =============================================================================

def drawdown(
    portfolio_value: pd.Series,
) -> pd.Series:
    """
    Compute portfolio drawdown.

    Parameters
    ----------
    portfolio_value : pd.Series
        Portfolio value.

    Returns
    -------
    pd.Series
        Drawdown series.
    """

    running_max = portfolio_value.cummax()

    dd = (
        portfolio_value - running_max
    ) / running_max

    dd.name = "Drawdown"

    return dd


# =============================================================================
# BACKTEST REPORT
# =============================================================================

def backtest_report(
    weights: np.ndarray,
    test_returns: pd.DataFrame,
    initial_investment: float = 1.0,
) -> pd.DataFrame:
    """
    Generate a complete backtest report.

    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights.

    test_returns : pd.DataFrame
        Test-period asset returns.

    initial_investment : float
        Initial portfolio value.

    Returns
    -------
    pd.DataFrame
        Daily portfolio performance.
    """

    returns = portfolio_backtest(
        weights,
        test_returns,
    )

    value = portfolio_value(
        returns,
        initial_investment,
    )

    cumulative = cumulative_returns(
        returns,
    )

    dd = drawdown(
        value,
    )

    report = pd.DataFrame(
        {
            "Portfolio Return": returns,
            "Portfolio Value": value,
            "Cumulative Return": cumulative,
            "Drawdown": dd,
        }
    )

    return report