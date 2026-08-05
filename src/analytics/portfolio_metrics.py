"""
Portfolio Performance Analytics

Author: Tichaona Peter Chiripa
Project: Portfolio Analytics Engine

This module computes standard portfolio performance and risk
metrics used in quantitative finance.
"""

import numpy as np
import pandas as pd

from scipy.stats import skew
from scipy.stats import kurtosis


# =============================================================================
# # =============================================================================
# PORTFOLIO RETURNS
# =============================================================================

def portfolio_returns(
    weights: np.ndarray,
    returns: pd.DataFrame,
) -> pd.Series:
    """
    Compute daily portfolio returns.

    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights.

    returns : pd.DataFrame
        Daily asset returns.

    Returns
    -------
    pd.Series
        Daily portfolio return series.
    """

    portfolio = returns.dot(weights)

    portfolio.name = "Portfolio Return"

    return portfolio


# =============================================================================
# ANNUALIZED RETURN
# =============================================================================

def annualized_return(
    portfolio_returns: pd.Series,
    trading_days: int = 252,
) -> float:
    """
    Compute annualized arithmetic return.

    Parameters
    ----------
    portfolio_returns : pd.Series
        Daily portfolio returns.

    trading_days : int, default=252
        Number of trading days per year.

    Returns
    -------
    float
        Annualized return.
    """

    return portfolio_returns.mean() * trading_days


# =============================================================================
# ANNUALIZED VOLATILITY
# =============================================================================

def annualized_volatility(
    portfolio_returns: pd.Series,
    trading_days: int = 252,
) -> float:
    """
    Compute annualized portfolio volatility.

    Parameters
    ----------
    portfolio_returns : pd.Series
        Daily portfolio returns.

    trading_days : int, default=252
        Number of trading days per year.

    Returns
    -------
    float
        Annualized volatility.
    """

    return portfolio_returns.std() * np.sqrt(trading_days)


# =============================================================================
# SHARPE RATIO
# =============================================================================

def sharpe_ratio(
    portfolio_returns: pd.Series,
    risk_free_rate: float = 0.0,
    trading_days: int = 252,
) -> float:
    """
    Compute the annualized Sharpe Ratio.

    Parameters
    ----------
    portfolio_returns : pd.Series
        Daily portfolio returns.

    risk_free_rate : float, default=0.0
        Annualized risk-free rate.

    trading_days : int, default=252
        Number of trading days per year.

    Returns
    -------
    float
        Annualized Sharpe Ratio.
    """

    annual_return = annualized_return(
        portfolio_returns,
        trading_days,
    )

    annual_volatility = annualized_volatility(
        portfolio_returns,
        trading_days,
    )

    if annual_volatility == 0:
        return np.nan

    sharpe = (
        annual_return - risk_free_rate
    ) / annual_volatility

    return sharpe


# =============================================================================
# SORTINO RATIO
# =============================================================================

def sortino_ratio(
    portfolio_returns: pd.Series,
    risk_free_rate: float = 0.0,
    trading_days: int = 252,
) -> float:
    """
    Compute the annualized Sortino Ratio.

    Parameters
    ----------
    portfolio_returns : pd.Series
        Daily portfolio returns.

    risk_free_rate : float, default=0.0
        Annualized risk-free rate.

    trading_days : int, default=252
        Number of trading days per year.

    Returns
    -------
    float
        Annualized Sortino Ratio.
    """

    annual_return = annualized_return(
        portfolio_returns,
        trading_days,
    )

    downside_returns = portfolio_returns[
        portfolio_returns < 0
    ]

    if len(downside_returns) == 0:
        return np.nan

    downside_deviation = (
        downside_returns.std()
        * np.sqrt(trading_days)
    )

    if downside_deviation == 0:
        return np.nan

    sortino = (
        annual_return - risk_free_rate
    ) / downside_deviation

    return sortino
# # =============================================================================
# MAXIMUM DRAWDOWN
# =============================================================================

def maximum_drawdown(
    portfolio_returns: pd.Series,
) -> float:
    """
    Compute the maximum portfolio drawdown.

    Parameters
    ----------
    portfolio_returns : pd.Series
        Daily portfolio returns.

    Returns
    -------
    float
        Maximum drawdown.
    """

    cumulative_returns = (
        1 + portfolio_returns
    ).cumprod()

    running_max = cumulative_returns.cummax()

    drawdowns = (
        cumulative_returns - running_max
    ) / running_max

    return drawdowns.min()


# =============================================================================
# CALMAR RATIO
# =============================================================================

def calmar_ratio(
    portfolio_returns: pd.Series,
    trading_days: int = 252,
) -> float:
    """
    Compute the annualized Calmar Ratio.

    Parameters
    ----------
    portfolio_returns : pd.Series
        Daily portfolio returns.

    trading_days : int, default=252
        Trading days per year.

    Returns
    -------
    float
        Calmar Ratio.
    """

    annual_return = annualized_return(
        portfolio_returns,
        trading_days,
    )

    max_dd = abs(
        maximum_drawdown(
            portfolio_returns,
        )
    )

    if max_dd == 0:
        return np.nan

    return annual_return / max_dd


# =============================================================================
# SKEWNESS
# =============================================================================

def skewness(
    portfolio_returns: pd.Series,
) -> float:
    """
    Compute return distribution skewness.

    Parameters
    ----------
    portfolio_returns : pd.Series
        Daily portfolio returns.

    Returns
    -------
    float
        Sample skewness.
    """

    return skew(
        portfolio_returns,
        bias=False,
    )


# =============================================================================
# EXCESS KURTOSIS
# =============================================================================

def excess_kurtosis(
    portfolio_returns: pd.Series,
) -> float:
    """
    Compute excess kurtosis.

    Parameters
    ----------
    portfolio_returns : pd.Series
        Daily portfolio returns.

    Returns
    -------
    float
        Excess kurtosis.
    """

    return kurtosis(
        portfolio_returns,
        fisher=True,
        bias=False,
    )


# =============================================================================
# HISTORICAL VALUE AT RISK
# =============================================================================

def value_at_risk(
    portfolio_returns: pd.Series,
    confidence_level: float = 0.95,
) -> float:
    """
    Compute Historical Value-at-Risk (VaR).

    Parameters
    ----------
    portfolio_returns : pd.Series
        Daily portfolio returns.

    confidence_level : float
        Confidence level.

    Returns
    -------
    float
        Historical VaR.
    """

    alpha = (
        1 - confidence_level
    ) * 100

    return np.percentile(
        portfolio_returns,
        alpha,
    )


# =============================================================================
# CONDITIONAL VALUE AT RISK
# =============================================================================

def conditional_value_at_risk(
    portfolio_returns: pd.Series,
    confidence_level: float = 0.95,
) -> float:
    """
    Compute Historical Conditional Value-at-Risk (CVaR).

    Parameters
    ----------
    portfolio_returns : pd.Series
        Daily portfolio returns.

    confidence_level : float
        Confidence level.

    Returns
    -------
    float
        Historical CVaR.
    """

    var = value_at_risk(
        portfolio_returns,
        confidence_level,
    )

    return portfolio_returns[
        portfolio_returns <= var
    ].mean()
# =============================================================================
# PORTFOLIO PERFORMANCE REPORT
# =============================================================================

def portfolio_report(
    weights: np.ndarray,
    returns: pd.DataFrame,
    risk_free_rate: float = 0.0,
    trading_days: int = 252,
) -> pd.DataFrame:
    """
    Generate a comprehensive portfolio performance report.

    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights.

    returns : pd.DataFrame
        Daily asset returns.

    risk_free_rate : float, default=0.0
        Annualized risk-free rate.

    trading_days : int, default=252
        Number of trading days per year.

    Returns
    -------
    pd.DataFrame
        Portfolio performance metrics.
    """

    # ---------------------------------------------------------
    # Portfolio Daily Returns
    # ---------------------------------------------------------

    daily_returns = portfolio_returns(
        weights,
        returns,
    )

    # ---------------------------------------------------------
    # Performance Metrics
    # ---------------------------------------------------------

    report = pd.DataFrame(
        {
            "Metric": [
                "Annual Return (%)",
                "Annual Volatility (%)",
                "Sharpe Ratio",
                "Sortino Ratio",
                "Maximum Drawdown (%)",
                "Calmar Ratio",
                "Skewness",
                "Excess Kurtosis",
                "Historical VaR (95%)",
                "Historical CVaR (95%)",
            ],
            "Value": [
                round(
                    annualized_return(
                        daily_returns,
                        trading_days,
                    ) * 100,
                    2,
                ),
                round(
                    annualized_volatility(
                        daily_returns,
                        trading_days,
                    ) * 100,
                    2,
                ),
                round(
                    sharpe_ratio(
                        daily_returns,
                        risk_free_rate,
                        trading_days,
                    ),
                    2,
                ),
                round(
                    sortino_ratio(
                        daily_returns,
                        risk_free_rate,
                        trading_days,
                    ),
                    2,
                ),
                round(
                    maximum_drawdown(
                        daily_returns,
                    ) * 100,
                    2,
                ),
                round(
                    calmar_ratio(
                        daily_returns,
                        trading_days,
                    ),
                    2,
                ),
                round(
                    skewness(
                        daily_returns,
                    ),
                    2,
                ),
                round(
                    excess_kurtosis(
                        daily_returns,
                    ),
                    2,
                ),
                round(
                    value_at_risk(
                        daily_returns,
                    ) * 100,
                    2,
                ),
                round(
                    conditional_value_at_risk(
                        daily_returns,
                    ) * 100,
                    2,
                ),
            ],
        }
    )

    return report