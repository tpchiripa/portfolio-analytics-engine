"""
Capital Asset Pricing Model (CAPM) Analytics

Author: Tichaona Peter Chiripa
Project: Portfolio Analytics Engine

This module computes CAPM-based performance measures used
to evaluate portfolios relative to a market benchmark.
"""

import numpy as np
import pandas as pd

# =============================================================================
# BETA
# =============================================================================

def beta(
    portfolio_returns: pd.Series,
    market_returns: pd.Series,
) -> float:
    """
    Compute CAPM beta.

    Parameters
    ----------
    portfolio_returns : pd.Series
        Portfolio return series.

    market_returns : pd.Series
        Market return series.

    Returns
    -------
    float
        Portfolio beta.
    """

    # ---------------------------------------------------------
    # Convert one-column DataFrames to Series if necessary
    # ---------------------------------------------------------

    if isinstance(portfolio_returns, pd.DataFrame):
        portfolio_returns = portfolio_returns.squeeze()

    if isinstance(market_returns, pd.DataFrame):
        market_returns = market_returns.squeeze()

    # ---------------------------------------------------------
    # Align dates
    # ---------------------------------------------------------

    portfolio_returns, market_returns = portfolio_returns.align(
        market_returns,
        join="inner",
    )

    covariance = np.cov(
        portfolio_returns,
        market_returns,
    )[0, 1]

    market_variance = np.var(
        market_returns,
        ddof=1,
    )

    return covariance / market_variance
# =============================================================================
# ALPHA
# =============================================================================

def alpha(
    portfolio_returns: pd.Series,
    market_returns: pd.Series,
    risk_free_rate: float = 0.0,
    trading_days: int = 252,
) -> float:
    """
    Compute Jensen's Alpha.

    Parameters
    ----------
    portfolio_returns : pd.Series
        Daily portfolio return series.

    market_returns : pd.Series
        Daily market (benchmark) return series.

    risk_free_rate : float, default=0.0
        Annualized risk-free rate.

    trading_days : int, default=252
        Number of trading days per year.

    Returns
    -------
    float
        Annualized Jensen's Alpha.
    """

    # ==========================================================
    # Convert one-column DataFrames to Series
    # ==========================================================

    if isinstance(portfolio_returns, pd.DataFrame):
        portfolio_returns = portfolio_returns.squeeze()

    if isinstance(market_returns, pd.DataFrame):
        market_returns = market_returns.squeeze()

    # ==========================================================
    # Align both return series on common dates
    # ==========================================================

    portfolio_returns, market_returns = portfolio_returns.align(
        market_returns,
        join="inner",
    )

    # ==========================================================
    # Annualized returns
    # ==========================================================

    portfolio_return = (
        portfolio_returns.mean()
        * trading_days
    )

    market_return = (
        market_returns.mean()
        * trading_days
    )

    # ==========================================================
    # Portfolio Beta
    # ==========================================================

    portfolio_beta = beta(
        portfolio_returns,
        market_returns,
    )

    # ==========================================================
    # Expected Return (CAPM)
    # ==========================================================

    expected_return = (
        risk_free_rate
        + portfolio_beta
        * (market_return - risk_free_rate)
    )

    # ==========================================================
    # Jensen's Alpha
    # ==========================================================

    return portfolio_return - expected_return

# =============================================================================
# EXPECTED CAPM RETURN
# =============================================================================

def expected_capm_return(
    beta_value: float,
    market_return: float,
    risk_free_rate: float = 0.0,
) -> float:
    """
    Compute expected return from CAPM.
    """

    return (
        risk_free_rate
        + beta_value
        * (market_return - risk_free_rate)
    )

# =============================================================================
# JENSEN'S ALPHA
# =============================================================================

def jensens_alpha(
    portfolio_returns: pd.Series,
    market_returns: pd.Series,
    risk_free_rate: float = 0.0,
    trading_days: int = 252,
) -> float:
    """
    Compute Jensen's Alpha.
    """

    return alpha(
        portfolio_returns,
        market_returns,
        risk_free_rate,
        trading_days,
    )

# =============================================================================
# TREYNOR RATIO
# =============================================================================

def treynor_ratio(
    portfolio_returns: pd.Series,
    market_returns: pd.Series,
    risk_free_rate: float = 0.0,
    trading_days: int = 252,
) -> float:
    """
    Compute Treynor Ratio.
    """

    portfolio_return = portfolio_returns.mean() * trading_days

    portfolio_beta = beta(
        portfolio_returns,
        market_returns,
    )

    return (
        portfolio_return - risk_free_rate
    ) / portfolio_beta

# =============================================================================
# TRACKING ERROR
# =============================================================================

def tracking_error(
    portfolio_returns: pd.Series,
    market_returns: pd.Series,
    trading_days: int = 252,
) -> float:
    """
    Compute annualized tracking error.
    """

    active_returns = (
        portfolio_returns
        - market_returns
    )

    return (
        active_returns.std()
        * np.sqrt(trading_days)
    )

# =============================================================================
# INFORMATION RATIO
# =============================================================================

def information_ratio(
    portfolio_returns: pd.Series,
    market_returns: pd.Series,
    trading_days: int = 252,
) -> float:
    """
    Compute Information Ratio.
    """

    active_return = (
        portfolio_returns.mean()
        - market_returns.mean()
    ) * trading_days

    te = tracking_error(
        portfolio_returns,
        market_returns,
        trading_days,
    )

    return active_return / te

# =============================================================================
# R-SQUARED
# =============================================================================

def r_squared(
    portfolio_returns: pd.Series,
    market_returns: pd.Series,
) -> float:
    """
    Compute the coefficient of determination (R²) between the
    portfolio and the market benchmark.

    Parameters
    ----------
    portfolio_returns : pd.Series
        Daily portfolio return series.

    market_returns : pd.Series
        Daily market (benchmark) return series.

    Returns
    -------
    float
        Coefficient of determination (R²).
    """

    # ==========================================================
    # Convert one-column DataFrames to Series
    # ==========================================================

    if isinstance(portfolio_returns, pd.DataFrame):
        portfolio_returns = portfolio_returns.squeeze()

    if isinstance(market_returns, pd.DataFrame):
        market_returns = market_returns.squeeze()

    # ==========================================================
    # Align both return series
    # ==========================================================

    portfolio_returns, market_returns = portfolio_returns.align(
        market_returns,
        join="inner",
    )

    # ==========================================================
    # Compute correlation
    # ==========================================================

    correlation = np.corrcoef(
        portfolio_returns,
        market_returns,
    )[0, 1]

    # ==========================================================
    # Coefficient of Determination
    # ==========================================================

    return correlation ** 2
# =============================================================================
# CAPM PERFORMANCE REPORT
# =============================================================================

def capm_report(
    portfolio_returns: pd.Series,
    market_returns: pd.Series,
    risk_free_rate: float = 0.0,
    trading_days: int = 252,
) -> pd.DataFrame:
    """
    Generate a comprehensive CAPM performance report.

    Parameters
    ----------
    portfolio_returns : pd.Series
        Daily portfolio returns.

    market_returns : pd.Series
        Daily benchmark returns.

    risk_free_rate : float, default=0.0
        Annualized risk-free rate.

    trading_days : int, default=252
        Number of trading days per year.

    Returns
    -------
    pd.DataFrame
        CAPM performance metrics.
    """

    # ==========================================================
    # Convert one-column DataFrames to Series
    # ==========================================================

    if isinstance(portfolio_returns, pd.DataFrame):
        portfolio_returns = portfolio_returns.squeeze()

    if isinstance(market_returns, pd.DataFrame):
        market_returns = market_returns.squeeze()

    # ==========================================================
    # Align both return series
    # ==========================================================

    portfolio_returns, market_returns = portfolio_returns.align(
        market_returns,
        join="inner",
    )

    # ==========================================================
    # Annualized Market Return
    # ==========================================================

    annual_market_return = (
        market_returns.mean()
        * trading_days
    )

    # ==========================================================
    # Portfolio Beta
    # ==========================================================

    portfolio_beta = beta(
        portfolio_returns,
        market_returns,
    )

    # ==========================================================
    # Portfolio Alpha
    # ==========================================================

    portfolio_alpha = alpha(
        portfolio_returns,
        market_returns,
        risk_free_rate,
        trading_days,
    )

    # ==========================================================
    # Expected CAPM Return
    # ==========================================================

    capm_return = expected_capm_return(
        portfolio_beta,
        annual_market_return,
        risk_free_rate,
    )

    # ==========================================================
    # Jensen's Alpha
    # ==========================================================

    jensen = jensens_alpha(
        portfolio_returns,
        market_returns,
        risk_free_rate,
        trading_days,
    )

    # ==========================================================
    # Treynor Ratio
    # ==========================================================

    treynor = treynor_ratio(
        portfolio_returns,
        market_returns,
        risk_free_rate,
        trading_days,
    )

    # ==========================================================
    # Tracking Error
    # ==========================================================

    te = tracking_error(
        portfolio_returns,
        market_returns,
        trading_days,
    )

    # ==========================================================
    # Information Ratio
    # ==========================================================

    info_ratio = information_ratio(
        portfolio_returns,
        market_returns,
        trading_days,
    )

    # ==========================================================
    # R-Squared
    # ==========================================================

    r2 = r_squared(
        portfolio_returns,
        market_returns,
    )

    # ==========================================================
    # Build Report
    # ==========================================================

    report = pd.DataFrame(
        {
            "Metric": [
                "Annual Market Return (%)",
                "Portfolio Beta",
                "Portfolio Alpha (%)",
                "Expected CAPM Return (%)",
                "Jensen's Alpha (%)",
                "Treynor Ratio",
                "Tracking Error (%)",
                "Information Ratio",
                "R-Squared",
            ],
            "Value": [
                round(float(annual_market_return) * 100, 2),
                round(float(portfolio_beta), 4),
                round(float(portfolio_alpha) * 100, 2),
                round(float(capm_return) * 100, 2),
                round(float(jensen) * 100, 2),
                round(float(treynor), 4),
                round(float(te) * 100, 2),
                round(float(info_ratio), 4),
                round(float(r2), 4),
            ],
        }
    )

    return report