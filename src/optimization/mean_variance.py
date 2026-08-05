"""
Mean-Variance Portfolio Optimization

Author: Tichaona Peter Chiripa
Project: Portfolio Analytics Engine
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize


# =============================================================================
# PORTFOLIO METRICS
# =============================================================================

def portfolio_performance(
    weights: np.ndarray,
    expected_returns: pd.Series,
    covariance_matrix: pd.DataFrame,
) -> tuple:
    """
    Calculate annualized portfolio return and volatility.

    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights.

    expected_returns : pd.Series
        Annualized expected returns.

    covariance_matrix : pd.DataFrame
        Annualized covariance matrix.

    Returns
    -------
    tuple
        (portfolio_return, portfolio_volatility)
    """

    portfolio_return = np.dot(weights, expected_returns)

    portfolio_volatility = np.sqrt(
        np.dot(
            weights.T,
            np.dot(covariance_matrix, weights),
        )
    )

    return portfolio_return, portfolio_volatility


def portfolio_variance(
    weights: np.ndarray,
    covariance_matrix: pd.DataFrame,
) -> float:
    """
    Calculate portfolio variance.
    """

    return np.dot(
        weights.T,
        np.dot(covariance_matrix, weights),
    )


def negative_sharpe_ratio(
    weights: np.ndarray,
    expected_returns: pd.Series,
    covariance_matrix: pd.DataFrame,
    risk_free_rate: float = 0.0,
) -> float:
    """
    Objective function for maximizing the Sharpe Ratio.
    """

    portfolio_return, portfolio_risk = portfolio_performance(
        weights,
        expected_returns,
        covariance_matrix,
    )

    sharpe_ratio = (
        portfolio_return - risk_free_rate
    ) / portfolio_risk

    return -sharpe_ratio


# =============================================================================
# PORTFOLIO OPTIMIZATION
# =============================================================================

def optimize_portfolio(
    expected_returns: pd.Series,
    covariance_matrix: pd.DataFrame,
    risk_free_rate: float = 0.0,
) -> dict:
    """
    Compute the Maximum Sharpe Ratio Portfolio.
    """

    n_assets = len(expected_returns)

    initial_weights = np.ones(n_assets) / n_assets

    bounds = tuple((0, 1) for _ in range(n_assets))

    constraints = (
        {
            "type": "eq",
            "fun": lambda w: np.sum(w) - 1,
        },
    )

    result = minimize(
        negative_sharpe_ratio,
        initial_weights,
        args=(
            expected_returns,
            covariance_matrix,
            risk_free_rate,
        ),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    if not result.success:
        raise ValueError(
            f"Optimization failed: {result.message}"
        )

    optimal_weights = result.x

    portfolio_return, portfolio_risk = portfolio_performance(
        optimal_weights,
        expected_returns,
        covariance_matrix,
    )

    sharpe_ratio = (
        portfolio_return - risk_free_rate
    ) / portfolio_risk

    return {
        "weights": optimal_weights,
        "return": round(portfolio_return, 4),
        "risk": round(portfolio_risk, 4),
        "sharpe": round(sharpe_ratio, 2),
    }


def minimum_variance_portfolio(
    expected_returns: pd.Series,
    covariance_matrix: pd.DataFrame,
    risk_free_rate: float = 0.0,
) -> dict:
    """
    Compute the Minimum Variance Portfolio.
    """

    n_assets = len(expected_returns)

    initial_weights = np.ones(n_assets) / n_assets

    bounds = tuple((0, 1) for _ in range(n_assets))

    constraints = (
        {
            "type": "eq",
            "fun": lambda w: np.sum(w) - 1,
        },
    )

    result = minimize(
        portfolio_variance,
        initial_weights,
        args=(covariance_matrix,),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    if not result.success:
        raise ValueError(
            f"Optimization failed: {result.message}"
        )

    optimal_weights = result.x

    portfolio_return, portfolio_risk = portfolio_performance(
        optimal_weights,
        expected_returns,
        covariance_matrix,
    )

    sharpe_ratio = (
        portfolio_return - risk_free_rate
    ) / portfolio_risk

    return {
        "weights": optimal_weights,
        "return": round(portfolio_return, 4),
        "risk": round(portfolio_risk, 4),
        "sharpe": round(sharpe_ratio, 2),
    }


def equal_weight_portfolio(
    expected_returns: pd.Series,
    covariance_matrix: pd.DataFrame,
    risk_free_rate: float = 0.0,
) -> dict:
    """
    Compute the Equal Weight Portfolio.
    """

    n_assets = len(expected_returns)

    weights = np.ones(n_assets) / n_assets

    portfolio_return, portfolio_risk = portfolio_performance(
        weights,
        expected_returns,
        covariance_matrix,
    )

    sharpe_ratio = (
        portfolio_return - risk_free_rate
    ) / portfolio_risk

    return {
        "weights": weights,
        "return": round(portfolio_return, 4),
        "risk": round(portfolio_risk, 4),
        "sharpe": round(sharpe_ratio, 2),
    }


# =============================================================================
# REPORTING FUNCTIONS
# =============================================================================

def portfolio_summary(
    results: dict,
    tickers: pd.Index,
) -> pd.DataFrame:
    """
    Generate portfolio allocation table.
    """

    allocation = pd.DataFrame(
        {
            "Asset": tickers,
            "Weight (%)": results["weights"] * 100,
        }
    )

    allocation["Weight (%)"] = (
        allocation["Weight (%)"]
        .round(2)
    )

    allocation = (
        allocation
        .sort_values(
            by="Weight (%)",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    allocation.index += 1
    allocation.index.name = "Rank"

    return allocation


def performance_summary(
    results: dict,
) -> pd.DataFrame:
    """
    Generate portfolio performance summary.
    """

    return pd.DataFrame(
        {
            "Metric": [
                "Expected Annual Return (%)",
                "Annual Volatility (%)",
                "Sharpe Ratio",
            ],
            "Value": [
                round(results["return"] * 100, 2),
                round(results["risk"] * 100, 2),
                round(results["sharpe"], 2),
            ],
        }
    )


def strategy_comparison(
    equal_weight: dict,
    minimum_variance: dict,
    maximum_sharpe: dict,
) -> pd.DataFrame:
    """
    Generate comparison table for portfolio strategies.
    """

    comparison = pd.DataFrame(
        {
            "Portfolio Strategy": [
                "Equal Weight",
                "Minimum Variance",
                "Maximum Sharpe",
            ],
            "Expected Return (%)": [
                round(equal_weight["return"] * 100, 2),
                round(minimum_variance["return"] * 100, 2),
                round(maximum_sharpe["return"] * 100, 2),
            ],
            "Risk (%)": [
                round(equal_weight["risk"] * 100, 2),
                round(minimum_variance["risk"] * 100, 2),
                round(maximum_sharpe["risk"] * 100, 2),
            ],
            "Sharpe Ratio": [
                round(equal_weight["sharpe"], 2),
                round(minimum_variance["sharpe"], 2),
                round(maximum_sharpe["sharpe"], 2),
            ],
        }
    )

    return comparison