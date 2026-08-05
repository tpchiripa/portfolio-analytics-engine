"""
Efficient Frontier Module

Author: Tichaona Peter Chiripa
Project: Portfolio Analytics Engine
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.optimization.mean_variance import portfolio_performance


def minimum_variance_target_return(
    target_return: float,
    expected_returns: pd.Series,
    covariance_matrix: pd.DataFrame,
) -> np.ndarray:
    """
    Compute the minimum variance portfolio for a specified target return.

    Parameters
    ----------
    target_return : float
        Desired annualized portfolio return.

    expected_returns : pd.Series
        Annualized expected asset returns.

    covariance_matrix : pd.DataFrame
        Annualized covariance matrix.

    Returns
    -------
    np.ndarray
        Optimal portfolio weights.
    """

    n_assets = len(expected_returns)

    initial_weights = np.ones(n_assets) / n_assets

    bounds = tuple((0, 1) for _ in range(n_assets))

    constraints = (
        {
            "type": "eq",
            "fun": lambda w: np.sum(w) - 1,
        },
        {
            "type": "eq",
            "fun": lambda w: np.dot(w, expected_returns) - target_return,
        },
    )

    def portfolio_variance(weights):
        """
        Portfolio variance objective function.
        """

        return (
            portfolio_performance(
                weights,
                expected_returns,
                covariance_matrix,
            )[1]
            ** 2
        )

    result = minimize(
        portfolio_variance,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    if not result.success:
        raise ValueError(
            f"Optimization failed: {result.message}"
        )

    return result.x


def efficient_frontier(
    expected_returns: pd.Series,
    covariance_matrix: pd.DataFrame,
    points: int = 150,
) -> pd.DataFrame:
    """
    Generate the Markowitz Efficient Frontier.

    Parameters
    ----------
    expected_returns : pd.Series
        Annualized expected returns.

    covariance_matrix : pd.DataFrame
        Annualized covariance matrix.

    points : int, default=150
        Number of portfolios along the frontier.

    Returns
    -------
    pd.DataFrame
        Efficient frontier statistics.
    """

    target_returns = np.linspace(
        expected_returns.min(),
        expected_returns.max(),
        points,
    )

    frontier = []

    for target in target_returns:

        weights = minimum_variance_target_return(
            target,
            expected_returns,
            covariance_matrix,
        )

        portfolio_return, portfolio_risk = portfolio_performance(
            weights,
            expected_returns,
            covariance_matrix,
        )

        frontier.append(
            {
                "Target Return (%)": round(
                    target * 100,
                    2,
                ),
                "Expected Return (%)": round(
                    portfolio_return * 100,
                    2,
                ),
                "Annualized Volatility (%)": round(
                    portfolio_risk * 100,
                    2,
                ),
                "Weights": weights,
            }
        )

    frontier = pd.DataFrame(frontier)

    return frontier