"""
Efficient Frontier Module

Author: Tichaona Peter Chiripa
Project: Portfolio Analytics Engine
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.optimization.mean_variance import portfolio_performance

def minimum_variance_portfolio(
    target_return,
    expected_returns,
    covariance_matrix,
):
    """
    Compute the minimum variance portfolio for a target return.
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
            "fun": lambda w:
                np.dot(w, expected_returns) - target_return,
        },
    )

    def portfolio_variance(weights):
        return portfolio_performance(
            weights,
            expected_returns,
            covariance_matrix,
        )[1] ** 2

    result = minimize(
        portfolio_variance,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    return result.x

def efficient_frontier(
    expected_returns,
    covariance_matrix,
    points=50,
):
    """
    Generate Efficient Frontier.
    """

    target_returns = np.linspace(
        expected_returns.min(),
        expected_returns.max(),
        points,
    )

    frontier = []

    for target in target_returns:

        weights = minimum_variance_portfolio(
            target,
            expected_returns,
            covariance_matrix,
        )

        portfolio_return, portfolio_risk = (
            portfolio_performance(
                weights,
                expected_returns,
                covariance_matrix,
            )
        )

        frontier.append(
            {
                "Return": portfolio_return * 100,
                "Risk": portfolio_risk * 100,
            }
        )

    return pd.DataFrame(frontier)