"""
Monte Carlo Portfolio Simulation

Author: Tichaona Peter Chiripa
Project: Portfolio Analytics Engine

This module generates random portfolios for visualizing the
risk-return opportunity set and validating portfolio optimization
results.
"""

import numpy as np
import pandas as pd

from src.optimization.mean_variance import portfolio_performance


# =============================================================================
# MONTE CARLO SIMULATION
# =============================================================================

def simulate_portfolios(
    expected_returns: pd.Series,
    covariance_matrix: pd.DataFrame,
    n_portfolios: int = 50000,
    risk_free_rate: float = 0.0,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generate random portfolios using Monte Carlo simulation.

    Parameters
    ----------
    expected_returns : pd.Series
        Annualized expected returns.

    covariance_matrix : pd.DataFrame
        Annualized covariance matrix.

    n_portfolios : int, default=50000
        Number of portfolios to simulate.

    risk_free_rate : float, default=0.0
        Annualized risk-free rate.

    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Monte Carlo simulation results.
    """

    np.random.seed(random_state)

    n_assets = len(expected_returns)

    simulation_results = []

    # ==========================================================
    # Generate Random Portfolios
    # ==========================================================

    for portfolio_id in range(1, n_portfolios + 1):

        # ------------------------------------------------------
        # Random portfolio weights
        # ------------------------------------------------------

        weights = np.random.random(n_assets)

        weights /= np.sum(weights)

        # ------------------------------------------------------
        # Portfolio statistics
        # ------------------------------------------------------

        portfolio_return, portfolio_risk = portfolio_performance(
            weights,
            expected_returns,
            covariance_matrix,
        )

        sharpe_ratio = (
            portfolio_return - risk_free_rate
        ) / portfolio_risk

        # ------------------------------------------------------
        # Store results
        # ------------------------------------------------------

        simulation_results.append(
            {
                "Portfolio": portfolio_id,
                "Expected Return (%)": portfolio_return * 100,
                "Risk (%)": portfolio_risk * 100,
                "Sharpe Ratio": sharpe_ratio,
            }
        )

    simulation = pd.DataFrame(simulation_results)

    simulation[
        [
            "Expected Return (%)",
            "Risk (%)",
            "Sharpe Ratio",
        ]
    ] = simulation[
        [
            "Expected Return (%)",
            "Risk (%)",
            "Sharpe Ratio",
        ]
    ].round(2)

    return simulation


# =============================================================================
# TABLE 5: MONTE CARLO SIMULATION SUMMARY
# =============================================================================

def simulation_summary(
    simulation_results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate Table 5:
    Summary statistics for Monte Carlo portfolio simulation.

    Parameters
    ----------
    simulation_results : pd.DataFrame
        Output from simulate_portfolios().

    Returns
    -------
    pd.DataFrame
        Summary statistics.
    """

    summary = (
        simulation_results[
            [
                "Expected Return (%)",
                "Risk (%)",
                "Sharpe Ratio",
            ]
        ]
        .describe()
        .round(2)
    )

    summary.index.name = "Statistic"

    return summary


# =============================================================================
# TABLE 6: TOP SHARPE PORTFOLIOS
# =============================================================================

def top_sharpe_portfolios(
    simulation_results: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Generate Table 6:
    Highest Sharpe Ratio portfolios from the Monte Carlo simulation.

    Parameters
    ----------
    simulation_results : pd.DataFrame
        Output from simulate_portfolios().

    top_n : int, default=10
        Number of portfolios to return.

    Returns
    -------
    pd.DataFrame
        Top-performing portfolios ranked by Sharpe Ratio.
    """

    top_portfolios = (
        simulation_results
        .sort_values(
            by="Sharpe Ratio",
            ascending=False,
        )
        .head(top_n)
        .reset_index(drop=True)
    )

    top_portfolios.index += 1
    top_portfolios.index.name = "Rank"

    return top_portfolios


# =============================================================================
# TABLE 7: MONTE CARLO RISK-RETURN BOUNDS
# =============================================================================

def simulation_bounds(
    simulation_results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate Table 7:
    Minimum and maximum values observed during
    the Monte Carlo simulation.

    Parameters
    ----------
    simulation_results : pd.DataFrame
        Output from simulate_portfolios().

    Returns
    -------
    pd.DataFrame
        Risk-return bounds.
    """

    bounds = pd.DataFrame(
        {
            "Metric": [
                "Minimum Expected Return (%)",
                "Maximum Expected Return (%)",
                "Minimum Risk (%)",
                "Maximum Risk (%)",
                "Minimum Sharpe Ratio",
                "Maximum Sharpe Ratio",
            ],
            "Value": [
                simulation_results["Expected Return (%)"].min(),
                simulation_results["Expected Return (%)"].max(),
                simulation_results["Risk (%)"].min(),
                simulation_results["Risk (%)"].max(),
                simulation_results["Sharpe Ratio"].min(),
                simulation_results["Sharpe Ratio"].max(),
            ],
        }
    )

    bounds["Value"] = bounds["Value"].round(2)

    return bounds