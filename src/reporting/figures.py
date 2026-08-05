"""
Utilities for generating and saving report figures.

Author: Tichaona Peter Chiripa
Project: Portfolio Analytics Engine
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# FIGURE 1: MARKOWITZ EFFICIENT FRONTIER
# =============================================================================

def plot_efficient_frontier(
    simulation_results: pd.DataFrame,
    frontier: pd.DataFrame,
    max_sharpe: dict,
    min_variance: dict,
    equal_weight: dict,
    filename: str = "Figure_1_Markowitz_Efficient_Frontier.png",
):
    """
    Generate Figure 1:
    Markowitz Efficient Frontier and Monte Carlo Portfolio Simulation.
    """

    fig, ax = plt.subplots(figsize=(12, 8))

    # ==========================================================
    # Monte Carlo Portfolio Universe
    # ==========================================================

    scatter = ax.scatter(
        simulation_results["Risk (%)"],
        simulation_results["Expected Return (%)"],
        c=simulation_results["Sharpe Ratio"],
        cmap="viridis",
        s=12,
        alpha=0.45,
    )

    cbar = plt.colorbar(
        scatter,
        ax=ax,
    )

    cbar.set_label(
        "Portfolio Sharpe Ratio",
        fontsize=12,
        fontweight="bold",
    )

    # ==========================================================
    # Efficient Frontier (Upper Branch Only)
    # ==========================================================

    minimum_variance_index = frontier[
        "Annualized Volatility (%)"
    ].idxmin()

    efficient_frontier = frontier.loc[
        minimum_variance_index:
    ]

    ax.plot(
        efficient_frontier["Annualized Volatility (%)"],
        efficient_frontier["Expected Return (%)"],
        color="black",
        linewidth=2.5,
        linestyle="-",
        label="Efficient Frontier",
        zorder=8,
    )

    # ==========================================================
    # Maximum Sharpe Portfolio
    # ==========================================================

    ax.scatter(
        max_sharpe["risk"] * 100,
        max_sharpe["return"] * 100,
        color="red",
        marker="*",
        s=380,
        edgecolors="black",
        linewidth=1.2,
        label="Maximum Sharpe Portfolio",
        zorder=10,
    )

    ax.annotate(
        "Maximum Sharpe",
        (
            max_sharpe["risk"] * 100,
            max_sharpe["return"] * 100,
        ),
        xytext=(15, 20),
        textcoords="offset points",
        fontsize=10,
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.30",
            fc="white",
            ec="black",
            alpha=0.95,
        ),
    )

    # ==========================================================
    # Minimum Variance Portfolio
    # ==========================================================

    ax.scatter(
        min_variance["risk"] * 100,
        min_variance["return"] * 100,
        color="royalblue",
        marker="D",
        s=170,
        edgecolors="black",
        linewidth=1.2,
        label="Minimum Variance Portfolio",
        zorder=10,
    )

    ax.annotate(
        "Minimum Variance",
        (
            min_variance["risk"] * 100,
            min_variance["return"] * 100,
        ),
        xytext=(15, -28),
        textcoords="offset points",
        fontsize=10,
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.30",
            fc="white",
            ec="black",
            alpha=0.95,
        ),
    )

    # ==========================================================
    # Equal Weight Portfolio
    # ==========================================================

    ax.scatter(
        equal_weight["risk"] * 100,
        equal_weight["return"] * 100,
        color="darkorange",
        marker="o",
        s=170,
        edgecolors="black",
        linewidth=1.2,
        label="Equal Weight Portfolio",
        zorder=10,
    )

    ax.annotate(
        "Equal Weight",
        (
            equal_weight["risk"] * 100,
            equal_weight["return"] * 100,
        ),
        xytext=(15, 15),
        textcoords="offset points",
        fontsize=10,
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.30",
            fc="white",
            ec="black",
            alpha=0.95,
        ),
    )

    # ==========================================================
    # Axis Formatting
    # ==========================================================

    ax.set_xlim(
        simulation_results["Risk (%)"].min() - 1,
        simulation_results["Risk (%)"].max() + 1,
    )

    ax.set_ylim(
        simulation_results["Expected Return (%)"].min() - 2,
        simulation_results["Expected Return (%)"].max() + 2,
    )

    figure_title = (
        "Figure 1.\n"
        "Markowitz Efficient Frontier and Monte Carlo Portfolio Simulation"
    )

    ax.set_title(
        figure_title,
        fontsize=18,
        fontweight="bold",
        pad=30,
    )

    ax.set_xlabel(
        "Annualized Volatility (%)",
        fontsize=13,
        fontweight="bold",
    )

    ax.set_ylabel(
        "Expected Annual Return (%)",
        fontsize=13,
        fontweight="bold",
    )

    ax.tick_params(
        axis="both",
        labelsize=11,
    )

    ax.grid(
        linestyle="--",
        linewidth=0.7,
        alpha=0.35,
    )

    ax.legend(
        loc="lower right",
        fontsize=10,
        frameon=True,
        fancybox=True,
        shadow=True,
    )

    plt.subplots_adjust(top=0.88)

    filepath = FIGURES_DIR / filename

    plt.savefig(
        filepath,
        dpi=600,
        bbox_inches="tight",
    )

    plt.show()

    print("=" * 70)
    print("Figure 1. Markowitz Efficient Frontier and Monte Carlo Portfolio Simulation")
    print("=" * 70)
    print(f"Saved to:\n{filepath}")

    return fig