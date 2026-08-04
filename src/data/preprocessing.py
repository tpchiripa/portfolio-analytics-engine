"""
Data preprocessing utilities.

Author: Tichaona Peter Chiripa
Project: Portfolio Analytics Engine
"""

import pandas as pd

from src.utils.config import (
    TRAIN_END,
    TEST_START,
)


def clean_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Remove missing observations.
    """

    prices = prices.sort_index()

    prices = prices.dropna(how="all")

    prices = prices.ffill()

    prices = prices.dropna()

    return prices


def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily simple returns.
    """

    returns = prices.pct_change()

    returns = returns.dropna()

    return returns


def split_train_test(returns: pd.DataFrame):
    """
    Split into training and test datasets.
    """

    train = returns.loc[:TRAIN_END]

    test = returns.loc[TEST_START:]

    return train, test


def annualized_mean_returns(train_returns: pd.DataFrame):
    """
    Annualized expected returns.
    """

    return train_returns.mean() * 252


def annualized_covariance(train_returns: pd.DataFrame):
    """
    Annualized covariance matrix.
    """

    return train_returns.cov() * 252


def summary_statistics(returns: pd.DataFrame):
    """
    Descriptive statistics.
    """

    return returns.describe().T