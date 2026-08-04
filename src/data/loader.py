"""
Functions for downloading market data.
"""

import yfinance as yf
from src.utils.config import (
    TICKERS,
    MARKET_INDEX,
    TRAIN_START,
    TEST_END,
    INTERVAL,
)


def download_stock_prices():
    """
    Download adjusted close prices for all portfolio assets.
    """

    prices = yf.download(
        TICKERS,
        start=TRAIN_START,
        end=TEST_END,
        interval=INTERVAL,
        auto_adjust=True,
        progress=False,
    )["Close"]

    return prices


def download_market_index():
    """
    Download S&P 500 index.
    """

    market = yf.download(
        MARKET_INDEX,
        start=TRAIN_START,
        end=TEST_END,
        interval=INTERVAL,
        auto_adjust=True,
        progress=False,
    )["Close"]

    return market