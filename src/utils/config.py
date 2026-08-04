"""
Project configuration settings.
"""

# Portfolio universe (Assignment assets)
TICKERS = [
    "TSLA",
    "WMT",
    "BAC",
    "GS",
    "LLY",
    "MRK",
    "GOOG",
    "META",
    "AAPL",
    "XOM",
]

# Benchmark
MARKET_INDEX = "^GSPC"

# Training Period
TRAIN_START = "2023-09-01"
TRAIN_END = "2025-09-30"

# Out-of-Sample Testing Period
TEST_START = "2025-10-01"
TEST_END = "2025-12-31"

# Data frequency
INTERVAL = "1d"

# Data directories
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"