"""
Utilities for generating and saving report tables.

Author: Tichaona Peter Chiripa
Project: Portfolio Analytics Engine
"""

from pathlib import Path
import pandas as pd


# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

REPORTS_DIR = PROJECT_ROOT / "reports" / "tables"


# =============================================================================
# TABLE UTILITIES
# =============================================================================

def save_table(
    dataframe: pd.DataFrame,
    table_number: int,
    title: str,
) -> None:
    """
    Save a formatted table to the project's reports/tables directory.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Table to save.

    table_number : int
        Table number used in the filename.

    title : str
        Table title.
    """

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    filename = (
        f"Table_{table_number}_"
        f"{title.replace(' ', '_')}.csv"
    )

    filepath = REPORTS_DIR / filename

    dataframe.to_csv(
        filepath,
        index=False,
    )

    print(f"\nTable {table_number}. {title}")
    print(dataframe)

    print(f"\nSaved to:\n{filepath}")