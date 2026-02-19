"""
Finance Package - Insurance financial calculations.

This package provides household-level and tract-level financial calculations
for flood insurance, including payouts, premiums, and OOP costs.

Submodules:
- core: apply_financials() and policy fill logic
- decisions: Decision file loading and gating
- premium: Premium calculation
- aggregation: Tract-level aggregation
- runner: run_finance_for_year() orchestrator

Public APIs:
- apply_financials(df, ...) -> pd.DataFrame
- aggregate_by_tract(hh, tract_col) -> pd.DataFrame
- detect_tract_column(df) -> str
- run_finance_for_year(...) -> (FIN_HH, FIN_TRACT)
- compute_premiums_per_household(df, ...) -> pd.DataFrame
"""

from .core import apply_financials
from .aggregation import aggregate_by_tract, detect_tract_column
from .runner import run_finance_for_year
from .premium import compute_premiums_per_household

# Backward compatibility: also expose from legacy finance.py (will be deprecated)
# For now, keep finance.py content as-is for any direct imports

__all__ = [
    "apply_financials",
    "aggregate_by_tract",
    "detect_tract_column",
    "run_finance_for_year",
    "compute_premiums_per_household",
]
