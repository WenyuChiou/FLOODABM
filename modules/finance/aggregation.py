"""
Finance Aggregation - Tract-level aggregation of household financials.

This module provides:
- detect_tract_column(): Auto-detect tract column name
- aggregate_by_tract(): Aggregate household-level data to tract level
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

__all__ = ["aggregate_by_tract", "detect_tract_column"]


def detect_tract_column(df: pd.DataFrame) -> Optional[str]:
    """
    Auto-detect tract column name from DataFrame.
    
    Searches for common column names like 'CensusTract', 'tract_geoid', 'GEOID'.
    """
    for cand in ["CensusTract", "tract_geoid", "tract_geoid10", "GEOID", "geoid"]:
        if cand in df.columns:
            return cand
    # try to detect GEOID-like
    for c in df.columns:
        if "tract" in c.lower() or "geoid" in c.lower():
            return c
    return None


def aggregate_by_tract(hh: pd.DataFrame, tract_col: str = "tract_geoid") -> pd.DataFrame:
    """
    Tract-level aggregation (unified kUSD standard).
    
    Aggregates:
    - Overall gross/payout/oop/premium (structure/contents/total)
    - owner_gross_total_kUSD, renter_gross_total_kUSD
    - owner_households, renter_households, households
    - policyholders, claims
    
    Args:
        hh: Household-level DataFrame from apply_financials()
        tract_col: Name of tract column (auto-detected if not found)
        
    Returns:
        Tract-level aggregated DataFrame
    """
    if hh is None or hh.empty:
        return pd.DataFrame(columns=[tract_col])

    df = hh.copy()

    # Tract column handling
    if tract_col not in df.columns:
        for cand in ("CensusTract", "GEOID", "tract", "geoid", "tract_geoid10"):
            if cand in df.columns:
                df = df.rename(columns={cand: tract_col})
                break
    df[tract_col] = df[tract_col].astype(str)

    # Identity flags
    ident = df.get("identity", "").astype(str).str.lower()
    df["_is_owner"] = (ident == "owner").astype(int)
    df["_is_renter"] = (ident == "renter").astype(int)

    # Helper: Ensure *_kUSD components exist
    def _ensure_kusd(modern: str, legacy: str):
        if modern in df.columns:
            df[modern] = pd.to_numeric(df[modern], errors="coerce").fillna(0.0)
        elif legacy in df.columns:
            df[modern] = pd.to_numeric(df[legacy], errors="coerce").fillna(0.0)
        else:
            df[modern] = 0.0

    # gross components
    _ensure_kusd("gross_structure_loss_kUSD", "gross_structure_loss")
    _ensure_kusd("gross_contents_loss_kUSD", "gross_contents_loss")

    # payout / oop / premium
    _ensure_kusd("payout_structure_kUSD", "payout_structure")
    _ensure_kusd("payout_contents_kUSD", "payout_contents")
    _ensure_kusd("oop_structure_kUSD", "oop_structure")
    _ensure_kusd("oop_contents_kUSD", "oop_contents")
    _ensure_kusd("premium_structure_kUSD", "premium_structure")
    _ensure_kusd("premium_contents_kUSD", "premium_contents")

    # Policy flag
    if "has_FI" in df.columns:
        df["has_FI"] = pd.to_numeric(df["has_FI"], errors="coerce").fillna(0).astype(int)
    else:
        df["has_FI"] = 0

    # has_claim
    if "has_claim" in df.columns:
        df["has_claim"] = pd.to_numeric(df["has_claim"], errors="coerce").fillna(0).astype(int)
    else:
        df["has_claim"] = 0

    # Per-row totals
    df["gross_total_row_kUSD"] = df["gross_structure_loss_kUSD"] + df["gross_contents_loss_kUSD"]
    df["owner_total_row_kUSD"] = df["gross_total_row_kUSD"] * df["_is_owner"]
    df["renter_total_row_kUSD"] = df["gross_total_row_kUSD"] * df["_is_renter"]

    # GroupBy aggregation
    grp = df.groupby(tract_col, dropna=False).agg(
        gross_structure_loss_kUSD=("gross_structure_loss_kUSD", "sum"),
        gross_contents_loss_kUSD=("gross_contents_loss_kUSD", "sum"),
        payout_structure_kUSD=("payout_structure_kUSD", "sum"),
        payout_contents_kUSD=("payout_contents_kUSD", "sum"),
        oop_structure_kUSD=("oop_structure_kUSD", "sum"),
        oop_contents_kUSD=("oop_contents_kUSD", "sum"),
        premium_structure_kUSD=("premium_structure_kUSD", "sum"),
        premium_contents_kUSD=("premium_contents_kUSD", "sum"),
        gross_total_kUSD=("gross_total_row_kUSD", "sum"),
        owner_gross_total_kUSD=("owner_total_row_kUSD", "sum"),
        renter_gross_total_kUSD=("renter_total_row_kUSD", "sum"),
        owner_households=("_is_owner", "sum"),
        renter_households=("_is_renter", "sum"),
        policyholders=("has_FI", "sum"),
        claims=("has_claim", "sum"),
    ).reset_index()

    # Derive totals
    grp["payout_total_kUSD"] = grp["payout_structure_kUSD"] + grp["payout_contents_kUSD"]
    grp["oop_total_kUSD"] = grp["oop_structure_kUSD"] + grp["oop_contents_kUSD"]
    grp["premium_total_kUSD"] = grp["premium_structure_kUSD"] + grp["premium_contents_kUSD"]
    grp["gross_total_kUSD"] = grp["gross_structure_loss_kUSD"] + grp["gross_contents_loss_kUSD"]
    grp["households"] = grp["owner_households"].astype(int) + grp["renter_households"].astype(int)

    # Convert to USD
    for col in [c for c in grp.columns if c.endswith("_kUSD")]:
        grp[col.replace("_kUSD", "_usd")] = pd.to_numeric(grp[col], errors="coerce").fillna(0.0) * 1000.0

    # Ensure count columns are integers
    for c in ("owner_households", "renter_households", "households", "policyholders", "claims"):
        if c in grp.columns:
            grp[c] = pd.to_numeric(grp[c], errors="coerce").fillna(0).astype(int)

    return grp
