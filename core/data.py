# -*- coding: utf-8 -*-
"""
Data Loading Module for FLOODABM
================================

Centralized data loading functions for households, depths, and psychology data.
This module provides consistent data loading across all entry points.

Data Files:
    - households_for_abm.csv: Household exposure database
    - depths: Flood depth data (JSON or CSV)
    - tract psychology: Initial TP/CP/SP by census tract
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# Household Data Loading
# =============================================================================

def load_households(path: Path) -> pd.DataFrame:
    """Load household exposure database from CSV.
    
    Args:
        path: Path to households CSV file
        
    Returns:
        DataFrame with standardized columns:
        - i: Unique household ID
        - tract_geoid: Census tract (11-digit string)
        - group: "owner" or "renter"
        - identity: Original tenure type
        
    Raises:
        FileNotFoundError: If file doesn't exist
    """
    if not path.exists():
        raise FileNotFoundError(f"Households file not found: {path}")
    
    df = pd.read_csv(path, dtype={"tract_geoid": str})
    
    # Ensure required columns
    if "i" not in df.columns:
        df.insert(0, "i", range(len(df)))
    
    # Normalize tract_geoid to 11 digits
    if "tract_geoid" in df.columns:
        df["tract_geoid"] = df["tract_geoid"].astype(str).str.zfill(11)
    
    # Normalize tenure/group column
    if "identity" in df.columns:
        df["group"] = df["identity"].astype(str).str.lower().replace({
            "owner-occupied": "owner",
            "owner occupied": "owner",
            "homeowner": "owner",
            "own": "owner",
            "renter-occupied": "renter",
            "renter occupied": "renter",
            "rent": "renter",
            "tenant": "renter",
        })
    
    return df.reset_index(drop=True)


def init_insurance_flags(
    df: pd.DataFrame,
    insurance_config: dict,
    seed: int = 42,
) -> pd.DataFrame:
    """Initialize has_FI (flood insurance) flags based on config.
    
    Args:
        df: Household DataFrame
        insurance_config: Insurance initialization config from YAML
        seed: Random seed for sampling
        
    Returns:
        DataFrame with has_FI column added/updated
    """
    df = df.copy()
    rng = np.random.default_rng(seed)
    
    # Initialize has_FI column
    if "has_FI" not in df.columns:
        df["has_FI"] = 0
    df["has_FI"] = pd.to_numeric(df["has_FI"], errors="coerce").fillna(0).astype(int)
    
    # Apply rates from config
    rate_overall = insurance_config.get("take_rate_overall")
    if rate_overall is not None:
        rate = float(rate_overall)
        n_total = len(df)
        n_insured = int(round(rate * n_total))
        if n_insured > 0:
            chosen = rng.choice(df.index, size=min(n_insured, n_total), replace=False)
            df.loc[chosen, "has_FI"] = 1
    
    return df


# =============================================================================
# Flood Depth Data Loading
# =============================================================================

def load_depths(path: Path) -> pd.DataFrame:
    """Load flood depth data from file.
    
    Supports both JSON and CSV formats.
    
    Args:
        path: Path to depths file
        
    Returns:
        DataFrame with columns:
        - year: Flood year
        - tract_geoid: Census tract (11-digit string)
        - depth_m: Flood depth in meters
        
    Raises:
        FileNotFoundError: If file doesn't exist
    """
    if not path.exists():
        raise FileNotFoundError(f"Depths file not found: {path}")
    
    suffix = path.suffix.lower()
    
    if suffix == ".json":
        df = pd.read_json(path).T.reset_index()
        df.columns = ["tract_geoid"] + list(df.columns[1:])
        # Unpivot years into long format
        df = df.melt(id_vars=["tract_geoid"], var_name="year", value_name="depth_m")
    else:
        # Assume CSV
        df = pd.read_csv(path, dtype={"tract_geoid": str})
    
    # Standardize columns
    df["tract_geoid"] = df["tract_geoid"].astype(str).str.zfill(11)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype(int)
    df["depth_m"] = pd.to_numeric(df["depth_m"], errors="coerce").fillna(0.0)
    
    return df.sort_values(["year", "tract_geoid"]).reset_index(drop=True)


def get_depths_for_year(depths_long: pd.DataFrame, year: int) -> pd.DataFrame:
    """Extract depth data for a specific year.
    
    Args:
        depths_long: Full depths DataFrame in long format
        year: Year to extract
        
    Returns:
        DataFrame with tract_geoid and depth_m columns for that year
    """
    mask = depths_long["year"] == year
    return depths_long.loc[mask, ["tract_geoid", "depth_m"]].copy()


def get_available_years(depths_long: pd.DataFrame) -> list[int]:
    """Get list of years available in depth data.
    
    Args:
        depths_long: Full depths DataFrame
        
    Returns:
        Sorted list of years
    """
    return sorted(depths_long["year"].unique().tolist())


# =============================================================================
# Tract Psychology Initialization
# =============================================================================

def init_tract_psychology(
    tracts: list[str],
    seed: int,
    rng: Optional[np.random.RandomState] = None,
) -> pd.DataFrame:
    """Initialize tract-level psychology (TP, CP, SP) values.

    Creates initial psychological state for each census tract with
    owner (homeowner) and renter values.

    Args:
        tracts: List of tract geoid strings
        seed: Random seed
        rng: Optional pre-existing RandomState

    Returns:
        DataFrame with columns:
        - tract_geoid, TP_owner, CP_owner, SP_owner, TP_renter, CP_renter, SP_renter
    """
    if rng is None:
        rng = np.random.RandomState(seed)

    n = len(tracts)

    # Initialize with uniform(0.3, 0.7) for moderate starting values
    data = {
        "tract_geoid": [str(t) for t in tracts],
        "TP_owner": rng.uniform(0.3, 0.7, n),
        "CP_owner": rng.uniform(0.3, 0.7, n),
        "SP_owner": rng.uniform(0.3, 0.7, n),
        "TP_renter": rng.uniform(0.3, 0.7, n),
        "CP_renter": rng.uniform(0.3, 0.7, n),
        "SP_renter": rng.uniform(0.3, 0.7, n),
    }

    return pd.DataFrame(data)


# =============================================================================
# Owner Share and Policy Loading
# =============================================================================

def load_owner_share(config: dict) -> dict[str, float]:
    """Load owner (homeowner) share by tract from config.

    Args:
        config: YAML config dict

    Returns:
        Dict mapping tract_geoid -> owner share (0-1)
    """
    section = config.get("owner_share", {}) or {}

    # Default share if not specified
    default_share = float(section.get("default", 0.7))

    # Load per-tract overrides
    by_tract = section.get("by_tract", {}) or {}

    shares = {str(t): float(v) for t, v in by_tract.items()}
    shares["_default"] = default_share

    return shares


def get_owner_share_for_tract(shares: dict, tract_geoid: str) -> float:
    """Get owner share for a specific tract.

    Args:
        shares: Owner share dict from load_owner_share()
        tract_geoid: Tract geoid string

    Returns:
        Owner share value (0-1)
    """
    return shares.get(str(tract_geoid), shares.get("_default", 0.7))

