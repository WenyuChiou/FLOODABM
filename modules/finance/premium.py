"""
Finance Premium - Insurance premium calculation.

This module calculates insurance premiums per household based on:
- Coverage amounts (structure, contents)
- Base rates and reserve factors
- Deductible adjustments
- Identity-based coverage rules (owner: structure, renter: contents)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

__all__ = ["compute_premiums_per_household", "_to_series"]


def _to_series(x, index, name) -> pd.Series:
    """Broadcast scalar or pass-through Series aligned to index."""
    if isinstance(x, (int, float)):
        return pd.Series(float(x), index=index, name=name)
    if isinstance(x, pd.Series):
        return pd.to_numeric(x, errors="coerce").reindex(index).astype(float).rename(name)
    raise TypeError(f"{name} must be scalar or pandas.Series aligned to df.")


def compute_premiums_per_household(
    df: pd.DataFrame,
    *,
    # Base rate ($/1k)
    rate_bldg_1k: float = 3.56,
    rate_cont_1k: float = 4.98,
    contents_share: float = 0.30,  # Default ratio for renters
    contents_share_owner: float = 0.57,  # Owner's contents:structure ratio
    owner_insures_both: bool = False,  # True: owner insures contents too
    reserve: float = 1.15,
    small_fee_usd: float = 100.0,
    # Column mapping
    col_identity: str = "identity",
    col_rcv_kUSD: str = "rcv_kUSD",
    col_contents_kUSD: str = "contents_kUSD",
    col_ded_struct_kUSD: str = "deductible_structure_kUSD",
    col_ded_cont_kUSD: str = "deductible_contents_kUSD",
    clip_floor: float = 0.001,
) -> pd.DataFrame:
    """
    Calculate insurance premiums per household.
    
    Premium = base_rate * (1 - deductible/coverage) * coverage * reserve + small_fee
    
    Args:
        df: Household DataFrame with coverage and identity columns
        rate_bldg_1k: Building rate per $1k of coverage
        rate_cont_1k: Contents rate per $1k of coverage
        contents_share: Default contents ratio for renters
        contents_share_owner: Default contents ratio for owners
        owner_insures_both: If True, owners also insure contents
        reserve: Reserve factor multiplier
        small_fee_usd: Fixed fee per policy
        
    Returns:
        DataFrame with premium_structure_kUSD, premium_contents_kUSD, premium_total_kUSD
    """
    out = df.copy()
    idx = out.index

    rb = _to_series(rate_bldg_1k, idx, "rate_bldg_1k")
    rc = _to_series(rate_cont_1k, idx, "rate_cont_1k")

    identity = (
        out[col_identity].astype(str).str.lower()
        if col_identity in out.columns
        else pd.Series("unknown", index=idx)
    )

    owner_insures_both_flag = bool(owner_insures_both)

    is_owner = identity.eq("owner")
    is_renter = identity.eq("renter")

    rcv_k = pd.to_numeric(out.get(col_rcv_kUSD, 0.0), errors="coerce").fillna(0.0)
    cont_k_raw = pd.to_numeric(out.get(col_contents_kUSD, np.nan), errors="coerce")

    # Fill missing contents_kUSD
    cont_k = cont_k_raw.copy()
    mask_owner_na = is_owner & cont_k.isna()
    cont_k.loc[mask_owner_na] = contents_share_owner * rcv_k.loc[mask_owner_na]
    mask_renter_na = is_renter & cont_k.isna()
    cont_k.loc[mask_renter_na] = contents_share * rcv_k.loc[mask_renter_na]
    cont_k = cont_k.fillna(0.0)

    # Coverage (by identity/settings)
    cov_struct_k = np.where(is_owner | (~is_owner & ~is_renter), rcv_k, 0.0)
    if owner_insures_both_flag:
        cov_cont_k = np.where(is_owner | is_renter | (~is_owner & ~is_renter), cont_k, 0.0)
    else:
        cov_cont_k = np.where(is_renter | (~is_owner & ~is_renter), cont_k, 0.0)

    ded_s = pd.to_numeric(out.get(col_ded_struct_kUSD, 0.0), errors="coerce").fillna(0.0)
    ded_c = pd.to_numeric(out.get(col_ded_cont_kUSD, 0.0), errors="coerce").fillna(0.0)

    fac_s = np.clip(1.0 - np.divide(ded_s, np.maximum(cov_struct_k, 1e-9)), clip_floor, 1.0)
    fac_c = np.clip(1.0 - np.divide(ded_c, np.maximum(cov_cont_k, 1e-9)), clip_floor, 1.0)

    small_fee_kUSD = float(small_fee_usd) / 1000.0

    prem_s_k = (rb / 1000.0) * fac_s * cov_struct_k * reserve
    prem_c_k = (rc / 1000.0) * fac_c * cov_cont_k * reserve

    # Small fixed fee
    fee_struct_mask = is_owner
    fee_cont_mask = is_renter | (is_owner & owner_insures_both_flag)

    fee_struct = small_fee_kUSD * fee_struct_mask.astype(float)
    fee_cont = small_fee_kUSD * fee_cont_mask.astype(float)

    # Robustly coerce "has_FI" to a boolean Series
    raw_insured = out["has_FI"] if "has_FI" in out.columns else 1

    if np.isscalar(raw_insured):
        insured_ser = pd.Series([raw_insured] * len(out), index=out.index)
    else:
        insured_ser = pd.Series(raw_insured, index=out.index)

    insured = pd.to_numeric(insured_ser, errors="coerce").fillna(1.0) > 0
    insured_f = insured.astype(float)

    prem_s_k = prem_s_k * insured_f
    prem_c_k = prem_c_k * insured_f
    fee_struct = fee_struct * insured_f
    fee_cont = fee_cont * insured_f

    out["premium_structure_kUSD"] = prem_s_k + fee_struct
    out["premium_contents_kUSD"] = prem_c_k + fee_cont
    out["premium_total_kUSD"] = out["premium_structure_kUSD"] + out["premium_contents_kUSD"]
    out["has_FI"] = insured.astype(int)

    return out
