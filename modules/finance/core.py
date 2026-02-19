"""
Finance Core - Policy fills and payout calculation.

This module provides the core insurance financial calculations:
- Policy column fills (limits, deductibles) with identity-based overrides
- Payout calculation: min(max(gross - deductible, 0), limit)
- Main API: apply_financials()

Units: all monetary values are in kUSD (thousands of USD).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Optional, Any

import numpy as np
import pandas as pd

__all__ = [
    "apply_financials",
    "KUSD_EPS",
    "_safe_series",
    "_payout",
    "_apply_policy_fills",
]

KUSD_EPS = 1e-9  # small epsilon to guard floating round-off in kUSD


def _import_vuln(modules_root: Path):
    """Import vulnerability module from project structure."""
    vul_root = modules_root / "vulnerability"
    sys.path.insert(0, str(vul_root))
    from vulnerability import Vulnerability  # type: ignore
    return Vulnerability


def _safe_series(df: pd.DataFrame, name: str, default: float) -> pd.Series:
    """Get a column from df as float Series, or return default-filled Series."""
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce").astype(float)
    return pd.Series([np.nan] * len(df), index=df.index, dtype=float)


def _payout(gross: pd.Series, limit_: pd.Series, ded: pd.Series) -> pd.Series:
    """Calculate insurance payout: min(max(gross - deductible, 0), limit)."""
    base = (gross - ded).clip(lower=0.0)
    return np.minimum(base, limit_)


def _apply_policy_fills(
    out: pd.DataFrame,
    identity: pd.Series,
    policy: Dict[str, Any],
    policy_overwrite: bool,
    colnames: Dict[str, str],
) -> None:
    """
    Fill/override policy columns from dict following precedence.
    
    Precedence (lowest to highest):
      (1) global defaults (limit=inf, deductible=0)
      (2) scalar constants in policy
      (3) identity-level fills (owner/renter specific)
      (4) keyed mapping (per household_id or tract)
    """
    # Ensure columns exist
    for key in ["limit_structure_kUSD", "deductible_structure_kUSD",
                "limit_contents_kUSD", "deductible_contents_kUSD"]:
        col = colnames[key]
        if col not in out.columns:
            out[col] = np.nan

    def _set_where(col: str, values: pd.Series, mask: Optional[pd.Series] = None):
        if mask is None:
            mask = pd.Series([True] * len(out), index=out.index)
        if policy_overwrite:
            out.loc[mask, col] = values.loc[mask]
        else:
            to_fill = mask & out[col].isna()
            out.loc[to_fill, col] = values.loc[to_fill]

    # (2) scalar constants
    scalar_keys = ["limit_structure_kUSD", "deductible_structure_kUSD",
                   "limit_contents_kUSD", "deductible_contents_kUSD"]
    for sk in scalar_keys:
        if sk in policy and isinstance(policy[sk], (int, float)):
            val = float(policy[sk])
            _set_where(colnames[sk], pd.Series(val, index=out.index))

    # (3) identity-level fills
    for id_key in ["owner", "renter", "unknown"]:
        if isinstance(policy.get(id_key), dict):
            mask = identity.eq(id_key)
            for sk in scalar_keys:
                if sk in policy[id_key]:
                    val = float(policy[id_key][sk])
                    _set_where(colnames[sk], pd.Series(val, index=out.index), mask=mask)

    # (4) keyed mapping
    if isinstance(policy.get("data"), dict):
        by = policy.get("by")
        if by is None:
            for cand in ["household_id", "hh_id", "CensusTract", "tract_geoid", "GEOID"]:
                if cand in out.columns:
                    by = cand
                    break
        if by and by in out.columns:
            pdata = policy["data"]
            series_by = out[by].astype(str)
            for sk in scalar_keys:
                col = colnames[sk]
                vals = out[col].copy()
                for idx, key_val in series_by.items():
                    entry = pdata.get(str(key_val))
                    if isinstance(entry, dict) and sk in entry:
                        val = float(entry[sk])
                        if policy_overwrite or pd.isna(vals.at[idx]):
                            vals.at[idx] = val
                out[col] = vals


def apply_financials(
    df: pd.DataFrame,
    *,
    colmap: Optional[Dict[str, str]] = None,
    policy: Optional[Dict[str, Any]] = None,
    policy_overwrite: bool = False,
    dec: Optional[pd.DataFrame] = None,
    idxer: Optional[pd.DataFrame] = None,
    year: Optional[int] = None,
    gate_by_decisions: bool = True,
    premium: Optional[Dict[str, Any]] = None,
    owner_insures_both: bool = False,
) -> pd.DataFrame:
    """
    Apply insurance financials at the household level.
    
    Steps:
      1) fill policy columns (limits/deductibles)
      2) compute RAW payouts per coverage (structure/contents), UNgated
      3) apply identity gating (owners: structure; renters: contents)
      4) apply decision gating (no FI => payouts=0)
      5) compute OOP = loss - payout
      6) compute premiums (structure/contents)
      7) derive totals + USD columns
      
    Returns:
        DataFrame with payout_*, oop_*, premium_* (kUSD & USD), has_claim, gross_total.
    """
    # Import at runtime to avoid circular imports
    from .decisions import _apply_decisions_gating, _infer_year_from_df
    from .premium import compute_premiums_per_household

    # ---- column map & copy ----
    cmap = {
        "identity": "identity",
        "gross_structure_loss": "gross_structure_loss_kUSD",
        "gross_contents_loss": "gross_contents_loss_kUSD",
        "limit_structure_kUSD": "limit_structure_kUSD",
        "deductible_structure_kUSD": "deductible_structure_kUSD",
        "limit_contents_kUSD": "limit_contents_kUSD",
        "deductible_contents_kUSD": "deductible_contents_kUSD",
    }
    if colmap:
        cmap.update(colmap)

    out = df.copy()

    # ---- identity & base losses (kUSD) ----
    identity = (
        out[cmap["identity"]].astype(str).str.lower()
        if cmap["identity"] in out.columns
        else pd.Series(["unknown"] * len(out), index=out.index)
    )
    is_owner = identity.eq("owner")
    is_renter = identity.eq("renter")

    g_struct = _safe_series(out, cmap["gross_structure_loss"], 0.0).astype(float).fillna(0.0)
    g_cont = _safe_series(out, cmap["gross_contents_loss"], 0.0).astype(float).fillna(0.0)

    # ---- ensure policy cols present ----
    for key in ["limit_structure_kUSD", "deductible_structure_kUSD",
                "limit_contents_kUSD", "deductible_contents_kUSD"]:
        col = cmap[key]
        if col not in out.columns:
            out[col] = np.nan

    # ---- fill policy from dict ----
    if policy:
        _apply_policy_fills(out, identity, policy, policy_overwrite, cmap)

    # default fills for missing policy numbers
    lim_struct = pd.to_numeric(out[cmap["limit_structure_kUSD"]], errors="coerce").fillna(np.inf)
    ded_struct = pd.to_numeric(out[cmap["deductible_structure_kUSD"]], errors="coerce").fillna(0.0)
    lim_cont = pd.to_numeric(out[cmap["limit_contents_kUSD"]], errors="coerce").fillna(np.inf)
    ded_cont = pd.to_numeric(out[cmap["deductible_contents_kUSD"]], errors="coerce").fillna(0.0)

    # ---- (2) RAW payouts per coverage (UNgated) ----
    payout_struct_all = _payout(g_struct, lim_struct, ded_struct)
    payout_cont_all = _payout(g_cont, lim_cont, ded_cont)

    # ---- (3) identity gating to coverage ----
    owner_insures_both_flag = bool(owner_insures_both)

    payout_structure = payout_struct_all.where(is_owner | (~is_owner & ~is_renter), 0.0)
    if owner_insures_both_flag:
        payout_contents = payout_cont_all.where(is_owner | is_renter | (~is_owner & ~is_renter), 0.0)
    else:
        payout_contents = payout_cont_all.where(is_renter | (~is_owner & ~is_renter), 0.0)

    # ---- (4) decision gating ----
    out["payout_structure_kUSD"] = pd.to_numeric(payout_structure, errors="coerce").fillna(0.0)
    out["payout_contents_kUSD"] = pd.to_numeric(payout_contents, errors="coerce").fillna(0.0)

    if gate_by_decisions:
        yr = year if year is not None else _infer_year_from_df(out)
        out = _apply_decisions_gating(out, dec=dec, idxer=idxer, year=yr)

    # Read back after gating
    payout_structure = pd.to_numeric(out.get("payout_structure_kUSD", 0.0), errors="coerce").fillna(0.0)
    payout_contents = pd.to_numeric(out.get("payout_contents_kUSD", 0.0), errors="coerce").fillna(0.0)

    # ---- (5) OOP after gating ----
    oop_structure = (g_struct - payout_structure).clip(lower=0.0)
    oop_contents = (g_cont - payout_contents).clip(lower=0.0)

    # ---- totals (kUSD) ----
    payout_total_kUSD = payout_structure + payout_contents
    oop_total_kUSD = oop_structure + oop_contents
    gross_total_kUSD = g_struct + g_cont

    # ---- policyholder flag ----
    out["has_FI"] = pd.to_numeric(out.get("has_FI", 0), errors="coerce").fillna(0).astype(int)
    out["policyholder"] = out["has_FI"]

    out["payout_structure_kUSD"] = payout_structure
    out["payout_contents_kUSD"] = payout_contents
    out["payout_total_kUSD"] = payout_total_kUSD

    out["oop_structure_kUSD"] = oop_structure
    out["oop_contents_kUSD"] = oop_contents
    out["oop_total_kUSD"] = oop_total_kUSD

    # ★ Uninsured enforcement
    _ins = pd.to_numeric(out.get("has_FI", 0), errors="coerce").fillna(0).astype(int).eq(1)
    _cols = ["payout_structure_kUSD", "payout_contents_kUSD", "payout_total_kUSD",
             "oop_structure_kUSD", "oop_contents_kUSD", "oop_total_kUSD"]
    for c in _cols:
        if c in out.columns:
            out.loc[~_ins, c] = 0.0
    if "payout_total_kUSD" in out.columns:
        out["has_claim"] = (_ins & (out["payout_total_kUSD"] > 0)).astype(int)

    out["gross_total_kUSD"] = gross_total_kUSD
    out["has_claim"] = (payout_total_kUSD > 0).astype(int)

    # ---- (6) Premiums ----
    if isinstance(premium, dict):
        out = compute_premiums_per_household(
            out,
            rate_bldg_1k=premium.get("rate_bldg_1k", 3.5),
            rate_cont_1k=premium.get("rate_cont_1k", 5.0),
            contents_share=premium.get("contents_share", 0.30),
            contents_share_owner=premium.get("contents_share_owner", 0.57),
            owner_insures_both=premium.get("owner_insures_both", owner_insures_both_flag),
            reserve=premium.get("reserve", 1.00),
            small_fee_usd=premium.get("small_fee_usd", 0.0),
            col_identity=cmap.get("identity", "identity"),
            col_rcv_kUSD=colmap.get("rcv_kUSD", "rcv_kUSD") if colmap else "rcv_kUSD",
            col_contents_kUSD=colmap.get("contents_kUSD", "contents_kUSD") if colmap else "contents_kUSD",
            col_ded_struct_kUSD=cmap.get("deductible_structure_kUSD", "deductible_structure_kUSD"),
            col_ded_cont_kUSD=cmap.get("deductible_contents_kUSD", "deductible_contents_kUSD"),
        )
    else:
        for col in ["premium_structure_kUSD", "premium_contents_kUSD", "premium_total_kUSD"]:
            if col not in out.columns:
                out[col] = 0.0

    # ---- (7) USD columns ----
    for col in [
        "gross_structure_loss_kUSD", "gross_contents_loss_kUSD", "gross_total_kUSD",
        "payout_structure_kUSD", "payout_contents_kUSD", "payout_total_kUSD",
        "oop_structure_kUSD", "oop_contents_kUSD", "oop_total_kUSD",
        "premium_structure_kUSD", "premium_contents_kUSD", "premium_total_kUSD",
    ]:
        if col in out.columns:
            out[col.replace("_kUSD", "_usd")] = pd.to_numeric(out[col], errors="coerce").fillna(0.0) * 1000.0

    # Ensure components exist
    out["gross_structure_loss_kUSD"] = pd.to_numeric(
        out.get("gross_structure_loss_kUSD", out.get("gross_structure_loss", 0.0)), errors="coerce"
    ).fillna(0.0)
    out["gross_contents_loss_kUSD"] = pd.to_numeric(
        out.get("gross_contents_loss_kUSD", out.get("gross_contents_loss", 0.0)), errors="coerce"
    ).fillna(0.0)

    out["gross_total_kUSD"] = out["gross_structure_loss_kUSD"] + out["gross_contents_loss_kUSD"]

    if ("payout_structure_kUSD" in out.columns) or ("payout_contents_kUSD" in out.columns):
        out["payout_structure_kUSD"] = pd.to_numeric(out.get("payout_structure_kUSD", 0.0), errors="coerce").fillna(0.0)
        out["payout_contents_kUSD"] = pd.to_numeric(out.get("payout_contents_kUSD", 0.0), errors="coerce").fillna(0.0)
        out["payout_total_kUSD"] = out["payout_structure_kUSD"] + out["payout_contents_kUSD"]

    # Sanity check
    _bad = (out["gross_total_kUSD"] - (out["gross_structure_loss_kUSD"] + out["gross_contents_loss_kUSD"])).abs() > 1e-6
    if _bad.any():
        print(f"[warn] {_bad.sum()} household rows with mismatched gross_total_kUSD; recomputed by sum of parts.")

    return out
