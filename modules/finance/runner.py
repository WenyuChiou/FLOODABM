"""
Finance Runner - High-level orchestrator for yearly finance processing.

This module provides run_finance_for_year() which:
- Calculates flood damage losses
- Applies insurance payouts
- Calculates premiums
- Aggregates results by tract
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .core import apply_financials
from .aggregation import aggregate_by_tract

__all__ = ["run_finance_for_year"]


def run_finance_for_year(
    *,
    year: int,
    state_event: pd.DataFrame,
    depth_map: Dict[str, float],
    ratio_by_tract: Dict[str, float],
    decisions: pd.DataFrame,
    policy: Dict,
    idxer: Optional[object] = None,
    gate_by_decisions: bool = True,
    renters_have_structure: bool = False,
    tract_col: str = "tract_geoid",
    modules_root: Path = Path("."),
    output_dir: Optional[Path] = None,
    save_csv: bool = True,
    compact_output: bool = False,
    premium: Optional[Dict[str, Any]] = None,
    owner_contents_ratio: float = 0.57,
    owner_insures_both: bool = False,
    ffe_ft: float = 0.5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply insurance financials for a single simulation year.
    
    Args:
        ...
        compact_output: If True, drop redundant columns (e.g. *_usd) to save space
        ...
    """
    df = state_event.copy()

    # ---- Column and type normalization ----
    if "group" in df.columns and "identity" not in df.columns:
        df = df.rename(columns={"group": "identity"})
    if "identity" not in df.columns:
        raise KeyError("state_event missing 'identity' column (must be 'owner' / 'renter').")

    df[tract_col] = df[tract_col].astype(str)
    df["identity"] = df["identity"].astype(str).str.lower()

    # Event reference
    df["event_year"] = year
    df["event_tract_geoid"] = df[tract_col].astype(str)

    # Map flood depths
    if "event_depth_m" not in df.columns:
        df["event_depth_m"] = df["event_tract_geoid"].map(
            lambda t: float(depth_map.get(str(t), np.nan))
        ).astype(float)

    df["event_loss_ratio"] = df["event_tract_geoid"].map(
        lambda t: float(ratio_by_tract.get(str(t), 0.0))
    ).astype(float)

    # Ensure monetary columns exist
    for col in ("rcv_kUSD", "contents_kUSD"):
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Fill missing owner contents
    is_owner = df["identity"].eq("owner")
    need_fill = is_owner & (pd.to_numeric(df["contents_kUSD"], errors="coerce").fillna(0.0) <= 0.0)
    if owner_contents_ratio is not None and need_fill.any():
        df.loc[need_fill, "contents_kUSD"] = owner_contents_ratio * df.loc[need_fill, "rcv_kUSD"]

    # Renter structure loss gating
    is_renter = df["identity"].eq("renter")
    if not renters_have_structure:
        df.loc[is_renter, "gross_structure_loss_kUSD"] = 0.0

    # Sync legacy column names
    df["gross_structure_loss"] = pd.to_numeric(df["gross_structure_loss_kUSD"], errors="coerce").fillna(0.0)
    df["gross_contents_loss"] = pd.to_numeric(df["gross_contents_loss_kUSD"], errors="coerce").fillna(0.0)

    # ---- Apply insurance financials ----
    idxer_for_fin = None if ("i" in getattr(decisions, "columns", [])) else idxer
    FIN_HH = apply_financials(
        df,
        policy=policy,
        dec=decisions,
        idxer=idxer_for_fin,
        year=year,
        gate_by_decisions=gate_by_decisions,
        premium=premium,
        owner_insures_both=owner_insures_both,
        colmap={
            "gross_structure_loss": "gross_structure_loss",
            "gross_contents_loss": "gross_contents_loss",
        },
    )

    # ---- Aggregate to tract level ----
    FIN_TRACT = aggregate_by_tract(FIN_HH, tract_col=tract_col)

    # ---- Save CSV outputs ----
    if save_csv and output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        out_hh = FIN_HH.copy()
        if compact_output:
            # Drop redundant *_usd columns (can be derived from *_kUSD)
            usd_cols = [c for c in out_hh.columns if c.endswith("_usd") or c.endswith("_USD")]
            # Also drop duplicated identity/loss columns if they exist
            redundant = [
                "identity_norm", "identity_dec", 
                "gross_structure_loss", "gross_contents_loss",
                "event_depth_m", "event_tract_geoid", "event_year" # often redundant with state
            ]
            cols_to_drop = set(usd_cols + redundant)
            out_hh = out_hh.drop(columns=[c for c in cols_to_drop if c in out_hh.columns])
            
        out_hh.to_csv(output_dir / f"finance_households_{year}.csv", index=False, encoding="utf-8-sig")
        FIN_TRACT.to_csv(output_dir / f"finance_tract_{year}.csv", index=False, encoding="utf-8-sig")

    # ---- Tract-level premium breakdown ----
    tmp = FIN_HH.copy()
    tmp["identity"] = tmp["identity"].astype(str).str.lower()
    tmp["tract_geoid"] = tmp["tract_geoid"].astype(str)

    tmp["prem_s_usd"] = pd.to_numeric(tmp.get("premium_structure_kUSD", 0.0), errors="coerce").fillna(0.0) * 1000.0
    tmp["prem_c_usd"] = pd.to_numeric(tmp.get("premium_contents_kUSD", 0.0), errors="coerce").fillna(0.0) * 1000.0

    o = (
        tmp[tmp["identity"] == "owner"]
        .groupby("tract_geoid")[["prem_s_usd", "prem_c_usd"]]
        .sum()
        .rename(columns={
            "prem_s_usd": "owner_premium_structure_usd",
            "prem_c_usd": "owner_premium_contents_usd",
        })
    )
    r = (
        tmp[tmp["identity"] == "renter"]
        .groupby("tract_geoid")[["prem_c_usd"]]
        .sum()
        .rename(columns={"prem_c_usd": "renter_premium_contents_usd"})
    )

    prem_tract = o.join(r, how="outer").fillna(0.0).reset_index()
    prem_tract["owner_premium_total_usd"] = (
        prem_tract["owner_premium_structure_usd"] + prem_tract["owner_premium_contents_usd"]
    )
    prem_tract["renter_premium_total_usd"] = prem_tract["renter_premium_contents_usd"]

    FIN_TRACT = FIN_TRACT.merge(prem_tract, on="tract_geoid", how="left")

    return FIN_HH, FIN_TRACT
