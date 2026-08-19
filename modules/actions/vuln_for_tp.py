# -*- coding: utf-8 -*-
"""
Vulnerability Integration for TP Updates
=========================================

This module bridges vulnerability calculations with TP (Trust in Policymaker) updates.
It provides functions to:

1. Load and process flood depth data by year
2. Calculate flood damage losses using vulnerability curves
3. Compute damage ratios by tract for TP shock calculations
4. Export tract-level flood damage summaries

The main entry point is `ratio_by_tract_from_vuln()` which calculates the
damage-to-value ratio used to determine TP shock intensity.

Example:
    >>> from modules.actions.vuln_for_tp import load_depths_all_years, ratio_by_tract_from_vuln
    >>> depths = load_depths_all_years(Path("config/depths_overall.json"))
    >>> ratios = ratio_by_tract_from_vuln(state_df, depths, 2021, modules_root)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# Vulnerability Module Import
# =============================================================================

_VULN_CLASS_CACHE = None

def _import_vuln(modules_root: Path):
    """Dynamically import Vulnerability class from vulnerability submodule.

    Args:
        modules_root: Path to modules/ directory

    Returns:
        Vulnerability class
    """
    global _VULN_CLASS_CACHE
    if _VULN_CLASS_CACHE is not None:
        return _VULN_CLASS_CACHE
    vul_root = modules_root / "vulnerability"
    if str(vul_root) not in sys.path:
        sys.path.insert(0, str(vul_root))
    from vulnerability import Vulnerability  # type: ignore
    _VULN_CLASS_CACHE = Vulnerability
    return Vulnerability


# =============================================================================
# Flood Depth Data Loading
# =============================================================================

def load_depths_all_years(path_overall_json: Path) -> pd.DataFrame:
    """Load flood depth data for all years from JSON file.
    
    Expects JSON format with columns like "11_mean", "14_mean", etc. for years.
    
    Args:
        path_overall_json: Path to overall_md_mean_by_tract_2011_2023.json
        
    Returns:
        Long-format DataFrame with columns:
        - tract_geoid: 11-digit census tract
        - year: 4-digit year (e.g., 2011)
        - depth_m: Mean flood depth in meters
    """
    df = pd.read_json(path_overall_json, dtype={"CensusTract": "string"})
    
    # Find year columns (format: "YY_mean" where YY is 2-digit year)
    year_cols = [c for c in df.columns if c.endswith("_mean") and c[:2].isdigit()]
    
    # Melt to long format
    long = df.melt(
        id_vars=["CensusTract"],
        value_vars=year_cols,
        var_name="yy",
        value_name="depth_m",
    )
    
    # Convert 2-digit year to 4-digit
    long["year"] = 2000 + long["yy"].str[:2].astype(int)
    long = long.drop(columns=["yy"]).rename(columns={"CensusTract": "tract_geoid"})
    
    # Clean up data types
    long["tract_geoid"] = long["tract_geoid"].astype(str).str.strip()
    long["depth_m"] = pd.to_numeric(long["depth_m"], errors="coerce").fillna(0.0).astype(float)
    
    return long[["tract_geoid", "year", "depth_m"]]


def depths_for_year(depths_long: pd.DataFrame, year: int) -> pd.DataFrame:
    """Extract flood depth data for a specific year.
    
    Args:
        depths_long: Long-format depths DataFrame from load_depths_all_years()
        year: Target year (4-digit)
        
    Returns:
        DataFrame with tract_geoid and mean depth_m for that year
    """
    d = depths_long.loc[depths_long["year"] == int(year), ["tract_geoid", "depth_m"]]
    return d.groupby("tract_geoid", as_index=False).mean()


# =============================================================================
# Vulnerability Input Preparation
# =============================================================================

def build_cat_input(
    state: pd.DataFrame,
    depths_y: pd.DataFrame,
    base_ffe_ft: float,
) -> pd.DataFrame:
    """Build catastrophe model input from state and depth data.
    
    Merges household state with flood depths and calculates effective FFE.
    
    Args:
        state: Household state DataFrame with columns:
            i, group, tract_geoid, rcv_kUSD, contents_kUSD, elev_ft_total
        depths_y: Depths DataFrame for current year
        base_ffe_ft: Base Finished Floor Elevation in feet
        
    Returns:
        DataFrame ready for vulnerability calculation
    """
    cat = state.merge(depths_y, on="tract_geoid", how="left").rename(columns={"group": "identity"})
    cat["depth_m"] = cat["depth_m"].fillna(0.0).astype(float)
    
    # Calculate effective FFE (base + any elevation improvements)
    elev_total = cat.get("elev_ft_total", 0.0)
    if isinstance(elev_total, pd.Series):
        elev_total = elev_total.fillna(0.0).astype(float)
    cat["ffe_ft_adjust"] = float(base_ffe_ft) + elev_total
    
    cols = ["i", "identity", "tract_geoid", "depth_m", "rcv_kUSD", "contents_kUSD", "ffe_ft_adjust"]
    return cat[cols].copy()


# =============================================================================
# Loss Calculation
# =============================================================================

def compute_losses_quick(
    cat_in: pd.DataFrame,
    modules_root: Path,
    FFE_ft: float,
) -> pd.DataFrame:
    """Compute flood losses for all households using vulnerability curves.
    
    Args:
        cat_in: Prepared input DataFrame from build_cat_input()
        modules_root: Path to modules/ directory
        FFE_ft: Base Finished Floor Elevation (used if per-row not available)
        
    Returns:
        DataFrame with original columns plus:
        - gross_structure_loss: Structure damage in kUSD
        - gross_contents_loss: Contents damage in kUSD
        - gross_total: Total damage in kUSD
    """
    Vulnerability = _import_vuln(modules_root)
    # Use ffe_ft=0 here since we pass per-row FFE in damage calls
    v = Vulnerability(ffe_ft=0.0)

    # Vectorized batch computation (Vulnerability accepts arrays)
    depth_m_arr = cat_in["depth_m"].to_numpy(dtype=float)
    ffe_arr = cat_in["ffe_ft_adjust"].to_numpy(dtype=float)
    rcv_arr = cat_in["rcv_kUSD"].to_numpy(dtype=float)
    con_arr = cat_in["contents_kUSD"].to_numpy(dtype=float)

    # Replace NaN with 0 for safe computation
    rcv_arr = np.nan_to_num(rcv_arr, nan=0.0)
    con_arr = np.nan_to_num(con_arr, nan=0.0)

    # Structure damage (batch)
    s_dmg, _ = v.structure_damage(rcv_arr, depth_m_arr, ffe_ft=ffe_arr)
    s_dmg = np.nan_to_num(s_dmg, nan=0.0)
    s_dmg[rcv_arr <= 0] = 0.0

    # Contents damage (batch)
    c_dmg, _ = v.contents_damage(con_arr, depth_m_arr, ffe_ft=ffe_arr)
    c_dmg = np.nan_to_num(c_dmg, nan=0.0)
    c_dmg[con_arr <= 0] = 0.0

    out = cat_in.reset_index(drop=True).copy()
    out["gross_structure_loss"] = s_dmg
    out["gross_contents_loss"] = c_dmg
    out["gross_total"] = s_dmg + c_dmg
    return out


# =============================================================================
# Tract-Level Ratio Calculation
# =============================================================================

def ratio_by_tract_from_vuln(
    state: pd.DataFrame,
    depths_long: pd.DataFrame,
    year: int,
    modules_root: Path,
    FFE_ft: float = 1.0,
    *,
    owner_contents_ratio: float = 0.57,
    renter_contents_fallback: float | None = None,
) -> dict[str, float]:
    """Calculate flood damage ratio by tract for TP shock calculation.
    
    Computes: ratio = total_loss / total_value per tract (0-1 scale)
    
    This ratio is used to determine the intensity of TP shock after floods.
    Higher ratios indicate more severe damage relative to property values.
    
    Args:
        state: Household state DataFrame
        depths_long: Long-format depths from load_depths_all_years()
        year: Target year
        modules_root: Path to modules/ directory
        FFE_ft: Base Finished Floor Elevation in feet
        owner_contents_ratio: Ratio to estimate owner contents if missing
        renter_contents_fallback: Optional ratio for missing renter contents
        
    Returns:
        Dict mapping tract_geoid to damage ratio (0-1)
    """
    # 1) Prepare state: normalize identity and fill missing values
    df = state.copy().reset_index(drop=True)
    df["identity"] = df.get("identity", df.get("group", "unknown")).astype(str).str.lower()
    df["tract_geoid"] = df["tract_geoid"].astype(str)
    
    df["rcv_kUSD"] = pd.to_numeric(df.get("rcv_kUSD", 0.0), errors="coerce").fillna(0.0)
    df["contents_kUSD"] = pd.to_numeric(df.get("contents_kUSD", np.nan), errors="coerce")
    
    is_owner = df["identity"].eq("owner")
    is_renter = df["identity"].eq("renter")
    
    # Fill missing owner contents using ratio
    need_owner_fill = is_owner & (df["contents_kUSD"].isna() | (df["contents_kUSD"] <= 0))
    df.loc[need_owner_fill, "contents_kUSD"] = owner_contents_ratio * df.loc[need_owner_fill, "rcv_kUSD"]
    
    # Optionally fill missing renter contents
    if renter_contents_fallback is not None:
        need_renter_fill = is_renter & (df["contents_kUSD"].isna() | (df["contents_kUSD"] <= 0))
        df.loc[need_renter_fill, "contents_kUSD"] = renter_contents_fallback * df.loc[need_renter_fill, "rcv_kUSD"]
    
    df["contents_kUSD"] = df["contents_kUSD"].fillna(0.0)
    
    # 2) Calculate losses using vulnerability model
    depths_y = depths_for_year(depths_long, year)
    cat_in = build_cat_input(df, depths_y, base_ffe_ft=FFE_ft)
    wl = compute_losses_quick(cat_in, modules_root, FFE_ft)
    
    # 3) Calculate loss and value by identity type
    # Owner: structure + contents loss/value
    # Renter: contents only loss/value
    loss_total = wl["gross_structure_loss"].where(is_owner, 0.0) + wl["gross_contents_loss"]
    value_total = df["rcv_kUSD"].where(is_owner, 0.0) + df["contents_kUSD"]
    
    tmp = pd.DataFrame({
        "tract_geoid": df["tract_geoid"],
        "loss_kUSD": pd.to_numeric(loss_total, errors="coerce").fillna(0.0),
        "value_kUSD": pd.to_numeric(value_total, errors="coerce").fillna(0.0),
    })
    
    # 4) Aggregate by tract and compute ratio
    g = tmp.groupby("tract_geoid", as_index=False).sum()
    g["ratio"] = np.where(g["value_kUSD"] > 0, (g["loss_kUSD"] / g["value_kUSD"]).clip(0.0, 1.0), 0.0)
    
    return dict(zip(g["tract_geoid"].astype(str), g["ratio"].astype(float)))


# =============================================================================
# Household Flood Damage Attachment
# =============================================================================

def _attach_hh_flood_damage(
    state_df: pd.DataFrame,
    year: int,
    depths_long: pd.DataFrame,
    modules_root: Path,
    ffe_ft: float = 1.0,
    renters_have_structure: bool = False,
    event_depth_m: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Attach flood damage amounts to household state DataFrame.
    
    Args:
        state_df: Household state DataFrame
        year: Target year
        depths_long: Long-format depths data
        modules_root: Path to modules/ directory
        ffe_ft: Base Finished Floor Elevation
        renters_have_structure: If False, set renter structure damage to 0
        
    Returns:
        DataFrame with added columns:
        - gross_structure_loss_kUSD
        - gross_contents_loss_kUSD
        - gross_total_kUSD
        - identity_norm
    """
    df = state_df.copy()
    
    # Normalize identity column
    if "identity" in df.columns:
        ident_raw = df["identity"]
    elif "group" in df.columns:
        ident_raw = df["group"]
    else:
        ident_raw = pd.Series(["renter"] * len(df), index=df.index)
    
    ident_norm = (
        ident_raw.astype(str).str.strip().str.lower()
        .replace({
            "owners": "owner", "homeowner": "owner", "homeowners": "owner",
            "renters": "renter", "tenant": "renter", "tenants": "renter",
        })
    )
    ident_norm = ident_norm.where(ident_norm.isin(["owner", "renter"]), "renter")
    df["identity_norm"] = ident_norm
    
    # Calculate losses
    dy = depths_for_year(depths_long, year)
    cat_in = build_cat_input(df, dy, base_ffe_ft=ffe_ft)
    if event_depth_m is not None:
        if not {"i", "event_depth_m"}.issubset(event_depth_m.columns):
            raise ValueError("event_depth_m override must contain i and event_depth_m")
        override = event_depth_m[["i", "event_depth_m"]].copy()
        override["i"] = pd.to_numeric(override["i"], errors="raise").astype(int)
        override["event_depth_m"] = pd.to_numeric(
            override["event_depth_m"], errors="raise"
        ).astype(float)
        invalid = (
            override["i"].duplicated().any()
            or not np.isfinite(override["event_depth_m"]).all()
            or (override["event_depth_m"] < 0).any()
        )
        if invalid:
            raise ValueError("event-depth overrides must be unique, finite, and nonnegative")
        df = df.drop(columns=["event_depth_m"], errors="ignore").merge(
            override,
            on="i",
            how="left",
            validate="one_to_one",
        )
        if df["event_depth_m"].isna().any():
            raise ValueError("event-depth override is incomplete")
        cat_in = cat_in.drop(columns=["depth_m"]).merge(
            override.rename(columns={"event_depth_m": "depth_m"}),
            on="i",
            how="left",
            validate="one_to_one",
        )
        if cat_in["depth_m"].isna().any():
            raise ValueError("event-depth override is incomplete")
    wl = compute_losses_quick(cat_in, modules_root, FFE_ft=ffe_ft)
    
    # Merge losses back to state
    out = df.merge(
        wl[["i", "gross_structure_loss", "gross_contents_loss", "gross_total"]],
        on="i",
        how="left",
    ).rename(columns={
        "gross_structure_loss": "gross_structure_loss_kUSD",
        "gross_contents_loss": "gross_contents_loss_kUSD",
        "gross_total": "gross_total_kUSD",
    }).fillna({
        "gross_structure_loss_kUSD": 0.0,
        "gross_contents_loss_kUSD": 0.0,
        "gross_total_kUSD": 0.0,
    })
    
    # Policy: renters don't have structure damage
    if not renters_have_structure:
        mask_r = out["identity_norm"].eq("renter")
        out.loc[mask_r, "gross_structure_loss_kUSD"] = 0.0
        out["gross_total_kUSD"] = out["gross_structure_loss_kUSD"] + out["gross_contents_loss_kUSD"]
    
    return out


# =============================================================================
# Tract-Level Damage Export
# =============================================================================

def _export_tract_flood_damage(
    state_with_damage: pd.DataFrame,
    year: int,
    out_root: Path,
) -> None:
    """Export tract-level flood damage summaries to CSV files.
    
    Creates files in out_root/vulnerability/flood_damage/:
    - flood_damage_tract_owner_breakdown_{year}.csv
    - flood_damage_tract_renter_breakdown_{year}.csv
    - flood_damage_tract_both_breakdown_{year}.csv
    - flood_damage_tract_owner_{year}.csv (simplified)
    - flood_damage_tract_renter_{year}.csv (simplified)
    - flood_damage_tract_both_{year}.csv (simplified)
    
    Args:
        state_with_damage: State DataFrame with damage columns
        year: Simulation year
        out_root: Root output directory
    """
    out_dir = out_root / "vulnerability" / "flood_damage"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    df = state_with_damage.copy()
    
    # Normalize tract_geoid column name
    if "tract_geoid" not in df.columns:
        for c in ("CensusTract", "GEOID", "tract", "geoid", "tract_geoid10"):
            if c in df.columns:
                df = df.rename(columns={c: "tract_geoid"})
                break
    df["tract_geoid"] = df["tract_geoid"].astype(str)
    
    # Normalize identity
    ident = (
        df.get("identity", df.get("group", "unknown"))
        .astype(str).str.lower()
        .map({"owner": "owner", "renter": "renter"})
        .fillna("unknown")
    )
    df["identity_norm"] = ident
    
    # Convert kUSD to USD
    s_usd = pd.to_numeric(df["gross_structure_loss_kUSD"], errors="coerce").fillna(0.0) * 1000.0
    c_usd = pd.to_numeric(df["gross_contents_loss_kUSD"], errors="coerce").fillna(0.0) * 1000.0
    
    # Owner breakdown
    o = df[df["identity_norm"] == "owner"].copy()
    o["structure_usd"] = s_usd.loc[o.index]
    o["contents_usd"] = c_usd.loc[o.index]
    owner = o.groupby("tract_geoid")[["structure_usd", "contents_usd"]].sum().reset_index()
    owner.insert(1, "year", int(year))
    owner["total_usd"] = owner["structure_usd"] + owner["contents_usd"]
    owner.sort_values("tract_geoid").to_csv(
        out_dir / f"flood_damage_tract_owner_breakdown_{year}.csv",
        index=False, encoding="utf-8-sig",
    )
    
    # Renter breakdown (contents only)
    r = df[df["identity_norm"] == "renter"].copy()
    r["contents_usd"] = c_usd.loc[r.index]
    renter = r.groupby("tract_geoid")[["contents_usd"]].sum().reset_index()
    renter.insert(1, "year", int(year))
    renter["total_usd"] = renter["contents_usd"]
    renter.sort_values("tract_geoid").to_csv(
        out_dir / f"flood_damage_tract_renter_breakdown_{year}.csv",
        index=False, encoding="utf-8-sig",
    )
    
    # Combined summary (owner structure+contents + renter contents)
    both = (
        pd.concat([
            o[["tract_geoid", "structure_usd", "contents_usd"]],
            r[["tract_geoid", "contents_usd"]].rename(columns={"contents_usd": "contents_usd_r"}),
        ], ignore_index=True)
        .groupby("tract_geoid").sum().reset_index()
    )
    both["contents_usd"] = both.get("contents_usd", 0.0) + both.get("contents_usd_r", 0.0)
    both = both.drop(columns=[c for c in ["contents_usd_r"] if c in both.columns])
    both.insert(1, "year", int(year))
    both["total_usd"] = both["structure_usd"].fillna(0.0) + both["contents_usd"].fillna(0.0)
    both.sort_values("tract_geoid").to_csv(
        out_dir / f"flood_damage_tract_both_breakdown_{year}.csv",
        index=False, encoding="utf-8-sig",
    )
    
    # Simplified files (backward compatible)
    owner[["tract_geoid", "year", "total_usd"]].rename(columns={"total_usd": "damage_usd"}).to_csv(
        out_dir / f"flood_damage_tract_owner_{year}.csv", index=False, encoding="utf-8-sig",
    )
    renter[["tract_geoid", "year", "total_usd"]].rename(columns={"total_usd": "damage_usd"}).to_csv(
        out_dir / f"flood_damage_tract_renter_{year}.csv", index=False, encoding="utf-8-sig",
    )
    both[["tract_geoid", "year", "total_usd"]].rename(columns={"total_usd": "damage_usd"}).to_csv(
        out_dir / f"flood_damage_tract_both_{year}.csv", index=False, encoding="utf-8-sig",
    )
