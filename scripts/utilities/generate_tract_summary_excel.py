# -*- coding: utf-8 -*-
"""
Task 0: Tract-Level Summary Table (Excel)
==========================================
Generates comprehensive tract-level summary Excel with:
- Tract identifiers + flood characteristics
- Action shares by year (owner: FI/EH/BP, renter: FI/RL)
- Cumulative financial: damage, payout, OOP, premium (baseline vs worst-case)
- Per-household damage, damage reduction %
- Post-flood behavioral change for 2014 and 2021 events

Usage:
    python generate_tract_summary_excel.py
"""
from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

# ── paths ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
BASE_DIR = ROOT / "outputs" / "baseline" / "baseline"
WORST_DIR = ROOT / "outputs" / "baseline" / "worst"
CONFIG = ROOT / "config" / "overall_md_mean_by_tract_2011_2023.json"
OUT_DIR = BASE_DIR / "excel"
OUT_FILE = OUT_DIR / "tract_summary_all.xlsx"

YEARS = list(range(2011, 2024))
FLOOD_EVENTS = [2014, 2021]  # Major flood years for pre/post analysis

# County FIPS mapping (NJ tracts in the Passaic River Basin)
COUNTY_MAP = {
    "34013": "Hudson",
    "34017": "Hudson",
    "34027": "Morris",
    "34031": "Passaic",
    "34003": "Bergen",
    "34023": "Middlesex",
    "34025": "Monmouth",
    "34029": "Ocean",
    "34035": "Somerset",
    "34039": "Union",
}


def get_county(geoid: str) -> str:
    """Extract county name from tract GEOID (first 5 digits = state+county FIPS)."""
    fips5 = str(geoid)[:5]
    return COUNTY_MAP.get(fips5, f"Unknown ({fips5})")


def load_flood_config() -> pd.DataFrame:
    """Load flood depth config and compute flood characteristics per tract."""
    with open(CONFIG, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for rec in data:
        geoid = rec["CensusTract"]
        depths = []
        flood_yrs = 0
        for yr in YEARS:
            key = f"{yr % 100:02d}_mean"
            d = rec.get(key, 0.0)
            depths.append(d)
            if d > 0:
                flood_yrs += 1
        mean_depth = np.mean([d for d in depths if d > 0]) if flood_yrs > 0 else 0.0
        rows.append({
            "tract_geoid": geoid,
            "county": get_county(geoid),
            "flood_prone": 1 if flood_yrs > 0 else 0,
            "mean_depth_m": round(mean_depth, 4),
            "n_flood_yrs": flood_yrs,
        })
    return pd.DataFrame(rows)


def load_action_shares() -> pd.DataFrame:
    """Load action shares and pivot to wide format (one row per tract)."""
    p = BASE_DIR / "decisions" / "action_share_owner_renter_tract_all_years.csv"
    df = pd.read_csv(p, encoding="utf-8")

    # Pivot: for each year, create oFI_YYYY, oEH_YYYY, oBP_YYYY, rFI_YYYY, rRL_YYYY
    rows = []
    for geoid, gdf in df.groupby("tract_geoid"):
        row = {"tract_geoid": str(geoid)}
        for _, r in gdf.iterrows():
            yr = int(r["year"])
            row[f"oFI_{yr}"] = round(r["owner_share_FI"], 4)
            row[f"oEH_{yr}"] = round(r["owner_share_EH"], 4)
            row[f"oBP_{yr}"] = round(r["owner_share_BP"], 4)
            row[f"rFI_{yr}"] = round(r["renter_share_FI"], 4)
            row[f"rRL_{yr}"] = round(r["renter_share_RL"], 4)
        rows.append(row)
    return pd.DataFrame(rows)


def load_finance_cumulative(fin_dir: Path, label: str) -> pd.DataFrame:
    """Load finance tract data and compute cumulative totals."""
    p = fin_dir / "finance" / "finance_tract_all_years.csv"
    if not p.exists():
        # Try concatenating individual year files
        files = sorted((fin_dir / "finance").glob("finance_tract_2*.csv"))
        if not files:
            print(f"  [WARN] No finance files found in {fin_dir / 'finance'}")
            return pd.DataFrame()
        df = pd.concat([pd.read_csv(f, encoding="utf-8") for f in files], ignore_index=True)
    else:
        df = pd.read_csv(p, encoding="utf-8")

    # Compute cumulative totals per tract (in USD)
    agg = df.groupby("tract_geoid").agg(
        cum_damage_usd=("gross_total_usd", "sum"),
        cum_owner_damage_usd=("owner_gross_total_usd", "sum"),
        cum_renter_damage_usd=("renter_gross_total_usd", "sum"),
        cum_payout_usd=("payout_total_usd", "sum"),
        cum_oop_usd=("oop_total_usd", "sum"),
        cum_premium_usd=("premium_total_usd", "sum"),
        total_households=("households", "mean"),  # avg households across years
    ).reset_index()

    # Per-household damage
    agg["per_hh_damage_usd"] = agg["cum_damage_usd"] / agg["total_households"]

    # Rename with label prefix
    rename = {}
    for col in agg.columns:
        if col not in ("tract_geoid", "total_households"):
            rename[col] = f"{label}_{col}"
    agg = agg.rename(columns=rename)
    agg["tract_geoid"] = agg["tract_geoid"].astype(str)
    return agg


def compute_damage_reduction(baseline: pd.DataFrame, worst: pd.DataFrame) -> pd.DataFrame:
    """Compute damage reduction % = (worst - baseline) / worst."""
    b = baseline[["tract_geoid", "b_cum_damage_usd"]].copy()
    w = worst[["tract_geoid", "w_cum_damage_usd"]].copy()
    merged = pd.merge(b, w, on="tract_geoid", how="outer")
    merged["damage_reduction_pct"] = np.where(
        merged["w_cum_damage_usd"] > 0,
        (merged["w_cum_damage_usd"] - merged["b_cum_damage_usd"]) / merged["w_cum_damage_usd"] * 100,
        0.0
    )
    return merged[["tract_geoid", "damage_reduction_pct"]].round(2)


def compute_post_flood_change() -> pd.DataFrame:
    """
    Compute pre/post flood behavioral change (DELTAS) relative to the
    year before each major flood event.

    Event 2014: baseline = 2013, deltas for 2014, 2015, 2016
    Event 2021: baseline = 2020, deltas for 2021, 2022, 2023

    For each event:
      delta_YYYY = share(YYYY) - share(baseline_year)
    Actions: owner FI/EH/BP, renter FI/RL
    """
    p = BASE_DIR / "decisions" / "action_share_owner_renter_tract_all_years.csv"
    df = pd.read_csv(p, encoding="utf-8")
    df["tract_geoid"] = df["tract_geoid"].astype(str)

    action_cols = [
        ("oFI", "owner_share_FI"),
        ("oEH", "owner_share_EH"),
        ("oBP", "owner_share_BP"),
        ("rFI", "renter_share_FI"),
        ("rRL", "renter_share_RL"),
    ]

    # Event config: {event_year: (baseline_year, [years to compute delta for])}
    EVENT_CONFIG = {
        2014: (2013, [2014, 2015, 2016]),
        2021: (2020, [2021, 2022, 2023]),
    }

    rows = []
    for geoid, gdf in df.groupby("tract_geoid"):
        gdf = gdf.set_index("year")
        row = {"tract_geoid": geoid}

        for event_yr, (base_yr, delta_yrs) in EVENT_CONFIG.items():
            eYY = f"e{event_yr % 100:02d}"

            for prefix, share_col in action_cols:
                # Baseline value (pre-flood year)
                v_base = gdf.loc[base_yr, share_col] if base_yr in gdf.index else np.nan
                row[f"{prefix}_{eYY}_base{base_yr}"] = round(v_base, 4) if not np.isnan(v_base) else np.nan

                # Raw shares and deltas for each post-event year
                for yr in delta_yrs:
                    v = gdf.loc[yr, share_col] if yr in gdf.index else np.nan
                    row[f"{prefix}_{eYY}_{yr}"] = round(v, 4) if not np.isnan(v) else np.nan

                    if not np.isnan(v_base) and not np.isnan(v):
                        row[f"{prefix}_{eYY}_d{yr}"] = round(v - v_base, 4)
                    else:
                        row[f"{prefix}_{eYY}_d{yr}"] = np.nan

        rows.append(row)
    return pd.DataFrame(rows)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Generating Tract-Level Summary Excel")
    print("=" * 60)

    # 1. Flood characteristics
    print("\n[1/6] Loading flood configuration …")
    flood_df = load_flood_config()
    print(f"  {len(flood_df)} tracts loaded")

    # 2. Action shares
    print("[2/6] Loading action shares …")
    action_df = load_action_shares()
    print(f"  {len(action_df)} tracts × {len(YEARS)} years")

    # 3. Baseline finance
    print("[3/6] Loading baseline finance …")
    base_fin = load_finance_cumulative(BASE_DIR, "b")
    print(f"  {len(base_fin)} tracts")

    # 4. Worst-case finance
    print("[4/6] Loading worst-case (no-adaptation) finance …")
    worst_fin = load_finance_cumulative(WORST_DIR, "w")
    print(f"  {len(worst_fin)} tracts")

    # 5. Damage reduction
    print("[5/6] Computing damage reduction …")
    reduction = compute_damage_reduction(base_fin, worst_fin)

    # 6. Post-flood behavioral change
    print("[6/6] Computing post-flood behavioral change …")
    postflood = compute_post_flood_change()

    # ── Merge all ──
    print("\nMerging all data …")
    flood_df["tract_geoid"] = flood_df["tract_geoid"].astype(str)
    action_df["tract_geoid"] = action_df["tract_geoid"].astype(str)

    result = flood_df.copy()
    result = result.merge(action_df, on="tract_geoid", how="left")
    result = result.merge(base_fin, on="tract_geoid", how="left")
    result = result.merge(worst_fin, on="tract_geoid", how="left")
    result = result.merge(reduction, on="tract_geoid", how="left")
    result = result.merge(postflood, on="tract_geoid", how="left")

    # ── Write Excel with multiple sheets ──
    print(f"\nWriting to {OUT_FILE} …")
    with pd.ExcelWriter(OUT_FILE, engine="openpyxl") as writer:
        # Sheet 1: Full summary (all columns)
        result.to_excel(writer, sheet_name="Summary", index=False)

        # Sheet 2: Action shares only (long format for easy plotting)
        p = BASE_DIR / "decisions" / "action_share_owner_renter_tract_all_years.csv"
        act_long = pd.read_csv(p, encoding="utf-8")
        act_long.to_excel(writer, sheet_name="ActionShares_Long", index=False)

        # Sheet 3: Finance baseline (long format)
        p = BASE_DIR / "finance" / "finance_tract_all_years.csv"
        fin_base_long = pd.read_csv(p, encoding="utf-8")
        fin_base_long.to_excel(writer, sheet_name="Finance_Baseline", index=False)

        # Sheet 4: Finance worst-case (long format)
        p = WORST_DIR / "finance" / "finance_tract_all_years.csv"
        if p.exists():
            fin_worst_long = pd.read_csv(p, encoding="utf-8")
            fin_worst_long.to_excel(writer, sheet_name="Finance_NoAdaptation", index=False)

        # Sheet 5: Post-flood change
        postflood.to_excel(writer, sheet_name="PostFlood_Change", index=False)

    print(f"\nDone! Output: {OUT_FILE}")
    print(f"  Sheets: Summary, ActionShares_Long, Finance_Baseline, Finance_NoAdaptation, PostFlood_Change")
    print(f"  Tracts: {len(result)}")
    print(f"  Columns in Summary: {len(result.columns)}")


if __name__ == "__main__":
    main()
