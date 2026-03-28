# -*- coding: utf-8 -*-
"""
Generate Household-Level Initial Conditions
=============================================

Adds to config/households_for_abm.csv:
  - TP_init, CP_init, SP_init, SC_init, PA_init  (Beta-distributed psych)
  - has_FI  (flood insurance flag, from YAML tract×group rates)

Each household gets an independent draw based on its group (owner/renter).
The output CSV is the fixed initial-condition file for the simulation.

Usage:
    python generate_household_psych.py [--seed 2025] [--csv path/to/households.csv]
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

from modules.actions.tp import (
    BETA_PARAMS_OWNER_DEFAULT,
    BETA_PARAMS_RENTER_DEFAULT,
)

PSYCH_COLS = ["TP", "CP", "SP", "SC", "PA"]


def _assign_has_fi(
    df: pd.DataFrame,
    insurance_cfg: dict,
    seed: int = 2025,
) -> pd.DataFrame:
    """Assign has_FI column based on per-tract-group take rates.

    Args:
        df: Household DataFrame with tract_geoid, group columns
        insurance_cfg: insurance_init section from abm_params.yaml
        seed: Random seed

    Returns:
        DataFrame with has_FI column added
    """
    rng = np.random.default_rng(int(insurance_cfg.get("seed", seed)))
    df["has_FI"] = 0

    rates = insurance_cfg.get("take_rate_by_tract_group", {})
    if not rates:
        print("[WARN] No take_rate_by_tract_group in config — all has_FI=0")
        return df

    tract_col = df["tract_geoid"].astype(str)
    group_col = df["group"].astype(str).str.lower()

    for tract_str, group_rates in rates.items():
        tract_str = str(tract_str).zfill(11)
        if not isinstance(group_rates, dict):
            continue
        for grp, rate in group_rates.items():
            rate = float(rate)
            if rate <= 0:
                continue
            mask = tract_col.eq(tract_str) & group_col.eq(grp)
            idx = df.index[mask].to_numpy()
            n = len(idx)
            if n == 0:
                continue
            k = max(0, min(int(round(rate * n)), n))
            if k > 0:
                chosen = rng.choice(idx, size=k, replace=False)
                df.loc[chosen, "has_FI"] = 1

    total = int(df["has_FI"].sum())
    n_own = int(df.loc[group_col.eq("owner"), "has_FI"].sum())
    n_ren = int(df.loc[group_col.eq("renter"), "has_FI"].sum())
    print(f"  has_FI: {total:,} total ({n_own:,} owner + {n_ren:,} renter)")
    return df


def generate_initial_conditions(
    csv_path: Path,
    yaml_path: Path | None = None,
    seed: int = 2025,
    beta_owner: dict | None = None,
    beta_renter: dict | None = None,
) -> pd.DataFrame:
    """Read households CSV, add psych + has_FI columns, write back.

    Args:
        csv_path: Path to households_for_abm.csv
        yaml_path: Path to abm_params.yaml (for insurance rates)
        seed: Random seed for reproducibility
        beta_owner: Beta(a,b) params per variable for owners
        beta_renter: Beta(a,b) params per variable for renters

    Returns:
        Updated DataFrame
    """
    beta_owner = beta_owner or BETA_PARAMS_OWNER_DEFAULT
    beta_renter = beta_renter or BETA_PARAMS_RENTER_DEFAULT

    df = pd.read_csv(csv_path, dtype={"tract_geoid": "string"}, encoding="utf-8")
    df["group"] = df["group"].astype(str).str.lower()

    rng = np.random.RandomState(seed)

    is_owner = df["group"].eq("owner")
    n = len(df)

    # --- Psychology columns ---
    for var in PSYCH_COLS:
        col = f"{var}_init"
        vals = np.empty(n, dtype=np.float64)

        a_o, b_o = beta_owner.get(var, (2.0, 2.0))
        n_own = is_owner.sum()
        if n_own > 0:
            vals[is_owner.values] = rng.beta(max(a_o, 1e-6), max(b_o, 1e-6), size=n_own)

        a_r, b_r = beta_renter.get(var, (2.0, 2.0))
        n_ren = (~is_owner).sum()
        if n_ren > 0:
            vals[~is_owner.values] = rng.beta(max(a_r, 1e-6), max(b_r, 1e-6), size=n_ren)

        df[col] = vals

    # --- Insurance column ---
    ins_cfg = {}
    if yaml_path and yaml_path.exists():
        cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        ins_cfg = cfg.get("insurance_init", {}) or {}
    df = _assign_has_fi(df, ins_cfg, seed=seed)

    # --- Write back ---
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"[OK] Wrote {len(df):,} rows × {len(df.columns)} cols to {csv_path}")
    for var in PSYCH_COLS:
        col = f"{var}_init"
        print(f"  {col}: mean={df[col].mean():.4f}  std={df[col].std():.4f}")
    return df


def main():
    ap = argparse.ArgumentParser(description="Generate household initial conditions")
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--csv", type=str, default=None)
    ap.add_argument("--yaml", type=str, default=None)
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent.parent  # project root
    csv_path = Path(args.csv) if args.csv else root / "config" / "households_for_abm.csv"
    yaml_path = Path(args.yaml) if args.yaml else root / "config" / "abm_params.yaml"
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    generate_initial_conditions(csv_path, yaml_path=yaml_path, seed=args.seed)


if __name__ == "__main__":
    main()
