"""Helper functions for main.py year-loop operations."""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from utils._helpers import (
    _standardize_has_FI_from_dec,
    _action_share_owner_renter_by_tract,
    finalize_has_fi,
)


def build_flood_bookkeeping_rows(
    year: int, TRACTS: list[str], ratio_raw: dict, ratio_used: dict, dmap: dict,
    THR_OWNER: float, THR_RENTER: float, DEPTH_THRESHOLD_M: float | None
) -> list[dict]:
    """Build flood status rows for each tract in this year."""
    rows = []
    for t in sorted(set(TRACTS) | set(map(str, ratio_raw.keys()))):
        r = float(ratio_raw.get(t, 0.0))
        ru = float(ratio_used.get(t, 0.0))
        dep = float(dmap.get(t, np.nan))
        is_shock = (ru > 0.0)
        is_flood_owner = (ru >= THR_OWNER)
        is_flood_renter = (ru >= THR_RENTER)
        is_flood_any = is_flood_owner or is_flood_renter
        if DEPTH_THRESHOLD_M is not None and np.isfinite(dep):
            is_flood_owner = is_flood_owner and (dep >= float(DEPTH_THRESHOLD_M))
            is_flood_renter = is_flood_renter and (dep >= float(DEPTH_THRESHOLD_M))
            is_flood_any = is_flood_owner or is_flood_renter
        rows.append({
            "year": year, "tract_geoid": t, "ratio": r, "ratio_used": ru,
            "depth_m": dep, "is_shock": is_shock,
            "is_flood_owner": is_flood_owner, "is_flood_renter": is_flood_renter, "is_flood_any": is_flood_any,
        })
    return rows


def save_decisions_and_update_prev(
    dec: pd.DataFrame, STATE: pd.DataFrame, year: int, DEC_DIR: Path
) -> pd.DataFrame:
    """Save decisions CSV and return standardized dec_prev for next year."""
    if {"i","i_orig","hh_idx_within_group"}.issubset(STATE.columns):
        dec = dec.merge(STATE[["i","i_orig","hh_idx_within_group"]], on="i", how="left")
    dec.to_csv(DEC_DIR / f"decisions_mgmix_{year}.csv", index=False, encoding="utf-8-sig")
    return _standardize_has_FI_from_dec(dec).sort_values("i").reset_index(drop=True)


def save_action_shares(
    dec: pd.DataFrame, STATE_NEXT: pd.DataFrame, year: int, DEC_DIR: Path
) -> None:
    """Compute and save per-tract action shares + EH coverage."""
    split_tbl = _action_share_owner_renter_by_tract(dec)
    st_next = STATE_NEXT.copy()
    st_next["tract_geoid"] = st_next["tract_geoid"].astype(str)
    st_next["eh_done"] = st_next.get("eh_done", False)
    owners = st_next[st_next["group"].str.lower().eq("owner")].copy()
    cov = (
        owners.groupby("tract_geoid", dropna=False)["eh_done"].agg(
            eh_covered=lambda s: s.astype(bool).sum(), n_owner="size"
        ).reset_index()
    )
    cov["eh_coverage_owner_cum"] = (cov["eh_covered"] / cov["n_owner"]).fillna(0.0)
    split_tbl = split_tbl.merge(cov[["tract_geoid","eh_coverage_owner_cum"]], on="tract_geoid", how="left")
    split_tbl["eh_coverage_owner_cum"] = split_tbl["eh_coverage_owner_cum"].fillna(0.0)
    split_tbl.to_csv(DEC_DIR / f"action_share_owner_renter_tract_{year}.csv", index=False, encoding="utf-8-sig")


def append_tp_trajectory(
    psy_df: pd.DataFrame, year: int, tp_rows: list[dict]
) -> None:
    """Append TP trajectory records for this year."""
    tprec = psy_df[["tract_geoid","TP_owner","TP_renter"]].copy()
    tprec["year"] = year
    tprec["phase"] = "after"
    tp_rows.extend(tprec[["year","tract_geoid","TP_owner","TP_renter","phase"]].to_dict("records"))


def advance_state_for_next_year(
    STATE: pd.DataFrame, STATE_NEXT: pd.DataFrame, year: int,
    OWNER_SHARE: dict | None = None,
) -> pd.DataFrame:
    """Mark backfill households and prepare state for next year.

    Backfill households (from RL/BP dynamics) get default psych values
    sampled from their group's Beta distributions.
    """
    prev_ids = set(STATE["i"].astype(int))
    STATE = STATE_NEXT.copy()
    STATE["tract_geoid"] = STATE["tract_geoid"].astype(str)
    if "elev_ft_total" not in STATE.columns:
        STATE["elev_ft_total"] = 0.0
    STATE["elev_ft_total"] = pd.to_numeric(STATE["elev_ft_total"], errors="coerce").fillna(0.0)
    new_mask = ~STATE["i"].astype(int).isin(prev_ids)
    if "is_backfill" not in STATE.columns:
        STATE["is_backfill"] = False
    if "year_created" not in STATE.columns:
        STATE["year_created"] = np.nan
    STATE.loc[new_mask, "is_backfill"] = True
    STATE.loc[new_mask, "year_created"] = year
    STATE.loc[new_mask, "elev_ft_total"] = 0.0

    # Initialize psych columns for backfill households
    if new_mask.any():
        from modules.actions.tp import (
            BETA_PARAMS_OWNER_DEFAULT, BETA_PARAMS_RENTER_DEFAULT,
        )
        rng_bf = np.random.RandomState(int(year) * 1000 + new_mask.sum())
        new_idx = STATE.index[new_mask]
        new_grp = STATE.loc[new_idx, "group"].astype(str).str.lower()
        for var in ["TP", "CP", "SP", "SC", "PA"]:
            init_col = f"{var}_init"
            if init_col not in STATE.columns:
                continue
            for grp_name, beta_dict in [("owner", BETA_PARAMS_OWNER_DEFAULT),
                                         ("renter", BETA_PARAMS_RENTER_DEFAULT)]:
                grp_mask = new_grp.eq(grp_name)
                n_grp = grp_mask.sum()
                if n_grp > 0:
                    a, b = beta_dict.get(var, (2.0, 2.0))
                    STATE.loc[new_idx[grp_mask], init_col] = rng_bf.beta(
                        max(a, 1e-6), max(b, 1e-6), size=n_grp
                    )
        # Set live TP = TP_init for backfill; t_clock = 0
        if "TP" in STATE.columns and "TP_init" in STATE.columns:
            STATE.loc[new_idx, "TP"] = STATE.loc[new_idx, "TP_init"].astype(float)
        if "t_clock" in STATE.columns:
            STATE.loc[new_idx, "t_clock"] = 0.0

    return STATE


def save_next_year_state(
    STATE: pd.DataFrame, year: int, dec_prev: pd.DataFrame, STATES_DIR: Path
) -> None:
    """Save next-year state snapshot with sticky FI."""
    STATE_OUT = STATE.copy()
    STATE_OUT["tract_geoid"] = STATE_OUT["tract_geoid"].astype(str)
    STATE_OUT.insert(1, "year", year + 1)
    STATE_OUT = finalize_has_fi(STATE_OUT, dec_prev=dec_prev, sticky=False)
    STATE_OUT.to_csv(STATES_DIR / f"state_next_{year+1}.csv", index=False, encoding="utf-8-sig")
    print(f"[OK] Final state (year {year+1}) saved to: {STATES_DIR / f'state_next_{year+1}.csv'}")


def handle_no_action_year(
    STATE: pd.DataFrame, year: int, DEC_DIR: Path, tp_rows: list[dict], TRACT_PSY: pd.DataFrame | None
) -> pd.DataFrame:
    """Create placeholder decisions/splits for NO_ACTION mode and return dec_prev."""
    dec = pd.DataFrame({
        "i": STATE["i"].astype(int),
        "tract_geoid": STATE["tract_geoid"].astype(str),
        "group": STATE["group"],
        "action": "None",
        "has_FI": pd.to_numeric(STATE.get("has_FI", 0), errors="coerce").fillna(0).astype(int),
    })
    dec.to_csv(DEC_DIR / f"decisions_mgmix_{year}.csv", index=False, encoding="utf-8-sig")

    split_tbl = pd.DataFrame({"tract_geoid": sorted(set(STATE["tract_geoid"].astype(str)))})
    for col in [
        "owner_share_FI","owner_share_EH","owner_share_BP","owner_share_DN",
        "renter_share_FI","renter_share_RL","renter_share_DN","eh_coverage_owner_cum",
    ]:
        split_tbl[col] = 0.0
    split_tbl.to_csv(DEC_DIR / f"action_share_owner_renter_tract_{year}.csv", index=False, encoding="utf-8-sig")

    if TRACT_PSY is not None:
        append_tp_trajectory(TRACT_PSY, year, tp_rows)

    return dec[["i","has_FI"]].sort_values("i").reset_index(drop=True)
