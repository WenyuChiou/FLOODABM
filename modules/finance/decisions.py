"""
Finance Decisions - Decision file loading and gating logic.

This module handles:
- Searching for decision files on disk (ENV vars, pattern matching)
- Standardizing decision DataFrames (has_FI, identity_dec)
- Applying decision gating (uninsured => payout=0)
"""

from __future__ import annotations

import os
from pathlib import Path as _Path
from typing import List, Optional

import numpy as np
import pandas as pd

__all__ = [
    "_infer_year_from_df",
    "_search_decision_files",
    "_standardize_decisions",
    "_apply_decisions_gating",
]

_DEFAULT_DECISION_PATTERNS = [
    "decision_mgmix_{year}.csv",
    "decisions_mgmix_{year}.csv",
    "decision_{year}.csv",
    "decisions_{year}.csv",
]


def _infer_year_from_df(_df: pd.DataFrame) -> Optional[int]:
    """Try to infer simulation year from DataFrame columns."""
    for key in ["year", "Year", "YR"]:
        if key in _df.columns:
            vals = pd.to_numeric(_df[key], errors="coerce").dropna().unique()
            if len(vals) == 1:
                try:
                    return int(vals[0])
                except Exception:
                    pass
    return None


def _search_decision_files(year: Optional[int]) -> List[_Path]:
    """
    Search for decision CSV files on disk.
    
    Priority:
      1) ABM_DECISION_FILE env var (exact file)
      2) ABM_DECISION_DIR / pattern matching
      3) Default search locations with year substitution
    """
    env_file = os.environ.get("ABM_DECISION_FILE", "").strip()
    if env_file and _Path(env_file).is_file():
        return [_Path(env_file)]

    mod_dir = _Path(__file__).resolve()
    modules_root = mod_dir.parents[1]  # .../modules
    roots = [
        _Path(os.environ.get("ABM_DECISION_DIR", "")).resolve()
        if os.environ.get("ABM_DECISION_DIR") else None,
        _Path.cwd() / "tests" / "decisions",
        modules_root / "actions" / "outputs",
        modules_root / "actions" / "tests",
        modules_root / "tests" / "decisions",
        modules_root / "actions",
        _Path.cwd() / "tests",
    ]
    roots = [r for r in roots if r and r.exists()]

    cand: List[_Path] = []
    if year is not None:
        patterns = (
            os.environ.get("ABM_DECISION_PATTERN", "").split(";")
            if os.environ.get("ABM_DECISION_PATTERN")
            else _DEFAULT_DECISION_PATTERNS
        )
        for root in roots:
            for pat in patterns:
                p = root / pat.format(year=year)
                if p.exists():
                    cand.append(p)
    else:
        globs = [
            "decision_mgmix_*.csv",
            "decisions_mgmix_*.csv",
            "decision_*.csv",
            "decisions_*.csv",
        ]
        for root in roots:
            for g in globs:
                cand.extend(root.glob(g))
        cand.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)

    # de-dup
    seen, out = set(), []
    for p in cand:
        s = str(p.resolve())
        if s not in seen:
            seen.add(s)
            out.append(p)
    return out


def _standardize_decisions(
    dec: pd.DataFrame,
    idxer: Optional[pd.DataFrame] = None,
    expected_len: Optional[int] = None,
) -> pd.DataFrame:
    """
    Standardize decision DataFrame to have 'i', 'has_FI', 'identity_dec' columns.
    
    - Attach household id "i" from various possible column names
    - Derive has_FI from various flag columns or action/policy columns
    - Extract identity override if present
    """
    d = dec.copy()

    # Attach household id "i"
    id_col = None
    for c in ["i", "household_id", "hh_id", "id"]:
        if c in d.columns:
            id_col = c
            break

    if id_col is None:
        # try "row" aligned with idxer
        row_col = None
        for c in ["row", "idx", "index", "row_id"]:
            if c in d.columns:
                row_col = c
                break
        if row_col is not None and isinstance(idxer, (pd.DataFrame, pd.Series)):
            if isinstance(idxer, pd.Series) and idxer.name != "i":
                ix = idxer.rename("i").reset_index(drop=True)
            elif isinstance(idxer, pd.Series):
                ix = idxer.reset_index(drop=True)
            else:
                if "i" in idxer.columns:
                    ix = idxer["i"].reset_index(drop=True)
                else:
                    ix = idxer.reset_index(drop=True).iloc[:, 0].rename("i")
            d = d.merge(
                pd.DataFrame({"row": np.arange(len(ix)), "i": ix}),
                left_on=row_col,
                right_on="row",
                how="left",
            )
            id_col = "i"
        elif (
            expected_len is not None
            and expected_len == len(d)
            and isinstance(idxer, (pd.Series, pd.DataFrame))
        ):
            # fallback: position-wise attach "i"
            if isinstance(idxer, pd.Series):
                i_series = idxer.rename("i").reset_index(drop=True)
            else:
                i_series = (
                    idxer["i"]
                    if "i" in idxer.columns
                    else idxer.reset_index(drop=True).iloc[:, 0]
                ).rename("i")
            d = d.reset_index(drop=True)
            d["i"] = i_series.values
            id_col = "i"

    if id_col and id_col != "i":
        d.rename(columns={id_col: "i"}, inplace=True)

    # has_FI
    fi_col = None
    for c in ["has_FI", "FI", "fi", "buy_insurance", "has_insurance"]:
        if c in d.columns:
            fi_col = c
            break
    if fi_col:
        d["has_FI"] = (pd.to_numeric(d[fi_col], errors="coerce").fillna(0.0) > 0).astype(int)
    else:
        if "action" in d.columns:
            d["has_FI"] = (d["action"].astype(str).str.upper().eq("FI")).astype(int)
        elif "POLICY_NAME" in d.columns:
            d["has_FI"] = d["POLICY_NAME"].astype(str).str.strip().ne("").astype(int)
        else:
            d["has_FI"] = 0

    # identity override
    ident_col = None
    for c in ["identity", "group", "status"]:
        if c in d.columns:
            ident_col = c
            break
    if ident_col:
        d["identity_dec"] = (
            d[ident_col]
            .astype(str)
            .str.lower()
            .str.strip()
            .map({"owner": "owner", "renter": "renter"})
            .fillna("")
        )
    else:
        d["identity_dec"] = ""

    keep = ["has_FI", "identity_dec"]
    if "i" in d.columns:
        keep = ["i"] + keep
    return d[keep]


def _apply_decisions_gating(
    out: pd.DataFrame,
    dec: Optional[pd.DataFrame],
    idxer: Optional[pd.DataFrame],
    year: Optional[int],
) -> pd.DataFrame:
    """
    Apply decision gating: uninsured households (has_FI=0) get payout=0.
    
    - Auto-searches for decision files if dec is None
    - Merges decisions by 'i' column
    - Zeros out payout columns for uninsured
    - Recomputes OOP after gating
    """
    if dec is None and "has_FI" not in out.columns:
        # try auto-search on disk
        cand = _search_decision_files(year)
        dec_df = None
        if len(cand) == 1:
            dec_df = pd.read_csv(cand[0])
        elif len(cand) > 1 and year is not None:
            for p in cand:
                if str(year) in p.stem:
                    dec_df = pd.read_csv(p)
                    break
        if dec_df is not None:
            dec = dec_df

    if dec is None:
        return out  # nothing to do

    std = _standardize_decisions(dec, idxer=idxer, expected_len=len(out))

    # Merge by 'i' when available; otherwise align by index length
    if "i" in std.columns and "i" in out.columns:
        merged = out.merge(std, on="i", how="left", suffixes=("", "_dec"))
    elif "i" in std.columns and "i" not in out.columns:
        merged = out.copy()
        merged["i"] = std["i"].values if len(std) == len(out) else np.nan
        merged = merged.merge(std, on="i", how="left", suffixes=("", "_dec"))
    else:
        merged = out.copy().reset_index(drop=True)
        std = std.reset_index(drop=True)
        for col in std.columns:
            merged[col] = std[col]

    # Unify 0/1
    merged["has_FI"] = pd.to_numeric(merged.get("has_FI", 0), errors="coerce").fillna(0).astype(int)

    # Decision priority: override with _dec if present
    if "has_FI_dec" in merged.columns:
        merged["has_FI"] = pd.to_numeric(merged["has_FI_dec"], errors="coerce").fillna(0).astype(int)
        merged = merged.drop(columns=["has_FI_dec"])
    elif "is_FI_dec" in merged.columns:
        merged["has_FI"] = pd.to_numeric(merged["is_FI_dec"], errors="coerce").fillna(0).astype(int)
        merged = merged.drop(columns=["is_FI_dec"])

    # optional identity override
    if "identity_dec" in merged.columns:
        mask = merged["identity_dec"].astype(str).str.lower().isin(["owner", "renter"])
        if "identity" in merged.columns:
            merged.loc[mask, "identity"] = merged.loc[mask, "identity_dec"].astype(str).str.lower()

    # gating: uninsured => payouts = 0
    m0 = merged["has_FI"].eq(0)

    _payout_cols = [
        "payout_structure", "payout_contents", "payout_total",
        "payout_structure_kUSD", "payout_contents_kUSD", "payout_total_kUSD",
    ]
    for c in _payout_cols:
        if c in merged.columns:
            merged.loc[m0, c] = 0.0

    # OOP = loss - payout
    if {"gross_structure_loss", "payout_structure"}.issubset(merged.columns):
        merged["oop_structure"] = (merged["gross_structure_loss"] - merged["payout_structure"]).clip(lower=0.0)
    if {"gross_contents_loss", "payout_contents"}.issubset(merged.columns):
        merged["oop_contents"] = (merged["gross_contents_loss"] - merged["payout_contents"]).clip(lower=0.0)
    if {"oop_structure", "oop_contents"}.issubset(merged.columns):
        merged["oop_total"] = merged["oop_structure"] + merged["oop_contents"]

    # gate OOP columns too
    _oop_cols = [
        "oop_structure", "oop_contents", "oop_total",
        "oop_structure_kUSD", "oop_contents_kUSD", "oop_total_kUSD",
    ]
    for c in _oop_cols:
        if c in merged.columns:
            merged.loc[m0, c] = 0.0

    # has_claim = insured AND has payout
    _payout_total = "payout_total_kUSD" if "payout_total_kUSD" in merged.columns else "payout_total"
    if _payout_total in merged.columns:
        merged["has_claim"] = ((merged["has_FI"] == 1) & (merged[_payout_total] > 0)).astype(int)

    return merged
