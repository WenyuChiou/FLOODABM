"""
Financial module for applying insurance policy terms to household-level flood gross losses,
with optional **decision gating** (only households that bought insurance receive payouts).

- Units: all monetary values are in kUSD (thousands of USD).
- Identity gating:
    * owner  -> only structure losses are insurable (payout_structure applies)
    * renter -> only contents  losses are insurable (payout_contents  applies)
    * unknown -> if both gross columns exist, both are computed; else whichever exists.

- Decision gating (NEW):
    If per-year decisions are provided (via `dec` argument) or discoverable, we set payouts to 0
    for uninsured households and recompute OOP. We also expose insured headcounts in aggregation.

Defaults (when policy columns are missing):
    - limit_*_kUSD = +inf
    - deductible_*_kUSD = 0

Formulas (per coverage):
    payout = min(max(gross - deductible, 0), limit)

Policy fills precedence (from lowest to highest; later overrides earlier when `policy_overwrite=True`):
  (1) global defaults (limit=inf, deductible=0)
  (2) scalar constants in `policy` (e.g., {"limit_structure_kUSD": 250})
  (3) identity-level fills in `policy`:
        {"owner": {"limit_structure_kUSD": 250, "deductible_structure_kUSD": 1},
         "renter":{"limit_contents_kUSD":  50, "deductible_contents_kUSD": 0.5}}
  (4) keyed mapping in `policy`:
        {"by": "household_id", "data": {"H001": {"limit_structure_kUSD": 200}}}
     If `by` is absent, we auto-pick the first available among
        ["household_id","hh_id","CensusTract","tract_geoid","GEOID"].
By default we **fill only missing values**. Set `policy_overwrite=True` to override existing ones.

Decision gating inputs (choose ONE; priority order):
  A) Direct `dec` DataFrame argument (recommended). Expected columns:
       - id column: "i" (preferred) or one of ["household_id","hh_id","id","row"]
       - action or flag: "has_FI" (0/1), or derive as (action == "FI") or (POLICY_NAME != "")
       - optional identity override: "identity" / "group" → owner/renter
     If `idxer` is provided and `dec` lacks "i", we will map rows to "i" via idxer.
  B) Auto-search a CSV on disk if `dec` is None (see _search_decision_files).

Key APIs:
    - apply_financials(df, *, colmap=None, policy=None, policy_overwrite=False,
                       dec=None, idxer=None, year=None, gate_by_decisions=True) -> pd.DataFrame
    - aggregate_by_tract(df, tract_col=None) -> pd.DataFrame
"""

from __future__ import annotations

import os
from pathlib import Path as _Path
from typing import Dict, Optional, Any, List, Tuple

import numpy as np
import pandas as pd
import sys

__all__ = ["apply_financials", "aggregate_by_tract", "detect_tract_column"]

KUSD_EPS = 1e-9  # small epsilon to guard floating round-off in kUSD


def _import_vuln(modules_root: Path):
    vul_root = modules_root / "vulnerability"
    sys.path.insert(0, str(vul_root))
    from vulnerability import Vulnerability  # type: ignore
    return Vulnerability

def _safe_series(df: pd.DataFrame, name: str, default: float) -> pd.Series:
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce").astype(float)
    return pd.Series([np.nan] * len(df), index=df.index, dtype=float)


def _payout(gross: pd.Series, limit_: pd.Series, ded: pd.Series) -> pd.Series:
    base = (gross - ded).clip(lower=0.0)
    return np.minimum(base, limit_)


def _apply_policy_fills(
    out: pd.DataFrame,
    identity: pd.Series,
    policy: Dict[str, Any],
    policy_overwrite: bool,
    colnames: Dict[str, str],
) -> None:
    """Fill/override policy columns from dict following precedence."""
    # Ensure columns exist
    for key in ["limit_structure_kUSD","deductible_structure_kUSD",
                "limit_contents_kUSD","deductible_contents_kUSD"]:
        col = colnames[key]
        if col not in out.columns:
            out[col] = np.nan

    def _set_where(col: str, values: pd.Series, mask: Optional[pd.Series]=None):
        if mask is None:
            mask = pd.Series([True]*len(out), index=out.index)
        if policy_overwrite:
            out.loc[mask, col] = values.loc[mask]
        else:
            to_fill = mask & out[col].isna()
            out.loc[to_fill, col] = values.loc[to_fill]

    # (2) scalar constants
    scalar_keys = ["limit_structure_kUSD","deductible_structure_kUSD",
                   "limit_contents_kUSD","deductible_contents_kUSD"]
    for sk in scalar_keys:
        if sk in policy and isinstance(policy[sk], (int, float)):
            val = float(policy[sk])
            _set_where(colnames[sk], pd.Series(val, index=out.index))

    # (3) identity-level fills
    for id_key in ["owner","renter","unknown"]:
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
            for cand in ["household_id","hh_id","CensusTract","tract_geoid","GEOID"]:
                if cand in out.columns:
                    by = cand
                    break
        if by and by in out.columns:
            pdata = policy["data"]
            series_by = out[by].astype(str)
            # Build per-row settings
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


# -------- Decisions (NEW) --------

_DEFAULT_DECISION_PATTERNS = [
    "decision_mgmix_{year}.csv",
    "decisions_mgmix_{year}.csv",
    "decision_{year}.csv",
    "decisions_{year}.csv",
]

def _infer_year_from_df(_df: pd.DataFrame) -> Optional[int]:
    for key in ["year", "Year", "YR"]:
        if key in _df.columns:
            vals = pd.to_numeric(_df[key], errors="coerce").dropna().unique()
            if len(vals) == 1:
                try: return int(vals[0])
                except Exception: pass
    return None

def _search_decision_files(year: Optional[int]) -> List[_Path]:
    env_file = os.environ.get("ABM_DECISION_FILE", "").strip()
    if env_file and _Path(env_file).is_file():
        return [_Path(env_file)]

    mod_dir = _Path(__file__).resolve()
    modules_root = mod_dir.parents[1]  # .../modules
    roots = [
        _Path(os.environ.get("ABM_DECISION_DIR","")).resolve()
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
        patterns = os.environ.get("ABM_DECISION_PATTERN","").split(";") \
                   if os.environ.get("ABM_DECISION_PATTERN") else _DEFAULT_DECISION_PATTERNS
        for root in roots:
            for pat in patterns:
                p = root / pat.format(year=year)
                if p.exists():
                    cand.append(p)
    else:
        globs = ["decision_mgmix_*.csv","decisions_mgmix_*.csv",
                 "decision_*.csv","decisions_*.csv"]
        for root in roots:
            for g in globs:
                cand.extend(root.glob(g))
        cand.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)

    # de-dup
    seen, out = set(), []
    for p in cand:
        s = str(p.resolve())
        if s not in seen:
            seen.add(s); out.append(p)
    return out

def _standardize_decisions(dec: pd.DataFrame,
                           idxer: Optional[pd.DataFrame] = None,
                           expected_len: Optional[int] = None) -> pd.DataFrame:
    d = dec.copy()

    # Attach household id "i"
    id_col = None
    for c in ["i","household_id","hh_id","id"]:
        if c in d.columns:
            id_col = c; break

    if id_col is None:
        # try "row" aligned with idxer (row → i)
        row_col = None
        for c in ["row","idx","index","row_id"]:
            if c in d.columns:
                row_col = c; break
        if row_col is not None and isinstance(idxer, (pd.DataFrame, pd.Series)):
            if isinstance(idxer, pd.Series) and idxer.name != "i":
                ix = idxer.rename("i").reset_index(drop=True)
            elif isinstance(idxer, pd.Series):
                ix = idxer.reset_index(drop=True)
            else:
                # DataFrame
                if "i" in idxer.columns:
                    ix = idxer["i"].reset_index(drop=True)
                else:
                    ix = idxer.reset_index(drop=True).iloc[:,0].rename("i")
            d = d.merge(pd.DataFrame({"row": np.arange(len(ix)), "i": ix}),
                        left_on=row_col, right_on="row", how="left")
            id_col = "i"
        elif expected_len is not None and expected_len == len(d) and isinstance(idxer, (pd.Series, pd.DataFrame)):
            # fallback: position-wise attach "i"
            if isinstance(idxer, pd.Series):
                i_series = idxer.rename("i").reset_index(drop=True)
            else:
                i_series = (idxer["i"] if "i" in idxer.columns
                            else idxer.reset_index(drop=True).iloc[:,0]).rename("i")
            d = d.reset_index(drop=True)
            d["i"] = i_series.values
            id_col = "i"
        else:
            # As last resort, keep without id; will align by index later
            pass

    if id_col and id_col != "i":
        d.rename(columns={id_col: "i"}, inplace=True)

    # has_FI
    fi_col = None
    for c in ["has_FI","FI","fi","buy_insurance","has_insurance"]:
        if c in d.columns:
            fi_col = c; break
    if fi_col:
        d["has_FI"] = (pd.to_numeric(d[fi_col], errors="coerce").fillna(0.0) > 0).astype(int)
    else:
        # derive from action / POLICY_NAME if present
        if "action" in d.columns:
            d["has_FI"] = (d["action"].astype(str).str.upper().eq("FI")).astype(int)
        elif "POLICY_NAME" in d.columns:
            d["has_FI"] = d["POLICY_NAME"].astype(str).str.strip().ne("").astype(int)
        else:
            d["has_FI"] = 0

    # identity override
    ident_col = None
    for c in ["identity","group","status"]:
        if c in d.columns:
            ident_col = c; break
    if ident_col:
        d["identity_dec"] = (d[ident_col].astype(str).str.lower().str.strip()
                             .map({"owner":"owner","renter":"renter"}).fillna(""))
    else:
        d["identity_dec"] = ""

    keep = ["has_FI","identity_dec"]
    if "i" in d.columns:
        keep = ["i"] + keep
    return d[keep]

def _apply_decisions_gating(out: pd.DataFrame,
                            dec: Optional[pd.DataFrame],
                            idxer: Optional[pd.DataFrame],
                            year: Optional[int]) -> pd.DataFrame:
    if dec is None and "has_FI" not in out.columns:
        # try auto-search on disk
        cand = _search_decision_files(year)
        dec_df = None
        if len(cand) == 1:
            dec_df = pd.read_csv(cand[0])
        elif len(cand) > 1 and year is not None:
            for p in cand:
                if str(year) in p.stem:
                    dec_df = pd.read_csv(p); break
        if dec_df is not None:
            dec = dec_df

    if dec is None:
        return out  # nothing to do

    std = _standardize_decisions(dec, idxer=idxer, expected_len=len(out))

    # Merge by 'i' when available; otherwise align by index length
    if "i" in std.columns and "i" in out.columns:
        merged = out.merge(std, on="i", how="left", suffixes=("","_dec"))
    elif "i" in std.columns and "i" not in out.columns:
        merged = out.copy()
        merged["i"] = std["i"].values if len(std) == len(out) else np.nan
        merged = merged.merge(std, on="i", how="left", suffixes=("","_dec"))
    else:
        merged = out.copy().reset_index(drop=True)
        std = std.reset_index(drop=True)
        for col in std.columns:
            merged[col] = std[col]

    # 統一 0/1
    merged["has_FI"] = pd.to_numeric(merged.get("has_FI", 0), errors="coerce").fillna(0).astype(int)

    # ★ 決策優先（覆蓋），不要 OR
    if "has_FI_dec" in merged.columns:
        merged["has_FI"] = pd.to_numeric(merged["has_FI_dec"], errors="coerce").fillna(0).astype(int)
        merged = merged.drop(columns=["has_FI_dec"])
    elif "is_FI_dec" in merged.columns:
        merged["has_FI"] = pd.to_numeric(merged["is_FI_dec"], errors="coerce").fillna(0).astype(int)
        merged = merged.drop(columns=["is_FI_dec"])
    # else: 沒有決策欄位，就維持 merged["has_FI"] 目前的 0/1



    # optional identity override
    if "identity_dec" in merged.columns:
        mask = merged["identity_dec"].astype(str).str.lower().isin(["owner","renter"])
        if "identity" in merged.columns:
            merged.loc[mask, "identity"] = merged.loc[mask, "identity_dec"].astype(str).str.lower()

    # gating
    m0 = merged["has_FI"].eq(0)

    # ★ Support both with and without _kUSD suffix
    _payout_cols = [
        "payout_structure", "payout_contents", "payout_total",
        "payout_structure_kUSD", "payout_contents_kUSD", "payout_total_kUSD"
    ]
    for c in _payout_cols:
        if c in merged.columns:
            merged.loc[m0, c] = 0.0

    # OOP = loss - payout (After calculating OOP, set to 0 if no insurance)
    if {"gross_structure_loss","payout_structure"}.issubset(merged.columns):
        merged["oop_structure"] = (merged["gross_structure_loss"] - merged["payout_structure"]).clip(lower=0.0)
    if {"gross_contents_loss","payout_contents"}.issubset(merged.columns):
        merged["oop_contents"]  = (merged["gross_contents_loss"]  - merged["payout_contents"]).clip(lower=0.0)
    if {"oop_structure","oop_contents"}.issubset(merged.columns):
        merged["oop_total"] = merged["oop_structure"] + merged["oop_contents"]

    # ★ Synchronize gate OOP (including _kUSD version)
    _oop_cols = [
        "oop_structure", "oop_contents", "oop_total",
        "oop_structure_kUSD", "oop_contents_kUSD", "oop_total_kUSD"
    ]
    for c in _oop_cols:
        if c in merged.columns:
            merged.loc[m0, c] = 0.0

    # has_claim is 1 only if insured and has payout (supports _kUSD)
    _payout_total = "payout_total_kUSD" if "payout_total_kUSD" in merged.columns else "payout_total"
    if _payout_total in merged.columns:
        merged["has_claim"] = ((merged["has_FI"]==1) & (merged[_payout_total] > 0)).astype(int)


    return merged

# -------- Premium calculation (NEW) --------
def _to_series(x, index, name):
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
    rate_bldg_1k = 3.56,
    rate_cont_1k = 4.98,
    contents_share = 0.30,        # <- Default ratio for renters (applied if contents_kUSD is missing)
    contents_share_owner = 0.57,  # <- Owner's contents:structure ratio (applied if contents_kUSD is missing)
    owner_insures_both: bool = False,  # <- True: owner insures contents too; False: insures structure only
    reserve = 1.15,
    small_fee_usd = 100.0,
    # Column mapping
    col_identity = "identity",
    col_rcv_kUSD = "rcv_kUSD",
    col_contents_kUSD = "contents_kUSD",
    col_ded_struct_kUSD = "deductible_structure_kUSD",
    col_ded_cont_kUSD = "deductible_contents_kUSD",
    clip_floor = 0.001
) -> pd.DataFrame:
    out = df.copy()
    idx = out.index

    rb = _to_series(rate_bldg_1k, idx, "rate_bldg_1k")
    rc = _to_series(rate_cont_1k, idx, "rate_cont_1k")

    identity = out[col_identity].astype(str).str.lower() if col_identity in out.columns else pd.Series("unknown", index=idx)

    owner_insures_both_flag = bool(owner_insures_both)
    
    is_owner  = identity.eq("owner")
    is_renter = identity.eq("renter")

    rcv_k = pd.to_numeric(out.get(col_rcv_kUSD, 0.0), errors="coerce").fillna(0.0)
    cont_k_raw = pd.to_numeric(out.get(col_contents_kUSD, np.nan), errors="coerce")

    # Fill missing contents_kUSD: use contents_share_owner for owner, contents_share for renter
    cont_k = cont_k_raw.copy()
    # Owner missing values
    mask_owner_na = is_owner & cont_k.isna()
    cont_k.loc[mask_owner_na] = contents_share_owner * rcv_k.loc[mask_owner_na]
    # Renter missing values
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
    ded_c = pd.to_numeric(out.get(col_ded_cont_kUSD,  0.0), errors="coerce").fillna(0.0)

    fac_s = np.clip(1.0 - np.divide(ded_s, np.maximum(cov_struct_k, 1e-9)), clip_floor, 1.0)
    fac_c = np.clip(1.0 - np.divide(ded_c, np.maximum(cov_cont_k,   1e-9)), clip_floor, 1.0)

    small_fee_kUSD = float(small_fee_usd) / 1000.0

    prem_s_k = (rb / 1000.0) * fac_s * cov_struct_k * reserve
    prem_c_k = (rc / 1000.0) * fac_c * cov_cont_k   * reserve

    # ✅ Small fixed fee: use "boolean mask" first, then convert to float
    fee_struct_mask = is_owner
    fee_cont_mask   = is_renter | (is_owner & owner_insures_both_flag)

    fee_struct = small_fee_kUSD * fee_struct_mask.astype(float)
    fee_cont   = small_fee_kUSD * fee_cont_mask.astype(float)

    # robustly coerce "has_FI" to a boolean Series aligned with out.index
    raw_insured = out["has_FI"] if "has_FI" in out.columns else 1

    if np.isscalar(raw_insured):
        insured_ser = pd.Series([raw_insured] * len(out), index=out.index)
    else:
        insured_ser = pd.Series(raw_insured, index=out.index)

    insured = pd.to_numeric(insured_ser, errors="coerce").fillna(1.0) > 0

    insured_f = insured.astype(float)

    prem_s_k   = prem_s_k   * insured_f
    prem_c_k   = prem_c_k   * insured_f
    fee_struct = fee_struct * insured_f
    fee_cont   = fee_cont   * insured_f

    out["premium_structure_kUSD"] = prem_s_k + fee_struct
    out["premium_contents_kUSD"]  = prem_c_k + fee_cont
    out["premium_total_kUSD"]     = out["premium_structure_kUSD"] + out["premium_contents_kUSD"]
    # Ensure has_FI is preserved for downstream use
    out["has_FI"] = insured.astype(int)
    return out


# -------- Public APIs --------

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
    Apply insurance financials at the household level. Steps:
      1) fill policy columns (limits/deductibles)
      2) compute RAW payouts per coverage (structure/contents), UNgated
      3) apply identity gating (owners: structure; renters: contents; owners' contents optional)
      4) apply decision gating (no FI => payouts=0)
      5) compute OOP = loss - payout
      6) compute premiums (structure/contents)
      7) derive totals + USD columns
    Returns df with payout_*, oop_*, premium_* (kUSD & USD), has_claim, gross_total.
    """
    # ---- column map & copy ----
    cmap = {
        "identity": "identity",
        "gross_structure_loss": "gross_structure_loss_kUSD",   # <== Default to use _kUSD column
        "gross_contents_loss":  "gross_contents_loss_kUSD",
        "limit_structure_kUSD": "limit_structure_kUSD",
        "deductible_structure_kUSD": "deductible_structure_kUSD",
        "limit_contents_kUSD":  "limit_contents_kUSD",
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
    is_owner  = identity.eq("owner")
    is_renter = identity.eq("renter")

    # If caller provides non-kUSD columns, still use the column pointed to by cmap
    g_struct = _safe_series(out, cmap["gross_structure_loss"], 0.0).astype(float).fillna(0.0)
    g_cont   = _safe_series(out, cmap["gross_contents_loss"],  0.0).astype(float).fillna(0.0)

    # ---- ensure policy cols present (limits/deductibles) ----
    for key in ["limit_structure_kUSD","deductible_structure_kUSD",
                "limit_contents_kUSD","deductible_contents_kUSD"]:
        col = cmap[key]
        if col not in out.columns:
            out[col] = np.nan

    # ---- fill policy from dict (owner/renter may differ) ----
    if policy:
        _apply_policy_fills(out, identity, policy, policy_overwrite, cmap)

    # default fills for missing policy numbers
    lim_struct = pd.to_numeric(out[cmap["limit_structure_kUSD"]], errors="coerce").fillna(np.inf)
    ded_struct = pd.to_numeric(out[cmap["deductible_structure_kUSD"]], errors="coerce").fillna(0.0)
    lim_cont   = pd.to_numeric(out[cmap["limit_contents_kUSD"]],      errors="coerce").fillna(np.inf)
    ded_cont   = pd.to_numeric(out[cmap["deductible_contents_kUSD"]], errors="coerce").fillna(0.0)

    # ---- (2) RAW payouts per coverage (UNgated) ----
    # Safe initialization to avoid missing assignment in any branch
    payout_struct_all = pd.Series(0.0, index=out.index, dtype=float)
    payout_cont_all   = pd.Series(0.0, index=out.index, dtype=float)

    payout_struct_all = _payout(g_struct, lim_struct, ded_struct)  # kUSD
    payout_cont_all   = _payout(g_cont,   lim_cont,   ded_cont)    # kUSD

    # ---- (3) identity gating to coverage ----
    # owners: structure; owners' contents depend on owner_insures_both switch; renters: contents only
    owner_insures_both_flag = bool(owner_insures_both)

    payout_structure = payout_struct_all.where(is_owner | (~is_owner & ~is_renter), 0.0)
    if owner_insures_both_flag:
        payout_contents = payout_cont_all.where(is_owner | is_renter | (~is_owner & ~is_renter), 0.0)
    else:
        payout_contents = payout_cont_all.where(is_renter | (~is_owner & ~is_renter), 0.0)

    # ---- (4) decision gating (no FI => payouts=0) ----
    # Attach current gated values to df first, upper level gating will find by column name and zero out
    out["payout_structure_kUSD"] = pd.to_numeric(payout_structure, errors="coerce").fillna(0.0)
    out["payout_contents_kUSD"]  = pd.to_numeric(payout_contents,  errors="coerce").fillna(0.0)

    if gate_by_decisions:
        yr = year if year is not None else _infer_year_from_df(out)
        # This function expects to zero out payout_*_kUSD based on has_FI / decisions (and may write has_FI)
        out = _apply_decisions_gating(out, dec=dec, idxer=idxer, year=yr)

    # Read back (in case modified after gating)
    payout_structure = pd.to_numeric(out.get("payout_structure_kUSD", 0.0), errors="coerce").fillna(0.0)
    payout_contents  = pd.to_numeric(out.get("payout_contents_kUSD",  0.0), errors="coerce").fillna(0.0)

    # ---- (5) OOP after gating ----
    # OOP = loss - payout; if no insurance/no payout, equals total loss
    oop_structure = (g_struct - payout_structure).clip(lower=0.0)
    oop_contents  = (g_cont   - payout_contents).clip(lower=0.0)

    # ---- totals (kUSD) ----
    payout_total_kUSD = payout_structure + payout_contents
    oop_total_kUSD    = oop_structure + oop_contents
    gross_total_kUSD  = g_struct + g_cont

    # ---- Add a clear policyholder flag (0/1) at household level ----
    out["has_FI"] = pd.to_numeric(out.get("has_FI", 0), errors="coerce").fillna(0).astype(int)
    out["policyholder"] = out["has_FI"]  # alias; explicit name for downstream use

    out["payout_structure_kUSD"] = payout_structure
    out["payout_contents_kUSD"]  = payout_contents
    out["payout_total_kUSD"]     = payout_total_kUSD

    out["oop_structure_kUSD"] = oop_structure
    out["oop_contents_kUSD"]  = oop_contents
    out["oop_total_kUSD"]     = oop_total_kUSD

    # ★ Final enforcement: Uninsured -> payout / OOP all 0, and has_claim is 1 only if insured and has payout
    _ins = pd.to_numeric(out.get("has_FI", 0), errors="coerce").fillna(0).astype(int).eq(1)
    _cols = ["payout_structure_kUSD","payout_contents_kUSD","payout_total_kUSD",
            "oop_structure_kUSD","oop_contents_kUSD","oop_total_kUSD"]
    for c in _cols:
        if c in out.columns:
            out.loc[~_ins, c] = 0.0
    if "payout_total_kUSD" in out.columns:
        out["has_claim"] = (_ins & (out["payout_total_kUSD"] > 0)).astype(int)


    out["gross_total_kUSD"]   = gross_total_kUSD
    out["has_claim"]          = (payout_total_kUSD > 0).astype(int)

    # ---- (6) Premiums (by coverage) ----
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
            col_identity=cmap.get("identity","identity"),
            col_rcv_kUSD=colmap.get("rcv_kUSD","rcv_kUSD") if colmap else "rcv_kUSD",
            col_contents_kUSD=colmap.get("contents_kUSD","contents_kUSD") if colmap else "contents_kUSD",
            col_ded_struct_kUSD=cmap.get("deductible_structure_kUSD","deductible_structure_kUSD"),
            col_ded_cont_kUSD=cmap.get("deductible_contents_kUSD","deductible_contents_kUSD"),
        )
    else:
        # If premium not calculated, ensure columns exist as 0
        for col in ["premium_structure_kUSD","premium_contents_kUSD","premium_total_kUSD"]:
            if col not in out.columns:
                out[col] = 0.0

    # ---- (7) USD columns (for compatibility with flood damage aggregation) ----
    # Convert common kUSD columns to USD equivalent columns
    for col in [
        "gross_structure_loss_kUSD","gross_contents_loss_kUSD","gross_total_kUSD",
        "payout_structure_kUSD","payout_contents_kUSD","payout_total_kUSD",
        "oop_structure_kUSD","oop_contents_kUSD","oop_total_kUSD",
        "premium_structure_kUSD","premium_contents_kUSD","premium_total_kUSD",
    ]:
        if col in out.columns:
            out[col.replace("_kUSD","_usd")] = pd.to_numeric(out[col], errors="coerce").fillna(0.0) * 1000.0
        else:
            # If caller uses non-kUSD original column names, ignore here
            pass

    # Ensure components exist (if already present, this just takes the value)
    out["gross_structure_loss_kUSD"] = pd.to_numeric(out.get("gross_structure_loss_kUSD", out.get("gross_structure_loss", 0.0)), errors="coerce").fillna(0.0)
    out["gross_contents_loss_kUSD"]  = pd.to_numeric(out.get("gross_contents_loss_kUSD",  out.get("gross_contents_loss",  0.0)), errors="coerce").fillna(0.0)

    # Sum components to get total (household level)
    out["gross_total_kUSD"] = out["gross_structure_loss_kUSD"] + out["gross_contents_loss_kUSD"]

    # Do the same for payout (if generated in your flow)
    if ("payout_structure_kUSD" in out.columns) or ("payout_contents_kUSD" in out.columns):
        out["payout_structure_kUSD"] = pd.to_numeric(out.get("payout_structure_kUSD", 0.0), errors="coerce").fillna(0.0)
        out["payout_contents_kUSD"]  = pd.to_numeric(out.get("payout_contents_kUSD",  0.0), errors="coerce").fillna(0.0)
        out["payout_total_kUSD"]     = out["payout_structure_kUSD"] + out["payout_contents_kUSD"]

    # (Optional) sanity check
    _bad = (out["gross_total_kUSD"] - (out["gross_structure_loss_kUSD"] + out["gross_contents_loss_kUSD"])).abs() > 1e-6
    if _bad.any():
        print(f"[warn] { _bad.sum() } household rows with mismatched gross_total_kUSD; recomputed by sum of parts.")
    # --- Then return out ---

    return out


def detect_tract_column(df: pd.DataFrame) -> Optional[str]:
    for cand in ["CensusTract", "tract_geoid", "tract_geoid10", "GEOID", "geoid"]:
        if cand in df.columns:
            return cand
    # try to detect GEOID-like if only one candidate exists
    for c in df.columns:
        if "tract" in c.lower() or "geoid" in c.lower():
            return c
    return None


# =========================================
# 3) aggregate_by_tract (REPLACE / Replace entire section)
# =========================================
def aggregate_by_tract(hh: pd.DataFrame, tract_col: str = "tract_geoid") -> pd.DataFrame:
    """
    Tract-level aggregation (unified kUSD standard)
    - overall gross/payout/oop/premium (structure/contents/total)
    - owner_gross_total_kUSD、renter_gross_total_kUSD
    - owner_households、renter_households、households (= sum of both)
    """
    if hh is None or hh.empty:
        return pd.DataFrame(columns=[tract_col])

    df = hh.copy()

    # Tract column handling (normalize column name)
    if tract_col not in df.columns:
        for cand in ("CensusTract", "GEOID", "tract", "geoid", "tract_geoid10"):
            if cand in df.columns:
                df = df.rename(columns={cand: tract_col})
                break
    df[tract_col] = df[tract_col].astype(str)

    # ----- Identity flag (for owner/renter aggregation and counting) -----
    ident = df.get("identity", "").astype(str).str.lower()
    df["_is_owner"]  = (ident == "owner").astype(int)
    df["_is_renter"] = (ident == "renter").astype(int)

    # Helper: Ensure *_kUSD components exist (sync from legacy column names if missing)
    def _ensure_kusd(modern: str, legacy: str):
        if modern in df.columns:
            df[modern] = pd.to_numeric(df[modern], errors="coerce").fillna(0.0)
        elif legacy in df.columns:
            df[modern] = pd.to_numeric(df[legacy], errors="coerce").fillna(0.0)
        else:
            df[modern] = 0.0

    # gross components (kUSD)
    _ensure_kusd("gross_structure_loss_kUSD", "gross_structure_loss")
    _ensure_kusd("gross_contents_loss_kUSD",  "gross_contents_loss")

    # payout / oop / premium (kUSD)
    _ensure_kusd("payout_structure_kUSD",   "payout_structure")
    _ensure_kusd("payout_contents_kUSD",    "payout_contents")
    _ensure_kusd("oop_structure_kUSD",      "oop_structure")
    _ensure_kusd("oop_contents_kUSD",       "oop_contents")
    _ensure_kusd("premium_structure_kUSD",  "premium_structure")
    _ensure_kusd("premium_contents_kUSD",   "premium_contents")

    # Policy flag
    if "has_FI" in df.columns:
        df["has_FI"] = pd.to_numeric(df["has_FI"], errors="coerce").fillna(0).astype(int)
    else:
        df["has_FI"] = 0

    # ★ Ensure has_claim exists and is int (1=has payout this year)
    if "has_claim" in df.columns:
        df["has_claim"] = pd.to_numeric(df["has_claim"], errors="coerce").fillna(0).astype(int)
    else:
        df["has_claim"] = 0

    # ----- Per-row total (sum components only, unified standard) -----
    df["gross_total_row_kUSD"] = df["gross_structure_loss_kUSD"] + df["gross_contents_loss_kUSD"]

    # Per-row owner/renter totals (split by identity flag)
    df["owner_total_row_kUSD"]  = df["gross_total_row_kUSD"] * df["_is_owner"]
    df["renter_total_row_kUSD"] = df["gross_total_row_kUSD"] * df["_is_renter"]

    # ----- GroupBy aggregation by tract -----
    grp = df.groupby(tract_col, dropna=False).agg(
        # overall gross
        gross_structure_loss_kUSD=("gross_structure_loss_kUSD", "sum"),
        gross_contents_loss_kUSD =("gross_contents_loss_kUSD",  "sum"),

        # overall payout / oop / premium
        payout_structure_kUSD    =("payout_structure_kUSD",     "sum"),
        payout_contents_kUSD     =("payout_contents_kUSD",      "sum"),
        oop_structure_kUSD       =("oop_structure_kUSD",        "sum"),
        oop_contents_kUSD        =("oop_contents_kUSD",         "sum"),
        premium_structure_kUSD   =("premium_structure_kUSD",    "sum"),
        premium_contents_kUSD    =("premium_contents_kUSD",     "sum"),

        # totals (overall)
        gross_total_kUSD         =("gross_total_row_kUSD",      "sum"),

        # owner / renter totals
        owner_gross_total_kUSD   =("owner_total_row_kUSD",      "sum"),
        renter_gross_total_kUSD  =("renter_total_row_kUSD",     "sum"),

        # owner / renter counts
        owner_households         =("_is_owner",                 "sum"),
        renter_households        =("_is_renter",                "sum"),

        # policyholders
        policyholders            =("has_FI",                    "sum"),

        # ★ claims (number of people with payout this year)
        claims                   =("has_claim",                 "sum"),
    ).reset_index()
    
    # Override totals by summing components (avoid inconsistency)
    grp["payout_total_kUSD"]  = grp["payout_structure_kUSD"]     + grp["payout_contents_kUSD"]
    grp["oop_total_kUSD"]     = grp["oop_structure_kUSD"]        + grp["oop_contents_kUSD"]
    grp["premium_total_kUSD"] = grp["premium_structure_kUSD"]    + grp["premium_contents_kUSD"]
    grp["gross_total_kUSD"]   = grp["gross_structure_loss_kUSD"] + grp["gross_contents_loss_kUSD"]

    # ★ households = owner + renter (avoid inconsistency with size)
    grp["households"] = grp["owner_households"].astype(int) + grp["renter_households"].astype(int)

    # Convert kUSD columns to USD (for compatibility)
    for col in [c for c in grp.columns if c.endswith("_kUSD")]:
        grp[col.replace("_kUSD", "_usd")] = pd.to_numeric(grp[col], errors="coerce").fillna(0.0) * 1000.0

    # Ensure count columns are integers
    for c in ("owner_households", "renter_households", "households", "policyholders", "claims"):
        if c in grp.columns:
            grp[c] = pd.to_numeric(grp[c], errors="coerce").fillna(0).astype(int)

    return grp





# === Damage-before-move finance for a single year ===
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd

# ================================
# 1) run_finance_for_year (REPLACE)
# ================================
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
    premium: Optional[Dict[str, Any]] = None,
    owner_contents_ratio: float = 0.57,   # owner contents 缺值以比例補
    owner_insures_both: bool = False,     # 影響賠付/保費 gating，不改毛損
    ffe_ft: float = 0.5,                  # base FFE（英尺）
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply insurance financials for a single simulation year.
    
    This function computes flood damage losses, applies insurance payouts,
    calculates premiums, and aggregates results by census tract.
    
    Key Concept - Damage-before-move:
        Losses for year Y are calculated using the household's pre-move
        tract location and that year's flood hazard at that tract.
    
    Args:
        year: Simulation year
        state_event: Household state DataFrame with flood damage
        depth_map: Dict mapping tract_geoid -> flood depth in meters
        ratio_by_tract: Dict mapping tract_geoid -> damage ratio (0-1)
        decisions: DataFrame with insurance decisions (has_FI column)
        policy: Insurance policy terms dict
        gate_by_decisions: If True, uninsured households get zero payout
        premium: Premium calculation parameters dict
        
    Returns:
        (FIN_HH, FIN_TRACT): Tuple of household-level and tract-level DataFrames
    """
    df = state_event.copy()

    # ---- Column and type normalization ----
    if "group" in df.columns and "identity" not in df.columns:
        df = df.rename(columns={"group": "identity"})
    if "identity" not in df.columns:
        raise KeyError("state_event missing 'identity' column (must be 'owner' / 'renter').")

    df[tract_col] = df[tract_col].astype(str)
    df["identity"] = df["identity"].astype(str).str.lower()

    # Event reference (damage happens first, then move): match current tract with hazard
    df["event_year"]        = year
    df["event_tract_geoid"] = df[tract_col].astype(str)

    # If per-household event_depth_m exists, use it; otherwise map from tract average
    if "event_depth_m" not in df.columns:
        df["event_depth_m"] = df["event_tract_geoid"].map(
            lambda t: float(depth_map.get(str(t), np.nan))
        ).astype(float)

    # (For diagnostics, not used in loss calculation) tract ratio
    df["event_loss_ratio"] = df["event_tract_geoid"].map(
        lambda t: float(ratio_by_tract.get(str(t), 0.0))
    ).astype(float)

    # Ensure monetary columns exist and are numeric
    for col in ("rcv_kUSD", "contents_kUSD"):
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Fill missing owner contents using ratio (consistent with vulnerability pipeline)
    is_owner = df["identity"].eq("owner")
    need_fill = is_owner & (pd.to_numeric(df["contents_kUSD"], errors="coerce").fillna(0.0) <= 0.0)
    if owner_contents_ratio is not None and need_fill.any():
        df.loc[need_fill, "contents_kUSD"] = owner_contents_ratio * df.loc[need_fill, "rcv_kUSD"]


    # --- Renter structure loss gating: always zero (renters don't own structure) ---
    is_renter = df["identity"].eq("renter")
    if not renters_have_structure:
        df.loc[is_renter, "gross_structure_loss_kUSD"] = 0.0

    # --- Create/sync legacy column names to avoid downstream inconsistencies ---
    df["gross_structure_loss"] = pd.to_numeric(df["gross_structure_loss_kUSD"], errors="coerce").fillna(0.0)
    df["gross_contents_loss"]  = pd.to_numeric(df["gross_contents_loss_kUSD"],  errors="coerce").fillna(0.0)

    # ---- Apply insurance financials with decision gating ----
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
            "gross_structure_loss": "gross_structure_loss",  # ※ 這裡傳 legacy 名稱，但內容已與 *_kUSD 同步
            "gross_contents_loss":  "gross_contents_loss",
        },
    )

    # ---- Aggregate to tract level ----
    FIN_TRACT = aggregate_by_tract(FIN_HH, tract_col=tract_col)

    # ---- (Optional) Save CSV outputs ----
    if save_csv and output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        FIN_HH.to_csv(output_dir / f"finance_households_{year}.csv", index=False, encoding="utf-8-sig")
        FIN_TRACT.to_csv(output_dir / f"finance_tract_{year}.csv",    index=False, encoding="utf-8-sig")

    # ---- Tract-level premium breakdown (USD aggregation) ----
    tmp = FIN_HH.copy()
    tmp["identity"]    = tmp["identity"].astype(str).str.lower()
    tmp["tract_geoid"] = tmp["tract_geoid"].astype(str)

    tmp["prem_s_usd"] = pd.to_numeric(tmp.get("premium_structure_kUSD", 0.0), errors="coerce").fillna(0.0) * 1000.0
    tmp["prem_c_usd"] = pd.to_numeric(tmp.get("premium_contents_kUSD",  0.0), errors="coerce").fillna(0.0) * 1000.0

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
    prem_tract["owner_premium_total_usd"]  = (
        prem_tract["owner_premium_structure_usd"] + prem_tract["owner_premium_contents_usd"]
    )
    prem_tract["renter_premium_total_usd"] = prem_tract["renter_premium_contents_usd"]

    FIN_TRACT = FIN_TRACT.merge(prem_tract, on="tract_geoid", how="left")

    return FIN_HH, FIN_TRACT
