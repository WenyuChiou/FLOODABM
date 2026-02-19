from __future__ import annotations
from pathlib import Path
from typing import Tuple, Dict, Any, Iterable
import yaml
import numpy as np
import pandas as pd

# ---------- YAML / path ----------
def load_yaml_cfg(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    if path.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError(f"Only YAML is allowed. Convert to .yaml: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data or {}

def resolve_path(base: Path, rel_or_abs: str | None, modules_root: Path) -> Path | None:
    if not rel_or_abs: 
        return None
    p = Path(rel_or_abs)
    if p.is_absolute():
        return p
    for cand in (base / rel_or_abs, modules_root / rel_or_abs, Path(rel_or_abs)):
        if cand.exists():
            return cand
    return Path(rel_or_abs)

# ---------- inline mg_share / policy ----------
def load_inline_mgshare_policy(cfg: dict) -> Tuple[Dict[str,float], Dict[str,Any]]:
    policy = cfg.get("policy") or {}
    mg_raw = cfg.get("mg_share")
    if not mg_raw:
        raise FileNotFoundError("Inline 'mg_share' not found in YAML. Add mg_share: {tract_geoid: share}.")
    if isinstance(mg_raw, dict):
        mg = {str(k): float(v) for k, v in mg_raw.items()}
    elif isinstance(mg_raw, list):
        tmp = {}
        for row in mg_raw:
            if not isinstance(row, dict): 
                continue
            t = row.get("tract_geoid") or row.get("tract") or row.get("geoid")
            s = row.get("share") or row.get("mg_share")
            if t is not None and s is not None:
                tmp[str(t)] = float(s)
        mg = tmp
    else:
        raise ValueError("Unsupported mg_share format in YAML.")
    if not policy:
        # allow empty policy, but provide safe minimal defaults
        policy = {
            "limit_structure_kUSD": 250.0,
            "deductible_structure_kUSD": 1.0,
            "limit_contents_kUSD": 100.0,
            "deductible_contents_kUSD": 1.0,
            "owner":  {"deductible_structure_kUSD": 1.0},
            "renter": {"deductible_contents_kUSD": 1.0},
            "coinsurance": {"enabled": False, "rate": 0.8},
        }
    return mg, policy

# ---------- finance (read-only; all from YAML) ----------
def read_finance_from_yaml(cfg: dict) -> tuple[dict, dict, dict]:
    fin = cfg.get("finance", {}) or {}
    default_premium = fin.get("default_premium", {}) or {}
    premium_by_state = fin.get("premium_by_state", {}) or {}
    owner_insurance  = fin.get("owner_insurance_contents", {}) or {}

    default_premium = {
        "rate_bldg_1k": float(default_premium.get("rate_bldg_1k", 3.6)),
        "rate_cont_1k": float(default_premium.get("rate_cont_1k", 5.0)),
        "contents_share": float(default_premium.get("contents_share", 0.30)),
        "reserve": float(default_premium.get("reserve", 1.15)),
        "small_fee_usd": float(default_premium.get("small_fee_usd", 100.0)),
    }

    for st, row in list(premium_by_state.items()):
        premium_by_state[st] = {
            "rate_bldg_1k": float(row.get("rate_bldg_1k", default_premium["rate_bldg_1k"])),
            "rate_cont_1k": float(row.get("rate_cont_1k", default_premium["rate_cont_1k"])),
        }

    owner_insurance = {
        "owner_insures_both": bool(owner_insurance.get("owner_insures_both", True)),
        "owner_contents_ratio": float(owner_insurance.get("owner_contents_ratio", 0.57)),
    }
    return premium_by_state, default_premium, owner_insurance



def _standardize_has_FI_from_dec(dec_df: pd.DataFrame) -> pd.DataFrame:
    d = dec_df.copy()
    if "i" not in d.columns:
        # If your decision output always has 'i', this part can be omitted; otherwise please align with your idxer
        d = d.reset_index().rename(columns={"index":"i"})
    if "has_FI" not in d.columns:
        # Infer from action: FI means purchased this year, but effective "next year"
        if "action" in d.columns:
            d["has_FI"] = (d["action"].astype(str).str.upper().eq("FI")).astype(int)
        else:
            d["has_FI"] = 0
    
    return d[["i","has_FI"]]

# ---------- depths (old loader) ----------
def load_depths_legacy(path_depths: Path) -> pd.DataFrame:
    from  modules.actions.vuln_for_tp import load_depths_all_years
    df = load_depths_all_years(path_depths)
    if "tract_geoid" not in df.columns and "tract" in df.columns:
        df = df.rename(columns={"tract": "tract_geoid"})
    if "year" not in df.columns:
        raise ValueError("Depths must include a 'year' column after loading.")
    df["tract_geoid"] = df["tract_geoid"].astype(str)
    return df

def years_from_cfg(cfg: dict, years_in_depths, path_events: Path | None) -> list[int]:
    import pandas as pd
    years_cfg = cfg.get("years", {}) or {"mode": "span", "start": 2011, "end": 2023}
    mode = str(years_cfg.get("mode", "span")).lower()
    years_in_depths = sorted(int(y) for y in years_in_depths)
    if mode == "events_only" and path_events and path_events.exists():
        ev = pd.read_csv(path_events)
        ycol = "year" if "year" in ev.columns else ("Year" if "Year" in ev.columns else None)
        years_from_events = sorted(pd.to_numeric(ev[ycol], errors="coerce").dropna().astype(int).unique().tolist()) if ycol else []
        return sorted(set(years_in_depths).intersection(years_from_events))
    if mode == "explicit":
        return sorted(set(int(y) for y in (years_cfg.get("list") or []) if str(y).strip()))
    y0 = int(years_cfg.get("start", min(years_in_depths)))
    y1 = int(years_cfg.get("end",   max(years_in_depths)))
    return [y for y in range(y0, y1+1) if y in years_in_depths]

def depths_for_year(depths_long: pd.DataFrame, year: int) -> pd.DataFrame:
    return depths_long.loc[depths_long["year"].eq(year), ["tract_geoid","depth_m"]].copy()

# ---------- tract psychology ----------
def init_tract_psych_safe(tracts, seed: int, rng) -> pd.DataFrame:
    from  modules.actions.mgmix_tp import init_tract_psych_mg_nmg
    attempts = [
        lambda: init_tract_psych_mg_nmg(tracts=tracts, seed=seed),
        lambda: init_tract_psych_mg_nmg(tracts=tracts, rng=rng),
        lambda: init_tract_psych_mg_nmg(tracts=tracts),
        lambda: init_tract_psych_mg_nmg(tracts, seed),
        lambda: init_tract_psych_mg_nmg(tracts, rng),
        lambda: init_tract_psych_mg_nmg(tracts),
    ]
    last_err = None
    for fn in attempts:
        try:
            df = fn()
            if "tract_geoid" not in df.columns and "tract" in df.columns:
                df = df.rename(columns={"tract": "tract_geoid"})
            for c in ("t_clock_MG","t_clock_NMG"):
                if c not in df.columns: df[c] = 0.0
            df["tract_geoid"] = df["tract_geoid"].astype(str)
            return df
        except TypeError as e:
            last_err = e
    raise TypeError(f"init_tract_psych_mg_nmg signature mismatch: {last_err}")

# ---------- households loader ----------
def load_households_csv(path: Path) -> pd.DataFrame:
    HH = pd.read_csv(path, dtype={"tract_geoid": "string"})
    HH["group"] = HH["group"].astype(str).str.lower().map({"owner":"owner","renter":"renter"}).fillna("owner")
    for col in ("rcv_kUSD","contents_kUSD"):
        if col not in HH.columns: HH[col] = 0.0
        HH[col] = pd.to_numeric(HH[col], errors="coerce").fillna(0.0)
    if "elev_ft_total" not in HH.columns: HH["elev_ft_total"] = 0.0
    HH["elev_ft_total"] = pd.to_numeric(HH["elev_ft_total"], errors="coerce").fillna(0.0)
    if "i" in HH.columns and not HH["i"].is_unique: HH = HH.rename(columns={"i": "i_orig"})
    if "i_orig" not in HH.columns: HH["i_orig"] = HH["i"] if "i" in HH.columns else np.arange(1, len(HH)+1)
    HH.insert(0, "i", np.arange(1, len(HH)+1, dtype=np.int64))
    if "hh_idx_within_group" not in HH.columns:
        HH["hh_idx_within_group"] = HH.groupby(["tract_geoid","group"]).cumcount() + 1
    if "policy_name" not in HH.columns: HH["policy_name"] = ""
    HH["tract_geoid"] = HH["tract_geoid"].astype(str)
    return HH[["i","group","tract_geoid","rcv_kUSD","contents_kUSD","elev_ft_total","policy_name","i_orig","hh_idx_within_group"]]

# ---------- TP group params from YAML ----------
def grp_params_from_yaml(cfg: dict, grp: str, TPGroupParams) -> Any:
    tp_decay = (cfg.get("tp_decay_params", {}) or {}).get(grp.upper(), {}) or {}
    # Updated defaults after (1-PA) formula change (2024-12 calibration)
    defaults = {"alpha":0.50, "beta":0.216667, "tau0":1.00, "tau_inf":32.18744, "k":0.03} if grp.upper()=="MG" else \
               {"alpha":0.383333, "beta":0.10, "tau0":1.00, "tau_inf":50.10013, "k":0.01}
    j = {**defaults, **{k: tp_decay.get(k, defaults[k]) for k in defaults}}
    return TPGroupParams(
        alpha=float(j["alpha"]),
        beta=float(j["beta"]),
        tau0=float(j["tau0"]),
        tau_inf=float(j["tau_inf"]),
        k=float(j["k"]),
    )



# ---------- action share summaries ----------
def _action_share_owner_renter_by_tract(dec: pd.DataFrame) -> pd.DataFrame:
    """
    Return: Annual x Per Tract "Owner specific column" + "Renter specific column" summary table
    Columns:
      year, tract_geoid,
      owner_N_total, owner_n_FI..owner_n_DN, owner_share_FI..owner_share_DN,
      renter_N_total, renter_n_FI..renter_n_DN, renter_share_FI..renter_share_DN
    """
    d = dec.copy()
    d["tract_geoid"] = d["tract_geoid"].astype(str)
    d["group"] = d["group"].str.lower().map({"owner": "owner", "renter": "renter"}).fillna("owner")
    d["action"] = d["action"].astype(str)

    OWNER_ACTIONS  = ["FI", "EH", "BP", "DN"]
    RENTER_ACTIONS = ["FI", "RL", "DN"]

    out_list = []
    for (yy, tg), gdf in d.groupby(["year", "tract_geoid"]):
        # owner
        g_owner = gdf[gdf["group"].isin(["owner", "homeowner"])]
        cnt_o = g_owner.groupby("action").size().reindex(OWNER_ACTIONS, fill_value=0)
        N_owner = int(cnt_o.sum())
        row = {"year": yy, "tract_geoid": tg, "owner_N_total": N_owner}
        for a in OWNER_ACTIONS:
            row[f"owner_n_{a}"] = int(cnt_o.get(a, 0))
            row[f"owner_share_{a}"] = (row[f"owner_n_{a}"] / N_owner) if N_owner > 0 else 0.0

        # renter
        g_renter = gdf[gdf["group"] == "renter"]
        cnt_r = g_renter.groupby("action").size().reindex(RENTER_ACTIONS, fill_value=0)
        N_renter = int(cnt_r.sum())
        row["renter_N_total"] = N_renter
        for a in RENTER_ACTIONS:
            row[f"renter_n_{a}"] = int(cnt_r.get(a, 0))
            row[f"renter_share_{a}"] = (row[f"renter_n_{a}"] / N_renter) if N_renter > 0 else 0.0

        out_list.append(row)

    return pd.DataFrame(out_list)

def finalize_has_fi(df: pd.DataFrame,
                    dec_prev: pd.DataFrame | None = None,
                    *,
                    i_col: str = "i",
                    sticky: bool = True,
                    drop_is_fi: bool = True,
                    inplace: bool = False) -> pd.DataFrame:
    """
    Organize has_FI at once (also clear *_dec; optionally merge this year's decision):
      - Merge has_FI_dec / is_FI_dec back to has_FI (OR), then delete suffix
      - If dec_prev is provided: align with global i, merge this year's FI (sticky=True -> OR; False -> Overwrite)
      - Finally keep only has_FI (optionally delete is_FI)
    """
    if not inplace:
        df = df.copy()

    def _b01_series(s: pd.Series | int | float | None, idx) -> pd.Series:
        if isinstance(s, pd.Series):
            return pd.to_numeric(s, errors="coerce").reindex(idx).fillna(0).astype(int)
        return pd.Series(0, index=idx, dtype=int)

    # 0) normalize base has_FI
    base = _b01_series(df.get("has_FI", 0), df.index)

    # 1) Consume *_dec -> OR back to has_FI, then delete
    if "has_FI_dec" in df.columns:
        base = (base | _b01_series(df["has_FI_dec"], df.index)).astype(int)
        df.drop(columns=["has_FI_dec"], inplace=True)
    if "is_FI_dec" in df.columns:
        base = (base | _b01_series(df["is_FI_dec"], df.index)).astype(int)
        df.drop(columns=["is_FI_dec"], inplace=True)

    # 2) Merge this year's decision (if dec_prev is provided)
    if dec_prev is not None and (i_col in df.columns) and (i_col in dec_prev.columns):
        # Source priority: has_FI > is_FI > action=='FI'
        if "has_FI" in dec_prev.columns:
            dec_bits = pd.to_numeric(dec_prev["has_FI"], errors="coerce").fillna(0).astype(int)
        elif "is_FI" in dec_prev.columns:
            dec_bits = pd.to_numeric(dec_prev["is_FI"], errors="coerce").fillna(0).astype(int)
        elif "action" in dec_prev.columns:
            dec_bits = dec_prev["action"].astype(str).str.upper().eq("FI").astype(int)
        else:
            dec_bits = None

        if dec_bits is not None:
            key = pd.to_numeric(dec_prev[i_col], errors="coerce")
            map_series = pd.Series(dec_bits.values, index=key).dropna()
            ids = pd.to_numeric(df[i_col], errors="coerce")
            dec_aligned = ids.map(map_series).fillna(0).astype(int)
            base = (base | dec_aligned).astype(int) if sticky else dec_aligned.astype(int)

    # 3) Write back has_FI; (optional) remove is_FI
    df["has_FI"] = base
    if drop_is_fi and "is_FI" in df.columns:
        df.drop(columns=["is_FI"], inplace=True)

    return df if not inplace else None
