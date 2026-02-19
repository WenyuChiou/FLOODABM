# -*- coding: utf-8 -*-
# modules/actions/mgmix_pipeline.py (fast)
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Callable, Dict, Tuple, List
from pathlib import Path
import json


from .mgmix_decision import (  # type: ignore
        build_state_indexer, features_take_by_indexer,
        mix_probs_by_weight, sequential_decision_fast
    )
from .mgmix_tp import (TPConfig, TPGroupParams, update_tract_tp_mg_nmg)  # type: ignore
from .mgmix_tp import update_tract_tp_mg_nmg



# --- utils & module-level cache for per-tract lognormal (NO class/self) ---
_BP_PARAMS_CACHE = None  # {"owner":{tract:{mu,sigma,T,N}}, "renter":{...}}

def _sigma_from_cv(cv: float):
    import numpy as np
    return float(np.sqrt(np.log1p(cv**2)))

def _mu_from_mean_sigma(mean_val: float, sigma: float):
    import numpy as np
    return float(np.log(mean_val) - 0.5 * sigma**2)

def build_pertract_lognorm_params_from_totals(
    state0, *, tract_col="tract_geoid", group_col="group",
    value_col_owner="rcv_kUSD", value_col_renter="contents_kUSD",
    cv_owner=0.694, cv_renter=1.8
):
    """
    Calculate lognormal parameters for owner (RCV) and renter (contents) based on
    'initial year' tract totals and household counts.
    Returns: {"owner":{tract:{mu,sigma,T,N}}, "renter":{...}}
    """
    df = state0.copy()
    df[tract_col] = df[tract_col].astype(str)
    df[group_col] = df[group_col].astype(str).str.lower().map({"owner":"owner","renter":"renter"}).fillna("owner")

    out = {"owner": {}, "renter": {}}

    # owner -> RCV
    dfo = df[df[group_col].eq("owner")].copy()
    dfo[value_col_owner] = pd.to_numeric(dfo.get(value_col_owner, 0.0), errors="coerce").fillna(0.0)
    stat_o = dfo.groupby(tract_col)[value_col_owner].agg(T="sum", N="size").reset_index()
    sig_o = _sigma_from_cv(float(cv_owner))
    for _, r in stat_o.iterrows():
        mean_x = max(float(r["T"]) / max(int(r["N"]), 1), 1e-6)
        mu = _mu_from_mean_sigma(mean_x, sig_o)
        out["owner"][str(r[tract_col])] = {"mu": mu, "sigma": sig_o, "T": float(r["T"]), "N": int(r["N"])}

    # renter -> contents
    dfr = df[df[group_col].eq("renter")].copy()
    dfr[value_col_renter] = pd.to_numeric(dfr.get(value_col_renter, 0.0), errors="coerce").fillna(0.0)
    stat_r = dfr.groupby(tract_col)[value_col_renter].agg(T="sum", N="size").reset_index()
    sig_r = _sigma_from_cv(float(cv_renter))
    for _, r in stat_r.iterrows():
        mean_x = max(float(r["T"]) / max(int(r["N"]), 1), 1e-6)
        mu = _mu_from_mean_sigma(mean_x, sig_r)
        out["renter"][str(r[tract_col])] = {"mu": mu, "sigma": sig_r, "T": float(r["T"]), "N": int(r["N"])}

    return out


def run_one_year_mgmix_fast(
    *,
    year: int,
    state: pd.DataFrame,
    tract_psy: pd.DataFrame,
    idxer,
    w_vec: np.ndarray,
    predictor_MG,
    predictor_NMG,
    params_MG,
    params_NMG,
    tp_cfg,
    ratio_by_tract: dict[str, float],
    overlay_policy_names: dict[str, str],
    rng: np.random.RandomState,
    years_step: float = 1.0,
    reset_clock_on_flood: bool = True,
    action_dyn: dict | None = None,
    depth_m_by_tract: dict[str, float] | None = None,
    rl_dest_k_best: int = 1,
):
    """
    Return (decisions_df, updated_tract_psy, state_next, changes) using fast indexer path.
    """
    # 0) Update tract psychology based on this year's flood (Note: ratio_by_tract is already filtered by effective threshold in main)
    psy_for_dec = update_tract_tp_mg_nmg(
        tract_psych=tract_psy,
        params_MG=params_MG,
        params_NMG=params_NMG,
        cfg=tp_cfg,
        ratio_by_tract=ratio_by_tract,
        shock_if_ratio_gt0=True,            # ratio > 0 is treated as flood shock (avoid secondary threshold)
        years_step=years_step,
        reset_clock_on_flood=reset_clock_on_flood,
        flood_ratio_threshold=0.0,          # Threshold not used here; already handled in main.py
    )

    # 1) Extract features
    TP_MG, CP_MG, SP_MG    = features_take_by_indexer(psy_for_dec, idxer, "MG")
    TP_NMG, CP_NMG, SP_NMG = features_take_by_indexer(psy_for_dec, idxer, "NMG")

    # 2) Model prediction (mu-only)
    pMG = predictor_MG(TP=TP_MG, CP=CP_MG, SP=SP_MG)
    pN  = predictor_NMG(TP=TP_NMG, CP=CP_NMG, SP=SP_NMG)

    # 3) Mix household probabilities weighted by tract MG share
    probs = mix_probs_by_weight(pMG, pN, w_vec)

    # 4) Hierarchical decision (owner: FI->EH->BP; renter: FI->RL)
    dec = sequential_decision_fast(
        state,
        probs,
        rng,
        overlay_policy_names=overlay_policy_names
    ).reset_index(drop=True)

    # 5) Assemble output decision table
    base = state[["i","group","tract_geoid"]].reset_index(drop=True)
    dec  = base.join(dec)
    dec.insert(3, "year", year)

    # 5.5) If depth is provided, select shallower destination for RL
    if depth_m_by_tract:
        dec = _attach_rl_dest_by_depth(dec, depth_m_by_tract, rng=rng, k_best=rl_dest_k_best)

    # 6) Tract psychology updated for the year
    psy_new = psy_for_dec

    # 7) Dynamics: RL/BP effective for next year's population (including backfill); EH one-time rule
    dyn = (action_dyn or {})
    eh_one_time = bool(dyn.get("eh_one_time", False))
    if eh_one_time:
        if "eh_done" not in state.columns:
            state = state.copy()
            state["eh_done"] = False
        done_map = state.set_index("i")["eh_done"].reindex(dec["i"]).fillna(False).to_numpy()
        is_owner = dec["group"].str.lower().eq("owner").to_numpy()
        repeat_eh = is_owner & dec["action"].eq("EH").to_numpy() & done_map
        if repeat_eh.any():
            dec.loc[repeat_eh, "action"] = "DN"
            dec.loc[repeat_eh, "EH_recounted_as_DN"] = True

    state_next, changes = _apply_action_dynamics(
        dec,
        state,
        apply_rl=bool(dyn.get("apply_RL", True)),
        rl_backfill=bool(dyn.get("rl_backfill", True)),
        apply_bp=bool(dyn.get("apply_BP", True)),
        bp_backfill=bool(dyn.get("bp_backfill", False)),
        rng=rng,
        bp_sampler=dyn.get("bp_sampler", "median"),
        cv_nj=dyn.get("cv_nj", {"rcv": 0.694, "contents": 1.8}),
        eh_step_ft=float(dyn.get("eh_step_ft", 1.0)),
        eh_cap_ft=float(dyn.get("eh_cap_ft", 8.0)),
        eh_one_time=eh_one_time,
        eh_once_min_ft=float(dyn.get("eh_once_min_ft", 3.0)),
        eh_once_max_ft=float(dyn.get("eh_once_max_ft", 5.0)),
    )

    return dec, psy_new, state_next, changes


def run_loop_mgmix_fast(
    years: List[int],
    state: pd.DataFrame,
    tract_psy: pd.DataFrame,
    mg_share_by_tract: Dict[str,float],
    predictor_MG: Callable[..., Dict[str, np.ndarray]],
    predictor_NMG: Callable[..., Dict[str, np.ndarray]],
    params_MG: TPGroupParams,
    params_NMG: TPGroupParams,
    tp_cfg: TPConfig,
    ratio_provider: Callable[[int], Dict[str,float]],
    overlay_policy_names={"owner":"owner_standard","renter":"renter_contents"},
    seed: int = 2025,
) -> Dict[int, pd.DataFrame]:
    """Multi-year loop using precomputed indexer and MG share vector for speed."""
    rng = np.random.RandomState(seed)
    decisions_by_year: Dict[int, pd.DataFrame] = {}

    idxer = build_state_indexer(state, tract_psy)
    w_vec = state["tract_geoid"].astype(str).map(mg_share_by_tract).to_numpy(float)

    psy = tract_psy.copy()
    for y in years:
        ratio_y = ratio_provider(y) if ratio_provider else None
        dec, psy = run_one_year_mgmix_fast(
            year=y,
            state=state,
            tract_psy=psy,
            idxer=idxer,
            w_vec=w_vec,
            predictor_MG=predictor_MG, predictor_NMG=predictor_NMG,
            params_MG=params_MG, params_NMG=params_NMG,
            tp_cfg=tp_cfg,
            ratio_by_tract=ratio_y,
            overlay_policy_names=overlay_policy_names,
            rng=rng,
        )
        decisions_by_year[y] = dec
    return decisions_by_year


def _load_action_dynamics_default() -> dict:
    """
    Backward compatibility: If action_dyn is not passed from runner, try reading JSON.
    Your new project has switched to YAML; this is only for minimal compatibility.
    """
    try:
        cfg_path = Path(__file__).resolve().parent / "config" / "abm_params.json"
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            dyn = cfg.get("action_dynamics", {}) or {}
            return {
                "apply_RL":     bool(dyn.get("apply_RL", True)),
                "rl_backfill":  bool(dyn.get("rl_backfill", True)),
                "apply_BP":     bool(dyn.get("apply_BP", True)),
                "bp_backfill":  bool(dyn.get("bp_backfill", False)),
            }
    except Exception:
        pass
    return {"apply_RL": True, "rl_backfill": True, "apply_BP": True, "bp_backfill": False}


def _apply_action_dynamics(
    dec: pd.DataFrame,
    state: pd.DataFrame,
    *,
    apply_rl: bool = True,
    rl_backfill: bool = True,
    apply_bp: bool = True,
    bp_backfill: bool = False,
    rng: np.random.RandomState | None = None,
    bp_sampler: str = "median",                 # 'median' | 'lognorm_mean' | 'lognorm_from_totals'
    cv_nj: dict | None = None,                  # {'rcv': float, 'contents': float}
    eh_step_ft: float = 1.0,                    # Annual increment cap
    eh_cap_ft: float = 8.0,                     # Cumulative cap
    eh_one_time: bool = False,
    eh_once_min_ft: float = 3.0,
    eh_once_max_ft: float = 5.0,

) -> tuple[pd.DataFrame, dict]:
    """
    Realize RL / BP for the current year into "next year's household distribution".
    - RL: Move renter to DEST_TRACT; optionally backfill same number of new renters in original tract (using tract renter median contents)
          and scale original household contents after moving: Normal(0.8, 0.1) multiplier, clipped to [0, 1]
    - BP: Remove owner; optionally backfill new owners in original tract (sampling rule see bp_sampler)
    """
    rng = rng or np.random.RandomState(0)
    st = state.copy().reset_index(drop=True)
    changes = {"moved_rl": 0, "backfilled_rl": 0, "removed_bp": 0, "backfilled_bp": 0}
    max_i = int(st["i"].max()) if "i" in st.columns else 0


    # -------- EH (owner elevation) --------
    # Accumulate ELEV_FT from this year's decisions to next year's state.elev_ft_total, applying single/cumulative caps
    if "elev_ft_total" not in st.columns:
        st["elev_ft_total"] = 0.0
    if "eh_done" not in st.columns:
        st["eh_done"] = False  # New status column, persists across years

    if "action" in dec.columns:
        eh_rows = dec[(dec["group"].str.lower() == "owner") & (dec["action"] == "EH")].copy()
        if not eh_rows.empty:
            st = st.set_index("i")

            if eh_one_time:
                # One-time mode: ignore dec.ELEV_FT, only elevate households doing EH "for the first time"
                has_done = st["eh_done"].reindex(eh_rows["i"]).fillna(False).to_numpy()
                first_mask = ~has_done
                first_ids = eh_rows.loc[first_mask, "i"].astype(int).tolist()

                if len(first_ids) > 0:
                    # Elevation amount: Uniform[min,max]
                    add_ft = rng.uniform(float(eh_once_min_ft), float(eh_once_max_ft), size=len(first_ids))
                    base = pd.to_numeric(st.loc[first_ids, "elev_ft_total"], errors="coerce").fillna(0.0).to_numpy()
                    st.loc[first_ids, "elev_ft_total"] = base + add_ft
                    st.loc[first_ids, "eh_done"] = True
                    changes["elevated_eh"] = int(len(first_ids))

                # Households that have already done EH, selecting EH this year are re-labeled as DN in run_one_year (for statistics)
                # No further elevation here, and no cap applied (spec: cap not effective in one-time mode)

                else:
                    # Old mechanism: annual step, cap limits...
                    if "ELEV_FT" in eh_rows.columns:
                        eh_i = (
                            eh_rows.assign(ELEV_FT=pd.to_numeric(eh_rows["ELEV_FT"], errors="coerce").fillna(0.0))
                                   .groupby("i", as_index=True)["ELEV_FT"].sum()
                                   .clip(lower=0.0, upper=eh_step_ft)
                        )
                        base = pd.to_numeric(st.get("elev_ft_total", 0.0), errors="coerce").fillna(0.0)
                        remaining = (eh_cap_ft - base).clip(lower=0.0)
                        inc_applied = np.minimum(eh_i, remaining.reindex(eh_i.index).fillna(0.0))

                        base.loc[inc_applied.index] = base.loc[inc_applied.index] + inc_applied
                        st["elev_ft_total"] = base.clip(upper=eh_cap_ft)
                        changes["elevated_eh"] = int((inc_applied > 0).sum())

                        # New: As long as there is actual elevation this year, mark as EH done, future logic changes to FI->BP->DN
                        st.loc[inc_applied.index, "eh_done"] = True

            st = st.reset_index()

    # General safety check (cap check only needed for old mechanism; one-time mechanism does not check cap)
    if not eh_one_time:
        over_cap = (st["elev_ft_total"] > eh_cap_ft).sum()
        if over_cap:
            print(f"[warn] {over_cap} households exceed EH cap after clamp")

        
            
    # -------- RL (renter relocate) --------
    if apply_rl and ("action" in dec.columns):
        rl = dec[
            (dec["group"].str.lower() == "renter")
            & (dec["action"] == "RL")
            & dec.get("DEST_TRACT", pd.Series(dtype=object)).notna()
        ].copy()
        if not rl.empty:
            dest_map = rl.set_index("i")["DEST_TRACT"].astype(str).to_dict()
            st = st.set_index("i")
            idx = st.index.intersection(dest_map.keys())
            if len(idx) > 0:
                st.loc[idx, "tract_geoid"] = pd.Series(dest_map)
                changes["moved_rl"] = int(len(idx))
            st = st.reset_index()

            # RL scaling: Normal(0.8, 0.1), clipped to [0, 1]
            if len(idx) > 0 and "contents_kUSD" in st.columns:
                # mult = rng.normal(loc=0.8, scale=0.1, size=len(idx))
                # mult = np.clip(mult, 0.0, 1.0)
                mult = 1
                st = st.set_index("i")
                st.loc[idx, "contents_kUSD"] = (
                    pd.to_numeric(st.loc[idx, "contents_kUSD"], errors="coerce").fillna(0.0) * mult
                )
                st = st.reset_index()

            # Backfill new renters (median contents)
            if rl_backfill:
                renters = st[st["group"].str.lower() == "renter"]
                med_con_by_tract = renters.groupby("tract_geoid", dropna=False)["contents_kUSD"].median()
                out_counts = rl.groupby("tract_geoid").size().to_dict()  # How many moved out from original tract
                new_rows = []
                for t, k in out_counts.items():
                    t = str(t); k = int(k)
                    c_med = float(med_con_by_tract.get(t, renters["contents_kUSD"].median()))
                    cur_max_idx = renters.loc[renters["tract_geoid"] == t, "hh_idx_within_group"].max()
                    cur_max_idx = int(cur_max_idx) if np.isfinite(cur_max_idx) else 0
                    for j in range(k):
                        max_i += 1
                        new_rows.append({
                            "i": max_i, "group": "renter", "tract_geoid": t,
                            "rcv_kUSD": 0.0, "contents_kUSD": c_med,
                            "elev_ft_total": 0.0, "policy_name": "",
                            "i_orig": max_i,
                            "hh_idx_within_group": int(cur_max_idx + j + 1),
                        })
                if new_rows:
                    st = pd.concat([st, pd.DataFrame(new_rows)], ignore_index=True)
                    changes["backfilled_rl"] = int(len(new_rows))

    # -------- BP (owner buyout) --------
    if apply_bp and ("action" in dec.columns):
        bp = dec[(dec["group"].str.lower() == "owner") & (dec["action"] == "BP")]
        if not bp.empty:
            rm_ids = set(bp["i"].astype(int).tolist())

            if not bp_backfill:
                # No backfill: directly remove
                st = st[~st["i"].astype(int).isin(rm_ids)].reset_index(drop=True)
                changes["removed_bp"] = len(rm_ids)
            else:
                # ------- Owner backfill -------
                owners = st[st["group"].str.lower() == "owner"].copy()
                owners["tract_geoid"] = owners["tract_geoid"].astype(str)

                # Totals / counts for each tract before moving
                sum_rcv_by_tract = owners.groupby("tract_geoid", dropna=False)["rcv_kUSD"].sum()
                sum_con_by_tract = owners.groupby("tract_geoid", dropna=False)["contents_kUSD"].sum()
                n_own_by_tract  = owners.groupby("tract_geoid", dropna=False).size()

                # How many to backfill per tract
                add_counts = bp.groupby(bp["tract_geoid"].astype(str)).size().to_dict()

                # CV (from runner's CV_NJ)
                cv_rcv = float((cv_nj or {}).get("rcv", 0.694))
                cv_con = float((cv_nj or {}).get("contents", 1.8))
                sig_r  = float(np.sqrt(np.log1p(cv_rcv**2)))
                sig_c  = float(np.sqrt(np.log1p(cv_con**2)))

                def _mu_from_mean(mean_val: float, cv: float) -> float | None:
                    if not np.isfinite(mean_val) or mean_val <= 0 or cv <= 0:
                        return None
                    return float(np.log(mean_val) - 0.5*np.log1p(cv**2))

                # Lazy build (mu, sigma) for each tract (using "initial totals" method), module level cache
                global _BP_PARAMS_CACHE
                if _BP_PARAMS_CACHE is None:
                    _BP_PARAMS_CACHE = build_pertract_lognorm_params_from_totals(
                        state0=state,  # If you have true initial year DataFrame, can switch to that
                        tract_col="tract_geoid",
                        group_col="group",
                        value_col_owner="rcv_kUSD",
                        value_col_renter="contents_kUSD",
                        cv_owner=cv_rcv,
                        cv_renter=cv_con,
                    )
                pertract = _BP_PARAMS_CACHE

                add_rows = []
                max_i = int(st["i"].max()) if "i" in st.columns else 0

                for t, k in add_counts.items():
                    t = str(t); k = int(k)
                    N = int(n_own_by_tract.get(t, 0))
                    mean_rcv = float(sum_rcv_by_tract.get(t, 0.0)) / N if N > 0 else 0.0
                    mean_con = float(sum_con_by_tract.get(t, 0.0)) / N if N > 0 else 0.0

                    if bp_sampler == "lognorm_from_totals":
                        # per-tract mu/sigma (fallback to mean+CV if not found)
                        ent = pertract.get("owner", {}).get(t)
                        if ent:
                            mu_r, mu_c = float(ent["mu"]), None  # contents not used for owner, can also use ent
                            r_vals = rng.lognormal(mean=mu_r, sigma=float(ent["sigma"]), size=k).tolist()  
                        else:
                            mu_r = _mu_from_mean(mean_rcv, cv_rcv)
                            r_vals = (np.exp(mu_r + sig_r * rng.normal(size=k)).tolist()
                                      if mu_r is not None else [0.0] * k)

                        # owner contents: maintain tract mean or also use lognormal (here using mean+CV)
                        mu_c = _mu_from_mean(mean_con, cv_con)
                        c_vals = (np.exp(mu_c + sig_c * rng.normal(size=k)).tolist()
                                  if mu_c is not None else [0.0] * k)

                    elif bp_sampler == "lognorm_mean":
                        mu_r = _mu_from_mean(mean_rcv, cv_rcv)
                        mu_c = _mu_from_mean(mean_con, cv_con)
                        r_vals = (np.exp(mu_r + sig_r * rng.normal(size=k)).tolist()
                                  if mu_r is not None else [0.0] * k)
                        c_vals = (np.exp(mu_c + sig_c * rng.normal(size=k)).tolist()
                                  if mu_c is not None else [0.0] * k)
                    else:
                        # Median backfill (old behavior)
                        r_med = float(owners.loc[owners["tract_geoid"] == t, "rcv_kUSD"].median())
                        c_med = float(owners.loc[owners["tract_geoid"] == t, "contents_kUSD"].median())
                        r_vals = [r_med] * k
                        c_vals = [c_med] * k

                    # Continuous hh_idx_within_group
                    cur_max_idx = owners.loc[owners["tract_geoid"] == t, "hh_idx_within_group"].max()
                    cur_max_idx = int(cur_max_idx) if np.isfinite(cur_max_idx) else 0

                    for j in range(k):
                        max_i += 1
                        add_rows.append({
                            "i": max_i,
                            "group": "owner",
                            "tract_geoid": t,
                            "rcv_kUSD": float(r_vals[j]),
                            "contents_kUSD": float(c_vals[j]),
                            "elev_ft_total": 0.0,
                            "policy_name": "",
                            "i_orig": max_i,
                            "hh_idx_within_group": int(cur_max_idx + j + 1),
                        })

                # Remove bought-out first, then backfill new households
                st = st[~st["i"].astype(int).isin(rm_ids)].reset_index(drop=True)
                if add_rows:
                    st = pd.concat([st, pd.DataFrame(add_rows)], ignore_index=True)
                changes["removed_bp"] = len(rm_ids)
                changes["backfilled_bp"] = int(len(add_rows))


    return st, changes


def _attach_rl_dest_by_depth(dec: pd.DataFrame,
                             depth_m_by_tract: dict[str, float],
                             *,
                             rng: np.random.RandomState | None = None,
                             k_best: int = 1) -> pd.DataFrame:
    """
    For (renter, action==RL) rows, fill in DEST_TRACT:
      1) First find shallower candidates with depth_dest < depth_origin (sorted shallow->deep)
      2) If no shallower and origin_depth==0: pick one from "other 0-depth" tracts
      3) Pick 1 from top k_best (when k_best=1 pick the shallowest)
      4) Stay in place only if no destination available
    """
    if dec is None or dec.empty or not depth_m_by_tract:
        return dec

    rng = rng or np.random.RandomState(0)
    dec = dec.copy()
    if "DEST_TRACT" not in dec.columns:
        dec["DEST_TRACT"] = pd.NA

    is_rl = (dec["group"].str.lower() == "renter") & (dec["action"] == "RL")
    if not is_rl.any():
        return dec

    dec.loc[is_rl, "orig_depth_m"] = dec.loc[is_rl, "tract_geoid"].astype(str)\
        .map(lambda t: float(depth_m_by_tract.get(str(t), np.nan)))

    items = [(str(t), float(d)) for t, d in depth_m_by_tract.items() if np.isfinite(d)]
    if not items:
        return dec
    items.sort(key=lambda x: x[1])
    zero_depth_tracts = [t for (t, d) in items if d == 0.0]

    def _ridx(n: int) -> int:
        return int(rng.randint(0, n))

    for idx in dec.index[is_rl]:
        t0 = str(dec.at[idx, "tract_geoid"])
        d0 = dec.at[idx, "orig_depth_m"]
        d0 = float(d0) if np.isfinite(d0) else np.nan
        if not np.isfinite(d0):
            continue

        lessers = [(t, d) for (t, d) in items if d < d0]
        if lessers:
            k = max(1, int(k_best))
            pool = lessers[:k] if k < len(lessers) else lessers
            pick = pool[0] if k == 1 else pool[_ridx(len(pool))]
            choice = pick[0]
            dec.at[idx, "DEST_TRACT"]    = choice
            dec.at[idx, "dest_depth_m"]  = float(depth_m_by_tract[choice])
            dec.at[idx, "depth_delta_m"] = float(d0 - depth_m_by_tract[choice])
            continue

        if d0 == 0.0:
            alt_zero = [t for t in zero_depth_tracts if t != t0]
            if alt_zero:
                choice = alt_zero[_ridx(len(alt_zero))]
                dec.at[idx, "DEST_TRACT"]    = choice
                dec.at[idx, "dest_depth_m"]  = 0.0
                dec.at[idx, "depth_delta_m"] = 0.0
            else:
                dec.at[idx, "RL_no_alt_zero"] = True
            continue

        dec.at[idx, "RL_no_lower_depth"] = True

    return dec
