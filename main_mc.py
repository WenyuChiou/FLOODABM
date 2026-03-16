# -*- coding: utf-8 -*-
"""
main_mc.py — Monte Carlo over INITIAL psychology only (population-denominator version)

Outputs (PNG only):
  - outputs/montecarlo/mc_initpsy_runs.csv
  - outputs/montecarlo/mc_initpsy_seeds.csv
  - outputs/montecarlo/fig/cumulated_actions_combined.png   # Left: Median (Cumulative), Right: Std (Cumulative); Population denominator
  - outputs/montecarlo/fig/cumulated_finance_combined.png   # Left: Mean (Cumulative), Right: Std (Cumulative)
  - outputs/montecarlo/fig/actions_timeseries_iqr_merged.png# Top: actions (Population denominator); Bottom: EH (EOY)
"""

from __future__ import annotations
from pathlib import Path
import sys, os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.ticker import MaxNLocator

# ---------- robust paths ----------
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODULES_ROOT = ROOT / "modules"
ACTIONS_DIR  = MODULES_ROOT / "actions"
CONFIG_DIR   = ROOT / "config"
CONFIG_PATH  = CONFIG_DIR / "abm_params.yaml"

OUTPUT_DIR = Path(os.environ.get("FLOODABM_OUTPUT_ROOT", ROOT / "outputs")).resolve()
MC_DIR     = OUTPUT_DIR / "montecarlo"
FIG_DIR    = MC_DIR / "fig"
for p in (MC_DIR, FIG_DIR):
    p.mkdir(parents=True, exist_ok=True)

# ---------- project imports (keep unchanged) ----------
from utils._helpers import (
    load_yaml_cfg, resolve_path, years_from_cfg,
    load_households_csv, load_depths_legacy,
    init_tract_psych_safe, grp_params_from_yaml,
    load_inline_owner_share_policy, read_finance_from_yaml,
)
from modules.actions.tp import TPConfig, TPGroupParams
from modules.actions.decision import load_predictors, build_state_indexer, _init_fi_flags_from_yaml
from modules.actions.vuln_for_tp import ratio_by_tract_from_vuln, _attach_hh_flood_damage
from modules.actions.pipeline import run_one_year_mgmix_fast
from modules.finance import run_finance_for_year

# ---------- publication style ----------
# Okabe–Ito colorblind-safe palette
PALETTE = {
    "blue":   "#0072B2",
    "orange": "#E69F00",
    "green":  "#009E73",
    "red":    "#D55E00",
    "purple": "#CC79A7",
    "brown":  "#8B6B4F",
    "gray":   "#999999",
    "yellow": "#F0E442",
    "black":  "#000000",
}

# ---- Paper style helpers ----
def _set_style():
    plt.rcParams.update({
        "figure.dpi": 300,          # Common resolution for journals
        "savefig.dpi": 300,
        "font.size": 16,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "axes.linewidth": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.grid": True,
        "grid.alpha": 0.2,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
        # Can switch to serif (Times); will auto fallback if font not found
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "CMU Serif"]
    })

def _panel_label(ax, label="(a)", x=-0.12, y=1.02):
    """Label subplots with (a) (b) ... at top-left corner"""
    ax.text(x, y, label, transform=ax.transAxes, fontsize=14, fontweight="bold",
            ha="left", va="bottom")

def _safe_div(a, b): return (a / b) if (b and b != 0) else 0.0

def _format_pct_axis(ax, decimals=0):
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=decimals))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, prune=None))
    ax.tick_params(axis="both", which="major", length=4)

def _format_usd_axis(ax, decimals=0):
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.0f}'))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, prune=None))
    ax.tick_params(axis="both", which="major", length=4)

def _apply_axes_cosmetics(ax):
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

def _band(ax, x, p25, p75, color, label=None, z=1, min_band=0.0, draw_midline=False):
    # When IQR is almost 0, choose a given minimum visible bandwidth
    if min_band > 0.0:
        mid = (p25 + p75) / 2.0
        too_thin = (p75 - p25) < min_band
        p25 = np.where(too_thin, mid - min_band/2, p25)
        p75 = np.where(too_thin, mid + min_band/2, p75)

    # Only fill band, do not draw lines
    ax.fill_between(x, p25, p75, alpha=0.15, color=color, linewidth=0, zorder=z)

    # Turn on this switch only if an extra "midline of the band" is needed (default False)
    if draw_midline:
        ax.plot(x, (p25 + p75) / 2.0, color=color, label=label, zorder=z+1)


# ---------- Legend plugin: Right centered, white background with black frame ----------
def legend_outside_right(fig, axes=None, *, right_rect=0.82, x_anchor=0.80, ncol=1):
    """
    Collect legends from one or more subplots, deduplicate, and place "Right centered, outside plot".
    Reserve space on the right using rect, legend uses white background with black frame.
    """
    if axes is None:
        axes = fig.get_axes()
    elif not isinstance(axes, (list, tuple)):
        axes = [axes]
    handles, labels, seen = [], [], set()
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for hi, li in zip(h, l):
            if not li or li in seen:
                continue
            seen.add(li); handles.append(hi); labels.append(li)

    # Reserve space for right legend
    fig.tight_layout(rect=[0, 0, right_rect, 1])
    leg = fig.legend(handles, labels,
                     loc="center left", bbox_to_anchor=(x_anchor, 0.5),
                     ncol=ncol, frameon=True, fancybox=False, framealpha=1.0)
    leg.get_frame().set_edgecolor("black")
    leg.get_frame().set_facecolor("white")

# ---------- per-run simulation ----------
def run_once_initpsy(init_seed: int, crn_seed: int) -> tuple[dict, pd.DataFrame]:
    CFG  = load_yaml_cfg(CONFIG_PATH)
    FILES           = CFG.get("files", {}) or {}
    PATH_DEPTHS     = resolve_path(ACTIONS_DIR, FILES.get("depths_overall"), MODULES_ROOT)
    PATH_EVENTS     = resolve_path(ACTIONS_DIR, FILES.get("flood_events"),  MODULES_ROOT)
    PATH_HOUSEHOLDS = resolve_path(ACTIONS_DIR, FILES.get("households"),    MODULES_ROOT)
    if not PATH_HOUSEHOLDS or not PATH_HOUSEHOLDS.exists():
        raise FileNotFoundError("Missing households CSV (files.households) in YAML.")
    if not PATH_DEPTHS or not PATH_DEPTHS.exists():
        raise FileNotFoundError("Missing depths_overall in YAML.")

    TP_CFG_JSON  = CFG.get("tp_config", {}) or {}
    FLOOD_JSON   = CFG.get("flood", {}) or {}
    DYN          = CFG.get("action_dynamics") or {}

    FLOOD_RATIO_THRESHOLD = float(FLOOD_JSON.get("ratio_threshold", 1e-3))
    FFE_FT                = float(FLOOD_JSON.get("FFE_ft", 0.5))
    SHOCK_MIN_RATIO       = float(TP_CFG_JSON.get("shock_min_ratio", FLOOD_RATIO_THRESHOLD))

    PREMIUM_CFG_BY_STATE, DEFAULT_PREMIUM, OWNER_INS = read_finance_from_yaml(CFG)
    OWNER_CONTENTS_RATIO = float(OWNER_INS.get("owner_contents_ratio", 0.57))
    OWNER_INSURES_BOTH   = bool(OWNER_INS.get("owner_insures_both", True))

    OWNER_SHARE, policy_dict = load_inline_owner_share_policy(CFG)

    DEPTHS_LONG = load_depths_legacy(PATH_DEPTHS)
    YEARS = years_from_cfg(CFG, DEPTHS_LONG["year"].unique(), PATH_EVENTS)

    BASE_SEED = int(crn_seed)
    STATE = load_households_csv(PATH_HOUSEHOLDS)
    TRACTS = sorted(set(DEPTHS_LONG["tract_geoid"].astype(str)) | set(STATE["tract_geoid"].astype(str)))

    # Fixed parameters
    PARAMS_OWNER  = grp_params_from_yaml(CFG, "owner",  TPGroupParams)
    PARAMS_RENTER = grp_params_from_yaml(CFG, "renter", TPGroupParams)
    TP_CFG = TPConfig(
        shock_scale_owner  = float(TP_CFG_JSON.get("shock_scale_owner", 1.0)),
        shock_scale_renter = float(TP_CFG_JSON.get("shock_scale_renter", 1.0)),
        shock_timing       = TP_CFG_JSON.get("shock_timing", "start"),
        clip_lo            = float(TP_CFG_JSON.get("clip_lo", 0.0)),
        clip_hi            = float(TP_CFG_JSON.get("clip_hi", 1.0)),
    )
    setattr(TP_CFG, "shock_mode", TP_CFG_JSON.get("shock_mode", "additive"))
    setattr(TP_CFG, "flood_ratio_threshold", FLOOD_RATIO_THRESHOLD)

    # Re-draw household-level psychology using init_seed (Monte Carlo variation)
    from modules.actions.tp import BETA_PARAMS_OWNER_DEFAULT, BETA_PARAMS_RENTER_DEFAULT
    psych_rng = np.random.RandomState(int(init_seed))
    is_owner = STATE["group"].astype(str).str.lower().eq("owner")
    n_hh = len(STATE)
    for var in ["TP", "CP", "SP", "SC", "PA"]:
        col = f"{var}_init"
        vals = np.empty(n_hh, dtype=np.float64)
        a_o, b_o = BETA_PARAMS_OWNER_DEFAULT.get(var, (2.0, 2.0))
        n_own = int(is_owner.sum())
        if n_own > 0:
            vals[is_owner.values] = psych_rng.beta(
                max(a_o, 1e-6), max(b_o, 1e-6), size=n_own
            )
        a_r, b_r = BETA_PARAMS_RENTER_DEFAULT.get(var, (2.0, 2.0))
        n_ren = n_hh - n_own
        if n_ren > 0:
            vals[~is_owner.values] = psych_rng.beta(
                max(a_r, 1e-6), max(b_r, 1e-6), size=n_ren
            )
        STATE[col] = vals

    # Re-draw initial insurance status using init_seed
    ins_cfg = CFG.get("insurance_init", {}) or {}
    fi_rng = np.random.default_rng(int(init_seed) + 1_000_000)
    STATE["has_FI"] = 0
    rates = ins_cfg.get("take_rate_by_tract_group", {})
    if rates:
        tract_col = STATE["tract_geoid"].astype(str)
        group_col = STATE["group"].astype(str).str.lower()
        for tract_str, group_rates in rates.items():
            tract_str = str(tract_str).zfill(11)
            if not isinstance(group_rates, dict):
                continue
            for grp, rate in group_rates.items():
                rate = float(rate)
                if rate <= 0:
                    continue
                mask = tract_col.eq(tract_str) & group_col.eq(grp)
                idx = STATE.index[mask].to_numpy()
                n = len(idx)
                if n == 0:
                    continue
                k = max(0, min(int(round(rate * n)), n))
                if k > 0:
                    chosen = fi_rng.choice(idx, size=k, replace=False)
                    STATE.loc[chosen, "has_FI"] = 1

    STATE["TP"] = STATE["TP_init"].astype(float)
    STATE["t_clock"] = 0.0

    predictor_owner, predictor_renter = load_predictors(ACTIONS_DIR)

    # Cumulative containers (legacy kept; population denominator added)
    act_counts_overall = {"FI": 0, "BP": 0, "RL": 0, "DN": 0}
    act_counts_owner   = {"FI": 0, "BP": 0, "DN": 0}
    act_counts_renter  = {"FI": 0, "RL": 0, "DN": 0}
    act_total_overall_all = 0
    act_total_owner_all   = 0
    act_total_owner_allEH = 0
    act_total_renter_all  = 0

    # Population denominator accumulation (numerator / denominator)
    tot_owner_pop = tot_renter_pop = tot_overall_pop = 0
    sum_owner_FI  = sum_owner_BP = 0
    sum_renter_FI = sum_renter_RL = 0
    sum_overall_RL = sum_overall_DN = 0

    ph_payout_sum = ph_oop_sum = 0.0
    
    ph_n = 0

    all_payout_sum = 0.0
    all_oop_sum    = 0.0
    all_damage_sum = 0.0
    all_n_households = 0

    rows_ts = []
    last_eh_cov = 0.0

    # ========= Annual Loop =========
    for y in YEARS:
        depths_y = DEPTHS_LONG[DEPTHS_LONG["year"].astype(int) == int(y)]
        dmap = dict(zip(depths_y["tract_geoid"].astype(str), depths_y["depth_m"].astype(float)))
        ratio_raw = ratio_by_tract_from_vuln(
            state=STATE.assign(year=y),
            depths_long=DEPTHS_LONG, year=y,
            modules_root=MODULES_ROOT, FFE_ft=FFE_FT,
            owner_contents_ratio=OWNER_CONTENTS_RATIO
        )
        ratio_used = {t: (float(r) if float(r) >= SHOCK_MIN_RATIO else 0.0) for t, r in ratio_raw.items()}

        dec, STATE_NEXT, _chg = run_one_year_mgmix_fast(
            year=y, state=STATE,
            predictor_owner=predictor_owner, predictor_renter=predictor_renter,
            params_owner=PARAMS_OWNER, params_renter=PARAMS_RENTER, tp_cfg=TP_CFG,
            ratio_by_tract=ratio_used,
            overlay_policy_names={"owner": "owner_standard", "renter": "renter_contents"},
            rng=np.random.RandomState(int(init_seed) * 100 + int(y)),
            years_step=1.0, reset_clock_on_flood=False, action_dyn=DYN,
            depth_m_by_tract=dmap, rl_dest_k_best=50,
            decision_threshold=float(CFG.get("decision_threshold", 0.5)),
            draw_bounds=CFG.get("draw_bounds", None),
        )

        # ----- Action Stats: Taken this year + Population denominator -----
        ac_col = next((c for c in dec.columns if c.lower() in ("action","act","decision")), None)
        if ac_col is None:
            cand = [c for c in dec.columns if c.lower() in ("fi","bp","rl","dn","eh")]
            if cand:
                tmp = dec[cand].astype(float)
                chosen = tmp.idxmax(axis=1).str.upper()
                dec = dec.assign(__action=chosen)
                ac_col = "__action"

        id_col = next((c for c in dec.columns if c.lower() in ("group","identity")), None)
        ident = dec[id_col].astype(str).str.lower() if id_col else pd.Series(["unknown"]*len(dec), index=dec.index)

        acts   = dec[ac_col].astype(str).str.upper().map({"FI":"FI","BP":"BP","RL":"RL","DN":"DN","EH":"EH"})
        mask_o = ident.eq("owner"); mask_r = ident.eq("renter")

        owners_n  = int(mask_o.sum())           # Population denominator (owner this year)
        renters_n = int(mask_r.sum())           # Population denominator (renter this year)
        overall_n = int(len(dec))               # Population denominator (overall this year)

        # Numerator (Taken this year)
        fi_o = int((acts[mask_o] == "FI").sum())
        bp_o = int((acts[mask_o] == "BP").sum())
        dn_o = int((acts[mask_o] == "DN").sum())
        fi_r = int((acts[mask_r] == "FI").sum())
        rl_r = int((acts[mask_r] == "RL").sum())
        dn_r = int((acts[mask_r] == "DN").sum())
        fi_all = int((acts == "FI").sum())
        bp_all = int((acts == "BP").sum())
        rl_all = int((acts == "RL").sum())
        dn_all = int((acts == "DN").sum())

        # ---- EH Coverage (End of Year STATE_NEXT denominator) ----
        if "is_EH" not in STATE.columns: STATE["is_EH"] = 0
        if "is_EH" not in STATE_NEXT.columns: STATE_NEXT["is_EH"] = 0

        ident_next = STATE_NEXT.get("group", STATE_NEXT.get("identity", "unknown")).astype(str).str.lower()
        mask_owner_eoy = ident_next.eq("owner")
        idx_owner_eoy = STATE_NEXT.index[mask_owner_eoy]

        idx_intersect = idx_owner_eoy.intersection(STATE.index)
        prev_eh = STATE.loc[idx_intersect, "is_EH"].astype(int)
        STATE_NEXT.loc[idx_intersect, "is_EH"] = (STATE_NEXT.loc[idx_intersect, "is_EH"].astype(int) | prev_eh)

        eh_col = next((c for c in dec.columns if c.lower()=="eh" or c.upper()=="EH"), None)
        new_eh = (pd.to_numeric(dec[eh_col], errors="coerce").fillna(0) > 0) if eh_col else acts.eq("EH")
        idx_owner_t = dec.index[mask_o]
        took_eh_on_t = new_eh.reindex(idx_owner_t, fill_value=False).astype(int)
        idx_new_eh_eoy = idx_owner_eoy.intersection(idx_owner_t)
        STATE_NEXT.loc[idx_new_eh_eoy, "is_EH"] = (
            STATE_NEXT.loc[idx_new_eh_eoy, "is_EH"].astype(int) |
            took_eh_on_t.reindex(idx_new_eh_eoy, fill_value=0).astype(int)
        )

        owner_count_eoy = int(mask_owner_eoy.sum())
        nEH_o_eoy = int(STATE_NEXT.loc[mask_owner_eoy, "is_EH"].astype(int).sum())
        eh_cov_rate = nEH_o_eoy / max(1, owner_count_eoy)
        last_eh_cov = float(eh_cov_rate)

        # Legacy denominators (kept to avoid breaking external dependencies)
        den_o_act   = max(1, fi_o + bp_o)
        den_o_all   = max(1, fi_o + bp_o + dn_o)
        den_o_allEH = max(1, fi_o + bp_o + dn_o + nEH_o_eoy)
        den_r_all   = max(1, fi_r + rl_r + dn_r)
        den_all     = max(1, fi_all + bp_all + rl_all + dn_all)

        # Cumulative (Legacy)
        act_counts_owner["FI"] += fi_o; act_counts_owner["BP"] += bp_o; act_counts_owner["DN"] += dn_o
        act_counts_renter["FI"] += fi_r; act_counts_renter["RL"] += rl_r; act_counts_renter["DN"] += dn_r
        act_counts_overall["FI"] += fi_all; act_counts_overall["BP"] += bp_all
        act_counts_overall["RL"] += rl_all; act_counts_overall["DN"] += dn_all
        act_total_owner_all   += den_o_all
        act_total_owner_allEH += den_o_allEH
        act_total_renter_all  += den_r_all
        act_total_overall_all += den_all

        # Cumulative (Population denominator)
        tot_owner_pop   += owners_n
        tot_renter_pop  += renters_n
        tot_overall_pop += overall_n
        sum_owner_FI    += fi_o
        sum_owner_BP    += bp_o
        sum_renter_FI   += fi_r
        sum_renter_RL   += rl_r
        sum_overall_RL  += rl_all
        sum_overall_DN  += dn_all

        # ---- Annual timeseries ----
        rows_ts.append({
            "year": int(y),
            # Legacy columns (kept)
            "owner_share_FI":       float(fi_o / den_o_act),
            "owner_share_BP":       float(bp_o / den_o_act),
            "renter_share_FI":      float(fi_r / den_r_all),
            "renter_share_RL":      float(rl_r / den_r_all),
            "owner_share_FI_all":   float(fi_o / den_o_all),
            "owner_share_BP_all":   float(bp_o / den_o_all),
            "owner_share_FI_allEH": float(fi_o / den_o_allEH),
            "owner_share_BP_allEH": float(bp_o / den_o_allEH),
            # Population denominator (Key)
            "owner_share_FI_pop":   float(fi_o / max(1, owners_n)),
            "owner_share_BP_pop":   float(bp_o / max(1, owners_n)),
            "renter_share_FI_pop":  float(fi_r / max(1, renters_n)),
            "renter_share_RL_pop":  float(rl_r / max(1, renters_n)),
            "overall_share_RL_pop": float(rl_all / max(1, overall_n)),
            "overall_share_DN_pop": float(dn_all / max(1, overall_n)),
            # EH (End of Year coverage)
            "owner_eh_cov_count":   int(nEH_o_eoy),
            "owner_eh_cov_rate":    float(eh_cov_rate),
            "owner_eh_cov_share":   float(eh_cov_rate),
        })

        # ---------- finance ----------
        # Calculate damage first
        state_with_damage = _attach_hh_flood_damage(
            state_df=STATE.assign(year=y),
            year=y,
            depths_long=DEPTHS_LONG,
            modules_root=MODULES_ROOT,
            ffe_ft=FFE_FT,
            renters_have_structure=False
        )

        FIN_HH, _ = run_finance_for_year(
            year=y, state_event=state_with_damage, depth_map=dmap,
            ratio_by_tract=ratio_used, decisions=dec, policy=policy_dict,
            idxer=None, gate_by_decisions=True, renters_have_structure=False,
            tract_col="tract_geoid", output_dir=None, save_csv=False,
            premium={**DEFAULT_PREMIUM,
                     "contents_share_owner": OWNER_CONTENTS_RATIO,
                     "owner_insures_both": OWNER_INSURES_BOTH},
            owner_contents_ratio=OWNER_CONTENTS_RATIO,
            owner_insures_both=OWNER_INSURES_BOTH,
        )
        has_fi = (pd.to_numeric(FIN_HH.get("has_FI", 0), errors="coerce").fillna(0).astype(int) > 0)
        payout = pd.to_numeric(FIN_HH.get("payout_total_usd",
                                          FIN_HH.get("payout_total_kUSD", 0.0)*1000.0), errors="coerce").fillna(0.0)
        oop    = pd.to_numeric(FIN_HH.get("oop_total_usd",
                                          FIN_HH.get("oop_total_kUSD", 0.0)*1000.0), errors="coerce").fillna(0.0)
        ph_payout_sum += float(payout.where(has_fi).sum(skipna=True))
        ph_oop_sum    += float(oop.where(has_fi).sum(skipna=True))
        ph_n          += int(has_fi.sum())

        all_payout_sum   += float(payout.sum(skipna=True))
        all_oop_sum      += float(oop.sum(skipna=True))
        all_damage_sum   += float((payout + oop).sum(skipna=True))
        all_n_households += int(len(FIN_HH))
        # ---------- roll ----------
        # Use advance_state_for_next_year to handle backfill psych initialization
        from utils.main_helpers import advance_state_for_next_year
        STATE = advance_state_for_next_year(STATE, STATE_NEXT, y)

    # ========= End of Annual Loop =========
    summary = {
        # Legacy (kept)
        "share_FI_all": _safe_div(act_counts_overall["FI"], act_total_overall_all),
        "share_BP_all": _safe_div(act_counts_overall["BP"], act_total_overall_all),
        "share_RL_all": _safe_div(act_counts_overall["RL"], act_total_overall_all),
        "share_DN_all": _safe_div(act_counts_overall["DN"], act_total_overall_all),
        "owner_share_FI_all":  _safe_div(act_counts_owner["FI"],  act_total_owner_all),
        "owner_share_BP_all":  _safe_div(act_counts_owner["BP"],  act_total_owner_all),
        "owner_share_DN_all":  _safe_div(act_counts_owner["DN"],  act_total_owner_all),
        "owner_share_FI_allEH": _safe_div(act_counts_owner["FI"],  act_total_owner_allEH),
        "owner_share_BP_allEH": _safe_div(act_counts_owner["BP"],  act_total_owner_allEH),
        "renter_share_FI_all": _safe_div(act_counts_renter["FI"], act_total_renter_all),
        "renter_share_RL_all": _safe_div(act_counts_renter["RL"], act_total_renter_all),
        "renter_share_DN_all": _safe_div(act_counts_renter["DN"], act_total_renter_all),

        # Population denominator (Focus: for cumulative plots)
        "owner_share_FI_pop":   _safe_div(sum_owner_FI,   tot_owner_pop),
        "owner_share_BP_pop":   _safe_div(sum_owner_BP,   tot_owner_pop),
        "renter_share_FI_pop":  _safe_div(sum_renter_FI,  tot_renter_pop),
        "renter_share_RL_pop":  _safe_div(sum_renter_RL,  tot_renter_pop),
        "overall_share_RL_pop": _safe_div(sum_overall_RL, tot_overall_pop),
        "overall_share_DN_pop": _safe_div(sum_overall_DN, tot_overall_pop),

        # finance（policyholders）
        "avg_payout_usd_ph": _safe_div(ph_payout_sum, ph_n),
        "avg_oop_usd_ph":    _safe_div(ph_oop_sum,    ph_n),
        "ph_n": ph_n,
        "avg_payout_usd_all": _safe_div(all_payout_sum,   all_n_households),
        "avg_oop_usd_all":    _safe_div(all_oop_sum,      all_n_households),
        "avg_damage_usd_all": _safe_div(all_damage_sum,   all_n_households),
        # EH Coverage (Last year; owner)
        "owner_eh_cov_last": float(last_eh_cov),
    }

    ts_df = pd.DataFrame(rows_ts).sort_values("year")
    return summary, ts_df

# ---------- helpers ----------
def running_center_std(arr: np.ndarray, center: str = "mean"):
    centers, stds = [], []
    for i in range(1, len(arr)+1):
        v = arr[:i]
        c = float(np.median(v)) if center == "median" else float(np.mean(v))
        s = float(np.std(v, ddof=1)) if i > 1 else 0.0
        centers.append(c); stds.append(s)
    return np.array(centers), np.array(stds)

# ---------- driver ----------
def main(n_runs: int = 10, init_seed0: int = 71001, crn_seed: int = 2025):
    _set_style()

    runs, seeds, ts_list = [], [], []
    for i in range(n_runs):
        init_seed = init_seed0 + i*37
        print(f"[InitPsyMC] Run {i+1}/{n_runs} (init_seed={init_seed}) ...")
        s, ts = run_once_initpsy(init_seed=init_seed, crn_seed=crn_seed)
        s["run_id"] = i+1; runs.append(s)
        ts["run_id"] = i+1; ts_list.append(ts)
        seeds.append({"run_id": i+1, "init_seed": init_seed})

    df_runs  = pd.DataFrame(runs).set_index("run_id")
    df_seeds = pd.DataFrame(seeds).set_index("run_id")
    df_runs.to_csv(MC_DIR / "mc_initpsy_runs.csv",  encoding="utf-8-sig")
    df_seeds.to_csv(MC_DIR / "mc_initpsy_seeds.csv", encoding="utf-8-sig")
    print("[OK] Saved:", MC_DIR / "mc_initpsy_runs.csv")
    print("[OK] Saved:", MC_DIR / "mc_initpsy_seeds.csv")

    # Derived finance metrics
    df_runs["avg_damage_usd_ph"] = df_runs["avg_payout_usd_ph"].fillna(0.0) + df_runs["avg_oop_usd_ph"].fillna(0.0)

    # Merge TS
    TS = pd.concat(ts_list, ignore_index=True)

    # ============== Figure 1: Cumulated Actions (Population Denominator) ==============
    x = np.arange(1, len(df_runs) + 1)
    series = {
        "Renter – FI":  (df_runs["renter_share_FI_pop"].to_numpy(float),  PALETTE["green"]),
        "Owner  – FI":  (df_runs["owner_share_FI_pop"].to_numpy(float),   PALETTE["blue"]),
        "Owner  – BP":  (df_runs["owner_share_BP_pop"].to_numpy(float),   PALETTE["orange"]),
        "Overall – RL": (df_runs["overall_share_RL_pop"].to_numpy(float), PALETTE["red"]),
        "Overall – DN": (df_runs["overall_share_DN_pop"].to_numpy(float), PALETTE["purple"]),
    }

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.8, 5.2), sharex=True, sharey=False)

    # Left: running median
    for label, (y, c) in series.items():
        m, _ = running_center_std(y, center="median")
        axL.plot(x, m, label=label, color=c)
    axL.set_title("(a) Actions — median")
    axL.set_xlabel("Number of runs [-]"); axL.set_ylabel("Share [-]")
    _format_pct_axis(axL, decimals=1); _apply_axes_cosmetics(axL)

    # Right: running std
    for label, (y, c) in series.items():
        _, s = running_center_std(y, center="median")
        axR.plot(x, s, label=label, color=c)
    axR.set_title("(b) Actions — std")
    axR.set_xlabel("Number of runs [-]"); axR.set_ylabel("Share [-]")
    _format_pct_axis(axR, decimals=1); _apply_axes_cosmetics(axR)

    # Legend: Right centered, outside plot, white background with black frame
    legend_outside_right(fig, [axL, axR])
    fig.savefig(FIG_DIR / "cumulated_actions_combined.png")
    plt.close(fig)

    # ============== Figure 2: Cumulated Finance ==============
    finance_series = {
        "Payout (Policyholder)":       (df_runs["avg_payout_usd_ph"].to_numpy(float), PALETTE["blue"]),
        "OOP (Policyholder)":          (df_runs["avg_oop_usd_ph"].to_numpy(float),    PALETTE["orange"]),
        "Flood damage (ALL)": (df_runs["avg_damage_usd_ph"].to_numpy(float), PALETTE["green"]),
    }
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.8, 5.2), sharex=True, sharey=False)

    def run_stat(arr):
        centers, stds = [], []
        for i in range(1, len(arr)+1):
            v = arr[:i]
            centers.append(float(np.mean(v)))
            stds.append(float(np.std(v, ddof=1)) if i > 1 else 0.0)
        return np.array(centers), np.array(stds)

    for label, (y, c) in finance_series.items():
        mean_c, std_c = run_stat(y)
        axL.plot(x, mean_c, label=label, color=c)
        axR.plot(x, std_c,  label=label, color=c)

    for ax, ttl in [(axL, "(a) Finance — mean "),
                    (axR, "(b) Finance — std ")]:
        ax.set_title(ttl)
        ax.set_xlabel("Number of runs [-]")
        _format_usd_axis(ax); _apply_axes_cosmetics(ax)

    axL.set_ylabel("USD per household"); axR.set_ylabel("USD per household")

    # Legend: Right centered, outside plot, white background with black frame
    legend_outside_right(fig, [axL, axR])
    fig.savefig(FIG_DIR / "cumulated_finance_combined.png")
    plt.close(fig)

    # ============== Figure 3: Timeseries (Actions vs EH) ==============
    TS["year"] = pd.to_numeric(TS["year"], errors="coerce").astype("Int64")
    first_year = int(TS["year"].min())
    TS = TS[TS["year"] != first_year].copy()

    def agg_band(ts: pd.DataFrame, col: str) -> pd.DataFrame:
        df = ts[["year", col]].copy()
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["year", col])
        if df.empty: 
            return pd.DataFrame(columns=["year","median","p25","p75"])
        g = df.groupby("year")[col]
        return pd.DataFrame({
            "year":   g.median().index.astype(int),
            "median": g.median().to_numpy(float),
            "p25":    g.quantile(0.25).to_numpy(float),
            "p75":    g.quantile(0.75).to_numpy(float),
        })

    # Population denominator columns
    ow_fi = agg_band(TS, "owner_share_FI_pop")
    ow_bp = agg_band(TS, "owner_share_BP_pop")
    rt_fi = agg_band(TS, "renter_share_FI_pop")
    rt_rl = agg_band(TS, "renter_share_RL_pop")
    eh_ts = agg_band(TS, "owner_eh_cov_rate")  # EOY coverage

    # Vertical: Top=actions, Bottom=EH; Legend unified outside right
    fig, (axA, axE) = plt.subplots(2, 1, figsize=(10.8, 8.6), sharex=False, sharey=False)

    def draw_band(ax, dfb, color, label, *, minband=0.01):
        """Draw IQR shadow + median line; minband increases bandwidth to avoid invisible when IQR approx 0"""
        if dfb.empty:
            return
        x_year = dfb["year"].to_numpy()
        _band(ax, x_year, dfb["p25"].to_numpy(), dfb["p75"].to_numpy(),
            color=color, label=None, min_band=minband, draw_midline=False)
        ax.plot(x_year, dfb["median"].to_numpy(), color=color, lw=2.6, label=label)

    # Top: actions (4 lines; increase minband for BP, RL to make shadow visible)
    draw_band(axA, ow_fi, PALETTE["blue"],   "Owner – FI", minband=0.01)
    draw_band(axA, ow_bp, PALETTE["orange"], "Owner – BP", minband=0.015)  # <- Make band more visible
    draw_band(axA, rt_fi, PALETTE["green"],  "Renter – FI", minband=0.01)
    draw_band(axA, rt_rl, PALETTE["red"],    "Renter – RL", minband=0.015) # <- Make band more visible
    axA.set_title("(a) Actions by year — median ± IQR")
    axA.set_xlabel("Year"); axA.set_ylabel("Share [-]")
    _format_pct_axis(axA, decimals=0); _apply_axes_cosmetics(axA)

    # Auto adjust top plot y range (leave some padding)
    bands_left = [b for b in (ow_fi, ow_bp, rt_fi, rt_rl) if not b.empty]
    if bands_left:
        y_min = min(float(b["p25"].min()) for b in bands_left)
        y_max = max(float(b["p75"].max()) for b in bands_left)
        span  = max(1e-6, y_max - y_min)
        pad   = 0.12 * span
        y0    = max(0.0, y_min - pad)
        y1    = min(1.0, y_max + pad)
        if y1 - y0 < 0.2:
            mid = 0.5 * (y0 + y1)
            y0, y1 = max(0.0, mid - 0.1), min(1.0, mid + 0.1)
        axA.set_ylim(y0, y1)

    # Bottom: EH coverage (Force headroom to avoid line hitting the edge)
    draw_band(axE, eh_ts, PALETTE["purple"], "EH cov (EOY)", minband=0.01)
    axE.set_title("(b) EH coverage (EOY) by year — median ± IQR")
    axE.set_xlabel("Year"); axE.set_ylabel("Share [-]")
    _format_pct_axis(axE, decimals=0); _apply_axes_cosmetics(axE)

    # -> Give bottom plot more space: top edge +5%, max 1.0
    if not eh_ts.empty:
        y_all = np.r_[eh_ts["p25"].to_numpy(), eh_ts["p75"].to_numpy(), eh_ts["median"].to_numpy()]
        y_top = float(np.nanmax(y_all))
        axE.set_ylim(0.2, min(1.0, y_top + 0.05))

    # Right centered, outside plot legend (white background black frame)
    legend_outside_right(fig, [axA, axE])
    fig.savefig(FIG_DIR / "actions_timeseries_iqr_merged.png")
    plt.close(fig)


    print("[DONE] Figures saved:",
          FIG_DIR / "cumulated_actions_combined.png",
          FIG_DIR / "cumulated_finance_combined.png",
          FIG_DIR / "actions_timeseries_iqr_merged.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Monte Carlo simulation for initial psychology.")
    parser.add_argument("--runs", type=int, default=10, help="Number of Monte Carlo runs (default: 10)")
    parser.add_argument("--init-seed", type=int, default=71001, help="Initial seed for psychology (default: 71001)")
    parser.add_argument("--crn-seed", type=int, default=2025, help="Common random number seed for simulation (default: 2025)")
    
    args = parser.parse_args()
    
    # Check flow; increase n_runs for robustness (e.g. 30/50/100)
    main(n_runs=args.runs, init_seed0=args.init_seed, crn_seed=args.crn_seed)
