# -*- coding: utf-8 -*-
"""
Threat Perception (TP) Update Module
=====================================

This module implements the Threat Perception (TP) psychological model
for agent behavior in flood risk response. TP represents how much a household
perceives flood threats and affects their adoption of mitigation actions.

Key Concepts:
    - TP Range: [0, 1] where 0 = no threat perception, 1 = high threat perception
    - Decay: TP naturally decays over time without flood events
    - Shock: Flood events boost TP (threat becomes salient)
    
Model Components:
    - TPConfig: Global configuration (shock strength, clipping, timing)
    - TPGroupParams: Per-group decay parameters (owner=Homeowner vs renter=Renter)
    - Decay Formula: TP' = TP * exp(-λ * Δt) where λ depends on PA, SC, τ(t)
    - Shock Formula: TP' = min(1, TP + ratio) [additive mode]

Example:
    >>> from modules.actions.tp import TPConfig, TPGroupParams, update_tract_tp_owner_renter
    >>> cfg = TPConfig(shock_scale_owner=0.7)
    >>> params_owner = TPGroupParams(alpha=0.5, beta=0.3, tau0=2.0, tau_inf=10.0, k=0.1)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# =============================================================================
# Constants
# =============================================================================

LN2 = np.log(2.0)

# Default Beta distribution parameters for initial psychology sampling
# Format: (alpha, beta) fitted from survey data (extract_tenure_data.py Phase 1)
BETA_PARAMS_OWNER_DEFAULT = {
    "TP": (6.270198, 4.635320),   # n=557, from Q19_1-Q19_11 (11 items)
    "CP": (7.857397, 5.051820),   # n=557, from Q21_1-2 + Q22_1,2,4,5,7,8 (8 items)
    "SP": (3.398335, 2.752034),   # n=557, from Q22_3, Q22_6, Q22_9 (3 items)
    "SC": (4.773929, 1.689776),   # n=557, from Q18_1-Q18_6 (6 items)
    "PA": (5.539009, 2.941853),   # n=557, from Q18_7-Q18_15 (9 items)
}

BETA_PARAMS_RENTER_DEFAULT = {
    "TP": (3.713441, 2.668174),   # n=379, from Q19_1-Q19_11 (11 items)
    "CP": (3.824367, 2.164083),   # n=379, from Q21_1-2 + Q22_1,2,4,5,7,8 (8 items)
    "SP": (1.759103, 1.103402),   # n=379, from Q22_3, Q22_6, Q22_9 (3 items)
    "SC": (2.794319, 1.035687),   # n=379, from Q18_1-Q18_6 (6 items)
    "PA": (2.512725, 1.322463),   # n=379, from Q18_7-Q18_15 (9 items)
}


# =============================================================================
# Configuration Classes
# =============================================================================

class TPConfig:
    """Global Trust in Policymaker configuration.
    
    Controls how flood shocks affect TP and value clipping.
    
    Attributes:
        shock_scale_owner: Shock multiplier for homeowners (default: 0.7)
        shock_scale_renter: Shock multiplier for renters (default: 0.7)
        clip_lo: Minimum TP value (default: 0.0)
        clip_hi: Maximum TP value (default: 1.0)
        shock_timing: "start" (shock before decay) or "end" (decay before shock)
    """
    
    def __init__(
        self,
        shock_scale_owner: float = 0.7,
        shock_scale_renter: float = 0.7,
        clip_lo: float = 0.0,
        clip_hi: float = 1.0,
        shock_timing: str = "start",
    ):
        self.shock_scale_owner = float(shock_scale_owner)
        self.shock_scale_renter = float(shock_scale_renter)
        self.clip_lo = float(clip_lo)
        self.clip_hi = float(clip_hi)
        self.shock_timing = str(shock_timing).lower()


class TPGroupParams:
    """TP decay parameters for a tenure group (owner or renter).
    
    The decay model uses a time-varying characteristic time τ(t):
        τ(t) = τ_inf - (τ_inf - τ0) * e^(-k*t)
    
    Where:
        τ0: Initial characteristic time (fast decay)
        τ_inf: Asymptotic characteristic time (slow decay)
        k: Rate of transition from τ0 to τ_inf
    
    Attributes:
        alpha: Weight for Policy Awareness (PA) in decay rate
        beta: Weight for Social Capital (SC) in decay rate
        tau0: Initial characteristic time (years)
        tau_inf: Asymptotic characteristic time (years)
        k: Transition rate parameter
    """
    
    def __init__(
        self,
        alpha: float,
        beta: float,
        tau0: float,
        tau_inf: float,
        k: float,
    ):
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.tau0 = float(tau0)
        self.tau_inf = float(tau_inf)
        self.k = float(k)


# =============================================================================
# Decay Model Functions
# =============================================================================

def _eff_closed_form(t, tau0: float, tau_inf: float, k: float) -> np.ndarray:
    """Compute effective decay integral using closed-form solution.
    
    Computes: Eff(t) = ∫₀ᵗ 1/τ(u) du
    
    Where τ(u) = τ_inf - (τ_inf - τ0) * e^(-k*u)
    
    Closed form: Eff(t) = t/τ_inf + (1/(k*τ_inf)) * ln((τ(t))/τ0)
    """
    t = np.asarray(t, dtype=float)
    num = tau_inf - (tau_inf - tau0) * np.exp(-k * np.clip(t, 0.0, None))
    num = np.maximum(num, 1e-12)
    tau0s = max(tau0, 1e-12)
    return (t / tau_inf) + (1.0 / (k * tau_inf)) * np.log(num / tau0s)


def _eff_interval(t0: float, t1: float, tau0: float, tau_inf: float, k: float) -> float:
    """Compute effective decay over time interval [t0, t1]."""
    return _eff_closed_form(t1, tau0, tau_inf, k) - _eff_closed_form(t0, tau0, tau_inf, k)


def _decay_step(
    TP: np.ndarray,
    PA: np.ndarray,
    SC: np.ndarray,
    t0: float,
    t1: float,
    params: TPGroupParams,
    clip_lo: float = 0.0,
    clip_hi: float = 1.0,
) -> np.ndarray:
    """Apply one decay step to TP values.
    
    Decay formula: TP' = TP * exp(-ln(2) * w * Eff)
    Where: w = α*(1-PA) + β*SC
    Note: Higher PA (Policy Awareness) → SLOWER decay (more trust in government response)
    
    Args:
        TP: Current TP values
        PA: Policy Awareness values
        SC: Social Capital values
        t0: Start time
        t1: End time
        params: Decay parameters for this group
        clip_lo: Minimum value after decay
        clip_hi: Maximum value after decay
        
    Returns:
        Updated TP values
    """
    # Higher PA means slower decay: (1-PA) term
    w = params.alpha * (1.0 - np.asarray(PA, float)) + params.beta * np.asarray(SC, float)
    Eff = _eff_interval(t0, t1, params.tau0, params.tau_inf, params.k)
    TPn = np.asarray(TP, float) * np.exp(-LN2 * w * Eff)
    return np.clip(TPn, clip_lo, clip_hi)


# =============================================================================
# Shock Functions
# =============================================================================

def _apply_shock(
    TP: np.ndarray,
    ratio: np.ndarray,
    scale: float,
    clip_lo: float,
    clip_hi: float,
    mode: str = "additive",
) -> np.ndarray:
    """Apply flood shock to TP values.
    
    Flood events increase TP by making government policy salient.
    
    Modes:
        "additive": TP' = min(1, TP + scale * ratio)
        "toward_one": TP' = TP + (1-TP) * [1 - (1-ratio)^scale]
    
    Args:
        TP: Current TP values
        ratio: Flood damage ratio per tract (0-1)
        scale: Shock strength multiplier
        clip_lo: Minimum after shock
        clip_hi: Maximum after shock
        mode: Shock application mode
        
    Returns:
        Updated TP values
    """
    r = np.clip(ratio, 0.0, 1.0).astype(float)
    if mode == "additive":
        eff = np.clip(scale * r, 0.0, 1.0)
        TP_new = TP + eff
    else:  # "toward_one"
        s = max(float(scale), 1e-6)
        f = 1.0 - np.power(1.0 - r, s)
        TP_new = TP + (1.0 - TP) * f
    return np.clip(TP_new, clip_lo, clip_hi)


# =============================================================================
# Tract Psychology Initialization
# =============================================================================

def init_tract_psych_owner_renter(
    tracts: list[str],
    rng: np.random.RandomState | None = None,
    mode: str = "beta",
    beta_params_owner: dict[str, tuple[float, float]] | None = None,
    beta_params_renter: dict[str, tuple[float, float]] | None = None,
) -> pd.DataFrame:
    """Initialize tract-level psychology for both owner and renter groups.

    Creates initial values for TP, CP, SP, SC, PA using Beta distributions.

    Args:
        tracts: List of census tract geoid strings
        rng: Random state for reproducibility
        mode: Initialization mode (currently only "beta" supported)
        beta_params_owner: Custom Beta parameters for owner group
        beta_params_renter: Custom Beta parameters for renter group

    Returns:
        DataFrame with columns:
        - tract_geoid
        - TP_owner, CP_owner, SP_owner, SC_owner, PA_owner
        - TP_renter, CP_renter, SP_renter, SC_renter, PA_renter
        - t_clock_owner, t_clock_renter (time since last flood)
    """
    rng = rng or np.random.RandomState(2025)
    cols = ["TP", "CP", "SP", "SC", "PA"]
    beta_params_owner = (beta_params_owner or BETA_PARAMS_OWNER_DEFAULT).copy()
    beta_params_renter = (beta_params_renter or BETA_PARAMS_RENTER_DEFAULT).copy()

    def _draw_one(beta_params: dict[str, tuple[float, float]]) -> dict[str, float]:
        """Draw one set of psychological values from Beta distributions."""
        out = {}
        for c in cols:
            a, b = beta_params.get(c, (2.0, 2.0))
            a = max(float(a), 1e-6)
            b = max(float(b), 1e-6)
            out[c] = rng.beta(a, b)
        return out

    rows = []
    for t in tracts:
        own = _draw_one(beta_params_owner)
        ren = _draw_one(beta_params_renter)
        rows.append({
            "tract_geoid": str(t),
            **{f"{c}_owner": own[c] for c in cols},
            **{f"{c}_renter": ren[c] for c in cols},
            "t_clock_owner": 0.0,
            "t_clock_renter": 0.0,
        })
    return pd.DataFrame(rows)


# Backward-compatible alias (deprecated — use init_tract_psych_owner_renter)
init_tract_psych_mg_nmg = init_tract_psych_owner_renter


# =============================================================================
# Flood Ratio Calculation
# =============================================================================

def build_ratio_by_tract_totaldamage(
    tract_loss_df: pd.DataFrame,
    state_value_df: pd.DataFrame,
    year: int,
) -> dict[str, float]:
    """Build flood damage ratio by tract for a given year.
    
    Ratio = total_loss / total_value per tract (0-1 scale).
    
    Args:
        tract_loss_df: DataFrame with tract losses (tract_geoid, year, total_loss_kUSD)
        state_value_df: DataFrame with household values (tract_geoid, rcv_kUSD, contents_kUSD)
        year: Year to calculate ratio for
        
    Returns:
        Dict mapping tract_geoid to damage ratio
    """
    val = state_value_df.copy()
    val["value"] = val["rcv_kUSD"].fillna(0) + val["contents_kUSD"].fillna(0)
    gval = val.groupby("tract_geoid", as_index=False)["value"].sum()
    
    loss = tract_loss_df.loc[
        tract_loss_df["year"] == year,
        ["tract_geoid", "total_loss_kUSD"]
    ].copy()
    
    g = gval.merge(loss, on="tract_geoid", how="left").fillna({"total_loss_kUSD": 0.0})
    g["ratio"] = np.where(g["value"] > 0, g["total_loss_kUSD"] / g["value"], 0.0).clip(0.0, 1.0)
    
    return dict(zip(g["tract_geoid"].astype(str), g["ratio"].astype(float)))


# =============================================================================
# Main TP Update Function
# =============================================================================

def update_tract_tp_owner_renter(
    tract_psych: pd.DataFrame,
    params_owner: TPGroupParams,
    params_renter: TPGroupParams,
    cfg: TPConfig,
    ratio_by_tract: dict[str, float],
    shock_if_ratio_gt0: bool = True,
    years_step: float = 1.0,
    reset_clock_on_flood: bool = False,
    flood_ratio_threshold: float = 0.0,
    flood_depth_m_by_tract: dict[str, float] | None = None,
    flood_depth_threshold_m: float | None = None,
) -> pd.DataFrame:
    """Update TP values for all tracts for one time step.

    Applies decay and optional flood shock to owner and renter TP values.

    Update order depends on cfg.shock_timing:
        "start": shock -> decay
        "end": decay -> shock

    Args:
        tract_psych: Current tract psychology DataFrame
        params_owner: Decay parameters for homeowner group
        params_renter: Decay parameters for renter group
        cfg: Global TP configuration
        ratio_by_tract: Flood damage ratio per tract
        shock_if_ratio_gt0: Whether to apply shock if ratio > 0
        years_step: Time step size in years
        reset_clock_on_flood: Reset decay clock on flood event
        flood_ratio_threshold: Minimum ratio to trigger flood
        flood_depth_m_by_tract: Optional depth-based flood trigger
        flood_depth_threshold_m: Depth threshold for flood trigger

    Returns:
        Updated tract psychology DataFrame
    """
    df = tract_psych.copy()

    thr_owner = float(getattr(cfg, "thr_owner", flood_ratio_threshold))
    thr_renter = float(getattr(cfg, "thr_renter", flood_ratio_threshold))
    use_depth = (flood_depth_m_by_tract is not None) and (flood_depth_threshold_m is not None)
    shock_mode = getattr(cfg, "shock_mode", "additive")
    timing = getattr(cfg, "shock_timing", "start")
    skip_decay_on_shock = bool(getattr(cfg, "skip_decay_on_shock", False))
    shock_scale_own = float(getattr(cfg, "shock_scale_owner", 1.0))
    shock_scale_ren = float(getattr(cfg, "shock_scale_renter", 1.0))
    clip_lo = float(getattr(cfg, "clip_lo", 0.0))
    clip_hi = float(getattr(cfg, "clip_hi", 1.0))

    # --- Extract arrays ---
    tracts = df["tract_geoid"].astype(str).to_numpy()
    ratio_arr = np.array([float(ratio_by_tract.get(t, 0.0)) for t in tracts])

    tp_own = df["TP_owner"].to_numpy(dtype=float)
    tp_ren = df["TP_renter"].to_numpy(dtype=float)
    pa_own = df["PA_owner"].to_numpy(dtype=float)
    sc_own = df["SC_owner"].to_numpy(dtype=float)
    pa_ren = df["PA_renter"].to_numpy(dtype=float)
    sc_ren = df["SC_renter"].to_numpy(dtype=float)
    if "t_clock_owner" not in df.columns or "t_clock_renter" not in df.columns:
        raise KeyError("tract_psych is missing t_clock columns — was init_tract_psych_owner_renter() called?")
    t_own = np.nan_to_num(df["t_clock_owner"].to_numpy(dtype=float), nan=0.0)
    t_ren = np.nan_to_num(df["t_clock_renter"].to_numpy(dtype=float), nan=0.0)

    # --- Determine flood masks ---
    if use_depth:
        depth_arr = np.array([float(flood_depth_m_by_tract.get(t, 0.0)) for t in tracts])
        flood_own = flood_ren = (depth_arr > float(flood_depth_threshold_m))
    elif shock_if_ratio_gt0:
        flood_own = ratio_arr >= thr_owner
        flood_ren = ratio_arr >= thr_renter
    else:
        flood_own = flood_ren = np.zeros(len(df), dtype=bool)

    # --- Vectorized decay helper (uses existing _decay_step) ---
    def _do_decay(tp, pa, sc, t_clock, params, skip_mask):
        """Apply decay to all tracts, skipping where skip_mask is True."""
        decayed = _decay_step(tp, pa, sc, t_clock, t_clock + years_step,
                              params, clip_lo=clip_lo, clip_hi=clip_hi)
        return np.where(skip_mask, tp, decayed)

    # --- Vectorized shock helper (uses existing _apply_shock) ---
    def _do_shock(tp, flood_mask, scale):
        """Apply shock where flood_mask is True."""
        if not flood_mask.any():
            return tp
        shocked = _apply_shock(tp, ratio_arr, scale, clip_lo, clip_hi, mode=shock_mode)
        return np.where(flood_mask, shocked, tp)

    # --- Apply updates based on timing ---
    skip_own = flood_own & skip_decay_on_shock
    skip_ren = flood_ren & skip_decay_on_shock

    if timing == "start":
        # Shock first, then decay
        tp_own = _do_shock(tp_own, flood_own, shock_scale_own)
        tp_ren = _do_shock(tp_ren, flood_ren, shock_scale_ren)
        tp_own = _do_decay(tp_own, pa_own, sc_own, t_own, params_owner, skip_own)
        tp_ren = _do_decay(tp_ren, pa_ren, sc_ren, t_ren, params_renter, skip_ren)
    else:
        # Decay first, then shock (end mode)
        tp_own = _do_decay(tp_own, pa_own, sc_own, t_own, params_owner, skip_own)
        tp_ren = _do_decay(tp_ren, pa_ren, sc_ren, t_ren, params_renter, skip_ren)
        tp_own = _do_shock(tp_own, flood_own, shock_scale_own)
        tp_ren = _do_shock(tp_ren, flood_ren, shock_scale_ren)

    # --- Update time clocks ---
    t_own = t_own + years_step
    t_ren = t_ren + years_step
    if reset_clock_on_flood:
        t_own = np.where(flood_own, 0.0, t_own)
        t_ren = np.where(flood_ren, 0.0, t_ren)

    # --- Write back ---
    df["TP_owner"] = tp_own
    df["TP_renter"] = tp_ren
    df["t_clock_owner"] = t_own
    df["t_clock_renter"] = t_ren

    return df


# Backward-compatible alias (deprecated — use update_tract_tp_owner_renter)
update_tract_tp_mg_nmg = update_tract_tp_owner_renter


# =============================================================================
# Household-Level TP Update
# =============================================================================

def update_household_tp(
    state: pd.DataFrame,
    params_owner: TPGroupParams,
    params_renter: TPGroupParams,
    cfg: TPConfig,
    ratio_by_tract: dict[str, float],
    *,
    shock_if_ratio_gt0: bool = True,
    years_step: float = 1.0,
    reset_clock_on_flood: bool = False,
    flood_ratio_threshold: float = 0.0,
) -> pd.DataFrame:
    """Update TP at the household level (not tract level).

    Each household has its own TP, PA, SC, t_clock values.
    Decay and shock use the same math as the tract-level version,
    but are applied to per-household arrays vectorized by group.

    Args:
        state: Household DataFrame with columns TP, PA_init, SC_init, t_clock, group, tract_geoid
        params_owner: Decay parameters for owner group
        params_renter: Decay parameters for renter group
        cfg: TPConfig with shock scales, thresholds, timing, etc.
        ratio_by_tract: Flood damage ratio per tract
        shock_if_ratio_gt0: Whether to apply shock when ratio > 0
        years_step: Time step (years)
        reset_clock_on_flood: Reset t_clock on flood event
        flood_ratio_threshold: Minimum ratio to trigger flood

    Returns:
        Updated state DataFrame (TP and t_clock modified in-place on copy)
    """
    df = state.copy()

    thr_owner = float(getattr(cfg, "thr_owner", flood_ratio_threshold))
    thr_renter = float(getattr(cfg, "thr_renter", flood_ratio_threshold))
    shock_mode = getattr(cfg, "shock_mode", "additive")
    timing = getattr(cfg, "shock_timing", "start")
    skip_decay_on_shock = bool(getattr(cfg, "skip_decay_on_shock", False))
    shock_scale_own = float(getattr(cfg, "shock_scale_owner", 1.0))
    shock_scale_ren = float(getattr(cfg, "shock_scale_renter", 1.0))
    clip_lo = float(getattr(cfg, "clip_lo", 0.0))
    clip_hi = float(getattr(cfg, "clip_hi", 1.0))

    # --- Lookup per-household flood ratio from tract ---
    tracts = df["tract_geoid"].astype(str).to_numpy()
    ratio_arr = np.array([float(ratio_by_tract.get(t, 0.0)) for t in tracts])

    is_owner = df["group"].astype(str).str.lower().eq("owner").to_numpy()
    is_renter = ~is_owner

    tp = df["TP"].to_numpy(dtype=float).copy()
    pa = df["PA_init"].to_numpy(dtype=float)
    sc = df["SC_init"].to_numpy(dtype=float)
    t_clock = df["t_clock"].to_numpy(dtype=float).copy()

    # --- Flood masks (per-household, based on their tract's ratio) ---
    if shock_if_ratio_gt0:
        flood_own = is_owner & (ratio_arr >= thr_owner)
        flood_ren = is_renter & (ratio_arr >= thr_renter)
    else:
        flood_own = np.zeros(len(df), dtype=bool)
        flood_ren = np.zeros(len(df), dtype=bool)

    # --- Helpers ---
    def _do_decay_hh(mask, params, skip_mask):
        """Decay TP for households in `mask`, skipping `skip_mask`."""
        idx = np.where(mask)[0]
        if len(idx) == 0:
            return
        decayed = _decay_step(
            tp[idx], pa[idx], sc[idx],
            t_clock[idx], t_clock[idx] + years_step,
            params, clip_lo=clip_lo, clip_hi=clip_hi,
        )
        apply = ~skip_mask[idx]
        tp[idx[apply]] = decayed[apply]

    def _do_shock_hh(flood_mask, scale):
        """Shock TP for households where flood_mask is True."""
        idx = np.where(flood_mask)[0]
        if len(idx) == 0:
            return
        shocked = _apply_shock(
            tp[idx], ratio_arr[idx], scale,
            clip_lo, clip_hi, mode=shock_mode,
        )
        tp[idx] = shocked

    # --- Apply based on timing ---
    skip_own = flood_own & skip_decay_on_shock
    skip_ren = flood_ren & skip_decay_on_shock

    if timing == "start":
        _do_shock_hh(flood_own, shock_scale_own)
        _do_shock_hh(flood_ren, shock_scale_ren)
        _do_decay_hh(is_owner, params_owner, skip_own)
        _do_decay_hh(is_renter, params_renter, skip_ren)
    else:
        _do_decay_hh(is_owner, params_owner, skip_own)
        _do_decay_hh(is_renter, params_renter, skip_ren)
        _do_shock_hh(flood_own, shock_scale_own)
        _do_shock_hh(flood_ren, shock_scale_ren)

    # --- Update clocks ---
    t_clock += years_step
    if reset_clock_on_flood:
        t_clock[flood_own | flood_ren] = 0.0

    df["TP"] = tp
    df["t_clock"] = t_clock

    return df
