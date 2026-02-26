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
    >>> from modules.actions.mgmix_tp import TPConfig, TPGroupParams, update_tract_tp_owner_renter
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
    "TP": (6.123, 4.480),   # n=694
    "CP": (5.430, 3.333),
    "SP": (2.666, 2.014),
    "SC": (3.368, 1.155),
    "PA": (3.575, 1.850),
}

BETA_PARAMS_RENTER_DEFAULT = {
    "TP": (4.765, 3.364),   # n=243
    "CP": (5.236, 3.155),
    "SP": (2.142, 1.474),
    "SC": (3.996, 1.576),
    "PA": (2.762, 1.428),
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


# Backward-compatible alias
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
    
    return {str(r.tract_geoid): float(r.ratio) for _, r in g.iterrows()}


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

    def _tau(t: float, tau0: float, tau_inf: float, k: float) -> float:
        """Time-varying characteristic decay time."""
        return tau_inf - (tau_inf - tau0) * np.exp(-k * t)

    def _decay(tp: float, t_clock: float, grp: TPGroupParams, pa: float, sc: float) -> float:
        """Apply exponential decay to TP value."""
        tau_t = _tau(t_clock, grp.tau0, grp.tau_inf, grp.k)
        mu = np.log(2.0) * (grp.alpha * (1.0 - pa) + grp.beta * sc) / max(tau_t, 1e-6)
        return float(tp * np.exp(-mu * years_step))

    use_depth = (flood_depth_m_by_tract is not None) and (flood_depth_threshold_m is not None)
    shock_mode = getattr(cfg, "shock_mode", "additive")
    timing = getattr(cfg, "shock_timing", "start")

    for i, r in df.iterrows():
        tract = str(r["tract_geoid"])
        ratio = float(ratio_by_tract.get(tract, 0.0))

        # Determine if flood occurred
        if use_depth:
            depth_m = float(flood_depth_m_by_tract.get(tract, 0.0))
            is_flood_owner = is_flood_renter = (depth_m > float(flood_depth_threshold_m))
        else:
            if shock_if_ratio_gt0:
                is_flood_owner = (ratio >= thr_owner)
                is_flood_renter = (ratio >= thr_renter)
            else:
                is_flood_owner = is_flood_renter = False

        # Extract current values
        tp_own = float(r["TP_owner"])
        tp_ren = float(r["TP_renter"])
        pa_own, sc_own = float(r["PA_owner"]), float(r["SC_owner"])
        pa_ren, sc_ren = float(r["PA_renter"]), float(r["SC_renter"])
        t_own = float(r.get("t_clock_owner", 0.0))
        t_ren = float(r.get("t_clock_renter", 0.0))

        # Get shock scale from config
        shock_scale_owner = float(getattr(cfg, 'shock_scale_owner', 1.0))
        shock_scale_renter = float(getattr(cfg, 'shock_scale_renter', 1.0))

        def _shock(tp: float, is_owner: bool = True) -> float:
            """Apply shock to TP using scale from config."""
            scale = shock_scale_owner if is_owner else shock_scale_renter
            if shock_mode == "additive":
                return min(1.0, tp + scale * ratio)
            return min(1.0, tp)

        # Get skip_decay_on_shock option
        skip_decay_on_shock = bool(getattr(cfg, "skip_decay_on_shock", False))

        # Apply updates based on timing
        if timing == "start":
            # Shock first, then decay
            if is_flood_owner:
                tp_own = _shock(tp_own, is_owner=True)
            if is_flood_renter:
                tp_ren = _shock(tp_ren, is_owner=False)
            if not (is_flood_owner and skip_decay_on_shock):
                tp_own = max(0.0, min(1.0, _decay(tp_own, t_own, params_owner, pa_own, sc_own)))
            if not (is_flood_renter and skip_decay_on_shock):
                tp_ren = max(0.0, min(1.0, _decay(tp_ren, t_ren, params_renter, pa_ren, sc_ren)))
        else:
            # Decay first, then shock (end mode)
            if not (is_flood_owner and skip_decay_on_shock):
                tp_own = max(0.0, min(1.0, _decay(tp_own, t_own, params_owner, pa_own, sc_own)))
            if not (is_flood_renter and skip_decay_on_shock):
                tp_ren = max(0.0, min(1.0, _decay(tp_ren, t_ren, params_renter, pa_ren, sc_ren)))
            if is_flood_owner:
                tp_own = _shock(tp_own, is_owner=True)
            if is_flood_renter:
                tp_ren = _shock(tp_ren, is_owner=False)

        # Update time clocks
        t_own += years_step
        t_ren += years_step
        if reset_clock_on_flood:
            if is_flood_owner:
                t_own = 0.0
            if is_flood_renter:
                t_ren = 0.0

        # Write back
        df.at[i, "TP_owner"] = tp_own
        df.at[i, "TP_renter"] = tp_ren
        df.at[i, "t_clock_owner"] = t_own
        df.at[i, "t_clock_renter"] = t_ren

    return df


# Backward-compatible alias
update_tract_tp_mg_nmg = update_tract_tp_owner_renter
