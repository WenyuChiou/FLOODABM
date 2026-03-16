# -*- coding: utf-8 -*-
"""
Agent Decision Module for FLOODABM
===================================

This module handles household-level decision making for flood mitigation actions.
It provides functions for:

1. Loading Bayesian predictors (fast .npz cache or fallback .pkl)
2. Building state indexers for efficient tract→household mapping
3. Making sequential decisions based on predicted action probabilities
4. Initializing flood insurance flags from YAML configuration

Decision Flow:
    - Owners: FI (Flood Insurance) → EH (Elevation) → BP (Barrier Protection) → DN (Do Nothing)
    - Renters: FI (Flood Insurance) → RL (Relocation) → DN (Do Nothing)

Example:
    >>> from modules.actions.decision import load_predictors, build_state_indexer
    >>> pred_owner, pred_renter = load_predictors(Path("modules/actions"))
    >>> idxer = build_state_indexer(state_df, tract_psy_df)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .bayes_fast_predictors import build_fast_predictors

# Fallback to compat loader if available
try:
    from .predictors_compat import build_bayes_predictors_from_actions_root as _compat_build
except Exception:
    _compat_build = None


# =============================================================================
# Predictor Loading
# =============================================================================

def load_predictors(actions_dir: Path) -> tuple:
    """Load Bayesian action predictors for owner and renter groups.

    Tries the fast .npz cache loader first, falls back to compat loader if needed.

    Args:
        actions_dir: Path to actions module directory containing models/

    Returns:
        (predictor_owner, predictor_renter): Two predictor functions

    Raises:
        RuntimeError: If both fast and compat loaders fail
    """
    try:
        return build_fast_predictors(actions_dir)
    except Exception as e_fast:
        if _compat_build is None:
            raise
        try:
            return _compat_build(actions_dir)
        except Exception as e_compat:
            raise RuntimeError(
                "Both predictor loaders failed.\n"
                f"- fast:   {e_fast}\n"
                f"- compat: {e_compat}\n"
            )


# =============================================================================
# State Indexer (Fast Tract Lookup)
# =============================================================================

def build_state_indexer(state: pd.DataFrame, tract_psy: pd.DataFrame) -> np.ndarray:
    """Build index array mapping each household to its tract row in tract_psy.
    
    This enables fast O(1) lookups for psychological features by household.
    
    Args:
        state: Household state DataFrame with tract_geoid column
        tract_psy: Tract psychology DataFrame with tract_geoid column
        
    Returns:
        NumPy array of indices where idxer[i] gives the row in tract_psy
        for household i
    """
    tr_to_pos = {t: i for i, t in enumerate(tract_psy["tract_geoid"].astype(str).to_numpy())}
    geo = state["tract_geoid"].astype(str).to_numpy()
    idxer = np.fromiter((tr_to_pos[g] for g in geo), dtype=np.int64, count=len(geo))
    return idxer


def features_take_by_indexer(
    tract_psy: pd.DataFrame,
    idxer: np.ndarray,
    suffix: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract TP, CP, SP features using pre-built indexer.

    Uses np.take for vectorized O(1) lookups instead of pandas.map.

    Args:
        tract_psy: Tract psychology DataFrame
        idxer: Index array from build_state_indexer()
        suffix: Group suffix ("owner" or "renter")

    Returns:
        (TP, CP, SP): Tuple of numpy arrays for each psychological feature
    """
    TP = tract_psy[f"TP_{suffix}"].to_numpy()[idxer]
    CP = tract_psy[f"CP_{suffix}"].to_numpy()[idxer]
    SP = tract_psy[f"SP_{suffix}"].to_numpy()[idxer]
    return TP, CP, SP


def predict_by_group(
    state: pd.DataFrame,
    predictor_owner,
    predictor_renter,
) -> dict[str, np.ndarray]:
    """Run group-specific predictors — no mixing.

    Each household uses its own group's predictor exclusively.
    Owner households get owner predictor outputs; renter households
    get renter predictor outputs.

    Args:
        state: Household DataFrame with TP, CP_init, SP_init, group columns
        predictor_owner: Owner predictor function(TP=, CP=, SP=)
        predictor_renter: Renter predictor function(TP=, CP=, SP=)

    Returns:
        Dict mapping action names (FI, EH, BP, RL) to probability arrays (N,)
    """
    N = len(state)
    is_owner = state["group"].astype(str).str.lower().eq("owner").to_numpy()

    tp = state["TP"].to_numpy(dtype=float)
    cp = state["CP_init"].to_numpy(dtype=float)
    sp = state["SP_init"].to_numpy(dtype=float)

    own_idx = np.where(is_owner)[0]
    ren_idx = np.where(~is_owner)[0]

    # Run each predictor only on its group's rows
    p_own = predictor_owner(TP=tp[own_idx], CP=cp[own_idx], SP=sp[own_idx])
    p_ren = predictor_renter(TP=tp[ren_idx], CP=cp[ren_idx], SP=sp[ren_idx])

    # Combine into full-length arrays
    all_acts = set(p_own) | set(p_ren)
    probs = {}
    for act in all_acts:
        combined = np.zeros(N, dtype=float)
        if act in p_own:
            combined[own_idx] = p_own[act]
        if act in p_ren:
            combined[ren_idx] = p_ren[act]
        probs[act] = combined

    return probs


def mix_probs_by_weight(
    p_owner: dict[str, np.ndarray],
    p_renter: dict[str, np.ndarray],
    w_owner_vec: np.ndarray,
) -> dict[str, np.ndarray]:
    """Mix owner and renter probabilities by household-level weights.

    Computes: p = w * p_owner + (1-w) * p_renter

    Args:
        p_owner: Dict mapping action names to owner probability arrays
        p_renter: Dict mapping action names to renter probability arrays
        w_owner_vec: Weight array (0-1) indicating owner share per household

    Returns:
        Dict mapping action names to mixed probability arrays
    """
    mix = {}
    w = w_owner_vec.astype(float)
    for act in p_owner.keys():
        a = np.asarray(p_owner[act], float)
        b = np.asarray(p_renter[act], float)
        mix[act] = np.clip(w * a + (1.0 - w) * b, 0.0, 1.0)
    return mix


# =============================================================================
# Sequential Decision Making
# =============================================================================

def sequential_decision_fast(
    state: pd.DataFrame,
    probs: dict[str, np.ndarray],
    rng: np.random.RandomState,
    overlay_policy_names: dict = None,
    decision_threshold: float = 0.0,
    draw_bounds: dict[str, tuple[float, float]] = None,
) -> pd.DataFrame:
    """Make sequential decisions for all households in vectorized fashion.

    Decision trees:
        - Owner:  FI → EH → BP → DN (if not already done EH)
        - Renter: FI → RL → DN

    Each step uses Monte Carlo sampling from the Bayesian posterior:
    a household adopts action *a* if a uniform draw u < p_a (Bernoulli trial).
    This naturally captures unobserved individual heterogeneity — even
    households with identical psychological profiles may make different
    choices due to unmodeled factors.

    Args:
        state: Household state DataFrame with 'group' column
        probs: Dict mapping action names (FI, EH, BP, RL) to probability arrays
        rng: NumPy RandomState for Monte Carlo sampling
        overlay_policy_names: Policy name overrides for FI decisions
        decision_threshold: Minimum probability floor (default 0.0).
            An action can only be adopted if p_a > this floor.
        draw_bounds: Action-specific bounds for the uniform draw.
            Dict mapping action name to (low, high) tuple.
            e.g. {"FI": (0.4, 0.6), "EH": (0.05, 0.25), ...}
            If None or action not in dict, defaults to (0, 1).

    Returns:
        DataFrame with columns: action, ELEV_FT, POLICY_NAME
    """
    if overlay_policy_names is None:
        overlay_policy_names = {"owner": "owner_standard", "renter": "renter_contents"}

    N = len(state)
    group = state["group"].astype(str).to_numpy()
    is_owner = (group == "owner")
    is_renter = ~is_owner

    # Initialize outputs
    out_action = np.full(N, "DN", dtype=object)
    out_elev = np.zeros(N, dtype=float)
    out_pol = np.full(N, "", dtype=object)

    # Extract probability arrays
    pFI = probs["FI"]
    pEH = probs.get("EH", np.zeros(N))
    pBP = probs.get("BP", np.zeros(N))
    pRL = probs.get("RL", np.zeros(N))

    # Handle EH one-time constraint (owners who already elevated cannot do it again)
    if "eh_done" not in state.columns:
        state = state.copy()
        state["eh_done"] = False

    eh_done = state["eh_done"].astype(bool).to_numpy()
    mask_no_eh = is_owner & eh_done
    if mask_no_eh.any():
        pEH = pEH.copy()
        pEH[mask_no_eh] = 0.0

    # Monte Carlo draws for each decision step (action-specific bounded ranges)
    if draw_bounds is None:
        draw_bounds = {}
    fi_lo, fi_hi = draw_bounds.get("FI", (0.0, 1.0))
    eh_lo, eh_hi = draw_bounds.get("EH", (0.0, 1.0))
    bp_lo, bp_hi = draw_bounds.get("BP", (0.0, 1.0))
    rl_lo, rl_hi = draw_bounds.get("RL", (0.0, 1.0))
    u_fi = rng.uniform(fi_lo, fi_hi, size=N)
    u_eh = rng.uniform(eh_lo, eh_hi, size=N)
    u_bp = rng.uniform(bp_lo, bp_hi, size=N)
    u_rl = rng.uniform(rl_lo, rl_hi, size=N)

    # ==================== Owner Decision Tree ====================
    # Step 1: FI decision (Bernoulli trial: adopt if u < p)
    take_FI = (u_fi < pFI) & (pFI > decision_threshold) & is_owner
    out_action[take_FI] = "FI"
    out_pol[take_FI] = overlay_policy_names.get("owner", "owner_standard")

    # Step 2: EH decision (only if didn't take FI)
    rem_owner = is_owner & ~take_FI
    take_EH = (u_eh < pEH) & (pEH > decision_threshold) & rem_owner
    out_action[take_EH] = "EH"
    out_elev[take_EH] = 5.0  # Record elevation decision (+5 ft)

    # Step 3: BP decision (only if didn't take FI or EH)
    rem_owner2 = rem_owner & ~take_EH
    take_BP = (u_bp < pBP) & (pBP > decision_threshold) & rem_owner2
    out_action[take_BP] = "BP"

    # ==================== Renter Decision Tree ====================
    # Step 1: FI decision (Bernoulli trial: adopt if u < p)
    take_FI_r = (u_fi < pFI) & (pFI > decision_threshold) & is_renter
    out_action[take_FI_r] = "FI"
    out_pol[take_FI_r] = overlay_policy_names.get("renter", "renter_contents")

    # Step 2: RL decision (only if didn't take FI)
    rem_renter = is_renter & ~take_FI_r
    take_RL = (u_rl < pRL) & (pRL > decision_threshold) & rem_renter
    out_action[take_RL] = "RL"

    return pd.DataFrame({
        "action": out_action,
        "ELEV_FT": out_elev,
        "POLICY_NAME": out_pol,
    })


# =============================================================================
# Insurance Flag Initialization
# =============================================================================

def _init_fi_flags_from_yaml(
    df: pd.DataFrame,
    ins_cfg: dict,
    *,
    state_col: str = "state",
    group_col: str = "identity",
    tract_col: str = "tract_geoid",
    seed: int = 42,
    overwrite: bool = True,
    assume_state_if_missing: str | None = None,
) -> pd.DataFrame:
    """Initialize has_FI (flood insurance) flags based on YAML configuration.
    
    Supports multiple rate specification methods (in priority order):
    1. take_rate_by_tract_group: {tract: {owner: rate, renter: rate}}
    2. take_rate_by_state: {state: rate}
    3. take_rate_by_group: {owner: rate, renter: rate}
    4. take_rate_overall: single rate for all households
    
    Args:
        df: Household DataFrame
        ins_cfg: Insurance configuration dict from YAML
        state_col: Column name for state (e.g., "NJ")
        group_col: Column name for tenure type
        tract_col: Column name for census tract
        seed: Random seed for sampling
        overwrite: If True, reset has_FI to 0 before applying rates
        assume_state_if_missing: Default state to use if column missing
        
    Returns:
        DataFrame with has_FI column updated
    """
    df = df.copy()
    rng = np.random.default_rng(int(ins_cfg.get("seed", seed)))
    
    def _norm_geoid(x):
        """Normalize tract geoid to 11-digit string."""
        s = "".join(ch for ch in str(x) if ch.isdigit())
        return s.zfill(11) if s else None
    
    def _apply_rate(mask: pd.Series, rate: float):
        """Apply insurance rate to masked subset of households."""
        if rate is None:
            return
        try:
            rate = float(rate)
        except (TypeError, ValueError):
            return
        if not np.isfinite(rate):
            return
        rate = min(max(rate, 0.0), 1.0)
        idx = df.index[mask.fillna(False)].to_numpy()
        n = idx.size
        if n == 0 or rate <= 0:
            return
        k = int(round(rate * n))
        k = max(0, min(k, n))
        if k == 0:
            return
        chosen = rng.choice(idx, size=k, replace=False)
        df.loc[chosen, "has_FI"] = 1
    
    # Initialize has_FI column
    if overwrite or ("has_FI" not in df.columns):
        df["has_FI"] = 0
    df["has_FI"] = pd.to_numeric(df["has_FI"], errors="coerce").fillna(0).astype("int8")
    
    # Normalize identity to owner/renter
    if group_col in df.columns:
        df[group_col] = (
            df[group_col].astype(str).str.strip().str.lower()
            .replace({
                "owner-occupied": "owner", "owner occupied": "owner",
                "homeowner": "owner", "own": "owner", "o": "owner",
                "1": "owner", "true": "owner", "yes": "owner",
                "renter-occupied": "renter", "renter occupied": "renter",
                "rent": "renter", "tenant": "renter", "r": "renter",
                "0": "renter", "false": "renter", "no": "renter",
            })
        )
    
    # Normalize tract geoid to 11 digits
    tract_ser = df[tract_col].map(_norm_geoid) if tract_col in df.columns else None
    
    # Extract rate configurations
    rate_by_tract_group = ins_cfg.get("take_rate_by_tract_group")
    rate_by_state = ins_cfg.get("take_rate_by_state")
    rate_by_group = ins_cfg.get("take_rate_by_group")
    rate_overall = ins_cfg.get("take_rate_overall")
    
    # 1) Apply tract × group rates
    if isinstance(rate_by_tract_group, dict) and tract_ser is not None:
        mapping = {}
        for k, v in rate_by_tract_group.items():
            nk = _norm_geoid(k)
            if nk:
                mapping[nk] = v
        
        if group_col in df.columns:
            gp = df[group_col].astype(str).str.lower()
            missing = []
            for geoid_key, pair in mapping.items():
                m_t = tract_ser.eq(geoid_key)
                if not m_t.any():
                    missing.append(geoid_key)
                    continue
                if isinstance(pair, dict):
                    o_rate = pair.get("owner")
                    r_rate = pair.get("renter")
                elif isinstance(pair, (list, tuple)) and len(pair) >= 2:
                    o_rate, r_rate = pair[0], pair[1]
                else:
                    o_rate = r_rate = pair
                _apply_rate(m_t & gp.eq("owner"), o_rate)
                _apply_rate(m_t & gp.eq("renter"), r_rate)
            if missing:
                print(f"[INFO] {len(missing)} mapping tracts not found (first 5): {missing[:5]}")
        else:
            for geoid_key, pair in mapping.items():
                m_t = tract_ser.eq(geoid_key)
                if isinstance(pair, dict):
                    rt = pair.get("owner") or pair.get("renter")
                elif isinstance(pair, (list, tuple)) and len(pair) >= 2:
                    rt = pair[0] if pair[0] is not None else pair[1]
                else:
                    rt = pair
                _apply_rate(m_t, rt)
    
    # 2) Apply state-level rates
    if isinstance(rate_by_state, dict) and len(rate_by_state) > 0:
        if state_col in df.columns:
            st = df[state_col].astype(str).str.upper()
            for s, rt in rate_by_state.items():
                _apply_rate(st.eq(str(s).upper()), rt)
        else:
            if assume_state_if_missing is not None:
                rt = rate_by_state.get(str(assume_state_if_missing).upper())
                if rt is not None:
                    _apply_rate(pd.Series(True, index=df.index), rt)
            elif len(rate_by_state) == 1:
                only_key = next(iter(rate_by_state))
                _apply_rate(pd.Series(True, index=df.index), rate_by_state[only_key])
    
    # 3) Apply group-level rates
    if isinstance(rate_by_group, dict) and group_col in df.columns:
        gp = df[group_col].astype(str).str.lower()
        for g, rt in rate_by_group.items():
            _apply_rate(gp.eq(str(g).lower()), rt)
    
    # 4) Apply overall rate
    if rate_overall is not None:
        _apply_rate(pd.Series(True, index=df.index), rate_overall)
    
    # Finalize
    df["has_FI"] = pd.to_numeric(df["has_FI"], errors="coerce").fillna(0).astype("int8")
    if "is_FI" in df.columns:
        df = df.drop(columns=["is_FI"])
    
    # Validation output
    total_ones = int(df["has_FI"].sum())
    print(f"[init] has_FI = {total_ones} households")
    if total_ones == 0:
        print("[HINT] Check: identity normalized to owner/renter, tract_geoid is 11 digits")
    
    return df
