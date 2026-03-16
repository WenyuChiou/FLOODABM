# -*- coding: utf-8 -*-
"""
FLOODABM Main Simulation Runner
===============================

This is the primary entry point for running FLOODABM simulations.

QUICK START
-----------
    # Run baseline scenario with optimized models
    python main.py --model-dir models_optimized --output-mode full

    # Run worst-case (no adaptation) scenario for comparison
    python main.py --scenario worst --model-dir models_optimized --output-mode full

    # Run both then analyze
    python analyze_themes.py

CLI ARGUMENTS
-------------
    --model-dir     : Model directory (default: 'models')
                      Auto-detects all 'models*' directories in modules/actions/
                      Available options shown in --help output
    
    --scenario      : Simulation scenario (default: 'baseline')
                      'baseline' = Normal simulation with agent decisions
                      'worst'    = No-action scenario (all agents do nothing)
    
    --output-mode   : Output storage mode
                      'full'    = All files (~250MB)
                      'summary' = Tract-level only (~60MB)
                      'minimal' = Essential only (~10MB)
    
    --out-root      : Custom output directory (default: outputs/)
    --thr-owner     : Override owner flood ratio threshold (default: 0.5)
    --thr-renter    : Override renter flood ratio threshold (default: 0.5)
    --no-plots      : Skip all visualization generation
    --deterministic : Use fixed RNG each year for reproducibility

KEY PARAMETERS (config/abm_params.yaml)
---------------------------------------
tp_config:
    shock_timing    : 'end' or 'start'
                      'end'   = Decay TP first, then apply flood shock (recommended)
                      'start' = Apply shock first, then decay
    
    skip_decay_on_shock : true/false
                      true  = Tracts hit by flood skip decay that year (shock preserved)
                      false = All tracts decay every year regardless of flood
    
    shock_mode      : 'additive' or 'toward_one'
                      'additive'   = TP += flood_ratio
                      'toward_one' = TP moves toward 1.0 asymptotically

insurance_init:
    take_rate_by_tract_group : Initial insurance uptake by tract
                      owner: 0.25 → Flood-prone tract
                      owner: 0.03 → Non-flood-prone tract

flood:
    ratio_threshold_owner  : Flood ratio trigger for owner group (default: 0.5)
    ratio_threshold_renter : Flood ratio trigger for renter group (default: 0.5)

ARCHITECTURE
------------
    1. Configuration: Load YAML config, parse CLI arguments
    2. Data Loading: Households, flood depths, tract psychology
    3. Annual Loop: For each year, compute vulnerability, make decisions, update TP
    4. Finalization: Aggregate outputs, generate visualization plots

KEY OUTPUTS
-----------
    decisions/     : Action decisions per household per year
    finance/       : Insurance payouts, premiums, OOP costs
    vulnerability/ : Flood damage by tract
    visualization/ : Generated plots
    states/        : Household state snapshots
    tp_traj.csv    : Threat Perception trajectory by tract/year

See config/abm_params.yaml for all configurable parameters.
"""
from __future__ import annotations
from pathlib import Path
import argparse
import os
import sys
import numpy as np
import pandas as pd

# Ensure local imports work when running as a script
THIS_FILE = Path(__file__).resolve()
ROOT = THIS_FILE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils._helpers import (
    load_yaml_cfg, resolve_path, years_from_cfg,
    load_households_csv, load_depths_legacy, depths_for_year,
    grp_params_from_yaml,
    load_inline_owner_share_policy, read_finance_from_yaml,
    finalize_has_fi,
)
from utils.main_helpers import (
    build_flood_bookkeeping_rows,
    save_decisions_and_update_prev,
    save_action_shares,
    append_tp_trajectory,
    advance_state_for_next_year,
    save_next_year_state,
    handle_no_action_year,
)
from modules.actions.vuln_for_tp import (
    ratio_by_tract_from_vuln,
    _attach_hh_flood_damage,
    _export_tract_flood_damage,
)
from modules.actions.tp import TPConfig, TPGroupParams
from modules.actions.decision import (
    load_predictors, _init_fi_flags_from_yaml,
)
from modules.finance import run_finance_for_year
from utils.sa_output_manager import SAOutputManager


# =============================================================================
# POST-SIMULATION: Aggregation & Visualization
# =============================================================================

from utils.finalize import finalize_and_plot


def run_single_scenario(
    argv: list[str] | None = None,
    scenario_override: str | None = None,
    skip_comparison_plots: bool = False,
) -> tuple[Path, "argparse.Namespace"]:
    """
    Run a single simulation scenario.
    
    Args:
        argv: Command line arguments
        scenario_override: Force a specific scenario ('baseline' or 'worst')
        skip_comparison_plots: If True, skip the compare-flood-or comparison plots
    
    Returns:
        Tuple of (output_dir, args) for use by caller
    """
    
    # =========================================================================
    # SECTION 1: PATH CONFIGURATION
    # =========================================================================
    MODULES_ROOT = ROOT / "modules"
    ACTIONS_DIR = MODULES_ROOT / "actions"
    CONFIG_DIR = ROOT / "config"
    CONFIG_PATH = CONFIG_DIR / "abm_params.yaml"

    # Load YAML first (so CLI can override it)
    CFG = load_yaml_cfg(CONFIG_PATH)

    # Auto-detect available model directories using ModelRegistry
    from utils.model_registry import ModelRegistry
    registry = ModelRegistry(ROOT)  # Models are at project root level
    available_models = registry.list_models()
    if not available_models:
        available_models = ["models"]  # fallback

    # ---------- CLI args (override YAML) ----------
    ap = argparse.ArgumentParser(description="FLOODABM main runner")
    ap.add_argument("--thr-owner", type=float, help="Override flood.ratio_threshold_owner")
    ap.add_argument("--thr-renter", type=float, help="Override flood.ratio_threshold_renter")
    ap.add_argument("--shock-owner", type=float, help="Override tp_config.shock_scale_owner")
    ap.add_argument("--shock-renter", type=float, help="Override tp_config.shock_scale_renter")
    ap.add_argument(
        "--scenario", choices=["baseline", "worst"],
        help="baseline = normal decisions/TP/actions; worst = no action (decisions off, TP off, finance not gated).",
    )
    ap.add_argument("--out-root", type=str,
                    help="Custom output root (default <repo>/outputs). The scenario name will be appended as a subfolder.")
    ap.add_argument("--compare-flood-or", action="store_true",
                    help="Plot grouped bars comparing Baseline vs Worst for Owner & Renter (two panels).")
    ap.add_argument("--compare-mode", choices=["total", "avg"], default="total",
                    help="Use 'total' USD or 'avg' USD per household.")
    ap.add_argument("--compare-severe-years", type=str, default="",
                    help="Comma-separated severe years to shade, e.g. '2011,2014,2021'.")
    ap.add_argument("--no-plots", action="store_true", help="Skip all plotting (override YAML output.enable_plots).")
    ap.add_argument("--deterministic", action="store_true", help="Use fixed RNG each year (no SEED+year offset) for reproducibility.")
    ap.add_argument("--output-mode", choices=["full", "summary", "minimal"], default=None,
                    help="Output storage mode: full (all files ~250MB), summary (~60MB), minimal (~10MB)")
    ap.add_argument("--model-dir", choices=available_models, default="baseline",
                    help=f"Model directory (auto-detected): {', '.join(available_models)}")
    ap.add_argument("--list-models", action="store_true",
                    help="List all available models and exit")
    ap.add_argument("--run-all", action="store_true",
                    help="Run ALL detected models sequentially")
    ap.add_argument("--years", type=int, default=None,
                    help="Limit simulation to first N years (for testing)")
    ap.add_argument("--decision-threshold", type=float, default=None,
                    help="Override decision_threshold (default 0.5). Action adopted if p > threshold.")
    args = ap.parse_args(argv)
    
    # Override scenario if specified (for batch run)
    if scenario_override:
        args.scenario = scenario_override
    
    # If running in batch mode, skip comparison plots in individual runs
    if skip_comparison_plots:
        args.compare_flood_or = False

    if args.thr_owner is not None:
        CFG.setdefault("flood", {})["ratio_threshold_owner"] = float(args.thr_owner)
    if args.thr_renter is not None:
        CFG.setdefault("flood", {})["ratio_threshold_renter"] = float(args.thr_renter)
    if args.decision_threshold is not None:
        CFG["decision_threshold"] = float(args.decision_threshold)
    # Shock scale overrides for TP configuration
    if args.shock_owner is not None:
        CFG.setdefault("tp_config", {})["shock_scale_owner"] = float(args.shock_owner)
    if args.shock_renter is not None:
        CFG.setdefault("tp_config", {})["shock_scale_renter"] = float(args.shock_renter)

    yaml_scen = str(CFG.get("scenario", "baseline")).strip().lower()
    SCEN = (args.scenario or yaml_scen or "baseline").lower()
    NO_ACTION = (SCEN == "worst")

    # =========================================================================
    # SECTION 2: OUTPUT DIRECTORY SETUP
    # =========================================================================
    # Priority: CLI arg > environment variable > default (repo/outputs)
    env_out = os.environ.get("FLOODABM_OUTPUT_ROOT", None)
    OUT_ROOT = Path(args.out_root or env_out or (ROOT / "outputs")).resolve()
    # Always include model name in output path for consistent structure
    OUTPUT_DIR = (OUT_ROOT / args.model_dir / SCEN).resolve()
    VIS_DIR = OUTPUT_DIR / "visualization"
    STATES_DIR = OUTPUT_DIR / "states"
    FIN_DIR = OUTPUT_DIR / "finance"
    DEC_DIR = OUTPUT_DIR / "decisions"
    for p in (OUTPUT_DIR, VIS_DIR, STATES_DIR, FIN_DIR, DEC_DIR):
        p.mkdir(parents=True, exist_ok=True)
    print(f"[scenario] {SCEN} | OUTPUT_DIR={OUTPUT_DIR}")

    # Initialize output manager (CLI flag > environment variable > default 'full')
    output_mode = args.output_mode or os.environ.get("FLOODABM_OUTPUT_MODE", "full")
    output_mgr = SAOutputManager(mode=output_mode, output_dir=OUTPUT_DIR)
    if output_mode != "full":
        print(f"[info] Output mode: {output_mode} (storage optimization enabled)")
        est_size = output_mgr.get_estimated_size()
        print(f"[info] Estimated storage: ~{est_size.get('total', 250):.0f} MB")

    # =========================================================================
    # SECTION 3: DATA FILE PATH RESOLUTION
    # =========================================================================
    # Resolve paths relative to actions directory or absolute from YAML
    FILES = CFG.get("files", {}) or {}
    PATH_DEPTHS = resolve_path(ACTIONS_DIR, FILES.get("depths_overall"), MODULES_ROOT)
    PATH_EVENTS = resolve_path(ACTIONS_DIR, FILES.get("flood_events"), MODULES_ROOT)
    PATH_HOUSEHOLDS = resolve_path(ACTIONS_DIR, FILES.get("households"), MODULES_ROOT)
    if not PATH_HOUSEHOLDS or not PATH_HOUSEHOLDS.exists():
        raise FileNotFoundError("Households exposure DB is required. Set files.households (CSV) in YAML.")
    if not PATH_DEPTHS or not PATH_DEPTHS.exists():
        raise FileNotFoundError("Depths file is required. Set files.depths_overall in YAML (JSON/CSV supported).")
    print(f"[info] Using households CSV: {PATH_HOUSEHOLDS}")
    print(f"[info] Using depths file:   {PATH_DEPTHS}")

    # =========================================================================
    # SECTION 4: CORE PARAMETER EXTRACTION
    # =========================================================================
    # Trust in Policymaker (TP) configuration
    TP_CFG_JSON = CFG.get("tp_config", {}) or {}
    FLOOD_JSON = CFG.get("flood", {}) or {}
    SEED = int(CFG.get("seed", 12345))
    rng = np.random.RandomState(SEED)
    YEARS_STEP = float(CFG.get("years_step", 1.0))
    DYN = CFG.get("action_dynamics") or {}
    INS_JSON = CFG.get("insurance_init", {}) or {}

    # ---------------------------------------------------------------------------
    # Flood Thresholds: Control when TP shock is triggered
    # - ratio_threshold: Minimum damage/value ratio to trigger TP shock
    # - depth_threshold_m: Optional depth-based trigger (meters)
    # ---------------------------------------------------------------------------
    FLOOD_RATIO_THRESHOLD = float(FLOOD_JSON.get("ratio_threshold", 0.5))  # legacy default
    DEPTH_THRESHOLD_M = FLOOD_JSON.get("depth_threshold_m", None)
    FFE_FT = float(FLOOD_JSON.get("FFE_ft", 0.5))
    SHOCK_TIMING = TP_CFG_JSON.get("shock_timing", "start")
    SHOCK_MODE = TP_CFG_JSON.get("shock_mode", "additive")
    # Owner/Renter specific thresholds for TP shock triggering
    _THR_OWNER = float(FLOOD_JSON.get("ratio_threshold_owner", FLOOD_RATIO_THRESHOLD))
    _THR_RENTER = float(FLOOD_JSON.get("ratio_threshold_renter", FLOOD_RATIO_THRESHOLD))

    # ---------------------------------------------------------------------------
    # Action Dynamics: Control how household actions are applied
    # - EH (Elevation): One-time or cumulative, with min/max/cap limits
    # - RL (Relocation): Whether to apply and backfill destination tracts
    # - BP (Buyout Program): Sampling method for protection levels
    # ---------------------------------------------------------------------------
    EH_ONE_TIME = bool(DYN.get("eh_one_time", False))
    EH_ONCE_MIN_FT = float(DYN.get("eh_once_min_ft", 3.0))
    EH_ONCE_MAX_FT = float(DYN.get("eh_once_max_ft", 5.0))
    EH_STEP_FT = float(DYN.get("eh_step_ft", 1.0))
    EH_CAP_FT = float(DYN.get("eh_cap_ft", 8.0))
    APPLY_RL = bool(DYN.get("apply_RL", True))
    RL_BACKFILL = bool(DYN.get("rl_backfill", True))
    APPLY_BP = bool(DYN.get("apply_BP", True))
    BP_BACKFILL = bool(DYN.get("bp_backfill", False))
    BP_SAMPLER = str(DYN.get("bp_sampler", "lognorm_from_totals"))

    CV_NJ = {"rcv": 0.694, "contents": 1.8}

    # ---------------------------------------------------------------------------
    # Decision Threshold: Probability cutoff for action adoption
    # Action is adopted if predicted probability > threshold (default 0.5)
    # ---------------------------------------------------------------------------
    DECISION_THRESHOLD = float(CFG.get("decision_threshold", 0.5))

    print(
        f"[cfg] ratio_threshold_owner={_THR_OWNER} | ratio_threshold_renter={_THR_RENTER} | "
        f"depth_threshold_m={DEPTH_THRESHOLD_M} | FFE_ft={FFE_FT} | shock_timing={SHOCK_TIMING} | "
        f"decision_threshold={DECISION_THRESHOLD}"
    )

    # ---------------------------------------------------------------------------
    # Finance Configuration: Insurance premiums and coverage
    # ---------------------------------------------------------------------------
    PREMIUM_CFG_BY_STATE, DEFAULT_PREMIUM, OWNER_INS = read_finance_from_yaml(CFG)
    def _premium_for_state_df(df_state: pd.DataFrame) -> dict:
        return dict(DEFAULT_PREMIUM)
    OWNER_CONTENTS_RATIO = float(OWNER_INS.get("owner_contents_ratio", 0.57))
    OWNER_INSURES_BOTH = bool(OWNER_INS.get("owner_insures_both", True))

    # Policy & owner share
    OWNER_SHARE, policy_dict = load_inline_owner_share_policy(CFG)

    # =========================================================================
    # SECTION 5: DATA LOADING
    # =========================================================================
    # Load flood depth data (long format: tract × year × depth_m)
    DEPTHS_LONG = load_depths_legacy(PATH_DEPTHS)
    YEARS = years_from_cfg(CFG, DEPTHS_LONG["year"].unique(), PATH_EVENTS)
    if args.years:
        YEARS = sorted(YEARS)[:args.years]
    print(f"[info] YEARS: {YEARS}")

    # Load household exposure database (psych + has_FI pre-baked in CSV)
    STATE = load_households_csv(PATH_HOUSEHOLDS)
    print(f"[info] Loaded households: {len(STATE):,} rows")
    if "has_FI" in STATE.columns:
        STATE["has_FI"] = pd.to_numeric(STATE["has_FI"], errors="coerce").fillna(0).astype("int8")
        print(f"[info] has_FI from CSV: {STATE['has_FI'].sum():,} households")
    else:
        # Fallback: runtime initialization (legacy path)
        STATE = _init_fi_flags_from_yaml(
            STATE,
            INS_JSON,
            group_col="identity",
            tract_col="tract_geoid",
            seed=INS_JSON.get("seed", CFG.get("seed", 42)),
            overwrite=True,
        )
        print(f"[info] Initial has_FI (runtime): {STATE['has_FI'].sum():,} households")

    # =========================================================================
    # SECTION 6: PSYCHOLOGY INITIALIZATION
    # =========================================================================
    # Initialize tract-level psychological factors (TP, CP, SP, SC, PA)
    # for both owner (homeowner) and renter groups
    TRACTS = sorted(
        set(DEPTHS_LONG["tract_geoid"].astype(str)) | set(STATE["tract_geoid"].astype(str))
    )
    PARAMS_OWNER = grp_params_from_yaml(CFG, "owner", TPGroupParams)
    PARAMS_RENTER = grp_params_from_yaml(CFG, "renter", TPGroupParams)
    TP_CFG = TPConfig(
        shock_scale_owner=float(TP_CFG_JSON.get("shock_scale_owner", 1.0)),
        shock_scale_renter=float(TP_CFG_JSON.get("shock_scale_renter", 1.0)),
        shock_timing=SHOCK_TIMING,
    )
    setattr(TP_CFG, "shock_mode", SHOCK_MODE)
    setattr(TP_CFG, "thr_owner", _THR_OWNER)
    setattr(TP_CFG, "thr_renter", _THR_RENTER)
    setattr(TP_CFG, "flood_ratio_threshold", FLOOD_RATIO_THRESHOLD)
    setattr(TP_CFG, "clip_lo", float(TP_CFG_JSON.get("clip_lo", 0.0)))
    setattr(TP_CFG, "clip_hi", float(TP_CFG_JSON.get("clip_hi", 1.0)))
    setattr(TP_CFG, "skip_decay_on_shock", bool(TP_CFG_JSON.get("skip_decay_on_shock", False)))

    # --- Household-level psychology initialization ---
    # Each household has its own TP/CP/SP/SC/PA drawn from Beta distributions
    # (generated by generate_household_psych.py, stored in CSV)
    psych_init_cols = ["TP_init", "CP_init", "SP_init", "SC_init", "PA_init"]
    missing_psych = [c for c in psych_init_cols if c not in STATE.columns]
    if missing_psych:
        raise ValueError(
            f"Household CSV is missing psych columns: {missing_psych}. "
            "Run: python generate_household_psych.py"
        )
    STATE["TP"] = STATE["TP_init"].astype(float)
    STATE["t_clock"] = 0.0

    MODEL_PATH = registry.get_path(args.model_dir)  # models/baseline/ or models/optimized/
    if MODEL_PATH is None:
        MODEL_PATH = ROOT / "models" / args.model_dir  # fallback
    predictor_owner, predictor_renter = load_predictors(MODEL_PATH)

    # =========================================================================
    # SECTION 7: ANNUAL SIMULATION LOOP
    # =========================================================================
    # Each year: vulnerability → finance → decisions → TP update → state advance
    tp_rows: list[dict] = []  # Collect TP trajectory data
    flood_rows: list[dict] = []
    INIT_FI_COL = "has_FI"
    dec_prev = STATE[["i"]].copy() if "i" in STATE.columns else STATE.reset_index()[["index"]].rename(columns={"index": "i"})
    dec_prev["has_FI"] = (
        pd.to_numeric(STATE[INIT_FI_COL], errors="coerce").fillna(0).astype(int)
        if INIT_FI_COL in STATE.columns else 0
    )

    for y in YEARS:
        # -----------------------------------------------------------------------
        # STEP 7.1: Prepare state snapshot for this year
        # Disaster happens first, then households may relocate
        # -----------------------------------------------------------------------
        STATE_V = STATE.copy()
        STATE_V = finalize_has_fi(STATE_V, dec_prev=dec_prev, sticky=False)
        STATE_V["tract_geoid"] = STATE_V["tract_geoid"].astype(str)
        STATE_V.insert(1, "year", y)
        STATES_DIR.mkdir(parents=True, exist_ok=True)
        if output_mgr.should_save('state_csvs'):
            STATE_V.to_csv(STATES_DIR / f"state_for_vuln_{y}.csv", index=False, encoding="utf-8-sig")

        # -----------------------------------------------------------------------
        # STEP 7.2: Load flood depth data for current year
        # Create tract → depth_m mapping for vulnerability calculation
        # -----------------------------------------------------------------------
        depths_y = depths_for_year(DEPTHS_LONG, y)
        dmap = dict(zip(depths_y["tract_geoid"].astype(str), depths_y["depth_m"].astype(float)))

        # -----------------------------------------------------------------------
        # STEP 7.3: Calculate flood damage using vulnerability curves
        # Attaches gross_structure_loss_kUSD and gross_contents_loss_kUSD
        # -----------------------------------------------------------------------
        STATE_V_WITH_DMG = _attach_hh_flood_damage(
            state_df=STATE_V.copy(), year=y, depths_long=DEPTHS_LONG,
            modules_root=ROOT / "modules", ffe_ft=FFE_FT,
        )
        _export_tract_flood_damage(state_with_damage=STATE_V_WITH_DMG, year=y, out_root=OUTPUT_DIR)

        # -----------------------------------------------------------------------
        # STEP 7.4: Compute damage ratio for TP shock calculation
        # ratio = total_loss / total_value per tract (0-1 scale)
        # -----------------------------------------------------------------------
        ratio_raw = ratio_by_tract_from_vuln(
            state=STATE_V, depths_long=DEPTHS_LONG, year=y,
            modules_root=ROOT / "modules", FFE_ft=FFE_FT,
            owner_contents_ratio=float(OWNER_CONTENTS_RATIO),
        )
        # Filter by minimum of owner/renter thresholds; actual group-specific filtering in TP update
        _min_thr = min(_THR_OWNER, _THR_RENTER)
        ratio_used = {str(t): (float(r) if float(r) >= _min_thr else 0.0) for t, r in ratio_raw.items()}
        print(f"ratio_used: {sum(v>0.0 for v in ratio_used.values()):,} tracts with ratio >= {_min_thr}")

        # -----------------------------------------------------------------------
        # STEP 7.5: Apply insurance financials
        # Computes payouts, premiums, and OOP costs based on decisions
        # -----------------------------------------------------------------------
        FIN_HH2, FIN_TRACT = run_finance_for_year(
            year=y,
            state_event=STATE_V_WITH_DMG,
            depth_map=dmap,
            ratio_by_tract=ratio_used,
            decisions=dec_prev,
            policy=policy_dict,
            idxer=None,
            gate_by_decisions=not NO_ACTION,
            renters_have_structure=False,
            tract_col="tract_geoid",
            modules_root=ROOT / "modules",
            output_dir=FIN_DIR,
            save_csv=output_mgr.should_save('finance_households'),
            compact_output=(output_mode != 'full'),
            premium={**_premium_for_state_df(STATE_V),
                     "contents_share_owner": OWNER_CONTENTS_RATIO,
                     "owner_insures_both": OWNER_INSURES_BOTH},
            owner_contents_ratio=OWNER_CONTENTS_RATIO,
            owner_insures_both=OWNER_INSURES_BOTH,
            ffe_ft=FFE_FT,
        )

        # Maintain ordering
        STATE = STATE.sort_values(["i"]).reset_index(drop=True)
        STATE["tract_geoid"] = STATE["tract_geoid"].astype(str)

        # Flood bookkeeping
        flood_rows.extend(build_flood_bookkeeping_rows(
            y, TRACTS, ratio_raw, ratio_used, dmap, _THR_OWNER, _THR_RENTER, DEPTH_THRESHOLD_M
        ))

        if not NO_ACTION:
            # -------------------------------------------------------------------
            # STEP 7.6: Run agent decision model (BASELINE MODE ONLY)
            # Uses Bayesian predictors to determine actions, updates TP
            # -------------------------------------------------------------------
            from modules.actions.pipeline import run_one_year_mgmix_fast
            # RNG: If --deterministic flag is true, use the same SEED every year (no +y offset)
            _year_seed = SEED if getattr(args, "deterministic", False) else (SEED + y)
            dec, STATE_NEXT_NO_EH, CHG = run_one_year_mgmix_fast(
                year=y, state=STATE,
                predictor_owner=predictor_owner, predictor_renter=predictor_renter,
                params_owner=PARAMS_OWNER, params_renter=PARAMS_RENTER, tp_cfg=TP_CFG,
                ratio_by_tract=ratio_used,
                overlay_policy_names={"owner": "owner_standard", "renter": "renter_contents"},
                rng=np.random.RandomState(_year_seed), years_step=YEARS_STEP,
                reset_clock_on_flood=True,
                action_dyn={
                    "eh_one_time": EH_ONE_TIME,
                    "eh_once_min_ft": EH_ONCE_MIN_FT,
                    "eh_once_max_ft": EH_ONCE_MAX_FT,
                    "eh_step_ft": EH_STEP_FT,
                    "eh_cap_ft": EH_CAP_FT,
                    "apply_RL": APPLY_RL,
                    "rl_backfill": RL_BACKFILL,
                    "apply_BP": APPLY_BP,
                    "bp_backfill": BP_BACKFILL,
                    "bp_sampler": BP_SAMPLER,
                    "cv_nj": CV_NJ,
                },
                depth_m_by_tract=dmap, rl_dest_k_best=50,
                decision_threshold=DECISION_THRESHOLD,
                draw_bounds=CFG.get("draw_bounds", None),
            )

            print(
                f"[year {y}] Households: {len(STATE):,} | Flooded tracts: {sum(v>0.0 for v in ratio_used.values()):,} | "
                f"Decisions: {len(dec):,} | Finance (hh): {len(FIN_HH2):,} | Finance (tract): {len(FIN_TRACT):,}"
            )
            FIN_TRACT.to_csv(FIN_DIR / f"finance_tract_{y}.csv", index=False, encoding="utf-8-sig")

            # Save decisions & update dec_prev
            dec_prev = save_decisions_and_update_prev(dec, STATE, y, DEC_DIR)

            # Per-tract action shares & EH coverage
            save_action_shares(dec, STATE_NEXT_NO_EH, y, DEC_DIR)

            # TP trajectory (aggregate from household-level to tract-level for output)
            _tp_agg = STATE_NEXT_NO_EH[["tract_geoid","group","TP"]].copy()
            _tp_agg["tract_geoid"] = _tp_agg["tract_geoid"].astype(str)
            _tp_own = _tp_agg[_tp_agg["group"]=="owner"].groupby("tract_geoid")["TP"].mean()
            _tp_ren = _tp_agg[_tp_agg["group"]=="renter"].groupby("tract_geoid")["TP"].mean()
            for tg in TRACTS:
                tp_rows.append({
                    "year": y, "tract_geoid": tg,
                    "TP_owner": float(_tp_own.get(tg, np.nan)),
                    "TP_renter": float(_tp_ren.get(tg, np.nan)),
                    "phase": "after",
                })

            # Advance to next-year state
            STATE = advance_state_for_next_year(STATE, STATE_NEXT_NO_EH, y)

            # Save next-year state snapshot
            if output_mgr.should_save('state_next_csvs'):
                save_next_year_state(STATE, y, dec_prev, STATES_DIR)
        else:
            # NO_ACTION mode
            print(f"[year {y}] NO_ACTION mode — decisions/TP skipped, finance gating off")
            dec_prev = handle_no_action_year(STATE, y, DEC_DIR, tp_rows, None)

            STATE_OUT = STATE.copy()
            STATE_OUT["tract_geoid"] = STATE_OUT["tract_geoid"].astype(str)
            STATE_OUT.insert(1, "year", y + 1)
            STATE_OUT = finalize_has_fi(STATE_OUT, dec_prev=dec_prev, sticky=False)
            if output_mgr.should_save('state_next_csvs'):
                STATE_OUT.to_csv(STATES_DIR / f"state_next_{y+1}.csv", index=False, encoding="utf-8-sig")

    # =========================================================================
    # SECTION 8: POST-SIMULATION FINALIZATION
    # =========================================================================
    # Always save TP trajectory and flood trigger data (needed for SA)
    tp_traj = pd.DataFrame(tp_rows)
    if output_mgr.should_save('tp_trajectory'):
        tp_traj.to_csv(OUTPUT_DIR / "tp_traj.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(flood_rows).to_csv(OUTPUT_DIR / "flood_years_by_tract.csv", index=False, encoding="utf-8-sig")

    # Aggregate results across years and generate visualization plots
    ENABLE_PLOTS_CFG = bool(CFG.get("output", {}).get("enable_plots", True))
    if ENABLE_PLOTS_CFG and not getattr(args, "no_plots", False):
        finalize_and_plot(YEARS, OUTPUT_DIR, FIN_DIR, DEC_DIR, VIS_DIR, CFG, OUT_ROOT, args, tp_rows, flood_rows, output_mgr)
    else:
        print("[info] plotting skipped (enable_plots YAML=", ENABLE_PLOTS_CFG, ", --no-plots=", getattr(args, 'no_plots', False), ")")
    
    # Save output manager summary report
    output_mgr.create_summary_report(OUTPUT_DIR / "output_summary.json")
    
    return OUTPUT_DIR, OUT_ROOT, VIS_DIR, args, CFG


def main(argv: list[str] | None = None) -> None:
    """
    Main simulation entry point.
    
    Orchestrates the full FLOODABM simulation flow:
    1. Parse configuration (YAML + CLI)
    2. Load household and flood depth data
    3. Initialize tract-level psychology
    4. Run yearly simulation loop
    5. Aggregate outputs and generate plots
    
    If --compare-flood-or is specified, automatically runs BOTH baseline and worst
    scenarios, then generates comparison plots.
    
    Args:
        argv: Optional command line arguments (for testing)
    """
    # Pre-parse to check if --compare-flood-or is specified
    ap_check = argparse.ArgumentParser(add_help=False)
    ap_check.add_argument("--compare-flood-or", action="store_true")
    ap_check.add_argument("--compare-all-models", action="store_true", 
                          help="Run baseline vs worst comparison for ALL registered models")
    ap_check.add_argument("--compare-severe-years", type=str, default="")
    ap_check.add_argument("--list-models", action="store_true")
    ap_check.add_argument("--run-all", action="store_true")
    ap_check.add_argument("--model-dir", type=str, default=None)
    known_args, remaining_argv = ap_check.parse_known_args(argv)
    
    # Handle --list-models: show available models and exit
    if known_args.list_models:
        from utils.model_registry import ModelRegistry
        PROJECT_ROOT = Path(__file__).resolve().parent
        registry = ModelRegistry(PROJECT_ROOT)  # Models at project root
        registry.print_summary()
        return
    
    # Handle --compare-all-models: run baseline vs worst for ALL models
    if known_args.compare_all_models:
        from utils.model_registry import ModelRegistry
        PROJECT_ROOT = Path(__file__).resolve().parent
        registry = ModelRegistry(PROJECT_ROOT)
        models = registry.list_models()
        
        print("=" * 70)
        print(f"[compare-all-models] Running Baseline vs Worst for ALL {len(models)} models...")
        print("=" * 70)
        
        for i, model_name in enumerate(models, 1):
            model_info = registry.get_model(model_name)
            print(f"\n{'='*70}")
            print(f"[{i}/{len(models)}] Model: {model_name} ({model_info['name']})")
            print("=" * 70)
            
            # Build new argv with --model-dir and --compare-flood-or
            # CRITICAL: Filter out recursive and multi-scenario flags to avoid argparse errors
            recursive_flags = {"--compare-all-models", "--run-all", "--compare-flood-or"}
            raw_argv = argv if argv is not None else sys.argv[1:]
            new_argv = [a for a in raw_argv if a not in recursive_flags]
            # Also filter out arguments that take values if they are related to comparison
            # (though currently compare-mode has a default, so it's safer to leave as is unless it causes issues)
            new_argv.extend(["--model-dir", model_name, "--compare-flood-or"])
            
            # Call main recursively with the new arguments
            main(argv=new_argv)
        
        print("\n" + "=" * 70)
        print("[compare-all-models] COMPLETE! All models processed.")
        print("=" * 70)
        return
    
    if known_args.run_all:
        from utils.model_registry import ModelRegistry
        PROJECT_ROOT = Path(__file__).resolve().parent
        registry = ModelRegistry(PROJECT_ROOT)  # Models at project root
        models = registry.list_models()
        
        print("=" * 70)
        print(f"[run-all] Running simulation for ALL {len(models)} models...")
        print("=" * 70)
        
        for i, model_name in enumerate(models, 1):
            model_info = registry.get_model(model_name)
            print(f"\n{'='*70}")
            print(f"[{i}/{len(models)}] Model: {model_name} ({model_info['name']})")
            print("=" * 70)
            
            # Build new argv with --model-dir set
            # CRITICAL: Filter out recursive flags
            recursive_flags = {"--compare-all-models", "--run-all", "--compare-flood-or"}
            raw_argv = argv if argv is not None else sys.argv[1:]
            new_argv = [a for a in raw_argv if a not in recursive_flags]
            new_argv.extend(["--model-dir", model_name])
            run_single_scenario(argv=new_argv, skip_comparison_plots=True)
        
        print("\n" + "=" * 70)
        print("[run-all] COMPLETE! All models processed.")
        print("=" * 70)
        return
    
    if known_args.compare_flood_or:
        # =====================================================================
        # BATCH MODE: Run both baseline and worst, then compare
        # =====================================================================
        print("=" * 70)
        print("[compare-flood-or] Running BOTH baseline and worst scenarios...")
        print("=" * 70)
        
        # Run baseline first (skip comparison plots in individual run)
        # CRITICAL: Filter out --compare-flood-or from argv to avoid recursion loop in argparse
        # We must ALSO filter out dependent args like --compare-mode or --compare-severe-years
        # if they are present in the raw argv to prevent the sub-run from seeing "invalid" flags
        recursive_flags = {"--compare-flood-or", "--compare-all-models", "--run-all"}
        raw_argv = argv if argv is not None else sys.argv[1:]
        sub_argv = []
        skip_next = False
        for i, a in enumerate(raw_argv):
            if skip_next:
                skip_next = False
                continue
            if a in recursive_flags:
                continue
            # Some flags take values - we should probably filter those too if they are comparison-specific
            if a in ("--compare-mode", "--compare-severe-years"):
                skip_next = True
                continue
            sub_argv.append(a)
        
        print("\n" + "=" * 70)
        print("[1/2] Running BASELINE scenario...")
        print("=" * 70)
        baseline_output_dir, out_root, vis_dir, args, cfg = run_single_scenario(
            argv=sub_argv, 
            scenario_override="baseline", 
            skip_comparison_plots=True
        )
        
        # Run worst scenario
        print("\n" + "=" * 70)
        print("[2/2] Running WORST scenario...")
        print("=" * 70)
        worst_output_dir, _, _, _, _ = run_single_scenario(
            argv=sub_argv, 
            scenario_override="worst", 
            skip_comparison_plots=True
        )
        
        # Generate comparison plots with both scenarios completed
        print("\n" + "=" * 70)
        print("[compare-flood-or] Generating Baseline vs Worst comparison plots...")
        print("=" * 70)
        
        from utils.plots_comparision_scenario import (
            plot_cum_payoutrate_and_damage_by_group,
            export_spatial_inputs_cumrate_perhh,
        )
        
        sev = None
        if known_args.compare_severe_years.strip():
            sev = [int(s) for s in known_args.compare_severe_years.split(",") if s.strip().isdigit()]
        
        try:
            # out_root should be at model_dir level (e.g., outputs/baseline/)
            model_level_root = baseline_output_dir.parent
            plot_cum_payoutrate_and_damage_by_group(
                out_root=model_level_root, vis_dir=vis_dir,
                severe_years=sev or [2011, 2014, 2021],
                save_name="cum_payoutrate_and_damage_by_group",
            )
            print("[OK] Comparison plot saved to:", vis_dir / "compare" / "cum_payoutrate_and_damage_by_group.png")
        except Exception as e:
            print("[warn] cum payout-rate + damage figure failed:", e)
        
        try:
            # out_root should be at model_dir level (e.g., outputs/baseline/)
            model_level_root = baseline_output_dir.parent
            export_spatial_inputs_cumrate_perhh(
                out_root=model_level_root,
                out_csv=vis_dir / "spatial_inputs_cumrate_perhh_early_late.csv",
                early=(2011, 2016), late=(2017, 2023), cfg=cfg, late_is_cumulative=False,
            )
            print("[OK] Spatial export saved to:", vis_dir / "spatial_inputs_cumrate_perhh_early_late.csv")
        except Exception as e:
            print("[warn] spatial export failed:", e)
        
        print("\n" + "=" * 70)
        print("[compare-flood-or] COMPLETE!")
        print(f"  Baseline output: {baseline_output_dir}")
        print(f"  Worst output:    {worst_output_dir}")
        print(f"  Comparison plots: {vis_dir}")
        print("=" * 70)
    else:
        # Single scenario mode (original behavior)
        run_single_scenario(argv=argv)


if __name__ == "__main__":
    main()