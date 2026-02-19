# sa_main.py — Sensitivity on MG/NMG flood ratio thresholds (0.1–0.8), reuse-aware
from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple, List
import sys

# ==== SA helpers & plotting (ensure utils/sa_dr/__init__.py exports the following items) ====
from utils.sa_dr import (
    run_experiment,
    series_cum_financial, series_cum_payout,
    totals_tradeoff_damage_payout_financial,
    dist_financial_per_household,
    plot_fin_focus_dashboard,
)

# Action/Behavior plots
from utils.sa_dr.plotting_actions import (
    plot_actions_grouped_boxplot,
    plot_actions_heatmaps_mg_nmg,
)

from utils.sa_dr.timeseries import (
    mean_over_years_policyholder_payout,
    mean_over_years_damage_per_household,
    ts_flood_damage_total,  # <- Use this to assemble damage_by_year (annual average flood damage)
    series_cum_payout_tenure
)

from utils.sa_dr.plotting import (
    plot_cumavg_damage_per_hh_owner_renter_delta,
    plot_rate_delta_cum_dual,
    plot_rate_delta_cum_tenure_series,
    series_cumavg_per_hh_damage,
    plot_sa_combined_dashboard_2x2
)
# ---- extra plotting helpers (reuse existing style & severe-year bands)
from utils.sa_dr.plotting import _set_style as _plt_set_style
from utils.sa_dr.plotting import _add_severe_year_bands as _plt_add_severe_year_bands
import matplotlib.pyplot as plt
from utils import cleanup_sa_outputs  # Import cleanup script module


ROOT = Path(__file__).resolve().parent
EXP_ROOT = ROOT / "outputs" / "experiments"


# ---------------- Utilities ----------------
def _normalize_run_dir(run_dir: Path) -> Path:
    """
    If outputs/.../baseline/* exists, use baseline as the real run root directory.
    Handles double-nested baseline/baseline cases by searching recursively.
    """
    run_dir = Path(run_dir)
    
    # Keep descending into baseline subdirectories until we find data or run out of baselines
    for _ in range(5):  # Limit recursion depth
        b = run_dir / "baseline"
        if not b.exists():
            break
        # Check if this baseline level has data directories
        if (b / "finance").exists() or (b / "decisions").exists() or (b / "vulnerability").exists():
            run_dir = b
            continue
        # Check if there's another baseline level below (even if current level has no data)
        if (b / "baseline").exists():
            run_dir = b
            continue
        break
    return run_dir


def _has_complete_outputs(run_dir: Path) -> bool:
    """
    Determine if it can be directly reused: allow finance in run_dir or run_dir/baseline, and tolerate nesting (recursive).
    """
    run_dir = Path(run_dir)

    # First try standard / baseline two finance locations
    finance_dirs = [
        run_dir / "finance",
        run_dir / "baseline" / "finance",
    ]
    for fdir in finance_dirs:
        if fdir.exists():
            if any(fdir.glob("finance_households_*.csv")):
                return True
            if (fdir / "finance_tract_all_years.csv").exists():
                return True

    # Fallback: Recursively find any finance_* files (tolerate more complex nested experiment structures)
    if list(run_dir.rglob("finance_households_*.csv")):
        return True
    if list(run_dir.rglob("finance_tract_all_years.csv")):
        return True

    # Vulnerability also tolerates baseline nesting
    vuln_candidates = [
        run_dir / "vulnerability" / "flood_damage" / "flood_damage_tract_ALL_years.csv",
        run_dir / "baseline" / "vulnerability" / "flood_damage" / "flood_damage_tract_ALL_years.csv",
    ]
    if any(p.exists() for p in vuln_candidates):
        return True

    # Existing dashboard png is also considered reusable
    if (run_dir / "SA_RT_financial_focus.png").exists():
        return True
    if (run_dir / "baseline" / "SA_RT_financial_focus.png").exists():
        return True

    return False


def _series_to_df(series_dict: Dict[str, Tuple[List[int], List[float]]],
                  value_name: str = "payout") -> pd.DataFrame:
    """
    Flatten { 'MG=0.2,NMG=0.3': ([years], [values]), ... } or { 'Shock=0.5': ([years], [values]), ... }
    into a long table.
    For MG/NMG format: columns = ['year','mg','nmg', value_name]
    For Shock format: columns = ['year','shock', value_name] (mg/nmg set to shock value)
    """
    rows = []
    for lab, (xs, ys) in series_dict.items():
        # Try to parse MG/NMG format first
        if "MG=" in lab and "NMG=" in lab:
            try:
                mg = float(lab.split(",")[0].split("=")[1])
                nmg = float(lab.split(",")[1].split("=")[1])
            except Exception:
                mg, nmg = np.nan, np.nan
        # Try to parse Shock format
        elif "Shock=" in lab:
            try:
                shock_val = float(lab.split("=")[1])
                mg = shock_val  # Use shock as pseudo mg/nmg for visualization
                nmg = shock_val
            except Exception:
                mg, nmg = np.nan, np.nan
        else:
            mg, nmg = np.nan, np.nan
        for x, y in zip(xs, ys):
            rows.append({"year": int(x), "mg": mg, "nmg": nmg, value_name: float(y)})
    return pd.DataFrame(rows)


def _damage_by_year_from_run(run_dir: Path) -> Optional[Dict[int, float]]:
    """
    Use ts_flood_damage_total(run_dir) to get annual flood damage, return {year: damage_value}.
    Priority columns: 'avg_damage_usd' -> 'damage_usd' -> 'total_damage_usd' (all annual basis).
    Return None if not found.
    """
    try:
        df = ts_flood_damage_total(_normalize_run_dir(run_dir))
    except Exception:
        return None
    if df is None or len(df) == 0:
        return None

    # Fault-tolerant column fetching
    cand_cols = ["avg_damage_usd", "damage_usd", "total_damage_usd"]
    dmg_col = None
    for c in cand_cols:
        if c in df.columns:
            dmg_col = c
            break
    if dmg_col is None:
        # Try to find numeric columns containing 'damage' and sum them
        num_damage_cols = [c for c in df.columns
                           if "damage" in c.lower() and pd.api.types.is_numeric_dtype(df[c])]
        if not num_damage_cols:
            return None
        df["__damage__"] = df[num_damage_cols].sum(axis=1)
        dmg_col = "__damage__"

    # year column fault tolerance
    ycol = "year" if "year" in df.columns else ("Year" if "Year" in df.columns else None)
    if ycol is None:
        return None

    # Clean and output dict
    out = {}
    for y, v in zip(df[ycol], pd.to_numeric(df[dmg_col], errors="coerce")):
        if pd.notna(v):
            out[int(y)] = float(v)
    return out if out else None


# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser(
        description="SA on MG/NMG flood ratio thresholds OR shock_scale with optional reuse of previous runs."
    )
    # Experiment type selection
    ap.add_argument(
        "--experiment-type", 
        choices=["flood-ratio", "shock-scale"], 
        default="flood-ratio",
        help="Type of SA experiment: 'flood-ratio' (MG/NMG thresholds) or 'shock-scale' (0.3/0.5/0.7)"
    )
    # Flood ratio parameters
    ap.add_argument("--min", type=float, default=0.1, help="min threshold (inclusive, for flood-ratio)")
    ap.add_argument("--max", type=float, default=0.8, help="max threshold (inclusive, for flood-ratio)")
    ap.add_argument("--step", type=float, default=0.1, help="step for thresholds (for flood-ratio)")
    # Shock scale parameters
    ap.add_argument(
        "--shock-values", type=str, default="0.3,0.4,0.5,0.6,0.7",
        help="Comma-separated shock_scale values to test for owner×renter grid (shock-scale experiment)"
    )
    ap.add_argument(
        "--fixed-ratio", type=float, default=0.5,
        help="Fixed flood ratio threshold for shock-scale experiment (default: 0.5)"
    )
    ap.add_argument(
        "--search-root", type=str, default=str(EXP_ROOT),
        help="Where to look for previous runs (default: outputs/experiments)."
    )
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--reuse", action="store_true", help="reuse existing outputs if present (default behavior)")
    g.add_argument("--rerun", action="store_true", help="force re-run simulations")
    ap.add_argument("--strict-reuse", action="store_true",
                    help="If set with --reuse, error out when no prior run is found for a scenario.")
    ap.add_argument("--no-plots", action="store_true", help="Skip all plotting.")
    ap.add_argument("--plot-only", action="store_true", help="Skip simulations and only generate plots from existing data.")
    ap.add_argument(
        "--output-mode", 
        choices=["full", "summary", "minimal"], 
        default="summary",
        help="Output detail level: 'full' (~325MB/run), 'summary' (~57KB/run, recommended for SA), 'minimal' (~2KB/run)"
    )
    args = ap.parse_args()

    reuse = args.reuse or not args.rerun
    search_root = Path(args.search_root).resolve()
    EXP_ROOT.mkdir(parents=True, exist_ok=True)
    print("[info] search_root =", search_root)
    print("[info] EXP_ROOT    =", EXP_ROOT)
    print(f"[info] experiment-type = {args.experiment_type}")
    
    # Create experiment-specific figure directory to avoid overwriting plots
    experiment_type = args.experiment_type
    if experiment_type == "flood-ratio":
        FIG_DIR = EXP_ROOT / "fig_flood_ratio"
    elif experiment_type == "shock-scale":
        FIG_DIR = EXP_ROOT / "fig_shock_scale"
    else:
        FIG_DIR = EXP_ROOT / "fig"  # fallback
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[info] Plots will be saved to: {FIG_DIR}")

    # Determine subdirectory for experiment outputs (cleanup: keep root clean)
    if experiment_type == "flood-ratio":
        case_subdir = "flood_ratio"
    elif experiment_type == "shock-scale":
        case_subdir = "shock_scale"
    else:
        case_subdir = "other"
    
    CASE_ROOT = EXP_ROOT / case_subdir
    CASE_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"[info] Simulation outputs will be organized in: {CASE_ROOT}")

    runs: List[Tuple[Tuple[float, float], Path]] = []


    # =====================================================================
    # SCANNING MODE (Plot Only)
    # =====================================================================
    if args.plot_only:
        print(f"[info] Plot-only mode: Scanning {EXP_ROOT} and {CASE_ROOT} for {experiment_type} results...")
        import re
        
        # Helper to scan a directory
        def _scan_dir(target_dir: Path, regex_pat):
            found = []
            if not target_dir.exists(): return found
            for d in target_dir.iterdir():
                if d.is_dir():
                    m = regex_pat.match(d.name)
                    if m and _has_complete_outputs(d):
                        vals = tuple(float(g.replace("_", ".")) for g in m.groups())
                        found.append((vals, d))
            return found

        if experiment_type == "flood-ratio":
            pat = re.compile(r"SA_RT_MG(\d+_\d+)_NMG(\d+_\d+)")
            runs.extend(_scan_dir(CASE_ROOT, pat)) # Priority: subdir
            runs.extend(_scan_dir(EXP_ROOT, pat))  # Fallback: root
            
        elif experiment_type == "shock-scale":
            pat = re.compile(r"SA_SHOCK_O(\d+_\d+)_R(\d+_\d+)")
            runs.extend(_scan_dir(CASE_ROOT, pat))
            runs.extend(_scan_dir(EXP_ROOT, pat))
        
        # Deduplicate by parameters (prefer subdir if duplicates)
        # runs list has (params, path). 
        unique_runs = {}
        for p, path in runs:
            # If we already have this param, keep the first one (subdir is scanned first so it wins)
            if p not in unique_runs:
                unique_runs[p] = path
            # Need to decide if we overwrite? 
            # If scanned in order: CASE_ROOT then EXP_ROOT.
            # CASE_ROOT runs are preferred. So if p already in dict, don't overwrite.
        
        runs = [(p, path) for p, path in unique_runs.items()]

        print(f"[info] Found {len(runs)} existing valid run directories.")
        if not runs:
            print("[warn] No results found. Exiting.")
            sys.exit(0)

    # =====================================================================
    # EXPERIMENT 1: Flood Ratio SA (original behavior)
    # =====================================================================
    elif experiment_type == "flood-ratio":
        mg_values = [round(x, 1) for x in np.arange(args.min, args.max + 1e-9, args.step)]
        nmg_values = [round(x, 1) for x in np.arange(args.min, args.max + 1e-9, args.step)]
        # Auto-include baseline (0.5) if not present, to ensure delta plots work
        if 0.5 not in mg_values:
            mg_values.append(0.5)
            mg_values.sort()
        if 0.5 not in nmg_values:
            nmg_values.append(0.5)
            nmg_values.sort()
        print(f"[info] Flood Ratio SA: MG={mg_values}, NMG={nmg_values}")

        for mg_thr in mg_values:
            for nmg_thr in nmg_values:
                run_dir: Optional[Path] = None
                label_name = f"SA_RT_MG{str(mg_thr).replace('.', '_')}_NMG{str(nmg_thr).replace('.', '_')}"
                
                # Check reuse (Subdir priority, then Flat fallback)
                if reuse:
                    # 1. Check structured path
                    candidate = search_root / case_subdir / label_name
                    if candidate.exists() and _has_complete_outputs(candidate):
                         print(f"[reuse] (structured) {candidate}")
                         run_dir = candidate
                    # 2. Check legacy flat path
                    elif (search_root / label_name).exists() and _has_complete_outputs(search_root / label_name):
                         print(f"[reuse] (legacy flat) {search_root / label_name}")
                         run_dir = search_root / label_name
                    elif args.strict_reuse:
                         raise FileNotFoundError(f"[strict-reuse] No outputs for {label_name}")

                if run_dir is None:
                    # Start new simulation in SUBDIR
                    # Determine exp_name relative to EXP_ROOT, usually run_experiment handles paths
                    # We pass "folder/label"
                    exp_rel_path = f"{case_subdir}/{label_name}"
                    
                    print(f"[rerun] start simulation for MG={mg_thr} / NMG={nmg_thr} -> {exp_rel_path}")
                    overrides = {
                        "flood": {
                            "ratio_threshold_MG": float(mg_thr),
                            "ratio_threshold_NMG": float(nmg_thr),
                        }
                    }
                    run_dir = Path(run_experiment(exp_rel_path, overrides, output_mode=args.output_mode))
                    print(f"[rerun] done: {run_dir}")
                    
                    # Auto-cleanup after run to save space
                    if args.output_mode == "summary":
                        print(f"[cleanup] Cleaning up {run_dir}...")
                        cleanup_sa_outputs.cleanup_scenario(run_dir, keep_viz=False)

                runs.append(((mg_thr, nmg_thr), run_dir))
                print(f"[info] MG={mg_thr:g} / NMG={nmg_thr:g} -> {run_dir}")

    # =====================================================================
    # EXPERIMENT 2: Shock Scale SA - shock_scale_owner × shock_scale_renter grid
    # Fixed flood ratio, varying shock_scale for owner and renter independently
    # =====================================================================
    elif experiment_type == "shock-scale":
        shock_values = [float(v.strip()) for v in args.shock_values.split(",")]
        # Auto-include baseline (0.5) if not present
        if 0.5 not in shock_values:
            shock_values.append(0.5)
            shock_values.sort()
        fixed_ratio = args.fixed_ratio
        print(f"[info] Shock Scale SA: owner_values={shock_values}, renter_values={shock_values}")
        print(f"[info] Grid: {len(shock_values)}×{len(shock_values)} = {len(shock_values)**2} combinations")
        print(f"[info] Fixed flood ratio: MG={fixed_ratio}, NMG={fixed_ratio}")

        # Nested loop: owner × renter combinations
        for shock_owner in shock_values:
            for shock_renter in shock_values:
                run_dir: Optional[Path] = None
                # Naming: SA_SHOCK_O{owner}_R{renter}
                label_name = f"SA_SHOCK_O{str(shock_owner).replace('.', '_')}_R{str(shock_renter).replace('.', '_')}"

                # Reuse logic: Subdir > Flat
                if reuse:
                    cand = search_root / case_subdir / label_name
                    cand_flat = search_root / label_name
                    if cand.exists() and _has_complete_outputs(cand):
                        print(f"[reuse] (structured) {cand}")
                        run_dir = cand
                    elif cand_flat.exists() and _has_complete_outputs(cand_flat):
                        print(f"[reuse] (legacy flat) {cand_flat}")
                        run_dir = cand_flat
                    elif args.strict_reuse:
                        raise FileNotFoundError(f"[strict-reuse] {label_name}")

                if run_dir is None:
                    # Rerun in SUBDIR
                    exp_rel_path = f"{case_subdir}/{label_name}"
                    print(f"[rerun] start simulation for shock_owner={shock_owner}, shock_renter={shock_renter} -> {exp_rel_path}")
                    overrides = {
                        "flood": {
                            "ratio_threshold_MG": float(fixed_ratio),
                            "ratio_threshold_NMG": float(fixed_ratio),
                        },
                        "tp_shock": {
                            "shock_scale_owner": float(shock_owner),
                            "shock_scale_renter": float(shock_renter),
                        }
                    }
                    run_dir = Path(run_experiment(exp_rel_path, overrides, output_mode=args.output_mode))
                    print(f"[rerun] done: {run_dir}")

                    # Auto-cleanup after run to save space
                    if args.output_mode == "summary":
                        print(f"[cleanup] Cleaning up {run_dir}...")
                        cleanup_sa_outputs.cleanup_scenario(run_dir, keep_viz=False)

                # Use (shock_owner, shock_renter) as the parameter tuple
                runs.append(((shock_owner, shock_renter), run_dir))
                print(f"[info] shock_owner={shock_owner:g} / shock_renter={shock_renter:g} -> {run_dir}")

    # ---- Finance and Summary Charts ----
    trade_points: Dict[str, Tuple[float, float]] = {}   # label -> (total_damage, total_payout)
    trade_bubbles: Dict[str, float] = {}               # label -> total_financial
    fin_series: Dict[str, Tuple[List[int], List[float]]] = {}  # label -> (years[], cum_avg_financial[])
    pay_series: Dict[str, Tuple[List[int], List[float]]] = {}  # label -> (years[], cum_avg_payout[])
    dist_fin: Dict[str, np.ndarray] = {}               # label -> np.ndarray (Annual average Premium+OOP per household)
    trade_points_avg: Dict[str, Tuple[float, float]] = {}  # label -> (avg_damage_per_hh_all, avg_payout_per_policyholder)
    owner_pay_series, renter_pay_series = {}, {}
    owner_dmg_series, renter_dmg_series = {}, {}
    bad_runs: List[str] = []


    # Read outputs run by run (tolerate baseline/ sub-layer)
    for (param1, param2), d_root in runs:
        # Generate label based on experiment type
        if experiment_type == "shock-scale":
            lab = f"NMG={param1:g},MG={param2:g}"  # param1 is shock_owner, param2 is shock_renter
        else:
            lab = f"MG={param1:g},NMG={param2:g}"  # param1 is mg_thr, param2 is nmg_thr
        d = _normalize_run_dir(d_root)

        try:
            # —— Cumulative financial / payout (follow original method) ——
            fin_series[lab] = series_cum_financial(d)
            pay_series[lab] = series_cum_payout(d)

            tot_damage, tot_payout, tot_financial = totals_tradeoff_damage_payout_financial(d)
            trade_points[lab]  = (tot_damage, tot_payout)
            trade_bubbles[lab] = tot_financial

            dist_fin[lab] = dist_financial_per_household(d)

            avg_x = mean_over_years_damage_per_household(d)
            avg_y = mean_over_years_policyholder_payout(d)
            trade_points_avg[lab] = (avg_x, avg_y)

            # —— Unified lab here too, use d directly (already normalized) ——
            own_pay_xy, ren_pay_xy = series_cum_payout_tenure(d)
            owner_pay_series[lab]  = own_pay_xy
            renter_pay_series[lab] = ren_pay_xy

            # —— Damage series for combined dashboard ——
            yrs_dmg, own_dmg, ren_dmg = series_cumavg_per_hh_damage(d)
            owner_dmg_series[lab]  = (yrs_dmg, own_dmg)
            renter_dmg_series[lab] = (yrs_dmg, ren_dmg)

        except Exception as e:
            print(f"[warn] skip {lab} due to read error: {e}")
            bad_runs.append(lab)
            continue

    # Dashboard (only use successful scenarios)
    fin_series_ok = {k: v for k, v in fin_series.items() if k not in bad_runs}
    pay_series_ok = {k: v for k, v in pay_series.items() if k not in bad_runs}
    trade_points_avg_ok = {k: v for k, v in trade_points_avg.items() if k not in bad_runs}
    trade_bubbles_ok = {k: v for k, v in trade_bubbles.items() if k not in bad_runs}
    owner_pay_series_ok  = {k: v for k, v in owner_pay_series.items()  if k not in bad_runs}
    renter_pay_series_ok = {k: v for k, v in renter_pay_series.items() if k not in bad_runs}
    owner_dmg_series_ok  = {k: v for k, v in owner_dmg_series.items()  if k not in bad_runs}
    renter_dmg_series_ok = {k: v for k, v in renter_dmg_series.items() if k not in bad_runs}

    # 2-Panel Dashboard: Cumulative household financial cost & payout
    # Disabled per user request (Code removed)

    # Get damage_by_year from a successful run
    damage_by_year: Optional[Dict[int, float]] = None
    for (param1, param2), d_root in runs:
        # Use correct label format based on experiment type
        if experiment_type == "shock-scale":
            lab = f"NMG={param1:g},MG={param2:g}"
        else:
            lab = f"MG={param1:g},NMG={param2:g}"
        if lab in fin_series_ok:  # Only pick successful scenarios
            damage_by_year = _damage_by_year_from_run(d_root)
            if damage_by_year:
                print("[info] loaded damage_by_year from", _normalize_run_dir(d_root))
            else:
                print("[warn] damage_by_year not found; plotting non-normalized deltas")
            break

    if not args.no_plots:
        # Determine baseline and highlights based on experiment type
        if experiment_type == "shock-scale":
            baseline_label = "NMG=0.3,MG=0.3"
            highlights_map = {
                "NMG=0.3,MG=0.3": {"color": "#1f77b4", "label": "NMG=0.3, MG=0.3"},
                "NMG=0.3,MG=0.7": {"color": "#2ca02c", "label": "NMG=0.3, MG=0.7"},
                "NMG=0.7,MG=0.3": {"color": "#d62728", "label": "NMG=0.7, MG=0.3"},
                "NMG=0.7,MG=0.7": {"color": "#9467bd", "label": "NMG=0.7, MG=0.7"},
            }
        else:
            baseline_label = "MG=0.5,NMG=0.5"
            highlights_map = {
                "MG=0.1,NMG=0.1": {"color": "#1f77b4", "label": "MG=0.1, NMG=0.1"},
                "MG=0.1,NMG=0.8": {"color": "#2ca02c", "label": "MG=0.1, NMG=0.8"},
                "MG=0.8,NMG=0.1": {"color": "#d62728", "label": "MG=0.8, NMG=0.1"},
                "MG=0.8,NMG=0.8": {"color": "#9467bd", "label": "MG=0.8, NMG=0.8"},
            }

        # Disabled per user request (focus on tenure-split version instead)
        # Note: plot_rate_delta_cum_dual logic removed


    if not args.no_plots:
        try:
            plot_rate_delta_cum_tenure_series(
                series_owner=owner_pay_series_ok,
                series_renter=renter_pay_series_ok,
                damage_by_year=damage_by_year,
                baseline=baseline_label,
                highlights=highlights_map,
                severe_years=(2011, 2014, 2021),
                out_png=EXP_ROOT / "fig" / "payout_rate_delta_cum_tenure.png",
                as_percent=True,
            )
        except Exception as e:
            print(f"[warn] plot_rate_delta_cum_tenure_series skipped: {e}")

    # ===== New figures: payout fan chart / endpoint heatmap / yearly delta =====
    # All disabled per user request (Code removed)

    # Summary table (per run)
    rows = []
    for (param1, param2), d_root in runs:
        # Generate label based on experiment type
        if experiment_type == "shock-scale":
            lab = f"NMG={param1:g},MG={param2:g}"
        else:
            lab = f"MG={param1:g},NMG={param2:g}"

        def _last(series_dict, key):
            ys = series_dict.get(key, ([], []))[1]
            return float(ys[-1]) if ys else np.nan

        tot_damage, tot_payout = trade_points.get(lab, (np.nan, np.nan))
        tot_financial = float(trade_bubbles.get(lab, np.nan))

        h = dist_fin.get(lab, np.array([]))
        h = h[~np.isnan(h)] if h.size else np.array([])
        hh_mean = float(np.mean(h)) if h.size else np.nan
        hh_p50  = float(np.percentile(h, 50)) if h.size else np.nan
        hh_p90  = float(np.percentile(h, 90)) if h.size else np.nan

        # Different columns based on experiment type
        if experiment_type == "shock-scale":
            rows.append({
                "shock_scale_NMG": param1,
                "shock_scale_MG": param2,
                "cum_avg_financial_usd": _last(fin_series, lab),
                "cum_avg_payout_usd": _last(pay_series, lab),
                "total_damage_usd": float(tot_damage) if tot_damage is not None else np.nan,
                "total_payout_usd": float(tot_payout) if tot_payout is not None else np.nan,
                "total_financial_usd": tot_financial,
                "hh_financial_mean_usd": hh_mean,
                "hh_financial_p50_usd": hh_p50,
                "hh_financial_p90_usd": hh_p90,
                "status": ("ok" if lab not in bad_runs else "failed"),
            })
        else:
            rows.append({
                "ratio_threshold_MG": param1,
                "ratio_threshold_NMG": param2,
                "cum_avg_financial_usd": _last(fin_series, lab),
                "cum_avg_payout_usd": _last(pay_series, lab),
                "total_damage_usd": float(tot_damage) if tot_damage is not None else np.nan,
                "total_payout_usd": float(tot_payout) if tot_payout is not None else np.nan,
                "total_financial_usd": tot_financial,
                "hh_financial_mean_usd": hh_mean,
                "hh_financial_p50_usd": hh_p50,
                "hh_financial_p90_usd": hh_p90,
                "status": ("ok" if lab not in bad_runs else "failed"),
            })

    # Save summary with appropriate filename
    if experiment_type == "shock-scale":
        summary_path = EXP_ROOT / "SA_SHOCK_financial_summary.csv"
    else:
        summary_path = EXP_ROOT / "SA_RT_financial_summary.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"[OK] wrote summary: {summary_path}")

    # === Action plots: grouped boxplot + MG×NMG heatmap ===
    # Generate run_map with appropriate labels
    if experiment_type == "shock-scale":
        run_map_all = {f"NMG={p1:g},MG={p2:g}": _normalize_run_dir(rdir) for ((p1, p2), rdir) in runs}
    else:
        run_map_all = {f"MG={p1:g},NMG={p2:g}": _normalize_run_dir(rdir) for ((p1, p2), rdir) in runs}
    run_map_ok  = {k: v for k, v in run_map_all.items() if k not in bad_runs}

    # === New: Cumulative "Average per-HH flood damage" (Owner / Renter) ===
    if not args.no_plots:
        try:
            # Determine baseline and highlights based on experiment type (reuse logic or define again if scope issue)
            if experiment_type == "shock-scale":
                baseline_label = "NMG=0.3,MG=0.3"
                highlights_map = {
                    "NMG=0.3,MG=0.3": {"color": "#1f77b4", "label": "NMG=0.3, MG=0.3"},
                    "NMG=0.3,MG=0.7": {"color": "#2ca02c", "label": "NMG=0.3, MG=0.7"},
                    "NMG=0.7,MG=0.3": {"color": "#d62728", "label": "NMG=0.7, MG=0.3"},
                    "NMG=0.7,MG=0.7": {"color": "#9467bd", "label": "NMG=0.7, MG=0.7"},
                }
            else:
                baseline_label = "MG=0.5,NMG=0.5"
                highlights_map = {
                    "MG=0.1,NMG=0.1": {"color": "#1f77b4", "label": "MG=0.1, NMG=0.1"},
                    "MG=0.1,NMG=0.8": {"color": "#2ca02c", "label": "MG=0.1, NMG=0.8"},
                    "MG=0.8,NMG=0.1": {"color": "#d62728", "label": "MG=0.8, NMG=0.1"},
                    "MG=0.8,NMG=0.8": {"color": "#9467bd", "label": "MG=0.8, NMG=0.8"},
                }

            FIG_DIR.mkdir(parents=True, exist_ok=True)
            plot_cumavg_damage_per_hh_owner_renter_delta(
                run_dirs_map=run_map_ok,
                baseline=baseline_label,
                highlights=highlights_map,
                severe_years=(2011, 2014, 2021),
                out_png=FIG_DIR / "cumavg_damage_per_hh_owner_renter_delta.png",
                legend_loc="upper left",
                annotate_endpoints=False,
            )
            print("[OK] wrote: cumavg_damage_per_hh_owner_renter_delta.png")
        except Exception as e:
            print("[warn] cumavg per-HH damage figure failed:", e)
    
    # ---- (2) Payout Rate Difference Chart (vs baseline, by tenure) ----
    if not args.no_plots:
        try:
            FIG_DIR.mkdir(parents=True, exist_ok=True)
            plot_rate_delta_cum_tenure_series(
                series_owner=owner_pay_series_ok,
                series_renter=renter_pay_series_ok,
                damage_by_year=damage_by_year,
                baseline=baseline_label,
                highlights=highlights_map,
                severe_years=(2011, 2014, 2021),
                out_png=FIG_DIR / "payout_rate_delta_cum_tenure.png",
                as_percent=True,
                legend_loc="upper left",
            )
            print("[OK] wrote: payout_rate_delta_cum_tenure.png")
        except Exception as e:
            print("[warn] payout rate delta figure failed:", e)

    # ---- (3) Combined 2x2 Dashboard (Damage + Payout) ----
    if not args.no_plots:
        try:
            FIG_DIR.mkdir(parents=True, exist_ok=True)
            plot_sa_combined_dashboard_2x2(
                dmg_owner=owner_dmg_series_ok,
                dmg_renter=renter_dmg_series_ok,
                pay_owner=owner_pay_series_ok,
                pay_renter=renter_pay_series_ok,
                damage_by_year=damage_by_year,
                baseline=baseline_label,
                highlights=highlights_map,
                severe_years=(2011, 2014, 2021),
                out_png = FIG_DIR / f"sa_{experiment_type.replace('-', '_')}_2x2.png"
            )
            print(f"[OK] wrote: {FIG_DIR / f'sa_{experiment_type.replace('-', '_')}_2x2.png'}")
        except Exception as e:
            print("[warn] combined 2x2 dashboard failed:", e)


    # Boxplot disabled per user request
    # if not args.no_plots:
    #     try:
    #         plot_actions_grouped_boxplot(
    #             run_dirs_map=run_map_ok,
    #             selections=[("Owner(FI)", "Owner", "FI"),
    #                         ("Renter(FI)", "Renter", "FI"),
    #                         ("BP", None, "BP"), ("RL", None, "RL"), ("DN", None, "DN")],
    #             pattern="",
    #             out_png=EXP_ROOT / "actions_grouped_boxplot.png",
    #             title="Tract-level action shares across MG/NMG scenarios (median over years)",
    #         )
    #     except Exception as e:
    #         print("[warn] grouped boxplot failed:", e)

    # Heatmap only for flood-ratio experiment (requires mg_values/nmg_values grid)
    # Disabled per user request
    # Note: plot_actions_heatmaps_mg_nmg logic removed




if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[abort] interrupted by user")
        sys.exit(130)
