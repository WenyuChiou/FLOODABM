# -*- coding: utf-8 -*-
"""
Phase 3: TP Decay Parameter Calibration (Owner/Renter)
======================================================
Fits the saturated-tau exponential decay model:
  TP(t) = TP0 * exp(-ln2 * w_eff * Eff(t))
  w_eff = alpha * (1 - PA) + beta * SC
  tau(t) = tau_inf - (tau_inf - tau0) * exp(-k*t)

Grid search over (alpha, beta), differential evolution for (tau0, delta, k).
Joint selection with cross-group constraints, fine-tuning, and bootstrap CIs.

Outputs: calibration_summary.xlsx, decay curves, CI ribbons, residual plots, heatmaps
"""
from __future__ import annotations
import sys, os
sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path
# Ensure this script's directory is on path for config import
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as cfg
import core.data as data
import core.optimizer as optimizer
import core.visualizer as visualizer
import pandas as pd
import json


def main():
    print("=" * 60)
    print("Phase 3: TP Decay Calibration — Owner/Renter")
    print("=" * 60)

    # 1. Load Data
    print(f"\nLoading data from: {cfg.DATA_FILE}")
    owner_df = data.load_group_sheet(cfg.DATA_FILE, "owner_cal")
    renter_df = data.load_group_sheet(cfg.DATA_FILE, "renter_cal")
    print(f"  Owner:  n={len(owner_df)}")
    print(f"  Renter: n={len(renter_df)}")

    # 2. Grid Search
    print("\nRunning Parallel Grid Optimization...")
    owner_grid = optimizer.run_grid_search(owner_df, "owner", n_jobs=-1)
    renter_grid = optimizer.run_grid_search(renter_df, "renter", n_jobs=-1)

    # 3. Selection & Fine-Tuning
    print("\nSelecting & Fine-Tuning Parameters...")
    try:
        best_rough = optimizer.select_best_joint_pair(owner_grid, renter_grid)
        best_final = optimizer.fine_tune_best_pair(best_rough, owner_df, renter_df)

        print("\n>>> FINAL RESULT <<<")
        for grp in ["owner", "renter"]:
            r = best_final[grp]
            print(f"  {grp:>6s}: alpha={r['alpha']:.4f}, beta={r['beta']:.4f}, "
                  f"tau0={r['tau0']:.2f}, tau_inf={r['tau_inf']:.2f}, k={r['k']:.4f}, "
                  f"RMSE={r['RMSE']:.4f}")

    except ValueError as e:
        print(f"ERROR: {e}")
        return

    # 4. Bootstrap (Confidence Intervals)
    print("\nRunning Bootstrap for Confidence Intervals (N=100)...")
    owner_boot = optimizer.run_bootstrap(
        owner_df, best_final["owner"]["alpha"], best_final["owner"]["beta"], "owner", n_boot=100
    )
    renter_boot = optimizer.run_bootstrap(
        renter_df, best_final["renter"]["alpha"], best_final["renter"]["beta"], "renter", n_boot=100
    )

    # 5. Save & Visualize
    vis_dir = cfg.OUTDIR
    vis_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving results to: {vis_dir}")

    visualizer.save_results(best_final, owner_grid, renter_grid, vis_dir)
    visualizer.plot_overlay(best_final, owner_df, renter_df, vis_dir)
    visualizer.plot_confidence_intervals(best_final, owner_boot, renter_boot, owner_df, renter_df, vis_dir)
    visualizer.plot_residuals(best_final, owner_df, renter_df, vis_dir)
    visualizer.plot_rmse_heatmap(owner_grid, "owner", vis_dir)
    visualizer.plot_rmse_heatmap(renter_grid, "renter", vis_dir)

    # 6. Export parameters for ABM config
    params_for_abm = {}
    for grp in ["owner", "renter"]:
        r = best_final[grp]
        params_for_abm[grp] = {
            "alpha": round(float(r["alpha"]), 4),
            "beta": round(float(r["beta"]), 4),
            "tau0": round(float(r["tau0"]), 4),
            "tau_inf": round(float(r["tau_inf"]), 4),
            "k": round(float(r["k"]), 4),
            "RMSE": round(float(r["RMSE"]), 4),
        }

    params_path = vis_dir / "tp_decay_params_for_abm.json"
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(params_for_abm, f, indent=2)
    print(f"\n[OK] ABM parameters exported to: {params_path}")

    # Also export as CSV for easy reading
    rows = []
    for grp in ["owner", "renter"]:
        r = best_final[grp]
        rows.append({
            "group": grp,
            "alpha": r["alpha"], "beta": r["beta"],
            "tau0": r["tau0"], "tau_inf": r["tau_inf"], "k": r["k"],
            "TP0_hat": r["TP0_hat"], "RMSE": r["RMSE"], "MAE": r["MAE"],
            "TP5_TP0": r["TP5_TP0"], "TP10_TP0": r["TP10_TP0"], "TP15_TP0": r["TP15_TP0"],
        })
    pd.DataFrame(rows).to_csv(vis_dir / "phase3_metrics_table.csv", index=False, encoding="utf-8-sig")
    print("[OK] phase3_metrics_table.csv")

    print("\n" + "=" * 60)
    print("Phase 3 complete. All outputs in:", vis_dir)
    print("=" * 60)


if __name__ == "__main__":
    main()
