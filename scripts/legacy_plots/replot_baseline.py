# -*- coding: utf-8 -*-
"""
Regenerate all baseline figures from existing output data.
Uses the corrected severe flood year markers (2011, 2014, 2021).
"""
import sys, pandas as pd
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

BASE = Path("outputs/baseline/baseline")
FIN_DIR = BASE / "finance"
VIS_DIR = BASE / "visualization"
TP_TRAJ = pd.read_csv(BASE / "tp_traj.csv")

print("=" * 60)
print("Regenerating baseline figures with corrected flood year markers")
print("=" * 60)

# 1) Main plot_all_outputs (TP, finance, decisions, damage)
from utils.plots import plot_all_outputs
plot_all_outputs(TP_TRAJ, FIN_DIR, VIS_DIR, poster=False)

# 2) Stacked finance (owner/renter separate) — the one that had (2014,2021,2022,2023)
from utils.plots import plot_fin_stacked_owner_renter_separate
try:
    plot_fin_stacked_owner_renter_separate(FIN_DIR, VIS_DIR)
    print("[OK] plot_fin_stacked_owner_renter_separate")
except Exception as e:
    print(f"[warn] plot_fin_stacked_owner_renter_separate: {e}")

# 3) Scenario comparison (baseline vs worst)
WORST_ROOT = Path("outputs/baseline/worst")
if WORST_ROOT.exists():
    from utils.plots_comparision_scenario import (
        plot_cum_payoutrate_and_damage_by_group,
        export_spatial_inputs_cumrate_perhh,
    )
    try:
        model_root = BASE.parent  # outputs/baseline/
        plot_cum_payoutrate_and_damage_by_group(
            out_root=model_root, vis_dir=VIS_DIR,
            severe_years=[2011, 2014, 2021],
            save_name="cum_payoutrate_and_damage_by_group",
        )
        print("[OK] scenario comparison figure")
    except Exception as e:
        print(f"[warn] scenario comparison: {e}")

# 4) New combined figures
print("\n--- New combined figures ---")
import subprocess
subprocess.run([sys.executable, "plot_fig4_dual_yaxis.py"], check=False)
subprocess.run([sys.executable, "plot_fig_finance_combined.py"], check=False)

print("\n" + "=" * 60)
print("All figures regenerated!")
print("=" * 60)
