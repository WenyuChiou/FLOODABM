# -*- coding: utf-8 -*-
"""
Generate publication-quality 4-panel NFIP validation figure.

Layout (vertical):
  (a) Essex County — standardized avg payout/claim time series
  (b) Morris County — standardized avg payout/claim time series
  (c) Passaic County — standardized avg payout/claim time series
  (d) Correlation scatter (z-score, 2013–2023, excl. no-claim years)

Data: previous validation CSVs (per-claim avg payout, standardized).
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# ── paths ──
ROOT = Path(__file__).resolve().parent
VAL_DIR = Path(r"C:\Users\wenyu\OneDrive - Lehigh University\Desktop\Lehigh\NSF-project\ABM\paper\validation")
TS_CSV = VAL_DIR / "sim_vs_actual_standardized - 複製.csv"   # all years (for time series)
SC_CSV = VAL_DIR / "sim_vs_actual_standardized.csv"           # filtered (for scatter)
OUT_DIR = ROOT / "outputs" / "baseline" / "baseline" / "visualization" / "validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COUNTIES = ["Essex", "Morris", "Passaic"]
WARMUP = [2011, 2012]

# ── load data ──
df_ts = pd.read_csv(TS_CSV)
df_sc = pd.read_csv(SC_CSV)

# ── print summary ──
print("=== Per-county R² (filtered, post-warmup, excl. no-claim) ===")
for county in COUNTIES:
    c = df_sc[df_sc["county"] == county].dropna(subset=["sim_z", "act_z"])
    r = np.corrcoef(c["sim_z"], c["act_z"])[0, 1]
    print(f"  {county:8s}: n={len(c):2d}, r={r:.3f}, R²={r**2:.3f}")

pw = df_sc.dropna(subset=["sim_z", "act_z"])
sl, ic, rv, pv, _ = stats.linregress(pw["sim_z"].values, pw["act_z"].values)
print(f"  {'Pooled':8s}: n={len(pw):2d},    r={rv:.3f}, R²={rv**2:.3f}, p={pv:.4f}")

# ── style ──
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.2,
    "grid.linestyle": "--",
})

# ── figure ──
fig, axes = plt.subplots(3, 1, figsize=(9, 10.5))

# ── panels (a)-(c): county time series (standardized avg payout/claim) ──
from matplotlib.patches import Patch
for idx, county in enumerate(COUNTIES):
    ax = axes[idx]
    c = df_ts[df_ts["county"] == county].sort_values("year")

    # Warm-up shading
    ax.axvspan(2010.5, 2012.5, color="0.5", alpha=0.12, zorder=0)

    # Simulated (blue solid, circles)
    ax.plot(c["year"], c["sim_z"], color="#2563eb", marker="o",
            markersize=6, linewidth=2.0, label="Simulated", zorder=3)

    # Actual (red dashed, squares) — skip NaN years
    c_act = c.dropna(subset=["act_z"])
    ax.plot(c_act["year"], c_act["act_z"], color="#ef4444", marker="s",
            markersize=6, linewidth=2.0, linestyle="--", label="Actual", zorder=3)

    ax.set_ylabel("z-score")
    ax.set_title(f"{county} County (2011\u20132023)", fontweight="bold")
    ax.set_xlim(2010.2, 2023.8)
    ax.set_xticks(list(range(2011, 2024)))
    ax.tick_params(length=5, width=1)

    # Pearson r annotation (top-center)
    c_filt = df_sc[df_sc["county"] == county].dropna(subset=["sim_z", "act_z"])
    r_val = np.corrcoef(c_filt["sim_z"], c_filt["act_z"])[0, 1]
    ax.text(0.5, 0.95, f"r = {r_val:.2f}", transform=ax.transAxes,
            fontsize=12, ha="center", va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.7", alpha=0.9))

    # Panel label
    ax.text(-0.08, 1.05, f"({chr(97 + idx)})", transform=ax.transAxes,
            fontsize=16, fontweight="bold", va="top")

    if idx == 0:
        handles, labels = ax.get_legend_handles_labels()
        handles.insert(0, Patch(facecolor="0.5", alpha=0.12, label="Warm-up"))
        labels.insert(0, "Warm-up")
        ax.legend(handles=handles, labels=labels, loc="upper right",
                  fontsize=11, frameon=True, framealpha=0.95)

    # Only show x-tick labels on panel (c)
    if idx < 2:
        ax.set_xticklabels([])
    else:
        ax.set_xlabel("Year")

fig.tight_layout(h_pad=1.5)
plt.subplots_adjust(hspace=0.25)

# Save
for ext in ["png", "pdf"]:
    out = OUT_DIR / f"nfip_validation_paper.{ext}"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"  -> {out}")

plt.close(fig)
print("Done.")
