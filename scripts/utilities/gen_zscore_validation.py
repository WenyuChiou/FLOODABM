# -*- coding: utf-8 -*-
"""
Generate z-score validation figure using:
- OLD actual data + act_z (unchanged)
- NEW simulated avg payout/claim from threshold=0.35 baseline
- Recompute sim_z on same filtered subset
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# ── paths ──
ROOT = Path(__file__).resolve().parent.parent.parent  # project root
OLD_TS = Path(r"C:\Users\wenyu\OneDrive - Lehigh University\Desktop\Lehigh\NSF-project\ABM\paper\validation\sim_vs_actual_standardized - 複製.csv")
OLD_SC = Path(r"C:\Users\wenyu\OneDrive - Lehigh University\Desktop\Lehigh\NSF-project\ABM\paper\validation\sim_vs_actual_standardized.csv")
NEW_COUNTY = ROOT / "outputs" / "baseline" / "baseline" / "visualization" / "validation" / "nfip_county_year_comparison.csv"
OUT_DIR = ROOT / "outputs" / "baseline" / "baseline" / "visualization" / "validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COUNTIES = ["Essex", "Morris", "Passaic"]

# ── load ──
ts = pd.read_csv(OLD_TS)   # all years (for time series plot)
sc = pd.read_csv(OLD_SC)   # filtered (for R² computation)
new = pd.read_csv(NEW_COUNTY)
new["sim_avg"] = np.where(new["sim_claims"] > 0, new["sim_payout"] / new["sim_claims"], np.nan)

# ── replace simulated in ts (all years) ──
ts = ts.merge(new[["county", "year", "sim_avg"]], on=["county", "year"], how="left")
ts["simulated"] = ts["sim_avg"]
ts.drop(columns=["sim_avg"], inplace=True)

# Recompute sim_z per county (on ALL years, same as old approach)
for county in COUNTIES:
    mask = ts["county"] == county
    vals = ts.loc[mask, "simulated"]
    mu, sd = vals.mean(), vals.std()
    ts.loc[mask, "sim_z"] = (vals - mu) / sd if sd > 0 else 0

# ── replace simulated in sc (filtered) ──
sc = sc.merge(new[["county", "year", "sim_avg"]], on=["county", "year"], how="left")
sc["simulated"] = sc["sim_avg"]
sc.drop(columns=["sim_avg"], inplace=True)

# Recompute sim_z per county on filtered subset
for county in COUNTIES:
    mask = sc["county"] == county
    vals = sc.loc[mask, "simulated"]
    mu, sd = vals.mean(), vals.std()
    sc.loc[mask, "sim_z"] = (vals - mu) / sd if sd > 0 else 0

# ── save updated CSVs ──
ts.to_csv(OUT_DIR / "sim_vs_actual_standardized_all.csv", index=False, encoding="utf-8-sig")
sc.to_csv(OUT_DIR / "sim_vs_actual_standardized_filtered.csv", index=False, encoding="utf-8-sig")
print("Saved updated CSVs.")

# ── print R² per county ──
print("\n=== Per-county R² (z-score, filtered subset) ===")
for county in COUNTIES:
    c = sc[sc["county"] == county].dropna(subset=["sim_z", "act_z"])
    if len(c) >= 2:
        r = np.corrcoef(c["sim_z"], c["act_z"])[0, 1]
        print(f"  {county:8s}: n={len(c):2d}, r={r:.3f}, R²={r**2:.3f}")

pw = sc.dropna(subset=["sim_z", "act_z"])
if len(pw) >= 2:
    sl, ic, rv, pv, _ = stats.linregress(pw["sim_z"].values, pw["act_z"].values)
    print(f"  {'Pooled':8s}: n={len(pw):2d}, r={rv:.3f}, R²={rv**2:.3f}, p={pv:.4f}")

# ── compare with old R² ──
print("\n=== Comparison with old stochastic R² ===")
print(f"  {'County':8s} {'Old R²':>8} {'New R²':>8}")
old_r2 = {"Essex": 0.50, "Morris": 0.41, "Passaic": 0.46}
for county in COUNTIES:
    c = sc[sc["county"] == county].dropna(subset=["sim_z", "act_z"])
    r = np.corrcoef(c["sim_z"], c["act_z"])[0, 1]
    print(f"  {county:8s} {old_r2[county]:>8.2f} {r**2:>8.2f}")

# ── publication figure (3-panel) ──
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
from matplotlib.patches import Patch

fig, axes = plt.subplots(3, 1, figsize=(9, 10.5))

for idx, county in enumerate(COUNTIES):
    ax = axes[idx]
    c = ts[ts["county"] == county].sort_values("year")

    # Warm-up shading
    ax.axvspan(2010.5, 2012.5, color="0.5", alpha=0.12, zorder=0)

    # Simulated (blue solid, circles)
    ax.plot(c["year"], c["sim_z"], color="#2563eb", marker="o",
            markersize=6, linewidth=2.0, label="Simulated", zorder=3)

    # Actual (red dashed, squares) — skip NaN
    c_act = c.dropna(subset=["act_z"])
    ax.plot(c_act["year"], c_act["act_z"], color="#ef4444", marker="s",
            markersize=6, linewidth=2.0, linestyle="--", label="Actual", zorder=3)

    ax.set_ylabel("z-score")
    ax.set_title(f"{county} County (2011\u20132023)", fontweight="bold")
    ax.set_xlim(2010.2, 2023.8)
    ax.set_xticks(list(range(2011, 2024)))
    ax.tick_params(length=5, width=1)

    # R² from filtered data
    cf = sc[sc["county"] == county].dropna(subset=["sim_z", "act_z"])
    if len(cf) >= 2:
        r = np.corrcoef(cf["sim_z"], cf["act_z"])[0, 1]
        r2 = r ** 2
        ax.text(0.97, 0.05, f"$R^2 = {r2:.2f}$",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=12, bbox=dict(boxstyle="round,pad=0.3",
                                       facecolor="white", edgecolor="black", linewidth=0.8))

    # Panel label
    ax.text(-0.08, 1.05, f"({chr(97 + idx)})", transform=ax.transAxes,
            fontsize=16, fontweight="bold", va="top")

    if idx == 0:
        handles, labels = ax.get_legend_handles_labels()
        handles.insert(0, Patch(facecolor="0.5", alpha=0.12, label="Warm-up"))
        labels.insert(0, "Warm-up")
        ax.legend(handles=handles, labels=labels, loc="upper right",
                  fontsize=11, frameon=True, framealpha=0.95)

    if idx < 2:
        ax.set_xticklabels([])
    else:
        ax.set_xlabel("Year")

fig.tight_layout(h_pad=1.5)
plt.subplots_adjust(hspace=0.25)

for ext in ["png", "pdf"]:
    out = OUT_DIR / f"nfip_zscore_validation_paper.{ext}"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"  -> {out}")

plt.close(fig)
print("\nDone.")
