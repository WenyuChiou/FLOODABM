# -*- coding: utf-8 -*-
"""
SI Figure: TP trajectories as side-by-side boxplots (owner vs renter) per year.
Shows tract-level distribution so 2021 Ida effect is visible in the spread.
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300,
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.labelsize": 11, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 10, "axes.linewidth": 0.8,
    "axes.grid": True, "grid.alpha": 0.15, "grid.linestyle": "--",
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
})

BASE = Path("outputs/baseline/baseline")
tp = pd.read_csv(BASE / "tp_traj.csv", dtype={"tract_geoid": str})
tp = tp[tp["phase"] == "after"]

YEARS = sorted(tp["year"].unique())
SEVERE = [2011, 2021]
OWNER_COLOR = "#1f77b4"
RENTER_COLOR = "#d62828"
SEV_COLOR = "#fff3cd"

# ── Build tract-level TP per year ──
owner_data = []
renter_data = []
for y in YEARS:
    ydf = tp[tp["year"] == y]
    owner_data.append(ydf.groupby("tract_geoid")["TP_owner"].mean().values)
    renter_data.append(ydf.groupby("tract_geoid")["TP_renter"].mean().values)

# ── Plot ──
fig, ax = plt.subplots(figsize=(10, 5))

width = 0.35
positions_o = np.arange(len(YEARS)) - width / 2
positions_r = np.arange(len(YEARS)) + width / 2

# Severe year shading
for i, y in enumerate(YEARS):
    if y in SEVERE:
        ax.axvspan(i - 0.5, i + 0.5, facecolor=SEV_COLOR, alpha=0.6, zorder=0)

bp_o = ax.boxplot(owner_data, positions=positions_o, widths=width * 0.85,
                  patch_artist=True, showfliers=True, zorder=2,
                  flierprops=dict(marker="o", ms=4, markerfacecolor=OWNER_COLOR, alpha=0.5),
                  medianprops=dict(color="white", lw=1.5))
bp_r = ax.boxplot(renter_data, positions=positions_r, widths=width * 0.85,
                  patch_artist=True, showfliers=True, zorder=2,
                  flierprops=dict(marker="s", ms=4, markerfacecolor=RENTER_COLOR, alpha=0.5),
                  medianprops=dict(color="white", lw=1.5))

for patch in bp_o["boxes"]:
    patch.set_facecolor(OWNER_COLOR)
    patch.set_alpha(0.7)
for patch in bp_r["boxes"]:
    patch.set_facecolor(RENTER_COLOR)
    patch.set_alpha(0.7)

# Also overlay median lines for continuity
o_meds = [np.median(d) for d in owner_data]
r_meds = [np.median(d) for d in renter_data]
ax.plot(np.arange(len(YEARS)), o_meds, "-", color=OWNER_COLOR, lw=1.5, alpha=0.6, zorder=1)
ax.plot(np.arange(len(YEARS)), r_meds, "-", color=RENTER_COLOR, lw=1.5, alpha=0.6, zorder=1)

# Legend
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor=OWNER_COLOR, alpha=0.7, label="Homeowner"),
                   Patch(facecolor=RENTER_COLOR, alpha=0.7, label="Renter")],
          loc="upper left", frameon=True, facecolor="white", edgecolor="black")

ax.set_xticks(np.arange(len(YEARS)))
ax.set_xticklabels([str(y) for y in YEARS])
ax.set_xlabel("Year")
ax.set_ylabel("Threat perception (TP)")
ax.set_title("Tract-Level Threat Perception Distribution by Housing Tenure (2011\u20132023)")
ax.set_ylim(0.0, 1.05)

fig.tight_layout()

# Save to both locations
out_dir = BASE / "visualization" / "si"
out_dir.mkdir(parents=True, exist_ok=True)
SM_DIR = Path(r"C:\Users\wenyu\OneDrive - Lehigh University\Desktop\Lehigh\NSF-project\ABM\paper\draft\v2_sections\SM")
SM_DIR.mkdir(parents=True, exist_ok=True)

for ext in ["png", "pdf"]:
    fig.savefig(out_dir / f"si_tp_boxplot.{ext}", dpi=300, bbox_inches="tight")
    fig.savefig(SM_DIR / f"Figure_S1_tp_boxplot.{ext}", dpi=300, bbox_inches="tight")

plt.close()
print("Done. Saved to both locations.")
