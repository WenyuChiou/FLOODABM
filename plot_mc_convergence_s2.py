# -*- coding: utf-8 -*-
"""
plot_mc_convergence_s2.py — Regenerate Figure S2 (MC convergence)
Style matches paper figures (white background, clean lines, 300 dpi).
Removes old ±2% tolerance band.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MC_DIR = ROOT / "outputs" / "montecarlo"
OUT_PATH = Path(r"C:\Users\wenyu\OneDrive - Lehigh University\Desktop\Lehigh\NSF-project\ABM\paper\draft\submission_files\Figure_S2.png")

# Paper style
plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "lines.linewidth": 2.0,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "axes.facecolor": "white",
    "figure.facecolor": "white",
})

# Load MC runs
df = pd.read_csv(MC_DIR / "mc_initpsy_runs.csv")
payout = df["avg_payout_usd_ph"].values
n_runs = len(payout)

# Running mean
running_mean = np.cumsum(payout) / np.arange(1, n_runs + 1)
grand_mean = payout.mean()

# CV at n=30
cv = payout.std() / payout.mean() * 100

fig, ax = plt.subplots(figsize=(7, 4.2))

# Running mean line
x = np.arange(1, n_runs + 1)
ax.plot(x, running_mean, color="#0072B2", marker="o", markersize=5,
        linewidth=1.8, label="Running mean", zorder=3)

# Grand mean dashed line
ax.axhline(grand_mean, color="black", linestyle="--", linewidth=1.0,
           label=f"Grand mean (${grand_mean:,.0f})", zorder=2)

ax.set_xlabel("Number of runs")
ax.set_ylabel("Payout per household (USD)")
ax.set_xlim(0.5, n_runs + 0.5)
ax.legend(loc="upper right", frameon=True, fancybox=False,
          edgecolor="gray", framealpha=0.9)

ax.set_axisbelow(True)

fig.tight_layout()
fig.savefig(OUT_PATH, bbox_inches="tight")
plt.close(fig)

print(f"[OK] Figure S2 saved: {OUT_PATH}")
print(f"     Grand mean: ${grand_mean:,.0f}, CV: {cv:.2f}%")
