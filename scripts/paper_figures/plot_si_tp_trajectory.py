# -*- coding: utf-8 -*-
"""
SI Figure: Threat Perception (TP) trajectories for homeowners vs renters.
Shows tract-level median ± IQR for both groups on the same panel.
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300,
    "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.labelsize": 12, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 10, "axes.linewidth": 0.8,
    "axes.grid": True, "grid.alpha": 0.15, "grid.linestyle": "--",
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
})

# ── Load data ──
BASE = Path("outputs/baseline/baseline")
tp = pd.read_csv(BASE / "tp_traj.csv", dtype={"tract_geoid": str})
tp = tp[tp["phase"] == "after"]  # post-update TP values

YEARS = sorted(tp["year"].unique())
SEVERE = [2011, 2021]
OWNER_COLOR = "#1f77b4"
RENTER_COLOR = "#d62828"
SEV_COLOR = "#fff3cd"

# ── Compute tract-level stats ──
def get_stats(df, col):
    medians, q25s, q75s = [], [], []
    for y in YEARS:
        vals = df[df["year"] == y].groupby("tract_geoid")[col].mean().values
        medians.append(np.median(vals))
        q25s.append(np.percentile(vals, 25))
        q75s.append(np.percentile(vals, 75))
    return np.array(medians), np.array(q25s), np.array(q75s)

o_med, o_q25, o_q75 = get_stats(tp, "TP_owner")
r_med, r_q25, r_q75 = get_stats(tp, "TP_renter")

# ── Plot ──
fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(len(YEARS))

# Severe year shading
for i, y in enumerate(YEARS):
    if y in SEVERE:
        ax.axvspan(i - 0.4, i + 0.4, facecolor=SEV_COLOR, alpha=0.8, zorder=0)

# Owner
ax.plot(x, o_med, "-o", ms=6, lw=2.2, color=OWNER_COLOR, label="Homeowner", zorder=3)
ax.fill_between(x, o_q25, o_q75, alpha=0.15, color=OWNER_COLOR, zorder=1)

# Renter
ax.plot(x, r_med, "-s", ms=6, lw=2.2, color=RENTER_COLOR, label="Renter", zorder=3)
ax.fill_between(x, r_q25, r_q75, alpha=0.15, color=RENTER_COLOR, zorder=1)

# Endpoint annotations
for med, color, name in [(o_med, OWNER_COLOR, "Owner"), (r_med, RENTER_COLOR, "Renter")]:
    for idx in [0, -1]:
        val = med[idx]
        offset = 0.012 if name == "Owner" else -0.025
        ax.annotate(f"{val:.2f}", xy=(x[idx], val),
                    xytext=(x[idx], val + offset),
                    fontsize=9, fontweight="bold", color=color, ha="center", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                              edgecolor=color, linewidth=0.6, alpha=1.0), zorder=5)

ax.set_ylabel("Threat perception (TP)")
ax.set_xlabel("Year")
ax.set_title("Threat Perception by Housing Tenure (2011\u20132023)")
ax.set_xticks(x)
ax.set_xticklabels([str(y) for y in YEARS], rotation=0)
ax.set_ylim(0.15, 0.75)
ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="black", framealpha=1.0)

fig.tight_layout()

out_dir = BASE / "visualization" / "si"
out_dir.mkdir(parents=True, exist_ok=True)
SM_DIR = Path(r"C:\Users\wenyu\OneDrive - Lehigh University\Desktop\Lehigh\NSF-project\ABM\paper\draft\v2_sections\SM")
SM_DIR.mkdir(parents=True, exist_ok=True)
for ext in ["png", "pdf"]:
    out = out_dir / f"si_tp_trajectory.{ext}"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved: {out}")
    # Also copy to SM directory
    sm_out = SM_DIR / f"Figure_S1_tp_trajectory.{ext}"
    fig.savefig(sm_out, dpi=300, bbox_inches="tight")
    print(f"Saved: {sm_out}")

plt.close()
print("Done.")
