# -*- coding: utf-8 -*-
"""
Fig 4 (new): Cumulative flood damage / cumulative actual loss ratio
Shows how much insurance and adaptation buffer the damage.
Higher ratio = more protection (damage is X times actual loss).
2 panels: (a) Homeowner, (b) Renter, each with baseline + no-adaptation lines.
"""
import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from pathlib import Path
import glob, re

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from utils.plots_modular.style import set_paper_style, panel_label, COLORS

set_paper_style()

OUT_ROOT = ROOT / "outputs" / "baseline" / "baseline"
WORST_ROOT = ROOT / "outputs" / "baseline" / "worst"
VIS = OUT_ROOT / "visualization" / "compare"
VIS.mkdir(parents=True, exist_ok=True)

SEVERE = [2011, 2021]
C_BASE = "#0077b6"
C_WORST = "#d62828"

def load_scenario(root):
    fin_frames = []
    for f in sorted(glob.glob(str(root / "finance/finance_tract_*.csv"))):
        m = re.search(r"tract_(\d{4})\.csv", f)
        if m:
            df = pd.read_csv(f, dtype={"tract_geoid": str})
            df["year"] = int(m.group(1))
            fin_frames.append(df)
    fin = pd.concat(fin_frames, ignore_index=True)

    fin_yr = fin.groupby("year").agg(
        owner_gross=("owner_gross_total_kUSD", "sum"),
        renter_gross=("renter_gross_total_kUSD", "sum"),
        payout_total=("payout_total_kUSD", "sum"),
        oop_total=("oop_total_kUSD", "sum"),
        owner_hh=("owner_households", "sum"),
        renter_hh=("renter_households", "sum"),
    ).reset_index()

    # Use finance gross damage (kUSD→USD) as tenure-specific damage
    fin_yr["owner_dmg"] = fin_yr["owner_gross"] * 1000
    fin_yr["renter_dmg"] = fin_yr["renter_gross"] * 1000

    merged = fin_yr.fillna(0).sort_values("year")

    total_gross = merged["owner_gross"] + merged["renter_gross"]
    merged["owner_payout_share"] = np.where(total_gross > 0,
                                             merged["owner_gross"] / total_gross, 0.5)
    merged["owner_payout"] = merged["payout_total"] * merged["owner_payout_share"] * 1000
    merged["renter_payout"] = merged["payout_total"] * (1 - merged["owner_payout_share"]) * 1000
    return merged


def cum_ratio(df, group):
    """Cumulative damage / cumulative actual loss ratio per year."""
    hh = np.maximum(df[f"{group}_hh"].values, 1)
    dmg = df[f"{group}_dmg"].values.astype(float)
    pay = df[f"{group}_payout"].values.astype(float)
    cum_dmg = np.cumsum(dmg / hh)
    cum_loss = np.cumsum((dmg - pay) / hh)
    # Avoid division by zero
    ratio = np.where(cum_loss > 0, cum_dmg / cum_loss, 1.0)
    return ratio, cum_dmg, cum_loss


base = load_scenario(OUT_ROOT)
worst = load_scenario(WORST_ROOT)
years = sorted(base["year"].unique())
x = np.arange(len(years))

# Compute ratios
rB_O, _, _ = cum_ratio(base, "owner")
rW_O, _, _ = cum_ratio(worst, "owner")
rB_R, _, _ = cum_ratio(base, "renter")
rW_R, _, _ = cum_ratio(worst, "renter")

# Print summary
print("=== Cumulative Damage / Actual Loss Ratio (2023) ===")
print(f"  Owner  baseline: {rB_O[-1]:.2f}x   no-adapt: {rW_O[-1]:.2f}x")
print(f"  Renter baseline: {rB_R[-1]:.2f}x   no-adapt: {rW_R[-1]:.2f}x")

# ── Plot ──
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

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)

for ax, rB, rW, title, plbl in [
    (ax1, rB_O, rW_O, "Homeowner", "(a)"),
    (ax2, rB_R, rW_R, "Renter", "(b)"),
]:
    # Severe year shading
    for i, y in enumerate(years):
        if y in SEVERE:
            ax.axvspan(i - 0.4, i + 0.4, facecolor="#fff3cd", alpha=0.6, zorder=0)

    # Reference line at 1.0
    ax.axhline(y=1.0, color="gray", ls=":", lw=1, alpha=0.5, zorder=0)

    l1, = ax.plot(x, rB, "-o", ms=6, lw=2.2, color=C_BASE, label="Baseline", zorder=3)
    l2, = ax.plot(x, rW, "-s", ms=6, lw=2.2, color=C_WORST, label="No-adaptation", zorder=3)

    # Shaded gap
    ax.fill_between(x, rB, rW, alpha=0.10, color="#22c55e", zorder=1)

    # Endpoint annotations
    for val, color, nudge in [(rB[-1], C_BASE, 0.05), (rW[-1], C_WORST, -0.08)]:
        ax.annotate(f"{val:.2f}x", xy=(x[-1], val),
                    xytext=(x[-1] + 0.3, val + nudge),
                    fontsize=10, fontweight="bold", color=color,
                    ha="left", va="center", zorder=5,
                    bbox=dict(boxstyle="round,pad=0.1", facecolor="white",
                              edgecolor="none", alpha=0.7))

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_xticks(x[::2])
    ax.set_xticklabels([str(y) for y in years[::2]])
    ax.text(-0.08, 1.05, plbl, transform=ax.transAxes, fontsize=14,
            fontweight="bold", ha="left", va="bottom")

    if ax == ax1:
        ax.set_ylabel("Cumulative damage / actual loss ratio")
    ax.legend(loc="upper left", fontsize=10, frameon=True, facecolor="white",
              edgecolor="black", framealpha=1.0)

fig.tight_layout()

# Save 2-panel version
SM_DIR = Path(r"C:\Users\wenyu\OneDrive - Lehigh University\Desktop\Lehigh\NSF-project\ABM\paper\draft\v2_sections\SM")
SM_DIR.mkdir(parents=True, exist_ok=True)
for ext in ["png", "pdf"]:
    fig.savefig(VIS / f"fig4_damage_loss_ratio.{ext}", dpi=300, bbox_inches="tight")
    fig.savefig(SM_DIR / f"fig4_damage_loss_ratio_2panel.{ext}", dpi=300, bbox_inches="tight")
plt.close()
print(f"\nSaved 2-panel: {VIS / 'fig4_damage_loss_ratio.png'}")

# ── Single-panel version: Homeowner vs Renter (baseline only) ──
OWNER_COLOR = "#1f77b4"
RENTER_COLOR = "#d62828"

fig2, ax = plt.subplots(figsize=(8, 5))

# Severe year shading
for i, y in enumerate(years):
    if y in SEVERE:
        ax.axvspan(i - 0.4, i + 0.4, facecolor="#fff3cd", alpha=0.6, zorder=0)

# Reference line at 1.0
ax.axhline(y=1.0, color="gray", ls=":", lw=1, alpha=0.5, zorder=0)

ax.plot(x, rB_O, "-o", ms=6, lw=2.2, color=OWNER_COLOR, label="Homeowner", zorder=3)
ax.plot(x, rB_R, "-s", ms=6, lw=2.2, color=RENTER_COLOR, label="Renter", zorder=3)

# Shaded gap between groups
ax.fill_between(x, rB_O, rB_R, alpha=0.10, color="#22c55e", zorder=1)

# Endpoint annotations
for val, color, nudge in [(rB_O[-1], OWNER_COLOR, 0.04), (rB_R[-1], RENTER_COLOR, -0.06)]:
    ax.annotate(f"{val:.2f}x", xy=(x[-1], val),
                xytext=(x[-1] + 0.3, val + nudge),
                fontsize=10, fontweight="bold", color=color,
                ha="left", va="center", zorder=5,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor=color, linewidth=0.6, alpha=1.0))

# Start annotations
for val, color, nudge in [(rB_O[0], OWNER_COLOR, 0.04), (rB_R[0], RENTER_COLOR, -0.06)]:
    ax.annotate(f"{val:.2f}x", xy=(x[0], val),
                xytext=(x[0] - 0.3, val + nudge),
                fontsize=10, fontweight="bold", color=color,
                ha="right", va="center", zorder=5,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor=color, linewidth=0.6, alpha=1.0))

ax.set_ylabel("Cumulative damage / actual loss ratio", fontsize=12)
ax.set_xlabel("Year")
ax.set_xticks(x[::2])
ax.set_xticklabels([str(y) for y in years[::2]])
ax.legend(loc="upper left", fontsize=10, frameon=True, facecolor="white",
          edgecolor="black", framealpha=1.0)

fig2.tight_layout()
for ext in ["png", "pdf"]:
    fig2.savefig(VIS / f"fig4_damage_loss_ratio_single.{ext}", dpi=300, bbox_inches="tight")
    fig2.savefig(SM_DIR / f"fig4_damage_loss_ratio.{ext}", dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved single-panel: {VIS / 'fig4_damage_loss_ratio_single.png'}")
