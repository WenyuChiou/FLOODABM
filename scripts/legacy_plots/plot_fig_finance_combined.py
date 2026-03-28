# -*- coding: utf-8 -*-
"""
Combined Finance Figure — Fixed tenure-specific OOP/payout calculation
Layout: 3 rows × 2 columns (Homeowner | Renter)
  Row 1: Avg flood damage per HH (bar)
  Row 2: Premiums + OOP (stacked bar) with OOP rate line
  Row 3: Payout (bar) with payout rate line

BUG FIX: Previous version used tract-level payout/OOP columns that are NOT
split by tenure, causing inflated renter OOP rates. Now aggregates from
household-level data where `identity` correctly separates owner/renter.
"""
import sys, re, glob
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300,
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 9, "axes.linewidth": 0.8,
    "axes.grid": True, "grid.alpha": 0.15, "grid.linestyle": "--",
    "axes.spines.top": False,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
})

BASE = Path("outputs/baseline/baseline")
SEVERE = [2011, 2021]
SEV_COLOR = "#e3f2fd"

def usd_fmt(x, _):
    ax = abs(x)
    if ax >= 1e6: return f"${x/1e6:.1f}M"
    if ax >= 1e3: return f"${x/1e3:.0f}K"
    return f"${x:.0f}"

# ── Load household-level data and aggregate by tenure ──
print("Loading household-level finance data...")
hh_frames = []
for f in sorted(glob.glob(str(BASE / "finance/finance_households_*.csv"))):
    m = re.search(r"households_(\d{4})\.csv", f)
    if m:
        df = pd.read_csv(f, usecols=[
            "identity", "year",
            "payout_structure_usd", "payout_contents_usd", "payout_total_usd",
            "oop_structure_usd", "oop_contents_usd", "oop_total_usd",
            "premium_structure_usd", "premium_contents_usd", "premium_total_usd",
            "gross_structure_loss_usd", "gross_contents_loss_usd", "gross_total_usd",
        ])
        hh_frames.append(df)
        print(f"  Loaded {m.group(1)}: {len(df)} households")

hh = pd.concat(hh_frames, ignore_index=True)

# Aggregate by year and tenure (all HH)
agg = hh.groupby(["year", "identity"]).agg(
    n_hh=("identity", "count"),
    premium_total=("premium_total_usd", "sum"),
    payout_total=("payout_total_usd", "sum"),
    oop_total=("oop_total_usd", "sum"),
    gross_total=("gross_total_usd", "sum"),
).reset_index()

# Aggregate insured-HH damage (for correct OOP rate denominator)
# OOP only counts insured HH gaps (deductible + excess), so denominator
# must also be insured-HH damage, not all-HH damage
insured = hh[hh["premium_total_usd"] > 0]
ins_agg = insured.groupby(["year", "identity"]).agg(
    insured_dmg=("gross_total_usd", "sum"),
).reset_index()
agg = agg.merge(ins_agg, on=["year", "identity"], how="left")
agg["insured_dmg"] = agg["insured_dmg"].fillna(0)

owner = agg[agg["identity"] == "owner"].set_index("year").sort_index()
renter = agg[agg["identity"] == "renter"].set_index("year").sort_index()

years = owner.index.values
x = np.arange(len(years))

# Per-HH calculations
o_hh = np.maximum(owner["n_hh"].values, 1)
r_hh = np.maximum(renter["n_hh"].values, 1)

o_premium = owner["premium_total"].values / o_hh
o_oop = owner["oop_total"].values / o_hh
o_payout = owner["payout_total"].values / o_hh
o_dmg = owner["gross_total"].values
o_dmg_per_hh = o_dmg / o_hh
o_ins_dmg = owner["insured_dmg"].values
o_oop_rate = np.where(o_ins_dmg > 0, owner["oop_total"].values / o_ins_dmg * 100, 0)
o_payout_rate = np.where(o_dmg > 0, owner["payout_total"].values / o_dmg * 100, 0)

r_premium = renter["premium_total"].values / r_hh
r_oop = renter["oop_total"].values / r_hh
r_payout = renter["payout_total"].values / r_hh
r_dmg = renter["gross_total"].values
r_dmg_per_hh = r_dmg / r_hh
r_ins_dmg = renter["insured_dmg"].values
r_oop_rate = np.where(r_ins_dmg > 0, renter["oop_total"].values / r_ins_dmg * 100, 0)
r_payout_rate = np.where(r_dmg > 0, renter["payout_total"].values / r_dmg * 100, 0)

# Print verification table
print("\n=== OOP Rate Verification ===")
print(f"{'Year':>6} {'Owner OOP%':>10} {'Renter OOP%':>12} {'Owner Payout%':>14} {'Renter Payout%':>15}")
for i, y in enumerate(years):
    print(f"{int(y):>6} {o_oop_rate[i]:>10.2f} {r_oop_rate[i]:>12.2f} {o_payout_rate[i]:>14.2f} {r_payout_rate[i]:>15.2f}")

# ── Plot 3×2 ──
fig, axs = plt.subplots(3, 2, figsize=(11, 10), sharex=True)

def shade_severe(ax):
    for i, y in enumerate(years):
        if y in SEVERE:
            ax.axvspan(i - 0.4, i + 0.4, color=SEV_COLOR, alpha=0.5, zorder=0)

# Colors
C_DMG = "#d62728"
C_PREM = "#1f77b4"
C_OOP = "#2ca02c"
C_PAY = "#ff7f0e"

# ── Row 1: Avg flood damage per HH ──
for col, (dmg, label, panel) in enumerate([
    (o_dmg_per_hh, "Homeowner", "(a)"),
    (r_dmg_per_hh, "Renter", "(b)"),
]):
    ax = axs[0, col]; shade_severe(ax)
    ax.bar(x, dmg, 0.7, label="Flood damage", color=C_DMG, alpha=0.8)
    ax.set_ylabel("Avg flood damage per HH ($)")
    ax.yaxis.set_major_formatter(FuncFormatter(usd_fmt))
    ax.set_title(label, pad=8)
    ax.text(-0.08, 1.03, panel, transform=ax.transAxes, fontsize=12, fontweight="bold",
            ha="left", va="bottom")
    # Clip y-axis to show detail in non-2011 years; annotate 2011 bar
    non_first = dmg[1:]  # exclude 2011
    ylim_top = max(non_first) * 1.5 if len(non_first) > 0 else dmg.max() * 1.1
    ax.set_ylim(0, ylim_top)
    # No annotation for clipped 2011 bar
    ax.legend(loc="upper right", fontsize=8)

# ── Row 2: Premium + OOP with OOP rate (insured-HH denominator) ──
# Shared OOP rate y-axis limit for (c) and (d)
oop_rate_ylim = max(o_oop_rate.max(), r_oop_rate.max()) * 1.3
oop_rate_ylim = max(oop_rate_ylim, 20)

for col, (prem, oop, oop_r, label, panel) in enumerate([
    (o_premium, o_oop, o_oop_rate, "Homeowner", "(c)"),
    (r_premium, r_oop, r_oop_rate, "Renter", "(d)"),
]):
    ax = axs[1, col]; shade_severe(ax)
    ax.bar(x, prem, 0.7, label="Premium", color=C_PREM, alpha=0.8)
    ax.bar(x, oop, 0.7, bottom=prem, label="OOP cost", color=C_OOP, alpha=0.8)
    ax.set_ylabel("Avg per HH ($)")
    ax.yaxis.set_major_formatter(FuncFormatter(usd_fmt))
    ax.text(-0.08, 1.03, panel, transform=ax.transAxes, fontsize=12, fontweight="bold",
            ha="left", va="bottom")

    ax2 = ax.twinx()
    ax2.plot(x, oop_r, "k--o", ms=4, lw=1.8, label="OOP rate")
    ax2.set_ylabel("OOP rate (%)")
    ax2.set_ylim(0, oop_rate_ylim)
    ax2.spines["top"].set_visible(False)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)

# ── Row 3: Payout ──
for col, (pay, pay_r, label, panel) in enumerate([
    (o_payout, o_payout_rate, "Homeowner", "(e)"),
    (r_payout, r_payout_rate, "Renter", "(f)"),
]):
    ax = axs[2, col]; shade_severe(ax)
    ax.bar(x, pay, 0.7, label="Payout", color=C_PAY, alpha=0.8)
    ax.set_ylabel("Avg payout per HH ($)")
    ax.yaxis.set_major_formatter(FuncFormatter(usd_fmt))
    ax.set_xlabel("Year")
    ax.set_xticks(x[::2])
    ax.set_xticklabels([str(int(y)) for y in years[::2]])
    ax.text(-0.08, 1.03, panel, transform=ax.transAxes, fontsize=12, fontweight="bold",
            ha="left", va="bottom")

    ax2 = ax.twinx()
    ax2.plot(x, pay_r, "k--s", ms=4, lw=1.8, label="Payout rate")
    ax2.set_ylabel("Payout rate (%)")
    ax2.set_ylim(0, 100)
    ax2.spines["top"].set_visible(False)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)

fig.tight_layout()

out_path = BASE / "visualization/finance/timeseries/fig_finance_combined_3x2.png"
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"\nSaved: {out_path}")
