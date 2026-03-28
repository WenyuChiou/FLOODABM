# -*- coding: utf-8 -*-
"""
Fig 4: Dual Y-axis — Cumulative flood damage (left) + Cumulative actual loss (right)
2-panel figure: (a) Homeowner, (b) Renter
All 4 lines clearly distinguishable with distinct styles + endpoint annotations.
"""
import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from pathlib import Path
import glob, re

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent.parent  # project root
sys.path.insert(0, str(ROOT))
from utils.plots_modular.style import set_paper_style, panel_label, COLORS

set_paper_style()

def usd_fmt(x, _):
    ax = abs(x)
    if ax >= 1e6: return f"${x/1e6:.1f}M"
    if ax >= 1e3: return f"${x/1e3:.0f}K"
    return f"${x:.0f}"

OUT_ROOT = ROOT / "outputs" / "baseline" / "baseline"
WORST_ROOT = ROOT / "outputs" / "baseline" / "worst"
VIS = OUT_ROOT / "visualization" / "compare"
VIS.mkdir(parents=True, exist_ok=True)

SEVERE = [2011, 2021]

# ── Load data ──
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

    dmg = pd.read_csv(root / "vulnerability/flood_damage/flood_damage_tract_ALL_years.csv",
                       dtype={"tract_geoid": str})
    dmg_yr = dmg.groupby("year").agg(
        owner_dmg=("owner_usd", "sum"),
        renter_dmg=("renter_usd", "sum"),
    ).reset_index()

    merged = fin_yr.merge(dmg_yr, on="year", how="outer").fillna(0).sort_values("year")

    total_gross = merged["owner_gross"] + merged["renter_gross"]
    merged["owner_payout_share"] = np.where(total_gross > 0,
                                             merged["owner_gross"] / total_gross, 0.5)
    merged["owner_payout"] = merged["payout_total"] * merged["owner_payout_share"] * 1000
    merged["renter_payout"] = merged["payout_total"] * (1 - merged["owner_payout_share"]) * 1000
    return merged


def cum_series(df, group):
    hh = np.maximum(df[f"{group}_hh"].values, 1)
    dmg = df[f"{group}_dmg"].values.astype(float)
    pay = df[f"{group}_payout"].values.astype(float)
    cum_dmg = np.cumsum(dmg / hh)
    cum_loss = np.cumsum((dmg - pay) / hh)
    return cum_dmg, cum_loss


base = load_scenario(OUT_ROOT)
worst = load_scenario(WORST_ROOT)
years = sorted(base["year"].unique())
x = np.arange(len(years))

cdB_O, clB_O = cum_series(base, "owner")
cdW_O, clW_O = cum_series(worst, "owner")
cdB_R, clB_R = cum_series(base, "renter")
cdW_R, clW_R = cum_series(worst, "renter")

# ── Plot ──
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

# Colors: blue for baseline, red for worst; solid=damage, dashed=actual loss
C_BASE = "#0077b6"
C_WORST = "#d62828"

for ax_main, group, cdB, cdW, clB, clW, plbl in [
    (ax1, "owner", cdB_O, cdW_O, clB_O, clW_O, "(a)"),
    (ax2, "renter", cdB_R, cdW_R, clB_R, clW_R, "(b)"),
]:
    title = "Homeowner" if group == "owner" else "Renter"

    # Severe flood shading
    for i, y in enumerate(years):
        if y in SEVERE:
            ax_main.axvspan(i - 0.4, i + 0.4,
                            color=COLORS["severe_shade"], alpha=0.25, zorder=0)

    # Single Y-axis: all 4 lines on same scale
    l1, = ax_main.plot(x, cdB, marker="o", ms=5, lw=2.2, color=C_BASE,
                       label="Baseline \u2014 Damage", zorder=3)
    l2, = ax_main.plot(x, cdW, marker="s", ms=5, lw=2.2, color=C_WORST,
                       label="No-adaptation \u2014 Damage", zorder=3)
    l3, = ax_main.plot(x, clB, marker="^", ms=5, lw=2.0, ls="--", color=C_BASE,
                       alpha=0.7, label="Baseline \u2014 Actual loss", zorder=2)
    l4, = ax_main.plot(x, clW, marker="D", ms=5, lw=2.0, ls="--", color=C_WORST,
                       alpha=0.7, label="No-adaptation \u2014 Actual loss", zorder=2)

    # Shaded areas: damage gap and actual-loss gap
    ax_main.fill_between(x, cdB, cdW, alpha=0.08, color=C_WORST)
    ax_main.fill_between(x, clB, clW, alpha=0.06, color="#22c55e")

    ax_main.set_ylabel("Cumulative per HH ($)")
    ax_main.yaxis.set_major_formatter(FuncFormatter(usd_fmt))
    ax_main.set_xlabel("Year")
    ax_main.set_xticks(x[::2])
    ax_main.set_xticklabels([str(y) for y in years[::2]])
    ax_main.set_title(title, fontsize=14)
    panel_label(ax_main, plbl)

    # Endpoint value annotations — sort by value descending, assign staggered offsets
    annotations = [
        (cdW[-1], C_WORST, "damage_worst"),
        (clW[-1], C_WORST, "loss_worst"),
        (cdB[-1], C_BASE,  "damage_base"),
        (clB[-1], C_BASE,  "loss_base"),
    ]
    annotations.sort(key=lambda t: -t[0])  # highest value first
    # Stagger: top gets +18, next +6, next -6, bottom -18
    y_offsets = [18, 6, -6, -18]
    x_offsets = [14, 14, 14, 14]
    for (val, color, _), yo, xo in zip(annotations, y_offsets, x_offsets):
        ax_main.annotate(
            f"{usd_fmt(val, None)}",
            xy=(x[-1], val), xytext=(xo, yo),
            textcoords="offset points",
            fontsize=10, color=color, fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=color, lw=0.5),
        )

    # Legend
    lines = [l1, l2, l3, l4]
    ax_main.legend(lines, [l.get_label() for l in lines],
                   loc="upper left", fontsize=10, framealpha=0.9,
                   handlelength=2.5)

fig.tight_layout()
out_path = VIS / "fig4_cum_damage_loss_dual_yaxis.png"
fig.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {out_path}")
