# -*- coding: utf-8 -*-
"""
Fig RQ1 Combined — 4-row layout for merged RQ1 (damage + finance)
  Row 1 (a,b): Cumulative damage + actual loss per HH — Owner | Renter  [old Fig4]
  Row 2 (c):   Damage/Loss ratio Owner vs Renter (single, full width)
  Row 3 (d,e): Premium + OOP stacked bar + OOP rate — Owner | Renter    [old Fig8]
  Row 4 (f,g): Payout bar + Payout rate — Owner | Renter               [old Fig8]
"""
import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
from pathlib import Path
import glob, re

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from utils.plots_modular.style import set_paper_style, COLORS
set_paper_style()

# ── Paths ──
BASE = ROOT / "outputs" / "baseline" / "baseline"
WORST = ROOT / "outputs" / "baseline" / "worst"
VIS = BASE / "visualization" / "compare"
VIS.mkdir(parents=True, exist_ok=True)
SM_DIR = Path(r"C:\Users\wenyu\OneDrive - Lehigh University\Desktop\Lehigh\NSF-project\ABM\paper\draft\v2_sections\SM")
SM_DIR.mkdir(parents=True, exist_ok=True)

SEVERE = [2011, 2021]

# ── Style ──
plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300,
    "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 10, "axes.linewidth": 0.8,
    "axes.grid": True, "grid.alpha": 0.12, "grid.linestyle": "--",
    "axes.spines.top": False,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
})

# ── Color palette (consistent semantic meaning throughout all panels) ──
# Owner = blue family, Renter = green family
# Within Row 1: solid = Baseline (adaptation), dashed-lighter = No-adaptation (worst)
C_OWN_BASE  = "#1a6faf"   # Owner  — Baseline     (solid blue)
C_OWN_WORST = "#6baed6"   # Owner  — No-adapt     (lighter blue)
C_REN_BASE  = "#1f7a3c"   # Renter — Baseline     (solid green)
C_REN_WORST = "#74c476"   # Renter — No-adapt     (lighter green)

# Finance panels: stacked bars use two tones within each group's hue
C_PREM_O  = "#1a6faf"     # Owner  premium  (blue)
C_OOP_O   = "#9ecae1"     # Owner  OOP      (pale blue)
C_PAY_O   = "#08519c"     # Owner  payout   (dark blue)
C_PREM_R  = "#1f7a3c"     # Renter premium  (green)
C_OOP_R   = "#a1d99b"     # Renter OOP      (pale green)
C_PAY_R   = "#006d2c"     # Renter payout   (dark green)

# Ratio panel uses owner/renter colors
C_OWNER  = C_OWN_BASE
C_RENTER = C_REN_BASE

# Severe year shading
C_SEVERE = "#cfe2f3"

def usd_fmt(x, _):
    ax = abs(x)
    if ax >= 1e6: return f"${x/1e6:.1f}M"
    if ax >= 1e3: return f"${x/1e3:.0f}K"
    return f"${x:.0f}"

def shade_severe(ax, years_list):
    for i, y in enumerate(years_list):
        if y in SEVERE:
            ax.axvspan(i - 0.45, i + 0.45, color=C_SEVERE, alpha=0.55, zorder=0,
                       label="_nolegend_")

def plabel(ax, text):
    """Bold panel label in upper-left, outside axes."""
    ax.text(-0.10, 1.04, text, transform=ax.transAxes, fontsize=11,
            fontweight="bold", ha="left", va="bottom")

def set_xticks(ax, x, years, step=2):
    ax.set_xticks(x[::step])
    ax.set_xticklabels([str(y) for y in years[::step]], rotation=0)
    ax.set_xlim(x[0] - 0.7, x[-1] + 1.2)   # right margin for annotations

def twin_ax_style(ax2, ylabel, ylim=None, n_ticks=5):
    """Consistent secondary y-axis formatting."""
    ax2.set_ylabel(ylabel, fontsize=11, color="#444444")
    ax2.tick_params(axis="y", labelsize=10, colors="#444444")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_color("#bbbbbb")
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=n_ticks, prune="both"))
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    if ylim is not None:
        ax2.set_ylim(0, ylim)

# ══════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════
def load_scenario_tract(root):
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

def cum_series(df, group):
    hh = np.maximum(df[f"{group}_hh"].values, 1)
    dmg = df[f"{group}_dmg"].values.astype(float)
    pay = df[f"{group}_payout"].values.astype(float)
    return np.cumsum(dmg / hh), np.cumsum((dmg - pay) / hh)

def cum_ratio_across_scenario(base_df, worst_df, group):
    """Ratio = Δ cumulative damage / Δ cumulative actual loss (across scenarios).
    Δ = no-adaptation minus baseline.  Shows: per $1 actual-loss reduction,
    how much total damage was avoided by adaptation."""
    cdB, clB = cum_series(base_df, group)
    cdW, clW = cum_series(worst_df, group)
    delta_dmg  = cdW - cdB   # damage avoided by adaptation
    delta_loss = clW - clB   # actual-loss avoided by adaptation
    return np.where(delta_loss > 0, delta_dmg / delta_loss, 1.0)

# Tract-level data for Row 1 + Row 2
base_t = load_scenario_tract(BASE)
worst_t = load_scenario_tract(WORST)
years = sorted(base_t["year"].unique())
x = np.arange(len(years))

cdB_O, clB_O = cum_series(base_t, "owner")
cdW_O, clW_O = cum_series(worst_t, "owner")
cdB_R, clB_R = cum_series(base_t, "renter")
cdW_R, clW_R = cum_series(worst_t, "renter")
ratio_O = cum_ratio_across_scenario(base_t, worst_t, "owner")
ratio_R = cum_ratio_across_scenario(base_t, worst_t, "renter")

print("=== Cumulative per HH (2023) ===")
print(f"  Owner  damage: ${cdB_O[-1]:,.0f} (base) / ${cdW_O[-1]:,.0f} (no-adapt)")
print(f"  Owner  loss:   ${clB_O[-1]:,.0f} (base) / ${clW_O[-1]:,.0f} (no-adapt)")
print(f"  Renter damage: ${cdB_R[-1]:,.0f} (base) / ${cdW_R[-1]:,.0f} (no-adapt)")
print(f"  Renter loss:   ${clB_R[-1]:,.0f} (base) / ${clW_R[-1]:,.0f} (no-adapt)")
print(f"\n  Owner  Δdmg={cdW_O[-1]-cdB_O[-1]:,.0f}  Δloss={clW_O[-1]-clB_O[-1]:,.0f}  ratio={ratio_O[-1]:.3f}")
print(f"  Renter Δdmg={cdW_R[-1]-cdB_R[-1]:,.0f}  Δloss={clW_R[-1]-clB_R[-1]:,.0f}  ratio={ratio_R[-1]:.3f}")

# ══════════════════════════════════════════════════════════
# FIGURE 4 — Cumulative damage/loss (2 panels) + Ratio (1 panel)
#             Layout: 1×3 (owner | renter | ratio)
# ══════════════════════════════════════════════════════════
import matplotlib.gridspec as gridspec

# --- Colors for cross-scenario distinction (CLEAR contrast) ---
C_DMG  = "#d62828"   # Damage = red (both scenarios)
C_LOSS = "#0077b6"   # Actual loss = blue (both scenarios)
# Scenario distinction: solid+filled = Baseline, dashed+open = No-adaptation

fig4 = plt.figure(figsize=(15, 4.8))
gs4 = fig4.add_gridspec(1, 3, wspace=0.32, width_ratios=[1, 1, 1.1])

for col, (cdB, cdW, clB, clW, title, plbl) in enumerate([
    (cdB_O, cdW_O, clB_O, clW_O, "Homeowner", "(a)"),
    (cdB_R, cdW_R, clB_R, clW_R, "Renter", "(b)"),
]):
    ax = fig4.add_subplot(gs4[0, col])
    shade_severe(ax, years)

    # Baseline (solid, filled markers)
    l1, = ax.plot(x, cdB, "-o",  ms=5, lw=2.2, color=C_DMG,
                  label="Baseline — Damage", zorder=4)
    l2, = ax.plot(x, clB, "-s",  ms=5, lw=2.2, color=C_LOSS,
                  label="Baseline — Actual loss", zorder=4)
    # No-adaptation (dashed, open markers)
    l3, = ax.plot(x, cdW, "o",  ms=5, lw=1.8, color=C_DMG,
                  linestyle="--", markerfacecolor="white", markeredgewidth=1.5,
                  label="No-adapt — Damage", zorder=3)
    l4, = ax.plot(x, clW, "s",  ms=5, lw=1.8, color=C_LOSS,
                  linestyle="--", markerfacecolor="white", markeredgewidth=1.5,
                  label="No-adapt — Actual loss", zorder=3)

    # Shaded gaps between scenarios
    ax.fill_between(x, cdB, cdW, alpha=0.08, color=C_DMG, zorder=1)
    ax.fill_between(x, clB, clW, alpha=0.08, color=C_LOSS, zorder=1)

    ax.set_ylabel("Cumulative per HH ($)", fontsize=11)
    ax.yaxis.set_major_formatter(FuncFormatter(usd_fmt))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune="both"))
    ax.set_title(title, fontsize=14, fontweight="bold", pad=8)
    set_xticks(ax, x, years)
    ax.set_xlabel("Year", fontsize=11)
    ax.spines["right"].set_visible(False)
    plabel(ax, plbl)
    ax.legend(handles=[l1, l3, l2, l4], loc="upper left", fontsize=9,
              framealpha=0.95, edgecolor="#cccccc", handlelength=2.5,
              labelspacing=0.3, borderpad=0.4, handletextpad=0.4)

# ── Panel (c): Insurance Leverage Ratio ──
ax_r = fig4.add_subplot(gs4[0, 2])
shade_severe(ax_r, years)

# Skip first year (no cross-scenario difference yet)
x_r = x[1:]
rO = ratio_O[1:]
rR = ratio_R[1:]
years_r = years[1:]

ax_r.plot(x_r, rO, "-o", ms=6, lw=2.4, color=C_OWN_BASE,  label="Homeowner", zorder=4)
ax_r.plot(x_r, rR, "-s", ms=6, lw=2.4, color=C_REN_BASE, label="Renter",    zorder=4)
ax_r.fill_between(x_r, rO, rR, alpha=0.10, color="#888888", zorder=1)

# Endpoint labels — directly next to the last data point
for val, color in [(rO[-1], C_OWN_BASE), (rR[-1], C_REN_BASE)]:
    ax_r.text(x_r[-1] + 0.3, val, f"{val:.2f}\u00d7",
              fontsize=11, fontweight="bold", color=color,
              ha="left", va="center", clip_on=False,
              bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                        edgecolor=color, linewidth=0.6, alpha=1.0))

ax_r.set_ylabel("\u0394 Cumul. damage / \u0394 Cumul. actual loss", fontsize=11)
ax_r.set_title("Insurance Leverage Ratio", fontsize=13, fontweight="bold", pad=8)
ax_r.set_xlabel("Year", fontsize=11)
ax_r.yaxis.set_major_locator(MaxNLocator(nbins=6, prune="both"))
set_xticks(ax_r, x_r, years_r)
r_all = np.concatenate([rO, rR])
r_pad = (r_all.max() - r_all.min()) * 0.25
ax_r.set_ylim(max(0, r_all.min() - r_pad), r_all.max() + r_pad)
ax_r.legend(loc="upper left", fontsize=10, frameon=True, facecolor="white",
            edgecolor="#cccccc", framealpha=1.0, handlelength=2.0)
ax_r.spines["right"].set_visible(False)
plabel(ax_r, "(c)")

# ── Endpoint annotations: dollar values at each line + percentage gap ──
for col, (cdB, cdW, clB, clW) in enumerate([
    (cdB_O, cdW_O, clB_O, clW_O),
    (cdB_R, cdW_R, clB_R, clW_R),
]):
    ax = fig4.axes[col]
    x_end = x[-1]

    # Dollar labels at 2023 endpoints only (no percentage)
    labels = [
        (cdW[-1], f"${cdW[-1]/1e3:.0f}K", C_DMG),
        (cdB[-1], f"${cdB[-1]/1e3:.0f}K", C_DMG),
        (clW[-1], f"${clW[-1]/1e3:.0f}K", C_LOSS),
        (clB[-1], f"${clB[-1]/1e3:.0f}K", C_LOSS),
    ]
    labels.sort(key=lambda t: t[0])
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    min_gap = y_range * 0.05
    ys = [l[0] for l in labels]
    for i in range(1, len(ys)):
        if ys[i] - ys[i-1] < min_gap:
            ys[i] = ys[i-1] + min_gap
    for i in range(len(ys)-2, -1, -1):
        if ys[i+1] - ys[i] < min_gap:
            ys[i] = ys[i+1] - min_gap

    for (_, txt, color), y_pos in zip(labels, ys):
        ax.text(x_end + 0.3, y_pos, txt,
                fontsize=10, fontweight="bold", color=color,
                ha="left", va="center", clip_on=False,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          edgecolor="none", alpha=0.85))

fig4.tight_layout()
for path in [VIS / "fig4_damage_ratio.png", VIS / "fig4_damage_ratio.pdf",
             SM_DIR / "fig4_damage_ratio.png", SM_DIR / "fig4_damage_ratio.pdf"]:
    fig4.savefig(path, dpi=300, bbox_inches="tight")
plt.close(fig4)
print(f"\nSaved Fig 4: {VIS / 'fig4_damage_ratio.png'}")

# Household-level finance data for Row 3 + Row 4
print("\nLoading household-level finance data...")
hh_frames = []
for f in sorted(glob.glob(str(BASE / "finance/finance_households_*.csv"))):
    m = re.search(r"households_(\d{4})\.csv", f)
    if m:
        hh_frames.append(pd.read_csv(f, usecols=[
            "identity", "year",
            "payout_total_usd", "oop_total_usd",
            "premium_total_usd", "gross_total_usd",
        ]))
hh = pd.concat(hh_frames, ignore_index=True)
agg = hh.groupby(["year", "identity"]).agg(
    n_hh=("identity", "count"),
    premium_total=("premium_total_usd", "sum"),
    payout_total=("payout_total_usd", "sum"),
    oop_total=("oop_total_usd", "sum"),
    gross_total=("gross_total_usd", "sum"),
).reset_index()
ins_agg = hh[hh["premium_total_usd"] > 0].groupby(["year", "identity"]).agg(
    insured_dmg=("gross_total_usd", "sum"),
).reset_index()
agg = agg.merge(ins_agg, on=["year", "identity"], how="left")
agg["insured_dmg"] = agg["insured_dmg"].fillna(0)

owner = agg[agg["identity"] == "owner"].set_index("year").sort_index()
renter = agg[agg["identity"] == "renter"].set_index("year").sort_index()
o_hh = np.maximum(owner["n_hh"].values, 1)
r_hh = np.maximum(renter["n_hh"].values, 1)
o_premium, o_oop = owner["premium_total"].values / o_hh, owner["oop_total"].values / o_hh
o_payout = owner["payout_total"].values / o_hh
o_ins_dmg, o_dmg = owner["insured_dmg"].values, owner["gross_total"].values
o_oop_rate = np.where(o_dmg > 0, owner["oop_total"].values / o_dmg * 100, 0)
o_payout_rate = np.where(o_dmg > 0, owner["payout_total"].values / o_dmg * 100, 0)

r_premium, r_oop = renter["premium_total"].values / r_hh, renter["oop_total"].values / r_hh
r_payout = renter["payout_total"].values / r_hh
r_ins_dmg, r_dmg = renter["insured_dmg"].values, renter["gross_total"].values
r_oop_rate = np.where(r_dmg > 0, renter["oop_total"].values / r_dmg * 100, 0)
r_payout_rate = np.where(r_dmg > 0, renter["payout_total"].values / r_dmg * 100, 0)

# ══════════════════════════════════════════════════════════
# FIGURE 8 — Finance: 2×2 (premium+OOP | payout)
# ══════════════════════════════════════════════════════════
fig8, axes8 = plt.subplots(2, 2, figsize=(12, 8.5))

# ── Row 1: Premium + OOP ──
oop_ylim = max(o_oop_rate.max(), r_oop_rate.max()) * 1.35
oop_ylim = max(oop_ylim, 20)

for col, (prem, oop, oop_r, c_prem, c_oop, title, plbl) in enumerate([
    (o_premium, o_oop, o_oop_rate, C_PREM_O, C_OOP_O, "Homeowner", "(a)"),
    (r_premium, r_oop, r_oop_rate, C_PREM_R, C_OOP_R, "Renter", "(b)"),
]):
    ax = axes8[0, col]
    shade_severe(ax, years)
    ax.bar(x, prem, 0.65, label="Premium",  color=c_prem, alpha=0.85, zorder=3)
    ax.bar(x, oop,  0.65, bottom=prem, label="OOP cost", color=c_oop,  alpha=0.85, zorder=3)
    ax.set_ylabel("Avg per HH ($)", fontsize=12)
    ax.yaxis.set_major_formatter(FuncFormatter(usd_fmt))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune="both"))
    ax.set_title(title, fontsize=14, fontweight="bold", pad=8)
    set_xticks(ax, x, years)
    ax.set_xlim(x[0] - 0.6, x[-1] + 0.6)
    plabel(ax, plbl)
    ax2 = ax.twinx()
    ax2.plot(x, oop_r, color="#444444", marker="o", ms=4, lw=1.6,
             linestyle="--", label="OOP rate", zorder=5)
    twin_ax_style(ax2, "Share of flood damage (%)", ylim=oop_ylim, n_ticks=4)
    h1, l1_ = ax.get_legend_handles_labels()
    h2, l2_ = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1_ + l2_, loc="upper left", fontsize=10,
              framealpha=0.92, edgecolor="#cccccc",
              handlelength=1.8, labelspacing=0.3, borderpad=0.4)

# ── Row 2: Payout ──
for col, (pay, pay_r, c_pay, title, plbl) in enumerate([
    (o_payout, o_payout_rate, C_PAY_O, "Homeowner", "(c)"),
    (r_payout, r_payout_rate, C_PAY_R, "Renter", "(d)"),
]):
    ax = axes8[1, col]
    shade_severe(ax, years)
    ax.bar(x, pay, 0.65, label="Payout", color=c_pay, alpha=0.85, zorder=3)
    ax.set_ylabel("Avg payout per HH ($)", fontsize=12)
    ax.yaxis.set_major_formatter(FuncFormatter(usd_fmt))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune="both"))
    ax.set_xlabel("Year", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=8)
    set_xticks(ax, x, years)
    ax.set_xlim(x[0] - 0.6, x[-1] + 0.6)
    plabel(ax, plbl)
    ax2 = ax.twinx()
    ax2.plot(x, pay_r, color="#444444", marker="s", ms=4, lw=1.6,
             linestyle="--", label="Payout rate", zorder=5)
    twin_ax_style(ax2, "Share of flood damage (%)", ylim=102, n_ticks=5)
    h1, l1_ = ax.get_legend_handles_labels()
    h2, l2_ = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1_ + l2_, loc="upper left", fontsize=10,
              framealpha=0.92, edgecolor="#cccccc",
              handlelength=1.8, labelspacing=0.3, borderpad=0.4)

# Removed: fig8.text (light blue shading note)
fig8.tight_layout()
for path in [VIS / "fig5_finance.png", VIS / "fig5_finance.pdf",
             SM_DIR / "fig5_finance.png", SM_DIR / "fig5_finance.pdf"]:
    fig8.savefig(path, dpi=300, bbox_inches="tight")
plt.close(fig8)
print(f"Saved Fig 5: {VIS / 'fig5_finance.png'}")
print("Done!")
