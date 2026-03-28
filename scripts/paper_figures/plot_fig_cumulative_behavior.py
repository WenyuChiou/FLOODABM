# -*- coding: utf-8 -*-
"""
Cumulative Behavior Composition — Paper-format version
Layout: 2 rows × 1 column (Homeowner | Renter), 100% stacked bars
Matches Figure 8 style: Times New Roman, consistent panel labels, severe year shading.
"""
import sys, re, glob
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.patches import Patch
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
SEVERE = []  # No shading — cumulative composition shows long-term trends, not event response
SEV_COLOR = "#fff3cd"  # light yellow, distinct from bar colors

# ── Load decisions and compute cumulative states ──
print("Loading decision data...")
file_map = {}
for f in sorted(glob.glob(str(BASE / "decisions/decisions_mgmix_*.csv"))):
    m = re.search(r"mgmix_(\d{4})\.csv", f)
    if m:
        file_map[int(m.group(1))] = f

years = sorted(file_map.keys())

# Track cumulative EH (permanent)
set_EH = set()
set_BP = set()

owner_stats = []
renter_stats = []

for i, y in enumerate(years):
    df = pd.read_csv(file_map[y])

    # ── Owners ──
    owners = df[df["group"] == "owner"]
    if i == 0:
        BASELINE_OWNER_N = len(owners)

    # Track cumulative EH and BP
    current_ehs = set(owners[owners["action"] == "EH"]["i"])
    if "ELEV_FT" in owners.columns:
        current_ehs.update(set(owners[owners["ELEV_FT"] > 0]["i"]))
    set_EH.update(current_ehs)

    current_bps = set(owners[owners["action"] == "BP"]["i"])
    set_BP.update(current_bps)
    n_bp_current = len(current_bps)

    # Classify stayers (non-BP this year)
    stayers = owners[owners["action"] != "BP"]
    is_fi_mask = (owners["action"] == "FI")
    if "POLICY_NAME" in owners.columns:
        has_pol = owners["POLICY_NAME"].notna() & (owners["POLICY_NAME"] != "")
        is_fi_mask = is_fi_mask | has_pol

    is_eh_status = stayers["i"].isin(set_EH)
    is_fi_status = is_fi_mask[stayers.index]

    n_both = (is_eh_status & is_fi_status).sum()
    n_eh = (is_eh_status & ~is_fi_status).sum()
    n_fi = (~is_eh_status & is_fi_status).sum()
    n_dn = (~is_eh_status & ~is_fi_status).sum()

    owner_stats.append({
        "year": y, "Both (EH+FI)": n_both, "EH": n_eh,
        "FI": n_fi, "BP": n_bp_current, "DN": n_dn,
    })

    # ── Renters ──
    renters = df[df["group"] == "renter"]
    r_fi = (renters["action"] == "FI").sum()
    if "POLICY_NAME" in renters.columns:
        has_pol = renters["POLICY_NAME"].notna() & (renters["POLICY_NAME"] != "")
        r_fi = ((renters["action"] == "FI") | has_pol).sum()
    r_rl = (renters["action"] == "RL").sum()
    r_dn = max(0, len(renters) - r_fi - r_rl)

    renter_stats.append({"year": y, "FI": r_fi, "RL": r_rl, "DN": r_dn})

df_own = pd.DataFrame(owner_stats)
df_rent = pd.DataFrame(renter_stats)

# ── Colors (matching original) ──
C = {
    "Both (EH+FI)": "#4f46e5",  # indigo
    "EH": "#059669",             # green
    "FI": "#60a5fa",             # light blue
    "BP": "#f59e0b",             # orange
    "DN": "#9ca3af",             # gray
    "RL": "#dc2626",             # red
}

owner_actions = ["Both (EH+FI)", "EH", "FI", "BP", "DN"]
renter_actions = ["FI", "RL", "DN"]

# ── Plot ──
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
x = np.arange(len(years))

def shade_severe(ax):
    for i, y in enumerate(years):
        if y in SEVERE:
            ax.axvspan(i - 0.4, i + 0.4, facecolor=SEV_COLOR, alpha=0.8, zorder=0,
                       edgecolor="#e65100", linewidth=1.2, linestyle="--")

def draw_stacked(ax, stats_df, acts):
    totals = stats_df[list(acts)].sum(axis=1).values
    bottom = np.zeros(len(stats_df))
    for a in acts:
        if a not in stats_df.columns:
            continue
        vals = stats_df[a].values
        heights = np.divide(vals, totals, out=np.zeros_like(totals, dtype=float),
                            where=totals != 0)
        ax.bar(x, heights, bottom=bottom, width=0.75, color=C.get(a, "#9ca3af"),
               edgecolor="white", linewidth=0.5, alpha=0.9, label=a)
        bottom += heights
    ax.set_ylabel("Household adaptation composition (%)")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_ylim(0, 1.0)

# (a) Homeowner
shade_severe(ax1)
draw_stacked(ax1, df_own, owner_actions)
ax1.set_title("Homeowner", pad=8)
ax1.text(-0.08, 1.03, "(a)", transform=ax1.transAxes, fontsize=12,
         fontweight="bold", ha="left", va="bottom")

# (b) Renter
shade_severe(ax2)
draw_stacked(ax2, df_rent, renter_actions)
ax2.set_title("Renter", pad=8)
ax2.text(-0.08, 1.03, "(b)", transform=ax2.transAxes, fontsize=12,
         fontweight="bold", ha="left", va="bottom")
ax2.set_xlabel("Year")
ax2.set_xticks(x[::2])
ax2.set_xticklabels([str(y) for y in years[::2]])

# Combined legend (collect unique handles from both panels)
all_h, all_l = [], []
seen = set()
for ax in [ax1, ax2]:
    h, l = ax.get_legend_handles_labels()
    for hi, li in zip(h, l):
        if li not in seen:
            seen.add(li)
            all_h.append(hi)
            all_l.append(li)
ax2.legend(handles=all_h, labels=all_l, loc="upper center",
           frameon=True, facecolor="white", edgecolor="black",
           framealpha=1.0, ncol=3, fontsize=9)

# ── Add 2023 end-value labels INSIDE the last bar ──
def annotate_last_bar(ax, stats_df, acts):
    """Add percentage labels at the midpoint of each segment in the last bar."""
    last_idx = len(stats_df) - 1
    total = stats_df[list(acts)].iloc[last_idx].sum()
    if total == 0:
        return
    bottom = 0.0
    for a in acts:
        if a not in stats_df.columns:
            continue
        val = stats_df[a].iloc[last_idx]
        height = val / total if total > 0 else 0
        if height >= 0.005:  # label segments >= 0.5%
            mid = bottom + height / 2
            pct = f"{height*100:.1f}%" if height < 0.05 else f"{height*100:.0f}%"
            fs = 7 if height < 0.03 else 9
            ax.text(last_idx, mid, pct, ha="center", va="center",
                    fontsize=fs, fontweight="bold", color="#222222",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                              edgecolor="#888888", linewidth=0.6, alpha=1.0))
        bottom += height

annotate_last_bar(ax1, df_own, owner_actions)
annotate_last_bar(ax2, df_rent, renter_actions)

# ── Also annotate FIRST bar (2011) for reference ──
def annotate_first_bar(ax, stats_df, acts):
    """Add percentage labels at the midpoint of each segment in the first bar."""
    total = stats_df[list(acts)].iloc[0].sum()
    if total == 0:
        return
    bottom = 0.0
    for a in acts:
        if a not in stats_df.columns:
            continue
        val = stats_df[a].iloc[0]
        height = val / total if total > 0 else 0
        if height >= 0.005:  # label segments >= 0.5%
            mid = bottom + height / 2
            pct = f"{height*100:.1f}%" if height < 0.05 else f"{height*100:.0f}%"
            fs = 7 if height < 0.03 else 9
            ax.text(0, mid, pct, ha="center", va="center",
                    fontsize=fs, fontweight="bold", color="#222222",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                              edgecolor="#888888", linewidth=0.6, alpha=1.0))
        bottom += height

annotate_first_bar(ax1, df_own, owner_actions)
annotate_first_bar(ax2, df_rent, renter_actions)

# ── Add total adoption rate annotation above 2011 and 2023 bars ──
def annotate_total_adoption(ax, stats_df, acts):
    """Add 'X% adopted' above first and last bars."""
    for idx, bar_x in [(0, 0), (-1, len(stats_df) - 1)]:
        total = stats_df[list(acts)].iloc[idx].sum()
        dn = stats_df["DN"].iloc[idx] if "DN" in stats_df.columns else 0
        adopted_pct = (total - dn) / total * 100 if total > 0 else 0
        ax.text(bar_x, 1.02, f"{adopted_pct:.0f}% adopted",
                ha="center", va="bottom", fontsize=9, fontweight="bold",
                color="#333333")

annotate_total_adoption(ax1, df_own, owner_actions)
annotate_total_adoption(ax2, df_rent, renter_actions)

fig.tight_layout()
out_path = BASE / "visualization/decisions/fig_cumulative_behavior_composition.png"
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"\nSaved: {out_path}")
