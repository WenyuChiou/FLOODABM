# -*- coding: utf-8 -*-
"""
Fig 6 — Tract-Level Action Share Trajectories (2011–2023)
5 panels: (a) Owner FI, (b) Owner EH, (c) Owner BP, (d) Renter FI, (e) Renter RL
Each panel shows individual tract lines (thin) + median (bold) + 2021 event line.
"""
import sys, re, glob, json
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
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
FLOOD_CFG = Path("config/overall_md_mean_by_tract_2011_2023.json")
SEVERE = [2011, 2021]

# ── Load flood-prone tract classification ──
with open(FLOOD_CFG, "r") as f:
    depth_data = json.load(f)

tract_mean = {}
for row in depth_data:
    tract = str(row["CensusTract"])
    vals = [v for k, v in row.items() if k.endswith("_mean") and v is not None]
    tract_mean[tract] = np.mean(vals) if vals else 0

fp_tracts = {t for t, m in tract_mean.items() if m > 0.05}
nfp_tracts = {t for t, m in tract_mean.items() if m <= 0.05}
print(f"Flood-prone: {len(fp_tracts)}, Non-flood-prone: {len(nfp_tracts)}")

# ── Load all decision files ──
file_map = {}
for f in sorted(glob.glob(str(BASE / "decisions/decisions_mgmix_*.csv"))):
    m = re.search(r"mgmix_(\d{4})\.csv", f)
    if m:
        file_map[int(m.group(1))] = f

YEARS = sorted(file_map.keys())
print(f"Years: {YEARS[0]}–{YEARS[-1]}")

# ── Compute per-tract action shares for all years ──
def compute_tract_shares(years, group, action):
    """Returns DataFrame: rows=tracts, cols=years, values=action share."""
    all_tracts = set()
    yearly = {}

    for y in years:
        df = pd.read_csv(file_map[y])
        grp = df[df["group"] == group].copy()
        grp["tract_geoid"] = grp["tract_geoid"].astype(str)
        tracts = sorted(grp["tract_geoid"].unique())
        all_tracts.update(tracts)

        shares = {}
        for t in tracts:
            t_df = grp[grp["tract_geoid"] == t]
            n = len(t_df)
            if n == 0:
                shares[t] = 0.0
                continue
            if action == "FI":
                is_act = (t_df["action"] == "FI")
                if "POLICY_NAME" in t_df.columns:
                    has_pol = t_df["POLICY_NAME"].notna() & (t_df["POLICY_NAME"] != "")
                    is_act = is_act | has_pol
                shares[t] = is_act.sum() / n
            elif action == "EH":
                is_act = (t_df["action"] == "EH")
                if "ELEV_FT" in t_df.columns:
                    is_act = is_act | (t_df["ELEV_FT"] > 0)
                shares[t] = is_act.sum() / n
            else:
                shares[t] = (t_df["action"] == action).sum() / n
        yearly[y] = shares

    # Build DataFrame
    all_tracts = sorted(all_tracts)
    data = {y: [yearly[y].get(t, 0.0) for t in all_tracts] for y in years}
    return pd.DataFrame(data, index=all_tracts)

# Compute all action DataFrames
panels = [
    ("owner", "FI", "Owner — Flood Insurance (FI)", "(a)"),
    ("owner", "EH", "Owner — Elevation (EH)", "(b)"),
    ("owner", "BP", "Owner — Buyout (BP)", "(c)"),
    ("renter", "FI", "Renter — Flood Insurance (FI)", "(d)"),
    ("renter", "RL", "Renter — Relocation (RL)", "(e)"),
    ("owner", "DN", "Owner — Do Nothing (DN)", ""),
    ("renter", "DN", "Renter — Do Nothing (DN)", ""),
]

share_dfs = {}
for group, action, _, _ in panels:
    key = f"{group}_{action}"
    if key not in share_dfs:
        print(f"  Computing {key}...")
        share_dfs[key] = compute_tract_shares(YEARS, group, action)

# ── Colors ──
OWNER_COLOR = "#1f77b4"   # blue
RENTER_COLOR = "#d62828"  # red
SEV_COLOR = "#fff3cd"

# ══════════════════════════════════════════════════════════════════
# Fig 6: Owner vs Renter Adoption Divergence (RQ2)
# 3 panels:
#   (a) FI — shared action, Owner vs Renter
#   (b) DN — shared inaction, Owner vs Renter
#   (c) Structural adaptation (EH, BP) vs Relocation (RL)
# ══════════════════════════════════════════════════════════════════

import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(13, 7))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.28)
ax_a = fig.add_subplot(gs[0, 0])  # top-left: FI
ax_b = fig.add_subplot(gs[1, 0])  # bottom-left: DN
ax_c = fig.add_subplot(gs[:, 1])  # right column spanning both rows: EH/BP/RL

x = np.arange(len(YEARS))

def shade_severe(ax):
    for i, y in enumerate(YEARS):
        if y in SEVERE:
            ax.axvspan(i - 0.4, i + 0.4, facecolor=SEV_COLOR, alpha=0.8, zorder=0)

def get_stats(df_shares):
    """Compute median, q25, q75 across tracts for each year."""
    medians, q25s, q75s = [], [], []
    for y in YEARS:
        v = df_shares[y].values.astype(float)
        medians.append(np.median(v))
        q25s.append(np.percentile(v, 25))
        q75s.append(np.percentile(v, 75))
    return np.array(medians), np.array(q25s), np.array(q75s)

def annotate_endpoints(ax, x_arr, medians, color, offset_y=0.02, side="both"):
    """Add percentage labels above/below data points, staying inside the plot.
    offset_y: vertical nudge in data coords. Positive = above, negative = below."""
    indices = {"both": [0, -1], "start": [0], "end": [-1]}[side]
    for idx in indices:
        val = medians[idx]
        # Place label above the point, centered horizontally on the marker
        ha = "center"
        ax.annotate(f"{val*100:.0f}%", xy=(x_arr[idx], val),
                    xytext=(x_arr[idx], val + offset_y),
                    fontsize=8, fontweight="bold", color=color, ha=ha, va="bottom",
                    clip_on=True, zorder=5,
                    bbox=dict(boxstyle="round,pad=0.1", facecolor="white",
                              edgecolor="none", alpha=0.7))

def plot_two_groups(ax, owner_key, renter_key, title, panel_label):
    shade_severe(ax)
    o_med, o_q25, o_q75 = get_stats(share_dfs[owner_key])
    r_med, r_q25, r_q75 = get_stats(share_dfs[renter_key])
    ax.plot(x, o_med, "-o", ms=5, lw=2.2, color=OWNER_COLOR, label="Homeowner", zorder=3)
    ax.fill_between(x, o_q25, o_q75, alpha=0.15, color=OWNER_COLOR, zorder=1)
    ax.plot(x, r_med, "-s", ms=5, lw=2.2, color=RENTER_COLOR, label="Renter", zorder=3)
    ax.fill_between(x, r_q25, r_q75, alpha=0.15, color=RENTER_COLOR, zorder=1)
    # Annotate start (2011) and end (2023) values
    # When values are close, put one above and one below to avoid overlap
    for side_idx, side in [(0, "start"), (-1, "end")]:
        o_val, r_val = o_med[side_idx], r_med[side_idx]
        if abs(o_val - r_val) < 0.05:
            # Close together: higher one above, lower one below
            if o_val >= r_val:
                annotate_endpoints(ax, x, o_med, OWNER_COLOR, offset_y=0.015, side=side)
                annotate_endpoints(ax, x, r_med, RENTER_COLOR, offset_y=-0.04, side=side)
            else:
                annotate_endpoints(ax, x, o_med, OWNER_COLOR, offset_y=-0.04, side=side)
                annotate_endpoints(ax, x, r_med, RENTER_COLOR, offset_y=0.015, side=side)
        else:
            annotate_endpoints(ax, x, o_med, OWNER_COLOR, offset_y=0.015, side=side)
            annotate_endpoints(ax, x, r_med, RENTER_COLOR, offset_y=0.015, side=side)
    ax.set_title(title, pad=8)
    ax.text(-0.08, 1.05, panel_label, transform=ax.transAxes, fontsize=12,
            fontweight="bold", ha="left", va="bottom")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_ylim(0, None)
    ax.set_ylabel("Annual adoption rate (%)")
    ax.set_xticks(x[::2])
    ax.set_xticklabels([str(y) for y in YEARS[::2]])
    ax.set_xlabel("Year")

# ── (a) Flood Insurance: Owner vs Renter ──
plot_two_groups(ax_a, "owner_FI", "renter_FI", "Flood Insurance", "(a)")
ax_a.legend(loc="upper right", fontsize=9, frameon=True, facecolor="white",
            edgecolor="black", framealpha=1.0)

# ── (b) Do Nothing: Owner vs Renter ──
plot_two_groups(ax_b, "owner_DN", "renter_DN", "No New Action", "(b)")

# ── (c) Structural Adaptation vs Relocation ──
shade_severe(ax_c)

ACTION_STYLES = {
    "owner_EH": {"color": "#2ca02c", "ls": "-",  "marker": "o", "label": "Elevation — Homeowner"},
    "owner_BP": {"color": "#f59e0b", "ls": "-",  "marker": "^", "label": "Buyout — Homeowner"},
    "renter_RL": {"color": "#d62828", "ls": "--", "marker": "s", "label": "Relocation — Renter"},
}

panel_c_meds = {}
for key, style in ACTION_STYLES.items():
    med, q25, q75 = get_stats(share_dfs[key])
    panel_c_meds[key] = med
    ax_c.plot(x, med, style["ls"], marker=style["marker"], ms=5, lw=2.2,
              color=style["color"], label=style["label"], zorder=3)
    ax_c.fill_between(x, q25, q75, alpha=0.12, color=style["color"], zorder=1)

# Annotate panel (c) endpoints
for key, style in ACTION_STYLES.items():
    med = panel_c_meds[key]
    annotate_endpoints(ax_c, x, med, style["color"], offset_y=0.008)

ax_c.set_title("Structural Adaptation and Relocation", pad=8)
ax_c.text(-0.05, 1.02, "(c)", transform=ax_c.transAxes, fontsize=12,
          fontweight="bold", ha="left", va="bottom")
ax_c.set_ylabel("Annual adoption rate (%)")
ax_c.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
ax_c.set_ylim(0, None)
ax_c.set_xticks(x[::2])
ax_c.set_xticklabels([str(y) for y in YEARS[::2]])
ax_c.set_xlabel("Year")
ax_c.legend(loc="lower left", fontsize=9, frameon=True, facecolor="white",
            edgecolor="black", framealpha=1.0)
out_path = BASE / "visualization/decisions/fig6_rq2_divergence.png"
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"\nSaved: {out_path}")
