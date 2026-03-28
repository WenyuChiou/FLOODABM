# -*- coding: utf-8 -*-
"""
Fig 7 v2 — 2-panel version (DN panel moved to SM as Figure S3)
  (a) FI — Owner vs Renter
  (b) Structural Adaptation (EH, BP) vs Relocation (RL)

Also generates Figure S3: DN — Owner vs Renter (for SM)
"""
import sys, re, glob, json
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent.parent  # project root
sys.path.insert(0, str(ROOT))
from utils.plots_modular.style import set_paper_style, panel_label as _panel_label, shade_severe_years, COLORS
set_paper_style()

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

# ── Load all decision files ──
file_map = {}
for f in sorted(glob.glob(str(BASE / "decisions/decisions_mgmix_*.csv"))):
    m = re.search(r"mgmix_(\d{4})\.csv", f)
    if m:
        file_map[int(m.group(1))] = f

YEARS = sorted(file_map.keys())
print(f"Years: {YEARS[0]}-{YEARS[-1]}")

# ── Compute per-tract action shares + population counts ──
def compute_tract_shares(years, group, action):
    all_tracts = set()
    yearly = {}
    yearly_n = {}          # tract population per year
    for y in years:
        df = pd.read_csv(file_map[y])
        grp = df[df["group"] == group].copy()
        grp["tract_geoid"] = grp["tract_geoid"].astype(str)
        tracts = sorted(grp["tract_geoid"].unique())
        all_tracts.update(tracts)
        shares = {}
        counts = {}
        for t in tracts:
            t_df = grp[grp["tract_geoid"] == t]
            n = len(t_df)
            counts[t] = n
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
        yearly_n[y] = counts
    all_tracts = sorted(all_tracts)
    data = {y: [yearly[y].get(t, 0.0) for t in all_tracts] for y in years}
    data_n = {y: [yearly_n[y].get(t, 0) for t in all_tracts] for y in years}
    df_shares = pd.DataFrame(data, index=all_tracts)
    df_pop = pd.DataFrame(data_n, index=all_tracts)
    return df_shares, df_pop

# Compute all needed DataFrames
keys_needed = [
    ("owner", "FI"), ("renter", "FI"),
    ("owner", "EH"), ("owner", "BP"), ("renter", "RL"),
    ("owner", "DN"), ("renter", "DN"),
]
share_dfs = {}
pop_dfs = {}
for group, action in keys_needed:
    key = f"{group}_{action}"
    print(f"  Computing {key}...")
    share_dfs[key], pop_dfs[key] = compute_tract_shares(YEARS, group, action)

OWNER_COLOR = "#2c3e50"   # dark navy (not blue)
RENTER_COLOR = "#d62828"  # red
x = np.arange(len(YEARS))

def shade_severe(ax):
    shade_severe_years(ax, YEARS, SEVERE)

def _weighted_quantile(values, weights, q):
    """Weighted quantile (linear interpolation)."""
    idx = np.argsort(values)
    sv, sw = values[idx], weights[idx]
    cum = np.cumsum(sw)
    cum_norm = (cum - 0.5 * sw) / cum[-1]      # midpoint CDF
    return np.interp(q, cum_norm, sv)

def get_stats(df_shares, df_pop=None):
    """Population-weighted mean + IQR (Q25/Q75) across tracts."""
    means, lo, hi = [], [], []
    for y in YEARS:
        v = df_shares[y].values.astype(float)
        if df_pop is not None and y in df_pop.columns:
            w = df_pop[y].values.astype(float)
            w = np.where(w > 0, w, 1e-6)  # avoid zero weights
            wmean = np.average(v, weights=w)
            q25 = _weighted_quantile(v, w, 0.25)
            q75 = _weighted_quantile(v, w, 0.75)
        else:
            wmean = np.mean(v)
            q25, q75 = np.percentile(v, [25, 75])
        means.append(wmean)
        lo.append(q25)
        hi.append(q75)
    return np.array(means), np.array(lo), np.array(hi)

def annotate_endpoints(ax, x_arr, medians, color, offset_y=0.02, side="both"):
    indices = {"both": [0, -1], "start": [0], "end": [-1]}[side]
    for idx in indices:
        val = medians[idx]
        ha = "center"
        fmt = f"{val*100:.1f}%" if val < 0.05 else f"{val*100:.0f}%"
        ax.annotate(fmt, xy=(x_arr[idx], val),
                    xytext=(x_arr[idx], val + offset_y),
                    fontsize=8, fontweight="bold", color=color, ha=ha, va="bottom",
                    clip_on=True, zorder=5,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                              edgecolor=color, linewidth=0.6, alpha=1.0))

def plot_two_groups(ax, owner_key, renter_key, title, panel_label):
    shade_severe(ax)
    o_med, o_q25, o_q75 = get_stats(share_dfs[owner_key], pop_dfs[owner_key])
    r_med, r_q25, r_q75 = get_stats(share_dfs[renter_key], pop_dfs[renter_key])
    ax.plot(x, o_med, "-o", ms=5, lw=2.2, color=OWNER_COLOR, label="Homeowner", zorder=3)
    ax.fill_between(x, o_q25, o_q75, alpha=0.15, color=OWNER_COLOR, zorder=1)
    ax.plot(x, r_med, "-s", ms=5, lw=2.2, color=RENTER_COLOR, label="Renter", zorder=3)
    ax.fill_between(x, r_q25, r_q75, alpha=0.15, color=RENTER_COLOR, zorder=1)
    for side_idx, side in [(0, "start"), (-1, "end")]:
        o_val, r_val = o_med[side_idx], r_med[side_idx]
        if abs(o_val - r_val) < 0.05:
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

# ══════════════════════════════════════════════════════════════════
# Figure 7: Left = FI (large), Right = RL (top) + EH/BP (bottom)
# ══════════════════════════════════════════════════════════════════
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(14, 7))
gs = gridspec.GridSpec(2, 2, width_ratios=[1.2, 1], hspace=0.35, wspace=0.30)
ax_a = fig.add_subplot(gs[:, 0])   # FI spans both rows
ax_b = fig.add_subplot(gs[0, 1])   # RL top-right
ax_c = fig.add_subplot(gs[1, 1])   # EH+BP bottom-right

EH_COLOR = "#2ca02c"
BP_COLOR = "#f59e0b"

def setup_ax(ax, title, plbl, fs_title=13):
    ax.set_title(title, fontsize=fs_title, pad=12)
    ax.text(-0.02, 1.08, plbl, transform=ax.transAxes, fontsize=14,
            fontweight="bold", ha="left", va="bottom")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_ylim(0, None)
    ax.set_ylabel("Annual adoption rate (%)", fontsize=11)
    ax.set_xticks(x[::2])
    ax.set_xticklabels([str(y) for y in YEARS[::2]], fontsize=10)
    ax.tick_params(axis="y", labelsize=10)
    ax.set_xlabel("Year", fontsize=11)
    ax.legend(loc="best", fontsize=10, frameon=True, facecolor="white",
              edgecolor="black", framealpha=1.0)

def plot_line_with_band(ax, key, color, label, marker="-o", ms=5):
    med, q25, q75 = get_stats(share_dfs[key], pop_dfs[key])
    ax.fill_between(x, q25, q75, alpha=0.15, color=color, zorder=1)
    ax.plot(x, med, marker, ms=ms, lw=2.2, color=color, label=label, zorder=3)
    return med

# (a) FI — Owner vs Renter (large left panel)
shade_severe(ax_a)
o_fi = plot_line_with_band(ax_a, "owner_FI", OWNER_COLOR, "Homeowner", "-o")
r_fi = plot_line_with_band(ax_a, "renter_FI", RENTER_COLOR, "Renter", "-s")
for side in ["start", "end"]:
    idx = 0 if side == "start" else -1
    gap = r_fi[idx] - o_fi[idx]
    if abs(gap) < 0.06:
        annotate_endpoints(ax_a, x, o_fi, OWNER_COLOR, offset_y=-0.08, side=side)
        annotate_endpoints(ax_a, x, r_fi, RENTER_COLOR, offset_y=0.04, side=side)
    else:
        annotate_endpoints(ax_a, x, o_fi, OWNER_COLOR, offset_y=0.025, side=side)
        annotate_endpoints(ax_a, x, r_fi, RENTER_COLOR, offset_y=0.025, side=side)
setup_ax(ax_a, "Flood Insurance (FI)", "(a)", fs_title=14)

# (b) RL — Renter (top-right)
shade_severe(ax_b)
r_rl = plot_line_with_band(ax_b, "renter_RL", RENTER_COLOR, "Renter", "--s")
annotate_endpoints(ax_b, x, r_rl, RENTER_COLOR, offset_y=0.02)
setup_ax(ax_b, "Relocation", "(b)")

# (c) EH + BP — Owner (bottom-right)
shade_severe(ax_c)
o_eh = plot_line_with_band(ax_c, "owner_EH", EH_COLOR, "Elevation", "-o")
o_bp = plot_line_with_band(ax_c, "owner_BP", BP_COLOR, "Buyout Prep.", "-^")
for side in ["start", "end"]:
    idx = 0 if side == "start" else -1
    gap = o_eh[idx] - o_bp[idx]
    if abs(gap) < 0.008:
        annotate_endpoints(ax_c, x, o_eh, EH_COLOR, offset_y=0.004, side=side)
        annotate_endpoints(ax_c, x, o_bp, BP_COLOR, offset_y=-0.008, side=side)
    else:
        annotate_endpoints(ax_c, x, o_eh, EH_COLOR, offset_y=0.004, side=side)
        annotate_endpoints(ax_c, x, o_bp, BP_COLOR, offset_y=0.004, side=side)
setup_ax(ax_c, "Elevation & Buyout Preparation", "(c)")
out7 = BASE / "visualization/decisions/fig7_rq2_divergence_2panel.png"
out7.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out7, dpi=600, bbox_inches="tight")
plt.close()
print(f"Saved Figure 7: {out7}")

# ══════════════════════════════════════════════════════════════════
# Figure S3 (SM): DN — Owner vs Renter
# ══════════════════════════════════════════════════════════════════
fig_s3, ax_s3 = plt.subplots(1, 1, figsize=(6, 4))
plot_two_groups(ax_s3, "owner_DN", "renter_DN", "Annual Do-Nothing Rate", "")
ax_s3.legend(loc="upper left", fontsize=9, frameon=True, facecolor="white",
             edgecolor="black", framealpha=1.0)
fig_s3.tight_layout()
out_s3 = BASE / "visualization/decisions/figure_s3_dn_annual.png"
fig_s3.savefig(out_s3, dpi=600, bbox_inches="tight")
plt.close()
print(f"Saved Figure S3: {out_s3}")
