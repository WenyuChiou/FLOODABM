# -*- coding: utf-8 -*-
"""
Task 4a: merge the 3 stacked county panels of Fig4_nfip_validation into ONE panel.

This REPLICATES the data pipeline of plot_nfip_validation_paper.py exactly (same
MC root, same (possibly swapped) COUNTY_MAP, same filtered-subset Pearson r) so the
only change is layout: 3 counties overlaid on a single time-series axis.

Outputs two draft candidates to the validation/ folder for review (does NOT
overwrite the paper Figure/ copy):
  Fig4_merged_timeseries.png  - 3 counties overlaid, sim(solid)+actual(dashed)
  Fig4_merged_scatter.png     - pooled sim vs actual scatter, colored by county
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

ROOT = Path(r"C:\Users\wenyu\OneDrive - Lehigh University\Desktop\Lehigh\NSF-project\ABM\paper\draft\mg_sensitivity\FLOODABM")
MC_ROOT = Path(r"C:\FLOODABM_mc50")
N_RUNS = 50
VAL_DIR = Path(r"C:\Users\wenyu\OneDrive - Lehigh University\Desktop\Lehigh\NSF-project\ABM\paper\validation")
TS_CSV = VAL_DIR / "sim_vs_actual_standardized - 複製.csv"
SC_CSV = VAL_DIR / "sim_vs_actual_standardized.csv"

COUNTIES = ["Essex", "Morris", "Passaic"]
WARMUP = [2011, 2012]
COUNTY_MAP = {"34013": "Essex", "34027": "Passaic", "34031": "Morris"}  # replicate as-is (flagged)
YEARS = list(range(2011, 2024))
# colorblind-safe county colors
CCOL = {"Essex": "#0072B2", "Morris": "#E69F00", "Passaic": "#009E73"}


def run_path(rid):
    return MC_ROOT / "baseline" / f"run_{rid:02d}" / "baseline" / "baseline"


def compute_sim_avg_by_county(rid):
    rd = run_path(rid)
    rows = []
    for y in YEARS:
        f = rd / "finance" / f"finance_tract_{y}.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f, dtype={"tract_geoid": str})
        df["year"] = y
        df["county"] = df["tract_geoid"].str[:5].map(COUNTY_MAP)
        rows.append(df[["county", "year", "payout_total_usd", "claims"]])
    if not rows:
        return None
    sim = pd.concat(rows, ignore_index=True)
    agg = sim.groupby(["county", "year"]).agg(
        sim_payout=("payout_total_usd", "sum"), sim_claims=("claims", "sum")).reset_index()
    agg["sim_avg"] = np.where(agg["sim_claims"] > 0, agg["sim_payout"] / agg["sim_claims"], np.nan)
    return agg[["county", "year", "sim_avg"]]


def zscore_by_county(df):
    out = df.copy()
    for c in COUNTIES:
        mask = out["county"] == c
        vals = out.loc[mask, "sim_avg"]
        mu, sd = vals.mean(skipna=True), vals.std(skipna=True)
        out.loc[mask, "sim_z"] = (vals - mu) / sd if sd and sd > 0 else 0.0
    return out


print(f"Loading {N_RUNS} MC runs from {MC_ROOT}...")
run_tables = []
for rid in range(1, N_RUNS + 1):
    raw = compute_sim_avg_by_county(rid)
    if raw is not None:
        run_tables.append(zscore_by_county(raw))
print(f"  loaded {len(run_tables)} runs")


def to_matrix(df):
    mat = np.full((len(COUNTIES), len(YEARS)), np.nan)
    for i, c in enumerate(COUNTIES):
        sub = df[df["county"] == c].set_index("year")
        for j, y in enumerate(YEARS):
            if y in sub.index:
                mat[i, j] = sub.loc[y, "sim_z"]
    return mat


sim_z_runs = np.stack([to_matrix(t) for t in run_tables])
sim_z_med = np.nanmedian(sim_z_runs, axis=0)
sim_z_q25 = np.nanpercentile(sim_z_runs, 25, axis=0)
sim_z_q75 = np.nanpercentile(sim_z_runs, 75, axis=0)

act = pd.read_csv(TS_CSV)
act_z_mat = np.full((len(COUNTIES), len(YEARS)), np.nan)
for i, c in enumerate(COUNTIES):
    sub = act[act["county"] == c].set_index("year")
    for j, y in enumerate(YEARS):
        if y in sub.index and "act_z" in sub.columns:
            v = sub.loc[y, "act_z"]
            act_z_mat[i, j] = v if pd.notna(v) else np.nan

# per-county filtered Pearson r (matches paper convention)
sc = pd.read_csv(SC_CSV)
filtered_years = {c: sorted(sc[sc["county"] == c]["year"].unique().tolist()) for c in COUNTIES}
r_per_county = {}
for i, c in enumerate(COUNTIES):
    years_c = filtered_years[c]
    run_sim_filt = []
    for run_df in run_tables:
        sub = run_df[(run_df["county"] == c) & (run_df["year"].isin(years_c))].sort_values("year")
        vals = sub["sim_avg"].values.astype(float)
        mu, sd = np.nanmean(vals), np.nanstd(vals, ddof=0)
        z = (vals - mu) / sd if sd > 0 else np.zeros_like(vals)
        run_sim_filt.append(z)
    med_filt = np.nanmedian(np.stack(run_sim_filt), axis=0)
    act_z_filt = sc[sc["county"] == c].sort_values("year")["act_z"].values.astype(float)
    valid = ~np.isnan(med_filt) & ~np.isnan(act_z_filt)
    r_per_county[c] = np.corrcoef(med_filt[valid], act_z_filt[valid])[0, 1] if valid.sum() >= 2 else np.nan
print("per-county r:", {c: round(r_per_county[c], 2) for c in COUNTIES})

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 14, "axes.labelsize": 18,
    "xtick.labelsize": 15, "ytick.labelsize": 15,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.2, "grid.linestyle": "--",
})
yrs = np.array(YEARS)

# ============ CANDIDATE 1: time-series overlay ============
fig, ax = plt.subplots(figsize=(11, 5.6))
ax.axvspan(2010.5, 2012.5, color="0.5", alpha=0.12, zorder=0)
for i, c in enumerate(COUNTIES):
    col = CCOL[c]
    ax.fill_between(yrs, sim_z_q25[i], sim_z_q75[i], color=col, alpha=0.10, zorder=1, linewidth=0)
    ax.plot(yrs, sim_z_med[i], color=col, marker="o", ms=6, lw=2.0, zorder=3)
    a = act_z_mat[i]
    v = ~np.isnan(a)
    ax.plot(yrs[v], a[v], color=col, marker="s", ms=6, lw=1.8, linestyle="--",
            markerfacecolor="white", markeredgewidth=1.5, zorder=3)
ax.set_xlabel("Year")
ax.set_ylabel("z-score (avg. payout per claim)")
ax.set_xlim(2010.2, 2023.8)
ax.set_xticks(YEARS)
# no embedded title (matches other main-text figures; caption carries the title).
# add top headroom so the two legends clear the data without a title above.
ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1] + 0.5)
county_handles = [Patch(facecolor=CCOL[c], label=f"{c} (r = {r_per_county[c]:.2f})") for c in COUNTIES]
style_handles = [
    Line2D([0], [0], color="0.3", marker="o", lw=2.0, label="Simulated (median)"),
    Line2D([0], [0], color="0.3", marker="s", lw=1.8, linestyle="--",
           markerfacecolor="white", markeredgewidth=1.5, label="Actual (NFIP)"),
    Patch(facecolor="0.5", alpha=0.12, label="Warm-up"),
]
# place both legends in the empty upper band (x>=2013, high z) so neither covers
# the warm-up shading / 2011 spikes on the LEFT.
leg1 = ax.legend(handles=county_handles, loc="upper center", bbox_to_anchor=(0.42, 1.0),
                 fontsize=11, frameon=True, framealpha=0.95, ncol=1)
ax.add_artist(leg1)
ax.legend(handles=style_handles, loc="upper right", fontsize=11, frameon=True, framealpha=0.95)
fig.tight_layout()
PAPER_FIG = Path(r"C:\Users\wenyu\OneDrive - Lehigh University\Desktop\Lehigh\NSF-project\ABM\paper\Figure")
fig.savefig(VAL_DIR / "Fig4_merged_timeseries.png", dpi=300, bbox_inches="tight")
# FINALIZE as the paper Fig4 (replaces the 3-panel version; layout-only change,
# county mapping replicated as-is per user decision "就基於原圖改").
for ext in ("png", "pdf"):
    fig.savefig(PAPER_FIG / f"Fig4_nfip_validation.{ext}", dpi=300, bbox_inches="tight")
plt.close(fig)
print("saved Fig4_merged_timeseries.png + Figure/Fig4_nfip_validation.{png,pdf}")

# ============ CANDIDATE 2: pooled scatter ============
fig, ax = plt.subplots(figsize=(6.4, 6.2))
allx, ally = [], []
for i, c in enumerate(COUNTIES):
    a = act_z_mat[i]; s = sim_z_med[i]
    v = ~np.isnan(a) & ~np.isnan(s) & ~np.isin(yrs, WARMUP)
    ax.scatter(a[v], s[v], color=CCOL[c], s=55, alpha=0.85, edgecolors="w", linewidth=0.6,
               label=f"{c} (r = {r_per_county[c]:.2f})", zorder=3)
    allx += list(a[v]); ally += list(s[v])
allx, ally = np.array(allx), np.array(ally)
r_all = np.corrcoef(allx, ally)[0, 1]
lims = [min(allx.min(), ally.min()) - 0.3, max(allx.max(), ally.max()) + 0.3]
ax.plot(lims, lims, color="0.5", linestyle=":", lw=1.5, zorder=1, label="1:1")
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel("Actual NFIP z-score")
ax.set_ylabel("Simulated z-score (median)")
ax.set_title("NFIP validation (post-warm-up, all counties)", fontweight="bold")
ax.text(0.05, 0.95, f"pooled r = {r_all:.2f}\nn = {len(allx)}", transform=ax.transAxes,
        ha="left", va="top", fontsize=14,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.7"))
ax.legend(loc="lower right", fontsize=11, frameon=True, framealpha=0.95)
fig.tight_layout()
fig.savefig(VAL_DIR / "Fig4_merged_scatter.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"saved Fig4_merged_scatter.png  (pooled r={r_all:.2f}, n={len(allx)})")
print("Done.")
