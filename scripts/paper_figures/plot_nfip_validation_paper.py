# -*- coding: utf-8 -*-
"""
Generate publication-quality 3-panel NFIP validation figure.

50-run Monte Carlo version:
  - Simulated values are computed per MC run (avg payout/claim, county-year)
  - Each run is z-standardized within its own county series
  - Blue line shows the 50-run median with an IQR shaded band
  - Red dashed line shows the reference NFIP actual z-scores (deterministic)
  - r annotation is the correlation between the median simulated z and the
    actual z, consistent with user preference (median-only, no brackets).
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path
import glob, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ── paths ──
ROOT = Path(__file__).resolve().parent.parent.parent  # project root
MC_ROOT = Path(r"C:\FLOODABM_mc50")
N_RUNS = 50

# Reference NFIP actual data — unchanged across runs.
# TS_CSV has all 13 years (for time-series plotting),
# SC_CSV has the filtered subset (post-warmup, excluding low-confidence years)
# used for the Pearson r annotation — matches the convention used in the
# paper's original Fig 10 (Essex: skip 2015/2016; Morris: skip 2016;
# Passaic: skip 2016/2018).
VAL_DIR = Path(r"C:\Users\wenyu\OneDrive - Lehigh University\Desktop\Lehigh"
               r"\NSF-project\ABM\paper\validation")
TS_CSV = VAL_DIR / "sim_vs_actual_standardized - 複製.csv"   # has county, year, act_z
SC_CSV = VAL_DIR / "sim_vs_actual_standardized.csv"           # filtered subset for r
OUT_DIR = ROOT / "outputs" / "baseline" / "baseline" / "visualization" / "validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PAPER_FIG = Path(r"C:\Users\wenyu\OneDrive - Lehigh University\Desktop\Lehigh"
                 r"\NSF-project\ABM\paper\Figure")

COUNTIES = ["Essex", "Morris", "Passaic"]
WARMUP = [2011, 2012]
COUNTY_MAP = {"34013": "Essex", "34027": "Passaic", "34031": "Morris"}
YEARS = list(range(2011, 2024))


def run_path(rid):
    return MC_ROOT / "baseline" / f"run_{rid:02d}" / "baseline" / "baseline"


def compute_sim_avg_by_county(rid):
    """For one run, return a DataFrame with columns [county, year, sim_avg]."""
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
        sim_payout=("payout_total_usd", "sum"),
        sim_claims=("claims", "sum"),
    ).reset_index()
    agg["sim_avg"] = np.where(agg["sim_claims"] > 0,
                                agg["sim_payout"] / agg["sim_claims"], np.nan)
    return agg[["county", "year", "sim_avg"]]


def zscore_by_county(df):
    """Standardize sim_avg within each county (all years)."""
    out = df.copy()
    for c in COUNTIES:
        mask = out["county"] == c
        vals = out.loc[mask, "sim_avg"]
        mu, sd = vals.mean(skipna=True), vals.std(skipna=True)
        out.loc[mask, "sim_z"] = (vals - mu) / sd if sd and sd > 0 else 0.0
    return out


# ── load 50 runs and stack sim_z ──
print(f"Loading {N_RUNS} MC runs...")
run_tables = []
for rid in range(1, N_RUNS + 1):
    raw = compute_sim_avg_by_county(rid)
    if raw is None:
        continue
    run_tables.append(zscore_by_county(raw))
print(f"  loaded {len(run_tables)} runs")

# Pivot each run into a (county × year) matrix and stack
def to_matrix(df):
    # rows = county in COUNTIES order, cols = year in YEARS order
    mat = np.full((len(COUNTIES), len(YEARS)), np.nan)
    for i, c in enumerate(COUNTIES):
        sub = df[df["county"] == c].set_index("year")
        for j, y in enumerate(YEARS):
            if y in sub.index:
                mat[i, j] = sub.loc[y, "sim_z"]
    return mat

sim_z_runs = np.stack([to_matrix(t) for t in run_tables])  # (n_runs, 3, 13)
sim_z_med = np.nanmedian(sim_z_runs, axis=0)
sim_z_q25 = np.nanpercentile(sim_z_runs, 25, axis=0)
sim_z_q75 = np.nanpercentile(sim_z_runs, 75, axis=0)

# ── load actual z-scores (deterministic reference) ──
act = pd.read_csv(TS_CSV)
act_z_mat = np.full((len(COUNTIES), len(YEARS)), np.nan)
for i, c in enumerate(COUNTIES):
    sub = act[act["county"] == c].set_index("year")
    for j, y in enumerate(YEARS):
        if y in sub.index:
            v = sub.loc[y, "act_z"] if "act_z" in sub.columns else np.nan
            act_z_mat[i, j] = v if pd.notna(v) else np.nan

# ── per-county Pearson r on the filtered subset ──
# Use the same filtered year list the paper originally used (from SC_CSV).
# Recompute sim_z on that restricted subset so mean/std are consistent with
# the filtered years (matching the original `gen_zscore_validation.py` logic).
sc = pd.read_csv(SC_CSV)
filtered_years = {
    c: sorted(sc[sc["county"] == c]["year"].unique().tolist()) for c in COUNTIES
}
print("\n=== Filtered year subset (matching paper convention) ===")
for c in COUNTIES:
    print(f"  {c}: {filtered_years[c]}")

print("\n=== Per-county Pearson r (50-run median sim_z vs actual z, filtered subset) ===")
r_per_county = {}
for i, c in enumerate(COUNTIES):
    years_c = filtered_years[c]
    # Re-standardize sim_avg over the filtered years only, per run
    run_sim_filt = []
    for run_df in run_tables:
        sub = run_df[(run_df["county"] == c) & (run_df["year"].isin(years_c))].sort_values("year")
        vals = sub["sim_avg"].values.astype(float)
        mu, sd = np.nanmean(vals), np.nanstd(vals, ddof=0)
        z = (vals - mu) / sd if sd > 0 else np.zeros_like(vals)
        run_sim_filt.append(z)
    run_sim_filt = np.stack(run_sim_filt)  # (n_runs, n_filtered_years)
    med_filt = np.nanmedian(run_sim_filt, axis=0)

    # Get act_z for the same filtered years from SC_CSV
    sc_c = sc[sc["county"] == c].sort_values("year")
    act_z_filt = sc_c["act_z"].values.astype(float)

    valid = ~np.isnan(med_filt) & ~np.isnan(act_z_filt)
    if valid.sum() >= 2:
        r = np.corrcoef(med_filt[valid], act_z_filt[valid])[0, 1]
    else:
        r = np.nan
    r_per_county[c] = r
    print(f"  {c:8s}: n={valid.sum()}, r={r:.3f}")

# ── style ──
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 14,
    "axes.labelsize": 18,
    "axes.titlesize": 19,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.2,
    "grid.linestyle": "--",
})

# ── figure ──
fig, axes = plt.subplots(3, 1, figsize=(9, 10.5))

for idx, county in enumerate(COUNTIES):
    ax = axes[idx]
    s_med = sim_z_med[idx]
    s_lo = sim_z_q25[idx]
    s_hi = sim_z_q75[idx]
    a = act_z_mat[idx]
    yrs = np.array(YEARS)

    # Warm-up shading
    ax.axvspan(2010.5, 2012.5, color="0.5", alpha=0.12, zorder=0)

    # Simulated 50-run IQR band (Q25-Q75) + median line
    ax.fill_between(yrs, s_lo, s_hi, color="#2563eb", alpha=0.22,
                    zorder=2, linewidth=0)
    ax.plot(yrs, s_med, color="#2563eb", marker="o",
            markersize=6, linewidth=2.0, label="Simulated (median)", zorder=3)

    # Actual (red dashed, squares) — skip NaN years
    valid = ~np.isnan(a)
    ax.plot(yrs[valid], a[valid], color="#ef4444", marker="s",
            markersize=6, linewidth=2.0, linestyle="--", label="Actual", zorder=3)

    ax.set_ylabel("z-score")
    ax.set_title(f"{county} County (2011\u20132023)", fontweight="bold")
    ax.set_xlim(2010.2, 2023.8)
    ax.set_xticks(list(range(2011, 2024)))
    ax.tick_params(length=5, width=1)

    # Pearson r annotation
    r_val = r_per_county[county]
    ax.text(0.5, 0.95, f"r = {r_val:.2f}", transform=ax.transAxes,
            fontsize=15, ha="center", va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="0.7", alpha=0.9))

    # Panel label
    ax.text(-0.08, 1.05, f"({chr(97 + idx)})", transform=ax.transAxes,
            fontsize=18, fontweight="bold", va="top")

    if idx == 0:
        handles, labels = ax.get_legend_handles_labels()
        handles.insert(0, Patch(facecolor="0.5", alpha=0.12, label="Warm-up"))
        labels.insert(0, "Warm-up")
        ax.legend(handles=handles, labels=labels, loc="upper right",
                  fontsize=14, frameon=True, framealpha=0.95)

    if idx < 2:
        ax.set_xticklabels([])
    else:
        ax.set_xlabel("Year")

fig.tight_layout(h_pad=1.5)
plt.subplots_adjust(hspace=0.25)

# Save
for ext in ["png", "pdf"]:
    fig.savefig(OUT_DIR / f"Fig4_nfip_validation.{ext}", dpi=300, bbox_inches="tight")
    fig.savefig(PAPER_FIG / f"Fig4_nfip_validation.{ext}", dpi=300, bbox_inches="tight")
print(f"Saved: {OUT_DIR / 'Fig4_nfip_validation.png'}")
print(f"Saved: {PAPER_FIG / 'Fig4_nfip_validation.png'}")

plt.close(fig)
print("Done.")
