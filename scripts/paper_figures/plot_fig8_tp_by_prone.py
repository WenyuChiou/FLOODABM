# -*- coding: utf-8 -*-
"""
Figure 8: TP median trajectories by flood-prone status (2-panel)
  (a) Homeowner TP — flood-prone (solid) vs non-prone (dashed)
  (b) Renter TP — flood-prone (solid) vs non-prone (dashed)

Matches Figure 7(c)-(f) style: population-weighted median + IQR band.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from utils.plots_modular.style import (
    set_paper_style, shade_severe_years,
    COLORS, COLOR_OWNER, COLOR_RENTER,
)

def panel_label(ax, label="(a)", x=-0.12, y=1.02):
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=18, fontweight="bold", va="bottom", ha="right")

# 50-run Monte Carlo baseline data lives on local disk.
MC_ROOT = Path(r"C:\FLOODABM_mc50_v2")
N_RUNS = 50

def run_path(scenario, run_id):
    return MC_ROOT / scenario / f"run_{run_id:02d}" / "baseline" / scenario

# Legacy single-run dir kept only as a fallback for flood_years_by_tract.csv.
BASE = ROOT / "outputs" / "baseline" / "baseline"
FLOOD_CFG = ROOT / "config" / "overall_md_mean_by_tract_2011_2023.json"
SEVERE_YEARS = [2011, 2021]
FP_THRESHOLD = 7  # tracts with depth > 0 in at least 7 of 13 years = flood-prone

FIG_DIR = Path(r"C:\Users\wenyu\OneDrive - Lehigh University\Desktop\Lehigh"
               r"\NSF-project\ABM\paper\Figure")


def classify_flood_prone(flood_years_csv: Path, threshold: int) -> tuple[set, set]:
    """Classify tracts as flood-prone or non-prone based on number of flood years."""
    df = pd.read_csv(flood_years_csv, dtype={"tract_geoid": str})
    # Count years where depth_m > 0 for each tract
    df["flooded"] = df["depth_m"] > 0
    flood_counts = df.groupby("tract_geoid")["flooded"].sum()
    fp = set(flood_counts[flood_counts >= threshold].index)
    nfp = set(flood_counts[flood_counts < threshold].index)
    print(f"Flood-prone: {len(fp)} tracts, Non-prone: {len(nfp)} tracts (threshold={threshold})")
    return fp, nfp


def weighted_stats(values, weights):
    """Population-weighted mean + Q25/Q75."""
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    w = np.where(weights > 0, weights, 1e-6)
    wmean = np.average(values, weights=w)
    # Weighted quantiles
    idx = np.argsort(values)
    sv, sw = values[idx], w[idx]
    cum = np.cumsum(sw)
    cum_norm = (cum - 0.5 * sw) / cum[-1]
    q25 = np.interp(0.25, cum_norm, sv)
    q75 = np.interp(0.75, cum_norm, sv)
    return wmean, q25, q75


def compute_run_series(run_dir, tp_col, group, fp_tracts, nfp_tracts):
    """For one run, return (fp_series, nfp_series) of length n_years.
    Each is the population-weighted mean TP across flood-prone (or non-prone)
    tracts, one value per year."""
    import glob, re
    tp_path = run_dir / "tp_traj.csv"
    if not tp_path.exists():
        return None, None, None
    tp = pd.read_csv(tp_path, dtype={"tract_geoid": str})
    tp = tp[tp["phase"] == "after"]
    years = sorted(tp["year"].unique())

    dec_dir = run_dir / "decisions"
    dec_files = {}
    for f in sorted(dec_dir.glob("decisions_mgmix_*.csv")):
        m = re.search(r"mgmix_(\d{4})\.csv", str(f))
        if m:
            dec_files[int(m.group(1))] = f

    fp_means, nfp_means = [], []
    for y in years:
        yr_tp = tp[tp["year"] == y].groupby("tract_geoid")[tp_col].mean()
        if y in dec_files:
            dec = pd.read_csv(dec_files[y], dtype={"tract_geoid": str})
            grp = dec[dec["group"] == group]
            pop = grp.groupby("tract_geoid").size()
        else:
            pop = pd.Series(1, index=yr_tp.index)

        fp_idx = yr_tp.index.intersection(fp_tracts)
        nfp_idx = yr_tp.index.intersection(nfp_tracts)
        fp_v = yr_tp.loc[fp_idx].values.astype(float)
        fp_w = pop.reindex(fp_idx).fillna(1).values.astype(float)
        wm_fp, _, _ = weighted_stats(fp_v, fp_w)
        fp_means.append(wm_fp)

        nfp_v = yr_tp.loc[nfp_idx].values.astype(float)
        nfp_w = pop.reindex(nfp_idx).fillna(1).values.astype(float)
        wm_nfp, _, _ = weighted_stats(nfp_v, nfp_w)
        nfp_means.append(wm_nfp)

    return years, np.array(fp_means), np.array(nfp_means)


def main():
    set_paper_style()

    # Classify tracts using any available run's flood_years_by_tract.csv
    # (the depth sequence is fixed across runs, so any run works).
    flood_csv = run_path("baseline", 1) / "flood_years_by_tract.csv"
    if not flood_csv.exists():
        flood_csv = BASE / "flood_years_by_tract.csv"
    fp_tracts, nfp_tracts = classify_flood_prone(flood_csv, FP_THRESHOLD)

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 6))

    for ax, tp_col, group, color, title, plbl in [
        (ax_a, "TP_owner", "owner", COLOR_OWNER, "Homeowner", "(a)"),
        (ax_b, "TP_renter", "renter", COLOR_RENTER, "Renter", "(b)"),
    ]:
        # Stack per-run population-weighted TP across 50 baseline runs.
        fp_runs, nfp_runs = [], []
        years_ref = None
        for rid in range(1, N_RUNS + 1):
            rd = run_path("baseline", rid)
            if not rd.exists():
                continue
            yrs, fp_s, nfp_s = compute_run_series(rd, tp_col, group,
                                                    fp_tracts, nfp_tracts)
            if fp_s is None:
                continue
            if years_ref is None:
                years_ref = yrs
            fp_runs.append(fp_s)
            nfp_runs.append(nfp_s)

        years = years_ref
        print(f"  {group}: {len(fp_runs)} runs loaded")

        fp_arr = np.stack(fp_runs)      # (n_runs, n_years)
        nfp_arr = np.stack(nfp_runs)

        # 50-run median + IQR across runs
        fp_mean = np.median(fp_arr, axis=0)
        fp_lo = np.percentile(fp_arr, 25, axis=0)
        fp_hi = np.percentile(fp_arr, 75, axis=0)
        nfp_mean = np.median(nfp_arr, axis=0)
        nfp_lo = np.percentile(nfp_arr, 25, axis=0)
        nfp_hi = np.percentile(nfp_arr, 75, axis=0)

        shade_severe_years(ax, years, SEVERE_YEARS)

        x = np.arange(len(years))

        # IQR bands underneath the median lines (flood-prone darker, non-prone lighter)
        ax.fill_between(x, fp_lo, fp_hi, color=color, alpha=0.30, zorder=2, linewidth=0,
                        label="_nolegend_")
        ax.fill_between(x, nfp_lo, nfp_hi, color=color, alpha=0.18, zorder=2, linewidth=0,
                        label="_nolegend_")

        # Flood-prone mean: solid
        ax.plot(x, fp_mean, "-o", ms=5, lw=2.5, color=color,
                label="Flood-prone", zorder=3)

        # Non-prone mean: dashed
        ax.plot(x, nfp_mean, "--s", ms=5, lw=2.5, color=color,
                label="Non-prone", zorder=3, alpha=0.8)

        # Endpoint annotations (value labels)
        for arr, style in [(fp_mean, "solid"), (nfp_mean, "dashed")]:
            for idx in [0, -1]:
                val = arr[idx]
                fmt = f"{val:.2f}"
                offset = 0.03 if style == "solid" else -0.05
                ax.annotate(fmt, xy=(x[idx], val),
                            xytext=(x[idx], val + offset),
                            fontsize=15, fontweight="bold", color=color,
                            ha="center", va="bottom",
                            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                                      edgecolor=color, linewidth=0.6, alpha=0.9))

        # Gap annotations at start and end
        gap_start = fp_mean[0] - nfp_mean[0]
        gap_end = fp_mean[-1] - nfp_mean[-1]
        mid_start = (fp_mean[0] + nfp_mean[0]) / 2
        mid_end = (fp_mean[-1] + nfp_mean[-1]) / 2
        ax.annotate("", xy=(x[0] + 0.5, fp_mean[0]), xytext=(x[0] + 0.5, nfp_mean[0]),
                    arrowprops=dict(arrowstyle="<->", color="0.3", lw=1.5))
        ax.text(x[0] + 1.3, mid_start, f"\u0394={gap_start:.2f}",
                fontsize=15, fontweight="bold", color="0.2", va="center",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          edgecolor="0.4", linewidth=0.6, alpha=1.0))
        ax.annotate("", xy=(x[-1] - 0.5, fp_mean[-1]), xytext=(x[-1] - 0.5, nfp_mean[-1]),
                    arrowprops=dict(arrowstyle="<->", color="0.3", lw=1.5))
        ax.text(x[-1] - 1.3, mid_end, f"\u0394={gap_end:.2f}",
                fontsize=15, fontweight="bold", color="0.2", va="center",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          edgecolor="0.4", linewidth=0.6, alpha=1.0))

        ax.set_title(title, pad=8, fontsize=19, fontweight="bold")
        panel_label(ax, plbl)
        ax.set_xlabel("Year", fontsize=18)
        ax.set_ylabel("Weighted tract-level mean TP", fontsize=18)
        ax.set_ylim(0.0, 1.05)
        ax.set_xticks(x[::2])
        ax.set_xticklabels([str(y) for y in years[::2]])
        ax.tick_params(axis="both", labelsize=15)
        ax.legend(loc="upper right", fontsize=15,
                  frameon=True, edgecolor="black", facecolor="white")

    fig.tight_layout(w_pad=2.5)

    # Save
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "pdf"]:
        out = FIG_DIR / f"Fig8_TP_by_prone.{ext}"
        fig.savefig(out, dpi=600, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
