# -*- coding: utf-8 -*-
"""Figure 10: TP distribution violin chart by year (two-panel, household-level)."""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import glob, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent  # project root
sys.path.insert(0, str(ROOT))
from utils.plots_modular.style import (
    set_paper_style, panel_label, shade_severe_years,
    COLORS, COLOR_OWNER, COLOR_RENTER,
)

# 50-run Monte Carlo baseline.
MC_ROOT = Path(r"C:\FLOODABM_mc50")
N_RUNS = 50

def run_path(rid):
    return MC_ROOT / "baseline" / f"run_{rid:02d}" / "baseline" / "baseline"

BASE = ROOT / "outputs" / "baseline" / "baseline"
SEVERE_YEARS = [2011, 2021]


def main():
    set_paper_style()

    # Pool tract-level mean TP across 50 baseline runs. Each (year, group)
    # gets 50 * n_tracts samples, yielding a more robust boxplot than a
    # single-run version that only has n_tracts samples per year.
    owner_pool = {}   # year -> list of tract means across runs
    renter_pool = {}
    n_loaded = 0
    years_ref = None
    for rid in range(1, N_RUNS + 1):
        rd = run_path(rid)
        tp_path = rd / "tp_traj.csv"
        if not tp_path.exists():
            continue
        tp = pd.read_csv(tp_path, dtype={"tract_geoid": str})
        tp = tp[tp["phase"] == "after"]
        yrs = sorted(tp["year"].unique())
        if years_ref is None:
            years_ref = yrs
        for y in yrs:
            yr_tp = tp[tp["year"] == y]
            owner_tract = yr_tp.groupby("tract_geoid")["TP_owner"].mean().values
            renter_tract = yr_tp.groupby("tract_geoid")["TP_renter"].mean().values
            owner_pool.setdefault(y, []).append(owner_tract)
            renter_pool.setdefault(y, []).append(renter_tract)
        n_loaded += 1

    years = years_ref
    owner_data = {y: np.concatenate(owner_pool[y]) for y in years}
    renter_data = {y: np.concatenate(renter_pool[y]) for y in years}
    print(f"Pooled tract-level TP from {n_loaded} MC runs ({years[0]}-{years[-1]})")
    print(f"  Samples per year: owner={len(owner_data[years[0]])}, renter={len(renter_data[years[0]])}")

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 5))

    for ax, data_dict, color, title, plbl in [
        (ax_a, owner_data, COLOR_OWNER, "Homeowner", "(a)"),
        (ax_b, renter_data, COLOR_RENTER, "Renter", "(b)"),
    ]:
        shade_severe_years(ax, years, SEVERE_YEARS)

        # Prepare data list
        data_list = [data_dict[y] for y in years]
        positions = np.arange(len(years))

        # Boxplot
        bp = ax.boxplot(
            data_list, positions=positions, widths=0.5,
            patch_artist=True, showfliers=True,
            flierprops=dict(marker="o", markersize=3, markerfacecolor="0.5",
                            markeredgecolor="0.5", alpha=0.4),
            medianprops=dict(color="white", lw=1.5),
            whiskerprops=dict(color="0.4", lw=1.0),
            capprops=dict(color="0.4", lw=1.0),
        )
        for box in bp["boxes"]:
            box.set_facecolor(color)
            box.set_alpha(0.6)
            box.set_edgecolor("0.3")
            box.set_linewidth(0.8)

        # Median trend line
        medians = [np.median(d) for d in data_list]
        ax.plot(positions, medians, "-", color=color, lw=1.5, alpha=0.5, zorder=1)

        ax.set_title(title, fontsize=14, fontweight="bold", pad=8)
        panel_label(ax, plbl)
        ax.set_xlabel("Year", fontsize=12)
        ax.set_ylabel("Tract-level mean threat perception", fontsize=12)
        ax.set_ylim(0.0, 1.05)
        ax.set_xticks(positions[::2])
        ax.set_xticklabels([str(y) for y in years[::2]])

    fig.tight_layout(w_pad=2.5)

    # Save — legacy visualization dir + paper Figure folder
    out_dir = BASE / "visualization" / "action"
    out_dir.mkdir(parents=True, exist_ok=True)
    paper_dir = Path(r"C:\Users\wenyu\OneDrive - Lehigh University\Desktop\Lehigh\NSF-project\ABM\paper\Figure")
    for ext in ["png", "pdf"]:
        fig.savefig(out_dir / f"tp_boxplot_by_year.{ext}", dpi=300, bbox_inches="tight")
        fig.savefig(paper_dir / f"tp_boxplot_by_year.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_dir / 'tp_boxplot_by_year.png'}")
    print(f"Saved: {paper_dir / 'tp_boxplot_by_year.png'}")


if __name__ == "__main__":
    main()
