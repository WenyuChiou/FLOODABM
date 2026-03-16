# -*- coding: utf-8 -*-
"""
2D Grid Sensitivity Analysis: decision_threshold (θ) × ratio_threshold
=======================================================================

9×9 = 81 combinations:
  - decision_threshold θ ∈ {0.1, 0.2, …, 0.9}
  - ratio_threshold   ∈ {0.1, 0.2, …, 0.9}  (symmetric: same for owner & renter)

Uses --output-mode summary to minimise disk usage (~1-2 MB per run).
θ = 0.0 is excluded because it is the baseline (pure stochastic).

Usage:
    python sa_decision_threshold.py              # run all 81 combinations
    python sa_decision_threshold.py --reuse      # skip runs with existing outputs
    python sa_decision_threshold.py --plot-only  # only regenerate plots
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── paths ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
SA_ROOT = Path("C:/temp/dt_rt_grid_sa")

DT_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]   # decision_threshold
RT_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]   # ratio_threshold


# ── helpers ────────────────────────────────────────────────────────────

def _run_tag(dt: float, rt: float) -> str:
    """DT01_RT01, DT01_RT02, …"""
    return f"DT{int(dt*10):02d}_RT{int(rt*10):02d}"


def _normalize_run_dir(run_dir: Path) -> Path:
    """Drill into nested baseline/ subdirectories."""
    for _ in range(5):
        b = run_dir / "baseline"
        if not b.exists():
            break
        if (b / "tp_traj.csv").exists() or (b / "finance").exists():
            run_dir = b
            continue
        if (b / "baseline").exists():
            run_dir = b
            continue
        break
    return run_dir


def _has_outputs(run_dir: Path) -> bool:
    rd = _normalize_run_dir(run_dir)
    return (rd / "tp_traj.csv").exists()


# ── simulation ─────────────────────────────────────────────────────────

def run_single(dt: float, rt: float, reuse: bool = False) -> Path:
    """Run one simulation with given (decision_threshold, ratio_threshold)."""
    tag = _run_tag(dt, rt)
    dst = SA_ROOT / tag
    if reuse and _has_outputs(dst):
        print(f"  [reuse] {tag}")
        return dst

    print(f"  [run]   {tag} — θ={dt}, RT={rt}")
    env = dict(os.environ)
    env["FLOODABM_OUTPUT_ROOT"] = str(dst)
    env["FLOODABM_OUTPUT_MODE"] = "summary"

    cmd = [
        sys.executable, "main.py",
        "--scenario", "baseline",
        "--decision-threshold", str(dt),
        "--thr-owner", str(rt),
        "--thr-renter", str(rt),
        "--output-mode", "summary",
    ]
    subprocess.check_call(cmd, cwd=str(ROOT), env=env)
    return dst


def run_all(reuse: bool = False) -> dict[tuple[float, float], Path]:
    """Run all 81 combinations. Returns {(dt, rt): run_dir}."""
    SA_ROOT.mkdir(parents=True, exist_ok=True)
    results = {}
    total = len(DT_VALUES) * len(RT_VALUES)
    done = 0
    for dt in DT_VALUES:
        for rt in RT_VALUES:
            results[(dt, rt)] = run_single(dt, rt, reuse=reuse)
            done += 1
            if done % 9 == 0:
                print(f"  progress: {done}/{total}")
    return results


# ── metric extraction (yearly time-series) ────────────────────────────

YEARS = list(range(2011, 2024))
SEVERE_YEARS = [2011, 2021]
BASELINE_DT = 0.1
BASELINE_RT = 0.1

# Extreme combos to highlight (mirrors RT grid SA style)
EXTREME_COMBOS = {
    (0.9, 0.9): (r"$\theta$=0.9, $\theta_r$=0.9", "#d62728"),   # red
    (0.9, 0.1): (r"$\theta$=0.9, $\theta_r$=0.1", "#2ca02c"),   # green
    (0.1, 0.9): (r"$\theta$=0.1, $\theta_r$=0.9", "#1f77b4"),   # blue
}


def load_scenario_data(run_dir: Path) -> pd.DataFrame:
    """Load finance + damage data for a single SA run (same logic as RT grid SA)."""
    rd = _normalize_run_dir(run_dir)

    fin_frames = []
    for year in YEARS:
        fp = rd / "finance" / f"finance_tract_{year}.csv"
        if fp.exists():
            df = pd.read_csv(fp, dtype={"tract_geoid": str})
            df["year"] = year
            fin_frames.append(df)
    if not fin_frames:
        return pd.DataFrame()

    fin = pd.concat(fin_frames, ignore_index=True)
    fin_yr = fin.groupby("year").agg(
        owner_gross_k=("owner_gross_total_kUSD", "sum"),
        renter_gross_k=("renter_gross_total_kUSD", "sum"),
        payout_total_k=("payout_total_kUSD", "sum"),
        owner_hh=("owner_households", "sum"),
        renter_hh=("renter_households", "sum"),
    ).reset_index()

    dmg_path = rd / "vulnerability" / "flood_damage" / "flood_damage_tract_ALL_years.csv"
    if dmg_path.exists():
        dmg = pd.read_csv(dmg_path, dtype={"tract_geoid": str})
        dmg_yr = dmg.groupby("year").agg(
            owner_dmg=("owner_usd", "sum"),
            renter_dmg=("renter_usd", "sum"),
        ).reset_index()
    else:
        dmg_yr = fin_yr[["year"]].copy()
        dmg_yr["owner_dmg"] = fin_yr["owner_gross_k"] * 1000
        dmg_yr["renter_dmg"] = fin_yr["renter_gross_k"] * 1000

    merged = fin_yr.merge(dmg_yr, on="year", how="outer").fillna(0).sort_values("year")

    total_gross = merged["owner_gross_k"] + merged["renter_gross_k"]
    owner_share = np.where(total_gross > 0, merged["owner_gross_k"] / total_gross, 0.5)
    merged["owner_payout"] = merged["payout_total_k"] * owner_share * 1000
    merged["renter_payout"] = merged["payout_total_k"] * (1 - owner_share) * 1000
    return merged


def compute_cumulative(df: pd.DataFrame, group: str):
    """Compute cumulative damage and actual loss per HH."""
    hh = np.maximum(df[f"{group}_hh"].values, 1)
    dmg = df[f"{group}_dmg"].values.astype(float)
    pay = df[f"{group}_payout"].values.astype(float)
    cum_dmg = np.cumsum(dmg / hh)
    cum_loss = np.cumsum((dmg - pay) / hh)
    return cum_dmg, cum_loss


# ── plotting: 2×2 Δ-from-baseline time-series ─────────────────────────

def plot_delta_timeseries(
    all_runs: dict[tuple[float, float], Path],
    out: Path,
):
    """
    2×2 figure showing Δ from baseline (DT=0.1, RT=0.1):
      (a) Homeowner — Change in Cumulative Damage per HH ($)
      (b) Renter    — Change in Cumulative Damage per HH ($)
      (c) Homeowner — Change in Cumulative Actual Loss per HH ($)
      (d) Renter    — Change in Cumulative Actual Loss per HH ($)

    Style matches Fig 9 (RT grid SA delta plot): gray spaghetti +
    3 colored extreme combos + flood year shading.
    """
    # Try importing shared style; fall back to sensible defaults
    try:
        sys.path.insert(0, str(ROOT))
        from utils.plots_modular.style import set_paper_style, panel_label, shade_severe_years
        set_paper_style()
        _has_style = True
    except ImportError:
        _has_style = False
        plt.rcParams.update({
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
            "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
            "legend.fontsize": 9, "axes.linewidth": 0.8,
            "axes.grid": True, "grid.alpha": 0.15, "grid.linestyle": "--",
        })

    # --- load all runs ---
    run_data = {}
    for (dt, rt), run_dir in all_runs.items():
        scenario = load_scenario_data(run_dir)
        if scenario.empty:
            continue
        o_dmg, o_loss = compute_cumulative(scenario, "owner")
        r_dmg, r_loss = compute_cumulative(scenario, "renter")
        run_data[(dt, rt)] = {
            "cum_owner_dmg": o_dmg,
            "cum_renter_dmg": r_dmg,
            "cum_owner_loss": o_loss,
            "cum_renter_loss": r_loss,
        }

    baseline_key = (BASELINE_DT, BASELINE_RT)
    if baseline_key not in run_data:
        print("  ERROR: baseline (0.1, 0.1) not found")
        return

    bl = run_data[baseline_key]
    x = np.arange(len(YEARS))

    # Compute deltas
    delta_data = {}
    for key, data in run_data.items():
        if key == baseline_key:
            continue
        delta_data[key] = {
            metric: data[metric] - bl[metric]
            for metric in ["cum_owner_dmg", "cum_renter_dmg", "cum_owner_loss", "cum_renter_loss"]
        }

    # --- figure ---
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    ax_a, ax_b = axes[0]
    ax_c, ax_d = axes[1]

    panels = [
        (ax_a, "cum_owner_dmg",  "Homeowner", "(a)"),
        (ax_b, "cum_renter_dmg", "Renter",    "(b)"),
        (ax_c, "cum_owner_loss", "Homeowner", "(c)"),
        (ax_d, "cum_renter_loss","Renter",    "(d)"),
    ]
    row_labels = {
        0: "Change in Cumulative Damage per HH",
        1: "Change in Cumulative Actual Loss per HH",
    }

    for idx, (ax, metric, col_title, plbl) in enumerate(panels):
        row = idx // 2

        # Severe year shading
        if _has_style:
            shade_severe_years(ax, YEARS, SEVERE_YEARS)
        else:
            for i, y in enumerate(YEARS):
                if y in SEVERE_YEARS:
                    ax.axvspan(i - 0.4, i + 0.4, facecolor="#cce5ff", alpha=0.5, zorder=0)

        # Zero line (baseline)
        ax.axhline(0, color="black", lw=2.0, zorder=5,
                    label=r"Baseline ($\theta$=0.1, $\theta_r$=0.1)")

        # Gray spaghetti for non-extreme runs
        for key, ddata in delta_data.items():
            if key in EXTREME_COMBOS:
                continue
            ax.plot(x, ddata[metric], color="#c0c0c0", lw=0.8, alpha=0.5, zorder=1)

        # Extreme combos (colored + labeled) — collect endpoints for anti-overlap
        endpoints = []
        for key, (label, color) in EXTREME_COMBOS.items():
            if key not in delta_data:
                continue
            vals = delta_data[key][metric]
            ax.plot(x, vals, "-o", ms=4, lw=2.0, color=color, label=label, zorder=8)
            end_val = vals[-1]
            if abs(end_val) < 1000:
                sign = "+" if end_val >= 0 else ""
                lbl = f"{sign}${end_val:.0f}"
            else:
                sign = "+" if end_val >= 0 else ""
                lbl = f"{sign}${end_val/1000:.0f}K"
            endpoints.append((end_val, lbl, color))

        # Sort endpoints by value and stagger if too close
        endpoints.sort(key=lambda t: t[0])
        MIN_GAP_PT = 18  # minimum vertical gap in offset points
        used_offsets = []
        for end_val, lbl, color in endpoints:
            y_off = 0
            # Check against previously placed annotations
            for prev_val, prev_off in used_offsets:
                # If endpoints are within ~2K of each other, stagger
                if abs(end_val - prev_val) < 3000:
                    y_off = prev_off + MIN_GAP_PT
            used_offsets.append((end_val, y_off))
            ax.annotate(
                lbl, xy=(x[-1], end_val),
                xytext=(8, y_off), textcoords="offset points",
                fontsize=11, fontweight="bold", color=color,
                ha="left", va="center", zorder=9,
            )

        ax.set_title(col_title, fontsize=14, fontweight="bold", pad=8)
        if _has_style:
            panel_label(ax, plbl)
        else:
            ax.text(-0.08, 1.05, plbl, transform=ax.transAxes, fontsize=12,
                    fontweight="bold", ha="left", va="bottom")

        ax.set_xlabel("Year", fontsize=12)
        ax.set_ylabel(f"{row_labels[row]} ($)", fontsize=11)
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, p: "$0" if v == 0 else f"${v/1000:+.0f}K"))
        ax.set_xticks(x[::2])
        ax.set_xticklabels([str(y) for y in YEARS[::2]])

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        legend_loc = "lower left" if row == 0 else "upper left"
        ax.legend(
            loc=legend_loc, fontsize=10, frameon=True,
            facecolor="white", edgecolor="0.8", framealpha=0.95,
            borderpad=0.5, labelspacing=0.3,
        )

    fig.tight_layout(h_pad=2.5, w_pad=2.0)
    fig.subplots_adjust(right=0.92)

    fig.savefig(str(out) + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(str(out) + ".pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out.stem}.png/.pdf")


# ── main ───────────────────────────────────────────────────────────────

def main():
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="2D grid SA: decision_threshold × ratio_threshold (9×9=81)")
    parser.add_argument("--reuse", action="store_true",
                        help="Skip runs whose outputs already exist")
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip simulations, only regenerate plots")
    args = parser.parse_args()

    n_total = len(DT_VALUES) * len(RT_VALUES)
    print("=" * 60)
    print("2D Grid SA: decision_threshold (θ) × ratio_threshold")
    print(f"  θ  values: {DT_VALUES}")
    print(f"  RT values: {RT_VALUES}")
    print(f"  total runs: {n_total}")
    print(f"  output dir: {SA_ROOT}")
    print(f"  output mode: summary (minimal disk usage)")
    print("=" * 60)

    # ── 1. Run simulations ──
    if args.plot_only:
        print("\n[plot-only mode] Locating existing outputs …")
        runs = {}
        for dt in DT_VALUES:
            for rt in RT_VALUES:
                d = SA_ROOT / _run_tag(dt, rt)
                if _has_outputs(d):
                    runs[(dt, rt)] = d
                else:
                    print(f"  [MISS] {_run_tag(dt, rt)}")
        if not runs:
            print("ERROR: no outputs found. Run without --plot-only first.")
            sys.exit(1)
        print(f"  found {len(runs)}/{n_total} runs")
    else:
        print(f"\n[1/2] Running {n_total} simulations …")
        runs = run_all(reuse=args.reuse)

    # ── 2. Generate Δ-from-baseline time-series figure ──
    print("\n[2/2] Generating Δ cumulative time-series figure …")
    plot_delta_timeseries(runs, SA_ROOT / "sa_dt_rt_grid_delta_2x2")

    # Estimate disk usage
    total_size = sum(f.stat().st_size for f in SA_ROOT.rglob("*") if f.is_file())
    print(f"\n  Total disk usage: {total_size / 1e6:.1f} MB")

    print("\n" + "=" * 60)
    print("Done!  All outputs in:")
    print(f"  {SA_ROOT}")
    print("=" * 60)


if __name__ == "__main__":
    main()
