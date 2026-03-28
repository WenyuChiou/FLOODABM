# -*- coding: utf-8 -*-
"""
RT_owner × RT_renter Grid Sensitivity Analysis
================================================

Asymmetric 9×9 sweep of ratio_threshold for owner and renter independently.
Produces 81 runs and a 2×2 Δ-from-baseline time-series figure.

Usage:
    python sa_rt_grid.py              # run all 25 combinations
    python sa_rt_grid.py --reuse      # skip runs with existing outputs
    python sa_rt_grid.py --plot-only  # skip simulation, only re-plot
"""
from __future__ import annotations

import argparse
import itertools
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── paths ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent  # project root
SA_ROOT = Path("C:/temp/rt_grid_sa")

RT_VALUES = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
BASELINE_RT_O = 0.10
BASELINE_RT_R = 0.10
YEARS = list(range(2011, 2024))

# Severe flood years (for gray bands in figure)
SEVERE_YEARS = [2011, 2012, 2014, 2021]


# ── helpers ────────────────────────────────────────────────────────────

def _run_tag(rt_o: float, rt_r: float) -> str:
    return f"RTo{rt_o:.2f}_RTr{rt_r:.2f}".replace(".", "")


def _normalize_run_dir(run_dir: Path) -> Path:
    for _ in range(5):
        b = run_dir / "baseline"
        if not b.exists():
            break
        if (b / "finance").exists() or (b / "decisions").exists():
            run_dir = b
            continue
        if (b / "baseline").exists():
            run_dir = b
            continue
        break
    return run_dir


def _has_outputs(run_dir: Path) -> bool:
    rd = _normalize_run_dir(run_dir)
    return (rd / "finance").exists() and len(list((rd / "finance").glob("*.csv"))) >= 13


# ── simulation ─────────────────────────────────────────────────────────

def run_single(rt_o: float, rt_r: float, reuse: bool = False) -> Path:
    tag = _run_tag(rt_o, rt_r)
    dst = SA_ROOT / tag
    if reuse and _has_outputs(dst):
        print(f"  [reuse] {tag}")
        return dst

    print(f"  [run]   {tag} — thr_owner={rt_o}, thr_renter={rt_r}")
    env = dict(os.environ)
    env["FLOODABM_OUTPUT_ROOT"] = str(dst)
    env["FLOODABM_OUTPUT_MODE"] = "summary"

    cmd = [
        sys.executable, "main.py",
        "--scenario", "baseline",
        "--thr-owner", str(rt_o),
        "--thr-renter", str(rt_r),
        "--output-mode", "summary",
    ]
    subprocess.check_call(cmd, cwd=str(ROOT), env=env)
    return dst


def run_all(reuse: bool = False) -> dict[tuple[float, float], Path]:
    SA_ROOT.mkdir(parents=True, exist_ok=True)
    results = {}
    combos = list(itertools.product(RT_VALUES, RT_VALUES))
    print(f"\nTotal runs: {len(combos)}")
    for i, (rt_o, rt_r) in enumerate(combos, 1):
        print(f"\n[{i}/{len(combos)}]")
        results[(rt_o, rt_r)] = run_single(rt_o, rt_r, reuse=reuse)
    return results


# ── metric extraction (yearly time series) ─────────────────────────────

def extract_yearly_damage(run_dir: Path) -> pd.DataFrame:
    """Extract yearly per-HH damage by tenure from finance CSVs."""
    rd = _normalize_run_dir(run_dir)
    records = []
    for year in YEARS:
        fp = rd / "finance" / f"finance_tract_{year}.csv"
        if not fp.exists():
            continue
        df = pd.read_csv(fp, encoding="utf-8")
        # Aggregate across tracts
        owner_damage = df["owner_gross_total_usd"].sum() if "owner_gross_total_usd" in df.columns else 0
        renter_damage = df["renter_gross_total_usd"].sum() if "renter_gross_total_usd" in df.columns else 0
        owner_hh = df["owner_households"].sum() if "owner_households" in df.columns else 1
        renter_hh = df["renter_households"].sum() if "renter_households" in df.columns else 1
        payout = df["payout_total_usd"].sum() if "payout_total_usd" in df.columns else 0
        gross = df["gross_total_usd"].sum() if "gross_total_usd" in df.columns else 1

        records.append({
            "year": year,
            "owner_damage_per_hh": owner_damage / max(owner_hh, 1),
            "renter_damage_per_hh": renter_damage / max(renter_hh, 1),
            "payout_rate": payout / max(gross, 1),
        })
    return pd.DataFrame(records)


def extract_yearly_actions(run_dir: Path) -> pd.DataFrame:
    """Extract yearly population-weighted action shares."""
    rd = _normalize_run_dir(run_dir)
    records = []
    for year in YEARS:
        fp = rd / "decisions" / f"action_share_owner_renter_tract_{year}.csv"
        if not fp.exists():
            continue
        df = pd.read_csv(fp, encoding="utf-8")
        row = {"year": year}
        ow = df["owner_N_total"].sum()
        rw = df["renter_N_total"].sum()
        if ow > 0:
            for col in ["owner_share_FI", "owner_share_EH", "owner_share_BP", "owner_share_DN"]:
                if col in df.columns:
                    row[col] = (df[col] * df["owner_N_total"]).sum() / ow
        if rw > 0:
            for col in ["renter_share_FI", "renter_share_RL", "renter_share_DN"]:
                if col in df.columns:
                    row[col] = (df[col] * df["renter_N_total"]).sum() / rw
        records.append(row)
    return pd.DataFrame(records)


# ── data loader (same logic as plot_fig4) ─────────────────────────────

def load_scenario_data(run_dir: Path) -> pd.DataFrame:
    """Load finance + damage data for a single SA run, matching Fig 4 logic."""
    rd = _normalize_run_dir(run_dir)

    # Finance CSVs
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

    # Flood damage CSVs (more accurate per-tenure damage)
    dmg_path = rd / "vulnerability" / "flood_damage" / "flood_damage_tract_ALL_years.csv"
    if dmg_path.exists():
        dmg = pd.read_csv(dmg_path, dtype={"tract_geoid": str})
        dmg_yr = dmg.groupby("year").agg(
            owner_dmg=("owner_usd", "sum"),
            renter_dmg=("renter_usd", "sum"),
        ).reset_index()
    else:
        # Fallback: use finance gross (in kUSD → USD)
        dmg_yr = fin_yr[["year"]].copy()
        dmg_yr["owner_dmg"] = fin_yr["owner_gross_k"] * 1000
        dmg_yr["renter_dmg"] = fin_yr["renter_gross_k"] * 1000

    merged = fin_yr.merge(dmg_yr, on="year", how="outer").fillna(0).sort_values("year")

    # Proportional payout split (same as Fig 4)
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


# ── plotting: 2×2 cumulative damage + actual loss ─────────────────────

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 9, "axes.linewidth": 0.8,
    "axes.grid": True, "grid.alpha": 0.15, "grid.linestyle": "--",
    "axes.spines.top": False,
})


def plot_delta_cumulative(
    all_runs: dict[tuple[float, float], Path],
    out: Path,
):
    """
    2×2 figure (matching Fig 4 metrics):
      (a) Homeowners — Cumulative Damage per HH ($)
      (b) Renters    — Cumulative Damage per HH ($)
      (c) Homeowners — Cumulative Actual Loss per HH ($)
      (d) Renters    — Cumulative Actual Loss per HH ($)

    Each panel shows 25 curves (one per RT combo). Baseline highlighted.
    """
    # --- load all runs ---
    run_data = {}
    for (rt_o, rt_r), run_dir in all_runs.items():
        scenario = load_scenario_data(run_dir)
        if scenario.empty:
            continue
        o_dmg, o_loss = compute_cumulative(scenario, "owner")
        r_dmg, r_loss = compute_cumulative(scenario, "renter")
        run_data[(rt_o, rt_r)] = {
            "cum_owner_dmg": o_dmg,
            "cum_renter_dmg": r_dmg,
            "cum_owner_loss": o_loss,
            "cum_renter_loss": r_loss,
        }

    if not run_data:
        print("ERROR: no valid runs")
        return

    x = np.arange(len(YEARS))
    baseline_key = (BASELINE_RT_O, BASELINE_RT_R)

    # --- figure ---
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax_a, ax_b = axes[0]
    ax_c, ax_d = axes[1]

    panels = [
        (ax_a, "cum_owner_dmg",  "Homeowner — Cumulative Damage per HH", "(a)"),
        (ax_b, "cum_renter_dmg", "Renter — Cumulative Damage per HH",    "(b)"),
        (ax_c, "cum_owner_loss", "Homeowner — Cumulative Actual Loss per HH", "(c)"),
        (ax_d, "cum_renter_loss","Renter — Cumulative Actual Loss per HH",    "(d)"),
    ]

    for ax, key, title, plbl in panels:
        # Severe year shading
        for i, y in enumerate(YEARS):
            if y in SEVERE_YEARS:
                ax.axvspan(i - 0.4, i + 0.4, facecolor="#fff3cd", alpha=0.6, zorder=0)

        # Gray spaghetti for non-baseline runs
        for (rt_o, rt_r), data in run_data.items():
            if (rt_o, rt_r) == baseline_key:
                continue
            ax.plot(x, data[key], color="#b0b0b0", lw=0.7, alpha=0.5, zorder=1)

        # Baseline (bold black)
        if baseline_key in run_data:
            bl = run_data[baseline_key][key]
            ax.plot(x, bl, "-o", ms=4, lw=2.5, color="black",
                    label=f"Baseline (RT = {BASELINE_RT_O:.2f})", zorder=10)
            # Endpoint annotation
            ax.annotate(f"${bl[-1]/1000:.0f}K", xy=(x[-1], bl[-1]),
                        xytext=(x[-1] - 1.5, bl[-1] * 1.08),
                        fontsize=9, fontweight="bold", color="black",
                        ha="center", va="bottom", zorder=11,
                        bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                                  edgecolor="none", alpha=0.8))

        ax.set_title(title, pad=8)
        ax.text(-0.08, 1.05, plbl, transform=ax.transAxes, fontsize=12,
                fontweight="bold", ha="left", va="bottom")
        ax.set_xlabel("Year")
        ax.set_ylabel("USD per household")
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, p: f"${v/1000:.0f}K"))
        ax.set_xticks(x[::2])
        ax.set_xticklabels([str(y) for y in YEARS[::2]])

    # Legend only on (a)
    ax_a.legend(loc="upper left", fontsize=9, frameon=True, facecolor="white",
                edgecolor="black", framealpha=1.0)

    # Add annotation for spread
    ax_a.annotate("81 RT combinations\n(gray lines)", xy=(0.95, 0.25),
                  xycoords="axes fraction", fontsize=8, color="#666666",
                  ha="right", va="center",
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="#f5f5f5",
                            edgecolor="#cccccc", alpha=0.9))

    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out.stem}.png/.pdf")


# ── main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RT_owner × RT_renter grid SA")
    parser.add_argument("--reuse", action="store_true",
                        help="Skip runs whose outputs already exist")
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip simulations, only regenerate plots")
    args = parser.parse_args()

    combos = list(itertools.product(RT_VALUES, RT_VALUES))
    print("=" * 60)
    print("RT_owner × RT_renter Grid Sensitivity Analysis")
    print(f"  RT values : {RT_VALUES}")
    print(f"  Total runs: {len(combos)}")
    print(f"  Baseline  : RT_o={BASELINE_RT_O}, RT_r={BASELINE_RT_R}")
    print(f"  Output    : {SA_ROOT}")
    print("=" * 60)

    # ── 1. Run simulations ──
    if args.plot_only:
        print("\n[plot-only mode] Locating existing outputs …")
        runs = {}
        for rt_o, rt_r in combos:
            d = SA_ROOT / _run_tag(rt_o, rt_r)
            if _has_outputs(d):
                runs[(rt_o, rt_r)] = d
            else:
                print(f"  [MISS] {_run_tag(rt_o, rt_r)}")
        if not runs:
            print("ERROR: no outputs found.")
            sys.exit(1)
        print(f"  Found {len(runs)}/{len(combos)} runs")
    else:
        print("\n[1/2] Running simulations …")
        runs = run_all(reuse=args.reuse)

    # ── 2. Generate figure ──
    print("\n[2/2] Generating Δ cumulative figure …")
    SA_ROOT.mkdir(parents=True, exist_ok=True)
    plot_delta_cumulative(runs, SA_ROOT / "sa_rt_grid_2x2")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
