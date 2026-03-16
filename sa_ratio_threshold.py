# -*- coding: utf-8 -*-
"""
Ratio-Threshold Sensitivity Analysis
=====================================

Symmetric sweep of ratio_threshold (applied to both owner and renter)
to analyse how the flood-damage trigger affects TP shocks, TP trajectory
shape, and action adoption rates.

Usage:
    python sa_ratio_threshold.py              # run all 7 thresholds
    python sa_ratio_threshold.py --reuse      # skip runs that already have outputs
    python sa_ratio_threshold.py --plot-only  # skip simulation, just re-plot
"""
from __future__ import annotations

import argparse
import sys
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize

# ── paths ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
SA_ROOT = ROOT / "outputs" / "experiments" / "ratio_threshold_sa"

THRESHOLDS = [0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]


# ── helpers ────────────────────────────────────────────────────────────

def _run_tag(val: float) -> str:
    """RT_0_01, RT_0_05, …"""
    return f"RT_{val:.2f}".replace(".", "_")


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
    """Check for complete outputs (must have tp_traj.csv from full mode)."""
    run_dir = _normalize_run_dir(run_dir)
    return (run_dir / "tp_traj.csv").exists()


# ── simulation ─────────────────────────────────────────────────────────

def run_single(thr: float, reuse: bool = False) -> Path:
    """Run one simulation for a given threshold value."""
    tag = _run_tag(thr)
    dst = SA_ROOT / tag
    if reuse and _has_outputs(dst):
        print(f"  [reuse] {tag} — outputs exist, skipping")
        return dst

    print(f"  [run]   {tag} — thr_owner={thr}, thr_renter={thr}")
    import os
    env = dict(os.environ)
    env["FLOODABM_OUTPUT_ROOT"] = str(dst)
    env["FLOODABM_OUTPUT_MODE"] = "summary"

    cmd = [
        sys.executable, "main.py",
        "--scenario", "baseline",
        "--thr-owner", str(thr),
        "--thr-renter", str(thr),
        "--output-mode", "summary",
    ]
    subprocess.check_call(cmd, cwd=str(ROOT), env=env)
    return dst


def run_all(reuse: bool = False) -> dict[float, Path]:
    """Run simulations for all threshold values. Returns {thr: run_dir}."""
    SA_ROOT.mkdir(parents=True, exist_ok=True)
    results = {}
    for thr in THRESHOLDS:
        results[thr] = run_single(thr, reuse=reuse)
    return results


# ── metric extraction ──────────────────────────────────────────────────

def extract_tp(run_dir: Path) -> pd.DataFrame:
    """Load tp_traj.csv → DataFrame[year, tract_geoid, TP_owner, TP_renter]."""
    rd = _normalize_run_dir(run_dir)
    return pd.read_csv(rd / "tp_traj.csv", encoding="utf-8")


def extract_flood_triggers(run_dir: Path) -> pd.DataFrame:
    """Load flood_years_by_tract.csv."""
    rd = _normalize_run_dir(run_dir)
    return pd.read_csv(rd / "flood_years_by_tract.csv", encoding="utf-8")


def extract_actions(run_dir: Path) -> pd.DataFrame:
    """Load the all-years action share CSV."""
    rd = _normalize_run_dir(run_dir)
    p = rd / "decisions" / "action_share_owner_renter_tract_all_years.csv"
    if not p.exists():
        # Try individual year files and concat
        files = sorted((rd / "decisions").glob("action_share_owner_renter_tract_2*.csv"))
        if not files:
            return pd.DataFrame()
        return pd.concat([pd.read_csv(f, encoding="utf-8") for f in files], ignore_index=True)
    return pd.read_csv(p, encoding="utf-8")


def summarise_run(thr: float, run_dir: Path) -> dict:
    """Extract summary metrics for one run."""
    tp = extract_tp(run_dir)
    ft = extract_flood_triggers(run_dir)
    act = extract_actions(run_dir)

    # TP summary (mean across tracts per year)
    tp_yearly = tp.groupby("year")[["TP_owner", "TP_renter"]].mean()
    final_year = tp_yearly.index.max()
    tp_final_owner = tp_yearly.loc[final_year, "TP_owner"]
    tp_final_renter = tp_yearly.loc[final_year, "TP_renter"]

    # Count years with TP increase (shock events)
    tp_diff_owner = tp_yearly["TP_owner"].diff()
    n_shock_years_owner = int((tp_diff_owner > 0).sum())
    tp_diff_renter = tp_yearly["TP_renter"].diff()
    n_shock_years_renter = int((tp_diff_renter > 0).sum())

    # Flood trigger counts
    n_tract_year_shocks = int(ft["is_shock"].sum()) if "is_shock" in ft.columns else 0
    shock_years = ft.loc[ft["is_shock"] == True, "year"].unique() if "is_shock" in ft.columns else []
    n_distinct_shock_years = len(shock_years)

    # Action shares at final year (population-weighted)
    row = {"threshold": thr,
           "tp_final_owner": round(tp_final_owner, 4),
           "tp_final_renter": round(tp_final_renter, 4),
           "n_shock_years_owner": n_shock_years_owner,
           "n_shock_years_renter": n_shock_years_renter,
           "n_tract_year_shocks": n_tract_year_shocks,
           "n_distinct_shock_years": n_distinct_shock_years}

    if not act.empty:
        a23 = act[act["year"] == final_year]
        if not a23.empty:
            # Population-weighted mean shares
            ow = a23["owner_N_total"].sum()
            if ow > 0:
                for col in ["owner_share_FI", "owner_share_EH", "owner_share_BP", "owner_share_DN"]:
                    if col in a23.columns:
                        row[col] = round((a23[col] * a23["owner_N_total"]).sum() / ow, 4)
            rw = a23["renter_N_total"].sum()
            if rw > 0:
                for col in ["renter_share_FI", "renter_share_RL", "renter_share_DN"]:
                    if col in a23.columns:
                        row[col] = round((a23[col] * a23["renter_N_total"]).sum() / rw, 4)

    return row


# ── plotting ───────────────────────────────────────────────────────────

def plot_tp_trajectory(runs: dict[float, Path], out: Path):
    """TP trajectory overlay — all thresholds on one plot (owner + renter)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    cmap = plt.cm.viridis
    norm = Normalize(vmin=min(THRESHOLDS), vmax=max(THRESHOLDS))

    for thr, run_dir in sorted(runs.items()):
        tp = extract_tp(run_dir)
        tp_yr = tp.groupby("year")[["TP_owner", "TP_renter"]].mean()
        color = cmap(norm(thr))
        axes[0].plot(tp_yr.index, tp_yr["TP_owner"], marker="o", ms=3,
                     color=color, label=f"{thr:.2f}")
        axes[1].plot(tp_yr.index, tp_yr["TP_renter"], marker="o", ms=3,
                     color=color, label=f"{thr:.2f}")

    for ax, title in zip(axes, ["Owner TP", "Renter TP"]):
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Year")
        ax.set_ylabel("Mean TP (across tracts)")
        ax.legend(title="Threshold", fontsize=8, title_fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("TP Trajectory by ratio_threshold", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out.name}")


def plot_flood_trigger_heatmap(runs: dict[float, Path], out: Path):
    """Heatmap: number of tracts shocked per year for each threshold."""
    records = []
    for thr, run_dir in sorted(runs.items()):
        ft = extract_flood_triggers(run_dir)
        if "is_shock" not in ft.columns:
            continue
        shock_by_year = ft[ft["is_shock"] == True].groupby("year").size()
        for yr, n in shock_by_year.items():
            records.append({"threshold": thr, "year": yr, "n_tracts_shocked": n})

    if not records:
        print("  [skip] flood_trigger_heatmap — no shock data")
        return

    df = pd.DataFrame(records)
    pivot = df.pivot_table(index="threshold", columns="year",
                           values="n_tracts_shocked", fill_value=0)
    # Ensure all years present
    all_years = sorted(set(df["year"]))
    for y in range(min(all_years), max(all_years) + 1):
        if y not in pivot.columns:
            pivot[y] = 0
    pivot = pivot[sorted(pivot.columns)]

    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd",
                   interpolation="nearest")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{t:.2f}" for t in pivot.index])
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(y) for y in pivot.columns], rotation=45, ha="right")
    ax.set_ylabel("ratio_threshold")
    ax.set_xlabel("Year")
    ax.set_title("Number of Tracts with TP Shock per Year", fontweight="bold")

    # Annotate cells
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = int(pivot.values[i, j])
            if v > 0:
                ax.text(j, i, str(v), ha="center", va="center",
                        fontsize=8, color="black" if v < pivot.values.max() / 2 else "white")

    fig.colorbar(im, ax=ax, label="# tracts shocked")
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out.name}")


def plot_action_share_comparison(runs: dict[float, Path], out: Path):
    """Bar chart: final-year action shares across thresholds."""
    records = []
    for thr, run_dir in sorted(runs.items()):
        act = extract_actions(run_dir)
        if act.empty:
            continue
        final_year = act["year"].max()
        a = act[act["year"] == final_year]
        ow = a["owner_N_total"].sum()
        rw = a["renter_N_total"].sum()
        row = {"threshold": f"{thr:.2f}"}
        if ow > 0:
            for c in ["owner_share_FI", "owner_share_EH", "owner_share_BP", "owner_share_DN"]:
                if c in a.columns:
                    row[c.replace("owner_share_", "O_")] = (a[c] * a["owner_N_total"]).sum() / ow
        if rw > 0:
            for c in ["renter_share_FI", "renter_share_RL", "renter_share_DN"]:
                if c in a.columns:
                    row[c.replace("renter_share_", "R_")] = (a[c] * a["renter_N_total"]).sum() / rw
        records.append(row)

    if not records:
        print("  [skip] action_share_comparison — no data")
        return

    df = pd.DataFrame(records).set_index("threshold")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Owner actions
    o_cols = [c for c in df.columns if c.startswith("O_")]
    if o_cols:
        df[o_cols].plot(kind="bar", stacked=True, ax=axes[0],
                        color=["#2196F3", "#4CAF50", "#FF9800", "#9E9E9E"])
        axes[0].set_title("Owner Action Shares (Final Year)", fontweight="bold")
        axes[0].set_ylabel("Share")
        axes[0].legend(title="Action", fontsize=8)
        axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)

    # Renter actions
    r_cols = [c for c in df.columns if c.startswith("R_")]
    if r_cols:
        df[r_cols].plot(kind="bar", stacked=True, ax=axes[1],
                        color=["#2196F3", "#E91E63", "#9E9E9E"])
        axes[1].set_title("Renter Action Shares (Final Year)", fontweight="bold")
        axes[1].set_ylabel("Share")
        axes[1].legend(title="Action", fontsize=8)
        axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)

    fig.suptitle("Action Share Comparison by ratio_threshold", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out.name}")


def plot_combined_dashboard(runs: dict[float, Path], summary_df: pd.DataFrame, out: Path):
    """4-panel combined dashboard."""
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # Panel 1: TP trajectory (owner)
    ax1 = fig.add_subplot(gs[0, 0])
    cmap = plt.cm.viridis
    norm = Normalize(vmin=min(THRESHOLDS), vmax=max(THRESHOLDS))
    for thr, run_dir in sorted(runs.items()):
        tp = extract_tp(run_dir)
        tp_yr = tp.groupby("year")["TP_owner"].mean()
        ax1.plot(tp_yr.index, tp_yr.values, marker="o", ms=3,
                 color=cmap(norm(thr)), label=f"{thr:.2f}")
    ax1.set_title("Owner TP Trajectory", fontweight="bold")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Mean TP")
    ax1.legend(title="Threshold", fontsize=7, title_fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)

    # Panel 2: TP trajectory (renter)
    ax2 = fig.add_subplot(gs[0, 1])
    for thr, run_dir in sorted(runs.items()):
        tp = extract_tp(run_dir)
        tp_yr = tp.groupby("year")["TP_renter"].mean()
        ax2.plot(tp_yr.index, tp_yr.values, marker="o", ms=3,
                 color=cmap(norm(thr)), label=f"{thr:.2f}")
    ax2.set_title("Renter TP Trajectory", fontweight="bold")
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Mean TP")
    ax2.legend(title="Threshold", fontsize=7, title_fontsize=8, ncol=2)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Summary metrics (shock counts)
    ax3 = fig.add_subplot(gs[1, 0])
    x = np.arange(len(summary_df))
    w = 0.25
    ax3.bar(x - w, summary_df["n_distinct_shock_years"], width=w,
            label="Distinct shock years", color="#2196F3")
    ax3.bar(x, summary_df["n_shock_years_owner"], width=w,
            label="Owner TP-increase years", color="#4CAF50")
    ax3.bar(x + w, summary_df["n_shock_years_renter"], width=w,
            label="Renter TP-increase years", color="#FF9800")
    ax3.set_xticks(x)
    ax3.set_xticklabels([f"{t:.2f}" for t in summary_df["threshold"]])
    ax3.set_xlabel("ratio_threshold")
    ax3.set_ylabel("Count")
    ax3.set_title("Shock Events by Threshold", fontweight="bold")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3, axis="y")

    # Panel 4: Final TP values
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(summary_df["threshold"], summary_df["tp_final_owner"],
             "o-", label="Owner", color="#2196F3", linewidth=2)
    ax4.plot(summary_df["threshold"], summary_df["tp_final_renter"],
             "s-", label="Renter", color="#FF9800", linewidth=2)
    ax4.set_xlabel("ratio_threshold")
    ax4.set_ylabel("Final TP (2023)")
    ax4.set_title("Final-Year TP vs Threshold", fontweight="bold")
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    fig.suptitle("ratio_threshold Sensitivity Analysis Dashboard",
                 fontsize=16, fontweight="bold", y=0.98)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out.name}")


# ── main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ratio_threshold sensitivity analysis")
    parser.add_argument("--reuse", action="store_true",
                        help="Skip runs whose outputs already exist")
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip simulations, only regenerate plots from existing outputs")
    args = parser.parse_args()

    print("=" * 60)
    print("ratio_threshold Sensitivity Analysis")
    print(f"  thresholds: {THRESHOLDS}")
    print(f"  output dir: {SA_ROOT}")
    print("=" * 60)

    # ── 1. Run simulations ──
    if args.plot_only:
        print("\n[plot-only mode] Locating existing outputs …")
        runs = {}
        for thr in THRESHOLDS:
            d = SA_ROOT / _run_tag(thr)
            if _has_outputs(d):
                runs[thr] = d
            else:
                print(f"  [MISS] {_run_tag(thr)} — no outputs found, skipping")
        if not runs:
            print("ERROR: no outputs found. Run without --plot-only first.")
            sys.exit(1)
    else:
        print("\n[1/3] Running simulations …")
        runs = run_all(reuse=args.reuse)

    # ── 2. Aggregate metrics ──
    print("\n[2/3] Extracting metrics …")
    rows = []
    for thr in sorted(runs):
        print(f"  extracting {_run_tag(thr)} …")
        rows.append(summarise_run(thr, runs[thr]))
    summary = pd.DataFrame(rows)
    summary.to_csv(SA_ROOT / "sa_summary.csv", index=False, encoding="utf-8")
    print(f"  saved → sa_summary.csv  ({len(summary)} rows)")
    print("\n  Summary table:")
    print(summary.to_string(index=False))

    # ── 3. Generate plots ──
    print("\n[3/3] Generating plots …")
    plot_tp_trajectory(runs, SA_ROOT / "tp_trajectory_comparison.png")
    plot_flood_trigger_heatmap(runs, SA_ROOT / "flood_trigger_heatmap.png")
    plot_action_share_comparison(runs, SA_ROOT / "action_share_comparison.png")
    plot_combined_dashboard(runs, summary, SA_ROOT / "sa_ratio_threshold_report.png")

    print("\n" + "=" * 60)
    print("Done!  All outputs in:")
    print(f"  {SA_ROOT}")
    print("=" * 60)


if __name__ == "__main__":
    main()
