# -*- coding: utf-8 -*-
"""
Sensitivity Analysis Visualization for FLOODABM
=================================================

Generates publication-quality figures from SA results:
  Figure SA-1: Tornado Plot (main text)
  Figure SA-2: TP Decay Time-Series (SI)
  Figure SA-3: RT × TP Decay Heatmap (main text or SI)
  Figure SA-4: SA Summary Table (SI)

Usage:
    python plot_sa_results.py                  # generate all figures
    python plot_sa_results.py --fig tornado    # single figure
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize

sys.stdout.reconfigure(encoding="utf-8")

# Add project root to path for imports
ROOT = Path(__file__).resolve().parent.parent.parent  # project root
sys.path.insert(0, str(ROOT))

from utils.plots_modular.style import (
    set_paper_style, panel_label, shade_severe_years,
    COLORS, COLOR_OWNER, COLOR_RENTER,
)

# ── paths ────────────────────────────────────────────────────────────────
SA_OUT = ROOT / "outputs" / "experiments" / "sensitivity_analysis"
RT_SA_SUMMARY = (ROOT / "outputs" / "experiments" /
                 "ratio_threshold_sa" / "sa_summary.csv")
FIG_DIR = SA_OUT / "figures"

SEVERE_YEARS = [2011, 2021]
YEARS = list(range(2011, 2024))

# ── data loading ─────────────────────────────────────────────────────────

def load_sa_metrics() -> pd.DataFrame:
    """Load the unified SA metrics CSV."""
    p = SA_OUT / "sa_all_metrics.csv"
    if not p.exists():
        print(f"[ERROR] {p} not found — run run_sensitivity_analysis.py first")
        sys.exit(1)
    return pd.read_csv(p, encoding="utf-8-sig")


def load_rt_summary() -> pd.DataFrame | None:
    """Load existing RT SA summary from sa_ratio_threshold.py outputs."""
    if RT_SA_SUMMARY.exists():
        return pd.read_csv(RT_SA_SUMMARY, encoding="utf-8")
    return None


# ── Figure SA-1: Tornado Plot ────────────────────────────────────────────

def _pct_change(series: pd.Series, baseline_val: float) -> tuple[float, float]:
    """Compute (min%, max%) change relative to baseline.

    Returns percentage change, e.g. (-30.5, +15.2).
    """
    if baseline_val == 0:
        return (0.0, 0.0)
    return (
        (series.min() - baseline_val) / baseline_val * 100,
        (series.max() - baseline_val) / baseline_val * 100,
    )


def plot_tornado(df: pd.DataFrame, rt_df: pd.DataFrame | None) -> None:
    """Horizontal tornado plot showing % change from baseline per SA parameter.

    All bars are normalised to percentage change from each parameter's own
    baseline, resolving the discrepancy between the RT SA and SA1-SA3 pipelines.

    3 panels: (a) Owner FI % Change, (b) Renter FI % Change,
              (c) Cumulative Payout % Change
    Y-axis: 4 parameters
    """
    set_paper_style()
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)

    # Collect normalised % change ranges for each SA
    params = []
    owner_pct = []   # (lo%, hi%) relative to own baseline
    renter_pct = []
    payout_pct = []

    # --- RT (from existing sa_summary.csv) ---
    if rt_df is not None and not rt_df.empty:
        # Baseline = threshold closest to 0.10
        rt_bl = rt_df.loc[(rt_df["threshold"] - 0.10).abs().idxmin()]
        params.append("Ratio Threshold\n(RT)")
        owner_pct.append(_pct_change(rt_df["owner_share_FI"],
                                     rt_bl["owner_share_FI"]))
        renter_pct.append(_pct_change(rt_df["renter_share_FI"],
                                      rt_bl["renter_share_FI"]))
        payout_pct.append((np.nan, np.nan))  # No payout in RT summary

    # --- SA1: TP Decay ---
    sa1 = df[df["sa"] == "sa1_tp_decay"].copy()
    if not sa1.empty and "owner_share_FI" in sa1.columns:
        sa1["value"] = pd.to_numeric(sa1["value"], errors="coerce")
        bl = sa1.loc[(sa1["value"] - 1.0).abs().idxmin()]
        params.append("TP Decay Rate\n(τ∞)")
        owner_pct.append(_pct_change(sa1["owner_share_FI"],
                                     bl["owner_share_FI"]))
        renter_pct.append(_pct_change(sa1["renter_share_FI"],
                                      bl["renter_share_FI"]))
        if "cumulative_payout_usd" in sa1.columns:
            payout_pct.append(_pct_change(sa1["cumulative_payout_usd"],
                                          bl["cumulative_payout_usd"]))
        else:
            payout_pct.append((np.nan, np.nan))

    # --- SA2: FI Rates ---
    sa2 = df[df["sa"] == "sa2_fi_rates"].copy()
    if not sa2.empty and "owner_share_FI" in sa2.columns:
        sa2["value"] = pd.to_numeric(sa2["value"], errors="coerce")
        bl = sa2.loc[(sa2["value"] - 1.0).abs().idxmin()]
        params.append("Initial FI Rate\n(Multiplier)")
        owner_pct.append(_pct_change(sa2["owner_share_FI"],
                                     bl["owner_share_FI"]))
        renter_pct.append(_pct_change(sa2["renter_share_FI"],
                                      bl["renter_share_FI"]))
        if "cumulative_payout_usd" in sa2.columns:
            payout_pct.append(_pct_change(sa2["cumulative_payout_usd"],
                                          bl["cumulative_payout_usd"]))
        else:
            payout_pct.append((np.nan, np.nan))

    # --- SA3: FP Threshold ---
    sa3 = df[df["sa"] == "sa3_fp_threshold"].copy()
    if not sa3.empty and "owner_share_FI" in sa3.columns:
        sa3["value"] = pd.to_numeric(sa3["value"], errors="coerce")
        bl = sa3.loc[(sa3["value"] - 7).abs().idxmin()]  # baseline = 7 years
        params.append("FP Threshold\n(Years)")
        owner_pct.append(_pct_change(sa3["owner_share_FI"],
                                     bl["owner_share_FI"]))
        renter_pct.append(_pct_change(sa3["renter_share_FI"],
                                      bl["renter_share_FI"]))
        if "cumulative_payout_usd" in sa3.columns:
            payout_pct.append(_pct_change(sa3["cumulative_payout_usd"],
                                          bl["cumulative_payout_usd"]))
        else:
            payout_pct.append((np.nan, np.nan))

    if not params:
        print("[SKIP] Tornado plot — no SA data found")
        return

    n = len(params)
    y = np.arange(n)
    bar_h = 0.5

    def _draw_panel(ax, pct_data, color, xlabel, panel_lbl, title):
        """Draw one tornado panel with % change bars."""
        for i, (lo, hi) in enumerate(pct_data):
            is_na = np.isnan(lo) or np.isnan(hi)
            is_zero = (not is_na) and abs(hi - lo) < 0.01
            if is_na:
                ax.barh(i, 0, height=bar_h, color="lightgray", alpha=0.3)
                ax.text(0, i, "N/A", ha="center", va="center",
                        fontsize=9, color="gray")
            elif is_zero:
                # Improvement 3: annotate zero-sensitivity bars
                ax.barh(i, 0, height=bar_h, color=color, alpha=0.15,
                        edgecolor="black", linewidth=0.5)
                ax.text(0.5, i, "0.0%", ha="left", va="center",
                        fontsize=8, color="gray", style="italic")
            else:
                ax.barh(i, hi - lo, left=lo, height=bar_h,
                        color=color, alpha=0.7,
                        edgecolor="black", linewidth=0.5)
        ax.axvline(0, color="black", linestyle="--", linewidth=1.0,
                   label="Baseline")
        ax.set_yticks(y)
        ax.set_yticklabels(params, fontsize=10)
        ax.set_xlabel(xlabel)
        # Improvement 2: title without panel label (panel_label adds it)
        ax.set_title(title, fontsize=12)
        panel_label(ax, panel_lbl)

    _draw_panel(axes[0], owner_pct, COLOR_OWNER,
                "Change from Baseline (%)",
                "(a)", "Owner FI Adoption")
    _draw_panel(axes[1], renter_pct, COLOR_RENTER,
                "Change from Baseline (%)",
                "(b)", "Renter FI Adoption")
    _draw_panel(axes[2], payout_pct, COLORS["orange"],
                "Change from Baseline (%)",
                "(c)", "Insurance Payout")

    # Add legend only to first panel
    axes[0].legend(fontsize=8, loc="lower right")

    fig.tight_layout()
    out = FIG_DIR / "SA_tornado_plot.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out}")


# ── Figure SA-2: TP Decay Time-Series ────────────────────────────────────

def plot_tp_timeseries(df: pd.DataFrame) -> None:
    """Time-series of mean TP across tau_inf scale factors.

    2 panels: (a) Owner TP, (b) Renter TP over 2011-2023
    """
    sa1 = df[df["sa"] == "sa1_tp_decay"].copy()
    if sa1.empty:
        print("[SKIP] TP time-series — no SA1 data")
        return

    set_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    # Build TP columns
    tp_owner_cols = [f"tp_owner_{yr}" for yr in YEARS]
    tp_renter_cols = [f"tp_renter_{yr}" for yr in YEARS]

    # Check if time-series columns exist
    available_o = [c for c in tp_owner_cols if c in sa1.columns]
    available_r = [c for c in tp_renter_cols if c in sa1.columns]
    if not available_o:
        print("[SKIP] TP time-series — no per-year TP columns in metrics")
        return

    cmap = plt.cm.viridis
    sa1["value"] = pd.to_numeric(sa1["value"], errors="coerce")
    scales = sorted(sa1["value"].dropna().unique())
    norm = Normalize(vmin=float(min(scales)), vmax=float(max(scales)))

    for _, row in sa1.iterrows():
        scale = float(row["value"])
        color = cmap(norm(scale))
        label = f"×{scale:.2f}"

        # Owner — no markers (Improvement 4: reduce clutter)
        vals_o = [float(row[c]) if c in row.index and pd.notna(row[c])
                  else np.nan for c in tp_owner_cols]
        yrs_o = [YEARS[i] for i, v in enumerate(vals_o) if not np.isnan(v)]
        vals_o = [v for v in vals_o if not np.isnan(v)]
        if vals_o:
            axes[0].plot(yrs_o, vals_o, color=color,
                         label=label, linewidth=1.8)

        # Renter
        vals_r = [float(row[c]) if c in row.index and pd.notna(row[c])
                  else np.nan for c in tp_renter_cols]
        yrs_r = [YEARS[i] for i, v in enumerate(vals_r) if not np.isnan(v)]
        vals_r = [v for v in vals_r if not np.isnan(v)]
        if vals_r:
            axes[1].plot(yrs_r, vals_r, color=color,
                         label=label, linewidth=1.8)

    # Improvement 4: unified legend position (upper right) in both panels
    for i, (ax, title) in enumerate(zip(axes, ["Owner Mean TP",
                                                "Renter Mean TP"])):
        panel_lbl = "(a)" if i == 0 else "(b)"
        ax.set_title(title, fontsize=12)
        panel_label(ax, panel_lbl)
        ax.set_xlabel("Year")
        if i == 0:
            ax.set_ylabel("Mean Threat Perception (TP)")
        ax.legend(title="τ∞ scale", fontsize=8, title_fontsize=9,
                  loc="upper right")
        ax.set_ylim(0, 0.7)

        # Shade severe flood years (using actual year values on x-axis)
        for sy in SEVERE_YEARS:
            ax.axvspan(sy - 0.45, sy + 0.45,
                       color=COLORS["severe_shade"], alpha=0.25, zorder=0)

    fig.tight_layout()
    out = FIG_DIR / "SA_tp_decay_timeseries.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out}")


# ── Figure SA-3: RT × TP Decay Heatmap ──────────────────────────────────

def plot_interaction_heatmap(df: pd.DataFrame) -> None:
    """3×3 heatmap of Owner/Renter FI rate for RT × TP decay interaction.

    2 panels: (a) Owner FI, (b) Renter FI
    """
    sa4 = df[df["sa"] == "sa4_interaction"].copy()
    if sa4.empty:
        print("[SKIP] Interaction heatmap — no SA4 data")
        return

    set_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    rt_vals = sorted(sa4["rt"].unique())
    decay_vals = sorted(sa4["decay_scale"].unique())

    panel_labels = ["(d)", "(e)"]
    for panel_idx, (metric, title, cmap_name) in enumerate([
        ("owner_share_FI", "Owner FI Rate", "Blues"),
        ("renter_share_FI", "Renter FI Rate", "Greens"),
    ]):
        ax = axes[panel_idx]

        # Build grid
        grid = np.full((len(decay_vals), len(rt_vals)), np.nan)
        for _, row in sa4.iterrows():
            ri = rt_vals.index(row["rt"])
            di = decay_vals.index(row["decay_scale"])
            if metric in row and pd.notna(row[metric]):
                grid[di, ri] = row[metric]

        im = ax.imshow(grid, aspect="auto", cmap=cmap_name,
                       interpolation="nearest",
                       vmin=np.nanmin(grid) * 0.95,
                       vmax=np.nanmax(grid) * 1.05)

        # Annotate cells
        for di in range(len(decay_vals)):
            for ri in range(len(rt_vals)):
                val = grid[di, ri]
                if not np.isnan(val):
                    # Highlight baseline cell
                    is_baseline = (abs(rt_vals[ri] - 0.10) < 1e-9
                                   and abs(decay_vals[di] - 1.0) < 1e-9)
                    weight = "bold" if is_baseline else "normal"
                    midpoint = (np.nanmax(grid) + np.nanmin(grid)) / 2
                    if is_baseline:
                        color = "red"
                    elif val > midpoint:
                        color = "white"
                    else:
                        color = "black"
                    ax.text(ri, di, f"{val:.3f}", ha="center", va="center",
                            fontsize=9, fontweight=weight, color=color)

        ax.set_xticks(range(len(rt_vals)))
        ax.set_xticklabels([f"{v:.2f}" for v in rt_vals])
        ax.set_yticks(range(len(decay_vals)))
        ax.set_yticklabels([f"×{v:.1f}" for v in decay_vals])
        ax.set_xlabel("Ratio Threshold (RT)")
        ax.set_ylabel("τ∞ Scale Factor")
        ax.set_title(title, fontsize=12)
        panel_label(ax, panel_labels[panel_idx])
        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.tight_layout()
    out = FIG_DIR / "SA_interaction_heatmap.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out}")


# ── Figure SA-4: Summary Table ──────────────────────────────────────────

def plot_summary_table(df: pd.DataFrame, rt_df: pd.DataFrame | None) -> None:
    """Render SA summary as a matplotlib table figure.

    Improvement 5: consistent column semantics (Owner FI Δ, Renter FI Δ,
    Payout Δ) and footnote for zero-sensitivity rows.
    """
    set_paper_style()

    rows = []

    # RT from existing data
    if rt_df is not None and not rt_df.empty:
        rt_bl = rt_df.loc[(rt_df["threshold"] - 0.10).abs().idxmin()]
        ofi_d = rt_df["owner_share_FI"].max() - rt_df["owner_share_FI"].min()
        rfi_d = rt_df["renter_share_FI"].max() - rt_df["renter_share_FI"].min()
        rows.append([
            "Ratio Threshold (RT)",
            "0.10",
            "[0.01, 0.50]",
            f"{ofi_d:.3f}",
            f"{rfi_d:.3f}",
            "N/A",
            "Strongest driver of FI"
        ])

    # SA1
    sa1 = df[df["sa"] == "sa1_tp_decay"].copy()
    if not sa1.empty:
        sa1["value"] = pd.to_numeric(sa1["value"], errors="coerce")
        ofi_d = sa1["owner_share_FI"].max() - sa1["owner_share_FI"].min()
        rfi_d = sa1["renter_share_FI"].max() - sa1["renter_share_FI"].min()
        bl_pay = sa1.loc[(sa1["value"] - 1.0).abs().idxmin(),
                         "cumulative_payout_usd"]
        pay_lo = (sa1["cumulative_payout_usd"].min() - bl_pay) / bl_pay * 100
        pay_hi = (sa1["cumulative_payout_usd"].max() - bl_pay) / bl_pay * 100
        rows.append([
            "TP Decay (τ∞)",
            "33.3 / 46.0",
            "[×0.5, ×2.0]",
            f"{ofi_d:.3f}",
            f"{rfi_d:.3f}",
            f"{pay_lo:+.1f}% to {pay_hi:+.1f}%",
            "Moderate; renters > owners"
        ])

    # SA2
    sa2 = df[df["sa"] == "sa2_fi_rates"].copy()
    if not sa2.empty:
        sa2["value"] = pd.to_numeric(sa2["value"], errors="coerce")
        bl_pay = sa2.loc[(sa2["value"] - 1.0).abs().idxmin(),
                         "cumulative_payout_usd"]
        pay_lo = (sa2["cumulative_payout_usd"].min() - bl_pay) / bl_pay * 100
        pay_hi = (sa2["cumulative_payout_usd"].max() - bl_pay) / bl_pay * 100
        rows.append([
            "Init FI Rate",
            "FP: 25%/8%",
            "[×0.5, ×1.5]",
            "0.000*",
            "0.000*",
            f"{pay_lo:+.1f}% to {pay_hi:+.1f}%",
            "Finance only; decisions invariant"
        ])

    # SA3
    sa3 = df[df["sa"] == "sa3_fp_threshold"].copy()
    if not sa3.empty:
        sa3["value"] = pd.to_numeric(sa3["value"], errors="coerce")
        bl_pay = sa3.loc[(sa3["value"] - 7).abs().idxmin(),
                         "cumulative_payout_usd"]
        pay_lo = (sa3["cumulative_payout_usd"].min() - bl_pay) / bl_pay * 100
        pay_hi = (sa3["cumulative_payout_usd"].max() - bl_pay) / bl_pay * 100
        rows.append([
            "FP Threshold",
            "7 years",
            "[1, 7, 13] yrs",
            "0.000*",
            "0.000*",
            f"{pay_lo:+.1f}% to {pay_hi:+.1f}%",
            "Bimodal tracts limit effect"
        ])

    if not rows:
        print("[SKIP] Summary table — no data")
        return

    col_labels = ["Parameter", "Baseline", "Range Tested",
                  "Owner FI Δ", "Renter FI Δ", "Payout Δ", "Key Finding"]

    fig, ax = plt.subplots(figsize=(15, 2.0 + 0.5 * len(rows)))
    ax.axis("off")
    ax.set_title("Sensitivity Analysis Summary", fontsize=14, fontweight="bold",
                 pad=20)

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        colWidths=[0.14, 0.10, 0.10, 0.09, 0.09, 0.16, 0.32],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)

    # Style header
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold")

    # Alternate row colors
    for i in range(1, len(rows) + 1):
        for j in range(len(col_labels)):
            cell = table[i, j]
            if i % 2 == 0:
                cell.set_facecolor("#ecf0f1")

    # Footnote for asterisk
    fig.text(0.02, 0.02,
             "* Decision shares are invariant because parameter does not "
             "enter the Bayesian decision engine.",
             fontsize=8, fontstyle="italic", color="#555555",
             transform=fig.transFigure)

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    out = FIG_DIR / "SA_summary_table.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out}")


# ── main ─────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="SA Visualization for FLOODABM")
    ap.add_argument(
        "--fig", choices=["tornado", "timeseries", "heatmap", "table", "all"],
        default="all", help="Which figure to generate"
    )
    args = ap.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = load_sa_metrics()
    rt_df = load_rt_summary()

    print("=" * 60)
    print("SA Visualization")
    print(f"  SA metrics: {len(df)} rows")
    print(f"  RT summary: {'loaded' if rt_df is not None else 'not found'}")
    print(f"  Output: {FIG_DIR}")
    print("=" * 60)

    fig_map = {
        "tornado": lambda: plot_tornado(df, rt_df),
        "timeseries": lambda: plot_tp_timeseries(df),
        "heatmap": lambda: plot_interaction_heatmap(df),
        "table": lambda: plot_summary_table(df, rt_df),
    }

    if args.fig == "all":
        for name, func in fig_map.items():
            print(f"\n--- {name} ---")
            func()
    else:
        fig_map[args.fig]()

    print(f"\nDone! Figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
