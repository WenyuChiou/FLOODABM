# utils/plots_finance_errorbar.py
"""
Finance plots with error bars for FLOODABM outputs.
Adds 95% CI error bars (across tracts) and legend inside plot.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pandas as pd


def _set_style():
    """Set paper-ready plot style."""
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 300, 
        "savefig.dpi": 300,
        "font.size": 20,
        "axes.titlesize": 22,
        "axes.titleweight": "bold", # Bold
        "axes.labelsize": 20,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 18,
        "axes.linewidth": 1.0,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "CMU Serif"],
    })


def _calc_ci(data: np.ndarray, confidence: float = 0.95) -> Tuple[float, float, float]:
    """Calculate mean and 95% CI, excluding outliers using IQR."""
    from scipy import stats
    
    if len(data) == 0:
        return 0, 0, 0
    
    # Remove outliers using IQR
    if len(data) > 4:
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        mask = (data >= q1 - 1.5 * iqr) & (data <= q3 + 1.5 * iqr)
        data = data[mask]
    
    if len(data) == 0:
        return 0, 0, 0
    
    mean = np.mean(data)
    if len(data) > 1:
        sem = stats.sem(data)
        ci = sem * stats.t.ppf(0.975, len(data) - 1)
    else:
        ci = 0
    
    return mean, ci, ci  # Returns mean, ci_lower, ci_upper (symmetric)


def plot_payout_with_errorbar(
    fin_dir: Path,
    fig_root: Path,
    severe_years: Tuple[int, ...] = (2011, 2014, 2021),
) -> None:
    """
    Plot Homeowner/Renter payout bar charts with error bars.
    
    - Uses tract-level data for error bar calculation
    - Error bars show 95% CI across tracts
    - Legend placed inside plot (upper left)
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    _set_style()
    
    # Look for tract-level finance data
    finance_csv = fin_dir / "finance_tract_all_years.csv"
    if not finance_csv.exists():
        print("[warn] No tract-level finance data for error bar plot")
        return
    
    df = pd.read_csv(finance_csv)
    years = sorted(df["year"].unique())
    
    # Check for payout columns
    payout_cols = [c for c in df.columns if "payout" in c.lower()]
    if not payout_cols:
        return
    
    # Calculate total payout per tract
    df["total_payout"] = df["payout_structure_kUSD"] + df["payout_contents_kUSD"]
    
    means, cis = [], []
    for year in years:
        year_data = df[df["year"] == year]["total_payout"].values
        mean, ci_low, ci_high = _calc_ci(year_data)
        means.append(mean)
        cis.append(ci_low)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(years))
    
    # Bar chart with error bars  
    bars = ax.bar(x, means, color="#F58518", alpha=0.8, edgecolor="black", linewidth=0.5,
                  yerr=cis, capsize=3, error_kw={"elinewidth": 1, "capthick": 1, "color": "black"})
    
    # Highlight severe years
    for i, year in enumerate(years):
        if year in severe_years:
            ax.axvspan(i - 0.4, i + 0.4, color="#93c5fd", alpha=0.2, zorder=0)
    
    # Formatting
    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in years], rotation=45, ha="right")
    ax.set_xlabel("Year")
    ax.set_ylabel("Avg Payout per Tract (kUSD)")
    ax.set_title("Insurance Payout by Year (± 95% CI)")
    
    # Legend inside plot - upper left
    legend_elements = [
        Patch(facecolor="#F58518", alpha=0.8, edgecolor="black", label="Payout"),
        Patch(facecolor="#93c5fd", alpha=0.2, edgecolor="none", label="Severe Flood Year"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", frameon=True, 
              framealpha=0.9, edgecolor="gray")
    
    plt.tight_layout()
    out_dir = fig_root / "finance"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "payout_with_errorbar.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path.name}")


def plot_oop_with_errorbar(
    fin_dir: Path,
    fig_root: Path,
    severe_years: Tuple[int, ...] = (2011, 2014, 2021),
) -> None:
    """
    Plot Out-of-Pocket cost bar chart with error bars.
    
    - Uses tract-level data for error bar calculation
    - Error bars show 95% CI across tracts
    - Legend placed inside plot (upper left)
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    _set_style()
    
    # Look for tract-level finance data
    finance_csv = fin_dir / "finance_tract_all_years.csv"
    if not finance_csv.exists():
        return
    
    df = pd.read_csv(finance_csv)
    years = sorted(df["year"].unique())
    
    # Check for OOP columns
    if "oop_structure_kUSD" not in df.columns:
        return
    
    df["total_oop"] = df["oop_structure_kUSD"] + df["oop_contents_kUSD"]
    
    means, cis = [], []
    for year in years:
        year_data = df[df["year"] == year]["total_oop"].values
        mean, ci_low, ci_high = _calc_ci(year_data)
        means.append(mean)
        cis.append(ci_low)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(years))
    
    # Bar chart with error bars
    bars = ax.bar(x, means, color="#54A24B", alpha=0.8, edgecolor="black", linewidth=0.5,
                  yerr=cis, capsize=3, error_kw={"elinewidth": 1, "capthick": 1, "color": "black"})
    
    # Highlight severe years
    for i, year in enumerate(years):
        if year in severe_years:
            ax.axvspan(i - 0.4, i + 0.4, color="#93c5fd", alpha=0.2, zorder=0)
    
    # Formatting
    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in years], rotation=45, ha="right")
    ax.set_xlabel("Year")
    ax.set_ylabel("Avg OOP per Tract (kUSD)")
    ax.set_title("Out-of-Pocket Cost by Year (± 95% CI)")
    
    # Legend inside plot - upper left
    legend_elements = [
        Patch(facecolor="#54A24B", alpha=0.8, edgecolor="black", label="OOP Cost"),
        Patch(facecolor="#93c5fd", alpha=0.2, edgecolor="none", label="Severe Flood Year"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", frameon=True, 
              framealpha=0.9, edgecolor="gray")
    
    plt.tight_layout()
    out_dir = fig_root / "finance"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "oop_with_errorbar.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path.name}")


def plot_payout_oop_combined_errorbar(
    fin_dir: Path,
    fig_root: Path,
    severe_years: Tuple[int, ...] = (2011, 2014, 2021),
) -> None:
    """
    Combined Payout + OOP bar chart with error bars (2x1 panels).
    
    - Error bars show 95% CI across tracts (IQR outlier exclusion)
    - Legend placed inside each subplot (upper left)
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    _set_style()
    
    # Look for tract-level finance data
    finance_csv = fin_dir / "finance_tract_all_years.csv"
    if not finance_csv.exists():
        return
    
    df = pd.read_csv(finance_csv)
    years = sorted(df["year"].unique())
    
    if "payout_structure_kUSD" not in df.columns or "oop_structure_kUSD" not in df.columns:
        return
    
    df["total_payout"] = df["payout_structure_kUSD"] + df["payout_contents_kUSD"]
    df["total_oop"] = df["oop_structure_kUSD"] + df["oop_contents_kUSD"]
    
    def get_stats(col):
        means, cis = [], []
        for year in years:
            year_data = df[df["year"] == year][col].values
            mean, ci, _ = _calc_ci(year_data)
            means.append(mean)
            cis.append(ci)
        return np.array(means), np.array(cis)
    
    payout_means, payout_cis = get_stats("total_payout")
    oop_means, oop_cis = get_stats("total_oop")
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    x = np.arange(len(years))
    
    # Panel (a): Payout
    ax = axes[0]
    ax.bar(x, payout_means, color="#F58518", alpha=0.8, edgecolor="black", linewidth=0.5,
           yerr=payout_cis, capsize=3, error_kw={"elinewidth": 1, "capthick": 1, "color": "black"})
    
    for i, year in enumerate(years):
        if year in severe_years:
            ax.axvspan(i - 0.4, i + 0.4, color="#93c5fd", alpha=0.2, zorder=0)
    
    ax.set_ylabel("Payout (kUSD)")
    ax.set_title("(a) Insurance Payout (± 95% CI)")
    ax.legend([Patch(facecolor="#F58518", alpha=0.8, edgecolor="black", label="Payout"),
               Patch(facecolor="#93c5fd", alpha=0.2, edgecolor="none", label="Severe Year")],
              ["Payout", "Severe Year"], loc="upper left", frameon=True, framealpha=0.9)
    
    # Panel (b): OOP
    ax = axes[1]
    ax.bar(x, oop_means, color="#54A24B", alpha=0.8, edgecolor="black", linewidth=0.5,
           yerr=oop_cis, capsize=3, error_kw={"elinewidth": 1, "capthick": 1, "color": "black"})
    
    for i, year in enumerate(years):
        if year in severe_years:
            ax.axvspan(i - 0.4, i + 0.4, color="#93c5fd", alpha=0.2, zorder=0)
    
    ax.set_xlabel("Year")
    ax.set_ylabel("OOP Cost (kUSD)")
    ax.set_title("(b) Out-of-Pocket Cost (± 95% CI)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in years], rotation=45, ha="right")
    ax.legend([Patch(facecolor="#54A24B", alpha=0.8, edgecolor="black", label="OOP"),
               Patch(facecolor="#93c5fd", alpha=0.2, edgecolor="none", label="Severe Year")],
              ["OOP Cost", "Severe Year"], loc="upper left", frameon=True, framealpha=0.9)
    
    plt.tight_layout()
    out_dir = fig_root / "finance"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "payout_oop_combined_errorbar.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path.name}")


def plot_finance_errorbar_all(output_dir: Path, fig_root: Path) -> None:
    """
    Main entry point for finance plots with error bars.
    Called from plot_all_outputs or plot_tract_analysis.
    """
    print("[finance-errorbar] Generating finance plots with error bars...")
    fin_dir = output_dir / "finance"
    
    try:
        plot_payout_with_errorbar(fin_dir, fig_root)
    except Exception as e:
        print(f"[warn] plot_payout_with_errorbar failed: {e}")
    
    try:
        plot_oop_with_errorbar(fin_dir, fig_root)
    except Exception as e:
        print(f"[warn] plot_oop_with_errorbar failed: {e}")
    
    try:
        plot_payout_oop_combined_errorbar(fin_dir, fig_root)
    except Exception as e:
        print(f"[warn] plot_payout_oop_combined_errorbar failed: {e}")
