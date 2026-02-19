# utils/plots_tract_analysis.py
"""
Tract-level analysis plots for FLOODABM outputs.
Three main dimensions:
1. Worst / Baseline comparison
    fig.suptitle("Worst vs Baseline by Tract", fontsize=18, fontweight="bold")
3. Damage / Financial outcomes
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd


def _set_style():
    """Set paper-ready plot style."""
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 300, 
        "savefig.dpi": 300,
        "font.size": 20,            # Standardized
        "axes.titlesize": 22,       # Standardized
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


def plot_worst_baseline_tract(spatial_csv: Path, fig_root: Path) -> None:
    """
    Dimension 1: Worst vs Baseline comparison by tract.
    Generates 2x2 panel figure.
    """
    if not spatial_csv.exists():
        return
    
    import matplotlib.pyplot as plt
    _set_style()
    
    df = pd.read_csv(spatial_csv)
    out_dir = fig_root / "compare"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Worst vs Baseline by Tract", fontsize=18, fontweight="bold")
    
    # Early period - damage ratio (owner)
    ax = axes[0, 0]
    early = df[df["period"] == "early"].copy()
    if "ratio_avg_damage_perhh_owner_bl_over_wr" in early.columns:
        early["ratio"] = early["ratio_avg_damage_perhh_owner_bl_over_wr"].fillna(1.0)
        early_sorted = early.sort_values("ratio").reset_index(drop=True)
        colors = ["#ef4444" if r > 1 else "#22c55e" for r in early_sorted["ratio"]]
        ax.barh(range(len(early_sorted)), early_sorted["ratio"], color=colors, alpha=0.7)
        ax.axvline(1.0, color="black", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Baseline / Worst Ratio")
    ax.set_ylabel("Tract Index")
    ax.set_title("(a) Early Period - Homeowner Damage Ratio")
    
    # Late period - damage ratio
    ax = axes[0, 1]
    late = df[df["period"] == "late"].copy()
    if "ratio_avg_damage_perhh_owner_bl_over_wr" in late.columns:
        late["ratio"] = late["ratio_avg_damage_perhh_owner_bl_over_wr"].fillna(1.0)
        late_sorted = late.sort_values("ratio").reset_index(drop=True)
        colors = ["#ef4444" if r > 1 else "#22c55e" for r in late_sorted["ratio"]]
        ax.barh(range(len(late_sorted)), late_sorted["ratio"], color=colors, alpha=0.7)
        ax.axvline(1.0, color="black", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Baseline / Worst Ratio")
    ax.set_title("(b) Late Period - Homeowner Damage Ratio")
    
    # Payout rate comparison
    ax = axes[1, 0]
    if "payout_rate_owner_baseline" in early.columns:
        width = 0.35
        x = range(len(early))
        ax.bar([i - width/2 for i in x], early["payout_rate_owner_baseline"], width, 
               label="Baseline", color="#3b82f6", alpha=0.7)
        ax.bar([i + width/2 for i in x], early["payout_rate_owner_worst"], width,
               label="Worst", color="#f97316", alpha=0.7)
        ax.legend()
    ax.set_xlabel("Tract Index")
    ax.set_ylabel("Payout Rate")
    ax.set_title("(c) Early Period - Payout Rate Comparison")
    
    # Flood-prone vs Non-flood-prone
    ax = axes[1, 1]
    if "flood_prone" in df.columns and "avg_damage_perhh_owner_baseline" in df.columns:
        fp = df[df["flood_prone"] == 1]
        nfp = df[df["flood_prone"] == 0]
        categories = ["Flood-Prone", "Non-Flood-Prone"]
        base_vals = [fp["avg_damage_perhh_owner_baseline"].mean(), 
                     nfp["avg_damage_perhh_owner_baseline"].mean()]
        worst_vals = [fp["avg_damage_perhh_owner_worst"].mean(),
                      nfp["avg_damage_perhh_owner_worst"].mean()]
        width = 0.35
        x = range(len(categories))
        ax.bar([i - width/2 for i in x], base_vals, width, label="Baseline", color="#3b82f6", alpha=0.7)
        ax.bar([i + width/2 for i in x], worst_vals, width, label="Worst", color="#f97316", alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
    ax.set_ylabel("Avg Damage per HH ($)")
    ax.set_title("(d) Flood-Prone vs Non-Flood-Prone")
    
    plt.tight_layout()
    out_path = out_dir / "worst_baseline_tract.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path.name}")


def plot_owner_renter_actions_tract(actions_csv: Path, fig_root: Path) -> None:
    """
    Dimension 2: Homeowner vs Renter action differences by tract.
    Generates 2x2 panel figure.
    """
    if not actions_csv.exists():
        return
    
    import matplotlib.pyplot as plt
    _set_style()
    
    df = pd.read_csv(actions_csv)
    out_dir = fig_root / "decisions"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Homeowner vs Renter Actions by Tract", fontsize=18, fontweight="bold")
    
    # Owner action distribution (pie)
    ax = axes[0, 0]
    owner_actions = ["owner_share_FI", "owner_share_EH", "owner_share_BP", "owner_share_DN"]
    owner_labels = ["FI", "EH", "BP", "DN"]
    owner_data = [df[c].mean() if c in df.columns else 0 for c in owner_actions]
    if sum(owner_data) > 0:
        colors = ["#3b82f6", "#22c55e", "#f97316", "#6b7280"]
        ax.pie(owner_data, labels=owner_labels, colors=colors, autopct="%1.1f%%", startangle=90)
    ax.set_title("(a) Homeowner Mean Action Distribution")
    
    # Renter action distribution (pie)
    ax = axes[0, 1]
    renter_actions = ["renter_share_FI", "renter_share_RL", "renter_share_DN"]
    renter_labels = ["FI", "RL", "DN"]
    renter_data = [df[c].mean() if c in df.columns else 0 for c in renter_actions]
    if sum(renter_data) > 0:
        colors = ["#3b82f6", "#ef4444", "#6b7280"]
        ax.pie(renter_data, labels=renter_labels, colors=colors, autopct="%1.1f%%", startangle=90)
    ax.set_title("(b) Renter Mean Action Distribution")
    
    # FI adoption by year
    ax = axes[1, 0]
    if "year" in df.columns:
        yearly = df.groupby("year")[["owner_share_FI", "renter_share_FI"]].mean()
        if "owner_share_FI" in df.columns:
            ax.plot(yearly.index, yearly["owner_share_FI"], marker="o", 
                    label="Homeowner", color="#3b82f6", linewidth=2)
        if "renter_share_FI" in df.columns:
            ax.plot(yearly.index, yearly["renter_share_FI"], marker="s",
                    label="Renter", color="#f97316", linewidth=2)
        ax.legend()
    ax.set_xlabel("Year")
    ax.set_ylabel("FI Adoption Rate")
    ax.set_title("(c) FI Adoption Over Time")
    
    # DN (Do Nothing) by year
    ax = axes[1, 1]
    if "year" in df.columns:
        yearly_dn = df.groupby("year")[["owner_share_DN", "renter_share_DN"]].mean()
        if "owner_share_DN" in df.columns:
            ax.plot(yearly_dn.index, yearly_dn["owner_share_DN"], marker="o",
                    label="Homeowner DN", color="#3b82f6", linewidth=2)
        if "renter_share_DN" in df.columns:
            ax.plot(yearly_dn.index, yearly_dn["renter_share_DN"], marker="s",
                    label="Renter DN", color="#f97316", linewidth=2)
        ax.legend()
    ax.set_xlabel("Year")
    ax.set_ylabel("DN (Do Nothing) Rate")
    ax.set_title("(d) Inaction Rate Over Time")
    
    plt.tight_layout()
    out_path = out_dir / "owner_renter_actions_tract.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path.name}")


def plot_damage_finance_tract(finance_csv: Path, fig_root: Path) -> None:
    """
    Dimension 3: Damage and financial outcomes by tract.
    Generates 2x2 panel figure.
    """
    if not finance_csv.exists():
        return
    
    import matplotlib.pyplot as plt
    _set_style()
    
    df = pd.read_csv(finance_csv)
    out_dir = fig_root / "finance"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Damage & Financial Outcomes by Tract", fontsize=18, fontweight="bold")
    
    # Total damage by tract (top 15)
    ax = axes[0, 0]
    if "gross_structure_loss_kUSD" in df.columns:
        tract_damage = df.groupby("tract_geoid").agg({
            "gross_structure_loss_kUSD": "sum",
            "gross_contents_loss_kUSD": "sum"
        }).reset_index()
        tract_damage["total_loss"] = (tract_damage["gross_structure_loss_kUSD"] + 
                                       tract_damage["gross_contents_loss_kUSD"])
        tract_damage = tract_damage.sort_values("total_loss", ascending=False).head(15)
        ax.barh(range(len(tract_damage)), tract_damage["total_loss"], color="#ef4444", alpha=0.7)
        ax.set_yticks(range(len(tract_damage)))
        ax.set_yticklabels(tract_damage["tract_geoid"].astype(str).str[-5:])
    ax.set_xlabel("Total Loss (kUSD)")
    ax.set_title("(a) Top 15 Tracts by Total Damage")
    
    # Payout by tract (top 15)
    ax = axes[0, 1]
    if "payout_structure_kUSD" in df.columns:
        tract_payout = df.groupby("tract_geoid").agg({
            "payout_structure_kUSD": "sum",
            "payout_contents_kUSD": "sum"
        }).reset_index()
        tract_payout["total_payout"] = (tract_payout["payout_structure_kUSD"] + 
                                         tract_payout["payout_contents_kUSD"])
        tract_payout = tract_payout.sort_values("total_payout", ascending=False).head(15)
        ax.barh(range(len(tract_payout)), tract_payout["total_payout"], color="#22c55e", alpha=0.7)
        ax.set_yticks(range(len(tract_payout)))
        ax.set_yticklabels(tract_payout["tract_geoid"].astype(str).str[-5:])
    ax.set_xlabel("Total Payout (kUSD)")
    ax.set_title("(b) Top 15 Tracts by Insurance Payout")
    
    # OOP cost over years
    ax = axes[1, 0]
    if "year" in df.columns and "oop_structure_kUSD" in df.columns:
        yearly_oop = df.groupby("year")[["oop_structure_kUSD", "oop_contents_kUSD"]].sum()
        yearly_oop["total_oop"] = yearly_oop["oop_structure_kUSD"] + yearly_oop["oop_contents_kUSD"]
        ax.bar(yearly_oop.index, yearly_oop["total_oop"], color="#f97316", alpha=0.7)
    ax.set_xlabel("Year")
    ax.set_ylabel("OOP Cost (kUSD)")
    ax.set_title("(c) Out-of-Pocket Cost Over Time")
    
    # Payout rate by year
    ax = axes[1, 1]
    if "year" in df.columns and "payout_structure_kUSD" in df.columns:
        yearly_fin = df.groupby("year").agg({
            "payout_structure_kUSD": "sum",
            "payout_contents_kUSD": "sum",
            "gross_structure_loss_kUSD": "sum",
            "gross_contents_loss_kUSD": "sum"
        })
        yearly_fin["payout_rate"] = (
            (yearly_fin["payout_structure_kUSD"] + yearly_fin["payout_contents_kUSD"]) /
            (yearly_fin["gross_structure_loss_kUSD"] + yearly_fin["gross_contents_loss_kUSD"] + 1e-6)
        )
        ax.plot(yearly_fin.index, yearly_fin["payout_rate"], marker="o", color="#3b82f6", linewidth=2)
        ax.set_ylim(0, 1)
    ax.set_xlabel("Year")
    ax.set_ylabel("Payout Rate")
    ax.set_title("(d) Payout Rate Over Time")
    
    plt.tight_layout()
    out_path = out_dir / "damage_finance_tract.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path.name}")


def plot_avg_damage_with_errorbar(finance_csv: Path, fig_root: Path,
                                    exclude_outliers: bool = True,
                                    severe_years: tuple = (2011, 2014, 2021)) -> None:
    """
    Plot average per-household damage by year with error bars.
    
    - Error bars show 95% CI across tracts (after removing outliers)
    - Outliers removed using IQR method (1.5 * IQR)
    - Legend placed inside the plot
    
    Args:
        finance_csv: Path to finance_tract_all_years.csv
        fig_root: Output directory for figures
        exclude_outliers: If True, exclude outliers using IQR method
        severe_years: Years to mark as severe flood years
    """
    if not finance_csv.exists():
        return
    
    import matplotlib.pyplot as plt
    from scipy import stats
    _set_style()
    
    df = pd.read_csv(finance_csv)
    out_dir = fig_root / "finance"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate per-tract, per-year damage
    if "gross_structure_loss_kUSD" not in df.columns:
        return
    
    df["total_loss_kUSD"] = df["gross_structure_loss_kUSD"] + df["gross_contents_loss_kUSD"]
    years = sorted(df["year"].unique())
    
    means, ci_lower, ci_upper = [], [], []
    
    for year in years:
        year_data = df[df["year"] == year]["total_loss_kUSD"].values
        
        if exclude_outliers and len(year_data) > 4:
            # Remove outliers using IQR method
            q1, q3 = np.percentile(year_data, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            year_data = year_data[(year_data >= lower_bound) & (year_data <= upper_bound)]
        
        if len(year_data) > 0:
            mean = np.mean(year_data)
            # 95% CI using t-distribution
            if len(year_data) > 1:
                sem = stats.sem(year_data)
                ci = sem * stats.t.ppf(0.975, len(year_data) - 1)
            else:
                ci = 0
            means.append(mean)
            ci_lower.append(mean - ci)
            ci_upper.append(mean + ci)
        else:
            means.append(0)
            ci_lower.append(0)
            ci_upper.append(0)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(9, 5))
    
    # Bar chart with error bars
    x = np.arange(len(years))
    errors = [np.array(means) - np.array(ci_lower), np.array(ci_upper) - np.array(means)]
    
    bars = ax.bar(x, means, color="#3b82f6", alpha=0.7, edgecolor="black", linewidth=0.5,
                  yerr=errors, capsize=3, error_kw={"elinewidth": 1, "capthick": 1})
    
    # Highlight severe years
    for i, year in enumerate(years):
        if year in severe_years:
            bars[i].set_color("#ef4444")
    
    # Formatting
    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in years], rotation=45, ha="right")
    ax.set_xlabel("Year")
    ax.set_ylabel("Avg Damage per Tract (kUSD)")
    ax.set_title("Average Flood Damage by Year (± 95% CI)")
    
    # Legend inside plot (upper right, not blocking main data)
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#3b82f6", alpha=0.7, edgecolor="black", label="Normal Year"),
        Patch(facecolor="#ef4444", alpha=0.7, edgecolor="black", label="Severe Flood Year"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", frameon=True, 
              framealpha=0.9, edgecolor="gray")
    
    plt.tight_layout()
    out_path = out_dir / "avg_damage_with_errorbar.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path.name}")


def plot_owner_renter_damage_errorbar(finance_csv: Path, fig_root: Path,
                                       exclude_outliers: bool = True) -> None:
    """
    Plot Homeowner vs Renter average damage by year with error bars.
    
    - Error bars show 95% CI across tracts
    - Outliers removed using IQR method
    - Legend placed inside the plot
    """
    if not finance_csv.exists():
        return
    
    import matplotlib.pyplot as plt
    from scipy import stats
    _set_style()
    
    df = pd.read_csv(finance_csv)
    out_dir = fig_root / "finance"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for owner/renter columns
    owner_col = None
    renter_col = None
    for c in df.columns:
        if "owner" in c.lower() and "gross" in c.lower():
            owner_col = c
        if "renter" in c.lower() and "gross" in c.lower():
            renter_col = c
    
    if not owner_col and not renter_col:
        # Try alternative column names
        if "gross_structure_loss_kUSD" in df.columns:
            # Use total as fallback
            return
    
    years = sorted(df["year"].unique())
    
    def calc_stats(data):
        if exclude_outliers and len(data) > 4:
            q1, q3 = np.percentile(data, [25, 75])
            iqr = q3 - q1
            data = data[(data >= q1 - 1.5 * iqr) & (data <= q3 + 1.5 * iqr)]
        if len(data) > 1:
            mean = np.mean(data)
            sem = stats.sem(data)
            ci = sem * stats.t.ppf(0.975, len(data) - 1)
            return mean, ci
        return np.mean(data) if len(data) > 0 else 0, 0
    
    owner_means, owner_cis = [], []
    renter_means, renter_cis = [], []
    
    for year in years:
        year_df = df[df["year"] == year]
        if owner_col:
            m, c = calc_stats(year_df[owner_col].values)
            owner_means.append(m)
            owner_cis.append(c)
        if renter_col:
            m, c = calc_stats(year_df[renter_col].values)
            renter_means.append(m)
            renter_cis.append(c)
    
    if not owner_means and not renter_means:
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(years))
    width = 0.35
    
    if owner_means:
        ax.bar(x - width/2, owner_means, width, label="Homeowner", color="#3b82f6", alpha=0.7,
               yerr=owner_cis, capsize=3, error_kw={"elinewidth": 1, "capthick": 1})
    if renter_means:
        ax.bar(x + width/2, renter_means, width, label="Renter", color="#f97316", alpha=0.7,
               yerr=renter_cis, capsize=3, error_kw={"elinewidth": 1, "capthick": 1})
    
    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in years], rotation=45, ha="right")
    ax.set_xlabel("Year")
    ax.set_ylabel("Avg Damage (kUSD)")
    ax.set_title("Homeowner vs Renter Damage (± 95% CI)")
    
    # Legend inside plot
    ax.legend(loc="upper right", frameon=True, framealpha=0.9, edgecolor="gray")
    
    plt.tight_layout()
    out_path = out_dir / "owner_renter_damage_errorbar.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path.name}")


def plot_tract_analysis(output_dir: Path, fig_root: Path) -> None:
    """
    Main entry point for tract-level analysis plots.
    Called from plot_all_outputs.
    
    Args:
        output_dir: The output directory (e.g., outputs/baseline/baseline)
        fig_root: The visualization directory (e.g., outputs/baseline/baseline/visualization)
    """
    print("[tract-analysis] Generating tract-level analysis plots...")
    
    # Dimension 1: Worst/Baseline comparison (requires spatial CSV)
    spatial_csv = fig_root / "spatial_inputs_cumrate_perhh_early_late.csv"
    if spatial_csv.exists():
        try:
            plot_worst_baseline_tract(spatial_csv, fig_root)
        except Exception as e:
            print(f"[warn] plot_worst_baseline_tract failed: {e}")
    
    # Dimension 2: Homeowner/Renter actions
    actions_csv = output_dir / "decisions" / "action_share_owner_renter_tract_all_years.csv"
    if actions_csv.exists():
        try:
            plot_owner_renter_actions_tract(actions_csv, fig_root)
        except Exception as e:
            print(f"[warn] plot_owner_renter_actions_tract failed: {e}")
    
    # Dimension 3: Damage/Finance
    finance_csv = output_dir / "finance" / "finance_tract_all_years.csv"
    if finance_csv.exists():
        try:
            plot_damage_finance_tract(finance_csv, fig_root)
        except Exception as e:
            print(f"[warn] plot_damage_finance_tract failed: {e}")
        
        # NEW: Average damage with error bars
        try:
            plot_avg_damage_with_errorbar(finance_csv, fig_root)
        except Exception as e:
            print(f"[warn] plot_avg_damage_with_errorbar failed: {e}")
        
        # NEW: Homeowner vs Renter damage with error bars
        try:
            plot_owner_renter_damage_errorbar(finance_csv, fig_root)
        except Exception as e:
            print(f"[warn] plot_owner_renter_damage_errorbar failed: {e}")
        
        # NEW: Finance plots with error bars (payout, OOP)
        try:
            from utils.plots_finance_errorbar import plot_finance_errorbar_all
            plot_finance_errorbar_all(output_dir, fig_root)
        except Exception as e:
            print(f"[warn] plot_finance_errorbar_all failed: {e}")

