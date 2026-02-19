"""
SA Results Visualization - TP Trajectory and Financial Comparison Plots
========================================================================
Generates:
1. TP Trajectory by Tract Type (MG vs NMG) with outlier removal
2. Financial Outcome Comparison (Total Damage, Payout, OOP by scenario)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.patches as mpatches

# Style settings
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 20,
    'axes.titlesize': 22,
    'axes.labelsize': 20,
    'legend.fontsize': 18,
    'figure.dpi': 300,
})

# ==============================================================================
# Configuration
# ==============================================================================

SHOCK_SCALE_ROOT = Path("outputs/experiments/shock_scale")
FLOOD_RATIO_ROOT = Path("outputs/experiments/flood_ratio")
FIG_DIR = Path("outputs/experiments/fig_shock_scale")

# Highlighted scenarios for Shock-Scale SA
SHOCK_HIGHLIGHTS = {
    "NMG=0.3,MG=0.3": {"color": "#1f77b4", "label": "NMG=0.3, MG=0.3 (Baseline)"},
    "NMG=0.3,MG=0.7": {"color": "#2ca02c", "label": "NMG=0.3, MG=0.7"},
    "NMG=0.7,MG=0.3": {"color": "#d62728", "label": "NMG=0.7, MG=0.3"},
    "NMG=0.7,MG=0.7": {"color": "#9467bd", "label": "NMG=0.7, MG=0.7"},
}

# ==============================================================================
# Helper Functions
# ==============================================================================

def get_scenario_dirs(root: Path, pattern: str = "SA_SHOCK_*") -> Dict[str, Path]:
    """Get all scenario directories matching pattern."""
    result = {}
    for d in root.glob(pattern):
        if d.is_dir():
            # Extract label from directory name
            name = d.name
            if "SA_SHOCK_O" in name:
                # Parse O0_3_R0_7 -> NMG=0.3,MG=0.7
                parts = name.replace("SA_SHOCK_O", "").split("_R")
                o_val = parts[0].replace("_", ".")
                r_val = parts[1].replace("_", ".")
                label = f"NMG={o_val},MG={r_val}"
            elif "SA_RT_MG" in name:
                # Parse MG0_3_NMG0_7 -> MG=0.3,NMG=0.7
                parts = name.replace("SA_RT_MG", "").split("_NMG")
                mg_val = parts[0].replace("_", ".")
                nmg_val = parts[1].replace("_", ".")
                label = f"MG={mg_val},NMG={nmg_val}"
            else:
                label = name
            
            # Find baseline directory
            baseline_path = d / "baseline" / "baseline"
            if baseline_path.exists():
                result[label] = baseline_path
    return result


def load_tp_traj(scenario_path: Path) -> pd.DataFrame:
    """Load TP trajectory data from a scenario."""
    tp_file = scenario_path / "tp_traj.csv"
    if not tp_file.exists():
        return pd.DataFrame()
    return pd.read_csv(tp_file)


def load_finance_summary(scenario_path: Path) -> pd.DataFrame:
    """Load finance summary from all years."""
    finance_file = scenario_path / "finance" / "finance_tract_all_years.csv"
    if not finance_file.exists():
        return pd.DataFrame()
    return pd.read_csv(finance_file)


def remove_outliers_iqr(series: pd.Series, k: float = 1.5) -> pd.Series:
    """Remove outliers using IQR method."""
    Q1, Q3 = series.quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower = Q1 - k * IQR
    upper = Q3 + k * IQR
    return series[(series >= lower) & (series <= upper)]


# ==============================================================================
# Plot 1: TP Trajectory by Tract Type
# ==============================================================================

def plot_tp_trajectory_by_group(
    scenarios: Dict[str, Path],
    highlights: Dict[str, Dict],
    out_png: Path,
    group: str = "MG",  # "MG" or "NMG"
    remove_outliers: bool = True,
    show_ci: bool = True,  # Show 95% confidence interval
):
    """
    Plot TP trajectory over years for a specific group (MG or NMG).
    Compares highlighted scenarios with 95% confidence interval bands.
    """
    from scipy import stats
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    col = f"TP_{group}"
    group_name = "Renters (MG)" if group == "MG" else "Homeowners (NMG)"
    
    all_data = {}
    
    for label, path in scenarios.items():
        df = load_tp_traj(path)
        if df.empty or col not in df.columns:
            continue
        
        # Remove outliers if requested
        if remove_outliers:
            df_clean = df.groupby("year")[col].apply(
                lambda x: remove_outliers_iqr(x)
            ).reset_index(level=0)
            df_clean.columns = ["year", col]
        else:
            df_clean = df[["year", col]].copy()
        
        # Calculate mean, std, count, and 95% CI
        yearly = df_clean.groupby("year")[col].agg(["mean", "std", "count"])
        
        # 95% CI = mean ± 1.96 * (std / sqrt(n))
        yearly["se"] = yearly["std"] / np.sqrt(yearly["count"])
        yearly["ci_lower"] = yearly["mean"] - 1.96 * yearly["se"]
        yearly["ci_upper"] = yearly["mean"] + 1.96 * yearly["se"]
        
        # Clip to valid TP range [0, 1]
        yearly["ci_lower"] = yearly["ci_lower"].clip(lower=0)
        yearly["ci_upper"] = yearly["ci_upper"].clip(upper=1)
        
        all_data[label] = yearly
    
    # Plot non-highlighted scenarios in grey (no CI)
    for label, yearly in all_data.items():
        if label not in highlights:
            ax.plot(yearly.index, yearly["mean"], 
                   color="#C8C8C8", lw=1, alpha=0.6, zorder=1)
    
    # Plot highlighted scenarios with 95% CI
    for label, cfg in highlights.items():
        if label in all_data:
            yearly = all_data[label]
            color = cfg["color"]
            
            # Mean line
            ax.plot(yearly.index, yearly["mean"],
                   color=color, lw=2.5, label=cfg["label"], zorder=3)
            
            # 95% CI band
            if show_ci:
                ax.fill_between(yearly.index, 
                               yearly["ci_lower"],
                               yearly["ci_upper"],
                               color=color, alpha=0.15, zorder=2)
    
    ax.set_xlabel("Year")
    ax.set_ylabel("Mean Threat Perception (TP)")
    ax.set_title(f"TP Trajectory - {group_name}\n(with 95% Confidence Interval)", fontweight="bold")
    ax.set_ylim(0, 1)
    ax.grid(True, ls="--", alpha=0.3)
    ax.legend(loc="upper left", frameon=True, framealpha=0.95)
    
    # Highlight severe flood years
    for y in [2011, 2014, 2021]:
        ax.axvspan(y-0.5, y+0.5, color="0.92", alpha=0.6, zorder=0)
    
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: {out_png}")


# ==============================================================================
# Plot 2: Financial Outcome Summary (Grouped Bar Chart)
# ==============================================================================

def plot_financial_summary_bars(
    scenarios: Dict[str, Path],
    highlights: Dict[str, Dict],
    out_png: Path,
    metric: str = "total_damage",  # "total_damage", "total_payout", "payout_rate"
):
    """
    Create grouped bar chart comparing financial metrics across highlighted scenarios.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Collect data for highlighted scenarios only
    data = {"Homeowner": {}, "Renter": {}}
    
    for label, path in scenarios.items():
        if label not in highlights:
            continue
            
        df = load_finance_summary(path)
        if df.empty:
            continue
        
        # Calculate totals (sum across all years and tracts)
        total_damage = (df["gross_structure_loss_kUSD"].sum() + 
                       df["gross_contents_loss_kUSD"].sum())
        total_payout = (df["payout_structure_kUSD"].sum() + 
                       df["payout_contents_kUSD"].sum())
        total_oop = (df["oop_structure_kUSD"].sum() + 
                    df["oop_contents_kUSD"].sum())
        
        if metric == "total_damage":
            data["Homeowner"][label] = total_damage / 1000  # Convert to $M
            data["Renter"][label] = total_damage / 1000  # Placeholder - need proper split
        elif metric == "total_payout":
            data["Homeowner"][label] = total_payout / 1000
            data["Renter"][label] = total_payout / 1000
        elif metric == "payout_rate":
            rate = total_payout / total_damage if total_damage > 0 else 0
            data["Homeowner"][label] = rate * 100
            data["Renter"][label] = rate * 100
    
    # Create bar positions
    labels = list(highlights.keys())
    x = np.arange(len(labels))
    width = 0.35
    
    values_owner = [data["Homeowner"].get(l, 0) for l in labels]
    colors = [highlights[l]["color"] for l in labels]
    
    bars = ax.bar(x, values_owner, width, color=colors, edgecolor="white", lw=1.5)
    
    # Labels
    ax.set_xlabel("Scenario")
    if metric == "total_damage":
        ax.set_ylabel("Total Damage ($M)")
        ax.set_title("Total Flood Damage by Scenario", fontweight="bold")
    elif metric == "total_payout":
        ax.set_ylabel("Total Insurance Payout ($M)")
        ax.set_title("Total Insurance Payout by Scenario", fontweight="bold")
    else:
        ax.set_ylabel("Payout Rate (%)")
        ax.set_title("Insurance Payout Rate by Scenario", fontweight="bold")
    
    ax.set_xticks(x)
    ax.set_xticklabels([highlights[l]["label"] for l in labels], rotation=15, ha="right")
    ax.grid(True, axis="y", ls="--", alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=9)
    
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: {out_png}")


# ==============================================================================
# Plot 3: Financial Outcome Heatmap (2D grid)
# ==============================================================================

def plot_financial_heatmap(
    scenarios: Dict[str, Path],
    out_png: Path,
    metric: str = "payout_rate",
):
    """
    Create 2D heatmap of financial metric across shock_homeowner × shock_renter grid.
    """
    import matplotlib.colors as mcolors
    
    # Extract values for each combination
    grid_data = {}
    for label, path in scenarios.items():
        if not label.startswith("NMG="): # or O= if still using old prefix
            if not label.startswith("O="):
                continue
        
        parts = label.split(",")
        o_val = float(parts[0].split("=")[1])
        r_val = float(parts[1].split("=")[1])
        
        df = load_finance_summary(path)
        if df.empty:
            continue
        
        total_damage = (df["gross_structure_loss_kUSD"].sum() + 
                       df["gross_contents_loss_kUSD"].sum())
        total_payout = (df["payout_structure_kUSD"].sum() + 
                       df["payout_contents_kUSD"].sum())
        
        if metric == "payout_rate":
            value = (total_payout / total_damage * 100) if total_damage > 0 else 0
        elif metric == "total_damage":
            value = total_damage / 1000
        else:
            value = total_payout / 1000
        
        grid_data[(o_val, r_val)] = value
    
    if not grid_data:
        print("[warn] No data for heatmap")
        return
    
    # Build grid
    o_vals = sorted(set(k[0] for k in grid_data.keys()))
    r_vals = sorted(set(k[1] for k in grid_data.keys()))
    
    matrix = np.zeros((len(r_vals), len(o_vals)))
    for i, r in enumerate(r_vals):
        for j, o in enumerate(o_vals):
            matrix[i, j] = grid_data.get((o, r), np.nan)
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    cmap = plt.cm.RdYlGn_r if metric == "total_damage" else plt.cm.RdYlGn
    im = ax.imshow(matrix, cmap=cmap, aspect="auto", origin="lower")
    
    # Labels
    ax.set_xticks(range(len(o_vals)))
    ax.set_xticklabels([f"{v:.1f}" for v in o_vals])
    ax.set_yticks(range(len(r_vals)))
    ax.set_yticklabels([f"{v:.1f}" for v in r_vals])
    ax.set_xlabel("NMG Shock Scale")
    ax.set_ylabel("MG Shock Scale")
    
    if metric == "payout_rate":
        ax.set_title("Insurance Payout Rate (%)\nby NMG/MG Shock Scale", fontweight="bold")
    elif metric == "total_damage":
        ax.set_title("Total Flood Damage ($M)\nby NMG/MG Shock Scale", fontweight="bold")
    else:
        ax.set_title("Total Insurance Payout ($M)\nby NMG/MG Shock Scale", fontweight="bold")
    
    # Add text annotations
    for i in range(len(r_vals)):
        for j in range(len(o_vals)):
            val = matrix[i, j]
            if not np.isnan(val):
                text_color = "white" if val > (matrix.max() + matrix.min()) / 2 else "black"
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", 
                       color=text_color, fontsize=9, fontweight="bold")
    
    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    if metric == "payout_rate":
        cbar.set_label("Payout Rate (%)")
    else:
        cbar.set_label("$M USD")
    
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: {out_png}")


# ==============================================================================
# Main Execution
# ==============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SA Results Visualization")
    print("=" * 60)
    
    # Get scenario directories
    shock_scenarios = get_scenario_dirs(SHOCK_SCALE_ROOT, "SA_SHOCK_*")
    print(f"Found {len(shock_scenarios)} Shock-Scale scenarios")
    
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. TP Trajectory plots
    print("\n--- Generating TP Trajectory Plots ---")
    plot_tp_trajectory_by_group(
        shock_scenarios, SHOCK_HIGHLIGHTS,
        FIG_DIR / "tp_trajectory_renters_MG.png",
        group="MG"
    )
    plot_tp_trajectory_by_group(
        shock_scenarios, SHOCK_HIGHLIGHTS,
        FIG_DIR / "tp_trajectory_homeowners_NMG.png",
        group="NMG"
    )
    
    # Note: Heatmaps and bar charts disabled per user request
    # Uncomment below to re-enable if needed
    # plot_financial_heatmap(shock_scenarios, FIG_DIR / "heatmap_payout_rate.png", metric="payout_rate")
    # plot_financial_heatmap(shock_scenarios, FIG_DIR / "heatmap_total_damage.png", metric="total_damage")
    # plot_financial_summary_bars(shock_scenarios, SHOCK_HIGHLIGHTS, FIG_DIR / "financial_summary_damage.png", metric="total_damage")
    
    print("\n" + "=" * 60)
    print("Done! Check outputs in:", FIG_DIR)
    print("=" * 60)
