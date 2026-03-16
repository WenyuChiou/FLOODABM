# -*- coding: utf-8 -*-
"""
FLOODABM Plotting Package
=========================

Modular visualization utilities organized by domain:
- style: Shared styling, colors, and formatting
- tp: Trust in Policymaker trajectory plots
- finance: Premium, payout, and loss visualizations
- flood: Flood damage and vulnerability plots
- comparison: Scenario comparison charts

The main entry point is `plot_all_outputs()` which orchestrates all plots.

Example:
    >>> from utils.plots import plot_all_outputs
    >>> plot_all_outputs(tp_traj_df, finance_dir, output_dir)
"""
from pathlib import Path
import pandas as pd

# Re-export main plotting functions for backward compatibility
from .style import set_paper_style, COLORS, LEGEND_FONTSIZE
from .tp import plot_tp_outputs
from .finance import plot_finance_outputs
from .flood import plot_flood_damage_outputs


def plot_all_outputs(
    tp_traj: pd.DataFrame,
    fin_dir: Path,
    vis_dir: Path,
    *,
    severe_years: list[int] = None,
) -> None:
    """Generate all visualization outputs.
    
    This is the main plotting entry point that creates all figures.
    
    Args:
        tp_traj: Trust trajectory DataFrame with year, tract_geoid, TP_owner, TP_renter
        fin_dir: Directory containing finance CSVs
        vis_dir: Output directory for visualizations
        severe_years: Optional list of severe flood years to highlight
    """
    vis_dir = Path(vis_dir)
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Apply paper style
    set_paper_style()
    
    # TP trajectory plots
    if not tp_traj.empty:
        plot_tp_outputs(tp_traj, vis_dir)
    
    # Finance plots (if data exists)
    if fin_dir and Path(fin_dir).exists():
        plot_finance_outputs(fin_dir, vis_dir, severe_years=severe_years)
    
    # Flood damage plots (handled by flood module)
    plot_flood_damage_outputs(vis_dir)


__all__ = [
    "plot_all_outputs",
    "set_paper_style",
    "COLORS",
    "plot_tp_outputs",
    "plot_finance_outputs",
    "plot_flood_damage_outputs",
]
