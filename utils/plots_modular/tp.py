# -*- coding: utf-8 -*-
"""
Trust in Policymaker (TP) Plotting Module
==========================================

Visualization functions for TP trajectory analysis.
Creates plots showing TP evolution over time for owner and renter groups.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .style import set_paper_style, COLORS, panel_label


def plot_tp_outputs(tp_traj: pd.DataFrame, fig_root: Path) -> None:
    """Generate all TP-related plots.
    
    Args:
        tp_traj: DataFrame with columns: year, tract_geoid, TP_owner, TP_renter, phase
        fig_root: Output directory for figures
    """
    if tp_traj.empty:
        print("[tp] No TP trajectory data, skipping plots")
        return
    
    fig_root = Path(fig_root)
    tp_dir = fig_root / "tp"
    tp_dir.mkdir(parents=True, exist_ok=True)
    
    set_paper_style()
    
    # Plot median with IQR for both groups
    _plot_median_iqr(tp_traj, "TP_owner", "Threat Perception (Owner)", tp_dir / "tp_owner_median_iqr.png")
    _plot_median_iqr(tp_traj, "TP_renter", "Threat Perception (Renter)", tp_dir / "tp_renter_median_iqr.png")

    # Plot area change
    _plot_area_change(tp_traj, "TP_owner", "TP Change Distribution (Owner)", tp_dir / "tp_owner_area_change.png")
    _plot_area_change(tp_traj, "TP_renter", "TP Change Distribution (Renter)", tp_dir / "tp_renter_area_change.png")
    
    # Combined plot
    _plot_combined_tp(tp_traj, tp_dir / "tp_combined.png")
    
    print(f"[tp] Saved TP plots to {tp_dir}")


def _plot_median_iqr(
    tp_traj: pd.DataFrame,
    col: str,
    title: str,
    out_path: Path,
) -> None:
    """Plot median TP with IQR band over time.
    
    Args:
        tp_traj: TP trajectory data
        col: Column to plot (TP_owner or TP_renter)
        title: Plot title
        out_path: Output file path
    """
    if col not in tp_traj.columns:
        return
    
    # Aggregate by year
    yearly = tp_traj.groupby("year")[col].agg(["median", lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)])
    yearly.columns = ["median", "q25", "q75"]
    years = yearly.index.values
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # IQR band
    ax.fill_between(years, yearly["q25"], yearly["q75"], alpha=0.3, color=COLORS["band"], label="IQR")
    
    # Median line
    ax.plot(years, yearly["median"], color=COLORS["median"], linewidth=2, marker="o", label="Median")
    
    ax.set_xlabel("Year")
    ax.set_ylabel(col.replace("_", " "))
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def _plot_area_change(
    tp_traj: pd.DataFrame,
    col: str,
    title: str,
    out_path: Path,
    q_low: int = 10,
    q_high: int = 90,
) -> None:
    """Plot TP distribution area over time.
    
    Args:
        tp_traj: TP trajectory data
        col: Column to plot
        title: Plot title
        out_path: Output path
        q_low: Lower percentile
        q_high: Upper percentile
    """
    if col not in tp_traj.columns:
        return
    
    # Aggregate by year with percentiles
    def agg_func(x):
        return pd.Series({
            "low": np.percentile(x, q_low),
            "mid": np.median(x),
            "high": np.percentile(x, q_high),
        })
    
    yearly = tp_traj.groupby("year")[col].apply(agg_func).unstack()
    years = yearly.index.values
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Percentile band
    ax.fill_between(years, yearly["low"], yearly["high"], alpha=0.3, color=COLORS["band"], 
                    label=f"{q_low}th-{q_high}th percentile")
    
    # Median line
    ax.plot(years, yearly["mid"], color=COLORS["median"], linewidth=2, label="Median")
    
    ax.set_xlabel("Year")
    ax.set_ylabel(col.replace("_", " "))
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def _plot_combined_tp(tp_traj: pd.DataFrame, out_path: Path) -> None:
    """Plot combined owner and renter TP trajectories.

    Args:
        tp_traj: TP trajectory data
        out_path: Output file path
    """
    if "TP_owner" not in tp_traj.columns or "TP_renter" not in tp_traj.columns:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, (col, group_name, color) in zip(axes, [
        ("TP_owner", "Owner (Homeowner)", COLORS["blue"]),
        ("TP_renter", "Renter", COLORS["green"]),
    ]):
        yearly = tp_traj.groupby("year")[col].agg(["median", lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)])
        yearly.columns = ["median", "q25", "q75"]
        years = yearly.index.values
        
        ax.fill_between(years, yearly["q25"], yearly["q75"], alpha=0.3, color=color)
        ax.plot(years, yearly["median"], color=color, linewidth=2, marker="o")
        
        ax.set_xlabel("Year")
        ax.set_ylabel("Trust in Policymaker")
        ax.set_title(f"{group_name}")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
    
    # Panel labels
    panel_label(axes[0], "(a)")
    panel_label(axes[1], "(b)")
    
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
