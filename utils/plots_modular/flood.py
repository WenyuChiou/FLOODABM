# -*- coding: utf-8 -*-
"""
Flood Damage Plotting Module
============================

Visualization functions for flood damage and vulnerability analysis.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .style import set_paper_style, COLORS, panel_label, format_usd_axis


def plot_flood_damage_outputs(fig_root: Path) -> None:
    """Generate flood damage plots if data exists.
    
    Looks for flood damage CSVs in the vulnerability/flood_damage subdirectory.
    
    Args:
        fig_root: Root output directory (e.g., outputs/baseline/visualization)
    """
    fig_root = Path(fig_root)
    
    # Check parent for vulnerability data
    vuln_dir = fig_root.parent / "vulnerability" / "flood_damage"
    if not vuln_dir.exists():
        return
    
    flood_fig_dir = fig_root / "flood"
    flood_fig_dir.mkdir(parents=True, exist_ok=True)
    
    set_paper_style()
    
    # Load aggregated flood damage
    all_years = vuln_dir / "flood_damage_tract_ALL_years.csv"
    if all_years.exists():
        df = pd.read_csv(all_years)
        _plot_annual_damage(df, flood_fig_dir)
        _plot_damage_by_group(df, flood_fig_dir)
        print(f"[flood] Saved flood damage plots to {flood_fig_dir}")


def _plot_annual_damage(df: pd.DataFrame, out_dir: Path) -> None:
    """Plot total flood damage by year.
    
    Args:
        df: Flood damage data with year and damage columns
        out_dir: Output directory
    """
    if "year" not in df.columns:
        return
    
    # Find damage columns
    damage_cols = [c for c in df.columns if "usd" in c.lower()]
    if not damage_cols:
        return
    
    # Use the 'both' or total column if available
    total_col = next((c for c in damage_cols if "both" in c.lower()), damage_cols[0])
    
    yearly = df.groupby("year")[total_col].sum().reset_index()
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.bar(yearly["year"], yearly[total_col] / 1e6, color=COLORS["red"], alpha=0.7)
    
    ax.set_xlabel("Year")
    ax.set_ylabel("Total Flood Damage (Million USD)")
    ax.set_title("Annual Flood Damage")
    ax.grid(True, alpha=0.3, axis="y")
    
    fig.tight_layout()
    fig.savefig(out_dir / "annual_flood_damage.png", dpi=300)
    plt.close(fig)


def _plot_damage_by_group(df: pd.DataFrame, out_dir: Path) -> None:
    """Plot flood damage by homeowner/renter group.
    
    Args:
        df: Flood damage data
        out_dir: Output directory
    """
    if "year" not in df.columns:
        return
    
    owner_col = next((c for c in df.columns if "owner" in c.lower()), None)
    renter_col = next((c for c in df.columns if "renter" in c.lower()), None)
    
    if not owner_col or not renter_col:
        return
    
    yearly = df.groupby("year").agg({
        owner_col: "sum",
        renter_col: "sum",
    }).reset_index()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    years = yearly["year"].values
    x = np.arange(len(years))
    width = 0.35
    
    ax.bar(x - width/2, yearly[owner_col] / 1e6, width, label="Homeowner", color=COLORS["owner"])
    ax.bar(x + width/2, yearly[renter_col] / 1e6, width, label="Renter", color=COLORS["renter"])
    
    ax.set_xlabel("Year")
    ax.set_ylabel("Flood Damage (Million USD)")
    ax.set_title("Flood Damage by Group")
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")
    
    fig.tight_layout()
    fig.savefig(out_dir / "flood_damage_by_group.png", dpi=300)
    plt.close(fig)
