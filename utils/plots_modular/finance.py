# -*- coding: utf-8 -*-
"""
Finance Plotting Module
=======================

Visualization functions for insurance and finance analysis.
Creates plots for premiums, payouts, and loss ratios.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .style import (
    set_paper_style, COLORS, COLOR_OWNER, COLOR_RENTER,
    panel_label, shade_severe_years, format_usd_axis,
)


def plot_finance_outputs(
    fin_dir: Path,
    fig_root: Path,
    *,
    severe_years: Optional[list[int]] = None,
) -> None:
    """Generate all finance-related plots.
    
    Args:
        fin_dir: Directory containing finance CSVs
        fig_root: Output directory for figures
        severe_years: Years to highlight as severe floods
    """
    fin_dir = Path(fin_dir)
    fig_root = Path(fig_root)
    fin_fig_dir = fig_root / "finance"
    fin_fig_dir.mkdir(parents=True, exist_ok=True)
    
    set_paper_style()
    
    # Load aggregated finance data
    tract_all = fin_dir / "finance_tract_all_years.csv"
    if tract_all.exists():
        df = pd.read_csv(tract_all)
        _plot_annual_payout_by_group(df, fin_fig_dir, severe_years)
        _plot_cumulative_payout(df, fin_fig_dir)
        print(f"[finance] Saved finance plots to {fin_fig_dir}")
    else:
        print(f"[finance] No aggregated finance data found at {tract_all}")


def _plot_annual_payout_by_group(
    df: pd.DataFrame,
    out_dir: Path,
    severe_years: Optional[list[int]] = None,
) -> None:
    """Plot annual payouts by homeowner/renter group.
    
    Args:
        df: Finance tract data with year, payout_owner_usd, payout_renter_usd
        out_dir: Output directory
        severe_years: Years to highlight
    """
    # Aggregate by year
    required_cols = {"year"}
    if not required_cols.issubset(df.columns):
        return
    
    # Find payout columns
    owner_col = next((c for c in df.columns if "owner" in c.lower() and "payout" in c.lower()), None)
    renter_col = next((c for c in df.columns if "renter" in c.lower() and "payout" in c.lower()), None)
    
    if not owner_col and not renter_col:
        return
    
    yearly = df.groupby("year").agg({
        col: "sum" for col in [owner_col, renter_col] if col
    }).reset_index()
    
    years = yearly["year"].values
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(years))
    width = 0.35
    
    if owner_col:
        ax.bar(x - width/2, yearly[owner_col] / 1e6, width, label="Homeowner", color=COLOR_OWNER)
    if renter_col:
        ax.bar(x + width/2, yearly[renter_col] / 1e6, width, label="Renter", color=COLOR_RENTER)
    
    # Shade severe years
    if severe_years:
        shade_severe_years(ax, years.tolist(), severe_years)
    
    ax.set_xlabel("Year")
    ax.set_ylabel("Payout (Million USD)")
    ax.set_title("Annual Insurance Payouts by Tenure Group")
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")
    
    fig.tight_layout()
    fig.savefig(out_dir / "annual_payout_by_group.png", dpi=300)
    plt.close(fig)


def _plot_cumulative_payout(df: pd.DataFrame, out_dir: Path) -> None:
    """Plot cumulative payouts over time.
    
    Args:
        df: Finance tract data
        out_dir: Output directory
    """
    if "year" not in df.columns:
        return
    
    # Find total payout column
    total_col = next((c for c in df.columns if "total" in c.lower() and "payout" in c.lower()), None)
    
    if not total_col:
        # Try to sum owner + renter
        owner_col = next((c for c in df.columns if "owner" in c.lower() and "payout" in c.lower()), None)
        renter_col = next((c for c in df.columns if "renter" in c.lower() and "payout" in c.lower()), None)
        
        if owner_col or renter_col:
            df = df.copy()
            df["_total"] = (df.get(owner_col, 0) if owner_col else 0) + (df.get(renter_col, 0) if renter_col else 0)
            total_col = "_total"
        else:
            return
    
    yearly = df.groupby("year")[total_col].sum().reset_index()
    yearly["cumulative"] = yearly[total_col].cumsum()
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.fill_between(yearly["year"], 0, yearly["cumulative"] / 1e6, alpha=0.3, color=COLORS["blue"])
    ax.plot(yearly["year"], yearly["cumulative"] / 1e6, color=COLORS["blue"], linewidth=2, marker="o")
    
    ax.set_xlabel("Year")
    ax.set_ylabel("Cumulative Payout (Million USD)")
    ax.set_title("Cumulative Insurance Payouts")
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(out_dir / "cumulative_payout.png", dpi=300)
    plt.close(fig)
