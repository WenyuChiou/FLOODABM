# -*- coding: utf-8 -*-
"""
Plotting Style Configuration
=============================

Shared styling, colors, and formatting functions for all FLOODABM plots.
Uses a consistent paper-ready style with customizable palettes.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from typing import Optional


# =============================================================================
# Color Palettes
# =============================================================================

COLORS = {
    # Primary colors for owner/renter distinction
    "owner": "#3b82f6",      # Blue-500
    "renter": "#22c55e",     # Green-500
    
    # Secondary colors
    "median": "#7c3aed",     # Violet-600
    "mean": "#0ea5e9",       # Sky-500
    "band": "#a78bfa",       # Violet-300
    
    # Flood severity
    "severe_shade": "#93c5fd",  # Blue-200
    "overflow": "#f59e0b",      # Amber-500
    
    # General palette (colorblind-friendly)
    "blue": "#0077BB",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "brown": "#8B6B4F",
    "gray": "#999999",
    "yellow": "#F0E442",
    "black": "#000000",
}

# Named shortcuts
COLOR_OWNER = COLORS["owner"]
COLOR_RENTER = COLORS["renter"]
COLOR_MEDIAN = COLORS["median"]
COLOR_MEAN = COLORS["mean"]
COLOR_BAND = COLORS["band"]


# =============================================================================
# Typography and Layout
# =============================================================================

LEGEND_FONTSIZE = 9
TITLE_FONTSIZE = 14
LABEL_FONTSIZE = 12
TICK_FONTSIZE = 11

LEGEND_AUTO_INSIDE = True  # Place legends inside axes with best location


# =============================================================================# Style Functions
# =============================================================================

def set_paper_style() -> None:
    """Apply publication-ready matplotlib style.
    
    Sets font sizes, line widths, and other parameters for
    clean, professional-looking figures.
    """
    plt.rcParams.update({
        # Font family — serif for journal consistency
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],

        # Font sizes
        "font.size": 12,
        "axes.titlesize": TITLE_FONTSIZE,
        "axes.labelsize": LABEL_FONTSIZE,
        "xtick.labelsize": TICK_FONTSIZE,
        "ytick.labelsize": TICK_FONTSIZE,
        "legend.fontsize": LEGEND_FONTSIZE,  # 9pt for readability
        
        # Line properties
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
        
        # Figure properties
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
        
        # Grid
        "axes.grid": False,
        "grid.alpha": 0.3,
        
        # Spines
        "axes.spines.top": False,
        "axes.spines.right": False,
        
        # Legend
        "legend.framealpha": 0.9,
        "legend.edgecolor": "0.8",
    })


def panel_label(ax, label: str = "(a)", x: float = -0.12, y: float = 1.02) -> None:
    """Add panel label (a), (b), etc. to subplot.
    
    Args:
        ax: Matplotlib axes
        label: Label text
        x: X position in axes coords
        y: Y position in axes coords
    """
    ax.text(
        x, y, label,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        va="bottom",
        ha="right",
    )


def create_legend_patch(color: str, label: str) -> Patch:
    """Create a colored patch for legend entries.
    
    Args:
        color: Fill color
        label: Legend text
        
    Returns:
        matplotlib Patch object
    """
    return Patch(facecolor=color, edgecolor="none", label=label)


def severe_year_patch(label: str = "Severe flood year") -> Patch:
    """Create a patch for severe flood year legend entry."""
    return Patch(facecolor=COLORS["severe_shade"], alpha=0.3, label=label)


def shade_severe_years(
    ax,
    years: list,
    severe_years: list[int],
    *,
    half_width: float = 0.48,
    color: Optional[str] = None,
    alpha: float = 0.25,
) -> None:
    """Add shaded bars for severe flood years.
    
    Args:
        ax: Matplotlib axes
        years: List of all years on x-axis
        severe_years: Years to highlight (can be 2-digit or 4-digit)
        half_width: Bar half-width
        color: Fill color (default: severe_shade)
        alpha: Fill transparency
    """
    if not severe_years:
        return
    
    color = color or COLORS["severe_shade"]
    
    # Normalize severe years to 4-digit
    sev_set = set()
    for y in severe_years:
        if y < 100:
            sev_set.add(2000 + y)
        else:
            sev_set.add(y)
    
    for i, yr in enumerate(years):
        if yr in sev_set:
            ax.axvspan(i - half_width, i + half_width, color=color, alpha=alpha, zorder=0)


def format_usd_axis(ax, axis: str = "y", decimals: int = 0) -> None:
    """Format axis ticks as USD currency.
    
    Args:
        ax: Matplotlib axes
        axis: "x" or "y"
        decimals: Decimal places
    """
    from matplotlib.ticker import FuncFormatter
    
    def formatter(x, pos):
        if abs(x) >= 1e6:
            return f"${x/1e6:.{decimals}f}M"
        elif abs(x) >= 1e3:
            return f"${x/1e3:.{decimals}f}K"
        else:
            return f"${x:.{decimals}f}"
    
    if axis == "y":
        ax.yaxis.set_major_formatter(FuncFormatter(formatter))
    else:
        ax.xaxis.set_major_formatter(FuncFormatter(formatter))


def format_pct_axis(ax, axis: str = "y", decimals: int = 0) -> None:
    """Format axis ticks as percentages.
    
    Args:
        ax: Matplotlib axes
        axis: "x" or "y"
        decimals: Decimal places
    """
    from matplotlib.ticker import FuncFormatter
    
    formatter = FuncFormatter(lambda x, _: f"{x:.{decimals}%}")
    
    if axis == "y":
        ax.yaxis.set_major_formatter(formatter)
    else:
        ax.xaxis.set_major_formatter(formatter)
