"""
Plotting Style - Shared style configuration and helper functions.

This module provides:
- Color palette (COLORS) and marker styles (MARKERS)
- _set_style(): Apply consistent matplotlib rcParams
- _panel_label(): Add panel labels like (a), (b)
- _style_axis(): Style individual axes
- _usd_fmt(): Format values as USD
- _rename_threshold_label(): Convert threshold to tau notation
- _add_severe_year_bands(): Add grey bands for severe flood years
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

__all__ = [
    "COLORS",
    "MARKERS",
    "_set_style",
    "_panel_label",
    "_style_axis",
    "_usd_fmt",
    "_rename_threshold_label",
    "_add_severe_year_bands",
]

# Color palette for plots
COLORS = ["#2563eb", "#ef4444", "#10b981", "#f59e0b", "#8b5cf6"]
MARKERS = ["o", "s", "D", "^", "v"]


def _set_style():
    """Apply consistent matplotlib style settings."""
    plt.rcParams.update({
        "figure.dpi": 300, 
        "savefig.dpi": 300,
        "font.size": 20,
        "axes.titlesize": 22,
        "axes.titleweight": "bold",
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


def _panel_label(ax, label: str = "(a)", x: float = -0.12, y: float = 1.02):
    """Add a panel label (e.g., '(a)', '(b)') to an axes."""
    ax.text(x, y, label, transform=ax.transAxes, fontsize=14, fontweight="bold",
            ha="left", va="bottom")


def _style_axis(ax: plt.Axes):
    """Apply consistent style to an axes object."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=5, width=1)


def _usd_fmt(x, _):
    """Format value as USD with appropriate suffix (K, M, B)."""
    absx = abs(x)
    if absx >= 1_000_000_000:
        return f"${x/1_000_000_000:.1f}B"
    if absx >= 1_000_000:
        return f"${x/1_000_000:.1f}M"
    if absx >= 1_000:
        return f"${x/1_000:.1f}K"
    return f"${x:.0f}"


def _rename_threshold_label(raw: str) -> str:
    """
    Convert threshold format to tau notation.
    
    Example: 'MG=0.2,NMG=0.3' -> 'τ_th(MG)=0.2, τ_th(NMG)=0.3'
    """
    lab = raw.strip()
    if lab.startswith("MG=") and "NMG=" in lab:
        try:
            mg = lab.split(",")[0].split("=")[1]
            nmg = lab.split(",")[1].split("=")[1]
            return f"τ_th(MG)={mg}, τ_th(NMG)={nmg}"
        except Exception:
            return raw
    return raw


def _add_severe_year_bands(
    ax: plt.Axes,
    years: List[int] | Tuple[int, ...],
    ymin=None,
    ymax=None
):
    """Add grey vertical bands for severe flood years."""
    if not years:
        return
    y0, y1 = ax.get_ylim() if (ymin is None or ymax is None) else (ymin, ymax)
    for y in years:
        ax.axvspan(y - 0.5, y + 0.5, color="0.9", alpha=0.6, linewidth=0)
    ax.set_ylim(y0, y1)
