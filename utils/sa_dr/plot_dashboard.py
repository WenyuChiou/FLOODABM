"""
Plotting Dashboard - Main financial dashboards and rate delta plots.

This module provides:
- plot_fin_focus_dashboard(): 2-panel financial cost and payout dashboard
- plot_rate_delta_cum_dual(): Rate delta vs baseline (cumulative)
- _plot_series_on_ax(): Helper for multi-line plots
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.patches as mpatches

from .plot_style import (
    COLORS, MARKERS,
    _set_style, _panel_label, _style_axis, _usd_fmt,
    _rename_threshold_label, _add_severe_year_bands,
)

__all__ = [
    "plot_fin_focus_dashboard",
    "plot_rate_delta_cum_dual",
    "_plot_series_on_ax",
]


def _plot_series_on_ax(
    ax: plt.Axes,
    series_by_label: Dict[str, Tuple[List[int], List[float]]],
    xlabel: str,
    ylabel: str,
    title: str,
    legend_inside: bool = True,
    legend_loc: str = "upper left",
    show_markers: bool = True,
    format_labels: bool = True,
):
    """Plot multiple (years, values) series on the same axes."""
    _style_axis(ax)
    for i, (lab, (xs, ys)) in enumerate(series_by_label.items()):
        lab2 = _rename_threshold_label(lab) if format_labels else lab
        ax.plot(
            xs, ys,
            color=COLORS[i % len(COLORS)],
            marker=MARKERS[i % len(MARKERS)] if show_markers else None,
            markersize=6.5,
            linewidth=2.6,
            label=lab2,
        )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_formatter(FuncFormatter(_usd_fmt))
    if legend_inside and series_by_label:
        ax.legend(
            loc=legend_loc, frameon=True, facecolor="white",
            framealpha=0.95, edgecolor="#e5e7eb", title="Scenarios (TP-decay thresholds)"
        )


def plot_fin_focus_dashboard(
    series_fin: Dict[str, Tuple[List[int], List[float]]],
    series_pay: Dict[str, Tuple[List[int], List[float]]],
    trade_points: Dict[str, Tuple[float, float]] | None = None,
    trade_bubbles: Dict[str, float] | None = None,
    out_png: Path | str = "SA_RT_financial_focus.png",
    subtitle: Optional[str] = None,
    figsize: Tuple[float, float] = (13.5, 10.5),
    legend_loc: str = "upper left",
    trade_title: str = "",
    trade_xlabel: str = "",
    trade_ylabel: str = "",
    panel_labels: Tuple[str, str] = ("(a)", "(b)"),
    severe_years: Tuple[int, ...] = (2011, 2014, 2021),
    format_legend_labels: bool = True,
):
    """
    Generate a 2-panel financial dashboard:
      (a) Cumulative avg household financial cost (premium+OOP)
      (b) Cumulative average payout
      
    Features:
      - Automatic tau notation for legend labels
      - Grey bands for severe flood years
    """
    _set_style()
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    fig, axs = plt.subplots(2, 1, figsize=figsize, constrained_layout=True)

    # (a) cumulative avg household financial cost
    _plot_series_on_ax(
        axs[0], series_fin,
        xlabel="Simulation year",
        ylabel="USD",
        title="Cumulative avg household financial cost (prem+OOP)",
        legend_inside=True, legend_loc=legend_loc, show_markers=True,
        format_labels=format_legend_labels,
    )
    if subtitle:
        axs[0].text(0.01, 1.02, subtitle, transform=axs[0].transAxes,
                    fontsize=12, ha="left", va="bottom")
    _add_severe_year_bands(axs[0], severe_years)

    # (b) cumulative average payout
    _plot_series_on_ax(
        axs[1], series_pay,
        xlabel="Simulation year",
        ylabel="USD",
        title="Cumulative average payout",
        legend_inside=True, legend_loc=legend_loc, show_markers=True,
        format_labels=format_legend_labels,
    )
    _add_severe_year_bands(axs[1], severe_years)

    _panel_label(axs[0], panel_labels[0])
    _panel_label(axs[1], panel_labels[1])

    fig.savefig(out_png, dpi=240)
    plt.close(fig)


def plot_rate_delta_cum_dual(
    series_fin: Dict[str, Tuple[List[int], List[float]]],
    series_pay: Dict[str, Tuple[List[int], List[float]]],
    damage_by_year: Dict[int, float],
    *,
    baseline: str = "MG=0.5,NMG=0.5",
    highlights: Optional[Dict[str, Dict]] = None,
    severe_years: Tuple[int, ...] = (2011, 2014, 2021),
    out_png: Path | str = "outputs/experiments/fig/rate_delta_cum_dual.png",
    as_percent: bool = True,
    legend_loc: str = "best",
) -> None:
    """
    Make a 2-panel figure showing rate delta vs baseline (cumulative).
    
    Left: Financial cost rate Δ vs baseline
    Right: Payout rate Δ vs baseline
    
    Features:
      - Non-highlighted scenarios in grey
      - Baseline as black dashed line
      - Severe flood years as grey bands
      - Optional percentage mode
    """

    def _cumavg_to_annual(xs: List[int], cumavg: List[float]) -> Tuple[List[int], List[float]]:
        xs = list(xs)
        ca = np.asarray(cumavg, dtype=float)
        n = np.arange(len(ca)) + 1
        totals = ca * n
        annual = np.empty_like(totals)
        annual[0] = totals[0]
        if len(totals) > 1:
            annual[1:] = totals[1:] - totals[:-1]
        return xs, annual.tolist()

    def _rate_map(series: Dict[str, Tuple[List[int], List[float]]]) -> Dict[str, Tuple[List[int], List[float]]]:
        out: Dict[str, Tuple[List[int], List[float]]] = {}
        for lab, (yrs, vals) in series.items():
            ys, annual_vals = _cumavg_to_annual(yrs, vals)
            rates: List[float] = []
            for y, v in zip(ys, annual_vals):
                d = damage_by_year.get(int(y), np.nan)
                rates.append(v / d if (d not in (None, 0) and np.isfinite(d)) else np.nan)
            out[lab] = (ys, rates)
        return out

    def _delta_cumsum(rate_map: Dict[str, Tuple[List[int], List[float]]]) -> Dict[str, Tuple[List[int], List[float]]]:
        if baseline not in rate_map:
            raise ValueError(f"Baseline '{baseline}' not found in provided series.")
        by, bv = rate_map[baseline]
        base = dict(zip(by, bv))

        out: Dict[str, Tuple[List[int], List[float]]] = {}
        for lab, (ys, rs) in rate_map.items():
            deltas = []
            for y, r in zip(ys, rs):
                b = base.get(y, np.nan)
                deltas.append(r - b if np.isfinite(b) else np.nan)
            arr = np.array(deltas, dtype=float)
            inc = np.where(np.isfinite(arr), arr, 0.0)
            cs = np.cumsum(inc).tolist()
            out[lab] = (ys, cs)
        return out

    fin_delta_cum = _delta_cumsum(_rate_map(series_fin))
    pay_delta_cum = _delta_cumsum(_rate_map(series_pay))

    def _to_percent(mp: Dict[str, Tuple[List[int], List[float]]]) -> Dict[str, Tuple[List[int], List[float]]]:
        if not as_percent:
            return mp
        out = {}
        for k, (ys, vs) in mp.items():
            out[k] = (ys, [v * 100 if np.isfinite(v) else np.nan for v in vs])
        return out

    fin_delta_cum = _to_percent(fin_delta_cum)
    pay_delta_cum = _to_percent(pay_delta_cum)

    # Plotting
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.2), constrained_layout=True)
    panels = [
        (fin_delta_cum, "Financial cost rate Δ vs baseline — cumulative over years"),
        (pay_delta_cum, "Payout rate Δ vs baseline — cumulative over years"),
    ]

    for ax, (mp, title) in zip(axs, panels):
        # Plot grey others first
        for lab, (yrs, vs) in mp.items():
            if (highlights and lab in highlights) or lab == baseline:
                continue
            ax.plot(yrs, vs, color="#BBBBBB", lw=1.1, alpha=0.7, zorder=1)

        # Baseline
        if baseline in mp:
            yrs_b, vs_b = mp[baseline]
            ax.plot(yrs_b, vs_b, color="#111111", lw=1.6, ls="--", alpha=0.9, label="Baseline", zorder=2)

        # Highlights
        if highlights:
            for lab, cfg in highlights.items():
                if lab not in mp:
                    continue
                color = cfg.get("color") or cfg.get("c") or "#000"
                label = cfg.get("label", lab.replace(",", ", "))
                yrs, vs = mp[lab]
                ax.plot(yrs, vs, color=color, lw=2.6, label=label, zorder=3)
                if len(yrs) and len(vs) and np.isfinite(vs[-1]):
                    ax.annotate(label, (yrs[-1], vs[-1]), xytext=(6, 0),
                                textcoords="offset points", va="center", fontsize=10, color=color)

        # Zero line
        ax.axhline(0, color="0.55", lw=1.0, zorder=0)

        # Severe flood bands
        band_handle = None
        for y in severe_years:
            patch = ax.axvspan(y - 0.5, y + 0.5, color="0.92", alpha=0.8, lw=0, zorder=0)
            band_handle = patch
        handles, labels = ax.get_legend_handles_labels()
        if band_handle is not None:
            band = mpatches.Patch(facecolor="0.92", edgecolor="none", alpha=0.8, label="Severe flood years")
            handles = [band] + handles
            labels = ["Severe flood years"] + labels

        ax.set_title(title)
        ax.set_xlabel("Year")
        ylabel = "Cumulative difference vs baseline (%)" if as_percent else "Cumulative ( (value / D$_t$) − (baseline / D$_t$) )"
        ax.set_ylabel(ylabel)
        if as_percent:
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}%"))
        ax.grid(True, ls="--", alpha=0.2)
        ax.legend(handles, labels, loc=legend_loc, framealpha=0.95)

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
