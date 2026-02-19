# -*- coding: utf-8 -*-
"""
Sensitivity Analysis and Data Reporting (SA_DR) Package
========================================================

Utilities for running sensitivity analyses and visualizing results.

Submodules:
    runner: Experiment runner infrastructure
    timeseries: Time-series data extraction (financial, payout, damage)
    collect: Data collection utilities
    plotting: Financial and comparison dashboards
    plotting_actions: Action share heatmaps and boxplots
    actions_core: Core action share calculations

Key Functions:
    - ts_financial_and_payout: Extract financial and payout time series
    - series_cum_financial: Cumulative financial cost series
    - plot_fin_focus_dashboard: 2-panel financial focus plot
    - plot_actions_heatmaps_mg_nmg: Action share heatmaps by MG/NMG

Example:
    >>> from utils.sa_dr import plot_fin_focus_dashboard, series_cum_financial
"""

from .runner import run_experiment

# Time-series metrics
from .timeseries import (
    ts_financial_and_payout,
    ts_flood_damage_total,
    series_cum_financial,
    series_cum_payout,
    series_cum_flood_damage,
    series_cum_premium_oop,
    final_cum_premium_oop,
    mean_over_years_flood_damage,
    mean_over_years_financial_and_payout,
    ts_premium_oop_avg_per_year,
    final_avg_premium_oop,
    totals_tradeoff_damage_payout_financial,
    dist_financial_per_household,
)

# Plotting: financial dashboards
from .plotting import plot_fin_focus_dashboard

# Plotting: action share visualizations
from .plotting_actions import (
    plot_actions_grouped_boxplot,
    plot_actions_heatmaps_mg_nmg,
)

__all__ = [
    # Runner
    "run_experiment",
    
    # Time-series metrics
    "ts_financial_and_payout",
    "ts_flood_damage_total",
    "series_cum_financial",
    "series_cum_payout",
    "series_cum_flood_damage",
    "series_cum_premium_oop",
    "final_cum_premium_oop",
    "mean_over_years_flood_damage",
    "mean_over_years_financial_and_payout",
    "ts_premium_oop_avg_per_year",
    "final_avg_premium_oop",
    "totals_tradeoff_damage_payout_financial",
    "dist_financial_per_household",
    
    # Plotting: dashboards
    "plot_fin_focus_dashboard",
    
    # Plotting: actions
    "plot_actions_grouped_boxplot",
    "plot_actions_heatmaps_mg_nmg",
]
