# -*- coding: utf-8 -*-
"""
Paper-ready action composition figures:
  Version A (main text): Stacked bar — yearly share (%), each year sums to 100%
  Version B (SI): Stacked bar — real household counts

Usage:
    python plot_action_composition_paper.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent.parent  # project root
sys.path.insert(0, str(ROOT))

from utils.plots_modular.style import set_paper_style, panel_label, COLORS

DECISIONS_DIR = ROOT / "outputs" / "baseline" / "baseline" / "decisions"
VIZ_DIR = ROOT / "outputs" / "baseline" / "baseline" / "visualization" / "decisions"
VIZ_DIR.mkdir(parents=True, exist_ok=True)

SEVERE_YEARS = [2011, 2021]

# Action colors — consistent, distinguishable
ACTION_COLORS = {
    "FI": "#3b82f6",   # blue
    "BP": "#f59e0b",   # amber
    "EH": "#22c55e",   # green
    "DN": "#a78bfa",   # purple
    "RL": "#ef4444",   # red
    "Protected": "#94a3b8",  # gray — already protected (cumulative BP+EH exits)
}

OWNER_ACTIONS = ["FI", "BP", "EH", "DN"]
RENTER_ACTIONS = ["FI", "RL", "DN"]


def load_data() -> pd.DataFrame:
    """Load tract-level decision data, aggregate to yearly totals."""
    files = sorted(DECISIONS_DIR.glob("action_share_owner_renter_tract_*.csv"))
    dfs = []
    for f in files:
        df = pd.read_csv(f, encoding="utf-8-sig")
        dfs.append(df)
    dec = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    if dec.empty:
        raise FileNotFoundError(f"No decision files in {DECISIONS_DIR}")
    return dec


def compute_yearly(dec: pd.DataFrame):
    """Compute yearly counts and shares for owner and renter."""
    years = sorted(dec["year"].unique())

    # Sum across tracts per year
    yearly_rows = []
    for yr in years:
        sub = dec[dec["year"] == yr]
        row = {"year": yr}
        # Owner counts (active decision pool)
        for a in OWNER_ACTIONS:
            row[f"owner_{a}"] = sub[f"owner_n_{a}"].sum()
        row["owner_active"] = sum(row[f"owner_{a}"] for a in OWNER_ACTIONS)
        # Renter counts
        for a in RENTER_ACTIONS:
            row[f"renter_{a}"] = sub[f"renter_n_{a}"].sum()
        row["renter_total"] = sum(row[f"renter_{a}"] for a in RENTER_ACTIONS)
        yearly_rows.append(row)

    df = pd.DataFrame(yearly_rows)

    # Owner total population = first year's active pool (before any exits)
    owner_pop = int(df.iloc[0]["owner_active"])
    # "Already protected" = cumulative BP+EH exits from decision pool
    df["owner_Protected"] = owner_pop - df["owner_active"]
    df["owner_total"] = owner_pop  # constant total population

    # Shares (%) — among active decision-makers only
    for a in OWNER_ACTIONS:
        df[f"owner_share_{a}"] = df[f"owner_{a}"] / df["owner_active"] * 100
    for a in RENTER_ACTIONS:
        df[f"renter_share_{a}"] = df[f"renter_{a}"] / df["renter_total"] * 100

    # Shares for headcount version (% of total pop including protected)
    for a in OWNER_ACTIONS:
        df[f"owner_pop_share_{a}"] = df[f"owner_{a}"] / owner_pop * 100
    df["owner_pop_share_Protected"] = df["owner_Protected"] / owner_pop * 100

    return df, years, owner_pop


def shade_severe(ax, years):
    for i, y in enumerate(years):
        if y in SEVERE_YEARS:
            ax.axvspan(i - 0.45, i + 0.45,
                       color=COLORS["severe_shade"], alpha=0.25, zorder=0,
                       clip_on=True)


def draw_stacked_bars(ax, df, years, group, actions, mode="share"):
    """
    Draw stacked bars.
    mode='share': Y = % (sums to 100%)
    mode='count': Y = household count
    """
    x = np.arange(len(years))
    width = 0.7
    bottom = np.zeros(len(years))

    for a in actions:
        if mode == "share":
            vals = df[f"{group}_share_{a}"].values
        else:
            vals = df[f"{group}_{a}"].values
        ax.bar(x, vals, width, bottom=bottom, color=ACTION_COLORS[a],
               label=a, edgecolor="white", linewidth=0.3)
        bottom += vals

    ax.set_xticks(x[::2])
    ax.set_xticklabels([str(y) for y in years[::2]], fontsize=9)
    ax.set_xlabel("Simulation Year")
    shade_severe(ax, years)


def main():
    set_paper_style()
    dec = load_data()
    df, years, owner_pop = compute_yearly(dec)

    # ── Version A: Share (%) among active decision-makers — main text ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    draw_stacked_bars(ax1, df, years, "owner", OWNER_ACTIONS, mode="share")
    ax1.set_ylabel("Action Share (%)")
    ax1.set_ylim(0, 100)
    ax1.set_title("Homeowner", fontsize=12)
    ax1.legend(fontsize=9, loc="upper right", ncol=2)
    panel_label(ax1, "(a)")

    draw_stacked_bars(ax2, df, years, "renter", RENTER_ACTIONS, mode="share")
    ax2.set_ylabel("Action Share (%)")
    ax2.set_ylim(0, 100)
    ax2.set_title("Renter", fontsize=12)
    ax2.legend(fontsize=9, loc="upper right")
    panel_label(ax2, "(b)")

    fig.tight_layout()
    out_a = VIZ_DIR / "action_composition_share_paper.png"
    fig.savefig(out_a, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_a}")

    # ── Version B: Headcount (full population) — SI ──
    # Owner: shows active pool {FI,BP,EH,DN} + "Already Protected" (gray)
    # so total bar height = constant 83K
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    owner_actions_full = OWNER_ACTIONS + ["Protected"]
    draw_stacked_bars(ax1, df, years, "owner", owner_actions_full, mode="count")
    ax1.set_ylabel("Households")
    ax1.set_title("Homeowner", fontsize=12)
    ax1.legend(fontsize=9, loc="center right", ncol=1)
    panel_label(ax1, "(a)")

    draw_stacked_bars(ax2, df, years, "renter", RENTER_ACTIONS, mode="count")
    ax2.set_ylabel("Households")
    ax2.set_title("Renter", fontsize=12)
    ax2.legend(fontsize=9, loc="upper right")
    panel_label(ax2, "(b)")

    fig.tight_layout()
    out_b = VIZ_DIR / "action_composition_counts_paper.png"
    fig.savefig(out_b, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_b}")


if __name__ == "__main__":
    main()
