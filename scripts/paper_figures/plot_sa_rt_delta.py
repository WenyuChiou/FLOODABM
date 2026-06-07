# -*- coding: utf-8 -*-
"""Plot SA RT grid: Δ from baseline with extreme combos labeled. Uses θ symbol.
Style matches RQ1/RQ2 figures via shared style module."""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import itertools
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent  # project root
sys.path.insert(0, str(ROOT))
from utils.plots_modular.style import set_paper_style, panel_label, shade_severe_years, COLORS

SA_ROOT = Path("C:/temp/rt_grid_sa")

RT_VALUES = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
BASELINE_RT_O = 0.10
BASELINE_RT_R = 0.10
YEARS = list(range(2011, 2024))
SEVERE_YEARS = [2011, 2021]

# Extreme combos to label (use mathtext for subscripts)
# Order: green first, blue second, red LAST (so red draws on top when overlapping)
EXTREME_COMBOS = {
    (0.90, 0.10): (r"$\theta_o$=0.90, $\theta_r$=0.10", "#2ca02c", "-o"),    # green solid
    (0.10, 0.90): (r"$\theta_o$=0.10, $\theta_r$=0.90", "#1f77b4", "-s"),    # blue square
    (0.90, 0.90): (r"$\theta_o$=0.90, $\theta_r$=0.90", "#d62728", "--D"),   # red dashed diamond
}


def _run_tag(rt_o, rt_r):
    return f"RTo{rt_o:.2f}_RTr{rt_r:.2f}".replace(".", "")


def _normalize_run_dir(run_dir):
    for _ in range(5):
        b = run_dir / "baseline"
        if not b.exists():
            break
        if (b / "finance").exists() or (b / "decisions").exists():
            run_dir = b
            continue
        if (b / "baseline").exists():
            run_dir = b
            continue
        break
    return run_dir


def load_scenario_data(run_dir):
    rd = _normalize_run_dir(run_dir)
    fin_frames = []
    for year in YEARS:
        fp = rd / "finance" / f"finance_tract_{year}.csv"
        if fp.exists():
            df = pd.read_csv(fp, dtype={"tract_geoid": str})
            df["year"] = year
            fin_frames.append(df)
    if not fin_frames:
        return pd.DataFrame()

    fin = pd.concat(fin_frames, ignore_index=True)
    fin_yr = fin.groupby("year").agg(
        owner_gross_k=("owner_gross_total_kUSD", "sum"),
        renter_gross_k=("renter_gross_total_kUSD", "sum"),
        payout_total_k=("payout_total_kUSD", "sum"),
        owner_hh=("owner_households", "sum"),
        renter_hh=("renter_households", "sum"),
    ).reset_index()

    dmg_path = rd / "vulnerability" / "flood_damage" / "flood_damage_tract_ALL_years.csv"
    if dmg_path.exists():
        dmg = pd.read_csv(dmg_path, dtype={"tract_geoid": str})
        dmg_yr = dmg.groupby("year").agg(
            owner_dmg=("owner_usd", "sum"),
            renter_dmg=("renter_usd", "sum"),
        ).reset_index()
    else:
        dmg_yr = fin_yr[["year"]].copy()
        dmg_yr["owner_dmg"] = fin_yr["owner_gross_k"] * 1000
        dmg_yr["renter_dmg"] = fin_yr["renter_gross_k"] * 1000

    merged = fin_yr.merge(dmg_yr, on="year", how="outer").fillna(0).sort_values("year")
    total_gross = merged["owner_gross_k"] + merged["renter_gross_k"]
    owner_share = np.where(total_gross > 0, merged["owner_gross_k"] / total_gross, 0.5)
    merged["owner_payout"] = merged["payout_total_k"] * owner_share * 1000
    merged["renter_payout"] = merged["payout_total_k"] * (1 - owner_share) * 1000
    return merged


def compute_cumulative(df, group):
    hh = np.maximum(df[f"{group}_hh"].values, 1)
    dmg = df[f"{group}_dmg"].values.astype(float)
    pay = df[f"{group}_payout"].values.astype(float)
    cum_dmg = np.cumsum(dmg / hh)
    cum_loss = np.cumsum((dmg - pay) / hh)
    return cum_dmg, cum_loss


def main():
    # Use the shared paper style (matches Fig 4, 5, 7, etc.)
    set_paper_style()

    combos = list(itertools.product(RT_VALUES, RT_VALUES))

    # Load all runs
    run_data = {}
    for rt_o, rt_r in combos:
        tag = _run_tag(rt_o, rt_r)
        d = SA_ROOT / tag
        if not d.exists():
            print(f"  [MISS] {tag}")
            continue
        scenario = load_scenario_data(d)
        if scenario.empty:
            continue
        o_dmg, o_loss = compute_cumulative(scenario, "owner")
        r_dmg, r_loss = compute_cumulative(scenario, "renter")
        run_data[(rt_o, rt_r)] = {
            "cum_owner_dmg": o_dmg,
            "cum_renter_dmg": r_dmg,
            "cum_owner_loss": o_loss,
            "cum_renter_loss": r_loss,
        }

    baseline_key = (BASELINE_RT_O, BASELINE_RT_R)
    if baseline_key not in run_data:
        print("ERROR: baseline not found")
        return

    bl = run_data[baseline_key]
    x = np.arange(len(YEARS))

    # Compute deltas
    delta_data = {}
    for key, data in run_data.items():
        if key == baseline_key:
            continue
        delta_data[key] = {
            metric: data[metric] - bl[metric]
            for metric in ["cum_owner_dmg", "cum_renter_dmg", "cum_owner_loss", "cum_renter_loss"]
        }

    # ── Plot ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    ax_a, ax_b = axes[0]
    ax_c, ax_d = axes[1]

    panels = [
        (ax_a, "cum_owner_dmg",  "Homeowner", "(a)"),
        (ax_b, "cum_renter_dmg", "Renter",    "(b)"),
        (ax_c, "cum_owner_loss", "Homeowner", "(c)"),
        (ax_d, "cum_renter_loss","Renter",    "(d)"),
    ]
    row_labels = {0: "Change in Cumulative GUL per HH", 1: "Change in Cumulative Actual Loss per HH"}

    for idx, (ax, metric, col_title, plbl) in enumerate(panels):
        row = idx // 2

        # Severe year shading (same style as RQ figures)
        shade_severe_years(ax, YEARS, SEVERE_YEARS)

        # Zero line (baseline)
        ax.axhline(0, color="black", lw=2.0, zorder=5, label=r"Baseline ($\theta$ = 0.10)")

        # Gray spaghetti for non-extreme runs
        for key, ddata in delta_data.items():
            if key in EXTREME_COMBOS:
                continue
            ax.plot(x, ddata[metric], color="#c0c0c0", lw=0.8, alpha=0.5, zorder=1)

        # Extreme combos (colored + labeled) — plot lines first
        endpoints = []  # collect (end_val, key, label_text, color)
        for key, (label, color, lstyle) in EXTREME_COMBOS.items():
            if key not in delta_data:
                continue
            vals = delta_data[key][metric]
            ax.plot(x, vals, lstyle, ms=5, lw=2.2, color=color, label=label, zorder=8)
            end_val = vals[-1]
            if abs(end_val) < 1000:
                sign = "+" if end_val >= 0 else ""
                lbl = f"{sign}${end_val:.0f}"
            else:
                sign = "+" if end_val >= 0 else ""
                lbl = f"{sign}${end_val/1000:.0f}K"
            endpoints.append((end_val, key, lbl, color))

        # Anti-overlap: sort DESCENDING by end_val so highest value gets
        # placed first (at top), then stagger downward for close values.
        endpoints.sort(key=lambda t: -t[0])
        min_gap_pts = 22
        placed_y = []  # (end_val, y_offset_pts)
        for end_val, key, lbl, color in endpoints:
            base_off = 12 if abs(end_val) < 2000 else 0
            y_off = base_off
            # Push DOWN if overlapping with previously placed (higher) annotation
            changed = True
            while changed:
                changed = False
                for prev_val, prev_off in placed_y:
                    if abs(end_val - prev_val) < 5000 and abs(y_off - prev_off) < min_gap_pts:
                        y_off = prev_off - min_gap_pts
                        changed = True
            placed_y.append((end_val, y_off))
            ax.annotate(
                lbl, xy=(x[-1], end_val),
                xytext=(8, y_off), textcoords="offset points",
                fontsize=11, fontweight="bold", color=color,
                ha="left", va="center", zorder=9,
            )

        # Title: column name only (row label goes on y-axis)
        ax.set_title(col_title, fontsize=14, fontweight="bold", pad=8)
        # Panel label — same style as other figures
        panel_label(ax, plbl)

        ax.set_xlabel("Year", fontsize=12)
        ax.set_ylabel(f"{row_labels[row]} ($)", fontsize=11)
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, p: "$0" if v == 0 else f"${v/1000:+.0f}K"))
        ax.set_xticks(x[::2])
        ax.set_xticklabels([str(y) for y in YEARS[::2]])

        # Right spine off (matching shared style); left/bottom on
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Top row (damage): lower-left; bottom row (actual loss): upper-left
        legend_loc = "lower left" if row == 0 else "upper left"
        ax.legend(
            loc=legend_loc, fontsize=10, frameon=True,
            facecolor="white", edgecolor="0.8", framealpha=0.95,
            borderpad=0.5, labelspacing=0.3,
        )

    fig.tight_layout(h_pad=2.5, w_pad=2.0)

    # Expand right margin for endpoint annotations
    fig.subplots_adjust(right=0.92)

    out = SA_ROOT / "sa_rt_grid_delta_2x2"
    fig.savefig(str(out) + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(str(out) + ".pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}.png/.pdf")


if __name__ == "__main__":
    main()
