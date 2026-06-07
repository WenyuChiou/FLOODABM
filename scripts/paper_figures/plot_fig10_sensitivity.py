# -*- coding: utf-8 -*-
"""
Figure 10: Sensitivity to Ratio Threshold (2×2)
=================================================

(a) Owner θ sweep — Homeowner Δ cumulative damage/HH
(b) Renter θ sweep — Renter Δ cumulative damage/HH
(c) Owner θ sweep — Homeowner Δ cumulative actual loss/HH
(d) Renter θ sweep — Renter Δ cumulative actual loss/HH

9 gray spaghetti lines + colored extreme (θ=0.9) per panel.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from utils.plots_modular.style import (
    set_paper_style, shade_severe_years,
    COLORS, COLOR_OWNER, COLOR_RENTER,
)

def panel_label(ax, label="(a)", x=-0.12, y=1.02):
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=18, fontweight="bold", va="bottom", ha="right")

RT_GRID_DIR = Path("C:/temp/rt_grid_sa_v2")
FIG_DIR = Path(r"C:\Users\wenyu\OneDrive - Lehigh University\Desktop\Lehigh"
               r"\NSF-project\ABM\paper\Figure")

RT_INTS = [10, 20, 30, 40, 50, 60, 70, 80, 90]
SEVERE_YEARS = [2011, 2021]
YEARS = list(range(2011, 2024))


def _normalize(run_dir: Path) -> Path:
    for _ in range(5):
        b = run_dir / "baseline"
        if not b.exists():
            break
        if (b / "finance").exists():
            run_dir = b
            continue
        if (b / "baseline").exists():
            run_dir = b
            continue
        break
    return run_dir


def load_yearly(run_dir: Path) -> dict | None:
    rd = _normalize(run_dir)
    fin_dir = rd / "finance"
    if not fin_dir.exists():
        return None

    rows = []
    for f in sorted(fin_dir.glob("finance_tract_2*.csv")):
        m = re.search(r"(\d{4})", f.name)
        if not m:
            continue
        df = pd.read_csv(f, dtype={"tract_geoid": str})
        # Use direct per-tenure payout columns when available; fall back to
        # proportional split only if missing (older outputs).
        if "owner_payout_total_usd" in df.columns:
            owner_pay = df["owner_payout_total_usd"].sum()
            renter_pay = df["renter_payout_total_usd"].sum()
        else:
            tot_pay = df["payout_total_kUSD"].sum() * 1000
            ow_g = df["owner_gross_total_kUSD"].sum() * 1000
            re_g = df["renter_gross_total_kUSD"].sum() * 1000
            tot = ow_g + re_g
            owner_pay = tot_pay * (ow_g / tot if tot > 0 else 0.5)
            renter_pay = tot_pay * (re_g / tot if tot > 0 else 0.5)
        rows.append({
            "year": int(m.group(1)),
            "owner_gross": df["owner_gross_total_kUSD"].sum() * 1000,
            "renter_gross": df["renter_gross_total_kUSD"].sum() * 1000,
            "owner_payout": owner_pay,
            "renter_payout": renter_pay,
            "owner_hh": df["owner_households"].sum(),
            "renter_hh": df["renter_households"].sum(),
        })
    if not rows:
        return None

    yr_df = pd.DataFrame(rows).sort_values("year")
    o_pay = yr_df["owner_payout"].values
    r_pay = yr_df["renter_payout"].values
    o_hh = np.maximum(yr_df["owner_hh"].values, 1)
    r_hh = np.maximum(yr_df["renter_hh"].values, 1)

    return {
        "owner_cum_dmg": np.cumsum(yr_df["owner_gross"].values / o_hh),
        "renter_cum_dmg": np.cumsum(yr_df["renter_gross"].values / r_hh),
        "owner_cum_loss": np.cumsum((yr_df["owner_gross"].values - o_pay) / o_hh),
        "renter_cum_loss": np.cumsum((yr_df["renter_gross"].values - r_pay) / r_hh),
    }


def main():
    set_paper_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    ax_a, ax_b = axes[0]
    ax_c, ax_d = axes[1]

    x = np.arange(len(YEARS))
    bl = load_yearly(RT_GRID_DIR / "RTo010_RTr010")

    # ── Load runs ──
    owner_sweep = []
    for rt_int in RT_INTS:
        if rt_int == 10:
            continue
        r = load_yearly(RT_GRID_DIR / f"RTo{rt_int:03d}_RTr010")
        if r:
            owner_sweep.append((rt_int / 100, r))

    renter_sweep = []
    for rt_int in RT_INTS:
        if rt_int == 10:
            continue
        r = load_yearly(RT_GRID_DIR / f"RTo010_RTr{rt_int:03d}")
        if r:
            renter_sweep.append((rt_int / 100, r))

    panels = [
        (ax_a, owner_sweep, bl, "owner_cum_dmg", COLOR_OWNER, "Homeowner",
         "Change in cumulative\nGUL per HH ($K)", "\u03b8\u2095"),
        (ax_b, renter_sweep, bl, "renter_cum_dmg", COLOR_RENTER, "Renter",
         "Change in cumulative\nGUL per HH ($K)", "\u03b8\u1d63"),
        (ax_c, owner_sweep, bl, "owner_cum_loss", COLOR_OWNER, None,
         "Change in cumulative\nactual loss per HH ($K)", "\u03b8\u2095"),
        (ax_d, renter_sweep, bl, "renter_cum_loss", COLOR_RENTER, None,
         "Change in cumulative\nactual loss per HH ($K)", "\u03b8\u1d63"),
    ]

    for i, (ax, runs, baseline, metric, color, title, ylabel, theta_sym) in enumerate(panels):
        plbl = f"({'abcd'[i]})"
        shade_severe_years(ax, YEARS, SEVERE_YEARS)
        ax.axhline(0, color="black", lw=1.5, zorder=2)

        # Gray spaghetti
        for val, data in runs:
            delta = (data[metric] - baseline[metric]) / 1000
            if val == 0.9:
                continue  # plot last
            ax.plot(x, delta, "-", lw=0.8, color="0.75", alpha=0.7, zorder=1)

        # Extreme (θ=0.9) — solid colored
        for val, data in runs:
            if val == 0.9:
                delta = (data[metric] - baseline[metric]) / 1000
                marker = "-o" if "owner" in metric else "-s"
                ax.plot(x, delta, marker, ms=4, lw=2.2, color=color,
                        label=f"{theta_sym} = 0.9", zorder=4)
                # Endpoint label — beside final point, with % of baseline
                v = delta[-1]
                baseline_val = baseline[metric][-1] / 1000
                pct = (v / baseline_val * 100) if baseline_val != 0 else 0
                sign = "+" if v >= 0 else ""
                fmt = f"{sign}${v:.0f}K ({sign}{pct:.0f}%)" if abs(v) >= 1 \
                    else f"{sign}${v:.1f}K ({sign}{pct:.0f}%)"
                ax.annotate(fmt, xy=(x[-1], v),
                            xytext=(5, 0),
                            textcoords="offset points",
                            fontsize=15, fontweight="bold", color=color,
                            va="center", ha="left",
                            annotation_clip=False,
                            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                                      edgecolor=color, linewidth=0.6, alpha=0.95))

        # Baseline label
        ax.plot([], [], "-", color="black", lw=1.5, label="Adaptation scenario (\u03b8 = 0.10)")

        if title:
            ax.set_title(title, fontsize=19, fontweight="bold", pad=8)
        ax.set_ylabel(ylabel, fontsize=18)
        ax.set_xlabel("Year", fontsize=18)
        ax.set_xticks(x[::2])
        ax.set_xticklabels([str(y) for y in YEARS[::2]])
        ax.tick_params(axis="both", labelsize=15)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # Legend position: damage row lower-left; loss row upper-left
        # (renter actual loss now increases at high θ, so upper-left avoids
        # overlapping the rising trajectory)
        if "dmg" in metric:
            loc = "lower left"
        else:
            loc = "upper left"
        ax.legend(loc=loc, fontsize=14,
                  frameon=True, edgecolor="black", facecolor="white")
        panel_label(ax, plbl)

    fig.tight_layout(h_pad=3, w_pad=4)
    fig.subplots_adjust(right=0.92)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "pdf"]:
        out = FIG_DIR / f"fig10_sensitivity.{ext}"
        fig.savefig(out, dpi=600, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
