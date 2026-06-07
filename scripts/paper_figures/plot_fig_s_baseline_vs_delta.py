# -*- coding: utf-8 -*-
"""
Supplementary Figure — Baseline FI share (2020) vs post-event change (2023).

Shows the ceiling effect directly:
- Tracts with 2020 FI share near 100% cluster at the right with delta ~ 0
  (no room to rise).
- Strongest positive delta concentrates at mid-baseline tracts (~30-60%).
- Non-prone tracts cluster tightly at low baseline with small deltas.

One panel per tenure group (homeowner / renter). Flood-prone vs non-prone
distinguished by marker shape. Points colored by 2023 delta sign.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8")

CSV = Path(r"C:\Users\wenyu\OneDrive - Lehigh University\Desktop\Lehigh"
           r"\NSF-project\ABM\paper\Figure\spatial_data"
           r"\Fig9_spatial_fi_delta_50run_median.csv")
FP_CSV = Path(__file__).resolve().parent.parent.parent \
    / "outputs" / "baseline" / "baseline" / "flood_impact_by_tract.csv"
OUT_DIR = Path(r"C:\Users\wenyu\OneDrive - Lehigh University\Desktop\Lehigh"
               r"\NSF-project\ABM\paper\Figure")

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from utils.plots_modular.style import set_paper_style, panel_label


def main():
    set_paper_style()

    df = pd.read_csv(CSV, dtype={"tract_geoid": str})
    fp = pd.read_csv(FP_CSV, dtype={"tract_geoid": str})[
        ["tract_geoid", "flood_prone"]]
    df = df.merge(fp, on="tract_geoid", how="left")

    # Convert to percentages / percentage points for display
    df["owner_fi_2020_pct"] = df["owner_fi_2020"] * 100
    df["renter_fi_2020_pct"] = df["renter_fi_2020"] * 100
    df["owner_delta_2023_pp"] = df["owner_delta_2023"] * 100
    df["renter_delta_2023_pp"] = df["renter_delta_2023"] * 100

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))

    for ax, tenure, plbl in [
        (axes[0], "owner", "(a) Homeowner"),
        (axes[1], "renter", "(b) Renter"),
    ]:
        x_col = f"{tenure}_fi_2020_pct"
        y_col = f"{tenure}_delta_2023_pp"

        # Reference lines
        ax.axhline(0, color="0.6", lw=1.0, zorder=1)
        ax.axvspan(90, 100, color="0.92", zorder=0,
                   label="Ceiling zone (baseline ≥ 90%)")

        prone = df[df["flood_prone"] == 1]
        nonp = df[df["flood_prone"] == 0]

        ax.scatter(nonp[x_col], nonp[y_col],
                   s=65, marker="o", facecolor="#d6d6d6",
                   edgecolor="#555555", lw=0.8, zorder=3,
                   label="Non-prone tract")
        ax.scatter(prone[x_col], prone[y_col],
                   s=90, marker="D", facecolor="#c72c48",
                   edgecolor="black", lw=0.9, zorder=4,
                   label="Flood-prone tract")

        # Label the most extreme tracts by GEOID (last 4 digits)
        cutoff_high = 5.0 if tenure == "owner" else 15.0
        cutoff_low = -5.0 if tenure == "owner" else -10.0
        for _, r in df.iterrows():
            val = r[y_col]
            if abs(val) >= cutoff_high or val <= cutoff_low:
                ax.annotate(
                    str(r["tract_geoid"])[-4:],
                    xy=(r[x_col], val),
                    xytext=(5, 4), textcoords="offset points",
                    fontsize=8, color="0.2")

        ax.set_xlim(0, 105)
        ax.set_xlabel("Pre-event FI share, 2020 (%)", fontsize=12)
        ax.set_ylabel("Change in FI share, 2023 vs 2020 (pp)", fontsize=12)
        ax.set_title(plbl, fontsize=13, fontweight="bold", pad=8)
        ax.grid(alpha=0.25, zorder=0)
        ax.legend(loc="upper right", fontsize=9, frameon=True,
                  facecolor="white", edgecolor="0.7")
        panel_label(ax, plbl.split(" ")[0])

    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "pdf"]:
        out = OUT_DIR / f"FigS_baseline_vs_delta_2023.{ext}"
        fig.savefig(out, dpi=600, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
