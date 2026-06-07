# -*- coding: utf-8 -*-
"""
Generate bivariate classification (1-9) per tract for Figure 9.

Two variables:
  X = 2020 baseline FI share       (low/mid/high: <30%, 30-70%, >70%)
  Y = 2023 delta FI share          (decrease/neutral/increase: <-3pp, -3 to +3pp, >+3pp)

Outputs:
  1. outputs/fig9/Fig9_bivariate_classes.csv
     columns: tract_geoid, owner_bivariate_2023, renter_bivariate_2023,
              plus original baseline + delta values for verification
  2. outputs/fig9/Fig9_bivariate_legend.png (+.pdf) — 3x3 legend for ArcGIS layout
  3. outputs/fig9/Fig9_bivariate_palette.md — color hex table for ArcGIS symbology

Class labeling convention (1..9):
  Row = Delta  (1 bottom = decrease,  3 top = increase)
  Col = Baseline (1 left = low,      3 right = high)

  Class = (Delta_row - 1) * 3 + Baseline_col

           Baseline
           Low   Mid   High
  Delta ↑
  Increase  7     8     9      (top row)
  Neutral   4     5     6      (mid row)
  Decrease  1     2     3      (bottom row)
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.stdout.reconfigure(encoding="utf-8")

INPUT_CSV = Path(r"C:\Users\wenyu\OneDrive - Lehigh University\Desktop\Lehigh"
                 r"\NSF-project\ABM\paper\Figure\spatial_data"
                 r"\Fig9_spatial_fi_delta_50run_median.csv")
OUT_DIR = Path(__file__).resolve().parent.parent.parent / "outputs" / "fig9"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Cutoffs
BASE_CUTS = [0.30, 0.70]        # in fraction, i.e. 30% / 70%
DELTA_CUTS = [-0.03, 0.03]      # in fraction, i.e. -3pp / +3pp

# Bivariate palette — Stevens-inspired systematic gradient
#   X axis (baseline) light -> dark left to right
#   Y axis (delta): red (decrease) - gray (neutral) - blue (increase)
#   Right column (high baseline) always deepest, anchoring the saturation signature
PALETTE = {
    # row 1: decrease (red family)
    1: "#f3b4b4",   # low-base, decrease        light pink
    2: "#c876b0",   # mid-base, decrease        medium magenta
    3: "#5a3a7a",   # high-base, decrease       deep purple
    # row 2: neutral (vivid green family — unchanged tracts)
    4: "#d4f0c8",   # low-base, neutral         lime mint
    5: "#7bc94a",   # mid-base, neutral         vivid lime
    6: "#2e8b1f",   # high-base, neutral        vivid deep green (SATURATION KEY)
    # row 3: increase (blue family)
    7: "#b4d4f0",   # low-base, increase        light blue
    8: "#7090c8",   # mid-base, increase        medium blue
    9: "#3a4a9a",   # high-base, increase       deep blue-indigo
}

LABELS = {
    1: "Low base · Decrease",
    2: "Mid base · Decrease",
    3: "High base · Decrease",
    4: "Low base · Neutral",
    5: "Mid base · Neutral",
    6: "High base · Neutral (saturated)",
    7: "Low base · Increase",
    8: "Mid base · Increase",
    9: "High base · Increase",
}


def baseline_class(val: float) -> int:
    """Return 1/2/3 for low/mid/high baseline."""
    if np.isnan(val):
        return 0
    if val < BASE_CUTS[0]:
        return 1
    if val < BASE_CUTS[1]:
        return 2
    return 3


def delta_class(val: float) -> int:
    """Return 1/2/3 for decrease/neutral/increase."""
    if np.isnan(val):
        return 0
    if val < DELTA_CUTS[0]:
        return 1
    if val < DELTA_CUTS[1]:
        return 2
    return 3


def bivariate_class(base_val: float, delta_val: float) -> int:
    bc = baseline_class(base_val)
    dc = delta_class(delta_val)
    if bc == 0 or dc == 0:
        return 0
    # class = (delta_row - 1) * 3 + baseline_col
    return (dc - 1) * 3 + bc


def build_classes(df: pd.DataFrame) -> pd.DataFrame:
    """Compute two families of bivariate classes:

    (A) Cumulative delta — each post-event year relative to the 2020 baseline.
        Columns: owner_bivariate_{2021,2022,2023}, renter_bivariate_{2021,2022,2023}

    (B) Year-over-year delta — each post-event year relative to the previous year.
        Columns: owner_bivariate_yoy_{2021,2022,2023}, renter_bivariate_yoy_{2021,2022,2023}

    In both families, the baseline axis of the bivariate class is the 2020 FI
    share (so the saturation / pre-condition dimension is consistent). Only the
    delta axis differs.
    """
    out = pd.DataFrame({"tract_geoid": df["tract_geoid"]})
    for tenure in ["owner", "renter"]:
        base = df[f"{tenure}_fi_2020"]
        out[f"{tenure}_baseline_2020"] = (base * 100).round(1)

        # --- (A) Cumulative delta (year vs 2020) ---
        for year in [2021, 2022, 2023]:
            delta = df[f"{tenure}_delta_{year}"]
            out[f"{tenure}_delta_{year}_pp"] = (delta * 100).round(1)
            out[f"{tenure}_bivariate_{year}"] = [
                bivariate_class(b, d) for b, d in zip(base, delta)]

        # --- (B) Year-over-year delta (year vs previous year) ---
        for year in [2021, 2022, 2023]:
            prev_year = year - 1
            yoy = df[f"{tenure}_fi_{year}"] - df[f"{tenure}_fi_{prev_year}"]
            out[f"{tenure}_delta_yoy_{year}_pp"] = (yoy * 100).round(1)
            out[f"{tenure}_bivariate_yoy_{year}"] = [
                bivariate_class(b, d) for b, d in zip(base, yoy)]
    return out


def draw_legend_png():
    fig, ax = plt.subplots(figsize=(4.5, 4.5))

    # Draw 3x3 color grid
    for delta_row in range(1, 4):           # 1 bottom, 3 top
        for base_col in range(1, 4):        # 1 left, 3 right
            cls = (delta_row - 1) * 3 + base_col
            color = PALETTE[cls]
            # matplotlib y axis origin bottom-left, so we use delta_row - 1 directly
            rect = Rectangle(
                (base_col - 1, delta_row - 1), 1, 1,
                facecolor=color, edgecolor="white", linewidth=1.5)
            ax.add_patch(rect)
            # class code in center
            ax.text(base_col - 0.5, delta_row - 0.5, str(cls),
                    ha="center", va="center",
                    color="white", fontsize=14, fontweight="bold")

    # Axes labels
    ax.set_xlim(-0.05, 3.05)
    ax.set_ylim(-0.05, 3.3)

    # x-axis (baseline)
    ax.set_xticks([0.5, 1.5, 2.5])
    ax.set_xticklabels(["< 30%", "30–70%", "> 70%"], fontsize=10)
    ax.set_xlabel("Pre-event (2020) FI share", fontsize=12, fontweight="bold")

    # y-axis (delta)
    ax.set_yticks([0.5, 1.5, 2.5])
    ax.set_yticklabels([
        "Δ < −3 pp\n(decrease)",
        "|Δ| ≤ 3 pp\n(neutral)",
        "Δ > +3 pp\n(increase)",
    ], fontsize=10)
    ax.set_ylabel("Change in FI share, 2023 vs 2020", fontsize=12, fontweight="bold")

    ax.set_aspect("equal")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title("Bivariate legend — Figure 9",
                 fontsize=12, fontweight="bold", pad=12)

    fig.tight_layout()
    for ext in ["png", "pdf"]:
        out = OUT_DIR / f"Fig9_bivariate_legend.{ext}"
        fig.savefig(out, dpi=600, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.close(fig)


def _hex_to_rgb(hex_code: str) -> str:
    """#rrggbb -> 'r, g, b'."""
    h = hex_code.lstrip("#")
    return f"{int(h[0:2], 16)}, {int(h[2:4], 16)}, {int(h[4:6], 16)}"


def write_palette_md():
    md = OUT_DIR / "Fig9_bivariate_palette.md"
    lines = []
    lines.append("# Figure 9 - Bivariate palette (for ArcGIS Unique Values symbology)\n")
    lines.append("- Baseline (2020 FI share) cutoffs: **30%, 70%**")
    lines.append("- Delta (each year vs 2020) cutoffs: **-3 pp, +3 pp**")
    lines.append("- Palette design: Stevens-inspired. Both axes progress from light to dark.")
    lines.append("  - X axis (baseline): light -> dark left to right")
    lines.append("  - Y axis (delta): red -> gray -> blue from bottom to top")
    lines.append("  - Class 6 (high baseline + neutral delta) is the **saturation signature**.")
    lines.append("")
    lines.append("| Class | Baseline | Delta | Hex | RGB | Semantic |")
    lines.append("|---|---|---|---|---|---|")
    for cls in range(1, 10):
        dc = (cls - 1) // 3 + 1
        bc = (cls - 1) % 3 + 1
        base_lab = {1: "Low (<30%)", 2: "Mid (30-70%)", 3: "High (>70%)"}[bc]
        delta_lab = {1: "Decrease (<-3pp)", 2: "Neutral (+/-3pp)", 3: "Increase (>+3pp)"}[dc]
        hex_code = PALETTE[cls]
        rgb = _hex_to_rgb(hex_code)
        lines.append(
            f"| {cls} | {base_lab} | {delta_lab} | "
            f"`{hex_code}` | {rgb} | {LABELS[cls]} |")
    lines.append("")
    lines.append("## Available bivariate columns in the CSV")
    lines.append("")
    lines.append("The CSV provides **two families** of bivariate class per tract per tenure.")
    lines.append("Both families share the same 2020 baseline (X axis) and +/-3 pp delta cutoff (Y axis);")
    lines.append("only the delta definition differs.")
    lines.append("")
    lines.append("### Family A - Cumulative delta (year vs 2020)")
    lines.append("Best for: \"how much has each tract changed since the pre-event baseline?\"")
    lines.append("- `owner_bivariate_2021` / `renter_bivariate_2021`   (delta = 2021 - 2020)")
    lines.append("- `owner_bivariate_2022` / `renter_bivariate_2022`   (delta = 2022 - 2020)")
    lines.append("- `owner_bivariate_2023` / `renter_bivariate_2023`   (delta = 2023 - 2020)")
    lines.append("")
    lines.append("### Family B - Year-over-year delta (year vs previous year)")
    lines.append("Best for: \"which years had incremental change?\" - highlights event-driven dynamics,")
    lines.append("useful when you want to show that most change happened in 2021 and later years are stable.")
    lines.append("- `owner_bivariate_yoy_2021` / `renter_bivariate_yoy_2021`   (delta = 2021 - 2020)")
    lines.append("- `owner_bivariate_yoy_2022` / `renter_bivariate_yoy_2022`   (delta = 2022 - 2021)")
    lines.append("- `owner_bivariate_yoy_2023` / `renter_bivariate_yoy_2023`   (delta = 2023 - 2022)")
    lines.append("")
    lines.append("(For 2021 the two families are identical by construction: 2021 - 2020.)")
    lines.append("")
    lines.append("## ArcGIS step-by-step")
    lines.append("")
    lines.append("1. Join `Fig9_bivariate_classes.csv` to your tract layer on `tract_geoid`.")
    lines.append("2. Right-click tract layer -> **Symbology** -> **Unique Values**.")
    lines.append("3. Value Field: pick the year/tenure you want to display, e.g. `owner_bivariate_2021`.")
    lines.append("4. Click **Add All Values** - classes 1 through 9 will appear (some years may only use a subset).")
    lines.append("5. For each class, double-click the color swatch and enter the hex or RGB value from the table above.")
    lines.append("   - In ArcGIS Pro: Format Symbol -> Color -> More Colors -> paste hex directly.")
    lines.append("   - In ArcMap: More Colors -> RGB tab -> paste R, G, B numbers.")
    lines.append("6. Remove or hide any class 0 symbol (tracts with missing data).")
    lines.append("7. Set `<all other values>` to transparent or light gray.")
    lines.append("8. Duplicate the layer for the other tenure / year combinations you need.")
    lines.append("9. Import `Fig9_bivariate_legend.png` into the layout as an image element.")
    lines.append("")
    lines.append("## Recommended figure layouts")
    lines.append("")
    lines.append("- **Minimal (1x2)**: Homeowner 2023 + Renter 2023 only. Simplest, best for main-text space.")
    lines.append("- **Full time series (2x3)**: Owner/Renter rows x 2021/2022/2023 cols, similar to current Fig 9 but with bivariate encoding. Same information as before plus baseline dimension.")
    lines.append("- **Hybrid**: Main text = 1x2 (2023 only); SM = 2x3 or 2x2 (2021/2022) for readers who want the time series.")
    lines.append("")
    lines.append("All layouts share the same bivariate legend (one instance is enough).")
    md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {md}")


def main():
    df = pd.read_csv(INPUT_CSV, dtype={"tract_geoid": str})
    cls_df = build_classes(df)

    out_csv = OUT_DIR / "Fig9_bivariate_classes.csv"
    cls_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"Saved: {out_csv}")

    # Print class counts for sanity check
    print("\n=== Class counts — Cumulative delta (vs 2020 baseline) ===")
    for tenure in ["owner", "renter"]:
        for year in [2021, 2022, 2023]:
            vc = cls_df[f"{tenure}_bivariate_{year}"].value_counts().sort_index()
            print(f"  {tenure} {year}: {dict(vc)}")

    print("\n=== Class counts — Year-over-year delta (vs previous year) ===")
    for tenure in ["owner", "renter"]:
        for year in [2021, 2022, 2023]:
            vc = cls_df[f"{tenure}_bivariate_yoy_{year}"].value_counts().sort_index()
            print(f"  {tenure} yoy-{year}: {dict(vc)}")

    draw_legend_png()
    write_palette_md()


if __name__ == "__main__":
    main()
