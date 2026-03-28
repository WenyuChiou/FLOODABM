# -*- coding: utf-8 -*-
"""
Publication-quality FI by Flood-Prone Status figure.
Replaces the auto-generated fi_by_flood_prone.png with a cleaner version.

Usage:
    python plot_fi_by_flood_prone_paper.py
"""
from __future__ import annotations
import sys, yaml
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import openpyxl

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent.parent  # project root
sys.path.insert(0, str(ROOT))

from utils.plots_modular.style import (
    set_paper_style, panel_label, COLORS, COLOR_OWNER, COLOR_RENTER,
)

# ── paths ────────────────────────────────────────────────────────────────
DECISIONS_DIR = (ROOT / "outputs" / "baseline" / "baseline" / "decisions")
VIZ_DIR = (ROOT / "outputs" / "baseline" / "baseline" / "visualization"
           / "decisions")
CONFIG = ROOT / "config" / "abm_params.yaml"
SEVERE_YEARS = [2011, 2021]

COLOR_FP = "#d62828"      # red — flood-prone
COLOR_NFP = "#0077b6"     # blue — non-flood-prone (colorblind-safe)


def load_decisions() -> pd.DataFrame:
    """Load tract-level decision shares for all years."""
    files = sorted(DECISIONS_DIR.glob("action_share_owner_renter_tract_*.csv"))
    dfs = []
    for f in files:
        df = pd.read_csv(f, encoding="utf-8-sig")
        dfs.append(df)
    dec = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    if dec.empty:
        raise FileNotFoundError(f"No decision files in {DECISIONS_DIR}")
    dec["tract_geoid"] = dec["tract_geoid"].astype(str)
    return dec


def get_flood_prone_tracts() -> list[str]:
    """Identify flood-prone tracts (owner init rate = 0.25 in config)."""
    with open(CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    rates = cfg.get("insurance_init", {}).get("take_rate_by_tract_group", {})
    return [str(t) for t, v in rates.items() if v.get("owner") == 0.25]


def main():
    set_paper_style()

    dec = load_decisions()
    fp_tracts = get_flood_prone_tracts()
    dec["flood_prone"] = dec["tract_geoid"].isin(fp_tracts)
    years = sorted(dec["year"].unique())

    # ── 2×2 figure ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # ── (a) Owner FI by flood-prone status ───────────────────────────────
    ax = axes[0, 0]
    for fp, label, color, ls in [
        (True,  "Flood-prone",     COLOR_FP,  "-"),
        (False, "Non-flood-prone", COLOR_NFP, "-"),
    ]:
        sub = dec[dec["flood_prone"] == fp]
        fi = sub.groupby("year")["owner_share_FI"].mean() * 100
        ax.plot(fi.index, fi.values, ls, color=color, linewidth=2,
                label=label)
    for sy in SEVERE_YEARS:
        ax.axvspan(sy - 0.4, sy + 0.4,
                   color=COLORS["severe_shade"], alpha=0.25, zorder=0)
    ax.set_ylabel("FI Adoption (%)")
    ax.set_title("Homeowner", fontsize=12)
    ax.set_ylim(0, 80)
    ax.legend(fontsize=9, loc="upper right")
    panel_label(ax, "(a)")

    # ── (b) Renter FI by flood-prone status ──────────────────────────────
    ax = axes[0, 1]
    for fp, label, color in [
        (True,  "Flood-prone",     COLOR_FP),
        (False, "Non-flood-prone", COLOR_NFP),
    ]:
        sub = dec[dec["flood_prone"] == fp]
        fi = sub.groupby("year")["renter_share_FI"].mean() * 100
        ax.plot(fi.index, fi.values, "-", color=color, linewidth=2,
                label=label)
    for sy in SEVERE_YEARS:
        ax.axvspan(sy - 0.4, sy + 0.4,
                   color=COLORS["severe_shade"], alpha=0.25, zorder=0)
    ax.set_title("Renter", fontsize=12)
    ax.set_ylim(0, 80)
    ax.legend(fontsize=9, loc="upper right")
    panel_label(ax, "(b)")

    # ── (c) Event-study: FI trajectory around floods (indexed to t-1=0%) ─
    # Window: t-1 (baseline=0%), t (flood year), t+1, t+2
    # Average across 2014 and 2021 events → shows sustained vs short-lived
    ax = axes[1, 0]
    FLOOD_EVENTS = [2021]
    WINDOW = [-1, 0, 1, 2]  # relative to flood year
    groups_c = [
        (True,  "owner_share_FI",  "FP Owner",   COLOR_FP,  "-",  "o"),
        (True,  "renter_share_FI", "FP Renter",  COLOR_FP,  "--", "s"),
        (False, "owner_share_FI",  "NFP Owner",  COLOR_NFP, "-",  "o"),
        (False, "renter_share_FI", "NFP Renter", COLOR_NFP, "--", "s"),
    ]
    def iqr_mean(values):
        """Remove outliers outside [Q1-1.5*IQR, Q3+1.5*IQR], then mean."""
        if len(values) < 4:
            return np.mean(values)
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        filtered = [v for v in values if lo <= v <= hi]
        return np.mean(filtered) if filtered else np.mean(values)

    excel_rows = []
    for fp, col, label, color, ls, marker in groups_c:
        sub = dec[dec["flood_prone"] == fp]
        tracts = sub["tract_geoid"].unique()
        avg_trajectory = []
        for offset in WINDOW:
            event_rates = []
            for sy in FLOOD_EVENTS:
                base_yr = sy - 1
                target_yr = sy + offset
                if base_yr not in years or target_yr not in years:
                    continue
                # Per-tract change rates, then IQR-filter before averaging
                tract_rates = []
                for tid in tracts:
                    t = sub[sub["tract_geoid"] == tid]
                    b = t[t["year"] == base_yr][col].values
                    v = t[t["year"] == target_yr][col].values
                    if len(b) > 0 and len(v) > 0 and b[0] != 0:
                        tract_rates.append((v[0] - b[0]) / b[0] * 100)
                rate = iqr_mean(tract_rates) if tract_rates else 0
                n_before = len(tract_rates)
                q1, q3 = (np.percentile(tract_rates, [25, 75])
                          if len(tract_rates) >= 4 else (np.nan, np.nan))
                iqr_val = q3 - q1 if pd.notna(q1) else np.nan
                excel_rows.append({
                    "group": label, "flood_year": sy,
                    "offset": offset, "year": target_yr,
                    "n_tracts": n_before,
                    "mean_rate_%": round(rate, 2),
                    "Q1": round(q1, 2) if pd.notna(q1) else "",
                    "Q3": round(q3, 2) if pd.notna(q3) else "",
                    "IQR": round(iqr_val, 2) if pd.notna(iqr_val) else "",
                })
                event_rates.append(rate)
            avg_trajectory.append(np.mean(event_rates) if event_rates else 0)
        ax.plot(WINDOW, avg_trajectory, ls, color=color, marker=marker,
                ms=6, lw=2, label=label)

    ax.axhline(0, color="black", linewidth=0.5, zorder=0)
    ax.axvline(0, color="gray", linewidth=0.8, ls=":", alpha=0.6, zorder=0)
    ax.set_xticks(WINDOW)
    ax.set_xticklabels(["t\u22121\n(base)", "t\n(flood)", "t+1", "t+2"],
                       fontsize=10)
    ax.set_ylabel("\u0394FI from Baseline (%)", fontsize=11)
    ax.set_xlabel("Relative Year", fontsize=11)
    ax.set_title("FI Response Around 2021 Flood (IQR-trimmed)", fontsize=12)
    ax.legend(fontsize=9, loc="lower left")
    panel_label(ax, "(c)")

    # ── Export Excel: summary sheet ──
    excel_path = VIZ_DIR / "fi_change_rate_after_floods.xlsx"
    excel_df = pd.DataFrame(excel_rows)

    # ── Tract-level table for ArcGIS Pro ──
    # Includes t-1(base), t(flood), t+1, t+2 for each event × owner/renter
    tract_rows = []
    for tract_id in dec["tract_geoid"].unique():
        t = dec[dec["tract_geoid"] == tract_id]
        fp_flag = tract_id in fp_tracts
        row = {"tract_geoid": tract_id, "flood_prone": int(fp_flag)}
        for sy in FLOOD_EVENTS:
            base_yr = sy - 1
            for col, prefix in [("owner_share_FI", "owner"), ("renter_share_FI", "renter")]:
                b = t[t["year"] == base_yr][col].values
                base_val = float(b[0]) * 100 if len(b) > 0 else np.nan
                row[f"{prefix}_base_{sy}"] = round(base_val, 2) if pd.notna(base_val) else np.nan
                # t, t+1, t+2 values and change rates
                for offset in [0, 1, 2]:
                    yr = sy + offset
                    v = t[t["year"] == yr][col].values
                    val = float(v[0]) * 100 if len(v) > 0 else np.nan
                    if pd.notna(base_val) and base_val != 0 and pd.notna(val):
                        rate = (val - base_val) / base_val * 100
                    else:
                        rate = np.nan
                    suffix = f"t{'+' + str(offset) if offset > 0 else ''}_{sy}"
                    row[f"{prefix}_{suffix}"] = round(val, 2) if pd.notna(val) else np.nan
                    row[f"{prefix}_chg_{suffix}"] = round(rate, 2) if pd.notna(rate) else np.nan
        # Average change rate at t+1 across 2 events (main metric for mapping)
        for prefix in ["owner", "renter"]:
            vals = [row.get(f"{prefix}_chg_t+1_{sy}") for sy in FLOOD_EVENTS]
            vals = [v for v in vals if pd.notna(v)]
            row[f"{prefix}_chg_t1_avg"] = round(np.mean(vals), 2) if vals else np.nan
            vals2 = [row.get(f"{prefix}_chg_t+2_{sy}") for sy in FLOOD_EVENTS]
            vals2 = [v for v in vals2 if pd.notna(v)]
            row[f"{prefix}_chg_t2_avg"] = round(np.mean(vals2), 2) if vals2 else np.nan
        tract_rows.append(row)

    tract_df = pd.DataFrame(tract_rows).sort_values("tract_geoid")

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        excel_df.to_excel(writer, index=False, sheet_name="Summary")
        tract_df.to_excel(writer, index=False, sheet_name="Tract_ArcGIS")
    print(f"Excel \u2192 {excel_path} (2 sheets: Summary + Tract_ArcGIS)")

    # ── (d) All groups FI trend comparison ───────────────────────────────
    ax = axes[1, 1]
    groups = [
        (True,  "owner_share_FI",  "FP Homeowner",     COLOR_FP,  "-",  2.0),
        (True,  "renter_share_FI", "FP Renter",        COLOR_FP,  "--", 1.5),
        (False, "owner_share_FI",  "Non-FP Homeowner", COLOR_NFP, "-",  2.0),
        (False, "renter_share_FI", "Non-FP Renter",    COLOR_NFP, "--", 1.5),
    ]
    for fp, col, label, color, ls, lw in groups:
        sub = dec[dec["flood_prone"] == fp]
        fi = sub.groupby("year")[col].mean() * 100
        ax.plot(fi.index, fi.values, ls, color=color, linewidth=lw,
                label=label)
    for sy in SEVERE_YEARS:
        ax.axvspan(sy - 0.4, sy + 0.4,
                   color=COLORS["severe_shade"], alpha=0.25, zorder=0)
    ax.set_ylabel("FI Adoption (%)", fontsize=11)
    ax.set_xlabel("Simulation Year", fontsize=11)
    ax.set_title("All Groups Comparison", fontsize=12)
    ax.set_ylim(0, 80)
    ax.legend(fontsize=9, loc="upper right", ncol=2)
    panel_label(ax, "(d)")

    # shared x-label for top row
    axes[0, 0].set_xlabel("Simulation Year")
    axes[0, 1].set_xlabel("Simulation Year")

    fig.tight_layout()
    out = VIZ_DIR / "fi_by_flood_prone.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
