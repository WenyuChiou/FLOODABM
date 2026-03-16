
# -*- coding: utf-8 -*-
"""
plots.py — Centralized visualization utilities for the ABM project.

UPDATE (beautified + 10× finer bins + overflow bins):
- Household premium histograms now use 10× smaller bin widths:
    * Homeowner: $50 bins, range [0, 5,000] + an overflow bin "5,000+"
    * Renter: $20 bins, range [0, 1,000] + an overflow bin "1,000+"
- Aesthetics improved (fonts, grid, line widths, colors) for BOTH premium charts and TP charts.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple, List
import numpy as np
import pandas as pd
import numpy as np
import pandas as pd

# Lazy loading for matplotlib to improve startup time
# import matplotlib.pyplot as plt
# from matplotlib.ticker import PercentFormatter
# from matplotlib.patches import Patch
# from matplotlib.ticker import FuncFormatter
# from matplotlib.lines import Line2D
# from matplotlib.gridspec import GridSpec
# import matplotlib.ticker as mtick



# ---- Paper style helpers ----
# Standard figure size for publications (consistent across all plots)
STANDARD_FIGSIZE = (12, 6)       # Single panel
STANDARD_FIGSIZE_2PANEL = (16, 12)  # Increased width (+4) and height

# Two standard sizes requested by user
FIGSIZE_PAPER = (15, 12)
FIGSIZE_POSTER = (16, 10)

def _set_style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 600,          # High resolution for publications
        "savefig.dpi": 600,
        "font.size": 20,            # Increased from 18 (+2pt)
        "axes.titlesize": 22,       # Increased from 20 (+2pt)
        "axes.labelsize": 20,       # Increased from 18 (+2pt)
        "axes.titleweight": "bold",
        "xtick.labelsize": 18,      # Increased from 16 (+2pt)
        "ytick.labelsize": 18,      # Increased from 16 (+2pt)
        "legend.fontsize": 18,      # Increased from 16 (+2pt)
        "axes.linewidth": 1.0,      # Slightly thicker for print
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.grid": True,
        "grid.alpha": 0.25,         # Slightly more visible grid
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
        # Can switch to serif (Times); will auto fallback if font not found
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "CMU Serif"],
        # Note: constrained_layout removed - conflicts with colorbar in some plots
    })


def _panel_label(ax, label="(a)", x=-0.10, y=1.02):
    """Label subplots with (a) (b) ... at top-left corner"""
    ax.text(x, y, label, transform=ax.transAxes, fontsize=14, fontweight="bold",
            ha="left", va="bottom")

SEV_SHADE = "#93c5fd"  # severe-year shade color

# Legend behavior toggles
LEGEND_AUTO_INSIDE = True  # place legends inside axes with best automatic location
LEGEND_FONTSIZE = 14       # Unified legend font size

def _legend_box_right(fig, handles, labels=None, *, title=None,
                      anchor=(1.01, 0.5), ncol=1):
    """Unified legend helper.
    If LEGEND_AUTO_INSIDE is True, place legend inside the last axes using loc='best'.
    Otherwise, keep the outside-right boxed legend.
    """
    if LEGEND_AUTO_INSIDE and fig.axes:
        ax = fig.axes[-1]
        lg = ax.legend(handles=handles, labels=labels, loc="best",
                       frameon=True, ncol=ncol, title=title,
                       fontsize=LEGEND_FONTSIZE, fancybox=False, 
                       framealpha=1.0, facecolor='white', edgecolor='black')
        return lg
    # fallback: box at right of the figure
    lg = fig.legend(handles=handles, labels=labels,
                    loc="center left", bbox_to_anchor=anchor,
                    frameon=True, ncol=ncol, title=title,
                    fontsize=LEGEND_FONTSIZE)
    fr = lg.get_frame()
    fr.set_facecolor("white"); fr.set_edgecolor("black"); fr.set_linewidth(0.8)
    return lg

def _severe_patch(label: str = "Severe flood year"):
    from matplotlib.patches import Patch
    return Patch(facecolor=SEV_SHADE, alpha=0.20, edgecolor="none", label=label)

# Palette
COLOR_OWNER  = "#cff720"  # blue-600
COLOR_RENTER = "#7ef7aa"  # green-600
COLOR_MEDIAN = "#7c3aed"  # violet-600
COLOR_BAND   = "#a78bfa"  # violet-300
COLOR_MEAN   = "#0ea5e9"  # sky-500

# ------------------------- directory helpers -------------------------

def _ensure_fig_dirs(fig_root: Path) -> Dict[str, Path]:
    sub = {
        "action":        fig_root / "action",
        "finance":       fig_root / "finance",
        "vulnerability": fig_root / "vulnerability",
        "decisions":     fig_root / "decisions",  # NEW: for behavioral analysis plots
    }
    sub["finance_premium"] = sub["finance"] / "premium_hist"

    for p in sub.values():
        p.mkdir(parents=True, exist_ok=True)
    return sub

def _ensure_vuln_damage_dir(fig_root: Path) -> Path:
    out_dir = fig_root / "vulnerability" / "damage"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


# ------------------------- TP (Action module) -------------------------

def _plot_median_iqr(tp_traj: pd.DataFrame, col: str, title: str, out_path: Path) -> None:
    if tp_traj is None or tp_traj.empty or col not in tp_traj.columns:
        return
    _set_style()
    years = sorted(tp_traj["year"].unique())
    med, q1, q3 = [], [], []
    for y in years:
        v = tp_traj.loc[tp_traj["year"] == y, col].to_numpy(float)
        if v.size == 0:
            med.append(np.nan); q1.append(np.nan); q3.append(np.nan)
        else:
            med.append(np.nanmedian(v))
            q1.append(np.nanpercentile(v, 25))
            q3.append(np.nanpercentile(v, 75))

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(years, med, color=COLOR_MEDIAN, linewidth=2.5, marker="o", label="median ")
    ax.fill_between(years, q1, q3, color=COLOR_BAND, alpha=0.35, label="IQR (25–75%)")
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Simulation Year", fontsize=15, fontweight="bold", labelpad=12); ax.set_ylabel(col); ax.set_title(title)
    ax.legend(frameon=True, ncol=2, loc="best", fontsize=LEGEND_FONTSIZE)
    fig.tight_layout(); fig.savefig(out_path); plt.close(fig)


def _plot_area_change(tp_traj: pd.DataFrame, col: str, title: str, out_path: Path,
                      q_low: int = 10, q_high: int = 90) -> None:
    if tp_traj is None or tp_traj.empty or col not in tp_traj.columns:
        return
    _set_style()
    years = sorted(tp_traj["year"].unique())
    wide = tp_traj.pivot_table(index="tract_geoid", columns="year", values=col, aggfunc="first").sort_index(axis=1)
    dwide = wide.diff(axis=1)

    ylist = [y for y in years if y in dwide.columns and not dwide[y].isna().all()]
    if not ylist: return

    mean = [dwide[y].mean(skipna=True) for y in ylist]
    qlo  = [dwide[y].quantile(q_low/100.0)  for y in ylist]
    qhi  = [dwide[y].quantile(q_high/100.0) for y in ylist]

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.fill_between(ylist, qlo, qhi, color=COLOR_BAND, alpha=0.35, label=f"Central {q_high-q_low}% band")
    ax.plot(ylist, mean, color=COLOR_MEAN, linewidth=2.5, marker="o", label="mean ΔTP")
    ax.axhline(0.0, color="#64748b", linestyle="--", linewidth=1)
    ax.set_xlabel("Simulation Year", fontsize=15, fontweight="bold", labelpad=12); ax.set_ylabel(f"Δ{col} (YoY)"); ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(out_path); plt.close(fig)


def plot_tp_outputs(tp_traj: pd.DataFrame, fig_root: Path) -> None:
    sub = _ensure_fig_dirs(fig_root)
    _plot_median_iqr(tp_traj, "TP_owner",  "TP (Owner) — Median ± IQR (after-update)",  sub["action"] / "owner_tp_median_iqr.png")
    _plot_median_iqr(tp_traj, "TP_renter", "TP (Renter) — Median ± IQR (after-update)", sub["action"] / "renter_tp_median_iqr.png")
    _plot_area_change(tp_traj, "TP_owner",  "ΔTP (Owner) — Mean with central 80% band",  sub["action"] / "owner_tp_change_area.png")
    _plot_area_change(tp_traj, "TP_renter", "ΔTP (Renter) — Mean with central 80% band", sub["action"] / "renter_tp_change_area.png")
    # New tract-level TP plots
    _plot_tp_heatmap(tp_traj, sub["action"] / "tp_tract_heatmap.png")
    _plot_tp_boxplot_by_year(tp_traj, sub["action"] / "tp_boxplot_by_year.png")


def _plot_tp_heatmap(tp_traj: pd.DataFrame, out_path: Path) -> None:
    """Plot tract-level TP (Threat Perception) heatmap over time."""
    if tp_traj is None or tp_traj.empty:
        return
    _set_style()
    import matplotlib.pyplot as plt
    
    # Average owner and renter
    tp_traj = tp_traj.copy()
    tp_traj["TP_avg"] = (tp_traj["TP_owner"] + tp_traj["TP_renter"]) / 2
    pivot = tp_traj.pivot_table(index="tract_geoid", columns="year", values="TP_avg", aggfunc="mean")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    
    # Labels
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(t)[-6:] for t in pivot.index], fontsize=8)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45)
    ax.set_xlabel("Simulation Year", fontsize=15, fontweight="bold", labelpad=12)
    ax.set_ylabel("Tract", fontsize=12)
    ax.set_title("TP (Threat Perception) by Tract and Year", fontsize=14, fontweight="bold")
    
    # Mark severe years
    severe_years = [2011, 2014, 2021]
    for sy in severe_years:
        if sy in pivot.columns:
            idx = list(pivot.columns).index(sy)
            ax.axvline(idx, color="green", linewidth=2, linestyle="--", alpha=0.7)
    
    fig.colorbar(im, ax=ax, label="TP (0-1)", shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path.name}")


def _plot_tp_boxplot_by_year(tp_traj: pd.DataFrame, out_path: Path) -> None:
    """Plot TP (Threat Perception) distribution boxplot by year with owner/renter comparison."""
    if tp_traj is None or tp_traj.empty:
        return
    _set_style()
    import matplotlib.pyplot as plt

    years = sorted(tp_traj["year"].unique())
    tp_owner_data = [tp_traj[tp_traj["year"] == y]["TP_owner"].dropna().values for y in years]
    tp_renter_data = [tp_traj[tp_traj["year"] == y]["TP_renter"].dropna().values for y in years]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("TP Distribution by Year (Tract-level)", fontsize=14, fontweight="bold")

    # Owner
    ax = axes[0]
    bp1 = ax.boxplot(tp_owner_data, labels=years, patch_artist=True)
    for patch in bp1["boxes"]:
        patch.set_facecolor("#3b82f6")
        patch.set_alpha(0.7)
    ax.set_xlabel("Simulation Year", fontsize=15, fontweight="bold", labelpad=12)
    ax.set_ylabel("TP_owner")
    ax.set_title("Owner (Homeowner)")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=45)

    # Mark severe years
    severe_idx = [years.index(y) + 1 for y in [2011, 2014, 2021] if y in years]
    for idx in severe_idx:
        ax.axvline(idx, color="red", alpha=0.2, linewidth=8)

    # Renter
    ax = axes[1]
    bp2 = ax.boxplot(tp_renter_data, labels=years, patch_artist=True)
    for patch in bp2["boxes"]:
        patch.set_facecolor("#22c55e")
        patch.set_alpha(0.7)
    ax.set_xlabel("Simulation Year", fontsize=15, fontweight="bold", labelpad=12)
    ax.set_ylabel("TP_renter")
    ax.set_title("Renter")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=45)
    for idx in severe_idx:
        ax.axvline(idx, color="red", alpha=0.2, linewidth=8)
    
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path.name}")

# ------------------------- Finance: household-level two-panel histograms -------------------------

def _ensure_dirs(fig_root: Path) -> Dict[str, Path]:
    d = {
        "finance": fig_root / "finance" / "premium_hist",
    }
    for p in d.values():
        p.mkdir(parents=True, exist_ok=True)
    return d

def _nice_step(xmax: int, target_ticks: int = 10) -> int:
    """
    Given xmax, return a nice tick step so that there are about target_ticks major ticks.
    Only returns common steps like 20/25/50/100/200/250/500/1000/2000/5000 ...
    """
    if xmax <= 0:
        return 1
    raw = max(1, int(np.ceil(xmax / max(2, target_ticks))))
    nice = [1, 2, 5]
    pow10 = 10 ** int(np.floor(np.log10(raw)))
    step = nice[0] * pow10
    for n in nice:
        cand = n * pow10
        if cand >= raw:
            step = cand
            break
    if xmax / step > target_ticks + 2:
        step *= 2
    return int(step)

def _resolve_identity(df: pd.DataFrame) -> pd.Series:
    """Map to {'owner','renter'}; fallback 'unknown'."""
    if "identity" in df.columns:
        s = df["identity"]
    elif "group" in df.columns:
        s = df["group"]
    else:
        s = pd.Series(["unknown"] * len(df), index=df.index)
    return (
        s.astype(str)
         .str.lower()
         .map({"owner": "owner", "renter": "renter"})
         .fillna("unknown")
    )

# Note: unified `_premium_total_usd` helper is defined later in this file (single source of truth)

def _extract_year_from(df: pd.DataFrame, fpath: Path) -> int:
    for col in ("year", "Year"):
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").dropna().unique()
            if len(vals) == 1:
                try:
                    return int(vals[0])
                except Exception:
                    pass
    # fallback: parse digits in filename
    try:
        return int("".join([ch for ch in fpath.stem if ch.isdigit()]))
    except Exception:
        return 0

# ------------------------ core draw ------------------------
def _draw_hist_axes(ax,
                    data: np.ndarray,
                    bin_edges: np.ndarray,
                    *,
                    bar_color: str,
                    xmax: int,
                    title: str,
                    tick_step: Optional[int] = None,
                    overflow_color: str = "#f59e0b"):
    """Draw one histogram with an extra overflow bin at xmax+."""
    overflow_mask = data > xmax
    data_in = data[~overflow_mask]
    data_over = data[overflow_mask]

    hist, edges = np.histogram(data_in, bins=bin_edges)
    centers = (edges[:-1] + edges[1:]) / 2.0
    width = np.diff(edges)

    ax.bar(centers, hist, width=width, align="center",
           color=bar_color, alpha=0.9, edgecolor="white", linewidth=0.3)

    # overflow bar
    over_n = int(data_over.size)
    overflow_center = xmax + width[-1] / 2.0
    ax.bar([overflow_center], [over_n], width=[width[-1]], align="center",
           color=overflow_color, alpha=0.95, edgecolor="white", linewidth=0.3)

    # x-range paddings
    right_pad = width[-1] * 1.75
    left_pad  = width[0] * 0.5
    ax.set_xlim(0 - left_pad, xmax + right_pad)

    # axes cosmetics
    ax.set_title(title)
    ax.set_ylabel("HHs")
    ax.grid(True, axis="y", linestyle="--", alpha=0.25)
    ax.tick_params(axis="x", pad=6)

    # ticks (auto nice step; last label is 'xmax+')
    if not tick_step or tick_step <= 0:
        tick_step = _nice_step(int(xmax), target_ticks=10)

    major_ticks = list(range(0, int(xmax) + 1, int(tick_step)))
    major_ticks.append(int(overflow_center))  # overflow tick at the bar center
    ax.set_xticks(major_ticks)

    labels = [f"{v:,}" for v in major_ticks[:-1]]
    labels.append(f"{int(xmax):,}+")
    ax.set_xticklabels(labels, rotation=0)

# ------------------------ public API ------------------------
def plot_household_premium_histograms(fin_dir: Path,
                                      fig_root: Path,
                                      tract_geoid: Optional[str] = None,
                                      *,
                                      owner_bar_color: str = "#3b82f6",   # Blue
                                      renter_bar_color: str = "#22c55e"): # Green
    """
    Output "Homeowner + Renter" two-panel histograms by year.
    - Read finance_households_*.csv under fin_dir
    - Only include policyholders (has_FI>0 or premium>0)
    - If tract_geoid is given, filter by that tract first
    - Output figures to fig_root/finance/premium_hist/
    """
    _set_style()
    sub = _ensure_dirs(fig_root)
    out_dir = sub["finance"]

    files = sorted(Path(fin_dir).glob("finance_households_*.csv"))
    if not files:
        print("[premium-hist] No CSV found.")
        return

    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"[premium-hist] Read failed: {f.name} — {e}")
            continue
        if df.empty:
            continue

        # tract filter
        if tract_geoid is not None:
            tcol = None
            for cand in ("tract_geoid", "tract", "CensusTract", "GEOID", "geoid"):
                if cand in df.columns:
                    tcol = cand; break
            if tcol is not None:
                df = df[df[tcol].astype(str) == str(tract_geoid)]
        if df.empty:
            continue

        year = _extract_year_from(df, f)
        title_prefix = f"{'Tract ' + str(tract_geoid) if tract_geoid else 'All tracts'} — {year}"
        out_png = out_dir / f"premium_histogram_{'tract_' + str(tract_geoid) if tract_geoid else 'ALL'}_{year}.png"

        # identity and premiums
        identity = _resolve_identity(df)
        premium_usd = _premium_total_usd(df)

        # policyholder mask: has_FI>0 first, otherwise premium>0
        has_fi = pd.to_numeric(df.get("has_FI", 0), errors="coerce").fillna(0).astype(int)
        if "has_FI" in df.columns:
            policy_mask = has_fi > 0
        else:
            policy_mask = pd.to_numeric(premium_usd, errors="coerce").fillna(0) > 0

        # split owner / renter
        owner_usd = pd.to_numeric(premium_usd.where(identity.eq("owner") & policy_mask), errors="coerce")
        renter_usd = pd.to_numeric(premium_usd.where(identity.eq("renter") & policy_mask), errors="coerce")
        owner_usd = owner_usd[np.isfinite(owner_usd) & (owner_usd > 0)].to_numpy(float)
        renter_usd = renter_usd[np.isfinite(renter_usd) & (renter_usd > 0)].to_numpy(float)

        if owner_usd.size == 0 and renter_usd.size == 0:
            print(f"[premium-hist] No policyholders for {title_prefix}. Skip.")
            continue

        # detect owner-insures-contents for title
        owner_insures_both = False
        if "owner_insures_both" in df.columns:
            try:
                owner_insures_both = bool(pd.Series(df["owner_insures_both"]).astype(bool).iloc[0])
            except Exception:
                pass
        elif "premium_contents_kUSD" in df.columns:
            owner_insures_both = bool(
                (identity.eq("owner") &
                 (pd.to_numeric(df["premium_contents_kUSD"], errors="coerce").fillna(0.0) > 0)).any()
            )

        owner_title = (f"{title_prefix} — Homeowner Premium for Building + Contents (USD, policyholders only)"
                       if owner_insures_both else
                       f"{title_prefix} — Homeowner Premium for Building (USD, policyholders only)")

        # owner xmax: at least 5,000, at most 10,000, then up to multiple of 50; also not less than p99
        if owner_usd.size:
            p99 = np.nanpercentile(owner_usd, 99)
        else:
            p99 = 5000
        owner_xmax = 13000
        owner_edges = np.arange(0, owner_xmax + 50, 50, dtype=float)

        # renter fixed to 1,000
        renter_xmax = 1000
        renter_edges = np.arange(0, renter_xmax + 20, 20, dtype=float)

        # ----- draw figure -----
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 9), sharex=False, constrained_layout=True)

        _draw_hist_axes(
            axes[0], owner_usd, owner_edges,
            bar_color=owner_bar_color, xmax=owner_xmax, title=owner_title,
            tick_step=1000
        )

        _draw_hist_axes(
            axes[1], renter_usd, renter_edges,
            bar_color=renter_bar_color, xmax=renter_xmax,
            title=f"{title_prefix} — Renter Premium for Contents (USD)",
            tick_step=100
        )
        axes[1].set_xlabel("Annual premium (USD)")

        fig.savefig(out_png, bbox_inches="tight")
        plt.close(fig)



# ------------------------- Vulnerability: tract-level flood damage CDF and box plots -------------------------



def _musd_formatter(x, _pos=None):
    return f"${x/1_000_000:,.1f}M"


def _kusd_formatter(x, _pos=None):
    # In thousands of USD
    return f"${x/1_000:,.0f}k"

# Placed near other utility functions at the top of the file (e.g. after _set_style)
# (Patch imported at top of file)

def _shade_year_bars(ax, years, severe_years, *, half_width=0.48,
                     color="#93c5fd", alpha=0.25):
    """Mark severe years with light blue on categorical x-axis (one bar per year).
    years: list[int] consistent with x-axis bars order
    severe_years: can provide years (e.g. 2021) or two-digit years (e.g. 21)
    """
    severe = set(int(y) for y in severe_years)
    # Support two-digit indication (e.g. 14, 21, 22, 23)
    severe_mod = {y % 100 for y in severe}
    for i, y in enumerate(years):
        if (int(y) in severe) or (int(y) % 100 in severe_mod):
            ax.axvspan(i - half_width, i + half_width,
                       color=color, alpha=alpha, zorder=0)


def _read_damage_all_tables(fig_root: Path):
    """
    Read data from single file flood_damage_tract_ALL_years.csv, and return
    (owner_df, renter_df, both_df) three long tables, each containing columns:
      ['year','tract_geoid','damage_usd']

    Supports two schemas:
      A) long: year, tract_geoid, identity(owner/renter), damage_usd
      B) wide: year, tract_geoid,
               owner_damage_usd(or damage_owner_usd ...),
               renter_damage_usd(or damage_renter_usd ...)

    Will try two root directories simultaneously:
      1) <fig_root>/vulnerability/flood_damage/
      2) <fig_root>/../vulnerability/flood_damage/
    """
    # Possible file locations
    cands = [
        fig_root / "vulnerability" / "flood_damage" / "flood_damage_tract_ALL_years.csv",
        fig_root.parent / "vulnerability" / "flood_damage" / "flood_damage_tract_ALL_years.csv",
    ]
    p = next((x for x in cands if x.exists()), None)
    if p is None:
        empty = pd.DataFrame(columns=["year", "tract_geoid", "damage_usd"])
        return empty.copy(), empty.copy(), empty.copy()

    df = pd.read_csv(p)

    # ---- Column normalization ----
    # year
    df["year"] = pd.to_numeric(df.get("year"), errors="coerce")
    df = df[df["year"].notna()].copy()
    df["year"] = df["year"].astype(int)

    # tract_geoid
    if "tract_geoid" not in df.columns:
        for c in ("CensusTract", "GEOID", "tract", "geoid", "tract_geoid10"):
            if c in df.columns:
                df = df.rename(columns={c: "tract_geoid"})
                break
    df["tract_geoid"] = df.get("tract_geoid", pd.Series([], dtype=str)).astype(str)

    # Column name index (case insensitive)
    lower_map = {c.lower(): c for c in df.columns}

    # ---- Schema A: Already long version (identity + damage_usd) ----
    if ("identity" in lower_map) and ("damage_usd" in lower_map):
        id_col = lower_map["identity"]
        dmg_col = lower_map["damage_usd"]
        long_df = df.rename(columns={id_col: "identity", dmg_col: "damage_usd"})[
            ["year", "tract_geoid", "identity", "damage_usd"]
        ].copy()
        long_df["identity"] = (
            long_df["identity"].astype(str).str.lower()
            .map({"owner": "owner", "renter": "renter"}).fillna("unknown")
        )
        long_df["damage_usd"] = pd.to_numeric(long_df["damage_usd"], errors="coerce").fillna(0.0)

        owner_df  = long_df[long_df["identity"] == "owner"][["year","tract_geoid","damage_usd"]].copy()
        renter_df = long_df[long_df["identity"] == "renter"][["year","tract_geoid","damage_usd"]].copy()
        both_df   = (long_df.groupby(["year","tract_geoid"], as_index=False)["damage_usd"].sum())
        return owner_df, renter_df, both_df

    # ---- Schema B: wide version (one column each for owner / renter) ----
    cand_owner  = [c for c in df.columns if c.lower() in (
        "owner_damage_usd", "damage_owner_usd", "owner_usd", "damage_owner"
    )]
    cand_renter = [c for c in df.columns if c.lower() in (
        "renter_damage_usd", "damage_renter_usd", "renter_usd", "damage_renter"
    )]

    if cand_owner and cand_renter:
        Ow, Rn = cand_owner[0], cand_renter[0]
        tmp = df[["year", "tract_geoid", Ow, Rn]].copy()
        tmp = tmp.rename(columns={Ow: "owner_damage_usd", Rn: "renter_damage_usd"})
        tmp["owner_damage_usd"]  = pd.to_numeric(tmp["owner_damage_usd"],  errors="coerce").fillna(0.0)
        tmp["renter_damage_usd"] = pd.to_numeric(tmp["renter_damage_usd"], errors="coerce").fillna(0.0)

        owner_df = tmp.rename(columns={"owner_damage_usd": "damage_usd"})[
            ["year", "tract_geoid", "damage_usd"]
        ].copy()
        renter_df = tmp.rename(columns={"renter_damage_usd": "damage_usd"})[
            ["year", "tract_geoid", "damage_usd"]
        ].copy()
        both_df = tmp.assign(damage_usd=tmp["owner_damage_usd"] + tmp["renter_damage_usd"])[
            ["year", "tract_geoid", "damage_usd"]
        ].copy()
        return owner_df, renter_df, both_df

    # If neither schema matches -> return empty
    empty = pd.DataFrame(columns=["year", "tract_geoid", "damage_usd"])
    return empty.copy(), empty.copy(), empty.copy()



def _owner_from_both_minus_renter(owner_df, renter_df, both_df):
    """If owner_df is empty, try to recover owner by subtracting renter from both per tract/year."""
    if not owner_df.empty:
        return owner_df
    if both_df.empty or renter_df.empty:
        return owner_df  # Return empty if really no data
    b = both_df.copy()
    r = renter_df.copy()
    m = b.merge(r, on=["year","tract_geoid"], how="left", suffixes=("_both","_rent"))
    m["damage_usd"] = (m["damage_usd_both"].fillna(0.0) - m["damage_usd_rent"].fillna(0.0)).clip(lower=0.0)
    return m[["year","tract_geoid","damage_usd"]]

def plot_flood_damage_by_year_box_from_all(fig_root: Path,
                                           drop_years=(),
                                           severe_years=(2011, 2014, 2021),
                                           show_rate: bool = False,
                                           poster: bool = False) -> None:
    owner_all, renter_all, both_all = _read_damage_all_tables(fig_root)
    if owner_all.empty and renter_all.empty and both_all.empty:
        return

    drop = set(int(y) for y in drop_years)
    filt = lambda df: df[(~df["year"].isin(drop))].copy() if not df.empty else df
    owner_all, renter_all, both_all = map(filt, (owner_all, renter_all, both_all))
    owner_all = _owner_from_both_minus_renter(owner_all, renter_all, both_all)

    def _stats_by_year(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=["year","mean","ci_lo","ci_hi"])
        g = df.groupby("year")["damage_usd"]
        agg = g.agg(
            n="count",
            mean=lambda s: float(np.nanmean(s)),
            std =lambda s: float(np.nanstd(s, ddof=1)),
        ).reset_index()
        agg["se"] = agg["std"] / np.sqrt(np.clip(agg["n"].astype(float), 1.0, None))
        agg["ci_lo"] = agg["mean"] - 1.96 * agg["se"]
        agg["ci_hi"] = agg["mean"] + 1.96 * agg["se"]
        for c in ("mean","ci_lo","ci_hi"):
            agg[c] = agg[c] / 1000.0  # kUSD
        return agg[["year","mean","ci_lo","ci_hi"]]

    stat_owner = _stats_by_year(owner_all)
    stat_renter = _stats_by_year(renter_all)
    stat_both  = _stats_by_year(both_all)

    _set_style()
    import matplotlib.pyplot as plt
    figsize = FIGSIZE_POSTER if poster else FIGSIZE_PAPER
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=figsize,
                             constrained_layout=True, sharex=False)

    def _auto_ylim_stat(stat: pd.DataFrame) -> tuple[float, float]:
        if stat.empty:
            return (0.0, 1.0)
        vmax = float(np.nanmax([stat["ci_hi"].max(), stat["mean"].max()]))
        pad  = 0.10 * max(1.0, vmax)
        return (0.0, vmax + pad)

    panels = [
        ("Homeowner", axes[0], stat_owner, COLOR_OWNER, "(a)"),
        ("Renter", axes[1], stat_renter, COLOR_RENTER, "(b)"),
        ("Both", axes[2], stat_both, "#2882ff", "(c)"),
    ]

    for title, ax, stat, color, plabel in panels:
        if stat.empty:
            ax.set_visible(False); continue

        years = stat["year"].astype(int).tolist()
        x = np.arange(len(years))

        _shade_year_bars(ax, years, severe_years, half_width=0.48,
                         color="#93c5fd", alpha=0.25)

        ax.bar(x, stat["mean"].to_numpy(), width=0.6,
               color=color, alpha=0.75,
               edgecolor="#374151", linewidth=1.2)

        lower = (stat["mean"] - stat["ci_lo"]).clip(lower=0.0).to_numpy()
        upper = (stat["ci_hi"]  - stat["mean"]).clip(lower=0.0).to_numpy()
        yerr  = np.vstack([lower, upper])
        ax.errorbar(
            x, stat["mean"].to_numpy(), yerr=yerr,
            fmt="none", ecolor="#9ca3af", elinewidth=1.2, capsize=2, zorder=3
        )
        ax.set_ylabel("Average per tract (kUSD)")
        ax.set_xlim(-0.6, len(x) - 0.4)
        # Only show even years
        even_years = [y for y in years if y % 2 == 0]
        even_indices = [i for i, y in enumerate(years) if y % 2 == 0]
        ax.set_xticks(even_indices)
        ax.set_xticklabels([str(y) for y in even_years])
        ax.set_ylim(*_auto_ylim_stat(stat))
        from matplotlib.ticker import FuncFormatter
        ax.yaxis.set_major_formatter(FuncFormatter(_kusd_formatter))
        ax.set_title(title)
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        _panel_label(ax, plabel)   # <- Top-left (a)/(b)/(c)


    axes[-1].set_xlabel("Simulation Year", labelpad=12)

    # Legend (auto inside when toggled)
    from matplotlib.patches import Patch
    severe_patch = Patch(facecolor="#93c5fd", alpha=0.25, edgecolor="none",
                         label="Severe flood year")
    _legend_box_right(fig, handles=[severe_patch], ncol=1)

    out_dir = _ensure_vuln_damage_dir(fig_root)
    fig.savefig(out_dir / "flood_damage_box_by_year_with2011_fromALL.png", bbox_inches="tight")
    print("Saved flood_damage_box_by_year_with2011_fromALL.png")
    plt.close(fig)



def plot_flood_damage_stacked_area_from_all(fig_root: Path, drop_years=()) -> None:
    """
    Stacked area plot: Stack tract-level damage totals for owner and renter by year, draw dashed line for total.
    Source: *_ALL.csv, 2011 not excluded.
    """
    owner_all, renter_all, both_all = _read_damage_all_tables(fig_root)
    if owner_all.empty and renter_all.empty and both_all.empty:
        return

    drop = set(int(y) for y in drop_years)
    filt = lambda df: df[(~df["year"].isin(drop))].copy() if not df.empty else df
    owner_all, renter_all, both_all = map(filt, (owner_all, renter_all, both_all))

    # If no owner_ALL, recover owner using both - renter (subtract per tract)
    owner_all = _owner_from_both_minus_renter(owner_all, renter_all, both_all)

    # Annual total
    own = owner_all.groupby("year")["damage_usd"].sum() if not owner_all.empty else pd.Series(dtype=float)
    ren = renter_all.groupby("year")["damage_usd"].sum() if not renter_all.empty else pd.Series(dtype=float)

    years = sorted(set(own.index.tolist()) | set(ren.index.tolist()))
    if not years:
        return
    y_arr  = np.array(years, dtype=int)
    own_y  = np.array([float(own.get(y, 0.0)) for y in years])
    ren_y  = np.array([float(ren.get(y, 0.0)) for y in years])
    tot_y  = own_y + ren_y

    _set_style()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.stackplot(y_arr, own_y, ren_y, labels=["Homeowner", "Renter"],
                 colors=[COLOR_OWNER, COLOR_RENTER], alpha=0.70)
    ax.plot(y_arr, tot_y, linestyle="--", linewidth=2.2, color="#111827", label="Total")

    ax.set_title("Annual Flood Damage", fontsize=16, pad=12, fontweight="bold")
    ax.legend(frameon=True, facecolor='white', edgecolor='black', 
              framealpha=1.0, loc="best", fontsize=LEGEND_FONTSIZE)

    out_dir = _ensure_vuln_damage_dir(fig_root)
    fig.savefig(out_dir / "flood_damage_stacked_area_with2011_fromALL.png", bbox_inches="tight")
    print("Saved flood_damage_stacked_area_with2011_fromALL.png")
    plt.close(fig)


def plot_flood_damage_stacked_area_cum_normalized_from_all(fig_root: Path,
                                                           drop_years=()) -> None:
    """
    Stacked area plot of 'cumulative' owner / renter flood damage by year, 'normalized' to 0-1 by final total.
    Source: *_ALL.csv; 2011 not excluded.
    """
    owner_all, renter_all, both_all = _read_damage_all_tables(fig_root)
    if owner_all.empty and renter_all.empty and both_all.empty:
        print("[WARN] No *_ALL.csv found under vulnerability/(flood_damage|flood damage)")
        return

    drop = set(int(y) for y in drop_years)
    filt = lambda df: df[(~df["year"].isin(drop))].copy() if not df.empty else df
    owner_all, renter_all, both_all = map(filt, (owner_all, renter_all, both_all))

    # If no owner_ALL, restore owner using both - renter
    owner_all = _owner_from_both_minus_renter(owner_all, renter_all, both_all)

    # Annual total
    own_y = owner_all.groupby("year")["damage_usd"].sum().sort_index()
    ren_y = renter_all.groupby("year")["damage_usd"].sum().sort_index()

    years = sorted(set(own_y.index.tolist()) | set(ren_y.index.tolist()))
    if not years:
        print("[WARN] No years left after dropping", drop_years)
        return

    y_arr = np.array(years, dtype=int)
    own = np.array([float(own_y.get(y, 0.0)) for y in years])
    ren = np.array([float(ren_y.get(y, 0.0)) for y in years])

    # Cumulative
    own_cum = np.cumsum(own)
    ren_cum = np.cumsum(ren)
    tot_cum = own_cum + ren_cum
    tot_final = float(tot_cum[-1]) if tot_cum.size else 0.0
    if tot_final <= 0:
        print("[WARN] Total cumulative damage is zero; skip plot.")
        return

    # Normalize to 0-1 (final total is 1)
    own_frac = own_cum / tot_final
    ren_frac = ren_cum / tot_final
    tot_frac = (own_cum + ren_cum) / tot_final

    _set_style()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

    ax.stackplot(y_arr, own_frac, ren_frac,
                 labels=["Homeowner (cumulative)", "Renter (cumulative)"],
                 colors=[COLOR_OWNER, COLOR_RENTER],
                 alpha=0.70)

    # Dashed line summing to 100%
    ax.plot(y_arr, tot_frac, linestyle="--", linewidth=2.2,
            color="#111827", label="Total (normalized)")

    ax.set_title("Cumulative Flood Damage", fontsize=16, pad=12, fontweight="bold")
    ax.legend(frameon=True, facecolor='white', edgecolor='black', 
              framealpha=1.0, loc="best", fontsize=LEGEND_FONTSIZE)

    out_dir = _ensure_vuln_damage_dir(fig_root)
    fig.savefig(out_dir / "flood_damage_stacked_area_cum_normalized_with2011_fromALL.png",
                bbox_inches="tight")
    print("Saved flood_damage_stacked_area_cum_normalized_with2011_fromALL.png")
    plt.close(fig)

# ==== NEW: household counts by tract-year (infer denominator from finance_households_YYYY.csv) ====
def _hh_counts_by_tract_year(fin_dir: Path) -> pd.DataFrame:
    """
    Return columns: ['year','tract_geoid','n_owner','n_renter','n_both']
    """
    tables = _load_finance_household_years(fin_dir)
    rows = []
    for df in tables:
        if "__year" not in df.columns:
            continue
        y = int(df["__year"].iloc[0])
        # tract column
        tcol = next((c for c in ("tract_geoid","tract","CensusTract","GEOID","geoid") if c in df.columns), None)
        if tcol is None:
            continue
        ident = _find_identity(df[ df.columns[df.columns.str.lower().isin(["group","identity"])][0] ]) \
                if df.columns.str.lower().isin(["group","identity"]).any() else pd.Series(["unknown"]*len(df))
        tmp = pd.DataFrame({"year": y, "tract_geoid": df[tcol].astype(str), "identity": ident})
        g = tmp.groupby(["year","tract_geoid","identity"]).size().rename("n").reset_index()
        rows.append(g)
    if not rows:
        return pd.DataFrame(columns=["year","tract_geoid","n_owner","n_renter","n_both"])
    n = (pd.concat(rows, ignore_index=True)
           .pivot_table(index=["year","tract_geoid"], columns="identity", values="n",
                        aggfunc="sum", fill_value=0).reset_index())
    n = n.rename(columns={"owner":"n_owner","renter":"n_renter"})
    for c in ("n_owner","n_renter"):
        if c not in n.columns: n[c] = 0
    n["n_both"] = n["n_owner"] + n["n_renter"]
    return n[["year","tract_geoid","n_owner","n_renter","n_both"]]

def _avg_damage_per_household(dmg_df: pd.DataFrame, n_df: pd.DataFrame, who: str) -> pd.DataFrame:
    """
    dmg_df: ['year','tract_geoid','damage_usd']; who: 'owner' | 'renter' | 'both'
    Return: ['year','avg_usd']
    """
    if dmg_df.empty: 
        return pd.DataFrame(columns=["year","avg_usd"])
    den_col = {"owner":"n_owner","renter":"n_renter","both":"n_both"}[who]
    m = dmg_df.merge(n_df, on=["year","tract_geoid"], how="left")
    m["den"] = pd.to_numeric(m.get(den_col, 0), errors="coerce")
    m = m[m["den"] > 0].copy()
    g = m.groupby("year").agg(total_damage=("damage_usd","sum"),
                              total_hh=("den","sum")).reset_index()
    g["avg_usd"] = np.where(g["total_hh"]>0, g["total_damage"]/g["total_hh"], np.nan)
    return g[["year","avg_usd"]].sort_values("year")

def plot_flood_damage_per_household(fig_root: Path, fin_dir: Path,
                                    drop_years=(),
                                    severe_years=(2011, 2014, 2021),
                                    include_both: bool = False,
                                    poster: bool = False) -> None:
    """
    (a) Homeowner、(b) Renter 的『每戶平均 flood damage (USD/HH)』。
    若 include_both=True，則加上 (c) Both 面板（預設不加）。
    """
    owner_all, renter_all, both_all = _read_damage_all_tables(fig_root)
    if owner_all.empty and renter_all.empty and both_all.empty:
        return

    # 分母：各年、各 tract 的 owner / renter 戶數
    n_df = _hh_counts_by_tract_year(fin_dir)

    # 過濾年份
    drop = set(int(y) for y in drop_years)
    filt = lambda d: d[(~d["year"].isin(drop))].copy() if not d.empty else d
    owner_all, renter_all, both_all = map(filt, (owner_all, renter_all, both_all))

    # 每戶平均（USD/HH）
    A = _avg_damage_per_household(owner_all, n_df, "owner")
    R = _avg_damage_per_household(renter_all, n_df, "renter")
    B = _avg_damage_per_household(both_all,  n_df, "both") if include_both else None

    # 面板列表（預設只畫 Homeowner, Renter）
    panels = [
        ("Homeowner", A, "(a)"),
        ("Renter", R, "(b)"),
    ]
    if include_both and B is not None and not B.empty:
        panels.append(("Both", B, "(c)"))

    # 繪圖
    _set_style()
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    from matplotlib.patches import Patch
    nrow = len(panels)
    # Custom figsize selection
    figsize = FIGSIZE_POSTER if poster else FIGSIZE_PAPER
    
    # If dynamic height is preferred for flood damage, we can override or stick to fixed.
    # User requested specific fixed sizes.
    fig, axes = plt.subplots(nrow, 1, figsize=figsize,
                             constrained_layout=True, sharex=False)
    if nrow == 1:
        axes = [axes]

    def _shade(ax, years):
        sev = set(int(s) for s in severe_years)
        for i, y in enumerate(years):
            if int(y) in sev:
                ax.axvspan(i - 0.48, i + 0.48, color="#93c5fd", alpha=0.25, zorder=0)

    # Custom formatter for y-axis: values displayed in thousands (k)
    def _k_formatter(x, pos):
        if x >= 1000:
            return f'${x/1000:.0f}k'
        else:
            return f'${x:.0f}'

    for ax, (title, df, tag) in zip(axes, panels):
        if df.empty:
            ax.set_visible(False); continue
        years = df["year"].astype(int).tolist()
        x = np.arange(len(years))
        _shade(ax, years)
        ax.bar(x, df["avg_usd"].to_numpy(float), width=0.6,
               color="#67b99a", alpha=0.80,
               edgecolor="black", linewidth=1.2)
        # Only show even years
        even_years = [y for y in years if y % 2 == 0]
        even_indices = [i for i, y in enumerate(years) if y % 2 == 0]
        ax.set_xticks(even_indices)
        ax.set_xticklabels([str(y) for y in even_years])
        ax.set_ylabel("Average per HH (USD)", labelpad=12)
        ax.yaxis.set_major_formatter(FuncFormatter(_k_formatter))
        ax.set_title(title, pad=12)
        ax.grid(True, axis="y", linestyle="--", alpha=0.30)
        ax.tick_params(axis='both', which='major')
        _panel_label(ax, tag)

    # Legend in the bottom panel, center
    bar_proxy = Patch(facecolor="#67b99a", edgecolor="black", linewidth=1.2, label="Average flood damage")
    handles = [bar_proxy, _severe_patch()]
    axes[-1].legend(handles=handles, loc="upper center",
                   frameon=True, facecolor='white', edgecolor='black',
                   framealpha=1.0, ncol=2)

    axes[-1].set_xlabel("Simulation Year", labelpad=12)

    out_dir = _ensure_vuln_damage_dir(fig_root)
    fname = "flood_damage_per_household_owner_renter.png" if not include_both \
            else "flood_damage_per_household_owner_renter_both.png"
    fig.savefig(out_dir / fname, bbox_inches="tight")
    plt.close(fig)


# ------------------------- Finance: household-level stacked bar plots -------------------------
# ---------- helpers ----------
def _ensure_fin_ts_dir(fig_root: Path) -> Path:
    out = fig_root / "finance" / "timeseries"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _find_identity(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().map({"owner":"owner","renter":"renter"}).fillna("unknown")

def _usd_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0.0)

def _sum_usd_cols(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    if not cols:
        return pd.Series(0.0, index=df.index)
    acc = pd.Series(0.0, index=df.index)
    for c in cols:
        if c in df.columns:
            acc = acc.add(_usd_series(df[c]), fill_value=0.0)
    return acc

# ---- replace these in plots.py ----

def _premium_total_usd(df: pd.DataFrame) -> pd.Series:
    # 1) 最優先：直接用 total 欄（避免重複相加）
    if "premium_total_usd" in df.columns:
        return pd.to_numeric(df["premium_total_usd"], errors="coerce").fillna(0.0)

    # 2) 沒有 total → 組合 structure + contents（USD）
    cols = []
    if "premium_structure_usd" in df.columns:
        cols.append("premium_structure_usd")
    if "premium_contents_usd" in df.columns:
        cols.append("premium_contents_usd")
    if cols:
        return pd.to_numeric(df[cols], errors="coerce").sum(axis=1).fillna(0.0)

    # 3) 只有 kUSD 版本
    if "premium_total_kUSD" in df.columns:
        return pd.to_numeric(df["premium_total_kUSD"], errors="coerce").fillna(0.0) * 1000.0
    cols_k = []
    if "premium_structure_kUSD" in df.columns:
        cols_k.append("premium_structure_kUSD")
    if "premium_contents_kUSD" in df.columns:
        cols_k.append("premium_contents_kUSD")
    if cols_k:
        return pd.to_numeric(df[cols_k], errors="coerce").sum(axis=1).fillna(0.0) * 1000.0

    # 4) 落空就回 0
    return pd.Series(0.0, index=df.index)

def _payout_total_usd(df: pd.DataFrame) -> pd.Series:
    if "payout_total_usd" in df.columns:
        return pd.to_numeric(df["payout_total_usd"], errors="coerce").fillna(0.0)
    if {"payout_structure_usd","payout_contents_usd"} <= set(df.columns):
        return (pd.to_numeric(df["payout_structure_usd"], errors="coerce").fillna(0.0) +
                pd.to_numeric(df["payout_contents_usd"], errors="coerce").fillna(0.0))
    if {"payout_structure_kUSD","payout_contents_kUSD"} <= set(df.columns):
        return (pd.to_numeric(df["payout_structure_kUSD"], errors="coerce").fillna(0.0) +
                pd.to_numeric(df["payout_contents_kUSD"], errors="coerce").fillna(0.0)) * 1000.0
    return pd.Series(0.0, index=df.index)

def _oop_total_usd(df: pd.DataFrame) -> pd.Series:
    if "oop_total_usd" in df.columns:
        return pd.to_numeric(df["oop_total_usd"], errors="coerce").fillna(0.0)
    if {"oop_structure_usd","oop_contents_usd"} <= set(df.columns):
        return (pd.to_numeric(df["oop_structure_usd"], errors="coerce").fillna(0.0) +
                pd.to_numeric(df["oop_contents_usd"], errors="coerce").fillna(0.0))
    if {"oop_structure_kUSD","oop_contents_kUSD"} <= set(df.columns):
        return (pd.to_numeric(df["oop_structure_kUSD"], errors="coerce").fillna(0.0) +
                pd.to_numeric(df["oop_contents_kUSD"], errors="coerce").fillna(0.0)) * 1000.0
    return pd.Series(0.0, index=df.index)



# `_musd_formatter` is already defined earlier; keep a single definition

def _load_finance_household_years(fin_dir: Path) -> list[pd.DataFrame]:
    files = sorted(fin_dir.glob("finance_households_*.csv"))
    out = []
    for f in files:
        try:
            df = pd.read_csv(f)
            # year
            if "year" in df.columns:
                y = pd.to_numeric(df["year"], errors="coerce").dropna().unique()
                if len(y)==1: df["__year"] = int(y[0])
            if "__year" not in df.columns:
                # 從檔名抓數字
                import re
                m = re.search(r"(20\d{2})", f.stem)
                df["__year"] = int(m.group(1)) if m else np.nan
            out.append(df)
        except Exception:
            continue
    return out


def _load_finance_tract_years(fin_dir: Path) -> list[pd.DataFrame]:
    """
    Load tract-level finance data (finance_tract_*.csv) as a fallback when
    household data is not available (e.g., in summary/minimal output modes).
    
    Returns a list of DataFrames with '__year' column added for consistency.
    """
    files = sorted(fin_dir.glob("finance_tract_*.csv"))
    # Exclude the aggregated file
    files = [f for f in files if 'all_years' not in f.stem.lower()]
    out = []
    for f in files:
        try:
            df = pd.read_csv(f)
            # Extract year from filename or column
            if "year" in df.columns:
                y_vals = pd.to_numeric(df["year"], errors="coerce").dropna().unique()
                if len(y_vals) == 1:
                    df["__year"] = int(y_vals[0])
            if "__year" not in df.columns:
                import re
                m = re.search(r"(20\d{2})", f.stem)
                if m:
                    df["__year"] = int(m.group(1))
            if "__year" in df.columns:
                out.append(df)
        except Exception:
            continue
    return out


def _load_finance_data_with_fallback(fin_dir: Path) -> tuple[list[pd.DataFrame], str]:
    """
    Load finance data, preferring household-level data but falling back to tract-level.
    
    Returns:
        tuple: (list of DataFrames, data_source: 'household' or 'tract')
    """
    # Try household data first
    tables = _load_finance_household_years(fin_dir)
    if tables:
        return tables, 'household'
    
    # Fallback to tract data
    tract_tables = _load_finance_tract_years(fin_dir)
    if tract_tables:
        print("[info] Using tract-level finance data for plots (household data not available)")
        return tract_tables, 'tract'
    
    return [], 'none'

# ---------- Figure 1: stacked owner/renter (premium+payout+oop) ----------
def _shade_severe_years(ax, years, severe_years):
    """在分類軸上用淡藍色標註嚴重洪年。"""
    severe = set(int(y) for y in severe_years)
    for i, y in enumerate(years):
        if int(y) in severe:
            ax.axvspan(i - 0.6, i + 0.6, color="#93c5fd", alpha=0.25, zorder=0)

def plot_fin_stacked_owner_renter_separate(
    fin_dir: Path,
    fig_root: Path,
    drop_years=(),                   # ✅ 不排除 2011
    severe_years=(2011, 2014, 2021),
    *,
    normalize: bool = False,         # True → 以比例堆疊（每年合計=1）
    panel_labels: tuple[str, str] = ("(a)", "(b)"),       # True→圖外右側；False→下方置中
    export_pdf: bool = True,         # 另存 PDF（向量）
    mono: bool = False,              # True→黑白期刊用（灰階+陰影）
) -> None:
    tables = _load_finance_household_years(fin_dir)
    if not tables:
        return

    # 蒐集：每年 × 身分 的 premium / payout / oop（總額）
    rows = []
    for df in tables:
        if "__year" not in df.columns:
            continue
        y = int(df["__year"].iloc[0])
        if y in set(drop_years):
            continue

        # identity 容錯
        if df.columns.str.lower().isin(["group", "identity"]).any():
            idcol = df.columns[df.columns.str.lower().isin(["group", "identity"])][0]
            ident = _find_identity(df[idcol])
        else:
            ident = pd.Series(["unknown"] * len(df))

        rows.append(pd.DataFrame({
            "year": y,
            "identity": ident,
            "premium_usd": _premium_total_usd(df),
            "payout_usd":  _payout_total_usd(df),
            "oop_usd":     _oop_total_usd(df),
        }))

    if not rows:
        return

    hh = pd.concat(rows, ignore_index=True)
    hh = hh[hh["identity"].isin(["owner", "renter"])]

    # 每年 × 身分 的總額
    tot = (hh.groupby(["year", "identity"], as_index=False)
             [["premium_usd", "payout_usd", "oop_usd"]].sum())

    # 每年 × 身分 的戶數（所有該身分戶，不限是否投保）
    cnt = (hh.groupby(["year", "identity"]).size()
             .rename("n_households").reset_index())

    # 轉成每戶平均
    grp = tot.merge(cnt, on=["year", "identity"], how="left")
    for col in ["premium_usd", "payout_usd", "oop_usd"]:
        grp[col] = np.where(grp["n_households"] > 0, grp[col] / grp["n_households"], 0.0)

    years = sorted(grp["year"].unique())
    if not years:
        return

    def _series(iden: str) -> dict[str, np.ndarray]:
        s = grp[grp["identity"].eq(iden)].set_index("year")
        prem = np.array([float(s["premium_usd"].get(y, 0.0)) for y in years])
        pay  = np.array([float(s["payout_usd"].get(y, 0.0))  for y in years])
        oop  = np.array([float(s["oop_usd"].get(y, 0.0))     for y in years])
        if normalize:
            t = prem + pay + oop
            t[t == 0] = 1.0
            prem, pay, oop = prem/t, pay/t, oop/t
        return {"prem": prem, "pay": pay, "oop": oop}

    OWN = _series("owner")
    REN = _series("renter")

    # ---- 論文樣式 ----
    _set_style()
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    import matplotlib.ticker as mtick
    from matplotlib.patches import Patch
    # 色票（Vega C10；mono 會轉灰階與陰影）
    C_PREM, C_PAY, C_OOP = ("#4C78A8", "#F58518", "#54A24B")
    if mono:
        C_PREM, C_PAY, C_OOP = ("#555555", "#9a9a9a", "#c9c9c9")

    fig_w = 7.0  # 雙欄寬（英吋）≈ 7.0；單欄可改 3.35
    fig_h = 4.6  # 兩面板總高
    fig, axes = plt.subplots(2, 1, figsize=(fig_w, fig_h), constrained_layout=True, sharex=True)
    x = np.arange(len(years)); w = 0.72

    def _shade_narrow(ax):
        severe = set(int(v) for v in severe_years)
        for i, y in enumerate(years):
            if int(y) in severe:
                ax.axvspan(i - 0.45, i + 0.45, color="#93c5fd", alpha=0.20, zorder=0)

    def _draw(ax, S, title, colors, label):
        _shade_narrow(ax)
        if mono:
            # 灰階 + 陰影（hatch）以利黑白列印
            ax.bar(x, S["prem"], width=w, color=colors[0], label="Premium",
                   edgecolor="black", linewidth=0.6, hatch="//", alpha=0.9)
            ax.bar(x, S["pay"],  width=w, bottom=S["prem"], color=colors[1], label="Payout",
                   edgecolor="black", linewidth=0.6, hatch="\\\\", alpha=0.9)
            ax.bar(x, S["oop"],  width=w, bottom=S["prem"]+S["pay"], color=colors[2], label="OOP",
                   edgecolor="black", linewidth=0.6, hatch="..", alpha=0.9)
        else:
            ax.bar(x, S["prem"], width=w, color=colors[0], label="Premium", alpha=0.95)
            ax.bar(x, S["pay"],  width=w, bottom=S["prem"],          color=colors[1], label="Payout", alpha=0.90)
            ax.bar(x, S["oop"],  width=w, bottom=S["prem"]+S["pay"], color=colors[2], label="OOP",    alpha=0.90)

        # 外框線
        ax.bar(x, S["prem"]+S["pay"]+S["oop"], width=w, fill=False,
               edgecolor="#111827", linewidth=0.8)

        # 標軸與格式
        ax.set_title(title)
        ax.set_xticks(x); ax.set_xticklabels([str(y) for y in years])
        ax.grid(True, axis="y", linestyle="--", alpha=0.25)
        if normalize:
            ax.set_ylabel("Share")
            ax.set_ylim(0, 1.0)
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        else:
            ax.set_ylabel("Average per HH (USD)")
            ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.0f}'))

        # 面板標籤
        _panel_label(ax, label)

    _draw(axes[0], OWN, "Homeowner", (C_PREM, C_PAY, C_OOP), panel_labels[0])
    _draw(axes[1], REN, "Renter", (C_PREM, C_PAY, C_OOP), panel_labels[1])
    axes[1].set_xlabel("Simulation Year", fontsize=15, fontweight="bold", labelpad=12)

    # 圖例：圖外右側或下方置中
    handles, labels = axes[0].get_legend_handles_labels()
    severe_patch = Patch(facecolor="#93c5fd", alpha=0.20, edgecolor="none", label="Severe flood year")
    handles = handles + [severe_patch]; labels = labels + ["Severe flood year"]

    handles, labels = axes[0].get_legend_handles_labels()
    handles += [_severe_patch()]
    labels  += ["Severe flood year"]
    # Place legend at top-left inside the bottom panel
    axes[1].legend(handles=handles, labels=labels, loc="upper left",
                   frameon=True, facecolor='white', edgecolor="black", 
                   framealpha=1.0, fontsize=LEGEND_FONTSIZE)

    out = _ensure_fin_ts_dir(fig_root)
    fname = ("stacked_premium_payout_oop_owner_renter_paper_norm" if normalize
             else "stacked_premium_payout_oop_owner_renter_paper")
    fig.savefig(out / f"{fname}.png", bbox_inches="tight")
    # PDF export disabled
    # if export_pdf:
    #     fig.savefig(out / f"{fname}.pdf", bbox_inches="tight")
    plt.close(fig)



def plot_fin_cost_owner_renter(
    fin_dir: Path,
    fig_root: Path,
    drop_years=(),
    severe_years=(2011, 2014, 2021),
    poster: bool = False,
) -> None:
    """
    (a) Homeowner / (b) Renter
    左軸：堆疊柱 Premium + OOP（每戶平均 USD）with 95% CI error bars
    右軸：OOP rate 折線（同群體：ΣOOP / Σflood damage）
          Homeowner 0–25%，Renter 0–10%
    """
    from scipy import stats
    
    tables = _load_finance_household_years(fin_dir)
    if not tables:
        return

    # --------- 蒐集 Premium / OOP（總額 + 戶數）---------
    rows = []
    for df in tables:
        if "__year" not in df.columns:
            continue
        y = int(df["__year"].iloc[0])
        if y in set(drop_years):
            continue

        # 身分欄（容錯）
        if df.columns.str.lower().isin(["group", "identity"]).any():
            idcol = df.columns[df.columns.str.lower().isin(["group", "identity"])][0]
            ident = _find_identity(df[idcol])
        else:
            ident = pd.Series(["unknown"] * len(df))
        
        # 加入 tract_geoid 以計算跨 tract 的 CI
        tract_col = None
        for c in ["tract_geoid", "tract", "GEOID", "geoid"]:
            if c in df.columns:
                tract_col = c
                break

        rows.append(pd.DataFrame({
            "year": y,
            "identity": ident,
            "tract": df[tract_col] if tract_col else "unknown",
            "premium_usd": _premium_total_usd(df),
            "oop_usd":     _oop_total_usd(df),
        }))

    if not rows:
        return

    hh = pd.concat(rows, ignore_index=True)
    hh = hh[hh["identity"].isin(["owner", "renter"])]

    # 年×身分：總額（給 OOP rate 用）
    tot = (hh.groupby(["year", "identity"], as_index=False)[["premium_usd", "oop_usd"]].sum())

    # 年×身分：每戶平均（左軸柱用）
    cnt = (hh.groupby(["year", "identity"]).size().rename("n_households").reset_index())
    grp = tot.merge(cnt, on=["year", "identity"], how="left")
    for c in ["premium_usd", "oop_usd"]:
        grp[c] = np.where(grp["n_households"] > 0, grp[c] / grp["n_households"], 0.0)

    years = sorted(grp["year"].unique())
    if not years:
        return
    x = np.arange(len(years)); w = 0.72

    def _series_mean(iden: str):
        s = grp[grp["identity"].eq(iden)].set_index("year")
        prem = np.array([float(s["premium_usd"].get(y, 0.0)) for y in years])
        oop  = np.array([float(s["oop_usd"].get(y, 0.0))     for y in years])
        return prem, oop
    
    def _series_ci(iden: str, n_bootstrap: int = 1000):
        """Calculate 95% CI using bootstrap resampling."""
        ci_list = []
        sub = hh[hh["identity"].eq(iden)]
        for y in years:
            year_data = sub[sub["year"] == y].copy()
            if len(year_data) > 0:
                tract_sums = year_data.groupby("tract")[["premium_usd", "oop_usd"]].sum()
                tract_sums["total"] = tract_sums["premium_usd"] + tract_sums["oop_usd"]
                values = tract_sums["total"].values
                
                # Exclude outliers using IQR
                if len(values) > 4:
                    q1, q3 = np.percentile(values, [25, 75])
                    iqr = q3 - q1
                    mask = (values >= q1 - 1.5 * iqr) & (values <= q3 + 1.5 * iqr)
                    values = values[mask]
                
                if len(values) > 1:
                    # Bootstrap resampling
                    np.random.seed(42)  # Reproducibility
                    boot_means = []
                    for _ in range(n_bootstrap):
                        sample = np.random.choice(values, size=len(values), replace=True)
                        boot_means.append(np.mean(sample))
                    boot_means = np.array(boot_means)
                    ci_low = np.percentile(boot_means, 2.5)
                    ci_high = np.percentile(boot_means, 97.5)
                    mean_val = np.mean(values)
                    ci = (ci_high - ci_low) / 2
                    # Normalize by average household count per tract
                    avg_hh_per_tract = len(year_data) / len(tract_sums) if len(tract_sums) > 0 else 1
                    ci_list.append(ci / max(avg_hh_per_tract, 1))
                else:
                    ci_list.append(0)
            else:
                ci_list.append(0)
        return np.array(ci_list)

    def _series_tot_oop(iden: str):
        s = tot[tot["identity"].eq(iden)].set_index("year")
        return np.array([float(s["oop_usd"].get(y, 0.0)) for y in years])

    prem_O, oop_O = _series_mean("owner")
    prem_R, oop_R = _series_mean("renter")
    ci_O = _series_ci("owner")
    ci_R = _series_ci("renter")
    oop_tot_O = _series_tot_oop("owner")
    oop_tot_R = _series_tot_oop("renter")

    # --------- 讀取 flood damage（分 owner / renter；總額）---------
    owner_all, renter_all, _both_all = _read_damage_all_tables(fig_root)
    dmg_O = (owner_all.groupby("year")["damage_usd"].sum() if not owner_all.empty
             else pd.Series(dtype=float))
    dmg_R = (renter_all.groupby("year")["damage_usd"].sum() if not renter_all.empty
             else pd.Series(dtype=float))

    rate_O = np.array([
        (oop_tot_O[i] / float(dmg_O.get(y))) if float(dmg_O.get(y, 0.0)) > 0 else np.nan
        for i, y in enumerate(years)
    ], dtype=float)
    rate_R = np.array([
        (oop_tot_R[i] / float(dmg_R.get(y))) if float(dmg_R.get(y, 0.0)) > 0 else np.nan
        for i, y in enumerate(years)
    ], dtype=float)

    # --------- 畫圖 ---------
    _set_style()
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    import matplotlib.ticker as mtick
    from matplotlib.patches import Patch
    C_PREM, C_OOP = "#4C78A8", "#54A24B"

    # Font size overrides for readability
    TITLE_FONTSIZE = 24
    LABEL_FONTSIZE = 22
    TICK_FONTSIZE = 20
    
    figsize = FIGSIZE_POSTER if poster else FIGSIZE_PAPER
    fig, axes = plt.subplots(2, 1, figsize=figsize, constrained_layout=True, sharex=True)

    def _shade(ax):
        sev = set(int(v) for v in severe_years)
        for i, y in enumerate(years):
            if int(y) in sev:
                ax.axvspan(i - 0.45, i + 0.45, color="#93c5fd", alpha=0.20, zorder=0)

    # Custom formatter for y-axis: values ≥1000 as "$Xk"
    def _k_formatter(x, pos):
        if x >= 1000:
            return f'${x/1000:.0f}k'
        else:
            return f'${x:.0f}'

    def _draw(ax, prem, oop, rate, title, label, y2_hi, show_legend=True):
        _shade(ax)
        bars1 = ax.bar(x, prem, width=w, color=C_PREM, alpha=0.95, label="Premium")
        bars2 = ax.bar(x, oop,  width=w, bottom=prem, color=C_OOP,  alpha=0.90, label="OOP")
        ax.bar(x, prem + oop, width=w, fill=False, edgecolor="#111827", linewidth=1.2)

        ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight="bold")
        ax.set_ylabel("Average per HH (USD)", fontsize=LABEL_FONTSIZE, labelpad=12)
        ax.yaxis.set_major_formatter(FuncFormatter(_k_formatter))
        ax.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
        # Only show even years
        even_years = [y for y in years if y % 2 == 0]
        even_indices = [i for i, y in enumerate(years) if y % 2 == 0]
        ax.set_xticks(even_indices)
        ax.set_xticklabels([str(y) for y in even_years], fontsize=TICK_FONTSIZE)
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        _panel_label(ax, label)

        # 右軸：OOP rate
        ax2 = ax.twinx()
        (line,) = ax2.plot(x, rate, linestyle="--", marker="o", linewidth=2.5, markersize=8,
                           color="#111827", label="OOP rate (OOP / flood damage)")
        ax2.set_ylim(0.0, y2_hi)   # ★ 依面板固定上限
        ax2.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
        ax2.set_ylabel("OOP rate", fontsize=LABEL_FONTSIZE, labelpad=12)
        ax2.tick_params(axis='y', labelsize=TICK_FONTSIZE)
        
        # Legend only on first panel
        if show_legend:
            sev_patch = Patch(facecolor="#93c5fd", alpha=0.20, edgecolor="none", label="Severe flood")
            handles = [bars1, bars2, line, sev_patch]
            ax.legend(handles=handles, loc="upper left", frameon=True, facecolor='white',
                      edgecolor="black", framealpha=1.0, fontsize=LEGEND_FONTSIZE)

        return bars1, bars2, line

    b1, b2, l1 = _draw(axes[0], prem_O, prem_O*0 + oop_O, rate_O,
                       "Homeowner", "(a)", y2_hi=0.25, show_legend=True)
    b3, b4, l2 = _draw(axes[1], prem_R, prem_R*0 + oop_R, rate_R,
                       "Renter", "(b)", y2_hi=0.10, show_legend=False)
    axes[1].set_xlabel("Simulation Year", fontsize=LABEL_FONTSIZE, fontweight="bold", labelpad=12)

    out = _ensure_fin_ts_dir(fig_root)
    fig.savefig(out / "stacked_financial_cost_premium_plus_oop_owner_renter.png",
                bbox_inches="tight")
    plt.close(fig)



def plot_payout_owner_renter(
    fin_dir: Path,
    fig_root: Path,
    drop_years=(),
    severe_years=(2011, 2014, 2021),
    *,
    show_rate: bool = False,           # ← 開關：是否加右軸 payout rate 折線
    rate_ylim: Optional[Tuple[float,float]] = (0, 0.4),  # 例如 (0,1.0)
    poster: bool = False,
) -> None:
    """
    (a) Homeowner / (b) Renter
    柱：每戶平均 payout（USD）with 95% CI error bars
    若 show_rate=True：右軸折線＝總 payout / 總 flood damage（同身份）。
    """
    from scipy import stats
    
    tables = _load_finance_household_years(fin_dir)
    if not tables: return

    rows = []
    for df in tables:
        if "__year" not in df.columns: continue
        y = int(df["__year"].iloc[0])
        if y in set(drop_years): continue

        if df.columns.str.lower().isin(["group","identity"]).any():
            idcol = df.columns[df.columns.str.lower().isin(["group","identity"])][0]
            ident = _find_identity(df[idcol])
        else:
            ident = pd.Series(["unknown"] * len(df))
        
        # 加入 tract_geoid 以計算跨 tract 的 CI
        tract_col = None
        for c in ["tract_geoid", "tract", "GEOID", "geoid"]:
            if c in df.columns:
                tract_col = c
                break

        rows.append(pd.DataFrame({
            "year": y,
            "identity": ident,
            "tract": df[tract_col] if tract_col else "unknown",
            "payout_usd": _payout_total_usd(df),
        }))

    if not rows: return
    hh  = pd.concat(rows, ignore_index=True)
    hh  = hh[hh["identity"].isin(["owner","renter"])]

    # 年×身分：總 payout 與每戶平均
    tot = (hh.groupby(["year","identity"], as_index=False)[["payout_usd"]].sum())
    cnt = (hh.groupby(["year","identity"]).size().rename("n_households").reset_index())
    grp = tot.merge(cnt, on=["year","identity"], how="left")
    grp["payout_avg"] = np.where(grp["n_households"]>0, grp["payout_usd"]/grp["n_households"], 0.0)

    years = sorted(grp["year"].unique()); x = np.arange(len(years)); w = 0.72
    if not years: return

    def _avg_series(iden: str) -> np.ndarray:
        s = grp[grp["identity"].eq(iden)].set_index("year")
        return np.array([float(s["payout_avg"].get(y,0.0)) for y in years])

    # 供 rate 使用的「總額」序列
    def _sum_series(iden: str) -> np.ndarray:
        s = grp[grp["identity"].eq(iden)].set_index("year")
        return np.array([float(s["payout_usd"].get(y,0.0)) for y in years])
    
    def _ci_series(iden: str, n_bootstrap: int = 1000) -> np.ndarray:
        """Calculate 95% CI for payout using bootstrap resampling."""
        ci_list = []
        sub = hh[hh["identity"].eq(iden)]
        for y in years:
            year_data = sub[sub["year"] == y].copy()
            if len(year_data) > 0:
                tract_sums = year_data.groupby("tract")[["payout_usd"]].sum()
                values = tract_sums["payout_usd"].values
                
                # Exclude outliers using IQR
                if len(values) > 4:
                    q1, q3 = np.percentile(values, [25, 75])
                    iqr = q3 - q1
                    mask = (values >= q1 - 1.5 * iqr) & (values <= q3 + 1.5 * iqr)
                    values = values[mask]
                
                if len(values) > 1:
                    # Bootstrap resampling
                    np.random.seed(42)  # Reproducibility
                    boot_means = []
                    for _ in range(n_bootstrap):
                        sample = np.random.choice(values, size=len(values), replace=True)
                        boot_means.append(np.mean(sample))
                    boot_means = np.array(boot_means)
                    ci_low = np.percentile(boot_means, 2.5)
                    ci_high = np.percentile(boot_means, 97.5)
                    ci = (ci_high - ci_low) / 2
                    # Normalize by average household count per tract
                    avg_hh_per_tract = len(year_data) / len(tract_sums) if len(tract_sums) > 0 else 1
                    ci_list.append(ci / max(avg_hh_per_tract, 1))
                else:
                    ci_list.append(0)
            else:
                ci_list.append(0)
        return np.array(ci_list)

    avg_O, avg_R = _avg_series("owner"), _avg_series("renter")
    sum_O, sum_R = _sum_series("owner"), _sum_series("renter")
    ci_O, ci_R = _ci_series("owner"), _ci_series("renter")

    # 讀取 flood damage（tract 級 → 年度總額）
    own_dmg, ren_dmg = None, None
    # ---------- 讀取 flood damage（tract 級 → 年度總額） ----------
    own_dmg, ren_dmg = None, None
    if show_rate:
        owner_all, renter_all, _ = _read_damage_all_tables(fig_root)
        own_dmg = owner_all.groupby("year")["damage_usd"].sum() if not owner_all.empty else pd.Series(dtype=float)
        ren_dmg = renter_all.groupby("year")["damage_usd"].sum() if not renter_all.empty else pd.Series(dtype=float)

        def _rate(sum_vec, dmg_series):
            vals = []
            for i, y in enumerate(years):
                den = float(dmg_series.get(y, np.nan)) if dmg_series is not None else np.nan
                num = float(sum_vec[i])
                if (not np.isfinite(den)) or den <= 0:
                    vals.append(np.nan)
                else:
                    vals.append(num / den)
            return np.array(vals, dtype=float)
        rate_O = _rate(sum_O, own_dmg)
        rate_R = _rate(sum_R, ren_dmg)

    _set_style()
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, PercentFormatter
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    C_BAR = "#F58518"   # payout 橘

    # Font size overrides for readability
    TITLE_FONTSIZE = 24
    LABEL_FONTSIZE = 22
    TICK_FONTSIZE  = 20

    figsize = FIGSIZE_POSTER if poster else FIGSIZE_PAPER
    fig, axes = plt.subplots(2, 1, figsize=figsize, constrained_layout=True, sharex=True)

    def _shade(ax):
        sev = set(int(v) for v in severe_years)
        for i, y in enumerate(years):
            if int(y) in sev:
                ax.axvspan(i - 0.45, i + 0.45, color="#93c5fd", alpha=0.20, zorder=0)

    def _k_formatter(x, pos):
        if x >= 1000: return f'${x/1000:.0f}k'
        return f'${x:.0f}'

    def _draw(ax, avg_vec, title, label, rate_vec=None, show_legend=True):
        _shade(ax)
        ax.bar(x, avg_vec, width=w, color=C_BAR, alpha=0.92, label="Payout")
        ax.bar(x, avg_vec, width=w, fill=False, edgecolor="#111827", linewidth=0.8)
        
        ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight="bold")
        ax.set_ylabel("Average per HH (USD)", fontsize=LABEL_FONTSIZE)
        ax.yaxis.set_major_formatter(FuncFormatter(_k_formatter))
        ax.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
        
        current_max = np.nanmax(avg_vec) if len(avg_vec) > 0 else 1
        ax.set_ylim(0, current_max * 1.35)
        
        even_indices = [i for i, y in enumerate(years) if y % 2 == 0]
        ax.set_xticks(even_indices)
        ax.set_xticklabels([str(y) for y in years if y % 2 == 0], fontsize=TICK_FONTSIZE)
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        _panel_label(ax, label)

        if show_rate and (rate_vec is not None):
            ax2 = ax.twinx()
            m = np.isfinite(rate_vec)
            line_handle = ax2.plot(x[m], rate_vec[m], color="black", linewidth=2.5,
                                   linestyle="-", marker="o", markersize=6.0, label="Payout rate")[0]
            ax2.set_ylabel("Payout rate", labelpad=12, color="black", fontsize=LABEL_FONTSIZE)
            ax2.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
            ax2.tick_params(axis='y', labelcolor='black', labelsize=TICK_FONTSIZE)
            ax2.set_ylim(0, 1.0)
            ax2.grid(False)

            if show_legend:
                payout_patch = Patch(facecolor=C_BAR, alpha=0.92, edgecolor="#111827", label="Payout")
                rate_line = Line2D([0], [0], color="black", linewidth=2.5, marker="o", label="Payout rate")
                sev_patch = Patch(facecolor="#93c5fd", alpha=0.20, label="Severe flood")
                ax.legend(handles=[payout_patch, rate_line, sev_patch], loc="upper left", 
                          frameon=True, facecolor='white', edgecolor="black", fontsize=LEGEND_FONTSIZE)

    _draw(axes[0], avg_O, "Homeowner", "(a)", rate_O if show_rate else None, show_legend=True)
    _draw(axes[1], avg_R, "Renter", "(b)", rate_R if show_rate else None, show_legend=False)
    axes[1].set_xlabel("Simulation Year", fontsize=LABEL_FONTSIZE, fontweight="bold", labelpad=12)

    out = _ensure_fin_ts_dir(fig_root)
    fname = "payout_owner_renter_with_rate.png" if show_rate else "payout_owner_renter.png"
    fig.savefig(out / fname, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fname}")

# (re-imports removed; already imported at top of file)

def plot_loss_attribution_stacked_percent(
    fin_dir: Path,
    fig_root: Path,
    *,
    deductible_usd: float = 1_000.0,
    owner_limit_usd: float = 250_000.0,
    renter_limit_usd: float = 100_000.0,
    severe_years: tuple[int, ...] = (2011, 2014, 2021),
    poster: bool = False,
    drop_years: tuple[int, ...] = (),
) -> None:
    """
    Homeowner / Renter 兩個面板：
    以「各年該身分類別的 *總* flood gross damage (=分母)」為基準，
    堆疊顯示其由 4 類所貢獻的比例（%）：
      1) 未投保戶(Uninsured)的損失
      2) 投保戶中：低於 deductible 的部分
      3) 投保戶中：介於 deductible–limit 的部分（保單承保層）
      4) 投保戶中：超過 limit 的部分
    視覺：右側外框 legend、嚴重年份以淡藍直條標示；與現有圖同 style。
    存檔：fig_root / "loss_attribution_stacked_percent_owner_renter.png"
    依賴的工具：_load_finance_household_years, _set_style, _legend_box_right, _find_identity（與你專案一致）
    """

    # ---------- 讀資料 ----------
    tables = _load_finance_household_years(fin_dir)
    if not tables:
        return

    rows = []
    for df in tables:
        if "__year" not in df.columns:
            continue
        y = int(df["__year"].iloc[0])
        if y in set(drop_years):
            continue

        # 身分欄（容錯，沿用你的識別器）
        if df.columns.str.lower().isin(["group", "identity", "tenure", "owner_renter"]).any():
            idcol = df.columns[df.columns.str.lower().isin(["group", "identity", "tenure", "owner_renter"])][0]
            identity = _find_identity(df[idcol])  # 需回傳 owner / renter / …
        else:
            identity = pd.Series(["unknown"] * len(df))

        # 是否投保（優先使用布林欄；否則以 premium>0 推斷）
        def _bool_col(cands):
            mask = df.columns.str.lower().isin([c.lower() for c in cands])
            if mask.any():
                c = df.columns[mask][0]
                s = df[c]
                return (s if s.dtype == bool else s.astype(str).str.lower().isin({"1","true","t","yes","y"}))
            return None

        insured = _bool_col(["insured","is_insured","has_insurance","insured_flag","policy"])
        if insured is None:
            # 以 premium 是否>0 判斷投保
            prem_col_mask = df.columns.str.lower().isin(["premium_total_usd","premium_usd","premium"])
            if prem_col_mask.any():
                pc = df.columns[prem_col_mask][0]
                insured = pd.to_numeric(df[pc], errors="coerce").fillna(0.0).gt(0.0)
            else:
                insured = pd.Series([False]*len(df))

        # Flood gross damage（優先使用 gross_total_kUSD）
        def _col(name_list):
            mask = df.columns.str.lower().isin([n.lower() for n in name_list])
            return df.columns[mask][0] if mask.any() else None

        col_gross_k = _col(["gross_total_kUSD","gross_total_kusd","gross_total_kusd "])
        col_gross   = _col(["gross_total_usd","gross_usd","flood_damage_usd","damage_usd"])
        if col_gross_k is not None:
            gross = pd.to_numeric(df[col_gross_k], errors="coerce").fillna(0.0) * 1000.0
        elif col_gross is not None:
            gross = pd.to_numeric(df[col_gross], errors="coerce").fillna(0.0)
        else:
            # 退而求其次：payout + OOP（可能略低估真正 gross）
            pc = _col(["payout_total_usd","payout_usd"])
            oc = _col(["oop_total_usd","oop_usd"])
            p = pd.to_numeric(df[pc], errors="coerce").fillna(0.0) if pc else 0.0
            o = pd.to_numeric(df[oc], errors="coerce").fillna(0.0) if oc else 0.0
            gross = p + o

        tmp = pd.DataFrame({
            "year": y,
            "identity": identity,
            "gross": gross,
            "insured": insured.astype(bool)
        })
        rows.append(tmp)

    if not rows:
        return

    dat = pd.concat(rows, ignore_index=True)
    dat = dat[dat["identity"].isin(["owner","renter"])]
    dat = dat[dat["gross"] > 0]  # 只統計有損失的 HH

    years = sorted(dat["year"].unique())
    if not years:
        return

    # ---------- 計算各年×身分 的比例堆疊 ----------
    out = []
    for (y, iden), g in dat.groupby(["year","identity"]):
        limit = owner_limit_usd if iden=="owner" else renter_limit_usd
        total = float(g["gross"].sum()) or 1.0

        # 未投保的損失 (= 全額落在「未投保」)
        uninsured_loss = float(g.loc[~g["insured"], "gross"].sum())

        # 已投保：依保單條款拆分
        gi = g.loc[g["insured"], "gross"].to_numpy(float)
        below = np.minimum(gi, deductible_usd).sum()
        inlayer = (np.clip(gi, deductible_usd, limit) - deductible_usd).clip(min=0).sum()
        above = np.maximum(gi - limit, 0).sum()

        out.append({
            "year": int(y),
            "identity": iden,
            "Uninsured loss": uninsured_loss/total*100.0,
            "Below deductible": below/total*100.0,
            "Within layer":     inlayer/total*100.0,
            "Above limit":      above/total*100.0,
        })

    agg = pd.DataFrame(out).sort_values(["identity","year"])

    # ---------- 視覺 (Stacked Area Chart) ----------
    _set_style()
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
    from matplotlib.patches import Patch

    # More harmonious color palette
    C_UNINS = "#6b7280"  # slate gray
    C_BELOW = "#fbbf24"  # warm amber
    C_MID   = "#34d399"  # emerald green
    C_ABOVE = "#f87171"  # coral red
    C_SEV   = "#93c5fd"  # light blue

    order  = ["Uninsured loss","Below deductible","Within layer","Above limit"]
    colors = [C_UNINS, C_BELOW, C_MID, C_ABOVE]

    # Figure setup
    figsize = FIGSIZE_POSTER if poster else FIGSIZE_PAPER
    fig, axes = plt.subplots(2, 1, figsize=figsize, constrained_layout=True, sharex=True)

    def _panel(ax, iden: str, title: str, tag: str, show_legend: bool = False):
        sub = agg[agg["identity"].eq(iden)].sort_values("year")

        # Prepare data for stacked area
        y_vals = []
        for key in order:
            vals = np.array([float(sub.loc[sub["year"].eq(y), key].values[0]) if (sub["year"]==y).any() else 0.0 for y in years])
            y_vals.append(vals)

        stack = ax.stackplot(years, y_vals, labels=order, colors=colors, alpha=0.85)

        # Shade severe years
        severe = set(int(v) for v in severe_years)
        for i, y in enumerate(years):
            if int(y) in severe:
                ax.axvline(y, color=C_SEV, alpha=0.3, linewidth=8, zorder=0)

        ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight="bold", pad=12)
        
        # Override Y-label with explicit fontsize
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
        ax.set_ylim(0, 100)
        ax.set_xlim(min(years), max(years))
        
        ax.set_ylabel("Share of total gross damage (%)", fontsize=LABEL_FONTSIZE, labelpad=10)
        ax.grid(True, axis="y", linestyle="--", alpha=0.35, zorder=1)
        ax.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
        
        # Set x-ticks to show even years only
        even_years = [y for y in years if y % 2 == 0]
        ax.set_xticks(even_years)
        ax.set_xticklabels([str(y) for y in even_years], fontsize=TICK_FONTSIZE)
        
        try:
            _panel_label(ax, tag)
        except Exception:
            pass
        
        # Legend inside Renter panel only - upper left, centered around 2013
        if show_legend:
            handles = [
                Patch(facecolor=C_UNINS, alpha=0.90, edgecolor="#374151", linewidth=1.0, label="Uninsured loss"),
                Patch(facecolor=C_BELOW, alpha=0.90, edgecolor="#374151", linewidth=1.0, label="Below deductible"),
                Patch(facecolor=C_MID,   alpha=0.90, edgecolor="#374151", linewidth=1.0, label="Covered by insurance"),
                Patch(facecolor=C_ABOVE, alpha=0.90, edgecolor="#374151", linewidth=1.0, label="Above limit"),
            ]
            # Position legend at upper left, more towards left edge
            legend = ax.legend(handles=handles, loc="upper left",
                               frameon=True, facecolor='white',
                               edgecolor="black", framealpha=1.0,
                               fontsize=LEGEND_FONTSIZE, ncol=1,
                               handlelength=1.5, handleheight=1.0)
            legend.get_frame().set_linewidth(1.2)

    # Figure setup
    figsize = FIGSIZE_POSTER if poster else FIGSIZE_PAPER
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True, constrained_layout=True)
    
    # Constants for aesthetics (match other fin plots)
    TITLE_FONTSIZE = 24
    LABEL_FONTSIZE = 22
    TICK_FONTSIZE = 20
    
    _panel(axes[0], "owner",  "Homeowner", "(a)", show_legend=False)
    _panel(axes[1], "renter", "Renter", "(b)", show_legend=True)
    axes[1].set_xlabel("Simulation Year", fontsize=LABEL_FONTSIZE, fontweight="bold", labelpad=12)

    out = _ensure_fin_ts_dir(fig_root)
    fig.savefig(out / "loss_attribution_stacked_percent_owner_renter.png", dpi=300, bbox_inches="tight")
    plt.close(fig)



# ------------------------- Decisions: action shares boxplots -------------------------

COLOR_FI = "#2563eb"   # blue
COLOR_EH = "#16a34a"   # green
COLOR_BP = "#f59e0b"   # amber
COLOR_RL = "#dc2626"   # red
COLOR_DN = "#7c3aed"   # violet

def _percentiles_from_per_year_arrays(per_year_arrays, p):
    """per_year_arrays: List[np.ndarray]; p: 百分位(0-100) or list"""
    if isinstance(p, (list, tuple)):
        res = []
        for pi in p:
            res.append(np.array([
                (np.percentile(v, pi) if (isinstance(v, np.ndarray) and v.size > 0) else np.nan)
                for v in per_year_arrays
            ]))
        return res
    else:
        return np.array([
            (np.percentile(v, p) if (isinstance(v, np.ndarray) and v.size > 0) else np.nan)
            for v in per_year_arrays
        ])
    


def _ensure_decision_fig_dir(fig_root: Path) -> Path:
    dec_dir = (fig_root / "decisions")
    dec_dir.mkdir(parents=True, exist_ok=True)
    return dec_dir

# 1) 欄位設定：Homeowner panel order: FI, BP, EH, DN (swapped EH and DN per user request)
OWNER_ACT     = ["FI", "BP", "EH coverage", "DN"]
OWNER_COLS    = ["owner_share_FI", "owner_share_BP", "eh_coverage_owner_cum", "owner_share_DN"]
OWNER_COLORS  = [COLOR_FI, COLOR_BP, COLOR_EH, COLOR_DN]
OWNER_YLABELS = ["FI share", "BP share", "EH coverage", "DN share"]

EH_RIBBON_COLOR = "#ceebd8"   # IQR 帶狀（amber-200 近似）
EH_LINE_COLOR   = "#16a34a"   # 中位數線（amber-600 近似）

def _percentiles_from_per_year_arrays(per_year_arrays: List[np.ndarray], p_list: List[float]) -> List[np.ndarray]:
    out = []
    for p in p_list:
        out.append(np.array([
            (np.percentile(v, p) if (isinstance(v, np.ndarray) and v.size > 0) else np.nan)
            for v in per_year_arrays
        ]))
    return out

# ======== Palette (替換你原本的顏色常數) ========
COLOR_FI = "#1f77b4"   # FI – calm blue
COLOR_BP = "#ff7f0e"   # BP – amber
COLOR_RL = "#d62728"   # RL – red
COLOR_DN = "#9467bd"   # DN – violet
COLOR_EH = "#2ca02c"   # EH – green

SEVERE_FILL = "#93c5fd"       # severe year shade
LINE_TOTAL = "#111827"        # overlay total-ratio line (黑灰)
EH_RIBBON_COLOR = "#bde7c3"   # EH IQR ribbon
EH_LINE_COLOR   = COLOR_EH

def plot_action_shares_owner_renter_separate_panels(
    decisions_dir: Path,
    fig_root: Path,
    severe_marks: Sequence[int] = (14, 21, 22, 23),
) -> None:
    """
    讀取 decisions/ 下的 tract-level 行為比例檔：
      - action_share_owner_renter_tract_YYYY.csv（畫箱型圖）
      - action_share_owner_renter_tract_all_years.csv（用來計算『Total ratio』折線）
    產出兩張圖：
      1) Renter：FI / RL / DN（三列）
      2) Homeowner ：FI / BP / DN / EH（四列；EH 為 IQR 帶＋折線）

    折線定義（年度）優先順序：
      1) Σ act_count / Σ group_count
      2) Σ(share * group_count) / Σ group_count
      3) mean(share)
    若仍全 NaN → 以箱型圖資料的年度平均值當保底，確保折線一定畫出來。
    """
    # ---------- 色系（箱 = 淺色；折線 = 深紅色虛線；EH 另配綠色） ----------
    C_FI_BOX = "#60a5fa"      # renter/owner FI box
    C_RL_BOX = "#fca5a5"      # renter RL box
    C_DN_BOX = "#c4b5fd"      # renter/owner DN box
    C_BP_BOX = "#fcd34d"      # owner BP box
    C_LINE   = "#b91c1c"      # ← 全部(除EH)折線統一深紅
    LINE_KW  = dict(color=C_LINE, linewidth=2.6, linestyle="--", marker="o", markersize=5.5, zorder=3)

    EH_RIBBON, EH_LINE = "#bbf7d0", "#15803d"  # EH IQR 與中位數線
    SHADE = "#93c5fd"                          # 嚴重洪年底色

    _set_style()
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    # ---------- 逐年檔（箱型圖來源） ----------
    files = sorted(decisions_dir.glob("action_share_owner_renter_tract_*.csv"))
    if not files:
        return

    year_tables: dict[int, pd.DataFrame] = {}
    for f in files:
        try:
            df = pd.read_csv(f)
            yvals = pd.to_numeric(df.get("year"), errors="coerce").dropna().unique()
            if len(yvals) == 0:
                continue
            y = int(yvals[0])
            # ✅ 不再排除 2011
            year_tables[y] = df
        except Exception:
            continue
    if not year_tables:
        return

    years = sorted(year_tables.keys())
    x = np.arange(len(years))

    # ---------- severe 年對應索引（可用 2 位數標示） ----------
    def _resolve_marks(marks: Sequence[int]) -> list[int]:
        idxs = set()
        idx_by_suffix: dict[int, list[int]] = {}
        for i, y in enumerate(years):
            idx_by_suffix.setdefault(y % 100, []).append(i)
        for m in marks:
            if m in years:
                idxs.add(years.index(m))
            elif 1 <= m <= len(years):
                idxs.add(m - 1)
            else:
                suf = m % 100
                idxs.update(idx_by_suffix.get(suf, []))
        return sorted(idxs)
    severe_idx = _resolve_marks(severe_marks)

    # ---------- 小工具 ----------
    def _shade(ax):
        for i in severe_idx:
            ax.axvspan(i - 0.48, i + 0.48, color=SHADE, alpha=0.28, zorder=0)

    def _first_col(df: pd.DataFrame, cands: list[str]) -> str | None:
        lower = {c.lower(): c for c in df.columns}
        for c in cands:
            if c in df.columns:
                return c
            if c.lower() in lower:
                return lower[c.lower()]
        return None

    def _collect_arrays(col: str) -> list[np.ndarray]:
        arrs = []
        for y in years:
            v = pd.to_numeric(year_tables[y].get(col, pd.Series(dtype=float)),
                              errors="coerce").dropna().to_numpy()
            arrs.append(v)
        return arrs

    def _tight_ylim(per_year_arrays: list[np.ndarray], min_span: float = 0.02) -> tuple[float, float]:
        vals = [v for v in per_year_arrays if isinstance(v, np.ndarray) and v.size > 0]
        if not vals:
            return (0.0, 1.0)
        a = np.concatenate(vals)
        q5, q95 = np.nanpercentile(a, 3), np.nanpercentile(a, 97)
        pad = 0.02 * max(1e-9, q95 - q5)
        lo, hi = max(0.0, q5 - pad), min(1.0, q95 + pad)
        if hi - lo < min_span:
            hi = min(1.0, lo + min_span)
        return (lo, hi)

    # ========== 從 all_years 檔案計算『Total ratio』 ==========
    all_years_path = decisions_dir / "action_share_owner_renter_tract_all_years.csv"
    if not all_years_path.exists():
        df_all = None
    else:
        df_all = pd.read_csv(all_years_path, dtype={"tract_geoid": str})
        df_all["year"] = pd.to_numeric(df_all.get("year"), errors="coerce")
        df_all = df_all.dropna(subset=["year"]).copy()
        df_all["year"] = df_all["year"].astype(int)
        # 對齊年份（不再主動移除 2011）
        df_all = df_all[df_all["year"].isin(years)].copy()

    def _total_ratio_from_all(df: pd.DataFrame, key: str, y: int) -> float:
        """key ∈ {owner_FI, owner_BP, owner_DN, renter_FI, renter_RL, renter_DN}"""
        d = df[df["year"] == y]
        if d.empty:
            return np.nan
        group = "owner" if key.startswith("owner_") else "renter"
        act   = key.split("_", 1)[1]  # FI/BP/DN/RL

        den_col = _first_col(d, [f"{group}_count", f"n_{group}", f"{group}_n", f"{group}_size", f"{group}s"])
        num_col = _first_col(d, [f"{group}_count_{act}", f"n_{group}_{act}", f"{act}_{group}_n",
                                 f"{act}_count_{group}", f"{group}_{act}_count"])
        share_col = _first_col(d, [f"{group}_share_{act}"])

        # 1) 明確計數
        if num_col and den_col:
            num = pd.to_numeric(d[num_col], errors="coerce").fillna(0.0).sum()
            den = pd.to_numeric(d[den_col], errors="coerce").fillna(0.0).sum()
            return float(num/den) if den > 0 else np.nan
        # 2) share × group 權重
        if share_col and den_col:
            s = pd.to_numeric(d[share_col], errors="coerce")
            w = pd.to_numeric(d[den_col],  errors="coerce").fillna(0.0)
            sw, ww = (s * w).sum(skipna=True), w.sum(skipna=True)
            return float(sw/ww) if ww > 0 else np.nan
        # 3) 未加權平均
        if share_col:
            s = pd.to_numeric(d[share_col], errors="coerce")
            return float(s.mean(skipna=True)) if s.notna().any() else np.nan
        return np.nan

    def _totals_line_from_all(key: str, fallback_arrays: list[np.ndarray]) -> np.ndarray:
        if df_all is None or df_all.empty:
            return np.array([
                (float(np.nanmean(v)) if isinstance(v, np.ndarray) and v.size > 0 else np.nan)
                for v in fallback_arrays
            ], dtype=float)
        vals = np.array([_total_ratio_from_all(df_all, key, y) for y in years], dtype=float)
        if np.isnan(vals).all():
            vals = np.array([
                (float(np.nanmean(v)) if isinstance(v, np.ndarray) and v.size > 0 else np.nan)
                for v in fallback_arrays
            ], dtype=float)
        return vals

    # 需要的箱型圖陣列（供保底 & y 軸自動）
    own_fi_arrays = _collect_arrays("owner_share_FI")
    own_bp_arrays = _collect_arrays("owner_share_BP")
    own_dn_arrays = _collect_arrays("owner_share_DN")
    rent_fi_arrays= _collect_arrays("renter_share_FI")
    rent_rl_arrays= _collect_arrays("renter_share_RL")
    rent_dn_arrays= _collect_arrays("renter_share_DN")

    # 從 all_years 得到每面板的折線（如果需要可啟用畫線）
    owner_fi_line  = _totals_line_from_all("owner_FI",  own_fi_arrays)
    owner_bp_line  = _totals_line_from_all("owner_BP",  own_bp_arrays)
    owner_dn_line  = _totals_line_from_all("owner_DN",  own_dn_arrays)
    renter_fi_line = _totals_line_from_all("renter_FI", rent_fi_arrays)
    renter_rl_line = _totals_line_from_all("renter_RL", rent_rl_arrays)
    renter_dn_line = _totals_line_from_all("renter_DN", rent_dn_arrays)

    # ========== 繪圖：箱型圖 + 折線（折線預設關閉；要開把註解取消） ==========
    def _draw_boxline(ax, arrays, box_color, title, totals, ylim=None):
        _shade(ax)
        bp = ax.boxplot(arrays, positions=x, widths=0.55, patch_artist=True,
                        showfliers=False, whis=[10, 90], zorder=1)
        for box in bp["boxes"]:
            box.set_facecolor(box_color); box.set_alpha(0.80)
            box.set_edgecolor("#374151"); box.set_linewidth(1.3)  # Darker edge
        for med in bp["medians"]:
            med.set_color("#111827"); med.set_linewidth(2.2)
        for w in bp["whiskers"] + bp["caps"]:
            w.set_color("#4b5563"); w.set_linewidth(1.2)

        # 若想顯示總比例折線，解除下面兩行註解
        # m = np.isfinite(totals)
        # if m.any(): ax.plot(x[m], totals[m], **LINE_KW)

        ax.set_ylim(*(ylim if ylim else _tight_ylim(arrays)))
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
        ax.set_xlim(-0.6, len(x) - 0.4)
        # Only show even years
        even_years = [y for y in years if y % 2 == 0]
        even_indices = [i for i, y in enumerate(years) if y % 2 == 0]
        ax.set_xticks(even_indices)
        ax.set_xticklabels([str(y) for y in even_years])
        ax.set_title(title, loc="center")
        ax.grid(True, axis="y", linestyle="--", alpha=0.25)

    # Shared Y-axes: compute unified range from both owner and renter for comparable panels
    _dn_shared_ylim = _tight_ylim(own_dn_arrays + rent_dn_arrays)
    _fi_shared_ylim = _tight_ylim(own_fi_arrays + rent_fi_arrays)

    # ===== RENTER =====
    fig_r, axs_r = plt.subplots(3, 1, figsize=(10, 8.5), sharex=True, constrained_layout=True)
    _draw_boxline(axs_r[0], rent_fi_arrays, C_FI_BOX, "", renter_fi_line, ylim=_fi_shared_ylim)
    axs_r[0].set_title('Renter', fontweight='bold', fontsize=16, loc='center')
    axs_r[0].set_ylabel("FI share")
    axs_r[0].text(-0.08, 1.05, '(e)', transform=axs_r[0].transAxes, fontweight='bold', fontsize=14, va='bottom', ha='left')
    _draw_boxline(axs_r[1], rent_rl_arrays, C_RL_BOX, "", renter_rl_line)
    axs_r[1].set_ylabel("RL share")
    axs_r[1].text(-0.08, 1.05, '(f)', transform=axs_r[1].transAxes, fontweight='bold', fontsize=14, va='bottom', ha='left')
    _draw_boxline(axs_r[2], rent_dn_arrays, C_DN_BOX, "", renter_dn_line, ylim=_dn_shared_ylim)
    axs_r[2].set_ylabel("DN share")
    axs_r[2].text(-0.08, 1.05, '(g)', transform=axs_r[2].transAxes, fontweight='bold', fontsize=14, va='bottom', ha='left')
    axs_r[-1].set_xlabel("Simulation Year", fontsize=15, fontweight="bold", labelpad=12)

    # Renter legend in upper right
    out_dir = _ensure_decision_fig_dir(fig_root)
    leg_handles_r = [
        Patch(facecolor=SHADE, alpha=0.28, edgecolor="none", label="Severe flood year"),
    ]
    axs_r[0].legend(handles=leg_handles_r, loc="upper right", frameon=True,
                    framealpha=0.95, edgecolor="#e5e7eb", fontsize=LEGEND_FONTSIZE)
    fig_r.savefig(out_dir / "action_shares_renter_separate_panels.png", bbox_inches="tight")
    plt.close(fig_r)

    # ===== OWNER =====
    fig_o, axs_o = plt.subplots(4, 1, figsize=(10, 11), sharex=True, constrained_layout=True)
    _draw_boxline(axs_o[0], own_fi_arrays, C_FI_BOX, "", owner_fi_line, ylim=_fi_shared_ylim)
    axs_o[0].set_title('Homeowner', fontweight='bold', fontsize=16, loc='center')
    axs_o[0].set_ylabel("FI share")
    axs_o[0].text(-0.08, 1.05, '(a)', transform=axs_o[0].transAxes, fontweight='bold', fontsize=14, va='bottom', ha='left')
    _draw_boxline(axs_o[1], own_bp_arrays, C_BP_BOX, "", owner_bp_line)
    axs_o[1].set_ylabel("BP share")
    axs_o[1].text(-0.08, 1.05, '(b)', transform=axs_o[1].transAxes, fontweight='bold', fontsize=14, va='bottom', ha='left')
    _draw_boxline(axs_o[2], own_dn_arrays, C_DN_BOX, "", owner_dn_line, ylim=_dn_shared_ylim)
    axs_o[2].set_ylabel("DN share")
    axs_o[2].text(-0.08, 1.05, '(c)', transform=axs_o[2].transAxes, fontweight='bold', fontsize=14, va='bottom', ha='left')

    # EH coverage（IQR + 中位折線 + 點）
    all_years_path2 = decisions_dir / "action_share_owner_renter_tract_all_years.csv"
    if all_years_path2.exists():
        df_all2 = pd.read_csv(all_years_path2, dtype={"tract_geoid": str})
        df_all2["year"] = pd.to_numeric(df_all2.get("year"), errors="coerce").astype("Int64")
        df_all2 = df_all2.dropna(subset=["year"]).copy()
        df_all2["year"] = df_all2["year"].astype(int)
        df_all2 = df_all2[df_all2["year"].isin(years)]
        
        # Calculate EH rate among eligible population: owner_share_EH / (1 - eh_coverage_owner_cum_prior)
        # For year t: eligible = 1 - coverage at year t-1
        # Approximate: eligible = 1 - (coverage_t - share_EH_t)
        eh_col = "owner_share_EH" if "owner_share_EH" in df_all2.columns else None
        cov_col = "eh_coverage_owner_cum"
        
        eh_eligible_arrays = []
        for y in years:
            ydf = df_all2[df_all2["year"] == y]
            if ydf.empty or eh_col is None:
                eh_eligible_arrays.append(np.array([]))
                continue
            
            share_eh = pd.to_numeric(ydf.get(eh_col, 0), errors="coerce").fillna(0).to_numpy()
            cov = pd.to_numeric(ydf.get(cov_col, 0), errors="coerce").fillna(0).to_numpy()
            # Estimate prior coverage: cov - share_eh (approx)
            prior_cov = np.clip(cov - share_eh, 0, 1)
            eligible = 1 - prior_cov
            # Mask out tracts where eligible population is too small (< 15%)
            # to avoid noisy rate calculations
            eh_rate_among_eligible = np.where(
                eligible >= 0.15,
                np.clip(share_eh / eligible, 0, 1),
                np.nan  # Mark as NaN when too few eligible
            )
            # Filter out NaN values
            valid_rates = eh_rate_among_eligible[~np.isnan(eh_rate_among_eligible)]
            eh_eligible_arrays.append(valid_rates if len(valid_rates) > 0 else np.array([0.0]))
    else:
        # Fallback: use raw EH share
        eh_eligible_arrays = _collect_arrays("owner_share_EH")

    ax = axs_o[3]
    _shade(ax)
    # Draw boxplot instead of IQR+line
    bp = ax.boxplot(eh_eligible_arrays, positions=x, widths=0.55, patch_artist=True,
                    showfliers=False, whis=[10, 90], zorder=1)
    for box in bp["boxes"]:
        box.set_facecolor(EH_RIBBON); box.set_alpha(0.80)
        box.set_edgecolor("#374151"); box.set_linewidth(1.3)
    for med in bp["medians"]:
        med.set_color(EH_LINE); med.set_linewidth(2.2)
    for w in bp["whiskers"] + bp["caps"]:
        w.set_color("#4b5563"); w.set_linewidth(1.2)
    
    ax.set_ylim(*_tight_ylim(eh_eligible_arrays))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_xlim(-0.6, len(x) - 0.4)
    # Only show even years
    even_years = [y for y in years if y % 2 == 0]
    even_indices = [i for i, y in enumerate(years) if y % 2 == 0]
    ax.set_xticks(even_indices)
    ax.set_xticklabels([str(y) for y in even_years])
    ax.set_title("", loc="center")  # No title, suptitle is enough
    ax.text(-0.08, 1.05, '(d)', transform=ax.transAxes, fontweight='bold', fontsize=14, va='bottom', ha='left')
    ax.set_ylabel("EH rate (eligible)")
    axs_o[-1].set_xlabel("Simulation Year", fontsize=15, fontweight="bold", labelpad=12)
    ax.grid(True, axis="y", linestyle="--", alpha=0.25)

    # Homeowner legend in upper right
    leg_items_o = [
        Patch(facecolor=SHADE, alpha=0.28, edgecolor="none", label="Severe flood year"),
    ]
    axs_o[0].legend(handles=leg_items_o, loc="upper right", frameon=True,
                    framealpha=0.95, edgecolor="#e5e7eb", fontsize=LEGEND_FONTSIZE)

    out_dir = _ensure_decision_fig_dir(fig_root)
    fig_o.savefig(out_dir / "action_shares_owner_separate_panels.png", bbox_inches="tight")
    plt.close(fig_o)
# ------------------------- Decisions: action counts stacked bar -------------------------

# (duplicate imports removed)

def plot_action_counts_stacked_from_split(
    decisions_dir: Path,
    fig_root: Path,
    *,
    drop_years: tuple[int, ...] = (),
    owner_actions: tuple[str, ...] = ("FI","EH","BP","DN"),
    renter_actions: tuple[str, ...] = ("FI","RL","DN"),
    dpi: int = 300,
) -> None:
    """
    Paper-ready stacked bars normalized by 2011 group size.
    Panels:
      (a) Homeowner actions
      (b) Renter actions

    Y-axis = action share vs. 2011 group = (#actions_y) / N_2011
    X-axis = Year
    """
    _set_style()  # Use consistent font and style
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    files = sorted(decisions_dir.glob("action_share_owner_renter_tract_*.csv"))
    if not files: 
        print("[stacked] No input files under", decisions_dir); 
        return

    rows = []
    owner_den_name_seen, renter_den_name_seen = None, None

    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception:
            continue

        if "year" not in df.columns:
            try:
                y = int("".join(ch for ch in f.stem if ch.isdigit())[-4:])
                df["year"] = y
            except Exception:
                continue

        df = df[~df["year"].isin(drop_years)].copy()
        if df.empty: 
            continue

        cols = {c.lower(): c for c in df.columns}
        def col_exact(name: str) -> str | None: return cols.get(name.lower())

        owner_den_col  = next((col_exact(c) for c in
                               ["owner_N_total","owner_n_total","owner_total","owner_count"]
                               if col_exact(c)), None)
        renter_den_col = next((col_exact(c) for c in
                               ["renter_N_total","renter_n_total","renter_total","renter_count"]
                               if col_exact(c)), None)
        if owner_den_col:  owner_den_name_seen  = owner_den_col
        if renter_den_col: renter_den_name_seen = renter_den_col

        def num_col(group: str, act: str) -> str | None:
            for cand in (f"{group}_n_{act}", f"{group}_{act}_n", f"{group}_count_{act}"):
                c = col_exact(cand)
                if c: return c
            return None

        need = ["year"]
        if owner_den_col:  need.append(owner_den_col)
        if renter_den_col: need.append(renter_den_col)
        for a in owner_actions:
            c = num_col("owner", a)
            if c: need.append(c)
        for a in renter_actions:
            c = num_col("renter", a)
            if c: need.append(c)

        agg = df[need].groupby("year").sum(numeric_only=True).reset_index()
        rows.append(agg)

    if not rows:
        print("[stacked] No usable rows after filtering."); 
        return

    yearly = pd.concat(rows, ignore_index=True).groupby("year").sum(numeric_only=True).reset_index()
    years = yearly["year"].astype(int).sort_values().tolist()
    if not years:
        print("[stacked] No years discovered."); 
        return

    # ---- 固定 baseline: 2011 ----
    FIXED_BASELINE_YEAR = 2011
    def _den_at(den_col: str | None, year: int) -> float:
        if not den_col or den_col not in yearly.columns: return 0.0
        row = yearly.loc[yearly["year"] == year]
        if row.empty: return 0.0
        try: return float(row[den_col].values[0])
        except Exception: return 0.0

    base_owner  = _den_at(owner_den_name_seen,  FIXED_BASELINE_YEAR)
    base_renter = _den_at(renter_den_name_seen, FIXED_BASELINE_YEAR)

    # ---- 矩陣：(#action_y) / N_2011 ----
    def stack_matrix(group: str, acts: tuple[str, ...], base: float) -> np.ndarray:
        mat = []
        for a in acts:
            c = next((cand for cand in (f"{group}_n_{a}", f"{group}_{a}_n", f"{group}_count_{a}")
                      if cand in yearly.columns), None)
            vec = (yearly[c] / max(base, 1.0)).values if c else np.zeros(len(yearly))
            mat.append(vec)
        return np.array(mat, float)

    M_owner  = stack_matrix("owner",  owner_actions,  base_owner)
    M_renter = stack_matrix("renter", renter_actions, base_renter)

    # ---- debug CSV ----
    debug = yearly.copy()
    for i,a in enumerate(owner_actions):  debug[f"owner_{a}_vs_2011"]  = M_owner[i]
    for i,a in enumerate(renter_actions): debug[f"renter_{a}_vs_2011"] = M_renter[i]
    out_dbg = fig_root / "decisions" / "_debug_stacked_from_split_yearly_with2011_paper.csv"
    out_dbg.parent.mkdir(parents=True, exist_ok=True)
    debug.to_csv(out_dbg, index=False, encoding="utf-8-sig")

    # ---- 繪圖 ----
    C = {"FI":"#56B4E9","EH":"#009E73","BP":"#E69F00","DN":"#CC79A7","RL":"#D55E00"}  # Okabe–Ito
    x = np.arange(len(years))
    fig, axes = plt.subplots(2, 1, figsize=(10, 7.6), sharex=True, constrained_layout=True)

    def draw(ax, M, acts, panel_label, panel_title):
        bottom = np.zeros(len(years))
        for i, a in enumerate(acts):
            ax.bar(x, M[i], bottom=bottom, width=0.72, color=C.get(a,"#9ca3af"),
                   edgecolor="white", linewidth=0.8, alpha=0.95, label=a)
            bottom += M[i]
        # (a)/(b)
        ax.text(-0.08, 1.04, panel_label, transform=ax.transAxes,
                fontsize=14, fontweight="bold", va="bottom", ha="left")
        # 軸標
        ax.set_ylabel("Action share \n(#actionsᵧ / N_2011)")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
        # Only show even years
        even_years = [y for y in years if y % 2 == 0]
        even_indices = [i for i, y in enumerate(years) if y % 2 == 0]
        ax.set_xticks(even_indices)
        ax.set_xticklabels([str(y) for y in even_years])
        # 精簡標題（保留 baseline 訊息）
        ax.set_title(panel_title + " — baseline scenario (normalized by 2011 group size) ")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    draw(axes[0], M_owner,  owner_actions, "(a)", "Homeowner actions")
    draw(axes[1], M_renter, renter_actions, "(b)", "Renter actions")
    axes[1].set_xlabel("Simulation Year", fontsize=15, fontweight="bold", labelpad=12)

    # Legend inside panel (a) (upper left)
    patches = [Patch(facecolor=C[a], edgecolor="#374151", linewidth=0.8, label=a) for a in ("FI","EH","BP","DN","RL")
               if a in set(owner_actions)|set(renter_actions)]
    patches.append(Patch(facecolor="#93c5fd", alpha=0.25, edgecolor="none", label="Severe flood year"))
    
    # Place legend in axes[0] (upper left), usually 2 columns to save vertical space
    axes[0].legend(handles=patches, loc="upper left", bbox_to_anchor=(0.0, 1.0),
                   frameon=True, ncol=2, fontsize=LEGEND_FONTSIZE,
                   handlelength=1.5, handleheight=1.0, columnspacing=1.0)

    out_dir = fig_root / "decisions"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "action_counts_stacked_baseline_normalized_with2011_paper.png",
                bbox_inches="tight")
    # PDF export disabled
    plt.close(fig)

# (duplicate imports removed)

def plot_action_panels_share_and_counts(
    decisions_dir: Path,
    fig_root: Path,
    *,
    drop_years: tuple[int, ...] = (),
    owner_actions: tuple[str, ...] = ("Both (EH+FI)", "EH", "FI", "BP", "DN"),
    renter_actions: tuple[str, ...] = ("FI", "RL", "DN"),
    dpi: int = 300,
    show_counts: bool = False,
    severe_years=(2011, 2014, 2021)
) -> None:
    """
    Plots action shares with cumulative status tracking for Owners.
    Source: decisions/decisions_mgmix_Y.csv (household-level).
    
    Logic:
      - Tracks unique Owner IDs to maintain 'Ever Elevated' status (set_EH) and 'Cumulative Buyouts' (BP).
      - Denominator: Fixed to 2011 Owner Population (N_2011) for stable 100% stack.
      - Categories (Priority Mutually Exclusive):
        1. BP (Cumulative): Agents who took BP in 2011..Y
        2. Both (EH+FI): In set_EH AND has FI
        3. EH (Only): In set_EH AND no FI
        4. FI (Only): Not in set_EH AND has FI
        5. DN: Remainder to 100%
    """
    _set_style()
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
    from matplotlib.patches import Patch

    # 1. Identify all years
    files = sorted(decisions_dir.glob("decisions_mgmix_*.csv"))
    if not files:
        print("[panels] No decisions_mgmix files found."); return

    # Parse years
    years = []
    file_map = {}
    for f in files:
        try:
            y = int(f.stem.split("_")[-1])
            if y not in drop_years:
                years.append(y)
                file_map[y] = f
        except: pass
    years.sort()
    if not years: return

    # 2. Track Cumulative States
    # We need a stable baseline N from 2011
    # However, we must ensure we strictly track IDs.
    
    # State containers
    set_EH = set()       # IDs who have EVER elevated
    set_BP = set()       # IDs who have taken BP (cumulative)
    
    owner_stats = []     # [{year: Y, "Both":n...}, ...]
    renter_stats = []

    # Constants
    BASELINE_OWNER_N = 0
    
    for i, y in enumerate(years):
        df = pd.read_csv(file_map[y])
        
        # Owners
        owners = df[df['group'] == 'owner']
        
        # Set baseline on first year
        if i == 0:
            BASELINE_OWNER_N = len(owners)
        
        # Current Year Actions (Flow)
        # BP this year: Those who relocate THIS YEAR (they are in the file with action='BP')
        n_bp_current = (owners['action'] == 'BP').sum()
        
        # Track BP IDs just for set_BP if needed (though not strictly necessary with the new logic, 
        # let's keep it if any diagnostic tools use it)
        current_bps = set(owners[owners['action'] == 'BP']['i'])
        set_BP.update(current_bps)
        
        # IDs taking EH *this year* OR clearly showing elevated logic
        # Note: We rely on 'action'=='EH' for the event, but we can also check ELEV_FT > 0 just in case.
        # But based on diagnostics, ELEV_FT seems to link to action.
        # Safe bet: Add anyone with action='EH' to set_EH.
        current_ehs = set(owners[owners['action'] == 'EH']['i'])
        # Also check ELEV_FT > 0 if available, to be robust
        if 'ELEV_FT' in owners.columns:
            elev_users = set(owners[owners['ELEV_FT'] > 0]['i'])
            current_ehs.update(elev_users)
        
        set_EH.update(current_ehs)
        
        # Identify Insured (FI)
        # Check POLICY_NAME or action
        is_fi_mask = (owners['action'] == 'FI')
        if 'POLICY_NAME' in owners.columns:
            # If not null and not empty, has insurance
            has_pol = owners['POLICY_NAME'].notna() & (owners['POLICY_NAME'] != '')
            is_fi_mask = is_fi_mask | has_pol
        
        ids_with_fi = set(owners[is_fi_mask]['i'])
        
        # Calculate Categories for CURRENT residents (excluding those who just BPed? 
        # Usually BP action means they are *leaving*. 
        # If they took BP this year, they are counted in n_BP (cumulative).
        # Residents remaining are Total - BPs?
        # Actually, the file contains everyone present *at decision time*.
        # So "BP" actors are in the file.
        # Strict Priority:
        # 1. BP (Cumulative count, not just this year's action)
        #    Wait, for the stack to sum to BASELINE_OWNER_N:
        #    Stack = (Pop in file) + (Cumulative Left via BP BEFORE this year?).
        #    Actually: 
        #    n_BP_total = len(set_BP)
        #    n_Active = BASELINE_OWNER_N - len(set_BP_prior_years) ?
        #    Let's stick to the counts of *categories*:
        
        # Classification for households PRESENT at the start of the year.
        # Those taking BP this year represent the "Relocatee" share of the current population.
        # Others are classified into Both/EH/FI/DN based on their status.
        
        stayers = owners[owners['action'] != 'BP']
        
        # Vectorized classification of stayers
        is_eh_status = stayers['i'].isin(set_EH)
        is_fi_status = is_fi_mask[stayers.index]
        
        n_both = (is_eh_status & is_fi_status).sum()
        n_eh = (is_eh_status & ~is_fi_status).sum()
        n_fi = (~is_eh_status & is_fi_status).sum()
        n_dn_active = (~is_eh_status & ~is_fi_status).sum()
        
        owner_stats.append({
            "year": y,
            "Both (EH+FI)": n_both,
            "EH": n_eh,
            "FI": n_fi,
            "BP": n_bp_current,  # Annual flow
            "DN": n_dn_active
        })

        # Renters (simplified, usually flow or single action)
        # Renter actions: FI, RL, DN
        renters = df[df['group'] == 'renter']
        r_fi = (renters['action'] == 'FI').sum()
        if 'POLICY_NAME' in renters.columns:
             has_pol = renters['POLICY_NAME'].notna() & (renters['POLICY_NAME'] != '')
             r_fi = ((renters['action'] == 'FI') | has_pol).sum()
             
        r_rl = (renters['action'] == 'RL').sum()
        r_dn = max(0, len(renters) - r_fi - r_rl)
        
        renter_stats.append({
            "year": y,
            "FI": r_fi,
            "RL": r_rl,
            "DN": r_dn,
            "N": len(renters) # Tracking N for renters
        })

    # Prepare DataFrames
    df_own = pd.DataFrame(owner_stats)
    df_rent = pd.DataFrame(renter_stats)
    
    # Debug output
    debug_dir = fig_root / "decisions"
    debug_dir.mkdir(parents=True, exist_ok=True)
    df_own.to_csv(debug_dir / "_debug_action_shares_owners_cumulative.csv", index=False)
    
    # ---- Plotting ----
    C = {
        "Both (EH+FI)": "#4f46e5", "EH": "#059669", "FI": "#60a5fa", 
        "BP": "#f59e0b", "DN": "#9ca3af", "RL": "#dc2626"
    }

    if show_counts:
        fig, axes = plt.subplots(2, 2, figsize=(11, 10), constrained_layout=True, sharex=True)
        ax11, ax12 = axes[0]; ax21, ax22 = axes[1]
        out_suffix = ""
    else:
        fig, (ax11, ax12) = plt.subplots(2, 1, figsize=(11, 10), constrained_layout=True, sharex=True)
        ax21 = ax22 = None
        out_suffix = "_share_only"

    def _panel_label(ax, text):
        ax.text(-0.08, 1.05, text, transform=ax.transAxes, fontsize=16, fontweight="bold", ha="left", va="bottom")

    def _shade_year_bars(ax, years, severe_years, half_width=0.48, color="#93c5fd", alpha=0.15):
        sev = set(int(v) for v in severe_years)
        for i, y in enumerate(years):
            if int(y) in sev:
                ax.axvspan(i - half_width, i + half_width, color=color, alpha=alpha, zorder=0)
    
    x = list(range(len(years)))
    x_ticks = [str(y) for y in years]
    
    # Helper to stack bars
    def _draw_stacked(ax, stats_df, acts, is_share=False, total_n=None):
        bottom = np.zeros(len(stats_df))
        
        # Calculate totals for share normalization
        if is_share:
            if total_n is None:
                # Sum of specified actions row-wise
                # For Owners, we forced sum to BASELINE_OWNER_N via DN, so row sum is constant.
                totals = stats_df[list(acts)].sum(axis=1).values
            else:
                totals = np.full(len(stats_df), total_n)
        
        for a in acts:
            if a not in stats_df.columns: continue
            vals = stats_df[a].values
            
            if is_share:
                # Avoid div by zero
                heights = np.divide(vals, totals, out=np.zeros_like(totals, dtype=float), where=totals!=0)
            else:
                heights = vals
                
            label = a
            color = C.get(a, "#9ca3af")
            
            ax.bar(x, heights, bottom=bottom, width=0.75, color=color, 
                   edgecolor="white", linewidth=0.5, alpha=0.9, label=label)
            bottom += heights

        if is_share:
            ax.set_ylabel("Cumulative Behavior Composition (%)", fontsize=14)
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
            ax.set_ylim(0, 1.0)
        else:
            ax.set_ylabel("Number of Households", fontsize=14)

        ax.set_xticks([i for i in x if years[i] % 2 == 0])
        ax.set_xticklabels([str(years[i]) for i in x if years[i] % 2 == 0], fontsize=12)
        ax.grid(True, axis="y", linestyle="--", alpha=0.2)

    # (a) Homeowner Share
    # Composition of staying population + this year's relocatees (Flow)
    _draw_stacked(ax11, df_own, owner_actions, is_share=True)
    ax11.set_title("Homeowner", fontsize=16, pad=12, fontweight="bold")
    _panel_label(ax11, "(a)")
    _shade_year_bars(ax11, years, severe_years)
    
    # (b) Renter Share
    # Renter denominator varies by year? Or fix to baseline?
    # Usually Renters churn. Normalizing by CURRENT year total N is typical for share.
    # Logic: df_rent has 'N'.
    # We use that as denominator row-by-row.
    denom_r = df_rent[list(renter_actions)].sum(axis=1).values # Or use stored 'N'
    _draw_stacked(ax12, df_rent, renter_actions, is_share=True) # Will calculate row sum automatically
    ax12.set_title("Renter", fontsize=16, pad=12, fontweight="bold")
    _panel_label(ax12, "(b)")
    _shade_year_bars(ax12, years, severe_years)

    if show_counts:
        # (c) Homeowner Counts
        _draw_stacked(ax21, df_own, owner_actions, is_share=False)
        ax21.set_title("Homeowner Householders (Counts)", fontsize=15, pad=10)
        _panel_label(ax21, "(c)")
        _shade_year_bars(ax21, years, severe_years, alpha=0.25)
        
        # (d) Renter Counts
        _draw_stacked(ax22, df_rent, renter_actions, is_share=False)
        ax22.set_title("Renter Householders (Counts)", fontsize=15, pad=10)
        _panel_label(ax22, "(d)")
        _shade_year_bars(ax22, years, severe_years, alpha=0.25)
        ax21.set_xlabel("Simulation Year", fontsize=15, fontweight="bold", labelpad=12)
        ax22.set_xlabel("Simulation Year", fontsize=15, fontweight="bold", labelpad=12)
    else:
        ax12.set_xlabel("Simulation Year", fontsize=15, fontweight="bold", labelpad=12)

    # Legend Inside (a)
    all_h, all_l = [], []
    # Collect unique handles
    seen = set()
    for ax in [ax11, ax12]:
        h, l = ax.get_legend_handles_labels()
        for hi, li in zip(h, l):
            if li not in seen:
                seen.add(li)
                all_h.append(hi)
                all_l.append(li)
    
    # Add severe year
    sev_handle = Patch(facecolor="#93c5fd", edgecolor="none", alpha=0.3, label="Severe flood year")
    all_h.append(sev_handle)
    all_l.append("Severe flood year")

    ax12.legend(handles=all_h, labels=all_l, loc='upper center', 
               frameon=True, facecolor='white', edgecolor='black', 
               framealpha=1.0, ncol=3, fontsize=14)

    out_dir = Path(fig_root) / "decisions"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"action_panels_share_and_counts{out_suffix}.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)




# ======================= Tract-based fallback plots =======================

def plot_cumulative_payout_from_tract(fin_dir: Path, fig_root: Path,
                                       severe_years=(2011, 2014, 2021)) -> None:
    """
    Plot cumulative payout using tract-level finance data.
    This is a fallback when household data is not available (summary/minimal mode).
    """
    tables = _load_finance_tract_years(fin_dir)
    if not tables:
        print("[info] No tract finance data available for cumulative payout plot")
        return
    
    rows = []
    for df in tables:
        if "__year" not in df.columns:
            continue
        y = int(df["__year"].iloc[0])
        
        # Get payout columns (try different naming conventions)
        payout_col = None
        for col in ["payout_total_usd", "payout_total_kUSD", "payout_usd"]:
            if col in df.columns:
                payout_col = col
                break
        
        if payout_col:
            payout_sum = pd.to_numeric(df[payout_col], errors="coerce").sum()
            if "_kUSD" in payout_col:
                payout_sum *= 1000  # Convert to USD
            rows.append({"year": y, "total_payout_usd": payout_sum})
    
    if not rows:
        print("[info] No payout data in tract files")
        return
    
    payout_df = pd.DataFrame(rows).sort_values("year")
    years = payout_df["year"].tolist()
    payout = payout_df["total_payout_usd"].to_numpy()
    cum_payout = np.cumsum(payout)
    
    _set_style()
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    x = np.arange(len(years))
    
    # Shade severe years
    sev = set(int(v) for v in severe_years)
    for i, y in enumerate(years):
        if int(y) in sev:
            ax.axvspan(i - 0.45, i + 0.45, color="#93c5fd", alpha=0.25, zorder=0)
    
    ax.bar(x, cum_payout / 1e6, width=0.7, color="#3b82f6", alpha=0.85, edgecolor="white")
    ax.set_ylabel("Cumulative Payout (Million USD)")
    ax.set_xlabel("Simulation Year", fontsize=15, fontweight="bold", labelpad=12)
    ax.set_title("Cumulative Insurance Payout Over Time")
    
    even_years = [y for y in years if y % 2 == 0]
    even_indices = [i for i, y in enumerate(years) if y % 2 == 0]
    ax.set_xticks(even_indices)
    ax.set_xticklabels([str(y) for y in even_years])
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    
    # Legend
    sev_patch = Patch(facecolor="#93c5fd", alpha=0.25, edgecolor="none", label="Severe flood year")
    ax.legend(handles=[sev_patch], loc="upper left", frameon=True, fontsize=LEGEND_FONTSIZE)
    
    out_dir = fig_root / "finance"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "cumulative_payout.png", bbox_inches="tight")
    print("[ok] Saved cumulative_payout.png (from tract data)")
    plt.close(fig)


def plot_flood_damage_by_group_from_tract(fin_dir: Path, fig_root: Path,
                                           severe_years=(2011, 2014, 2021)) -> None:
    """
    Plot annual flood damage by owner/renter group using tract data.
    Fallback when household data is not available.
    """
    tables = _load_finance_tract_years(fin_dir)
    if not tables:
        print("[info] No tract finance data for flood damage by group plot")
        return
    
    rows = []
    for df in tables:
        if "__year" not in df.columns:
            continue
        y = int(df["__year"].iloc[0])
        
        # Get owner/renter gross damage columns
        owner_dmg = 0.0
        renter_dmg = 0.0
        
        for col in ["owner_gross_total_kUSD", "owner_gross_total_usd"]:
            if col in df.columns:
                val = pd.to_numeric(df[col], errors="coerce").sum()
                owner_dmg = val * 1000 if "kUSD" in col else val
                break
        
        for col in ["renter_gross_total_kUSD", "renter_gross_total_usd"]:
            if col in df.columns:
                val = pd.to_numeric(df[col], errors="coerce").sum()
                renter_dmg = val * 1000 if "kUSD" in col else val
                break
        
        rows.append({"year": y, "owner_usd": owner_dmg, "renter_usd": renter_dmg})
    
    if not rows:
        return
    
    dmg_df = pd.DataFrame(rows).sort_values("year")
    years = dmg_df["year"].tolist()
    owner = dmg_df["owner_usd"].to_numpy()
    renter = dmg_df["renter_usd"].to_numpy()
    
    _set_style()
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    
    fig, ax = plt.subplots(figsize=(16, 8), constrained_layout=True)
    x = np.arange(len(years))
    w = 0.35
    
    # Shade severe years
    sev = set(int(v) for v in severe_years)
    for i, y in enumerate(years):
        if int(y) in sev:
            ax.axvspan(i - 0.5, i + 0.5, color="#93c5fd", alpha=0.25, zorder=0)
    
    ax.bar(x - w/2, owner / 1e6, width=w, color=COLOR_OWNER, alpha=0.85, 
           edgecolor="white", label="Owner")
    ax.bar(x + w/2, renter / 1e6, width=w, color=COLOR_RENTER, alpha=0.85,
           edgecolor="white", label="Renter")
    
    ax.set_ylabel("Flood Damage (Million USD)")
    ax.set_xlabel("Simulation Year", fontsize=15, fontweight="bold", labelpad=12)
    ax.set_title("Annual Flood Damage by Group")
    
    even_years = [y for y in years if y % 2 == 0]
    even_indices = [i for i, y in enumerate(years) if y % 2 == 0]
    ax.set_xticks(even_indices)
    ax.set_xticklabels([str(y) for y in even_years])
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    
    sev_patch = Patch(facecolor="#93c5fd", alpha=0.25, edgecolor="none", label="Severe flood year")
    handles, labels = ax.get_legend_handles_labels()
    handles.append(sev_patch)
    labels.append("Severe flood year")
    ax.legend(handles=handles, labels=labels, loc="upper left", frameon=True, fontsize=LEGEND_FONTSIZE)
    
    out_dir = fig_root / "flood"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "flood_damage_by_group.png", bbox_inches="tight")
    print("[ok] Saved flood_damage_by_group.png (from tract data)")
    plt.close(fig)


def plot_annual_flood_damage_from_tract(fin_dir: Path, fig_root: Path,
                                         severe_years=(2011, 2014, 2021)) -> None:
    """
    Plot total annual flood damage using tract data.
    """
    tables = _load_finance_tract_years(fin_dir)
    if not tables:
        print("[info] No tract finance data for annual flood damage plot")
        return
    
    rows = []
    for df in tables:
        if "__year" not in df.columns:
            continue
        y = int(df["__year"].iloc[0])
        
        # Get total gross damage
        total_dmg = 0.0
        for col in ["gross_total_kUSD", "gross_total_usd"]:
            if col in df.columns:
                val = pd.to_numeric(df[col], errors="coerce").sum()
                total_dmg = val * 1000 if "kUSD" in col else val
                break
        
        if total_dmg == 0:
            # Fall back to sum of owner + renter
            for col in ["owner_gross_total_kUSD", "renter_gross_total_kUSD"]:
                if col in df.columns:
                    total_dmg += pd.to_numeric(df[col], errors="coerce").sum() * 1000
            for col in ["owner_gross_total_usd", "renter_gross_total_usd"]:
                if col in df.columns:
                    total_dmg += pd.to_numeric(df[col], errors="coerce").sum()
        
        rows.append({"year": y, "total_damage_usd": total_dmg})
    
    if not rows:
        return
    
    dmg_df = pd.DataFrame(rows).sort_values("year")
    years = dmg_df["year"].tolist()
    damage = dmg_df["total_damage_usd"].to_numpy()
    
    _set_style()
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    
    fig, ax = plt.subplots(figsize=(16, 16), constrained_layout=True)
    x = np.arange(len(years))
    
    # Shade severe years
    sev = set(int(v) for v in severe_years)
    for i, y in enumerate(years):
        if int(y) in sev:
            ax.axvspan(i - 0.45, i + 0.45, color="#93c5fd", alpha=0.25, zorder=0)
    
    ax.bar(x, damage / 1e6, width=0.7, color="#ef4444", alpha=0.85, edgecolor="white")
    ax.set_ylabel("Total Flood Damage (Million USD)")
    ax.set_xlabel("Simulation Year", fontsize=15, fontweight="bold", labelpad=12)
    ax.set_title("Annual Total Flood Damage")
    
    even_years = [y for y in years if y % 2 == 0]
    even_indices = [i for i, y in enumerate(years) if y % 2 == 0]
    ax.set_xticks(even_indices)
    ax.set_xticklabels([str(y) for y in even_years])
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    
    sev_patch = Patch(facecolor="#93c5fd", alpha=0.25, edgecolor="none", label="Severe flood year")
    ax.legend(handles=[sev_patch], loc="upper left", frameon=True, fontsize=LEGEND_FONTSIZE)
    
    out_dir = fig_root / "flood"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "annual_flood_damage.png", bbox_inches="tight")
    print("[ok] Saved annual_flood_damage.png (from tract data)")
    plt.close(fig)


# ------------------------- Combined entry -------------------------

# ======================= Behavioral Analysis Plots =======================

def plot_fi_by_flood_prone(decisions_dir: Path, fig_root: Path, 
                            flood_years_file: Path = None,
                            severe_years=(2011, 2014, 2021)) -> None:
    """
    Plot FI adoption comparison: Flood-prone vs Non-flood-prone areas.
    Includes 4-panel analysis: Homeowner FP/Non-FP, Renter FP/Non-FP with before/after flood markers.
    Identifies flood-prone tracts as those with ratio_used >= 0.5 in any year.
    """
    import matplotlib.pyplot as plt
    _set_style()
    
    # Load decisions
    all_years_file = decisions_dir / "action_share_owner_renter_tract_all_years.csv"
    if not all_years_file.exists():
        print("[warn] No all_years decision file for flood-prone analysis")
        return
    
    dec = pd.read_csv(all_years_file)
    if 'year' not in dec.columns:
        # Load from individual files
        import glob, re
        files = list(decisions_dir.glob("action_share_owner_renter_tract_*.csv"))
        dfs = []
        for f in files:
            match = re.search(r'tract_(\d{4})\.csv', str(f))
            if match:
                df = pd.read_csv(f)
                df['year'] = int(match.group(1))
                dfs.append(df)
        dec = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    
    if dec.empty:
        return
    
    dec['tract_geoid'] = dec['tract_geoid'].astype(str)
    
    # Identify flood-prone tracts from config (owner init rate = 0.25)
    config_file = decisions_dir.parent.parent.parent / "config" / "abm_params.yaml"
    if not config_file.exists():
        config_file = Path("config/abm_params.yaml")
    
    high_flood_tracts = []
    if config_file.exists():
        import yaml
        with open(config_file, encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        take_rate = cfg.get('insurance_init', {}).get('take_rate_by_tract_group', {})
        high_flood_tracts = [str(t) for t, v in take_rate.items() if v.get('owner') == 0.25]
    
    if not high_flood_tracts:
        # Fallback to ratio_used method
        flood_file = flood_years_file or (decisions_dir.parent / "flood_years_by_tract.csv")
        if flood_file.exists():
            flood_df = pd.read_csv(flood_file)
            flood_df['tract_geoid'] = flood_df['tract_geoid'].astype(str)
            high_flood_tracts = flood_df[flood_df['ratio_used'] >= 0.5]['tract_geoid'].unique().tolist()
    
    dec['flood_prone'] = dec['tract_geoid'].isin(high_flood_tracts)
    years = sorted(dec['year'].unique())
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('FI Adoption by Flood-Prone Status: Homeowner vs Renter\n(Severe floods marked with dashed lines)', 
                 fontsize=14, fontweight='bold')
    
    plot_configs = [
        (0, 0, 'owner_share_FI', 'Homeowner', 'FI (%)'),
        (0, 1, 'renter_share_FI', 'Renter', 'FI (%)'),
    ]
    
    colors = {'Flood-Prone': '#ef4444', 'Non-Flood-Prone': '#22c55e'}
    
    for row, col, fi_col, title_grp, ylabel in plot_configs:
        ax = axes[row, col]
        for fp, label in [(True, 'Flood-Prone'), (False, 'Non-Flood-Prone')]:
            subset = dec[dec['flood_prone'] == fp]
            if subset.empty:
                continue
            fi = subset.groupby('year')[fi_col].mean() * 100
            ax.plot(fi.index, fi.values, 'o-', label=label, linewidth=2.5, 
                    color=colors[label], markersize=6)
        
        for sy in severe_years:
            ax.axvline(sy, color='gray', alpha=0.4, linewidth=2, linestyle='--')
        
        ax.set_xlabel("Simulation Year", fontsize=15, fontweight="bold", labelpad=12)
        ax.set_ylabel(ylabel)
        ax.set_title(f'{title_grp} FI Adoption')
        ax.legend(loc='best')
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
    
    # Bottom left: FI Delta after floods for Non-FP
    ax = axes[1, 0]
    delta_data = []
    for sy in severe_years:
        if sy+1 in years and sy+2 in years:
            for grp, fi_col in [('Homeowner', 'owner_share_FI'), ('Renter', 'renter_share_FI')]:
                nfp = dec[dec['flood_prone'] == False]
                before = nfp[nfp['year']==sy][fi_col].mean() * 100
                after1 = nfp[nfp['year']==sy+1][fi_col].mean() * 100
                after2 = nfp[nfp['year']==sy+2][fi_col].mean() * 100
                delta_data.append({'Period': f'{sy}→{sy+1}', 'Group': grp, 'Delta': after1-before})
                delta_data.append({'Period': f'{sy}→{sy+2}', 'Group': grp, 'Delta': after2-before})
    
    if delta_data:
        delta_df = pd.DataFrame(delta_data)
        x = np.arange(len(delta_df['Period'].unique()))
        width = 0.35
        periods = list(delta_df['Period'].unique())
        
        owner_deltas = [delta_df[(delta_df['Period']==p) & (delta_df['Group']=='Homeowner')]['Delta'].values[0] for p in periods]
        renter_deltas = [delta_df[(delta_df['Period']==p) & (delta_df['Group']=='Renter')]['Delta'].values[0] for p in periods]
        
        ax.bar(x - width/2, owner_deltas, width, label='Homeowner', color='#3b82f6')
        ax.bar(x + width/2, renter_deltas, width, label='Renter', color='#f59e0b')
        ax.axhline(0, color='black', linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(periods, rotation=45, ha='right')
        ax.set_ylabel('ΔFI (pp)')
        ax.set_title('Non-Flood-Prone: FI Change After Floods')
        ax.legend()
    
    # Bottom right: Cumulative trend comparison
    ax = axes[1, 1]
    nfp_owner = dec[dec['flood_prone']==False].groupby('year')['owner_share_FI'].mean() * 100
    nfp_renter = dec[dec['flood_prone']==False].groupby('year')['renter_share_FI'].mean() * 100
    fp_owner = dec[dec['flood_prone']==True].groupby('year')['owner_share_FI'].mean() * 100
    fp_renter = dec[dec['flood_prone']==True].groupby('year')['renter_share_FI'].mean() * 100
    
    ax.plot(nfp_owner.index, nfp_owner.values, 'o-', label='Non-FP Homeowner', color='#22c55e', linewidth=2)
    ax.plot(nfp_renter.index, nfp_renter.values, 's--', label='Non-FP Renter', color='#22c55e', linewidth=2, alpha=0.7)
    ax.plot(fp_owner.index, fp_owner.values, 'o-', label='FP Homeowner', color='#ef4444', linewidth=2)
    ax.plot(fp_renter.index, fp_renter.values, 's--', label='FP Renter', color='#ef4444', linewidth=2, alpha=0.7)
    
    for sy in severe_years:
        ax.axvline(sy, color='gray', alpha=0.4, linewidth=2, linestyle='--')
    
    ax.set_xlabel("Simulation Year", fontsize=15, fontweight="bold", labelpad=12)
    ax.set_ylabel('FI (%)')
    ax.set_title('All Groups FI Trend Comparison')
    ax.legend(loc='best', fontsize=9)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    out_dir = _ensure_fig_dirs(fig_root)["decisions"]
    out_path = out_dir / "fi_by_flood_prone.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path.name}")


def plot_fi_delta_after_floods(decisions_dir: Path, fig_root: Path,
                                severe_years=(2011, 2014, 2021)) -> None:
    """
    Plot FI change (delta) after severe flood years for homeowner/renter.
    Shows before→after comparison.
    """
    import matplotlib.pyplot as plt
    _set_style()
    
    # Load decisions
    all_years_file = decisions_dir / "action_share_owner_renter_tract_all_years.csv"
    if not all_years_file.exists():
        return
    
    dec = pd.read_csv(all_years_file)
    if 'year' not in dec.columns:
        import glob, re
        files = list(decisions_dir.glob("action_share_owner_renter_tract_*.csv"))
        dfs = []
        for f in files:
            match = re.search(r'tract_(\d{4})\.csv', str(f))
            if match:
                df = pd.read_csv(f)
                df['year'] = int(match.group(1))
                dfs.append(df)
        dec = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    
    if dec.empty:
        return
    
    years = sorted(dec['year'].unique())
    
    # Calculate deltas
    delta_data = []
    for sy in severe_years:
        if sy in years and sy+1 in years:
            for grp, label in [('owner_share_FI', 'Homeowner'), ('renter_share_FI', 'Renter')]:
                before = dec[dec['year']==sy][grp].mean() * 100
                after = dec[dec['year']==sy+1][grp].mean() * 100
                delta_data.append({
                    'Period': f'{sy}→{sy+1}',
                    'Group': label,
                    'Before': before,
                    'After': after,
                    'Delta': after - before
                })
    
    if not delta_data:
        return
    
    delta_df = pd.DataFrame(delta_data)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('FI Change Before/After Severe Floods', fontsize=14, fontweight='bold')
    
    # Before/After comparison
    ax = axes[0]
    x = np.arange(len(delta_df['Period'].unique()))
    width = 0.35
    periods = delta_df['Period'].unique()
    
    owner_before = [delta_df[(delta_df['Period']==p) & (delta_df['Group']=='Owner')]['Before'].values[0] for p in periods]
    owner_after = [delta_df[(delta_df['Period']==p) & (delta_df['Group']=='Owner')]['After'].values[0] for p in periods]
    
    ax.bar(x - width/2, owner_before, width, label='Before Flood', color='#a1a1aa')
    ax.bar(x + width/2, owner_after, width, label='After Flood', color='#3b82f6')
    ax.set_xticks(x)
    ax.set_xticklabels(periods)
    ax.set_ylabel('Homeowner FI Share (%)')
    ax.set_title('Homeowner FI: Before/After Floods')
    ax.legend()
    
    # Delta bars
    ax = axes[1]
    colors = ['#3b82f6' if d >= 0 else '#ef4444' for d in delta_df[delta_df['Group']=='Owner']['Delta'].values]
    ax.bar(x, delta_df[delta_df['Group']=='Owner']['Delta'].values, color=colors)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(periods)
    ax.set_ylabel('ΔFI Share (pp)')
    ax.set_title('Homeowner FI Change After Floods')
    
    plt.tight_layout()
    out_dir = _ensure_fig_dirs(fig_root)["decisions"]
    out_path = out_dir / "fi_delta_after_floods.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path.name}")


def plot_owner_vs_renter_fi(decisions_dir: Path, fig_root: Path,
                             severe_years=(2011, 2014, 2021)) -> None:
    """
    Plot homeowner vs renter FI adoption over time.
    """
    import matplotlib.pyplot as plt
    _set_style()
    
    all_years_file = decisions_dir / "action_share_owner_renter_tract_all_years.csv"
    if not all_years_file.exists():
        return
    
    dec = pd.read_csv(all_years_file)
    if 'year' not in dec.columns:
        import glob, re
        files = list(decisions_dir.glob("action_share_owner_renter_tract_*.csv"))
        dfs = []
        for f in files:
            match = re.search(r'tract_(\d{4})\.csv', str(f))
            if match:
                df = pd.read_csv(f)
                df['year'] = int(match.group(1))
                dfs.append(df)
        dec = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    
    if dec.empty:
        return
    
    years = sorted(dec['year'].unique())
    owner_fi = dec.groupby('year')['owner_share_FI'].mean() * 100
    renter_fi = dec.groupby('year')['renter_share_FI'].mean() * 100
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(owner_fi.index, owner_fi.values, 'o-', label='Homeowner', linewidth=2.5, 
            color='#3b82f6', markersize=8)
    ax.plot(renter_fi.index, renter_fi.values, 's-', label='Renter', linewidth=2.5,
            color='#f59e0b', markersize=8)
    
    for sy in severe_years:
        ax.axvline(sy, color='green', alpha=0.15, linewidth=12)
    
    ax.set_xlabel("Simulation Year", fontsize=15, fontweight="bold", labelpad=12)
    ax.set_ylabel('FI Share (%)', fontsize=12)
    ax.set_title('Flood Insurance Adoption: Homeowner vs Renter', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.set_ylim(0, 80)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    out_dir = _ensure_fig_dirs(fig_root)["decisions"]
    out_path = out_dir / "owner_vs_renter_fi.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path.name}")


def plot_all_outputs(tp_traj: pd.DataFrame,
                     fin_dir: Path,
                     fig_root: Path,
                     household_hist_for_tracts: Optional[Sequence[str]] = None,
                     poster: bool = False) -> None:
    """
    Centralized entry-point for visuals.
    - Action module (TP): saved under <fig_root>/tp
    - Finance module: household-level or tract-level plots under <fig_root>/finance
    - Flood damage: plots under <fig_root>/flood
    - Decisions/Actions: plots under <fig_root>/decisions

    Gracefully handles missing data by falling back to tract-level data when
    household-level data is not available (e.g., in summary/minimal output modes).
    """
    # ======== TP (Trust in Policymaker) plots ========
    plot_tp_outputs(tp_traj, fig_root)
    
    # ======== Check data availability ========
    has_household_data = bool(_load_finance_household_years(fin_dir))
    has_tract_data = bool(_load_finance_tract_years(fin_dir))
    
    if has_household_data:
        # Full household-level plots
        plot_household_premium_histograms(fin_dir, fig_root, tract_geoid=None)
        if household_hist_for_tracts:
            for geoid in household_hist_for_tracts:
                plot_household_premium_histograms(fin_dir, fig_root, tract_geoid=str(geoid))
        
        # Finance timeseries plots (user requested)
        try:
            plot_fin_cost_owner_renter(fin_dir, fig_root, drop_years=(), severe_years=(2011, 2014, 2021))
        except Exception as e:
            print(f"[warn] plot_fin_cost_owner_renter failed: {e}")
        
        try:
            plot_payout_owner_renter(fin_dir, fig_root, drop_years=(), severe_years=(2011, 2014, 2021),
                                    show_rate=True, rate_ylim=(0, 1.0), poster=poster)
        except Exception as e:
            print(f"[warn] plot_payout_owner_renter failed: {e}")
        
        try:
            plot_loss_attribution_stacked_percent(fin_dir, fig_root, severe_years=(2011, 2014, 2021))
        except Exception as e:
            print(f"[warn] plot_loss_attribution_stacked_percent failed: {e}")
    
    elif has_tract_data:
        # Fallback to tract-level plots
        print("[info] Using tract-level data for finance plots (summary/minimal mode)")
        plot_cumulative_payout_from_tract(fin_dir, fig_root, severe_years=(2011, 2014, 2021))
        plot_flood_damage_by_group_from_tract(fin_dir, fig_root, severe_years=(2011, 2014, 2021))
        plot_annual_flood_damage_from_tract(fin_dir, fig_root, severe_years=(2011, 2014, 2021))
    else:
        print("[warn] No finance data available for plotting")
    
    # ======== Decisions/Actions plots ========
    decisions_dir = fin_dir.parent / "decisions"
    if decisions_dir.exists() and any(decisions_dir.glob("*.csv")):
        try:
            plot_action_shares_owner_renter_separate_panels(decisions_dir, fig_root,
                                                            severe_marks=(11, 14, 21))
        except Exception as e:
            print(f"[warn] plot_action_shares failed: {e}")
        
        try:
            plot_action_counts_stacked_from_split(decisions_dir, fig_root, drop_years=())
        except Exception as e:
            print(f"[warn] plot_action_counts failed: {e}")
        
        try:
            plot_action_panels_share_and_counts(decisions_dir, fig_root)
        except Exception as e:
            print(f"[warn] plot_action_panels failed: {e}")
        
        # ======== NEW: Behavioral Analysis Plots ========
        try:
            plot_fi_by_flood_prone(decisions_dir, fig_root)
        except Exception as e:
            print(f"[warn] plot_fi_by_flood_prone failed: {e}")
        
        # Disabled per user request
        # try:
        #     plot_fi_delta_after_floods(decisions_dir, fig_root)
        # except Exception as e:
        #     print(f"[warn] plot_fi_delta_after_floods failed: {e}")
        
        # try:
        #     plot_owner_vs_renter_fi(decisions_dir, fig_root)
        # except Exception as e:
        #     print(f"[warn] plot_owner_vs_renter_fi failed: {e}")
        
        # ======== NEW: Flood Impact CSV for ArcGIS ========
        try:
            output_flood_impact_csv(decisions_dir, fig_root.parent)
        except Exception as e:
            print(f"[warn] output_flood_impact_csv failed: {e}")
    else:
        print("[info] No decisions data available for action plots")
    
    # ======== Vulnerability/Flood damage plots (from *_ALL.csv files) ========
    try:
        plot_flood_damage_by_year_box_from_all(fig_root)
    except Exception as e:
        print(f"[warn] plot_flood_damage_by_year_box failed: {e}")
    
    try:
        plot_flood_damage_per_household(fig_root, fin_dir, severe_years=(2011, 2014, 2021))
    except Exception as e:
        print(f"[warn] plot_flood_damage_per_household failed: {e}")
    
    # ======== Tract-level Analysis Plots (3 dimensions) ========
    try:
        # from utils.plots_tract_analysis import plot_tract_analysis
        output_dir = fin_dir.parent  # Go up from finance/ to output root
        # plot_tract_analysis(output_dir, fig_root)
    except Exception as e:
        print(f"[warn] plot_tract_analysis failed: {e}")
    
    print("[OK] plot_all_outputs completed")


def plot_baseline_vs_worst_comparison(baseline_dir: Path, worst_dir: Path, 
                                       output_dir: Path,
                                       periods=('early', 'late')) -> None:
    """
    Create comparison bar charts for poster:
    (a) Difference of payout (Baseline - Worst) by homeowner/renter
    (b) Ratio of avg flood damage (Worst / Baseline) - always >= 1.0
    
    Args:
        baseline_dir: Path to baseline output directory
        worst_dir: Path to worst-case output directory  
        output_dir: Path to save output plots
    """
    import matplotlib.pyplot as plt
    _set_style()
    
    # Define periods
    early_years = range(2011, 2017)  # 2011-2016
    late_years = range(2017, 2024)   # 2017-2023
    
    def load_finance_data(scenario_dir):
        """Load and aggregate finance data."""
        fin_dir = scenario_dir / "finance"
        if not fin_dir.exists():
            return None
        
        files = list(fin_dir.glob("finance_tract_*.csv"))
        if not files:
            return None
        
        dfs = []
        for f in files:
            if 'all_years' in f.name:
                continue
            try:
                df = pd.read_csv(f)
                # Extract year from filename
                import re
                match = re.search(r'(\d{4})', f.name)
                if match:
                    df['year'] = int(match.group(1))
                    dfs.append(df)
            except Exception:
                pass
        
        return pd.concat(dfs, ignore_index=True) if dfs else None
    
    # Load data
    baseline_fin = load_finance_data(baseline_dir)
    worst_fin = load_finance_data(worst_dir)
    
    if baseline_fin is None or worst_fin is None:
        print("[warn] Cannot create baseline vs worst comparison - missing data")
        return
    
    # Calculate metrics by period
    def calc_metrics(df, years):
        sub = df[df['year'].isin(years)]
        if sub.empty:
            return {}
        
        # Try different column names
        payout_cols = ['payout_total_usd', 'payout_usd', 'payout_owner', 'total_payout']
        damage_cols = ['gross_structure_loss_usd', 'flood_damage_usd', 'damage_usd', 'total_damage']
        
        payout = 0
        for col in payout_cols:
            if col in sub.columns:
                payout = sub[col].sum()
                break
        
        damage = 0
        for col in damage_cols:
            if col in sub.columns:
                damage = sub[col].sum()
                break
        
        # Owner/Renter specific if available
        owner_payout = sub['payout_owner'].sum() if 'payout_owner' in sub.columns else payout * 0.6
        renter_payout = sub['payout_renter'].sum() if 'payout_renter' in sub.columns else payout * 0.4
        owner_damage = sub['damage_owner'].sum() if 'damage_owner' in sub.columns else damage * 0.6
        renter_damage = sub['damage_renter'].sum() if 'damage_renter' in sub.columns else damage * 0.4
        
        return {
            'payout_owner': owner_payout,
            'payout_renter': renter_payout,
            'damage_owner': owner_damage,
            'damage_renter': renter_damage,
        }
    
    # Calculate for early and late periods
    b_early = calc_metrics(baseline_fin, early_years)
    b_late = calc_metrics(baseline_fin, late_years)
    w_early = calc_metrics(worst_fin, early_years)
    w_late = calc_metrics(worst_fin, late_years)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.2)) # Increased from 12, 5
    fig.suptitle('Baseline vs Worst Scenario Comparison', fontsize=20, fontweight='bold')
    
    x = np.arange(2)  # Owner, Renter
    width = 0.35
    colors = {'early': '#60a5fa', 'late': '#f97316'}
    
    # (a) Difference of payout (Baseline - Worst)
    ax = axes[0]
    
    # Early period difference
    early_diff_owner = (w_early.get('payout_owner', 0) - b_early.get('payout_owner', 0)) / 1e6
    early_diff_renter = (w_early.get('payout_renter', 0) - b_early.get('payout_renter', 0)) / 1e6
    
    # Late period difference  
    late_diff_owner = (w_late.get('payout_owner', 0) - b_late.get('payout_owner', 0)) / 1e6
    late_diff_renter = (w_late.get('payout_renter', 0) - b_late.get('payout_renter', 0)) / 1e6
    
    bars1 = ax.bar(x - width/2, [early_diff_owner, early_diff_renter], width, 
                   label='Early (2011-2016)', color=colors['early'])
    bars2 = ax.bar(x + width/2, [late_diff_owner, late_diff_renter], width,
                   label='Late (2017-2023)', color=colors['late'])
    
    ax.set_ylabel('Difference of Payout ($M)\n(Worst - Baseline)', fontsize=18)
    ax.set_xticks(x)
    ax.set_xticklabels(['Homeowner', 'Renter'], fontsize=16)
    ax.set_title('(a) Payout Difference', fontsize=18)
    ax.legend(fontsize=14)
    ax.axhline(0, color='black', linewidth=0.5)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       ha='center', va='bottom' if height >= 0 else 'top',
                       fontsize=12)
    
    # (b) Ratio of flood damage (Worst / Baseline) - clamp to >= 1.0
    ax = axes[1]
    
    def safe_ratio(worst_val, baseline_val):
        """Calculate ratio, ensuring result >= 1.0"""
        if baseline_val <= 0:
            return 1.0
        ratio = worst_val / baseline_val
        return max(ratio, 1.0)  # Clamp to minimum 1.0
    
    early_ratio_owner = safe_ratio(w_early.get('damage_owner', 0), b_early.get('damage_owner', 1))
    early_ratio_renter = safe_ratio(w_early.get('damage_renter', 0), b_early.get('damage_renter', 1))
    late_ratio_owner = safe_ratio(w_late.get('damage_owner', 0), b_late.get('damage_owner', 1))
    late_ratio_renter = safe_ratio(w_late.get('damage_renter', 0), b_late.get('damage_renter', 1))
    
    bars1 = ax.bar(x - width/2, [early_ratio_owner, early_ratio_renter], width,
                   label='Early (2011-2016)', color=colors['early'])
    bars2 = ax.bar(x + width/2, [late_ratio_owner, late_ratio_renter], width,
                   label='Late (2017-2023)', color=colors['late'])
    
    ax.set_ylabel('Ratio of Avg Flood Damage\n(Worst / Baseline)', fontsize=18)
    ax.set_xticks(x)
    ax.set_xticklabels(['Homeowner', 'Renter'], fontsize=16)
    ax.set_title('(b) Damage Ratio', fontsize=18)
    ax.legend(fontsize=14)
    ax.axhline(1.0, color='black', linewidth=0.5, linestyle='--')
    ax.set_ylim(0.8, max(early_ratio_owner, early_ratio_renter, late_ratio_owner, late_ratio_renter) * 1.2)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       ha='center', va='bottom',
                       fontsize=12)
    
    plt.tight_layout()
    
    # Save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "baseline_vs_worst_comparison.png"
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] Saved: baseline_vs_worst_comparison.png")


def output_flood_impact_csv(decisions_dir: Path, output_dir: Path,
                             flood_years=(2014, 2021)) -> None:
    """
    Output tract-level flood impact CSV for ArcGIS visualization.
    Combines 2014 and 2021 flood impacts using pre-flood year as baseline.
    
    Output columns:
        - tract_geoid
        - flood_prone (0/1)
        - delta_fi_owner_y1, delta_fi_owner_y2 (averaged across flood events)
        - delta_fi_renter_y1, delta_fi_renter_y2 (averaged across flood events)
        - Individual flood year deltas
    """
    import yaml
    
    # Load decisions
    all_years_file = decisions_dir / "action_share_owner_renter_tract_all_years.csv"
    if not all_years_file.exists():
        return
    
    dec = pd.read_csv(all_years_file)
    if 'year' not in dec.columns:
        import re
        files = list(decisions_dir.glob("action_share_owner_renter_tract_*.csv"))
        dfs = []
        for f in files:
            match = re.search(r'tract_(\d{4})\.csv', str(f))
            if match:
                df = pd.read_csv(f)
                df['year'] = int(match.group(1))
                dfs.append(df)
        dec = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    
    if dec.empty:
        return
    
    dec['tract_geoid'] = dec['tract_geoid'].astype(str)
    tracts = dec['tract_geoid'].unique()
    
    # Load flood-prone classification from config
    config_file = Path("config/abm_params.yaml")
    fp_tracts = []
    if config_file.exists():
        with open(config_file, encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        take_rate = cfg.get('insurance_init', {}).get('take_rate_by_tract_group', {})
        fp_tracts = [str(t) for t, v in take_rate.items() if v.get('owner') == 0.25]
    
    # Calculate deltas for each tract
    rows = []
    for tract in tracts:
        t_dec = dec[dec['tract_geoid'] == tract]
        row = {'tract_geoid': tract, 'flood_prone': 1 if tract in fp_tracts else 0}
        
        # Calculate for each flood year
        for fy in flood_years:
            base_year = fy - 1
            if base_year not in t_dec['year'].values or fy not in t_dec['year'].values:
                continue
            
            base = t_dec[t_dec['year'] == base_year]
            
            for col, prefix in [('owner_share_FI', 'owner'), ('renter_share_FI', 'renter')]:
                base_val = base[col].values[0] if len(base) > 0 else 0
                
                for offset, suffix in [(1, 'y1'), (2, 'y2')]:
                    target_year = fy + offset - 1  # Y+1 = flood year, Y+2 = flood year + 1
                    target = t_dec[t_dec['year'] == target_year]
                    if len(target) > 0:
                        delta = (target[col].values[0] - base_val) * 100  # in percentage points
                        row[f'delta_fi_{prefix}_{suffix}_{fy}'] = round(delta, 2)
        
        rows.append(row)
    
    result = pd.DataFrame(rows)
    
    # Calculate average across flood years
    for prefix in ['owner', 'renter']:
        for suffix in ['y1', 'y2']:
            cols = [c for c in result.columns if c.startswith(f'delta_fi_{prefix}_{suffix}_')]
            if cols:
                result[f'delta_fi_{prefix}_{suffix}_avg'] = result[cols].mean(axis=1).round(2)
    
    # Reorder columns
    base_cols = ['tract_geoid', 'flood_prone']
    avg_cols = [c for c in result.columns if '_avg' in c]
    detail_cols = [c for c in result.columns if c not in base_cols + avg_cols]
    result = result[base_cols + avg_cols + sorted(detail_cols)]
    
    # Save to CSV
    out_path = output_dir / "flood_impact_by_tract.csv"
    result.to_csv(out_path, index=False)
    print(f"[OK] Saved: flood_impact_by_tract.csv ({len(result)} tracts)")

