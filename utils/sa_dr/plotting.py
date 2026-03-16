# utils/sa_dr/plotting.py
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# --------- 配色與樣式 ---------
COLORS  = ["#2563eb", "#ef4444", "#10b981", "#f59e0b", "#8b5cf6"]
MARKERS = ["o", "s", "D", "^", "v"]

# ---- style helpers ----
def _set_style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 300, 
        "savefig.dpi": 300,
        "font.size": 20,
        "axes.titlesize": 22,       # Standardized
        "axes.titleweight": "bold", # Bold
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

def _panel_label(ax, label="(a)", x=-0.12, y=1.02):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=16, fontweight="bold",
            ha="left", va="bottom")

def _style_axis(ax: plt.Axes):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=5, width=1)

def _usd_fmt(x, _):
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
    把 'MG=0.2,NMG=0.3' → 'τ_th(MG)=0.2, τ_th(NMG)=0.3'
    若不是該格式，原樣返回（保相容）。
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

def _add_severe_year_bands(ax: plt.Axes, years: list[int] | tuple[int, ...], ymin=None, ymax=None):
    """
    在圖上畫出重災年直帶；若 ymin/ymax 為 None 則自動取當前 y 限。
    """
    if not years:
        return
    y0, y1 = ax.get_ylim() if (ymin is None or ymax is None) else (ymin, ymax)
    for y in years:
        ax.axvspan(y - 0.5, y + 0.5, color="0.9", alpha=0.6, linewidth=0)
    # 重畫一次 y 限，避免被 axvspan 影響
    ax.set_ylim(y0, y1)

from matplotlib.ticker import FuncFormatter

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
    """把多條 (years, values) 依 label 繪到同一個 axes。"""
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

# ---------------- Main entry（2 子圖） ----------------
def plot_fin_focus_dashboard(
    series_fin: Dict[str, Tuple[List[int], List[float]]],
    series_pay: Dict[str, Tuple[List[int], List[float]]],
    trade_points: Dict[str, Tuple[float, float]] | None = None,  # 保留參數避免舊程式壞掉（不再使用）
    trade_bubbles: Dict[str, float] | None = None,               # 保留參數避免舊程式壞掉（不再使用）
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
    產生 2 個子圖：
      (a) 累積 household financial cost（premium+OOP）
      (b) 累積 average payout
    內建：
      - 依你的 style 設定
      - 圖例自動把 'MG=.,NMG=.' 轉為 τ_th(...) 標示
      - 灰帶標出 severe years
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

# ========= NEW: baseline-difference dual line plot =========


from pathlib import Path
from typing import Dict, Tuple, List, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.patches as mpatches


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
    Make a 2-panel figure:

    Left  : Financial cost rate Δ vs baseline — cumulative over years
    Right : Payout rate Δ vs baseline — cumulative over years

    Definitions per year t (using total flood damage D_t as denominator):
      1) Convert cumulative-AVERAGE series to ANNUAL values correctly:
           total_i   = cum_avg_i * (i+1)
           annual_i  = total_i - total_{i-1}  (annual_0 = total_0)
      2) rate_t     = annual_value_t / D_t
      3) delta_t    = rate_t - baseline_rate_t
      4) cum_delta  = cumulative sum of delta_t over years (NaN treated as 0 increment)

    Other features:
      - All non-highlighted scenarios are plotted in grey.
      - Baseline is a black dashed line (always 0).
      - Severe flood years are shown as grey bands AND included in the legend.
      - If `as_percent=True`, y is shown in percentage points (×100).
    """

    def _cumavg_to_annual(xs: List[int], cumavg: List[float]) -> Tuple[List[int], List[float]]:
        xs = list(xs)
        ca = np.asarray(cumavg, dtype=float)
        n = np.arange(len(ca)) + 1          # 1,2,3,... number of years up to i
        totals = ca * n                      # back out cumulative totals
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
            inc = np.where(np.isfinite(arr), arr, 0.0)   # NaN treated as 0 increment
            cs = np.cumsum(inc).tolist()
            out[lab] = (ys, cs)
        return out

    fin_delta_cum = _delta_cumsum(_rate_map(series_fin))
    pay_delta_cum = _delta_cumsum(_rate_map(series_pay))

    # convert to percentage points if needed (×100)
    def _to_percent(mp: Dict[str, Tuple[List[int], List[float]]]) -> Dict[str, Tuple[List[int], List[float]]]:
        if not as_percent:
            return mp
        out = {}
        for k, (ys, vs) in mp.items():
            out[k] = (ys, [v * 100 if np.isfinite(v) else np.nan for v in vs])
        return out

    fin_delta_cum = _to_percent(fin_delta_cum)
    pay_delta_cum = _to_percent(pay_delta_cum)

    # ---- plotting ----
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.2), constrained_layout=True)
    panels = [
        (fin_delta_cum, "Financial cost rate Δ vs baseline — cumulative over years"),
        (pay_delta_cum, "Payout rate Δ vs baseline — cumulative over years"),
    ]

    for ax, (mp, title) in zip(axs, panels):
        # plot grey others first
        for lab, (yrs, vs) in mp.items():
            if (highlights and lab in highlights) or lab == baseline:
                continue
            ax.plot(yrs, vs, color="#BBBBBB", lw=1.1, alpha=0.7, zorder=1)

        # baseline (always ~0)
        if baseline in mp:
            yrs_b, vs_b = mp[baseline]
            ax.plot(yrs_b, vs_b, color="#111111", lw=1.6, ls="--", alpha=0.9, label="Baseline", zorder=2)

        # highlights
        if highlights:
            for lab, cfg in highlights.items():
                if lab not in mp:
                    continue
                color = cfg.get("color") or cfg.get("c") or "#000"
                label = cfg.get("label", lab.replace(",", ", "))
                yrs, vs = mp[lab]
                ax.plot(yrs, vs, color=color, lw=2.6, label=label, zorder=3)
                # end label
                if len(yrs) and len(vs) and np.isfinite(vs[-1]):
                    ax.annotate(label, (yrs[-1], vs[-1]), xytext=(6, 0),
                                textcoords="offset points", va="center", fontsize=10, color=color)

        # zero line
        ax.axhline(0, color="0.55", lw=1.0, zorder=0)

        # severe flood bands (also add to legend)
        band_handle = None
        for y in severe_years:
            patch = ax.axvspan(y - 0.5, y + 0.5, color="0.92", alpha=0.8, lw=0, zorder=0)
            band_handle = patch  # last patch is fine for legend handle
        handles, labels = ax.get_legend_handles_labels()
        if band_handle is not None:
            band = mpatches.Patch(facecolor="0.92", edgecolor="none", alpha=0.8, label="Severe flood years")
            handles = [band] + handles
            labels = ["Severe flood years"] + labels

        # labels & grid
        ax.set_title(title)
        ax.set_xlabel("Year")
        ylabel = "Cumulative difference vs baseline (%)" if as_percent \
                 else "Cumulative ( (value / D$_t$) − (baseline / D$_t$) )"
        ax.set_ylabel(ylabel)
        if as_percent:
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}%"))
        ax.grid(True, ls="--", alpha=0.2)
        ax.legend(handles, labels, loc=legend_loc, framealpha=0.95)

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=400)
    plt.close(fig)


# ========= NEW: cumulative average per-HH flood damage by tenure (owner / renter) =========
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.patches as mpatches

# 我們沿用 collect._pick 的「欄位名容錯」策略
from utils.sa_dr.collect import _pick  # :contentReference[oaicite:4]{index=4}

def _align_and_diff(
    base_xy: Tuple[List[int], List[float]],
    other_xy: Tuple[List[int], List[float]],
) -> Tuple[List[int], List[float]]:
    """與 baseline 對齊年份，回傳 (years, other - baseline)。"""
    by, bv = base_xy
    oy, ov = other_xy
    base_map = {int(y): float(v) if np.isfinite(v) else np.nan for y, v in zip(by, bv)}
    other_map = {int(y): float(v) if np.isfinite(v) else np.nan for y, v in zip(oy, ov)}
    years = [y for y in by if y in other_map]  # 以 baseline 年份為主
    diffs = []
    for y in years:
        b = base_map.get(y, np.nan)
        o = other_map.get(y, np.nan)
        diffs.append(o - b if np.isfinite(o) and np.isfinite(b) else np.nan)
    return years, diffs

def _read_homeowner_renter_damage_by_year(run_dir: Path) -> pd.DataFrame:
    """
    讀 vulnerability/flood_damage/flood_damage_tract_ALL_years.csv，
    產出每年的 owner_usd、renter_usd（跨 tract 加總）。
    """
    p = Path(run_dir) / "vulnerability" / "flood_damage" / "flood_damage_tract_ALL_years.csv"
    if not p.exists():
        raise FileNotFoundError(f"not found: {p}")
    df = pd.read_csv(p)
    # 欄位容錯：owner_usd / renter_usd / both_usd 在彙整檔中由 main_v2 統一輸出
    ycol = _pick(df, ["year","Year"])
    ownc = _pick(df, ["owner_usd","owner_damage_usd"], required=False)
    renc = _pick(df, ["renter_usd","renter_damage_usd"], required=False)
    if ownc is None and renc is None:
        # 若只給 both_usd，這裡就無法拆 owner/renter；直接報錯更安全
        raise KeyError("need owner_usd/renter_usd in flood_damage_tract_ALL_years.csv")
    # 避免 NaN
    if ownc is None: df["owner_usd"]  = 0.0; ownc = "owner_usd"
    if renc is None: df["renter_usd"] = 0.0; renc = "renter_usd"
    g = (df.groupby(ycol)[[ownc, renc]].sum()
           .rename(columns={ownc:"owner_total_usd", renc:"renter_total_usd"})
           .reset_index()
           .sort_values(ycol))
    g = g.rename(columns={ycol: "year"})
    g["year"] = pd.to_numeric(g["year"], errors="coerce").astype(int)
    return g[["year","owner_total_usd","renter_total_usd"]]

def _read_homeowner_renter_households_by_year(run_dir: Path) -> pd.DataFrame:
    """
    讀 finance/finance_tract_all_years.csv，產出每年 owner_households、renter_households（跨 tract 加總）。
    """
    p = Path(run_dir) / "finance" / "finance_tract_all_years.csv"
    if not p.exists():
        raise FileNotFoundError(f"not found: {p}")
    df = pd.read_csv(p)
    ycol = _pick(df, ["year","Year"])
    ohh  = _pick(df, ["owner_households","n_owner","owner_n_households","owner_n_hh"])
    rhh  = _pick(df, ["renter_households","n_renter","renter_n_households","renter_n_hh"])
    g = (df.groupby(ycol)[[ohh, rhh]].sum()
           .rename(columns={ohh:"owner_hh", rhh:"renter_hh"})
           .reset_index()
           .sort_values(ycol))
    g = g.rename(columns={ycol: "year"})
    g["year"] = pd.to_numeric(g["year"], errors="coerce").astype(int)
    return g[["year","owner_hh","renter_hh"]]

def series_cumavg_per_hh_damage(run_dir: Path) -> Tuple[List[int], List[float], List[float]]:
    """
    輸出：(years, owner_cumavg_per_hh, renter_cumavg_per_hh)
      per_hh(year) = total_damage_usd(year) / n_households(year)
      cumavg       = 逐年 per_hh 的累加（同 payout/financial 累積平均的作法）
    """
    dmg = _read_homeowner_renter_damage_by_year(run_dir)
    hh  = _read_homeowner_renter_households_by_year(run_dir)
    m = pd.merge(dmg, hh, on="year", how="inner").sort_values("year")
    # 年度每戶（分母=0 → NaN）
    m["owner_per_hh"]  = np.where(m["owner_hh"]  > 0, m["owner_total_usd"]  / m["owner_hh"],  np.nan)
    m["renter_per_hh"] = np.where(m["renter_hh"] > 0, m["renter_total_usd"] / m["renter_hh"], np.nan)
    # 累積平均（逐年加總；遇 NaN 不加）——與 ts_financial_and_payout 的 cumsum 精神一致 :contentReference[oaicite:7]{index=7}
    def _nan_cumsum(x: pd.Series) -> List[float]:
        tot = 0.0; out = []
        for v in pd.to_numeric(x, errors="coerce"):
            if np.isfinite(v): tot += float(v)
            out.append(tot)
        return out
    years = m["year"].astype(int).tolist()
    owner_cum  = _nan_cumsum(m["owner_per_hh"])
    renter_cum = _nan_cumsum(m["renter_per_hh"])
    return years, owner_cum, renter_cum

def plot_cumavg_damage_per_hh_owner_renter_delta(
    run_dirs_map: Dict[str, Path],
    *,
    baseline: str = "MG=0.5,NMG=0.5",
    highlights: Optional[Dict[str, Dict]] = None,
    severe_years: Tuple[int, ...] = (2011, 2014, 2021),
    out_png: Path | str = "outputs/experiments/fig/cumavg_damage_per_hh_owner_renter_delta.png",
    legend_loc: str = "upper left",
    annotate_endpoints: bool = False,
) -> None:
    """
    Difference vs baseline of cumulative per-HH flood damage (Homeowner/Renter).
    y = scenario(cumavg per-HH) − baseline(cumavg per-HH), year by year.
    Baseline shown as y=0 dashed line. Highlights in color, others grey.
    Legend is placed outside (right-center).
    """
    _set_style()

    # ---- 1) load cumulative-average per-HH series (owner/renter) ----
    series_owner: Dict[str, Tuple[List[int], List[float]]]  = {}
    series_renter: Dict[str, Tuple[List[int], List[float]]] = {}
    for raw_lab, rdir in run_dirs_map.items():
        lab = _rename_threshold_label(raw_lab)
        try:
            yrs, own, ren = series_cumavg_per_hh_damage(Path(rdir))
            series_owner[lab]  = (yrs, own)
            series_renter[lab] = (yrs, ren)
        except Exception as e:
            print(f"[skip] {lab}: {e}")

    base_key = _rename_threshold_label(baseline)
    if base_key not in series_owner or base_key not in series_renter:
        print(f"[warn] baseline '{baseline}' not found; skip plotting.")
        return

    # ---- 2) align to baseline and take differences ----
    def _align_and_diff(base_xy, other_xy):
        by, bv = base_xy; oy, ov = other_xy
        bm = {int(y): float(v) if np.isfinite(v) else np.nan for y, v in zip(by, bv)}
        om = {int(y): float(v) if np.isfinite(v) else np.nan for y, v in zip(oy, ov)}
        years = [y for y in by if y in om]
        diffs = [(om[y] - bm[y]) if (np.isfinite(om[y]) and np.isfinite(bm[y])) else np.nan for y in years]
        return years, diffs

    base_owner, base_renter = series_owner[base_key], series_renter[base_key]

    diff_owner: Dict[str, Tuple[List[int], List[float]]]  = {}
    diff_renter: Dict[str, Tuple[List[int], List[float]]] = {}
    for lab, xy in series_owner.items():
        if lab == base_key:
            yrs = base_owner[0]; diff_owner[lab] = (yrs, [0.0] * len(yrs))
        else:
            diff_owner[lab] = _align_and_diff(base_owner, xy)
    for lab, xy in series_renter.items():
        if lab == base_key:
            yrs = base_renter[0]; diff_renter[lab] = (yrs, [0.0] * len(yrs))
        else:
            diff_renter[lab] = _align_and_diff(base_renter, xy)

    # ---- 3) plot (legend inside on right panel) ----
    fig, axs = plt.subplots(1, 2, figsize=(13.5, 5.2), constrained_layout=True)
    # No suptitle - title info goes in individual panel titles

    panels = [
        (axs[0], diff_owner,  "Homeowner", "Cumulative Flood Damage per HH"),
        (axs[1], diff_renter, "Renter", "Cumulative Flood Damage per HH"),
    ]
    legend_ax = axs[0]  # put legend only on the left panel
    hl_set = { _rename_threshold_label(k) for k in (highlights or {}) }

    for ax, mp, bold_title, subtitle in panels:
        # grey others (exclude highlights & baseline)
        for lab, (xs, ys) in mp.items():
            if lab == base_key or lab in hl_set: 
                continue
            ax.plot(xs, ys, color="#BDBDBD", lw=1.1, alpha=0.85, zorder=1)

        # baseline = y=0 dashed line
        base_handle = ax.axhline(0.0, color="black", linestyle="--", linewidth=1.6, label="Baseline", zorder=2)

        # highlights
        if highlights:
            for key, cfg in highlights.items():
                lab = _rename_threshold_label(key)
                if lab not in mp:
                    continue
                xs, ys = mp[lab]
                color = cfg.get("color") or cfg.get("c") or "#2563eb"
                label = cfg.get("label", lab)
                ax.plot(xs, ys, color=color, lw=2.6, label=label, zorder=3)
                if annotate_endpoints and len(xs) and np.isfinite(ys[-1]):
                    ax.annotate(label, (xs[-1], ys[-1]), xytext=(6, 0),
                                textcoords="offset points", va="center", fontsize=10, color=color)

        # severe years (only 2011/2014/2021)
        band_handle = None
        for y in severe_years:
            band_handle = ax.axvspan(y-0.5, y+0.5, color="0.92", alpha=0.8, lw=0, zorder=0)

        # Bold title with subtitle
        ax.set_title(f"{bold_title}\n{subtitle}", fontweight='bold', fontsize=14)
        ax.set_xlabel("Year", fontsize=12)
        ax.set_ylabel(r"$\Delta$ Cumulative Damage per HH (USD)", fontsize=12)
        ax.yaxis.set_major_formatter(FuncFormatter(_usd_fmt))
        ax.grid(True, ls="--", alpha=0.2)

        # legend inside right panel
        if ax is legend_ax:
            handles, labels = ax.get_legend_handles_labels()
            if band_handle is not None:
                handles = [mpatches.Patch(facecolor="0.92", edgecolor="none", alpha=0.8, label="Severe flood years")] + handles
                labels  = ["Severe flood years"] + labels
            if base_handle and "Baseline" not in labels:
                handles = [base_handle] + handles
                labels  = ["Baseline"] + labels
            if handles:
                ax.legend(
                    handles, labels,
                    loc=legend_loc,
                    frameon=True, framealpha=0.95,
                    fontsize=10,
                )

    _panel_label(axs[0], "(a)")
    _panel_label(axs[1], "(b)")

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=400, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"[OK] wrote: {out_png}")

# ========= NEW: payout rate Δ vs baseline, split by tenure (Homeowners / Renters) =========


# 方便容錯找欄位（你專案已有）
from utils.sa_dr.collect import _pick

# 放在 plotting.py 頂部 imports 附近補上：
from utils.sa_dr.collect import _pick
import pandas as pd
import numpy as np
from pathlib import Path

def _series_cumavg_payout_by_tenure(run_dir: Path):
    """
    讀取 Homeowner / Renter 的「每保戶 payout」逐年值，然後做累積和（cumulative sum of annual per-policyholder payout）。

    讀取策略（與前面用的聚合邏輯一致）：
      1) 走 finance/finance_households_*.csv 分片（逐檔讀進、合併）
      2) 欄位容錯：
         - 年份: year / Year
         - 身份: is_owner / owner / tenure_is_owner / tenure (Owner/Renter)
         - 保戶旗標: is_policyholder / policyholder / has_policy / ph
                     → 若無，fallback: premium_* 欄位總和 > 0
                     → 再無，fallback: payout_* > 0
         - payout: payout_total_usd / payout_usd / claim_payout_usd
                   → 若沒有 total，則用 payout_structure_usd + payout_contents_usd
    回傳：((years, cum_owner_pp), (years, cum_renter_pp))
    """
    run_dir = Path(run_dir)
    hh_files = sorted((run_dir / "finance").glob("finance_households_*.csv"))
    if not hh_files:
        raise RuntimeError("No household shards found for payout-by-tenure.")

    agg = {}  # year -> {opay, rpay, oph, rph}
    for fp in hh_files:
        df = pd.read_csv(fp)

        # 年份
        ycol = _pick(df, ["year", "Year"])

        # payout 欄位
        payout_col = None
        for c in ["payout_total_usd", "payout_usd", "claim_payout_usd"]:
            if c in df.columns:
                payout_col = c; break
        if payout_col is None:
            ps = [c for c in df.columns if c.lower() in ("payout_structure_usd","payout_contents_usd")]
            if ps:
                df["__pout__"] = df[ps].sum(axis=1, min_count=1)
                payout_col = "__pout__"
            else:
                continue  # 這片沒有 payout 資訊，跳過

        # Owner / Renter 判定
        own_bool = None
        for c in ["is_owner", "owner", "tenure_is_owner"]:
            if c in df.columns:
                s = pd.to_numeric(df[c], errors="coerce")
                own_bool = s.fillna(0) > 0 if s.dtype != bool else df[c].astype(bool)
                break
        if own_bool is None:
            if "tenure" in df.columns:
                own_bool = df["tenure"].astype(str).str.lower().str.contains("owner")
            else:
                continue  # 無 tenure 資訊

        # Policyholder 判定（容錯順序）
        ph_bool = None
        for c in ["is_policyholder", "policyholder", "has_policy", "ph"]:
            if c in df.columns:
                ph_bool = df[c].astype(bool); break
        if ph_bool is None:
            prem_cols = [c for c in df.columns if "premium" in c.lower() and c.lower().endswith("_usd")]
            if prem_cols:
                ph_bool = df[prem_cols].sum(axis=1, min_count=1).fillna(0) > 0
            else:
                ph_bool = pd.to_numeric(df[payout_col], errors="coerce").fillna(0) > 0  # 最後兜底

        yr = pd.to_numeric(df[ycol], errors="coerce")
        pay = pd.to_numeric(df[payout_col], errors="coerce").fillna(0.0)
        o_mask = own_bool.fillna(False)
        r_mask = ~o_mask

        tmp = pd.DataFrame({
            "year": yr,
            "opay": np.where(o_mask, pay, 0.0),
            "rpay": np.where(r_mask, pay, 0.0),
            "oph":  np.where(o_mask & ph_bool, 1.0, 0.0),
            "rph":  np.where(r_mask & ph_bool, 1.0, 0.0),
        }).dropna(subset=["year"])

        g = tmp.groupby("year").sum(numeric_only=True)
        for y, row in g.iterrows():
            d = agg.setdefault(int(y), {"opay":0.0,"rpay":0.0,"oph":0.0,"rph":0.0})
            d["opay"] += float(row["opay"]); d["rpay"] += float(row["rpay"])
            d["oph"]  += float(row["oph"]);  d["rph"]  += float(row["rph"])

    years = sorted(agg.keys())
    owner_pp  = [(agg[y]["opay"]/agg[y]["oph"]) if agg[y]["oph"]>0 else np.nan for y in years]
    renter_pp = [(agg[y]["rpay"]/agg[y]["rph"]) if agg[y]["rph"]>0 else np.nan for y in years]

    def _cum(x):
        s = 0.0; out = []
        for v in x:
            if np.isfinite(v): s += float(v)
            out.append(s)
        return out

    return (years, _cum(owner_pp)), (years, _cum(renter_pp))


def _cum_to_annual(xs: List[int], ys_cum: List[float]) -> Tuple[List[int], List[float]]:
    """把 cum（逐年加總）還原成年值：annual[0]=cum[0]；其後為差分。"""
    annual = []
    prev = 0.0
    for v in ys_cum:
        if np.isfinite(v):
            annual.append(v - prev)
            prev = v
        else:
            annual.append(np.nan)
    return xs, annual

# utils/sa_dr/plotting.py
def plot_rate_delta_cum_tenure_series(
    *,
    series_owner: dict[str, tuple[list[int], list[float]]],
    series_renter: dict[str, tuple[list[int], list[float]]],
    damage_by_year: dict[int, float],
    baseline: str = "MG=0.5,NMG=0.5",
    highlights: dict[str, dict] | None = None,
    severe_years: tuple[int, ...] = (2011, 2014, 2021),
    out_png: Path | str = "outputs/experiments/fig/payout_rate_delta_cum_tenure.png",
    as_percent: bool = True,
    legend_loc: str = "upper left",
) -> None:
    """
    用現成的 series_owner / series_renter 畫「Payout rate Δ vs baseline（累積）」；左 Homeowner、右 Renter。
    去掉 fig.suptitle；字級與上一版一致；右側 legend 不被切。
    """
    import matplotlib.patches as mpatches
    _set_style()

    # ---- baseline 對齊（允許 "MG=..." 或 τ_th 標籤）----
    def _match(k: str) -> bool:
        return _rename_threshold_label(k) == _rename_threshold_label(baseline)

    base_key = next((k for k in series_owner if _match(k)), None)
    if base_key is None:
        print(f"[warn] baseline '{baseline}' not found in series_owner."); return
    if base_key not in series_renter:
        print(f"[warn] baseline '{baseline}' not found in series_renter."); return

    # ---- 累積→年度，再算「率」，最後做累積差 ----
    def _cum_to_annual(xs, ys_cum):
        prev, out = 0.0, []
        for v in ys_cum:
            out.append(v - prev if np.isfinite(v) else np.nan)
            prev = v if np.isfinite(v) else prev
        return xs, out

    def _rate_delta_cum(series_map):
        bx, by_cum = series_map[base_key]
        bx, b_annual = _cum_to_annual(bx, by_cum)
        b_rate = []
        for y, v in zip(bx, b_annual):
            d = damage_by_year.get(int(y), np.nan)
            b_rate.append(v / d if (np.isfinite(v) and np.isfinite(d) and d != 0) else np.nan)

        out = {}
        for lab, (xs, ys_cum) in series_map.items():
            xs, annual = _cum_to_annual(xs, ys_cum)
            s, cum = 0.0, []
            for y, v, br in zip(xs, annual, b_rate):
                d = damage_by_year.get(int(y), np.nan)
                r = (v / d) if (np.isfinite(v) and np.isfinite(d) and d != 0) else np.nan
                if np.isfinite(r) and np.isfinite(br):
                    s += (r - br)
                cum.append(s)
            out[lab] = (xs, cum)
        return out

    diff_O = _rate_delta_cum(series_owner)
    diff_R = _rate_delta_cum(series_renter)

    # ---- plot (no suptitle; legend inside right panel) ----
    fig, axs = plt.subplots(1, 2, figsize=(13.5, 5.2), constrained_layout=True)

    panels = [(axs[0], diff_O, "Homeowner", "Cumulative Payout Rate"), (axs[1], diff_R, "Renter", "Cumulative Payout Rate")]
    legend_ax = axs[0]
    hl_set = { _rename_threshold_label(k) for k in (highlights or {}) }
    # No suptitle - title info goes in individual panel titles

    for ax, mp, bold_title, subtitle in panels:
        # 其他灰線
        for lab, (xs, ys) in mp.items():
            if _rename_threshold_label(lab) == _rename_threshold_label(baseline) or _rename_threshold_label(lab) in hl_set:
                continue
            ax.plot(xs, ys, color="#C8C8C8", lw=1.1, alpha=0.75, zorder=1)

        # baseline
        base_handle = ax.axhline(0, color="black", ls="--", lw=1.6, label="Baseline", zorder=2)

        # highlights
        if highlights:
            for key, cfg in highlights.items():
                lab = next((k for k in mp if _rename_threshold_label(k) == _rename_threshold_label(key)), None)
                if not lab: 
                    continue
                xs, ys = mp[lab]
                color = cfg.get("color") or cfg.get("c") or "#2563eb"
                label = cfg.get("label", _rename_threshold_label(lab))
                ax.plot(xs, ys, color=color, lw=2.6, label=label, zorder=3)

        # severe years
        band_handle = None
        for y in severe_years:
            band_handle = ax.axvspan(y-0.5, y+0.5, color="0.92", alpha=0.8, lw=0, zorder=0)

        # Bold title with subtitle
        ax.set_title(f"{bold_title}\n{subtitle}", fontweight='bold', fontsize=14)
        ax.set_xlabel("Year", fontsize=12)
        ax.set_ylabel(r"$\Delta$ Cumulative Payout Rate (%)" if as_percent else "Difference", fontsize=12)
        if as_percent:
            # Multiplier was incorrect (100 is enough if values are fractions)
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v*100:.2f}%"))
        ax.grid(True, ls="--", alpha=0.2)

        # legend inside right panel
        if ax is legend_ax:
            handles, labels = ax.get_legend_handles_labels()
            if band_handle is not None:
                handles = [mpatches.Patch(facecolor="0.92", edgecolor="none", alpha=0.8, label="Severe flood years")] + handles
                labels  = ["Severe flood years"] + labels
            if base_handle and "Baseline" not in labels:
                handles = [base_handle] + handles
                labels  = ["Baseline"] + labels
            ax.legend(
                handles, labels,
                loc=legend_loc,
                frameon=True, framealpha=0.95,
                fontsize=10,
            )

    _panel_label(axs[0], "(a)"); _panel_label(axs[1], "(b)")

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    # tight + pad 避免 legend 被裁切
    fig.savefig(out_png, dpi=400, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"[OK] wrote: {out_png}")
def plot_sa_combined_dashboard_2x2(
    *,
    dmg_owner: dict[str, tuple[list[int], list[float]]],
    dmg_renter: dict[str, tuple[list[int], list[float]]],
    pay_owner: dict[str, tuple[list[int], list[float]]],
    pay_renter: dict[str, tuple[list[int], list[float]]],
    damage_by_year: dict[int, float],
    baseline: str = "MG=0.5,NMG=0.5",
    highlights: dict[str, dict] | None = None,
    severe_years: tuple[int, ...] = (2011, 2014, 2021),
    out_png: Path | str = "outputs/experiments/fig/sa_combined_dashboard_2x2.png",
    legend_loc: str = "upper left",
) -> None:
    """
    Combined 2x2 dashboard:
    Row 1 (a,b): Cumulative Flood Damage per HH (Homeowner, Renter)
    Row 2 (c,d): Cumulative Payout Rate (Homeowner, Renter)
    """
    import matplotlib.patches as mpatches
    _set_style()
    
    # 1. Processing Logic (Reuse logic from individual functions)
    def _match(k: str) -> bool:
        return _rename_threshold_label(k) == _rename_threshold_label(baseline)

    base_key = next((k for k in dmg_owner if _match(k)), None)
    if not base_key: 
        print(f"[warn] baseline '{baseline}' not found."); return

    # Damage Diffs
    def _get_diffs(mp):
        bx, bv = mp[base_key]
        out = {}
        for k, xy in mp.items():
            out[k] = _align_and_diff((bx, bv), xy)
        return out
    
    diff_dmg_O = _get_diffs(dmg_owner)
    diff_dmg_R = _get_diffs(dmg_renter)

    # Payout Diffs (Rate based)
    def _cum_to_annual_local(xs, ys_cum):
        prev, out = 0.0, []
        for v in ys_cum:
            out.append(v - prev if np.isfinite(v) else np.nan)
            prev = v if np.isfinite(v) else prev
        return xs, out

    def _rate_delta_cum_local(series_map):
        bx, by_cum = series_map[base_key]
        bx, b_annual = _cum_to_annual_local(bx, by_cum)
        b_rate = []
        for y, v in zip(bx, b_annual):
            d = damage_by_year.get(int(y), np.nan)
            b_rate.append(v / d if (np.isfinite(v) and np.isfinite(d) and d != 0) else np.nan)
        out = {}
        for lab, (xs, ys_cum) in series_map.items():
            xs, annual = _cum_to_annual_local(xs, ys_cum)
            s, cum = 0.0, []
            for y, v, br in zip(xs, annual, b_rate):
                d = damage_by_year.get(int(y), np.nan)
                r = (v / d) if (np.isfinite(v) and np.isfinite(d) and d != 0) else np.nan
                if np.isfinite(r) and np.isfinite(br): s += (r - br)
                cum.append(s)
            out[lab] = (xs, cum)
        return out

    diff_pay_O = _rate_delta_cum_local(pay_owner)
    diff_pay_R = _rate_delta_cum_local(pay_renter)

    # 2. Plot Construction
    fig, axs = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)
    hl_set = { _rename_threshold_label(k) for k in (highlights or {}) }
    
    # Map panel data
    # Row 1: Damage
    # Row 2: Payout
    panels = [
        (axs[0,0], diff_dmg_O, "Homeowner", r"$\Delta$ Cumulative Damage per HH (USD)", False),
        (axs[0,1], diff_dmg_R, "Renter", r"$\Delta$ Cumulative Damage per HH (USD)", False),
        (axs[1,0], diff_pay_O, "Homeowner", r"$\Delta$ Cumulative Payout Rate (%)", True),
        (axs[1,1], diff_pay_R, "Renter", r"$\Delta$ Cumulative Payout Rate (%)", True),
    ]
    labels = ["(a)", "(b)", "(c)", "(d)"]

    for i, (ax, mp, tenure, ylabel, is_rate) in enumerate(panels):
        # Grey lines
        for lab, (xs, ys) in mp.items():
            if _rename_threshold_label(lab) == _rename_threshold_label(baseline) or _rename_threshold_label(lab) in hl_set:
                continue
            ax.plot(xs, ys, color="#C8C8C8", lw=0.9, alpha=0.6, zorder=1)
        
        # Baseline
        base_handle = ax.axhline(0, color="black", ls="--", lw=1.5, label="Baseline", zorder=2)
        
        # Highlights
        if highlights:
            for key, cfg in highlights.items():
                lab = next((k for k in mp if _rename_threshold_label(k) == _rename_threshold_label(key)), None)
                if not lab: continue
                xs, ys = mp[lab]
                color = cfg.get("color") or cfg.get("c") or "#2563eb"
                ax.plot(xs, ys, color=color, lw=2.5, label=cfg.get("label", _rename_threshold_label(lab)), zorder=3)

        # Severe bands
        band_handle = None
        for y in severe_years:
            band_handle = ax.axvspan(y-0.5, y+0.5, color="0.94", alpha=0.8, lw=0, zorder=0)

        # Style
        ax.set_title(f"{tenure}\n{ylabel.split('(')[0].strip()}", fontweight='bold', fontsize=16)
        ax.set_xlabel("Year", fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        if is_rate:
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v*100:.2f}%"))
        else:
            ax.yaxis.set_major_formatter(FuncFormatter(_usd_fmt))
        
        _panel_label(ax, labels[i], x=-0.14)
        
        # Legend on the bottom-left subplot as requested
        if i == 2:
            handles, labs = ax.get_legend_handles_labels()
            if band_handle:
                handles = [mpatches.Patch(facecolor="0.94", edgecolor="none", alpha=0.8, label="Severe flood years")] + handles
                labs = ["Severe flood years"] + labs
            ax.legend(handles, labs, loc="lower left", frameon=True, framealpha=0.95, fontsize=12)

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=400, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"[OK] wrote: {out_png}")
