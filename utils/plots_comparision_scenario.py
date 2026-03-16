# utils/plots_comparision_scenario.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
from typing import Iterable, Tuple, Any, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

# ============================ Paper style & helpers ============================

def _set_style():
    plt.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.size": 12,            # Decreased from 18 (-2pt)
        "axes.titlesize": 16,       # Decreased from 20 (-2pt)
        "axes.titleweight": "bold", 
        "axes.labelsize": 12,       # Decreased from 18 (-2pt)
        "xtick.labelsize": 12,      # Decreased from 16 (-2pt)
        "ytick.labelsize": 12,      # Decreased from 16 (-2pt)
        "legend.fontsize": 12,      # Decreased from 16 (-2pt)
        "axes.linewidth": 1.0,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.grid": True,
        "grid.alpha": 0.2,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "CMU Serif"],
    })

def _panel_label(ax, label="(a)", x=-0.10, y=1.02):
    ax.text(
        x, y, label,
        transform=ax.transAxes,
        fontsize=16, fontweight="bold",
        ha="left", va="bottom",
        zorder=10, clip_on=False,
    )

def _usd_fmt(x, _):
    absx = abs(x)
    if absx >= 1_000_000_000:
        return f"${x/1_000_000_000:.1f}B"
    if absx >= 1_000_000:
        return f"${x/1_000_000:.1f}M"
    if absx >= 1_000:
        return f"${x/1_000:.1f}K"
    return f"${x:.0f}"
    if absx >= 1_000_000:
        return f"${x/1_000_000:.1f}M"
    if absx >= 1_000:
        return f"${x/1_000:.1f}K"
    return f"${x:.0f}"

SEV_SHADE = "#93c5fd"  # severe-year shade color
LEGEND_FONTSIZE = 9   # Compact legend
def _legend_box_right(fig, handles, labels=None, *, title=None,
                      anchor=(1.01, 0.5), ncol=1):
    lg = fig.legend(handles=handles, labels=labels,
                    loc="center left", bbox_to_anchor=anchor,
                    frameon=True, ncol=ncol, title=title, fontsize=LEGEND_FONTSIZE)
    fr = lg.get_frame()
    if fr is not None:
        fr.set_facecolor("white"); fr.set_edgecolor("black"); fr.set_linewidth(0.8)
    return lg

def _severe_patch(label: str = "Severe flood year") -> Patch:
    return Patch(facecolor=SEV_SHADE, alpha=0.20, edgecolor="none", label=label)

# Palette
COLOR_OWNER_LINE   = "#0ea5e9"  # Baseline 線
COLOR_WORST_LINE   = "#7c3aed"  # Worst 線
COLOR_DAMAGE_FILL  = "#a7f3d0"
COLOR_DAMAGE_LINE  = "#111827"
COLOR_OWNER  = "#cff720"
COLOR_RENTER = "#7ef7aa"

COLOR_MEDIAN = "#7c3aed"  # violet-600
COLOR_BAND   = "#a78bfa"  # violet-300
COLOR_MEAN   = "#0ea5e9"  # sky-500

def _ensure_out(vis_dir: Path, sub: str = "compare") -> Path:
    out = Path(vis_dir) / sub
    out.mkdir(parents=True, exist_ok=True)
    return out

# ============================ Readers & reducers ==============================

def _find_tract_col(df: pd.DataFrame) -> str | None:
    for c in ("tract_geoid","tract","CensusTract","GEOID","geoid","tract_geoid10"):
        if c in df.columns:
            return c
    return None

def _identity_series(df: pd.DataFrame) -> pd.Series:
    cand = None
    for c in ("identity","group","tenure","owner_renter"):
        if c in df.columns: cand = c; break
        lc = {x.lower(): x for x in df.columns}
        if c in lc: cand = lc[c]; break
    if cand is None:
        return pd.Series(["unknown"]*len(df))
    return (df[cand].astype(str).str.lower()
            .map({"owner":"owner","homeowner":"owner","renter":"renter"}).fillna("unknown"))

def _payout_total_row(df: pd.DataFrame) -> pd.Series:
    if "payout_total_usd" in df.columns:
        return pd.to_numeric(df["payout_total_usd"], errors="coerce").fillna(0.0)
    if {"payout_structure_usd","payout_contents_usd"} <= set(df.columns):
        return (pd.to_numeric(df["payout_structure_usd"], errors="coerce").fillna(0.0) +
                pd.to_numeric(df["payout_contents_usd"],  errors="coerce").fillna(0.0))
    if {"payout_structure_kUSD","payout_contents_kUSD"} <= set(df.columns):
        return (pd.to_numeric(df["payout_structure_kUSD"], errors="coerce").fillna(0.0) +
                pd.to_numeric(df["payout_contents_kUSD"],  errors="coerce").fillna(0.0)) * 1000.0
    for alt in ("payout_usd","Payout_usd"):
        if alt in df.columns:
            return pd.to_numeric(df[alt], errors="coerce").fillna(0.0)
    return pd.Series(0.0, index=df.index)

def _year_from_name(p: Path) -> int | None:
    try:
        s = p.stem
        digits = "".join(ch for ch in s if ch.isdigit())
        if len(digits) >= 4:
            return int(digits[-4:])
    except Exception:
        pass
    return None

# ---------- 場景輸出：EARLY vs LATE（含 owner/renter、baseline/worst、比值與 flood_prone） ----------

def _flood_prone_from_cfg_for(out_tracts: Iterable[str], cfg: Dict[str, Any] | None) -> pd.Series:
    """
    用「輸出表 out 的 tract 列表」來標記 flood-prone。
    規則：只要該 tract 的 owner 初始投保率 == 0.25 → 1，否則 0。
    支援三種 YAML 結構：
      1) insurance_init 是數字/字串數字
      2) insurance_init = {"owner": <數字>} 或 {"owner": {tract: <數字>, ...}}
      3) insurance_init = {"take_rate_by_tract_group": {tract: {owner: <數字>, ...}, ...}}
    """
    tracts = pd.Index([str(t) for t in out_tracts], name="tract_geoid")
    INS = (cfg or {}).get("insurance_init", {}) or {}

    def is_025(x) -> bool:
        try:
            return abs(float(x) - 0.25) < 1e-12
        except Exception:
            return False

    s = pd.Series(0, index=tracts, dtype=int)

    # 1) 直接數值
    if isinstance(INS, (int, float, str)):
        if is_025(INS):
            s[:] = 1
        return s

    if not isinstance(INS, dict):
        return s

    # 2) owner: 單值或 per-tract
    if "owner" in INS:
        owner_val = INS.get("owner")
        if isinstance(owner_val, (int, float, str)):
            if is_025(owner_val):
                s[:] = 1
            return s
        if isinstance(owner_val, dict):
            for k, v in owner_val.items():
                if is_025(v):
                    kk = str(k)
                    if kk in s.index:
                        s.loc[kk] = 1
            return s

    # 3) 你目前使用的形式
    if "take_rate_by_tract_group" in INS:
        tr_map = INS.get("take_rate_by_tract_group") or {}
        for k, v in tr_map.items():
            try:
                owner_take = (v or {}).get("owner", None)
            except Exception:
                owner_take = None
            if is_025(owner_take):
                kk = str(k)
                if kk in s.index:
                    s.loc[kk] = 1
        return s

    return s

def export_spatial_inputs_cumrate_perhh(
    out_root: Path,
    out_csv: Path,
    *,
    early: Tuple[int,int] = (2011, 2016),
    late:  Tuple[int,int] = (2017, 2023),
    cfg: Dict[str, Any] | None = None,
    late_is_cumulative: bool = False,  # False=early/late互斥；True=late為 early_start→late_end 的累積
) -> Path:
    """
    產出每個 tract × period（early/late）一列，欄位（皆分 owner / renter）：
      - total_payout_*_{baseline,worst}
      - total_damage_*_{baseline,worst}
      - avg_damage_perhh_*_{baseline,worst} = (Σdamage) / (Σhh)
      - payout_rate_*_{baseline,worst}      = (Σpayout) / (Σdamage) ; damage=0 → NaN
      - ratio（baseline / worst）：
          ratio_payout_rate_*_bl_over_wr
          ratio_avg_damage_perhh_*_bl_over_wr
      - flood_prone（由 YAML 的 insurance_init 判定 owner 初始=0.25）
    """

    out_root = Path(out_root)

    # ---- helpers：讀 (year, tract, identity) 的 finance/damage ----
    def _owner_renter_finance_by_tract_year(fin_dir: Path) -> pd.DataFrame:
        fin_dir = Path(fin_dir)
        rows = []
        for f in sorted(fin_dir.glob("finance_households_*.csv")):
            y = _year_from_name(f)
            if y is None:
                continue
            try:
                df = pd.read_csv(f)
            except Exception:
                continue
            tcol = _find_tract_col(df)
            if tcol is None or df.empty:
                continue
            tract = df[tcol].astype(str)
            ident = _identity_series(df)
            payout = _payout_total_row(df)
            g = (pd.DataFrame({"year": y, "tract_geoid": tract, "identity": ident, "payout_usd": payout})
                   .groupby(["year","tract_geoid","identity"], as_index=False)
                   .agg(payout_usd=("payout_usd","sum"), hh=("identity","size")))
            g = g[g["identity"].isin(["owner","renter"])]
            rows.append(g)
        if not rows:
            return pd.DataFrame(columns=["year","tract_geoid","identity","payout_usd","hh"])
        return pd.concat(rows, ignore_index=True)

    def _owner_renter_damage_by_tract_year(fd_all_path: Path) -> pd.DataFrame:
        p = Path(fd_all_path)
        if not p.exists():
            return pd.DataFrame(columns=["year","tract_geoid","identity","damage_usd"])
        df = pd.read_csv(p)
        df["year"] = pd.to_numeric(df.get("year"), errors="coerce")
        df = df.dropna(subset=["year"]).copy()
        df["year"] = df["year"].astype(int)

        tcol = _find_tract_col(df)
        if tcol is None:
            return pd.DataFrame(columns=["year","tract_geoid","identity","damage_usd"])
        df[tcol] = df[tcol].astype(str)

        lower = {c.lower(): c for c in df.columns}
        # long
        if ("identity" in lower) and ("damage_usd" in lower):
            ic, dc = lower["identity"], lower["damage_usd"]
            tmp = df[["year", tcol, ic, dc]].copy()
            tmp[ic] = tmp[ic].astype(str).str.lower().map({"owner":"owner","renter":"renter"})
            tmp[dc] = pd.to_numeric(tmp[dc], errors="coerce").fillna(0.0)
            tmp = tmp[tmp[ic].isin(["owner","renter"])]
            g = (tmp.groupby(["year", tcol, ic], as_index=False)[dc].sum()
                   .rename(columns={tcol:"tract_geoid", ic:"identity", dc:"damage_usd"}))
            return g
        # wide
        cand_o = [c for c in df.columns if c.lower() in ("owner_damage_usd","damage_owner_usd","owner_usd","damage_owner")]
        cand_r = [c for c in df.columns if c.lower() in ("renter_damage_usd","damage_renter_usd","renter_usd","damage_renter")]
        if cand_o and cand_r:
            ocol, rcol = cand_o[0], cand_r[0]
            for c in (ocol, rcol):
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
            go = (df.groupby(["year", tcol], as_index=False)[ocol].sum()
                    .rename(columns={tcol:"tract_geoid", ocol:"damage_usd"}))
            go["identity"] = "owner"
            gr = (df.groupby(["year", tcol], as_index=False)[rcol].sum()
                    .rename(columns={tcol:"tract_geoid", rcol:"damage_usd"}))
            gr["identity"] = "renter"
            return pd.concat([go, gr], ignore_index=True)
        return pd.DataFrame(columns=["year","tract_geoid","identity","damage_usd"])

    def _agg_period(df_fin: pd.DataFrame, df_dmg: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
        f = df_fin[(df_fin["year"] >= start) & (df_fin["year"] <= end)].copy()
        d = df_dmg[(df_dmg["year"] >= start) & (df_dmg["year"] <= end)].copy()
        f = f[f["identity"].isin(["owner","renter"])]
        d = d[d["identity"].isin(["owner","renter"])]

        f_sum = (f.groupby(["tract_geoid","identity"], as_index=False)
                   .agg(total_payout=("payout_usd","sum"), hh=("hh","sum")))
        d_sum = (d.groupby(["tract_geoid","identity"], as_index=False)
                   .agg(total_damage=("damage_usd","sum")))

        m = pd.merge(f_sum, d_sum, on=["tract_geoid","identity"], how="outer").fillna(0.0)
        m["avg_damage_perhh"] = np.where(m["hh"] > 0, m["total_damage"] / m["hh"], np.nan)
        m["payout_rate"] = np.where(m["total_damage"] > 0, m["total_payout"] / m["total_damage"], np.nan)

        piv = m.pivot(index="tract_geoid", columns="identity",
                      values=["total_payout","total_damage","avg_damage_perhh","payout_rate"])

        def _col(tup, default=np.nan):
            try:
                return piv[tup]
            except Exception:
                return pd.Series(default, index=piv.index)

        out = pd.DataFrame({
            "tract_geoid": piv.index.astype(str),
            # totals
            "total_payout_owner": _col(("total_payout","owner"), 0.0).to_numpy(),
            "total_payout_renter": _col(("total_payout","renter"), 0.0).to_numpy(),
            "total_damage_owner": _col(("total_damage","owner"), 0.0).to_numpy(),
            "total_damage_renter": _col(("total_damage","renter"), 0.0).to_numpy(),
            # per-HH damage
            "avg_damage_perhh_owner": _col(("avg_damage_perhh","owner"), np.nan).to_numpy(),
            "avg_damage_perhh_renter": _col(("avg_damage_perhh","renter"), np.nan).to_numpy(),
            # payout rate
            "payout_rate_owner": _col(("payout_rate","owner"), np.nan).to_numpy(),
            "payout_rate_renter": _col(("payout_rate","renter"), np.nan).to_numpy(),
        })
        return out

    # ---- 讀兩個 scenario 的年表 ----
    def _scenario_block(scen: str):
        scen_dir = out_root / scen
        fin = _owner_renter_finance_by_tract_year(scen_dir / "finance")
        dmg = _owner_renter_damage_by_tract_year(
            scen_dir / "vulnerability" / "flood_damage" / "flood_damage_tract_ALL_years.csv"
        )
        tracts = sorted(set(pd.concat([fin["tract_geoid"], dmg["tract_geoid"]], ignore_index=True).astype(str)))
        return fin, dmg, tracts

    finB, dmgB, trB = _scenario_block("baseline")
    finW, dmgW, trW = _scenario_block("worst")
    all_tracts = pd.Index(sorted(set(trB) | set(trW)), name="tract_geoid")

    # ---- period 定義 ----
    early_s, early_e = int(early[0]), int(early[1])
    late_s,  late_e  = int(late[0]),  int(late[1])
    late_start = early_s if late_is_cumulative else late_s

    # ---- 聚合：baseline / worst × early / late ----
    eB = _agg_period(finB, dmgB, early_s,    early_e).rename(columns={
        "total_payout_owner":"total_payout_owner_baseline",
        "total_payout_renter":"total_payout_renter_baseline",
        "total_damage_owner":"total_damage_owner_baseline",
        "total_damage_renter":"total_damage_renter_baseline",
        "avg_damage_perhh_owner":"avg_damage_perhh_owner_baseline",
        "avg_damage_perhh_renter":"avg_damage_perhh_renter_baseline",
        "payout_rate_owner":"payout_rate_owner_baseline",
        "payout_rate_renter":"payout_rate_renter_baseline",
    })
    eW = _agg_period(finW, dmgW, early_s,    early_e).rename(columns={
        "total_payout_owner":"total_payout_owner_worst",
        "total_payout_renter":"total_payout_renter_worst",
        "total_damage_owner":"total_damage_owner_worst",
        "total_damage_renter":"total_damage_renter_worst",
        "avg_damage_perhh_owner":"avg_damage_perhh_owner_worst",
        "avg_damage_perhh_renter":"avg_damage_perhh_renter_worst",
        "payout_rate_owner":"payout_rate_owner_worst",
        "payout_rate_renter":"payout_rate_renter_worst",
    })
    lB = _agg_period(finB, dmgB, late_start, late_e).rename(columns={
        "total_payout_owner":"total_payout_owner_baseline",
        "total_payout_renter":"total_payout_renter_baseline",
        "total_damage_owner":"total_damage_owner_baseline",
        "total_damage_renter":"total_damage_renter_baseline",
        "avg_damage_perhh_owner":"avg_damage_perhh_owner_baseline",
        "avg_damage_perhh_renter":"avg_damage_perhh_renter_baseline",
        "payout_rate_owner":"payout_rate_owner_baseline",
        "payout_rate_renter":"payout_rate_renter_baseline",
    })
    lW = _agg_period(finW, dmgW, late_start, late_e).rename(columns={
        "total_payout_owner":"total_payout_owner_worst",
        "total_payout_renter":"total_payout_renter_worst",
        "total_damage_owner":"total_damage_owner_worst",
        "total_damage_renter":"total_damage_renter_worst",
        "avg_damage_perhh_owner":"avg_damage_perhh_owner_worst",
        "avg_damage_perhh_renter":"avg_damage_perhh_renter_worst",
        "payout_rate_owner":"payout_rate_owner_worst",
        "payout_rate_renter":"payout_rate_renter_worst",
    })

    # ---- 合併 early / late（同時帶入 worst/baseline）----
    early_df = pd.merge(eB, eW, on="tract_geoid", how="outer")
    early_df.insert(1, "period", "early")
    late_label = "late_cum" if late_is_cumulative else "late"
    late_df  = pd.merge(lB, lW, on="tract_geoid", how="outer")
    late_df.insert(1, "period", late_label)

    out = pd.concat([early_df, late_df], ignore_index=True)

    # ---- 場景比值（baseline / worst）：payout rate 與 avg_damage_perhh ----
    def _safe_div(num, den):
        num = pd.to_numeric(num, errors="coerce")
        den = pd.to_numeric(den, errors="coerce")
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(den > 0, num / den, np.nan)
        return r

    out["ratio_payout_rate_owner_bl_over_wr"]  = _safe_div(
        out["payout_rate_owner_baseline"],  out["payout_rate_owner_worst"])
    out["ratio_payout_rate_renter_bl_over_wr"] = _safe_div(
        out["payout_rate_renter_baseline"], out["payout_rate_renter_worst"])

    out["ratio_avg_damage_perhh_owner_bl_over_wr"]  = _safe_div(
        out["avg_damage_perhh_owner_baseline"],  out["avg_damage_perhh_owner_worst"])
    out["ratio_avg_damage_perhh_renter_bl_over_wr"] = _safe_div(
        out["avg_damage_perhh_renter_baseline"], out["avg_damage_perhh_renter_worst"])

    # ---- flood_prone：用 out 表本身的 tract 清單來對應，避免鍵不對 ----
    out["tract_geoid"] = out["tract_geoid"].astype(str)
    flood_prone_map = _flood_prone_from_cfg_for(out["tract_geoid"].unique(), cfg)
    out["flood_prone"] = out["tract_geoid"].map(flood_prone_map).fillna(0).astype(int)

    # ---- 欄位順序 ----
    ordered_cols = [
        "tract_geoid","period",
        # payout totals
        "total_payout_owner_baseline","total_payout_owner_worst",
        "total_payout_renter_baseline","total_payout_renter_worst",
        # damage totals
        "total_damage_owner_baseline","total_damage_owner_worst",
        "total_damage_renter_baseline","total_damage_renter_worst",
        # avg per HH
        "avg_damage_perhh_owner_baseline","avg_damage_perhh_owner_worst",
        "avg_damage_perhh_renter_baseline","avg_damage_perhh_renter_worst",
        # payout rate
        "payout_rate_owner_baseline","payout_rate_owner_worst",
        "payout_rate_renter_baseline","payout_rate_renter_worst",
        # scenario ratios (baseline / worst)
        "ratio_payout_rate_owner_bl_over_wr","ratio_payout_rate_renter_bl_over_wr",
        "ratio_avg_damage_perhh_owner_bl_over_wr","ratio_avg_damage_perhh_renter_bl_over_wr",
        # flag
        "flood_prone",
    ]
    for c in ordered_cols:
        if c not in out.columns:
            out[c] = np.nan
    out = out[ordered_cols].sort_values(["tract_geoid","period"]).reset_index(drop=True)

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[ok] spatial inputs written → {out_csv}")
    return out_csv


# ========= 2×2 圖：累積 payout rate + 累積 per-HH flood damage（Owner/Renter） =========

def _owner_renter_payout_and_counts(fin_dir: Path) -> pd.DataFrame:
    fin_dir = Path(fin_dir)
    rows = []
    for f in sorted(fin_dir.glob("finance_households_*.csv")):
        y = _year_from_name(f)
        if y is None: continue
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        ident = _identity_series(df)
        pay   = _payout_total_row(df)
        g = (pd.DataFrame({"identity":ident, "payout":pay})
               .groupby("identity").agg(payout=("payout","sum"),
                                        n=("payout","size")))
        rows.append({
            "year": y,
            "owner_payout_usd": float(g["payout"].get("owner", 0.0)),
            "renter_payout_usd": float(g["payout"].get("renter", 0.0)),
            "owner_hh": int(g["n"].get("owner", 0)),
            "renter_hh": int(g["n"].get("renter", 0)),
        })
    if not rows:
        return pd.DataFrame(columns=["year","owner_payout_usd","renter_payout_usd","owner_hh","renter_hh"])
    return pd.DataFrame(rows).sort_values("year")

def _owner_renter_damage_year_totals(fin_allyears_path: Path) -> pd.DataFrame:
    """
    從 finance_tract_all_years.csv 讀 owner/renter 的年度『總毛損』(USD)。
    需要欄位：year, owner_gross_total_kUSD, renter_gross_total_kUSD
    """
    p = Path(fin_allyears_path)
    if not p.exists():
        return pd.DataFrame(columns=["year","owner_damage_usd","renter_damage_usd"])
    df = pd.read_csv(p)

    need = ["year", "owner_gross_total_kUSD", "renter_gross_total_kUSD"]
    for c in need:
        if c not in df.columns:
            return pd.DataFrame(columns=["year","owner_damage_usd","renter_damage_usd"])

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"]).copy()
    df["year"] = df["year"].astype(int)
    for c in ["owner_gross_total_kUSD","renter_gross_total_kUSD"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    g = (df.groupby("year", as_index=False)
            [["owner_gross_total_kUSD","renter_gross_total_kUSD"]].sum()
            .sort_values("year"))

    g["owner_damage_usd"]  = g["owner_gross_total_kUSD"]  * 1000.0
    g["renter_damage_usd"] = g["renter_gross_total_kUSD"] * 1000.0
    return g[["year","owner_damage_usd","renter_damage_usd"]]


def _owner_renter_damage_from_households(finance_dir: Path) -> pd.DataFrame:
    """
    Aggregates owner/renter damage directly from finance_households_*.csv files.
    This bypasses potential issues with pre-aggregated summary files.
    """
    data = []
    # Find all finance_households_YYYY.csv
    for p in sorted(finance_dir.glob("finance_households_*.csv")):
        try:
            # Extract year from filename
            stem = p.stem  # e.g. "finance_households_2011"
            parts = stem.split("_")
            if not parts[-1].isdigit():
                continue
            year = int(parts[-1])
            
            # Read only necessary columns
            # Using lambda to be robust against column presence
            df = pd.read_csv(p, usecols=lambda c: c in ["identity", "gross_total_usd"])
            
            if "gross_total_usd" not in df.columns or "identity" not in df.columns:
                 continue

            # Normalize identity
            df["norm_id"] = df["identity"].astype(str).str.lower().map({
                "owner": "owner", "homeowner": "owner", 
                "renter": "renter"
            }).fillna("unknown")
            
            # Filter
            sub = df[df["norm_id"].isin(["owner", "renter"])]
            grp = sub.groupby("norm_id")[["gross_total_usd"]].sum()
            
            row = {"year": year, "owner_damage_usd": 0.0, "renter_damage_usd": 0.0}
            if "owner" in grp.index:
                row["owner_damage_usd"] = grp.loc["owner", "gross_total_usd"]
            if "renter" in grp.index:
                row["renter_damage_usd"] = grp.loc["renter", "gross_total_usd"]
                
            data.append(row)
        except Exception as e:
            print(f"Warning processing {p.name}: {e}")
            
    if not data:
        return pd.DataFrame(columns=["year", "owner_damage_usd", "renter_damage_usd"])
        
    return pd.DataFrame(data).sort_values("year")


def plot_cum_payoutrate_and_damage_by_group(
    out_root: Path,
    vis_dir: Path,
    *,
    severe_years: list[int] | None = None,
    save_name: str = "cum_payoutrate_and_damage_by_group",
) -> None:
    if severe_years is None:
        severe_years = [2011, 2014, 2021]

    def _scenario(scen: str):
        scen_dir = Path(out_root) / scen
        fin = _owner_renter_payout_and_counts(scen_dir / "finance")
        dmg = _owner_renter_damage_from_households(scen_dir / "finance")
        years = sorted(set(pd.to_numeric(fin["year"], errors="coerce").dropna().astype(int))
                    | set(pd.to_numeric(dmg["year"], errors="coerce").dropna().astype(int)))
        # 對齊到共同 years；少的補 0（對年度累積是合理處理）
        f2 = (fin.astype({"year": int})
                .set_index("year")
                .reindex(years)
                .fillna(0.0)
                .reset_index())
        d2 = (dmg.astype({"year": int})
                .set_index("year")
                .reindex(years)
                .fillna(0.0)
                .reset_index())
        return years, f2, d2


    yB, fB, dB = _scenario("baseline")
    yW, fW, dW = _scenario("worst")
    years = sorted(set(yB) | set(yW))
    if not years:
        print("[warn] no years for cum payoutrate/damage figure"); return

    def _build(f: pd.DataFrame, d: pd.DataFrame, group: str, *, debug=False):
        # 取出年度向量（已在 _scenario 對齊）
        yrs = f["year"].to_numpy(dtype=int)

        if group == "owner":
            payout = f["owner_payout_usd"].to_numpy(dtype=float)
            hh     = f["owner_hh"].to_numpy(dtype=float)
            damage = d["owner_damage_usd"].to_numpy(dtype=float)
        else:
            payout = f["renter_payout_usd"].to_numpy(dtype=float)
            hh     = f["renter_hh"].to_numpy(dtype=float)
            damage = d["renter_damage_usd"].to_numpy(dtype=float)

        # 年度比率 r_t（僅供檢查/理解，不畫）
        with np.errstate(divide="ignore", invalid="ignore"):
            r_t = np.where(damage > 0, payout / damage, np.nan)

        # ---- 逐年累積（你要的定義）
        cum_pay = np.cumsum(payout)
        cum_dmg = np.cumsum(damage)
        with np.errstate(divide="ignore", invalid="ignore"):
            cum_rate = np.where(cum_dmg > 0, cum_pay / cum_dmg, np.nan)

        # per-HH 當年 → 累積（加總年度 per-HH）
        per_hh = np.where(hh > 0, damage / hh, np.nan)
        cum_per_hh = np.nancumsum(per_hh)

        if debug:
            dbg = pd.DataFrame({
                "year": yrs,
                "P_t": payout,
                "G_t": damage,
                "r_t = P_t/G_t": r_t,
                "ΣP_t": cum_pay,
                "ΣG_t": cum_dmg,
                "R_t = ΣP/ΣG": cum_rate,
                "perHH_t = G_t/HH_t": per_hh,
                "cum_perHH": cum_per_hh,
            })
            print("\n[debug] annual & cumulative table (", group, "):\n", dbg.to_string(index=False))

        return cum_rate, cum_per_hh


    rateB_O, cumDamB_O = _build(fB, dB, "owner",  debug=False)
    rateW_O, cumDamW_O = _build(fW, dW, "owner",  debug=False)
    rateB_R, cumDamB_R = _build(fB, dB, "renter", debug=False)
    rateW_R, cumDamW_R = _build(fW, dW, "renter", debug=False)


    # Use raw cumulative values (no visual scaling) so figure matches text numbers
    cumDamB_O_viz = cumDamB_O
    cumDamW_O_viz = cumDamW_O
    cumDamB_R_viz = cumDamB_R
    cumDamW_R_viz = cumDamW_R


    # ========= 視覺化專用縮放（結束）=========
    _set_style()
    # Journal-quality figure: standard single-column width ~7.5in, double-column ~15in
    # Using landscape 2x2 layout for clarity
    fig, axs = plt.subplots(2, 2, figsize=(10, 6),
                            constrained_layout=True, sharex=True)
    
    # Enhanced colors for print
    C1, C2 = "#0077b6", "#d62828"  # Blue for Baseline, Red for No-adaptation
    x = np.arange(len(years))
    
    def _shade(ax):
        sev = set(int(s) for s in severe_years)
        for i, y in enumerate(years):
            if y in sev:
                ax.axvspan(i - 0.5, i + 0.5, color=SEV_SHADE, alpha=0.15, zorder=0)

    # (a) Homeowner
    ax = axs[0, 0]; _shade(ax)
    ax.fill_between(x, cumDamB_O_viz, cumDamW_O_viz, alpha=0.20, color="#ef4444",
                    label="Difference")
    ax.plot(x, cumDamB_O_viz, marker="o", linewidth=2.2, color=C1, label="Baseline")
    ax.plot(x, cumDamW_O_viz, marker="s", linewidth=2.2, color=C2, label="No-adaptation")
    _panel_label(ax, "(a)")
    ax.set_title("Homeowner", pad=12)
    ax.set_ylabel("Cumulative flood damage\nper HH ($)")
    ax.yaxis.set_major_formatter(FuncFormatter(_usd_fmt))
    
    # (b) Homeowner — Cumulative Real Loss per HH = Σ(Damage - Payout) / HH
    ax = axs[0, 1]; _shade(ax)
    damage_O_B = dB["owner_damage_usd"].to_numpy()
    damage_O_W = dW["owner_damage_usd"].to_numpy()
    payout_O_B = fB["owner_payout_usd"].to_numpy()
    payout_O_W = fW["owner_payout_usd"].to_numpy()
    hh_O_B = np.maximum(fB["owner_hh"].to_numpy(), 1)
    hh_O_W = np.maximum(fW["owner_hh"].to_numpy(), 1)
    
    _actual_B_O = (damage_O_B - payout_O_B) / hh_O_B
    _actual_W_O = (damage_O_W - payout_O_W) / hh_O_W
    real_loss_B_O = np.cumsum(_actual_B_O)
    real_loss_W_O = np.cumsum(_actual_W_O)

    ax.fill_between(x, real_loss_B_O, real_loss_W_O, alpha=0.25, color="#22c55e")
    ax.plot(x, real_loss_B_O, marker="o", linewidth=2.2, color=C1)
    ax.plot(x, real_loss_W_O, marker="s", linewidth=2.2, color=C2)
    _panel_label(ax, "(b)")
    ax.set_title("Homeowner", pad=12)
    ax.set_ylabel("Cumulative actual loss\nper HH ($)")
    ax.yaxis.set_major_formatter(FuncFormatter(_usd_fmt))

    # (c) Renter
    ax = axs[1, 0]; _shade(ax)
    ax.fill_between(x, cumDamB_R_viz, cumDamW_R_viz, alpha=0.20, color="#ef4444")
    ax.plot(x, cumDamB_R_viz, marker="o", linewidth=2.2, color=C1)
    ax.plot(x, cumDamW_R_viz, marker="s", linewidth=2.2, color=C2)
    _panel_label(ax, "(c)")
    ax.set_title("Renter", pad=12)
    ax.set_ylabel("Cumulative flood damage\nper HH ($)")
    ax.yaxis.set_major_formatter(FuncFormatter(_usd_fmt))

    # (d) Renter — Cumulative Real Loss per HH = Σ(Damage - Payout) / HH
    ax = axs[1, 1]; _shade(ax)
    damage_R_B = dB["renter_damage_usd"].to_numpy()
    damage_R_W = dW["renter_damage_usd"].to_numpy()
    payout_R_B = fB["renter_payout_usd"].to_numpy()
    payout_R_W = fW["renter_payout_usd"].to_numpy()
    hh_R_B = np.maximum(fB["renter_hh"].to_numpy(), 1)
    hh_R_W = np.maximum(fW["renter_hh"].to_numpy(), 1)
    
    _actual_B_R = (damage_R_B - payout_R_B) / hh_R_B
    _actual_W_R = (damage_R_W - payout_R_W) / hh_R_W
    real_loss_B_R = np.cumsum(_actual_B_R)
    real_loss_W_R = np.cumsum(_actual_W_R)
    
    ax.fill_between(x, real_loss_B_R, real_loss_W_R, alpha=0.25, color="#22c55e")
    ax.plot(x, real_loss_B_R, marker="o", linewidth=2.2, color=C1)
    ax.plot(x, real_loss_W_R, marker="s", linewidth=2.2, color=C2)
    _panel_label(ax, "(d)")
    ax.set_title("Renter", pad=12)
    ax.set_ylabel("Cumulative actual loss\nper HH ($)")
    ax.yaxis.set_major_formatter(FuncFormatter(_usd_fmt))


    for ax in axs[-1, :]:
        ax.set_xlabel("Year")
    ticks_even   = [i for i, y in enumerate(years) if (y % 2) == 0]
    labels_even  = [str(y) for y in years if (y % 2) == 0]

    for ax in axs.ravel():
        ax.set_xticks(ticks_even)
        ax.set_xticklabels(labels_even, rotation=0)  # 偶數年很少，0°更清楚

    # Align y-axis limits between (a,b) for Owner row and (c,d) for Renter row
    # Owner row (row 0)
    ymax_owner = max(axs[0, 0].get_ylim()[1], axs[0, 1].get_ylim()[1])
    ymin_owner = min(axs[0, 0].get_ylim()[0], axs[0, 1].get_ylim()[0])
    axs[0, 0].set_ylim(ymin_owner, ymax_owner)
    axs[0, 1].set_ylim(ymin_owner, ymax_owner)
    
    # Renter row (row 1)
    ymax_renter = max(axs[1, 0].get_ylim()[1], axs[1, 1].get_ylim()[1])
    ymin_renter = min(axs[1, 0].get_ylim()[0], axs[1, 1].get_ylim()[0])
    axs[1, 0].set_ylim(ymin_renter, ymax_renter)
    axs[1, 1].set_ylim(ymin_renter, ymax_renter)

    handles = [
        plt.Line2D([], [], color=C1, marker="o", linewidth=2.2, label="Baseline"),
        plt.Line2D([], [], color=C2, marker="s", linewidth=2.2, label="No-adaptation"),
        Patch(facecolor=SEV_SHADE, alpha=0.20, edgecolor="none", label="Severe flood year"),
    ]
    # Place legend inside panel (b) upper-left
    axs[0, 1].legend(handles=handles, loc="upper left", frameon=True, fontsize=LEGEND_FONTSIZE)

    out_dir = _ensure_out(vis_dir, "compare")
    fig.savefig(out_dir / f"{save_name}.png", dpi=300, bbox_inches="tight")
    # PDF output disabled: fig.savefig(out_dir / f"{save_name}.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[ok] wrote:", out_dir / f"{save_name}.png")