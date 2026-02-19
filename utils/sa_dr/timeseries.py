# utils/sa_dr/timeseries.py
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd

# -------------------- internal helpers --------------------

def _pick(df: pd.DataFrame, names: List[str], required: bool = True) -> Optional[str]:
    """Pick the first existing column name from candidates."""
    for n in names:
        if n in df.columns:
            return n
    if required:
        raise KeyError(f"missing column; tried {names}")
    return None

def _concat_household_years(fin_dir: Path) -> Optional[pd.DataFrame]:
    """Concatenate finance_households_YYYY.csv under fin_dir (if any)."""
    fin_dir = Path(fin_dir)
    pats = sorted(fin_dir.glob("finance_households_*.csv"))
    if not pats:
        return None
    buf = []
    for p in pats:
        try:
            df = pd.read_csv(p)
            if "year" not in df.columns:
                # infer from filename if possible
                try:
                    y = int(p.stem.split("_")[-1])
                    df.insert(0, "year", y)
                except Exception:
                    pass
            buf.append(df)
        except Exception:
            continue
    return pd.concat(buf, ignore_index=True) if buf else None

# -------------------- public time series --------------------

def ts_financial_and_payout(run_dir: Path) -> pd.DataFrame:
    """
    Return per-year average (per household): financial (premium+OOP) and payout,
    plus cumulative sums.
    """
    run_dir = Path(run_dir)
    fdir = run_dir / "finance"

    hh = _concat_household_years(fdir)
    if hh is not None and not hh.empty:
        # premium + OOP
        prem = _pick(hh, ["premium_usd", "premium", "premium_total_usd"], required=False)
        oop  = _pick(hh, ["oop_usd", "out_of_pocket_usd", "total_oop_usd", "oop"], required=False)
        if oop is None:
            loss   = _pick(hh, ["loss_usd","total_loss_usd","gross_total_usd","total_damage_usd","damage_usd"], required=False)
            payout = _pick(hh, ["payout_usd","total_payout_usd","payout_total_usd","indemnity_usd","indemnity_total_usd"], required=False)
            if loss is not None and payout is not None:
                hh["_oop_fallback_"] = (pd.to_numeric(hh[loss], errors="coerce")
                                       - pd.to_numeric(hh[payout], errors="coerce")).clip(lower=0.0).fillna(0.0)
                oop = "_oop_fallback_"
            else:
                hh["_oop_zero_"] = 0.0; oop = "_oop_zero_"
        payout_col = _pick(hh, ["payout_usd","total_payout_usd","payout_total_usd","indemnity_usd","indemnity_total_usd"], required=False)
        if payout_col is None:
            hh["_payout_zero_"] = 0.0; payout_col = "_payout_zero_"

        grp = (hh.groupby("year")[[prem, oop, payout_col]].mean()
                 .rename(columns={prem:"avg_premium", oop:"avg_oop", payout_col:"avg_payout"})
                 .reset_index().sort_values("year"))

        grp["avg_financial_usd"] = grp["avg_premium"] + grp["avg_oop"]
        grp["cum_avg_financial_usd"] = grp["avg_financial_usd"].cumsum()
        grp["cum_avg_payout_usd"]    = grp["avg_payout"].cumsum()
        return grp[["year", "avg_financial_usd", "avg_payout", "cum_avg_financial_usd", "cum_avg_payout_usd"]]

    # fallback: tract-level
    tpath = fdir / "finance_tract_all_years.csv"
    if not tpath.exists():
        raise FileNotFoundError(f"no finance csvs in {fdir}")
    df = pd.read_csv(tpath)
    n_hh = _pick(df, ["n_households","households","n_hh"])
    prem = _pick(df, ["premium_usd","premium_total_usd","total_premium_usd"], required=False)
    oop  = _pick(df, ["oop_usd","out_of_pocket_usd","total_oop_usd","oop"], required=False)
    pay  = _pick(df, ["payout_usd","total_payout_usd","payout_total_usd","indemnity_usd"], required=False)
    if prem is None: df["_prem_zero_"] = 0.0; prem = "_prem_zero_"
    if oop  is None: df["_oop_zero_"]  = 0.0;  oop  = "_oop_zero_"
    if pay  is None: df["_pay_zero_"]  = 0.0;  pay  = "_pay_zero_"

    df["avg_premium"] = pd.to_numeric(df[prem], errors="coerce") / pd.to_numeric(df[n_hh], errors="coerce").replace({0:np.nan})
    df["avg_oop"]     = pd.to_numeric(df[oop],  errors="coerce") / pd.to_numeric(df[n_hh], errors="coerce").replace({0:np.nan})
    df["avg_payout"]  = pd.to_numeric(df[pay],  errors="coerce") / pd.to_numeric(df[n_hh], errors="coerce").replace({0:np.nan})
    g = (df.groupby("year")[["avg_premium","avg_oop","avg_payout"]].mean()
           .reset_index().sort_values("year"))
    g["avg_financial_usd"] = g["avg_premium"] + g["avg_oop"]
    g["cum_avg_financial_usd"] = g["avg_financial_usd"].cumsum()
    g["cum_avg_payout_usd"]    = g["avg_payout"].cumsum()
    return g[["year", "avg_financial_usd", "avg_payout", "cum_avg_financial_usd", "cum_avg_payout_usd"]]

def ts_flood_damage_total(run_dir: Path) -> pd.DataFrame:
    """
    Return per-year tract-level 'community' flood damage averages (owner+renter),
    with cumulative column.
    """
    run_dir = Path(run_dir)
    p = run_dir / "vulnerability" / "flood_damage" / "flood_damage_tract_ALL_years.csv"
    if not p.exists():
        # 若不存在就給空表（避免中斷）
        return pd.DataFrame({"year": [], "avg_damage_usd": [], "cum_avg_damage_usd": []})
    df = pd.read_csv(p)
    # 以 year 平均每 tract 的 damage（owner+renter 合計已由 pipeline 輸出）
    g = (df.groupby("year")["both_usd"]
           .mean()
           .rename("avg_damage_usd")
           .reset_index()
           .sort_values("year"))
    g["cum_avg_damage_usd"] = g["avg_damage_usd"].cumsum()
    return g

# ---------- cumulative series exposed ----------

def series_cum_financial(run_dir: Path) -> Tuple[List[int], List[float]]:
    df = ts_financial_and_payout(run_dir)
    return df["year"].tolist(), df["cum_avg_financial_usd"].tolist()

def series_cum_payout(run_dir: Path) -> Tuple[List[int], List[float]]:
    df = ts_financial_and_payout(run_dir)
    return df["year"].tolist(), df["cum_avg_payout_usd"].tolist()

def series_cum_flood_damage(run_dir: Path) -> Tuple[List[int], List[float]]:
    df = ts_flood_damage_total(run_dir)
    return df["year"].tolist(), df["cum_avg_damage_usd"].tolist()

# ---------- premium / oop (cumulative series & final cumulative) ----------

def series_cum_premium_oop(run_dir: Path) -> Tuple[List[int], List[float], List[float]]:
    """
    Return (years, cum_avg_premium_usd, cum_avg_oop_usd). Prefer household files; fallback to tract-level.
    """
    run_dir = Path(run_dir)
    fdir = run_dir / "finance"
    hh = _concat_household_years(fdir)

    if hh is not None and not hh.empty:
        prem = _pick(hh, ["premium_usd","premium","premium_total_usd"], required=False)
        oop  = _pick(hh, ["oop_usd","out_of_pocket_usd","total_oop_usd","oop"], required=False)
        if oop is None:
            loss   = _pick(hh, ["loss_usd","total_loss_usd","gross_total_usd","total_damage_usd","damage_usd"], required=False)
            payout = _pick(hh, ["payout_usd","total_payout_usd","payout_total_usd","indemnity_usd","indemnity_total_usd"], required=False)
            if loss is not None and payout is not None:
                hh["_oop_fallback_"] = (pd.to_numeric(hh[loss], errors="coerce")
                                       - pd.to_numeric(hh[payout], errors="coerce")).clip(lower=0.0).fillna(0.0)
                oop = "_oop_fallback_"
            else:
                hh["_oop_zero_"] = 0.0; oop = "_oop_zero_"

        grp = (hh.groupby("year")[[prem, oop]].mean()
                 .rename(columns={prem:"avg_premium", oop:"avg_oop"})
                 .reset_index().sort_values("year"))
        grp["cum_premium"] = grp["avg_premium"].cumsum()
        grp["cum_oop"]     = grp["avg_oop"].cumsum()
        return grp["year"].tolist(), grp["cum_premium"].tolist(), grp["cum_oop"].tolist()

    # fallback: tract-level
    t = fdir / "finance_tract_all_years.csv"
    if not t.exists():
        raise FileNotFoundError(f"no finance csvs in {fdir}")
    df = pd.read_csv(t)
    prem = _pick(df, ["premium_usd","premium_total_usd","total_premium_usd"], required=False)
    oop  = _pick(df, ["oop_usd","out_of_pocket_usd","total_oop_usd","oop"], required=False)
    n_hh = _pick(df, ["n_households","households","n_hh"])
    if prem is None: df["_prem_zero_"] = 0.0; prem = "_prem_zero_"
    if oop  is None:
        loss   = _pick(df, ["loss_usd","total_loss_usd","gross_total_usd","total_damage_usd","damage_usd"], required=False)
        payout = _pick(df, ["payout_usd","total_payout_usd","payout_total_usd","indemnity_usd"], required=False)
        if loss is not None and payout is not None:
            df["_oop_est_"] = (pd.to_numeric(df[loss], errors="coerce")
                              - pd.to_numeric(df[payout], errors="coerce")).clip(lower=0.0)
            oop = "_oop_est_"
        else:
            df["_oop_zero_"] = 0.0; oop = "_oop_zero_"

    df["avg_premium"] = pd.to_numeric(df[prem], errors="coerce") / pd.to_numeric(df[n_hh], errors="coerce").replace({0:np.nan})
    df["avg_oop"]     = pd.to_numeric(df[oop],  errors="coerce") / pd.to_numeric(df[n_hh], errors="coerce").replace({0:np.nan})
    g = (df.groupby("year")[["avg_premium","avg_oop"]].mean()
           .reset_index().sort_values("year"))
    g["cum_premium"] = g["avg_premium"].cumsum()
    g["cum_oop"]     = g["avg_oop"].cumsum()
    return g["year"].tolist(), g["cum_premium"].tolist(), g["cum_oop"].tolist()

def final_cum_premium_oop(run_dir: Path) -> Tuple[float, float]:
    """Return (final_cum_premium_usd, final_cum_oop_usd)."""
    _, cp, co = series_cum_premium_oop(run_dir)
    return (float(cp[-1]) if cp else 0.0, float(co[-1]) if co else 0.0)

# ---------- annual mean helpers (for bottom row) ----------

def mean_over_years_flood_damage(run_dir: Path) -> float:
    """Annual mean of community flood damage (owner+renter combined)."""
    df = ts_flood_damage_total(run_dir)
    return float(pd.to_numeric(df["avg_damage_usd"], errors="coerce").mean()) if not df.empty else 0.0

def mean_over_years_financial_and_payout(run_dir: Path) -> Tuple[float, float]:
    """(financial_mean, payout_mean) — both per-household annual means."""
    df = ts_financial_and_payout(run_dir)
    if df.empty:
        return 0.0, 0.0
    fin_mean = float(pd.to_numeric(df["avg_financial_usd"], errors="coerce").mean())
    pay_mean = float(pd.to_numeric(df["avg_payout"],        errors="coerce").mean())
    return fin_mean, pay_mean

def ts_premium_oop_avg_per_year(run_dir: Path) -> pd.DataFrame:
    """Per-year (avg_premium, avg_oop) — per household; source: household finance or tract fallback."""
    run_dir = Path(run_dir)
    fdir = run_dir / "finance"
    hh = _concat_household_years(fdir)
    if hh is not None and not hh.empty:
        prem = _pick(hh, ["premium_usd","premium","premium_total_usd"], required=False)
        oop  = _pick(hh, ["oop_usd","out_of_pocket_usd","total_oop_usd","oop"], required=False)
        if oop is None:
            loss   = _pick(hh, ["loss_usd","total_loss_usd","gross_total_usd","total_damage_usd","damage_usd"], required=False)
            payout = _pick(hh, ["payout_usd","total_payout_usd","payout_total_usd","indemnity_usd","indemnity_total_usd"], required=False)
            if loss is not None and payout is not None:
                hh["_oop_fallback_"] = (pd.to_numeric(hh[loss], errors="coerce")
                                       - pd.to_numeric(hh[payout], errors="coerce")).clip(lower=0.0).fillna(0.0)
                oop = "_oop_fallback_"
            else:
                hh["_oop_zero_"] = 0.0; oop = "_oop_zero_"
        grp = (hh.groupby("year")[[prem, oop]].mean()
                 .rename(columns={prem:"avg_premium", oop:"avg_oop"})
                 .reset_index().sort_values("year"))
        return grp[["year", "avg_premium", "avg_oop"]]

    # fallback: tract-level
    t = fdir / "finance_tract_all_years.csv"
    df = pd.read_csv(t)
    prem = _pick(df, ["premium_usd","premium_total_usd","total_premium_usd"], required=False)
    oop  = _pick(df, ["oop_usd","out_of_pocket_usd","total_oop_usd","oop"], required=False)
    n_hh = _pick(df, ["n_households","households","n_hh"])
    if prem is None: df["_prem_zero_"] = 0.0; prem = "_prem_zero_"
    if oop  is None: df["_oop_zero_"]  = 0.0;  oop  = "_oop_zero_"
    df["avg_premium"] = pd.to_numeric(df[prem], errors="coerce") / pd.to_numeric(df[n_hh], errors="coerce").replace({0:np.nan})
    df["avg_oop"]     = pd.to_numeric(df[oop],  errors="coerce") / pd.to_numeric(df[n_hh], errors="coerce").replace({0:np.nan})
    g = (df.groupby("year")[["avg_premium","avg_oop"]].mean()
           .reset_index().sort_values("year"))
    return g

def final_avg_premium_oop(run_dir: Path) -> Tuple[float, float]:
    """(avg_premium_mean, avg_oop_mean) — per-household annual means."""
    df = ts_premium_oop_avg_per_year(run_dir)
    if df.empty:
        return 0.0, 0.0
    return float(df["avg_premium"].mean()), float(df["avg_oop"].mean())

# ===== NEW: totals for trade-off (damage/payout/financial) =====

def totals_tradeoff_damage_payout_financial(run_dir: Path) -> tuple[float, float, float]:
    """
    回傳 (total_damage_usd, total_payout_usd, total_financial_usd)，皆為「累積總額」。
    來源：
      - damage：vulnerability/flood_damage/flood_damage_tract_ALL_years.csv
      - payout / financial：finance/finance_tract_all_years.csv（payout、premium、oop）
    """
    run_dir = Path(run_dir)

    # --- damage total（ALL tracts, ALL years）
    fd = run_dir / "vulnerability" / "flood_damage" / "flood_damage_tract_ALL_years.csv"
    total_damage = 0.0
    if fd.exists():
        d = pd.read_csv(fd)
        if {"owner_usd","renter_usd"}.issubset(d.columns):
            d["damage_usd"] = pd.to_numeric(d["owner_usd"], errors="coerce").fillna(0.0) \
                            + pd.to_numeric(d["renter_usd"], errors="coerce").fillna(0.0)
        elif "both_usd" in d.columns:
            d["damage_usd"] = pd.to_numeric(d["both_usd"], errors="coerce").fillna(0.0)
        else:
            d["damage_usd"] = pd.to_numeric(d.get("damage_usd", 0.0), errors="coerce").fillna(0.0)
        total_damage = float(d["damage_usd"].sum())

    # --- payout/financial total（用 tract 匯總）
    ft = run_dir / "finance" / "finance_tract_all_years.csv"
    total_payout = 0.0
    total_financial = 0.0
    if ft.exists():
        f = pd.read_csv(ft)
        pay = _pick(f, ["payout_usd","total_payout_usd","payout_total_usd","indemnity_usd"], required=False) or "_pay_zero_"
        if pay == "_pay_zero_": f[pay] = 0.0
        prem = _pick(f, ["premium_usd","premium_total_usd","total_premium_usd"], required=False) or "_prem_zero_"
        if prem == "_prem_zero_": f[prem] = 0.0
        oop  = _pick(f, ["oop_usd","out_of_pocket_usd","total_oop_usd","oop"], required=False) or "_oop_zero_"
        if oop == "_oop_zero_": f[oop] = 0.0

        total_payout    = float(pd.to_numeric(f[pay],  errors="coerce").fillna(0.0).sum())
        total_financial = float(
            pd.to_numeric(f[prem], errors="coerce").fillna(0.0).sum() +
            pd.to_numeric(f[oop],  errors="coerce").fillna(0.0).sum()
        )

    return total_damage, total_payout, total_financial


# ===== NEW: household-level distribution of (premium+OOP) annual mean =====

def dist_financial_per_household(run_dir: Path) -> np.ndarray:
    """
    回傳每戶「Premium+OOP 的年平均」分布（1D array, USD/HH）。
    讀取 finance_households_*.csv，對每戶先做 (premium+oop)，再在年份上取平均。
    若沒有 OOP 欄，會用 fallback: max(loss - payout, 0)。
    """
    run_dir = Path(run_dir)
    fdir = run_dir / "finance"
    hh = _concat_household_years(fdir)
    if hh is None or hh.empty:
        return np.array([])

    prem = _pick(hh, ["premium_usd","premium","premium_total_usd"])
    oop  = _pick(hh, ["oop_usd","out_of_pocket_usd","total_oop_usd","oop"], required=False)
    if oop is None:
        loss   = _pick(hh, ["loss_usd","total_loss_usd","gross_total_usd","total_damage_usd","damage_usd"], required=False)
        payout = _pick(hh, ["payout_usd","total_payout_usd","payout_total_usd","indemnity_usd","indemnity_total_usd"], required=False)
        if loss is not None and payout is not None:
            hh["_oop_fallback_"] = (pd.to_numeric(hh[loss], errors="coerce")
                                   - pd.to_numeric(hh[payout], errors="coerce")).clip(lower=0.0).fillna(0.0)
            oop = "_oop_fallback_"
        else:
            hh["_oop_zero_"] = 0.0
            oop = "_oop_zero_"

    # household id 欄位容錯
    hid = "i" if "i" in hh.columns else ("i_orig" if "i_orig" in hh.columns else None)
    if hid is None:
        hh = hh.reset_index().rename(columns={"index":"i"})
        hid = "i"

    hh["_fin_"] = pd.to_numeric(hh[prem], errors="coerce").fillna(0.0) + \
                  pd.to_numeric(hh[oop],  errors="coerce").fillna(0.0)

    g = hh.groupby(hid)["_fin_"].mean()  # 每戶年平均
    return g.to_numpy(dtype=float)

# ---------- NEW: averages for trade-off panel (requested) ----------
from pathlib import Path
import pandas as pd

def _policyholder_mask(hh: pd.DataFrame) -> pd.Series:
    """
    盡量識別 policyholders：
      1) 優先找 has_policy / insured / has_insurance / has_fi / buy_fi 等欄位
      2) 若無，fallback: premium > 0
    """
    cand_cols = ["has_policy","insured","has_insurance","has_fi","buy_fi",
                 "policyholder","is_policyholder"]
    for c in cand_cols:
        if c in hh.columns:
            try:
                s = pd.to_numeric(hh[c])  # 嚴格轉，若含非數字會丟例外
            except Exception:
                s = hh[c]                 # 維持原型，交給下面的 coerce 路徑

            if s.dtype == bool:
                return s

            # 0/1 或可數字化的字串 → 視為 True/False；不可數字者變 NaN，再當作 False
            return pd.to_numeric(hh[c], errors="coerce").fillna(0) > 0

    # fallback via premium
    prem = _pick(hh, ["premium_usd","premium","premium_total_usd"], required=False)
    if prem is not None:
        return pd.to_numeric(hh[prem], errors="coerce").fillna(0) > 0
    # 沒線索就全 False
    return pd.Series(False, index=hh.index)


def mean_over_years_policyholder_payout(run_dir: Path) -> float:
    """
    回傳「每年投保戶平均 payout」再跨年取平均。
    優先使用 finance_households_YYYY.csv；若缺，回傳 0.0（避免亂猜 policyholder 數）。
    """
    run_dir = Path(run_dir)
    fdir = run_dir / "finance"
    hh = _concat_household_years(fdir)
    if hh is None or hh.empty:
        return 0.0

    payout_col = _pick(hh, ["payout_usd","total_payout_usd","payout_total_usd",
                            "indemnity_usd","indemnity_total_usd"], required=False)
    if payout_col is None:
        return 0.0

    mask = _policyholder_mask(hh)
    if not mask.any():
        return 0.0

    hh["_payout_"] = pd.to_numeric(hh[payout_col], errors="coerce").fillna(0.0)
    hh["_is_pol_"] = mask.astype(bool)

    by = (hh[hh["_is_pol_"]]
            .groupby("year")["_payout_"]
            .mean()
            .rename("avg_payout_per_policyholder"))
    return float(by.mean()) if not by.empty else 0.0


def mean_over_years_damage_per_household(run_dir: Path) -> float:
    """
    回傳「每年全體住戶(ALL HH)的平均洪水損失」再跨年取平均。
    優先 household 檔；若缺，fallback 用 tract 總損失 / 該年平均戶數。
    """
    run_dir = Path(run_dir)
    fdir = run_dir / "finance"
    hh = _concat_household_years(fdir)
    if hh is not None and not hh.empty:
        loss = _pick(hh, ["loss_usd","total_loss_usd","gross_total_usd",
                          "gross_total_loss_usd","total_damage_usd","damage_usd"], required=False)
        if loss is not None:
            g = (pd.to_numeric(hh[loss], errors="coerce")
                   .groupby(hh["year"])
                   .mean())
            return float(g.mean()) if not g.empty else 0.0

    # tract-level fallback
    fd = run_dir / "vulnerability" / "flood_damage" / "flood_damage_tract_ALL_years.csv"
    ft = run_dir / "finance" / "finance_tract_all_years.csv"
    if fd.exists() and ft.exists():
        import numpy as np
        d = pd.read_csv(fd)
        if {"owner_usd","renter_usd"}.issubset(d.columns):
            d["damage_usd"] = pd.to_numeric(d["owner_usd"], errors="coerce").fillna(0.0) + \
                              pd.to_numeric(d["renter_usd"], errors="coerce").fillna(0.0)
        elif "both_usd" in d.columns:
            d["damage_usd"] = pd.to_numeric(d["both_usd"], errors="coerce").fillna(0.0)
        else:
            d["damage_usd"] = pd.to_numeric(d.get("damage_usd", 0.0), errors="coerce").fillna(0.0)

        f = pd.read_csv(ft)
        n_hh = _pick(f, ["n_households","households","n_hh"])
        nmap = (pd.to_numeric(f[n_hh], errors="coerce")
                  .groupby(f["year"]).mean().replace({0: np.nan}))

        dmg_per_hh_by_year = (d.groupby("year")["damage_usd"].mean() / nmap).dropna()
        return float(dmg_per_hh_by_year.mean()) if not dmg_per_hh_by_year.empty else 0.0

    return 0.0


def series_cum_payout_tenure(run_dir: Path) -> Tuple[
    Tuple[List[int], List[float]],  # (years, owners: cumulative per-policyholder payout)
    Tuple[List[int], List[float]],  # (years, renters: cumulative per-policyholder payout)
]:
    """
    先嘗試用 household 分片聚合（與你們前段讀法一致）。
    若無可用分片，回退到 tract 檔，以 owner/renter 的損失或保費比例拆分 total payout 與 policyholders。

    回傳: ((years, owner_cum_pp), (years, renter_cum_pp))
    """
    run_dir = Path(run_dir)
    fin_dir = run_dir / "finance"

    def _nan_cumsum(vals: List[float]) -> List[float]:
        s = 0.0; out = []
        for v in vals:
            if np.isfinite(v):
                s += float(v)
            out.append(s)
        return out

    # ===================== A) HH 分片路徑（優先） =====================
    hh_files = sorted(fin_dir.glob("finance_households_*.csv"))
    agg: Dict[int, Dict[str, float]] = {}
    if hh_files:
        for fp in hh_files:
            df = pd.read_csv(fp)

            # year
            ycol = _pick(df, ["year", "Year"])
            yr = pd.to_numeric(df[ycol], errors="coerce")

            # payout
            payout_col = None
            for c in ["payout_total_usd", "payout_usd", "claim_payout_usd"]:
                if c in df.columns:
                    payout_col = c; break
            if payout_col is None:
                pcs = [c for c in df.columns if c.lower() in ("payout_structure_usd", "payout_contents_usd")]
                if pcs:
                    df["__pout__"] = df[pcs].sum(axis=1, min_count=1)
                    payout_col = "__pout__"
                else:
                    # 這片沒有 payout，跳過
                    continue
            pay = pd.to_numeric(df[payout_col], errors="coerce").fillna(0.0)

            # tenure (owner vs renter)
            own_mask = None
            for c in ["is_owner", "owner", "tenure_is_owner"]:
                if c in df.columns:
                    s = pd.to_numeric(df[c], errors="coerce")
                    own_mask = (s.fillna(0) > 0) if s.dtype != bool else df[c].astype(bool)
                    break
            if own_mask is None:
                if "tenure" in df.columns:
                    own_mask = df["tenure"].astype(str).str.lower().str.contains("owner")
                else:
                    continue
            own_mask = own_mask.fillna(False)
            rent_mask = ~own_mask

            # policyholder flag（容錯：is_policyholder > premium_sum>0 > payout>0）
            ph_bool = None
            for c in ["is_policyholder", "policyholder", "has_policy", "ph"]:
                if c in df.columns:
                    col = df[c]
                    ph_bool = (pd.to_numeric(col, errors="coerce").fillna(0) > 0) if col.dtype != bool else col.astype(bool)
                    break
            if ph_bool is None:
                prem_cols = [c for c in df.columns if ("premium" in c.lower()) and c.lower().endswith("_usd")]
                if prem_cols:
                    ph_bool = df[prem_cols].sum(axis=1, min_count=1).fillna(0) > 0
                else:
                    ph_bool = pay.fillna(0) > 0  # 兜底

            tmp = pd.DataFrame({
                "year": yr,
                "opay": np.where(own_mask,  pay, 0.0),
                "rpay": np.where(rent_mask, pay, 0.0),
                "oph":  np.where(own_mask  & ph_bool, 1.0, 0.0),
                "rph":  np.where(rent_mask & ph_bool, 1.0, 0.0),
            }).dropna(subset=["year"])

            g = tmp.groupby("year").sum(numeric_only=True)
            for y, row in g.iterrows():
                d = agg.setdefault(int(y), {"opay":0.0,"rpay":0.0,"oph":0.0,"rph":0.0})
                d["opay"] += float(row.get("opay", 0.0))
                d["rpay"] += float(row.get("rpay", 0.0))
                d["oph"]  += float(row.get("oph",  0.0))
                d["rph"]  += float(row.get("rph",  0.0))

        if agg:  # 有成功聚合，直接回傳
            years = sorted(agg.keys())
            owner_pp  = [(agg[y]["opay"]/agg[y]["oph"]) if agg[y]["oph"]>0 else np.nan for y in years]
            renter_pp = [(agg[y]["rpay"]/agg[y]["rph"]) if agg[y]["rph"]>0 else np.nan for y in years]
            return (years, _nan_cumsum(owner_pp)), (years, _nan_cumsum(renter_pp))

    # ===================== B) Tract 檔回退（拆分法） =====================
    # 用 owner/renter 的「損失或保費占比」拆分 total payout 與 policyholders。
    tract = fin_dir / "finance_tract_all_years.csv"
    if not tract.exists():
        raise RuntimeError("series_cum_payout_tenure: aggregation is empty (no usable rows).")

    df = pd.read_csv(tract)
    ycol = _pick(df, ["year", "Year"])

    # 年度總 payout、總 policyholders
    payout_tot = _pick(df, ["payout_total_usd"])
    ph_tot_col = _pick(df, ["policyholders"])

    # owner/renter 的分拆基礎：優先用「各自的 gross（結構+內容）」，否則用各自的 premium_total
    # gross
    own_gross = None; rent_gross = None
    if "owner_gross_total_usd" in df.columns and "renter_gross_total_usd" in df.columns:
        own_gross  = pd.to_numeric(df["owner_gross_total_usd"], errors="coerce").fillna(0.0)
        rent_gross = pd.to_numeric(df["renter_gross_total_usd"], errors="coerce").fillna(0.0)
    else:
        own_gross = rent_gross = None

    # premium
    own_prem = df["owner_premium_total_usd"] if "owner_premium_total_usd" in df.columns else None
    rent_prem = df["renter_premium_total_usd"] if "renter_premium_total_usd" in df.columns else None
    if own_prem is not None:
        own_prem = pd.to_numeric(own_prem, errors="coerce").fillna(0.0)
    if rent_prem is not None:
        rent_prem = pd.to_numeric(rent_prem, errors="coerce").fillna(0.0)

    # households（最後兜底）
    own_hh = df["owner_households"] if "owner_households" in df.columns else None
    rent_hh = df["renter_households"] if "renter_households" in df.columns else None
    if own_hh is not None:
        own_hh = pd.to_numeric(own_hh, errors="coerce").fillna(0.0)
    if rent_hh is not None:
        rent_hh = pd.to_numeric(rent_hh, errors="coerce").fillna(0.0)

    # 逐年 groupby
    g = (df.groupby(ycol, as_index=False)
            .agg(payout_total=(payout_tot, "sum"),
                 policyholders=(ph_tot_col, "sum"),
                 owner_gross=("owner_gross_total_usd", "sum") if own_gross is not None else ("tract_geoid", "size"),
                 renter_gross=("renter_gross_total_usd", "sum") if rent_gross is not None else ("tract_geoid", "size"),
                 owner_prem=("owner_premium_total_usd", "sum") if own_prem is not None else ("tract_geoid", "size"),
                 renter_prem=("renter_premium_total_usd", "sum") if rent_prem is not None else ("tract_geoid", "size"),
                 owner_hh=("owner_households", "sum") if own_hh is not None else ("tract_geoid", "size"),
                 renter_hh=("renter_households", "sum") if rent_hh is not None else ("tract_geoid", "size"))
            .sort_values(ycol))

    # 抽出序列
    years = pd.to_numeric(g[ycol], errors="coerce").astype(int).tolist()
    Ptot  = pd.to_numeric(g["payout_total"], errors="coerce").fillna(0.0).to_numpy()
    PH    = pd.to_numeric(g["policyholders"], errors="coerce").fillna(0.0).to_numpy()

    # 拆分權重：優先 gross，其次 premium，最後 households
    w_pay_owner = None; w_ph_owner = None
    if "owner_gross" in g and "renter_gross" in g and g[["owner_gross","renter_gross"]].to_numpy().sum() > 0:
        og = pd.to_numeric(g["owner_gross"], errors="coerce").fillna(0.0).to_numpy()
        rg = pd.to_numeric(g["renter_gross"], errors="coerce").fillna(0.0).to_numpy()
        denom = np.where((og + rg) > 0, (og + rg), 1.0)
        w_pay_owner = og / denom
    elif "owner_prem" in g and "renter_prem" in g and g[["owner_prem","renter_prem"]].to_numpy().sum() > 0:
        op = pd.to_numeric(g["owner_prem"], errors="coerce").fillna(0.0).to_numpy()
        rp = pd.to_numeric(g["renter_prem"], errors="coerce").fillna(0.0).to_numpy()
        denom = np.where((op + rp) > 0, (op + rp), 1.0)
        w_pay_owner = op / denom
    else:
        oh = pd.to_numeric(g["owner_hh"], errors="coerce").fillna(0.0).to_numpy()
        rh = pd.to_numeric(g["renter_hh"], errors="coerce").fillna(0.0).to_numpy()
        denom = np.where((oh + rh) > 0, (oh + rh), 1.0)
        w_pay_owner = oh / denom

    # policyholders 拆分：優先 premium，最後 households
    if "owner_prem" in g and "renter_prem" in g and g[["owner_prem","renter_prem"]].to_numpy().sum() > 0:
        op = pd.to_numeric(g["owner_prem"], errors="coerce").fillna(0.0).to_numpy()
        rp = pd.to_numeric(g["renter_prem"], errors="coerce").fillna(0.0).to_numpy()
        denom = np.where((op + rp) > 0, (op + rp), 1.0)
        w_ph_owner = op / denom
    else:
        oh = pd.to_numeric(g["owner_hh"], errors="coerce").fillna(0.0).to_numpy()
        rh = pd.to_numeric(g["renter_hh"], errors="coerce").fillna(0.0).to_numpy()
        denom = np.where((oh + rh) > 0, (oh + rh), 1.0)
        w_ph_owner = oh / denom

    # 拆成 owner / renter 的 payout 與 policyholders
    owner_pay = w_pay_owner * Ptot
    renter_pay = (1.0 - w_pay_owner) * Ptot
    owner_ph = w_ph_owner * PH
    renter_ph = (1.0 - w_ph_owner) * PH

    # 每年每保戶 payout
    owner_pp = np.where(owner_ph > 0, owner_pay / owner_ph, np.nan)
    renter_pp = np.where(renter_ph > 0, renter_pay / renter_ph, np.nan)

    # 累積
    owner_cum  = _nan_cumsum(owner_pp.tolist())
    renter_cum = _nan_cumsum(renter_pp.tolist())

    return (years, owner_cum), (years, renter_cum)