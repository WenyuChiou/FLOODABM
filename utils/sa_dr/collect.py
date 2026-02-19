# utils/sa/collect.py
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

def _pick(df, names, required=True):
    """
    先大小寫不敏感精確比對；不行再用模糊包含比對。
    """
    cols = list(df.columns)
    lower_map = {c.lower(): c for c in cols}

    # 精確（不分大小寫）
    for n in names:
        c = lower_map.get(n.lower())
        if c is not None:
            return c

    # 模糊（包含關鍵字）
    for n in names:
        key = n.lower()
        for c in cols:
            if key in c.lower():
                return c

    if required:
        raise KeyError(f"missing column; tried {names} ; available={cols}")
    return None




def cumulative_avg_financial(run_dir: Path) -> pd.DataFrame:
    """
    回傳 columns=[group, cum_avg_financial_usd]（owner/renter）。
    以 household-level 每年 (premium+OOP) 平均後跨年加總。
    若沒有 household 檔，退而用 tract 匯總近似。
    """
    fdir = Path(run_dir) / "finance"
    parts = sorted(fdir.glob("finance_households_*.csv"))

    # ── 優先使用 household 檔 ──────────────────────────────────────────
    if parts:
        dfs = []
        for p in parts:
            try:
                dfs.append(pd.read_csv(p))
            except Exception:
                pass
        if dfs:
            hh = pd.concat(dfs, ignore_index=True)

            grp  = _pick(hh, ["identity", "group", "owner_renter"])
            prem = _pick(hh, ["premium_usd", "premium", "premium_total_usd"])
            year = _pick(hh, ["year", "YEAR"])

            # 先找 OOP 欄位；找不到就用 (loss - payout) 推回；再不行當 0
            oop  = _pick(hh, ["oop_usd","out_of_pocket_usd","total_oop_usd","oop"], required=False)
            if oop is None:
                loss   = _pick(hh, ["loss_usd","total_loss_usd","gross_total_usd","gross_total_loss_usd",
                                    "total_damage_usd","damage_usd"], required=False)
                payout = _pick(hh, ["payout_usd","total_payout_usd","payout_total_usd",
                                    "indemnity_usd","indemnity_total_usd"], required=False)
                if loss is not None and payout is not None:
                    val = (pd.to_numeric(hh[loss], errors="coerce")
                         - pd.to_numeric(hh[payout], errors="coerce")).clip(lower=0.0)
                    hh["_oop_fallback_"] = val.fillna(0.0)
                    oop = "_oop_fallback_"
                else:
                    print("[warn] OOP column not found and cannot infer from (loss - payout). Treat OOP=0.")
                    hh["_oop_zero_"] = 0.0
                    oop = "_oop_zero_"

            hh["financial_usd"] = (
                pd.to_numeric(hh[prem], errors="coerce").fillna(0.0)
              + pd.to_numeric(hh[oop],  errors="coerce").fillna(0.0)
            )

            yrgrp = (hh.groupby([year, grp])["financial_usd"].mean().reset_index())
            out   = (yrgrp.groupby(grp)["financial_usd"].sum().reset_index())
            out.columns = ["group", "cum_avg_financial_usd"]
            return out

    # ── 回退：用 tract 匯總近似（(premium+OOP)/n_households，逐年加總） ─────────
    tfile = fdir / "finance_tract_all_years.csv"
    if not tfile.exists():
        raise FileNotFoundError(f"no finance households or tract csv found in {fdir}")

    df   = pd.read_csv(tfile)
    grp  = _pick(df, ["identity","group","owner_renter"])
    prem = _pick(df, ["premium_usd","premium_total_usd","total_premium_usd"], required=False)
    oop  = _pick(df, ["oop_usd","out_of_pocket_usd","total_oop_usd","oop"], required=False)
    n_hh = _pick(df, ["n_households","households","n_hh"])

    if prem is None:
        df["_prem_zero_"] = 0.0
        prem = "_prem_zero_"
    if oop is None:
        # 後備：用 (loss - payout)/n_households 估 OOP；再不行就 0
        loss   = _pick(df, ["loss_usd","total_loss_usd","gross_total_usd","total_damage_usd","damage_usd"], required=False)
        payout = _pick(df, ["payout_usd","total_payout_usd","payout_total_usd","indemnity_usd"], required=False)
        if loss is not None and payout is not None:
            val = (pd.to_numeric(df[loss], errors="coerce")
                 - pd.to_numeric(df[payout], errors="coerce")).clip(lower=0.0)
            denom = pd.to_numeric(df[n_hh], errors="coerce").replace({0: np.nan})
            df["_oop_est_"] = (val / denom).fillna(0.0)
            oop = "_oop_est_"
        else:
            df["_oop_zero_"] = 0.0
            oop = "_oop_zero_"

    df["avg_financial"] = (
        pd.to_numeric(df[prem], errors="coerce").fillna(0.0)
      + pd.to_numeric(df[oop],  errors="coerce").fillna(0.0)
    ) / pd.to_numeric(df[n_hh], errors="coerce").replace({0: np.nan})

    out = df.groupby(grp)["avg_financial"].sum().reset_index()
    out.columns = ["group", "cum_avg_financial_usd"]
    return out


def cumulative_avg_flood_damage(run_dir: Path) -> pd.DataFrame:
    """
    回傳 columns=[group, cum_avg_damage_usd]（owner/renter）。
    以 tract-level 每年平均後跨年加總。
    """
    fdir = Path(run_dir) / "vulnerability" / "flood_damage"
    all_years = fdir / "flood_damage_tract_ALL_years.csv"
    if all_years.exists():
        dfw = pd.read_csv(all_years)
        rows = []
        if "owner_usd" in dfw.columns:
            a = dfw[["year","tract_geoid","owner_usd"]].rename(columns={"owner_usd":"damage_usd"}); a["group"]="owner"; rows.append(a)
        if "renter_usd" in dfw.columns:
            b = dfw[["year","tract_geoid","renter_usd"]].rename(columns={"renter_usd":"damage_usd"}); b["group"]="renter"; rows.append(b)
        df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    else:
        def _read_all(tag):
            rows = []
            for p in fdir.glob(f"flood_damage_tract_{tag}_*.csv"):
                if p.stem.endswith("_ALL") or p.stem.endswith("_years"): continue
                df = pd.read_csv(p, usecols=["year","tract_geoid","damage_usd"])
                df["group"] = tag
                rows.append(df)
            return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        df = pd.concat([_read_all("owner"), _read_all("renter")], ignore_index=True)

    if df.empty:
        raise FileNotFoundError(f"no flood damage csvs in {fdir}")
    by = df.groupby(["year","group"])["damage_usd"].mean().reset_index()
    out = by.groupby("group")["damage_usd"].sum().reset_index()
    out.columns = ["group","cum_avg_damage_usd"]
    return out
