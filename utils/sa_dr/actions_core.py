# utils/sa_dr/actions_core.py
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import re
import numpy as np
import pandas as pd

# --------- helpers ---------
def _col_map(df: pd.DataFrame) -> Dict[str, str]:
    return {c.lower(): c for c in df.columns}

def _col_like(df: pd.DataFrame, *names: str) -> Optional[str]:
    m = _col_map(df)
    for n in names:
        if n.lower() in m:
            return m[n.lower()]
    return None

def _is_share_name(name: str) -> bool:
    cl = name.lower()
    return any(k in cl for k in ["share","ratio","frac","prop","pct","percent"])

def _is_count_name(name: str) -> bool:
    cl = name.lower()
    return ("_n_" in cl or cl.endswith("_n") or "n_total" in cl or
            "count" in cl or "num" in cl)

def _ensure_DN_cols(df: pd.DataFrame) -> pd.DataFrame:
    for a in ["FI","EH","BP","RL"]:
        if a not in df.columns:
            df[a] = 0.0
        df[a] = pd.to_numeric(df[a], errors="coerce").fillna(0.0).clip(0,1)
    if "DN" not in df.columns or df["DN"].isna().all():
        df["DN"] = (1.0 - (df["FI"]+df["EH"]+df["BP"]+df["RL"])).clip(0,1)
    else:
        df["DN"] = pd.to_numeric(df["DN"], errors="coerce").fillna(0.0).clip(0,1)
    return df

def _pick(df, names, required=True):
    for n in names:
        if n in df.columns: return n
    if required: raise KeyError(f"missing column, tried {names}")
    return None

# --------- core read ---------
def read_action_share_all(run_dir: Path, pattern: str = "action_share_owner_renter_tract_*.csv") -> pd.DataFrame:
    """
    Normalize decisions/action_share_* files to long tidy format:
      columns: year, tract_geoid, group (Owner/Renter), FI, EH, BP, RL, DN
    Priority: pick share/ratio/frac/prop/pct columns; ignore *_n_* counts;
    if only counts exist (n + *_n_total), derive share = n/total.
    """
    run_dir = Path(run_dir)
    decdir = run_dir / "decisions"
    p_all = decdir / "action_share_owner_renter_tract_all_years.csv"

    # read
    if p_all.exists():
        df = pd.read_csv(p_all)
    else:
        parts = sorted(decdir.glob(pattern))
        if not parts:
            raise FileNotFoundError(f"No action-share CSV under {decdir}")
        buf = []
        for f in parts:
            try:
                d = pd.read_csv(f)
                if "year" not in d.columns:
                    m = re.search(r"(19|20)\d{2}", f.stem)
                    if m: d.insert(0, "year", int(m.group(0)))
                buf.append(d)
            except Exception:
                pass
        df = pd.concat(buf, ignore_index=True) if buf else pd.DataFrame()

    if df.empty:
        raise FileNotFoundError("Action-share file is empty")

    # standardize columns
    df = df.rename(columns={c: c.strip() for c in df.columns})
    ycol = _col_like(df, "year") or "year"
    if ycol not in df.columns:
        df.insert(0, "year", np.nan); ycol = "year"
    tcol = _col_like(df, "tract_geoid","tract","geoid","tractid")
    if tcol is None:
        raise KeyError("Missing tract id column")

    # A. narrow table: has group column
    gcol = _col_like(df, "group","identity","grp")
    if gcol is not None:
        out = df.copy()
        out[gcol] = out[gcol].astype(str).str.strip().str.lower()
        out[gcol] = out[gcol].map(lambda s: "Owner" if s.startswith("own") else ("Renter" if s.startswith("rent") else s.title()))
        for a in ["FI","EH","BP","RL","DN"]:
            ac = _col_like(out, a)
            out[a] = pd.to_numeric(out[ac], errors="coerce").fillna(0.0) if ac else 0.0
        out = out.rename(columns={ycol:"year", tcol:"tract_geoid", gcol:"group"})
        out = out[["year","tract_geoid","group","FI","EH","BP","RL","DN"]].copy()
        out = _ensure_DN_cols(out)
        out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
        out["tract_geoid"] = out["tract_geoid"].astype(str)
        return out

    # B. wide: parse names to (group, action)
    triples = []  # (group, action, column)
    for c in df.columns:
        if c in {ycol, tcol}: 
            continue
        cl = c.strip().lower()
        m = re.search(r'(owner|renter).*?(fi|eh|bp|rl|dn)', cl) or \
            re.search(r'(fi|eh|bp|rl|dn).*?(owner|renter)', cl)
        if not m: 
            continue
        a, g = m.group(1), m.group(2)
        if a in {"owner","renter"} and g in {"fi","eh","bp","rl","dn"}:
            a, g = g, a
        if g in {"owner","renter"} and a in {"fi","eh","bp","rl","dn"}:
            group = "Owner" if g == "owner" else "Renter"
            action = a.upper()
            triples.append((group, action, c))

    # choose best column per (group,action)
    chosen: Dict[Tuple[str,str], str] = {}
    for (group, action, c) in triples:
        key = (group, action)
        if key not in chosen:
            chosen[key] = c
        else:
            best = chosen[key]
            if (_is_share_name(c) and not _is_share_name(best)) or (_is_share_name(c) and _is_share_name(best) and len(c)<len(best)):
                chosen[key] = c
            elif (_is_count_name(best) and not _is_count_name(c)):
                chosen[key] = c

    base_keep = [ycol, tcol]
    long_parts = []

    for (group, action), colname in chosen.items():
        if _is_count_name(colname):
            total_key = f"{group.lower()}_n_total"
            total_col = _col_map(df).get(total_key)
            if total_col is None:
                continue
            tmp = df[base_keep + [colname, total_col]].copy()
            num = pd.to_numeric(tmp[colname], errors="coerce")
            den = pd.to_numeric(tmp[total_col], errors="coerce").replace({0: np.nan})
            val = (num/den).clip(0,1)
            tmp = tmp.drop(columns=[colname, total_col])
            tmp["group"] = group; tmp["action"] = action; tmp["value"] = val
            long_parts.append(tmp)
        else:
            tmp = df[base_keep + [colname]].copy()
            val = pd.to_numeric(tmp[colname], errors="coerce").fillna(0.0)
            if val.max() > 1.0:
                val = (val/100.0).clip(0,1)
            tmp = tmp.drop(columns=[colname])
            tmp["group"] = group; tmp["action"] = action; tmp["value"] = val
            long_parts.append(tmp)

    if not long_parts:
        raise KeyError("Cannot infer wide-format share columns.")

    long = pd.concat(long_parts, ignore_index=True)
    out = (long.pivot_table(index=base_keep + ["group"], columns="action", values="value", aggfunc="first")
                .reset_index())
    out = out.rename(columns={ycol:"year", tcol:"tract_geoid"})
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    out["tract_geoid"] = out["tract_geoid"].astype(str)
    out["group"] = out["group"].astype(str)
    out = _ensure_DN_cols(out)
    return out[["year","tract_geoid","group","FI","EH","BP","RL","DN"]].copy()

# --------- aggregations for plotting ---------
def _tract_yearly_mean_by_action(run_dir: Path, pattern: str) -> pd.DataFrame:
    df = read_action_share_all(run_dir, pattern)
    g = (df.groupby(["tract_geoid","group"])[["FI","EH","BP","RL","DN"]]
           .mean().reset_index())
    return g

def selection_vectors(run_dir: Path,
                      selections: List[Tuple[str, Optional[str], str]],
                      pattern: str) -> Dict[str, np.ndarray]:
    """
    Return per-selection tract vectors (values in [0,1]); each value is
    the mean over simulation years for that tract. If group is None,
    take unweighted mean of Owner/Renter per tract.
    """
    base = _tract_yearly_mean_by_action(run_dir, pattern)
    all_tracts = base["tract_geoid"].unique()
    out: Dict[str, np.ndarray] = {}
    for label, grp, col in selections:
        if grp is None:
            tmp = (base.pivot(index="tract_geoid", columns="group", values=col)
                        .reindex(index=all_tracts, columns=["Owner","Renter"]))
            vec = tmp.mean(axis=1, skipna=True).to_numpy(dtype=float)
        else:
            vec = (base.loc[base["group"]==grp, ["tract_geoid", col]]
                        .set_index("tract_geoid")[col]
                        .reindex(all_tracts).to_numpy(dtype=float))
        vec = vec[~np.isnan(vec)]
        out[label] = vec
    return out

def heatmap_matrix(run_dirs_map: Dict[str, Path],
                   selections: List[Tuple[str, Optional[str], str]],
                   pattern: str) -> pd.DataFrame:
    """
    rows = selection labels; cols = scenario labels; cell = mean over tracts
    of the per-tract mean-over-years share for that selection.
    """
    rows = []
    for sel_label, grp, col in selections:
        r = {"metric": sel_label}
        for scen, rdir in run_dirs_map.items():
            vecs = selection_vectors(rdir, [(sel_label, grp, col)], pattern)
            v = vecs[sel_label]
            r[scen] = float(np.nanmean(v)) if v.size else np.nan
        rows.append(r)
    return pd.DataFrame(rows).set_index("metric")

def eh_coverage_series(
    run_dir: Path,
    pattern: str = "action_share_owner_renter_tract_*.csv",  # ← 新增，支援第二個位置參數
    *,
    county_fips: str | None = None,
    group_priority=("Owner","Renter"),
    threshold: float = 1e-9,   
) -> pd.Series:
    df = read_action_share_all(run_dir, pattern)   # ← 把 pattern 傳進去

    # 只取某縣（optional）
    if county_fips:
        df = df[df["tract_geoid"].astype(str).str.startswith(county_fips)].copy()

    ycol = _pick(df, ["year"])
    tcol = _pick(df, ["tract_geoid","tract","geoid","tractid"])

    # 取 EH share 欄（先找 owner，再找 renter，再找通用命名）
    eh_col = None
    for cand in ["owner_share_EH", "renter_share_EH", "EH", "eh_share", "share_EH"]:
        if cand in df.columns:
            eh_col = cand
            break
    if eh_col is None:
        # 若是寬表，前面的 read_action_share_all 已經幫你整理出 EH 欄
        eh_col = "EH"
        if eh_col not in df.columns:
            raise KeyError("No EH share column found.")

    base = df[[ycol, tcol, eh_col]].copy()
    base.columns = ["year","tract_geoid","eh_share"]
    base["eh_share"] = pd.to_numeric(base["eh_share"], errors="coerce").fillna(0.0).clip(0,1)

    # 每個 tract 逐年累積，封頂 1
    base = base.sort_values(["tract_geoid","year"])
    base["cov"] = base.groupby("tract_geoid")["eh_share"].cumsum().clip(upper=1.0)

    # 每年對所有 tract 平均 → 覆蓋率
    s = base.groupby("year")["cov"].mean().sort_index()
    try: s.index = s.index.astype(int)
    except: pass
    return s
