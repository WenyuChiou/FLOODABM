# core/data.py
# Data Loading and Imputation Module

import pandas as pd
import numpy as np
from pathlib import Path
import re, math
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg

def _auto01(s: pd.Series) -> pd.Series:
    """Normalize series to [0,1] if it looks like a 1-5 scale."""
    vals = pd.to_numeric(s, errors="coerce")
    if vals.max(skipna=True) > 1.5:
        vals = vals / 5.0
    return vals.clip(0, 1)

def parse_q15_value(x):
    """Parses messy time strings like '10-20', '10+', '>=5', etc."""
    if x is None: return (math.nan, math.nan)
    try:
        xv = float(x)
        if np.isfinite(xv): return (xv, xv)
    except Exception:
        pass
    s = str(x).strip().lower()
    m = re.match(r'^\s*(\d+)\s*[-~\u2013]\s*(\d+)\s*$', s)
    if m: return (float(m.group(1)), float(m.group(2)))
    m = re.match(r'^\s*(?:>=|>|at least)?\s*(\d+)\s*\+?\s*$', s)
    if m:
        lo = float(m.group(1)); return (lo, max(lo+5, lo))
    nums = re.findall(r'\d+', s)
    if nums:
        lo = float(nums[0]); return (lo, max(lo+5, lo))
    return (math.nan, math.nan)

def impute_time_stochastic(series: pd.Series, mean: float, std: float, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    series = series.copy()
    nulls = series.isna()
    if nulls.any():
        n_miss = nulls.sum()
        fill_vals = rng.normal(loc=mean, scale=std, size=n_miss)
        fill_vals = np.clip(fill_vals, 1.0, 15.0)
        series.loc[nulls] = fill_vals
    return series

def load_group_sheet(path: Path, sheet: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_excel(path, sheet_name=sheet)
    cols = {c.lower(): c for c in df.columns}

    def get_col_data(alias_list):
        for name in alias_list:
            if name.lower() in cols: return df[cols[name.lower()]]
        return None

    PA = _auto01(get_col_data(cfg.COLS["PA"]))
    SC = _auto01(get_col_data(cfg.COLS["SC"]))
    TP = _auto01(get_col_data(cfg.COLS["TP"]))

    if PA is None or SC is None or TP is None:
        raise ValueError(f"Missing PA, SC, or TP columns in sheet {sheet}")

    t_lo_series = get_col_data(cfg.COLS["T_LO"])
    t_hi_series = get_col_data(cfg.COLS["T_HI"])

    if t_lo_series is not None:
        t_lo = pd.to_numeric(t_lo_series, errors='coerce')
        if t_hi_series is not None:
            t_hi = pd.to_numeric(t_hi_series, errors='coerce')
        else:
            t_hi = t_lo.copy()
    elif cfg.COL_Q15_RAW in cols:
        pairs = [parse_q15_value(v) for v in df[cols[cfg.COL_Q15_RAW]]]
        t_lo = pd.Series([p[0] for p in pairs])
        t_hi = pd.Series([p[1] for p in pairs])
    else:
        t_lo = pd.Series([np.nan] * len(df))
        t_hi = pd.Series([np.nan] * len(df))

    t_lo = impute_time_stochastic(t_lo, cfg.IMPUTE_MEAN, cfg.IMPUTE_STD, cfg.RANDOM_SEED)
    t_hi = t_hi.fillna(t_lo)

    return pd.DataFrame(dict(PA=PA, SC=SC, TP=TP, t_lo=t_lo, t_hi=t_hi))
