# -*- coding: utf-8 -*-
"""
Generate Table S_MC_variance — per-year spread of key metrics across the
50 Monte Carlo baseline runs. Serves as defensive evidence for reviewers
asking "is your MC actually doing anything?"

Metrics covered (matching main-text figures):
  Fig. 4 (c): Insurance leverage ratio (owner, renter)
  Fig. 5   : Owner/Renter OOP rate, Payout rate
  Fig. 7   : FI/EH/RL adoption (flood-prone vs non-prone, owner + renter)
  Fig. 8   : Weighted tract-level mean TP (flood-prone vs non-prone, owner + renter)

Outputs:
  Table_S_MC_variance_long.csv   — one row per (metric, year) with
                                    min/q25/median/q75/max across 50 runs
  Table_S_MC_variance_endpoint.csv — compact wide table for the SI showing
                                      2011, 2021, 2023 only
"""
from __future__ import annotations
import sys
import glob
import re
import json
from pathlib import Path
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

MC = Path(r"C:\FLOODABM_mc50_v2")
N_RUNS = 50
YEARS = list(range(2011, 2024))
FP_THRESHOLD = 7  # matches fig8 flood-prone classification

ROOT = Path(r"C:\Users\wenyu\OneDrive - Lehigh University\Desktop\Lehigh"
            r"\NSF-project\ABM\paper\draft\mg_sensitivity\FLOODABM")
OUT_DIR = Path(r"C:\Users\wenyu\OneDrive - Lehigh University\Desktop\Lehigh"
               r"\NSF-project\ABM\paper\Figure\SI")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def run_path(rid):
    return MC / "baseline" / f"run_{rid:02d}" / "baseline" / "baseline"


# -- 1. Flood-prone classification (matches plot_fig8_tp_by_prone.py) ---------
def classify_flood_prone():
    df = pd.read_csv(run_path(1) / "flood_years_by_tract.csv",
                     dtype={"tract_geoid": str})
    df["flooded"] = df["depth_m"] > 0
    counts = df.groupby("tract_geoid")["flooded"].sum()
    return (set(counts[counts >= FP_THRESHOLD].index),
            set(counts[counts < FP_THRESHOLD].index))


# -- 2. Finance metrics per run (fig4/fig5) ---------------------------------
def load_finance_run(rd, columns):
    rows = []
    for y in YEARS:
        f = rd / "finance" / f"finance_tract_{y}.csv"
        if not f.exists():
            return None
        df = pd.read_csv(f, dtype={"tract_geoid": str})
        df["year"] = y
        rows.append(df)
    fin = pd.concat(rows, ignore_index=True)
    agg = fin.groupby("year").agg(
        owner_gross=("owner_gross_total_usd", "sum"),
        renter_gross=("renter_gross_total_usd", "sum"),
        owner_payout=("owner_payout_total_usd", "sum"),
        renter_payout=("renter_payout_total_usd", "sum"),
        owner_oop=("owner_oop_total_usd", "sum"),
        renter_oop=("renter_oop_total_usd", "sum"),
        owner_hh=("owner_households", "sum"),
        renter_hh=("renter_households", "sum"),
    ).reset_index().fillna(0).sort_values("year").reset_index(drop=True)
    return agg


# -- 3. Tract-level adoption rates per run (fig7 c-f) -----------------------
def weighted_mean(values, weights):
    w = np.where(weights > 0, weights, 1e-6)
    return np.average(values, weights=w) if len(values) else np.nan


def adoption_rates(rd, group, action, fp, nfp):
    """Return (fp_array, nfp_array) of weighted-mean adoption rate per year."""
    fp_r, nfp_r = [], []
    for y in YEARS:
        f = rd / "decisions" / f"decisions_mgmix_{y}.csv"
        df = pd.read_csv(f, dtype={"tract_geoid": str})
        g = df[df["group"] == group].copy()
        g["tract_geoid"] = g["tract_geoid"].astype(str)
        rates, pops = {}, {}
        for t, sub in g.groupby("tract_geoid"):
            n = len(sub)
            if n == 0: continue
            pops[t] = n
            if action == "FI":
                is_a = (sub["action"] == "FI")
                if "POLICY_NAME" in sub.columns:
                    has_pol = sub["POLICY_NAME"].notna() & (sub["POLICY_NAME"].astype(str).str.len() > 0)
                    is_a = is_a | has_pol
                rates[t] = is_a.sum() / n
            elif action == "EH":
                is_a = (sub["action"] == "EH")
                if "ELEV_FT" in sub.columns:
                    is_a = is_a | (sub["ELEV_FT"] > 0)
                rates[t] = is_a.sum() / n
            else:
                rates[t] = (sub["action"] == action).sum() / n
        fp_vals = [rates[t] for t in rates if t in fp]
        fp_ws   = [pops[t]  for t in rates if t in fp]
        nfp_vals = [rates[t] for t in rates if t in nfp]
        nfp_ws   = [pops[t]  for t in rates if t in nfp]
        fp_r.append(weighted_mean(np.array(fp_vals), np.array(fp_ws, dtype=float)))
        nfp_r.append(weighted_mean(np.array(nfp_vals), np.array(nfp_ws, dtype=float)))
    return np.array(fp_r), np.array(nfp_r)


# -- 4. TP weighted means per run (fig8) ------------------------------------
def tp_weighted(rd, tp_col, group, fp, nfp):
    tp = pd.read_csv(rd / "tp_traj.csv", dtype={"tract_geoid": str})
    tp = tp[tp["phase"] == "after"]
    fp_r, nfp_r = [], []
    for y in YEARS:
        yr = tp[tp["year"] == y].groupby("tract_geoid")[tp_col].mean()
        dec = pd.read_csv(rd / "decisions" / f"decisions_mgmix_{y}.csv",
                          dtype={"tract_geoid": str})
        pop = dec[dec["group"] == group].groupby("tract_geoid").size()
        fp_i = yr.index.intersection(fp)
        nfp_i = yr.index.intersection(nfp)
        fp_r.append(weighted_mean(yr.loc[fp_i].values,
                                   pop.reindex(fp_i).fillna(1).values.astype(float)))
        nfp_r.append(weighted_mean(yr.loc[nfp_i].values,
                                    pop.reindex(nfp_i).fillna(1).values.astype(float)))
    return np.array(fp_r), np.array(nfp_r)


# ============================================================================
print(f"Loading {N_RUNS} MC runs and computing metrics...")
FP, NFP = classify_flood_prone()
print(f"  Flood-prone tracts: {len(FP)}  Non-prone: {len(NFP)}")

# Containers: dict[metric_key] -> list of (n_years,) arrays across runs
metrics = {}


def add(key, arr):
    metrics.setdefault(key, []).append(arr)


for rid in range(1, N_RUNS + 1):
    rd = run_path(rid)
    if not rd.exists(): continue

    # Finance — per-HH cumulative and rates
    fin = load_finance_run(rd, None)
    if fin is not None:
        o_hh = np.maximum(fin["owner_hh"].values, 1)
        r_hh = np.maximum(fin["renter_hh"].values, 1)
        o_g = fin["owner_gross"].values
        r_g = fin["renter_gross"].values
        # Fig6 = Financial outcomes (paper Figure 6: premium + OOP + payout)
        add("Fig6_owner_OOP_rate_pct",
            np.where(o_g > 0, fin["owner_oop"].values / o_g * 100, 0))
        add("Fig6_renter_OOP_rate_pct",
            np.where(r_g > 0, fin["renter_oop"].values / r_g * 100, 0))
        add("Fig6_owner_payout_rate_pct",
            np.where(o_g > 0, fin["owner_payout"].values / o_g * 100, 0))
        add("Fig6_renter_payout_rate_pct",
            np.where(r_g > 0, fin["renter_payout"].values / r_g * 100, 0))

    # Fig7 c-f: owner FI, renter FI, owner EH, renter RL
    for (grp, act, prefix) in [
        ("owner", "FI", "Fig7_owner_FI"),
        ("renter", "FI", "Fig7_renter_FI"),
        ("owner", "EH", "Fig7_owner_EH"),
        ("renter", "RL", "Fig7_renter_RL"),
    ]:
        fp_r, nfp_r = adoption_rates(rd, grp, act, FP, NFP)
        add(f"{prefix}_floodprone_rate_pct", fp_r * 100)
        add(f"{prefix}_nonprone_rate_pct", nfp_r * 100)

    # Fig8: TP weighted mean flood-prone vs non-prone
    for (tp_col, group, prefix) in [
        ("TP_owner", "owner", "Fig8_owner_TP"),
        ("TP_renter", "renter", "Fig8_renter_TP"),
    ]:
        fp_tp, nfp_tp = tp_weighted(rd, tp_col, group, FP, NFP)
        add(f"{prefix}_floodprone", fp_tp)
        add(f"{prefix}_nonprone", nfp_tp)

    if rid % 10 == 0:
        print(f"  run {rid}/{N_RUNS}")

# Fig4 leverage ratio needs worst-scenario finance too
def load_worst_run(rid):
    rd = MC / "worst" / f"run_{rid:02d}" / "baseline" / "worst"
    return load_finance_run(rd, None) if rd.exists() else None


base_fin_runs = [load_finance_run(run_path(rid), None) for rid in range(1, N_RUNS + 1)]
worst_fin_runs = [load_worst_run(rid) for rid in range(1, N_RUNS + 1)]

for group in ["owner", "renter"]:
    ratios = []
    cum_dmg_base = []
    cum_dmg_worst = []
    cum_loss_base = []
    cum_loss_worst = []
    for b, w in zip(base_fin_runs, worst_fin_runs):
        if b is None or w is None:
            continue
        hh_b = np.maximum(b[f"{group}_hh"].values, 1)
        hh_w = np.maximum(w[f"{group}_hh"].values, 1)
        cdB = np.cumsum(b[f"{group}_gross"].values / hh_b)
        clB = np.cumsum((b[f"{group}_gross"].values - b[f"{group}_payout"].values) / hh_b)
        cdW = np.cumsum(w[f"{group}_gross"].values / hh_w)
        clW = np.cumsum((w[f"{group}_gross"].values - w[f"{group}_payout"].values) / hh_w)
        ddg = cdW - cdB
        dls = clW - clB
        ratio = np.where(dls > 0, ddg / dls, np.nan)
        ratios.append(ratio)
        cum_dmg_base.append(cdB)
        cum_dmg_worst.append(cdW)
        cum_loss_base.append(clB)
        cum_loss_worst.append(clW)
    # Fig5 cumulative per-HH series (paper Figure 5 panels a/b)
    metrics[f"Fig5_{group}_cum_damage_baseline_USD"] = cum_dmg_base
    metrics[f"Fig5_{group}_cum_damage_noadapt_USD"] = cum_dmg_worst
    metrics[f"Fig5_{group}_cum_actual_loss_baseline_USD"] = cum_loss_base
    metrics[f"Fig5_{group}_cum_actual_loss_noadapt_USD"] = cum_loss_worst
    # Leverage ratio (paper Figure 5 panel c)
    metrics[f"Fig5_{group}_leverage_ratio"] = ratios

# ============================================================================
# Build the single wide-format table reported in Text S6 / Table S6.
def unit_for(key):
    """Return the unit string for a metric key (used for the `unit` column)."""
    k = key.lower()
    if "rate_pct" in k or "_pct" in k:
        return "percent"
    if "_usd" in k:
        return "$K/household"  # dollar values reported in thousands of USD
    if "leverage_ratio" in k:
        return "dimensionless"
    if "_tp_" in k or k.endswith("tp_floodprone") or k.endswith("tp_nonprone"):
        return "TP (0–1)"
    return ""

def value_scale(key):
    """Scale factor for display: dollar metrics are divided by 1000 → $K."""
    return 1e-3 if "_USD" in key else 1.0

# Long-format CSV with 4 figure groups: one row per (metric × year),
# columns = figure | metric | unit | year | median | q25 | q75.
# Dollar metrics are reported in $K/household (USD ÷ 1000) for readability.
# A blank row is inserted between figure groups for visual separation in Excel.
def figure_of(key):
    if key.startswith("Fig5"): return "Fig5"
    if key.startswith("Fig6"): return "Fig6"
    if key.startswith("Fig7"): return "Fig7"
    if key.startswith("Fig8"): return "Fig8"
    return ""

def short_name(key):
    """Strip the FigN_ prefix and trailing _USD to keep metric name readable."""
    s = key
    for prefix in ["Fig5_", "Fig6_", "Fig7_", "Fig8_"]:
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    if s.endswith("_USD"):
        s = s[:-4]
    return s

# Sort metrics by figure then by name, so each figure group is contiguous.
ordered_keys = sorted(metrics.keys(),
                       key=lambda k: (figure_of(k), short_name(k)))

rows = []
prev_fig = None
for key in ordered_keys:
    fig_id = figure_of(key)
    if prev_fig is not None and fig_id != prev_fig:
        # Blank separator row between figure groups
        rows.append({c: "" for c in
                     ["figure", "metric", "unit", "year", "median", "q25", "q75"]})
    prev_fig = fig_id
    arr = np.stack(metrics[key])  # (n_runs, n_years)
    scale = value_scale(key)
    u = unit_for(key)
    nm = short_name(key)
    for j, y in enumerate(YEARS):
        col = arr[:, j] * scale
        rows.append({
            "figure": fig_id,
            "metric": nm,
            "unit": u,
            "year": int(y),
            "median": round(float(np.nanmedian(col)), 3),
            "q25":    round(float(np.nanpercentile(col, 25)), 3),
            "q75":    round(float(np.nanpercentile(col, 75)), 3),
        })

out_df = pd.DataFrame(rows, columns=["figure", "metric", "unit", "year", "median", "q25", "q75"])
out_path = OUT_DIR / "Table_S_stochastic_variance.csv"
out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
print(f"\nSaved: {out_path}")
print(f"  data rows: {(out_df['year'] != '').sum()}  (26 metrics × 13 years)")
print(f"  total rows incl. separators: {len(out_df)}")

# Quick preview of 2023 row for each metric
print("\n=== 2023 sample (median, q25, q75) ===")
sample = out_df[out_df["year"] == 2023]
for _, r in sample.iterrows():
    print(f"  {r['figure']} | {r['metric']:42s} | {r['unit']:14s} | {r['median']:>10} [{r['q25']}–{r['q75']}]")
