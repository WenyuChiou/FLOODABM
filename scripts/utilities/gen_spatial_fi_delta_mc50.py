# -*- coding: utf-8 -*-
"""
Generate tract-level FI take-up rate deltas (relative to pre-event year 2020)
aggregated across 50 Monte Carlo baseline runs.

Methodology — fixed-denominator:
    For each (tract, year, run), FI rate is computed as
        fi_rate = FI_count[year] / pop_2020[tract]
    i.e. the denominator is pinned to the 2020 (pre-event) tract population.
    This isolates behavioral changes in FI adoption from demographic shifts
    caused by RL relocation or BP buyouts, which would otherwise alter the
    per-tract household count between years.

Output: a single wide-format CSV with one row per tract and six data columns,
ready for direct import into ArcGIS / QGIS to color the Fig 9 spatial maps.

    tract_geoid,
    owner_delta_2021, owner_delta_2022, owner_delta_2023,
    renter_delta_2021, renter_delta_2022, renter_delta_2023,
    owner_fi_2020 ... owner_fi_2023,
    renter_fi_2020 ... renter_fi_2023
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

MC_ROOT = Path(r"C:\FLOODABM_mc50_v2")
N_RUNS = 50
BASE_YEAR = 2020
DELTA_YEARS = [2021, 2022, 2023]
ALL_YEARS = sorted(set([BASE_YEAR] + DELTA_YEARS))
GROUPS = ["owner", "renter"]

OUT_DIR = Path(r"C:\Users\wenyu\OneDrive - Lehigh University\Desktop\Lehigh"
               r"\NSF-project\ABM\paper\Figure\spatial_data")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def run_path(rid: int) -> Path:
    return MC_ROOT / "baseline" / f"run_{rid:02d}" / "baseline" / "baseline"


def load_per_tract(file_path: Path, group: str) -> pd.DataFrame:
    """Return DataFrame indexed by tract_geoid with columns [n, fi_count]
    for one year + one tenure group."""
    df = pd.read_csv(file_path, dtype={"tract_geoid": str})
    grp = df[df["group"] == group].copy()
    rows = []
    for t, sub in grp.groupby("tract_geoid"):
        n = len(sub)
        if n == 0:
            continue
        is_fi = (sub["action"] == "FI")
        if "POLICY_NAME" in sub.columns:
            has_pol = sub["POLICY_NAME"].notna() & (sub["POLICY_NAME"].astype(str).str.len() > 0)
            is_fi = is_fi | has_pol
        rows.append({"tract_geoid": str(t), "n": n, "fi_count": int(is_fi.sum())})
    out = pd.DataFrame(rows).set_index("tract_geoid")
    return out


print(f"Loading FI counts and tract populations from {N_RUNS} MC runs (fixed-denom)...")
collected: dict[tuple[str, int], list[pd.DataFrame]] = {}
n_loaded = 0
for rid in range(1, N_RUNS + 1):
    rd = run_path(rid)
    if not rd.exists():
        continue
    for y in ALL_YEARS:
        f = rd / "decisions" / f"decisions_mgmix_{y}.csv"
        if not f.exists():
            continue
        for g in GROUPS:
            collected.setdefault((g, y), []).append(load_per_tract(f, g))
    n_loaded += 1
print(f"  loaded {n_loaded} runs")


# Capture 2020 population per (group, run) for the fixed denominator
pop_2020_per_run: dict[str, list[pd.Series]] = {g: [] for g in GROUPS}
for g in GROUPS:
    for run_df in collected[(g, BASE_YEAR)]:
        pop_2020_per_run[g].append(run_df["n"])

tracts_ref = sorted(set().union(*[set(s.index) for s in pop_2020_per_run["renter"]]))
print(f"  tracts: {len(tracts_ref)}")

# Per (group, year): 50-run median of fixed-denom rate per tract
medians: dict[tuple[str, int], pd.Series] = {}
for g in GROUPS:
    for y in ALL_YEARS:
        rate_runs = []
        for run_idx, run_df in enumerate(collected[(g, y)]):
            fi_y = run_df["fi_count"].reindex(tracts_ref)
            pop_2020 = pop_2020_per_run[g][run_idx].reindex(tracts_ref)
            r = fi_y / pop_2020.replace(0, np.nan)
            rate_runs.append(r)
        df_rate = pd.concat(rate_runs, axis=1)
        medians[(g, y)] = df_rate.median(axis=1)


# Build wide-format output
out = pd.DataFrame({"tract_geoid": tracts_ref})
for g in GROUPS:
    base = medians[(g, BASE_YEAR)]
    for y in DELTA_YEARS:
        cur = medians[(g, y)]
        out[f"{g}_delta_{y}"] = (cur.values - base.values)

for g in GROUPS:
    out[f"{g}_fi_{BASE_YEAR}"] = medians[(g, BASE_YEAR)].values
    for y in DELTA_YEARS:
        out[f"{g}_fi_{y}"] = medians[(g, y)].values

out_wide = OUT_DIR / "Fig9_spatial_fi_delta_50run_median.csv"
out.to_csv(out_wide, index=False, encoding="utf-8-sig")
print(f"\nSaved wide CSV: {out_wide}")
print(f"  rows (tracts): {len(out)}")
print(f"  cols: {list(out.columns)}")

# Sanity prints — min/max delta per column
print("\n=== Delta summary (50-run median per tract, pp = percentage points) ===")
for col in [c for c in out.columns if "delta" in c]:
    v = out[col].values
    print(f"  {col:24s}: min={v.min()*100:+6.2f}pp  max={v.max()*100:+6.2f}pp  mean={v.mean()*100:+6.2f}pp")
