# -*- coding: utf-8 -*-
"""
Generate tract-level population change (renter and owner) relative to the
2020 pre-event baseline, aggregated across 50 Monte Carlo baseline runs.

Purpose: supplementary evidence for Fig 9. Renter relocation (RL) drives
inflow/outflow between flood-prone and non-prone tracts; owner population
should remain near-stable (only small BP-driven attrition). Visualizing this
flow pattern justifies the fixed-2020-denominator methodology used in Fig 9
to isolate behavioral changes in FI take-up from demographic shifts.

Output: a single wide-format CSV with one row per tract:

    tract_geoid,
    owner_dpct_2021, owner_dpct_2022, owner_dpct_2023,    (% change)
    renter_dpct_2021, renter_dpct_2022, renter_dpct_2023, (% change)
    owner_n_2020, owner_n_2021, owner_n_2022, owner_n_2023,
    renter_n_2020, renter_n_2021, renter_n_2022, renter_n_2023,
    owner_dn_2021, owner_dn_2022, owner_dn_2023,          (Δ count)
    renter_dn_2021, renter_dn_2022, renter_dn_2023        (Δ count)

ArcGIS / QGIS layout:
    Use the *_dpct_* columns with an unclassed RdBu diverging color ramp
    centered at 0. Suggested range: −25% to +25%.
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


def load_pop_per_tract(file_path: Path, group: str) -> pd.Series:
    """Return pd.Series indexed by tract_geoid, value = household count for
    the given tenure group in this year/run."""
    df = pd.read_csv(file_path, dtype={"tract_geoid": str})
    grp = df[df["group"] == group]
    counts = grp.groupby("tract_geoid").size()
    counts.name = f"{group}_n"
    return counts


print(f"Loading per-tract population from {N_RUNS} MC runs...")
collected: dict[tuple[str, int], list[pd.Series]] = {}
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
            collected.setdefault((g, y), []).append(load_pop_per_tract(f, g))
    n_loaded += 1
print(f"  loaded {n_loaded} runs")


# Per (group, year): 50-run median per-tract count
tracts_ref = sorted(set().union(*[set(s.index) for s in collected[("renter", BASE_YEAR)]]))
print(f"  tracts: {len(tracts_ref)}")

medians: dict[tuple[str, int], pd.Series] = {}
for (g, y), lst in collected.items():
    df = pd.concat([s.reindex(tracts_ref).fillna(0) for s in lst], axis=1)
    medians[(g, y)] = df.median(axis=1)


# Build wide-format output
out = pd.DataFrame({"tract_geoid": tracts_ref})

# % change relative to 2020
for g in GROUPS:
    base = medians[(g, BASE_YEAR)].replace(0, np.nan)
    for y in DELTA_YEARS:
        cur = medians[(g, y)]
        out[f"{g}_dpct_{y}"] = ((cur - medians[(g, BASE_YEAR)]) / base * 100).values

# Raw counts for verification / SM table
for g in GROUPS:
    out[f"{g}_n_{BASE_YEAR}"] = medians[(g, BASE_YEAR)].astype(int).values
    for y in DELTA_YEARS:
        out[f"{g}_n_{y}"] = medians[(g, y)].astype(int).values

# Δ count
for g in GROUPS:
    for y in DELTA_YEARS:
        out[f"{g}_dn_{y}"] = (medians[(g, y)] - medians[(g, BASE_YEAR)]).astype(int).values


out_path = OUT_DIR / "Fig9_spatial_pop_delta_50run_median.csv"
out.to_csv(out_path, index=False, encoding="utf-8-sig")
print(f"\nSaved: {out_path}")
print(f"  rows: {len(out)} | cols: {list(out.columns)}")

print("\n=== Δ% summary (50-run median per tract) ===")
for col in [c for c in out.columns if "_dpct_" in c]:
    v = out[col].dropna().values
    print(f"  {col:24s}: min={v.min():+7.2f}%  max={v.max():+7.2f}%  "
          f"mean={v.mean():+5.2f}%")
