# -*- coding: utf-8 -*-
"""
SM Table S_Y: Non-FI renter pool trajectory, flood-prone vs non-prone.

Evidence for the selection-effect argument in Fig 7 Panel (d):
in flood-prone tracts, renters are progressively removed from the
non-FI pool via FI uptake, leaving a residual population. In non-prone
tracts the pool composition remains roughly constant.

Columns:
    year,
    prone_renter_N_mean, prone_nonFI_share_median (% of renters not holding FI),
    nonprone_renter_N_mean, nonprone_nonFI_share_median,
    prone_TP_renter_median, nonprone_TP_renter_median

Computed as 50-run MC median.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

MC_ROOT = Path(r"C:\FLOODABM_mc50_v2\baseline")
N_RUNS = 50
YEARS = list(range(2011, 2024))

FP = pd.read_csv(
    r"C:\Users\wenyu\OneDrive - Lehigh University\Desktop\Lehigh"
    r"\NSF-project\ABM\paper\draft\mg_sensitivity\FLOODABM"
    r"\outputs\baseline\baseline\flood_impact_by_tract.csv",
    dtype={"tract_geoid": str},
)[["tract_geoid", "flood_prone"]]

OUT = Path(__file__).resolve().parent.parent.parent / "outputs" / "sm_tables"
OUT.mkdir(parents=True, exist_ok=True)


def _load_run_year(rid: int, y: int) -> pd.DataFrame | None:
    fp = (MC_ROOT / f"run_{rid:02d}" / "baseline" / "baseline"
          / "decisions" / f"action_share_owner_renter_tract_{y}.csv")
    if not fp.exists():
        return None
    d = pd.read_csv(fp, dtype={"tract_geoid": str})
    d = d[["tract_geoid", "renter_N_total", "renter_n_FI",
           "renter_share_FI"]].copy()
    d["year"] = y
    d["run"] = rid
    # Non-FI pool: renters not holding FI this year
    d["nonFI_share"] = 1.0 - d["renter_share_FI"]
    return d


frames = []
for rid in range(1, N_RUNS + 1):
    for y in YEARS:
        r = _load_run_year(rid, y)
        if r is not None:
            frames.append(r)

print(f"Loaded {len(frames)} (run, year) frames")
df = pd.concat(frames, ignore_index=True)
df = df.merge(FP, on="tract_geoid", how="left")

# Weighted mean non-FI share per tract-group per year (weight by renter_N_total)
# We want the fraction of non-FI renters in the group, not the simple mean of per-tract shares.
def _pool_summary(sub: pd.DataFrame) -> pd.Series:
    # Per run, sum across tracts within group, then compute share.
    per_run = sub.groupby(["run", "year"]).apply(
        lambda g: pd.Series({
            "renter_N_group": g["renter_N_total"].sum(),
            "renter_nonFI_N": (g["renter_N_total"]
                               - g["renter_n_FI"]).sum(),
        }),
        include_groups=False,
    ).reset_index()
    per_run["nonFI_share"] = per_run["renter_nonFI_N"] / per_run["renter_N_group"]
    return per_run


prone_per_run = _pool_summary(df[df["flood_prone"] == 1])
nonp_per_run = _pool_summary(df[df["flood_prone"] == 0])

# Median across 50 runs per year
prone_median = prone_per_run.groupby("year")[["renter_N_group",
                                              "renter_nonFI_N",
                                              "nonFI_share"]].median()
nonp_median = nonp_per_run.groupby("year")[["renter_N_group",
                                            "renter_nonFI_N",
                                            "nonFI_share"]].median()

# Tract-level TP trajectory (population-average TP; supplementary evidence)
tp_frames = []
for rid in range(1, N_RUNS + 1):
    fp = MC_ROOT / f"run_{rid:02d}" / "baseline" / "baseline" / "tp_traj.csv"
    if not fp.exists():
        continue
    t = pd.read_csv(fp, dtype={"tract_geoid": str})
    t = t[t["phase"] == "after"].copy()
    t["run"] = rid
    tp_frames.append(t)
tp_df = pd.concat(tp_frames, ignore_index=True)
tp_df = tp_df.merge(FP, on="tract_geoid", how="left")

# Median TP per group per year (unweighted tract median, then median across runs)
tp_per_run = tp_df.groupby(["run", "year", "flood_prone"])[
    ["TP_renter"]].median().reset_index()
tp_prone = tp_per_run[tp_per_run["flood_prone"] == 1].groupby("year")[
    "TP_renter"].median()
tp_nonp = tp_per_run[tp_per_run["flood_prone"] == 0].groupby("year")[
    "TP_renter"].median()

# Build the summary table
out = pd.DataFrame({
    "year": YEARS,
    "prone_renter_N": prone_median["renter_N_group"].astype(int).values,
    "prone_nonFI_share_%": (prone_median["nonFI_share"] * 100).round(1).values,
    "prone_TP_renter": tp_prone.round(3).values,
    "nonprone_renter_N": nonp_median["renter_N_group"].astype(int).values,
    "nonprone_nonFI_share_%": (nonp_median["nonFI_share"] * 100).round(1).values,
    "nonprone_TP_renter": tp_nonp.round(3).values,
})

out_csv = OUT / "TableSY_non_fi_pool_trajectory.csv"
out.to_csv(out_csv, index=False, encoding="utf-8-sig")
print(f"\nSaved: {out_csv}\n")
print(out.to_string(index=False))
