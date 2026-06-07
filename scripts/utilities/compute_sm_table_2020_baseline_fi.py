# -*- coding: utf-8 -*-
"""
SM Table: 2020 pre-event FI take-up rate and post-event Δ by tract (50-run MC median).

Reads the canonical Fig 9 spatial CSV (Fig9_spatial_fi_delta_50run_median.csv),
which uses the fixed 2020 tract population as the denominator. This keeps Table
S7 consistent with the Fig 9 spatial maps.

Columns:
    tract_geoid, flood_prone,
    fi_share_owner_2020, fi_share_renter_2020,
    delta_owner_2021, delta_renter_2021,
    delta_owner_2022, delta_renter_2022,
    delta_owner_2023, delta_renter_2023
All Δ in percentage points (pp). Sorted by flood_prone desc, then 2020 owner share desc.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

OUT = Path(__file__).resolve().parent.parent.parent / "outputs" / "sm_tables"
OUT.mkdir(parents=True, exist_ok=True)

FIG9_CSV = Path(r"C:\Users\wenyu\OneDrive - Lehigh University\Desktop\Lehigh"
                r"\NSF-project\ABM\paper\Figure\spatial_data"
                r"\Fig9_spatial_fi_delta_50run_median.csv")

# flood_prone classification (from baseline run)
FP = pd.read_csv(
    r"C:\Users\wenyu\OneDrive - Lehigh University\Desktop\Lehigh"
    r"\NSF-project\ABM\paper\draft\mg_sensitivity\FLOODABM"
    r"\outputs\baseline\baseline\flood_impact_by_tract.csv",
    dtype={"tract_geoid": str},
)[["tract_geoid", "flood_prone"]]

# Read the canonical Fig 9 spatial CSV (already fixed-denom, 50-run median)
src = pd.read_csv(FIG9_CSV, dtype={"tract_geoid": str})

# Reshape into the column names the rest of this script expects
wide = pd.DataFrame({"tract_geoid": src["tract_geoid"]})
for tenure in ["owner", "renter"]:
    wide[f"{tenure}_share_FI_2020"] = src[f"{tenure}_fi_2020"]
    for y in [2021, 2022, 2023]:
        wide[f"{tenure}_share_FI_{y}"] = src[f"{tenure}_fi_{y}"]
        # Δ in percentage points (delta in source CSV is fraction, so * 100)
        wide[f"delta_{tenure}_{y}"] = src[f"{tenure}_delta_{y}"] * 100

print(f"Loaded {len(wide)} tracts from {FIG9_CSV.name}")

# Add baseline class (Low/Mid/High) for direct cross-reference with Figure 9
def _baseline_class(share: float) -> str:
    if share < 0.30:
        return "Low"
    if share < 0.70:
        return "Mid"
    return "High"

wide["owner_baseline_class"] = wide["owner_share_FI_2020"].apply(_baseline_class)
wide["renter_baseline_class"] = wide["renter_share_FI_2020"].apply(_baseline_class)

# Attach flood_prone
out = wide.merge(FP, on="tract_geoid", how="left")

# Tidy column order
cols = ["tract_geoid", "flood_prone",
        "owner_share_FI_2020", "owner_baseline_class",
        "renter_share_FI_2020", "renter_baseline_class",
        "delta_owner_2021", "delta_renter_2021",
        "delta_owner_2022", "delta_renter_2022",
        "delta_owner_2023", "delta_renter_2023"]
out = out[cols]

# Sort: flood_prone desc, then 2020 owner share desc
out = out.sort_values(["flood_prone", "owner_share_FI_2020"],
                      ascending=[False, False]).reset_index(drop=True)

# Round for display; flag saturation (>=99.5%) with "~100"
disp = out.copy()

def _fmt_share(v: float) -> str:
    pct = v * 100
    if pct >= 99.5:
        return "~100"
    return f"{pct:.1f}"

disp["owner_share_FI_2020"] = disp["owner_share_FI_2020"].apply(_fmt_share)
disp["renter_share_FI_2020"] = disp["renter_share_FI_2020"].apply(_fmt_share)
for c in ["delta_owner_2021", "delta_renter_2021",
          "delta_owner_2022", "delta_renter_2022",
          "delta_owner_2023", "delta_renter_2023"]:
    disp[c] = disp[c].round(1)

# Rename for readability in the CSV
disp = disp.rename(columns={
    "owner_share_FI_2020": "Homeowner_FI_2020_%",
    "owner_baseline_class": "Homeowner_baseline_class",
    "renter_share_FI_2020": "Renter_FI_2020_%",
    "renter_baseline_class": "Renter_baseline_class",
    "delta_owner_2021": "dH_2021_pp",
    "delta_renter_2021": "dR_2021_pp",
    "delta_owner_2022": "dH_2022_pp",
    "delta_renter_2022": "dR_2022_pp",
    "delta_owner_2023": "dH_2023_pp",
    "delta_renter_2023": "dR_2023_pp",
})

out_csv = OUT / "TableSX_tract_FI_2020_baseline_and_delta.csv"
disp.to_csv(out_csv, index=False, encoding="utf-8-sig")
print(f"Saved: {out_csv}")

print()
print("FLOOD-PRONE (N=13):")
print(disp[disp["flood_prone"] == 1].to_string(index=False))
print()
print("NON-PRONE (N=14):")
print(disp[disp["flood_prone"] == 0].to_string(index=False))
