# -*- coding: utf-8 -*-
"""
Extract augmented calibration data for TP decay model.
Uses ALL survey respondents (not just flood-experienced).

Flood-experienced (Q15 not NaN): standard decay observations with time-since-flood
Non-flood (Q15 NaN): t→∞ observations where TP → TP_base

Output: augmented_cal_data.xlsx with sheets:
  - owner_augmented (all owners with has_flood flag)
  - renter_augmented (all renters with has_flood flag)
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import numpy as np
from pathlib import Path

_ABM_ROOT = Path(r"C:\Users\wenyu\OneDrive - Lehigh University\Desktop\Lehigh\NSF-project\ABM")
RAW_DATA = _ABM_ROOT / "Calibration" / "Basyiean regression mode" / "data" / "data_ori.xlsx"
CAL_DATA = _ABM_ROOT / "Calibration" / "Updating threat perception" / "cal_data.xlsx"
OUTPUT   = Path(__file__).resolve().parent / "augmented_cal_data.xlsx"


def compute_variables(df):
    """Compute SC, PA, TP, CP, SP from raw Q items (1-5 scale).

    Note: Excel column headers use re-numbered Q# from MG/NMG sheets.
    Original survey Q numbers (from 999_label.xlsx):
      SC = Q18_1:Q18_6   (Excel: Q21_1:Q21_6)
      PA = Q18_7:Q18_15  (Excel: Q21_7:Q21_15)
      TP = Q19_1:Q19_11  (Excel: Q22_1:Q22_11)
      CP = Q21_1-2 + Q22_1,2,4,5,7,8  (Excel: Q24_1-2 + Q25_1,2,4,5,7,8)
      SP = Q22_3, Q22_6, Q22_9  (Excel: Q25_3, Q25_6, Q25_9)
    """
    # SC = mean(Q21_1:Q21_6) — original Q18_1:Q18_6
    sc_cols = [f"Q21_{i}" for i in range(1, 7)]
    df["SC_mean"] = df[sc_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)

    # PA = mean(Q21_7:Q21_15) — original Q18_7:Q18_15
    pa_cols = [f"Q21_{i}" for i in range(7, 16)]
    df["PA_mean"] = df[pa_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)

    # TP = mean(Q22_1:Q22_11) — original Q19_1:Q19_11
    tp_cols = [f"Q22_{i}" for i in range(1, 12)]
    df["TP_mean"] = df[tp_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)

    # CP = mean(Q25_1:Q25_8) — original Q22_1,2,4,5,7,8 + Q21_1,2
    cp_cols = [f"Q25_{i}" for i in range(1, 9)]
    available_cp = [c for c in cp_cols if c in df.columns]
    df["CP_mean"] = df[available_cp].apply(pd.to_numeric, errors="coerce").mean(axis=1)

    # SP = mean of SP items (Q25_7:Q25_9) — original Q22_3, Q22_6, Q22_9
    sp_cols = [f"Q25_{i}" for i in range(7, 10)]
    available_sp = [c for c in sp_cols if c in df.columns]
    if available_sp:
        df["SP_mean"] = df[available_sp].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    else:
        df["SP_mean"] = np.nan

    return df


def parse_q15_year(q15_val):
    """Convert Q15 (Excel column, original Q12) answer to approximate years-since-flood."""
    if pd.isna(q15_val):
        return np.nan
    try:
        v = int(q15_val)
        # Q15 (original Q12) mapping: 1=<1yr, 2=1-5yr, 3=5-10yr, 4=10-15yr, 5=15-20yr, 6=20-25yr, 7=>25yr
        mapping = {1: 1, 2: 3, 3: 8, 4: 13, 5: 18, 6: 23, 7: 25}
        return mapping.get(v, 5)
    except (ValueError, TypeError):
        return 5


def main():
    print(f"Loading raw data from: {RAW_DATA}")

    # Load both sheets
    mg = pd.read_excel(RAW_DATA, sheet_name="MG")
    nmg = pd.read_excel(RAW_DATA, sheet_name="NMG")
    mg["_source"] = "MG"
    nmg["_source"] = "NMG"
    all_data = pd.concat([mg, nmg], ignore_index=True)
    print(f"Total respondents: {len(all_data)}")

    # Compute psychological variables
    all_data = compute_variables(all_data)

    # Q3: 1=owner, 2=renter
    all_data["tenure"] = all_data["Q3"].map({1: "owner", 2: "renter"})

    # Flood experience (Excel Q15 = original survey Q12)
    all_data["has_flood"] = all_data["Q15"].notna().astype(int)
    all_data["Q15_year"] = all_data["Q15"].apply(parse_q15_year)

    # Drop rows with missing key variables
    key_cols = ["SC_mean", "PA_mean", "TP_mean"]
    valid = all_data.dropna(subset=key_cols + ["tenure"])
    print(f"Valid respondents (SC, PA, TP not NaN): {len(valid)}")

    # Split by tenure
    results = {}
    for tenure in ["owner", "renter"]:
        grp = valid[valid["tenure"] == tenure].copy()
        flood = grp[grp["has_flood"] == 1]
        no_flood = grp[grp["has_flood"] == 0]
        print(f"\n{tenure.upper()}: total={len(grp)}, flood={len(flood)}, no_flood={len(no_flood)}")
        print(f"  TP (flood):    mean={flood['TP_mean'].mean():.4f}, std={flood['TP_mean'].std():.4f}")
        print(f"  TP (no flood): mean={no_flood['TP_mean'].mean():.4f}, std={no_flood['TP_mean'].std():.4f}")
        print(f"  SC: mean={grp['SC_mean'].mean():.4f}, PA: mean={grp['PA_mean'].mean():.4f}")

        out_df = grp[["SC_mean", "PA_mean", "TP_mean", "CP_mean", "SP_mean",
                       "has_flood", "Q15", "Q15_year", "_source"]].copy()
        out_df = out_df.reset_index(drop=True)
        results[f"{tenure}_augmented"] = out_df

    # Save
    with pd.ExcelWriter(OUTPUT, engine="openpyxl") as writer:
        for sheet_name, df in results.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"\nSaved: {OUTPUT}")
    print(f"  Sheets: {list(results.keys())}")

    # Also print TP_base estimates
    print("\n=== TP_base Estimates (mean of non-flood respondents, 1-5 scale) ===")
    for tenure in ["owner", "renter"]:
        grp = valid[valid["tenure"] == tenure]
        no_flood = grp[grp["has_flood"] == 0]
        tp_base_raw = no_flood["TP_mean"].mean()
        tp_base_01 = tp_base_raw / 5.0
        print(f"  {tenure}: TP_base = {tp_base_raw:.4f} (raw) = {tp_base_01:.4f} (0-1 scale), n={len(no_flood)}")


if __name__ == "__main__":
    main()
