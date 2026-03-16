# -*- coding: utf-8 -*-
import sys; sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import pandas as pd
from pathlib import Path

OLD_TS = Path(r"C:\Users\wenyu\OneDrive - Lehigh University\Desktop\Lehigh\NSF-project\ABM\paper\validation\sim_vs_actual_standardized - 複製.csv")
OLD_SC = Path(r"C:\Users\wenyu\OneDrive - Lehigh University\Desktop\Lehigh\NSF-project\ABM\paper\validation\sim_vs_actual_standardized.csv")
COUNTIES = ["Essex", "Morris", "Passaic"]

scenarios = [
    ("Stochastic", "outputs/baseline/baseline"),
    ("θ=0.30", "outputs/thr030/baseline/baseline"),
    ("θ=0.35", "outputs/thr035/baseline/baseline"),
    ("θ=0.40", "outputs/thr040/baseline/baseline"),
    ("θ=0.45", "outputs/thr045/baseline/baseline"),
    ("θ=0.50", "outputs/thr050/baseline/baseline"),
]

print(f"{'Mechanism':<14} {'Essex R²':>9} {'Morris R²':>10} {'Passaic R²':>11} {'Pooled R²':>10}")
print("-" * 58)

for label, base in scenarios:
    val_csv = Path(base) / "visualization/validation/nfip_county_year_comparison.csv"
    if not val_csv.exists():
        print(f"{label:<14} — no validation data")
        continue
    new = pd.read_csv(val_csv)
    new["sim_avg"] = np.where(new["sim_claims"] > 0, new["sim_payout"] / new["sim_claims"], np.nan)

    sc = pd.read_csv(OLD_SC)
    sc = sc.merge(new[["county", "year", "sim_avg"]], on=["county", "year"], how="left")
    sc["simulated"] = sc["sim_avg"]
    for county in COUNTIES:
        mask = sc["county"] == county
        vals = sc.loc[mask, "simulated"]
        mu, sd = vals.mean(), vals.std()
        sc.loc[mask, "sim_z"] = (vals - mu) / sd if sd > 0 else 0

    r2s = {}
    for county in COUNTIES:
        c = sc[sc["county"] == county].dropna(subset=["sim_z", "act_z"])
        if len(c) >= 2:
            r = np.corrcoef(c["sim_z"], c["act_z"])[0, 1]
            r2s[county] = r ** 2
    pw = sc.dropna(subset=["sim_z", "act_z"])
    r_pool = np.corrcoef(pw["sim_z"], pw["act_z"])[0, 1]
    r2s["Pooled"] = r_pool ** 2

    print(f"{label:<14} {r2s.get('Essex',0):>9.2f} {r2s.get('Morris',0):>10.2f} {r2s.get('Passaic',0):>11.2f} {r2s.get('Pooled',0):>10.2f}")
