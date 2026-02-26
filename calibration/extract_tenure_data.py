"""
Extract Owner/Renter survey data from MG+NMG raw sheets.
Split by Q3: Q3=2 → renter, everything else → owner.

Steps:
  1.1  Create owner_variable / renter_variable sheets in data_ori.xlsx
  1.2  Create owner_cal / renter_cal sheets in cal_data.xlsx
  1.3  Fit Beta distributions for 5 psychological variables
  1.4  Compute action adoption rates
  1.5  Generate all plots for professor review

Usage:
    python extract_tenure_data.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import os
import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist
from scipy.stats import kstest
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

# ── Paths ──────────────────────────────────────────────────────────────
BASE = Path(r"C:\Users\wenyu\OneDrive - Lehigh University\Desktop\Lehigh\NSF-project\ABM\Calibration")
DATA_ORI = BASE / "Basyiean regression mode" / "data" / "data_ori.xlsx"
CAL_DATA = BASE / "Updating threat perception" / "cal_data.xlsx"
OUT_DIR  = BASE / "tenure_distributions"
OUT_DIR.mkdir(exist_ok=True)

# ── Column definitions (verified against Excel formulas) ───────────────
TP_ITEMS = [f"Q22_{i}" for i in range(1, 12)]           # 11 items
CP_ITEMS = ["Q24_1", "Q24_2", "Q25_1", "Q25_2",
            "Q25_4", "Q25_5", "Q25_7", "Q25_8"]          # 8 items
SP_ITEMS = ["Q25_3", "Q25_6", "Q25_9"]                   # 3 items
SC_ITEMS = [f"Q21_{i}" for i in range(1, 7)]              # 6 items
PA_ITEMS = [f"Q21_{i}" for i in range(7, 16)]             # 9 items

ACTION_MAP = {"FI": "Q27", "EH": "Q29", "BP": "Q31", "RL": "Q33"}

Q15_YEAR_MAP = {1: 1, 2: 3, 3: 8, 4: 15, 5: 25, 6: 35, 7: 45}


def compute_variables(raw_df):
    """Compute perception means and action variables from raw survey data."""
    out = pd.DataFrame()
    out["FI"] = raw_df[ACTION_MAP["FI"]]
    out["EH"] = raw_df[ACTION_MAP["EH"]]
    out["BP"] = raw_df[ACTION_MAP["BP"]]
    out["RL"] = raw_df[ACTION_MAP["RL"]]
    out["TP_mean"] = raw_df[TP_ITEMS].mean(axis=1)
    out["CP_mean"] = raw_df[CP_ITEMS].mean(axis=1)
    out["SP_mean"] = raw_df[SP_ITEMS].mean(axis=1)
    return out


def compute_cal_variables(raw_df):
    """Compute calibration variables (Q15 + perceptions) from raw FE data."""
    out = pd.DataFrame()
    out["Q15"] = raw_df["Q15"]
    out["Q15_year"] = raw_df["Q15"].map(Q15_YEAR_MAP)
    out["SC_mean"] = raw_df[SC_ITEMS].mean(axis=1)
    out["PA_mean"] = raw_df[PA_ITEMS].mean(axis=1)
    out["TP_mean"] = raw_df[TP_ITEMS].mean(axis=1)
    out["CP_mean"] = raw_df[CP_ITEMS].mean(axis=1)
    out["SP_mean"] = raw_df[SP_ITEMS].mean(axis=1)
    return out


def split_by_tenure(df, q3_series):
    """Split dataframe by Q3: Q3==2 → renter, everything else → owner."""
    is_renter = q3_series == 2
    owner = df[~is_renter].reset_index(drop=True)
    renter = df[is_renter].reset_index(drop=True)
    return owner, renter


# ══════════════════════════════════════════════════════════════════════
# STEP 1.1 — Create owner_variable / renter_variable in data_ori.xlsx
# ══════════════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 1.1: Extract owner/renter variable sheets")
print("=" * 60)

# Read raw sheets
nmg_raw = pd.read_excel(DATA_ORI, sheet_name="NMG")
mg_raw  = pd.read_excel(DATA_ORI, sheet_name="MG")
print(f"NMG raw: {len(nmg_raw)} rows, MG raw: {len(mg_raw)} rows")

# Combine all respondents
all_raw = pd.concat([nmg_raw, mg_raw], ignore_index=True)
q3_all  = all_raw["Q3"]
print(f"Combined: {len(all_raw)} rows")
print(f"Q3 distribution: {q3_all.value_counts(dropna=False).to_dict()}")

# Compute variables from raw data
all_vars = compute_variables(all_raw)
owner_var, renter_var = split_by_tenure(all_vars, q3_all)

print(f"\nOwner:  {len(owner_var)} rows")
print(f"Renter: {len(renter_var)} rows")

# Add Source column (all Original — synthetic augmentation can be done later)
owner_var["Source"] = "Original"
renter_var["Source"] = "Original"

# Save to data_ori.xlsx as new sheets
print(f"\nWriting to {DATA_ORI}...")
with pd.ExcelWriter(DATA_ORI, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
    owner_var.to_excel(writer, sheet_name="owner_variable", index=False)
    renter_var.to_excel(writer, sheet_name="renter_variable", index=False)
print("  → Saved sheets: owner_variable, renter_variable")

# Verify
print("\n--- Owner variable summary ---")
print(owner_var[["FI", "EH", "BP", "RL", "TP_mean", "CP_mean", "SP_mean"]].describe().round(3))
print("\n--- Renter variable summary ---")
print(renter_var[["FI", "EH", "BP", "RL", "TP_mean", "CP_mean", "SP_mean"]].describe().round(3))


# ══════════════════════════════════════════════════════════════════════
# STEP 1.2 — Create owner_cal / renter_cal in cal_data.xlsx
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 1.2: Create TP decay calibration data by tenure")
print("=" * 60)

nmg_fe = pd.read_excel(CAL_DATA, sheet_name="NMG_FE")
mg_fe  = pd.read_excel(CAL_DATA, sheet_name="MG_FE")
print(f"NMG_FE: {len(nmg_fe)} rows, MG_FE: {len(mg_fe)} rows")

# Combine FE data
all_fe = pd.concat([nmg_fe, mg_fe], ignore_index=True)
q3_fe  = all_fe["Q3"]
print(f"Combined FE: {len(all_fe)} rows")
print(f"Q3 distribution: {q3_fe.value_counts(dropna=False).to_dict()}")

# Compute cal variables from raw FE data
all_cal = compute_cal_variables(all_fe)

# Drop rows with Q15_year NaN (Q15 value not in map)
before = len(all_cal)
all_cal = all_cal.dropna(subset=["Q15_year"])
q3_fe_clean = q3_fe.loc[all_cal.index]
print(f"After dropping unmapped Q15: {len(all_cal)} rows (dropped {before - len(all_cal)})")

owner_cal, renter_cal = split_by_tenure(all_cal.reset_index(drop=True),
                                         q3_fe_clean.reset_index(drop=True))
print(f"\nOwner cal:  {len(owner_cal)} rows")
print(f"Renter cal: {len(renter_cal)} rows")

print(f"\nWriting to {CAL_DATA}...")
with pd.ExcelWriter(CAL_DATA, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
    owner_cal.to_excel(writer, sheet_name="owner_cal", index=False)
    renter_cal.to_excel(writer, sheet_name="renter_cal", index=False)
print("  → Saved sheets: owner_cal, renter_cal")

print("\n--- Owner cal summary ---")
print(owner_cal.describe().round(3))
print("\n--- Renter cal summary ---")
print(renter_cal.describe().round(3))


# ══════════════════════════════════════════════════════════════════════
# STEP 1.3 — Fit Beta distributions for 5 psychological variables
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 1.3: Fit Beta distributions")
print("=" * 60)

PSYCH_VARS = ["TP_mean", "CP_mean", "SP_mean"]
# SC and PA are only in cal data, but we can also compute from the full raw
# For Beta fitting, use the full survey dataset (all_raw)
all_vars_full = pd.DataFrame()
all_vars_full["TP_mean"] = all_raw[TP_ITEMS].mean(axis=1)
all_vars_full["CP_mean"] = all_raw[CP_ITEMS].mean(axis=1)
all_vars_full["SP_mean"] = all_raw[SP_ITEMS].mean(axis=1)
all_vars_full["SC_mean"] = all_raw[SC_ITEMS].mean(axis=1)
all_vars_full["PA_mean"] = all_raw[PA_ITEMS].mean(axis=1)
all_vars_full["Q3"] = q3_all.values

PSYCH_ALL = ["TP_mean", "CP_mean", "SP_mean", "SC_mean", "PA_mean"]

owner_full = all_vars_full[all_vars_full["Q3"] != 2].copy()
renter_full = all_vars_full[all_vars_full["Q3"] == 2].copy()

# Normalize to 0-1: divide by 5 (Likert 1-5 → 0.2-1.0 range)
# Then clip to (epsilon, 1-epsilon) for Beta fitting
EPS = 1e-6

beta_params = {}
for group_name, group_df in [("owner", owner_full), ("renter", renter_full)]:
    beta_params[group_name] = {}
    print(f"\n--- {group_name.upper()} (n={len(group_df)}) ---")
    for var in PSYCH_ALL:
        vals = group_df[var].dropna().values / 5.0  # normalize to 0-1
        vals = np.clip(vals, EPS, 1 - EPS)
        a, b, loc, scale = beta_dist.fit(vals, floc=0, fscale=1)
        mean_val = a / (a + b)
        std_val = np.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))

        # KS test
        ks_stat, ks_p = kstest(vals, "beta", args=(a, b))

        beta_params[group_name][var] = {
            "alpha": a, "beta": b, "mean": mean_val, "std": std_val,
            "n": len(vals), "ks_stat": ks_stat, "ks_p": ks_p
        }
        short = var.replace("_mean", "")
        print(f"  {short:2s}: α={a:.4f}  β={b:.4f}  mean={mean_val:.4f}  "
              f"std={std_val:.4f}  n={len(vals)}  KS p={ks_p:.4f}")

# Build summary table
rows = []
for grp in ["owner", "renter"]:
    for var in PSYCH_ALL:
        p = beta_params[grp][var]
        rows.append({
            "Group": grp, "Variable": var.replace("_mean", ""),
            "n": p["n"], "alpha": round(p["alpha"], 6), "beta": round(p["beta"], 6),
            "mean": round(p["mean"], 4), "std": round(p["std"], 4),
            "KS_stat": round(p["ks_stat"], 4), "KS_p": round(p["ks_p"], 4),
        })
summary_df = pd.DataFrame(rows)
summary_df.to_csv(OUT_DIR / "beta_parameters_summary.csv", index=False, encoding="utf-8")
print(f"\n→ Beta parameters saved to {OUT_DIR / 'beta_parameters_summary.csv'}")


# ══════════════════════════════════════════════════════════════════════
# STEP 1.4 — Action adoption rates
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 1.4: Action adoption rates")
print("=" * 60)

# For actions, use the owner_var / renter_var from Step 1.1
# Adoption = action value is not NaN (respondent chose this action)
# Mean Likert score among adopters
action_stats = []
for group_name, group_df in [("owner", owner_var), ("renter", renter_var)]:
    print(f"\n--- {group_name.upper()} (n={len(group_df)}) ---")
    for act in ["FI", "EH", "BP", "RL"]:
        non_nan = group_df[act].dropna()
        n_total = len(group_df)
        n_respond = len(non_nan)
        adoption_rate = n_respond / n_total if n_total > 0 else 0
        mean_likert = non_nan.mean() if len(non_nan) > 0 else np.nan
        std_likert = non_nan.std() if len(non_nan) > 1 else np.nan

        action_stats.append({
            "Group": group_name, "Action": act,
            "n_total": n_total, "n_respond": n_respond,
            "response_rate": round(adoption_rate, 4),
            "mean_likert": round(mean_likert, 3) if not np.isnan(mean_likert) else np.nan,
            "std_likert": round(std_likert, 3) if not np.isnan(std_likert) else np.nan,
        })
        print(f"  {act}: responded={n_respond}/{n_total} ({adoption_rate:.1%}), "
              f"mean={mean_likert:.2f}" if not np.isnan(mean_likert) else f"  {act}: no data")

action_df = pd.DataFrame(action_stats)
action_df.to_csv(OUT_DIR / "action_adoption_rates.csv", index=False, encoding="utf-8")
print(f"\n→ Action rates saved to {OUT_DIR / 'action_adoption_rates.csv'}")


# ══════════════════════════════════════════════════════════════════════
# STEP 1.5 — Plots
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 1.5: Generate plots")
print("=" * 60)

# ── 1.5a: Beta distribution plots (5 vars × 2 groups) ────────────────
fig, axes = plt.subplots(5, 2, figsize=(14, 22))
x = np.linspace(0.001, 0.999, 500)

for i, var in enumerate(PSYCH_ALL):
    for j, (grp_name, grp_df) in enumerate([("owner", owner_full), ("renter", renter_full)]):
        ax = axes[i, j]
        vals = grp_df[var].dropna().values / 5.0
        vals = np.clip(vals, EPS, 1 - EPS)

        p = beta_params[grp_name][var]
        a, b = p["alpha"], p["beta"]

        # Histogram
        ax.hist(vals, bins=30, density=True, alpha=0.5, color="steelblue",
                edgecolor="white", label="Data")
        # Fitted Beta PDF
        y = beta_dist.pdf(x, a, b)
        ax.plot(x, y, "r-", lw=2, label=f"Beta({a:.2f}, {b:.2f})")
        # Mean line
        ax.axvline(p["mean"], color="darkred", ls="--", lw=1.5,
                   label=f"mean={p['mean']:.3f}")

        short = var.replace("_mean", "")
        ax.set_title(f"{short} — {grp_name} (n={p['n']}, KS p={p['ks_p']:.3f})",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("Normalized value (0–1)")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

plt.suptitle("Beta Distribution Fits: Owner vs Renter\n5 Psychological Variables",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
fig.savefig(OUT_DIR / "beta_distributions_owner_renter.png", dpi=200, bbox_inches="tight")
fig.savefig(OUT_DIR / "beta_distributions_owner_renter.pdf", bbox_inches="tight")
plt.close(fig)
print("  → Saved beta_distributions_owner_renter.png/pdf")

# ── 1.5b: Overlay comparison (same axes for owner vs renter) ─────────
fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))
colors = {"owner": "#2196F3", "renter": "#FF5722"}

for i, var in enumerate(PSYCH_ALL):
    ax = axes[i]
    for grp_name in ["owner", "renter"]:
        p = beta_params[grp_name][var]
        a, b = p["alpha"], p["beta"]
        y = beta_dist.pdf(x, a, b)
        ax.plot(x, y, color=colors[grp_name], lw=2.5,
                label=f"{grp_name} (α={a:.2f}, β={b:.2f})")
        ax.fill_between(x, y, alpha=0.15, color=colors[grp_name])
        ax.axvline(p["mean"], color=colors[grp_name], ls="--", lw=1, alpha=0.7)

    short = var.replace("_mean", "")
    ax.set_title(short, fontsize=13, fontweight="bold")
    ax.set_xlabel("Normalized (0–1)")
    if i == 0:
        ax.set_ylabel("Density")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

plt.suptitle("Owner vs Renter: Psychological Variable Distributions",
             fontsize=14, fontweight="bold")
plt.tight_layout()
fig.savefig(OUT_DIR / "overlay_comparison.png", dpi=200, bbox_inches="tight")
fig.savefig(OUT_DIR / "overlay_comparison.pdf", bbox_inches="tight")
plt.close(fig)
print("  → Saved overlay_comparison.png/pdf")

# ── 1.5c: Action adoption bar chart ──────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
actions = ["FI", "EH", "BP", "RL"]
owner_rates = []
renter_rates = []
owner_means = []
renter_means = []

for act in actions:
    o = action_df[(action_df["Group"] == "owner") & (action_df["Action"] == act)]
    r = action_df[(action_df["Group"] == "renter") & (action_df["Action"] == act)]
    owner_rates.append(o["response_rate"].values[0] if len(o) > 0 else 0)
    renter_rates.append(r["response_rate"].values[0] if len(r) > 0 else 0)
    owner_means.append(o["mean_likert"].values[0] if len(o) > 0 else 0)
    renter_means.append(r["mean_likert"].values[0] if len(r) > 0 else 0)

x_pos = np.arange(len(actions))
width = 0.35

bars1 = ax.bar(x_pos - width/2, owner_rates, width, label="Owner",
               color="#2196F3", alpha=0.8)
bars2 = ax.bar(x_pos + width/2, renter_rates, width, label="Renter",
               color="#FF5722", alpha=0.8)

# Add value labels
for bar, rate in zip(bars1, owner_rates):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{rate:.1%}", ha="center", va="bottom", fontsize=9)
for bar, rate in zip(bars2, renter_rates):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{rate:.1%}", ha="center", va="bottom", fontsize=9)

ax.set_xlabel("Adaptation Action", fontsize=12)
ax.set_ylabel("Response Rate", fontsize=12)
ax.set_title("Action Response Rates: Owner vs Renter", fontsize=14, fontweight="bold")
ax.set_xticks(x_pos)
ax.set_xticklabels(["Flood Insurance\n(FI)", "Elevation/Hardening\n(EH)",
                     "Building Protection\n(BP)", "Relocation\n(RL)"])
ax.legend(fontsize=11)
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(0, max(max(owner_rates), max(renter_rates)) + 0.15)

plt.tight_layout()
fig.savefig(OUT_DIR / "action_rates_owner_renter.png", dpi=200, bbox_inches="tight")
fig.savefig(OUT_DIR / "action_rates_owner_renter.pdf", bbox_inches="tight")
plt.close(fig)
print("  → Saved action_rates_owner_renter.png/pdf")

# ── 1.5d: Action mean Likert bar chart ───────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

bars1 = ax.bar(x_pos - width/2, owner_means, width, label="Owner",
               color="#2196F3", alpha=0.8)
bars2 = ax.bar(x_pos + width/2, renter_means, width, label="Renter",
               color="#FF5722", alpha=0.8)

for bar, m in zip(bars1, owner_means):
    if not np.isnan(m):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                f"{m:.2f}", ha="center", va="bottom", fontsize=9)
for bar, m in zip(bars2, renter_means):
    if not np.isnan(m):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                f"{m:.2f}", ha="center", va="bottom", fontsize=9)

ax.set_xlabel("Adaptation Action", fontsize=12)
ax.set_ylabel("Mean Likert Score (1–5)", fontsize=12)
ax.set_title("Mean Action Intensity: Owner vs Renter", fontsize=14, fontweight="bold")
ax.set_xticks(x_pos)
ax.set_xticklabels(["Flood Insurance\n(FI)", "Elevation/Hardening\n(EH)",
                     "Building Protection\n(BP)", "Relocation\n(RL)"])
ax.legend(fontsize=11)
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(0, 5.5)

plt.tight_layout()
fig.savefig(OUT_DIR / "action_means_owner_renter.png", dpi=200, bbox_inches="tight")
fig.savefig(OUT_DIR / "action_means_owner_renter.pdf", bbox_inches="tight")
plt.close(fig)
print("  → Saved action_means_owner_renter.png/pdf")


# ── Print PerceptionSampler-ready parameters ─────────────────────────
print("\n" + "=" * 60)
print("READY-TO-USE: PerceptionSampler Beta parameters")
print("=" * 60)
print("""
params = {
    'owner': {""")
for var in PSYCH_ALL:
    p = beta_params["owner"][var]
    short = var.replace("_mean", "")
    print(f"        '{short}': {{'alpha': {p['alpha']:.9f}, 'beta': {p['beta']:.9f}}},")
print("""    },
    'renter': {""")
for var in PSYCH_ALL:
    p = beta_params["renter"][var]
    short = var.replace("_mean", "")
    print(f"        '{short}': {{'alpha': {p['alpha']:.9f}, 'beta': {p['beta']:.9f}}},")
print("    }")
print("}")

print("\n✓ Phase 1 complete. Review plots in:", OUT_DIR)
