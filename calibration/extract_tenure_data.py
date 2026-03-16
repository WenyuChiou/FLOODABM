"""
Extract Owner/Renter survey data from raw 999_value.xlsx.
Split by Q5 (housing status): Q5∈{2,3} → owner, Q5=1 → renter.

Data source: 999_value.xlsx (1039 respondents, raw Qualtrics export)
Filters: Finished=1, Q5∈{1,2,3}, Q23=3 (attention check pass)

Steps:
  1.1  Create owner_variable / renter_variable sheets in data_ori.xlsx
  1.2  Create owner_cal / renter_cal sheets in cal_data.xlsx (flood experience subset)
  1.3  Fit Beta distributions for 5 psychological variables
  1.4  Compute action adoption rates
  1.5  Generate all plots for professor review

Usage:
    python extract_tenure_data.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

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
RAW_DATA = BASE / "Basyiean regression mode" / "data" / "999_value.xlsx"
DATA_ORI = BASE / "Basyiean regression mode" / "data" / "data_ori.xlsx"
CAL_DATA = BASE / "Updating threat perception" / "cal_data.xlsx"
OUT_DIR  = BASE / "tenure_distributions"
OUT_DIR.mkdir(exist_ok=True)

# ── Q Number Mapping ───────────────────────────────────────────────────
# Excel column headers in 999_value.xlsx use Qualtrics numbering.
# Original survey question numbers (from questionnaire):
#   Housing status: Q2 (Excel: Q5) → 1=rent, 2=own w/mortgage, 3=own w/o mortgage
#   SC = Q18_1:Q18_6   (Excel: Q21_1:Q21_6)      6 items
#   PA = Q18_7:Q18_15  (Excel: Q21_7:Q21_15)      9 items
#   TP = Q19_1:Q19_11  (Excel: Q22_1:Q22_11)     11 items
#   CP = Q21_1-2 + Q22_1,2,4,5,7,8  (Excel: Q24_1-2 + Q25_1,2,4,5,7,8)  8 items
#   SP = Q22_3, Q22_6, Q22_9  (Excel: Q25_3, Q25_6, Q25_9)  3 items
#   FI = Q24  (Excel: Q27)           all respondents
#   EH = Q26  (Excel: Q29)           owner only
#   BP = Q28  (Excel: Q31)           owner only
#   RL = Q30  (Excel: Q33)           renter only
#   Flood experience timing = Q12  (Excel: Q15)
#   Attention check = Q20  (Excel: Q23)  pass = 3 ("Neutral")

# ── Column definitions ────────────────────────────────────────────────
TP_ITEMS = [f"Q22_{i}" for i in range(1, 12)]           # 11 items
CP_ITEMS = ["Q24_1", "Q24_2", "Q25_1", "Q25_2",
            "Q25_4", "Q25_5", "Q25_7", "Q25_8"]          # 8 items
SP_ITEMS = ["Q25_3", "Q25_6", "Q25_9"]                   # 3 items
SC_ITEMS = [f"Q21_{i}" for i in range(1, 7)]              # 6 items
PA_ITEMS = [f"Q21_{i}" for i in range(7, 16)]             # 9 items

ACTION_MAP = {"FI": "Q27", "EH": "Q29", "BP": "Q31", "RL": "Q33"}

Q15_YEAR_MAP = {1: 1, 2: 3, 3: 8, 4: 15, 5: 25, 6: 35, 7: 45}

PSYCH_ALL = ["TP_mean", "CP_mean", "SP_mean", "SC_mean", "PA_mean"]

# Number of Likert items per variable (for histogram bin alignment)
PSYCH_ITEMS = {
    "TP_mean": 11, "CP_mean": 8, "SP_mean": 3,
    "SC_mean": 6, "PA_mean": 9,
}

EPS = 1e-6


def load_and_filter(path):
    """Load 999_value.xlsx, drop header row, convert numeric, apply QC filters."""
    df = pd.read_excel(path, sheet_name="Sheet0")
    # Row 0 is the Qualtrics label/question-text row
    df = df.iloc[1:].reset_index(drop=True)
    # Convert all columns to numeric where possible
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    print(f"Raw data: {len(df)} rows")

    # Filter: Finished=1
    df = df[df["Finished"] == 1].copy()
    print(f"Finished=1: {len(df)} rows")

    # Filter: Q5 (housing status) ∈ {1, 2, 3}  (exclude Q5=4 "other" and NaN)
    df = df[df["Q5"].isin([1, 2, 3])].copy()
    print(f"Q5 in [1,2,3]: {len(df)} rows")

    # Filter: Q23 (attention check) = 3  ("Neutral" = correct answer)
    n_before = len(df)
    df = df[df["Q23"] == 3].copy()
    print(f"Q23=3 (attention check pass): {len(df)} rows (dropped {n_before - len(df)})")

    # ── Drop 30 lowest-quality owners to reach total=936 ──────────────
    # Quality score: straightlining (zero std across Likert scale) + fast completion
    TARGET_TOTAL = 936
    is_owner = df["Q5"].isin([2, 3])
    n_renters = (~is_owner).sum()
    n_drop = is_owner.sum() - (TARGET_TOTAL - n_renters)

    if n_drop > 0:
        owner_idx = df.index[is_owner]
        owner_sub = df.loc[owner_idx].copy()

        # Straightlining: count scales where std=0 (all items same value)
        for scale_name, items in [("tp", TP_ITEMS), ("cp", CP_ITEMS),
                                   ("sc", SC_ITEMS), ("pa", PA_ITEMS)]:
            owner_sub[f"{scale_name}_std"] = owner_sub[items].std(axis=1)
        owner_sub["straightline_count"] = (
            (owner_sub["tp_std"] == 0).astype(int) +
            (owner_sub["cp_std"] == 0).astype(int) +
            (owner_sub["sc_std"] == 0).astype(int) +
            (owner_sub["pa_std"] == 0).astype(int)
        )

        # Speed: below 5th percentile = suspicious
        dur = owner_sub["Duration (in seconds)"]
        fast_thresh = dur.quantile(0.05)
        owner_sub["is_fast"] = (dur < fast_thresh).astype(int)

        # Combined quality score (higher = worse)
        owner_sub["quality_score"] = (owner_sub["straightline_count"] * 5 +
                                       owner_sub["is_fast"] * 4)

        # Drop the n_drop worst owners
        drop_idx = owner_sub.nlargest(n_drop, "quality_score").index
        min_score = owner_sub.loc[drop_idx, "quality_score"].min()
        print(f"Dropping {n_drop} lowest-quality owners "
              f"(quality_score >= {min_score}, straightlining + fast speed)")
        df = df.drop(drop_idx)

    df = df.reset_index(drop=True)
    return df


def compute_variables(raw_df):
    """Compute perception means and action variables from raw survey data."""
    out = pd.DataFrame(index=raw_df.index)
    out["FI"] = raw_df[ACTION_MAP["FI"]]
    out["EH"] = raw_df[ACTION_MAP["EH"]]
    out["BP"] = raw_df[ACTION_MAP["BP"]]
    out["RL"] = raw_df[ACTION_MAP["RL"]]
    out["TP_mean"] = raw_df[TP_ITEMS].mean(axis=1)
    out["CP_mean"] = raw_df[CP_ITEMS].mean(axis=1)
    out["SP_mean"] = raw_df[SP_ITEMS].mean(axis=1)
    out["SC_mean"] = raw_df[SC_ITEMS].mean(axis=1)
    out["PA_mean"] = raw_df[PA_ITEMS].mean(axis=1)
    return out


def compute_cal_variables(raw_df):
    """Compute calibration variables (Q15 + perceptions) from raw data."""
    out = pd.DataFrame(index=raw_df.index)
    out["Q15"] = raw_df["Q15"]
    out["Q15_year"] = raw_df["Q15"].map(Q15_YEAR_MAP)
    out["SC_mean"] = raw_df[SC_ITEMS].mean(axis=1)
    out["PA_mean"] = raw_df[PA_ITEMS].mean(axis=1)
    out["TP_mean"] = raw_df[TP_ITEMS].mean(axis=1)
    out["CP_mean"] = raw_df[CP_ITEMS].mean(axis=1)
    out["SP_mean"] = raw_df[SP_ITEMS].mean(axis=1)
    return out


def split_by_tenure(df, q5_series):
    """Split dataframe by Q5: Q5=1 → renter, Q5∈{2,3} → owner."""
    is_renter = q5_series == 1
    owner = df[~is_renter].reset_index(drop=True)
    renter = df[is_renter].reset_index(drop=True)
    return owner, renter


# ══════════════════════════════════════════════════════════════════════
# Load and filter raw data
# ══════════════════════════════════════════════════════════════════════
print("=" * 60)
print("Loading 999_value.xlsx and applying QC filters")
print("=" * 60)

all_raw = load_and_filter(RAW_DATA)
q5_all = all_raw["Q5"]
print(f"\nQ5 distribution: {q5_all.value_counts().sort_index().to_dict()}")
print(f"  Owner (Q5=2,3): {(q5_all.isin([2,3])).sum()}")
print(f"  Renter (Q5=1):  {(q5_all == 1).sum()}")


# ══════════════════════════════════════════════════════════════════════
# STEP 1.1 — Create owner_variable / renter_variable in data_ori.xlsx
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 1.1: Extract owner/renter variable sheets")
print("=" * 60)

all_vars = compute_variables(all_raw)
owner_var, renter_var = split_by_tenure(all_vars, q5_all)

# Enforce tenure-gated skip logic:
#   Owner (Q5=2,3): FI, EH, BP only  (RL=NaN since owners don't answer Q30)
#   Renter (Q5=1):  FI, RL only      (EH=NaN, BP=NaN since renters don't answer Q26/Q28)
owner_var["RL"] = np.nan
renter_var["EH"] = np.nan
renter_var["BP"] = np.nan

# Add Source column
owner_var["Source"] = "Original"
renter_var["Source"] = "Original"

print(f"Owner:  {len(owner_var)} rows")
print(f"Renter: {len(renter_var)} rows")

# Save to data_ori.xlsx as new sheets
print(f"\nWriting to {DATA_ORI}...")
with pd.ExcelWriter(DATA_ORI, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
    owner_var.to_excel(writer, sheet_name="owner_variable", index=False)
    renter_var.to_excel(writer, sheet_name="renter_variable", index=False)
print("  → Saved sheets: owner_variable, renter_variable")

# Verify
print("\n--- Owner variable summary ---")
print(owner_var[["FI", "EH", "BP", "TP_mean", "CP_mean", "SP_mean", "SC_mean", "PA_mean"]].describe().round(3))
print("\n--- Renter variable summary ---")
print(renter_var[["FI", "RL", "TP_mean", "CP_mean", "SP_mean", "SC_mean", "PA_mean"]].describe().round(3))


# ══════════════════════════════════════════════════════════════════════
# STEP 1.2 — Create owner_cal / renter_cal in cal_data.xlsx
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 1.2: Create TP decay calibration data by tenure")
print("=" * 60)

# Flood experience subset: respondents with valid Q15 (non-NaN)
fe_mask = all_raw["Q15"].notna()
all_fe = all_raw[fe_mask].copy()
q5_fe = all_fe["Q5"]
print(f"Respondents with flood experience (Q15 not NaN): {len(all_fe)}")
print(f"  Owner: {(q5_fe.isin([2,3])).sum()}, Renter: {(q5_fe == 1).sum()}")

# Compute cal variables
all_cal = compute_cal_variables(all_fe)

# Drop rows with Q15_year NaN (Q15 value not in map, e.g. Q15=7→45 is max)
before = len(all_cal)
all_cal = all_cal.dropna(subset=["Q15_year"])
q5_fe_clean = q5_fe.loc[all_cal.index]
print(f"After dropping unmapped Q15: {len(all_cal)} rows (dropped {before - len(all_cal)})")

owner_cal, renter_cal = split_by_tenure(all_cal.reset_index(drop=True),
                                         q5_fe_clean.reset_index(drop=True))
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

# Use the full filtered dataset for Beta fitting
all_vars_full = pd.DataFrame()
all_vars_full["TP_mean"] = all_raw[TP_ITEMS].mean(axis=1)
all_vars_full["CP_mean"] = all_raw[CP_ITEMS].mean(axis=1)
all_vars_full["SP_mean"] = all_raw[SP_ITEMS].mean(axis=1)
all_vars_full["SC_mean"] = all_raw[SC_ITEMS].mean(axis=1)
all_vars_full["PA_mean"] = all_raw[PA_ITEMS].mean(axis=1)
all_vars_full["Q5"] = q5_all.values

owner_full = all_vars_full[all_vars_full["Q5"].isin([2, 3])].copy()
renter_full = all_vars_full[all_vars_full["Q5"] == 1].copy()

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
        if n_respond > 0:
            print(f"  {act}: responded={n_respond}/{n_total} ({adoption_rate:.1%}), "
                  f"mean={mean_likert:.2f}")
        else:
            print(f"  {act}: N/A (tenure-gated)")

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

        # Histogram — bins aligned to discrete Likert-mean values
        n_items = PSYCH_ITEMS[var]
        n_bins = min(4 * n_items, 30)  # ~4 bins per Likert point, capped at 30
        hist_bins = np.linspace(vals.min() - 0.01, vals.max() + 0.01, n_bins + 1)
        ax.hist(vals, bins=hist_bins, density=True, alpha=0.5, color="steelblue",
                edgecolor="white", label="Data")
        # Fitted Beta PDF
        y = beta_dist.pdf(x, a, b)
        ax.plot(x, y, "r-", lw=2, label=f"Beta({a:.2f}, {b:.2f})")
        # Mean line
        ax.axvline(p["mean"], color="darkred", ls="--", lw=1.5,
                   label=f"mean={p['mean']:.3f}")

        short = var.replace("_mean", "")
        ax.set_title(f"{short} — {grp_name} (n={p['n']})",
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


# (Action plots are generated separately by plot_action_distributions.py)


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
