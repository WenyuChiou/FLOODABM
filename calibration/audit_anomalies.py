import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
from scipy import stats

DATA1 = "C:/Users/wenyu/OneDrive - Lehigh University/Desktop/Lehigh/NSF-project/ABM/Calibration/Basyiean regression mode/data/data_ori.xlsx"
DATA2 = "C:/Users/wenyu/OneDrive - Lehigh University/Desktop/Lehigh/NSF-project/ABM/Calibration/Updating threat perception/cal_data.xlsx"

owner_var  = pd.read_excel(DATA1, sheet_name='owner_variable')
renter_var = pd.read_excel(DATA1, sheet_name='renter_variable')
owner_cal  = pd.read_excel(DATA2, sheet_name='owner_cal')
renter_cal = pd.read_excel(DATA2, sheet_name='renter_cal')

LIKERT_PERCEPTION = ['TP_mean', 'CP_mean', 'SP_mean']
LIKERT_ACTIONS    = ['FI', 'EH', 'BP', 'RL']
VALID_Q15_YEARS   = {1, 3, 8, 15, 25, 35, 45}

EXPECTED_SURVEY_TOTAL = 937
EXPECTED_CAL_TOTAL    = 213
THRESHOLD = 30
SKEW_THRESH = 2.0
KURT_THRESH = 7.0

SEP = "=" * 72

def section(title):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)

def subsection(title):
    print(f"\n  --- {title} ---")

# ===================================================================
# CHECK 9  Cross-check row counts
# ===================================================================
section("CHECK 9 - CROSS-CHECK: EXPECTED ROW COUNTS")
n_ov  = len(owner_var)
n_rv  = len(renter_var)
n_oc  = len(owner_cal)
n_rc  = len(renter_cal)
survey_total = n_ov + n_rv
cal_total    = n_oc + n_rc

print(f"  owner_variable  rows : {n_ov:>4}   (expected 694)")
print(f"  renter_variable rows : {n_rv:>4}   (expected 243)")
survey_ok = "OK" if survey_total == EXPECTED_SURVEY_TOTAL else "*** MISMATCH ***"
print(f"  Survey total         : {survey_total:>4}   (expected {EXPECTED_SURVEY_TOTAL}) -> {survey_ok}")
print(f"  owner_cal rows       : {n_oc:>4}   (expected 170)")
print(f"  renter_cal rows      : {n_rc:>4}   (expected  43)")
cal_ok = "OK" if cal_total == EXPECTED_CAL_TOTAL else "*** MISMATCH ***"
print(f"  Cal total            : {cal_total:>4}   (expected {EXPECTED_CAL_TOTAL}) -> {cal_ok}")

# ===================================================================
# CHECK 1  Missing values
# ===================================================================
section("CHECK 1 - MISSING VALUES (NaN count per column)")
for label, df in [("owner_variable", owner_var), ("renter_variable", renter_var),
                  ("owner_cal", owner_cal), ("renter_cal", renter_cal)]:
    subsection(label)
    miss = df.isnull().sum()
    miss_pct = (miss / len(df) * 100).round(2)
    tbl = pd.DataFrame({'NaN_count': miss, 'NaN_%': miss_pct})
    tbl = tbl[tbl.NaN_count > 0]
    if tbl.empty:
        print("    [PASS] No missing values.")
    else:
        for col, row in tbl.iterrows():
            flag = " *** WARNING >5%" if row['NaN_%'] > 5 else ""
            print(f"    {col:<20} NaN={int(row.NaN_count):>4}  ({row['NaN_%']:>5.1f}%){flag}")

# ===================================================================
# CHECK 2  Out-of-range values
# ===================================================================
section("CHECK 2 - OUT-OF-RANGE VALUES")

def check_range(df, label, cols, lo, hi, integer_only=False):
    subsection(f"{label}  (expected [{lo},{hi}]  integer_check={integer_only})")
    any_issue = False
    for col in cols:
        if col not in df.columns:
            print(f"    {col:<20} [SKIP] column not present")
            continue
        s = df[col].dropna()
        out = s[(s < lo) | (s > hi)]
        if integer_only:
            non_int = s[s != s.round(0)]
            if not non_int.empty:
                print(f"    {col:<20} non-integer values: {len(non_int)}")
                any_issue = True
        if not out.empty:
            print(f"    {col:<20} out-of-range [{lo},{hi}]: n={len(out)}  min={out.min():.4f}  max={out.max():.4f}")
            any_issue = True
    if not any_issue:
        print("    [PASS] All values in range.")

check_range(owner_var,  "owner_variable  - perception means",  LIKERT_PERCEPTION, 1, 5)
check_range(renter_var, "renter_variable - perception means",  LIKERT_PERCEPTION, 1, 5)
check_range(owner_cal,  "owner_cal       - all means", LIKERT_PERCEPTION + ['SC_mean','PA_mean'], 1, 5)
check_range(renter_cal, "renter_cal      - all means", LIKERT_PERCEPTION + ['SC_mean','PA_mean'], 1, 5)
check_range(owner_var,  "owner_variable  - action cols (FI EH BP RL)", LIKERT_ACTIONS, 1, 5, integer_only=True)
check_range(renter_var, "renter_variable - action cols (FI EH BP RL)", LIKERT_ACTIONS, 1, 5, integer_only=True)

# ===================================================================
# CHECK 3  Outliers (|z| > 2)
# ===================================================================
section("CHECK 3 - OUTLIERS  (values > 2 sigma from group mean)")

def check_outliers(df, label, cols):
    subsection(label)
    any_flag = False
    for col in cols:
        if col not in df.columns:
            continue
        s = df[col].dropna()
        if len(s) < 3:
            continue
        z = np.abs(stats.zscore(s))
        out_idx = s.index[z > 2]
        if not out_idx.empty:
            vals = s[out_idx]
            print(f"    {col:<20} outliers (|z|>2): n={len(out_idx)}")
            print(f"      values: {sorted(vals.tolist())[:15]}")
            any_flag = True
    if not any_flag:
        print("    [PASS] No outliers beyond 2 sigma.")

check_outliers(owner_var,  "owner_variable  - perception means", LIKERT_PERCEPTION)
check_outliers(renter_var, "renter_variable - perception means", LIKERT_PERCEPTION)
check_outliers(owner_cal,  "owner_cal       - all numeric", LIKERT_PERCEPTION + ['SC_mean','PA_mean'])
check_outliers(renter_cal, "renter_cal      - all numeric", LIKERT_PERCEPTION + ['SC_mean','PA_mean'])

# ===================================================================
# CHECK 4  Zero-variance columns
# ===================================================================
section("CHECK 4 - ZERO-VARIANCE COLUMNS")
for label, df in [("owner_variable", owner_var), ("renter_variable", renter_var),
                  ("owner_cal", owner_cal), ("renter_cal", renter_cal)]:
    subsection(label)
    num_cols = df.select_dtypes(include='number').columns
    zero_var = [c for c in num_cols if df[c].dropna().std() == 0]
    if zero_var:
        for c in zero_var:
            print(f"    *** ZERO VARIANCE: {c}  (all values = {df[c].dropna().iloc[0]})")
    else:
        print("    [PASS] No zero-variance columns.")

# ===================================================================
# CHECK 5  Duplicate rows
# ===================================================================
section("CHECK 5 - DUPLICATE ROWS (exact duplicates)")
for label, df in [("owner_variable", owner_var), ("renter_variable", renter_var),
                  ("owner_cal", owner_cal), ("renter_cal", renter_cal)]:
    subsection(label)
    dups = df.duplicated().sum()
    if dups > 0:
        print(f"    *** {dups} duplicate rows found!")
        dup_rows = df[df.duplicated(keep=False)]
        print(f"    Sample (first 6):")
        print(dup_rows.head(6).to_string(index=True))
    else:
        print("    [PASS] No exact duplicates.")

# ===================================================================
# CHECK 6  Q15_year valid mapping
# ===================================================================
section("CHECK 6 - Q15_year VALID VALUES  (expected: 1,3,8,15,25,35,45)")
for label, df in [("owner_cal", owner_cal), ("renter_cal", renter_cal)]:
    subsection(label)
    if 'Q15_year' not in df.columns:
        print("    [SKIP] column Q15_year not found")
        continue
    vals = df['Q15_year'].dropna()
    unique_vals = set(vals.unique())
    invalid = unique_vals - VALID_Q15_YEARS
    print(f"    Unique Q15_year values : {sorted(unique_vals)}")
    print(f"    Value counts:")
    vc = vals.value_counts().sort_index()
    for idx, cnt in vc.items():
        print(f"      Q15_year={idx:<4}  n={cnt}")
    if invalid:
        print(f"    *** INVALID values found: {sorted(invalid)}")
    else:
        print("    [PASS] All Q15_year values are valid.")
    if 'Q15' in df.columns:
        q15_vals = df['Q15'].dropna().unique()
        print(f"    Unique Q15 category values : {sorted(q15_vals)}")
        # Check for Q15/Q15_year cross-consistency (Q15 should be ordinal 1-5 mapping to year)
        cross = df[['Q15','Q15_year']].dropna()
        mapping = cross.groupby('Q15')['Q15_year'].unique()
        print("    Q15 -> Q15_year mapping (each Q15 category maps to):")
        for q, yrs in mapping.items():
            print(f"      Q15={q}  ->  Q15_year={sorted(yrs)}")

# ===================================================================
# CHECK 7  Sample size adequacy per column/bucket >= 30
# ===================================================================
section("CHECK 7 - SAMPLE SIZE ADEQUACY  (flag < 30 non-null)")

subsection("owner_variable - action columns")
for col in LIKERT_ACTIONS:
    if col not in owner_var.columns: continue
    n = owner_var[col].notna().sum()
    flag = " *** BELOW THRESHOLD" if n < THRESHOLD else ""
    print(f"    {col:<6}  n={n:>4}{flag}")

subsection("renter_variable - action columns  [HIGH RISK: total n=243]")
for col in LIKERT_ACTIONS:
    if col not in renter_var.columns: continue
    n = renter_var[col].notna().sum()
    flag = " *** BELOW THRESHOLD" if n < THRESHOLD else ""
    print(f"    {col:<6}  n={n:>4}{flag}")

subsection("owner_cal - by Q15_year bucket")
if 'Q15_year' in owner_cal.columns:
    grp = owner_cal.groupby('Q15_year').size()
    for yr, cnt in grp.items():
        flag = " *** BELOW THRESHOLD" if cnt < THRESHOLD else ""
        print(f"    Q15_year={yr:<4}  n={cnt:>4}{flag}")

subsection("renter_cal - by Q15_year bucket  [HIGH RISK: total n=43]")
if 'Q15_year' in renter_cal.columns:
    grp = renter_cal.groupby('Q15_year').size()
    for yr, cnt in grp.items():
        flag = " *** BELOW THRESHOLD" if cnt < THRESHOLD else ""
        print(f"    Q15_year={yr:<4}  n={cnt:>4}{flag}")

# ===================================================================
# CHECK 8  Distribution shape
# ===================================================================
section("CHECK 8 - DISTRIBUTION SHAPE  (skewness > 2 or kurtosis > 7 = severe)")

def check_distribution(df, label, cols):
    subsection(label)
    for col in cols:
        if col not in df.columns: continue
        s = df[col].dropna()
        if len(s) < 4: continue
        sk = float(stats.skew(s))
        ku = float(stats.kurtosis(s, fisher=True))
        flags = []
        if abs(sk) > SKEW_THRESH: flags.append(f"skewness={sk:.3f}")
        if abs(ku) > KURT_THRESH: flags.append(f"kurtosis={ku:.3f}")
        if flags:
            print(f"    *** {col:<20} {', '.join(flags)}")
        else:
            print(f"        {col:<20} skewness={sk:+.3f}  kurtosis={ku:+.3f}  [OK]")

check_distribution(owner_var,  "owner_variable  - perception + action", LIKERT_PERCEPTION + LIKERT_ACTIONS)
check_distribution(renter_var, "renter_variable - perception + action", LIKERT_PERCEPTION + LIKERT_ACTIONS)
check_distribution(owner_cal,  "owner_cal       - all numeric", ['SC_mean','PA_mean'] + LIKERT_PERCEPTION)
check_distribution(renter_cal, "renter_cal      - all numeric", ['SC_mean','PA_mean'] + LIKERT_PERCEPTION)

# ===================================================================
# SUMMARY DESCRIPTIVE STATS
# ===================================================================
section("SUMMARY - KEY DESCRIPTIVE STATISTICS")
for label, df in [("owner_variable", owner_var), ("renter_variable", renter_var),
                  ("owner_cal", owner_cal), ("renter_cal", renter_cal)]:
    subsection(label)
    num_df = df.select_dtypes(include='number')
    desc = num_df.describe().T[['count','mean','std','min','max']]
    desc['count'] = desc['count'].astype(int)
    print(desc.round(4).to_string())

print(f"\n{SEP}")
print("  AUDIT COMPLETE")
print(SEP)
