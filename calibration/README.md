# Calibration Pipeline: Owner/Renter Behavioral Parameters

This directory contains the complete calibration pipeline for the FLOODABM behavioral parameters,
organized by housing tenure (homeowner vs. renter).

## Directory Structure

```
calibration/
├── data/                           # Source data
│   ├── data_ori.xlsx               # Raw survey (NMG/MG sheets + owner/renter variable sheets)
│   ├── data.xlsx                   # Processed survey data (Bayesian engine reads this)
│   └── cal_data.xlsx               # TP decay calibration data (owner_cal, renter_cal sheets)
│
├── extract_tenure_data.py          # Phase 1: Extract & fit distributions from survey
├── audit_anomalies.py              # Data quality audit script
│
├── bayesian_engine/                # Phase 2: Bayesian Beta Regression
│   ├── main.py                     # Entry point: run all group × action models
│   ├── model.py                    # NumPyro Beta regression model definition
│   ├── config.py                   # GROUPS = [("owner_variable","owner"), ("renter_variable","renter")]
│   ├── data.py                     # Data loading from Excel sheets
│   ├── calibration.py              # Post-hoc probability calibration (Platt/isotonic/beta)
│   ├── evaluation.py               # Out-of-fold evaluation
│   ├── metrics.py                  # Brier score, ECE, calibration metrics
│   ├── strategy.py                 # Training strategy patterns
│   └── utils.py                    # Utility functions
│
├── phase1_tenure_distributions/    # Phase 1 outputs
│   ├── beta_parameters_summary.csv # Alpha/Beta params for 5 vars × 2 groups (+ KS test)
│   ├── action_adoption_rates.csv   # FI/EH/BP/RL adoption rates by tenure
│   ├── beta_distributions_owner_renter.png/pdf  # 10-panel histograms + Beta fits
│   ├── overlay_comparison.png/pdf               # Owner vs Renter overlay
│   ├── action_rates_owner_renter.png/pdf        # Adoption rate bar charts
│   ├── action_means_owner_renter.png/pdf        # Mean Likert score comparison
│   ├── Phase1_Owner_Renter_Report.docx          # Comprehensive Word report
│   └── generate_report.py                       # Script to regenerate report
│
└── phase2_bayesian_regression/     # Phase 2 outputs
    ├── all_results_summary_v2.xlsx # All 8 models: coefficients, convergence, metrics
    ├── best_calibrators.json       # Selected calibration method per (group, action)
    ├── best_calibrators_v2.json    # Updated calibrator selection
    ├── Validation.docx             # Validation report
    ├── SUPPLEMENT_bundle.xlsx      # Supplementary data for paper
    ├── owner_group/                # Per-action results for owners
    │   └── {BP,EH,FI,RL}/         # prob_metrics.xlsx per action
    └── renter_group/               # Per-action results for renters
        └── {BP,EH,FI,RL}/         # prob_metrics.xlsx per action
```

## Pipeline Overview

### Phase 1: Survey Data Extraction & Distribution Fitting
**Script:** `extract_tenure_data.py`

1. Reads raw survey data from `data_ori.xlsx` (NMG + MG sheets)
2. Splits respondents by Q3 (housing tenure): Q3=2 → renter, else → owner
3. Computes 5 psychological variables:
   - TP (Threat Perception) = mean(Q22_1:Q22_11) / 5
   - CP (Coping Perception) = mean(Q24_1,Q24_2,Q25_1,Q25_2,Q25_4,Q25_5,Q25_7,Q25_8) / 5
   - SP (Social Perception) = mean(Q25_3,Q25_6,Q25_9) / 5
   - SC (Social Capital) = mean(Q21_1:Q21_6) / 5
   - PA (Place Attachment) = mean(Q21_7:Q21_15) / 5
4. Extracts action columns: FI=Q27, EH=Q29, BP=Q31, RL=Q33
5. Fits Beta(α,β) distributions via MLE
6. Generates plots and summary statistics

**Sample sizes:** Owner=694, Renter=243

### Phase 2: Bayesian Beta Regression
**Script:** `bayesian_engine/main.py`

1. Fits Beta regression: P(action) ~ sigmoid(β₀ + β_TP·TP + β_CP·CP + β_SP·SP)
2. Uses NumPyro NUTS sampler (4 chains, 4000 warmup, 1200 samples)
3. Trains 8 models: owner×{FI,EH,BP,RL} + renter×{FI,EH,BP,RL}
4. Applies post-hoc calibration (Platt scaling, temperature, isotonic, beta)
5. Outputs `.pkl` (full model) and `.npz` (weights only) files

**Models are saved to:** `../models/baseline/` and `../models/baseline_fast/`

### Phase 3: TP Decay Calibration (pending)
Uses `cal_data.xlsx` (owner_cal: n=170, renter_cal: n=43) to calibrate
the TP exponential decay parameters (α, β, τ₀, τ∞, k) per tenure group.

## Key Results

### Beta Distribution Parameters (Phase 1)
| Variable | Owner α | Owner β | Renter α | Renter β |
|----------|---------|---------|----------|----------|
| TP       | 6.123   | 4.480   | 4.765    | 3.364    |
| CP       | 5.430   | 3.333   | 5.236    | 3.155    |
| SP       | 2.666   | 2.014   | 2.142    | 1.474    |
| SC       | 3.368   | 1.155   | 3.996    | 1.576    |
| PA       | 3.575   | 1.850   | 2.762    | 1.428    |

### Bayesian Regression Coefficients (Phase 2, posterior means)
| Model      | β_TP  | β_CP  | β_SP  | β₀    |
|------------|-------|-------|-------|-------|
| owner_FI   | 3.05  | 0.42  | 0.13  | -2.87 |
| owner_EH   | 2.76  | -0.11 | 0.53  | -3.32 |
| owner_BP   | 2.88  | -0.62 | 0.85  | -3.56 |
| owner_RL   | 2.90  | -0.81 | -0.02 | -2.69 |
| renter_FI  | 2.90  | 1.52  | -0.39 | -3.38 |
| renter_EH  | 2.36  | 1.06  | 2.96  | -5.55 |
| renter_BP  | 2.71  | -0.96 | 1.94  | -3.85 |
| renter_RL  | 3.10  | 0.94  | -0.69 | -3.29 |

All 8 models: 0 divergences, R-hat = 1.00, ESS > 400.
