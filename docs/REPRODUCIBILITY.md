# Reproducing the revised WRR simulation

These steps reproduce the simulation that used tract-area SFHA shares, not the earlier NSI-by-tenure run. The shipped inputs are synthetic household inventories, processed tract flood depths and area shares, fitted Bayesian models, and the fixed configuration. The inputs describe 52,141 households in 27 tracts; 6,485 start in the SFHA and 2,861 start with flood insurance. No survey workbook is needed.

## Run and verify from a fresh clone

Use a clean Python 3.14.0 environment and run the following from the repository root. The output directory must be new and outside the clone. This example runs the first draw for both scenarios across all 13 years (2011–2023).

```sh
python -m pip install -r requirements-benchmark.txt
python scripts/utilities/validate_sfha_baseline.py
python scripts/utilities/run_mc100_local.py --out-root ../wrr-check --start 1 --end 1
python scripts/utilities/verify_wrr_finance.py --mc-root ../wrr-check --start 1 --end 1
```

For the complete ensemble, use a different output directory and omit `--start` and `--end` on both commands (50 draws, two scenarios). The runner uses the same fixed initial states and event-depth seed for every draw. The decision seeds and posterior indices are recorded in [`wrr_mc50_finance_reference.csv`](../data/supplementary/wrr_mc50_finance_reference.csv).

The finance check compares all 13 annual tract finance CSVs against one archived checksum per run and scenario. It normalizes line endings; numeric formatting changes across library versions can still cause a strict mismatch. The reference is the archived tract-area WRR benchmark. This check covers simulation finance. See [checking reproduced WRR figures](WRR_FIGURE_CHECKS.md) for the separate comparison against the final figure exports and the limits of figure regeneration.

## Rebuild the initial states

```sh
python scripts/utilities/build_sfha_area_initialization.py --households config/households_for_abm_sfha.csv --area-shares config/sfha_area_share_by_tract.csv --output-dir ../wrr-initialization --replay
```

The published household CSV is the only agent inventory in the checkout. In
replay mode the builder preserves its base attributes and recalculates
`sfha_share`, `inside_SFHA`, and `has_FI` using the tract-area data and fixed
seeds. The manifest records the input and output checksums. For an independent
reconstruction from the original unassigned synthetic inventory, retrieve it
from the pinned Git commit recorded under `historical_unassigned_households`
in the manifest, pass it to `--households`, and omit `--replay`.

See the [input inventory](DATA_INVENTORY.md) for the roles of the retained and
retired data files.

## Annual ensemble medians and interquartile ranges

Supporting Information Text S7 points to
[`data/supplementary/Table_S_MC_variance.csv`](../data/supplementary/Table_S_MC_variance.csv).
It contains 455 entries across Figures 5, 6, 8, and 9, using the revised
50-draw ensemble. Reassemble and verify it from the five bundled aggregate
sources:

```sh
python scripts/utilities/build_wrr_mc_variance_table.py --check
python scripts/utilities/build_wrr_mc_variance_table.py --output ../Table_S_MC_variance.csv
```

Each row identifies the unit and how the series was calculated. Figure 6a–b reports cumulative
loss through each year divided by the initial tenure count. Figure 6c–f
uses current-year counts for per-household dollar amounts and tenure GUL
for the OOP and payout shares. The combined premium-plus-OOP quartiles
use within-draw totals; the stacked bars use separate medians. Figure 8
rates and Figure 6 shares are fractions, not percentages. The table does
not assign annual IQRs to the pooled exceedance curves (Figures 7 and 11),
the median-only tract map (Figure 10), or to the observed data in Figure 5.

The earlier seed list and variance table remain in Git history. Survey
workbooks for refitting are not shipped; re-estimating the fitted models
requires separately authorized calibration data.
