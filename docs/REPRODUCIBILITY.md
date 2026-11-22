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

## Recompute finance figure data from all 50 paired draws

Once the 50 paired draws complete, run:

```sh
python scripts/utilities/rebuild_wrr_finance_series.py --mc-root ../wrr-ensemble --out-dir ../wrr-recomputed-finance
```

The command verifies the 100 scenario/draw finance bundles. It then computes
the 39 simulated county-year payout bands and their three validation
correlations (Figure 5), all 260 Figure 6 rows, and both 2,400-point AEP
curves (Figures 7 and 11). Figure 5 uses the corrected county identifiers
and the bundled observed z-scores and selected-year flags; the command does
not refetch the underlying claims.
The 2011–2023 cumulative profiles use fixed initial tenure counts; the
annual dollar series use each run-year's tenure counts. The AEPs pool only
2012–2023 adaptation results, with 600 run-year values per tenure and
metric, divided by each run-year's household count before ranking. GUL
and actual loss are ranked separately. Differences at the same exceedance
probability are differences between two curve values; they are not paired
payouts from the same simulated run-year. Figure 11 divides the ranked
dollar values by the corresponding median tenure income.

The outputs are compared to the bundled figure data at every coordinate
before writing. Existing directories are not overwritten. This rebuild
verifies the plotted data; replaying the final Figure 6/7 layout still
requires the [vector components](WRR_FIGURE_CHECKS.md#replay-from-the-archived-pdf-components).

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
Figure 9's 2023 estimate for renters in flood-prone tracts uses 49 valid
draws: one draw has no post-action TP for a tract with a positive
decision-time renter weight. The published aggregation omits that draw
for this stratum; the other 51 Figure 9 estimates use all 50 draws.

The earlier seed list and variance table remain in Git history. Survey
workbooks for refitting are not shipped; re-estimating the fitted models
requires separately authorized calibration data.
