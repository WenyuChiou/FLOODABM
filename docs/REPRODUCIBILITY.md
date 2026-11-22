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

Earlier Monte Carlo seed and variance tables, and the survey workbooks for refitting, are not part of the revised distribution. Prior tracked copies remain in Git history. Re-estimating coefficients or threat-perception decay requires separately authorized calibration data and is not part of this benchmark.
