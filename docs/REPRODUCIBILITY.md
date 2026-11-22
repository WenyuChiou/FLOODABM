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

The finance check compares all 13 annual tract finance CSVs against one archived checksum per run and scenario. It normalizes line endings; numeric formatting changes across library versions can still cause a strict mismatch. The reference is the archived tract-area WRR benchmark. The check covers simulation finance, not spatial layouts, manuscript formatting, all historical sensitivity runs, or figures supplied by separate tools.

## Rebuild the initial states

```sh
python scripts/utilities/build_sfha_area_initialization.py --households config/households_for_abm.csv --area-shares config/sfha_area_share_by_tract.csv --output-dir ../wrr-initialization
```

`households_for_abm.csv` is still needed here: it is the input synthetic inventory, not an alternate benchmark. The builder outputs an SFHA-tagged inventory, tract shares, assignment summaries, and the initialization seeds and hashes. The shipped `config/sfha_assignment_manifest.json` checks these inputs and outputs without using original machine paths.

Earlier Monte Carlo seed and variance tables, and the survey workbooks for refitting, are not part of the revised distribution. Prior tracked copies remain in Git history. Re-estimating coefficients or threat-perception decay requires separately authorized calibration data and is not part of this benchmark.
