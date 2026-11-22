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
verifies the plotted data. The PNG renderers for
[Figure 6](WRR_FIGURE_CHECKS.md#main-figure-6-directly-from-the-loss-and-finance-series)
and [Figure 7](WRR_FIGURE_CHECKS.md#main-figure-7-directly-from-the-exceedance-series)
can redraw the submitted plots directly from this output without archived components.

### Recompute the annualized loss tables (Table S7 and response Table R3)

The last points in the Figure 6 cumulative profiles are the median losses
through 2023 per initial household. Multiply these medians by the initial
tenure counts (41,517 owners or 10,624 renters) for aggregate losses, or
divide by 13 for annualized losses per household. The builder keeps the
full precision until it formats each table in its own units.

```sh
python scripts/utilities/build_wrr_annualized_loss_tables.py --out-dir ../wrr-loss-tables
```

The command writes `Table_S7.csv` and `Table_R3.csv` from the bundled
cumulative series. To use the series recomputed from the 50 paired draws
above, add `--finance-series-dir ../wrr-recomputed-finance`. The builder
checks all 104 cumulative-series rows against the revised reference before
writing either table. No new data files or external archives are needed
for the bundled route.

## Redraw adoption and threat-perception trajectories

Follow the [PNG commands](WRR_FIGURE_CHECKS.md#render-from-the-bundled-data)
to redraw Figures 8 and 9 from the bundled aggregate series. This verifies
the figures against the saved results; this distribution does not include a
command to rebuild those aggregates from individual household decisions.

## Recompute the data for the spatial maps

```sh
python scripts/utilities/rebuild_wrr_spatial_series.py --mc-root ../wrr-ensemble --out-dir ../wrr-spatial-data
```

The builder reads 200 adaptation tract-action CSVs (50 draws × 2020–2023).
For Figure 10 it divides each year's insured household count by that draw's
2020 count for the same tract and tenure. It takes medians of these rates
and, separately, of the within-draw changes from 2020. The supplied data
record proportions; multiply by 100 for the baseline percent and changes in
percentage points. The three baseline classes use 30% and 70% cutoffs; the
central change class includes both −3 and +3 percentage points.

For Supporting Figure S7 it takes the median **population count** for each
tract, tenure, and year before calculating relative change from the 2020
median. These six map fields are already in percent. The verifier compares the
full Figure 10 CSV and the six plotted S7 fields against checksums from the
revised map inputs (S7 at the archived nine-decimal precision). Both CSVs join
on 11-digit tract GEOID. This command verifies the map values and categories;
it does not reproduce the GIS layout.

## Recompute the SFHA area shares

The study archive contains the tract geometry `study_tracts_svi.gpkg` and
the clipped National Flood Hazard Layer extract
`nj_nfhl_sfha_study_tracts.geojson`. Neither layer is bundled. With these
two unchanged files and the optional Windows x64 Python 3.14.0 environment
in `requirements-sfha-areas.txt`, run:

```sh
python -m pip install -r requirements-sfha-areas.txt
python scripts/utilities/rebuild_wrr_sfha_areas.py --tracts /path/to/study_tracts_svi.gpkg --sfha /path/to/nj_nfhl_sfha_study_tracts.geojson --output-dir ../wrr-sfha-areas
```

The command checks both input hashes, unions the archived flood-zone polygons
to avoid double-counting overlaps, and intersects them with the 27 tracts.
It checks both output hashes before writing:

- `sfha_area_share_by_tract.csv` uses NAD83 / UTM zone 18N (EPSG:26918)
  and matches the model input under `config/`.
- `sfha_tract_area_coverage.csv` uses NAD83 / Conus Albers (EPSG:5070)
  and matches the archived Figure 1 coverage table.

The shares differ slightly between the two projections, but all 27 tracts
are in the same Figure 1 classes. Use the model CSV for initialization
replay, not the map CSV. The overlay reproduces the values behind Figure 1,
not its basemap or final layout.

To rebuild **only Figure 1's coverage table** with no study-archive inputs,
use the same Python environment and download both verified public layers:

```sh
python scripts/utilities/rebuild_wrr_sfha_areas.py --public-tracts --public-nfhl --output-dir ../wrr-figure1-areas
```

This repeats the August 17, 2026 extraction from the
[NJDEP-hosted FEMA NFHL layer](https://mapsdep.nj.gov/arcgis/rest/services/Features/Hydrography/MapServer/43).
The command accepts the result only if it matches the archived bytes. If the
service changes, replace `--public-nfhl` with `--sfha` and the unchanged
archived extract. For offline use, point `--public-source-dir` to a verified
cache containing `fema_tracts.geojson` and `nj_nfhl_sfha_study_tracts.geojson`.

The public-tract route verifies all 27 geometries and writes only the
hash-checked `sfha_tract_area_coverage.csv`. To regenerate the
full-precision model CSV, use `--tracts` with the archived tract GeoPackage.
Either verified NFHL source works with that mode; the public tract geometry
does not reproduce the exact model CSV.

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

The bundled-series check compares the whole CSV exactly. Checking from
independently rebuilt results requires both the finance and behavior series;
this distribution provides the finance rebuild only. When both directories
are supplied, the checker matches every row's identifiers, units, and basis
exactly and compares the quantiles at an absolute tolerance of `1e-9` in
their reported units.

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

## Recompute the threshold sensitivity series (Figures S4 and S5)

Use the first ten draws of the verified main ensemble as controls. The archived
case roots and reference inputs are not included in the repository:

```sh
python scripts/utilities/rebuild_wrr_action_intervals.py --mc-root ../wrr-ensemble --case-root ../ex01-raw --case-registry ../ex01-case-registry.csv --annual-metrics ../ex01-annual-metrics.csv --reference-dir ../r1 --out-dir ../wrr-s4-recomputed
python scripts/utilities/rebuild_wrr_gul_threshold_series.py --mc-root ../wrr-ensemble --case-root ../ex02-raw --reference-dir ../r2 --out-dir ../wrr-s5-recomputed
```

S4 uses the revised EX01 full-ten-draw archive
(`20260824_EX01_sfha_area_full10_01`) and its separate interval registry.
Its final reference folder contains `Figure_R1_data.csv` and
`paired_outcomes_by_run.csv`. The builder checks 20 action-interval settings
with ten draws each. Renter insurance and relocation results are shown for
renters; the other settings are shown for homeowners.

S5 uses the revised EX02 archive (`20260824_EX02_sfha_area_full10_02`),
including the 16 threshold settings with ten draws each. Its reference folder
contains `annual_metrics.csv`, `paired_fixed_initial_by_run.csv`, and
`Figure_R2_fixed_initial_data.csv`. Each setting uses the matching draw of the
0.10 control, for the tenure whose threshold was varied.

Both builders sum annual losses, divide by the initial tenure count, and
subtract the normalized loss of the same draw and tenure in the control.
The reported median and IQR summarize these paired differences. Each builder
checks annual finance results before checking the paired and plotted values.
The verified CSVs are plotting data; no PNG or new simulation is produced.

## Recompute the flood-sequence sensitivity series (Figure S6)

Use the audited flood-order study archive and its companion `r3` reference
directory. The suite must contain `raw/` and `processed/run_registry.csv`; the
reference directory must contain `raw_finance_hashes.json`,
`fixed_initial_trajectory_by_run.csv`, and `Figure_R3_fixed_initial_plot_data.csv`.
These archives are not shipped in the repo.

```sh
python scripts/utilities/rebuild_wrr_sequence_series.py --suite-root ../wrr-sequence-archive --reference-dir ../wrr-r3-references --out-dir ../wrr-s6-recomputed
```

The builder verifies 2,860 annual finance files before rebuilding cumulative
losses per initial household. It checks both scenarios' trajectory tables.
Only adaptation runs enter the final plot data: observed losses use the
median and IQR across ten draws, and shuffled losses use the median and range
of the ten sequence medians. This rebuild verifies the 104 plotted rows and
their 5,720 source trajectory rows; it does not rerender the manuscript image
or rerun the sensitivity simulations.

## Recompute the paired sensitivity tables (Tables S8 and S9)

The repository includes 80 cumulative finance records (four cases, ten draws,
and two tenure groups), without individual or tract-level records. To
recompute and check Tables S8–S9 against the final Supporting Information and
response Tables R5–R6, install `requirements-benchmark.txt` and run:

```sh
python scripts/utilities/rebuild_wrr_sensitivity_tables.py --out-dir ../wrr-s8-s9-recomputed
```

S8 compares triangular and truncated-normal threshold draws with the uniform
control. S9 compares the same-year FI-first-EH decision rule with the original
annual rule. For each case, pair the first ten draws with the matching control
draws (same decision seed and posterior index). Sum GUL and payouts from 2011
through 2023. Actual loss is GUL minus payouts, not out-of-pocket cost plus
premium. For each draw and tenure, subtract the control's cumulative loss
from the alternative's, divide by the **initial** cohort size (41,517 owners
or 10,624 renters), and convert to thousands of dollars per household. The
tables report the median and linear 25th and 75th percentiles of these
paired differences. Positive values indicate greater loss under the alternative.

To check the table inputs against raw finance outputs, also supply the
root of the separate WRR reviewer-response analysis archive and the verified
main ensemble:

```sh
python scripts/utilities/rebuild_wrr_sensitivity_tables.py --archive-root ../wrr-review-archive --mc-root ../wrr-ensemble --out-dir ../wrr-s8-s9-from-raw
```

The archive root must retain the original relative EX03 and EX04 case paths.
The command checks all 520 raw annual finance files against the audited
four-case fingerprints and compares the rebuilt totals against the bundled
totals rounded to cents. Both routes check all 12 displayed cells at the
submitted one-decimal precision. This is reconstruction from recorded model
outputs, not a rerun of the sensitivity simulations.
