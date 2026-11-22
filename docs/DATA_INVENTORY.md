# WRR data inventory

The revised benchmark uses one published household inventory and one set of
full Bayesian posteriors. The [reproduction guide](REPRODUCIBILITY.md) explains
how to rerun the benchmark and check its numerical outputs.

Start with `config/abm_params.yaml`. Use `config/households_for_abm_sfha.csv`
for both simulation and initialization replay. The prior unassigned household
file is recorded in the manifest for provenance only.

## Data retained in the checkout

| File | Consumer or purpose |
| --- | --- |
| `config/households_for_abm_sfha.csv` | The simulation reads the synthetic household attributes, SFHA status, and initial flood-insurance status from this file. |
| `config/abm_params.yaml` | Current model parameters and input paths. |
| `config/overall_md_mean_by_tract_2011_2023.json` | Tract-year flood depths used for threat perception and flood-prone classification. |
| `config/depth_distribution_by_tract_year.csv` | Distributions used to sample household flood depths. These and the tract-year means serve different functions. |
| `config/sfha_area_share_by_tract.csv` | Source tract-area fractions for initialization. |
| `config/sfha_shares.csv` | Tract shares in the schema read by the simulator, including during relocation. |
| `config/sfha_assignment_summary.csv` | Initialization audit by tract and tenure. |
| `config/initial_fi_summary.csv` | Initial insurance counts and rates by tenure and SFHA status. |
| `config/sfha_assignment_manifest.json` | Initialization seeds, checksums, and provenance. |
| `config/models/models_config.yaml` | Registry of the shipped baseline model. |
| `data/supplementary/wrr_mc50_finance_reference.csv` | Seed schedule and archived finance checksums for the revised 50-draw ensemble. |
| `data/supplementary/wrr_main_figures_reference.json` | Dimensions and decoded pixel checksums for the final revised main figures. |
| `data/supplementary/Table_S_MC_variance.csv` | Current Text S7 table: 455 yearly entries of medians and quartiles for the revised ensemble. |
| `data/figures/Figure8_adaptation_trajectories.csv` | Final group-level median and interquartile ranges plotted in Figure 8; no household records. |
| `data/figures/Figure11_income_aep.csv` | Final pooled and sorted income-normalized AEP series plotted in Figure 11; no household records. |
| `data/figures/Figure5_nfip_validation_corrected_county_labels.csv` | Final simulated summary and observed standardized payouts by county and year, with corrected labels. |
| `data/figures/Figure5_metrics_corrected_county_labels.csv` | Correlations and year counts in Figure 5. |
| `data/figures/Figure6_cumulative_loss.csv` | Final cumulative loss series for Figure 6a–b, normalized by fixed initial tenure counts. |
| `data/figures/Figure6_annual_finance.csv` | Final finance series for Figure 6c–f, archived under the former filename `Figure7_annual_finance.csv` before the panels were renumbered. |
| `data/figures/Figure7_aep.csv` | Final pooled run-year GUL and actual-loss exceedance curves in dollars per household. No individual household records. |
| `data/figures/Figure9_tp_by_prone.csv` | Final annual group-level threat perception medians and interquartile ranges. |
| `data/figures/FigureS4_action_threshold_intervals.csv`, `FigureS5_gul_ratio_threshold.csv`, and `FigureS6_flood_event_sequence.csv` | Accepted group-level sensitivity plot tables (20, 16, and 104 rows). These contain no individual household records and can be replaced at render time with the verified rebuilt tables. |
| `data/supplementary/wrr_s8_s9_cumulative_finance.csv` | 80 tenure-level cumulative GUL and payout totals for Tables S8–S9; no individual or tract-level records. |
| `models/baseline/{owner_BP,owner_EH,owner_FI,renter_FI,renter_RL}.npz` | One full posterior per action and tenure; shared by the ensemble and posterior predictive query. |
| `models/baseline_fast/{owner_BP,owner_EH,owner_FI,renter_FI,renter_RL}.npz` | Smaller posterior-mean caches used by the default single-run simulation. These are not duplicates of the full posteriors. |
| `models/baseline/best_calibrators.json` | Fallback calibrators for the canonical baseline predictors. |
| `calibration/phase3_tp_decay/outputs/tp_decay_params_for_abm.json` | Reference result read by the optional TP-calibration report and augmented-calibration tools. The published simulation reads its active settings from `config/abm_params.yaml`. |

## Superseded data removed from the checkout

| File | Reason |
| --- | --- |
| `config/households_for_abm.csv` | Historical pre-SFHA snapshot. The retained household file can replay the assignment; the original import remains available at the pinned commit in the manifest. |
| `calibration/phase2_bayesian_regression/models/owner_BP.npz` | Identical to `models/baseline/owner_BP.npz`. |
| `calibration/phase2_bayesian_regression/models/owner_EH.npz` | Identical to `models/baseline/owner_EH.npz`. |
| `calibration/phase2_bayesian_regression/models/owner_FI.npz` | Identical to `models/baseline/owner_FI.npz`. |
| `calibration/phase2_bayesian_regression/models/renter_FI.npz` | Identical to `models/baseline/renter_FI.npz`. |
| `calibration/phase2_bayesian_regression/models/renter_RL.npz` | Identical to `models/baseline/renter_RL.npz`. |
| `modules/actions/calibrators/best_calibrators.json` | Old calibrator export; the model loader does not search this directory. |
| `modules/actions/calibrators/best_calibrators.yaml` | Old calibrator export with no remaining reader. |
| `calibration/data/data.xlsx` and `calibration/data/cal_data.xlsx` | Optional historical refitting inputs removed in the preceding cleanup. Fitted simulation models do not require these workbooks. |
| `data/supplementary/mc_initpsy_seeds.csv` and the former 341-row `Table_S_MC_variance.csv` | Earlier ensemble seed list and variance summary. The revised draw schedule and 455-row Text S7 table replace them. |
| Root `core/` and `utils/config_loader.py` | Unused old settings, data, path, and CLI loaders. The calibration-specific `core/` package is separate and retained. |
| `modules/actions/MODELS_VERSION_GUIDE.md` | Documented unshipped model variants and workstation-specific paths. Use the reproduction guide for the shipped model. |
| `main_mc.py`, `scripts/utilities/run_montecarlo.py`, `scripts/utilities/run_mc_extend.py` | Obsolete ensemble drivers. The revised benchmark uses the fixed draw schedule in `scripts/utilities/run_mc100_local.py`. |
| `generate_paper_figures.py`, `scripts/paper_figures/plot_mc_convergence_s2.py`, `scripts/utilities/gen_table_s_mc_variance.py` | Superseded plotting registry and summaries for earlier runs. See the [revised figure guide](WRR_FIGURE_CHECKS.md) for current render routes. |
| `scripts/utilities/gen_fig9_bivariate_classes.py`, `gen_spatial_fi_delta_mc50.py`, `gen_spatial_pop_delta_mc50.py` (same directory) | Earlier map preparation paths with workstation-specific inputs and obsolete figure numbering. The revised builder verifies Figure 10 and Supporting Figure S7 map data from the canonical ensemble. |
| `scripts/utilities/compute_sm_table_2020_baseline_fi.py`, `compute_sm_table_non_fi_pool.py` (same directory) | Earlier exploratory tables depend on workstation paths and old ensemble outputs. Use the revised spatial-series builder for the data behind Figure 10. The exploratory tables remain in Git history. |
| `scripts/utilities/_calc_zscore_r2.py`, `gen_zscore_validation.py` (same directory) | Validation scripts for obsolete single-run threshold experiments. Use the paired-draw finance builder and final Figure 5 PNG route for the revised validation. |
| Former `scripts/paper_figures/` | These scripts used older ensembles and figure numbering, or produced exploratory figures that are not in the revised manuscript, supporting information, or response. Use the [revised figure guide](WRR_FIGURE_CHECKS.md) and [threshold sensitivity reconstruction commands](REPRODUCIBILITY.md#recompute-the-threshold-sensitivity-series-figures-s4-and-s5) instead. The historical scripts remain in Git. |
| Former `scripts/sensitivity/` | Older standalone sweeps and the initial-insurance experiments predate the revised sensitivity design. The older combined runner could temporarily rewrite the benchmark YAML and household CSV. Use the [reconstruction guide](REPRODUCIBILITY.md#recompute-the-threshold-sensitivity-series-figures-s4-and-s5) for the published analyses or the [scenario guide](SCENARIOS.md#4-other-scenario-levers-cli-flags-and-parameters) for new runs. The old tools remain in Git history. |

The published YAML contains only the shipped model settings and input paths.
It no longer lists unused Beta parameters, a separate owner-share table,
unshipped hazard defaults, or unshipped model variants.
The action pipeline takes its settings from the simulation runner; it does not
read a separate JSON configuration.

Unchanged dates on fitted models or source data do not make them stale. Use the
consumer and verification path above to distinguish active dependencies and
reference artifacts from obsolete copies.
