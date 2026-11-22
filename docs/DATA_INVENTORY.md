# WRR data inventory

The revised benchmark uses one published household inventory and one set of
full Bayesian posteriors. The [reproduction guide](REPRODUCIBILITY.md) explains
how to rerun the benchmark and check its numerical outputs.

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
| `data/figures/Figure8_adaptation_trajectories.csv` | Final group-level median and interquartile ranges plotted in Figure 8; no household records. |
| `data/figures/Figure11_income_aep.csv` | Final pooled and sorted income-normalized AEP series plotted in Figure 11; no household records. |
| `data/figures/Figure5_nfip_validation_corrected_county_labels.csv` | Final simulated summary and observed standardized payouts by county and year, with corrected labels. |
| `data/figures/Figure5_metrics_corrected_county_labels.csv` | Correlations and year counts in Figure 5. |
| `data/figures/Figure9_tp_by_prone.csv` | Final annual group-level threat perception medians and interquartile ranges. |
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
| `data/supplementary/mc_initpsy_seeds.csv` and `data/supplementary/Table_S_MC_variance.csv` | Superseded ensemble seed list and variance summary removed in the preceding cleanup. |
| Root `core/` and `utils/config_loader.py` | Unused old settings, data, path, and CLI loaders. The calibration-specific `core/` package is separate and retained. |
| `modules/actions/MODELS_VERSION_GUIDE.md` | Documented unshipped model variants and workstation-specific paths. Use the reproduction guide for the shipped model. |
| `main_mc.py`, `scripts/utilities/run_montecarlo.py`, `scripts/utilities/run_mc_extend.py` | Obsolete ensemble drivers. The revised benchmark uses the fixed draw schedule in `scripts/utilities/run_mc100_local.py`. |
| `generate_paper_figures.py`, `scripts/paper_figures/plot_mc_convergence_s2.py`, `scripts/utilities/gen_table_s_mc_variance.py` | Superseded plotting registry and summaries for earlier runs. See the [revised figure guide](WRR_FIGURE_CHECKS.md) for current render routes. |

The published YAML contains only the shipped model settings and input paths.
It no longer lists unused Beta parameters, a separate owner-share table,
unshipped hazard defaults, or unshipped model variants.

Unchanged dates on fitted models or source data do not make them stale. Use the
consumer and verification path above to distinguish active dependencies and
reference artifacts from obsolete copies.
