# FLOODABM

A coupled agent-based and catastrophe flood modeling framework (ABM-CAT) for simulating household flood adaptation in the Passaic River Basin, New Jersey (2011-2023).

## Overview

FLOODABM simulates how homeowners and renters make flood adaptation decisions across 27 census tracts over 13 years of historical flood events. The framework distinguishes four adaptive actions available by homeownership status:

| Action | Available to | Mechanism |
|--------|-------------|-----------|
| Flood Insurance (FI) | Both | Homeowners insure structure and contents, renters insure contents only (NFIP) |
| Elevation and Hardening (EH) | Homeowners | Raise first-floor height to reduce flood vulnerability |
| Buyout Program (BP) | Homeowners | Government acquisition that removes the structure from the floodplain |
| Relocation (RL) | Renters | Move to a different tract to reduce flood exposure |

Each action modifies exposure or vulnerability in the catastrophe flood model. Simulated flood damage feeds back into household threat perception (TP), which drives subsequent adaptation decisions. This two-way coupling is the central mechanism of the framework.

## Project Structure

```
FLOODABM/
├── main.py                      # Single-run simulation
├── main_mc.py                   # Monte Carlo batch runner
├── run_sensitivity_analysis.py  # Sensitivity analysis runner
├── sa_ratio_threshold.py        # SA: ratio threshold sweep
├── sa_decision_threshold.py     # SA: decision threshold sweep
├── requirements.txt
│
├── config/                      # Configuration and input data
│   ├── abm_params.yaml                        # All simulation parameters
│   ├── households_for_abm.csv                  # Household inventory
│   ├── mg_share_by_tract.json                  # Renter share by tract (ACS, legacy filename)
│   └── overall_md_mean_by_tract_2011_2023.json # Mean flood depths by tract-year
│
├── core/                        # Framework infrastructure
│   ├── cli.py                   # Command-line arguments
│   ├── config.py                # Configuration loader
│   ├── data.py                  # Data loading utilities
│   └── paths.py                 # Path management
│
├── modules/
│   ├── actions/                 # Household decision models
│   │   ├── pipeline.py          # Action pipeline orchestration
│   │   ├── decision.py          # Decision logic (FI/EH/BP/RL)
│   │   ├── tp.py                # Threat perception dynamics (ODE)
│   │   ├── vuln_for_tp.py       # Vulnerability input for TP
│   │   └── bayes_fast_predictors.py  # Fast Bayesian prediction
│   ├── finance/                 # Financial outcome calculations
│   │   ├── core.py              # Damage and payout
│   │   ├── premium.py           # NFIP premium calculation
│   │   ├── decisions.py         # Insurance decision logic
│   │   ├── aggregation.py       # Tract-level aggregation
│   │   └── runner.py            # Finance pipeline runner
│   └── vulnerability/           # Flood vulnerability functions
│       └── vulnerability.py
│
├── models/
│   └── baseline_fast/           # Pre-trained Bayesian models (.npz)
│
├── calibration/                 # Calibration pipeline (3 phases)
│   ├── phase1_tenure_distributions/  # Action distributions by homeownership status
│   ├── phase2_bayesian_regression/   # Bayesian logistic regression
│   └── phase3_tp_decay/             # TP decay parameter estimation
│
├── utils/                       # Visualization and output
│   ├── main_helpers.py          # Simulation loop helpers
│   ├── finalize.py              # Post-simulation output
│   ├── plots.py                 # Core plotting functions
│   ├── plots_modular/           # Modular plot components
│   └── sa_dr/                   # SA visualization utilities
│
├── plot_*.py                    # Paper figure generation scripts
└── validate_nfip.py             # NFIP validation against observed data
```

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.10+.

## Data Requirements

All input data required to run the simulation are included in the `config/` directory:

- `abm_params.yaml` -- simulation parameters (TP decay, action bounds, NFIP rates)
- `households_for_abm.csv` -- household inventory (52,141 households across 27 tracts)
- `mg_share_by_tract.json` -- renter share per tract derived from ACS data
- `overall_md_mean_by_tract_2011_2023.json` -- mean flood depths per tract-year from the CAT model

No external data download is needed. The `data/` directory is used for runtime caching and is not tracked by git.

## Quick Start

```bash
# Run baseline scenario (with adaptation)
python main.py --scenario baseline --output-mode full

# Run no-adaptation scenario (--scenario worst disables all adaptive actions)
python main.py --scenario worst --output-mode full

# Run both and generate comparison plots
python main.py --compare-flood-or --output-mode full
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--scenario` | `baseline` | `baseline` (with adaptation) or `worst` (no-adaptation) |
| `--output-mode` | `full` | `full` / `summary` / `minimal` |
| `--out-root` | `outputs/` | Output directory |
| `--deterministic` | off | Fixed RNG seed for reproducibility |
| `--no-plots` | off | Skip figure generation |
| `--compare-flood-or` | off | Run both scenarios and compare |

### Monte Carlo and Sensitivity Analysis

```bash
# Monte Carlo (30 replicates by default)
python main_mc.py --runs 30

# Sensitivity analysis
python sa_ratio_threshold.py
python sa_decision_threshold.py
```

## Configuration

All parameters are defined in `config/abm_params.yaml`, organized into:

- **Threat perception** -- shock amplitude, decay rates (owner vs renter)
- **Action dynamics** -- elevation caps, buyout/relocation toggles, draw bounds
- **Finance** -- NFIP premium rates, deductibles, coverage limits by homeownership status
- **Insurance initialization** -- take-up rates by tract and homeownership status
- **Flood hazard** -- depth thresholds, event masking

## Simulation Flow

1. Load configuration and household inventory
2. For each year (2011-2023):
   - Compute flood hazard per tract (depths and damage ratios)
   - Calculate damage, insurance premiums, payouts, and out-of-pocket costs
   - Determine household actions (FI/EH/BP/RL) based on threat perception
   - Apply actions to update vulnerability, exposure, and financial state
   - Update threat perception (shock in flood years, decay otherwise)
3. Write outputs and generate figures

## Outputs

Results are saved under `outputs/<scenario>/`:

| Directory | Contents |
|-----------|----------|
| `finance/` | Tract- and household-level financial summaries |
| `decisions/` | Household actions and adaptation shares by tract |
| `states/` | Year-end household state snapshots |
| `vulnerability/` | Flood damage ratio metrics |
| `visualization/` | Figures (when plotting is enabled) |

## Reproducing Paper Figures

Each `plot_*.py` script generates one figure from the paper:

| Script | Figure |
|--------|--------|
| `plot_nfip_validation_paper.py` | Fig 3: NFIP validation |
| `plot_fig4_damage_loss_ratio.py` | Fig 4: Damage and loss reduction |
| `plot_fig_finance_combined.py` | Fig 5: Financial outcomes (2x2) |
| `plot_fig6_combined.py` | Fig 6: Adaptation stock and flow |
| *(Fig 7: generated in ArcGIS)* | Fig 7: Spatial flood insurance map |
| `plot_fig10_tp_boxplot.py` | Fig 8: Threat perception dynamics |
| `plot_sa_results.py` | Fig 9: Sensitivity analysis |

## Runtime

A single baseline run completes in approximately 2 minutes on a standard laptop. Monte Carlo simulation with 30 replicates takes approximately 60 minutes.

## Citing

If you use this code, please cite:

> Chiou, W., Yang, Y.C.E., Tanaka, T., Jamrussri, S., & Feng, S. Household Flood Adaptation and Financial Outcomes for Homeowners and Renters: A Coupled Agent-Based and Catastrophe Flood Modeling Framework. *Journal of Hydrology* (submitted 2025).

## License

MIT License
