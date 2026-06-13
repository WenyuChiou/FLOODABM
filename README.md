# FLOODABM

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20587985.svg)](https://doi.org/10.5281/zenodo.20587985)

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
├── main.py                      # Single-run simulation (2011-2023)
├── main_mc.py                   # Monte Carlo batch runner
├── generate_paper_figures.py    # Unified paper figure pipeline (see Reproducing Paper Figures)
├── requirements.txt
│
├── config/                      # Configuration and input data
│   ├── abm_params.yaml          # All simulation parameters
│   ├── households_for_abm.csv   # Household inventory (52,141 households)
│   └── overall_md_mean_by_tract_2011_2023.json  # Flood depths by tract-year
│
├── core/                        # Framework infrastructure
│   ├── config.py                # Configuration loader
│   ├── data.py                  # Data loading utilities
│   ├── cli.py                   # Command-line arguments
│   └── paths.py                 # Path management
│
├── modules/
│   ├── actions/                 # Household decision models (Bayesian + TP)
│   ├── finance/                 # NFIP premium, payout, OOP calculations
│   └── vulnerability/           # Flood damage functions
│
├── models/
│   └── baseline_fast/           # Pre-trained Bayesian models (.npz)
│
├── calibration/                 # Calibration pipeline (3 phases)
│   ├── phase1_tenure_distributions/  # Beta distribution fitting
│   ├── phase2_bayesian_regression/   # Bayesian logistic regression
│   └── phase3_tp_decay/             # TP decay parameter estimation
│
├── utils/                       # Visualization helpers and output
│
└── scripts/                     # Organized scripts (not called directly)
    ├── paper_figures/           # Individual figure scripts (called by generate_paper_figures.py)
    ├── examples/                # Example helper scripts (e.g. future-depth generator)
    ├── sensitivity/             # Sensitivity analysis runners
    ├── validation/              # NFIP validation
    ├── utilities/               # One-time data generation and export tools
    └── legacy_plots/            # Superseded figure scripts (kept for reference)
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
- `overall_md_mean_by_tract_2011_2023.json` -- mean flood depths per tract-year from the CAT model

No external data download is needed to run the model. Processed outputs that support published tables and figures are provided under `data/supplementary/`, for example `Table_S_MC_variance.csv` (the annual median and interquartile range across the 50 stochastic runs reported in the Supporting Information) and `mc_initpsy_seeds.csv` (the 50-run Monte Carlo seed list). Other contents of `data/` are runtime caches and are not tracked by git.

## Quick Start

Install, then run the baseline experiment. Use `--no-plots` to write all data outputs without the optional plotting step — this is the recommended way to run the model and the entry point for adding new scenarios (e.g. climate projections):

```bash
pip install -r requirements.txt

# Baseline scenario: households adapt via flood insurance, elevation, buyout, relocation
python main.py --scenario baseline --no-plots --output-mode full

# No-adaptation baseline (--scenario worst disables all adaptive actions)
python main.py --scenario worst --no-plots --output-mode full
```

A baseline run takes ~2 minutes and writes results under `outputs/baseline/`:
`decisions/` (household actions and adoption shares by tract), `finance/` (premiums, payouts, OOP, per-tract financial summaries), `vulnerability/` (flood damage ratios), and `states/` (year-end household state). The `worst` scenario writes to `outputs/worst/`.

> Plotting is optional and separate from running the experiment. The inline plots (omit `--no-plots`) and the paper-figure scripts need `scipy` + `openpyxl` with `matplotlib < 3.9` (pinned in `requirements.txt`); some inline plots are still fragile. For running the model and analysing results, use `--no-plots` and read the output CSVs.

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
# Monte Carlo stochastic runs (the analyses in the paper use 50 runs)
python main_mc.py --runs 50

# Sensitivity analysis
python scripts/sensitivity/sa_ratio_threshold.py
python scripts/sensitivity/sa_decision_threshold.py
```

## Documentation

Guides for running your own scenarios and extending the model (e.g. climate
projections, future-behavior + CAT studies):

| Guide | For |
|---|---|
| [docs/SCENARIOS.md](docs/SCENARIOS.md) | Running climate / CAT hazard scenarios — swap the flood-depth file (`--scenario` is adaptation on/off, **not** the hazard) |
| [docs/FUTURE_SIMULATION.md](docs/FUTURE_SIMULATION.md) | Running beyond 2023 (future-behavior + CAT) — **read the silent-truncation warning** |
| [docs/DATA_FORMATS.md](docs/DATA_FORMATS.md) | Exact input schemas (flood-depth file, household file) + the GEOID-alignment requirement |
| [docs/PARAMETERS.md](docs/PARAMETERS.md) | Calibrated (keep-frozen) vs tunable parameters; reuse the shipped models vs recalibrate |
| `scripts/examples/make_future_depths.py` | Starter script to build a future flood-depth file |

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

Use the unified pipeline to regenerate the main-text figures from a Monte Carlo output tree:

```bash
python generate_paper_figures.py          # run all registered figure scripts
python generate_paper_figures.py --figs 6 7   # specific figures only
python generate_paper_figures.py --list   # show the figure registry
```

The figure scripts live in `scripts/paper_figures/` and read from a 50-run Monte Carlo output tree produced by `scripts/utilities/run_mc100_local.py`. Registered figures follow the manuscript numbering:

| Manuscript figure | Script |
|--------|--------|
| Fig 5 — NFIP avg payout per claim (z-score), 3 counties on one panel, 50-run median + IQR | `plot_fig5_nfip_validation.py` |
| Fig 6 — cumulative ground-up and actual loss per household | `plot_fig_rq1_combined.py` |
| Fig 7 — financial outcomes and loss AEP curves | `plot_fig_rq1_combined.py` (same run as Fig 6) |
| Fig 9 — tract-level mean threat perception, flood-prone vs non-prone | `plot_fig8_tp_by_prone.py` |
| Fig 11 — income-normalized AEP of household loss (% of tenure median income) | `plot_fig11_income_aep.py` |

Adaptation-trajectory (Fig 8), threat-perception distribution, ratio-threshold sensitivity, and supporting ceiling-effect figures are produced by the other scripts in `scripts/paper_figures/`. The supporting tables (including `data/supplementary/Table_S_MC_variance.csv`) are generated by the scripts in `scripts/utilities/` (`gen_table_s_mc_variance.py`, `compute_sm_table_*.py`).

Prepared outside this pipeline: the study-area map (Fig 1), the Bayesian-procedure diagram (Fig 2), the framework flowchart (Fig 3), and the flood-model diagnostics (Fig 4). The spatial flood-insurance maps (Fig 10) and the supporting population-change maps (SI Fig S4) are drawn in ArcGIS from the tract-level CSVs written by `scripts/utilities/gen_spatial_fi_delta_mc50.py` and `gen_spatial_pop_delta_mc50.py`.

> Note: the Monte Carlo driver (`scripts/utilities/run_mc100_local.py`) writes under `FLOODABM_MC_ROOT` (default `outputs/mc50/`); set that variable to relocate the output tree, which the figure scripts then read. The 50-run seed list is published at `data/supplementary/mc_initpsy_seeds.csv`. Each run draws a distinct Bayesian posterior sample (the posterior `.npz` ship in `models/baseline/`), so the 50-run posterior spread is reproduced directly; single runs use the posterior-mean cache in `models/baseline_fast/`.

## Runtime

A single baseline run (2011–2023) completes in approximately 2 minutes on a standard laptop. A 50-run Monte Carlo batch scales roughly linearly with the number of runs.

## Citing

If you use this code, please cite both the paper and the archived software:

> Chiou, W., Yang, Y. C. E., Tanaka, T., Jamrussri, S., & Feng, S. (2026). Household flood adaptation and financial outcomes: A coupled human-flood modeling analysis of homeowners and renters. *Water Resources Research* (under review).

> Chiou, W., Yang, Y. C. E., Tanaka, T., Jamrussri, S., & Feng, S. (2026). *FLOODABM* (v1.0.0) [Software]. Zenodo. https://doi.org/10.5281/zenodo.20587986

## License

MIT License
