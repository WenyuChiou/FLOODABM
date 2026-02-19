# FLOODABM

A coupled agent-based and catastrophe modeling framework (ABM-CAT) for simulating household flood adaptation in the Passaic River Basin, New Jersey (2011-2023).

## Overview

FLOODABM models how homeowners and renters make protective decisions across historical flood events. The framework distinguishes four adaptive actions:

- **Flood Insurance (FI)**: Homeowners insure structure + contents; renters insure contents only (per NFIP rules)
- **Elevation (EH)**: Homeowners raise first-floor height to reduce flood vulnerability
- **Buyout (BP)**: Homeowners participate in government acquisition programs
- **Relocation (RL)**: Renters relocate to reduce exposure

Each action modifies a specific component of the catastrophe model. Updated flood damage feeds back into household risk perception (threat perception), which drives subsequent adaptation decisions. This two-way coupling is the central mechanism of the framework.

## Project Structure

```
FLOODABM/
├── main.py                 # Primary simulation entry point
├── main_mc.py              # Monte Carlo batch runner
├── sa_main.py              # Sensitivity analysis runner
├── requirements.txt        # Python dependencies
├── config/
│   ├── abm_params.yaml           # All simulation parameters
│   ├── households_for_abm.csv    # Household inventory (tract, tenure, value)
│   ├── mg_share_by_tract.json    # Renter share by census tract (ACS)
│   └── overall_md_mean_by_tract_2011_2023.json  # Mean flood depths by tract
├── core/                   # Core framework
│   ├── cli.py              # Command-line argument parsing
│   ├── config.py           # Configuration loading and validation
│   ├── data.py             # Data loading utilities
│   └── paths.py            # Path management
├── modules/                # Simulation components
│   ├── actions/            # Household decision models
│   │   ├── mgmix_pipeline.py     # Action pipeline
│   │   ├── mgmix_decision.py     # Decision logic per action type
│   │   ├── mgmix_tp.py           # Threat perception dynamics
│   │   ├── vuln_for_tp.py        # Vulnerability input for TP
│   │   └── bayes_fast_predictors.py  # Fast Bayesian prediction
│   ├── finance/            # Financial outcome calculations
│   │   ├── core.py         # Damage and payout calculations
│   │   ├── decisions.py    # Insurance decision logic
│   │   ├── premium.py      # NFIP premium calculation
│   │   ├── aggregation.py  # Tract-level aggregation
│   │   └── runner.py       # Finance pipeline runner
│   └── vulnerability/      # Flood vulnerability functions
│       └── vulnerability.py
├── models/
│   └── baseline_fast/      # Pre-trained Bayesian models (.npz)
├── utils/                  # Visualization and output utilities
│   ├── main_helpers.py     # Simulation loop helpers
│   ├── finalize.py         # Post-simulation finalization
│   ├── plots.py            # Plotting functions
│   ├── plots_modular/      # Modular plot components
│   ├── plots_comparision_scenario.py  # Baseline vs worst-case comparison
│   └── sa_dr/              # Sensitivity analysis utilities
├── data/                   # Runtime data (not tracked)
├── fig/                    # Generated figures (not tracked)
└── logs/                   # Simulation logs (not tracked)
```

## Installation

```bash
pip install -r requirements.txt
```

**Requirements**: Python 3.10+, numpy, pandas, matplotlib, PyYAML

## Usage

### Run baseline scenario (with adaptation)

```bash
python main.py --scenario baseline --output-mode full
```

### Run worst-case scenario (no adaptation)

```bash
python main.py --scenario worst --output-mode full
```

### Run both scenarios and generate comparison plots

```bash
python main.py --compare-flood-or --output-mode full
```

### Key CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--scenario` | `baseline` | `baseline` (with adaptation) or `worst` (no adaptation) |
| `--output-mode` | `full` | `full` / `summary` / `minimal` |
| `--out-root` | `outputs/` | Output directory |
| `--deterministic` | off | Fixed RNG for reproducibility |
| `--no-plots` | off | Skip visualization |
| `--compare-flood-or` | off | Run both scenarios and compare |
| `--compare-severe-years` | `2011,2014,2021` | Severe flood years for comparison plots |

### Monte Carlo Simulation

```bash
python main_mc.py --runs 100
```

### Sensitivity Analysis

```bash
# Recommended: summary mode (76% storage savings)
python sa_main.py --min 0.1 --max 0.8 --step 0.1 --output-mode summary

# Full mode for publication
python sa_main.py --min 0.1 --max 0.8 --step 0.1 --output-mode full
```

## Configuration

All simulation parameters are defined in `config/abm_params.yaml`:

- **Threat perception**: Shock and decay models for risk perception dynamics
- **Action dynamics**: Elevation caps, buyout/relocation toggles
- **Finance**: NFIP premium rates, deductibles, coverage limits by tenure
- **Insurance initialization**: Take-up rates by tract and tenure
- **Flood hazard**: Depth thresholds, event masking

## Simulation Flow

1. Load configuration and household data
2. For each year (2011-2023):
   - Compute flood hazard (depths and ratios per tract)
   - Calculate damage, premiums, payouts, and uncompensated costs
   - Determine household actions (FI/EH/BP/RL) based on threat perception
   - Apply actions: update vulnerability, exposure, and financial state
   - Update threat perception (decay in non-flood years; shock in flood years)
   - Advance state to next year
3. Generate output figures and summary statistics

### Threat Perception Model

Threat perception (TP) drives household decisions. It increases via flood shocks and decays over time:

- **Shock**: `TP' = min(1.0, TP + shock_scale * damage_ratio)` (applied in flood years)
- **Decay**: `TP' = TP * exp(-ln(2) * w * Eff)` where `w = alpha*(1-PA) + beta*SC`

Higher policy awareness (PA) slows TP decay, reflecting that informed households maintain risk perception longer.

## Outputs

Under `outputs/<scenario>/`:

| Directory | Contents |
|-----------|----------|
| `finance/` | Tract- and household-level financial summaries |
| `decisions/` | Household actions and per-tract action shares |
| `states/` | Year-end state snapshots |
| `vulnerability/` | Flood ratio metrics |
| `visualization/` | Figures (if plotting enabled) |

## Citing

If you use this code, please cite the associated paper:

> [Citation to be added upon publication]

## License

[To be added]
