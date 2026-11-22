# FLOODABM

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20587985.svg)](https://doi.org/10.5281/zenodo.20587985)

A coupled agent-based and catastrophe flood modeling framework (ABM-CAT) for simulating household flood adaptation in the Passaic River Basin, New Jersey (2011–2023).

[How the annual simulation links flood losses and household decisions](#simulation-flow)

## Overview

FLOODABM simulates how homeowners and renters make flood adaptation decisions across 27 census tracts over 13 years of historical flood events. The framework distinguishes four adaptive actions available by homeownership status:

| Action | Available to | Mechanism |
|--------|-------------|-----------|
| Flood Insurance (FI) | Both | Homeowners insure structure and contents, renters insure contents only (NFIP) |
| Elevation and Hardening (EH) | Homeowners | Raise first-floor height to reduce flood vulnerability |
| Buyout Program (BP) | Homeowners | Government acquisition that removes the structure from the floodplain |
| Relocation (RL) | Renters | Move to a different tract to reduce flood exposure |

The flood model calculates household losses. A separate tract-level ground-up loss (GUL) ratio informs the household threat-perception (TP) update. Adaptation decisions then change insurance, vulnerability, or exposure for the next year.

## Project Structure

```
FLOODABM/
├── main.py                      # Single-run simulation (2011-2023)
├── requirements.txt
│
├── config/                      # Configuration and input data
│   ├── abm_params.yaml          # All simulation parameters
│   ├── households_for_abm_sfha.csv  # Household inventory with inside_SFHA and initial FI status
│   └── overall_md_mean_by_tract_2011_2023.json  # Flood depths by tract-year
│
├── modules/
│   ├── actions/                 # Household decision models (Bayesian + TP)
│   ├── finance/                 # NFIP premium, payout, OOP calculations
│   └── vulnerability/           # Flood damage functions
│
├── models/
│   ├── baseline/                # Fitted posterior draws for the revised ensemble
│   └── baseline_fast/           # Posterior-mean caches for single runs
│
├── calibration/                 # Calibration provenance and fitted model query
│   ├── bayesian_engine/          # Phase 2 fitting code retained for provenance
│   ├── phase2_bayesian_regression/ # Query the shipped posterior models
│   └── phase3_tp_decay/          # Optional TP decay refitting with external input
│
├── utils/                       # Visualization helpers and output
│
└── scripts/
    ├── examples/                # Example helper scripts (e.g. future-depth generator)
    └── utilities/               # Ensemble runner, figure rebuilds, and verification
```

## Installation

Use a clean Python 3.14.0 environment for the revised WRR benchmark.

```bash
python -m pip install -r requirements-benchmark.txt
```

The separate `requirements.txt` serves the older optional plotting and data-preparation tools. It is not the installation file for the revised benchmark.

## Data Requirements

All input data required to run the simulation are included in the `config/` directory:

- `abm_params.yaml` -- simulation parameters (TP decay, action bounds, NFIP rates)
- `households_for_abm_sfha.csv` -- current household inventory (52,141 households across 27 tracts), including the binary `inside_SFHA` attribute and the SFHA-specific initial FI status
- `sfha_area_share_by_tract.csv` -- input tract-area fractions used by the initialization builder
- `sfha_shares.csv` -- one SFHA area share per tract, used for both tenure groups
- `sfha_assignment_manifest.json` -- checksums, seeds, initial FI rates and initialization counts
- `overall_md_mean_by_tract_2011_2023.json` -- mean flood depths per tract-year from the CAT model
- `depth_distribution_by_tract_year.csv` -- grid-cell depth distributions used to sample household depths

No survey workbook or external data download is needed to rerun the model with the published calibration. The benchmark uses the fitted models and the tract-area initialization in this repository. See [Reproducing the revised benchmark](docs/REPRODUCIBILITY.md) for the exact run and output checks, and how those differ from refitting the model or regenerating the paper's graphics.

## Quick Start

After installing the benchmark dependencies, run either scenario. Use `--no-plots` to write the simulation data without generating figures:

```bash
# Baseline scenario: households adapt via flood insurance, elevation, buyout, relocation
python main.py --scenario baseline --no-plots --output-mode full

# No-adaptation baseline (--scenario worst disables all adaptive actions)
python main.py --scenario worst --no-plots --output-mode full
```

With the default model (`baseline`), the adaptation run writes to `outputs/baseline/baseline/` and the no-adaptation run writes to `outputs/baseline/worst/`. Each contains `decisions/`, `finance/`, `vulnerability/`, and `states/`. The directory pattern is `outputs/<model>/<scenario>/`.

> To recreate the paper figures separately as PNGs, follow the [revised manuscript and supporting figure guide](docs/WRR_FIGURE_CHECKS.md).

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
# Local 50-run ensemble used for the paper figures (100 scenario runs)
python scripts/utilities/run_mc100_local.py --out-root ../FLOODABM_mc50 --scenarios baseline worst
```

For the revised paper, see [Figures S4–S5 threshold sensitivity](docs/REPRODUCIBILITY.md#recompute-the-threshold-sensitivity-series-figures-s4-and-s5),
[Figure S6 flood-sequence sensitivity](docs/REPRODUCIBILITY.md#recompute-the-flood-sequence-sensitivity-series-figure-s6),
and [Tables S8–S9 paired sensitivities](docs/REPRODUCIBILITY.md#recompute-the-paired-sensitivity-tables-tables-s8-and-s9).
These reproduce the published summaries from the archived cases. For a new
experiment, use the [scenario options](docs/SCENARIOS.md#4-other-scenario-levers-cli-flags-and-parameters).

## Documentation

Guides for running your own scenarios and extending the model (e.g. climate
projections, future-behavior + CAT studies):

| Guide | For |
|---|---|
| [docs/SCENARIOS.md](docs/SCENARIOS.md) | Running climate / CAT hazard scenarios — swap the flood-depth file (`--scenario` is adaptation on/off, **not** the hazard) |
| [docs/FUTURE_SIMULATION.md](docs/FUTURE_SIMULATION.md) | Running beyond 2023 (future-behavior + CAT) — **read the silent-truncation warning** |
| [docs/DATA_FORMATS.md](docs/DATA_FORMATS.md) | Exact input schemas (flood-depth file, household file) + the GEOID-alignment requirement |
| [docs/PARAMETERS.md](docs/PARAMETERS.md) | Calibrated (keep-frozen) vs tunable parameters; reuse the shipped models vs recalibrate |
| [docs/KNOWN_LIMITATIONS.md](docs/KNOWN_LIMITATIONS.md) | Engineering caveats for reusers/extenders: inert config, two-entrypoint differences, hardcoded action set, silent-failure modes |
| `scripts/examples/make_future_depths.py` | Starter script to build a future flood-depth file |

## Configuration

All parameters are defined in `config/abm_params.yaml`, organized into:

- **Threat perception** -- shock amplitude, decay rates (owner vs renter)
- **Action dynamics** -- elevation caps, buyout/relocation toggles, draw bounds
- **Finance** -- NFIP premium rates, deductibles, coverage limits by homeownership status
- **Insurance initialization** -- take-up rates by homeowner/renter group and SFHA status
- **SFHA initialization** -- SFHA area share by tract, with the same share used for homeowners and renters. Initial FI rates are 30% for homeowners inside the SFHA, 3% for homeowners outside, 8% for renters inside, and 2% for renters outside. Homeowners use FI draw bounds U(0, 0.10) inside the SFHA and U(0.35, 0.55) outside; renters use U(0.70, 0.90) regardless of SFHA status.
- **Flood hazard** -- depth thresholds, event masking

The formal ensemble uses `scripts/utilities/run_mc100_local.py` to run both
scenarios with the fixed revised design. See the [reproduction guide](docs/REPRODUCIBILITY.md).

## Simulation Flow

1. Load the revised configuration and initial household inventory.
2. For each year from 2011 through 2023, sample flood depths for each
   household from its tract's depth distribution. Calculate ground-up loss
   and financial outcomes under the insurance status in effect that year.
3. In the adaptation scenario, use the tract-level GUL ratio to update TP,
   then use the behavioral model to evaluate the actions available to each
   household group. The no-adaptation scenario skips household decisions.
4. Carry the resulting insurance, first-floor height, or exposure changes
   into the next year's simulation. Save annual losses and decisions for
   the figure and table rebuilds.

## Outputs

Results are saved under `outputs/<scenario>/`:

| Directory | Contents |
|-----------|----------|
| `finance/` | Tract- and household-level financial summaries |
| `decisions/` | Household actions and adaptation shares by tract |
| `states/` | Year-end household state snapshots |
| `vulnerability/` | Flood damage ratio metrics |
| `visualization/` | Figures (when plotting is enabled) |

## Reproduce and check the revised figures

From a fresh clone, generate Figures 5–8 and 11 directly as PNGs from the
bundled results. The exact-render environment uses Python 3.14.0 and local
Times New Roman:

```sh
python -m pip install -r requirements-benchmark.txt -r requirements-vector-figures.txt
python scripts/utilities/render_wrr_figures.py --figure-dir ../wrr-png --figures 5 6 7 8 11
python scripts/utilities/verify_wrr_figures.py --figure-dir ../wrr-png --figures 5 6 7 8 11
```

Figure 9 and Supporting Figures S4–S6 also redraw from bundled results in
their separate pinned environment. The [figure guide](docs/WRR_FIGURE_CHECKS.md)
lists the PNG-only commands and distinguishes these plots from archive checks.
The standard workflow does not regenerate maps or manuscript illustrations
and requires no GIS software. The simulation still uses the bundled tract
flood depths and special flood hazard area (SFHA) shares.
Use the [benchmark instructions](docs/REPRODUCIBILITY.md) to rerun the underlying simulations.

> Note: the formal Monte Carlo driver requires `--out-root` to point outside the repository. Across the 50 runs, it varies the action-decision seed and Bayesian posterior draw while keeping the SFHA assignment, initial FI status, household attributes, flood sequence, and flood-depth sampling seed fixed. The posterior `.npz` files are stored in `models/baseline/`; single runs use the posterior-mean cache in `models/baseline_fast/`.

## Runtime

A single baseline run covers 2011–2023. The formal ensemble uses 50 draws per scenario (100 simulations when both baseline and worst are run); runtime depends on hardware and output mode.

## Citing

If you use this code, please cite both the paper and the archived software:

> Chiou, W., Yang, Y. C. E., Tanaka, T., Jamrussri, S., & Feng, S. (2026). Household flood adaptation and financial outcomes: A coupled human-flood modeling analysis of homeowners and renters. *Water Resources Research* (under review).

> Chiou, W., Yang, Y. C. E., Tanaka, T., Jamrussri, S., & Feng, S. (2026). *FLOODABM* (v1.0.0) [Software]. Zenodo. https://doi.org/10.5281/zenodo.20587986

## License

MIT License
