# Running Scenarios (incl. climate / CAT hazard scenarios)

This guide explains how to run FLOODABM under scenarios beyond the two built-in
`--scenario` modes, including **climate / catastrophe (CAT) flood-hazard
scenarios**. No model code changes are required.

> The behavioral model (the Bayesian adaptation-decision model in `models/`) is a
> property of the surveyed Passaic River Basin population, not of the hazard. It is
> **reused unchanged** across hazard scenarios. You change the *flood input*, not
> the model. See [PARAMETERS.md](PARAMETERS.md) for what is safe to change.

## 1. What `--scenario baseline | worst` actually means

`--scenario` is **not** a hazard switch. It toggles household adaptation
(`core/cli.py`, `choices=["baseline", "worst"]`):

| `--scenario` | Meaning |
|---|---|
| `baseline` | Households adapt (flood insurance, elevation, buyout, relocation), threat perception updates over time. |
| `worst`    | No-action counterfactual: decisions and threat-perception updates are off; used as the no-adaptation comparison. |

A **climate / CAT scenario is a different flood hazard**, which is injected
through the flood-depth inputs (below), independently of `--scenario`.

## 2. Injecting a climate / CAT flood scenario (the flood-depth files)

The flood hazard enters the model through two matching per-tract, per-year inputs
set in `config/abm_params.yaml`:

```yaml
files:
  depths_overall: config/overall_md_mean_by_tract_2011_2023.json
  depth_distributions: config/depth_distribution_by_tract_year.csv
```

To run your own hazard scenario:

1. **Prepare both depth inputs.** The aggregate file follows the same schema as
   the shipped
   (`config/overall_md_mean_by_tract_2011_2023.json`). See
   [DATA_FORMATS.md](DATA_FORMATS.md#flood-depth-file) for the exact schema. In
   short: a JSON array of one record per census tract,
   `{"CensusTract": "<11-digit GEOID>", "11_mean": <depth_m>, "12_mean": ..., ...}`,
   where each `YY_mean` column is the mean flood depth (meters) for year `20YY`.
   The distribution CSV contains `year`, `tract_geoid`, and `depth_values_m`,
   where `depth_values_m` is a JSON array of within-tract grid-cell depths in meters.
   Both files must have matching tract-year coverage.
2. **Point the model at both files** by editing `files.depths_overall` and
   `files.depth_distributions`, or keep a separate params file per scenario.
3. **Run** (the `--no-plots` data path is the reliable one — see the main README):
   ```bash
   python main.py --scenario baseline --no-plots --output-mode full
   ```
   Outputs land under `outputs/baseline/` (`decisions/`, `finance/`,
   `vulnerability/`, `states/`).

That is the entire hazard-scenario mechanism. The same Bayesian behavioral model
and all behavioral parameters are reused; only the flood-depth inputs differ.

> **GEOID alignment:** the `CensusTract` GEOIDs in your
> depth files must match the `tract_geoid` values in `config/households_for_abm_sfha.csv`
> and `config/sfha_shares.csv`. The model stops when the
> two depth inputs have different tract-year coverage or inconsistent tract means.
> Verify household GEOID alignment before trusting a new-area run.

## 3. For future years (post-2023)

Running beyond 2023 (e.g. a 2024–2050 climate projection) works through the same
depths file, but there is a **silent-truncation trap** in the default year mode.
This is important enough to have its own page: see
[FUTURE_SIMULATION.md](FUTURE_SIMULATION.md) before running future years.

## 4. Other scenario levers (CLI flags and parameters)

Besides the hazard, these are designed, intentional knobs:

| Lever | How | Effect |
|---|---|---|
| Flood-loss decision threshold | `--thr-owner`, `--thr-renter` | Loss-ratio at which threat perception rises and actions trigger |
| Threat-perception reactivity | `--shock-owner`, `--shock-renter` | Multiplier on how much a flood raises threat perception |
| Adoption stochasticity | `--decision-threshold` (and `draw_bounds` in YAML) | Adoption acceptance window |
| Monte Carlo ensemble | `scripts/utilities/run_mc100_local.py` | Varies the action-decision seed and Bayesian posterior draw while keeping initialization and the flood-depth seed fixed |

The sensitivity-analysis scripts under `scripts/sensitivity/` are the canonical
list of parameters the study treats as designed knobs. For the full
calibrated-vs-tunable map (what is safe to change vs what would invalidate the
calibration), see [PARAMETERS.md](PARAMETERS.md).

## See also
- [FUTURE_SIMULATION.md](FUTURE_SIMULATION.md) — running beyond 2023 (read the warning)
- [DATA_FORMATS.md](DATA_FORMATS.md) — exact input-file schemas
- [PARAMETERS.md](PARAMETERS.md) — calibrated vs tunable parameters; when to recalibrate
