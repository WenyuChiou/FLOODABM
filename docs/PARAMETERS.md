# Parameters, calibration, and when to recalibrate

This page answers: *which parameters can an external team change, which must stay
frozen, and do they need to recalibrate or can they reuse the shipped models?*

Short answer for a **climate / CAT scenario in the same study area (Passaic River
Basin)**: **reuse everything as-is** — change only the flood-depth input
([SCENARIOS.md](SCENARIOS.md)) and, optionally, the scenario levers in the "tunable"
table below. No recalibration is needed; the behavioral model is a property of the
surveyed population, not the hazard.

## 1. Keep frozen — calibrated / region-specific

Changing these without recalibrating silently invalidates the published results.

| Parameter (in `config/abm_params.yaml` unless noted) | Provenance | Why frozen |
|---|---|---|
| Initial household psychology | Phase-1 survey fit (557 homeowners and 379 renters) | The published household CSV stores the fixed draws. |
| `tp_decay_params` (per-tenure α, β, tau0, tau_inf, k) | Phase-3 grid search | Empirical threat-perception decay (shipped truth: owner `tau_inf=33.2589`, renter `46.0431`) |
| `finance` premium rates, `reserve` (1.15), `owner_contents_ratio` (0.57) | NFIP actuarial | Rescale every premium, affordability, and CAT damage result |
| Preassigned `inside_SFHA` and `has_FI` in `households_for_abm_sfha.csv`; `sfha_shares.csv` | Tract SFHA area fraction and literature-informed initial FI rates | Defines the revised agent initialization and must remain fixed for reproduction |
| Household tenure in `households_for_abm_sfha.csv` | Synthetic household inventory | Determines homeowner and renter counts by tract; no separate tenure-share setting is used. |
| `flood.FFE_ft` (1.0), `files.*` | Data bindings / first-floor-elevation assumption | Calibration constants |

> The Beta shapes used to generate new initial psychology draws are defined in
> `modules/actions/tp.py` (`BETA_PARAMS_OWNER_DEFAULT` and
> `BETA_PARAMS_RENTER_DEFAULT`). The benchmark reads the fixed values in
> `config/households_for_abm_sfha.csv`.

> **Two different "alpha/beta".** `tp_decay_params.alpha/beta` are decay-weighting
> constants; the Beta-distribution shape parameters describe the
> initial psychology draws. They are unrelated — do not conflate them.

## 2. Safe to tune — scenario levers

Each has a CLI flag or a sensitivity-analysis script confirming it is a designed
knob. These are what you sweep for a scenario study; they do not require
recalibration.

| Lever | Where | Purpose |
|---|---|---|
| `tp_config.shock_scale_owner/renter`, `shock_timing` | YAML / `--shock-owner`, `--shock-renter` | Threat-perception reactivity to floods |
| `flood.ratio_threshold(_owner/_renter)` | YAML / `--thr-owner`, `--thr-renter` | Loss-ratio that triggers action |
| `draw_bounds` (FI/EH/BP/RL), `decision_threshold` | YAML / `--decision-threshold` | Adoption acceptance window / stochasticity |
| `policy` (deductibles, limits, coinsurance) | YAML | The insurance design you sweep |
| `action_dynamics` toggles, `eh_*` elevation policy | YAML | Which actions are active / elevation rules |
| `files.depths_overall`, `files.depth_distributions` | YAML | A matched alternative flood hazard for a scenario analysis |
| Action-decision seed and Bayesian posterior draw | `scripts/utilities/run_mc100_local.py` | Formal 50-draw ensemble; initialization and flood-depth seed remain fixed |
| `years` | YAML | Simulated years — **only with matching depth columns** ([FUTURE_SIMULATION.md](FUTURE_SIMULATION.md)) |

## 3. Reuse vs recalibrate

**Reuse the shipped Passaic models as-is** when region, population, and survey
instrument are unchanged and only the hazard/climate scenario varies. Treat
`calibration/` as a read-only provenance record.

For another study area, population, or survey instrument, reassess whether
the Passaic calibration applies. Use a separate checkout for custom settings:

1. **Own numbers, no rerun.** Overwrite `tp_decay_params` in the YAML and drop
   replacement `.npz` model files into `models/baseline/` (and `models/baseline_fast/`).
   For the psychology Beta shapes, edit `modules/actions/tp.py`
   (`BETA_PARAMS_*_DEFAULT`) in your custom model, then run
   `python scripts/utilities/generate_household_psych.py --csv INPUT.csv --output NEW.csv`.
   Use a separate configuration whose `files.households` points to the new CSV.
   Keep the shipped inventory unchanged for benchmark reproduction. Adding
   `--yaml CUSTOM.yaml` uses your configuration when validating or assigning initial
   insurance status; it does not override the Python Beta shapes (§1).
2. **Own survey.** Derive any new parameters from authorized data and a
   documented fitting workflow. The published refitting source and input
   requirements are described below.

### Write-back map (where each calibration output goes)
- Phase 1 (Beta shapes) → `modules/actions/tp.py:BETA_PARAMS_*_DEFAULT` **(not the YAML)** + regenerate the household CSV.
- Phase 2 (Bayesian regression) → `.npz` into `models/baseline/` + `models/baseline_fast/`.
- Phase 3 (TP decay) → `tp_decay_params` in `config/abm_params.yaml` (and `calibration/phase3_tp_decay/outputs/tp_decay_params_for_abm.json`).

## 4. Refitting constraints

The inputs and code needed for a full training rerun are not included:

- Phase 1 used instrument-specific survey preprocessing. The original survey
  is not distributed, and the workstation-specific tools remain in Git history.
- Phase 2 fitting source is retained for provenance. It depends on the
  original `data.xlsx` (`owner_variable` and `renter_variable` sheets) and
  the `src.util_2.load_data` loader; neither is included. Use the
  [posterior predictive query](../calibration/README.md) to examine the
  shipped models.
- Optional Phase 3 fitting requires an authorized `cal_data.xlsx` in
  `calibration/data/` (`owner_cal` and `renter_cal` sheets). Run it in a
  separate checkout to keep the shipped parameters and reference files intact.

The [revised benchmark](REPRODUCIBILITY.md) runs with its shipped models
and input files.

## See also
- [SCENARIOS.md](SCENARIOS.md) · [FUTURE_SIMULATION.md](FUTURE_SIMULATION.md) · [DATA_FORMATS.md](DATA_FORMATS.md)
- [Calibration notes](../calibration/README.md) — fitted parameters and historical methods
