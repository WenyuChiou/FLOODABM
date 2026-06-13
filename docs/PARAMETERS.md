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
| Psychology Beta shapes (`beta_parameters`) | Phase-1 survey fit (N = 557 owner / 379 renter) | Population property. **See the warning below — the YAML block is not the live source.** |
| `tp_decay_params` (per-tenure α, β, tau0, tau_inf, k) | Phase-3 grid search | Empirical threat-perception decay (shipped truth: owner `tau_inf=33.2589`, renter `46.0431`) |
| `finance` premium rates, `reserve` (1.15), `owner_contents_ratio` (0.57) | NFIP actuarial | Rescale every premium, affordability, and CAT damage result |
| `insurance_init.take_rate_by_tract_group` | Empirical NFIP penetration | Sets initial insurance; **also acts as the de-facto flood-prone classifier** (some plots key off the literal `owner == 0.25` value), so editing it can relabel headline figures |
| `owner_share` | ACS tenure split | Sets the owner/renter agent counts per tract |
| `flood.FFE_ft` (1.0), `files.*` | Data bindings / first-floor-elevation assumption | Calibration constants |

> **`beta_parameters` is effectively dead config.** The live psychology Beta
> shapes are in `modules/actions/tp.py` (`BETA_PARAMS_OWNER_DEFAULT` /
> `BETA_PARAMS_RENTER_DEFAULT`), and the per-household initial values are already
> baked into `config/households_for_abm.csv`. Editing the `beta_parameters` YAML
> block alone will **not** change a run. To change the psychology distribution you
> must edit `tp.py` and regenerate the household file (see §3).

> **Two different "alpha/beta".** `tp_decay_params.alpha/beta` are decay-weighting
> constants; `beta_parameters` are Beta-distribution shape parameters for the
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
| `hazard.*` method choices | YAML | Hazard-to-damage method options |
| `seed`, `insurance_init.seed` | YAML | New Monte Carlo realization (distribution preserved) |
| `years` | YAML | Simulated years — **only with matching depth columns** ([FUTURE_SIMULATION.md](FUTURE_SIMULATION.md)) |

## 3. Reuse vs recalibrate

**Reuse the shipped Passaic models as-is** when region, population, and survey
instrument are unchanged and only the hazard/climate scenario varies. Treat
`calibration/` as a read-only provenance record.

**Recalibrate (full Phase 1 → 2 → 3)** only when **any** of these differ: study
region/basin, population demographics, or survey instrument. Three plug-in tiers:

1. **Own numbers, no rerun.** Overwrite `tp_decay_params` in the YAML and drop
   replacement `.npz` model files into `models/baseline/` (and `models/baseline_fast/`).
   For the psychology Beta shapes, edit `modules/actions/tp.py`
   (`BETA_PARAMS_*_DEFAULT`) and regenerate `config/households_for_abm.csv` via
   `scripts/utilities/generate_household_psych.py` — editing the YAML alone does
   nothing (§1).
2. **Own survey, reuse Phase 2/3.** Build `data.xlsx` (sheets `owner_variable` /
   `renter_variable` with columns FI, EH, BP, RL, TP_mean, CP_mean, SP_mean,
   SC_mean, PA_mean, Source) and `cal_data.xlsx` (`owner_cal` / `renter_cal`), then
   run Phases 2–3. **See the known limitation below.**
3. **Own raw survey.** Phase 1 is not reusable as shipped (hardcoded paths and
   instrument-specific question maps).

### Write-back map (where each calibration output goes)
- Phase 1 (Beta shapes) → `modules/actions/tp.py:BETA_PARAMS_*_DEFAULT` **(not the YAML)** + regenerate the household CSV.
- Phase 2 (Bayesian regression) → `.npz` into `models/baseline/` + `models/baseline_fast/`.
- Phase 3 (TP decay) → `tp_decay_params` in `config/abm_params.yaml` (and `calibration/phase3_tp_decay/outputs/tp_decay_params_for_abm.json`).

## 4. Known calibration-pipeline limitations (documented, not fixed)

These are recorded so an external team is not surprised; they are **not** changed
here because this version is under peer review.

- **Phase 1 is not runnable as shipped:** its raw survey input is not in the repo
  and the path points at a personal location.
- **Phase 2 has a broken import:** `calibration/bayesian_engine/data.py` does
  `from src.util_2 import load_data`, but no `src/` package or `util_2` module is
  shipped, so `python -m bayesian_engine.main` raises `ImportError` immediately.
  To re-run Phase 2 from the processed `data.xlsx`/`cal_data.xlsx`, supply your own
  `load_data` loader.

Reusing the shipped models (the common case for climate scenarios) is unaffected by
either limitation.

## See also
- [SCENARIOS.md](SCENARIOS.md) · [FUTURE_SIMULATION.md](FUTURE_SIMULATION.md) · [DATA_FORMATS.md](DATA_FORMATS.md)
- `calibration/README.md` — the original calibration pipeline notes
