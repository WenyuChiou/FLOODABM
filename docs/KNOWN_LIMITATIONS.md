# Known limitations and engineering caveats

This page is honest documentation for anyone **reusing or extending** FLOODABM.
It records places where the code is rigid, where a configuration knob looks live
but is not, and where a behavior is fixed in code rather than exposed in
`config/abm_params.yaml`.

**Scope:** these are *engineering / extensibility* caveats. The published Passaic
River Basin results are produced by the shipped configuration and shipped models
as-is; the items below matter when you change the study area, swap the hazard,
add an action, or otherwise depart from the shipped setup. None of them changes a
default-configuration run.

If you only want to run a climate/CAT scenario in the same study area, you do not
need most of this — see [SCENARIOS.md](SCENARIOS.md) and
[PARAMETERS.md](PARAMETERS.md). This page is for people who want to change *how
the model works*, not just *what hazard it sees*.

---

## 1. Configuration for the revised benchmark

`main.py` reads `config/abm_params.yaml` using `utils._helpers.load_yaml_cfg`.
Tract means and household depths load from `files.depths_overall` and
`files.depth_distributions`, respectively. The benchmark preflight requires
`hazard.depths_mode: grid_hist`. The revised run uses the 2011–2023 span; a
separate `files.flood_events` calendar is optional input for `years.mode: events_only`.
The household inventory supplies tenure directly.

---

## 2. Single-run and ensemble entry points

`main.py` runs a single scenario; the revised ensemble invokes it through
`scripts/utilities/run_mc100_local.py`. A standalone default run and a numbered
ensemble draw use different Bayesian coefficient selections. Use the
[reproduction guide](REPRODUCIBILITY.md) for the paired scenario comparison
and the seeded design; the earlier Monte Carlo drivers remain in Git history.

---

## 3. The action set is hardcoded, not a registry

The four actions (`FI`, `EH`, `BP`, `RL`) and the per-tenure decision trees
(owner: FI→EH→BP; renter: FI→RL) are not data-driven. Adding, removing, or moving
an action between tenures is a multi-layer source edit, not a config change. The
action strings and per-tenure structure are re-encoded in **several layers**:

- inference action list (`modules/actions/bayes_fast_predictors.py` `ACTIONS`) **and**
  a separate training action list (`calibration/bayesian_engine/config.py`, with
  ~15 RL-specific calibration special-cases);
- the decision trees (`modules/actions/decision.py` `sequential_decision_fast`);
- per-action physical effects (`modules/actions/pipeline.py` `_apply_action_dynamics`
  — EH elevation, BP buyout-removal, RL relocation are each bespoke);
- finance (FI-only: `has_FI` is derived from `action == "FI"` across
  `modules/finance/`; EH/BP/RL have no financial coupling);
- the simulation output writers under `utils/`, and the figure-specific
  summaries and renderers.

Realistic blast radius for a new action: roughly 25–35 sites across these layers
plus newly trained model files. There is no shortcut.

**The one clean extension seam:** `modules/actions/decision.py` `load_predictors`.
You can replace the Bayesian predictor with your own model (random
forest, neural net, rule table, …) by returning two callables that honor the
`predict(*, TP, CP, SP) -> {action: prob_array}` contract — no retraining of the
shipped engine required. This is clean **only** for a like-for-like predictor that
emits the same four actions with the same meanings.

---

## 4. The decision model has a fixed 3-feature set

The Bayesian adoption model uses exactly `[TP, CP, SP]`
(`modules/actions/bayes_fast_predictors.py` `FEATURES`). The trained weight files
in `models/baseline/` carry coefficients for only these three drivers.

- The survey measured five psychological variables (TP, CP, SP, SC, PA). `SC` and
  `PA` are loaded but feed **only** the TP-decay rate (`modules/actions/tp.py`),
  never the decision.
- Adding a new decision driver (e.g. income, prior losses) requires **retraining**
  to produce new weight files; editing `FEATURES` alone yields zero weights for the
  new driver (a silent no-effect — see §5).
- The retraining pipeline is **not runnable as shipped**: `calibration/bayesian_engine/data.py`
  imports `from src.util_2 import load_data`, but no `src/` package is included, so
  Phase 2 raises `ImportError` on import. Phase 1 raw survey input is also not in
  the repo. See [PARAMETERS.md](PARAMETERS.md) §4. Reusing the shipped models (the
  common case) is unaffected.

---

## 5. Silent-failure modes when extending

These do not affect a default run, but will produce **wrong results with no error**
if you extend the model without knowing them:

- **Untrained action → p = 0.5.** `modules/actions/bayes_fast_predictors.py`
  `_load_group_models` skips an action whose model file is missing, leaving zero
  weights (`sigmoid(0) = 0.5`). This is intentional for tenure-unavailable actions
  (`owner_RL`, `renter_EH`, `renter_BP` are deliberately absent), but it means a new
  action added to `ACTIONS` without a trained model silently adopts at ~50%.
- **New household column dropped at load.** `utils/_helpers.py` `load_households_csv`
  keeps a whitelist of psychology columns (`TP_init … PA_init`, `has_FI`); any other
  per-household column you add to `households_for_abm_sfha.csv` is silently discarded
  before it reaches the simulation.
- **Unmapped action dropped from outputs.** The simulation output writers under
  `utils/` map a fixed set of action strings; an action not in the map
  becomes `NaN` and disappears from summaries, Excel exports, and figures, even if it
  was simulated.
---

## 6. Modeling settings that live in code, not config

A few analysis/behavior settings are fixed in source rather than exposed in
`abm_params.yaml`. These are the values used in the published runs; change them by
editing the cited location.

| Setting | Where | Note |
|---|---|---|
| Flood-prone threshold (positive flood depth in at least 7 of 13 years) | `utils/flood_prone.py` (`FLOOD_PRONE_MIN_YEARS`) | Defines flood-prone and non-prone tracts for the benchmark. The repository plots Figures 8 and 9 from the saved group-level summaries. |
| Tenure median incomes (owner 148,097 / renter 63,699) | `scripts/utilities/rebuild_wrr_finance_series.py` (`MEDIAN_INCOME_USD`) | ACS income values for normalizing losses in Figure 11. Use values from the new study area when adapting this analysis. |
| Relocation contents retention | `modules/actions/pipeline.py` (`_attach_rl_dest_by_depth` region) | Relocated renters currently retain 100% of contents value; the contents-reduction draw described in the docstring is disabled in the shipped behavior. |
| Recorded elevation amount | `modules/actions/decision.py` `sequential_decision_fast` (inline literal `5.0` on the `out_elev[take_EH]` line) | In the default `eh_one_time` mode the actual elevation is drawn from `action_dynamics.eh_once_min/max_ft` and this recorded value is not used. |
| Decision-tree topology and RL destination count | `modules/actions/decision.py`, `main.py` (`rl_dest_k_best`) | Per-tenure action order and the relocation candidate count are fixed in code; only the per-step draw windows/thresholds are config-tunable. |

---

## See also
- [SCENARIOS.md](SCENARIOS.md) · [FUTURE_SIMULATION.md](FUTURE_SIMULATION.md) · [DATA_FORMATS.md](DATA_FORMATS.md) · [PARAMETERS.md](PARAMETERS.md)
