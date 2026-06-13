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

## 1. Inert ("dead") configuration surfaces

Some files and keys look like live configuration but are not read at runtime. The
live config path is the inline YAML read in `main.py` / `main_mc.py` via
`utils._helpers.load_yaml_cfg`. These are marked in-source as well.

| Surface | Status | Live source instead |
|---|---|---|
| `core/config.py`, `core/cli.py`, `utils/config_loader.py` | **Not imported at runtime.** A typed-dataclass loader + a second CLI parser that the live runners never call. | `main.py` builds its own inline argparse parser; YAML via `utils._helpers.load_yaml_cfg`. |
| `abm_params.yaml: beta_parameters` block | **Dead.** Editing it has no effect. | Live initial-psychology Beta shapes are in `modules/actions/tp.py` (`BETA_PARAMS_OWNER_DEFAULT` / `BETA_PARAMS_RENTER_DEFAULT`) and pre-baked into `config/households_for_abm.csv`. See [PARAMETERS.md](PARAMETERS.md). |
| `abm_params.yaml: hazard` block, `flood.events_mode`, `flood.min_trigger`, `files.grid_depths_yaml` | **No live consumer.** | Depths load from `files.depths_overall`. |

Practical effect: do not wire a new tool to `core/config.py` or `core/cli.py`, and
do not expect edits to the keys above to change a run.

---

## 2. Two entry points that do not honor the same settings

There are two runners, and they are **not** behaviorally identical:

- `main.py` — single deterministic/stochastic run.
- `main_mc.py` — the Monte Carlo driver that **produces every paper figure**
  (via `generate_paper_figures.py`), using its own fixed internal settings
  (documented below) that match the published-results configuration.

Differences a reuser must know:

| Setting | `main.py` | `main_mc.py` (figures) |
|---|---|---|
| `years_step` | read from YAML | **hardcoded to 1.0** (YAML value ignored) |
| `flood.ratio_threshold_owner` / `_renter` | read (per-tenure TP-shock trigger) | **not used**; the MC path gates on a single `tp_config.shock_min_ratio` key that is **absent from the shipped YAML**, falling back to `flood.ratio_threshold` |
| `reset_clock_on_flood` | hardcoded `True` | hardcoded `False` |

Consequence: tuning some YAML keys changes a `main.py` run but **not** the
figure-producing MC runs. If your goal is to alter the published-style results,
verify the knob is honored in `main_mc.py`, not just `main.py`.

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
- **two** output/figure stacks (`utils/` driven by `main.py`, plus `main_mc.py`),
  and the per-paper-figure scripts under `scripts/paper_figures/`.

Realistic blast radius for a new action: roughly 25–35 sites across these layers
plus newly trained model files. There is no shortcut.

**The one clean extension seam:** `modules/actions/decision.py` `load_predictors`.
You can replace the Bayesian predictor wholesale with your own model (random
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
  per-household column you add to `households_for_abm.csv` is silently discarded
  before it reaches the simulation.
- **Unmapped action dropped from outputs.** The output tallies (e.g. `main_mc.py`
  and `utils/` writers) map a fixed set of action strings; an action not in the map
  becomes `NaN` and disappears from summaries, Excel exports, and figures, even if it
  was simulated.
- **GEOID mismatch dropped silently.** Tracts present in one input but not another
  (depths file / household file / `owner_share` / `insurance_init`) are dropped with
  no error. See [DATA_FORMATS.md](DATA_FORMATS.md).

---

## 6. Modeling settings that live in code, not config

A few analysis/behavior settings are fixed in source rather than exposed in
`abm_params.yaml`. These are the values used in the published runs; change them by
editing the cited location.

| Setting | Where | Note |
|---|---|---|
| Flood-prone threshold (a tract is "flood-prone" if it floods in ≥ N of 13 years) | `scripts/paper_figures/plot_fig8_tp_by_prone.py` (`FP_THRESHOLD = 7`) | Drives the flood-prone vs non-prone partition in the TP figures. |
| Tenure median incomes (owner 148,097 / renter 63,699) | `scripts/paper_figures/plot_fig11_income_aep.py` | Drives the income-normalized equity figure; ACS values for the shipped study area. Re-targeting the model requires updating these. |
| Relocation contents retention | `modules/actions/pipeline.py` (`_attach_rl_dest_by_depth` region) | Relocated renters currently retain 100% of contents value; the contents-reduction draw described in the docstring is disabled in the shipped behavior. |
| Recorded elevation amount | `modules/actions/decision.py` `sequential_decision_fast` (inline literal `5.0` on the `out_elev[take_EH]` line) | In the default `eh_one_time` mode the actual elevation is drawn from `action_dynamics.eh_once_min/max_ft` and this recorded value is not used. |
| Decision-tree topology and RL destination count | `modules/actions/decision.py`, `main.py` (`rl_dest_k_best`) | Per-tenure action order and the relocation candidate count are fixed in code; only the per-step draw windows/thresholds are config-tunable. |

---

## See also
- [SCENARIOS.md](SCENARIOS.md) · [FUTURE_SIMULATION.md](FUTURE_SIMULATION.md) · [DATA_FORMATS.md](DATA_FORMATS.md) · [PARAMETERS.md](PARAMETERS.md)
