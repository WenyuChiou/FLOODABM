# Running FLOODABM beyond 2023 (future-behavior + CAT simulation)

The model is not hardwired to 2011–2023. The simulated years are **derived from
the flood-depth file** (`utils/_helpers.py:years_from_cfg`), so a future-climate
run is: supply future flood depths, then select the future years. The behavioral
model and parameters are reused unchanged.

There is one trap that fails **silently**. Read §1 before running.

## 1. ⚠️ The silent-truncation trap (read this first)

`config/abm_params.yaml` has a `years` block with a `mode`:

- **`mode: span`** (the shipped default) returns
  `[y for y in range(start, end+1) if y in <years present in the depths file>]`
  (`utils/_helpers.py:133`). It **silently intersects** your requested span with
  the years that actually exist in the depths file.

  → If you set `years.end: 2050` but your depths file still only has
  `11_mean`…`23_mean`, the run **silently executes 2011–2023 with no error and no
  warning.** You get a "future" run that is not future.

- **`mode: explicit`** returns the sorted, deduplicated contents of `years.list`,
  **without intersecting against the depths file** (`utils/_helpers.py:130`). This
  is the safe, explicit path for future years — but every listed year **must** have
  a matching `YY_mean` column in the depths file, or the downstream load will error
  / produce NaNs. (Order does not matter; the list is sorted internally.)

## 2. Recipe — a 2024–2050 climate run

1. **Build a future depths file.** Same schema as
   `config/overall_md_mean_by_tract_2011_2023.json`
   (see [DATA_FORMATS.md](DATA_FORMATS.md#flood-depth-file)): one record per tract,
   with `YY_mean` columns for every future year — `24_mean`, `25_mean`, … `50_mean`
   (years are decoded as `2000 + YY`, so two-digit years cover 2000–2099). Depths in
   meters, from your CAT model under the chosen RCP/SSP pathway. Keep the same
   `CensusTract` GEOIDs as the household file (see the GEOID-alignment warning in
   [SCENARIOS.md](SCENARIOS.md#2-injecting-a-climate--cat-flood-scenario-the-flood-depth-file)).
   - A starter helper that scales the historical depths into a future file is in
     [`scripts/examples/make_future_depths.py`](../scripts/examples/make_future_depths.py).
2. **Point the model at it** in `config/abm_params.yaml`:
   ```yaml
   files:
     depths_overall: config/your_future_depths.json
   ```
3. **Select the years explicitly** (the safe path):
   ```yaml
   years:
     mode: explicit
     list: [2024, 2025, 2026, ..., 2050]
   ```
4. **Run:**
   ```bash
   python main.py --scenario baseline --no-plots --output-mode full
   ```
   Confirm the printed `[info] YEARS: [...]` line matches your intended future
   years — this is your check that the depths file actually covers them.

## 3. What carries the trajectory into the future

- **Initial state** comes from `config/households_for_abm.csv` (`TP_init`,
  `has_FI`, `rcv_kUSD`, …). For a true future run that continues from a past run,
  seed the initial state from the last historical year's `states/` output (or set
  the initial columns accordingly).
- **Threat perception** evolves each year via decay (`tp_decay_params`) between
  floods and a shock on flood years (`tp_config.shock_scale_*`). Decay parameters
  are calibrated (keep frozen); the shock multipliers are tunable scenario knobs.
- **Adaptation decisions** update through the Bayesian model in `models/baseline/`
  as threat perception, coping, and stakeholder perceptions change. No retraining
  is needed for a future hazard in the same region.

## 4. Economic-parameter caveat (state it in your methods)

NFIP premium/deductible rates, coverage limits, and replacement-cost values are
**not year-indexed** — they are held at their calibrated (≈present-day) levels for
all simulated years. A future run therefore assumes current insurance terms and
property values unless you change them deliberately (see
[PARAMETERS.md](PARAMETERS.md)). Document this assumption in any future-projection
study.

## See also
- [SCENARIOS.md](SCENARIOS.md) — the hazard-injection mechanism
- [DATA_FORMATS.md](DATA_FORMATS.md) — depths-file and household-file schemas
- [PARAMETERS.md](PARAMETERS.md) — calibrated vs tunable parameters
