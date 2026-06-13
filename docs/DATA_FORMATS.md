# Input data formats

To run FLOODABM on a new hazard scenario, future years, or a new study area, you
replace one or both of these inputs. All paths are set in `config/abm_params.yaml`.

## Flood-depth file

Set by `files.depths_overall` in `config/abm_params.yaml`. Shipped example:
`config/overall_md_mean_by_tract_2011_2023.json`. Parsed by
`modules/actions/vuln_for_tp.py:load_depths_all_years` (via `pd.read_json`).

**Schema** — a JSON array, one object per census tract:

```json
[
  {"CensusTract": "34013021300", "11_mean": 0.0, "12_mean": 0.0, "13_mean": 0.0,
   "14_mean": 0.0, "15_mean": 0.0, "16_mean": 0.0, "17_mean": 0.0, "18_mean": 0.0,
   "19_mean": 0.0, "20_mean": 0.0, "21_mean": 1.83, "22_mean": 0.0, "23_mean": 0.0},
  ...
]
```

| Field | Type | Meaning |
|---|---|---|
| `CensusTract` | string | 11-digit census tract GEOID (keep as a string to preserve leading digits) |
| `YY_mean` | float | Mean flood depth (**meters**) for the tract in year `20YY`. One column per simulated year. |

- The loader keeps every column matching `^\d\d_mean$`, melts to a long table
  (`tract_geoid`, `year`, `depth_m`), and decodes the year as `2000 + YY`. So
  `24_mean` → 2024, `50_mean` → 2050 (two-digit years span 2000–2099).
- Missing/blank depths are coerced to `0.0`.
- For a future-climate file, add `24_mean … NN_mean` columns. The set of years the
  model runs is derived from this file — see [FUTURE_SIMULATION.md](FUTURE_SIMULATION.md).

A CSV with columns `tract_geoid, year, depth_m` is also accepted by the legacy
loader path; the JSON wide format above is the canonical, tested one.

## Household file

Set by the household-inventory path in `config/abm_params.yaml`. Shipped:
`config/households_for_abm.csv`. One row per household agent.

| Column | Type | Units / range | Meaning |
|---|---|---|---|
| `tract_geoid` | int/str | 11-digit GEOID | Census tract the household is in |
| `state` | str | e.g. `NJ` | State label |
| `i` | int | — | Household index |
| `group` | str | `owner` / `renter` | Tenure (drives which actions are available and the finance model) |
| `hh_idx_within_group` | int | — | Index within tenure group |
| `rcv_kUSD` | float | thousands USD | Structure replacement-cost value (owners) |
| `contents_kUSD` | float | thousands USD | Contents value |
| `TP_init` | float | 0–1 | Initial threat perception |
| `CP_init` | float | 0–1 | Initial coping perception |
| `SP_init` | float | 0–1 | Initial stakeholder perception |
| `SC_init` | float | 0–1 | Initial social capital (moderates threat-perception decay) |
| `PA_init` | float | 0–1 | Initial place attachment (moderates threat-perception decay) |
| `has_FI` | 0/1 | binary | Whether the household starts with flood insurance |

`TP_init`, `CP_init`, `SP_init` feed the Bayesian adaptation-decision model;
`SC_init` and `PA_init` moderate how threat perception decays between floods.
For a continued future run, seed these from the last historical year's `states/`
output (see [FUTURE_SIMULATION.md](FUTURE_SIMULATION.md)).

## GEOID alignment (required, unchecked)

The census-tract GEOIDs must be consistent across **all** of:

- the depths file (`CensusTract`),
- `config/households_for_abm.csv` (`tract_geoid`),
- `owner_share` and `insurance_init.take_rate_by_tract_group` in `config/abm_params.yaml`.

Tracts that appear in one input but not another are **silently dropped** — the
model does not raise an error. When swapping in a new study area or scenario,
verify the tract sets match before trusting the results.

## See also
- [SCENARIOS.md](SCENARIOS.md) · [FUTURE_SIMULATION.md](FUTURE_SIMULATION.md) · [PARAMETERS.md](PARAMETERS.md)
