# Scripts

The current benchmark runner is `utilities/run_mc100_local.py`. The
[figure checks](../docs/WRR_FIGURE_CHECKS.md) refer to the revised figures.
The former `legacy_plots/` directory is available in Git history at
`4ca700d020389cffcb4eb063263afc79af7aaff8`; those plots do not reproduce the
revised paper.

## utilities/

One-time data generation, export, and analysis scripts. Run as needed.

| Script | Purpose |
|--------|---------|
| `generate_household_psych.py` | Generate household psychometric parameters for the household file configured by `files.households` |
| `export_arcgis_excel.py` | Export tract-level data to Excel for ArcGIS Pro |
| `generate_tract_summary_excel.py` | Create comprehensive tract summary table |
| `gen_zscore_validation.py` | Generate z-score validation figure (internal) |
| `_calc_zscore_r2.py` | Compute R2 across threshold scenarios |
| `run_mc_extend.py` | Extend Monte Carlo from 15 to 30 runs |
| `run_montecarlo.py` | Standalone MC runner (replaced by `main_mc.py`) |
