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
| `run_mc100_local.py` | Run the revised paired scenarios with the canonical draw schedule; use `--start` and `--end` to select draws |
| `verify_wrr_finance.py` | Verify annual tract finance output against the revised benchmark |
| `rebuild_wrr_finance_series.py` | Recompute and check the simulated Figure 5–7 and 11 numeric series from the 50 paired draws |
| `build_wrr_mc_variance_table.py` | Assemble and check the 455 annual quantile entries cited in Supporting Information Text S7 |
| `render_wrr_figures.py` | Redraw the four supported figures from bundled aggregates |
| `render_wrr_vector_figures.py` | Reproduce Figures 4, 6, and 7 from the pinned external study-archive PDFs |
| `verify_wrr_figures.py` | Compare decoded PNG pixels to the submitted figure reference |
| `generate_household_psych.py` | Generate household psychometric parameters for the household file configured by `files.households` |
| `export_arcgis_excel.py` | Export tract-level data to Excel for ArcGIS Pro |
| `generate_tract_summary_excel.py` | Create comprehensive tract summary table |
| `gen_zscore_validation.py` | Generate z-score validation figure (internal) |
| `_calc_zscore_r2.py` | Compute R2 across threshold scenarios |
