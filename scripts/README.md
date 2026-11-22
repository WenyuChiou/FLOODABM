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
| `rebuild_wrr_spatial_series.py` | Recompute and verify the tract data plotted in Figure 10 and Supporting Figure S7 |
| `rebuild_wrr_action_intervals.py` | Recompute Supporting Figure S4 from the audited interval cases and matched main-ensemble controls |
| `rebuild_wrr_gul_threshold_series.py` | Recompute Supporting Figure S5 from the audited ratio-threshold cases and matched main-ensemble controls |
| `rebuild_wrr_sequence_series.py` | Recompute Supporting Figure S6 curves and bands from the verified external sensitivity archive |
| `build_wrr_mc_variance_table.py` | Assemble and check the 455 annual quantile entries cited in Supporting Information Text S7 |
| `render_wrr_figures.py` | Redraw Figures 5–9 and 11 as PNGs from bundled or explicitly selected rebuilt aggregates; use Figure 9's separate environment |
| `render_wrr_aep_png.py` | Redraw Figure 7 as one PNG directly from exceedance data and verify every pixel |
| `render_wrr_supporting_figures.py` | Reproduce and verify the final S4–S6 PNGs from bundled or explicitly selected rebuilt plot tables |
| `render_wrr_supporting_raster.py` | Reproduce Figure 10 and final S1–S3 and S7 PNGs from pinned archive layers; S1–S3 use installed Windows Arial fonts |
| `export_wrr_spatial_fi_png.py` | Verify the 162 Figure 10 classes or S7 population changes and export the exact revised PNG from the corresponding editable ArcGIS archive |
| `render_wrr_vector_figures.py` | Optional archive replay: read the pinned study components and produce PNGs for Figures 4, 6, or 7 |
| `verify_wrr_figures.py` | Compare decoded PNG pixels to the submitted figure reference |
| `generate_household_psych.py` | Generate psychometric parameters for a custom inventory with `--csv INPUT.csv --output NEW.csv`; preserves the input and the shipped inventory |
| `export_arcgis_excel.py` | Export tract-level data to Excel for ArcGIS Pro |
| `generate_tract_summary_excel.py` | Create comprehensive tract summary table |
