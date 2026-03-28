# Scripts

Plotting and utility scripts organized by purpose. These are **not** part of
the main paper pipeline (`generate_paper_figures.py`) but are kept for
reference and future reuse.

## legacy_plots/

Older or alternative figure scripts, superseded by the current paper pipeline.
Use these if you need a specific visualization style that was dropped from the
final figures.

| Script | What it draws | Replaced by |
|--------|--------------|-------------|
| `plot_fig4_damage_loss_ratio.py` | Cumulative damage/loss ratio | `plot_fig_rq1_combined.py` (Fig 4+5) |
| `plot_fig4_dual_yaxis.py` | Dual Y-axis damage + loss | `plot_fig_rq1_combined.py` (Fig 4+5) |
| `plot_fig6_combined.py` | 2x2 cumulative + annual | `plot_fig_cumulative_behavior.py` (Fig 6) |
| `plot_fig6_event_study_allactions.py` | Tract-level action trajectories | `plot_fig_cumulative_behavior.py` (Fig 6) |
| `plot_fig_finance_combined.py` | Finance 3x2 layout | `plot_fig_rq1_combined.py` (Fig 5) |
| `plot_fi_by_flood_prone_paper.py` | FI by flood-prone status | `plot_fig_rq1_combined.py` |
| `plot_montecarlo_results.py` | MC visualization | `plot_mc_convergence_s2.py` |
| `plot_action_composition_paper.py` | Stacked bar action shares | `plot_fig_cumulative_behavior.py` |
| `plot_draw_bounds_explainer.py` | Bounded draw mechanism diagram | (conceptual, not in paper) |
| `replot_baseline.py` | Regenerate all baseline figures | `generate_paper_figures.py` |

## utilities/

One-time data generation, export, and analysis scripts. Run as needed.

| Script | Purpose |
|--------|---------|
| `generate_household_psych.py` | Generate household psychometric parameters for `config/households_for_abm.csv` |
| `export_arcgis_excel.py` | Export tract-level data to Excel for ArcGIS Pro |
| `generate_tract_summary_excel.py` | Create comprehensive tract summary table |
| `gen_zscore_validation.py` | Generate z-score validation figure (internal) |
| `_calc_zscore_r2.py` | Compute R2 across threshold scenarios |
| `run_mc_extend.py` | Extend Monte Carlo from 15 to 30 runs |
| `run_montecarlo.py` | Standalone MC runner (replaced by `main_mc.py`) |
