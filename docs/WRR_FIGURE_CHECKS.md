# Checking reproduced WRR figures

The figure reference uses the final high-resolution reviewer-revision PNGs, not
the copies embedded in Word. Render or export figures at the same resolution
and name them `Figure_01.png` through `Figure_11.png`. To compare them:

```sh
python scripts/utilities/verify_wrr_figures.py --figure-dir ../wrr-rendered
```

The checker accepts only static PNGs. It compares their decoded RGBA pixels and
dimensions, ignoring lossless encoding differences, color profiles, and other
metadata. By default it requires every figure. To check a subset, add
`--figures 6 7 11`. This command does not regenerate figures. Matching pixels
do not establish the provenance of the candidate image.

## Render from the bundled data

Figures 5, 8, and 11 can be redrawn from their published aggregate series. Install
`requirements-benchmark.txt` in Python 3.14.0. Exact pixel reproduction was
tested on Windows with Times New Roman; another font or renderer can change
the output pixels.

```sh
python scripts/utilities/render_wrr_figures.py --figure-dir ../wrr-rendered
python scripts/utilities/verify_wrr_figures.py --figure-dir ../wrr-rendered --figures 5 8 11
```

The two commands reproduce the plotting stage from the saved group-level
series, not the ensemble aggregation that produced the series.

Figure 9 uses a separate Windows x64 Conda environment because the accepted
render depends on Matplotlib 3.10.0 and FreeType 2.12.1. The lock pins the
renderer builds; it does not change the simulation dependencies. Use Times
New Roman as with the other figures.

```sh
conda create -n wrr-figure9 --file environment-figure9-win64.lock
conda run -n wrr-figure9 python scripts/utilities/render_wrr_figures.py --figure-dir ../wrr-rendered --figures 9
python scripts/utilities/verify_wrr_figures.py --figure-dir ../wrr-rendered --figures 9
```

## Replay from the archived PDF components

Place the following unchanged study-archive files in one local directory.
Their SHA-256 hashes are pinned in `scripts/utilities/render_wrr_vector_figures.py`;
earlier exports with the same filename are rejected.

| Source file | Final output |
| --- | --- |
| `Fig04_flood_inudation.pdf` (final-label export; archive filename spelling) | Figure 4 at 600 dpi |
| `Fig06_cumulative_loss.pdf` and `Fig07_finance_per_household.pdf` (label-corrected vector components) | Figure 6: merge the cumulative panel pair and the annual finance panels at 300 dpi |
| `Fig07_finance_per_household.pdf` (same input as Figure 6) | Figure 7: separate the loss AEP panels, keep all four vector curves, and apply the final protection-margin labels at 600 dpi |

```sh
python -m pip install -r requirements-vector-figures.txt
python scripts/utilities/render_wrr_vector_figures.py --components ../wrr-pdf-components --figure-dir ../wrr-rendered
python scripts/utilities/verify_wrr_figures.py --figure-dir ../wrr-rendered --figures 4 6 7
```

The renderer verifies every selected output against the final PNG reference
before writing any images and refuses to overwrite existing files. To replay
one figure, pass `--figures 4`, `--figures 6`, or `--figures 7`. Figures 6 and
7 use the fonts already embedded in the finance PDF, so no separate font files
are needed. The PDFs are not bundled. This route reproduces the layout and
rendering stage; it does not rerun the flood diagnostics or regenerate the
finance series from simulation CSVs.

## Final revision sources and current reproduction coverage

| Final figure | Source and reproduction boundary |
| --- | --- |
| 1 | Study-area map with tract SFHA shares and white-outlined county labels. The displayed layout layers resolved locally, and the 27 tract classes agreed with the model area fractions. The live ArcGIS re-export did not match the final pixels. |
| 2 | Coupled framework; final diagrams.net drawing. |
| 3 | Adoption-probability diagram; editable PowerPoint artwork and mathematical insets. |
| 4 | Flood-model diagnostics. The final-label PowerPoint PDF was replayed to the exact accepted pixels using the vector renderer. |
| 5 | County NFIP diagnostic. The bundled renderer uses the corrected Morris/Passaic labels and the final aggregate series; the regenerated PNG matches the final pixels. |
| 6 | SQ1 six-panel loss and finance figure. The vector renderer recomposes the two PDF components from the study archive to the exact final pixels. |
| 7 | SQ2 two-panel aggregate exceedance probability figure, with the revised protection-margin labels. The vector renderer reproduces the exact final pixels from the original finance PDF. |
| 8 | Adoption trajectories. The included renderer uses the final aggregate series and the last legend and annotation edits; its regenerated PNG matches the final pixels. |
| 9 | Threat-perception trajectories. The final aggregate series and 17-point legend reproduce the exact PNG with the separate pinned environment. |
| 10 | Spatial flood-insurance comparison. The displayed ArcGIS layout layers resolved locally and the visual dimensions matched after restoring the original trim and border. The re-export still differs in a small region near the legend. |
| 11 | Income-normalized AEP, with the revised income-relative protection-margin labels. Redrawing it from the bundled aggregate series matches the final PNG pixels. |

The [simulation benchmark](REPRODUCIBILITY.md) and this image check are
separate. A clone contains the simulation inputs, figure reference checksums,
and the series needed to redraw Figures 5, 8, 9, and 11. Replaying Figures 4, 6,
and 7 requires the separate archived PDFs above. The complete native GIS and
office export workflows are not included.

For the annualized losses in Table S7 and response Table R3, use the respective
full-precision aggregate medians and divide by 13 years and the **initial**
number of households in that tenure group (41,517 owners; 10,624 renters).
For the annual decision-rule comparison in Table S9 and response Table R6,
pair the ten runs by seed, divide cumulative differences by the same initial
group counts, convert to thousands of dollars, and summarize the paired
differences using the median and IQR. Using current-year counts instead does
not reproduce those tables.
