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

Figures 8 and 11 can be redrawn from their published aggregate series. Install
`requirements-benchmark.txt` in Python 3.14.0. Exact pixel reproduction was
tested on Windows with Times New Roman; another font or renderer can change
the output pixels.

```sh
python scripts/utilities/render_wrr_figures.py --figure-dir ../wrr-rendered
python scripts/utilities/verify_wrr_figures.py --figure-dir ../wrr-rendered --figures 8 11
```

The two commands reproduce the plotting stage from the saved group-level
series, not the ensemble aggregation that produced the series.

## Final revision sources and current reproduction coverage

| Final figure | Source and reproduction boundary |
| --- | --- |
| 1 | Study-area map with tract SFHA shares and white-outlined county labels; ArcGIS Pro layout and geodata. The archived project has unresolved layer paths and its re-export did not match the final PNG. |
| 2 | Coupled framework; final diagrams.net drawing. |
| 3 | Adoption-probability diagram; editable PowerPoint artwork and mathematical insets. |
| 4 | Flood-model diagnostics; editable PowerPoint layout using the original maps and hydrograph panels. |
| 5 | County NFIP diagnostic; use the corrected Morris/Passaic labels, not the original CSV ordering. A render with the corrected inputs did not match the final export pixels. |
| 6 | SQ1 six-panel loss and finance figure. Recomposition of the late-review component vectors from the study archive matched the final PNG pixels. The component assets and exact fonts are not bundled here. |
| 7 | SQ2 two-panel aggregate exceedance probability figure, with the revised protection-margin labels. Recomposition from the archived vector matched the final PNG pixels; those files are not bundled here. |
| 8 | Adoption trajectories. The included renderer uses the final aggregate series and the last legend and annotation edits; its regenerated PNG matches the final pixels. |
| 9 | Threat-perception trajectories; current aggregate series plus late typography and layout edits. The tested renderer did not match the final pixels. |
| 10 | Spatial flood-insurance comparison; ArcGIS Pro layout, joined tract geometry, and aggregate scenario results. The archived project has unresolved layer paths and its re-export did not match the final PNG. |
| 11 | Income-normalized AEP, with the revised income-relative protection-margin labels. Redrawing it from the bundled aggregate series matches the final PNG pixels. |

The [simulation benchmark](REPRODUCIBILITY.md) and this image check are
separate. A clone contains the simulation inputs, figure reference checksums,
and the series needed to redraw Figures 8 and 11. It does not contain the complete final-generation GIS projects,
office drawings, intermediate vectors, or late layout assets. Historical
plotting scripts do not fill those gaps.

For the annualized losses in Table S7 and response Table R3, use the respective
full-precision aggregate medians and divide by 13 years and the **initial**
number of households in that tenure group (41,517 owners; 10,624 renters).
For the annual decision-rule comparison in Table S9 and response Table R6,
pair the ten runs by seed, divide cumulative differences by the same initial
group counts, convert to thousands of dollars, and summarize the paired
differences using the median and IQR. Using current-year counts instead does
not reproduce those tables.
