# Checking reproduced WRR figures

The figure reference uses the final high-resolution reviewer-revision PNGs, not
the copies embedded in Word. The clone-only result plots are Figures 5–9 and
11 and Supporting Figures S4–S6. The standard workflow skips maps and
manuscript illustrations. The archive sections document optional checks for
previously produced material, including Figure 4's map and hydrograph panels.

Use the direct render instructions below. To verify the six main result plots
at the same resolution and with their original figure numbers:

```sh
python scripts/utilities/verify_wrr_figures.py --figure-dir ../wrr-rendered --figures 5 6 7 8 9 11
```

The checker accepts only static PNGs. It compares their decoded RGBA pixels and
dimensions, ignoring lossless encoding differences, color profiles, and other
metadata. By default it requires every main figure. Keep the explicit selection
when checking just the result plots. This command verifies existing images;
matching pixels do not establish their provenance.

Use the direct PNG routes below for new runs. They do not write PDF files.
The optional historical PDF replay reads existing archive components only;
it is not a prerequisite for the direct renderers.

## Render from the bundled data

Figures 5–8 and 11 can be redrawn from their published aggregate series.
Install `requirements-benchmark.txt` and `requirements-vector-figures.txt`
in Python 3.14.0 (the second file supplies the in-memory drawing dependency for Figures 6 and 7).
Exact pixel reproduction was
tested on Windows with Times New Roman; another font or renderer can change
the output pixels.

```sh
python scripts/utilities/render_wrr_figures.py --figure-dir ../wrr-rendered --figures 5 6 7 8 11
python scripts/utilities/verify_wrr_figures.py --figure-dir ../wrr-rendered --figures 5 6 7 8 11
```

The two commands reproduce the plotting stage from the saved group-level
series. To render Figures 5–7 or 11 from refreshed finance results, pass
`--series-dir` with the output directory of the
[finance rebuild](REPRODUCIBILITY.md#recompute-finance-figure-data-from-all-50-paired-draws).
Select only the figures whose required series are in that directory; the
renderer does not fill missing files from the bundled data.

Figure 9 uses a separate Windows x64 Conda environment because the accepted
render depends on Matplotlib 3.10.0 and FreeType 2.12.1. The lock pins the
renderer builds; it does not change the simulation dependencies. Use Times
New Roman as with the other figures.

```sh
conda create -n wrr-figure9 --file environment-figure9-win64.lock
conda run -n wrr-figure9 python scripts/utilities/render_wrr_figures.py --figure-dir ../wrr-rendered --figures 9
python scripts/utilities/verify_wrr_figures.py --figure-dir ../wrr-rendered --figures 9
```

The same optional `--series-dir` works with Figure 9 in its pinned environment.

### Main Figure 6 directly from the loss and finance series

With the same pinned Python and Times New Roman setup as the batch command,
run `render_wrr_figures.py --figure-dir ../wrr-rendered --figures 6`. The
renderer draws (a–b) from `Figure6_cumulative_loss.csv` and (c–f) from
`Figure6_annual_finance.csv`. It checks the entire 3,550 × 4,373 PNG against
the submitted pixels before writing it. The only output is `Figure_06.png`;
the command needs no archived components. To use the table data rebuilt from
the ensemble, pass their folder with `--series-dir`.

### Main Figure 7 directly from the exceedance series

Install both `requirements-benchmark.txt` and `requirements-vector-figures.txt`
in Python 3.14.0. With the installed Windows Times New Roman fonts, run:

```sh
python scripts/utilities/render_wrr_aep_png.py --figure-dir ../wrr-rendered
```

This redraws all four paths from the bundled `Figure7_aep.csv`, calculates
the endpoint labels and protection margins from the loss values, and checks
every decoded pixel against the submitted image before writing
`Figure_07.png`. Add `--series-dir ../wrr-recomputed-finance` to use the
[series rebuilt from the ensemble](REPRODUCIBILITY.md#recompute-finance-figure-data-from-all-50-paired-draws).
The command uses local regular and bold Times New Roman; `--font-dir`
overrides the default Windows Fonts folder. No archived PDF is needed and no
PDF is written. The PNG file encoding may differ even when its pixels match.

### Supporting Figures S4–S6

Use the same pinned Windows x64 environment and Times New Roman as Figure 9.
The repository includes the three accepted plot tables. Drawing from them
does not require the study archives:

```sh
conda run -n wrr-figure9 python scripts/utilities/render_wrr_supporting_figures.py --figure-dir ../wrr-rendered
```

The renderer verifies the exact final PNG bytes before writing any of the three
images and refuses to overwrite existing files. To replay a subset, use
`--figures 4 6` for Figures S4 and S6, for example. To render from the
[recomputed sensitivity series](REPRODUCIBILITY.md#recompute-the-threshold-sensitivity-series-figures-s4-and-s5),
specify the verified CSV files explicitly:

```sh
conda run -n wrr-figure9 python scripts/utilities/render_wrr_supporting_figures.py --figure-dir ../wrr-rendered-from-rebuild --s4-series ../wrr-s4-recomputed/Figure_R1_data.csv --s5-series ../wrr-s5-recomputed/Figure_R2_fixed_initial_data.csv --s6-series ../wrr-s6-recomputed/Figure_R3_fixed_initial_plot_data.csv
```

The rebuilt-data route needs the archived sensitivity cases. Drawing from the
bundled tables reproduces the plotting stage without rerunning those cases.

### Supporting Figures S1–S3 from the study archive

Use the original pre-labeling PNGs in `Sources/review_v2_20260909/sm_fonts/before/`
of the study figure archive. They are not included in a clone. The command
matches each input's SHA-256 before applying the last label and legend edits.
Use Python 3.14.0, Pillow 12.0.0 (pinned in `requirements-benchmark.txt`), and
the installed Windows Arial regular and bold fonts.

```sh
python scripts/utilities/render_wrr_supporting_raster.py --components ../wrr-raster-components --figure-dir ../wrr-rendered --figures 1
```

The command redraws only the approved label regions and checks the S1 PNG
against the final standalone supporting export. It preserves the prior
bootstrap curves; it does not recompute the calibration intervals. For an
optional archive check, `--figures 2 3` replays the corresponding map edits.
It does not simulate flood depths or recreate Word's embedded JPEG for S3.

Rebuilding these figures from their scientific inputs would require additional
sources. This checkout does not include the matching inputs and plotting
procedure for those builds:

| Figure | Inputs needed beyond the archived image |
| --- | --- |
| S1 | Bootstrap results for the plotted threat-perception decay medians and 95% intervals, or the inputs and procedure that produced those results. |
| S2 | Stage IV-forced and WRF-forced simulated depths for Hurricane Irene, the reference depth grid, and the grid comparison and plotting procedure. |
| S3 | The 2012–2023 annual maximum-depth grids, flood-model domain boundary, and map settings. |

### Supporting Figure S7 from the study archive

Use the pre-crop population map in the study figure archive,
`SM/FigS4_population_change.png` (historical numbering). With the same
Python and Pillow versions as above, run:

```sh
python scripts/utilities/render_wrr_supporting_raster.py --figures 7 --components ../wrr-s7-map-component --figure-dir ../wrr-rendered
```

The command crops the map and moves the flood-prone legend entry, reproducing
the final `FigS07_population_change.png` exactly. It uses the lettering already
in the archived raster and requires no fonts. The archive input is not bundled.
The [spatial data builder](REPRODUCIBILITY.md#recompute-the-data-for-the-spatial-maps)
separately rebuilds all six population-change fields from the model outputs;
this raster replay does not regenerate the GIS map layout from those fields.
Use the [editable ArcGIS route below](#supporting-figure-s7-from-the-editable-arcgis-project)
if you have the project and its linked layers.

### Main Figure 10 from the study archive

Use the full-page export in `Sources/maps/centered_key_exports/Fig10_spatial_fi.png`,
not the similarly named cropped export. With Python 3.14.0 and Pillow 12.0.0:

```sh
python scripts/utilities/render_wrr_supporting_raster.py --main-map 10 --components ../wrr-fig10-map-component --figure-dir ../wrr-rendered
```

The renderer checks the archived source, trims the full page, and adds the
94-pixel border to reproduce the accepted `Figure_10.png` exactly. The
[main figure checker](#checking-reproduced-wrr-figures) accepts that PNG
by its decoded pixels when run with `--figures 10`. This replay starts from
the archived GIS map export;
the [map data builder](REPRODUCIBILITY.md#recompute-the-data-for-the-spatial-maps)
verifies the underlying insurance data separately.

### Both spatial maps from public GIS data

Use this route if you have the two verified layout-only `.pagx` files. They
are not bundled in this checkout. The command needs their approved bytes
(names may differ), the two CSVs rebuilt by the
[spatial data builder](REPRODUCIBILITY.md#recompute-the-data-for-the-spatial-maps),
and a licensed ArcGIS Pro Python environment. The verified setup uses Windows,
ArcGIS Pro 3.2, and separate Python 3.14.0 with
`requirements-sfha-areas.txt` and Pillow 12.0.0. Use the blank project provided
by ArcGIS Pro 3.2 at `Resources/ArcToolBox/Services/routingservices/data/Blank.aprx`
within its installation. The command verifies that exact stock template and
checks the ArcGIS Pro 3.2 / Python 3.9 worker versions before opening a project.

```sh
python scripts/utilities/render_wrr_public_maps.py --series-dir ../wrr-spatial-data --figure10-layout /path/to/Figure10.pagx --s7-layout /path/to/FigureS7.pagx --arcgis-python /path/to/ArcGIS/python.exe --blank-project /path/to/Blank.aprx --output-dir ../wrr-spatial-png
```

The command rebuilds the tract, river and marker layers from public FEMA and
2022 Census TIGER/Line sources without opening the archived study geodatabase
or projects. It writes `Figure_10.png` and `FigS07_population_change.png` only
after all 324 plotted values and both revised PNG hashes match. The output
directory must be new; no PDF or GIS workspace is retained there.

To use a prepared offline cache, pass `--public-source-dir` containing
`fema_tracts.geojson`, `tl_2022_34_tract.zip`, and the
`tl_2022_<county>_<type>.zip` files for `34013`, `34027`, and `34031`,
with type `areawater` and `linearwater`. Both the online and cache routes
verify the revision's source identities.

### Main Figure 10 from the editable ArcGIS project

If you have the study archive's `Sources/maps/Figure10/Figure10_ready.aprx`
and its linked local layers, you can generate the exact Figure 10 PNG without
the archived bitmap. Install the benchmark dependencies in Python 3.14.0
and use a separate licensed ArcGIS Pro Python environment (verified with
ArcGIS Pro 3.2). Rebuild `Figure10_spatial_fi.csv` using the
[spatial data builder](REPRODUCIBILITY.md#recompute-the-data-for-the-spatial-maps),
then run:

```sh
python scripts/utilities/export_wrr_spatial_fi_png.py --project /path/to/Figure10_ready.aprx --series ../wrr-spatial-data/Figure10_spatial_fi.csv --arcgis-python /path/to/ArcGIS/python.exe --figure-dir ../wrr-rendered
```

Keep the project and the linked layers together as supplied. The command
checks the 162 bivariate classes used by the six visible map panels against
the rebuilt CSV. It positions the legend marker in memory without saving
any modification to the project, then exports, crops, and verifies the PNG.
The only retained output is `Figure_10.png`; no PDF is involved. The editable
project, linked layers, and ArcGIS Pro are external to the clone.

### Supporting Figure S7 from the editable ArcGIS project

Use `Sources/maps/FigureS4/FigureS4_ready.aprx` from the study archive
(the filename uses historical numbering). Keep its linked local layers
together and use the same benchmark Python and ArcGIS Pro environments.
First rebuild `FigureS7_population_change.csv` with the spatial data builder,
then run:

```sh
python scripts/utilities/export_wrr_spatial_fi_png.py --figure S7 --project /path/to/FigureS4_ready.aprx --series ../wrr-spatial-data/FigureS7_population_change.csv --arcgis-python /path/to/ArcGIS/python.exe --figure-dir ../wrr-rendered
```

The exporter checks the field used by each of the six visible map panels and
matches all 162 tract population changes to the rebuilt values at nine decimal
places. It exports the final legend already stored in the project, crops the
page, and writes `FigS07_population_change.png` only if its bytes match the
revised figure. The project is not saved. This route needs the external editable
project and linked layers, but does not need the archived raster.

## Verify the archived Figure 4 PNG

The hydrologic validation figure uses inundation maps and hydrographs from
outside the household simulation. If you have the final standalone PNG
`Fig04_flood_inudation.png` from the study archive, place a copy named
`Figure_04.png` in the verification directory and run:

```sh
python scripts/utilities/verify_wrr_figures.py --figure-dir ../wrr-rendered --figures 4
```

This verifies the supplied export, not the hydrologic model or its source
data. No slide application is needed for this check.

## Optional archive route (PNG output)

Place the following unchanged study-archive files in one local directory.
Their SHA-256 hashes are pinned in `scripts/utilities/render_wrr_vector_figures.py`;
earlier exports with the same filename are rejected. This route reads the archived
components and writes PNGs only. The archive route for Figures 6 and 7 is optional;
the direct CSV-to-PNG command above does not need these inputs.

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

## Figure 1 data coverage and PNG verification

The [overlay rebuild](REPRODUCIBILITY.md#recompute-the-sfha-area-shares)
reproduces all 27 revised tract coverage values using the two pinned layers.
In a controlled ArcGIS test, verified public FEMA tract geometries replaced the
27 drawn tract polygons. The resulting pixels matched the original layer's
render with the online basemap turned off and the other layers unchanged.

The [Esri World Topographic Map style](https://r.esri.com/arcgisutils/reference/arc_portal_urls.html)
is an online dependency of the detail frame. The live ArcGIS re-export does not
match the revised PNG. For exact image verification, supply the archived revised
`Figure_01.png` to the [figure checker](#checking-reproduced-wrr-figures)
with `--figures 1`.

## Final revision sources and current reproduction coverage

| Final figure | Source and reproduction boundary |
| --- | --- |
| 1 | Study-area map with tract SFHA shares and white-outlined county labels. See the [Figure 1 checks](#figure-1-data-coverage-and-png-verification). |
| 2–3 | Framework and adoption-probability illustrations. The checker accepts the submitted PNGs; these illustrations are not outputs of a simulation run. |
| 4 | Flood-model diagnostics. [Verify the final archived PNG](#verify-the-archived-figure-4-png) separately from the household simulation. |
| 5 | County NFIP diagnostic. The bundled renderer uses the corrected Morris/Passaic labels and the final aggregate series; the regenerated PNG matches the final pixels. |
| 6 | SQ1 six-panel loss and finance figure. The [CSV-to-PNG renderer](#main-figure-6-directly-from-the-loss-and-finance-series) rebuilds all six panels at the exact submitted pixels. The study-archive replay remains an alternative. |
| 7 | SQ2 two-panel aggregate exceedance probability figure. The [data-driven PNG renderer](#main-figure-7-directly-from-the-exceedance-series) rebuilds the curves and protection-margin labels at the exact submitted pixels. The finance PDF replay remains an alternative. |
| 8 | Adoption trajectories. The included renderer uses the final aggregate series and the last legend and annotation edits; its regenerated PNG matches the final pixels. |
| 9 | Threat-perception trajectories. The final aggregate series and 17-point legend reproduce the exact PNG with the separate pinned environment. |
| 10 | Spatial flood-insurance comparison. The [map data builder](REPRODUCIBILITY.md#recompute-the-data-for-the-spatial-maps) reproduces all 162 bivariate cell assignments. Both the pinned archive replay and the [verified native ArcGIS export](#main-figure-10-from-the-editable-arcgis-project) produce the exact final PNG. |
| 11 | Income-normalized AEP, with the revised income-relative protection-margin labels. Redrawing it from the bundled aggregate series matches the final PNG pixels. |

The [simulation benchmark](REPRODUCIBILITY.md) and this image check are
separate. A clone contains the simulation inputs, figure reference checksums,
and the sources needed to redraw Figures 5–9 and 11. Figures 1, 4, 10, S1–S3,
and S7 require separately supplied materials for the stated checks. The
public GIS route for Figures 10 and S7 needs only their layout files, rebuilt
series, and public GIS inputs. The layouts, native GIS projects, and
hydrologic modeling projects are not included in the clone.

The same spatial builder reconstructs the six percentage-change fields
plotted in Supporting Figure S7. The original GIS inputs contain ancillary
columns that are not used in that figure.

The [annualized loss table builder](REPRODUCIBILITY.md#recompute-the-annualized-loss-tables-table-s7-and-response-table-r3)
derives Table S7 and response Table R3 from the last points of the Figure 6
cumulative profiles, retaining the distinct units and precision of each table.
The [paired sensitivity table builder](REPRODUCIBILITY.md#recompute-the-paired-sensitivity-tables-tables-s8-and-s9)
reconstructs Tables S8–S9 and response Tables R5–R6 from the initial-cohort
denominators and the ten matched draws. Using current-year counts instead
does not reproduce those tables.
