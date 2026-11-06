# Dependency map — from STARdbi-coupled code to a standalone package

## The result in one line

All five manuscript features regenerate from the trap scans plus bounding-box
coordinates, with **no STARdbi, Django or database involvement**, matching STARdbi's
stored values to floating-point noise.

## Validation (150 boxes, 3 traps, against the STARdbi export)

| feature | max abs difference | interpretation |
|---|---|---|
| `insect_size[pixels]` | **0** | segmentation is bit-identical |
| `log_insect_size` | 1.8e-15 | float rounding only |
| `rb_ratio` | 1.8e-15 | float rounding only |
| `rg_ratio` | 2.2e-16 | float rounding only |
| `svd_ratio01` | 7.7e-07 | LAPACK singular-value tolerance |
| `svd_ratio12` | 8.0e-06 | LAPACK singular-value tolerance |

The table above compares against the full-precision export. The deposited detection table
records some rows at about nine significant digits, so a check against it (notebook section
1.3) is bounded at roughly 1e-8 for the colour and size features rather than 1e-15. The
segmentation remains bit-identical either way.

Features must be computed from the **original trap scans**, not the exported crop
JPEGs. Recomputing from the annotated crops gives r = 0.996-0.9993 rather than
equality, because those crops are lossy re-encodings. This decides the Mendeley
deposit: it must carry the trap images.

## What `gpsvdi/` contains

| file | origin | change |
|---|---|---|
| `_config.py` | `helpers/biodiversity_config.py` (STARdbi) | constants copied verbatim |
| `features.py` | `helpers/biodiversity_features.py` (STARdbi) | one import rewritten; formulas untouched |
| `segmentation.py` | `insect_coverage.py` (repository root) | added the missing `crop()` helper |
| `extract.py` | **new** | glue: trap scan + boxes -> five features |
| `__init__.py` | **new** | public surface: `extract_trap_features`, `FEATURE_SET` |

## What stays in `BiodiversityUtils`

These import cleanly and have no STARdbi coupling, so they are reused rather than
copied: `entropy_estimators.py` (Grid/KDE/KDE-loo), `biodiversity_estimators/`,
`biodiversity_estimation.py`, `Taxonomy_utils/`, `weighted_correlation.py`,
`analysing_utils/`.

## What is excluded, and why

| file | reason |
|---|---|
| `helpers/biodiversity_hooks.py` | writes features back to the STARdbi database (`traps_images.models`) |
| `models_static_functions.py` | STARdbi server-side Django model code |
| `Engineered_features_utils/calculate_features.py` | abandoned stub: syntax error at line 17, superseded by `gpsvdi.extract` |
| `Taxonomy_utils.get_df_with_features` | dead code, and computes the wrong features (per-channel SVD entropy, no `log_insect_size`) |

## Defects found along the way

1. **`merge_dataframes.py` line 38** calls `calculate_features(insect_images_df)` — a name
   neither imported nor defined. Crashes the merge path.
2. **`associate_insects_with_traps.py` line 5** indexes with a tuple rather than a list of
   column names. Even once fixed, it rebinds a local name and returns nothing, so it
   cannot affect its caller — it is a no-op containing a crash.
3. **`insect_detection_functions.py` is missing** from the repository, though
   `insect_coverage.py` imports it. Only its `crop()` helper is used; `gpsvdi/segmentation.py`
   now supplies it.

Items 1 and 2 were neutralised in memory for the baseline run. Neither has been edited
on disk.

## Cost

~0.29 s per box, dominated by `cv2.fastNlMeansDenoisingColored` in the segmentation.
For the 8,200 annotated insects that is roughly 40 minutes single-threaded — acceptable
for a one-off notebook run with a cached intermediate, but it should be checkpointed to
parquet so reviewers need not repeat it.
