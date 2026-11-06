# GPS-VDI — visual diversity of flying insects from sticky-trap scans

This repository holds the analysis code for a method that estimates the diversity of flying
insects directly from scanned yellow sticky traps, without identifying the specimens. Each
detected arthropod is reduced to five visual features; the spread of those features across a
trap is summarised by a grid-based entropy estimator, giving a per-trap visual diversity index.
The index is validated against expert taxonomic identification of a 40-trap subset, and then
applied to the full set of field traps.

It accompanies the manuscript by Cabra-Leykin et al. The trap images, the expert
identification table and the ecological analysis scripts are deposited separately
(see **Data**).

## Reproducing the analysis

Fetch the data, then run one notebook.

```
pip install -r requirements.txt
python fetch_data.py --dest ./data
```

`fetch_data.py` pulls the deposit from Mendeley Data over its public API — no account, token
or key is needed. By default it takes the two tables and the 40 annotated trap scans, which
is everything the notebook reads; `--what tables` takes the tables alone (about 30 MB) and
`--what all` adds the field scans (about 6 GB, not needed here). Every file is checked against
the SHA-256 published in the deposit listing, and files already present are skipped, so an
interrupted fetch resumes.

Then run `GPS_VDI_reproduction.ipynb` with two environment variables set:

```
GPSVDI_DATA_ROOT   the folder fetch_data.py populated   (required)
GPSVDI_REPO_ROOT   path to this repository              (defaults to the working directory)
GPSVDI_WORK        where intermediate files are written (defaults to <repo>/gpsvdi_output)
```

### In Colab

The notebook detects Colab, clones this repository, installs the two extra packages and runs
`fetch_data.py` for you, so the only thing to do is run the cells.

One thing is worth setting first. A Colab runtime deletes everything under `/content` when the
session ends, and a full pass through this notebook is a few hours of computation, so an
interrupted session otherwise starts again from nothing. Every expensive step writes its result
to disk and reloads it on a later run, but only if that disk survives. Setting

```python
DRIVE_CACHE = "/content/drive/MyDrive/gpsvdi"
```

at the top of the setup cell keeps the downloaded data and the intermediates in your own Google
Drive, so a second session resumes instead of restarting. Left empty, as it ships, nothing is
mounted and no permission is requested - the notebook simply runs, and anything it computes is
lost with the runtime.

Section 3's sweeps are the long part and are better run on a local machine.

The notebook is organised in five sections:

| section | what it does |
|---|---|
| 0 | setup and analysis parameters |
| 1 | recomputes the five visual features from the trap scans and verifies them |
| 2 | computes GPS-VDI and compares it with expert taxonomy |
| 3 | sensitivity tests: estimator choice, feature ablation, label permutation, grid search |
| 4 | field application across the full trap set, ending in a table for the R analyses |

Section 1 recomputes features for every annotated arthropod, which takes roughly a quarter of
an hour; the result is cached as `features_annotated.parquet` in the work directory, so later
runs skip it. Section 3 is the expensive part: the cost of the grid estimator grows as the
fifth power of the number of grid positions per dimension, so the notebook ships with
deliberately small defaults for the sweeps. The behaviour does not change with larger values —
only the runtime — and the cells say so where the values are set.

Section 4 does not re-segment the field scans. It reads the deposited detection table, whose
features were produced by the same pipeline; section 1.3 is the check that the two agree.

## Repository layout

```
GPS_VDI_reproduction.ipynb   the end-to-end analysis, start here
fetch_data.py                populates a data folder from the Mendeley deposit
gpsvdi/                      feature extraction from trap scans (segmentation + features)
BiodiversityUtils/           diversity estimators, grid machinery, data handling
plot_metadata.csv            plot-level site, habitat, crop and climate metadata
dependency_map.md            where each piece of the pipeline came from
test_reproduction.py         regression test: gpsvdi reproduces the reference features
R/README_R.md                the columns the ecological analyses expect
```

`gpsvdi` is self-contained: given a trap scan, a list of bounding boxes and the scan
resolution, `extract_trap_features()` returns the five features used throughout. It replaces an
earlier dependency on an external database system, and `dependency_map.md` records what came
from where.

The ecological mixed models behind the manuscript's figures are in R and are deposited with
the data rather than kept here. The notebook's interface to them is a single file: it writes
`per_trap_diversity.csv` and stops. `R/README_R.md` documents the columns those scripts expect.

## Data

The trap scans, the expert identification table, the detection table and the ecological
analysis scripts are deposited in Mendeley Data:

> Cabra-Leykin, L. and Keasar, C. *Insect biodiversity assessment*. Mendeley Data.
> https://doi.org/10.17632/6pbgs93kdf

That DOI follows the current version of the deposit. `fetch_data.py` names the deposit version
this release was verified against and says so if the deposit has since moved on.

`plot_metadata.csv` in this repository supplies the plot-level metadata used to attach habitat
and season; it is the manuscript's Table 1 in machine-readable form.

## Releases

Each release is a frozen state of the analysis that was run end to end against a stated version
of the deposit, and carries its own DOI. The manuscript cites the release that produced its
numbers, so that citation stays reproducible no matter how the method develops afterwards.

Later releases may change both the code and the results. When they do, the release notes say
what changed and whether any published number moved. A reader who arrives at an older release
through the manuscript will see on its archive page that newer ones exist.

Cutting one:

1. Run the notebook end to end against the deposit version the release will name, and record
   what it produced in `CHANGELOG.md`.
2. Set `version` in `.zenodo.json` to the release number.
3. Tag `vMAJOR.MINOR.PATCH` and give the GitHub release a title that names what the release
   is for — for v1.0.0, the paper it accompanies.
4. Paste the `CHANGELOG.md` entry into the release description.
5. Once the accompanying article has a DOI, add it to `related_identifiers` in
   `.zenodo.json` as `isSupplementTo`, beside the data DOI, so the archive links the three
   objects together.

## Current limitations

- **The sensitivity sweeps ship at reduced settings** so the notebook completes in a reasonable
  time. The values used for the published figures are given in the comments beside them.
- **Embedding-based estimators are present but unused.** `BiodiversityUtils` can score traps
  from neural-network embeddings; the manuscript does not, and that path needs packages this
  analysis does not otherwise require.

## Citation and licence

Please cite the manuscript, the Mendeley Data deposit above, and this software. `CITATION.cff`
carries the machine-readable form.

This code is released under the BSD 3-Clause licence; see `LICENSE`.
