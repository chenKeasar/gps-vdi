# Changelog

## v1.0.0 — the state published with Cabra-Leykin et al. (2026)

The commit that produced the manuscript's figures and tables. It was run end to end against
version 2 of the data deposit, <https://doi.org/10.17632/6pbgs93kdf>. A complete cold run takes
roughly two hours on a sixteen-core workstation; section 3's sensitivity sweeps are the larger
part of that, and every expensive step caches its result, so an interrupted run resumes.

What that run establishes:

- **The features are reproduced from the pixels.** Section 1.3 recomputes all five visual
  features from the trap scans and compares them with the values the production pipeline
  stored, for the 6,312 annotated arthropods on the 30 validation traps that are also field
  traps. Segmentation is bit-identical — `insect_size[pixels]` differs by exactly zero. The
  colour and size features agree to about 5e-9, which is the precision at which the deposited
  table records them rather than a limit of the computation, and the singular-value ratios to
  about 3e-5, where LAPACK implementations differ.
- **The validation covers the whole annotated set.** Section 2 compares the index with expert
  taxonomy across all 40 annotated traps and 8,200 identified arthropods.
- **The field application covers the whole survey.** Section 4 reads the deposited detection
  table: 91,984 detections, 91,981 of them with complete features, across 322 traps.
- **All twelve correlations in Table 2** round to the values printed in the manuscript.

### Reproducing it

```
pip install -r requirements.txt
python fetch_data.py --dest ./data
```

then run `GPS_VDI_reproduction.ipynb`. `fetch_data.py` reads the deposit over its public API;
no account, token or key is required. In Colab the notebook does all of this itself.
