# R analyses — mixed models for habitat and season

This directory holds the R code behind **manuscript Figures 6 and 7** and the GLMM results in
section 3.4. It is deliberately separate from the Python notebook.

## The interface

The notebook writes exactly one file, and this code reads exactly that file:

```
per_trap_diversity.csv
```

Nothing else crosses the boundary. The notebook does not call R, does not import anything from
this directory, and does not read anything these scripts produce. **Deleting this entire
directory leaves the notebook fully runnable.**

## Column contract

One row per sticky trap.

| column | type | meaning | example |
|---|---|---|---|
| `trap_id` | integer | STARdbi identifier for the trap scan | 140 |
| `trap_name` | text | scan filename, `site_plot_replicate_dates.jpg` | 6_1_2_10-15.8.22.jpg |
| `experiment` | text | sampling round as recorded in STARdbi | Agrovoltaic round 2 2022 |
| `location` | text | site name as recorded in STARdbi | Merom Golan |
| `placement_date` | date `YYYY-MM-DD` | date the trap was set out | 2022-08-10 |
| `site` | text | site name matched to manuscript Table 1 | Merom Golan |
| `site_n` | integer | site number parsed from the filename | 6 |
| `plot_n` | integer | plot within the site (1-3) | 1 |
| `rep_n` | integer | replicate trap within the plot (1-3) | 2 |
| `n_plots_in_site` | integer | plots at this site; 3 means it has a natural plot | 3 |
| `habitat` | factor | `Natural` or `Agriculture` | Agriculture |
| `season` | factor | `spring` (Apr-Jun) or `summer` (Jul-Sep) | summer |
| `n_insects` | integer | arthropods detected on the trap — the abundance response | 101 |
| `visual_diversity` | double | GPS-VDI index for the trap | 4.3863 |
| `diversity_z` | double | `visual_diversity` z-transformed over all traps | 2.0313 |

Notes for the models:

1. **Random effect.** `site` is the grouping factor used in the manuscript ("site identity was
   treated as a random effect"). Use `site`, not `location` — `site` is reconciled against
   Table 1, whereas `location` is the raw STARdbi string and its spelling varies.
2. **Pairing.** Manuscript section 2.6 restricts the models to paired traps (n = 140). That
   subset is **not** pre-computed here, because the rule that produces 140 is not recorded
   anywhere in the data. Whatever rule you apply, apply it in R from these columns.
3. **Traps that cannot be placed.** A trap whose filename does not follow
   `site_plot_replicate` has `site`, `plot_n` and `habitat` empty. The notebook prints a
   warning listing any such trap rather than dropping it silently. Decide explicitly in R
   whether to exclude them.
4. **Distributions**, following the manuscript: diversity with a Gaussian family and identity
   link; abundance with a Poisson family and log link.

## Software versions used in the manuscript

| | |
|---|---|
| R | 4.2.2 |
| lme4 | Bates et al. 2015 |
| lmerTest | Kuznetsova et al. 2017 |

## Status of this directory

The scripts themselves live in the Mendeley deposit, folder **"R script and data"**, and should
be copied here so that code and data stay together. Until then this README is the contract they
must honour.

## How to change or remove this

Three cases, all safe:

1. **Edit the scripts in place.** Nothing in the notebook depends on their internals. Only the
   column names above are load-bearing.
2. **Replace them entirely** — a different modelling approach, a different language, a
   collaborator's own version. Anything that reads `per_trap_diversity.csv` works.
3. **Remove this directory.** The notebook still runs end to end and still produces
   `per_trap_diversity.csv`; the manuscript would then cite the mixed models as performed
   elsewhere.

If the column contract itself needs to change, that is the one case requiring a notebook edit —
section 4.4 is where the file is written.
