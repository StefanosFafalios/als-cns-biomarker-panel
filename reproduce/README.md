# ALS Biomarker Discovery — Reproducibility

This folder contains everything needed to reproduce the figures, tables, and
statistics in the accompanying manuscript and supplementary, starting from
public GEO/SRA inputs.

```
reproduce/
├── README.md                  # this file
├── FIGURE_MANIFEST.md         # figure / table -> producing script mapping
├── environment_als.yml        # conda environment (Python 3.13, LightGBM 4.5, etc.)
├── download_geo_data.sh       # one-shot GEO downloader (-> ../resources/)
├── quantify_srp064478.sh      # Salmon quantification for SRP064478 (~1 h, optional)
├── run_all.sh                 # top-level entry point — runs every step
└── logs/                      # per-step stdout/stderr written here at runtime
```

## Quick start

```bash
# 1. Build the conda environment (Python 3.13, LightGBM, SHAP, gseapy, ...)
conda env create -f reproduce/environment_als.yml
conda activate als-cns-panel

# 2. Download GEO and ArrayExpress data (~3 GB)
bash reproduce/download_geo_data.sh

# 3. Optionally quantify SRP064478 with Salmon (~1 h; needed for Step 22)
bash reproduce/quantify_srp064478.sh

# 3b. Optional: place the two single-nucleus .h5 cohorts (GSE219280, GSE212630)
#     under ../resources/<GSE>/suppl/ — see Provenance. Steps 31/32 skip without them.

# 4. Run all manuscript-cited analyses (≈ 30 h end-to-end, single 24-core box)
bash reproduce/run_all.sh
```

## The critical panel is defined in exactly one place

`panel_critical.csv` is the single source of truth for panel membership. It is
written by step **24a** (`greedy_significance_stop.py`) and read by every
downstream script through `utils.critical_panel_symbols()`,
`critical_panel_features()`, `load_critical_panel()`, and
`critical_panel_directions()`. No script hardcodes gene lists, panel size, or
up/down direction of effect.

Two consequences when resuming a partial run:

- Step 24a is **load-bearing**. `run_all.sh from 25` onwards assumes
  `panel_critical.csv` already exists; if it does not, those steps fail fast with
  a pointer back to 24a rather than silently using stale membership.
- Panel size is *derived*, never a constant. Changing the non-inferiority margin
  in `greedy_significance_stop.py` changes membership, and every figure, caption
  and table regenerates to match without further edits. Read the size with
  `utils.critical_panel_size()` and the label with `critical_panel_label()`;
  `test_critical_panel_label.py` fails the build if any script types it out
  instead. (This bullet itself said "not fixed at 15 in perpetuity" until
  2026-08-10, months after the panel became 18 — the literal it warns against,
  in the document warning against it.)

The same rule covers derived *values*, not just membership: the 25-gene panel's
within-cohort CV AUC comes from `utils.panel25_cv_auc()`, which reads the
incremental curve at k=25. It was a hardcoded `0.9621` in four scripts and
reached the abstract one full run after the pipeline started producing 0.9611.

## Run modes

`run_all.sh` accepts three optional modes. It is the only runnable pipeline definition in
this repository — targeted reruns use `only` rather than a second script:

| Mode | Command | Effect |
|---|---|---|
| Full (default) | `bash run_all.sh` | Runs everything end-to-end. The two BayesianOptimizer steps (1, 2b) are not in this repository; their tuned hyperparameters ship as `src/lgbm_best_params.json` and `src/lgbm_top500_best_params.json`. |
| Fast | `bash run_all.sh fast` | Skips Step 2 — needs `src/lgbm_prefilter_X.npy` and `src/lgbm_prefilter_names.txt` (~160 MB, not tracked). All downstream steps still rerun. |
| Resume | `bash run_all.sh from 24` | Skips every step whose numeric prefix is below `24` (base-10 numeric compare; the argument may be a bare number like `24` or a full step ID like `02c`). **Read the warning below before using this.** |
| Subset | `bash run_all.sh only 24a 25 07` | Runs **only** the listed step IDs, still in file order. Use after a bug fix invalidates part of the pipeline. This exists so a targeted rerun never needs a second script with its own copy of the dependency graph — `run_all.sh` is the only runnable pipeline definition in the repo. |

### Two gotchas that have both cost a wasted run

**`from` compares numbers; the file does not run in numeric order.** Section E
(steps 24--26a) sits *above* section F (06, 08, 13) in `run_all.sh`, so
"resume from where it stopped" and "resume from this number" are different
requests. If the run died at step 06, `from 06` re-runs 21--26a — which had
already finished — and skips nothing useful. **To resume after an
interruption, list the remaining step IDs with `only`**, which respects file
order; `from` is only safe when the numeric prefix and the file position happen
to agree.

**`only` matches the step ID prefix before the first underscore.** `only 24a`
is correct; `only 24a_greedy_signif_stop` matches nothing, and because no step
matches, the run prints "All steps completed" in under a second and exits 0.
A silent no-op looks exactly like a successful run — check the log count.

### Step 24a caches its scoring

`greedy_significance_stop.py` separates expensive scoring from cheap reporting. If
`greedy_significance_stop.csv` already contains every panel size it reuses those scores and
regenerates the report in seconds; scoring from scratch takes ~8 h. Because the cache keys on
nothing, it will happily be reused after a change that should have invalidated it —
**set `GREEDY_STOP_REFIT=1` to force a full refit** whenever the model, the cohorts, or the
25-gene panel change:

```bash
GREEDY_STOP_REFIT=1 python greedy_significance_stop.py
```

Each step writes outputs to `src/` alongside the script
(figures as `.png`, statistics as `.txt`, CSV tables as `.csv`) and logs to
`reproduce/logs/<step_id>.log`.

## Checking for stale outputs

Step 99 runs `panel_staleness_audit.py`, which flags any analysis output older than the
`panel_critical.csv` it depends on. It resolves the dependency by AST — direct readers of the
`utils` panel accessors, plus anything that transitively imports one — rather than by grepping
for gene symbols, because three separate rounds of stale results in this study were missed by
symbol-based greps. Run it on its own at any time:

```bash
python panel_staleness_audit.py
```

A clean tree prints no `STALE` rows. Two other flags exist so that permanent,
un-actionable rows do not train you to ignore the column that matters:

| flag | meaning |
|---|---|
| `n/a` | keyed to the 25-gene panel (`lgbm_core25_panel.csv`); a critical-panel redefinition does not affect it |
| `expl` | exploratory script that `run_all.sh` never invokes, so it is permanently older than `panel_critical.csv` |
| `?` | the script produces no artefact at all (only `panel_staleness_audit.py` itself should ever show this) |

Pipeline membership for the `expl` flag is derived by parsing `run_all.sh`, not
kept as a second list — adding or retiring a step needs no edit to the auditor.
Artefacts are located both by name prefix *and* by the filenames a script names
in its own source, because four steps write outputs not named after themselves
(`explainability_figures` → `explainability_covariates.png`, and similar) and
were silently unauditable — reported `?` rather than a freshness verdict — until
2026-08-10. **A `?` on anything other than the auditor itself means the audit
cannot see that step, not that the step is fine.** `STALE` means "prove this was rerun", not
"this is definitely wrong": transitively importing a panel-aware module is not the same as
depending on panel *membership*, so the report prints the import chain for each row.

## Rebuilding the PDFs

After `run_all.sh` finishes, regenerate the PDFs:

```bash
cd src/manuscript
python build_plos_variants.py          # derive the two PLOS files from manuscript.tex
for pass in 1 2; do                    # 2 passes each for xr cross-refs
  for f in supplementary manuscript manuscript_plos manuscript_plos_preview; do
    xelatex -interaction=nonstopmode "$f.tex"
  done
done
```

Then export the submission figures:

```bash
python export_plos_figures.py     # -> plos_submission/Fig1.tif ... Fig4.tif
```

PLOS is the submission target. `manuscript_plos.pdf` is deliberately
figure-free and is submitted alongside the numbered TIFFs; `manuscript_plos_preview.pdf`
embeds the figures for review only and is not submitted. The exporter reads
figure order from the preview document (so numbering follows first citation
automatically), flattens transparency onto white, and enforces PLOS's 300 dpi and
2.63--7.5 inch width limits, downscaling anything wider.

**`manuscript_plos.tex` and `manuscript_plos_preview.tex` are generated, not
edited.** They are `manuscript.tex` plus six PLOS-specific transformations
(`lineno`/`setspace`, the PLOS title block, `\linenumbers`, supplementary
cross-refs rewritten to the "S1 Appendix" convention, `Figure~\ref{fig:...}`
rewritten to PLOS's "Fig N" style, and figures stripped for the submission
variant). Editing them directly is how they previously drifted a
month behind the main manuscript; run `build_plos_variants.py` after any change
to `manuscript.tex` instead. Every substitution in that script asserts an
expected count, so a partial rewrite fails loudly rather than shipping.

`xelatex` (not `pdflatex`) is required because `supplementary.tex` uses Unicode
glyphs (★, ▲, ✓) in figure captions. Two passes per document are required
because `manuscript.tex` and `supplementary.tex` cross-reference each other via
the `xr` LaTeX package (`\externaldocument{...}` in each preamble).

## Step layout

`run_all.sh` is grouped into sections A–P by pipeline stage. The manuscript
(2026-10-05 rewrite) draws on these steps:

| Manuscript section | Steps |
|---|---|
| 3.1 Discovery and the 25-gene panel | 02, 02c, 02d, 03, 30b, 30c (the Bayesian searches, 01 and 02b, are not in this repository) |
| 3.2 Validation in nine external datasets | 04, 05, 05b, 22, 30a, 40, 24b, 24c, 24d |
| 3.3 Trimming to the critical panel | 24, 24a |
| 3.4 Composition, per-cell change, bulk direction | 11, 11a, 11b, 11c, 41, 41a, 42 |
| 3.5 GSTM1 deletion | 07b |
| 3.6 Specificity and clinical correlates | 10, 35, 38 (gated) |
| 3.7 Drug direction | 07, 07ac, 07c |
| Supplement: robustness, blood, context | 21a, 27, 27a, 28, 28aa, 28c, 14, 06, 07a, 08, 15a, 15b |

See `FIGURE_MANIFEST.md` for a per-figure / per-table lookup table mapping
manuscript artifacts to producing scripts.

## Dependencies on external services

Several steps query external APIs and require network access at runtime.
For reproducibility under API drift, all API calls log the response payload
to `reproduce/logs/<step_id>.log`.

| API | Used by | Notes |
|---|---|---|
| Enrichr | pathway_ora.py, upstream_regulators.py | Pathway ORA, TF inference |
| DGIdb v5 (GraphQL) | drug_targets.py, drug_targets_surrogates.py, druggable_replacement_targets.py | Drug-target interaction database |
| OpenTargets (GraphQL) | drug_targets.py, drug_targets_surrogates.py | Tractability / neurodegeneration disease scoring |
| MyGeneInfo | annotate_core25.py, cross_cohort_substitution.py, replaceability_figures.py, adaptive_panel_validation.py | Symbol / biotype resolution |
| STRING v12 (REST) | string_network.py | Protein-protein interaction network |
| GTEx v8 (REST) | shap500_artifact_audit.py, gtex_tissue_specificity.py | Per-tissue expression baselines |

## Provenance

| Item | Role | Source | Fetched by |
|---|---|---|---|
| GSE153960 (training + GPL16791) | discovery + held-out platform | NYGC ALS Consortium, GEO | `download_geo_data.sh` |
| GSE76220 (lumbar SC LCM) | external CNS validation | GEO | `download_geo_data.sh` |
| GSE122649 (motor cortex) | external CNS validation | GEO | `download_geo_data.sh` |
| SRP064478 (cervical SC) | external CNS validation | SRA | `quantify_srp064478.sh` (Salmon) |
| GSE234297 (blood) | negative control (peripheral) | GEO | `download_geo_data.sh` |
| GSE67196 (regional ALS) | regional scoring | GEO | `download_geo_data.sh` |
| GSE125583 (Alzheimer's, fusiform gyrus) | cross-disease specificity | GEO | `download_geo_data.sh` |
| GSE13162 (FTLD-TDP) | cross-disease specificity | GEO | `download_geo_data.sh` |
| GSE64810 (Huntington's, BA9) | cross-disease specificity | GEO | `download_geo_data.sh` |
| GSE68719 (Parkinson's, BA9) | cross-disease specificity | GEO | `download_geo_data.sh` |
| GSE255683 (lumbar SC, RNA-seq) | independent validation | GEO | `download_geo_data.sh` |
| GSE287256 (cervical SC, RNA-seq) | independent validation | GEO | `download_geo_data.sh` |
| GSE287257 (cervical SC, snRNA-seq; GSE287256 donors) | spinal-cord cell-type localisation | GEO | `download_geo_data.sh` |
| GSE26927 (Illumina arrays; ALS cervical SC + AD/PD/HD/MS/SCZ arms) | independent validation + specificity | GEO | `download_geo_data.sh` |
| E-MTAB-8635 (spinal cord, Agilent, raw GenePix) | independent validation | ArrayExpress | `download_geo_data.sh` |
| E-MTAB-2325 (motor cortex, Agilent; 35 donors shared with E-MTAB-8635) | independent validation | ArrayExpress | `download_geo_data.sh` |
| GSE219280 (C9ALS snRNA, ~4.6 GB) | per-cell validation (step 31) | GEO, **manual** | — see below |
| GSE212630 (non-C9 snRNA, ~1 GB) | per-cell validation (step 32) | GEO, **manual** | — see below |

The two single-nucleus cohorts are per-GSM supplementary `.h5` matrices that
`download_geo_data.sh` does not fetch. Place them in
`resources/GSE219280/suppl/` and `.../GSE212630/suppl/`
respectively (filenames as served by GEO). Steps 31 and 32 are **gated on those
files being present** and are skipped with a message otherwise, so a run without
them completes rather than aborting; the manuscript's per-cell FCN3 result comes
from these two steps and will simply be absent.
| Software | Python 3.13, LightGBM 4.5+, SHAP 0.52+, scikit-learn 1.5+ (full pin list in `environment_als.yml`) |
