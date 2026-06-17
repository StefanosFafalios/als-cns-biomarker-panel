# ALS Biomarker Discovery — Reproducibility

Everything needed to reproduce the figures, tables, and statistics of the study
from public GEO/SRA inputs. See the repository-root `README.md` for an overview;
this document details the pipeline.

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
# 1. Build the conda environment
conda env create -f reproduce/environment_als.yml
conda activate als-cns-panel

# 2. Download GEO data (~2 GB) into ../resources/
bash reproduce/download_geo_data.sh

# 3. Optionally quantify SRP064478 with Salmon (~1 h; needed for Step 22)
bash reproduce/quantify_srp064478.sh

# 4. Run all analyses
bash reproduce/run_all.sh
```

## Run modes

| Mode | Command | Effect |
|---|---|---|
| Full (default) | `bash reproduce/run_all.sh` | Runs everything end-to-end. Step 2 (decontamination) regenerates the ~160 MB feature matrix `src/lgbm_prefilter_X.npy` (~2 h). |
| Fast | `bash reproduce/run_all.sh fast` | Skips Step 2 — assumes `src/lgbm_prefilter_X.npy` already exists. Regenerate it once with `python src/lgbm_iterative_pipeline.py`; the cached `src/lgbm_best_params.json` / `src/lgbm_top500_best_params.json` are already provided. |
| Resume | `bash reproduce/run_all.sh from 24` | Skips every step with a numeric prefix lexicographically below `24`. |

> **Hyperparameter search is omitted from this repository.** The full-space and
> top-500 BayesianOptimizer searches used a separate AutoML library; their outputs
> are provided as cached JSON (`src/lgbm_best_params.json`,
> `src/lgbm_top500_best_params.json`), so the pipeline reproduces from them.

Each step writes outputs to `src/` alongside the script (figures as `.png`,
statistics as `.txt`, tables as `.csv`) and logs to `reproduce/logs/<step_id>.log`.

## Step layout

`run_all.sh` is grouped into sections by analysis role:

| Group | Role | Scripts |
|---|---|---|
| A | Core model development — decontamination + panel definition (bayesopt omitted) | 5 |
| B | Gene-level annotation + artifact auditing | 6 |
| C | Stability + multi-algorithm robustness | 3 |
| D | External CNS validation | 6 (+SRP, gated on Salmon) |
| E | Cross-cohort LOO + critical-panel definition | 4 |
| F | Pathway / network / upstream regulators | 8 |
| G | Conditional independence / Markov blanket | 3 |
| H | Drug-target annotation | 4 |
| I | Cell-type deconvolution + progression proxies | 3 |
| J | Disease specificity + subtype | 4 |
| K | Regional scoring + blood validation | 2 |
| L | Gene replacement + cross-cohort substitution | 9 |
| M | Calibration + linear-model sensitivity | 6 |
| N | With-replacement validation (negative control) | 1 |
| O | Case-strengthening robustness (reference deconvolution, random-panel null, genetic support, single-nucleus) | 6 |

See `FIGURE_MANIFEST.md` for a per-figure / per-table lookup.

## External services

Several steps query public APIs (network required at runtime); responses are
logged to `reproduce/logs/<step_id>.log`.

| API | Used by | Notes |
|---|---|---|
| Enrichr | pathway_ora.py, upstream_regulators.py | Pathway ORA, TF inference |
| DGIdb v5 (GraphQL) | drug_targets.py, drug_targets_surrogates.py, druggable_replacement_targets.py | Drug–target interactions |
| OpenTargets (GraphQL) | drug_targets.py, drug_targets_surrogates.py, genetic_support_opentargets.py | Tractability / genetic association |
| MyGeneInfo | annotate_core25.py, cross_cohort_substitution.py, replaceability_figures.py, adaptive_panel_validation.py | Symbol / biotype resolution |
| STRING v12 (REST) | string_network.py | Protein–protein interaction network |
| GTEx v8 (REST) | shap500_artifact_audit.py, gtex_tissue_specificity.py | Per-tissue expression baselines |

## Provenance

| Item | Source |
|---|---|
| GSE153960 (discovery + GPL16791) | NYGC ALS Consortium, GEO |
| GSE76220 (lumbar SC LCM) | GEO |
| GSE122649 (motor cortex) | GEO |
| SRP064478 (cervical SC) | SRA |
| GSE234297 (blood) | GEO |
| GSE67196 (regional ALS) | GEO |
| GSE219280, GSE212630 (single-nucleus) | GEO |
| Software | Python 3.13, LightGBM 4.5+, SHAP 0.52+, scikit-learn 1.5+ (full pins in `environment_als.yml`) |
