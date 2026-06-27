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
conda env create -f als_analysis/GSE153960/reproduce/environment_als.yml
conda activate coffeeBreak

# 2. Download GEO data (~2 GB)
bash als_analysis/GSE153960/reproduce/download_geo_data.sh

# 3. Optionally quantify SRP064478 with Salmon (~1 h; needed for Step 22)
bash als_analysis/GSE153960/reproduce/quantify_srp064478.sh

# 4. Run all manuscript-cited analyses (≈ 30 h end-to-end, single 24-core box)
bash als_analysis/GSE153960/reproduce/run_all.sh
```

## Run modes

`run_all.sh` accepts two optional flags:

| Mode | Command | Effect |
|---|---|---|
| Full (default) | `bash run_all.sh` | Runs everything end-to-end. The two BayesianOptimizer steps (1 and 2b) account for ~15 h of the total. |
| Fast | `bash run_all.sh fast` | Skips Steps 1, 2, 2b — assumes cached `lgbm_best_params.json`, `lgbm_top500_best_params.json`, and `lgbm_prefilter_X.npy`/`names.txt` are present (these are committed to the repo). All downstream steps still rerun. |
| Resume | `bash run_all.sh from 24` | Skips every step with a numeric prefix lexicographically below `24`. Use to resume after interruption. |

Each step writes outputs to `als_analysis/GSE153960/` alongside the script
(figures as `.png`, statistics as `.txt`, CSV tables as `.csv`) and logs to
`reproduce/logs/<step_id>.log`.

## Rebuilding the PDFs

After `run_all.sh` finishes, regenerate both PDFs:

```bash
cd als_analysis/GSE153960/manuscript
xelatex supplementary.tex && xelatex manuscript.tex
xelatex supplementary.tex && xelatex manuscript.tex   # 2nd pass for xr refs
```

`xelatex` (not `pdflatex`) is required because `supplementary.tex` uses Unicode
glyphs (★, ▲, ✓) in figure captions. Two passes per document are required
because `manuscript.tex` and `supplementary.tex` cross-reference each other via
the `xr` LaTeX package (`\externaldocument{...}` in each preamble).

## Step layout

`run_all.sh` is grouped into 14 sections (A–N) by manuscript role:

| Group | Manuscript section | Scripts |
|---|---|---|
| A | Core model development (Methods, §3.1) | 5 scripts (bayesopt, decontamination, panel definition) |
| B | Gene-level annotation + artifact auditing (§3.3, §3.8) | 6 scripts |
| C | Stability + multi-algorithm robustness (§3.1, S19) | 3 scripts |
| D | External CNS validation (§3.4) | 6 scripts (+SRP gated on Salmon) |
| E | Cross-cohort LOO + critical panel (§3.4.2) | 4 scripts (greedy elim, bootstrap CI, eval) |
| F | Pathway / network / upstream (§3.6, §3.11, S5, S12, S15) | 8 scripts |
| H | Drug target annotation (§3.13) | 3 scripts |
| I | Cell-type deconvolution + progression (§3.9, S7, deconv-robustness) | 3 scripts (incl. cell-type-assignment bootstrap) |
| J | Disease specificity + subtype (§3.10, §3.7, S6) | 4 scripts |
| K | Regional scoring + blood validation (§3.4.3, §3.4.4) | 2 scripts |
| L | Gene replacement + cross-cohort substitution (§3.8) | 9 scripts (incl. 2 new analyses) |
| M | Calibration + linear-model sensitivity (S17, S18, ComBat) | 6 scripts (incl. ComBat batch correction) |
| N | With-replacement validation — §3.8 negative control (canonical KRTAP/family-filtered replacement assignment; steps 05/05b mirror it) | 1 script |

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

| Item | Source |
|---|---|
| GSE153960 (training + GPL16791) | NYGC ALS Consortium, GEO |
| GSE76220 (lumbar SC LCM) | GEO |
| GSE122649 (motor cortex) | GEO |
| SRP064478 (cervical SC) | SRA |
| GSE234297 (blood) | GEO |
| GSE67196 (regional ALS) | GEO |
| Software | Python 3.13, LightGBM 4.5+, SHAP 0.52+, scikit-learn 1.5+ (full pin list in `environment_als.yml`) |
