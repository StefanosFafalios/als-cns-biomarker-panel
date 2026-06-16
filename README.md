# ALS CNS Biomarker Panel

Reproducibility code for an **in-silico biomarker-discovery study** of amyotrophic
lateral sclerosis (ALS) using post-mortem central-nervous-system (CNS) RNA-seq.
A LightGBM + SHAP pipeline defines a **25-gene CNS panel** (and a 15-gene critical
sub-panel) that distinguishes ALS-spectrum motor-neuron disease from
non-neurological controls. The panel is validated **zero-shot** across multiple
independent cohorts and platforms, and is characterised with cell-type
deconvolution, single-nucleus validation, a genome-wide gene-replacement screen,
and druggable-target annotation.

> **Discovery cohort:** GSE153960 (NYGC ALS Consortium, n = 874 post-mortem CNS).
>
> **Disclaimer:** All analyses are computational. Findings are *associative* and
> require experimental validation; nothing here is clinical advice.

## Repository layout

```
.
├── utils.py                  # shared data loaders (GEO series-matrix / supplementary parsers)
├── GSE153960/
│   ├── *.py                  # analysis scripts (one concern each, ~72 scripts)
│   ├── *.json *.csv *.txt    # cached model hyperparameters, panels, and result tables
│   ├── *.png                 # manuscript / supplementary figures
│   └── reproduce/
│       ├── run_all.sh         # master pipeline (ordered, with per-step runtimes)
│       ├── FIGURE_MANIFEST.md # every figure/table → producing-script mapping
│       ├── environment_als.yml
│       ├── download_geo_data.sh
│       └── quantify_srp064478.sh
└── resources/                # (not tracked) raw GEO data — see "Data" below
```

## Setup

```bash
conda env create -f GSE153960/reproduce/environment_als.yml
conda activate als-cns-panel
```

The pipeline depends only on the standard scientific-Python stack
(numpy, pandas, scikit-learn, lightgbm, shap, scipy, statsmodels, matplotlib,
h5py). It does **not** depend on any external AutoML library.

## Data

Expression data are public and are **not** stored in this repository. Download
them into `resources/`:

```bash
bash GSE153960/reproduce/download_geo_data.sh      # GEO cohorts → resources/
bash GSE153960/reproduce/quantify_srp064478.sh     # (optional) SRP064478 Salmon quantification
```

Single-nucleus cohorts (GSE219280, GSE212630) and the BRETIGEA brain cell-type
marker table have their own download notes inside `GSE153960/reproduce/run_all.sh`.

## Reproducing the analysis

```bash
bash GSE153960/reproduce/run_all.sh          # full pipeline
bash GSE153960/reproduce/run_all.sh fast     # skip the feature-matrix rebuild (needs it cached)
bash GSE153960/reproduce/run_all.sh from 24  # resume from a given step
```

`GSE153960/reproduce/FIGURE_MANIFEST.md` maps every figure and table to the
script that produces it, so any single result can be regenerated without running
the whole pipeline.

### The feature matrix (`lgbm_prefilter_X.npy`)

The pre-filtered feature matrix (874 × 47,822, ~160 MB) is **not** tracked here.
It is regenerated — without any external library — by the step-2 pipeline from
the downloaded GEO data:

```bash
conda run -n als-cns-panel python GSE153960/lgbm_iterative_pipeline.py
```

This reads the provided cached hyperparameters (`lgbm_best_params.json`) and
writes `lgbm_prefilter_X.npy`, after which `run_all.sh fast` runs end-to-end.

## Provenance of the cached hyperparameters

LightGBM hyperparameter optimisation (full-space and top-500 Bayesian search) was
performed with a separate AutoML library and is **not** part of this repository.
Those two steps are omitted; their results are provided as cached artifacts
(`lgbm_best_params.json`, `lgbm_top500_best_params.json`) and the rest of the
pipeline reproduces deterministically from them.

## Data availability

GEO / SRA accessions referenced by the scripts include: **GSE153960** (discovery),
GSE76220, GSE122649, GSE67196, GSE234297, GSE219280, GSE212630, and SRP064478.

## Citation

Manuscript in preparation; preprint forthcoming. Please cite this repository and
the accompanying manuscript when using the panel or code.

## License

[MIT](LICENSE) © 2026 Stefanos Fafalios
