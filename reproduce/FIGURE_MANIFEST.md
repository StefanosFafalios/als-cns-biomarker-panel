# Figure & Table Manifest

Every figure and table in `manuscript.tex` / `supplementary.tex` (2026-10-05 rewrite),
with the script and `run_all.sh` step that produce it. PNG and statistics paths are
relative to `src/`.

## Main text

| Label | Artefact | Producing script (step) |
|---|---|---|
| tab:cohorts | dataset sizes and panel-gene coverage | `cohort_validation_table.py` (24b) |
| fig:beeswarm | `lgbm_core25_shap_beeswarm.png` | `lgbm_core25_panel.py` (02d) |
| tab:panel25 | `lgbm_core25_panel.csv`, `gene_profiling_summary.csv` | `lgbm_core25_panel.py` (02d), `gene_profiling.py` (03) |
| tab:validation | `cohort_validation_table.{csv,txt}` | `cohort_validation_table.py` (24b) |
| fig:validation | `cohort_validation.png` | `cohort_validation_figure.py` (24d) |
| fig:trim | `greedy_significance_stop.png` | `greedy_significance_stop.py` (24a) |
| fig:direction, tab:direction | `drug_direction.png`, `drug_direction_{table.csv,statistics.txt}` | `drug_direction_figure.py` (07c; needs 07, 41, 41a, 42) |

## Supplementary

| Label | Artefact | Producing script (step) |
|---|---|---|
| tab:supp_hyper | `lgbm_top500_best_params.json` | Bayesian search (02b, not in this repository; the JSON ships in `src/`) |
| tab:supp_random | `random_panel_null_cohorts_statistics.txt` | `random_panel_null_cohorts.py` (30a) |
| S3 pooled null | `pooled_cv_random_null_statistics.txt` | `pooled_cv_random_null.py` (24c) |
| tab:supp_nested | `nested_panel_cv_statistics.txt`, `nested_cv_decomposition_statistics.txt` | `nested_panel_cv.py` (30b), `nested_cv_decomposition.py` (30c) |
| tab:supp_trim, fig:supp_trim | `iterative_panel_elimination{.png,_statistics.txt}`, `greedy_significance_stop_statistics.txt` | `iterative_panel_elimination.py` (24), `greedy_significance_stop.py` (24a) |
| tab:supp_deconv | `cell_type_deconvolution_statistics.txt`, `deconv_robustness.txt` | `cell_type_deconvolution.py` (11), `deconv_robustness.py` (11a) |
| BRETIGEA paragraph | `deconv_reference_bretigea_statistics.txt` | `deconv_reference_bretigea.py` (11b) |
| tab:supp_incremental | `incremental_over_composition_statistics.txt` | `incremental_over_composition.py` (11c) |
| tab:supp_percell, fig:supp_snrna | `snrna_spinal_cord{.png,_percell.csv}`, `snrna_cortex_percell_statistics.txt` | `snrna_spinal_cord.py` (41), `snrna_cortex_percell.py` (41a) |
| tab:supp_meta | `multicohort_meta_analysis{.csv,_statistics.txt}` | `multicohort_meta_analysis.py` (42) |
| tab:supp_gstm1 | `gstm1_null_check_statistics.txt` | `gstm1_null_check.py` (07b) |
| tab:supp_crossdisease | `cross_disease_specificity_statistics.txt` | `cross_disease_specificity.py` (35) |
| S10 within-cohort FTD | `disease_specificity_statistics.txt` | `disease_specificity.py` (10) |
| S10 blood paragraph | `blood_validation_gse234297_statistics.txt` | `blood_validation_gse234297.py` (14) |
| S11 clinical | `nygc_clinical_association_statistics.txt` | `nygc_clinical_association.py` (38, gated) |
| tab:supp_drugs | `drug_targets_summary.csv` | `drug_targets.py` (07), `genetic_support_opentargets.py` (07ac) |
| S13 context | `panel_mb_*`, `upstream_regulators*`, `pathway_ora*`, `gtex_tissue_specificity*` | 15a/15b, 08, 06, 07a |
| S14 robustness | `calibration_analysis_statistics.txt`, `lr_baseline_statistics.txt`, `combat_batch_correction_statistics.txt`, `stability_selection_statistics.txt`, `gene_replacement*` | 28, 28c, 28aa, 21a, 27/27a |
| S15 coffeeBreak | `coffeebreak_panel_statistics.txt` | `coffeebreak_panel_loo.py` (39, gated; needs coffeeBreak) |
| S3 newer cohorts | `new_cohort_validation_statistics.txt` | `new_cohort_validation.py` (40) |

## Exploratory scripts not in `run_all.sh`

The following scripts exist in `src/` but are NOT cited
in the manuscript or supplementary. They are kept for historical context and
internal sensitivity analyses but are not required to reproduce the published
results:

```
biomarker_discovery_lr_bayesopt.py    # Alternative LR-based discovery
loc112268270_greedy.py                # LOC112268270 lysosomal probe (deep-dive cut; now 1 sentence in main text)
cmap_drug_repurposing.py              # Enrichr LINCS overlap with an older 15-gene list; no FDR-significant hits; not the per-cell screen proposed in the Discussion
multicohort_lr_extended.py            # LR vs LGBM on GSE122649/GSE234297/SRP064478; own (raw-count) blood loader, not cited
reciprocal_discovery.py               # Reciprocal discovery on GSE122649 (n=38); omitted: underpowered, panel re-discovery overfits (transfer 0.54). Due-diligence only.
ses_markov_blanket.py                 # IAMB Markov-blanket selection; removed from paper (alpha-dependent sensitivity analysis)
iamb_alpha_grid.py                    # IAMB alpha-tuning grid; removed from paper
iamb_strict_alpha.py                  # IAMB strict-alpha membership; removed from paper
lr_svm_gpl16791_validation.py         # LR/SVM comparison on GPL16791
lr_svm_panel_analysis.py              # LR/SVM analysis on panel
lr_svm_srp064478_validation.py        # LR/SVM on SRP064478
universal_replacement_profile.py      # Universal replacement profiling
validation_panel_loo.py               # Legacy panel LOO (superseded by panel_loo_zeroshot.py v2)
cross_cohort_panel_loo.py             # Legacy cross-cohort LOO (superseded by iterative_panel_elimination.py)
colsample_sensitivity.py              # colsample_bytree transfer sensitivity; internal check, not cited
```
