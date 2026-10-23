# Figure & Table Manifest

Every figure and table in `manuscript.tex` / `supplementary.tex` (2026-10-23 rewrite: 50-gene panel, 21-gene critical panel, stable core),
with the script and `run_all.sh` step that produce it. PNG and statistics paths are
relative to `src/`.

## Main text

| Label | Artefact | Producing script (step) |
|---|---|---|
| tab:cohorts | dataset sizes and panel-gene coverage | `cohort_validation_table.py` (24b) |
| fig:beeswarm | `lgbm_core25_shap_beeswarm.png` | `lgbm_core25_panel.py` (02d) |
| tab:panel | `lgbm_core25_panel.csv`, `gene_profiling_summary.csv` | `lgbm_core25_panel.py` (02d), `gene_profiling.py` (03) |
| tab:validation | `cohort_validation_table.{csv,txt}` | `cohort_validation_table.py` (24b) |
| fig:validation | `cohort_validation.png` | `cohort_validation_figure.py` (24d) |
| fig:trim | `greedy_significance_stop.png` | `greedy_significance_stop.py` (24a) |
| tab:core | `lodo_trim_statistics.txt`, `critical_replacement_pooled_statistics.txt`, `multicohort_meta_analysis.csv`, `core_gene_covariates_statistics.txt`, `cross_disease_gene_direction.csv` | `lodo_trim.py` (24e), `critical_replacement_pooled.py` (27j), `multicohort_meta_analysis.py` (42), `core_gene_covariates.py` (38a, gated), `cross_disease_gene_direction.py` (35a) |
| fig:direction, tab:direction | `drug_direction.png`, `drug_direction_{table.csv,statistics.txt}` | `drug_direction_figure.py` (07c; needs 07, 41, 41a, 42) |

## Supplementary

| Label | Artefact | Producing script (step) |
|---|---|---|
| tab:supp_hyper | `utils.SELECTED_LGBM_PARAMS` | `utils.py` (the selected configuration) |
| sec:supp_size (panel size) | `lgbm_top500_final_curve.csv` | `lgbm_top500_final_analysis.py` (02c) |
| tab:supp_random | `random_panel_null_cohorts_statistics.txt` | `random_panel_null_cohorts.py` (30a) |
| tab:supp_nested, substitute re-selection | `nested_panel_cv_statistics.txt`, `nested_cv_decomposition_statistics.txt`, `substitute_reselection_statistics.txt` | `nested_panel_cv.py` (30b), `nested_cv_decomposition.py` (30c), `substitute_reselection.py` (27l) |
| tab:supp_covariates | `core_gene_covariates{.csv,_statistics.txt}` | `core_gene_covariates.py` (38a, gated on the Zenodo ST1 table) |
| sec:supp_trim_null | `pooled_cv_random_null_statistics.txt` | `pooled_cv_random_null.py` (24c) |
| leave-one-dataset-out trim | `lodo_trim{.csv,_statistics.txt}` | `lodo_trim.py` (24e, ~26 h) |
| tab:supp_trim | `iterative_panel_elimination_statistics.txt`, `greedy_significance_stop_statistics.txt` | `iterative_panel_elimination.py` (24), `greedy_significance_stop.py` (24a) |
| tab:supp_deconv | `cell_type_deconvolution_statistics.txt`, `deconv_robustness.txt` | `cell_type_deconvolution.py` (11), `deconv_robustness.py` (11a) |
| BRETIGEA paragraph | `deconv_reference_bretigea_statistics.txt` | `deconv_reference_bretigea.py` (11b) |
| tab:supp_incremental | `incremental_over_composition_statistics.txt` | `incremental_over_composition.py` (11c) |
| tab:supp_percell | `snrna_spinal_cord_percell.csv`, `snrna_cortex_percell_statistics.txt` | `snrna_spinal_cord.py` (41), `snrna_cortex_percell.py` (41a) |
| tab:supp_purity | `snrna_typing_purity{.csv,_statistics.txt}`, `snrna_percell_v2_statistics.txt` | `snrna_typing_purity.py` (41e), `snrna_percell_v2.py` (41c) |
| tab:supp_spatial | `spatial_spinal_cord{.csv,_statistics.txt}` | `spatial_spinal_cord.py` (41b) |
| tab:supp_pineda | `pineda_cortex_counts{.csv,_statistics.txt}`, `pineda_cortex_lookup.csv` | `pineda_cortex_counts.py` (41f), `pineda_cortex_lookup.py` (41d) |
| tab:supp_lodo | `lodo_trim{.csv,_statistics.txt}` | `lodo_trim.py` (24e) |
| tab:supp_meta | `multicohort_meta_analysis{.csv,_statistics.txt}` | `multicohort_meta_analysis.py` (42) |
| tab:supp_crossdisease | `cross_disease_specificity_statistics.txt` | `cross_disease_specificity.py` (35) |
| tab:supp_crossgene, Results 3.6 and Discussion (Huntington's) | `cross_disease_gene_direction{.csv,_statistics.txt}`, `cross_disease_specificity_loo.csv` | `cross_disease_gene_direction.py` (35a), `cross_disease_specificity.py` (35) |
| sec:supp_specificity within-cohort FTD | `disease_specificity_statistics.txt` | `disease_specificity.py` (10) |
| sec:supp_specificity blood paragraph | `blood_validation_gse234297_statistics.txt` | `blood_validation_gse234297.py` (14) |
| sec:supp_clinical | `nygc_clinical_association_statistics.txt` | `nygc_clinical_association.py` (38, gated) |
| tab:supp_drugs | `drug_targets_summary.csv` | `drug_targets.py` (07), `genetic_support_opentargets.py` (07ac) |
| tab:supp_replace, tab:supp_subst, Results 3.7 substitutes | `critical_replacement_pooled{.csv,_statistics.txt}`, `substitute_percell{.csv,_statistics.txt}` | `critical_replacement_pooled.py` (27j), `substitute_percell.py` (27k) |
| Supp. connectivity-map section, Results 3.7 screen | `cmap_direction_screen{.csv,_statistics.txt}` | `cmap_direction_screen.py` (07d; Enrichr LINCS libraries cached in `resources/LINCS`) |
| sec:supp_context | `panel_mb_*`, `upstream_regulators*`, `pathway_ora*`, `gtex_tissue_specificity*` | 15a/15b, 08, 06, 07a |
| sec:supp_robust, sec:supp_modelchoice LR | `calibration_analysis_statistics.txt`, `lr_baseline_statistics.txt`, `combat_batch_correction_statistics.txt`, `stability_selection_statistics.txt`, `multi_algorithm_robustness_statistics.txt` | 28, 28c, 28aa, 21a, 21 |
| sec:supp_random newer cohorts, GSE26927 arms | `new_cohort_validation_statistics.txt` | `new_cohort_validation.py` (40) |

