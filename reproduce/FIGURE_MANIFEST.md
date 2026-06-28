# Figure & Table Manifest

Mapping every `\includegraphics` reference in `manuscript.tex` /
`supplementary.tex` to the script that produces it. Use this to verify a
specific result without re-running the full pipeline.

PNG paths are relative to `als_analysis/GSE153960/`. Each script writes its
output(s) to that same directory.

## Main-text figures

| Label | File | Producing script | Section |
|---|---|---|---|
| fig:iter_perf | `lgbm_iter_performance.png` | `lgbm_iter_plots.py` | §3.1 |
| fig:core25_curve | `lgbm_top500_final_curve_top50.png` | `lgbm_top500_final_analysis.py` | §3.1 |
| fig:core25_loo | `lgbm_core25_loo.png` | `lgbm_core25_panel.py` | §3.1 |
| fig:core25_curve_zoom | `lgbm_core25_curve.png` | `lgbm_core25_panel.py` | §3.1 |
| fig:core25_shap | `lgbm_core25_shap_beeswarm.png` | `lgbm_core25_panel.py` | §3.3 |
| fig:gpl16791 | `external_validation_gpl16791.png` (in `external_validation_gpl16791.py` output) | `external_validation_gpl16791.py` | §3.4 |
| fig:gse76220 | `blood_validation_gse76220.png` | `external_validation_gse76220.py` | §3.4.1 |
| fig:gse122649 | `additional_cohort_gse122649.png` | `additional_cohort_gse122649.py` | §3.4.1 |
| fig:srp064478 | `srp064478_validation.png` | `srp064478_figure.py` (depends on `srp064478_validation.py`) | §3.4.1 |
| fig:cross_cohort_loo | `panel_loo_zeroshot.png` | `panel_loo_zeroshot.py` (v2) | §3.4.2 |
| fig:validations | `combined_validation_roc.png` | `combined_validation_roc.py` (step 22b; depends on all external validations) | §3.4 |
| fig:gse67196 | `gse67196_regional_scoring.png` | `gse67196_regional_scoring.py` | §3.4.3 |
| fig:blood | `blood_validation_gse234297.png` | `blood_validation_gse234297.py` | §3.4.4 |
| fig:pathway_ora_up | `pathway_ora_up_in_als.png` | `pathway_ora.py` | §3.6 |
| fig:pathway_ora_down | `pathway_ora_down_in_als.png` | `pathway_ora.py` | §3.6 |
| fig:pathway_ora_np | `pathway_ora_non_predictive.png` | `pathway_ora.py` | §3.6 |
| fig:gene_profiling_auc | `gene_profiling_auc.png` | `gene_profiling.py` | §3.3 |
| fig:gene_profiling_tissue | `gene_profiling_tissue.png` | `gene_profiling.py` | §3.3 |
| fig:gene_profiling_volcano | `gene_profiling_volcano.png` | `gene_profiling.py` | §3.3 |
| fig:c9 | `c9als_sals_differential.png` | `c9als_sals_differential.py` | §3.7 |
| fig:replacement | `gene_replacement.png` | `gene_replacement.py` (cleaned: `gene_replacement_plot15.py`) | §3.8 |
| fig:repl_scatter | `replaceability_scatter.png` | `replaceability_scatter.py` | §3.8 |
| fig:deconv | `cell_type_deconvolution.png` | `cell_type_deconvolution.py` | §3.9 |
| fig:specificity | `disease_specificity.png` | `disease_specificity.py` | §3.10 |
| fig:string | `string_network.png` | `string_network.py` | §3.13 |
| fig:causal | `causal_panel_network.png` | `causal_panel_network.py` | §3.13 |

## Supplementary figures

| Label | File | Producing script | Supplementary section |
|---|---|---|---|
| fig:repl_greedy | `repl_vs_greedy.png` | `replaceability_figures.py` | S1.7 |
| fig:repl_box | `repl_candidate_boxplot.png` | `replaceability_figures.py` | S1.7 |
| fig:repl_heatmap | `repl_module_heatmap.png` | `replaceability_figures.py` | S1.7 |
| fig:replacement_loo | `replacement_panel_loo.png` | `replacement_panel_loo.py` | S14 |
| fig:panelmb | `panel_mb_network.png` | `panel_mb_network.py` + `panel_mb_postprocess.py` | S15 |
| fig:loo_sensitivity | `panel_loo_zeroshot.png` (shared) | `panel_loo_zeroshot.py` | S22 |
| fig:calibration | `calibration_reliability.png` | `calibration_analysis.py` | S17 |
| fig:dca | `decision_curve.png` | `decision_curve_analysis.py` | S22 |
| fig:tfs | `upstream_regulators.png` | `upstream_regulators.py` | S5 |
| fig:stability | `stability_selection.png` | `stability_selection.py` | S19 |
| fig:stability_module | `stability_module.png` | `stability_module.py` | S19 |
| fig:progression | `als_progression_proxies.png` | `als_progression_proxies.py` | S7 |
| fig:deconv_corr | `cell_type_deconvolution_corr.png` | `cell_type_deconvolution.py` | (auxiliary) |
| fig:panel_variant_cv | `panel_variant_cv.png` | `panel_variant_cv.py` | S13 |
| fig:panel_variant_greedy | `panel_variant_greedy.png` | `panel_variant_greedy.py` | S13 |
| fig:robustness | `multi_algorithm_robustness.png` | `multi_algorithm_robustness.py` | S19 (moved from main) |
| fig:sub_bar | `cross_cohort_substitution.png` | `cross_cohort_substitution.py` | S1.7 (moved from main) |
| fig:sub_roc | `substitution_roc_all.png` | `replaceability_figures.py` | S1.7 (moved from main) |
| fig:combat | `combat_batch_correction.png` | `combat_batch_correction.py` | S (ComBat batch correction) |
| fig:bretigea | `deconv_reference_bretigea.png` | `deconv_reference_bretigea.py` | S (robustness: composition) |
| fig:random_null | `random_panel_null.png` | `random_panel_null.py` | S (robustness: transfer specificity) |
| fig:snrna | `snrna_percell_validation.png` | `snrna_percell_validation.py` (GSE219280 snRNA; #1) | S (robustness: single-nucleus) |
| fig:snrna2 | `snrna_validation_gse212630.png` | `snrna_validation_gse212630.py` (GSE212630 non-C9 snRNA; #2 generalisation) | S (robustness: single-nucleus) |

## Tables

Most tables are hardcoded in the .tex files, but their data values come from
the following statistics files. Updating a table number = rerunning that
producing script.

| Table | Data source | Producing script |
|---|---|---|
| tab:core25_short | `lgbm_core25_panel.csv` | `lgbm_core25_panel.py` |
| tab:validation_summary | `lr_vs_lgbm_zeroshot.txt` + `panel_critical_eval_statistics.txt` | `lr_vs_lgbm_zeroshot.py` + `panel_17gene_eval.py` |
| tab:bio_evidence, tab:druggability | `drug_targets_statistics.txt` + `cell_type_deconvolution_statistics.txt` + `drug_targets_surrogates.txt` (Tier-S surrogate rows) | `drug_targets.py` + `cell_type_deconvolution.py` + `drug_targets_surrogates.py` |
| tab:specificity | `disease_specificity_statistics.txt` | `disease_specificity.py` |
| tab:tfs | `upstream_regulators_statistics.txt` | `upstream_regulators.py` |
| tab:coverage_matrix | `lr_vs_lgbm_zeroshot.txt` | `lr_vs_lgbm_zeroshot.py` |
| tab:multicohort | `lr_vs_lgbm_zeroshot.txt` + `multicohort_baselines.txt` | `lr_vs_lgbm_zeroshot.py` + `multicohort_baselines.py` |
| tab:ctd_sensitivity | `compartment_regression_sensitivity_statistics.txt` | `compartment_regression_sensitivity.py` |
| tab:lr_baseline | `lr_baseline_statistics.txt` | `lr_baseline.py` |
| tab:loo_sensitivity | `panel_loo_zeroshot_statistics.txt` + `cross_cohort_loo_sensitivity.txt` | `panel_loo_zeroshot.py` + `cross_cohort_loo_sensitivity.py` |
| tab:deconv_robust | `deconv_robustness.txt` / `.csv` | `deconv_robustness.py` (cell-type assignment bootstrap; also sets composition-driver labels in tab:bio_evidence) |
| tab:drugrepl | `druggable_replacement_targets.csv` / `.txt` | `druggable_replacement_targets.py` (DGIdb druggable co-regulated surrogates; depends on `gene_replacement.py`) |
| tab:combat | `combat_batch_correction_statistics.txt` | `combat_batch_correction.py` |
| tab:incremental | `incremental_over_composition_statistics.txt` | `incremental_over_composition.py` (panel vs composition; #2) |
| (sec:genetic_support, text) | `genetic_support_opentargets_statistics.txt` | `genetic_support_opentargets.py` (OpenTargets ALS genetic support; #6) |

## Statistics-text references in narrative

The manuscript narrative cites a handful of specific numbers (AUCs, CIs,
p-values) that come from these statistics files. If only the corresponding
script is rerun, those numbers will refresh automatically.

| Section | Numbers come from |
|---|---|
| Abstract (5-fold CV AUC = 0.9621) | `lgbm_top500_final_curve.csv` |
| §3.4.2 W.mean trajectory | `iterative_panel_elimination_statistics.txt` |
| §3.4.2 D_ZS CIs | `panel_loo_zeroshot_statistics.txt` |
| §3.8 cross-cohort substitution Δ AUC | `cross_cohort_substitution_statistics.txt` |
| §3.10 ALS vs FTLD/PSP AUC = 0.9067 | `disease_specificity_statistics.txt` |
| §3.13 DGIdb counts | `drug_targets_statistics.txt` |

## Exploratory scripts not in `run_all.sh`

The following scripts exist in `als_analysis/GSE153960/` but are NOT cited
in the manuscript or supplementary. They are kept for historical context and
internal sensitivity analyses but are not required to reproduce the published
results:

```
biomarker_discovery_lr_bayesopt.py    # Alternative LR-based discovery
loc112268270_greedy.py                # LOC112268270 lysosomal probe (deep-dive cut; now 1 sentence in main text)
cmap_drug_repurposing.py              # CMap/LINCS repurposing (section removed: negative result, no FDR-significant hits)
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
```
