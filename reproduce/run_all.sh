#!/usr/bin/env bash
# =============================================================================
# ALS Biomarker Discovery — Full Reproducibility Script
# Dataset: GSE153960 (NYGC ALS Consortium, n=874 post-mortem CNS)
#
# Reproduces every figure and table in:
#   manuscript.tex
#   supplementary.tex
#
# Usage (from repository root):
#   bash reproduce/run_all.sh           # run everything
#   bash reproduce/run_all.sh fast      # skip Step 2 (feature-matrix rebuild)
#   bash reproduce/run_all.sh from 24   # resume from step 24
#
# Prerequisites:
#   1. conda env: conda env create -f reproduce/environment_als.yml
#   2. GEO data in resources/    (bash download_geo_data.sh)
#   3. SRP064478 Salmon quantification         (bash quantify_srp064478.sh) -- Step 22 only
#
# Each step writes its outputs to src/ alongside the script.
# Step-by-step runtimes are indicated next to each invocation (wall-clock,
# 24-core workstation). The two BayesianOptimizer steps are omitted (outputs cached as JSON); Step 2 regenerates lgbm_prefilter_X.npy (~2h).
# =============================================================================
set -euo pipefail

PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONDA_ENV="als-cns-panel"
RUN="conda run -n $CONDA_ENV python"
SCRIPT_DIR="$PROJ_ROOT/src"
LOG_DIR="$PROJ_ROOT/reproduce/logs"
mkdir -p "$LOG_DIR"

# Argument parsing: support "fast" mode and "from N" resume
MODE="${1:-full}"
RESUME_FROM=""
if [[ "$MODE" == "from" && -n "${2:-}" ]]; then
    RESUME_FROM="$2"
    MODE="full"
fi

run_step() {
    local step="$1"
    local script="$2"
    local note="${3:-}"

    # Resume gate (string compare on numeric prefix)
    if [[ -n "$RESUME_FROM" ]]; then
        local id="${step%%_*}"
        if [[ "$id" < "$RESUME_FROM" ]]; then
            echo "[SKIP] $step  (resuming from $RESUME_FROM)"
            return 0
        fi
    fi

    echo "======================================================"
    echo "[$step] $(date '+%Y-%m-%d %H:%M:%S')  $script  $note"
    echo "======================================================"
    $RUN "$SCRIPT_DIR/$script" 2>&1 | tee "$LOG_DIR/${step}.log"
}

cd "$PROJ_ROOT"

# =============================================================================
# A. CORE MODEL DEVELOPMENT (Methods + Results §3.1)
# =============================================================================
if [[ "$MODE" != "fast" ]]; then
    run_step "02_iterative_decontam"  lgbm_iterative_pipeline.py            "~2h"
    run_step "02_iter_plots"          lgbm_iter_plots.py                    "<5m"
else
    echo "[FAST MODE] Skipping Step 2 (needs lgbm_prefilter_X.npy; rebuild once via lgbm_iterative_pipeline.py)
fi
run_step "02c_top500_final"           lgbm_top500_final_analysis.py         "~30m"
run_step "02c_top50_summary"          lgbm_top50_summary.py                 "<5m"
run_step "02d_core25_panel"           lgbm_core25_panel.py                  "~20m"

# =============================================================================
# B. GENE-LEVEL ANNOTATION AND ARTIFACT AUDITING (§3.3, §3.8)
# =============================================================================
run_step "03_gene_profiling"          gene_profiling.py                     "~10m"
run_step "03a_annotate_core25"        annotate_core25.py                    "<5m (MyGeneInfo)"
run_step "03b_synergy_core25"         synergy_analysis.py                   "~15m"
run_step "03c_synergy_top500"         synergy_top500.py                     "~30m"
run_step "03d_pseudogene_audit"       pseudogene_artifact_check.py          "<5m"
run_step "03e_shap500_artifact"       shap500_artifact_audit.py             "<5m (GTEx API)"

# =============================================================================
# C. STABILITY AND MULTI-ALGORITHM ROBUSTNESS (§3.1, S19)
# =============================================================================
run_step "21_multi_algorithm"         multi_algorithm_robustness.py         "~30m"
run_step "21a_stability_selection"    stability_selection.py                "~45m"
run_step "21b_stability_module"       stability_module.py                   "~5m"

# =============================================================================
# D. EXTERNAL VALIDATION — CNS COHORTS (§3.4)
# =============================================================================
run_step "04_external_gpl16791"       external_validation_gpl16791.py       "~10m"
run_step "05_external_gse76220"       external_validation_gse76220.py       "~5m (with-replacement dict mirrors step 29 canonical assignment)"
run_step "05b_additional_gse122649"   additional_cohort_gse122649.py        "~10m (with-replacement dict mirrors step 29)"

# Step 22: SRP064478 requires prior Salmon quantification (see quantify_srp064478.sh)
if find "$PROJ_ROOT/resources/SRP064478/quant" -name "quant.sf" 2>/dev/null | \
   wc -l | grep -q "^15$"; then
    run_step "22_external_srp064478"  srp064478_validation.py               "~5m"
    run_step "22a_srp064478_figure"   srp064478_figure.py                   "<5m"
else
    echo "[22_external_srp064478] SKIPPED — run quantify_srp064478.sh first (15/15 quant.sf required)"
fi

# Multi-cohort comparison (used in tab:validation_summary, tab:multicohort)
run_step "23_lr_vs_lgbm_zeroshot"     lr_vs_lgbm_zeroshot.py                "~5m"
run_step "23a_multicohort_baselines"  multicohort_baselines.py              "~10m"
run_step "23b_multicohort_lr_ext"     multicohort_lr_extended.py            "~15m"

# =============================================================================
# E. CROSS-COHORT LOO AND CRITICAL PANEL DEFINITION (§3.4.2)
# Greedy backward elimination + v2 bootstrap-CI define the 15-crit panel.
# =============================================================================
run_step "24_iterative_panel_elim"    iterative_panel_elimination.py        "~25m"
run_step "25_panel_loo_zeroshot"      panel_loo_zeroshot.py                 "~60m  (20-seed ensemble + B=2000 bootstrap)"
run_step "26_panel_critical_eval"     panel_17gene_eval.py                  "~5m"
run_step "26a_cross_cohort_loo_sens"  cross_cohort_loo_sensitivity.py       "~10m"

# =============================================================================
# F. PATHWAY / NETWORK / UPSTREAM REGULATOR ANALYSIS (§3.6, §3.11, S5, S12, S15)
# =============================================================================
run_step "06_pathway_ora"             pathway_ora.py                        "~10m (Enrichr API)"
run_step "08_upstream_regulators"     upstream_regulators.py                "~15m (Enrichr API)"
run_step "13_gsea_shap"               gsea_shap_ranking.py                  "~5m"
run_step "13b_de_gsea_robustness"     de_gsea_robustness.py                 "~10m"
run_step "12a_string_network"         string_network.py                     "~5m (STRING API)"
run_step "12b_causal_panel_network"   causal_panel_network.py               "~10m"
run_step "15a_panel_mb_network"       panel_mb_network.py                   "~20m (LASSO neighbourhood)"
run_step "15b_panel_mb_postprocess"   panel_mb_postprocess.py               "<5m"

# =============================================================================
# G. CONDITIONAL INDEPENDENCE / MARKOV BLANKET (§3.11, S16)
# =============================================================================
run_step "19_iamb_markov_blanket"     ses_markov_blanket.py                 "~30m"
run_step "19a_iamb_alpha_grid"        iamb_alpha_grid.py                    "~20m"
run_step "19b_iamb_strict_alpha"      iamb_strict_alpha.py                  "~10m"

# =============================================================================
# H. DRUG TARGET ANNOTATION (§3.13, tab:bio_evidence, tab:druggability)
# =============================================================================
run_step "07_drug_targets"            drug_targets.py                       "~10m (DGIdb + OpenTargets)"
run_step "07aa_drug_surrogates"       drug_targets_surrogates.py            "~3m (DGIdb/OT for non-coding-member surrogates)"
run_step "07ab_druggable_replacements" druggable_replacement_targets.py     "~3m (DGIdb druggable co-regulated surrogates; needs 27)"
run_step "07a_gtex_specificity"       gtex_tissue_specificity.py            "~10m (GTEx API)"

# =============================================================================
# I. CELL-TYPE DECONVOLUTION AND PROGRESSION PROXIES (§3.9, S7)
# =============================================================================
run_step "11_cell_type_deconv"        cell_type_deconvolution.py            "~10m"
run_step "11a_deconv_robustness"      deconv_robustness.py                  "~5m (bootstrap CI on cell-type assignment; needs 11 + 03)"
run_step "12_progression_proxies"     als_progression_proxies.py            "~5m"

# =============================================================================
# J. DISEASE SPECIFICITY AND SUBTYPE ANALYSIS (§3.10, §3.7, S6)
# =============================================================================
run_step "10_disease_specificity"     disease_specificity.py                "~15m  (5-fold + fold-averaged FTLD/PSP)"
run_step "10a_subtype_probe"          subtype_probe.py                      "~5m"
run_step "10b_stmn2_tdp43_check"      stmn2_tdp43_check.py                  "<5m"
run_step "15_c9als_sals_de"           c9als_sals_differential.py            "~10m"

# =============================================================================
# K. REGIONAL SCORING AND BLOOD VALIDATION (§3.4.3, §3.4.4)
# =============================================================================
run_step "20_gse67196_regional"       gse67196_regional_scoring.py          "~5m"
run_step "14_blood_gse234297"         blood_validation_gse234297.py         "~10m"

# =============================================================================
# L. GENE REPLACEMENT AND CROSS-COHORT SUBSTITUTION (§3.8, NEW)
# Genome-wide non-inferiority screen + replaceability-informed substitution.
# =============================================================================
run_step "27_gene_replacement"        gene_replacement.py                   "~3h (genome-wide screen)"
run_step "27a_gene_replacement_clean" gene_replacement_clean.py             "~5m (artifact correction)"
run_step "27b_gene_replacement_plot"  gene_replacement_plot15.py            "<5m"
run_step "27c_replaceability_scatter" replaceability_scatter.py             "<5m"
run_step "27d_panel_variant_cv"       panel_variant_cv.py                   "~20m"
run_step "27e_panel_variant_greedy"   panel_variant_greedy.py               "~30m"
run_step "27f_replacement_panel_loo"  replacement_panel_loo.py              "~15m"
run_step "27g_cross_cohort_subst"     cross_cohort_substitution.py          "~10m (depends on 27)"
run_step "27h_replaceability_figs"    replaceability_figures.py             "<5m (depends on 24, 27, 27g)"

# =============================================================================
# M. CALIBRATION AND LINEAR-MODEL SENSITIVITY (S17, S18)
# =============================================================================
run_step "28_calibration_analysis"    calibration_analysis.py               "~10m"
run_step "28aa_combat_batch_corr"     combat_batch_correction.py            "~5m (ComBat batch-correction sensitivity; Limitation 4)"
run_step "28a_decision_curve"         decision_curve_analysis.py            "~5m"
run_step "28b_compartment_reg_sens"   compartment_regression_sensitivity.py "~15m"
run_step "28c_lr_baseline"            lr_baseline.py                        "~10m"
run_step "28d_small_cohort_loo"       small_cohort_loo_refit.py             "~5m"

# =============================================================================
# N. WITH-REPLACEMENT VALIDATION (§3.8 negative control)
# Canonical derivation of the de-duplicated clean replacement assignment
# (confirmed artifacts AND artifact gene families excluded -- olfactory/taste/
# keratin/KRTAP/defensin/LCE/SPRR/VN1R, aligned with cross_cohort_substitution
# and druggable_replacement_targets). The hardcoded dicts in steps 05/05b
# mirror this output; the A->B zero-shot deltas are the §3.8 negative control.
# Depends on 27 (gene_replacement) + 03e (shap500 artifact audit).
# =============================================================================
run_step "29_adaptive_panel"          adaptive_panel_validation.py          "~10m (canonical KRTAP/family-filtered replacement assignment)"

# =============================================================================
# O. CASE-STRENGTHENING / ROBUSTNESS (in-silico)
# #3 reference-based deconvolution (BRETIGEA), #2 incremental value over
# composition, #4 random-panel null for transfer, #6 OpenTargets genetic support.
# Prereq for #3/#2: BRETIGEA R package + exported markers --
#   Rscript -e 'install.packages("BRETIGEA"); library(BRETIGEA); \
#     write.csv(markers_df_brain,"src/bretigea_markers.csv",row.names=FALSE)'
# #3 depends on 11 (legacy z-score concordance); #2 imports the estimator from #3.
# =============================================================================
run_step "11b_deconv_bretigea"        deconv_reference_bretigea.py          "~5m (needs bretigea_markers.csv + step 11)"
run_step "11c_incremental_composition" incremental_over_composition.py       "~8m (imports estimator from 11b)"
run_step "30_random_panel_null"       random_panel_null.py                  "~15m (B=1000 x2 nulls; GPL24676->GPL16791)"
run_step "07ac_genetic_support"       genetic_support_opentargets.py        "~3m (OpenTargets ALS genetic_association)"
# #1 single-nucleus per-cell validation. Prereq: download the GSE219280 frontal+
# motor-cortex C9ALS+Control filtered .h5 (12+12 samples, ~4.6GB) to
# resources/GSE219280/suppl/ (per-GSM files at
# ftp.ncbi.nlm.nih.gov/geo/samples/GSM...; cellBender_corrected_filtered.h5).
run_step "31_snrna_percell"           snrna_percell_validation.py           "~5m (needs GSE219280 .h5; h5py)"
# #2 generalisation to a non-C9 cohort. Prereq: download GSE212630 Control + TDPhigh
# filtered_feature_bc_matrix.h5 (7+8 samples, ~1GB) to resources/GSE212630/suppl/.
run_step "32_snrna_gse212630"         snrna_validation_gse212630.py         "~4m (needs GSE212630 .h5; non-C9 generalisation)"

echo ""
echo "======================================================"
echo "All steps completed: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Outputs in: $SCRIPT_DIR"
echo "Logs in:    $LOG_DIR"
echo ""
echo "Manifest of figure -> script mapping in:"
echo "  $PROJ_ROOT/reproduce/FIGURE_MANIFEST.md"
echo ""
echo "======================================================"
