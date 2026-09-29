#!/usr/bin/env bash
# =============================================================================
# ALS Biomarker Discovery — Full Reproducibility Script
# Dataset: GSE153960 (NYGC ALS Consortium, n=874 post-mortem CNS)
#
# Reproduces every figure and table in:
#   manuscript.tex
#   supplementary.tex
#
# Usage (from coffeeBreak project root):
#   bash .../run_all.sh                     # run everything
#   bash .../run_all.sh fast                # skip Step 2 (feature-matrix rebuild)
#   bash .../run_all.sh from 24             # resume from step 24
#   bash .../run_all.sh only 07 25 24a      # run ONLY these steps, in file order
#
# `only` exists so that a targeted rerun -- after a bug fix invalidates a subset
# of results -- uses this file's ordering rather than a second script with its
# own copy of the dependency graph. There is deliberately only one runnable
# pipeline definition in this repository.
#
# Prerequisites:
#   1. conda env: conda env create -f reproduce/environment_als.yml
#   2. GEO data in resources/    (bash download_geo_data.sh)
#   3. SRP064478 Salmon quantification         (bash quantify_srp064478.sh) -- Step 22 only
#
# Each step writes its outputs to src/ alongside the script.
# Step-by-step runtimes are indicated next to each invocation (wall-clock,
# 24-core workstation). The two BayesianOptimizer steps (1, 2b) are not in this
# repository; their tuned hyperparameters ship as src/lgbm_*best_params.json.
# =============================================================================
set -euo pipefail

PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONDA_ENV="als-cns-panel"
RUN="conda run -n $CONDA_ENV python"
SCRIPT_DIR="$PROJ_ROOT/src"
LOG_DIR="$PROJ_ROOT/reproduce/logs"
mkdir -p "$LOG_DIR"

# Argument parsing: "fast" mode, "from N" resume, "only <ids...>" subset.
MODE="${1:-full}"
RESUME_FROM=""
ONLY_STEPS=""
if [[ "$MODE" == "from" && -n "${2:-}" ]]; then
    RESUME_FROM="$2"
    MODE="full"
elif [[ "$MODE" == "only" ]]; then
    shift
    if [[ $# -eq 0 ]]; then
        echo "usage: run_all.sh only <step-id> [step-id ...]" >&2
        exit 2
    fi
    ONLY_STEPS=" $* "
    MODE="full"
fi

run_step() {
    local step="$1"
    local script="$2"
    local note="${3:-}"

    # Subset gate. Matches on the step ID prefix before the first underscore, so
    # `only 24a` selects 24a_greedy_signif_stop. Steps still run in file order,
    # which is what keeps the dependency graph in one place.
    if [[ -n "$ONLY_STEPS" ]]; then
        local sid="${step%%_*}"
        if [[ "$ONLY_STEPS" != *" $sid "* ]]; then
            return 0
        fi
    fi

    # Resume gate: compare the numeric prefix as a base-10 integer, with a
    # lexicographic tie-break on any letter suffix (step IDs look like 02, 02b,
    # 07aa, 24, ...). A bare string compare would mis-skip e.g. "10" < "5".
    if [[ -n "$RESUME_FROM" ]]; then
        local id="${step%%_*}"
        local id_digits="${id//[!0-9]/}"
        local from_digits="${RESUME_FROM//[!0-9]/}"
        local id_num=$((10#${id_digits:-0}))
        local from_num=$((10#${from_digits:-0}))
        if (( id_num < from_num )) ||
           { (( id_num == from_num )) && [[ "$id" < "$RESUME_FROM" ]]; }; then
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
    echo "[FAST MODE] Skipping Step 2 — needs lgbm_prefilter_X.npy from a previous full run"
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
# 21a/21b run after step 27 (section K): the module-aware stability reads the
# replacement screen's gene_replacement_results.csv.

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
    run_step "22b_combined_roc"       combined_validation_roc.py            "<5m (Fig3 combined 4-cohort ROC grid; needs all external validations)"
else
    echo "[22_external_srp064478] SKIPPED — run quantify_srp064478.sh first (15/15 quant.sf required)"
fi

# Multi-cohort comparison (used in tab:validation_summary, tab:multicohort)
run_step "23_lr_vs_lgbm_zeroshot"     lr_vs_lgbm_zeroshot.py                "~5m"
run_step "23a_multicohort_baselines"  multicohort_baselines.py              "23s (measured 2026-08-04)"
run_step "23b_multicohort_lr_ext"     multicohort_lr_extended.py            "~15m"

# =============================================================================
# E. CROSS-COHORT LOO AND CRITICAL PANEL DEFINITION (§3.4.2)
# Greedy backward elimination + a paired-bootstrap non-inferiority stopping rule
# define the critical panel.
#
# Step 24a is load-bearing for everything after it: it writes panel_critical.csv,
# the single source of truth that every downstream script reads through
# utils.critical_panel_symbols() / critical_panel_features(). No script hardcodes
# panel membership, so 24a must succeed before any of them are run.
#
# 24a caches: if greedy_significance_stop.csv already holds all 25 panel sizes it
# reuses them and only regenerates the report (seconds). Scoring from scratch is
# ~8h. Set GREEDY_STOP_REFIT=1 to force a full refit -- required if the model,
# the cohorts, or the 25-gene panel change, since the cache keys on none of those.
# =============================================================================
run_step "24_iterative_panel_elim"    iterative_panel_elimination.py        "~25m"
run_step "24a_greedy_signif_stop"     greedy_significance_stop.py           "~4m full refit / seconds cached (measured 2026-08-04; writes panel_critical.csv)"
run_step "25_panel_loo_zeroshot"      panel_loo_zeroshot.py                 "~22m on an idle box (measured 2026-08-05). A 65h wall was observed once under heavy external memory/CPU contention -- that figure is contention, not cost."
run_step "26_panel_critical_eval"     panel_critical_eval.py                  "~5m"
run_step "26a_cross_cohort_loo_sens"  cross_cohort_loo_sensitivity.py       "~10m"
# Donor-grouped internal CV (the manuscript's headline within-cohort AUC) and the
# GPL16791 donor-disjoint zero-shot sensitivity. Needs panel_critical.csv (24a) and
# combined_validation_roc_statistics.txt (22b), whose GPL16791 AUCs it must reproduce.
run_step "26b_donor_leakage"          donor_leakage_sensitivity.py          "~4m"

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
# G. DRUG TARGET ANNOTATION (§3.13, tab:bio_evidence, tab:druggability)
# =============================================================================
run_step "07_drug_targets"            drug_targets.py                       "21s (DGIdb + OpenTargets; network-bound)"
# 07ab runs after step 27 (section K): it reads gene_replacement_results.csv.
run_step "07a_gtex_specificity"       gtex_tissue_specificity.py            "~10m (GTEx API)"

# =============================================================================
# H. CELL-TYPE DECONVOLUTION AND PROGRESSION PROXIES (§3.9, S7)
# =============================================================================
run_step "11_cell_type_deconv"        cell_type_deconvolution.py            "~10m"
run_step "11a_deconv_robustness"      deconv_robustness.py                  "14s (bootstrap CI on cell-type assignment; needs 11 + 03)"
run_step "12_progression_proxies"     als_progression_proxies.py            "~5m"

# =============================================================================
# I. DISEASE SPECIFICITY AND SUBTYPE ANALYSIS (§3.10, §3.7, S6)
# =============================================================================
run_step "10_disease_specificity"     disease_specificity.py                "~15m  (5-fold + fold-averaged FTLD/PSP)"
run_step "10a_subtype_probe"          subtype_probe.py                      "~5m"
run_step "10b_stmn2_tdp43_check"      stmn2_tdp43_check.py                  "<5m"
run_step "15_c9als_sals_de"           c9als_sals_differential.py            "~10m"

# =============================================================================
# J. REGIONAL SCORING AND BLOOD VALIDATION (§3.4.3, §3.4.4)
# =============================================================================
run_step "20_gse67196_regional"       gse67196_regional_scoring.py          "~5m"
run_step "14_blood_gse234297"         blood_validation_gse234297.py         "~10m"

# =============================================================================
# K. GENE REPLACEMENT AND CROSS-COHORT SUBSTITUTION (§3.8, NEW)
# Genome-wide non-inferiority screen + replaceability-informed substitution.
# =============================================================================
run_step "27_gene_replacement"        gene_replacement.py                   "~1.5h (genome-wide screen; measured 2026-09-28)"
run_step "27a_gene_replacement_clean" gene_replacement_clean.py             "~5m (artifact correction)"
run_step "27b_gene_replacement_plot"  gene_replacement_plot15.py            "<5m"
run_step "27c_replaceability_scatter" replaceability_scatter.py             "<5m"
run_step "27d_panel_variant_cv"       panel_variant_cv.py                   "42s"
run_step "27e_panel_variant_greedy"   panel_variant_greedy.py               "38s"
# 27e picks its swaps on GSE122649, so 27f tests them on cohorts the search never saw
# (donor-disjoint GPL16791 primary, non-inferiority margin 0.01). It replaces the
# replacement-panel LOO, which weighted GSE122649 at 38/58 and so re-used the selection cohort.
run_step "27f_replacement_heldout"    replacement_panel_heldout.py          "~5m (reads 27e swaps)"
run_step "27g_cross_cohort_subst"     cross_cohort_substitution.py          "~13m (depends on 27)"
# 07aa must follow 27g: the surrogate assignments are read from
# cross_cohort_substitution_statistics.txt, not hardcoded (they were until
# 2026-08-04, and two of three were wrong).
run_step "07aa_drug_surrogates"       drug_targets_surrogates.py            "2s (surrogates derived from 27g output; must follow it)"
run_step "27h_replaceability_figs"    replaceability_figures.py             "33s (depends on 24, 27, 27g)"
run_step "07ab_druggable_replacements" druggable_replacement_targets.py     "~3m (DGIdb druggable co-regulated surrogates; needs 27)"
# 21a reruns the published SHAP ranking on 100 donor-level bootstraps and saves
# each top 25; 21b reads those draws plus the step-27 replacement modules.
run_step "21a_stability_selection"    stability_selection.py                "~5m (100 donor-level bootstraps x published ranking)"
run_step "21b_stability_module"       stability_module.py                   "<1m (reads 21a draws and 27 modules)"

# =============================================================================
# L. CALIBRATION AND LINEAR-MODEL SENSITIVITY (S17, S18)
# =============================================================================
run_step "28_calibration_analysis"    calibration_analysis.py               "19s"
run_step "28aa_combat_batch_corr"     combat_batch_correction.py            "23s (ComBat batch-correction sensitivity; Limitation 4)"
run_step "28a_decision_curve"         decision_curve_analysis.py            "~5m"
run_step "28b_compartment_reg_sens"   compartment_regression_sensitivity.py "~15m"
run_step "28c_lr_baseline"            lr_baseline.py                        "~10m"
run_step "28d_small_cohort_loo"       small_cohort_loo_refit.py             "~5m"

# =============================================================================
# M. WITH-REPLACEMENT VALIDATION (§3.8 negative control)
# Canonical derivation of the de-duplicated clean replacement assignment
# (confirmed artifacts AND artifact gene families excluded -- olfactory/taste/
# keratin/KRTAP/defensin/LCE/SPRR/VN1R, aligned with cross_cohort_substitution
# and druggable_replacement_targets). The hardcoded dicts in steps 05/05b
# mirror this output; the A->B zero-shot deltas are the §3.8 negative control.
# Depends on 27 (gene_replacement) + 03e (shap500 artifact audit).
# =============================================================================
run_step "29_adaptive_panel"          adaptive_panel_validation.py          "30s (canonical KRTAP/family-filtered replacement assignment)"

# =============================================================================
# N. CASE-STRENGTHENING / ROBUSTNESS (in-silico)
# #3 reference-based deconvolution (BRETIGEA), #2 incremental value over
# composition, #4 random-panel null for transfer, #6 OpenTargets genetic support.
# Prereq for #3/#2: BRETIGEA R package + exported markers --
#   Rscript -e 'install.packages("BRETIGEA"); library(BRETIGEA); \
#     write.csv(markers_df_brain,"src/bretigea_markers.csv",row.names=FALSE)'
# #3 depends on 11 (legacy z-score concordance); #2 imports the estimator from #3.
# =============================================================================
run_step "11b_deconv_bretigea"        deconv_reference_bretigea.py          "~5m (needs bretigea_markers.csv + step 11)"
run_step "11c_incremental_composition" incremental_over_composition.py       "~8m (imports estimator from 11b)"
# Transfer against random gene sets (main-text Figure/Table, S random-null section).
# 30 writes the annotated 17-gene panel (MyGeneInfo); 30a scores the 25-gene panel and
# the annotated 17 against 1,000 random gene sets per test set with the manuscript's
# zero-shot model (GPL16791 all and donor-disjoint, GSE76220, GSE122649, SRP064478).
run_step "30_panel_annotation"        panel_annotation.py                   "<1m (MyGeneInfo; writes lgbm_core25_panel_annotated.csv)"
run_step "30a_random_panel_null"      random_panel_null_cohorts.py          "~20m (25 + annotated 17; needs 22 for SRP064478)"
# Nested donor-grouped CV of the discovery procedure and its decomposition by step
# (S nested-CV section, Limitation 2). Within-cohort model of the main text.
run_step "30b_nested_panel_cv"        nested_panel_cv.py                    "~8m (15 outer folds)"
run_step "30c_nested_cv_decomp"       nested_cv_decomposition.py            "~13m"
# IAMB Markov blanket (S IAMB section). Needs the coffeeBreak library (IAMBSelector).
run_step "07ac_genetic_support"       genetic_support_opentargets.py        "~3m (OpenTargets ALS genetic_association)"
# The two single-nucleus steps need .h5 matrices that download_geo_data.sh does
# NOT fetch (they are per-GSM supplementary files, ~5.6 GB combined). They are
# gated on the data being present, exactly like step 22 is gated on Salmon
# output: without a gate, a fresh run following the README dies here under
# `set -euo pipefail` after ~28 h of completed work, on data nothing told the
# user to download.
#
# #1 single-nucleus per-cell validation. Prereq: the GSE219280 frontal+motor-cortex
# C9ALS+Control filtered .h5 (12+12 samples, ~4.6GB) in resources/GSE219280/suppl/
# (per-GSM files at ftp.ncbi.nlm.nih.gov/geo/samples/GSM...;
# cellBender_corrected_filtered.h5).
if compgen -G "$PROJ_ROOT/resources/GSE219280/suppl/*.h5" > /dev/null; then
    run_step "31_snrna_percell"       snrna_percell_validation.py           "~5m (needs GSE219280 .h5; h5py)"
else
    echo "[31_snrna_percell] SKIPPED — no .h5 in resources/GSE219280/suppl/ (see README Provenance)"
fi
# #2 generalisation to a non-C9 cohort. Prereq: GSE212630 Control + TDPhigh
# filtered_feature_bc_matrix.h5 (7+8 samples, ~1GB) in resources/GSE212630/suppl/.
if compgen -G "$PROJ_ROOT/resources/GSE212630/suppl/*.h5" > /dev/null; then
    run_step "32_snrna_gse212630"     snrna_validation_gse212630.py         "24s (needs GSE212630 .h5; non-C9 generalisation)"
else
    echo "[32_snrna_gse212630] SKIPPED — no .h5 in resources/GSE212630/suppl/ (see README Provenance)"
fi

# =============================================================================
# O. SHARED-MECHANISM APPRAISAL AND CROSS-DISEASE SPECIFICITY
# Prompted by Zhou et al. 2026 (Cell 189:1-17), which reports a TOP1-mediated
# somatic-damage mechanism shared by ALS, FTD and Alzheimer's disease.
# Step 34 tests that study's DNA-repair and cryptic-exon gene sets inside
# GSE153960 (it publishes no per-gene damage map and no open data, so no direct
# integration is possible). Step 35 scores the panel on four non-ALS
# neurodegenerative cohorts, each against ITS OWN in-cohort controls.
# Step 35 downloads its four cohorts on demand (~125 MB) if download_geo_data.sh
# has not already fetched them.
# =============================================================================
run_step "34_top1_axis_check"         top1_axis_check.py                    "~5m (needs mygene; network)"
run_step "35_cross_disease_spec"      cross_disease_specificity.py          "~40m (AD/FTLD-TDP/HD/PD + ALS positive control; 5-fold CV fold-averaged scoring + per-cohort LOO; network on first run)"
run_step "36_cross_disease_covar"     cross_disease_covariates.py           "41s (PMI/RIN/age confounding check; depends on 35)"
run_step "37_explainability_figs"     explainability_figures.py             "<1m (covariate/cross-disease/deconv explainability panels; needs 35, 36, 11a)"

# C9orf72 / onset / duration association (main-text C9 paragraph). Needs the NYGC
# consortium donor table (Zenodo 20025792, NYGC_Flagship_ST_1_3_Metadata.xlsx).
if [[ -f "$PROJ_ROOT/resources/NYGC_zenodo_20025792/NYGC_Flagship_ST_1_3_Metadata.xlsx" ]]; then
    run_step "38_nygc_clinical"       nygc_clinical_association.py          "<1m (needs the Zenodo ST1 table)"
else
    echo "[38_nygc_clinical] SKIPPED — Zenodo 20025792 ST1 table not in resources/NYGC_zenodo_20025792/"
fi
# coffeeBreak-derived panel and its random gene-set baseline (S coffeeBreak section).
# Needs the coffeeBreak search output (coffeebreak_equiv_run/deployment_mask.csv), which
# is not reproduced here (see README: the coffeeBreak searches ship their outputs).
if [[ -f "$SCRIPT_DIR/coffeebreak_equiv_run/deployment_mask.csv" ]]; then
    :  # keeps this branch valid where the public-repo sync drops the coffeeBreak step
else
    echo "[39_coffeebreak_panel] SKIPPED — coffeebreak_equiv_run/deployment_mask.csv not present"
fi
# =============================================================================
# P. INDEPENDENT SPINAL-CORD / MOTOR-CORTEX VALIDATION (Step 101)
# Five cohorts no panel-selection step saw (GSE255683, GSE287256, GSE26927,
# E-MTAB-8635, E-MTAB-2325), plus GSE26927's other disease arms, and the
# single-nucleus localisation of the panel in spinal cord (GSE287257).
# =============================================================================
run_step "40_new_cohort_validation"   new_cohort_validation.py              "~35m (5 cohorts x 2 panels x 1,000 random sets)"
run_step "41_snrna_spinal_cord"       snrna_spinal_cord.py                  "~5m (12 CellRanger .h5; extracts GSE287257_RAW.tar)"
# =============================================================================
# Final check. Not an analysis: it reports any output that is older than the
# panel_critical.csv it transitively depends on, which is how three separate
# batches of stale results were caught during this study. A clean run should
# show no STALE rows; investigate any that appear before trusting the PDFs.
# =============================================================================
run_step "99_panel_staleness_audit"   panel_staleness_audit.py              "<1s (verification only; no outputs)"

echo ""
echo "======================================================"
echo "All steps completed: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Outputs in: $SCRIPT_DIR"
echo "Logs in:    $LOG_DIR"
echo ""
echo "Manifest of figure -> script mapping in:"
echo "  $SCRIPT_DIR/reproduce/FIGURE_MANIFEST.md"
echo ""
echo "Now rebuild PDFs (xelatex required: supplementary.tex uses Unicode glyphs):"
echo "  cd $SCRIPT_DIR/manuscript"
echo "  xelatex supplementary.tex && xelatex manuscript.tex"
echo "  xelatex supplementary.tex && xelatex manuscript.tex   # 2nd pass for xr refs"
echo "======================================================"
