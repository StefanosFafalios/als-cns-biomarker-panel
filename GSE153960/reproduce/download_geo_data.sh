#!/usr/bin/env bash
# =============================================================================
# Download all GEO datasets required for the ALS analysis.
# Run from the coffeeBreak project root before running run_all.sh.
#
# Total download size: ~2 GB (series matrices + supplementary count files)
# =============================================================================
set -euo pipefail

RES="$(cd "$(dirname "$0")/../../.." && pwd)/als_analysis/resources"

dl() {
    local dest="$1"; local url="$2"
    mkdir -p "$(dirname "$dest")"
    [[ -f "$dest" ]] && echo "  already exists: $(basename "$dest")" && return
    echo "  downloading: $(basename "$dest")"
    wget -q --show-progress -O "$dest" "$url"
}

GEO_FTP="https://ftp.ncbi.nlm.nih.gov/geo/series"

echo "[GSE153960] NYGC ALS Consortium (training + GPL16791 cross-platform)"
mkdir -p "$RES/GSE153960"/{matrix,soft}
dl "$RES/GSE153960/matrix/GSE153960-GPL24676_series_matrix.txt.gz" \
   "$GEO_FTP/GSE153nnn/GSE153960/matrix/GSE153960-GPL24676_series_matrix.txt.gz"
dl "$RES/GSE153960/matrix/GSE153960-GPL16791_series_matrix.txt.gz" \
   "$GEO_FTP/GSE153nnn/GSE153960/matrix/GSE153960-GPL16791_series_matrix.txt.gz"
dl "$RES/GSE153960/soft/GSE153960_family.soft.gz" \
   "$GEO_FTP/GSE153nnn/GSE153960/soft/GSE153960_family.soft.gz"

echo "[GSE76220] LCM lumbar motor neuron cohort (Step 5)"
mkdir -p "$RES/GSE76220"/{matrix,soft,suppl}
dl "$RES/GSE76220/matrix/GSE76220_series_matrix.txt.gz" \
   "$GEO_FTP/GSE76nnn/GSE76220/matrix/GSE76220_series_matrix.txt.gz"
dl "$RES/GSE76220/soft/GSE76220_family.soft.gz" \
   "$GEO_FTP/GSE76nnn/GSE76220/soft/GSE76220_family.soft.gz"
dl "$RES/GSE76220/suppl/GSE76220_ALS_LCM_MPR_gene_expression.txt.gz" \
   "$GEO_FTP/GSE76nnn/GSE76220/suppl/GSE76220_ALS_LCM_MPR_gene_expression.txt.gz"

echo "[GSE122649] Motor cortex cohort — additional_cohort_gse122649.py downloads RAW.tar automatically"

echo "[GSE234297] Whole-blood cohort (Step 14)"
mkdir -p "$RES/GSE234297"/{matrix,suppl}
dl "$RES/GSE234297/matrix/GSE234297_series_matrix.txt.gz" \
   "$GEO_FTP/GSE234nnn/GSE234297/matrix/GSE234297_series_matrix.txt.gz"
dl "$RES/GSE234297/suppl/GSE234297_gene_raw_counts.txt.gz" \
   "$GEO_FTP/GSE234nnn/GSE234297/suppl/GSE234297_gene_raw_counts.txt.gz"
dl "$RES/GSE234297/suppl/GSE234297_gene_offset_matrix.txt.gz" \
   "$GEO_FTP/GSE234nnn/GSE234297/suppl/GSE234297_gene_offset_matrix.txt.gz"

echo "[GSE67196] Regional ALS scoring (Step 20)"
mkdir -p "$RES/GSE67196"
dl "$RES/GSE67196/GSE67196_rawcount.txt.gz" \
   "$GEO_FTP/GSE67nnn/GSE67196/suppl/GSE67196_rawcount.txt.gz"

echo "Done. All GEO files downloaded to $RES"
echo "Next: run als_analysis/GSE153960/reproduce/run_all.sh"
