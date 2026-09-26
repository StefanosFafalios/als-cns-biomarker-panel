#!/usr/bin/env bash
# =============================================================================
# Download all GEO datasets required for the ALS analysis.
# Run from the repository root before running run_all.sh.
#
# Total download size: ~2 GB (series matrices + supplementary count files)
# =============================================================================
set -euo pipefail

RES="$(cd "$(dirname "$0")/.." && pwd)/resources"

# Download once, and only once. Files are fetched to a .part sibling and moved
# into place atomically, so an interrupted transfer can never leave a truncated
# file that a later run would mistake for a completed download. Anything
# smaller than MIN_BYTES is treated as a failed fetch (NCBI serves an HTML
# error page under throttling, which is small and is NOT valid gzip).
MIN_BYTES=1000

dl() {
    local dest="$1"; local url="$2"
    mkdir -p "$(dirname "$dest")"
    if [[ -f "$dest" ]] && [[ $(stat -c%s "$dest" 2>/dev/null || echo 0) -gt $MIN_BYTES ]]; then
        echo "  already exists: $(basename "$dest")"
        return
    fi
    [[ -f "$dest" ]] && echo "  re-fetching undersized file: $(basename "$dest")"
    echo "  downloading: $(basename "$dest")"
    wget -q --show-progress -O "$dest.part" "$url"
    if [[ $(stat -c%s "$dest.part" 2>/dev/null || echo 0) -le $MIN_BYTES ]]; then
        rm -f "$dest.part"
        echo "  ERROR: $(basename "$dest") came back undersized (throttled?)" >&2
        return 1
    fi
    mv "$dest.part" "$dest"
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

echo "[GSE122649] Motor cortex cohort (validation)"
mkdir -p "$RES/GSE122649"/{soft,suppl}
# Fetched here rather than at analysis time: eleven scripts read this SOFT file,
# and re-fetching it per run made them fail under NCBI throttling.
dl "$RES/GSE122649/soft/GSE122649_family.soft.gz" \
   "$GEO_FTP/GSE122nnn/GSE122649/soft/GSE122649_family.soft.gz"
dl "$RES/GSE122649/GSE122649_RAW.tar" \
   "$GEO_FTP/GSE122nnn/GSE122649/suppl/GSE122649_RAW.tar"

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
   "$GEO_FTP/GSE67nnn/GSE67196/suppl/GSE67196_Petrucelli2015_ALS_genes.rawcount.txt.gz"

# ---------------------------------------------------------------------------
# Cross-disease specificity cohorts (Step 35, cross_disease_specificity.py)
# The script downloads these itself if absent; fetching them here keeps the
# one-shot download path complete.  Adds ~125 MB.
# ---------------------------------------------------------------------------

echo "[GSE125583] Alzheimer's disease, fusiform gyrus (AD 219 / ctrl 70)"
mkdir -p "$RES/GSE125583"/{matrix,suppl}
dl "$RES/GSE125583/matrix/GSE125583_series_matrix.txt.gz" \
   "$GEO_FTP/GSE125nnn/GSE125583/matrix/GSE125583_series_matrix.txt.gz"
dl "$RES/GSE125583/suppl/GSE125583_RAW.tar" \
   "$GEO_FTP/GSE125nnn/GSE125583/suppl/GSE125583_RAW.tar"

echo "[GSE13162] FTLD-U / FTLD-TDP microarray (39 FTLD-U / ctrl 17)"
mkdir -p "$RES/GSE13162/matrix"
dl "$RES/GSE13162/matrix/GSE13162_series_matrix.txt.gz" \
   "$GEO_FTP/GSE13nnn/GSE13162/matrix/GSE13162_series_matrix.txt.gz"
# GPL571 probe->symbol annotation is fetched by the script from the GEO
# acc.cgi text view (not available under the FTP platform annot/ path).

echo "[GSE64810] Huntington's disease, BA9 (HD 20 / ctrl 49)"
mkdir -p "$RES/GSE64810"/{matrix,suppl}
dl "$RES/GSE64810/matrix/GSE64810_series_matrix.txt.gz" \
   "$GEO_FTP/GSE64nnn/GSE64810/matrix/GSE64810_series_matrix.txt.gz"
dl "$RES/GSE64810/suppl/GSE64810_mlhd_DESeq2_norm_counts_adjust.txt.gz" \
   "$GEO_FTP/GSE64nnn/GSE64810/suppl/GSE64810_mlhd_DESeq2_norm_counts_adjust.txt.gz"

echo "[GSE68719] Parkinson's disease, BA9 (PD 29 / ctrl 44)"
mkdir -p "$RES/GSE68719"/{matrix,suppl}
dl "$RES/GSE68719/matrix/GSE68719_series_matrix.txt.gz" \
   "$GEO_FTP/GSE68nnn/GSE68719/matrix/GSE68719_series_matrix.txt.gz"
dl "$RES/GSE68719/suppl/GSE68719_mlpd_PCG_DESeq2_norm_counts.txt.gz" \
   "$GEO_FTP/GSE68nnn/GSE68719/suppl/GSE68719_mlpd_PCG_DESeq2_norm_counts.txt.gz"

echo "Done. All GEO files downloaded to $RES"
echo "Next: run reproduce/run_all.sh"
