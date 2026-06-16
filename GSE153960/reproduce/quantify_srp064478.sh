#!/usr/bin/env bash
# =============================================================================
# Salmon quantification pipeline for SRP064478 (Ladd et al. 2017).
# Required for Step 22 (srp064478_validation.py).
#
# Prerequisites:
#   conda env: als_pipeline (must contain salmon)
#   Disk: ~100 GB for FASTQ downloads (auto-deleted after quantification)
#   Salmon index: built from Ensembl GRCh38 release 111 cDNA (see below)
#
# Build Salmon index (one-time, ~15 min, ~8 GB disk):
#   mkdir -p als_analysis/resources/SRP064478/index
#   wget -O /tmp/hg38_cdna.fa.gz \
#     https://ftp.ensembl.org/pub/release-111/fasta/homo_sapiens/cdna/Homo_sapiens.GRCh38.cdna.all.fa.gz
#   conda run -n als_pipeline salmon index \
#     -t /tmp/hg38_cdna.fa.gz \
#     -i als_analysis/resources/SRP064478/index/salmon_hg38 \
#     --gencode -p 8
#
# Run from coffeeBreak project root:
#   bash als_analysis/GSE153960/reproduce/quantify_srp064478.sh
# =============================================================================
set -euo pipefail

PIPELINE="$(cd "$(dirname "$0")/../../.." && pwd)/als_analysis/resources/SRP064478/run_parallel.sh"

if [[ ! -f "$PIPELINE" ]]; then
    echo "ERROR: run_parallel.sh not found at $PIPELINE"
    exit 1
fi

echo "[SRP064478] Launching parallel Salmon quantification (3 samples at a time)..."
echo "  Estimated time: 12-16 hours depending on network speed"
bash "$PIPELINE"
echo "[SRP064478] Done — check quant/ for 15 quant.sf files"
