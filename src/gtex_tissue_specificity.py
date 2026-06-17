"""
GTEx Tissue Specificity — ALS Panel Protein-Coding Genes

Queries GTEx v8 median TPM per tissue for each of the 15 protein-coding panel
genes. Computes CNS specificity scores and validates the per-tissue biological
story (CKMT2 motor-cortex depletion, SERTAD1 CNS expression, CLDN5/VWF
endothelial specificity, etc.).
"""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import requests

SCRIPT_DIR = Path(__file__).parent

GTEX_API = "https://gtexportal.org/api/v2"
DATASET = "gtex_v8"

_PROTEIN_CODING = [
    "MECOM", "SERTAD1", "FCN3", "PROS1", "ANGPT2", "EMP1",
    "TINAGL1", "CKMT2", "VWF", "CLDN5", "NR4A1", "SOHLH2",
    "HEXB", "MCEE", "SLC37A2",
]

_DIRECTION: dict[str, str] = {
    "MECOM": "down", "SERTAD1": "up", "FCN3": "up", "PROS1": "up",
    "ANGPT2": "down", "EMP1": "up", "TINAGL1": "down", "CKMT2": "down",
    "VWF": "down", "CLDN5": "down", "NR4A1": "down", "SOHLH2": "up",
    "HEXB": "up", "MCEE": "up", "SLC37A2": "up",
}

# Canonical CNS tissues in GTEx v8 (tissueSiteDetailId)
_CNS_TISSUES = {
    "Brain_Cortex", "Brain_Frontal_Cortex_BA9", "Brain_Anterior_cingulate_cortex_BA24",
    "Brain_Spinal_cord_cervical_c-1", "Brain_Amygdala", "Brain_Caudate_basal_ganglia",
    "Brain_Cerebellum", "Brain_Cerebellar_Hemisphere", "Brain_Hippocampus",
    "Brain_Hypothalamus", "Brain_Nucleus_accumbens_basal_ganglia",
    "Brain_Putamen_basal_ganglia", "Brain_Substantia_nigra", "Brain_Putamen_basal_ganglia",
}


def _get_gencode_id(symbol: str) -> str | None:
    """Look up GTEx v26 gencodeId for a gene symbol."""
    try:
        r = requests.get(f"{GTEX_API}/reference/gene",
                         params={"geneId": symbol}, timeout=15)
        r.raise_for_status()
        data = r.json().get("data", [])
        if data:
            return data[0]["gencodeId"]
    except Exception:
        pass
    return None


def _get_median_expression(gencode_id: str) -> list[dict]:
    """Return list of {tissueSiteDetailId, median} for a gencodeId."""
    for attempt in range(3):
        try:
            r = requests.get(
                f"{GTEX_API}/expression/medianGeneExpression",
                params={"gencodeId": gencode_id, "datasetId": DATASET},
                timeout=20,
            )
            r.raise_for_status()
            return r.json().get("data", [])
        except Exception:
            if attempt == 2:
                return []
            time.sleep(2 ** attempt)
    return []


def _cns_specificity(tpm_dict: dict[str, float]) -> float:
    """Ratio of mean CNS TPM to mean non-CNS TPM. >1 = CNS-enriched."""
    cns = [v for k, v in tpm_dict.items() if k in _CNS_TISSUES]
    non_cns = [v for k, v in tpm_dict.items() if k not in _CNS_TISSUES]
    if not cns or not non_cns or np.mean(non_cns) < 1e-6:
        return float("nan")
    return float(np.mean(cns) / np.mean(non_cns))


def main() -> None:
    print("=" * 65)
    print("GTEx Tissue Specificity — ALS Panel Genes")
    print("=" * 65)

    all_records: dict[str, dict[str, float]] = {}
    gencode_ids: dict[str, str] = {}
    specs: dict[str, float] = {}

    for gene in _PROTEIN_CODING:
        print(f"  [{gene}] looking up gencodeId ...", end=" ", flush=True)
        gcid = _get_gencode_id(gene)
        if not gcid:
            print("NOT FOUND — skipping")
            continue
        gencode_ids[gene] = gcid
        print(f"{gcid}", end=" | ", flush=True)

        tpm_data = _get_median_expression(gcid)
        if not tpm_data:
            print("no expression data")
            continue
        tpm_dict = {x["tissueSiteDetailId"]: float(x["median"]) for x in tpm_data}
        all_records[gene] = tpm_dict
        spec = _cns_specificity(tpm_dict)
        specs[gene] = spec
        print(f"n_tissues={len(tpm_dict)}, CNS_spec={spec:.2f}")
        time.sleep(0.3)

    if not all_records:
        print("No data retrieved from GTEx.")
        return

    # Build expression matrix: tissues × genes
    all_tissues = sorted({t for d in all_records.values() for t in d})
    expr_df = pd.DataFrame(
        {gene: [all_records.get(gene, {}).get(t, np.nan) for t in all_tissues]
         for gene in _PROTEIN_CODING if gene in all_records},
        index=all_tissues,
    )
    spec_df = (
        pd.DataFrame({"Gene": list(specs.keys()), "CNS_specificity": list(specs.values())})
        .sort_values("CNS_specificity", ascending=False)
        .reset_index(drop=True)
    )

    # Save
    expr_df.to_csv(SCRIPT_DIR / "gtex_expression_matrix.csv")
    spec_df.to_csv(SCRIPT_DIR / "gtex_cns_specificity.csv", index=False)

    _plot(expr_df, spec_df, all_records)
    _write_statistics(expr_df, spec_df, all_records, gencode_ids)
    print("\nDone. Outputs: gtex_tissue_specificity.png  gtex_tissue_specificity_statistics.txt")


def _plot(
    expr_df: pd.DataFrame,
    spec_df: pd.DataFrame,
    all_records: dict[str, dict[str, float]],
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(22, 9))

    # Panel A — CNS specificity ratio (bar chart)
    ax = axes[0]
    genes = spec_df["Gene"].values
    vals = spec_df["CNS_specificity"].values
    colors = ["#C62828" if _DIRECTION.get(g) == "up" else "#1565C0" for g in genes]
    ax.barh(genes[::-1], vals[::-1], color=colors[::-1])
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1, label="CNS = body avg")
    ax.set_xlabel("CNS specificity ratio (CNS TPM / non-CNS TPM)")
    ax.set_title("GTEx CNS specificity\nred=up, blue=down in ALS")
    ax.legend(fontsize=8)
    ax.tick_params(axis="y", labelsize=8)

    # Panel B — expression heatmap (log1p TPM, top CNS tissues + selected body tissues)
    ax = axes[1]
    highlight_tissues = [
        t for t in expr_df.index if "Brain" in t or "Spinal" in t
        or t in {"Liver", "Kidney_Cortex", "Heart_Left_Ventricle",
                 "Lung", "Artery_Tibial", "Blood", "Muscle_Skeletal"}
    ]
    sub = np.log1p(expr_df.loc[highlight_tissues].fillna(0))
    im = ax.imshow(sub.T, aspect="auto", cmap="YlOrRd",
                   vmin=0, vmax=sub.max().max())
    ax.set_xticks(range(len(highlight_tissues)))
    ax.set_xticklabels(
        [t.replace("Brain_", "").replace("_", " ")[:20] for t in highlight_tissues],
        rotation=90, fontsize=9,
    )
    ax.set_yticks(range(len(sub.columns)))
    ax.set_yticklabels(sub.columns, fontsize=8)
    plt.colorbar(im, ax=ax, label="log1p(TPM)")
    ax.set_title("log1p(Median TPM) — CNS + selected tissues")

    # Panel C — per-gene expression across brain regions for priority targets
    ax = axes[2]
    priority = ["SERTAD1", "FCN3", "CKMT2", "CLDN5", "VWF"]
    brain_tissues = sorted([t for t in expr_df.index if "Brain" in t or "Spinal" in t])
    for i, gene in enumerate(p for p in priority if p in expr_df.columns):
        vals_gene = [all_records.get(gene, {}).get(t, np.nan) for t in brain_tissues]
        ax.plot(range(len(brain_tissues)), vals_gene,
                marker="o", markersize=4, linewidth=1.5, label=gene)
    ax.set_xticks(range(len(brain_tissues)))
    ax.set_xticklabels(
        [t.replace("Brain_", "").replace("_", " ")[:15] for t in brain_tissues],
        rotation=90, fontsize=9,
    )
    ax.set_yscale("log")
    ax.set_ylabel("Median TPM (GTEx v8, log scale)")
    ax.set_title("Brain / spinal cord expression\n(priority drug targets)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, axis="y", which="both", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(SCRIPT_DIR / "gtex_tissue_specificity.png", dpi=150, bbox_inches="tight")
    plt.close()


def _write_statistics(
    expr_df: pd.DataFrame,
    spec_df: pd.DataFrame,
    all_records: dict[str, dict[str, float]],
    gencode_ids: dict[str, str],
) -> None:
    lines = [
        "GTEx v8 Tissue Specificity — ALS Panel Protein-Coding Genes",
        "=" * 65,
        f"Dataset: GTEx v8  |  Genes queried: {len(_PROTEIN_CODING)}",
        f"Genes with data: {len(all_records)}",
        "",
        "CNS SPECIFICITY (CNS mean TPM / non-CNS mean TPM)",
        "-" * 65,
        f"{'Gene':<12} {'CNS_spec':>10} {'Direction':<10} {'gencodeId':<20}",
        "-" * 65,
    ]
    for _, row in spec_df.iterrows():
        gene = row["Gene"]
        lines.append(
            f"{gene:<12} {row['CNS_specificity']:>10.3f} "
            f"{_DIRECTION.get(gene, ''):<10} {gencode_ids.get(gene, ''):<20}"
        )

    # Per-gene top-5 tissues by expression
    lines += ["", "TOP-5 EXPRESSING TISSUES PER GENE", "-" * 65]
    for gene in _PROTEIN_CODING:
        if gene not in all_records:
            continue
        tpm_sorted = sorted(all_records[gene].items(), key=lambda x: x[1], reverse=True)
        top5 = tpm_sorted[:5]
        lines.append(f"\n{gene} ({_DIRECTION.get(gene, '')} in ALS):")
        for tissue, tpm in top5:
            cns_flag = " [CNS]" if tissue in _CNS_TISSUES else ""
            lines.append(f"  {tissue:<50} TPM={tpm:.2f}{cns_flag}")

    (SCRIPT_DIR / "gtex_tissue_specificity_statistics.txt").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
