"""Supplementary robustness check: DE-based GSEA on all 47,822 pre-filtered genes.

Ranks all 47,822 pre-filtered genes by signed Welch t-statistic (ALS vs Control)
and runs gseapy.prerank with the same gene sets and settings as Step 13 (SHAP-GSEA).

Purpose: validate that SelectFromModel did not introduce pathway-level bias.
If top pathways substantially overlap with Step 13, the 3,704-gene SHAP approach
is unbiased with respect to pathway discovery.  Divergences identify biology present
in the full transcriptome but filtered out by SFM — a useful Discussion point.

Outputs
-------
de_gsea_robustness/               directory with gseapy enrichment plots
de_gsea_robustness_results.csv    full NES / FDR table
de_gsea_robustness_statistics.txt human-readable summary with Step-13 comparison
"""

from __future__ import annotations

import sys
import time
import pathlib
import warnings

import mygene
import numpy as np
import pandas as pd

ALS_DIR = pathlib.Path(__file__).parents[1]
SCRIPT_DIR = pathlib.Path(__file__).parent

sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))
sys.path.insert(0, str(ALS_DIR))

from utils import load_dataset  # noqa: E402

_PREFILTER_X_NPY = SCRIPT_DIR / "lgbm_prefilter_X.npy"
_PREFILTER_NAMES = SCRIPT_DIR / "lgbm_prefilter_names.txt"

# Step-13 reference: top KEGG / Reactome hits for comparison
_STEP13_KEGG_SIG = {
    "Hematopoietic cell lineage",
    "C-type lectin receptor signaling pathway",
    "Calcium signaling pathway",
    "Influenza A",
    "Phospholipase D signaling pathway",
    "MAPK signaling pathway",
}
_STEP13_REACTOME_SIG = {"Sensory Perception R-HSA-9709957"}

OUT_DIR = SCRIPT_DIR / "de_gsea_robustness"
OUT_CSV = SCRIPT_DIR / "de_gsea_robustness_results.csv"
OUT_TXT = SCRIPT_DIR / "de_gsea_robustness_statistics.txt"

GENE_SETS = [
    "GO_Biological_Process_2023",
    "KEGG_2021_Human",
    "Reactome_2022",
]
FDR_THRESH = 0.25
TOP_N = 15
PLATFORM = "GPL24676"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_signed_t(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Vectorised Welch t-statistic (positive = higher in ALS).

    Args:
        X: float32 array (n_samples, n_features), already log1p + PC-residualised.
        y: int array (n_samples,), 1=ALS 0=Control.

    Returns:
        Array of shape (n_features,) with signed t-statistic per gene.
    """
    X = X.astype(np.float64)
    als = X[y == 1]
    ctrl = X[y == 0]
    n_a, n_c = als.shape[0], ctrl.shape[0]

    mu_a = als.mean(axis=0)
    mu_c = ctrl.mean(axis=0)
    var_a = als.var(axis=0, ddof=1)
    var_c = ctrl.var(axis=0, ddof=1)

    se = np.sqrt(var_a / n_a + var_c / n_c)
    se = np.where(se == 0, np.finfo(float).tiny, se)  # avoid /0
    return (mu_a - mu_c) / se


def _map_ensembl_to_symbols(versioned_ids: list[str]) -> dict[str, str]:
    """Map versioned ENSG IDs → HGNC symbols via MyGeneInfo (batched, with retry)."""
    bases = [e.split(".")[0] for e in versioned_ids]
    mg_client = mygene.MyGeneInfo()

    batch_size = 500
    base_map: dict[str, str] = {}

    for start in range(0, len(bases), batch_size):
        chunk = bases[start : start + batch_size]
        batch_end = min(start + batch_size, len(bases))
        print(f"    MyGeneInfo {start + 1}–{batch_end} / {len(bases)} ...", flush=True)

        for attempt in range(3):
            try:
                hits = mg_client.querymany(
                    chunk,
                    scopes="ensembl.gene",
                    fields="symbol",
                    species="human",
                    verbose=False,
                )
                for h in hits:
                    if "symbol" in h:
                        base_map[h["query"]] = h["symbol"]
                break
            except Exception as exc:
                wait = 5 * (2**attempt)
                print(f"    Attempt {attempt + 1} failed: {exc}. Retry in {wait}s ...")
                time.sleep(wait)
        else:
            print(f"    WARNING: all retries failed for batch {start}–{batch_end}")

    return {v: base_map.get(b, b) for v, b in zip(versioned_ids, bases)}


def _build_prerank_series(
    signed_t: np.ndarray,
    feature_names: list[str],
    ensg_to_sym: dict[str, str],
) -> pd.Series:
    """Convert per-feature signed t-statistic to HGNC-symbol-indexed Series.

    Multi-ENSG → same symbol: keep the one with the largest |t|.
    Drop unresolved IDs (still look like ENSG...).
    """
    sym_to_t: dict[str, float] = {}
    for feat, t_val in zip(feature_names, signed_t):
        sym = ensg_to_sym.get(feat, feat)
        if sym.startswith("ENSG"):
            continue
        if sym not in sym_to_t or abs(t_val) > abs(sym_to_t[sym]):
            sym_to_t[sym] = t_val

    return pd.Series(sym_to_t).sort_values(ascending=False)


def _run_gsea(rnk: pd.Series) -> pd.DataFrame:
    """Run gseapy.prerank for all GENE_SETS and return concatenated results."""
    import gseapy as gp

    OUT_DIR.mkdir(exist_ok=True)
    all_results: list[pd.DataFrame] = []

    for gene_set in GENE_SETS:
        print(f"  Running GSEA: {gene_set} ...", flush=True)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            pre_res = gp.prerank(
                rnk=rnk,
                gene_sets=gene_set,
                outdir=str(OUT_DIR / gene_set),
                permutation_num=1000,
                seed=42,
                verbose=False,
                min_size=10,
                max_size=500,
            )
        df = pre_res.res2d.copy()
        df.insert(0, "gene_set_library", gene_set)
        all_results.append(df)

    return pd.concat(all_results, ignore_index=True)


def _col(df: pd.DataFrame, *candidates: str) -> str | None:
    for c in candidates:
        match = next((x for x in df.columns if x.lower() == c.lower()), None)
        if match:
            return match
    return None


def _write_statistics(
    results: pd.DataFrame,
    rnk: pd.Series,
    n_features: int,
    n_mapped: int,
    signed_t: np.ndarray,
    lines: list[str],
) -> None:
    """Append per-library tables and Step-13 comparison to lines."""
    for gsl in GENE_SETS:
        sub = results[results["gene_set_library"] == gsl].copy()
        fdr_col = _col(sub, "fdr q-val", "fdr", "padj")
        nes_col = _col(sub, "nes")
        term_col = _col(sub, "term") or sub.columns[1]

        if fdr_col is None or nes_col is None:
            lines.append(f"\n{gsl}: unrecognised columns {sub.columns.tolist()}")
            continue

        sub = sub.sort_values(fdr_col)
        sig = sub[sub[fdr_col] < FDR_THRESH]

        lines.append(f"\n{'='*70}")
        lines.append(f"Gene set library: {gsl}")
        lines.append(
            f"Significant pathways (FDR < {FDR_THRESH}): {len(sig)} / {len(sub)}"
        )
        lines.append(f"{'Term':<55} {'NES':>7} {'FDR':>10}")
        lines.append("-" * 74)

        show = sig.head(TOP_N) if len(sig) > 0 else sub.head(TOP_N)
        for _, row in show.iterrows():
            term = str(row[term_col])
            if len(term) > 54:
                term = term[:51] + "..."
            lines.append(
                f"{term:<55} {float(row[nes_col]):>7.3f} {float(row[fdr_col]):>10.3e}"
            )
        if len(sig) == 0:
            lines.append("  (no pathways met FDR threshold; showing top 15 by FDR)")

    # Step-13 comparison
    lines.append(f"\n{'='*70}")
    lines.append("COMPARISON TO STEP 13 (SHAP-GSEA on 3,704 genes)")
    lines.append("-" * 70)

    kegg_sub = results[results["gene_set_library"] == "KEGG_2021_Human"].copy()
    react_sub = results[results["gene_set_library"] == "Reactome_2022"].copy()

    kegg_fdr = _col(kegg_sub, "fdr q-val", "fdr", "padj")
    react_fdr = _col(react_sub, "fdr q-val", "fdr", "padj")
    kegg_term = _col(kegg_sub, "term") or kegg_sub.columns[1]
    react_term = _col(react_sub, "term") or react_sub.columns[1]

    if kegg_fdr:
        kegg_sig_de = set(
            kegg_sub.loc[kegg_sub[kegg_fdr] < FDR_THRESH, kegg_term]
        )
        overlap_kegg = kegg_sig_de & _STEP13_KEGG_SIG
        lines.append(
            f"KEGG sig (DE-GSEA, FDR<{FDR_THRESH}): {len(kegg_sig_de)} pathways"
        )
        lines.append(
            f"KEGG overlap with Step-13 sig set (n=6): {len(overlap_kegg)} / 6"
        )
        if overlap_kegg:
            for p in sorted(overlap_kegg):
                lines.append(f"  [MATCH]  {p}")
        missing_kegg = _STEP13_KEGG_SIG - kegg_sig_de
        if missing_kegg:
            for p in sorted(missing_kegg):
                lines.append(f"  [ABSENT] {p}")

    if react_fdr:
        react_sig_de = set(
            react_sub.loc[react_sub[react_fdr] < FDR_THRESH, react_term]
        )
        overlap_react = react_sig_de & _STEP13_REACTOME_SIG
        lines.append(
            f"\nReactome sig (DE-GSEA): {len(react_sig_de)} pathways"
        )
        lines.append(
            f"Reactome overlap with Step-13 sig set (n=1): {len(overlap_react)} / 1"
        )
        if overlap_react:
            lines.append(f"  [MATCH]  Sensory Perception")
        else:
            lines.append(f"  [ABSENT] Sensory Perception R-HSA-9709957")

    lines.append("")
    lines.append(
        "Interpretation: overlap confirms SFM did not introduce pathway bias;"
    )
    lines.append(
        "divergences identify biology present in full transcriptome but filtered"
    )
    lines.append("by SelectFromModel (see Discussion).")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run DE-based GSEA on all 47,822 pre-filtered genes."""
    print("=" * 65)
    print("DE-GSEA supplementary robustness check")
    print("47,822 pre-filtered genes | signed Welch t-statistic")
    print("=" * 65)

    # Load prefilter matrix and y labels
    print("\n[1] Loading prefilter matrix and labels ...")
    X = np.load(_PREFILTER_X_NPY)  # (874, 47822)
    feature_names: list[str] = _PREFILTER_NAMES.read_text().splitlines()
    assert X.shape == (874, len(feature_names)), (
        f"Shape mismatch: X={X.shape}, names={len(feature_names)}"
    )

    (ds,) = load_dataset(
        "GSE153960", platform=PLATFORM, resources_dir=ALS_DIR / "resources"
    )
    y = ds.y.values.astype(int)
    print(f"  X: {X.shape} | ALS: {y.sum()} | Control: {(y==0).sum()}")

    # Compute signed Welch t-statistic
    print("\n[2] Computing signed t-statistics ...")
    signed_t = _compute_signed_t(X, y)
    print(f"  t-stat range: [{signed_t.min():.2f}, {signed_t.max():.2f}]")
    print(f"  Positive (ALS-up): {(signed_t > 0).sum()} | Negative: {(signed_t < 0).sum()}")

    # Map ENSG IDs to HGNC symbols
    print("\n[3] Mapping ENSG IDs → HGNC symbols via MyGeneInfo ...")
    ensg_to_sym = _map_ensembl_to_symbols(feature_names)
    n_mapped = sum(1 for v in ensg_to_sym.values() if not v.startswith("ENSG"))
    print(f"  Resolved {n_mapped} / {len(feature_names)} features to HGNC symbols")

    # Build prerank series
    rnk = _build_prerank_series(signed_t, feature_names, ensg_to_sym)
    print(f"  Prerank list: {len(rnk)} unique symbols")
    print(f"  Top-5 ALS-up: {list(rnk.head().items())}")
    print(f"  Top-5 ctrl-up: {list(rnk.tail().items())}")

    # Run GSEA
    print("\n[4] Running preranked GSEA ...")
    results = _run_gsea(rnk)
    results.to_csv(OUT_CSV, index=False)
    print(f"  Full results saved: {OUT_CSV}")

    # Write statistics
    lines: list[str] = [
        "DE-GSEA Robustness Check — 47,822 Pre-filtered Genes",
        "=" * 70,
        f"Feature space: {len(feature_names)} pre-filtered genes (lgbm_prefilter_X.npy)",
        f"Ranking metric: signed Welch t-statistic (ALS vs Control, positive = ALS-up)",
        f"Samples: n={len(y)} | ALS={y.sum()} | Control={(y==0).sum()}",
        f"HGNC-mapped symbols: {n_mapped} / {len(feature_names)} ({n_mapped/len(feature_names)*100:.1f}%)",
        f"Prerank list after deduplication: {len(rnk)} unique symbols",
        f"Gene set libraries: {', '.join(GENE_SETS)}",
        "Permutations: 1000 | Min gene-set size: 10 | Max: 500",
        f"FDR significance threshold: {FDR_THRESH}",
        "",
        "PRERANK LIST STATISTICS",
        f"  Positive scores (ALS-up): {(rnk > 0).sum()}",
        f"  Negative scores (ctrl-up): {(rnk < 0).sum()}",
        f"  Score range: [{rnk.min():.3f}, {rnk.max():.3f}]",
        f"  Note: unlike SHAP-GSEA (Step 13), all genes have non-zero scores",
        f"  (t-stat is defined for every gene with sufficient variance).",
    ]

    _write_statistics(results, rnk, len(feature_names), n_mapped, signed_t, lines)

    lines += [
        "",
        "=" * 70,
        "OUTPUT FILES",
        f"  Full NES/FDR table: {OUT_CSV}",
        f"  Enrichment plots:   {OUT_DIR}/",
        f"  Statistics:         {OUT_TXT}",
    ]

    OUT_TXT.write_text("\n".join(lines) + "\n")
    print(f"\n  Statistics written: {OUT_TXT}")
    print("Done.")


if __name__ == "__main__":
    main()
