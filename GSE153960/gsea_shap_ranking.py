"""GSEA on 3,704-gene SHAP-ranked list (Step 13).

Preranked GSEA using signed mean SHAP values from the final iterative
LightGBM model as the ranking metric.  Signed SHAP (positive → ALS,
negative → control) provides a biologically meaningful directional metric
for pathway enrichment — unlike a raw unsigned importance score.

Outputs
-------
gsea_shap_ranking/          directory with gseapy enrichment plots
gsea_shap_ranking_results.csv   full NES / FDR table (all gene sets)
gsea_shap_ranking_statistics.txt  human-readable top pathways per gene set
"""

from __future__ import annotations

import pathlib
import textwrap
import urllib.request

import mygene
import numpy as np
import pandas as pd

SCRIPT_DIR = pathlib.Path(__file__).parent

SHAP_CSV = SCRIPT_DIR / "lgbm_iter_final_shap_ranking.csv"
SHAP_NPY = SCRIPT_DIR / "lgbm_iter_final_shap_values.npy"

OUT_DIR = SCRIPT_DIR / "gsea_shap_ranking"
OUT_CSV = SCRIPT_DIR / "gsea_shap_ranking_results.csv"
OUT_TXT = SCRIPT_DIR / "gsea_shap_ranking_statistics.txt"

# Gene sets available in gseapy Enrichr library
GENE_SETS = [
    "GO_Biological_Process_2023",
    "KEGG_2021_Human",
    "Reactome_2022",
]

FDR_THRESH = 0.25  # standard GSEA FDR cutoff
TOP_N_PER_SET = 15  # pathways to report per gene set


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_signed_shap() -> pd.Series:
    """Return pd.Series(mean signed SHAP, index=versioned ENSG feature names).

    The .npy matrix has shape (n_samples, 3704) with columns in SHAP rank order
    (rank 1 = highest mean |SHAP|).  The CSV feature column gives the same order.
    """
    shap_df = pd.read_csv(SHAP_CSV)
    features: list[str] = shap_df["feature"].tolist()

    # Raw signed SHAP matrix — columns align with feature order in CSV
    shap_matrix: np.ndarray = np.load(SHAP_NPY)  # (n_samples, 3704)
    if shap_matrix.shape[1] != len(features):
        raise ValueError(
            f"Shape mismatch: npy has {shap_matrix.shape[1]} features, "
            f"CSV has {len(features)}"
        )

    mean_signed = shap_matrix.mean(axis=0)  # (3704,)
    return pd.Series(mean_signed, index=features)


def _map_ensembl_to_symbols(versioned_ids: list[str]) -> dict[str, str]:
    """Map versioned ENSG IDs → HGNC symbols via MyGeneInfo.

    Unmapped IDs fall back to the base ENSG ID (no version suffix).
    Batch requests 500 at a time with up to 3 retries per batch.
    """
    import time

    bases = [e.split(".")[0] for e in versioned_ids]
    mg_client = mygene.MyGeneInfo()

    batch_size = 500
    base_map: dict[str, str] = {}

    for start in range(0, len(bases), batch_size):
        chunk = bases[start : start + batch_size]
        batch_end = min(start + batch_size, len(bases))
        print(f"    Querying MyGeneInfo {start + 1}–{batch_end} / {len(bases)} ...")

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
                break  # success
            except Exception as exc:
                wait = 5 * (2 ** attempt)
                print(f"    Attempt {attempt + 1} failed: {exc}. Retrying in {wait}s ...")
                time.sleep(wait)
        else:
            print(f"    WARNING: all retries failed for batch {start}–{batch_end}; skipping")

    return {
        versioned: base_map.get(base, base)
        for versioned, base in zip(versioned_ids, bases)
    }


def _build_prerank_series(
    signed_shap: pd.Series, ensg_to_sym: dict[str, str]
) -> pd.Series:
    """Convert ENSG-indexed signed SHAP series to symbol-indexed prerank series.

    Handles:
    - Multiple ENSG IDs mapping to the same symbol → keep the one with the
      largest absolute SHAP value (most discriminative)
    - ENSG IDs that failed to resolve to a real symbol → drop (cannot be used
      by gseapy pathway libraries which expect HGNC symbols)
    """
    sym_to_shap: dict[str, float] = {}
    for versioned, shap_val in signed_shap.items():
        sym = ensg_to_sym.get(str(versioned), str(versioned))
        # Drop unresolved IDs (still look like ENSG...)
        if sym.startswith("ENSG"):
            continue
        if sym not in sym_to_shap or abs(shap_val) > abs(sym_to_shap[sym]):
            sym_to_shap[sym] = shap_val

    rnk = pd.Series(sym_to_shap).sort_values(ascending=False)
    return rnk


def _run_gsea(rnk: pd.Series) -> pd.DataFrame:
    """Run gseapy.prerank for all GENE_SETS, return concatenated results."""
    try:
        import gseapy as gp
    except ImportError as exc:
        raise ImportError(
            "gseapy is required: pip install gseapy"
        ) from exc

    OUT_DIR.mkdir(exist_ok=True)
    all_results: list[pd.DataFrame] = []

    for gene_set in GENE_SETS:
        print(f"  Running GSEA: {gene_set} ...")
        pre_res = gp.prerank(
            rnk=rnk,
            gene_sets=gene_set,
            outdir=str(OUT_DIR / gene_set.replace(" ", "_")),
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


def _write_statistics(results: pd.DataFrame, rnk: pd.Series, lines: list[str]) -> None:
    """Append top-pathway tables to lines list."""
    for gsl in GENE_SETS:
        sub = results[results["gene_set_library"] == gsl].copy()
        # Normalise column names (gseapy may vary slightly between versions)
        fdr_col = next(
            (c for c in sub.columns if "fdr" in c.lower()), None
        )
        nes_col = next(
            (c for c in sub.columns if c.lower() == "nes"), None
        )
        term_col = next(
            (c for c in sub.columns if "term" in c.lower()), "Term"
        )
        if fdr_col is None or nes_col is None:
            lines.append(f"\n{gsl}: could not parse result columns: {sub.columns.tolist()}")
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

        show = sig.head(TOP_N_PER_SET) if len(sig) > 0 else sub.head(TOP_N_PER_SET)
        for _, row in show.iterrows():
            term = str(row[term_col])
            if len(term) > 54:
                term = term[:51] + "..."
            nes_val = row[nes_col]
            fdr_val = row[fdr_col]
            lines.append(f"{term:<55} {float(nes_val):>7.3f} {float(fdr_val):>10.3e}")

        if len(sig) == 0:
            lines.append("  (no pathways met FDR threshold; showing top 15 by FDR)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run preranked GSEA on 3,704 SHAP-ranked genes."""
    print("Loading signed SHAP values ...")
    signed_shap = _load_signed_shap()
    print(f"  Features loaded: {len(signed_shap)}")
    print(f"  Signed SHAP range: [{signed_shap.min():.4f}, {signed_shap.max():.4f}]")

    print("Mapping ENSG IDs → HGNC symbols via MyGeneInfo ...")
    ensg_to_sym = _map_ensembl_to_symbols(signed_shap.index.tolist())
    n_mapped = sum(
        1 for v in ensg_to_sym.values() if not str(v).startswith("ENSG")
    )
    print(f"  Resolved {n_mapped} / {len(ensg_to_sym)} features to HGNC symbols")

    rnk = _build_prerank_series(signed_shap, ensg_to_sym)
    print(f"  Prerank list size: {len(rnk)} unique symbols")
    print(f"  Top-5 positive (ALS-associated): {rnk.head().to_dict()}")
    print(f"  Top-5 negative (control-associated): {rnk.tail().to_dict()}")

    print("Running preranked GSEA ...")
    results = _run_gsea(rnk)
    results.to_csv(OUT_CSV, index=False)
    print(f"  Full results saved: {OUT_CSV}")

    lines: list[str] = [
        "GSEA Prerank — 3,704 SHAP-Ranked Genes (Step 13)",
        "=" * 70,
        f"Prerank list: {len(rnk)} HGNC symbols (3,704 ENSG → {n_mapped} mapped)",
        f"Ranking metric: signed mean SHAP (positive = ALS, negative = control)",
        f"Gene set libraries: {', '.join(GENE_SETS)}",
        f"Permutations: 1000 | Min gene-set size: 10 | Max: 500",
        f"FDR significance threshold: {FDR_THRESH}",
        "",
        f"PRERANK LIST STATISTICS",
        f"  Total symbols: {len(rnk)}",
        f"  Positive scores (ALS-associated): {(rnk > 0).sum()}",
        f"  Negative scores (control-associated): {(rnk < 0).sum()}",
        f"  Score range: [{rnk.min():.4f}, {rnk.max():.4f}]",
    ]
    _write_statistics(results, rnk, lines)

    lines += [
        "",
        "=" * 70,
        "OUTPUT FILES",
        f"  Full NES/FDR table: {OUT_CSV}",
        f"  Enrichment plots:   {OUT_DIR}/",
    ]

    OUT_TXT.write_text("\n".join(lines) + "\n")
    print(f"  Statistics written: {OUT_TXT}")
    print("Done.")


if __name__ == "__main__":
    main()
