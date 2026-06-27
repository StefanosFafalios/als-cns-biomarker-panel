# ruff: noqa: E402
"""Pathway over-representation analysis (ORA) using Enrichr via gseapy.

Gene sets
---------
  up_in_ALS      : panel genes with direction = up_in_ALS AND BH-adj p < 0.05
  down_in_ALS    : panel genes with direction = down_in_ALS AND BH-adj p < 0.05
  full_panel     : all 25 panel genes regardless of direction / significance
  non_predictive : the 9 genes with univariate AUC < 0.60 (interaction pool)

Libraries queried (Enrichr)
---------------------------
  GO_Biological_Process_2023
  KEGG_2021_Human
  Reactome_2022
  WikiPathway_2023_Human
  DisGeNET

Outputs
-------
  pathway_ora_results.csv          — all enriched terms (all gene sets, all libs)
  pathway_ora_up.png               — top terms for up-in-ALS genes
  pathway_ora_down.png             — top terms for down-in-ALS genes
  pathway_ora_full.png             — top terms for full panel
  pathway_ora_statistics.txt       — human-readable summary
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ALS_DIR = Path(__file__).parents[1]
sys.path.insert(0, str(ALS_DIR))

SCRIPT_DIR = Path(__file__).parent

_PROFILING_CSV = SCRIPT_DIR / "gene_profiling_summary.csv"
_PANEL_CSV = SCRIPT_DIR / "lgbm_core25_panel.csv"

PADJ_THRESHOLD = 0.05       # BH-adj p cutoff for directional gene sets
AUC_NON_PRED = 0.60         # univariate AUC threshold for non-predictive pool
ENRICHR_PADJ = 0.05         # Enrichr Adjusted P-value cutoff for display
TOP_TERMS = 15              # top terms per library to display in plots

# 12 protein-coding members of the 15-gene greedy-tail critical panel
# (greedy backward elimination peak at k=15, W.mean = 0.8921; equal cohort weights)
_CRITICAL_CODING_GENES: frozenset[str] = frozenset({
    "FCN3", "PROS1", "ANGPT2", "TINAGL1", "CKMT2", "CLDN5",
    "NR4A1", "SOHLH2", "HEXB", "MCEE", "SLC37A2", "SERTAD1",
})

_LIBRARIES: list[str] = [
    "GO_Biological_Process_2023",
    "KEGG_2021_Human",
    "Reactome_2022",
    "WikiPathways_2023_Human",
    "DisGeNET",
]

_REQUEST_DELAY = 1.2  # seconds between Enrichr calls


# ---------------------------------------------------------------------------
# Gene set builders
# ---------------------------------------------------------------------------


def _build_gene_sets(prof_path: Path) -> dict[str, list[str]]:
    """Return gene-set dicts from profiling CSV."""
    import pandas as pd

    prof = pd.read_csv(prof_path)

    # Restrict to the 11 protein-coding critical-panel genes (D_ZS < 0)
    critical = prof[prof["symbol"].isin(_CRITICAL_CODING_GENES)]

    up = critical.loc[
        (critical["direction"] == "up_in_ALS") & (critical["mwu_padj"] < PADJ_THRESHOLD),
        "symbol",
    ].tolist()

    down = critical.loc[
        (critical["direction"] == "down_in_ALS") & (critical["mwu_padj"] < PADJ_THRESHOLD),
        "symbol",
    ].tolist()

    full = critical["symbol"].tolist()

    non_pred = critical.loc[critical["univariate_auc"] < AUC_NON_PRED, "symbol"].tolist()

    # Drop ENSG-only symbols (no gene symbol → can't query Enrichr)
    def clean(lst: list[str]) -> list[str]:
        return [s for s in lst if not s.startswith("ENSG") and not s.startswith("LOC")]

    sets = {
        "up_in_ALS": clean(up),
        "down_in_ALS": clean(down),
        "full_panel": clean(full),
        "non_predictive": clean(non_pred),
    }

    print("Gene sets:")
    for name, genes in sets.items():
        print(f"  {name:<20}: {len(genes)} genes  ({', '.join(genes)})")
    return sets


# ---------------------------------------------------------------------------
# Enrichr query
# ---------------------------------------------------------------------------


def _enrichr_query(gene_list: list[str], library: str) -> "pd.DataFrame":
    """Run gseapy.enrichr for one gene list and one library."""
    import gseapy

    try:
        enr = gseapy.enrichr(
            gene_list=gene_list,
            gene_sets=library,
            outdir=None,
            no_plot=True,
            verbose=False,
        )
        df = enr.results
        if df is None or df.empty:
            return _empty_df()
        return df
    except Exception as exc:
        print(f"    [warn] Enrichr failed for {library}: {exc}")
        return _empty_df()


def _empty_df() -> "pd.DataFrame":
    import pandas as pd
    return pd.DataFrame(columns=["Term", "Overlap", "P-value", "Adjusted P-value",
                                  "Odds Ratio", "Combined Score", "Genes"])


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def _plot_ora(results: "pd.DataFrame", set_name: str, padj_col: str = "Adjusted P-value") -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    # Filter to significant and take top terms by combined score, one per library
    sig = results[results[padj_col] < ENRICHR_PADJ].copy()
    if sig.empty:
        print(f"  No significant terms for {set_name} at Adj.P < {ENRICHR_PADJ}")
        return

    sig = sig.sort_values("Combined Score", ascending=False)
    # Keep top TOP_TERMS across all libraries
    top = sig.head(TOP_TERMS).copy()
    n = len(top)

    fig, ax = plt.subplots(figsize=(11, max(0.5 * n + 2, 5)))
    y_pos = list(range(n - 1, -1, -1))

    colours = {
        "GO_Biological_Process_2023": "#1f77b4",
        "KEGG_2021_Human": "#ff7f0e",
        "Reactome_2022": "#2ca02c",
        "WikiPathways_2023_Human": "#9467bd",
        "DisGeNET": "#d62728",
    }

    bar_colours = [colours.get(str(row["Gene_set"]), "#8c564b")
                   for _, row in top[::-1].iterrows()]
    scores = np.log10(top["Combined Score"].values[::-1] + 1)

    ax.barh(y_pos, scores, color=bar_colours, edgecolor="none")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(
        [f"{row['Term'][:60]}  [{row['Gene_set'].split('_')[0]}]"
         for _, row in top[::-1].iterrows()],
        fontsize=7.5,
    )
    ax.set_xlabel("log10(Combined Score + 1)  [Enrichr]")
    ax.set_title(
        f"Pathway ORA — {set_name} gene set\n"
        f"GSE153960 GPL24676 core 25-gene panel · Adj.P < {ENRICHR_PADJ}",
        fontsize=9,
    )

    from matplotlib.patches import Patch
    # Legend lists only the libraries actually present among the plotted bars
    present_libs = list(dict.fromkeys(str(g) for g in top["Gene_set"]))
    legend_handles = [
        Patch(color=colours.get(lib, "#8c564b"), label=lib.split("_")[0])
        for lib in present_libs
    ]
    ax.legend(handles=legend_handles, fontsize=7, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    slug = set_name.lower().replace(" ", "_")
    out = SCRIPT_DIR / f"pathway_ora_{slug}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"  Saved → {out.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    import matplotlib
    import pandas as pd

    matplotlib.use("Agg")

    print("\n" + "=" * 65)
    print("Pathway ORA (Enrichr)")
    print("=" * 65)

    gene_sets = _build_gene_sets(_PROFILING_CSV)

    all_results: list[pd.DataFrame] = []

    for set_name, genes in gene_sets.items():
        if not genes:
            print(f"\n[{set_name}] empty gene list — skipping")
            continue
        print(f"\n[{set_name}] querying {len(_LIBRARIES)} libraries ...")
        for library in _LIBRARIES:
            print(f"  {library} ...", end=" ", flush=True)
            df = _enrichr_query(genes, library)
            if not df.empty:
                df["gene_set_name"] = set_name
                df["Gene_set"] = library
                all_results.append(df)
                sig_count = (df["Adjusted P-value"] < ENRICHR_PADJ).sum()
                print(f"{sig_count} significant terms")
            else:
                print("no results")
            time.sleep(_REQUEST_DELAY)

    if not all_results:
        print("\nNo results returned from Enrichr.")
        return

    combined = pd.concat(all_results, ignore_index=True)

    # Save full results
    out_csv = SCRIPT_DIR / "pathway_ora_results.csv"
    combined.to_csv(out_csv, index=False)
    print(f"\nSaved all results → {out_csv.name}")

    # Plots per gene set
    print("\n--- Plots ---")
    for set_name in gene_sets:
        subset = combined[combined["gene_set_name"] == set_name]
        if not subset.empty:
            _plot_ora(subset, set_name)

    # Human-readable statistics file
    lines = [
        "Pathway ORA — Enrichr Results",
        "=" * 65,
        "",
        "Libraries: " + ", ".join(_LIBRARIES),
        f"Significance threshold: Adj.P < {ENRICHR_PADJ}",
        "",
    ]
    for set_name in gene_sets:
        subset = combined[combined["gene_set_name"] == set_name]
        sig = subset[subset["Adjusted P-value"] < ENRICHR_PADJ].sort_values(
            "Combined Score", ascending=False
        )
        lines.append(f"{'─' * 55}")
        lines.append(f"Gene set: {set_name}  ({len(gene_sets[set_name])} genes)")
        lines.append(f"Significant terms: {len(sig)}")
        if not sig.empty:
            lines.append("Top 10 by Combined Score:")
            for _, row in sig.head(10).iterrows():
                lines.append(
                    f"  [{row['Gene_set'].split('_')[0]:<12}] "
                    f"{row['Term'][:55]:<55}  "
                    f"AdjP={row['Adjusted P-value']:.2e}  "
                    f"Score={row['Combined Score']:.1f}  "
                    f"Genes={row['Genes']}"
                )
        lines.append("")

    stat_out = SCRIPT_DIR / "pathway_ora_statistics.txt"
    stat_out.write_text("\n".join(lines))
    print(f"Saved → {stat_out.name}")

    print("\n" + "=" * 65)
    print("DONE")
    print("=" * 65)


def _replot_all() -> None:
    """Regenerate the ORA bar figures from cached results (no Enrichr query)."""
    import matplotlib
    import pandas as pd

    matplotlib.use("Agg")
    combined = pd.read_csv(SCRIPT_DIR / "pathway_ora_results.csv")
    for set_name in combined["gene_set_name"].unique():
        subset = combined[combined["gene_set_name"] == set_name]
        if not subset.empty:
            _plot_ora(subset, set_name)


if __name__ == "__main__":
    import sys

    if "--replot" in sys.argv:
        _replot_all()
    else:
        main()
