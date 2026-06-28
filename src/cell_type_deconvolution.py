# ruff: noqa: E402
"""Cell-type deconvolution of GSE153960 GPL24676 bulk RNA-seq.

Method: z-score module scoring (equivalent to ssGSEA mean-rank approach).
For each sample, the score for a cell type is the mean z-score of its
marker genes. This gives a relative estimate of cell-type abundance /
transcriptional activity that is robust when a full single-cell reference
matrix is unavailable.

Marker gene sets from:
  Zhang et al. 2016 (PMID 26638076) — purified human brain cell types
  Allen Brain Cell Atlas consensus markers
  Jurga et al. 2021 (PMID 33580054) — human spinal cord cell-type markers

Outputs
-------
  cell_type_deconvolution_scores.csv  — per-sample cell-type scores
  cell_type_deconvolution.png         — violins: ALS vs Control per cell type
  cell_type_deconvolution_corr.png    — heatmap: panel genes × cell types
  cell_type_deconvolution_statistics.txt
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ALS_DIR = Path(__file__).parents[1]
sys.path.insert(0, str(ALS_DIR))

from utils import _load_suppl_expression, _parse_series_matrix

SCRIPT_DIR = Path(__file__).parent
RESOURCES_DIR = ALS_DIR / "resources"

_MATRIX_PATH = (
    RESOURCES_DIR / "GSE153960" / "matrix" / "GSE153960-GPL24676_series_matrix.txt.gz"
)
_PANEL_CSV = SCRIPT_DIR / "lgbm_core25_panel.csv"
_PROFILING_CSV = SCRIPT_DIR / "gene_profiling_summary.csv"

RANDOM_STATE = 42

# 12 protein-coding members of the 15-gene greedy-tail critical panel
# (greedy backward elimination peak at k=15, W.mean=0.9029; equal cohort weights)
# Pseudogenes (HERC2P8, SMG1P5) and snoRNA (SNORD97) excluded from cell-type analyses
_CRITICAL_CODING_GENES: frozenset[str] = frozenset(
    {
        "FCN3",
        "PROS1",
        "ANGPT2",
        "TINAGL1",
        "CKMT2",
        "CLDN5",
        "NR4A1",
        "SOHLH2",
        "HEXB",
        "MCEE",
        "SLC37A2",
        "SERTAD1",
    }
)

# Genes that receive a cell-type composition-driver label in the manuscript
# (tab:bio_evidence): the 12 critical-panel members plus EMP1, which is
# greedy-dropped from the operational panel but retained as a strict-CI-core
# biological / drug-target candidate.
_DECONV_GENES: frozenset[str] = _CRITICAL_CODING_GENES | {"EMP1"}

# ---------------------------------------------------------------------------
# Cell-type marker sets (consensus from Zhang 2016, Allen Brain, Jurga 2021)
# Deliberately broad — missing genes handled gracefully at runtime
# ---------------------------------------------------------------------------

_MARKERS: dict[str, list[str]] = {
    "Neuron": [
        "SNAP25",
        "SYP",
        "SYN1",
        "SYN2",
        "NEFL",
        "NEFM",
        "NEFH",
        "MAP2",
        "RBFOX3",
        "VSNL1",
        "GAD1",
        "GAD2",
        "SLC17A7",
        "CAMK2A",
        "NRGN",
        "STMN2",
    ],
    "Astrocyte": [
        "GFAP",
        "S100B",
        "AQP4",
        "SLC1A2",
        "SLC1A3",
        "ALDH1L1",
        "GJA1",
        "VIM",
        "SOX9",
        "FABP7",
        "CLU",
    ],
    "Oligodendrocyte": [
        "MBP",
        "PLP1",
        "MOG",
        "MAG",
        "MOBP",
        "CNP",
        "OPALIN",
        "CLDN11",
        "ERMN",
        "KLK6",
    ],
    "Microglia": [
        "AIF1",
        "CX3CR1",
        "TMEM119",
        "P2RY12",
        "HEXB",
        "C1QA",
        "C1QB",
        "C1QC",
        "TREM2",
        "CSF1R",
        "OLFML3",
        "SALL1",
        "FCRLS",
    ],
    "Endothelial": [
        "CLDN5",
        "VWF",
        "PECAM1",
        "CDH5",
        "FLT1",
        "ESAM",
        "PODXL",
        "ERG",
        "KDR",
        "CLEC14A",
    ],
    "OPC": [
        "PDGFRA",
        "CSPG4",
        "OLIG1",
        "OLIG2",
        "SOX10",
        "GPR17",
        "PCDH15",
        "LHFPL3",
    ],
    "Pericyte": [
        "PDGFRB",
        "RGS5",
        "ANPEP",
        "KCNJ8",
        "ABCC9",
        "NOTCH3",
        "DES",
    ],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _map_symbols_to_ensembl(symbols: list[str]) -> dict[str, str]:
    """Query MyGeneInfo to map HGNC symbols → base Ensembl IDs."""
    try:
        import mygene

        mg = mygene.MyGeneInfo()
        results = mg.querymany(
            symbols,
            scopes="symbol",
            fields="ensembl.gene",
            species="human",
            returnall=False,
        )
        mapping: dict[str, str] = {}
        for r in results:
            if "ensembl" in r:
                ens = r["ensembl"]
                if isinstance(ens, list):
                    ens = ens[0]
                mapping[r["query"]] = ens.get("gene", "")
        return mapping
    except Exception as exc:
        print(f"  [WARN] MyGeneInfo lookup failed: {exc}")
        return {}


def _build_ensg_to_symbol(feature_names: list[str]) -> dict[str, str]:
    """Map base Ensembl ID → index in feature_names list."""
    return {f.split(".")[0]: i for i, f in enumerate(feature_names)}


def _compute_module_scores(
    X_z: np.ndarray,  # (n_samples, n_genes) z-scored
    ensg_index: dict[str, int],
    sym_to_ensg: dict[str, str],
    markers: dict[str, list[str]],
) -> pd.DataFrame:
    """Return DataFrame (n_samples, n_cell_types) of module scores."""
    scores: dict[str, np.ndarray] = {}
    coverage: dict[str, tuple[int, int]] = {}

    for cell_type, syms in markers.items():
        idxs = []
        for sym in syms:
            ensg = sym_to_ensg.get(sym, "")
            if ensg and ensg in ensg_index:
                idxs.append(ensg_index[ensg])
        coverage[cell_type] = (len(idxs), len(syms))
        if idxs:
            scores[cell_type] = X_z[:, idxs].mean(axis=1)
        else:
            scores[cell_type] = np.full(X_z.shape[0], np.nan)

    for ct, (found, total) in coverage.items():
        print(f"    {ct:<18} {found}/{total} markers matched")

    return pd.DataFrame(scores)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import mannwhitneyu, spearmanr

    print("=" * 60)
    print("Cell-type deconvolution — GSE153960 GPL24676")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load expression
    # ------------------------------------------------------------------
    print("\n[1] Loading expression data ...")
    meta_df, _ = _parse_series_matrix(_MATRIX_PATH)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        expr_df = _load_suppl_expression("GSE153960", meta_df, RESOURCES_DIR)

    common = meta_df.index.intersection(expr_df.columns)
    meta_df = meta_df.loc[common]
    expr_df = expr_df[common]
    print(f"  Samples: {len(common)}, Genes: {expr_df.shape[0]}")

    feature_names = list(expr_df.index)
    ensg_index = _build_ensg_to_symbol(feature_names)

    grp = meta_df["group"].fillna("").astype(str)
    als_mask = grp.str.match(r"^ALS Spectrum MND$", na=False) | grp.str.match(
        r"^ALS Spectrum MND, Other Neurological Dis", na=False
    )
    ctrl_mask = grp.str.match(r"^Non-Neurological Control", na=False)
    train_mask = als_mask | ctrl_mask

    als_idx = [i for i, m in enumerate(train_mask) if m and als_mask.iloc[i]]
    ctrl_idx = [i for i, m in enumerate(train_mask) if m and ctrl_mask.iloc[i]]
    print(f"  ALS: {len(als_idx)}, Control: {len(ctrl_idx)}")

    # log1p transform
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        X_log = np.log1p(expr_df.values.T.astype(np.float32))  # (n, g)

    # z-score across ALL samples (fit on all, apply to all)
    mu = X_log.mean(axis=0, keepdims=True)
    sd = X_log.std(axis=0, keepdims=True) + 1e-8
    X_z = (X_log - mu) / sd

    # ------------------------------------------------------------------
    # 2. Map markers to Ensembl IDs
    # ------------------------------------------------------------------
    print("\n[2] Mapping cell-type markers via MyGeneInfo ...")
    all_symbols = sorted({s for syms in _MARKERS.values() for s in syms})
    sym_to_ensg = _map_symbols_to_ensembl(all_symbols)
    print(f"  Mapped {len(sym_to_ensg)}/{len(all_symbols)} symbols")

    # ------------------------------------------------------------------
    # 3. Compute module scores
    # ------------------------------------------------------------------
    print("\n[3] Computing cell-type module scores ...")
    score_df = _compute_module_scores(X_z, ensg_index, sym_to_ensg, _MARKERS)
    score_df.index = list(common)
    score_df["group"] = [
        "ALS" if als_mask.iloc[i] else ("Control" if ctrl_mask.iloc[i] else "Other")
        for i in range(len(common))
    ]

    score_df.to_csv(SCRIPT_DIR / "cell_type_deconvolution_scores.csv")

    # ------------------------------------------------------------------
    # 4. Statistical comparison: ALS vs Control
    # ------------------------------------------------------------------
    print("\n[4] ALS vs Control — Mann-Whitney U per cell type ...")
    cell_types = [ct for ct in _MARKERS if not score_df[ct].isna().all()]
    stats_lines: list[str] = [
        "Cell-type Deconvolution — GSE153960 GPL24676",
        "=" * 60,
        "Method: z-score module scoring",
        f"ALS n={len(als_idx)}, Control n={len(ctrl_idx)}",
        "",
        f"{'Cell type':<20} {'ALS median':>12} {'Ctrl median':>12} "
        f"{'Δ (ALS-Ctrl)':>14} {'p (MWU)':>12} {'Direction':>12}",
        "-" * 85,
    ]

    als_scores = score_df.iloc[als_idx]
    ctrl_scores = score_df.iloc[ctrl_idx]

    mwu_results: dict[str, dict] = {}
    for ct in cell_types:
        a = als_scores[ct].dropna().values
        c = ctrl_scores[ct].dropna().values
        if len(a) < 5 or len(c) < 5:
            continue
        stat, p = mannwhitneyu(a, c, alternative="two-sided")
        med_als, med_ctrl = np.median(a), np.median(c)
        delta = med_als - med_ctrl
        direction = "UP in ALS" if delta > 0 else "DOWN in ALS"
        mwu_results[ct] = {
            "median_als": med_als,
            "median_ctrl": med_ctrl,
            "delta": delta,
            "p": p,
            "direction": direction,
        }
        line = (
            f"{ct:<20} {med_als:>12.4f} {med_ctrl:>12.4f} "
            f"{delta:>+14.4f} {p:>12.3e} {direction:>12}"
        )
        print(f"  {line}")
        stats_lines.append(line)

    # ------------------------------------------------------------------
    # 5. Panel gene × cell-type correlation
    # ------------------------------------------------------------------
    print("\n[5] Panel gene × cell-type Spearman correlations ...")
    panel_df = pd.read_csv(_PANEL_CSV)
    profiling_df = pd.read_csv(_PROFILING_CSV)

    # Merge direction; correlation analysis covers the 13 labelled panel genes
    # (12 critical-panel members + EMP1)
    panel_info = panel_df.merge(
        profiling_df[["feature", "direction"]],
        on="feature",
        how="left",
    )
    panel_info = panel_info[panel_info["symbol"].isin(_DECONV_GENES)]

    # Co-expression correlations are computed over all profiled GPL24676 samples
    # (n=999; 684 ALS + 190 control + 125 other-neurological), which maximises the
    # stability of the rank-correlation estimate used for cell-type assignment.
    corr_rows: list[dict] = []
    for _, row in panel_info.iterrows():
        feat = str(row["feature"])
        base = feat.split(".")[0]
        sym = str(row.get("symbol", feat))
        if base not in ensg_index:
            continue
        g_idx = ensg_index[base]
        gene_expr = X_log[:, g_idx]
        for ct in cell_types:
            ct_scores = score_df[ct].values
            valid = ~np.isnan(ct_scores)
            rho, p_rho = spearmanr(gene_expr[valid], ct_scores[valid])
            corr_rows.append(
                {
                    "symbol": sym,
                    "cell_type": ct,
                    "rho": rho,
                    "p": p_rho,
                    "direction": row.get("direction", ""),
                }
            )

    corr_df = pd.DataFrame(corr_rows)

    # Pivot for heatmap
    pivot = corr_df.pivot(index="symbol", columns="cell_type", values="rho")
    pivot = pivot.reindex(columns=cell_types)

    stats_lines += [
        "",
        "Panel gene × cell-type Spearman ρ (|ρ| > 0.30 flagged)",
        "-" * 60,
    ]
    high_corr = corr_df[corr_df["rho"].abs() > 0.30].sort_values(
        "rho", key=abs, ascending=False
    )
    for _, r in high_corr.iterrows():
        stats_lines.append(
            f"  {r['symbol']:<20} × {r['cell_type']:<18} "
            f"ρ={r['rho']:+.3f}  p={r['p']:.2e}"
        )

    # ------------------------------------------------------------------
    # 6. Plots
    # ------------------------------------------------------------------
    print("\n[6] Generating plots ...")

    # --- Plot A: violin plots (2 × 4 grid) ---
    n_ct = len(cell_types)
    ncols = 4
    nrows = int(np.ceil(n_ct / ncols))
    fig_v, axes_v = plt.subplots(
        nrows, ncols, figsize=(ncols * 5.5, nrows * 5.5), sharey=False
    )
    axes_flat = axes_v.flatten()
    # Hide any unused slots
    for ax in axes_flat[n_ct:]:
        ax.set_visible(False)

    palette = {"ALS": "#E64B35", "Control": "#4DBBD5"}
    for ax, ct in zip(axes_flat, cell_types):
        for grp_name, color in palette.items():
            vals = score_df[score_df["group"] == grp_name][ct].dropna().values
            parts = ax.violinplot(
                [vals],
                positions=[list(palette).index(grp_name)],
                showmedians=True,
                showextrema=False,
            )
            for pc in parts["bodies"]:
                pc.set_facecolor(color)
                pc.set_alpha(0.7)
            parts["cmedians"].set_color("black")
            parts["cmedians"].set_linewidth(2)

        r = mwu_results.get(ct, {})
        p_val = r.get("p", 1.0)
        p_str = (
            "***"
            if p_val < 0.001
            else "**"
            if p_val < 0.01
            else "*"
            if p_val < 0.05
            else "n.s."
        )
        ax.set_title(f"{ct}\n{p_str}", fontsize=11)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["ALS", "Ctrl"], fontsize=10)
        ax.set_ylabel("Module score (z)", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)

    fig_v.suptitle(
        "Cell-type module scores — ALS vs Non-neurological Control\n"
        "GSE153960 GPL24676 (n=874)",
        fontsize=12,
    )
    fig_v.tight_layout()
    fig_v.savefig(SCRIPT_DIR / "cell_type_deconvolution.png", dpi=150)
    plt.close(fig_v)

    # --- Plot B: correlation heatmap ---
    fig_h, ax_h = plt.subplots(figsize=(max(6, n_ct * 1.1), max(6, len(pivot) * 0.45)))
    import matplotlib.colors as mcolors

    vmax = min(0.8, pivot.abs().max().max())
    cmap = plt.cm.RdBu_r
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax_h.imshow(pivot.values, cmap=cmap, norm=norm, aspect="auto")

    ax_h.set_xticks(range(len(cell_types)))
    ax_h.set_xticklabels(cell_types, rotation=40, ha="right", fontsize=9)
    ax_h.set_yticks(range(len(pivot)))
    ax_h.set_yticklabels(pivot.index, fontsize=8)

    # Annotate cells
    for i in range(len(pivot)):
        for j in range(len(cell_types)):
            val = pivot.values[i, j]
            if not np.isnan(val) and abs(val) > 0.20:
                ax_h.text(
                    j,
                    i,
                    f"{val:+.2f}",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="white" if abs(val) > 0.45 else "black",
                )

    plt.colorbar(im, ax=ax_h, label="Spearman ρ", shrink=0.6)
    ax_h.set_title(
        "Panel gene × cell-type Spearman ρ\nGSE153960 GPL24676 (n=874)",
        fontsize=10,
    )
    fig_h.tight_layout()
    fig_h.savefig(SCRIPT_DIR / "cell_type_deconvolution_corr.png", dpi=150)
    plt.close(fig_h)

    # ------------------------------------------------------------------
    # 7. Write statistics
    # ------------------------------------------------------------------
    stats_path = SCRIPT_DIR / "cell_type_deconvolution_statistics.txt"
    stats_path.write_text("\n".join(stats_lines) + "\n")
    print(f"\n  Statistics → {stats_path.name}")
    print("  Violin plot → cell_type_deconvolution.png")
    print("  Correlation → cell_type_deconvolution_corr.png")
    print("  Scores CSV  → cell_type_deconvolution_scores.csv")
    print("\nDone.")


if __name__ == "__main__":
    main()
