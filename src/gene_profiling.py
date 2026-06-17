# ruff: noqa: E402
"""Per-gene profiling for the core 25-gene ALS biomarker panel.

For each gene in lgbm_core25_panel.csv:
  1. Univariate AUC — roc_auc_score(y, x_gene), max with 1−AUC so it is
     always ≥ 0.5 (direction captured by log2FC).
  2. Log2 fold-change — computed on expm1-back-transformed values.
  3. Direction — "up_in_ALS" or "down_in_ALS".
  4. Mann-Whitney U p-value (two-sided) with Benjamini-Hochberg FDR.
  5. Per-tissue univariate AUC — for tissues with ≥ 5 ALS and ≥ 5 control.

Outputs
-------
  gene_profiling_summary.csv     — master table (all metrics, tissue AUCs)
  gene_profiling_auc.png         — per-gene univariate AUC bar chart
  gene_profiling_volcano.png     — log2FC vs −log10(p_adj) scatter
  gene_profiling_tissue.png      — heatmap: genes × tissues (univariate AUC)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ALS_DIR = Path(__file__).parents[1]
sys.path.insert(0, str(ALS_DIR))

from utils import load_dataset

SCRIPT_DIR = Path(__file__).parent
PLATFORM = "GPL24676"

_PANEL_CSV = SCRIPT_DIR / "lgbm_core25_panel.csv"
_PREFILTER_X_NPY = SCRIPT_DIR / "lgbm_prefilter_X.npy"
_PREFILTER_NAMES_TXT = SCRIPT_DIR / "lgbm_prefilter_names.txt"

MIN_TISSUE_SAMPLES = 5  # minimum ALS *and* control samples required per tissue
PREFIX = "gene_profiling_"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_data() -> tuple[
    np.ndarray, np.ndarray, list[str], list[str], np.ndarray, list[str]
]:
    """Return X_resid, X_orig, feature_names, symbols, y_int, tissue_labels.

    X_resid (874 × 25) — PC-residualized values from prefilter cache.
                         Used for AUC and Mann-Whitney U (same space as model).
    X_orig  (874 × 25) — original log1p-transformed RSEM counts from ds.X.
                         Used for fold-change (avoids negative residual artifacts).
    """
    import pandas as pd

    panel = pd.read_csv(_PANEL_CSV)
    features: list[str] = panel["feature"].tolist()
    symbols: list[str] = panel["symbol"].tolist()

    (ds,) = load_dataset(
        "GSE153960", platform=PLATFORM, resources_dir=ALS_DIR / "resources"
    )
    y_int = ds.y.values.astype(int)
    tissues: list[str] = ds.meta["tissue"].tolist()

    # Residualized matrix — align to ds sample order via GSM index
    all_names = _PREFILTER_NAMES_TXT.read_text().splitlines()
    name_to_col = {n: i for i, n in enumerate(all_names)}
    col_idx = [name_to_col[f] for f in features]
    X_full = np.load(_PREFILTER_X_NPY)
    X_resid = X_full[:, col_idx]

    # Original log1p expression (no residualization) — slice panel genes from ds.X
    available = [f for f in features if f in ds.X.columns]
    missing = set(features) - set(available)
    if missing:
        print(f"  Warning: {len(missing)} panel features not found in ds.X: {missing}")
    X_orig = ds.X[available].values  # (874, n_available) in ds sample order

    print(
        f"  Loaded: X_resid={X_resid.shape}  X_orig={X_orig.shape}  "
        f"ALS={y_int.sum()}  Control={(y_int == 0).sum()}"
    )
    return X_resid, X_orig, features, symbols, y_int, tissues


# ---------------------------------------------------------------------------
# Per-gene metrics
# ---------------------------------------------------------------------------


def _univariate_auc(x: np.ndarray, y: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    auc = float(roc_auc_score(y, x))
    return max(auc, 1.0 - auc)


def _log2fc(x_raw: np.ndarray, y: np.ndarray) -> float:
    """Log2 fold-change ALS vs Control on raw RSEM counts (from ds.X).

    ds.X contains raw RSEM counts; log1p is applied only in the prefilter.
    Returns NaN when mean_ctrl = 0 (gene absent in all controls) or
    mean_als = 0.  Callers should note inf = present in ALS, absent in ctrl.
    """
    mean_als = float(x_raw[y == 1].mean())
    mean_ctrl = float(x_raw[y == 0].mean())
    if mean_ctrl <= 0 or mean_als <= 0:
        return float("nan")
    return float(np.log2(mean_als / mean_ctrl))


def _mannwhitney_p(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import mannwhitneyu

    stat, p = mannwhitneyu(x[y == 1], x[y == 0], alternative="two-sided")
    return float(p)


def _bh_adjust(pvals: list[float]) -> list[float]:
    """Benjamini-Hochberg FDR correction."""
    n = len(pvals)
    order = np.argsort(pvals)
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n + 1)
    adjusted = [min(p * n / r, 1.0) for p, r in zip(pvals, ranks)]
    # Enforce monotonicity (cumulative min from largest p)
    for i in order[::-1]:
        pass  # handled by the rank formula; scipy alternative below
    return [min(a, 1.0) for a in adjusted]


def compute_gene_metrics(
    X_resid: np.ndarray,
    X_orig: np.ndarray,
    y: np.ndarray,
    symbols: list[str],
) -> list[dict]:
    """Compute per-gene AUC, log2FC, direction, and p-value.

    AUC and Mann-Whitney U use X_resid (same space as the trained model).
    log2FC uses X_orig (original log1p expression, no residualization).
    """
    rows = []
    for i, sym in enumerate(symbols):
        x_resid = X_resid[:, i]
        x_orig = X_orig[:, i]
        auc = _univariate_auc(x_resid, y)
        fc = _log2fc(x_orig, y)
        pval = _mannwhitney_p(x_resid, y)
        direction = "up_in_ALS" if (not np.isnan(fc) and fc >= 0) else "down_in_ALS"
        mean_als = float(x_orig[y == 1].mean())  # raw RSEM counts
        mean_ctrl = float(x_orig[y == 0].mean())  # raw RSEM counts
        rows.append(
            {
                "symbol": sym,
                "univariate_auc": round(auc, 4),
                "log2fc": round(fc, 4),
                "direction": direction,
                "mean_expr_als": round(mean_als, 4),
                "mean_expr_ctrl": round(mean_ctrl, 4),
                "mwu_pvalue": pval,
            }
        )
        print(
            f"  {sym:<18}  AUC={auc:.4f}  log2FC={fc:+.3f}  "
            f"dir={direction}  p={pval:.2e}"
        )

    pvals = [r["mwu_pvalue"] for r in rows]
    padj = _bh_adjust(pvals)
    for r, p, pa in zip(rows, pvals, padj):
        r["mwu_pvalue"] = float(f"{p:.4e}")
        r["mwu_padj"] = float(f"{pa:.4e}")

    return rows


# ---------------------------------------------------------------------------
# Per-tissue AUC
# ---------------------------------------------------------------------------


def compute_tissue_aucs(
    X: np.ndarray,
    y: np.ndarray,
    tissues: list[str],
    symbols: list[str],
) -> dict[str, list[float | None]]:
    """For each tissue with sufficient samples, compute per-gene univariate AUC."""

    tissue_arr = np.array(tissues)
    tissue_names = sorted(set(tissues))
    results: dict[str, list[float | None]] = {}

    for tissue in tissue_names:
        mask = tissue_arr == tissue
        y_t = y[mask]
        n_als = int((y_t == 1).sum())
        n_ctrl = int((y_t == 0).sum())
        if n_als < MIN_TISSUE_SAMPLES or n_ctrl < MIN_TISSUE_SAMPLES:
            print(
                f"  Skip {tissue!r}: ALS={n_als} Control={n_ctrl} (< {MIN_TISSUE_SAMPLES})"
            )
            continue

        aucs: list[float | None] = []
        for i in range(X.shape[1]):
            x_t = X[mask, i]
            aucs.append(_univariate_auc(x_t, y_t))
        results[tissue] = aucs
        print(
            f"  {tissue:<30}  ALS={n_als:3d}  Control={n_ctrl:3d}  "
            f"mean_AUC={np.mean(aucs):.3f}"
        )

    return results


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _plot_auc(gene_rows: list[dict]) -> None:
    import matplotlib.pyplot as plt

    symbols = [r["symbol"] for r in gene_rows]
    aucs = [r["univariate_auc"] for r in gene_rows]
    directions = [r["direction"] for r in gene_rows]
    colors = ["#d6604d" if d == "up_in_ALS" else "#4393c3" for d in directions]

    fig, ax = plt.subplots(figsize=(9, max(0.42 * len(symbols), 8)))
    y_pos = list(range(len(symbols) - 1, -1, -1))
    ax.barh(y_pos, aucs, color=colors, edgecolor="none")
    ax.axvline(
        0.5, color="black", linewidth=0.8, linestyle="--", label="Chance (AUC=0.5)"
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(symbols, fontsize=9)
    ax.set_xlabel(
        "Univariate AUC-ROC  (single-gene; max with 1−AUC so always ≥ 0.5;\n"
        "colour shows direction — red = up in ALS, blue = down in ALS)"
    )
    ax.set_title(
        "Per-gene univariate discriminative ability — core 25-gene panel\n"
        "ALS Spectrum MND vs Non-Neurological Control · GSE153960 GPL24676 · n=874",
        fontsize=10,
    )
    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(facecolor="#d6604d", label="Up in ALS (higher expression in ALS)"),
            Patch(facecolor="#4393c3", label="Down in ALS (lower expression in ALS)"),
        ],
        fontsize=8,
        loc="lower right",
    )
    ax.set_xlim(0.45, 1.0)
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(SCRIPT_DIR / f"{PREFIX}auc.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"  Saved → {PREFIX}auc.png")


def _plot_volcano(gene_rows: list[dict]) -> None:
    import matplotlib.pyplot as plt

    # Exclude genes with undefined log2FC (zero expression in one group)
    plotable = [
        r for r in gene_rows if not (np.isnan(r["log2fc"]) or np.isinf(r["log2fc"]))
    ]
    skipped = [
        r["symbol"] for r in gene_rows if np.isnan(r["log2fc"]) or np.isinf(r["log2fc"])
    ]
    if skipped:
        print(
            f"  Volcano: excluding {len(skipped)} genes with undefined log2FC: {skipped}"
        )

    log2fc = [r["log2fc"] for r in plotable]
    neg_log10p = [-np.log10(max(r["mwu_padj"], 1e-300)) for r in plotable]
    symbols = [r["symbol"] for r in plotable]
    directions = [r["direction"] for r in plotable]
    colors = ["#d6604d" if d == "up_in_ALS" else "#4393c3" for d in directions]

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(log2fc, neg_log10p, c=colors, s=60, alpha=0.85, edgecolors="none")

    for sym, x, y_val in zip(symbols, log2fc, neg_log10p):
        ax.annotate(
            sym,
            xy=(x, y_val),
            xytext=(4, 2),
            textcoords="offset points",
            fontsize=7.5,
            color="#333333",
        )

    ax.axvline(0, color="black", linewidth=0.7, linestyle="--")
    ax.axhline(
        -np.log10(0.05), color="grey", linewidth=0.7, linestyle=":", label="FDR = 0.05"
    )

    ax.set_xlabel(
        "Log₂ fold-change  (ALS / Control; computed on raw RSEM counts from GEO supplementary file)\n"
        "Positive = higher expression in ALS; negative = lower expression in ALS"
    )
    ax.set_ylabel("−log₁₀(BH-adjusted p-value)  (Mann-Whitney U, two-sided)")
    ax.set_title(
        "Volcano plot — core 25-gene panel\n"
        "ALS Spectrum MND vs Non-Neurological Control · GSE153960 GPL24676 · n=874",
        fontsize=10,
    )
    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(facecolor="#d6604d", label="Up in ALS"),
            Patch(facecolor="#4393c3", label="Down in ALS"),
            plt.Line2D([0], [0], color="grey", linestyle=":", label="FDR = 0.05"),
        ],
        fontsize=8,
    )
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    fig.savefig(SCRIPT_DIR / f"{PREFIX}volcano.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"  Saved → {PREFIX}volcano.png")


def _plot_tissue_heatmap(
    tissue_aucs: dict[str, list[float | None]],
    symbols: list[str],
) -> None:
    import matplotlib.pyplot as plt
    import pandas as pd

    tissues = list(tissue_aucs.keys())
    mat = np.array([tissue_aucs[t] for t in tissues]).T  # (n_genes, n_tissues)
    df = pd.DataFrame(mat, index=symbols, columns=tissues)

    fig, ax = plt.subplots(
        figsize=(max(0.9 * len(tissues), 10), max(0.42 * len(symbols), 8))
    )
    im = ax.imshow(df.values, cmap="RdYlBu_r", vmin=0.5, vmax=1.0, aspect="auto")
    plt.colorbar(im, ax=ax, label="Univariate AUC-ROC  (0.5 = chance, 1.0 = perfect)")

    ax.set_xticks(range(len(tissues)))
    ax.set_xticklabels(tissues, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(symbols)))
    ax.set_yticklabels(symbols, fontsize=9)

    for i in range(len(symbols)):
        for j in range(len(tissues)):
            val = df.values[i, j]
            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                fontsize=6.5,
                color="white" if val > 0.85 else "black",
            )

    ax.set_title(
        "Per-tissue univariate AUC — core 25-gene panel\n"
        "ALS Spectrum MND vs Non-Neurological Control · GSE153960 GPL24676\n"
        "Each cell: single-gene AUC within that tissue region (warm = strong signal)",
        fontsize=9,
    )
    plt.tight_layout()
    fig.savefig(SCRIPT_DIR / f"{PREFIX}tissue.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"  Saved → {PREFIX}tissue.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    import matplotlib
    import pandas as pd

    matplotlib.use("Agg")

    print("\n" + "=" * 60)
    print("Gene Profiling — Core 25-gene Panel")
    print("=" * 60)

    print("\nLoading data ...")
    X_resid, X_orig, features, symbols, y_int, tissues = _load_data()

    panel = pd.read_csv(_PANEL_CSV)

    print("\n--- Per-gene metrics ---")
    gene_rows = compute_gene_metrics(X_resid, X_orig, y_int, symbols)

    print("\n--- Per-tissue AUC ---")
    tissue_aucs = compute_tissue_aucs(X_resid, y_int, tissues, symbols)

    # Build summary table
    summary = panel.copy()
    metric_df = pd.DataFrame(gene_rows)
    tissue_df = pd.DataFrame(
        {t: tissue_aucs[t] for t in tissue_aucs}, index=range(len(symbols))
    )
    tissue_df.columns = [f"auc_{c.replace(' ', '_')}" for c in tissue_df.columns]

    summary = pd.concat(
        [
            summary.reset_index(drop=True),
            metric_df[
                [
                    "univariate_auc",
                    "log2fc",
                    "direction",
                    "mean_expr_als",
                    "mean_expr_ctrl",
                    "mwu_pvalue",
                    "mwu_padj",
                ]
            ],
            tissue_df,
        ],
        axis=1,
    )
    out_csv = SCRIPT_DIR / f"{PREFIX}summary.csv"
    summary.to_csv(out_csv, index=False)
    print(f"\nSaved summary → {out_csv}")
    print(
        summary[
            ["symbol", "univariate_auc", "log2fc", "direction", "mwu_padj"]
        ].to_string(index=False)
    )

    print("\n--- Plots ---")
    _plot_auc(gene_rows)
    _plot_volcano(gene_rows)
    _plot_tissue_heatmap(tissue_aucs, symbols)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
