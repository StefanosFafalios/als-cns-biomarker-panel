# ruff: noqa: E402
"""Generate all summary plots from lgbm_iterative_pipeline.py outputs.

Produces
--------
  lgbm_iter_final_curve.png       — incremental CV AUC vs. number of SHAP-ranked features
  lgbm_iter_final_shap_bar.png    — mean |SHAP| bar chart, top-50 features (gene symbols)
  lgbm_iter_final_shap_beeswarm.png — SHAP beeswarm, top-20 features (gene symbols)
  lgbm_iter_final_loo.png         — LOO Δ AUC bar chart, top-50 features (gene symbols)

All inputs are read from the same directory as this script.
"""

from __future__ import annotations

import sys
from pathlib import Path

ALS_DIR = Path(__file__).parents[1]
sys.path.insert(0, str(ALS_DIR))

import matplotlib
import mygene
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).parent

SHAP_RANKING_CSV = SCRIPT_DIR / "lgbm_iter_final_shap_ranking.csv"
SHAP_VALUES_NPY = SCRIPT_DIR / "lgbm_iter_final_shap_values.npy"
CURVE_CSV = SCRIPT_DIR / "lgbm_iter_final_curve.csv"
LOO_CSV = SCRIPT_DIR / "lgbm_iter_final_loo_summary.csv"
PREFILTER_X_NPY = SCRIPT_DIR / "lgbm_prefilter_X.npy"
PREFILTER_NAMES_TXT = SCRIPT_DIR / "lgbm_prefilter_names.txt"

# Final SFM features come from the last (clean) iteration cache
_SFM_CACHE = SCRIPT_DIR / "lgbm_iter_2_sfm_features.txt"

TOP_SHAP = 50
TOP_BEESWARM = 20
TOP_LOO = 50


# ---------------------------------------------------------------------------
# Gene symbol resolution
# ---------------------------------------------------------------------------


def _resolve_symbols(ensg_ids: list[str]) -> dict[str, str]:
    """Map versioned ENSG IDs → gene symbols via MyGeneInfo."""
    bases = [e.split(".")[0] for e in ensg_ids]
    mg = mygene.MyGeneInfo()
    hits = mg.querymany(
        bases, scopes="ensembl.gene", fields="symbol", species="human", verbose=False
    )
    base_map = {h["query"]: h.get("symbol", h["query"]) for h in hits}
    return {
        versioned: base_map.get(base, base) for versioned, base in zip(ensg_ids, bases)
    }


def _label(sym_map: dict[str, str], feature: str, max_len: int = 18) -> str:
    sym = sym_map.get(feature, feature.split(".")[0])
    return sym if len(sym) <= max_len else sym[:max_len]


# ---------------------------------------------------------------------------
# Plot 1: Incremental CV curve
# ---------------------------------------------------------------------------


def plot_curve() -> None:
    df = pd.read_csv(CURVE_CSV)
    k = df["n_features"].values
    auc = df["mean_auc"].values

    best_k = int(k[auc.argmax()])
    best_auc = float(auc.max())

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(k, auc, color="steelblue", linewidth=1.5, label="5-fold CV AUC")
    ax.axvline(
        best_k,
        color="crimson",
        linestyle="--",
        linewidth=1.2,
        label=f"Peak: k={best_k}, AUC={best_auc:.4f}",
    )
    ax.set_xlabel(
        "Genes included  (added in SHAP rank order, most discriminative first)"
    )
    ax.set_ylabel("Mean AUC-ROC  (5-fold stratified CV, random_state=42)")
    ax.set_title(
        "Incremental CV curve — LightGBM (top-500 SHAP features)\n"
        "ALS Spectrum MND vs Non-Neurological Control · GSE153960 GPL24676 · n=874",
        fontsize=10,
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = SCRIPT_DIR / "lgbm_iter_final_curve.png"
    fig.savefig(out, dpi=150)
    plt.close("all")
    print(f"  Saved → {out}")


# ---------------------------------------------------------------------------
# Plot 2: SHAP bar chart (top-50)
# ---------------------------------------------------------------------------


def plot_shap_bar(sym_map: dict[str, str]) -> None:
    df = pd.read_csv(SHAP_RANKING_CSV).head(TOP_SHAP)
    features = df["feature"].tolist()
    shap_vals = df["mean_abs_shap"].values
    labels = [_label(sym_map, f) for f in features]

    fig, ax = plt.subplots(figsize=(7, 12))
    colors = ["#2166ac" if v >= shap_vals[0] * 0.5 else "#92c5de" for v in shap_vals]
    y = range(len(labels) - 1, -1, -1)
    ax.barh(list(y), shap_vals[::-1], color=colors[::-1], edgecolor="none")
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels[::-1], fontsize=8)
    ax.set_xlabel(
        "Mean |SHAP| value  (average per-sample contribution magnitude;\n"
        "bar length = importance, not direction — see beeswarm for direction)"
    )
    ax.set_title(
        f"Top-{TOP_SHAP} features ranked by discriminative contribution (mean |SHAP|)\n"
        "ALS Spectrum MND vs Non-Neurological Control · GSE153960 GPL24676 · n=874",
        fontsize=10,
    )
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    out = SCRIPT_DIR / "lgbm_iter_final_shap_bar.png"
    fig.savefig(out, dpi=150)
    plt.close("all")
    print(f"  Saved → {out}")


# ---------------------------------------------------------------------------
# Plot 3: SHAP beeswarm (top-20)
# ---------------------------------------------------------------------------


def plot_shap_beeswarm(sym_map: dict[str, str]) -> None:
    import shap

    shap_ranking = pd.read_csv(SHAP_RANKING_CSV)
    top_features = shap_ranking["feature"].head(TOP_BEESWARM).tolist()

    # Reconstruct X_final from pre-filter cache + SFM feature list
    sfm_names = [s.strip() for s in _SFM_CACHE.read_text().splitlines() if s.strip()]
    all_names = PREFILTER_NAMES_TXT.read_text().splitlines()
    name_to_prefilter_col = {n: i for i, n in enumerate(all_names)}
    sfm_col_in_prefilter = [name_to_prefilter_col[n] for n in sfm_names]

    X_prefilter = np.load(PREFILTER_X_NPY)
    X_final = X_prefilter[:, sfm_col_in_prefilter]

    shap_vals = np.load(SHAP_VALUES_NPY)  # (n_samples, n_sfm_features)

    # Map top_features → column indices in sfm_names order
    sfm_col_map = {n: i for i, n in enumerate(sfm_names)}
    top_cols = [sfm_col_map[f] for f in top_features]

    shap_top = shap_vals[:, top_cols]
    X_top = X_final[:, top_cols]
    labels = [_label(sym_map, f) for f in top_features]

    fig, ax = plt.subplots(figsize=(8, 7))
    shap.summary_plot(
        shap_top,
        X_top,
        feature_names=labels,
        max_display=TOP_BEESWARM,
        show=False,
        plot_type="dot",
        color_bar_label="Gene expression\n(red = high, blue = low)",
    )
    ax = plt.gca()
    ax.set_xlabel(
        "SHAP value  (positive → ALS Spectrum MND; negative → Control)\n"
        "Dot colour = gene expression level in that sample"
    )
    ax.set_title(
        f"SHAP beeswarm — top-{TOP_BEESWARM} features ranked by mean |SHAP|\n"
        "ALS Spectrum MND vs Non-Neurological Control · GSE153960 GPL24676 · n=874\n"
        "Red dots right of zero: high expression associates with ALS; "
        "red dots left: high expression associates with Control",
        fontsize=9,
    )
    plt.tight_layout()
    out = SCRIPT_DIR / "lgbm_iter_final_shap_beeswarm.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"  Saved → {out}")


# ---------------------------------------------------------------------------
# Plot 4: LOO Δ AUC bar chart
# ---------------------------------------------------------------------------


def plot_loo(sym_map: dict[str, str]) -> None:
    df = pd.read_csv(LOO_CSV).sort_values("rank")
    features = df["feature"].tolist()
    deltas = df["delta_auc"].values
    labels = [_label(sym_map, f) for f in features]

    colors = ["#d6604d" if d < 0 else "#4393c3" for d in deltas]
    order = np.argsort(deltas)[::-1]

    fig, ax = plt.subplots(figsize=(7, 12))
    y = range(len(labels))
    ax.barh(
        [len(labels) - 1 - i for i in range(len(labels))],
        deltas[order],
        color=[colors[i] for i in order],
        edgecolor="none",
    )
    ax.set_yticks(list(y))
    ax.set_yticklabels([labels[i] for i in order[::-1]], fontsize=8)
    from matplotlib.patches import Patch

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(
        "Δ AUC = Baseline AUC − Leave-one-out AUC\n"
        "(positive = gene is irreplaceable; negative = gene is redundant / captured by others)"
    )
    ax.set_title(
        f"LOO feature irreplaceability — top-{TOP_LOO} SHAP features\n"
        f"ALS Spectrum MND vs Non-Neurological Control · GSE153960 GPL24676 · "
        f"baseline AUC={df['baseline_auc'].iloc[0]:.4f}  (5-fold stratified CV, random_state=42)",
        fontsize=9,
    )
    ax.legend(
        handles=[
            Patch(facecolor="#4393c3", label="Irreplaceable: removal degrades AUC"),
            Patch(
                facecolor="#d6604d", label="Redundant: signal captured by other genes"
            ),
        ],
        fontsize=8,
        loc="lower right",
    )
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    out = SCRIPT_DIR / "lgbm_iter_final_loo.png"
    fig.savefig(out, dpi=150)
    plt.close("all")
    print(f"  Saved → {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("Resolving gene symbols ...")
    shap_df = pd.read_csv(SHAP_RANKING_CSV)
    loo_df = pd.read_csv(LOO_CSV)
    all_features = list(
        dict.fromkeys(
            shap_df["feature"].head(TOP_SHAP).tolist() + loo_df["feature"].tolist()
        )
    )
    sym_map = _resolve_symbols(all_features)
    print(f"  Resolved {len(sym_map)} symbols")

    print("\nPlot 1: Incremental CV curve")
    plot_curve()

    print("\nPlot 2: SHAP bar chart (top-50)")
    plot_shap_bar(sym_map)

    print("\nPlot 3: SHAP beeswarm (top-20)")
    plot_shap_beeswarm(sym_map)

    print("\nPlot 4: LOO Δ AUC bar chart")
    plot_loo(sym_map)

    print("\nDone.")


if __name__ == "__main__":
    main()
