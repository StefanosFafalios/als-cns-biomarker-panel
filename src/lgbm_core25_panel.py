# ruff: noqa: E402
"""Core 25-gene ALS biomarker panel — final focused analysis.

Takes the top-25 SHAP-ranked features from lgbm_top500_final_shap_ranking.csv,
runs 5-fold CV and LOO within the 25-feature context, and generates all
panel-specific plots.

Inputs (must exist)
-------------------
  lgbm_top500_final_shap_ranking.csv   — SHAP-ranked features (top-25 taken)
  lgbm_top500_final_shap_values.npy    — SHAP values (874 × 500)
  lgbm_top500_final_curve.csv          — incremental curve (k=1..500)
  lgbm_top500_best_params.json         — model hyperparameters
  lgbm_prefilter_X.npy / names.txt     — pre-filtered feature matrix

Outputs (prefix: lgbm_core25_)
-------------------------------
  lgbm_core25_panel.csv                — final panel (rank, ENSG, symbol, SHAP, LOO Δ)
  lgbm_core25_loo_summary.csv          — LOO within 25-feature context
  lgbm_core25_curve.png                — incremental curve k=1..25 with gene labels
  lgbm_core25_shap_bar.png             — mean |SHAP| bar chart
  lgbm_core25_shap_beeswarm.png        — SHAP beeswarm
  lgbm_core25_loo.png                  — LOO Δ AUC bar chart
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

ALS_DIR = Path(__file__).parents[1]
sys.path.insert(0, str(ALS_DIR))

from utils import load_dataset

SCRIPT_DIR = Path(__file__).parent
PLATFORM = "GPL24676"
CV_FOLDS = 5
CV_RANDOM_STATE = 42
PANEL_SIZE = 25
PREFIX = "lgbm_core25_"

_PARAMS_PATH = SCRIPT_DIR / "lgbm_top500_best_params.json"
_SHAP_RANKING_SRC = SCRIPT_DIR / "lgbm_top500_final_shap_ranking.csv"
_SHAP_VALUES_NPY = SCRIPT_DIR / "lgbm_top500_final_shap_values.npy"
_CURVE_SRC = SCRIPT_DIR / "lgbm_top500_final_curve.csv"
_PREFILTER_X_NPY = SCRIPT_DIR / "lgbm_prefilter_X.npy"
_PREFILTER_NAMES_TXT = SCRIPT_DIR / "lgbm_prefilter_names.txt"

LOO_CSV = SCRIPT_DIR / f"{PREFIX}loo_summary.csv"
LOO_CKPT = SCRIPT_DIR / f"{PREFIX}loo_ckpt.csv"
PANEL_CSV = SCRIPT_DIR / f"{PREFIX}panel.csv"

LGBM_PARAMS: dict = json.loads(_PARAMS_PATH.read_text())
print(f"Loaded params from {_PARAMS_PATH}")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def _load_panel_matrix() -> tuple[np.ndarray, list[str], np.ndarray]:
    """Return X (874 × 25), feature_names (in SHAP rank order), y_int."""
    import pandas as pd

    shap_df = pd.read_csv(_SHAP_RANKING_SRC)
    top_features: list[str] = shap_df["feature"].head(PANEL_SIZE).tolist()

    all_names = _PREFILTER_NAMES_TXT.read_text().splitlines()
    name_to_col = {n: i for i, n in enumerate(all_names)}
    col_idx = [name_to_col[f] for f in top_features]

    X_full = np.load(_PREFILTER_X_NPY)
    X = X_full[:, col_idx]

    (ds,) = load_dataset(
        "GSE153960", platform=PLATFORM, resources_dir=ALS_DIR / "resources"
    )
    y_int = ds.y.values.astype(int)

    print(
        f"  Panel matrix: {X.shape}  |  ALS={y_int.sum()}  Control={(y_int == 0).sum()}"
    )
    return X, top_features, y_int


def _resolve_symbols(features: list[str]) -> dict[str, str]:
    import mygene

    bases = [f.split(".")[0] for f in features]
    mg = mygene.MyGeneInfo()
    hits = mg.querymany(
        bases, scopes="ensembl.gene", fields="symbol", species="human", verbose=False
    )
    base_map = {h["query"]: h.get("symbol", h["query"]) for h in hits}
    return {f: base_map.get(b, b) for f, b in zip(features, bases)}


def _sym(sym_map: dict[str, str], feat: str) -> str:
    return sym_map.get(feat, feat.split(".")[0])[:20]


# ---------------------------------------------------------------------------
# CV baseline
# ---------------------------------------------------------------------------


def _cv_baseline(X: np.ndarray, y: np.ndarray) -> float:
    from lightgbm import LGBMClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=CV_RANDOM_STATE)
    fold_aucs = []
    for tr, te in skf.split(X, y):
        clf = LGBMClassifier(**LGBM_PARAMS)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            clf.fit(X[tr], y[tr])
        fold_aucs.append(float(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1])))
    auc = float(np.mean(fold_aucs))
    print(
        f"  Baseline AUC ({PANEL_SIZE} features): {auc:.4f}  folds={[round(a, 4) for a in fold_aucs]}"
    )
    return auc


# ---------------------------------------------------------------------------
# LOO within 25-feature context
# ---------------------------------------------------------------------------


def _run_loo(
    X: np.ndarray, y: np.ndarray, feature_names: list[str], baseline_auc: float
) -> list[dict]:
    import pandas as pd
    from lightgbm import LGBMClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=CV_RANDOM_STATE)
    all_mask = np.ones(PANEL_SIZE, dtype=bool)

    done: dict[str, dict] = {}
    if LOO_CKPT.exists():
        for _, row in pd.read_csv(LOO_CKPT).iterrows():
            done[row["feature"]] = row.to_dict()
        print(f"  Resuming LOO ({len(done)} cached)")

    results: list[dict] = []
    for rank_i, feat in enumerate(feature_names, 1):
        if feat in done:
            results.append(done[feat])
            print(
                f"  [{rank_i:2d}/{PANEL_SIZE}] {feat[:36]:<36} Δ={done[feat]['delta_auc']:+.4f} (cached)"
            )
            continue

        mask = all_mask.copy()
        mask[rank_i - 1] = False
        fold_aucs = []
        for tr, te in skf.split(X[:, mask], y):
            clf = LGBMClassifier(**LGBM_PARAMS)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                clf.fit(X[tr][:, mask], y[tr])
            fold_aucs.append(
                float(roc_auc_score(y[te], clf.predict_proba(X[te][:, mask])[:, 1]))
            )

        loo_auc = float(np.mean(fold_aucs))
        delta = round(baseline_auc - loo_auc, 4)
        row = {
            "rank": rank_i,
            "feature": feat,
            "baseline_auc": round(baseline_auc, 4),
            "loo_auc": round(loo_auc, 4),
            "delta_auc": delta,
            **{f"fold_{j + 1}_auc": round(a, 4) for j, a in enumerate(fold_aucs)},
        }
        results.append(row)
        done[feat] = row
        print(
            f"  [{rank_i:2d}/{PANEL_SIZE}] {feat[:36]:<36} AUC={loo_auc:.4f}  Δ={delta:+.4f}"
        )
        pd.DataFrame(list(done.values())).to_csv(LOO_CKPT, index=False)

    return results


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


_PROTEIN_CODING_PANEL = {
    "MECOM", "SERTAD1", "FCN3", "PROS1", "ANGPT2", "EMP1", "TINAGL1",
    "CKMT2", "VWF", "CLDN5", "NR4A1", "SOHLH2", "HEXB", "MCEE", "SLC37A2",
}
_CLR_PC = "#d62728"   # red — protein-coding transferable panel
_CLR_NC = "#aec7e8"   # light blue — non-coding elements


def _plot_curve(feature_names: list[str], sym_map: dict[str, str]) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
    import pandas as pd

    df = pd.read_csv(_CURVE_SRC).head(PANEL_SIZE)
    k = df["n_features"].values
    auc = df["mean_auc"].values
    best_k, best_auc = int(k[auc.argmax()]), float(auc.max())
    labels = [_sym(sym_map, f) for f in feature_names]
    is_pc = [lbl in _PROTEIN_CODING_PANEL for lbl in labels]

    fig, ax = plt.subplots(figsize=(14, 6))

    # Draw line first (neutral colour), then overlay coloured scatter markers
    ax.plot(k, auc, color="#7fb3d3", linewidth=2, zorder=1)
    for ki, auci, pc in zip(k, auc, is_pc):
        ax.scatter(
            ki, auci,
            color=_CLR_PC if pc else _CLR_NC,
            s=55, zorder=2, edgecolors="white", linewidths=0.5,
        )

    ax.axvline(
        best_k,
        color="crimson",
        linestyle="--",
        linewidth=1.2,
        label=f"Peak: k={best_k}, AUC={best_auc:.4f}",
    )

    ax.set_xticks(k)
    tick_labels = ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    for tlbl, pc in zip(tick_labels, is_pc):
        tlbl.set_color(_CLR_PC if pc else "#555555")
        if pc:
            tlbl.set_fontweight("bold")

    ax.set_ylabel("Mean AUC-ROC  (5-fold stratified CV, random_state=42)")
    ax.set_title(
        f"Incremental CV curve — core {PANEL_SIZE}-gene panel (LightGBM top-500 model)\n"
        "ALS Spectrum MND vs Non-Neurological Control · GSE153960 GPL24676 · n=874\n"
        f"Each tick = one gene added in SHAP rank order; panel cut at k={PANEL_SIZE} (91.5% of max AUC gain)",
        fontsize=9,
    )

    legend_pc = mlines.Line2D(
        [], [], color=_CLR_PC, marker="o", linestyle="None", markersize=7,
        label="Protein-coding (transferable panel, n=15)",
    )
    legend_nc = mlines.Line2D(
        [], [], color=_CLR_NC, marker="o", linestyle="None", markersize=7,
        label="Non-coding element (n=10)",
    )
    legend_peak = mlines.Line2D(
        [], [], color="crimson", linestyle="--", linewidth=1.2,
        label=f"Peak: k={best_k}, AUC={best_auc:.4f}",
    )
    ax.legend(handles=[legend_pc, legend_nc, legend_peak], fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.subplots_adjust(top=0.84)
    fig.savefig(SCRIPT_DIR / f"{PREFIX}curve.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"  Saved → {PREFIX}curve.png")


def _plot_shap_bar(feature_names: list[str], sym_map: dict[str, str]) -> None:
    import matplotlib.pyplot as plt
    import pandas as pd

    shap_df = pd.read_csv(_SHAP_RANKING_SRC).head(PANEL_SIZE)
    vals = shap_df["mean_abs_shap"].values
    labels = [_sym(sym_map, f) for f in feature_names]
    colors = ["#2166ac" if v >= vals[0] * 0.5 else "#92c5de" for v in vals]

    row_h = max(0.42 * PANEL_SIZE, 8)
    fig, ax = plt.subplots(figsize=(9, row_h))
    y = list(range(PANEL_SIZE - 1, -1, -1))
    ax.barh(y, vals[::-1], color=colors[::-1], edgecolor="none")
    ax.set_yticks(y)
    ax.set_yticklabels(labels[::-1], fontsize=9)
    ax.set_xlabel(
        "Mean |SHAP| value  (average per-sample contribution magnitude;\n"
        "bar length = importance, not direction — see beeswarm for direction)"
    )
    ax.set_title(
        f"Core {PANEL_SIZE}-gene panel ranked by discriminative contribution (mean |SHAP|)\n"
        "ALS Spectrum MND vs Non-Neurological Control · GSE153960 GPL24676 · n=874",
        fontsize=10,
    )
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(SCRIPT_DIR / f"{PREFIX}shap_bar.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"  Saved → {PREFIX}shap_bar.png")


def _plot_beeswarm(
    X: np.ndarray, feature_names: list[str], sym_map: dict[str, str]
) -> None:
    import matplotlib.pyplot as plt

    # Load SHAP values for the top-500 space and slice to our 25 columns
    # The SHAP values are in the order of the top-500 SHAP ranking
    import pandas as pd
    import shap

    top500_ranking = pd.read_csv(_SHAP_RANKING_SRC)
    top500_features = top500_ranking["feature"].tolist()
    shap_vals_500 = np.load(_SHAP_VALUES_NPY)  # (874, 500) in top-500 input order

    # Map from input feature order → SHAP value columns
    # SHAP values saved in _compute_shap are in the ORDER of feature_names (input order to model)
    # but the ranking CSV is sorted by mean |SHAP|. We need the column index in the SHAP matrix.
    # The SHAP matrix columns correspond to the INPUT order (feature_names passed to model),
    # which is the top-500 SHAP-ranked order from lgbm_iter_final_shap_ranking.csv.
    top500_col_map = {f: i for i, f in enumerate(top500_features)}
    cols = [top500_col_map[f] for f in feature_names]

    shap_top = shap_vals_500[:, cols]
    labels = [_sym(sym_map, f) for f in feature_names]

    row_h = max(0.42 * PANEL_SIZE + 3, 10)
    fig, _ = plt.subplots(figsize=(10, row_h))
    shap.summary_plot(
        shap_top,
        X,
        feature_names=labels,
        max_display=PANEL_SIZE,
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
        f"SHAP beeswarm — core {PANEL_SIZE}-gene panel ranked by mean |SHAP|\n"
        "ALS Spectrum MND vs Non-Neurological Control · GSE153960 GPL24676 · n=874\n"
        "Red dots right of zero: high expression associates with ALS; "
        "red dots left: high expression associates with Control",
        fontsize=9,
    )
    plt.tight_layout()
    plt.savefig(SCRIPT_DIR / f"{PREFIX}shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"  Saved → {PREFIX}shap_beeswarm.png")


def _plot_loo(
    loo_results: list[dict], sym_map: dict[str, str], baseline_auc: float
) -> None:
    import matplotlib.pyplot as plt

    deltas = np.array([r["delta_auc"] for r in loo_results])
    labels = [_sym(sym_map, r["feature"]) for r in loo_results]
    order = np.argsort(deltas)[::-1]
    colors = ["#d6604d" if deltas[i] < 0 else "#4393c3" for i in order]

    n = len(labels)
    row_h = max(0.42 * n, 8)
    fig, ax = plt.subplots(figsize=(9, row_h))
    ax.barh(
        [n - 1 - i for i in range(n)], deltas[order], color=colors, edgecolor="none"
    )
    ax.set_yticks(list(range(n)))
    ax.set_yticklabels([labels[i] for i in order[::-1]], fontsize=9)
    from matplotlib.patches import Patch

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(
        "Δ AUC = Baseline AUC − Leave-one-out AUC\n"
        "(positive = gene is irreplaceable; negative = gene is redundant / captured by others)"
    )
    ax.set_title(
        f"LOO feature irreplaceability — core {PANEL_SIZE}-gene panel\n"
        "ALS Spectrum MND vs Non-Neurological Control · GSE153960 GPL24676 · "
        f"baseline AUC={baseline_auc:.4f}  (5-fold stratified CV, random_state=42)",
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
    fig.savefig(SCRIPT_DIR / f"{PREFIX}loo.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"  Saved → {PREFIX}loo.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    import matplotlib
    import pandas as pd

    matplotlib.use("Agg")

    print("\n" + "=" * 60)
    print(f"Core {PANEL_SIZE}-gene Panel Analysis")
    print("=" * 60)

    X, feature_names, y_int = _load_panel_matrix()

    print("\nResolving gene symbols ...")
    sym_map = _resolve_symbols(feature_names)
    for i, (f, s) in enumerate([(f, sym_map[f]) for f in feature_names], 1):
        print(f"  {i:2d}. {f}  →  {s}")

    print("\n--- 5-fold CV baseline ---")
    baseline_auc = _cv_baseline(X, y_int)

    print(f"\n--- LOO within {PANEL_SIZE}-feature context ---")
    if LOO_CSV.exists() and not LOO_CKPT.exists():
        loo_results = pd.read_csv(LOO_CSV).to_dict("records")
        print(f"  Loaded completed LOO ({len(loo_results)} features)")
    else:
        loo_results = _run_loo(X, y_int, feature_names, baseline_auc)
        pd.DataFrame(loo_results).to_csv(LOO_CSV, index=False)
        LOO_CKPT.unlink(missing_ok=True)
        print(f"  Saved → {LOO_CSV}")

    # Save final panel CSV
    shap_df = pd.read_csv(_SHAP_RANKING_SRC).head(PANEL_SIZE)
    loo_map = {r["feature"]: r["delta_auc"] for r in loo_results}
    panel_df = pd.DataFrame(
        {
            "rank": range(1, PANEL_SIZE + 1),
            "feature": feature_names,
            "symbol": [sym_map[f] for f in feature_names],
            "mean_abs_shap": shap_df["mean_abs_shap"].values,
            "loo_delta_auc": [loo_map.get(f, float("nan")) for f in feature_names],
        }
    )
    panel_df.to_csv(PANEL_CSV, index=False)
    print(f"\nPanel saved → {PANEL_CSV}")
    print(panel_df.to_string(index=False))

    print("\n--- Plots ---")
    _plot_curve(feature_names, sym_map)
    _plot_shap_bar(feature_names, sym_map)
    _plot_beeswarm(X, feature_names, sym_map)
    _plot_loo(loo_results, sym_map, baseline_auc)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
