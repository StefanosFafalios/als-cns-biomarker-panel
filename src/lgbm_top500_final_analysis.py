# ruff: noqa: E402
"""Final analysis on the top-500 SHAP-ranked features with the top-500 optimised model.

Pipeline
--------
1. Load top-500 features from lgbm_iter_final_shap_ranking.csv
2. Slice pre-filter cache to those 500 features
3. Train LightGBM (lgbm_top500_best_params.json) on all 874 samples
4. Compute SHAP values (TreeExplainer)
5. 5-fold CV incremental curve (k = 1 … 500, SHAP-ranked order)
6. LOO on top-50 SHAP features

Outputs (prefix: lgbm_top500_final_)
--------------------------------------
  lgbm_top500_final_shap_ranking.csv
  lgbm_top500_final_shap_values.npy
  lgbm_top500_final_curve.csv
  lgbm_top500_final_loo_summary.csv
  lgbm_top500_final_shap_bar.png
  lgbm_top500_final_shap_beeswarm.png
  lgbm_top500_final_curve.png
  lgbm_top500_final_loo.png
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
TOP_N = 500
TOP_LOO = 50
TOP_BEESWARM = 20
PREFIX = "lgbm_top500_final_"

_PARAMS_PATH = SCRIPT_DIR / "lgbm_top500_best_params.json"
_SHAP_RANKING_SRC = SCRIPT_DIR / "lgbm_iter_final_shap_ranking.csv"
_PREFILTER_X_NPY = SCRIPT_DIR / "lgbm_prefilter_X.npy"
_PREFILTER_NAMES_TXT = SCRIPT_DIR / "lgbm_prefilter_names.txt"

SHAP_CSV = SCRIPT_DIR / f"{PREFIX}shap_ranking.csv"
SHAP_NPY = SCRIPT_DIR / f"{PREFIX}shap_values.npy"
CURVE_CSV = SCRIPT_DIR / f"{PREFIX}curve.csv"
CURVE_CKPT = SCRIPT_DIR / f"{PREFIX}curve_ckpt.csv"
LOO_CSV = SCRIPT_DIR / f"{PREFIX}loo_summary.csv"
LOO_CKPT = SCRIPT_DIR / f"{PREFIX}loo_ckpt.csv"

LGBM_PARAMS: dict = json.loads(_PARAMS_PATH.read_text())
print(f"Loaded params from {_PARAMS_PATH}")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_top500() -> tuple[np.ndarray, list[str], np.ndarray]:
    """Return X (874 × 500), feature_names, y_int."""
    import pandas as pd

    shap_df = pd.read_csv(_SHAP_RANKING_SRC)
    top_features: list[str] = shap_df["feature"].head(TOP_N).tolist()

    all_names = _PREFILTER_NAMES_TXT.read_text().splitlines()
    name_to_col = {n: i for i, n in enumerate(all_names)}
    missing = set(top_features) - set(all_names)
    if missing:
        raise ValueError(f"{len(missing)} SHAP features missing from pre-filter cache")

    col_idx = [name_to_col[f] for f in top_features]
    X_full = np.load(_PREFILTER_X_NPY)
    X = X_full[:, col_idx]

    print("Loaded GSE153960 labels ...")
    (ds,) = load_dataset(
        "GSE153960", platform=PLATFORM, resources_dir=ALS_DIR / "resources"
    )
    y_int = ds.y.values.astype(int)

    print(
        f"  Feature matrix: {X.shape}  |  ALS={y_int.sum()}  Control={(y_int == 0).sum()}"
    )
    return X, top_features, y_int


# ---------------------------------------------------------------------------
# Model / SHAP
# ---------------------------------------------------------------------------


def _train(X: np.ndarray, y: np.ndarray) -> object:
    from lightgbm import LGBMClassifier

    clf = LGBMClassifier(**LGBM_PARAMS)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        clf.fit(X, y)
    return clf


def _compute_shap(
    model: object, X: np.ndarray, feature_names: list[str]
) -> tuple[np.ndarray, list[str], np.ndarray]:
    import shap

    print("  Computing SHAP values (TreeExplainer) ...")
    explainer = shap.TreeExplainer(model)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        shap_vals = explainer.shap_values(X)

    if isinstance(shap_vals, list):
        sv = np.array(shap_vals[1])
    elif shap_vals.ndim == 3:
        sv = shap_vals[:, :, 1]
    else:
        sv = shap_vals

    mean_abs = np.abs(sv).mean(axis=0)
    rank_idx = np.argsort(mean_abs)[::-1]
    ranked_names = [feature_names[i] for i in rank_idx]
    return sv, ranked_names, mean_abs[rank_idx]


# ---------------------------------------------------------------------------
# Incremental CV curve
# ---------------------------------------------------------------------------


def _run_curve(
    X: np.ndarray, y: np.ndarray, ranked_names: list[str], feature_names: list[str]
) -> list[float]:
    import pandas as pd
    from lightgbm import LGBMClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    name_to_col = {n: i for i, n in enumerate(feature_names)}
    rank_cols = [name_to_col[n] for n in ranked_names]
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=CV_RANDOM_STATE)

    scores: list[float] = []
    start_k = 1
    if CURVE_CKPT.exists():
        scores = pd.read_csv(CURVE_CKPT)["mean_auc"].tolist()
        start_k = len(scores) + 1
        print(f"  Resuming curve from k={start_k}")

    for k in range(start_k, len(ranked_names) + 1):
        X_k = X[:, rank_cols[:k]]
        fold_aucs = []
        for tr, te in skf.split(X_k, y):
            clf = LGBMClassifier(**LGBM_PARAMS)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                clf.fit(X_k[tr], y[tr])
            fold_aucs.append(
                float(roc_auc_score(y[te], clf.predict_proba(X_k[te])[:, 1]))
            )
        scores.append(float(np.mean(fold_aucs)))
        print(
            f"  k={k:3d}/{len(ranked_names)}  {ranked_names[k - 1][:28]:<28}  AUC={scores[-1]:.4f}"
        )
        pd.DataFrame(
            {"n_features": range(1, len(scores) + 1), "mean_auc": scores}
        ).to_csv(CURVE_CKPT, index=False)

    return scores


# ---------------------------------------------------------------------------
# LOO
# ---------------------------------------------------------------------------


def _run_loo(
    X: np.ndarray,
    y: np.ndarray,
    top_names: list[str],
    feature_names: list[str],
    baseline_auc: float,
) -> list[dict]:
    import pandas as pd
    from lightgbm import LGBMClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    name_to_col = {n: i for i, n in enumerate(feature_names)}
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=CV_RANDOM_STATE)
    all_mask = np.ones(X.shape[1], dtype=bool)

    done: dict[str, dict] = {}
    if LOO_CKPT.exists():
        for _, row in pd.read_csv(LOO_CKPT).iterrows():
            done[row["feature"]] = row.to_dict()
        print(f"  Resuming LOO ({len(done)} cached)")

    results: list[dict] = []
    for rank_i, feat in enumerate(top_names, 1):
        if feat in done:
            results.append(done[feat])
            print(
                f"  [{rank_i:2d}/{TOP_LOO}] {feat[:36]:<36} Δ={done[feat]['delta_auc']:+.4f} (cached)"
            )
            continue

        mask = all_mask.copy()
        mask[name_to_col[feat]] = False
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
            f"  [{rank_i:2d}/{TOP_LOO}] {feat[:36]:<36} AUC={loo_auc:.4f}  Δ={delta:+.4f}"
        )
        pd.DataFrame(list(done.values())).to_csv(LOO_CKPT, index=False)

    return results


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


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
    s = sym_map.get(feat, feat.split(".")[0])
    return s[:18]


def _plot_curve(scores: list[float]) -> None:
    import matplotlib.pyplot as plt

    k = np.arange(1, len(scores) + 1)
    auc = np.array(scores)
    best_k, best_auc = int(k[auc.argmax()]), float(auc.max())

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
        "Incremental CV curve — LightGBM top-500 model\n"
        "ALS Spectrum MND vs Non-Neurological Control · GSE153960 GPL24676 · n=874",
        fontsize=10,
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(SCRIPT_DIR / f"{PREFIX}curve.png", dpi=150)
    plt.close("all")
    print(f"  Saved → {PREFIX}curve.png")


def _plot_curve_top50(
    scores: list[float], ranked_names: list[str], sym_map: dict[str, str]
) -> None:
    import matplotlib.pyplot as plt

    k50 = np.arange(1, 51)
    auc50 = np.array(scores[:50])
    best_k, best_auc = int(k50[auc50.argmax()]), float(auc50.max())
    labels = [_sym(sym_map, ranked_names[i]) for i in range(50)]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(k50, auc50, color="steelblue", linewidth=1.8, marker="o", markersize=4)
    ax.axvline(
        best_k,
        color="crimson",
        linestyle="--",
        linewidth=1.2,
        label=f"Peak: k={best_k}, AUC={best_auc:.4f}",
    )
    ax.set_xticks(k50)
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylabel("Mean AUC-ROC  (5-fold stratified CV, random_state=42)")
    ax.set_title(
        "Incremental CV curve — top-50 SHAP features (LightGBM top-500 model)\n"
        "ALS Spectrum MND vs Non-Neurological Control · GSE153960 GPL24676 · n=874",
        fontsize=10,
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(SCRIPT_DIR / f"{PREFIX}curve_top50.png", dpi=150)
    plt.close("all")
    print(f"  Saved → {PREFIX}curve_top50.png")


def _plot_shap_bar(shap_df, sym_map: dict[str, str]) -> None:
    import matplotlib.pyplot as plt

    df = shap_df.head(50)
    vals = df["mean_abs_shap"].values
    labels = [_sym(sym_map, f) for f in df["feature"]]
    colors = ["#2166ac" if v >= vals[0] * 0.5 else "#92c5de" for v in vals]

    fig, ax = plt.subplots(figsize=(7, 12))
    y = list(range(len(labels) - 1, -1, -1))
    ax.barh(y, vals[::-1], color=colors[::-1], edgecolor="none")
    ax.set_yticks(y)
    ax.set_yticklabels(labels[::-1], fontsize=8)
    ax.set_xlabel(
        "Mean |SHAP| value  (average per-sample contribution magnitude;\n"
        "bar length = importance, not direction — see beeswarm for direction)"
    )
    ax.set_title(
        "Top-50 features ranked by discriminative contribution (mean |SHAP|)\n"
        "ALS Spectrum MND vs Non-Neurological Control · GSE153960 GPL24676 · n=874",
        fontsize=10,
    )
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(SCRIPT_DIR / f"{PREFIX}shap_bar.png", dpi=150)
    plt.close("all")
    print(f"  Saved → {PREFIX}shap_bar.png")


def _plot_beeswarm(
    shap_vals: np.ndarray,
    X: np.ndarray,
    ranked_names: list[str],
    sym_map: dict[str, str],
) -> None:
    import matplotlib.pyplot as plt
    import shap

    top = ranked_names[:TOP_BEESWARM]
    feat_to_col = {n: i for i, n in enumerate(ranked_names)}
    cols = [feat_to_col[n] for n in top]
    labels = [_sym(sym_map, f) for f in top]

    fig, _ = plt.subplots(figsize=(8, 7))
    shap.summary_plot(
        shap_vals[:, cols],
        X[:, cols],
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

    fig, ax = plt.subplots(figsize=(7, 12))
    n = len(labels)
    ax.barh(
        [n - 1 - i for i in range(n)], deltas[order], color=colors, edgecolor="none"
    )
    ax.set_yticks(list(range(n)))
    ax.set_yticklabels([labels[i] for i in order[::-1]], fontsize=8)
    from matplotlib.patches import Patch

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(
        "Δ AUC = Baseline AUC − Leave-one-out AUC\n"
        "(positive = gene is irreplaceable; negative = gene is redundant / captured by others)"
    )
    ax.set_title(
        f"LOO feature irreplaceability — top-{TOP_LOO} SHAP features\n"
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
    fig.savefig(SCRIPT_DIR / f"{PREFIX}loo.png", dpi=150)
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
    print("LightGBM Top-500 Final Analysis")
    print("=" * 60)

    X, feature_names, y_int = _load_top500()

    # SHAP
    if SHAP_CSV.exists() and SHAP_NPY.exists():
        print("\nLoading cached SHAP ...")
        shap_vals = np.load(SHAP_NPY)
        shap_df = pd.read_csv(SHAP_CSV)
        ranked_names = shap_df["feature"].tolist()
        ranked_shap = shap_df["mean_abs_shap"].values
    else:
        print("\nTraining LightGBM for SHAP ...")
        model = _train(X, y_int)
        shap_vals, ranked_names, ranked_shap = _compute_shap(model, X, feature_names)
        np.save(SHAP_NPY, shap_vals)
        pd.DataFrame(
            {
                "rank": range(1, len(ranked_names) + 1),
                "feature": ranked_names,
                "mean_abs_shap": ranked_shap,
            }
        ).to_csv(SHAP_CSV, index=False)
        print(f"  Saved SHAP ranking → {SHAP_CSV}")

    shap_df = pd.read_csv(SHAP_CSV)

    # CV AUC baseline (full 500 features)
    from lightgbm import LGBMClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=CV_RANDOM_STATE)
    fold_aucs = []
    for tr, te in skf.split(X, y_int):
        clf = LGBMClassifier(**LGBM_PARAMS)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            clf.fit(X[tr], y_int[tr])
        fold_aucs.append(
            float(roc_auc_score(y_int[te], clf.predict_proba(X[te])[:, 1]))
        )
    baseline_auc = float(np.mean(fold_aucs))
    print(
        f"\nBaseline AUC (all 500 features): {baseline_auc:.4f}  folds={[round(a, 4) for a in fold_aucs]}"
    )

    # Incremental curve
    print("\n--- Incremental CV curve ---")
    if CURVE_CSV.exists() and not CURVE_CKPT.exists():
        curve_scores = pd.read_csv(CURVE_CSV)["mean_auc"].tolist()
        print(f"  Loaded completed curve ({len(curve_scores)} points)")
    else:
        curve_scores = _run_curve(X, y_int, ranked_names, feature_names)
        pd.DataFrame(
            {"n_features": range(1, len(curve_scores) + 1), "mean_auc": curve_scores}
        ).to_csv(CURVE_CSV, index=False)
        CURVE_CKPT.unlink(missing_ok=True)
        print(f"  Saved → {CURVE_CSV}")

    # LOO
    print("\n--- LOO ---")
    top50 = ranked_names[:TOP_LOO]
    if LOO_CSV.exists() and not LOO_CKPT.exists():
        loo_results = pd.read_csv(LOO_CSV).to_dict("records")
        print(f"  Loaded completed LOO ({len(loo_results)} features)")
    else:
        loo_results = _run_loo(X, y_int, top50, feature_names, baseline_auc)
        pd.DataFrame(loo_results).to_csv(LOO_CSV, index=False)
        LOO_CKPT.unlink(missing_ok=True)
        print(f"  Saved → {LOO_CSV}")

    # Gene symbols
    print("\nResolving gene symbols ...")
    all_features = list(
        dict.fromkeys(ranked_names[:50] + [r["feature"] for r in loo_results])
    )
    sym_map = _resolve_symbols(all_features)

    # Plots
    print("\n--- Plots ---")
    _plot_curve(curve_scores)
    _plot_curve_top50(curve_scores, ranked_names, sym_map)
    _plot_shap_bar(shap_df, sym_map)
    _plot_beeswarm(shap_vals, X, ranked_names, sym_map)
    _plot_loo(loo_results, sym_map, baseline_auc)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
