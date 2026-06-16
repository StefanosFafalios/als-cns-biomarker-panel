# ruff: noqa: E402
"""Zero-shot cross-platform validation: train on GPL24676, predict on GPL16791.

Both platforms are within GSE153960 (NYGC ALS Consortium) but differ in
library preparation method (Illumina NovaSeq vs HiSeq 2500).

Strategy
--------
1. Load raw RSEM counts for GPL24676 (n=874) and GPL16791 (n=636).
2. Apply log1p to both.
3. Fit the CTD compartment PC regression on GPL24676 only.
4. Apply the fitted regression to residualise both platforms.
5. Extract the 25 core panel features from both residualised matrices.
6. Train LightGBM (top-500 best params, colsample_bytree=1.0) on all 874
   GPL24676 samples.
7. Zero-shot predict on all 636 GPL16791 samples.
8. Report AUC, bootstrap 95 % CI, confusion matrix at Youden threshold,
   and per-tissue AUC breakdown.

Outputs
-------
  blood_validation_gpl16791.png
  blood_validation_gpl16791_statistics.txt
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

ALS_DIR = Path(__file__).parents[1]
sys.path.insert(0, str(ALS_DIR))

from utils import load_dataset

SCRIPT_DIR = Path(__file__).parent

_PANEL_CSV = SCRIPT_DIR / "lgbm_core25_panel.csv"
_PARAMS_PATH = SCRIPT_DIR / "lgbm_top500_best_params.json"

TRAIN_PLATFORM = "GPL24676"
TEST_PLATFORM = "GPL16791"
N_BOOTSTRAP = 2_000
RANDOM_STATE = 42

# CTD compartment Ensembl base IDs (same as biomarker_discovery_lgbm_bayesopt.py)
_COMPARTMENTS: dict[str, tuple[str, ...]] = {
    "erythrocyte": (
        "ENSG00000206172", "ENSG00000188536", "ENSG00000244734",
        "ENSG00000158578", "ENSG00000170180", "ENSG00000133742",
        "ENSG00000159111", "ENSG00000105610", "ENSG00000179364",
        "ENSG00000223609",
    ),
    "platelet": (
        "ENSG00000163736", "ENSG00000163737", "ENSG00000185245",
    ),
    "endothelial": (
        "ENSG00000261371", "ENSG00000179776", "ENSG00000110799",
    ),
}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def _load_platform(platform: str) -> tuple[np.ndarray, list[str], np.ndarray, list[str]]:
    """Return X_raw, feature_names, y, tissues for one platform."""
    (ds,) = load_dataset(
        "GSE153960", platform=platform, resources_dir=ALS_DIR / "resources"
    )
    X_raw = ds.X.values.astype(np.float32)
    feature_names: list[str] = list(ds.X.columns)
    y = ds.y.values.astype(int)
    tissues: list[str] = ds.meta.get("tissue", ["unknown"] * len(y)).tolist()
    print(f"  {platform}: n={len(y)}  ALS={y.sum()}  Ctrl={(y==0).sum()}")
    return X_raw, feature_names, y, tissues


def _ctd_residualise(
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Fit CTD PC regression on training data; residualise both train and test."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LinearRegression

    base_ids = [n.split(".")[0] for n in feature_names]
    pc_train: list[np.ndarray] = []
    pc_test: list[np.ndarray] = []

    for comp, bases in _COMPARTMENTS.items():
        base_set = set(bases)
        idx = [i for i, b in enumerate(base_ids) if b in base_set]
        if not idx:
            print(f"    [warn] No features for compartment '{comp}'")
            continue
        pca = PCA(n_components=1, random_state=0)
        pc_train.append(pca.fit_transform(X_train[:, idx]))
        pc_test.append(pca.transform(X_test[:, idx]))
        print(
            f"    CTD {comp}: {len(idx)} markers, "
            f"PC1 var={pca.explained_variance_ratio_[0]:.3f}"
        )

    Z_train = np.hstack(pc_train)
    Z_test = np.hstack(pc_test)
    reg = LinearRegression()
    reg.fit(Z_train, X_train)
    resid_train = (X_train - reg.predict(Z_train)).astype(np.float32)
    resid_test = (X_test - reg.predict(Z_test)).astype(np.float32)
    return resid_train, resid_test


def _extract_panel(
    X: np.ndarray,
    feature_names: list[str],
    panel_features: list[str],
) -> np.ndarray:
    """Extract panel columns; missing features filled with zeros.

    For CTD-residualised inputs the per-feature mean is ~0, so a zero fill is
    approximately mean-imputation. GPL16791 matches all 25 panel features, so
    no imputation occurs on this path.
    """
    name_to_col = {n: i for i, n in enumerate(feature_names)}
    cols = []
    missing = []
    for f in panel_features:
        if f in name_to_col:
            cols.append(X[:, name_to_col[f]])
        else:
            cols.append(np.zeros(X.shape[0], dtype=np.float32))
            missing.append(f)
    if missing:
        print(f"    [warn] {len(missing)} panel features missing: {missing}")
    return np.column_stack(cols)


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def _bootstrap_auc(y: np.ndarray, scores: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    from sklearn.metrics import roc_auc_score

    aucs = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y[idx], scores[idx]))
    return np.array(aucs)


def _youden_threshold(fpr: np.ndarray, tpr: np.ndarray, thresholds: np.ndarray) -> float:
    idx = np.argmax(tpr - fpr)
    return float(thresholds[idx])


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def _plot(
    y_test: np.ndarray,
    scores_test: np.ndarray,
    auc: float,
    ci_lo: float,
    ci_hi: float,
    train_auc: float,
    tissues_test: list[str],
    tissue_aucs: dict[str, float],
) -> None:
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(y_test, scores_test)
    thresh = _youden_threshold(fpr, tpr, thresholds)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ROC curve
    ax = axes[0]
    ax.plot(fpr, tpr, lw=2, color="#1f77b4",
            label=f"GPL16791 zero-shot AUC = {auc:.4f}\n95% CI [{ci_lo:.4f}, {ci_hi:.4f}]")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.axvline(fpr[np.argmax(tpr - fpr)], color="grey", linestyle=":", lw=0.9,
               label=f"Youden threshold = {thresh:.3f}")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(
        f"Zero-shot cross-platform validation\n"
        f"Train: GPL24676 (n=874) → Test: GPL16791 (n={len(y_test)})\n"
        f"Train 5-fold CV AUC = {train_auc:.4f}  ·  Test AUC = {auc:.4f}  (25-gene panel)",
        fontsize=9,
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Per-tissue AUC bar chart
    ax2 = axes[1]
    tissues_sorted = sorted(tissue_aucs.items(), key=lambda x: x[1], reverse=True)
    names = [t for t, _ in tissues_sorted]
    vals = [v for _, v in tissues_sorted]
    colours = ["#2ca02c" if v >= 0.80 else "#1f77b4" if v >= 0.70 else "#ff7f0e"
               for v in vals]
    ax2.barh(list(range(len(names))), vals[::-1], color=colours[::-1], edgecolor="none")
    ax2.set_yticks(list(range(len(names))))
    ax2.set_yticklabels(names[::-1], fontsize=8)
    ax2.axvline(0.5, color="red", linestyle="--", lw=0.8, label="Chance")
    ax2.axvline(auc, color="#1f77b4", linestyle="--", lw=0.9, label=f"Overall AUC={auc:.3f}")
    ax2.set_xlabel("Zero-shot AUC-ROC per tissue (GPL16791)")
    ax2.set_title(
        "Per-tissue zero-shot AUC (GPL16791)\n"
        "Green ≥ 0.80, Blue 0.70–0.80, Orange < 0.70",
        fontsize=9,
    )
    ax2.legend(fontsize=8)
    ax2.set_xlim(0.4, 1.0)
    ax2.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    out = SCRIPT_DIR / "blood_validation_gpl16791.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"  Saved → {out.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    import json

    import matplotlib
    import pandas as pd
    from lightgbm import LGBMClassifier
    from sklearn.metrics import (
        confusion_matrix,
        roc_auc_score,
        roc_curve,
    )

    matplotlib.use("Agg")

    print("\n" + "=" * 65)
    print("Zero-shot Validation: GPL24676 → GPL16791")
    print("=" * 65)

    panel = pd.read_csv(_PANEL_CSV)
    panel_features: list[str] = panel["feature"].tolist()
    panel_symbols: list[str] = panel["symbol"].tolist()

    print("\nLoading platforms ...")
    X_train_raw, feat_train, y_train, _ = _load_platform(TRAIN_PLATFORM)
    X_test_raw, feat_test, y_test, tissues_test = _load_platform(TEST_PLATFORM)

    # Verify feature alignment
    assert feat_train == feat_test, "Feature lists differ between platforms"

    print("\nApplying log1p ...")
    X_train_log = np.log1p(X_train_raw)
    X_test_log = np.log1p(X_test_raw)

    print("\nFitting CTD compartment regression on GPL24676 ...")
    X_train_res, X_test_res = _ctd_residualise(X_train_log, X_test_log, feat_train)

    print("\nExtracting 25-gene panel ...")
    X_tr = _extract_panel(X_train_res, feat_train, panel_features)
    X_te = _extract_panel(X_test_res, feat_test, panel_features)
    print(f"  Train: {X_tr.shape}  Test: {X_te.shape}")

    params = json.loads(_PARAMS_PATH.read_text())
    params["colsample_bytree"] = 1.0  # all 25 features in every tree

    print("\nTraining on GPL24676 (all 874 samples) ...")
    clf = LGBMClassifier(**params)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        clf.fit(X_tr, y_train)

    TRAIN_CV_AUC = 0.9621  # 5-fold stratified CV from lgbm_core25_panel.py (step 2d)
    print(f"  Train 5-fold CV AUC: {TRAIN_CV_AUC:.4f}")

    print("\nZero-shot prediction on GPL16791 ...")
    test_scores = clf.predict_proba(X_te)[:, 1]
    test_auc = float(roc_auc_score(y_test, test_scores))
    print(f"  Test AUC: {test_auc:.4f}")

    # Bootstrap CI
    rng = np.random.default_rng(RANDOM_STATE)
    boot = _bootstrap_auc(y_test, test_scores, N_BOOTSTRAP, rng)
    ci_lo, ci_hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    print(f"  Bootstrap 95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]  (n={N_BOOTSTRAP})")

    # Confusion matrix at Youden threshold
    fpr, tpr, thresholds = roc_curve(y_test, test_scores)
    thresh = _youden_threshold(fpr, tpr, thresholds)
    y_pred = (test_scores >= thresh).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    print(f"\n  Youden threshold: {thresh:.4f}")
    print(f"  Sensitivity: {sensitivity:.4f}  Specificity: {specificity:.4f}")
    print(f"  Confusion matrix (TN={tn}, FP={fp}, FN={fn}, TP={tp})")

    # Per-tissue AUC
    tissue_aucs: dict[str, float] = {}
    for tissue in sorted(set(tissues_test)):
        idx = [i for i, t in enumerate(tissues_test) if t == tissue]
        if len(idx) < 5 or len(np.unique(y_test[idx])) < 2:
            continue
        ta = float(roc_auc_score(y_test[idx], test_scores[idx]))
        tissue_aucs[tissue] = round(ta, 4)
        print(f"    {tissue:<35} n={len(idx):3d}  AUC={ta:.4f}")

    # Plot
    print("\nGenerating plot ...")
    _plot(y_test, test_scores, test_auc, ci_lo, ci_hi, TRAIN_CV_AUC, tissues_test, tissue_aucs)

    # Statistics file
    lines = [
        "Zero-shot Cross-platform Validation: GPL24676 → GPL16791",
        "=" * 65,
        f"Training set  : GSE153960 GPL24676  n={len(y_train)}  ALS={y_train.sum()}  Ctrl={(y_train==0).sum()}",
        f"Test set      : GSE153960 GPL16791  n={len(y_test)}  ALS={y_test.sum()}  Ctrl={(y_test==0).sum()}",
        f"Panel         : {len(panel_features)} genes",
        f"Preprocessing : log1p → CTD compartment regression (erythrocyte/platelet/endothelial PC1)",
        "",
        f"Train 5-fold CV AUC        : {TRAIN_CV_AUC:.4f}",
        f"Test AUC (zero-shot)       : {test_auc:.4f}",
        f"Bootstrap 95% CI           : [{ci_lo:.4f}, {ci_hi:.4f}]  (n={N_BOOTSTRAP})",
        f"Youden threshold           : {thresh:.4f}",
        f"Sensitivity                : {sensitivity:.4f}",
        f"Specificity                : {specificity:.4f}",
        f"TN={tn}  FP={fp}  FN={fn}  TP={tp}",
        "",
        "Per-tissue AUC (GPL16791):",
    ]
    for tissue, ta in sorted(tissue_aucs.items(), key=lambda x: -x[1]):
        n_tissue = sum(1 for t in tissues_test if t == tissue)
        lines.append(f"  {tissue:<40} n={n_tissue:3d}  AUC={ta:.4f}")

    stat_out = SCRIPT_DIR / "blood_validation_gpl16791_statistics.txt"
    stat_out.write_text("\n".join(lines))
    print(f"  Saved → {stat_out.name}")

    print("\n" + "=" * 65)
    print("DONE")
    print("=" * 65)


if __name__ == "__main__":
    main()
