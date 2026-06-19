"""Calibration analysis for the GPL24676-trained panels on validation cohorts.

Computes discrimination (AUC) and calibration (Brier score, log-loss,
expected calibration error) for the zero-shot predictions of BOTH the
25-gene panel and the 15-gene critical sub-panel on:
  - GPL16791 (n=636, primary cross-platform validation)
Both panels are evaluated identically: the same LightGBM model (top-500
hyperparameters) on CTD-residualised data, with 5-fold CV self-calibration
on the GPL24676 training cohort. The two reliability curves are overlaid in
each cohort panel for direct comparison.

Outputs:
  - calibration_summary.csv        (per-panel, per-cohort discrimination + calibration metrics)
  - calibration_reliability.png    (reliability diagram per cohort; 25-gene + 15-gene critical overlaid)
  - calibration_analysis_statistics.txt
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

ALS_DIR = Path(__file__).parents[1]
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(ALS_DIR))

from utils import load_dataset  # noqa: E402

_PANEL_CSV = SCRIPT_DIR / "lgbm_core25_panel.csv"
_PARAMS_PATH = SCRIPT_DIR / "lgbm_top500_best_params.json"
_PREFILTER_NAMES = SCRIPT_DIR / "lgbm_prefilter_names.txt"

# 15-gene critical panel: indices into the 25-gene ordered panel
# (greedy backward-elimination peak, W.mean AUC = 0.8921; see panel_17gene_eval.py).
_CRITICAL_IDX = [1, 2, 3, 5, 6, 7, 10, 11, 12, 15, 16, 18, 20, 23, 24]


def _expected_calibration_error(
    y_true: np.ndarray, scores: np.ndarray, n_bins: int = 10
) -> float:
    """Equal-width bin ECE = sum_b (n_b / N) * |acc_b - conf_b|."""
    n = len(y_true)
    if n == 0:
        return float("nan")
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (scores >= lo) & (scores < hi if i < n_bins - 1 else scores <= hi)
        if mask.sum() == 0:
            continue
        acc = y_true[mask].mean()
        conf = scores[mask].mean()
        ece += mask.sum() / n * abs(acc - conf)
    return float(ece)


def _reliability(y_true: np.ndarray, scores: np.ndarray, n_bins: int = 10):
    """Return per-bin observed accuracy and mean confidence (for plotting)."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    accs, confs, counts = [], [], []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (scores >= lo) & (scores < hi if i < n_bins - 1 else scores <= hi)
        if mask.sum() == 0:
            continue
        accs.append(y_true[mask].mean())
        confs.append(scores[mask].mean())
        counts.append(int(mask.sum()))
    return np.array(confs), np.array(accs), np.array(counts)


def _ctd_residualise(X_train_log, X_test_log, feat_names):
    """Replicate the CTD compartment regression used in the main pipeline."""
    from external_validation_gpl16791 import _ctd_residualise as _impl
    return _impl(X_train_log, X_test_log, feat_names)


def _load_platform(platform: str):
    from external_validation_gpl16791 import _load_platform as _impl
    return _impl(platform)


def _extract_panel(X, feat_names, panel_features):
    feat_to_idx = {f: i for i, f in enumerate(feat_names)}
    cols = [feat_to_idx[f] for f in panel_features if f in feat_to_idx]
    return X[:, cols]


def _fit_lgbm(X_tr, y_train):
    from lightgbm import LGBMClassifier
    params = json.loads(_PARAMS_PATH.read_text())
    params["colsample_bytree"] = 1.0
    clf = LGBMClassifier(**params)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        clf.fit(X_tr, y_train)
    return clf


def _zeroshot_and_cv(X_tr, X_te, y_train):
    """Fit on the full train cohort and score the test cohort (zero-shot);
    also return 5-fold stratified CV scores on the training cohort.

    Args:
        X_tr: Training-cohort panel matrix.
        X_te: Test-cohort panel matrix (same columns).
        y_train: Training labels.

    Returns:
        (train_cv_scores, test_scores) probability arrays.
    """
    from sklearn.model_selection import StratifiedKFold

    clf_full = _fit_lgbm(X_tr, y_train)
    test_scores = clf_full.predict_proba(X_te)[:, 1]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_cv_scores = np.zeros(len(y_train))
    for tr_idx, val_idx in cv.split(X_tr, y_train):
        cv_clf = _fit_lgbm(X_tr[tr_idx], y_train[tr_idx])
        train_cv_scores[val_idx] = cv_clf.predict_proba(X_tr[val_idx])[:, 1]
    return train_cv_scores, test_scores


def main() -> None:
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

    print("\n" + "=" * 70)
    print("Calibration Analysis — 25-gene panel + 15-gene critical panel (zero-shot)")
    print("=" * 70)

    panel = pd.read_csv(_PANEL_CSV)
    panel_features = panel["feature"].tolist()

    print("\nLoading GPL24676 (train) and GPL16791 (test) ...")
    X_train_raw, feat_train, y_train, _ = _load_platform("GPL24676")
    X_test_raw, feat_test, y_test, tissues_test = _load_platform("GPL16791")

    assert feat_train == feat_test
    X_train_log = np.log1p(X_train_raw)
    X_test_log = np.log1p(X_test_raw)

    print("Applying CTD compartment regression ...")
    X_train_res, X_test_res = _ctd_residualise(X_train_log, X_test_log, feat_train)

    # 25-gene panel
    X_tr = _extract_panel(X_train_res, feat_train, panel_features)
    X_te = _extract_panel(X_test_res, feat_test, panel_features)
    print(f"  25-gene panel matched: train {X_tr.shape}, test {X_te.shape}")

    # 15-gene critical panel (subset of the 25-gene panel)
    crit_features = [panel_features[i] for i in _CRITICAL_IDX]
    X_tr_c = _extract_panel(X_train_res, feat_train, crit_features)
    X_te_c = _extract_panel(X_test_res, feat_test, crit_features)
    print(f"  critical panel matched: train {X_tr_c.shape}, test {X_te_c.shape}")

    print("\nComputing 5-fold CV + zero-shot scores (both panels) ...")
    train_cv_scores, test_scores = _zeroshot_and_cv(X_tr, X_te, y_train)
    train_cv_scores_c, test_scores_c = _zeroshot_and_cv(X_tr_c, X_te_c, y_train)
    print(f"  25-gene  train CV AUC = {roc_auc_score(y_train, train_cv_scores):.4f}")
    print(f"  critical train CV AUC = {roc_auc_score(y_train, train_cv_scores_c):.4f}")

    # Save score arrays for follow-up
    np.savez(
        SCRIPT_DIR / "calibration_scores.npz",
        train_cv_scores=train_cv_scores, y_train=y_train,
        gpl16791_scores=test_scores, gpl16791_y=y_test,
        train_cv_scores_crit=train_cv_scores_c, gpl16791_scores_crit=test_scores_c,
    )

    # Calibration metrics for both panels on each cohort
    results = []
    for panel_name, sc_tr, sc_te in [
        ("25-gene", train_cv_scores, test_scores),
        ("15-gene critical", train_cv_scores_c, test_scores_c),
    ]:
        for label, y, s in [
            ("GPL24676_5foldCV", y_train, sc_tr),
            ("GPL16791_zeroshot", y_test, sc_te),
        ]:
            brier = float(brier_score_loss(y, s))
            ll = float(log_loss(y, np.clip(s, 1e-7, 1 - 1e-7)))
            auc = float(roc_auc_score(y, s))
            ece = _expected_calibration_error(y, s, n_bins=10)
            results.append({
                "panel": panel_name, "cohort": label, "n": len(y),
                "n_pos": int(y.sum()), "AUC": round(auc, 4),
                "brier": round(brier, 4), "log_loss": round(ll, 4),
                "ECE_10bin": round(ece, 4),
                "mean_score_pos": round(float(s[y == 1].mean()), 4),
                "mean_score_neg": round(float(s[y == 0].mean()), 4),
            })

    df_res = pd.DataFrame(results)
    df_res.to_csv(SCRIPT_DIR / "calibration_summary.csv", index=False)
    print("\nCalibration summary:")
    print(df_res.to_string(index=False))

    # Reliability plots — 25-gene and 15-gene critical overlaid per cohort
    panel_curves = [
        ("25-gene panel", "#1f77b4", train_cv_scores, test_scores),
        ("15-gene critical", "#d62728", train_cv_scores_c, test_scores_c),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    cohorts = [
        ("Train cohort (GPL24676; 5-fold CV)", y_train, 0),
        ("GPL16791 cross-platform (zero-shot)", y_test, 1),
    ]
    for ax, (title, y, which) in zip(axes, cohorts):
        ax.plot([0, 1], [0, 1], "--", color="gray", lw=1.0, label="perfect")
        for pname, color, tr_s, te_s in panel_curves:
            s = tr_s if which == 0 else te_s
            confs, accs, _ = _reliability(y, s, n_bins=10)
            brier = float(brier_score_loss(y, s))
            auc = float(roc_auc_score(y, s))
            ax.plot(confs, accs, "o-", color=color, lw=1.5, markersize=5,
                    label=f"{pname}: AUC={auc:.3f}, Brier={brier:.3f}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed fraction of ALS")
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(alpha=0.3)
        ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    fig.savefig(SCRIPT_DIR / "calibration_reliability.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    print("\nSaved -> calibration_reliability.png")

    # Statistics file
    lines = [
        "Calibration analysis (25-gene panel + 15-gene critical panel)",
        "=" * 60,
        "Discrimination + calibration on training and primary external cohort.",
        "Both panels use the same LightGBM model (top-500 hyperparameters) on",
        "CTD-residualised data; the critical panel is the 15-gene greedy-elimination",
        "subset of the 25-gene panel.",
        "",
        "Brier score (smaller is better; 0 = perfect, 0.25 = uninformative).",
        "Expected Calibration Error (ECE; 10 equal-width bins).",
        "Log-loss (cross-entropy; smaller is better).",
        "",
        df_res.to_string(index=False),
        "",
        "Interpretation:",
        "  GPL24676 5-fold CV: in-sample calibration of the model used for all transfer.",
        "  GPL16791 zero-shot: real-world calibration when applied to a different platform.",
        "  ECE quantifies the gap between predicted probability and observed frequency;",
        "  values > 0.05 are typically considered miscalibrated for clinical use.",
    ]
    (SCRIPT_DIR / "calibration_analysis_statistics.txt").write_text("\n".join(lines))
    print("Saved -> calibration_analysis_statistics.txt")


if __name__ == "__main__":
    main()
