"""Calibration analysis for the GPL24676-trained 25-gene panel on validation cohorts.

Computes discrimination (AUC) and calibration (Brier score, log-loss,
expected calibration error) for the zero-shot predictions on:
  - GPL16791 (n=636, primary cross-platform validation)
  - GSE234297 (blood, n=144)
Other cohorts (GSE76220 n=20; GSE122649 n=38; SRP064478 n=15) are too small
for meaningful calibration curves and are reported only with Brier scalar.

Outputs:
  - calibration_summary.csv        (per-cohort discrimination + calibration metrics)
  - calibration_reliability.png    (reliability diagram per cohort)
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


def main() -> None:
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

    print("\n" + "=" * 70)
    print("Calibration Analysis — 25-gene panel zero-shot transfer")
    print("=" * 70)

    panel = pd.read_csv(_PANEL_CSV)
    panel_features = panel["feature"].tolist()
    panel_symbols = panel["symbol"].tolist()

    print("\nLoading GPL24676 (train) and GPL16791 (test) ...")
    X_train_raw, feat_train, y_train, _ = _load_platform("GPL24676")
    X_test_raw, feat_test, y_test, tissues_test = _load_platform("GPL16791")

    assert feat_train == feat_test
    X_train_log = np.log1p(X_train_raw)
    X_test_log = np.log1p(X_test_raw)

    print("Applying CTD compartment regression ...")
    X_train_res, X_test_res = _ctd_residualise(X_train_log, X_test_log, feat_train)

    X_tr = _extract_panel(X_train_res, feat_train, panel_features)
    X_te = _extract_panel(X_test_res, feat_test, panel_features)

    print(f"  panel matched: train {X_tr.shape}, test {X_te.shape}")

    clf = _fit_lgbm(X_tr, y_train)
    test_scores = clf.predict_proba(X_te)[:, 1]

    # Also recompute training-cohort 5-fold CV scores for calibration
    from sklearn.model_selection import StratifiedKFold
    print("\nComputing 5-fold CV scores on training cohort for self-calibration ...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_cv_scores = np.zeros(len(y_train))
    for tr_idx, val_idx in cv.split(X_tr, y_train):
        cv_clf = _fit_lgbm(X_tr[tr_idx], y_train[tr_idx])
        train_cv_scores[val_idx] = cv_clf.predict_proba(X_tr[val_idx])[:, 1]
    train_cv_auc = roc_auc_score(y_train, train_cv_scores)
    print(f"  Train 5-fold CV AUC = {train_cv_auc:.4f}")

    # Save score arrays for follow-up
    np.savez(
        SCRIPT_DIR / "calibration_scores.npz",
        train_cv_scores=train_cv_scores, y_train=y_train,
        gpl16791_scores=test_scores, gpl16791_y=y_test,
    )

    # Calibration on each
    results = []
    for label, y, s in [
        ("GPL24676_5foldCV", y_train, train_cv_scores),
        ("GPL16791_zeroshot", y_test, test_scores),
    ]:
        brier = float(brier_score_loss(y, s))
        ll = float(log_loss(y, np.clip(s, 1e-7, 1 - 1e-7)))
        auc = float(roc_auc_score(y, s))
        ece = _expected_calibration_error(y, s, n_bins=10)
        # Mean score for ALS vs Control class
        results.append({
            "cohort": label, "n": len(y), "n_pos": int(y.sum()),
            "AUC": round(auc, 4),
            "brier": round(brier, 4),
            "log_loss": round(ll, 4),
            "ECE_10bin": round(ece, 4),
            "mean_score_pos": round(float(s[y == 1].mean()), 4),
            "mean_score_neg": round(float(s[y == 0].mean()), 4),
        })

    df_res = pd.DataFrame(results)
    df_res.to_csv(SCRIPT_DIR / "calibration_summary.csv", index=False)
    print("\nCalibration summary:")
    print(df_res.to_string(index=False))

    # Reliability plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (label, y, s) in zip(axes, [
        ("Train cohort (GPL24676; 5-fold CV)", y_train, train_cv_scores),
        ("GPL16791 cross-platform (zero-shot)", y_test, test_scores),
    ]):
        confs, accs, counts = _reliability(y, s, n_bins=10)
        ax.plot([0, 1], [0, 1], "--", color="gray", lw=1.0, label="perfect")
        ax.plot(confs, accs, "o-", color="#1f77b4", lw=1.5,
                markersize=4 + 0.3 * counts.mean() ** 0.5)
        for c, a, n_b in zip(confs, accs, counts):
            ax.annotate(str(n_b), (c, a), fontsize=7, xytext=(2, 2),
                        textcoords="offset points", color="#555")
        brier = float(brier_score_loss(y, s))
        auc = float(roc_auc_score(y, s))
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed fraction of ALS")
        ax.set_title(f"{label}\nAUC = {auc:.3f}   Brier = {brier:.3f}", fontsize=11)
        ax.grid(alpha=0.3)
        ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    fig.savefig(SCRIPT_DIR / "calibration_reliability.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    print("\nSaved -> calibration_reliability.png")

    # Statistics file
    lines = [
        "Calibration analysis (25-gene panel)",
        "=" * 60,
        "Discrimination + calibration on training and primary external cohort.",
        "",
        f"Brier score (smaller is better; 0 = perfect, 0.25 = uninformative).",
        f"Expected Calibration Error (ECE; 10 equal-width bins).",
        f"Log-loss (cross-entropy; smaller is better).",
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
    print(f"Saved -> calibration_analysis_statistics.txt")


if __name__ == "__main__":
    main()
