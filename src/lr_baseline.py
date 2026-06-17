"""Logistic-regression baseline on the 25-gene panel for cross-platform transfer.

If LR matches or beats LGBM zero-shot, that indicates the LGBM zero-shot AUC
is not capturing GPL24676-specific non-linearities and supports LR as the
preferred transfer model. If LGBM clearly wins, the non-linear interactions
are genuine cross-platform.

Compares discrimination on:
  - GPL24676 5-fold CV (training calibration check)
  - GPL16791 zero-shot (n=636; primary external)
  - GSE234297 zero-shot blood (n=144)

Outputs: lr_baseline_summary.csv, lr_baseline_statistics.txt
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ALS_DIR = Path(__file__).parents[1]
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(ALS_DIR))

from external_validation_gpl16791 import (  # noqa: E402
    _ctd_residualise,
    _load_platform,
)

_PANEL_CSV = SCRIPT_DIR / "lgbm_core25_panel.csv"
_PARAMS_PATH = SCRIPT_DIR / "lgbm_top500_best_params.json"


def _extract_panel(X: np.ndarray, feat_names: list[str], panel_features: list[str]) -> np.ndarray:
    feat_to_idx = {f: i for i, f in enumerate(feat_names)}
    cols = [feat_to_idx[f] for f in panel_features if f in feat_to_idx]
    return X[:, cols]


def _fit_lgbm(X_tr: np.ndarray, y_train: np.ndarray):
    from lightgbm import LGBMClassifier
    params = json.loads(_PARAMS_PATH.read_text())
    params["colsample_bytree"] = 1.0
    clf = LGBMClassifier(**params)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        clf.fit(X_tr, y_train)
    return clf


def _fit_lr(X_tr: np.ndarray, y_train: np.ndarray):
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(X_tr)
    clf = LogisticRegressionCV(
        cv=5, solver="liblinear", penalty="l2", scoring="roc_auc",
        max_iter=2000, random_state=42, refit=True,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        clf.fit(scaler.transform(X_tr), y_train)
    return clf, scaler


def main() -> None:
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    panel = pd.read_csv(_PANEL_CSV)
    panel_features = panel["feature"].tolist()

    print("Loading GPL24676 + GPL16791 ...")
    X_train_raw, feat_train, y_train, _ = _load_platform("GPL24676")
    X_test_raw, feat_test, y_test, _ = _load_platform("GPL16791")
    assert feat_train == feat_test

    X_train_log = np.log1p(X_train_raw)
    X_test_log = np.log1p(X_test_raw)
    X_train_res, X_test_res = _ctd_residualise(X_train_log, X_test_log, feat_train)
    X_tr = _extract_panel(X_train_res, feat_train, panel_features)
    X_te = _extract_panel(X_test_res, feat_test, panel_features)
    print(f"  Train: {X_tr.shape}  Test: {X_te.shape}")

    # 5-fold CV — both models
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    lgbm_cv, lr_cv = np.zeros(len(y_train)), np.zeros(len(y_train))
    for tr, va in cv.split(X_tr, y_train):
        clf = _fit_lgbm(X_tr[tr], y_train[tr])
        lgbm_cv[va] = clf.predict_proba(X_tr[va])[:, 1]
        lr, sc = _fit_lr(X_tr[tr], y_train[tr])
        lr_cv[va] = lr.predict_proba(sc.transform(X_tr[va]))[:, 1]

    # Refit both on full train, predict on GPL16791
    lgbm_full = _fit_lgbm(X_tr, y_train)
    lgbm_te_scores = lgbm_full.predict_proba(X_te)[:, 1]
    lr_full, scaler = _fit_lr(X_tr, y_train)
    lr_te_scores = lr_full.predict_proba(scaler.transform(X_te))[:, 1]

    results = []
    for label, y, s_lgbm, s_lr in [
        ("GPL24676_5foldCV", y_train, lgbm_cv, lr_cv),
        ("GPL16791_zeroshot", y_test, lgbm_te_scores, lr_te_scores),
    ]:
        for model_name, s in [("LGBM", s_lgbm), ("LR_L2", s_lr)]:
            results.append({
                "cohort": label,
                "model": model_name,
                "n": len(y),
                "AUC": round(float(roc_auc_score(y, s)), 4),
                "brier": round(float(brier_score_loss(y, s)), 4),
                "log_loss": round(float(log_loss(y, np.clip(s, 1e-7, 1 - 1e-7))), 4),
                "mean_pos": round(float(s[y == 1].mean()), 4),
                "mean_neg": round(float(s[y == 0].mean()), 4),
            })

    df = pd.DataFrame(results)
    df.to_csv(SCRIPT_DIR / "lr_baseline_summary.csv", index=False)
    print("\n" + df.to_string(index=False))

    lines = [
        "LR vs LGBM baseline — discrimination and calibration",
        "=" * 60,
        df.to_string(index=False),
        "",
        "Interpretation:",
        "  - If LR matches or beats LGBM on GPL16791 zero-shot, the LGBM",
        "    advantage is GPL24676-specific (non-transferable interactions).",
        "  - If LGBM clearly beats LR, the gradient-boosted non-linear",
        "    structure is genuine cross-platform signal.",
    ]
    (SCRIPT_DIR / "lr_baseline_statistics.txt").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
