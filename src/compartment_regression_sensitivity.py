"""Sensitivity to CTD compartment regression.

Reviewer M3 concern: the CTD compartment PCs are fit on the full training
matrix and OLS-regressed out before SHAP/SFM/Bayesian search; this could
inflate in-cohort CV AUC if any fold-specific leakage occurs, and the
test-time projection may behave poorly cross-platform.

This script tests the simpler hypothesis: how much does removing
compartment regression change the headline numbers?

For both LGBM and L2-LR on the 25-gene panel:
  (a) +CTD regression  vs  (b) raw log1p
on GPL24676 5-fold CV and GPL16791 zero-shot transfer.

If the with/without comparison shows large changes, the residualisation is a
load-bearing step that needs CV-aware refitting. If changes are small, the
panel signal is not driven by the residualisation, and the leakage concern
is bounded.

Outputs: compartment_regression_sensitivity.csv, .txt
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
    fmap = {f: i for i, f in enumerate(feat_names)}
    return X[:, [fmap[f] for f in panel_features if f in fmap]]


def _fit_lgbm(X, y):
    from lightgbm import LGBMClassifier
    params = json.loads(_PARAMS_PATH.read_text())
    params["colsample_bytree"] = 1.0
    clf = LGBMClassifier(**params)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        clf.fit(X, y)
    return clf


def _fit_lr(X, y):
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X)
    clf = LogisticRegressionCV(
        cv=5, solver="liblinear", penalty="l2", scoring="roc_auc",
        max_iter=2000, random_state=42, refit=True,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        clf.fit(sc.transform(X), y)
    return clf, sc


def _cv_scores(X, y, model_fn):
    """5-fold stratified CV scores for either lgbm or lr."""
    from sklearn.model_selection import StratifiedKFold
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    out = np.zeros(len(y))
    for tr, va in cv.split(X, y):
        if model_fn == "lgbm":
            clf = _fit_lgbm(X[tr], y[tr])
            out[va] = clf.predict_proba(X[va])[:, 1]
        else:
            clf, sc = _fit_lr(X[tr], y[tr])
            out[va] = clf.predict_proba(sc.transform(X[va]))[:, 1]
    return out


def main() -> None:
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

    panel = pd.read_csv(_PANEL_CSV)
    panel_features = panel["feature"].tolist()

    print("Loading GPL24676 + GPL16791 ...")
    X_train_raw, feat_train, y_train, _ = _load_platform("GPL24676")
    X_test_raw, feat_test, y_test, _ = _load_platform("GPL16791")
    assert feat_train == feat_test

    X_train_log = np.log1p(X_train_raw)
    X_test_log = np.log1p(X_test_raw)

    # +CTD condition
    X_train_res, X_test_res = _ctd_residualise(X_train_log, X_test_log, feat_train)
    X_tr_ctd = _extract_panel(X_train_res, feat_train, panel_features)
    X_te_ctd = _extract_panel(X_test_res, feat_test, panel_features)

    # raw (no CTD) condition
    X_tr_raw = _extract_panel(X_train_log, feat_train, panel_features)
    X_te_raw = _extract_panel(X_test_log, feat_test, panel_features)

    print(f"  shapes: CTD-train {X_tr_ctd.shape}  raw-train {X_tr_raw.shape}")

    results = []
    for cond_name, X_tr, X_te in [("+CTD", X_tr_ctd, X_te_ctd), ("raw", X_tr_raw, X_te_raw)]:
        # In-cohort 5-fold CV
        for model_name, model_fn in [("LGBM", "lgbm"), ("LR_L2", "lr")]:
            cv_s = _cv_scores(X_tr, y_train, model_fn)
            results.append({
                "condition": cond_name,
                "model": model_name,
                "evaluation": "GPL24676_5foldCV",
                "n": len(y_train),
                "AUC": round(float(roc_auc_score(y_train, cv_s)), 4),
                "brier": round(float(brier_score_loss(y_train, cv_s)), 4),
                "log_loss": round(float(log_loss(y_train, np.clip(cv_s, 1e-7, 1 - 1e-7))), 4),
            })

        # Zero-shot
        lgbm = _fit_lgbm(X_tr, y_train)
        te_s_lgbm = lgbm.predict_proba(X_te)[:, 1]
        lr, sc = _fit_lr(X_tr, y_train)
        te_s_lr = lr.predict_proba(sc.transform(X_te))[:, 1]

        for model_name, te_s in [("LGBM", te_s_lgbm), ("LR_L2", te_s_lr)]:
            results.append({
                "condition": cond_name,
                "model": model_name,
                "evaluation": "GPL16791_zeroshot",
                "n": len(y_test),
                "AUC": round(float(roc_auc_score(y_test, te_s)), 4),
                "brier": round(float(brier_score_loss(y_test, te_s)), 4),
                "log_loss": round(float(log_loss(y_test, np.clip(te_s, 1e-7, 1 - 1e-7))), 4),
            })

    df = pd.DataFrame(results)
    df.to_csv(SCRIPT_DIR / "compartment_regression_sensitivity.csv", index=False)
    print("\n" + df.to_string(index=False))

    # Delta table
    delta_rows = []
    for ev in ("GPL24676_5foldCV", "GPL16791_zeroshot"):
        for mdl in ("LGBM", "LR_L2"):
            a = df[(df["condition"] == "+CTD") & (df["evaluation"] == ev) & (df["model"] == mdl)].iloc[0]
            b = df[(df["condition"] == "raw") & (df["evaluation"] == ev) & (df["model"] == mdl)].iloc[0]
            delta_rows.append({
                "evaluation": ev, "model": mdl,
                "dAUC_ctd_minus_raw": round(a["AUC"] - b["AUC"], 4),
                "dBrier_ctd_minus_raw": round(a["brier"] - b["brier"], 4),
            })
    delta_df = pd.DataFrame(delta_rows)
    print("\n+CTD minus raw (a positive AUC delta means CTD helps):")
    print(delta_df.to_string(index=False))

    lines = [
        "Compartment-regression sensitivity (M3)",
        "=" * 60,
        "Compares +CTD regression vs raw log1p for both LGBM and LR-L2",
        "on 5-fold CV (GPL24676) and zero-shot transfer (GPL16791).",
        "",
        df.to_string(index=False),
        "",
        "Delta (+CTD minus raw):",
        delta_df.to_string(index=False),
        "",
        "Interpretation:",
        "  - Small |dAUC| (<0.01) means CTD residualisation has no meaningful",
        "    effect on the panel-level discrimination/calibration; the leakage",
        "    concern is bounded.",
        "  - Large |dAUC| means the residualisation is load-bearing and",
        "    needs to be refitted in a CV-aware manner.",
    ]
    (SCRIPT_DIR / "compartment_regression_sensitivity.txt").write_text("\n".join(lines))
    print(f"\nSaved -> compartment_regression_sensitivity.{{csv,txt}}")


if __name__ == "__main__":
    main()
