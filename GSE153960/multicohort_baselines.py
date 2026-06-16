"""Multi-cohort LR baseline + Brier/calibration across all validation cohorts.

Extends the GPL16791-only LR baseline to GSE76220 (LCM SC), GSE122649
(motor cortex), SRP064478 (cervical SC), GSE234297 (blood). For each cohort,
reports zero-shot AUC + Brier + log-loss + mean control score for both
LightGBM and L2-LR on the 25-gene panel.

The training pipeline is the existing one (with CTD residualisation).
Test cohorts that aren't matched by Ensembl ID (HGNC-only) use
symbol-based matching with column-mean imputation for missing genes,
matching the validation_panel_loo.py pipeline.

Outputs: multicohort_baselines.csv, .txt
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
    _ctd_residualise, _load_platform,
)
from utils import load_dataset  # noqa: E402

_PANEL_CSV = SCRIPT_DIR / "lgbm_core25_panel.csv"
_PARAMS_PATH = SCRIPT_DIR / "lgbm_top500_best_params.json"


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


def _extract_panel(X, feat_names, panel_features):
    fmap = {f: i for i, f in enumerate(feat_names)}
    return X[:, [fmap[f] for f in panel_features if f in fmap]]


def _score_cohort(name, X_te, y_te, lgbm, lr, lr_scaler):
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
    s_lgbm = lgbm.predict_proba(X_te)[:, 1]
    s_lr = lr.predict_proba(lr_scaler.transform(X_te))[:, 1]
    out = []
    for model_name, s in [("LGBM", s_lgbm), ("LR_L2", s_lr)]:
        out.append({
            "cohort": name, "model": model_name,
            "n": len(y_te), "n_pos": int(y_te.sum()),
            "AUC": round(float(roc_auc_score(y_te, s)), 4),
            "brier": round(float(brier_score_loss(y_te, s)), 4),
            "log_loss": round(float(log_loss(y_te, np.clip(s, 1e-7, 1 - 1e-7))), 4),
            "mean_pos": round(float(s[y_te == 1].mean()), 4),
            "mean_neg": round(float(s[y_te == 0].mean()), 4) if (y_te == 0).any() else float("nan"),
        })
    return out


def _load_external_by_symbol(gse_id: str, platform: str, panel_symbols: list[str]):
    """Load external cohort and match panel by HGNC symbol (with column-mean fill)."""
    try:
        (ds,) = load_dataset(gse_id, platform=platform,
                             resources_dir=ALS_DIR / "resources")
    except Exception as exc:
        print(f"  [warn] {gse_id}/{platform}: {exc}")
        return None
    cols = list(ds.X.columns)
    # symbol-based matching: external cohort columns may already be symbols
    # Try direct symbol match first
    sym_lookup = {c: i for i, c in enumerate(cols)}
    X = ds.X.values.astype(np.float32)
    y = ds.y.values.astype(int)
    matched = []
    matched_cols = []
    missing = []
    for sym in panel_symbols:
        if sym in sym_lookup:
            matched.append(sym)
            matched_cols.append(sym_lookup[sym])
        else:
            missing.append(sym)
    if missing:
        print(f"  [{gse_id}] missing {len(missing)} symbols: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if not matched_cols:
        return None
    X_panel = np.log1p(np.nan_to_num(X[:, matched_cols], nan=0.0))
    # Build full 25-column matrix with col-mean fill for missing
    result = np.zeros((X_panel.shape[0], len(panel_symbols)), dtype=np.float32)
    col_means = X_panel.mean(axis=0) if X_panel.shape[0] else None
    fill_value = float(np.nan_to_num(col_means.mean(), nan=0.0)) if col_means is not None else 0.0
    j = 0
    for i, sym in enumerate(panel_symbols):
        if sym in sym_lookup:
            result[:, i] = np.nan_to_num(X_panel[:, j], nan=fill_value)
            j += 1
        else:
            result[:, i] = fill_value
    return result, y


def main() -> None:
    panel = pd.read_csv(_PANEL_CSV)
    panel_features = panel["feature"].tolist()
    panel_symbols = panel["symbol"].tolist()

    print("Loading GPL24676 training set ...")
    X_train_raw, feat_train, y_train, _ = _load_platform("GPL24676")
    X_train_log = np.log1p(X_train_raw)
    # CTD residualisation against itself
    X_train_res, _ = _ctd_residualise(X_train_log, X_train_log, feat_train)
    X_tr = _extract_panel(X_train_res, feat_train, panel_features)
    print(f"  shape: {X_tr.shape}")

    print("\nFitting LGBM and LR-L2 on training data ...")
    lgbm = _fit_lgbm(X_tr, y_train)
    lr, scaler = _fit_lr(X_tr, y_train)

    all_results = []

    # Cohort 1: GPL16791 (Ensembl ID match, full pipeline)
    print("\nGPL16791 ...")
    X_test_raw, feat_test, y_test, _ = _load_platform("GPL16791")
    X_test_log = np.log1p(X_test_raw)
    _, X_test_res = _ctd_residualise(X_train_log, X_test_log, feat_train)
    X_te = _extract_panel(X_test_res, feat_test, panel_features)
    all_results.extend(_score_cohort("GPL16791", X_te, y_test, lgbm, lr, scaler))

    # External cohorts via symbol matching (no CTD residualisation possible)
    for gse, plat in [("GSE76220", None), ("GSE122649", None), ("GSE234297", None)]:
        print(f"\n{gse} ...")
        loaded = _load_external_by_symbol(gse, plat, panel_symbols)
        if loaded is None:
            print(f"  skipped")
            continue
        X_ext, y_ext = loaded
        all_results.extend(_score_cohort(gse, X_ext, y_ext, lgbm, lr, scaler))

    df = pd.DataFrame(all_results)
    df.to_csv(SCRIPT_DIR / "multicohort_baselines.csv", index=False)
    print("\n" + df.to_string(index=False))

    lines = [
        "Multi-cohort LR-L2 vs LGBM baseline + calibration",
        "=" * 60,
        df.to_string(index=False),
        "",
        "Interpretation:",
        "  - AUC parity (within 0.02) between LR and LGBM cross-cohort indicates",
        "    that most of the panel's discriminative signal is linearly extractable.",
        "  - Brier and log-loss directly compare calibration: lower is better.",
        "  - Mean-neg score gives the average ALS probability assigned to control",
        "    samples; a value near the training prevalence indicates good calibration.",
    ]
    (SCRIPT_DIR / "multicohort_baselines.txt").write_text("\n".join(lines))
    print(f"\nSaved -> multicohort_baselines.{{csv,txt}}")


if __name__ == "__main__":
    main()
